# map_matching.py
"""
Mixin com todo o módulo de Localização Absoluta (map matching) baseado na
dissertação de Bruno Dantas (ITA 2023, Seção 3.2.3).

A classe MapMatchingMixin é herdada por OdometriaVisual e acessa atributos
da classe principal via self (self.config, self.map_dataset, etc.).
"""

import logging

import numpy as np
import cv2 as cv
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import Affine
from pyproj import Transformer
from geopy.distance import distance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constantes (Tabela 3.1 da dissertação)
# ---------------------------------------------------------------------------

# Limiares mínimos de inliers: (limiar_posicao, limiar_angulo_escala)
_INLIER_THRESHOLDS = {
    'LOFTR':       (20, 40),
    'MATCHFORMER': (16, 20),
    'SUPERPOINT':  (16, 20),
    'ORB':         (10, 15),
    'AKAZE':       (10, 15),
}
_DEFAULT_INLIER_THRESHOLDS = (16, 20)

# Tamanho alvo (px) para o patch e para a imagem aérea no map matching
_MAP_MATCH_TARGET_PX = 480


# ---------------------------------------------------------------------------
# Função auxiliar (usada apenas pelo mixin)
# ---------------------------------------------------------------------------

def _rotacionar_imagem(img, angulo_graus):
    """
    Rotaciona imagem em torno do centro sem cortar bordas.
    Retorna (img_rotacionada, M_inv) onde M_inv é a matriz afim inversa
    para mapear pixels de volta à imagem original.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv.getRotationMatrix2D((cx, cy), angulo_graus, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0
    img_rot = cv.warpAffine(img, M, (new_w, new_h))
    M_inv = cv.invertAffineTransform(M)
    return img_rot, M_inv


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------

class MapMatchingMixin:
    """
    Mixin que adiciona o módulo de Localização Absoluta à OdometriaVisual.

    Atributos esperados na classe hospedeira (definidos em __init__):
        self.config, self.map_dataset, self._map_transformer,
        self.roi_margin_factor, self.escala_atual, self.scale_search_step,
        self.angle_search_range_deg, self.angle_search_candidates,
        self.inlier_thr_position, self.inlier_thr_angle_scale,
        self.fx, self.cx  (para _inicializar_escala)

    self.roi_size_m e self.gsd_voo_efetivo são calculados dinamicamente por
    _inicializar_escala (não são lidos diretamente do config — ver método).

    Métodos esperados na classe hospedeira:
        self._obter_correspondencias(...)
        self.calcula_distancia_latlon(...)
    """

    # ------------------------------------------------------------------
    # Inicialização do mapa e escala
    # ------------------------------------------------------------------

    def _inicializar_mapa(self):
        """Abre o arquivo GeoTIFF de forma lazy (sem carregar tudo na RAM)."""
        map_path = self.config['paths']['map_tif_path']
        logger.info("Carregando mapa base para Map Matching: %s", map_path)
        self.map_dataset = rasterio.open(map_path)
        map_crs = self.map_dataset.crs
        if map_crs and not map_crs.is_geographic:
            self._map_transformer = Transformer.from_crs(
                "EPSG:4326", map_crs, always_xy=True
            )
            logger.info("CRS do GeoTIFF: %s (projetado — reproj ativado)", map_crs.to_string())
        else:
            self._map_transformer = None
            logger.info("CRS do GeoTIFF: %s (geográfico — sem reproj)", map_crs)

    def _inicializar_escala(self, altura_inicial):
        """
        Calcula o GSD de referência do voo e deriva roi_size_m dinamicamente
        a partir dele (em vez de um valor fixo em metros no config).

        gsd_voo_efetivo já é ajustado para o tamanho real do patch usado no
        matching (_MAP_MATCH_TARGET_PX=480px), não o tamanho original da
        câmera (ex. 640px) — sem esse ajuste a escala de referência ficaria
        sistematicamente ~(largura_original/TARGET) mais fina do que deveria.

        A partir desta inicialização, a conversão de resolução mapa→voo é
        feita explicitamente em _preparar_patch_satelital (reamostragem do
        crop bruto para casar com gsd_voo_efetivo); 'escala' deixa de
        carregar essa conversão de unidades e passa a ser só o fator de
        ajuste fino (~1.0) que a busca gulosa (_busca_escala,
        ±scale_search_step por frame) refina frame a frame.
        """
        largura_camera_original = 2.0 * self.cx
        gsd_voo = altura_inicial / self.fx
        self.gsd_voo_efetivo = gsd_voo * (largura_camera_original / _MAP_MATCH_TARGET_PX)

        self.roi_size_m = _MAP_MATCH_TARGET_PX * self.gsd_voo_efetivo * self.roi_margin_factor

        self.escala_atual = 1.0
        self._escala_inicializada = True
        logger.info(
            "GSD do voo (efetivo, TARGET=%dpx): %.3fm/px | roi_size_m derivado: "
            "%.1fm (roi_margin_factor=%.2f) | escala inicial: 1.0",
            _MAP_MATCH_TARGET_PX, self.gsd_voo_efetivo, self.roi_size_m, self.roi_margin_factor
        )

    # ------------------------------------------------------------------
    # Preparação do patch satelital
    # ------------------------------------------------------------------

    def _bbox_ao_redor(self, lat, lon, half):
        """Bbox geodésica (minx, miny, maxx, maxy) de lado 2*half (m) centrada em (lat, lon)."""
        p_min_lat = distance(meters=half).destination((lat, lon), bearing=180)
        p_min_lon = distance(meters=half).destination((lat, lon), bearing=270)
        p_max_lon = distance(meters=half).destination((lat, lon), bearing=90)
        p_max_lat = distance(meters=half).destination((lat, lon), bearing=0)
        return (p_min_lon.longitude, p_min_lat.latitude,
                p_max_lon.longitude, p_max_lat.latitude)

    def _recortar_roi_mapa(self, lat, lon, margem_extra=1.5, lat_real=None, lon_real=None):
        """
        Recorta um patch do GeoTIFF centrado na posição estimada (lat/lon).
        O parâmetro margem_extra amplia a ROI para acomodar rotação posterior.

        Se (lat_real, lon_real) for informado (modo diagnóstico
        roi_center_mode='real_bbox'), a ROI passa a ser a menor bbox que
        contém tanto a posição estimada quanto a real, cada uma com a
        margem normal ao redor — em vez de uma bbox de raio fixo em torno
        de um único ponto. Isso garante que a posição real esteja sempre
        dentro do recorte pesquisado, independente da deriva acumulada do
        dead-reckoning; usado apenas para isolar a qualidade do
        matching/correção do problema de "onde procurar" sem GPS.
        """
        half = (self.roi_size_m * margem_extra) / 2.0
        minx, miny, maxx, maxy = self._bbox_ao_redor(lat, lon, half)

        if lat_real is not None and lon_real is not None:
            minx2, miny2, maxx2, maxy2 = self._bbox_ao_redor(lat_real, lon_real, half)
            minx, miny = min(minx, minx2), min(miny, miny2)
            maxx, maxy = max(maxx, maxx2), max(maxy, maxy2)

        if self._map_transformer is not None:
            minx, miny = self._map_transformer.transform(minx, miny)
            maxx, maxy = self._map_transformer.transform(maxx, maxy)

        window = from_bounds(minx, miny, maxx, maxy, self.map_dataset.transform)

        # boundless=True é essencial: sem isso, uma janela que ultrapassa os
        # limites do GeoTIFF é silenciosamente recortada pelo rasterio, o
        # que desloca o centro geográfico real do patch em relação ao centro
        # pedido (lat, lon) sem gerar erro — corrompendo a geo-referência
        # (T_final) mesmo com matching/inliers aparentemente corretos.
        ds_w, ds_h = self.map_dataset.width, self.map_dataset.height
        if (window.col_off < 0 or window.row_off < 0 or
                window.col_off + window.width > ds_w or
                window.row_off + window.height > ds_h):
            logger.warning(
                "ROI do map matching ultrapassa os limites do GeoTIFF perto de "
                "(%.6f, %.6f) — roi_size_m=%.0fm pode ser grande demais para a "
                "cobertura do mapa base nesta posição; preenchendo com zero "
                "fora da área coberta.", lat, lon, self.roi_size_m
            )
        patch_transform = self.map_dataset.window_transform(window)
        precisa_cor = getattr(self.detector_absoluto, 'requires_color', False)

        if precisa_cor and self.map_dataset.count >= 3:
            patch = self.map_dataset.read([1, 2, 3], window=window, boundless=True, fill_value=0)
            patch = np.transpose(patch, (1, 2, 0))  # (3,H,W) -> (H,W,3), ordem RGB
            if patch.size == 0:
                return np.zeros((1, 1, 3), dtype=np.uint8), patch_transform
            patch_8u = np.stack([
                cv.normalize(patch[..., c], None, alpha=0, beta=255,
                             norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
                for c in range(3)
            ], axis=-1)
            # PIL/RomaDetector espera RGB; o resto do pipeline (cv.resize,
            # warpAffine, copyMakeBorder) é agnóstico à ordem dos canais —
            # cv.cvtColor(BGR2RGB) em RomaDetector.match faz a troca correta
            # só se a origem for de fato BGR. Mapa base já está em RGB (bandas
            # 1,2,3 do GeoTIFF), então convertemos aqui para BGR e manter a
            # mesma convenção do resto do pipeline (cv.imread/frame de voo).
            patch_8u = cv.cvtColor(patch_8u, cv.COLOR_RGB2BGR)
            return patch_8u, patch_transform

        if precisa_cor:
            logger.warning(
                "map_matching.absolute_detector exige cor, mas o mapa base "
                "(%s) tem só %d banda(s) — usando grayscale replicado em 3 "
                "canais como fallback degradado (não é cor real).",
                self.config['paths'].get('map_tif', '?'), self.map_dataset.count
            )
            patch = self.map_dataset.read(1, window=window, boundless=True, fill_value=0)
            if patch.size == 0:
                return np.zeros((1, 1, 3), dtype=np.uint8), patch_transform
            patch_8u = cv.normalize(patch, None, alpha=0, beta=255,
                                    norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
            if patch_8u is None:
                patch_8u = np.zeros_like(patch, dtype=np.uint8)
            return cv.cvtColor(patch_8u, cv.COLOR_GRAY2BGR), patch_transform

        patch = self.map_dataset.read(1, window=window, boundless=True, fill_value=0)
        if patch.size == 0:
            return np.zeros((1, 1), dtype=np.uint8), patch_transform

        patch_8u = cv.normalize(patch, None, alpha=0, beta=255,
                                norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        if patch_8u is None:
            patch_8u = np.zeros_like(patch, dtype=np.uint8)
        return patch_8u, patch_transform

    def _preparar_patch_satelital(self, lat, lon, angulo_graus, escala, gsd_voo=None,
                                   lat_real=None, lon_real=None):
        """
        Prepara patch satelital para comparação (Seção 3.2.3 da dissertação):
        1. Recorta patch maior com margem (_recortar_roi_mapa)
        2. Reamostra o crop bruto para que sua resolução (m/px) case com o
           GSD da imagem aérea (gsd_voo) — não apenas "cabe em TARGET px"
           como antes (esse resize por tamanho, dissociado do GSD, era parte
           do bug de escala: com ROI grande sobre mapa de alta resolução
           nativa, degradava a resolução do patch antes de a escala atuar)
        3. Rotaciona pelo angulo_graus (_rotacionar_imagem)
        4. Aplica escala fina (ajuste da busca gulosa) e crop/padding central
           para TARGET×TARGET

        Retorna: (patch_final, T_final, M_rot_inv, shape_antes_rot)
        onde T_final mapeia pixels do patch_final para coordenadas geo.
        """
        TARGET = _MAP_MATCH_TARGET_PX
        MAX_PATCH_PX = 4000  # cap de segurança contra upsample explosivo
        escala = max(escala, 0.01)
        gsd_voo = gsd_voo if gsd_voo is not None else self.gsd_voo_efetivo

        # 1. Recortar com margem
        patch_big, T_big = self._recortar_roi_mapa(
            lat, lon, margem_extra=1.5, lat_real=lat_real, lon_real=lon_real
        )

        # 1b. (opcional, experimento "varredura de GSD do mapa base") degrada o
        # crop bruto para simular uma fonte de satélite com GSD mais grosseiro,
        # SEM trocar a fonte nem a altitude: reamostra o crop para o GSD-alvo
        # (down com INTER_AREA) e de volta ao tamanho original (up com
        # INTER_LINEAR), o que é equivalente a borrar a imagem à resolução-alvo
        # preservando shape e T_big (a geo-referência não muda). Só atua quando
        # o GSD-alvo é mais grosseiro que a resolução nativa do mapa.
        gsd_alvo = getattr(self, 'base_map_degrade_gsd', None)
        if gsd_alvo:
            res_atual = abs(T_big.a)
            if gsd_alvo > res_atual:
                f = res_atual / gsd_alvo  # < 1
                h0, w0 = patch_big.shape[:2]
                wd, hd = max(1, int(round(w0 * f))), max(1, int(round(h0 * f)))
                patch_big = cv.resize(
                    cv.resize(patch_big, (wd, hd), interpolation=cv.INTER_AREA),
                    (w0, h0), interpolation=cv.INTER_LINEAR
                )

        # 2. Reamostrar para casar a resolução do crop com o GSD do voo
        h_big, w_big = patch_big.shape[:2]
        res_mapa_m = abs(T_big.a)
        fator_resample = (res_mapa_m / gsd_voo) if gsd_voo > 0 else 1.0
        novo_w = max(1, int(round(w_big * fator_resample)))
        novo_h = max(1, int(round(h_big * fator_resample)))
        if max(novo_w, novo_h) > MAX_PATCH_PX:
            cap = MAX_PATCH_PX / max(novo_w, novo_h)
            novo_w = max(1, int(novo_w * cap))
            novo_h = max(1, int(novo_h * cap))
            fator_resample *= cap
            logger.warning(
                "Patch satelital excedeu %dpx na reamostragem; aplicado cap "
                "de segurança (roi_size_m/gsd_voo incoerentes?)", MAX_PATCH_PX
            )
        interp = cv.INTER_AREA if fator_resample < 1.0 else cv.INTER_LINEAR
        patch_resized = cv.resize(patch_big, (novo_w, novo_h), interpolation=interp)
        T_resized = Affine(
            T_big.a / fator_resample, T_big.b / fator_resample, T_big.c,
            T_big.d / fator_resample, T_big.e / fator_resample, T_big.f
        )

        shape_antes_rot = patch_resized.shape

        # 3. Rotacionar
        patch_rot, M_rot_inv = _rotacionar_imagem(patch_resized, angulo_graus)

        # 4. Aplicar escala fina e crop/padding central para TARGET×TARGET
        h_rot, w_rot = patch_rot.shape[:2]
        new_w_s = max(1, int(round(w_rot * escala)))
        new_h_s = max(1, int(round(h_rot * escala)))
        patch_scaled = cv.resize(patch_rot, (new_w_s, new_h_s))

        h_s, w_s = patch_scaled.shape[:2]
        pad_top = pad_left = 0
        if h_s < TARGET or w_s < TARGET:
            logger.debug(
                "Patch satelital menor que TARGET (%dx%d) — roi_size_m pode "
                "estar pequeno para o gsd_voo atual; aplicando padding.",
                w_s, h_s
            )
            pad_h = max(0, TARGET - h_s)
            pad_w = max(0, TARGET - w_s)
            pad_top, pad_left = pad_h // 2, pad_w // 2
            patch_scaled = cv.copyMakeBorder(
                patch_scaled, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
                cv.BORDER_CONSTANT, value=0
            )
            h_s, w_s = patch_scaled.shape[:2]

        y1 = max(0, (h_s - TARGET) // 2)
        x1 = max(0, (w_s - TARGET) // 2)
        y2 = min(h_s, y1 + TARGET)
        x2 = min(w_s, x1 + TARGET)
        patch_final = patch_scaled[y1:y2, x1:x2]

        # Compor transformação inversa completa: patch_final → geo
        # Origem do crop relativa a patch_scaled ANTES do padding (padding só
        # estende bordas com zero, não desloca a origem geo do conteúdo real)
        x1_sem_pad = x1 - pad_left
        y1_sem_pad = y1 - pad_top
        # M_cs: crop+escala inverso (patch_final → patch_rot)
        M_cs = Affine(1.0 / escala, 0.0, x1_sem_pad / escala,
                      0.0, 1.0 / escala, y1_sem_pad / escala)
        # M_rot_affine: rotação inversa (patch_rot → patch_resized)
        M_rot_affine = Affine(
            M_rot_inv[0, 0], M_rot_inv[0, 1], M_rot_inv[0, 2],
            M_rot_inv[1, 0], M_rot_inv[1, 1], M_rot_inv[1, 2]
        )
        # T_final: patch_final → geo
        T_final = T_resized * M_rot_affine * M_cs

        return patch_final, T_final, M_rot_inv, shape_antes_rot

    # ------------------------------------------------------------------
    # Normalização fotométrica (experimento: reduzir o gap de domínio
    # tonalidade/brilho/detalhe entre foto de voo e mapa base satelital)
    # ------------------------------------------------------------------

    @staticmethod
    def _para_gray(img):
        if img is not None and img.ndim == 3:
            return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def _match_histograms_cdf(src, ref):
        """Casa o histograma de src ao de ref (ambos grayscale uint8) via CDF,
        sem depender de scikit-image (ausente no venv). Aproxima o brilho/
        contraste global da foto de voo ao do patch satelital."""
        src_hist = np.bincount(src.ravel(), minlength=256).astype(np.float64)
        ref_hist = np.bincount(ref.ravel(), minlength=256).astype(np.float64)
        src_cdf = np.cumsum(src_hist)
        ref_cdf = np.cumsum(ref_hist)
        if src_cdf[-1] == 0 or ref_cdf[-1] == 0:
            return src
        src_cdf /= src_cdf[-1]
        ref_cdf /= ref_cdf[-1]
        lut = np.interp(src_cdf, ref_cdf, np.arange(256)).astype(np.uint8)
        return lut[src]

    @staticmethod
    def _grad_mag(img):
        """Magnitude do gradiente Sobel normalizada para uint8 — casa
        estrutura (bordas) em vez de intensidade bruta, insensível a
        diferenças de tonalidade/brilho entre as fontes."""
        gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=3)
        gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=3)
        mag = cv.magnitude(gx, gy)
        return cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

    def _preprocessar_para_match(self, img_aerea, patch):
        """Aplica a normalização fotométrica configurada (self.photometric_norm)
        às DUAS imagens antes do matching. Modos combináveis por '+':
        'none' (default, passa direto), 'clahe', 'histmatch', 'gradient'.
        Não altera geometria (só intensidades), então as homografias/pontos
        continuam válidos em coordenadas de pixel."""
        modo = getattr(self, 'photometric_norm', 'none')
        if not modo or modo == 'none':
            return img_aerea, patch
        a = self._para_gray(img_aerea)
        p = self._para_gray(patch)
        if 'histmatch' in modo:
            a = self._match_histograms_cdf(a, p)
        if 'clahe' in modo:
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            a = clahe.apply(a)
            p = clahe.apply(p)
        if 'gradient' in modo:
            a = self._grad_mag(a)
            p = self._grad_mag(p)
        return a, p

    # ------------------------------------------------------------------
    # Busca de escala
    # ------------------------------------------------------------------

    def _busca_escala(self, img_aerea, lat, lon, angulo_graus,
                              escala_anterior, lat_real=None, lon_real=None):
        """
        Testa 3 escalas candidatas e retorna a que produz mais inliers
        (Seção 3.2.3 da dissertação).

        Returns:
            (patch_otimo, escala_otima, pts1_inliers, pts2_inliers,
             n_inliers, patch_transform, M_rot_inv, shape_orig)
        """
        passo = self.scale_search_step
        candidatas = [
            max(escala_anterior * (1.0 - passo), 0.01),
            max(escala_anterior, 0.01),
            max(escala_anterior * (1.0 + passo), 0.01),
        ]

        melhor = None
        for escala in candidatas:
            patch, T, M_rot_inv, shape_orig = self._preparar_patch_satelital(
                lat, lon, angulo_graus, escala, lat_real=lat_real, lon_real=lon_real
            )
            img_a_proc, patch_proc = self._preprocessar_para_match(img_aerea, patch)
            pts1, pts2, _, _, _, _, _ = self._obter_correspondencias(
                img_a_proc, patch_proc, usar_absoluto=True
            )
            if pts1 is None or len(pts1) < 4:
                continue
            _, mask = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0)
            if mask is None:
                continue
            n_inliers = int(np.sum(mask))
            if melhor is None or n_inliers > melhor[4]:
                pts1_inl = pts1[mask.ravel() == 1]
                pts2_inl = pts2[mask.ravel() == 1]
                melhor = (patch, escala, pts1_inl, pts2_inl,
                          n_inliers, T, M_rot_inv, shape_orig)

        if melhor is None:
            patch, T, M_rot_inv, shape_orig = self._preparar_patch_satelital(
                lat, lon, angulo_graus, escala_anterior, lat_real=lat_real, lon_real=lon_real
            )
            return (patch, escala_anterior, np.array([]), np.array([]),
                    0, T, M_rot_inv, shape_orig)

        return melhor

    # ------------------------------------------------------------------
    # Busca de ângulo
    # ------------------------------------------------------------------

    def _busca_angulo(self, img_aerea, lat, lon, angulo_base, escala,
                       lat_real=None, lon_real=None):
        """
        Testa candidatos de ÂNGULO (não só escala) e retorna o que produz
        mais inliers — mesmo padrão de _busca_escala, mas para o ângulo.

        Motivação (2026-07-08, achado da investigação do "runaway"): o
        ângulo usado pra rotacionar o patch satelital vinha sempre fixo do
        yaw acumulado da odometria (sem suavização), que fica justamente
        mais ruidoso em curvas (matriz essencial mal condicionada). Sem
        nenhuma busca própria de ângulo, um yaw ruidoso numa curva
        rotacionava o patch errado, derrubava os inliers abaixo do limiar,
        e como a única correção de yaw existente depende de um match já
        bem-sucedido, o erro nunca era corrigido — persistindo e compondo
        nos frames seguintes. Testar um bracket largo de ângulos ao redor
        do yaw estimado permite recuperar o match mesmo quando o yaw da
        odometria está bem errado.

        candidatos = angulo_base + offset, para offset uniformemente
        espaçado em ±self.angle_search_range_deg
        (self.angle_search_candidates valores, incluindo offset=0).

        Returns:
            (patch_otimo, angulo_otimo, pts1_inliers, pts2_inliers,
             n_inliers, patch_transform, M_rot_inv, shape_orig)
        """
        n = max(1, self.angle_search_candidates)
        alcance = self.angle_search_range_deg
        if n == 1:
            offsets = [0.0]
        else:
            offsets = list(np.linspace(-alcance, alcance, n))

        melhor = None
        for offset in offsets:
            angulo = angulo_base + offset
            patch, T, M_rot_inv, shape_orig = self._preparar_patch_satelital(
                lat, lon, angulo, escala, lat_real=lat_real, lon_real=lon_real
            )
            img_a_proc, patch_proc = self._preprocessar_para_match(img_aerea, patch)
            pts1, pts2, _, _, _, _, _ = self._obter_correspondencias(
                img_a_proc, patch_proc, usar_absoluto=True
            )
            if pts1 is None or len(pts1) < 4:
                continue
            _, mask = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0)
            if mask is None:
                continue
            n_inliers = int(np.sum(mask))
            if melhor is None or n_inliers > melhor[4]:
                pts1_inl = pts1[mask.ravel() == 1]
                pts2_inl = pts2[mask.ravel() == 1]
                melhor = (patch, angulo, pts1_inl, pts2_inl,
                          n_inliers, T, M_rot_inv, shape_orig)

        if melhor is None:
            patch, T, M_rot_inv, shape_orig = self._preparar_patch_satelital(
                lat, lon, angulo_base, escala, lat_real=lat_real, lon_real=lon_real
            )
            return (patch, angulo_base, np.array([]), np.array([]),
                    0, T, M_rot_inv, shape_orig)

        return melhor

    # ------------------------------------------------------------------
    # Estimação de posição e ângulo via homografia
    # ------------------------------------------------------------------

    def _estimar_posicao_pela_homografia(self, pts_aerea, pts_satelital,
                                          patch_transform, img_aerea_shape):
        """
        Pseudocódigo 1 da dissertação (Seção 3.1.5):
        Estima posição geográfica pelo centro da imagem aérea projetado
        no patch via homografia.
        Retorna (lat, lon) ou None em caso de falha.
        """
        if pts_aerea is None or len(pts_aerea) < 4:
            return None

        H, _ = cv.findHomography(pts_aerea, pts_satelital, cv.RANSAC, 5.0)
        if H is None:
            return None

        h, w = img_aerea_shape[:2]
        centro = np.array([[[w / 2.0, h / 2.0]]], dtype=np.float32)
        ponto_no_patch = cv.perspectiveTransform(centro, H)[0][0]

        if not (np.isfinite(ponto_no_patch[0]) and np.isfinite(ponto_no_patch[1])):
            return None

        lon_corr, lat_corr = patch_transform * (
            float(ponto_no_patch[0]), float(ponto_no_patch[1])
        )

        # Se o mapa está em CRS projetado, converter de volta para WGS84
        if self._map_transformer is not None:
            lon_corr, lat_corr = self._map_transformer.transform(
                lon_corr, lat_corr, direction='INVERSE'
            )

        if not (np.isfinite(lat_corr) and np.isfinite(lon_corr)):
            return None
        if not (-90.0 <= lat_corr <= 90.0 and -180.0 <= lon_corr <= 180.0):
            return None

        return lat_corr, lon_corr

    def _extrair_angulo_da_homografia(self, H):
        """
        Equação 3.1 da dissertação:
            theta = atan2(H[0,1], H[0,0]) * 180 / pi
        Retorna 0.0 se H for None.
        """
        if H is None:
            return 0.0
        return float(np.degrees(np.arctan2(H[0, 1], H[0, 0])))

    # ------------------------------------------------------------------
    # Correção principal
    # ------------------------------------------------------------------

    def _corrigir_posicao_pelo_mapa(self, curr_img, lat_est, lon_est,
                                     yaw_acumulado, i, lat_real=None, lon_real=None):
        """
        Módulo de Localização Absoluta (Seção 3.2.3 da dissertação).
        Retorna (lat, lon, yaw, escala, n_inliers, map_ok).

        (lat_real, lon_real): ver docstring de _recortar_roi_mapa — só usado
        quando roi_center_mode='real_bbox' (modo diagnóstico).
        """
        TARGET = _MAP_MATCH_TARGET_PX
        h, w = curr_img.shape[:2]
        img_aerea = cv.resize(curr_img, (TARGET, TARGET)) if (h != TARGET or w != TARGET) else curr_img

        # Etapa 1: busca de ÂNGULO (escala fixa em escala_atual) — encontra
        # um ângulo validado por inliers antes de refinar a escala. Ver
        # docstring de _busca_angulo para a motivação (fix do "runaway").
        (_, angulo_otimo, _, _,
         n_inliers_ang, _, _, _) = self._busca_angulo(
            img_aerea, lat_est, lon_est, yaw_acumulado, self.escala_atual,
            lat_real=lat_real, lon_real=lon_real
        )
        logger.debug(
            "Frame %d: busca de ângulo escolheu %.1f° (base yaw=%.1f°, "
            "offset=%.1f°, inliers=%d)",
            i, angulo_otimo, yaw_acumulado, angulo_otimo - yaw_acumulado, n_inliers_ang
        )

        # Etapa 2: busca de ESCALA, agora usando o ângulo validado da etapa 1
        # (em vez do yaw cru da odometria) como base de rotação do patch.
        (patch_otimo, escala_otima, pts1, pts2,
         n_inliers, patch_transform, M_rot_inv,
         shape_orig) = self._busca_escala(
            img_aerea, lat_est, lon_est, angulo_otimo, self.escala_atual,
            lat_real=lat_real, lon_real=lon_real
        )

        # Armazena sempre para debug, independente do resultado
        self._last_debug_data = (img_aerea, patch_otimo)

        if n_inliers < self.inlier_thr_position:
            return lat_est, lon_est, yaw_acumulado, self.escala_atual, n_inliers, False

        pos = self._estimar_posicao_pela_homografia(
            pts1, pts2, patch_transform, img_aerea.shape
        )
        if pos is None:
            return lat_est, lon_est, yaw_acumulado, self.escala_atual, n_inliers, False

        lat_corr, lon_corr = pos
        # angulo_otimo já é um ângulo validado por inliers (n_inliers >=
        # inlier_thr_position aqui) — substitui yaw_acumulado como base do
        # yaw retornado mesmo sem o refinamento fino da homografia abaixo.
        # Isso quebra o loop causal do "runaway": mesmo um match "só ok"
        # já puxa o yaw de volta pra perto do correto, em vez de deixá-lo
        # preso no valor ruidoso vindo da odometria.
        novo_yaw    = angulo_otimo % 360
        nova_escala = self.escala_atual

        if n_inliers >= self.inlier_thr_angle_scale:
            H_final, _ = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0)
            delta_ang = self._extrair_angulo_da_homografia(H_final)
            novo_yaw    = (angulo_otimo + delta_ang) % 360
            nova_escala = escala_otima

        self._last_map_match_data = (img_aerea, patch_otimo, pts1, pts2, n_inliers)
        return lat_corr, lon_corr, novo_yaw, nova_escala, n_inliers, True

    # ------------------------------------------------------------------
    # Visualização
    # ------------------------------------------------------------------

    def _visualizar_map_matching(self, i, lat_antes, lon_antes, lat_depois, lon_depois):
        """Janela OpenCV side-by-side: imagem aérea (esq.) vs patch satelital (dir.)."""
        if not hasattr(self, '_last_map_match_data'):
            return

        img_aerea, patch, pts1, pts2, n_inliers = self._last_map_match_data

        kp1 = [cv.KeyPoint(x=float(p[0]), y=float(p[1]), size=4) for p in pts1]
        kp2 = [cv.KeyPoint(x=float(p[0]), y=float(p[1]), size=4) for p in pts2]
        matches = [cv.DMatch(k, k, 0) for k in range(len(kp1))]

        img_vis = cv.drawMatches(
            img_aerea, kp1, patch, kp2, matches, None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        corr_dist = self.calcula_distancia_latlon(
            lat_antes, lon_antes, lat_depois, lon_depois
        )
        h_vis = img_vis.shape[0]
        cv.putText(img_vis,
                   f"Frame {i} | Inliers: {n_inliers} | Correcao: {corr_dist:.1f} m",
                   (10, 22), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv.putText(img_vis, "Imagem Aerea",
                   (10, h_vis - 8), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv.putText(img_vis, "Patch Satelital",
                   (img_aerea.shape[1] + 10, h_vis - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv.imshow("Map Matching — Correspondencias", img_vis)
        cv.waitKey(1)

    def _exibir_debug_patch(self, i, n_inliers, map_ok):
        """Janela de debug: imagem aérea e patch satelital lado a lado, sempre."""
        if not hasattr(self, '_last_debug_data'):
            return

        img_aerea, patch = self._last_debug_data

        def para_bgr(img, h_alvo):
            out = cv.cvtColor(img, cv.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
            if out.shape[0] != h_alvo:
                s = h_alvo / out.shape[0]
                out = cv.resize(out, (max(1, int(out.shape[1] * s)), h_alvo))
            return out

        h_ref = max(img_aerea.shape[0], patch.shape[0])
        img_a = para_bgr(img_aerea, h_ref)
        img_p = para_bgr(patch, h_ref)
        sep   = np.full((h_ref, 4, 3), 128, dtype=np.uint8)
        vis   = np.hstack([img_a, sep, img_p])

        status = "MAP OK" if map_ok else "SEM CORRESPONDENCIA"
        cor    = (0, 200, 0) if map_ok else (0, 0, 220)
        cv.putText(vis, f"Frame {i} | Inliers: {n_inliers} | {status}",
                   (10, 22), cv.FONT_HERSHEY_SIMPLEX, 0.65, cor, 2)
        cv.putText(vis, "Imagem Aerea",
                   (10, h_ref - 8), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv.putText(vis, "Patch Satelital (recortado + rotac.)",
                   (img_a.shape[1] + 14, h_ref - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv.imshow("Map Matching — Debug", vis)
        logger.info("Frame %d | Inliers: %d | %s | Pressione qualquer tecla na janela para continuar...",
                    i, n_inliers, status)
        cv.waitKey(0)
