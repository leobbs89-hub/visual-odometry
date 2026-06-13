# map_matching.py
"""
Mixin com todo o módulo de Localização Absoluta (map matching) baseado na
dissertação de Bruno Dantas (ITA 2023, Seção 3.2.3).

A classe MapMatchingMixin é herdada por OdometriaVisual e acessa atributos
da classe principal via self (self.config, self.map_dataset, etc.).
"""

import numpy as np
import cv2 as cv
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import Affine
from pyproj import Transformer
from geopy.distance import distance


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
        self.roi_size_m, self.escala_atual, self.scale_search_step,
        self.inlier_thr_position, self.inlier_thr_angle_scale,
        self.fx  (para _inicializar_escala)

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
        print(f"Carregando mapa base para Map Matching: {map_path}")
        self.map_dataset = rasterio.open(map_path)
        map_crs = self.map_dataset.crs
        if map_crs and not map_crs.is_geographic:
            self._map_transformer = Transformer.from_crs(
                "EPSG:4326", map_crs, always_xy=True
            )
            print(f"[MAP] CRS do GeoTIFF: {map_crs.to_string()} (projetado — reproj ativado)")
        else:
            self._map_transformer = None
            print(f"[MAP] CRS do GeoTIFF: {map_crs} (geográfico — sem reproj)")

    def _inicializar_escala(self, altura_inicial):
        """Estima escala inicial entre imagem aérea e mapa satelital."""
        pixel_size = abs(self.map_dataset.transform.a)
        if self.map_dataset.crs and self.map_dataset.crs.is_geographic:
            res_mapa_m = pixel_size * np.radians(1) * 6_371_000
        else:
            res_mapa_m = pixel_size
        gsd_voo = altura_inicial / self.fx
        self.escala_atual = max(gsd_voo / res_mapa_m, 0.01)
        self._escala_inicializada = True
        print(f"[MAP] Escala inicial: {self.escala_atual:.4f} "
              f"(GSD={gsd_voo:.3f}m/px, res_mapa={res_mapa_m:.3f}m/px)")

    # ------------------------------------------------------------------
    # Preparação do patch satelital
    # ------------------------------------------------------------------

    def _recortar_roi_mapa(self, lat, lon, margem_extra=1.5):
        """
        Recorta um patch do GeoTIFF centrado na posição estimada (lat/lon).
        O parâmetro margem_extra amplia a ROI para acomodar rotação posterior.
        """
        half = (self.roi_size_m * margem_extra) / 2.0
        p_min_lat = distance(meters=half).destination((lat, lon), bearing=180)
        p_min_lon = distance(meters=half).destination((lat, lon), bearing=270)
        p_max_lon = distance(meters=half).destination((lat, lon), bearing=90)
        p_max_lat = distance(meters=half).destination((lat, lon), bearing=0)

        minx = p_min_lon.longitude
        miny = p_min_lat.latitude
        maxx = p_max_lon.longitude
        maxy = p_max_lat.latitude

        if self._map_transformer is not None:
            minx, miny = self._map_transformer.transform(minx, miny)
            maxx, maxy = self._map_transformer.transform(maxx, maxy)

        window = from_bounds(minx, miny, maxx, maxy, self.map_dataset.transform)
        patch = self.map_dataset.read(1, window=window)
        patch_transform = self.map_dataset.window_transform(window)

        if patch.size == 0:
            return np.zeros((1, 1), dtype=np.uint8), patch_transform

        patch_8u = cv.normalize(patch, None, alpha=0, beta=255,
                                norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        if patch_8u is None:
            patch_8u = np.zeros_like(patch, dtype=np.uint8)
        return patch_8u, patch_transform

    def _preparar_patch_satelital(self, lat, lon, angulo_graus, escala):
        """
        Prepara patch satelital para comparação (Seção 3.2.3 da dissertação):
        1. Recorta patch maior com margem (_recortar_roi_mapa)
        2. Redimensiona proporcionalmente se maior que TARGET
        3. Rotaciona pelo angulo_graus (_rotacionar_imagem)
        4. Aplica escala e crop central para TARGET×TARGET

        Retorna: (patch_final, T_final, M_rot_inv, shape_antes_rot)
        onde T_final mapeia pixels do patch_final para coordenadas geo.
        """
        TARGET = _MAP_MATCH_TARGET_PX
        escala = max(escala, 0.01)

        # 1. Recortar com margem
        patch_big, T_big = self._recortar_roi_mapa(lat, lon, margem_extra=1.5)

        # 2. Redimensionar se maior que TARGET
        h_big, w_big = patch_big.shape[:2]
        max_dim = max(h_big, w_big)
        if max_dim > TARGET:
            scale_r = TARGET / max_dim
            new_w = max(1, int(w_big * scale_r))
            new_h = max(1, int(h_big * scale_r))
            patch_resized = cv.resize(patch_big, (new_w, new_h))
            T_resized = Affine(
                T_big.a / scale_r, T_big.b / scale_r, T_big.c,
                T_big.d / scale_r, T_big.e / scale_r, T_big.f
            )
        else:
            patch_resized = patch_big
            T_resized = T_big

        shape_antes_rot = patch_resized.shape

        # 3. Rotacionar
        patch_rot, M_rot_inv = _rotacionar_imagem(patch_resized, angulo_graus)

        # 4. Aplicar escala e crop central para TARGET×TARGET
        h_rot, w_rot = patch_rot.shape[:2]
        new_w_s = max(1, int(w_rot * escala))
        new_h_s = max(1, int(h_rot * escala))
        patch_scaled = cv.resize(patch_rot, (new_w_s, new_h_s))

        h_s, w_s = patch_scaled.shape[:2]
        y1 = max(0, (h_s - TARGET) // 2)
        x1 = max(0, (w_s - TARGET) // 2)
        y2 = min(h_s, y1 + TARGET)
        x2 = min(w_s, x1 + TARGET)
        patch_final = patch_scaled[y1:y2, x1:x2]

        # Compor transformação inversa completa: patch_final → geo
        # M_cs: crop+escala inverso (patch_final → patch_rot)
        M_cs = Affine(1.0 / escala, 0.0, x1 / escala,
                      0.0, 1.0 / escala, y1 / escala)
        # M_rot_affine: rotação inversa (patch_rot → patch_resized)
        M_rot_affine = Affine(
            M_rot_inv[0, 0], M_rot_inv[0, 1], M_rot_inv[0, 2],
            M_rot_inv[1, 0], M_rot_inv[1, 1], M_rot_inv[1, 2]
        )
        # T_final: patch_final → geo
        T_final = T_resized * M_rot_affine * M_cs

        return patch_final, T_final, M_rot_inv, shape_antes_rot

    # ------------------------------------------------------------------
    # Busca de escala
    # ------------------------------------------------------------------

    def _busca_escala(self, img_aerea, lat, lon, angulo_graus,
                              escala_anterior):
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
                lat, lon, angulo_graus, escala
            )
            pts1, pts2, _, _, _, _ = self._obter_correspondencias(
                img_aerea, patch, usar_absoluto=True
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
                lat, lon, angulo_graus, escala_anterior
            )
            return (patch, escala_anterior, np.array([]), np.array([]),
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
                                     yaw_acumulado, i):
        """
        Módulo de Localização Absoluta (Seção 3.2.3 da dissertação).
        Retorna (lat, lon, yaw, escala, n_inliers, map_ok).
        """
        TARGET = _MAP_MATCH_TARGET_PX
        h, w = curr_img.shape[:2]
        img_aerea = cv.resize(curr_img, (TARGET, TARGET)) if (h != TARGET or w != TARGET) else curr_img

        (patch_otimo, escala_otima, pts1, pts2,
         n_inliers, patch_transform, M_rot_inv,
         shape_orig) = self._busca_escala(
            img_aerea, lat_est, lon_est, yaw_acumulado, self.escala_atual
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
        novo_yaw  = yaw_acumulado
        nova_escala = self.escala_atual

        if n_inliers >= self.inlier_thr_angle_scale:
            H_final, _ = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0)
            delta_ang = self._extrair_angulo_da_homografia(H_final)
            novo_yaw    = (yaw_acumulado + delta_ang) % 360
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
        print(f"[DEBUG] Frame {i} | Inliers: {n_inliers} | {status}"
              " | Pressione qualquer tecla na janela para continuar...")
        cv.waitKey(0)
