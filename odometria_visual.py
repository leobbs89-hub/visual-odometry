# odometria_visual.py

"""
Módulo contendo a lógica principal para a Odometria Visual Monocular.

Este módulo define a classe `OdometriaVisual`, que encapsula todos os passos
necessários para estimar a trajetória de uma câmera a partir de uma sequência
de imagens, comparando-a com dados de GPS.
"""
import sys
import os
import time
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from pandas import read_csv, DataFrame
from pyproj import Geod
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
from geopy.distance import geodesic, distance


# ---------------------------------------------------------------------------
# Imports opcionais — redes neurais
# ---------------------------------------------------------------------------

# LightGlue / SuperPoint
caminho_lightglue = os.path.join(os.path.dirname(__file__), 'LightGlue')
if caminho_lightglue not in sys.path:
    sys.path.insert(0, caminho_lightglue)

try:
    import torch
    from lightglue.superpoint import SuperPoint
    from lightglue.lightglue import LightGlue
    from lightglue.utils import match_pair
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# LoFTR (via kornia)
try:
    import kornia
    from kornia.feature import LoFTR
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False

# MatchFormer
caminho_matchformer = os.path.join(os.path.dirname(__file__), 'MatchFormer')
if caminho_matchformer not in sys.path:
    sys.path.insert(0, caminho_matchformer)

try:
    from matchformer import MatchFormer
    MATCHFORMER_AVAILABLE = True
except ImportError:
    MATCHFORMER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Funções auxiliares (independentes de estado)
# ---------------------------------------------------------------------------

def estima_latlon(start_lat, start_lon, yaw, distancia):
    """
    Estima a coordenada geográfica final a partir de um ponto inicial,
    azimute (yaw) e distância.
    """
    g = Geod(ellps='WGS84')
    end_lon, end_lat, _ = g.fwd(start_lon, start_lat, yaw, distancia,
                                 return_back_azimuth=False)
    return end_lat, end_lon


def calcula_distancia_latlon(start_lat, start_lon, end_lat, end_lon):
    """
    Calcula a distância geodésica entre dois pontos (Lat/Lon) em metros.
    """
    g = Geod(ellps='WGS84')
    _, _, dist = g.inv(start_lon, start_lat, end_lon, end_lat)
    return dist


def criar_caminho_kml(lat_list, lon_list, file_path):
    """
    Cria um arquivo KML a partir de listas de coordenadas para visualização
    no Google Earth.
    """
    kml_template = '''<?xml version="1.0" encoding="UTF-8"?>
    <kml xmlns="http://www.opengis.net/kml/2.2">
    <Document>
        <name>{route_name}</name>
        <Style id="yellowLine"><LineStyle><color>7f00ffff</color><width>4</width></LineStyle></Style>
        <Placemark>
            <name>Trajetoria</name>
            <styleUrl>#yellowLine</styleUrl>
            <LineString><tessellate>1</tessellate><coordinates>{point_elements}</coordinates></LineString>
        </Placemark>
    </Document>
    </kml>'''

    point_elements = "".join(
        f'{lon},{lat},0 ' for lat, lon in zip(lat_list, lon_list)
    )
    xml_content = kml_template.format(
        route_name=os.path.basename(file_path),
        point_elements=point_elements.strip()
    )

    with open(file_path, 'w') as f:
        f.write(xml_content)
    print(f"Arquivo KML salvo em: {file_path}")


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class OdometriaVisual:
    """
    Encapsula o pipeline de odometria visual monocular.
    """

    def __init__(self, config):
        """
        Inicializa a classe com os parâmetros de configuração.

        Args:
            config (dict): Dicionário contendo todas as configurações.
                           Gerado por main.py a partir do config.yaml.
        """
        self.config = config
        self.detector_type = config['detector_type'].upper()

        # --- Resolve device UMA única vez para todos os detectores neurais ---
        # Opções via config.yaml:  device: auto | cpu | cuda
        self.device = self._resolver_device(config.get('device', 'auto'))

        # --- Map Matching ---
        self.use_map_matching = config.get('use_map_matching', False)
        if self.use_map_matching:
            self.map_match_interval = config.get('map_match_interval', 1)
            self.roi_size_m = config.get('roi_size_m', 1000)
            self._inicializar_mapa()

        # --- Parâmetros da câmera ---
        cam = config['camera_params']
        self.fx = cam['fx']
        self.fy = cam['fy']
        self.cx = cam['cx']
        self.cy = cam['cy']

        # --- Listas de estado ---
        self.imgs_list      = []
        self.lat_real_list  = []
        self.lon_real_list  = []
        self.height_list    = []
        self.yaw_real_list  = []
        self.gsd_list       = []

        self._inicializar_detector_matcher()

    # ------------------------------------------------------------------
    # Helpers de inicialização
    # ------------------------------------------------------------------

    @staticmethod
    def _resolver_device(preferencia: str):
        """
        Resolve o dispositivo PyTorch com base na preferência do config.yaml.

        Valores aceitos:
            'auto' — usa CUDA se disponível, senão CPU (padrão)
            'cpu'  — força CPU independentemente da GPU disponível
            'cuda' — força CUDA; lança erro se não houver GPU

        Returns:
            torch.device
        """
        if not TORCH_AVAILABLE:
            # Sem PyTorch instalado; retorna None e os detectores neurais
            # emitirão ImportError descritivo ao ser selecionados.
            return None

        pref = str(preferencia).lower().strip()

        if pref == 'cpu':
            dev = torch.device('cpu')
        elif pref == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "device: cuda foi solicitado no config.yaml, mas nenhuma "
                    "GPU CUDA foi encontrada. Altere para 'auto' ou 'cpu'."
                )
            dev = torch.device('cuda')
        else:  # 'auto'
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"[INFO] Device para redes neurais: {dev}")
        return dev

    def _inicializar_mapa(self):
        """Abre o arquivo GeoTIFF de forma lazy (sem carregar tudo na RAM)."""
        map_path = self.config['paths']['map_tif_path']
        print(f"Carregando mapa base para Map Matching: {map_path}")
        self.map_dataset = rasterio.open(map_path)

    def _inicializar_detector_matcher(self):
        """
        Instancia o detector e o matcher de acordo com detector_type.
        Usa self.device (já resolvido) para todos os detectores neurais.
        """
        print(f"Inicializando detector: {self.detector_type}")

        if self.detector_type == 'ORB':
            params = self.config['detector_params']['orb']
            self.detector = cv.ORB_create(**params)
            self.matcher  = cv.DescriptorMatcher_create(
                cv.DescriptorMatcher_BRUTEFORCE_HAMMING
            )

        elif self.detector_type == 'AKAZE':
            params = self.config['detector_params']['akaze']
            self.detector = cv.AKAZE_create(**params)
            self.matcher  = cv.DescriptorMatcher_create(
                cv.DescriptorMatcher_BRUTEFORCE_HAMMING
            )

        elif self.detector_type == 'SUPERPOINT':
            if not TORCH_AVAILABLE:
                raise ImportError(
                    "PyTorch e LightGlue são necessários para SuperPoint.\n"
                    "Instale com: pip install -r requirements-neural.txt\n"
                    "  e clone:  git clone https://github.com/cvg/LightGlue.git\n"
                    "Funciona em CPU — GPU não é obrigatória."
                )
            params = self.config['detector_params'].get('superpoint', {})
            self.detector = SuperPoint(**params).eval().to(self.device)
            self.matcher  = LightGlue(features='superpoint').eval().to(self.device)
            print(f"  SuperPoint + LightGlue prontos em [{self.device}]")

        elif self.detector_type == 'LOFTR':
            if not KORNIA_AVAILABLE:
                raise ImportError(
                    "Kornia e PyTorch são necessários para LoFTR.\n"
                    "Instale com: pip install -r requirements-neural.txt\n"
                    "Funciona em CPU — GPU não é obrigatória."
                )
            params = self.config['detector_params'].get('loftr', {})
            self.matcher = LoFTR(**params).eval().to(self.device)
            print(f"  LoFTR pronto em [{self.device}]")

        elif self.detector_type == 'MATCHFORMER':
            if not MATCHFORMER_AVAILABLE:
                raise ImportError(
                    "A biblioteca MatchFormer não foi encontrada.\n"
                    "Clone com: git clone https://github.com/gaopengcuhk/MatchFormer.git\n"
                    "Instale dependências: pip install -r requirements-neural.txt\n"
                    "Funciona em CPU — GPU não é obrigatória."
                )
            params = self.config['detector_params'].get('matchformer', {})
            self.matcher = MatchFormer(**params).eval().to(self.device)
            print(f"  MatchFormer pronto em [{self.device}]")

        else:
            raise ValueError(
                f"Detector desconhecido: '{self.detector_type}'. "
                "Escolha: ORB | AKAZE | SUPERPOINT | LOFTR | MATCHFORMER"
            )

    # ------------------------------------------------------------------
    # Carregamento de dados
    # ------------------------------------------------------------------

    def _carregar_dados(self):
        """Carrega as imagens e os dados de ground truth (GPS)."""

        image_path = self.config['paths']['image_path']
        for file in sorted(os.listdir(image_path)):
            img_path = os.path.join(image_path, file)
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if img is not None:
                self.imgs_list.append(img)

        if not self.imgs_list:
            raise FileNotFoundError(
                f"Nenhuma imagem encontrada em: {image_path}"
            )
        print(f"{len(self.imgs_list)} imagens carregadas.")

        # Ground truth
        gt_path = self.config['paths']['ground_truth_path']
        df = read_csv(gt_path, sep=',')
        self.lat_real_list  = np.array(df["Lat"])
        self.lon_real_list  = np.array(df["Long"])
        self.height_list    = np.array(df["Altura"])
        self.yaw_real_list  = np.array(df["Proa"])

        # GSD — Ground Sampling Distance (m/pixel)
        self.gsd_list = (1.0 * self.height_list) / self.fx
        print("Dados de GPS e GSD carregados.")

        # Projeção UTM local
        self._R_EARTH = 6_371_000
        self.origin_lat = self.lat_real_list[0]
        self.origin_lon = self.lon_real_list[0]
        self._cos_origin_lat = np.cos(np.radians(self.origin_lat))

        zone = int((self.lon_real_list[0] + 180) / 6) + 1
        hemisphere = 'north' if self.lat_real_list[0] >= 0 else 'south'
        epsg = 32600 + zone if hemisphere == 'north' else 32700 + zone

        self._transformer = Transformer.from_crs(
            "EPSG:4326", f"EPSG:{epsg}", always_xy=True
        )
        self._origin_x, self._origin_y = self._transformer.transform(
            self.lon_real_list[0], self.lat_real_list[0]
        )

    # ------------------------------------------------------------------
    # Map Matching
    # ------------------------------------------------------------------

    def _recortar_roi_mapa(self, lat, lon):
        """
        Recorta um patch do GeoTIFF centrado na posição estimada (lat/lon).
        """
        half = self.roi_size_m / 2.0
        p_min_lat = distance(meters=half).destination((lat, lon), bearing=180)
        p_min_lon = distance(meters=half).destination((lat, lon), bearing=270)
        p_max_lon = distance(meters=half).destination((lat, lon), bearing=90)
        p_max_lat = distance(meters=half).destination((lat, lon), bearing=0)

        minx = p_min_lon.longitude
        miny = p_min_lat.latitude
        maxx = p_max_lon.longitude
        maxy = p_max_lat.latitude

        window = from_bounds(minx, miny, maxx, maxy, self.map_dataset.transform)
        patch = self.map_dataset.read(1, window=window)
        patch_transform = self.map_dataset.window_transform(window)

        patch_8u = cv.normalize(patch, None, alpha=0, beta=255,
                                norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        return patch_8u, patch_transform

    def _corrigir_posicao_pelo_mapa(self, curr_img, lat_est, lon_est):
        """
        Tenta corrigir o drift via homografia câmera ↔ ortomosaico.
        Retorna (lat, lon) possivelmente corrigidos.
        """
        patch_mapa, patch_transform = self._recortar_roi_mapa(lat_est, lon_est)

        pts_cam, pts_mapa, _, _, num_matches = self._obter_correspondencias(
            curr_img, patch_mapa
        )

        if pts_cam is None or len(pts_cam) < 15:
            print(f"Map Matching: poucos matches ({num_matches}). "
                  "Mantendo posição atual.")
            return lat_est, lon_est

        H, mask = cv.findHomography(pts_cam, pts_mapa, cv.RANSAC, 5.0)
        if H is None:
            print("Map Matching: homografia não encontrada. Mantendo posição.")
            return lat_est, lon_est

        num_inliers = int(np.sum(mask))
        if num_inliers < 10:
            print(f"Map Matching: poucos inliers ({num_inliers}). "
                  "Mantendo posição.")
            return lat_est, lon_est

        h_cam, w_cam = curr_img.shape
        centro_cam = np.array([[[w_cam / 2.0, h_cam / 2.0]]], dtype=np.float32)
        centro_no_patch = cv.perspectiveTransform(centro_cam, H)[0][0]

        lon_corrigida, lat_corrigida = patch_transform * (
            centro_no_patch[0], centro_no_patch[1]
        )

        # Filtro complementar: 30% mapa, 70% odometria
        alfa = 0.3
        lat_final = lat_est * (1.0 - alfa) + lat_corrigida * alfa
        lon_final = lon_est * (1.0 - alfa) + lon_corrigida * alfa

        return lat_final, lon_final

    # ------------------------------------------------------------------
    # Correspondências de features
    # ------------------------------------------------------------------

    def _obter_correspondencias(self, img1, img2):
        """
        Detecta features e retorna pontos correspondentes entre img1 e img2.

        Returns:
            pts1, pts2        : arrays Nx2 float32
            kpt_count1/2      : número de keypoints detectados
            match_count       : número de matches aceitos
        """
        if self.detector_type in ('ORB', 'AKAZE'):
            kp1, des1 = self.detector.detectAndCompute(img1, None)
            kp2, des2 = self.detector.detectAndCompute(img2, None)

            if des1 is None or des2 is None:
                return None, None, len(kp1), len(kp2), 0

            nn_matches = self.matcher.knnMatch(des1, des2, k=2)
            ratio = self.config['matcher_params']['nn_match_ratio']
            good = [m for m, n in nn_matches if m.distance < ratio * n.distance]

            pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
            return pts1, pts2, len(kp1), len(kp2), len(good)

        elif self.detector_type == 'SUPERPOINT':
            t1 = torch.from_numpy(img1).float().to(self.device).unsqueeze(0).unsqueeze(0) / 255.
            t2 = torch.from_numpy(img2).float().to(self.device).unsqueeze(0).unsqueeze(0) / 255.

            feats1, feats2, matches01 = match_pair(
                self.detector, self.matcher, t1, t2
            )
            kp1     = feats1['keypoints']
            kp2     = feats2['keypoints']
            matches = matches01['matches']

            pts1 = kp1[matches[:, 0]].cpu().numpy()
            pts2 = kp2[matches[:, 1]].cpu().numpy()
            return pts1, pts2, len(kp1), len(kp2), len(matches)

        elif self.detector_type in ('LOFTR', 'MATCHFORMER'):
            t1 = torch.from_numpy(img1).float().to(self.device).unsqueeze(0).unsqueeze(0) / 255.
            t2 = torch.from_numpy(img2).float().to(self.device).unsqueeze(0).unsqueeze(0) / 255.

            with torch.inference_mode():
                corr = self.matcher({'image0': t1, 'image1': t2})

            pts1 = corr['keypoints0'].cpu().numpy()
            pts2 = corr['keypoints1'].cpu().numpy()
            n = len(pts1)
            return pts1, pts2, n, n, n

        return None, None, 0, 0, 0

    # ------------------------------------------------------------------
    # Helpers de cálculo
    # ------------------------------------------------------------------

    @staticmethod
    def _calcula_deslocamento_escala(pts1, pts2, gsd):
        """
        Estima a distância percorrida (metros) a partir do deslocamento
        mediano dos pontos correspondentes (inliers) e do GSD.
        """
        deslocamentos = [np.linalg.norm(p1 - p2) * gsd
                         for p1, p2 in zip(pts1, pts2)]
        return float(np.median(deslocamentos)) if deslocamentos else 0.0

    def _latlon_to_xy(self, lat, lon):
        """
        Converte (lat, lon) para coordenadas XY locais em metros,
        relativas à origem da trajetória, via projeção UTM.
        """
        x, y = self._transformer.transform(lon, lat)
        return x - self._origin_x, y - self._origin_y

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------

    def executar(self):
        """Executa o pipeline completo de odometria visual."""
        self._carregar_dados()

        # --- Inicialização ---
        lat_est_list = [self.lat_real_list[0]]
        lon_est_list = [self.lon_real_list[0]]
        yaw_acumulado_est = float(self.yaw_real_list[0])
        dist_real_total = 0.0
        dist_est_total  = 0.0

        x0_est,  y0_est  = self._latlon_to_xy(lat_est_list[0],       lon_est_list[0])
        x0_real, y0_real = self._latlon_to_xy(self.lat_real_list[0], self.lon_real_list[0])
        x_est_list,  y_est_list  = [x0_est],  [y0_est]
        x_real_list, y_real_list = [x0_real], [y0_real]

        # --- Plotagem em tempo real ---
        show_plot = self.config['display']['show_plot']
        if show_plot:
            plt.ion()
            fig, ax = plt.subplots()
            ax.set_title("Trajetória Estimada")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            line_est,  = ax.plot(x_est_list,  y_est_list,  'r-', label='Estimado')
            line_real, = ax.plot(x_real_list, y_real_list, 'b-', label='Real')
            ax.legend()

        print("\nIniciando o loop de odometria...")
        colunas = ["IMG","KPT1","KPT2","MATCHES","INLIERS",
                   "DIST_REAL","DIST_EST","ERRO_ACUM(m)","TEMPO(s)"]
        Resultados = DataFrame(columns=colunas)
        print("IMG\tKPT1\tKPT2\tMATCHES\tINLIERS\t\tDIST_REAL\tDIST_EST\tERRO_ACUM(m)\tTEMPO(s)")

        # --- Loop principal ---
        num_frames = len(self.imgs_list)
        for i in range(num_frames - 1):
            t0 = time.time()

            prev_img = self.imgs_list[i]
            curr_img = self.imgs_list[i + 1]

            # 1. Correspondências
            pts1, pts2, kpt1, kpt2, n_matches = self._obter_correspondencias(
                prev_img, curr_img
            )
            if pts1 is None or len(pts1) < 8:
                print(f"Frame {i}: poucas correspondências. Pulando.")
                continue

            # 2. Matriz Essencial → pose
            E, mask_e = cv.findEssentialMat(
                pts1, pts2, focal=self.fx, pp=(self.cx, self.cy),
                method=cv.RANSAC, prob=0.999, threshold=1.0
            )
            inl1 = pts1[mask_e.ravel() == 1]
            inl2 = pts2[mask_e.ravel() == 1]
            n_inliers = len(inl1)

            if n_inliers < 5:
                print(f"Frame {i}: poucos inliers após RANSAC. Pulando.")
                continue

            _, R, t, _ = cv.recoverPose(
                E, inl1, inl2, focal=self.fx, pp=(self.cx, self.cy)
            )

            # 3. Escala via GSD
            dist_est_atual = self._calcula_deslocamento_escala(
                inl1, inl2, self.gsd_list[i]
            )
            dist_est_total += dist_est_atual

            # 4. Yaw acumulado
            euler = Rotation.from_matrix(R).as_euler('zyx', degrees=True)
            yaw_delta = euler[0]
            if i <= 2 and abs(yaw_delta) > 45:
                yaw_delta = 0.0
            yaw_acumulado_est = (yaw_acumulado_est - yaw_delta) % 360

            # 5. Nova posição estimada
            est_lat, est_lon = estima_latlon(
                lat_est_list[i], lon_est_list[i],
                yaw_acumulado_est, dist_est_atual
            )

            # 6. Correção por map matching (opcional)
            if self.use_map_matching and (i % self.map_match_interval == 0):
                print(f"Frame {i}: executando Map Matching...")
                lat_c, lon_c = self._corrigir_posicao_pelo_mapa(
                    curr_img, est_lat, est_lon
                )
                if abs(lat_c - est_lat) > 1e-8:
                    print(f"  [MAP MATCHING] Posição corrigida no frame {i + 1}")
                est_lat, est_lon = lat_c, lon_c

            lat_est_list.append(est_lat)
            lon_est_list.append(est_lon)

            # 7. Métricas
            dist_real_atual = calcula_distancia_latlon(
                self.lat_real_list[i],     self.lon_real_list[i],
                self.lat_real_list[i + 1], self.lon_real_list[i + 1]
            )
            dist_real_total += dist_real_atual
            erro_acum = calcula_distancia_latlon(
                self.lat_real_list[i + 1], self.lon_real_list[i + 1],
                est_lat, est_lon
            )

            elapsed = time.time() - t0
            print(f'{i}\t{kpt1}\t{kpt2}\t{n_matches}\t{n_inliers}\t\t'
                  f'{dist_real_atual:8.2f}\t{dist_est_atual:8.2f}\t'
                  f'{erro_acum:8.2f}\t\t{elapsed:.2f}')
            Resultados.loc[i] = [i, kpt1, kpt2, n_matches, n_inliers,
                                  dist_real_atual, dist_est_atual,
                                  erro_acum, elapsed]

            # 8. Plotagem
            x_r, y_r = self._latlon_to_xy(
                self.lat_real_list[i + 1], self.lon_real_list[i + 1]
            )
            x_real_list.append(x_r)
            y_real_list.append(y_r)

            x_e, y_e = self._latlon_to_xy(lat_est_list[-1], lon_est_list[-1])
            x_est_list.append(x_e)
            y_est_list.append(y_e)

            if show_plot:
                line_est.set_data(x_est_list, y_est_list)
                line_real.set_data(x_real_list, y_real_list)
                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.001)

            if self.config['display']['show_images']:
                self._exibir_correspondencias(prev_img, curr_img, inl1, inl2)
                if cv.waitKey(10) & 0xFF == ord('q'):
                    break

        # --- Finalização ---
        if self.config['display']['show_images']:
            cv.destroyAllWindows()
        if show_plot:
            plt.ioff()
            plt.show()

        output_dir = self.config['paths']['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        Resultados.to_csv(
            os.path.join(output_dir, f'resultados_{self.detector_type}.csv'),
            index=False
        )
        criar_caminho_kml(
            self.lat_real_list, self.lon_real_list,
            os.path.join(output_dir, 'trajetoria_real.kml')
        )
        criar_caminho_kml(
            lat_est_list, lon_est_list,
            os.path.join(output_dir,
                         f'trajetoria_estimada_{self.detector_type}.kml')
        )
        print("\nProcesso de odometria finalizado.")

    def _exibir_correspondencias(self, prev_img, curr_img, inl1, inl2,
                                  max_draw=150):
        """Desenha correspondências inliers e exibe em janela OpenCV."""
        n = len(inl1)
        idx = (np.linspace(0, n - 1, max_draw, dtype=int)
               if n > max_draw else range(n))

        kp1 = [cv.KeyPoint(x=float(inl1[j][0]), y=float(inl1[j][1]), size=2)
               for j in idx]
        kp2 = [cv.KeyPoint(x=float(inl2[j][0]), y=float(inl2[j][1]), size=2)
               for j in idx]
        matches = [cv.DMatch(k, k, 0) for k in range(len(idx))]

        img_display = cv.drawMatches(
            prev_img, kp1, curr_img, kp2, matches, None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        scale = 1200 / img_display.shape[1]
        if scale < 1:
            img_display = cv.resize(img_display, (0, 0), fx=scale, fy=scale)

        cv.imshow(
            f'Correspondencias Inliers — {self.detector_type}', img_display
        )
