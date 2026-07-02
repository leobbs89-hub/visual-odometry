# odometria_visual.py
"""
Módulo contendo a lógica principal para a Odometria Visual Monocular.

Este módulo define a classe `OdometriaVisual`, que encapsula todos os passos
necessários para estimar a trajetória de uma câmera a partir de uma sequência
de imagens, comparando-a com dados de GPS.

Módulos complementares:
    map_matching.py — Localização Absoluta via correspondência com mapa satelital
    utils.py              — Funções utilitárias puras (KML, etc.)
"""

import os
import time
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from pandas import read_csv, DataFrame
from pyproj import Geod, Transformer

from utils import criar_caminho_kml, mach_to_kml_color, _KML_COLOR_REAL
from map_matching import MapMatchingMixin, _INLIER_THRESHOLDS, _DEFAULT_INLIER_THRESHOLDS
from detectors import criar_detector, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class OdometriaVisual(MapMatchingMixin):
    """
    Encapsula o pipeline de odometria visual monocular.

    O módulo de Localização Absoluta (map matching) é herdado de
    MapMatchingMixin e ativado via config['use_map_matching'].
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
        self.absolute_detector_type = config.get(
            'absolute_detector_type', self.detector_type
        ).upper()

        # --- Resolve device UMA única vez para todos os detectores neurais ---
        self.device = self._resolver_device(config.get('device', 'auto'))

        # --- Map Matching ---
        self.use_map_matching = config.get('use_map_matching', False)
        if self.use_map_matching:
            self.map_match_interval = config.get('map_match_interval', 1)
            self.roi_size_m         = config.get('roi_size_m', 1000)
            self.scale_search_step  = config.get('scale_search_step', 0.05)

            mm_params = config.get('map_matching_params', {})
            default_thr = _INLIER_THRESHOLDS.get(
                self.absolute_detector_type, _DEFAULT_INLIER_THRESHOLDS
            )
            self.inlier_thr_position    = mm_params.get('inlier_thr_position',    default_thr[0])
            self.inlier_thr_angle_scale = mm_params.get('inlier_thr_angle_scale', default_thr[1])

            self.escala_atual        = 1.0
            self._escala_inicializada = False

            self._inicializar_mapa()
            self._inicializar_detector_absoluto()

        # --- Parâmetros da câmera ---
        cam = config['camera_params']
        self.fx = cam['fx']
        self.fy = cam['fy']
        self.cx = cam['cx']
        self.cy = cam['cy']

        # --- Listas de estado ---
        self.imgs_list     = []
        self.lat_real_list = []
        self.lon_real_list = []
        self.height_list   = []
        self.yaw_real_list = []
        self.gsd_list      = []

        self.geod = Geod(ellps='WGS84')

        # --- Filtro de yaw (ver executar(), passo 4) ---
        self._yaw_max_initial_frames  = config.get('yaw_filter_max_initial_frames', 2)
        self._yaw_max_initial_deg     = config.get('yaw_filter_max_initial_yaw_deg', 45)
        self._yaw_min_inlier_ratio    = config.get('yaw_filter_min_inlier_ratio', 0.10)

        self._inicializar_detector_odometria()

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
        """
        if not TORCH_AVAILABLE:
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

    def _inicializar_detector_odometria(self):
        """Instancia o FeatureDetector do módulo de odometria."""
        self.detector_odometria = criar_detector(
            self.detector_type, self.config['detector_params'],
            self.config['matcher_params'], self.device, papel='odometria'
        )

    def _inicializar_detector_absoluto(self):
        """Instancia o FeatureDetector do módulo de localização absoluta."""
        self.detector_absoluto = criar_detector(
            self.absolute_detector_type, self.config['detector_params'],
            self.config['matcher_params'], self.device, papel='absoluto'
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
            raise FileNotFoundError(f"Nenhuma imagem encontrada em: {image_path}")
        print(f"{len(self.imgs_list)} imagens carregadas.")

        gt_path = self.config['paths']['ground_truth_path']
        df = read_csv(gt_path, sep=',')
        self.lat_real_list = np.array(df["Lat"])
        self.lon_real_list = np.array(df["Long"])
        self.height_list   = np.array(df["Altura"])
        self.yaw_real_list = np.array(df["Proa"])

        # GSD — Ground Sampling Distance (m/pixel)
        self.gsd_list = (1.0 * self.height_list) / self.fx
        print("Dados de GPS e GSD carregados.")

        # Projeção UTM local
        self._R_EARTH      = 6_371_000
        self.origin_lat    = self.lat_real_list[0]
        self.origin_lon    = self.lon_real_list[0]
        self._cos_origin_lat = np.cos(np.radians(self.origin_lat))

        zone       = int((self.lon_real_list[0] + 180) / 6) + 1
        hemisphere = 'north' if self.lat_real_list[0] >= 0 else 'south'
        epsg       = 32600 + zone if hemisphere == 'north' else 32700 + zone

        self._transformer = Transformer.from_crs(
            "EPSG:4326", f"EPSG:{epsg}", always_xy=True
        )
        self._origin_x, self._origin_y = self._transformer.transform(
            self.lon_real_list[0], self.lat_real_list[0]
        )

        if self.use_map_matching:
            self._inicializar_escala(self.height_list[0])

    # ------------------------------------------------------------------
    # Correspondências de features
    # ------------------------------------------------------------------

    def _obter_correspondencias(self, img1, img2, prev_features=None,
                                 usar_absoluto=False):
        """
        Detecta features e retorna pontos correspondentes entre img1 e img2.

        Args:
            usar_absoluto: se True usa detector_absoluto;
                           se False usa detector_odometria.

        Returns:
            pts1, pts2, kpt_count1, kpt_count2, match_count, curr_features
        """
        det = self.detector_absoluto if usar_absoluto else self.detector_odometria
        r = det.match(img1, img2, prev_state=prev_features)
        return r.pts1, r.pts2, r.kpt_count1, r.kpt_count2, r.match_count, r.state

    # ------------------------------------------------------------------
    # Helpers de cálculo geográfico
    # ------------------------------------------------------------------

    def estima_latlon(self, start_lat, start_lon, yaw, distancia):
        """Estima coordenada final a partir de ponto inicial, azimute e distância."""
        end_lon, end_lat, _ = self.geod.fwd(
            start_lon, start_lat, yaw, distancia, return_back_azimuth=False
        )
        return end_lat, end_lon

    def calcula_distancia_latlon(self, start_lat, start_lon, end_lat, end_lon):
        """Calcula a distância geodésica entre dois pontos (Lat/Lon) em metros."""
        _, _, dist = self.geod.inv(start_lon, start_lat, end_lon, end_lat)
        return dist

    @staticmethod
    def _calcula_deslocamento_escala(pts1, pts2, gsd):
        """Estima distância percorrida (metros) pelo deslocamento mediano dos inliers."""
        deslocamentos = [np.linalg.norm(p1 - p2) * gsd for p1, p2 in zip(pts1, pts2)]
        return float(np.median(deslocamentos)) if deslocamentos else 0.0

    def _latlon_to_xy(self, lat, lon):
        """Converte (lat, lon) para coordenadas XY locais em metros via projeção UTM."""
        x, y = self._transformer.transform(lon, lat)
        return x - self._origin_x, y - self._origin_y

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------

    def _calculate_metrics(self, i, est_lat, est_lon):
        """Calcula as métricas de distância real e erro acumulado."""
        dist_real_atual = self.calcula_distancia_latlon(
            self.lat_real_list[i],     self.lon_real_list[i],
            self.lat_real_list[i + 1], self.lon_real_list[i + 1]
        )
        erro_acum = self.calcula_distancia_latlon(
            self.lat_real_list[i + 1], self.lon_real_list[i + 1],
            est_lat, est_lon
        )
        return dist_real_atual, erro_acum

    def _process_frame_pose(self, pts1, pts2):
        """Calcula a pose (R, t) e inliers a partir das correspondências."""
        E, mask_e = cv.findEssentialMat(
            pts1, pts2, focal=self.fx, pp=(self.cx, self.cy),
            method=cv.RANSAC, prob=0.999, threshold=1.0
        )
        inl1 = pts1[mask_e.ravel() == 1]
        inl2 = pts2[mask_e.ravel() == 1]

        if len(inl1) < 5:
            return None, None, None, None

        _, R, t, _ = cv.recoverPose(
            E, inl1, inl2, focal=self.fx, pp=(self.cx, self.cy)
        )
        return inl1, inl2, R, t

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
        debug_mm  = self.config['display'].get('debug_map_matching', False)
        if show_plot:
            plt.ion()
            fig, ax = plt.subplots()
            ax.set_title("Trajetória Estimada")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            line_est,  = ax.plot(x_est_list,  y_est_list,  'r-', label='Estimado')
            line_real, = ax.plot(x_real_list, y_real_list, 'b-', label='Real')
            line_map,  = ax.plot([], [], 'g^', markersize=8, label='Map Matching', zorder=5)
            ax.legend()

        print("\nIniciando o loop de odometria...")
        xy_map_corr    = []
        resultados_list = []
        if self.config['display'].get('print_console', True):
            print("IMG\tKPT1\tKPT2\tMATCHES\tINLIERS\t\tDIST_REAL\tDIST_EST\tERRO_ACUM(m)\tTEMPO(s)\tMAP_OK")

        # --- Loop principal ---
        num_frames    = len(self.imgs_list)
        prev_features = None

        for i in range(num_frames - 1):
            t0 = time.time()

            prev_img = self.imgs_list[i]
            curr_img = self.imgs_list[i + 1]

            # 1. Correspondências (odometria)
            pts1, pts2, kpt1, kpt2, n_matches, curr_features = self._obter_correspondencias(
                prev_img, curr_img, prev_features, usar_absoluto=False
            )
            prev_features = curr_features
            if pts1 is None or len(pts1) < 8:
                print(f"Frame {i}: poucas correspondências. Pulando.")
                continue

            # 2. Matriz Essencial → pose
            result_pose = self._process_frame_pose(pts1, pts2)
            if result_pose[0] is None:
                print(f"Frame {i}: poucos inliers após RANSAC. Pulando.")
                continue
            inl1, inl2, R, t = result_pose
            n_inliers = len(inl1)

            # 3. Escala via GSD
            dist_est_atual  = self._calcula_deslocamento_escala(inl1, inl2, self.gsd_list[i])
            dist_est_total += dist_est_atual

            # 4. Yaw acumulado
            euler     = Rotation.from_matrix(R).as_euler('zyx', degrees=True)
            yaw_delta = euler[0]
            inlier_ratio = n_inliers / n_matches if n_matches > 0 else 0
            # Ignora rotação se: ângulo absurdo nos primeiros frames,
            # OU taxa de inliers muito baixa (RANSAC divergiu — ver
            # pipeline.yaw_filter em config.yaml)
            if (i <= self._yaw_max_initial_frames and abs(yaw_delta) > self._yaw_max_initial_deg) \
                    or inlier_ratio < self._yaw_min_inlier_ratio:
                yaw_delta = 0.0
            yaw_acumulado_est = (yaw_acumulado_est - yaw_delta) % 360

            # 5. Nova posição estimada
            est_lat, est_lon = self.estima_latlon(
                lat_est_list[-1], lon_est_list[-1], yaw_acumulado_est, dist_est_atual
            )

            # 6. Correção por map matching (opcional)
            n_inliers_map = 0
            map_ok        = False
            lat_antes_corr, lon_antes_corr = est_lat, est_lon
            ran_map_matching = self.use_map_matching and (i % self.map_match_interval == 0)
            if ran_map_matching:
                (est_lat, est_lon,
                 yaw_acumulado_est,
                 self.escala_atual,
                 n_inliers_map,
                 map_ok) = self._corrigir_posicao_pelo_mapa(
                    curr_img, est_lat, est_lon, yaw_acumulado_est, i
                )

            if map_ok:
                if self.config['display'].get('show_map_matching', True):
                    self._visualizar_map_matching(
                        i, lat_antes_corr, lon_antes_corr, est_lat, est_lon
                    )
                x_mc, y_mc = self._latlon_to_xy(est_lat, est_lon)
                xy_map_corr.append((x_mc, y_mc))

            if debug_mm:
                if ran_map_matching:
                    self._exibir_debug_patch(i, n_inliers_map, map_ok)
                else:
                    input(f"[DEBUG] Frame {i} (map matching não executado neste intervalo)"
                          " | Pressione ENTER para continuar...")

            lat_est_list.append(est_lat)
            lon_est_list.append(est_lon)

            # 7. Métricas
            dist_real_atual, erro_acum  = self._calculate_metrics(i, est_lat, est_lon)
            dist_real_total            += dist_real_atual

            elapsed = time.time() - t0
            if self.config['display'].get('print_console', True):
                print(f'{i}\t{kpt1}\t{kpt2}\t{n_matches}\t{n_inliers}\t\t'
                      f'{dist_real_atual:8.2f}\t{dist_est_atual:8.2f}\t'
                      f'{erro_acum:8.2f}\t\t{elapsed:.2f}\t\t{map_ok}')

            resultados_list.append({
                "IMG": i, "KPT1": kpt1, "KPT2": kpt2,
                "MATCHES": n_matches, "INLIERS": n_inliers,
                "DIST_REAL": dist_real_atual, "DIST_EST": dist_est_atual,
                "ERRO_ACUM(m)": erro_acum, "TEMPO(s)": elapsed,
                "MAP_OK": map_ok,
            })

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
                if xy_map_corr:
                    xs, ys = zip(*xy_map_corr)
                    line_map.set_data(xs, ys)
                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.001)

            if self.config['display']['show_images']:
                self._exibir_correspondencias(prev_img, curr_img, inl1, inl2)
                if cv.waitKey(10) & 0xFF == ord('q'):
                    break

        # --- Finalização ---
        Resultados = DataFrame(resultados_list) if resultados_list else DataFrame()

        if (self.config['display']['show_images'] or
                self.config['display'].get('show_map_matching', False) or
                debug_mm):
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
        mach = self.config.get('mach')
        criar_caminho_kml(
            self.lat_real_list, self.lon_real_list,
            os.path.join(output_dir, 'trajetoria_real.kml'),
            color=_KML_COLOR_REAL
        )
        criar_caminho_kml(
            lat_est_list, lon_est_list,
            os.path.join(output_dir, f'trajetoria_estimada_{self.detector_type}.kml'),
            color=mach_to_kml_color(mach)
        )
        print("\nProcesso de odometria finalizado.")

    def _exibir_correspondencias(self, prev_img, curr_img, inl1, inl2,
                                  max_draw=150):
        """Desenha correspondências inliers e exibe em janela OpenCV."""
        n   = len(inl1)
        idx = (np.linspace(0, n - 1, max_draw, dtype=int) if n > max_draw else range(n))

        kp1 = [cv.KeyPoint(x=float(inl1[j][0]), y=float(inl1[j][1]), size=2) for j in idx]
        kp2 = [cv.KeyPoint(x=float(inl2[j][0]), y=float(inl2[j][1]), size=2) for j in idx]
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

        cv.imshow(f'Correspondencias Inliers — {self.detector_type}', img_display)
