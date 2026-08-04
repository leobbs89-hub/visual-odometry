# detectors.py
"""
Interface comum para os detectores/matchers de features suportados pelo
pipeline de Odometria Visual (ORB, AKAZE, SIFT, SUPERPOINT, LOFTR, MATCHFORMER).

Cada classe encapsula exatamente a lógica de inicialização e matching que
antes vivia espalhada em OdometriaVisual._instanciar_detector() e
OdometriaVisual._obter_correspondencias() — sem alterar nenhum cálculo
numérico, apenas reorganizando em um objeto por detector.
"""

import sys
import os
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


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
    from kornia.feature import LoFTR
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False

# MatchFormer
caminho_matchformer = os.path.join(os.path.dirname(__file__), 'MatchFormer')
if caminho_matchformer not in sys.path:
    sys.path.insert(0, caminho_matchformer)

try:
    from model.matchformer import Matchformer
    MATCHFORMER_AVAILABLE = True
except ImportError:
    MATCHFORMER_AVAILABLE = False

# RoMa (backbone DINOv2, treinado em RGB — único detector deste módulo que
# usa cor de verdade; ver RomaDetector abaixo)
try:
    from romatch import roma_outdoor
    from PIL import Image
    ROMA_AVAILABLE = True
except ImportError:
    ROMA_AVAILABLE = False


# ---------------------------------------------------------------------------
# cv2 — só usado pelos detectores OpenCV (import tardio evita custo p/
# quem só usa detectores neurais, mas cv2 é dependência base do projeto)
# ---------------------------------------------------------------------------
import cv2 as cv


def _img_to_tensor(img, device):
    """Converte imagem numpy para tensor PyTorch preparado para modelos neurais."""
    return torch.from_numpy(img).float().to(device).unsqueeze(0).unsqueeze(0) / 255.


@dataclass
class MatchResult:
    pts1: "np.ndarray | None"
    pts2: "np.ndarray | None"
    kpt_count1: int
    kpt_count2: int
    match_count: int
    state: object = None   # cache de (kp, des) do frame atual — só ORB/AKAZE usam
    confidences: "np.ndarray | None" = None  # confiança por match (alinhada a pts1/pts2) — só detectores neurais


class FeatureDetector:
    """Interface comum para os detectores/matchers de features."""
    name: str
    # True apenas para detectores que exploram cor de verdade (hoje só RoMa).
    # Controla se OdometriaVisual entrega img1/img2 em BGR (H,W,3) ou
    # grayscale (H,W) — ver odometria_visual.py::_carregar_dados.
    requires_color: bool = False

    def match(self, img1, img2, prev_state=None) -> MatchResult:
        raise NotImplementedError


class OrbDetector(FeatureDetector):
    name = "ORB"

    def __init__(self, params, matcher_params, device=None, papel="odometria"):
        self.det = cv.ORB_create(**params)
        self.mat = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
        self.nn_match_ratio = matcher_params['nn_match_ratio']
        logger.info("ORB [%s] pronto", papel)

    def match(self, img1, img2, prev_state=None):
        kp1, des1 = prev_state if prev_state is not None else self.det.detectAndCompute(img1, None)
        kp2, des2 = self.det.detectAndCompute(img2, None)
        curr_state = (kp2, des2)

        if des1 is None or des2 is None:
            return MatchResult(
                None, None,
                len(kp1) if kp1 else 0,
                len(kp2) if kp2 else 0,
                0, curr_state,
            )

        good = [m for m, n in self.mat.knnMatch(des1, des2, k=2)
                if m.distance < self.nn_match_ratio * n.distance]

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return MatchResult(pts1, pts2, len(kp1), len(kp2), len(good), curr_state)


class AkazeDetector(FeatureDetector):
    name = "AKAZE"

    def __init__(self, params, matcher_params, device=None, papel="odometria"):
        self.det = cv.AKAZE_create(**params)
        self.mat = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
        self.nn_match_ratio = matcher_params['nn_match_ratio']
        logger.info("AKAZE [%s] pronto", papel)

    def match(self, img1, img2, prev_state=None):
        kp1, des1 = prev_state if prev_state is not None else self.det.detectAndCompute(img1, None)
        kp2, des2 = self.det.detectAndCompute(img2, None)
        curr_state = (kp2, des2)

        if des1 is None or des2 is None:
            return MatchResult(
                None, None,
                len(kp1) if kp1 else 0,
                len(kp2) if kp2 else 0,
                0, curr_state,
            )

        good = [m for m, n in self.mat.knnMatch(des1, des2, k=2)
                if m.distance < self.nn_match_ratio * n.distance]

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return MatchResult(pts1, pts2, len(kp1), len(kp2), len(good), curr_state)


class SiftDetector(FeatureDetector):
    name = "SIFT"

    def __init__(self, params, matcher_params, device=None, papel="odometria"):
        self.det = cv.SIFT_create(**params)
        self.mat = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)
        self.nn_match_ratio = matcher_params['nn_match_ratio']
        logger.info("SIFT [%s] pronto", papel)

    def match(self, img1, img2, prev_state=None):
        kp1, des1 = prev_state if prev_state is not None else self.det.detectAndCompute(img1, None)
        kp2, des2 = self.det.detectAndCompute(img2, None)
        curr_state = (kp2, des2)

        if des1 is None or des2 is None:
            return MatchResult(
                None, None,
                len(kp1) if kp1 else 0,
                len(kp2) if kp2 else 0,
                0, curr_state,
            )

        good = [m for m, n in self.mat.knnMatch(des1, des2, k=2)
                if m.distance < self.nn_match_ratio * n.distance]

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return MatchResult(pts1, pts2, len(kp1), len(kp2), len(good), curr_state)


class SuperPointDetector(FeatureDetector):
    name = "SUPERPOINT"

    def __init__(self, params, matcher_params, device, papel="odometria"):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch e LightGlue são necessários para SuperPoint.\n"
                "Instale com: pip install -r requirements-neural.txt\n"
                "  e clone:  git clone https://github.com/cvg/LightGlue.git\n"
                "Funciona em CPU — GPU não é obrigatória."
            )
        self.device = device
        self.det = SuperPoint(**params).eval().to(device)
        self.mat = LightGlue(features='superpoint').eval().to(device)
        logger.info("SuperPoint + LightGlue [%s] prontos em [%s]", papel, device)

    def match(self, img1, img2, prev_state=None):
        t1 = _img_to_tensor(img1, self.device)
        t2 = _img_to_tensor(img2, self.device)

        feats1, feats2, matches01 = match_pair(self.det, self.mat, t1, t2)
        kp1     = feats1['keypoints']
        kp2     = feats2['keypoints']
        matches = matches01['matches']

        pts1 = kp1[matches[:, 0]].cpu().numpy()
        pts2 = kp2[matches[:, 1]].cpu().numpy()
        confidences = matches01['matching_scores0'][matches[:, 0]].cpu().numpy()
        return MatchResult(pts1, pts2, len(kp1), len(kp2), len(matches), None, confidences)


class LoftrDetector(FeatureDetector):
    name = "LOFTR"

    def __init__(self, params, matcher_params, device, papel="odometria"):
        if not KORNIA_AVAILABLE:
            raise ImportError(
                "Kornia e PyTorch são necessários para LoFTR.\n"
                "Instale com: pip install -r requirements-neural.txt\n"
                "Funciona em CPU — GPU não é obrigatória."
            )
        self.device = device
        self.mat = LoFTR(**params).eval().to(device)
        logger.info("LoFTR [%s] pronto em [%s]", papel, device)

    def match(self, img1, img2, prev_state=None):
        t1 = _img_to_tensor(img1, self.device)
        t2 = _img_to_tensor(img2, self.device)
        with torch.inference_mode():
            corr = self.mat({'image0': t1, 'image1': t2})
        pts1 = corr['keypoints0'].cpu().numpy()
        pts2 = corr['keypoints1'].cpu().numpy()
        confidences = corr['confidence'].cpu().numpy()
        n = len(pts1)
        return MatchResult(pts1, pts2, n, n, n, None, confidences)


class MatchFormerDetector(FeatureDetector):
    name = "MATCHFORMER"

    def __init__(self, params, matcher_params, device, papel="odometria"):
        if not MATCHFORMER_AVAILABLE:
            raise ImportError(
                "A biblioteca MatchFormer não foi encontrada.\n"
                "Clone com: git clone https://github.com/InSAI-Lab/MatchFormer.git\n"
                "Instale dependências: pip install -r requirements-neural.txt\n"
                "Funciona em CPU — GPU não é obrigatória."
            )
        params = dict(params)
        ckpt_path = params.pop('ckpt_path', None)
        if not ckpt_path:
            raise ValueError(
                "detector_params.matchformer.ckpt_path não configurado em config.yaml.\n"
                "Baixe um checkpoint pré-treinado (ex: outdoor-large-LA.ckpt) em "
                "https://github.com/InSAI-Lab/MatchFormer e aponte o caminho em ckpt_path — "
                "sem isso o modelo roda com pesos aleatórios (não representativo)."
            )
        ckpt_path_abs = ckpt_path if os.path.isabs(ckpt_path) else os.path.join(
            os.path.dirname(caminho_matchformer), ckpt_path
        )
        if not os.path.isfile(ckpt_path_abs):
            raise FileNotFoundError(
                f"Checkpoint do MatchFormer não encontrado em '{ckpt_path_abs}'.\n"
                "Baixe o .ckpt pré-treinado (ex: outdoor-large-LA.ckpt) do Google Drive "
                "oficial do repositório e coloque no caminho configurado em ckpt_path."
            )

        self.device = device
        self.mat = Matchformer(params).eval().to(device)
        state_dict = {
            k.replace('matcher.', ''): v
            for k, v in torch.load(ckpt_path_abs, map_location='cpu').items()
        }
        self.mat.load_state_dict(state_dict)
        logger.info(
            "MatchFormer [%s] pronto em [%s] — checkpoint carregado de [%s]",
            papel, device, ckpt_path_abs
        )

    def match(self, img1, img2, prev_state=None):
        t1 = _img_to_tensor(img1, self.device)
        t2 = _img_to_tensor(img2, self.device)
        data = {'image0': t1, 'image1': t2}
        with torch.inference_mode():
            self.mat(data)  # modifica data in-place, não retorna nada
        pts1 = data['mkpts0_f'].cpu().numpy()
        pts2 = data['mkpts1_f'].cpu().numpy()
        confidences = data['mconf'].cpu().numpy()
        n = len(pts1)
        return MatchResult(pts1, pts2, n, n, n, None, confidences)


class RomaDetector(FeatureDetector):
    name = "ROMA"
    requires_color = True

    def __init__(self, params, matcher_params, device, papel="odometria"):
        if not ROMA_AVAILABLE:
            raise ImportError(
                "RoMa (pacote romatch) não foi encontrado.\n"
                "Instale com: pip install romatch\n"
                "Atenção: romatch traz albumentations, que instala "
                "opencv-python-headless e sobrescreve silenciosamente o "
                "opencv-python/opencv-contrib-python já instalado (mesmo "
                "namespace cv2). Depois de instalar, rode:\n"
                "  pip uninstall -y opencv-python-headless\n"
                "  pip install --force-reinstall --no-deps "
                "opencv-python==<versão original> "
                "opencv-contrib-python==<versão original>\n"
                "Funciona em CPU — GPU não é obrigatória, mas é bem mais lento "
                "(backbone DINOv2)."
            )
        params = dict(params)
        self.num_matches   = params.pop('num_matches', 5000)
        sample_thresh      = params.pop('sample_thresh', None)
        self.device = device
        self.model  = roma_outdoor(device=device, **params)
        if sample_thresh is not None:
            self.model.sample_thresh = sample_thresh
        logger.info("RoMa [%s] pronto em [%s]", papel, device)

    def match(self, img1, img2, prev_state=None):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        # img1/img2 chegam em BGR (cv.imread/rasterio já convertidos) —
        # PIL espera RGB.
        im_a = Image.fromarray(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
        im_b = Image.fromarray(cv.cvtColor(img2, cv.COLOR_BGR2RGB))

        warp, certainty = self.model.match(im_a, im_b, device=self.device)
        matches, conf = self.model.sample(warp, certainty, num=self.num_matches)
        kpts1, kpts2 = self.model.to_pixel_coordinates(matches, h1, w1, h2, w2)

        pts1 = kpts1.cpu().numpy()
        pts2 = kpts2.cpu().numpy()
        confidences = conf.cpu().numpy()
        n = len(pts1)
        return MatchResult(pts1, pts2, n, n, n, None, confidences)


_REGISTRY = {
    'ORB':         OrbDetector,
    'AKAZE':       AkazeDetector,
    'SIFT':        SiftDetector,
    'SUPERPOINT':  SuperPointDetector,
    'LOFTR':       LoftrDetector,
    'MATCHFORMER': MatchFormerDetector,
    'ROMA':        RomaDetector,
}


def criar_detector(tipo, detector_params, matcher_params, device, papel='odometria'):
    """
    Instancia o FeatureDetector correspondente a `tipo`.

    Args:
        tipo (str): 'ORB' | 'AKAZE' | 'SIFT' | 'SUPERPOINT' | 'LOFTR' | 'MATCHFORMER' | 'ROMA'
        detector_params (dict): config['detector_params'] completo (a classe
            extrai a chave que precisa, ex: detector_params['orb']).
        matcher_params (dict): config['matcher_params'] (usado por ORB/AKAZE/SIFT).
        device: torch.device ou None.
        papel (str): 'odometria' ou 'absoluto' (apenas para log).

    Returns:
        FeatureDetector
    """
    logger.info("Inicializando detector [%s]: %s", papel, tipo)

    cls = _REGISTRY.get(tipo)
    if cls is None:
        raise ValueError(
            f"Detector desconhecido: '{tipo}'. "
            "Escolha: ORB | AKAZE | SIFT | SUPERPOINT | LOFTR | MATCHFORMER | ROMA"
        )

    key = tipo.lower()
    params = detector_params.get(key, {}) if tipo not in ('ORB', 'AKAZE', 'SIFT') else detector_params[key]
    return cls(params, matcher_params, device, papel)
