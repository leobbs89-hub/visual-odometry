# detectors.py
"""
Interface comum para os detectores/matchers de features suportados pelo
pipeline de Odometria Visual (ORB, AKAZE, SUPERPOINT, LOFTR, MATCHFORMER).

Cada classe encapsula exatamente a lógica de inicialização e matching que
antes vivia espalhada em OdometriaVisual._instanciar_detector() e
OdometriaVisual._obter_correspondencias() — sem alterar nenhum cálculo
numérico, apenas reorganizando em um objeto por detector.
"""

import sys
import os
from dataclasses import dataclass

import numpy as np


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


class FeatureDetector:
    """Interface comum para os detectores/matchers de features."""
    name: str

    def match(self, img1, img2, prev_state=None) -> MatchResult:
        raise NotImplementedError


class OrbDetector(FeatureDetector):
    name = "ORB"

    def __init__(self, params, matcher_params, device=None, papel="odometria"):
        self.det = cv.ORB_create(**params)
        self.mat = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
        self.nn_match_ratio = matcher_params['nn_match_ratio']
        print(f"  ORB [{papel}] pronto")

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
        print(f"  AKAZE [{papel}] pronto")

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
        print(f"  SuperPoint + LightGlue [{papel}] prontos em [{device}]")

    def match(self, img1, img2, prev_state=None):
        t1 = _img_to_tensor(img1, self.device)
        t2 = _img_to_tensor(img2, self.device)

        feats1, feats2, matches01 = match_pair(self.det, self.mat, t1, t2)
        kp1     = feats1['keypoints']
        kp2     = feats2['keypoints']
        matches = matches01['matches']

        pts1 = kp1[matches[:, 0]].cpu().numpy()
        pts2 = kp2[matches[:, 1]].cpu().numpy()
        return MatchResult(pts1, pts2, len(kp1), len(kp2), len(matches), None)


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
        print(f"  LoFTR [{papel}] pronto em [{device}]")

    def match(self, img1, img2, prev_state=None):
        t1 = _img_to_tensor(img1, self.device)
        t2 = _img_to_tensor(img2, self.device)
        with torch.inference_mode():
            corr = self.mat({'image0': t1, 'image1': t2})
        pts1 = corr['keypoints0'].cpu().numpy()
        pts2 = corr['keypoints1'].cpu().numpy()
        n = len(pts1)
        return MatchResult(pts1, pts2, n, n, n, None)


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
        self.device = device
        self.mat = Matchformer(params).eval().to(device)
        print(f"  MatchFormer [{papel}] pronto em [{device}]")

    def match(self, img1, img2, prev_state=None):
        t1 = _img_to_tensor(img1, self.device)
        t2 = _img_to_tensor(img2, self.device)
        data = {'image0': t1, 'image1': t2}
        with torch.inference_mode():
            self.mat(data)  # modifica data in-place, não retorna nada
        pts1 = data['mkpts0_f'].cpu().numpy()
        pts2 = data['mkpts1_f'].cpu().numpy()
        n = len(pts1)
        return MatchResult(pts1, pts2, n, n, n, None)


_REGISTRY = {
    'ORB':         OrbDetector,
    'AKAZE':       AkazeDetector,
    'SUPERPOINT':  SuperPointDetector,
    'LOFTR':       LoftrDetector,
    'MATCHFORMER': MatchFormerDetector,
}


def criar_detector(tipo, detector_params, matcher_params, device, papel='odometria'):
    """
    Instancia o FeatureDetector correspondente a `tipo`.

    Args:
        tipo (str): 'ORB' | 'AKAZE' | 'SUPERPOINT' | 'LOFTR' | 'MATCHFORMER'
        detector_params (dict): config['detector_params'] completo (a classe
            extrai a chave que precisa, ex: detector_params['orb']).
        matcher_params (dict): config['matcher_params'] (usado por ORB/AKAZE).
        device: torch.device ou None.
        papel (str): 'odometria' ou 'absoluto' (apenas para log).

    Returns:
        FeatureDetector
    """
    print(f"Inicializando detector [{papel}]: {tipo}")

    cls = _REGISTRY.get(tipo)
    if cls is None:
        raise ValueError(
            f"Detector desconhecido: '{tipo}'. "
            "Escolha: ORB | AKAZE | SUPERPOINT | LOFTR | MATCHFORMER"
        )

    key = tipo.lower()
    params = detector_params.get(key, {}) if tipo not in ('ORB', 'AKAZE') else detector_params[key]
    return cls(params, matcher_params, device, papel)
