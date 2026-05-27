# main.py

"""
Ponto de entrada do pipeline de Odometria Visual Monocular.

Uso:
    python main.py                              # usa config.yaml padrão
    python main.py --config outro.yaml
    python main.py --detector AKAZE             # sobrescreve só o detector
    python main.py --detector SUPERPOINT        # roda rede neural em CPU
    python main.py --detector SUPERPOINT --device cpu   # força CPU explicitamente
"""

import sys
import argparse
import numpy as np
import cv2 as cv
import yaml
from pathlib import Path


# ---------------------------------------------------------------------------
# Constantes OpenCV referenciadas como string no YAML
# ---------------------------------------------------------------------------
_ORB_SCORE = {
    "HARRIS": cv.ORB_HARRIS_SCORE,
    "FAST":   cv.ORB_FAST_SCORE,
}
_AKAZE_DESC = {
    "MLDB":         cv.AKAZE_DESCRIPTOR_MLDB,
    "MLDB_UPRIGHT": cv.AKAZE_DESCRIPTOR_MLDB_UPRIGHT,
}
_KAZE_DIFF = {
    "PM_G1":       cv.KAZE_DIFF_PM_G1,
    "PM_G2":       cv.KAZE_DIFF_PM_G2,
    "WEICKERT":    cv.KAZE_DIFF_WEICKERT,
    "CHARBONNIER": cv.KAZE_DIFF_CHARBONNIER,
}


# ---------------------------------------------------------------------------
# Funções de preparação de config
# ---------------------------------------------------------------------------

def carregar_yaml(caminho: str) -> dict:
    """Lê o arquivo YAML e retorna o dicionário de configuração."""
    with open(caminho, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolver_caminhos(cfg: dict) -> dict:
    """
    Constrói caminhos absolutos a partir do base_path do YAML.
    Retorna o dicionário 'paths' esperado por OdometriaVisual.
    """
    base = Path(cfg["paths"]["base_path"])
    return {
        "image_path":        str(base / cfg["paths"]["images"]),
        "ground_truth_path": str(base / cfg["paths"]["ground_truth"]),
        "output_dir":        str(base / cfg["paths"]["output"]),
        "map_tif_path":      str(base / cfg["paths"]["map_tif"]),
    }


def resolver_camera(cfg: dict) -> dict:
    """Calcula fx, fy, cx, cy a partir das dimensões e FOV da câmera."""
    cam = cfg["camera"]
    w, h = cam["width"], cam["height"]
    fx = w / 2 / np.tan(np.deg2rad(cam["h_fov"]) / 2)
    fy = h / 2 / np.tan(np.deg2rad(cam["v_fov"]) / 2)
    return {"fx": fx, "fy": fy, "cx": w / 2, "cy": h / 2}


def resolver_detector_params(cfg: dict) -> dict:
    """
    Converte strings legíveis (ex: 'HARRIS') pelas constantes numéricas
    do OpenCV, mantendo os demais parâmetros intactos.
    """
    p = cfg.get("detector_params", {})

    if "orb" in p:
        p["orb"]["scoreType"] = _ORB_SCORE.get(
            str(p["orb"].get("scoreType", "HARRIS")).upper(),
            cv.ORB_HARRIS_SCORE,
        )
    if "akaze" in p:
        p["akaze"]["descriptor_type"] = _AKAZE_DESC.get(
            str(p["akaze"].get("descriptor_type", "MLDB")).upper(),
            cv.AKAZE_DESCRIPTOR_MLDB,
        )
        p["akaze"]["diffusivity"] = _KAZE_DIFF.get(
            str(p["akaze"].get("diffusivity", "PM_G2")).upper(),
            cv.KAZE_DIFF_PM_G2,
        )
    return p


def montar_config(cfg: dict) -> dict:
    """
    Transforma o dicionário lido do YAML no formato esperado pela classe
    OdometriaVisual.
    """
    mm = cfg.get("map_matching", {})
    return {
        "detector_type":      cfg["detector"],
        # 'auto' usa GPU se disponível, senão CPU automaticamente.
        # 'cpu'  força CPU mesmo que haja GPU.
        # 'cuda' força GPU e lança erro se não houver CUDA.
        "device":             cfg.get("device", "auto"),
        "paths":              resolver_caminhos(cfg),
        "camera_params":      resolver_camera(cfg),
        "matcher_params":     cfg.get("matcher", {"nn_match_ratio": 0.8}),
        "detector_params":    resolver_detector_params(cfg),
        "display":            cfg.get("display", {"show_plot": True, "show_images": False}),
        "use_map_matching":   mm.get("enabled", False),
        "map_match_interval": mm.get("interval", 1),
        "roi_size_m":         mm.get("roi_size_m", 1000),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de Odometria Visual Monocular",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Caminho para o arquivo de configuração YAML\n(padrão: config.yaml)",
    )
    parser.add_argument(
        "--detector", "-d",
        choices=["ORB", "AKAZE", "SUPERPOINT", "LOFTR", "MATCHFORMER"],
        help="Sobrescreve o campo 'detector' do config.yaml",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        help=(
            "Dispositivo para detectores neurais (SUPERPOINT/LOFTR/MATCHFORMER).\n"
            "  auto  usa GPU se disponível, senão CPU (padrão)\n"
            "  cpu   força CPU — funciona sem placa de vídeo\n"
            "  cuda  força GPU; erro se CUDA não estiver disponível\n"
            "Sobrescreve o campo 'device' do config.yaml."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"[ERRO] Arquivo de configuração não encontrado: {config_path}")

    cfg = carregar_yaml(str(config_path))

    # Sobrescreve campos via linha de comando, se fornecidos
    if args.detector:
        cfg["detector"] = args.detector
        print(f"[INFO] Detector sobrescrito via argumento: {args.detector}")

    if args.device:
        cfg["device"] = args.device
        print(f"[INFO] Device sobrescrito via argumento: {args.device}")

    config = montar_config(cfg)

    from odometria_visual import OdometriaVisual
    try:
        odometria = OdometriaVisual(config)
        odometria.executar()
    except FileNotFoundError as e:
        sys.exit(f"[ERRO] Arquivo ou diretório não encontrado: {e}")
    except ImportError as e:
        sys.exit(f"[ERRO] Dependência ausente:\n{e}")
    except RuntimeError as e:
        sys.exit(f"[ERRO] Configuração de device inválida:\n{e}")
    except Exception as e:
        sys.exit(f"[ERRO] Falha na execução: {e}")


if __name__ == "__main__":
    main()
