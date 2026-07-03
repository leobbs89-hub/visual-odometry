"""
Roda todos os experimentos de odometria visual:
  - ORB, AKAZE, SIFT:       1 run por combinação  →  2 rotas × 7 machs × 3 = 42 runs
  - SUPERPOINT, LOFTR:      2 runs (GPU + CPU)    →  2 rotas × 7 machs × 2 × 2 = 56 runs
  Total: 98 runs

Uso:
    python rodar_experimentos.py                    # todas as combinações
    python rodar_experimentos.py --rota FLORESTA    # só FLORESTA
    python rodar_experimentos.py --detector ORB     # só ORB
    python rodar_experimentos.py --rota FLORESTA --detector ORB   # validação
    python rodar_experimentos.py --no-cpu           # pula variantes CPU dos descritores neurais
"""
import sys
import os
import shutil
import traceback
import argparse
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_DIR))

from main import montar_config
from odometria_visual import OdometriaVisual

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

BASE_TESE = Path(r"C:\Users\bbs_l\OneDrive\Leandro\ITA\MESTRADO\Tese\rotas_quadradas")
ROTAS     = ["FLORESTA", "URBANO"]
MACHS     = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
DETECTORS = ["ORB", "AKAZE", "SIFT", "SUPERPOINT", "LOFTR", "MATCHFORMER"]

NEURAL_DETECTORS = {"SUPERPOINT", "LOFTR", "MATCHFORMER"}
# (tag usada no nome da pasta, device passado ao pipeline)
NEURAL_RUNS = [("GPU", "auto"), ("CPU", "cpu")]

CAMERA = {"width": 640, "height": 640, "h_fov": 60, "v_fov": 60}

_DET_PARAMS = {
    "ORB": {
        "orb": {
            "nfeatures":     3000,
            "scaleFactor":   1.1,
            "nlevels":       4,
            "edgeThreshold": 15,
            "firstLevel":    0,
            "WTA_K":         2,
            "scoreType":     "HARRIS",
            "patchSize":     31,
        }
    },
    "AKAZE": {
        "akaze": {
            "descriptor_type":     "MLDB",
            "descriptor_size":     0,
            "descriptor_channels": 3,
            "threshold":           0.0001,
            "nOctaves":            2,
            "nOctaveLayers":       6,
            "diffusivity":         "PM_G1",
        }
    },
    "SIFT": {
        "FLORESTA": {
            "sift": {
                "nfeatures":          3000,
                "nOctaveLayers":       3,
                "contrastThreshold":   0.04,
                "edgeThreshold":       10,
                "sigma":                1.6,
            }
        },
        "URBANO": {
            "sift": {
                "nfeatures":          3000,
                "nOctaveLayers":       4,
                "contrastThreshold":   0.01,
                "edgeThreshold":       20,
                "sigma":                1.2,
            }
        },
    },
    "SUPERPOINT": {
        "FLORESTA": {
            "superpoint": {
                "detection_threshold": 0.0005,
                "nms_radius":          2,
                "max_num_keypoints":   1024,
            }
        },
        "URBANO": {
            "superpoint": {
                "detection_threshold": 0.0005,
                "nms_radius":          4,
                "max_num_keypoints":   4096,
            }
        },
    },
    "LOFTR": {
        "loftr": {"pretrained": "outdoor"}
    },
}

_MATCHFORMER_PARAMS = {
    "matchformer": {
        "backbone_type":           "largela",
        "ckpt_path":               "MatchFormer/weights/outdoor-large-LA.ckpt",
        "resolution":              [8, 2],
        "fine_window_size":        5,
        "fine_concat_coarse_feat": True,
        "coarse": {"d_model": 256, "d_ffn": 256},
        "fine":   {"d_model": 128, "d_ffn": 128},
        "match_coarse": {
            "thr":                  0.2,
            "border_rm":            0,
            "match_type":           "dual_softmax",
            "dsmax_temperature":    0.1,
            "train_coarse_percent": 0.2,
            "train_pad_num_gt_min": 200,
        },
    }
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ms(mach: float) -> str:
    return str(mach)


def preparar_pasta_imagens(rota: str, mach: float) -> str:
    """Retorna caminho de imagens, criando Resized_clean se houver extras."""
    frames_dir    = BASE_TESE / rota / f"mach_{ms(mach)}" / "frames"
    resized_clean = frames_dir / "Resized_clean"
    resized       = frames_dir / "Resized"

    if resized_clean.exists():
        return str(resized_clean)

    all_pngs   = [f for f in os.listdir(resized) if f.endswith(".png")]
    clean_pngs = [f for f in all_pngs if f.startswith("-")]
    extra_pngs = [f for f in all_pngs if not f.startswith("-")]

    if extra_pngs:
        print(f"  Criando Resized_clean ({len(extra_pngs)} extras excluídos)...")
        os.makedirs(resized_clean, exist_ok=True)
        for f in clean_pngs:
            shutil.copy2(resized / f, resized_clean / f)
        return str(resized_clean)

    return str(resized)


def montar_cfg(rota: str, mach: float, detector: str,
               device: str, output_folder: str) -> dict:
    """
    Monta config no formato esperado por montar_config() de main.py.
    output_folder: nome da subpasta dentro de results/ (ex: 'ORB', 'SUPERPOINT_GPU').
    """
    if detector in ("SUPERPOINT", "SIFT"):
        det_params = dict(_DET_PARAMS[detector][rota])
    elif detector == "MATCHFORMER":
        det_params = {}
    else:
        det_params = dict(_DET_PARAMS[detector])

    return {
        "detector": detector,
        "device":   device,
        "mach":     mach,
        "paths": {
            "base_path":    str(BASE_TESE / rota / f"mach_{ms(mach)}"),
            "images":       preparar_pasta_imagens(rota, mach),
            "ground_truth": f"Coord-Heading-Elev_1500_{ms(mach)}.csv",
            "output":       f"results/{output_folder}",
            "map_tif":      ".",   # dummy — map_matching desativado
        },
        "camera":          CAMERA,
        "detector_params": {**det_params, **_MATCHFORMER_PARAMS},
        "matcher":         {"nn_match_ratio": 0.9},
        "display":         {"show_plot": False, "show_images": False, "print_console": True},
        "map_matching":    {"enabled": False},
    }


def rodar_experimento(rota: str, mach: float, detector: str,
                      device_tag, device: str) -> str:
    """Roda um experimento. Retorna 'ok' ou 'error'."""
    output_folder = f"{detector}_{device_tag}" if device_tag else detector
    tag = f"{rota}/mach_{ms(mach)}/{output_folder}"

    print(f"\n{'='*60}")
    print(f"[RUN] {tag}")
    print(f"{'='*60}")

    try:
        cfg_raw = montar_cfg(rota, mach, detector, device, output_folder)
        config  = montar_config(cfg_raw)
        OdometriaVisual(config).executar()
        print(f"[OK] {tag}")
        return "ok"
    except Exception:
        print(f"[ERRO] {tag}")
        traceback.print_exc()
        return "error"


def build_combinacoes(rotas, machs, detectors, no_cpu=False):
    """Retorna lista de (rota, mach, detector, device_tag, device)."""
    neural_runs = [r for r in NEURAL_RUNS if r[0] != "CPU"] if no_cpu else NEURAL_RUNS
    combos = []
    for rota in rotas:
        for mach in machs:
            for detector in detectors:
                if detector in NEURAL_DETECTORS:
                    for device_tag, device in neural_runs:
                        combos.append((rota, mach, detector, device_tag, device))
                else:
                    combos.append((rota, mach, detector, None, "auto"))
    return combos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Roda experimentos de odometria visual")
    parser.add_argument("--rota",     choices=ROTAS,     default=None)
    parser.add_argument("--detector", choices=DETECTORS, default=None)
    parser.add_argument("--mach",     type=float,        default=None,
                        help="Rodar apenas este valor de mach (ex: 1.2)")
    parser.add_argument("--no-cpu",   action="store_true",
                        help="Para descritores neurais, roda apenas GPU (ignora variante CPU)")
    return parser.parse_args()


def main():
    args = parse_args()

    rotas     = [args.rota]     if args.rota     else ROTAS
    detectors = [args.detector] if args.detector else DETECTORS
    machs     = [args.mach]     if args.mach     else MACHS

    combinacoes = build_combinacoes(rotas, machs, detectors, no_cpu=args.no_cpu)
    print(f"Total de runs: {len(combinacoes)}")

    resultados = {"ok": [], "error": []}

    for rota, mach, detector, device_tag, device in combinacoes:
        status = rodar_experimento(rota, mach, detector, device_tag, device)
        label  = f"{rota}/mach_{ms(mach)}/{detector}{'_'+device_tag if device_tag else ''}"
        resultados[status].append(label)

    print(f"\n{'='*60}")
    print("RESUMO FINAL")
    print(f"{'='*60}")
    print(f"OK:    {len(resultados['ok'])}")
    print(f"Erros: {len(resultados['error'])}")

    if resultados["error"]:
        print("\nFalharam:")
        for r in resultados["error"]:
            print(f"  ✗ {r}")


if __name__ == "__main__":
    main()
