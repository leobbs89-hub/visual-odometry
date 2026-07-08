"""
Compara odometria visual COM e SEM map matching na rota CHAMPAIGN, usando o
mapa base NAIP (baixar_mapa_naip.py) como fonte diferente da imagem "voada"
(captura no Google Earth, gerada por criar_rota_champaign.py).

Pré-requisitos antes de rodar:
    1. criar_rota_champaign.py já executado (CSV + frames/Resized existem).
    2. baixar_mapa_naip.py já executado, gerando:
       Tese/rotas_quadradas/CHAMPAIGN/map.tif

Uso:
    python rodar_teste_map_matching.py --mach 1.0
    python rodar_teste_map_matching.py --mach 1.0 --detector SUPERPOINT
    python rodar_teste_map_matching.py --mach 1.0 --detector ORB AKAZE SUPERPOINT
"""

import sys
import argparse
import traceback
from pathlib import Path

import pandas as pd

# main.py / odometria_visual.py moram na raiz de visual-odometry/, um nível
# acima desta pasta (map_matching_naip/)
PIPELINE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE_DIR))

from main import montar_config
from odometria_visual import OdometriaVisual

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

BASE = Path(r"C:\Users\bbs_l\OneDrive\Leandro\ITA\MESTRADO\Tese\rotas_quadradas\CHAMPAIGN")
MAP_TIF = BASE / "map.tif"   # compartilhado entre todos os machs (mesma área)

CAMERA = {"width": 640, "height": 640, "h_fov": 60, "v_fov": 60}

DEFAULT_DETECTORS = ["ORB", "AKAZE", "SUPERPOINT"]

# Parâmetros reaproveitados dos valores já otimizados em rodar_experimentos.py
_DET_PARAMS = {
    "ORB": {
        "orb": {
            "nfeatures": 3000, "scaleFactor": 1.1, "nlevels": 4,
            "edgeThreshold": 15, "firstLevel": 0, "WTA_K": 2,
            "scoreType": "HARRIS", "patchSize": 31,
        }
    },
    "AKAZE": {
        "akaze": {
            "descriptor_type": "MLDB", "descriptor_size": 0, "descriptor_channels": 3,
            "threshold": 0.0001, "nOctaves": 2, "nOctaveLayers": 6, "diffusivity": "PM_G1",
        }
    },
    # Champaign é terreno misto (urbano + rural) -> parâmetros próximos do
    # ajuste feito para URBANO em rodar_experimentos.py.
    "SUPERPOINT": {
        "superpoint": {"detection_threshold": 0.0005, "nms_radius": 4, "max_num_keypoints": 4096}
    },
    "LOFTR": {
        "loftr": {"pretrained": "outdoor"}
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def montar_cfg(mach, detector, map_matching_on, frames_dir, csv_name, output_folder):
    return {
        "detector": detector,
        "device":   "auto",
        "mach":     mach,
        "paths": {
            "base_path":    str(BASE / f"mach_{mach}"),
            "images":       frames_dir,
            "ground_truth": csv_name,
            "output":       f"results/{output_folder}",
            "map_tif":      str(MAP_TIF),
        },
        "camera":          CAMERA,
        "detector_params": _DET_PARAMS.get(detector, {}),
        "matcher":         {"nn_match_ratio": 0.9},
        "display": {
            "show_plot": False, "show_images": False,
            "print_console": True, "show_map_matching": False,
        },
        "map_matching": {
            "enabled":            map_matching_on,
            "absolute_detector":  "LOFTR",
            "interval":           1,
            # roi_size_m não é mais fixo — espelha config.yaml da raiz do
            # projeto (roi_margin_factor), derivado do GSD real do voo em
            # _inicializar_escala(). Ver comentário lá antes de mudar aqui.
            "roi_margin_factor":  1.3,
            "scale_search_step":  0.05,
        },
    }


def preparar_pasta_imagens(mach):
    """Mesma lógica de Resized_clean usada em rodar_experimentos.py."""
    frames_dir    = BASE / f"mach_{mach}" / "frames"
    resized_clean = frames_dir / "Resized_clean"
    resized       = frames_dir / "Resized"

    if resized_clean.exists():
        return str(resized_clean)
    if not resized.exists():
        sys.exit(
            f"[erro] Não encontrei {resized}. Rode criar_rota_champaign.py "
            "e capture as imagens no Google Earth antes de continuar."
        )
    return str(resized)


def rodar_par(mach, detector):
    """Roda o mesmo detector COM e SEM map matching. Retorna dict com erro final de cada variante."""
    frames_dir = preparar_pasta_imagens(mach)
    csv_name   = f"Coord-Heading-Elev_1500_{mach}.csv"

    resultado = {}
    for ligado, tag in [(False, "sem_mapmatch"), (True, "com_mapmatch")]:
        output_folder = f"{detector}_{tag}"
        print(f"\n{'='*60}\n[RUN] CHAMPAIGN/mach_{mach}/{output_folder}\n{'='*60}")
        try:
            cfg_raw = montar_cfg(mach, detector, ligado, frames_dir, csv_name, output_folder)
            config  = montar_config(cfg_raw)
            OdometriaVisual(config).executar()

            csv_path = BASE / f"mach_{mach}" / "results" / output_folder / f"resultados_{detector}.csv"
            df = pd.read_csv(csv_path)
            resultado[tag] = float(df["ERRO_ACUM(m)"].iloc[-1]) if len(df) else None
        except Exception:
            print(f"[ERRO] {detector} {tag}")
            traceback.print_exc()
            resultado[tag] = None

    return resultado


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mach", type=float, required=True, help="Valor de mach a testar (deve ter CSV/frames gerados)")
    parser.add_argument("--detector", nargs="+", default=DEFAULT_DETECTORS,
                         choices=["ORB", "AKAZE", "SIFT", "SUPERPOINT", "LOFTR", "MATCHFORMER"])
    args = parser.parse_args()

    if not MAP_TIF.exists():
        sys.exit(f"[erro] Mapa base não encontrado: {MAP_TIF}\nRode baixar_mapa_naip.py primeiro.")

    linhas = []
    for det in args.detector:
        r = rodar_par(args.mach, det)
        sem, com = r.get("sem_mapmatch"), r.get("com_mapmatch")
        delta = (com - sem) if (sem is not None and com is not None) else None
        linhas.append((det, sem, com, delta))

    print(f"\n{'='*60}\nRESUMO — Erro final acumulado (m), mach {args.mach}\n{'='*60}")
    print(f"{'Detector':<12}{'Sem MM':>12}{'Com MM':>12}{'Delta':>12}")
    for det, sem, com, delta in linhas:
        fmt = lambda v: f"{v:.1f}" if v is not None else "erro"
        sinal = "" if delta is None else ("+" if delta > 0 else "")
        print(f"{det:<12}{fmt(sem):>12}{fmt(com):>12}{sinal + fmt(delta) if delta is not None else 'erro':>12}")


if __name__ == "__main__":
    main()
