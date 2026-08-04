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

    # Comparar descritores do MAP MATCHING (absolute_detector) com o detector
    # principal da odometria fixo, numa rota/mapa diferentes (ex. CHAMPAIGN_GRANDE):
    python rodar_teste_map_matching.py --mach 1.0 --detector AKAZE \
        --abs-detector ORB AKAZE LOFTR MATCHFORMER \
        --base-dir "...\Tese\rotas_quadradas\CHAMPAIGN_GRANDE" \
        --roi-margin-factor 3.0
"""

import sys
import time
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
from utils import camera_do_recorte

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

BASE = Path(r"C:\Users\bbs_l\OneDrive\Leandro\ITA\MESTRADO\Tese\rotas_quadradas\CHAMPAIGN")
MAP_TIF = BASE / "map.tif"   # compartilhado entre todos os machs (mesma área)

# Derivado do recorte vigente (utils.FRAME_RECORTE_PX) em vez de hardcoded:
# este dict é importado por testar_map_matching_puro.py, diagnosticar_contraste.py,
# diagnosticar_offset_centro.py e verificar_alinhamento_patch.py, e ficar
# dessincronizado do recorte real reintroduz silenciosamente um erro de escala/centro.
CAMERA = camera_do_recorte()

DEFAULT_DETECTORS = ["ORB", "AKAZE", "SUPERPOINT"]
DEFAULT_ABS_DETECTORS = ["LOFTR"]

# Parâmetros reaproveitados dos valores já otimizados em rodar_experimentos.py.
# Único dict com TODAS as chaves (orb/akaze/superpoint/loftr/matchformer) --
# tanto o detector principal quanto o absolute_detector do map matching lêem
# a chave que precisam do mesmo config['detector_params'] (ver
# detectors.py::criar_detector), então os dois têm que estar presentes ao
# mesmo tempo quando são detectores diferentes.
DETECTOR_PARAMS = {
    "orb": {
        "nfeatures": 3000, "scaleFactor": 1.1, "nlevels": 4,
        "edgeThreshold": 15, "firstLevel": 0, "WTA_K": 2,
        "scoreType": "HARRIS", "patchSize": 31,
    },
    "akaze": {
        "descriptor_type": "MLDB", "descriptor_size": 0, "descriptor_channels": 3,
        "threshold": 0.0001, "nOctaves": 2, "nOctaveLayers": 6, "diffusivity": "PM_G1",
    },
    # Mesmos defaults do config.yaml da raiz do projeto (perfil FLORESTA).
    "sift": {
        "nfeatures": 3000, "nOctaveLayers": 3,
        "contrastThreshold": 0.04, "edgeThreshold": 10, "sigma": 1.6,
    },
    # Champaign é terreno misto (urbano + rural) -> parâmetros próximos do
    # ajuste feito para URBANO em rodar_experimentos.py.
    "superpoint": {"detection_threshold": 0.0005, "nms_radius": 4, "max_num_keypoints": 4096},
    "loftr": {"pretrained": "outdoor"},
    # Mesmos defaults do config.yaml da raiz do projeto.
    "matchformer": {
        "backbone_type": "largela",
        "ckpt_path": "MatchFormer/weights/outdoor-large-LA.ckpt",
        "resolution": [8, 2],
        "fine_window_size": 5,
        "fine_concat_coarse_feat": True,
        "coarse": {"d_model": 256, "d_ffn": 256},
        "fine": {"d_model": 128, "d_ffn": 128},
        "match_coarse": {
            "thr": 0.2, "border_rm": 0, "match_type": "dual_softmax",
            "dsmax_temperature": 0.1, "train_coarse_percent": 0.2,
            "train_pad_num_gt_min": 200,
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def montar_cfg(mach, detector, map_matching_on, frames_dir, csv_name, output_folder,
                base_dir, map_tif, abs_detector, roi_margin_factor, interval,
                roi_center_mode="estimado"):
    return {
        "detector": detector,
        "device":   "auto",
        "mach":     mach,
        "paths": {
            "base_path":    str(base_dir / f"mach_{mach}"),
            "images":       frames_dir,
            "ground_truth": csv_name,
            "output":       f"results/{output_folder}",
            "map_tif":      str(map_tif),
        },
        "camera":          CAMERA,
        "detector_params": DETECTOR_PARAMS,
        "matcher":         {"nn_match_ratio": 0.9},
        "display": {
            "show_plot": False, "show_images": False,
            "print_console": True, "show_map_matching": False,
        },
        "map_matching": {
            "enabled":            map_matching_on,
            "absolute_detector":  abs_detector,
            "interval":           interval,
            # roi_size_m não é mais fixo — espelha config.yaml da raiz do
            # projeto (roi_margin_factor), derivado do GSD real do voo em
            # _inicializar_escala(). Ver comentário lá antes de mudar aqui.
            "roi_margin_factor":  roi_margin_factor,
            "roi_center_mode":    roi_center_mode,
            "scale_search_step":  0.05,
            # Fix do "runaway" (2026-07-08) — ver map_matching.py::_busca_angulo.
            # Testado range=15° (2026-07-11), piorou (ver
            # project_map_matching_offset_yaw.md) — revertido para 30°.
            "angle_search_range_deg":  30.0,
            "angle_search_candidates": 5,
        },
    }


def preparar_pasta_imagens(base_dir, mach):
    """Mesma lógica de Resized_clean usada em rodar_experimentos.py."""
    frames_dir    = base_dir / f"mach_{mach}" / "frames"
    resized_clean = frames_dir / "Resized_clean"
    resized       = frames_dir / "Resized"

    if resized_clean.exists():
        return str(resized_clean)
    if not resized.exists():
        sys.exit(
            f"[erro] Não encontrei {resized}. Rode o script de geração da rota "
            "e capture as imagens no Google Earth antes de continuar."
        )
    return str(resized)


def resolver_csv_name(base_dir, mach):
    """
    Nome do CSV de ground truth (Coord-Heading-Elev_<ALTITUDE>_<MACH>.csv) --
    a altitude no nome varia por rota (1500 em CHAMPAIGN/CHAMPAIGN_GRANDE,
    3500 em CHAMPAIGN_ALT3500, etc.), então descobre o nome real por glob em
    vez de assumir 1500 fixo.
    """
    route_dir = base_dir / f"mach_{mach}"
    candidatos = list(route_dir.glob(f"Coord-Heading-Elev_*_{mach}.csv"))
    if not candidatos:
        sys.exit(f"[erro] Nenhum CSV 'Coord-Heading-Elev_*_{mach}.csv' encontrado em {route_dir}")
    if len(candidatos) > 1:
        sys.exit(f"[erro] Mais de um CSV candidato em {route_dir}: {[c.name for c in candidatos]} -- ambíguo.")
    return candidatos[0].name


def rodar_par(mach, detector, base_dir, map_tif, abs_detector, roi_margin_factor, interval,
              roi_center_mode="estimado"):
    """Roda o mesmo par (detector, abs_detector) COM e SEM map matching.
    Retorna dict com erro final e tempo (s) de execução de cada variante."""
    frames_dir = preparar_pasta_imagens(base_dir, mach)
    csv_name   = resolver_csv_name(base_dir, mach)

    resultado = {}
    for ligado, tag in [(False, "sem_mapmatch"), (True, "com_mapmatch")]:
        output_folder = f"{detector}_abs-{abs_detector}_{roi_center_mode}_{tag}"
        print(f"\n{'='*60}\n[RUN] {base_dir.name}/mach_{mach}/{output_folder}\n{'='*60}")
        try:
            cfg_raw = montar_cfg(mach, detector, ligado, frames_dir, csv_name, output_folder,
                                  base_dir, map_tif, abs_detector, roi_margin_factor, interval,
                                  roi_center_mode=roi_center_mode)
            config  = montar_config(cfg_raw)
            t0 = time.time()
            OdometriaVisual(config).executar()
            tempo_s = time.time() - t0

            csv_path = base_dir / f"mach_{mach}" / "results" / output_folder / f"resultados_{detector}.csv"
            df = pd.read_csv(csv_path)
            resultado[tag] = float(df["ERRO_ACUM(m)"].iloc[-1]) if len(df) else None
            resultado[f"{tag}_tempo_s"] = tempo_s
        except Exception:
            print(f"[ERRO] {detector} abs={abs_detector} {tag}")
            traceback.print_exc()
            resultado[tag] = None
            resultado[f"{tag}_tempo_s"] = None

    return resultado


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mach", type=float, required=True, help="Valor de mach a testar (deve ter CSV/frames gerados)")
    parser.add_argument("--detector", nargs="+", default=DEFAULT_DETECTORS,
                         choices=["ORB", "AKAZE", "SIFT", "SUPERPOINT", "LOFTR", "MATCHFORMER"],
                         help="Detector PRINCIPAL da odometria (eixo padrão de comparação)")
    parser.add_argument("--abs-detector", nargs="+", default=DEFAULT_ABS_DETECTORS,
                         choices=["ORB", "AKAZE", "SIFT", "SUPERPOINT", "LOFTR", "MATCHFORMER"],
                         help="Detector do MAP MATCHING (absolute_detector), independente do --detector. "
                              "Passe vários pra comparar descritores de map matching com --detector fixo.")
    parser.add_argument("--base-dir", type=Path, default=BASE,
                         help="Pasta da rota (contém mach_X/ e map.tif). Default: CHAMPAIGN original.")
    parser.add_argument("--map", type=Path, default=None,
                         help="Caminho do map.tif. Default: <base-dir>/map.tif")
    parser.add_argument("--roi-margin-factor", type=float, default=1.3,
                         help="map_matching.roi_margin_factor (default: 1.3, igual ao config.yaml original)")
    parser.add_argument("--interval", type=int, default=1,
                         help="map_matching.interval — a cada quantos frames rodar a correção (default: 1)")
    parser.add_argument("--roi-center-mode", nargs="+", default=["estimado"],
                         choices=["estimado", "real_bbox"],
                         help="map_matching.roi_center_mode. 'estimado' = comportamento normal "
                              "(default). 'real_bbox' = modo DIAGNÓSTICO que ancora a ROI na "
                              "coordenada real (GT), garantindo que ela esteja sempre dentro do "
                              "recorte pesquisado — não representa um cenário realista sem GPS. "
                              "Passe vários pra comparar os dois modos na mesma rodada.")
    args = parser.parse_args()

    base_dir = args.base_dir
    map_tif  = args.map if args.map is not None else base_dir / "map.tif"
    if not map_tif.exists():
        sys.exit(f"[erro] Mapa base não encontrado: {map_tif}\nRode baixar_mapa_naip.py primeiro.")

    linhas = []
    for det in args.detector:
        for abs_det in args.abs_detector:
            for roi_mode in args.roi_center_mode:
                r = rodar_par(args.mach, det, base_dir, map_tif, abs_det,
                              args.roi_margin_factor, args.interval, roi_center_mode=roi_mode)
                sem, com = r.get("sem_mapmatch"), r.get("com_mapmatch")
                delta = (com - sem) if (sem is not None and com is not None) else None
                tempo_com = r.get("com_mapmatch_tempo_s")
                linhas.append((det, abs_det, roi_mode, sem, com, delta, tempo_com))

    print(f"\n{'='*60}\nRESUMO — Erro final acumulado (m), mach {args.mach}, "
          f"roi_margin_factor={args.roi_margin_factor}, mapa={map_tif.name}\n{'='*60}")
    print(f"{'Detector':<12}{'AbsDetector':<14}{'ROI':<12}{'Sem MM':>12}{'Com MM':>12}{'Delta':>12}{'Tempo(s)':>12}")
    for det, abs_det, roi_mode, sem, com, delta, tempo_com in linhas:
        fmt = lambda v: f"{v:.1f}" if v is not None else "erro"
        sinal = "" if delta is None else ("+" if delta > 0 else "")
        print(f"{det:<12}{abs_det:<14}{roi_mode:<12}{fmt(sem):>12}{fmt(com):>12}"
              f"{(sinal + fmt(delta)) if delta is not None else 'erro':>12}{fmt(tempo_com):>12}")


if __name__ == "__main__":
    main()
