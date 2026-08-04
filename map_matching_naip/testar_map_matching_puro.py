"""
Testa o módulo de map matching (Localização Absoluta, map_matching.py) ISOLADO
da odometria: para cada frame, em vez de usar a posição estimada por
dead-reckoning (que carrega deriva acumulada), usa diretamente a posição REAL
(GPS) daquele frame como entrada de `_corrigir_posicao_pelo_mapa`. Isso isola
a precisão do matching/geo-referenciamento em si -- sem o problema de "onde
procurar" (raio de busca vs. deriva acumulada), que é uma questão separada já
investigada em `CLAUDE.md` desta pasta.

Duas estratégias de YAW são comparadas, no mesmo frame / mesma posição:

  - "busca_angulo": yaw base carregado só pelo próprio map matching, frame a
    frame, SEM nenhuma contribuição de odometria -- começa no yaw real do
    frame 0 (mesmo init que `OdometriaVisual.executar()` usa) e, a partir
    daí, cada frame usa o `novo_yaw` devolvido pela correção do frame
    anterior como base para a busca de ângulo do frame seguinte. Isso
    reaproduz exatamente o mecanismo de busca de ângulo (`_busca_angulo`,
    ver map_matching.py) já usado pelo pipeline principal, só que sem a
    contribuição de yaw_delta da odometria entre correções.
  - "yaw_real": yaw base = Proa real do CSV daquele frame, direto, SEM
    encadeamento entre frames -- oráculo de yaw perfeito, para medir o
    piso de precisão do matching quando o yaw de entrada nunca está errado.

Em ambos os casos a busca de ângulo (`_busca_angulo`) e de escala
(`_busca_escala`) já embutidas em `_corrigir_posicao_pelo_mapa` continuam
rodando normalmente (mesmo bracket ±angle_search_range_deg /
scale_search_step do config) -- a única coisa que muda entre as duas
variantes é o valor de yaw_acumulado usado como base do offset.

Saída: um CSV por abs_detector com o erro de posição (m) por frame para as
duas variantes, mais um resumo com o erro médio e % de frames com
MAP_OK=True.

Uso:
    python testar_map_matching_puro.py --mach 1.0
    python testar_map_matching_puro.py --mach 1.0 --abs-detector ORB AKAZE SUPERPOINT
    python testar_map_matching_puro.py --mach 1.0 --abs-detector LOFTR \
        --base-dir "...\\CHAMPAIGN_GRANDE" --roi-margin-factor 3.0
"""

import sys
import copy
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
from rodar_teste_map_matching import CAMERA, DETECTOR_PARAMS, preparar_pasta_imagens, resolver_csv_name, BASE

# Detector de odometria nunca é usado neste teste (não chamamos executar()) --
# só precisa ser um valor válido para o __init__ de OdometriaVisual instanciar
# um FeatureDetector barato.
DETECTOR_ODOM_DUMMY = "ORB"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def montar_cfg_puro(mach, abs_detector, frames_dir, csv_name, base_dir, map_tif, roi_margin_factor,
                    photometric_norm="none", degrade_base_gsd=None):
    return {
        "detector": DETECTOR_ODOM_DUMMY,
        "device":   "auto",
        "mach":     mach,
        "paths": {
            "base_path":    str(base_dir / f"mach_{mach}"),
            "images":       frames_dir,
            "ground_truth": csv_name,
            "output":       "results/_map_matching_puro",  # não usado (não chamamos executar())
            "map_tif":      str(map_tif),
        },
        "camera":          CAMERA,
        # deepcopy: resolver_detector_params (main.py) converte strings tipo
        # "HARRIS"/"MLDB" para constantes cv2 IN PLACE -- sem cópia, chamar
        # montar_config várias vezes (uma por abs_detector) mutaria o dict
        # module-level compartilhado de rodar_teste_map_matching.
        "detector_params": copy.deepcopy(DETECTOR_PARAMS),
        "matcher":         {"nn_match_ratio": 0.9},
        "display": {
            "show_plot": False, "show_images": False,
            "print_console": False, "show_map_matching": False,
        },
        "map_matching": {
            "enabled":            True,
            "absolute_detector":  abs_detector,
            "interval":           1,
            "roi_margin_factor":  roi_margin_factor,
            "roi_center_mode":    "estimado",
            "scale_search_step":  0.05,
            "angle_search_range_deg":  30.0,
            "angle_search_candidates": 5,
            "photometric_norm":    photometric_norm,
            "base_map_degrade_gsd": degrade_base_gsd,
        },
    }


# ---------------------------------------------------------------------------
# Núcleo do teste
# ---------------------------------------------------------------------------

def testar_abs_detector(mach, abs_detector, base_dir, map_tif, roi_margin_factor,
                        photometric_norm="none", degrade_base_gsd=None):
    """Roda o map matching isolado (sem odometria) para todos os frames da
    rota, comparando as duas estratégias de yaw. Retorna um DataFrame com uma
    linha por frame."""
    frames_dir = preparar_pasta_imagens(base_dir, mach)
    csv_name   = resolver_csv_name(base_dir, mach)

    cfg_raw = montar_cfg_puro(mach, abs_detector, frames_dir, csv_name,
                               base_dir, map_tif, roi_margin_factor,
                               photometric_norm=photometric_norm,
                               degrade_base_gsd=degrade_base_gsd)
    config = montar_config(cfg_raw)

    ov = OdometriaVisual(config)
    ov._carregar_dados()

    num_frames = len(ov.imgs_list)
    if num_frames < 2:
        raise RuntimeError(f"Só {num_frames} imagem(ns) carregada(s) em {frames_dir} -- preciso de pelo menos 2.")

    # Init igual ao de OdometriaVisual.executar(): yaw_acumulado começa no
    # yaw real do frame 0.
    yaw_encadeado = float(ov.yaw_real_list[0])
    escala_busca  = ov.escala_atual  # já inicializada em 1.0 por _inicializar_escala
    escala_real   = ov.escala_atual

    print(f"\n{'='*90}\n[MAP MATCHING PURO] abs_detector={abs_detector} | mach={mach} | "
          f"{base_dir.name} | roi_margin_factor={roi_margin_factor} | {num_frames} frames\n{'='*90}")
    print(f"{'FRAME':<7}{'INL_ba':>8}{'OK_ba':>7}{'ERRO_ba(m)':>12}    "
          f"{'INL_yr':>8}{'OK_yr':>7}{'ERRO_yr(m)':>12}")

    linhas = []
    for i in range(num_frames - 1):
        lat_gt = float(ov.lat_real_list[i + 1])
        lon_gt = float(ov.lon_real_list[i + 1])
        yaw_gt = float(ov.yaw_real_list[i + 1])

        img_abs = ov._frame_para_detector(i + 1, ov.detector_absoluto)

        # --- Variante A: busca de ângulo encadeada (yaw só de map matching) ---
        yaw_base_a = yaw_encadeado
        ov.escala_atual = escala_busca
        (lat_a, lon_a, yaw_a, escala_a,
         n_inl_a, ok_a) = ov._corrigir_posicao_pelo_mapa(
            img_abs, lat_gt, lon_gt, yaw_base_a, i
        )
        erro_a = ov.calcula_distancia_latlon(lat_gt, lon_gt, lat_a, lon_a)
        yaw_encadeado = yaw_a
        escala_busca  = escala_a

        # --- Variante B: yaw real do CSV, direto, sem encadeamento ---
        ov.escala_atual = escala_real
        (lat_b, lon_b, yaw_b, escala_b,
         n_inl_b, ok_b) = ov._corrigir_posicao_pelo_mapa(
            img_abs, lat_gt, lon_gt, yaw_gt, i
        )
        erro_b = ov.calcula_distancia_latlon(lat_gt, lon_gt, lat_b, lon_b)
        escala_real = escala_b

        print(f"{i:<7}{n_inl_a:>8}{str(ok_a):>7}{erro_a:>12.2f}    "
              f"{n_inl_b:>8}{str(ok_b):>7}{erro_b:>12.2f}")

        linhas.append({
            "FRAME":                     i + 1,
            "LAT_GT":                    lat_gt,
            "LON_GT":                    lon_gt,
            "YAW_GT":                    yaw_gt,
            "YAW_BASE_BUSCA_ANGULO":     yaw_base_a,
            "YAW_ESTIMADO_BUSCA_ANGULO": yaw_a,
            "INLIERS_BUSCA_ANGULO":      n_inl_a,
            "MAP_OK_BUSCA_ANGULO":       bool(ok_a),
            "ERRO_BUSCA_ANGULO(m)":      erro_a,
            "INLIERS_YAW_REAL":          n_inl_b,
            "MAP_OK_YAW_REAL":           bool(ok_b),
            "ERRO_YAW_REAL(m)":          erro_b,
        })

    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mach", type=float, required=True, help="Valor de mach a testar (deve ter CSV/frames gerados)")
    parser.add_argument("--abs-detector", nargs="+", default=["SUPERPOINT"],
                         choices=["ORB", "AKAZE", "SIFT", "SUPERPOINT", "LOFTR", "MATCHFORMER", "ROMA"],
                         help="Detector do MAP MATCHING (absolute_detector) a testar. Passe vários pra comparar.")
    parser.add_argument("--base-dir", type=Path, default=BASE,
                         help="Pasta da rota (contém mach_X/ e o mapa base). Default: CHAMPAIGN original.")
    parser.add_argument("--map", type=Path, default=None,
                         help="Caminho do mapa base (.tif/.vrt). Default: <base-dir>/map_v2.vrt se existir, senão map.tif")
    parser.add_argument("--roi-margin-factor", type=float, default=1.3,
                         help="map_matching.roi_margin_factor (default: 1.3)")
    parser.add_argument("--out-dir", type=Path, default=None,
                         help="Pasta para salvar os CSVs de saída (default: <base-dir>/mach_X/results)")
    parser.add_argument("--photometric-norm", default="none",
                         help="Normalização fotométrica antes do match: none|clahe|histmatch|gradient "
                              "(combináveis por '+', ex. clahe+gradient). Default: none.")
    parser.add_argument("--degrade-base-gsd", type=float, default=None,
                         help="Degrada o mapa base para este GSD (m/px) antes do match, simulando uma "
                              "fonte mais grosseira sem trocar fonte/altitude. Default: off.")
    parser.add_argument("--tag", default=None,
                         help="Sufixo no nome do CSV de saída (ex. 'clahe') para não sobrescrever o baseline.")
    args = parser.parse_args()

    base_dir = args.base_dir
    if args.map is not None:
        map_tif = args.map
    else:
        candidato_vrt = base_dir / "map_v2.vrt"
        map_tif = candidato_vrt if candidato_vrt.exists() else base_dir / "map.tif"
    if not map_tif.exists():
        sys.exit(f"[erro] Mapa base não encontrado: {map_tif}\nRode baixar_mapa_naip.py primeiro ou passe --map.")

    resumo = []
    for abs_det in args.abs_detector:
        try:
            df = testar_abs_detector(args.mach, abs_det, base_dir, map_tif, args.roi_margin_factor,
                                     photometric_norm=args.photometric_norm,
                                     degrade_base_gsd=args.degrade_base_gsd)
        except Exception:
            print(f"[ERRO] abs_detector={abs_det}")
            traceback.print_exc()
            continue

        out_dir = args.out_dir if args.out_dir is not None else base_dir / f"mach_{args.mach}" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        sufixo = f"_{args.tag}" if args.tag else ""
        out_path = out_dir / f"mapmatch_puro_{abs_det}_mach_{args.mach}{sufixo}.csv"
        df.to_csv(out_path, index=False)

        # Erro médio "bruto" (todas as linhas) é enganoso: quando MAP_OK=False,
        # _corrigir_posicao_pelo_mapa devolve a própria posição de entrada
        # (aqui, sempre a posição real -- ver docstring do módulo), então uma
        # falha reporta erro 0.0, não uma penalidade. O erro médio condicionado
        # a MAP_OK=True é a métrica que reflete a precisão real do matching
        # quando ele de fato encontra correspondência.
        media_a       = df["ERRO_BUSCA_ANGULO(m)"].mean()
        media_b       = df["ERRO_YAW_REAL(m)"].mean()
        media_a_ok    = df.loc[df["MAP_OK_BUSCA_ANGULO"], "ERRO_BUSCA_ANGULO(m)"].mean()
        media_b_ok    = df.loc[df["MAP_OK_YAW_REAL"],     "ERRO_YAW_REAL(m)"].mean()
        pct_ok_a      = 100.0 * df["MAP_OK_BUSCA_ANGULO"].mean()
        pct_ok_b      = 100.0 * df["MAP_OK_YAW_REAL"].mean()
        resumo.append((abs_det, media_a, media_a_ok, pct_ok_a,
                        media_b, media_b_ok, pct_ok_b, len(df), out_path))

    print(f"\n{'='*90}\nRESUMO -- erro médio de posição por frame (m), mach {args.mach}, "
          f"{base_dir.name}, roi_margin_factor={args.roi_margin_factor}\n{'='*90}")
    print("ba = busca_angulo (yaw encadeado só via map matching) | yr = yaw_real (Proa do CSV)")
    print("ErroMedio_* = média sobre TODOS os frames (falha reporta 0.0, ver docstring) | "
          "ErroMedio_*_OK = média só sobre frames com MAP_OK=True (métrica confiável)")
    print(f"{'AbsDetector':<14}{'ErroMedio_ba':>13}{'ErroOK_ba':>11}{'MAP_OK%_ba':>11}  "
          f"{'ErroMedio_yr':>13}{'ErroOK_yr':>11}{'MAP_OK%_yr':>11}{'Nframes':>9}")
    for abs_det, media_a, media_a_ok, pct_ok_a, media_b, media_b_ok, pct_ok_b, n, out_path in resumo:
        print(f"{abs_det:<14}{media_a:>13.2f}{media_a_ok:>11.2f}{pct_ok_a:>10.1f}%  "
              f"{media_b:>13.2f}{media_b_ok:>11.2f}{pct_ok_b:>10.1f}%{n:>9}")
        print(f"    -> {out_path}")


if __name__ == "__main__":
    main()
