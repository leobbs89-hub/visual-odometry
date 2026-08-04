"""
Verifica se o CENTRO GEOGRAFICO do patch satelital (pixel 240,240, apos
_busca_angulo + _busca_escala) realmente coincide com (lat_gt, lon_gt) --
ou seja, se o recorte em si esta correto -- e compara com a posicao que o
MATCHING de features (homografia + inliers) estimaria na pratica.

Motivacao (2026-07-14): usuario reportou visualmente que os centros das
imagens aerea/satelite em diagnosticar_contraste.py nao parecem coincidir.
Investigacao numerica (script de uso unico em scratchpad) mostrou que o
CROP em si esta correto (pixel central do patch mapeia de volta pra GT
com erro de ~1-2m, puro arredondamento) -- mas a posicao ESTIMADA PELA
HOMOGRAFIA (o que o pipeline de verdade usa pra corrigir a posicao) fica
400-750m longe do GT mesmo apos a busca de angulo/escala escolher o melhor
candidato por contagem de inliers. Ou seja: o "desalinhamento" que se ve
a olho nu comparando as duas imagens nao e um bug de coordenadas -- e a
manifestacao visual da baixa acuracia geometrica do matching (o angulo/
escala que maximiza nº de inliers nao garante que o conteudo visual dos
dois patches se sobreponha ponto-a-ponto; feature matches na periferia da
imagem sao muito mais sensiveis a erro angular residual do que o pixel
central). Este script marca uma cruz no centro geometrico de cada imagem
(prova visual de que o crop esta centrado no GT) e reporta os dois numeros
lado a lado pra deixar isso inequivoco.

Uso:
    python verificar_alinhamento_patch.py --mach 1.0
    python verificar_alinhamento_patch.py --mach 1.0 --frames 1 7 13 19 25
    python verificar_alinhamento_patch.py --mach 1.0 --abs-detector SUPERPOINT
"""

import sys
import copy
import argparse
from pathlib import Path

import numpy as np
import cv2 as cv
from geopy.distance import distance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PIPELINE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE_DIR))

from main import montar_config
from odometria_visual import OdometriaVisual
from rodar_teste_map_matching import CAMERA, DETECTOR_PARAMS, preparar_pasta_imagens, BASE

COR_AEREA = "#2a78d6"
COR_SATELITE = "#1baf7a"
COR_CRUZ = "#e34948"  # slot 6 (red) -- destaque, nao entra em conflito com as duas series acima


def montar_cfg(mach, abs_detector, base_dir, map_tif, roi_margin_factor):
    return {
        "detector": "ORB",
        "device": "auto",
        "mach": mach,
        "paths": {
            "base_path": str(base_dir / f"mach_{mach}"),
            "images": preparar_pasta_imagens(base_dir, mach),
            "ground_truth": f"Coord-Heading-Elev_1500_{mach}.csv",
            "output": "results/_verificacao_alinhamento",
            "map_tif": str(map_tif),
        },
        "camera": CAMERA,
        "detector_params": copy.deepcopy(DETECTOR_PARAMS),
        "matcher": {"nn_match_ratio": 0.9},
        "display": {"show_plot": False, "show_images": False, "print_console": False, "show_map_matching": False},
        "map_matching": {
            "enabled": True,
            "absolute_detector": abs_detector,
            "interval": 1,
            "roi_margin_factor": roi_margin_factor,
            "roi_center_mode": "estimado",
            "scale_search_step": 0.05,
            "angle_search_range_deg": 30.0,
            "angle_search_candidates": 5,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mach", type=float, required=True)
    parser.add_argument("--abs-detector", default="ORB")
    parser.add_argument("--base-dir", type=Path, default=BASE)
    parser.add_argument("--map", type=Path, default=None)
    parser.add_argument("--roi-margin-factor", type=float, default=1.3)
    parser.add_argument("--frames", type=int, nargs="+", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--angulo", choices=["busca", "yaw_gt"], default="busca",
                         help="'busca' (default) usa _busca_angulo (busca por inliers); "
                              "'yaw_gt' usa a Proa do CSV diretamente, sem busca de angulo "
                              "(so busca de escala roda em cima do yaw_gt fixo).")
    args = parser.parse_args()

    base_dir = args.base_dir
    if args.map is not None:
        map_tif = args.map
    else:
        candidato_vrt = base_dir / "map_v2.vrt"
        map_tif = candidato_vrt if candidato_vrt.exists() else base_dir / "map.tif"
    if not map_tif.exists():
        sys.exit(f"[erro] Mapa base nao encontrado: {map_tif}")

    cfg_raw = montar_cfg(args.mach, args.abs_detector, base_dir, map_tif, args.roi_margin_factor)
    config = montar_config(cfg_raw)
    ov = OdometriaVisual(config)
    ov._carregar_dados()

    num_frames = len(ov.imgs_list)
    if args.frames is not None:
        indices = args.frames
    else:
        indices = sorted(set(int(round(x)) for x in np.linspace(1, num_frames - 1, 5)))

    TARGET = 480
    pares = []
    for idx in indices:
        if not (1 <= idx <= num_frames - 1):
            print(f"[aviso] frame {idx} fora do intervalo -- pulando")
            continue
        lat_gt = float(ov.lat_real_list[idx])
        lon_gt = float(ov.lon_real_list[idx])
        yaw_gt = float(ov.yaw_real_list[idx])
        img_aerea_r = cv.resize(ov.imgs_list[idx], (TARGET, TARGET))

        if args.angulo == "busca":
            (_, angulo_otimo, _, _, _, _, _, _) = ov._busca_angulo(
                img_aerea_r, lat_gt, lon_gt, yaw_gt, ov.escala_atual
            )
        else:
            angulo_otimo = yaw_gt
        (patch, escala_otima, pts1, pts2, n_inliers, T_final, _, _) = ov._busca_escala(
            img_aerea_r, lat_gt, lon_gt, angulo_otimo, ov.escala_atual
        )
        if patch.ndim == 3:
            patch = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)

        # 1) o crop em si esta centrado no GT? (independe do matching)
        lon_c, lat_c = T_final * (TARGET / 2.0, TARGET / 2.0)
        if ov._map_transformer is not None:
            lon_c, lat_c = ov._map_transformer.transform(lon_c, lat_c, direction='INVERSE')
        d_crop = distance((lat_gt, lon_gt), (lat_c, lon_c)).meters

        # 2) posicao que o matching por homografia estimaria de verdade
        d_hom = None
        if pts1 is not None and len(pts1) >= 4:
            res = ov._estimar_posicao_pela_homografia(pts1, pts2, T_final, img_aerea_r.shape)
            if res is not None:
                d_hom = distance((lat_gt, lon_gt), res).meters

        print(f"Frame {idx}: angulo={angulo_otimo:.1f} escala={escala_otima:.3f} n_inliers={n_inliers} "
              f"| centro_do_crop_vs_GT={d_crop:.1f}m "
              f"| posicao_estimada_por_homografia_vs_GT={'%.1fm' % d_hom if d_hom is not None else 'N/A'}")

        pares.append((idx, img_aerea_r, patch, d_crop, d_hom, n_inliers))

    if not pares:
        sys.exit("[erro] Nenhum frame valido.")

    n = len(pares)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4.2 * n))
    if n == 1:
        axes = axes[None, :]

    cx = cy = TARGET / 2.0
    for row, (idx, img_a, img_s, d_crop, d_hom, n_inl) in enumerate(pares):
        for col, (img, titulo, cor) in enumerate([
            (img_a, f"Frame {idx} -- Aerea", COR_AEREA),
            (img_s, f"Frame {idx} -- Satelite", COR_SATELITE),
        ]):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.axhline(cy, color=COR_CRUZ, lw=0.8, alpha=0.9)
            ax.axvline(cx, color=COR_CRUZ, lw=0.8, alpha=0.9)
            ax.set_title(titulo, fontsize=9)
            ax.axis("off")
        d_hom_txt = f"{d_hom:.0f}m" if d_hom is not None else "N/A"
        axes[row, 0].text(
            0.0, -0.08,
            f"centro do crop vs GT: {d_crop:.1f}m   |   posicao estimada (homografia, {n_inl} inliers) vs GT: {d_hom_txt}",
            transform=axes[row, 0].transAxes, fontsize=8, color="#52514e"
        )

    fig.tight_layout()
    sufixo = "" if args.angulo == "busca" else "_yaw_gt"
    out_path = args.out if args.out is not None else base_dir / f"mach_{args.mach}" / "results" / f"verificacao_alinhamento_patch{sufixo}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"\nFigura salva em: {out_path}")


if __name__ == "__main__":
    main()
