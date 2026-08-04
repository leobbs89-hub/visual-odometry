"""
Mede o deslocamento LOCAL entre a imagem aerea e o patch satelital perto do
centro, via correlacao de fase (cv.phaseCorrelate) -- independe de
feature-matching/RANSAC, mede diretamente "quanto preciso mover o patch
satelital pra alinhar o conteudo visual com a aerea nessa regiao".

Motivacao (2026-07-14): usuario notou um erro visual no centro de TODAS as
imagens comparadas (ver verificar_alinhamento_patch.py) e suspeita que seja
um problema da fonte NAIP (georreferenciamento do mosaico/VRT). Antes de
trocar de fonte (trabalho grande: nova busca STAC, download, mosaico), este
script testa se o deslocamento e SISTEMATICO (mesma direcao/magnitude em
metros reais em todos os frames -- assinatura de bug de georreferenciamento
no map_v2.vrt) ou ALEATORIO (direcao/magnitude variam por frame -- mais
consistente com diferenca real de conteudo entre as fontes, ex. construcao
nova, vegetacao, sombra, data de captura diferente).

Para isolar do angulo/escala escolhidos pela busca por inliers (que ja tem
seu proprio erro, ver conversa), usa o MESMO patch alinhado (busca de
angulo + escala por inliers, igual ao pipeline real) e mede o residuo LOCAL
que sobra depois disso -- ou seja, "dado o melhor angulo/escala que o
pipeline acha, ainda sobra deslocamento fino no centro?"

O deslocamento em pixels e convertido pra metros (gsd_voo_efetivo) e
tambem para East/North no referencial do MUNDO (desfazendo a rotacao do
angulo escolhido), pra poder comparar a DIRECAO geografica entre frames
com headings diferentes -- se a direcao E/N for parecida entre frames
apesar de headings diferentes, isso aponta pra um bug fixo no mosaico
(ex.: erro de registro entre os 2 tiles NAIP, ou half-pixel na VRT).

Uso:
    python diagnosticar_offset_centro.py --mach 1.0
    python diagnosticar_offset_centro.py --mach 1.0 --frames 1 7 13 19 25
"""

import sys
import copy
import argparse
from pathlib import Path

import numpy as np
import cv2 as cv

PIPELINE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE_DIR))

from main import montar_config
from odometria_visual import OdometriaVisual
from rodar_teste_map_matching import CAMERA, DETECTOR_PARAMS, preparar_pasta_imagens, BASE


def montar_cfg(mach, abs_detector, base_dir, map_tif, roi_margin_factor):
    return {
        "detector": "ORB",
        "device": "auto",
        "mach": mach,
        "paths": {
            "base_path": str(base_dir / f"mach_{mach}"),
            "images": preparar_pasta_imagens(base_dir, mach),
            "ground_truth": f"Coord-Heading-Elev_1500_{mach}.csv",
            "output": "results/_diag_offset_centro",
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


def chip_central(img, lado):
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    r = lado // 2
    return img[cy - r:cy + r, cx - r:cx + r]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mach", type=float, required=True)
    parser.add_argument("--abs-detector", default="ORB")
    parser.add_argument("--base-dir", type=Path, default=BASE)
    parser.add_argument("--map", type=Path, default=None)
    parser.add_argument("--roi-margin-factor", type=float, default=1.3)
    parser.add_argument("--frames", type=int, nargs="+", default=None)
    parser.add_argument("--chip", type=int, default=200, help="Lado do chip central (px) usado na correlacao de fase.")
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
    gsd = ov.gsd_voo_efetivo
    resultados = []

    for idx in indices:
        if not (1 <= idx <= num_frames - 1):
            continue
        lat_gt = float(ov.lat_real_list[idx])
        lon_gt = float(ov.lon_real_list[idx])
        yaw_gt = float(ov.yaw_real_list[idx])
        img_aerea_r = cv.resize(ov.imgs_list[idx], (TARGET, TARGET))

        (_, angulo_otimo, _, _, _, _, _, _) = ov._busca_angulo(
            img_aerea_r, lat_gt, lon_gt, yaw_gt, ov.escala_atual
        )
        (patch, escala_otima, _, _, n_inliers, _, _, _) = ov._busca_escala(
            img_aerea_r, lat_gt, lon_gt, angulo_otimo, ov.escala_atual
        )
        if patch.ndim == 3:
            patch = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
        aerea = img_aerea_r if img_aerea_r.ndim == 2 else cv.cvtColor(img_aerea_r, cv.COLOR_BGR2GRAY)

        chip_a = chip_central(aerea, args.chip).astype(np.float32)
        chip_s = chip_central(patch, args.chip).astype(np.float32)

        win = cv.createHanningWindow((args.chip, args.chip), cv.CV_32F)
        (dx, dy), resposta = cv.phaseCorrelate(chip_a * win, chip_s * win)

        # (dx,dy) em pixels da imagem alinhada (ja rotacionada por angulo_otimo).
        # Desfazer a rotacao pra obter deslocamento no referencial do MUNDO
        # (leste/norte) -- permite comparar direcao entre frames com headings
        # diferentes. angulo_otimo e o quanto o patch satelital (originalmente
        # norte-up) foi girado; a imagem aerea ja "nasce" nessa orientacao.
        rad = np.radians(angulo_otimo)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        # rotacao inversa (imagem -> mundo norte-up); dy de imagem cresce pra
        # baixo (sul), por isso o sinal invertido no norte
        leste = dx * cos_a - (-dy) * sin_a
        norte = dx * sin_a + (-dy) * cos_a

        dx_m, dy_m = dx * gsd, dy * gsd
        leste_m, norte_m = leste * gsd, norte * gsd
        mag_m = float(np.hypot(dx_m, dy_m))

        resultados.append((idx, dx, dy, dx_m, dy_m, leste_m, norte_m, mag_m, resposta, n_inliers, angulo_otimo))
        print(f"Frame {idx}: angulo={angulo_otimo:6.1f} n_inliers={n_inliers:4d} | "
              f"offset_imagem=({dx:+6.2f},{dy:+6.2f})px = ({dx_m:+7.2f},{dy_m:+7.2f})m | "
              f"offset_mundo(E,N)=({leste_m:+7.2f},{norte_m:+7.2f})m | |offset|={mag_m:6.2f}m | "
              f"confianca_corr={resposta:.3f}")

    if not resultados:
        sys.exit("[erro] Nenhum frame valido.")

    leste_arr = np.array([r[5] for r in resultados])
    norte_arr = np.array([r[6] for r in resultados])
    mag_arr = np.array([r[7] for r in resultados])
    print(f"\n{'='*70}\nRESUMO ({len(resultados)} frames)\n{'='*70}")
    print(f"|offset| medio = {mag_arr.mean():.2f}m  (desvio {mag_arr.std():.2f}m)")
    print(f"offset E medio = {leste_arr.mean():+.2f}m  (desvio {leste_arr.std():.2f}m)")
    print(f"offset N medio = {norte_arr.mean():+.2f}m  (desvio {norte_arr.std():.2f}m)")
    print(f"\nInterpretacao: se o desvio padrao de E/N for MUITO menor que a media "
          f"(offset aponta sempre pra mesma direcao geografica, independente do "
          f"heading de cada frame), e forte indicio de erro de georreferenciamento "
          f"FIXO no mosaico NAIP (map_v2.vrt). Se E/N variar bastante frame a frame "
          f"(sem direcao preferencial), e mais consistente com diferenca real de "
          f"conteudo entre as fontes (Google Earth x NAIP), nao um bug de mapa.")


if __name__ == "__main__":
    main()
