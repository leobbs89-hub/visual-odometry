"""
Diagnóstico de contraste/exposição: compara a imagem aérea (Google Earth,
frames/Resized/*.png) contra o patch satelital correspondente (recortado do
mapa base NAIP), usando a MESMA busca de ângulo (_busca_angulo) e escala
(_busca_escala) que o pipeline principal roda em _corrigir_posicao_pelo_mapa
-- não yaw_gt/escala=1.0 direto. Um erro de poucos graus no ângulo já desloca
visivelmente as features (mesmo com o centro geográfico correto) e pode ser
confundido com um problema de coordenadas -- ver nota abaixo.

Motivação: usuário notou visualmente que parte das imagens de satélite (NAIP)
aparecem "estouradas" (highlights sem detalhe) e com contraste mais baixo que
as imagens do Google Earth. Este script mede isso numericamente (média, desvio
padrão, percentis, fração de pixels saturados) em vez de só inspecionar
visualmente, e gera uma figura lado a lado com os histogramas para confirmar
ou refutar a suspeita antes de decidir qual técnica de equalização usar.

Nota sobre alinhamento (2026-07-14): a v1 deste script usava
`_preparar_patch_satelital(lat, lon, yaw_gt, escala=1.0)` direto -- mesmo
(lat, lon) center para as duas imagens, mas SEM a busca de ângulo/escala.
Isso pareceu (visualmente) um problema de coordenadas ("os locais não são
iguais"), mas era só o ângulo bruto (Proa do CSV) ficando alguns graus longe
do ótimo (ex.: frame 13 da rota CHAMPAIGN mach 1.0: yaw_gt=156.4° vs
ângulo ótimo=171.4°, 9 vs 17 inliers ORB) -- diferença de rotação, não de
posição. Corrigido para sempre rodar a busca real antes de comparar.

Nota sobre normalização: o patch satelital que sai de `_recortar_roi_mapa` já
passa por um `cv.normalize(..., NORM_MINMAX)` -- um alongamento linear que
estica o mínimo/máximo já presentes na janela recortada para 0-255. Isso já
reduz (mas não elimina) o efeito de "contraste baixo" se a causa for só janela
de valores estreita; NÃO recupera pixels que já saturaram (clipping) no
processamento original do NAIP -- esses continuam achatados em 255 mesmo
depois do stretch. A imagem aérea (Google Earth) não passa por nenhuma
equalização.

Uso:
    python diagnosticar_contraste.py --mach 1.0
    python diagnosticar_contraste.py --mach 1.0 --frames 1 6 12 18 24
"""

import sys
import copy
import argparse
from pathlib import Path

import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PIPELINE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE_DIR))

from main import montar_config
from odometria_visual import OdometriaVisual
from rodar_teste_map_matching import CAMERA, DETECTOR_PARAMS, preparar_pasta_imagens, BASE

# Paleta categórica (dataviz skill): slot 1 = azul, slot 2 = aqua.
COR_AEREA      = "#2a78d6"
COR_SATELITE   = "#1baf7a"


def montar_cfg_diag(mach, base_dir, map_tif, roi_margin_factor):
    return {
        "detector": "ORB",
        "device":   "auto",
        "mach":     mach,
        "paths": {
            "base_path":    str(base_dir / f"mach_{mach}"),
            "images":       preparar_pasta_imagens(base_dir, mach),
            "ground_truth": f"Coord-Heading-Elev_1500_{mach}.csv",
            "output":       "results/_diagnostico_contraste",  # não usado
            "map_tif":      str(map_tif),
        },
        "camera":          CAMERA,
        "detector_params": copy.deepcopy(DETECTOR_PARAMS),
        "matcher":         {"nn_match_ratio": 0.9},
        "display": {
            "show_plot": False, "show_images": False,
            "print_console": False, "show_map_matching": False,
        },
        "map_matching": {
            "enabled":            True,
            "absolute_detector":  "ORB",
            "interval":           1,
            "roi_margin_factor":  roi_margin_factor,
            "roi_center_mode":    "estimado",
            "scale_search_step":  0.05,
            "angle_search_range_deg":  30.0,
            "angle_search_candidates": 5,
        },
    }


def estatisticas(img):
    """Estatísticas de contraste/exposição de uma imagem grayscale uint8."""
    vals = img.astype(np.float64).ravel()
    return {
        "media":        float(vals.mean()),
        "desvio":       float(vals.std()),
        "min":          float(vals.min()),
        "max":          float(vals.max()),
        "p1":           float(np.percentile(vals, 1)),
        "p99":          float(np.percentile(vals, 99)),
        "frac_escuro":  float(np.mean(vals <= 5)),    # fração de pixels quase-pretos
        "frac_estourado": float(np.mean(vals >= 250)),  # fração de pixels quase-brancos (saturados)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mach", type=float, required=True)
    parser.add_argument("--base-dir", type=Path, default=BASE)
    parser.add_argument("--map", type=Path, default=None)
    parser.add_argument("--roi-margin-factor", type=float, default=1.3)
    parser.add_argument("--frames", type=int, nargs="+", default=None,
                         help="Índices de frame (1-based, mesma numeração do CSV) a inspecionar. "
                              "Default: 5 amostras uniformemente espaçadas na rota.")
    parser.add_argument("--out", type=Path, default=None,
                         help="Caminho do PNG de saída (default: <base-dir>/mach_X/results/diagnostico_contraste.png)")
    args = parser.parse_args()

    base_dir = args.base_dir
    if args.map is not None:
        map_tif = args.map
    else:
        candidato_vrt = base_dir / "map_v2.vrt"
        map_tif = candidato_vrt if candidato_vrt.exists() else base_dir / "map.tif"
    if not map_tif.exists():
        sys.exit(f"[erro] Mapa base não encontrado: {map_tif}")

    cfg_raw = montar_cfg_diag(args.mach, base_dir, map_tif, args.roi_margin_factor)
    config = montar_config(cfg_raw)
    ov = OdometriaVisual(config)
    ov._carregar_dados()

    num_frames = len(ov.imgs_list)
    if args.frames is not None:
        indices = args.frames
    else:
        n_amostras = 5
        indices = sorted(set(
            int(round(x)) for x in np.linspace(1, num_frames - 1, n_amostras)
        ))

    linhas_stats = []
    pares = []
    for idx in indices:
        if not (1 <= idx <= num_frames - 1):
            print(f"[aviso] frame {idx} fora do intervalo [1, {num_frames - 1}] -- pulando")
            continue
        img_aerea_raw = ov.imgs_list[idx]
        lat_gt = float(ov.lat_real_list[idx])
        lon_gt = float(ov.lon_real_list[idx])
        yaw_gt = float(ov.yaw_real_list[idx])

        TARGET = 480
        img_aerea_r = cv.resize(img_aerea_raw, (TARGET, TARGET))

        # Alinhamento de VERDADE: mesma busca de ângulo (_busca_angulo) +
        # escala (_busca_escala) que o pipeline principal roda em
        # _corrigir_posicao_pelo_mapa. Usar yaw_gt/escala=1.0 direto (como a
        # v1 deste script fazia) subestima o alinhamento real -- um erro de
        # poucos graus no ângulo já desloca visivelmente as features e pode
        # ser confundido com "GPS não bate" (ver conversa/CLAUDE.md).
        (_, angulo_otimo, _, _, _, _, _, _) = ov._busca_angulo(
            img_aerea_r, lat_gt, lon_gt, yaw_gt, ov.escala_atual
        )
        (patch, escala_otima, _, _, n_inliers, _, _, _) = ov._busca_escala(
            img_aerea_r, lat_gt, lon_gt, angulo_otimo, ov.escala_atual
        )
        if patch.ndim == 3:
            patch = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
        print(f"  [alinhamento] angulo={angulo_otimo:.1f} (Proa={yaw_gt:.1f}) "
              f"escala={escala_otima:.3f} inliers={n_inliers}")

        st_aerea = estatisticas(img_aerea_r)
        st_sat   = estatisticas(patch)
        linhas_stats.append((idx, st_aerea, st_sat))
        pares.append((idx, img_aerea_r, patch))

        print(f"Frame {idx}:")
        print(f"  Aérea    -- media={st_aerea['media']:.1f} desvio={st_aerea['desvio']:.1f} "
              f"p1-p99=[{st_aerea['p1']:.0f},{st_aerea['p99']:.0f}] "
              f"estourado={100*st_aerea['frac_estourado']:.2f}% escuro={100*st_aerea['frac_escuro']:.2f}%")
        print(f"  Satélite -- media={st_sat['media']:.1f} desvio={st_sat['desvio']:.1f} "
              f"p1-p99=[{st_sat['p1']:.0f},{st_sat['p99']:.0f}] "
              f"estourado={100*st_sat['frac_estourado']:.2f}% escuro={100*st_sat['frac_escuro']:.2f}%")

    if not linhas_stats:
        sys.exit("[erro] Nenhum frame válido para inspecionar.")

    # Médias agregadas
    media_desvio_aerea = np.mean([s[1]["desvio"] for s in linhas_stats])
    media_desvio_sat   = np.mean([s[2]["desvio"] for s in linhas_stats])
    media_estourado_aerea = np.mean([s[1]["frac_estourado"] for s in linhas_stats])
    media_estourado_sat   = np.mean([s[2]["frac_estourado"] for s in linhas_stats])
    print(f"\n{'='*70}\nRESUMO ({len(linhas_stats)} frames)\n{'='*70}")
    print(f"Desvio padrão médio     -- Aérea: {media_desvio_aerea:.1f}   Satélite: {media_desvio_sat:.1f}")
    print(f"Fração saturada (>=250) -- Aérea: {100*media_estourado_aerea:.2f}%   Satélite: {100*media_estourado_sat:.2f}%")

    # --- Figura: imagem + histograma por frame amostrado ---
    n = len(pares)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.2 * n))
    if n == 1:
        axes = axes[None, :]

    for row, (idx, img_a, img_s) in enumerate(pares):
        axes[row, 0].imshow(img_a, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title(f"Frame {idx} -- Aérea (Google Earth)", fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(img_s, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title(f"Frame {idx} -- Satélite (NAIP)", fontsize=9)
        axes[row, 1].axis("off")

        ax_h = axes[row, 2]
        ax_h.hist(img_a.ravel(), bins=64, range=(0, 255), alpha=0.6,
                   color=COR_AEREA, label="Aérea", density=True)
        ax_h.hist(img_s.ravel(), bins=64, range=(0, 255), alpha=0.6,
                   color=COR_SATELITE, label="Satélite", density=True)
        ax_h.set_xlim(0, 255)
        ax_h.set_yticks([])
        ax_h.tick_params(labelsize=7)
        if row == 0:
            ax_h.legend(fontsize=7)
        ax_h.set_title("Histograma de intensidade", fontsize=9)

    fig.tight_layout()
    out_path = args.out if args.out is not None else base_dir / f"mach_{args.mach}" / "results" / "diagnostico_contraste.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"\nFigura salva em: {out_path}")


if __name__ == "__main__":
    main()
