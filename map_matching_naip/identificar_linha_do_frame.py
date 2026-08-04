"""
A qual linha do CSV cada frame BRUTO do Google Earth realmente corresponde?

Pergunta concreta que motivou este script: o Movie Maker grava o primeiro
render duas vezes (`-000000.png` e `-000001.png` são byte-idênticos). O
terceiro arquivo gravado (`-000002.png`) corresponde à SEGUNDA ou à TERCEIRA
coordenada do CSV? Isso separa dois modelos:

  - relógio do tour PAUSADO durante a escrita duplicada
        -> `-00000k.png` mostra t = k-1  -> a rotulagem atual está certa
  - relógio do tour CORRENDO durante a escrita duplicada
        -> `-00000k.png` mostra t = k    -> tudo a partir do índice 1 está
           um frame adiantado (o lag medido em varredura_lag_captura.py)

Método: recorta o frame bruto centrado no nadir, leva a norte-acima e mede
por correlação de fase o resíduo contra a janela satelital georreferenciada
centrada em CADA linha candidata do CSV. A linha certa é a que zera o resíduo.

Só funciona em rota que ainda tenha o `-000000.png` bruto (não apagado).

Uso:
    python identificar_linha_do_frame.py --base-dir "...\\CHAMPAIGN" --mach 1.0 \
        --map "...\\CHAMPAIGN\\map_v2.vrt"
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2 as cv
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))  # visual-odometry/
from utils import (recortar_frame_google_earth, FRAME_RECORTE_PX,  # noqa: E402
                   GE_VIEWPORT_H_FOV_DEG, FRAME_BRUTO_GE)
from comparar_fontes_satelite import (  # noqa: E402
    janela_norte_acima, frame_norte_acima, offset_m,
)

N_PX = 480


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", required=True)
    p.add_argument("--mach", type=float, default=1.0)
    p.add_argument("--map", required=True)
    p.add_argument("--n-frames", type=int, default=6,
                   help="Quantos frames brutos do início testar")
    p.add_argument("--n-linhas", type=int, default=8,
                   help="Quantas linhas do CSV oferecer como candidatas")
    args = p.parse_args()

    base_dir = Path(args.base_dir)
    route_dir = base_dir / f"mach_{args.mach}"
    raw_dir = route_dir / "frames"
    df = pd.read_csv(next(route_dir.glob("Coord-Heading-Elev_*.csv")))

    brutos = sorted(
        (int(re.search(r"(\d+)\.png$", f.name).group(1)), f)
        for f in raw_dir.glob("-*.png") if re.search(r"(\d+)\.png$", f.name)
    )[:args.n_frames]
    if not brutos:
        sys.exit(f"[erro] Nenhum bruto em {raw_dir}")
    if brutos[0][0] != 0:
        print(f"[aviso] O primeiro bruto é '-{brutos[0][0]:06d}.png', não "
              "'-000000.png' — esta rota já teve o frame inicial apagado, "
              "então o teste perde justamente o caso de interesse.")

    fx = FRAME_BRUTO_GE[1] / 2.0 / np.tan(np.deg2rad(GE_VIEWPORT_H_FOV_DEG) / 2.0)
    gsd_voo = float(df["Altura"].min()) / fx
    gsd = gsd_voo * FRAME_RECORTE_PX / N_PX

    ds = rasterio.open(args.map)
    to_dst = Transformer.from_crs(CRS.from_epsg(4326), ds.crs, always_xy=True)
    han = cv.createHanningWindow((N_PX, N_PX), cv.CV_32F)

    n_lin = min(args.n_linhas, len(df))
    print(f"\n{'='*88}")
    print(f"[FRAME BRUTO -> LINHA DO CSV] {base_dir.name} | mach {args.mach}")
    print(f"  GSD do voo {gsd_voo:.3f} m/px | janela {gsd_voo*FRAME_RECORTE_PX:.0f} m")
    print(f"  Convenção atual do pipeline: '-00000k.png' <-> linha k-1 do CSV")
    print(f"{'='*88}")
    # A janela satelital só depende da linha do CSV — extrair uma vez e
    # reusar entre os frames (a reprojeção é o passo caro).
    sat_cache = []
    for j in range(n_lin):
        row = df.iloc[j]
        x0, y0 = to_dst.transform(float(row["Long"]), float(row["Lat"]))
        sat_cache.append(janela_norte_acima(ds, ds.crs, x0, y0, gsd, N_PX))

    print("\n|residuo| em metros — menor valor da linha = coordenada verdadeira\n")
    print("bruto".ljust(14) + "".join(f"linha{j:>2}".rjust(9) for j in range(n_lin)))
    print("-" * (14 + 9 * n_lin))

    achados = []
    for k, caminho in brutos:
        raw = cv.imread(str(caminho))
        if raw is None or raw.shape[:2] != FRAME_BRUTO_GE:
            print(f"[aviso] {caminho.name} não é {FRAME_BRUTO_GE} — pulando")
            continue
        rec = recortar_frame_google_earth(raw)

        vals = []
        for j in range(n_lin):
            row = df.iloc[j]
            aer = frame_norte_acima(rec, float(row["Proa"]), gsd_voo,
                                    (rec.shape[0] / 2.0, rec.shape[1] / 2.0), gsd, N_PX)
            dE, dN, resp = offset_m(aer, sat_cache[j], gsd, han)
            vals.append((float(np.hypot(dE, dN)), resp))

        # Confiança = quão isolado está o mínimo. Champaign é malha urbana
        # regular + talhões agrícolas, ou seja MUITO auto-similar: uma busca
        # discreta sobre dezenas de linhas encontra mínimos espúrios com
        # facilidade. Sem essa razão, um "melhor" de 14 m sobre um segundo
        # melhor de 16 m parece tão bom quanto um de 3 m sobre 335 m.
        mags = sorted(v[0] for v in vals)
        melhor = int(np.argmin([v[0] for v in vals]))
        razao = (mags[1] / mags[0]) if mags[0] > 0 else np.inf
        achados.append((k, melhor, vals[melhor][0], razao))
        linha = f"-{k:06d}.png".ljust(14)
        for j, (m, _) in enumerate(vals):
            marca = "*" if j == melhor else " "
            linha += f"{m:8.0f}{marca}"
        print(linha)

    RAZAO_MIN = 5.0
    print("\n" + "=" * 88)
    print(f"CONCLUSÃO (só frames com mínimo isolado: 2o melhor >= {RAZAO_MIN:.0f}x o melhor)")
    print("=" * 88)
    confiaveis = []
    for k, j, m, razao in achados:
        atual = k - 1
        if razao < RAZAO_MIN:
            print(f"  -{k:06d}.png  DESCARTADO — mínimo não é isolado "
                  f"(melhor {m:.0f} m, 2o melhor só {razao:.1f}x) ")
            continue
        confiaveis.append((k, j))
        veredito = "confere" if j == atual else f"DESLOCADO {j - atual:+d}"
        print(f"  -{k:06d}.png  -> linha {j} do CSV (residuo {m:.0f} m, "
              f"isolamento {razao:.0f}x) | convenção atual diz linha {atual} "
              f"-> {veredito}")

    deslocs = [j - (k - 1) for k, j in confiaveis if k >= 1]
    if deslocs:
        from collections import Counter
        c = Counter(deslocs)
        print(f"\n  {len(confiaveis)} de {len(achados)} frames tiveram mínimo isolado.")
        print(f"  deslocamento predominante (frames k>=1): "
              f"{c.most_common(1)[0][0]:+d}  {dict(c)}")
    else:
        print("\n  Nenhum frame com mínimo isolado — busca discreta não é "
              "conclusiva nesta rota/trecho (terreno auto-similar). Use a "
              "varredura local de delta (varredura_lag_captura.py).")


if __name__ == "__main__":
    main()
