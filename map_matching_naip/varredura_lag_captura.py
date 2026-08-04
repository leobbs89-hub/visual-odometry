"""
Varredura do lag de captura (delta, em frames) entre a foto de voo gravada no
Google Earth e a linha do CSV de ground truth a que ela é atribuída.

Hipótese testada (causa B de Tese/investigacao_offset_google_earth_causa_raiz.md):
a imagem salva como "frame N" não corresponde à posição da linha N do CSV, e
sim à de N+delta. Com a correção do centro já aplicada (causa A), o resíduo
medido entre foto de voo e mapa base satelital é ~1 espaçamento de frame, o
que aponta delta ~ 1.

Método: para cada delta candidato, reinterpola posição/proa/altura em
t = N + delta, monta a janela satelital norte-acima CENTRADA nessa posição
interpolada, e mede por correlação de fase o resíduo contra a foto de voo
(também levada a norte-acima pela proa interpolada). O delta correto é o que
zera o resíduo.

Não depende de nenhuma captura nova — é só reinterpretação do CSV existente.

Saídas:
  - varredura por delta: |resíduo| médio e componente ao longo da proa
  - delta implícito por frame (along / espaçamento) + ajuste linear, para
    testar se o lag é CONSTANTE ou tem deriva temporal ao longo da rota

Uso:
    python varredura_lag_captura.py --base-dir "...\\CHAMPAIGN" --mach 1.0 \
        --map "...\\CHAMPAIGN\\map_v2.vrt"
"""

import argparse
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
from utils import FRAME_RECORTE_PX, GE_VIEWPORT_H_FOV_DEG, FRAME_BRUTO_GE  # noqa: E402
from comparar_fontes_satelite import (  # noqa: E402
    janela_norte_acima, frame_norte_acima, offset_m,
)

N_PX = 480


def _interp_ang(a0, a1, f):
    """Interpola ângulo em graus pelo caminho curto (evita salto 359->0)."""
    d = (a1 - a0 + 180.0) % 360.0 - 180.0
    return (a0 + f * d) % 360.0


def amostrar(df, t):
    """Estado (lat, lon, proa, altura) no instante fracionário t, em frames."""
    n = len(df)
    if t <= 0:
        i0, f = 0, 0.0
    elif t >= n - 1:
        i0, f = n - 2, 1.0
    else:
        i0 = int(np.floor(t))
        f = t - i0
    a, b = df.iloc[i0], df.iloc[i0 + 1]
    return (
        float(a["Lat"]) + f * (float(b["Lat"]) - float(a["Lat"])),
        float(a["Long"]) + f * (float(b["Long"]) - float(a["Long"])),
        _interp_ang(float(a["Proa"]), float(b["Proa"]), f),
        float(a["Altura"]) + f * (float(b["Altura"]) - float(a["Altura"])),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", required=True)
    p.add_argument("--mach", type=float, default=1.0)
    p.add_argument("--map", required=True)
    p.add_argument("--deltas", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    p.add_argument("--delta-linear", type=float, nargs=2, metavar=("A", "B"),
                   action="append", default=None,
                   help="Avalia também o modelo com deriva delta(N) = A + B*N. "
                        "Repetível. Para validação cruzada, passe o ajuste "
                        "obtido em OUTRA rota — ajustar e avaliar na mesma "
                        "rota é circular.")
    p.add_argument("--resp-min", type=float, default=0.05,
                   help="Descarta frames cuja correlação de fase não achou pico")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    base_dir = Path(args.base_dir)
    route_dir = base_dir / f"mach_{args.mach}"
    resized_dir = route_dir / "frames" / "Resized"
    csv_cand = list(route_dir.glob("Coord-Heading-Elev_*.csv"))
    if not csv_cand:
        sys.exit(f"[erro] Nenhum CSV em {route_dir}")
    df = pd.read_csv(csv_cand[0])

    fx = FRAME_BRUTO_GE[1] / 2.0 / np.tan(np.deg2rad(GE_VIEWPORT_H_FOV_DEG) / 2.0)
    gsd_voo = float(df["Altura"].min()) / fx
    janela_m = gsd_voo * FRAME_RECORTE_PX
    gsd = janela_m / N_PX

    ds = rasterio.open(args.map)
    to_dst = Transformer.from_crs(CRS.from_epsg(4326), ds.crs, always_xy=True)
    han = cv.createHanningWindow((N_PX, N_PX), cv.CV_32F)

    # Espaçamento real entre frames consecutivos (m), para converter o resíduo
    # ao longo da proa em "quantos frames de atraso".
    g_lat = 111320.0
    esp = []
    for i in range(len(df) - 1):
        dlat = (df["Lat"].iloc[i + 1] - df["Lat"].iloc[i]) * g_lat
        dlon = ((df["Long"].iloc[i + 1] - df["Long"].iloc[i]) * g_lat
                * np.cos(np.deg2rad(df["Lat"].iloc[i])))
        esp.append(float(np.hypot(dlat, dlon)))
    esp.append(esp[-1])

    print(f"\n{'='*84}")
    print(f"[VARREDURA DE LAG] {base_dir.name} | mach {args.mach} | {len(df)} frames")
    print(f"  fx={fx:.4f} px | GSD do voo {gsd_voo:.3f} m/px | janela {janela_m:.0f} m "
          f"| espaçamento medio {np.mean(esp):.1f} m/frame")
    print(f"{'='*84}")

    imgs = {}
    for idx in range(len(df)):
        im = cv.imread(str(resized_dir / df.iloc[idx]["File"]))
        if im is not None:
            imgs[idx] = im
    if not imgs:
        sys.exit(f"[erro] Nenhum frame lido de {resized_dir}")

    # Cada "modelo" é um rótulo + uma função frame -> delta. O delta constante
    # é só o caso particular B=0.
    modelos = [(d, (lambda n, d=d: d)) for d in args.deltas]
    for a, b in (args.delta_linear or []):
        modelos.append((f"lin {a:.3f}{b:+.5f}N", lambda n, a=a, b=b: a + b * n))

    linhas = []
    for rotulo, fdelta in modelos:
        for idx, im in imgs.items():
            delta = fdelta(idx)
            t = idx + delta
            if t > len(df) - 1:
                continue
            lat, lon, proa, _ = amostrar(df, t)
            x0, y0 = to_dst.transform(lon, lat)
            sat = janela_norte_acima(ds, ds.crs, x0, y0, gsd, N_PX)
            aer = frame_norte_acima(im, proa, gsd_voo,
                                    (im.shape[0] / 2.0, im.shape[1] / 2.0), gsd, N_PX)
            dE, dN, resp = offset_m(aer, sat, gsd, han)
            along = dN * np.cos(np.deg2rad(proa)) + dE * np.sin(np.deg2rad(proa))
            cross = dE * np.cos(np.deg2rad(proa)) - dN * np.sin(np.deg2rad(proa))
            linhas.append({
                "MODELO": rotulo, "DELTA": delta, "FRAME": idx, "PROA": proa,
                "ALONG_M": along, "CROSS_M": cross,
                "OFFSET_M": float(np.hypot(dE, dN)), "RESP": resp,
                "ESPACAMENTO_M": esp[min(idx, len(esp) - 1)],
            })

    out = pd.DataFrame(linhas)
    ok = out[out["RESP"] > args.resp_min]

    print(f"\n{'modelo':>18}{'n':>5}{'|resid| medio':>15}{'along medio':>13}"
          f"{'|along| medio':>15}{'desvio along':>14}")
    print("-" * 81)
    resumo = []
    for rotulo, _ in modelos:
        s = ok[ok["MODELO"] == rotulo]
        if s.empty:
            continue
        resumo.append({
            "modelo": rotulo, "n": len(s),
            "resid_medio": s["OFFSET_M"].mean(),
            "along_medio": s["ALONG_M"].mean(),
            "abs_along_medio": s["ALONG_M"].abs().mean(),
            "along_desvio": s["ALONG_M"].std(),
        })
        r = resumo[-1]
        rot = f"{rotulo:.2f}" if isinstance(rotulo, float) else str(rotulo)
        print(f"{rot:>18}{r['n']:>5}{r['resid_medio']:>15.1f}"
              f"{r['along_medio']:>13.1f}{r['abs_along_medio']:>15.1f}"
              f"{r['along_desvio']:>14.1f}")

    res = pd.DataFrame(resumo)
    melhor = res.loc[res["abs_along_medio"].idxmin()]
    print(f"\n  melhor modelo pela componente ao longo da proa: "
          f"{melhor['modelo']} (|along| medio {melhor['abs_along_medio']:.1f} m)")

    # Delta implícito por frame, a partir da medição em delta=0: quantos frames
    # de deslocamento o conteúdo da imagem está adiantado.
    base = ok[ok["MODELO"] == 0.0].copy()
    if len(base) >= 3:
        base["DELTA_IMPLICITO"] = base["ALONG_M"] / base["ESPACAMENTO_M"]
        coef = np.polyfit(base["FRAME"], base["DELTA_IMPLICITO"], 1)
        print(f"\n  delta implícito por frame (medido em delta=0, {len(base)} frames):")
        print(f"    media {base['DELTA_IMPLICITO'].mean():.3f} frames  "
              f"(mediana {base['DELTA_IMPLICITO'].median():.3f}, "
              f"desvio {base['DELTA_IMPLICITO'].std():.3f})")
        print(f"    ajuste linear: delta(N) = {coef[1]:.3f} + {coef[0]:+.5f}*N")
        print(f"    -> {'CONSTANTE' if abs(coef[0]) < 0.005 else 'COM DERIVA'} "
              f"ao longo da rota (limiar 0.005 frames/frame)")

    out_path = Path(args.out) if args.out else route_dir / "results" / "varredura_lag_captura.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
