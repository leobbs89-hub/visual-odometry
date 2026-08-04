"""
Varredura de GSD do mapa base: testa a hipótese "uma fonte de satélite com
GSD mais grosseiro AJUDA o map matching?" SEM trocar de fonte nem de altitude.

Para cada GSD-alvo em {baseline, 1, 3, 5, 10} m/px, roda o map matching
isolado (testar_map_matching_puro) na MESMA rota/voo, degradando o mapa base
para aquele GSD antes do matching (map_matching.base_map_degrade_gsd). Isola
o efeito da resolução da fonte do efeito da altitude — que numa troca real de
fonte (NAIP 0.3m → Sentinel-2 10m) vêm confundidos.

Hipótese que motiva o teste: se o gap residual vem de detalhes finos que
DIFEREM entre Google Earth e satélite, borrar esses detalhes (fonte mais
grosseira) poderia deixar só as estruturas grandes que casam melhor,
reduzindo o erro. Ou o contrário: menos detalhe = menos inliers = pior.

Métrica: erro médio de posição vs GPS condicionado a MAP_OK=True, na variante
yaw_real (oráculo de proa) — o piso de precisão do matching+georreferência.

Uso:
    python varredura_gsd_base.py --mach 1.0 --abs-detector AKAZE SUPERPOINT \
        --base-dir "...\\CHAMPAIGN" --map "...\\CHAMPAIGN\\map_v2.vrt"
"""

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from testar_map_matching_puro import testar_abs_detector
from rodar_teste_map_matching import BASE

GSDS_DEFAULT = [None, 1.0, 3.0, 5.0, 10.0]  # None = baseline (resolução nativa)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mach", type=float, default=1.0)
    p.add_argument("--abs-detector", nargs="+", default=["AKAZE", "SUPERPOINT"],
                    choices=["ORB", "AKAZE", "SIFT", "SUPERPOINT", "LOFTR", "MATCHFORMER", "ROMA"])
    p.add_argument("--base-dir", type=Path, default=BASE)
    p.add_argument("--map", type=Path, default=None)
    p.add_argument("--roi-margin-factor", type=float, default=1.3)
    p.add_argument("--gsds", type=float, nargs="+", default=None,
                    help="GSDs-alvo (m/px). Default: 1 3 5 10 (+ baseline nativo).")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    base_dir = args.base_dir
    if args.map is not None:
        map_tif = args.map
    else:
        cand = base_dir / "map_v2.vrt"
        map_tif = cand if cand.exists() else base_dir / "map.tif"
    if not map_tif.exists():
        sys.exit(f"[erro] Mapa base não encontrado: {map_tif}")

    gsds = GSDS_DEFAULT if args.gsds is None else ([None] + list(args.gsds))
    out_dir = args.out_dir if args.out_dir is not None else base_dir / f"mach_{args.mach}" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    resultados = []  # (abs_det, gsd, erro_ok_yr, pct_ok_yr, erro_ok_ba, pct_ok_ba)
    for abs_det in args.abs_detector:
        for gsd in gsds:
            rotulo = "nativo" if gsd is None else f"{gsd:g}m"
            print(f"\n########## {abs_det} | GSD base = {rotulo} ##########")
            try:
                dfr = testar_abs_detector(args.mach, abs_det, base_dir, map_tif,
                                          args.roi_margin_factor, degrade_base_gsd=gsd)
            except Exception:
                print(f"[ERRO] {abs_det} GSD={rotulo}")
                traceback.print_exc()
                continue
            tag = "nativo" if gsd is None else f"gsd{gsd:g}"
            dfr.to_csv(out_dir / f"mapmatch_puro_{abs_det}_mach_{args.mach}_{tag}.csv", index=False)
            erro_ok_yr = dfr.loc[dfr["MAP_OK_YAW_REAL"], "ERRO_YAW_REAL(m)"].mean()
            pct_ok_yr = 100.0 * dfr["MAP_OK_YAW_REAL"].mean()
            erro_ok_ba = dfr.loc[dfr["MAP_OK_BUSCA_ANGULO"], "ERRO_BUSCA_ANGULO(m)"].mean()
            pct_ok_ba = 100.0 * dfr["MAP_OK_BUSCA_ANGULO"].mean()
            resultados.append((abs_det, rotulo, erro_ok_yr, pct_ok_yr, erro_ok_ba, pct_ok_ba))

    print(f"\n{'='*88}\nVARREDURA DE GSD DO MAPA BASE — {base_dir.name} | mach {args.mach}\n{'='*88}")
    print("ErroOK = erro médio (m) só nos frames com MAP_OK=True | yr=yaw_real (oráculo) | ba=busca_angulo")
    print(f"{'AbsDet':<12}{'GSD_base':>10}{'ErroOK_yr':>12}{'MAP_OK%_yr':>12}{'ErroOK_ba':>12}{'MAP_OK%_ba':>12}")
    for abs_det, rotulo, e_yr, ok_yr, e_ba, ok_ba in resultados:
        e_yr_s = f"{e_yr:.2f}" if e_yr == e_yr else "nan"
        e_ba_s = f"{e_ba:.2f}" if e_ba == e_ba else "nan"
        print(f"{abs_det:<12}{rotulo:>10}{e_yr_s:>12}{ok_yr:>11.1f}%{e_ba_s:>12}{ok_ba:>11.1f}%")
    print(f"\nCSVs em: {out_dir}")


if __name__ == "__main__":
    main()
