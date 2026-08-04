"""
Sonda de desalinhamento da malha (Google Earth) × mapa base satelital.

Para cada frame, prepara a foto de voo e o patch satelital georreferenciado
CENTRADO NA POSIÇÃO GPS REAL do frame (rotacionado pela Proa real), do mesmo
jeito que o map matching prepara — mas SEM matching/homografia. Em vez disso,
mede por CORRELAÇÃO DE FASE (cv.phaseCorrelate) o deslocamento translacional
residual (dx, dy) em pixels entre as duas imagens, e converte para metros via
o GSD do voo.

Objetivo: quantificar a suspeita de que a malha 3D do Google Earth e o mapa
base de satélite (nadir) NÃO coincidem — se houver um offset sistemático e
direcional (não só ruído aleatório frame a frame), isso é evidência de
desalinhamento de fonte, e explica parte do erro residual do map matching que
nenhuma troca de matcher/estratégia fechou.

A correlação de fase mede só translação; para reduzir a sensibilidade ao gap
de tonalidade/brilho entre as fontes (que degradaria o pico de correlação),
por padrão as duas imagens passam por CLAHE + gradiente Sobel antes da
correlação (casa estrutura, não intensidade). Desligável com --sem-preproc.

Saída: CSV por frame (dx, dy, |offset| em m, resposta do pico) + um resumo
com offset médio/mediano, o VETOR médio (dx, dy) — que revela viés
direcional — e o desvio padrão.

Uso:
    python sonda_desalinhamento.py --base-dir "...\\CHAMPAIGN" --mach 1.0 \
        --map "...\\CHAMPAIGN\\map_v2.vrt"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2 as cv

# Console do Windows é cp1252 por padrão e quebra em caracteres fora dele
# (ex. seta '->'); força utf-8 com fallback pra não derrubar o resumo final.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))  # visual-odometry/
from gerar_mosaico_diagnostico import ConstrutorDePatch  # reusa o stub da mixin

_TARGET = 480


def _para_gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _grad(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=3)
    return cv.magnitude(gx, gy)


def _preproc(img, ativo):
    g = _para_gray(img)
    if not ativo:
        return g.astype(np.float32)
    g = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    return _grad(g).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", required=True, help="Pasta da rota (contém mach_X/)")
    parser.add_argument("--mach", type=float, default=1.0)
    parser.add_argument("--map", required=True, help="Caminho do mapa base (.vrt/.tif)")
    parser.add_argument("--roi-margin-factor", type=float, default=1.3)
    parser.add_argument("--h-fov", type=float, default=60.0)
    parser.add_argument("--sensor-px", type=int, default=640)
    parser.add_argument("--sem-preproc", action="store_true",
                         help="Correlaciona intensidade bruta (default: CLAHE+gradiente)")
    parser.add_argument("--out", default=None, help="CSV de saída (default: <base-dir>/mach_X/results/sonda_desalinhamento.csv)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    route_dir = base_dir / f"mach_{args.mach}"
    resized_dir = route_dir / "frames" / "Resized"
    csv_cand = list(route_dir.glob("Coord-Heading-Elev_*.csv"))
    if not csv_cand:
        sys.exit(f"[erro] Nenhum CSV em {route_dir}")
    df = pd.read_csv(csv_cand[0])

    fx = args.sensor_px / 2.0 / np.tan(np.deg2rad(args.h_fov) / 2.0)
    cx = args.sensor_px / 2.0

    construtor = ConstrutorDePatch(args.map, args.roi_margin_factor, fx, cx, cor=False)
    construtor._inicializar_mapa()
    construtor._inicializar_escala(float(df["Altura"].iloc[0]))
    gsd = construtor.gsd_voo_efetivo  # m/px do patch (e da aérea reamostrada p/ 480)
    preproc_on = not args.sem_preproc

    han = cv.createHanningWindow((_TARGET, _TARGET), cv.CV_32F)

    print(f"\n{'='*80}\n[SONDA DESALINHAMENTO] {base_dir.name} | mach {args.mach} | "
          f"GSD={gsd:.3f} m/px | preproc={'CLAHE+grad' if preproc_on else 'bruto'}\n{'='*80}")
    print(f"{'FRAME':<7}{'dx_px':>9}{'dy_px':>9}{'offset_m':>11}{'resp':>8}")

    linhas = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        img_path = resized_dir / row["File"]
        img = cv.imread(str(img_path))
        if img is None:
            continue
        img_a = cv.resize(img, (_TARGET, _TARGET)) if img.shape[:2] != (_TARGET, _TARGET) else img
        patch, _, _, _ = construtor._preparar_patch_satelital(
            lat=float(row["Lat"]), lon=float(row["Long"]),
            angulo_graus=float(row["Proa"]), escala=1.0,
        )
        patch = cv.resize(patch, (_TARGET, _TARGET)) if patch.shape[:2] != (_TARGET, _TARGET) else patch

        a = _preproc(img_a, preproc_on)
        p = _preproc(patch, preproc_on)
        # phaseCorrelate: shift que leva 'a' a casar 'p'. dx>0 => 'a' está
        # deslocada +x em relação a 'p' (convenção do OpenCV).
        (dx, dy), resp = cv.phaseCorrelate(a, p, han)
        offset_m = float(np.hypot(dx, dy) * gsd)

        print(f"{idx:<7}{dx:>9.2f}{dy:>9.2f}{offset_m:>11.2f}{resp:>8.3f}")
        linhas.append({
            "FRAME": idx, "LAT": float(row["Lat"]), "LON": float(row["Long"]),
            "PROA": float(row["Proa"]),
            "DX_PX": dx, "DY_PX": dy,
            "DX_M": dx * gsd, "DY_M": dy * gsd,
            "OFFSET_M": offset_m, "RESP": resp,
        })

    out = pd.DataFrame(linhas)
    out_path = Path(args.out) if args.out else route_dir / "results" / "sonda_desalinhamento.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # Vetor médio (dx,dy) revela viés SISTEMÁTICO; |offset| médio mede magnitude
    # total (inclui ruído). Se o vetor médio ~ |offset| médio, o desalinhamento
    # é direcional (sistemático); se vetor médio << |offset| médio, é ruído.
    dxm, dym = out["DX_M"].mean(), out["DY_M"].mean()
    print(f"\n{'='*80}\nRESUMO — {base_dir.name} | {len(out)} frames | GSD={gsd:.3f} m/px\n{'='*80}")
    print(f"|offset| médio:   {out['OFFSET_M'].mean():.2f} m  (mediano {out['OFFSET_M'].median():.2f} m, "
          f"máx {out['OFFSET_M'].max():.2f} m)")
    print(f"vetor médio (dx,dy): ({dxm:.2f}, {dym:.2f}) m  -> |vetor médio| = {np.hypot(dxm, dym):.2f} m")
    print(f"desvio padrão dx/dy: ({out['DX_M'].std():.2f}, {out['DY_M'].std():.2f}) m")
    print(f"resposta de pico média: {out['RESP'].mean():.3f}  (0=ruído, 1=casamento perfeito)")
    frac = np.hypot(dxm, dym) / out['OFFSET_M'].mean() if out['OFFSET_M'].mean() else 0
    print(f"|vetor médio| / |offset| médio = {frac:.2f}  "
          f"({'SISTEMÁTICO/direcional' if frac > 0.5 else 'majoritariamente RUÍDO'})")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
