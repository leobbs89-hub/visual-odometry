"""
Mosaico de conferência: MESMO ponto do solo visto por N fontes de satélite
independentes + a foto de voo do Google Earth, todos reprojetados para a
MESMA janela norte-acima, MESMO GSD e MESMA cobertura em metros.

Objetivo (duas perguntas separadas, que a sonda de desalinhamento não separa):

  (1) As fontes de satélite concordam ENTRE SI na georreferência?
      Se NAIP × ESRI × Sentinel-2 casam dentro de poucos metros, a
      georreferência do lado satelital está sadia e o offset sistemático
      medido em `sonda_desalinhamento.py` só pode vir do lado Google Earth
      (posição/instante/centro do frame), não do mapa base.

  (2) O recorte salvo em Resized/ está centrado no nadir?
      Os PNGs brutos gravados no Google Earth são 700x640, e o nadir (= a
      coordenada GPS do CSV, já que o tour usa tilt=0) cai no centro do
      viewport, linha 350. O recorte antigo (`img[:640, :]`, que descartava
      só o rodapé) deixava esse nadir na linha 350 de um frame de 640 linhas,
      cujo centro é a 320 — 30 px de erro ao longo da proa, herdado por todo
      o pipeline. Este script mede isso comparando duas colunas:
        - "GEresized" : o arquivo REAL de frames/Resized/, tratado como o
                        pipeline trata (centro = centro da imagem). Valida o
                        recorte que estiver em vigor, sem reimplementá-lo.
        - "GEraw"     : o frame bruto 700x640 com centro em (320,350), o
                        nadir verdadeiro — referência.
      Com o recorte correto as duas colunas devem coincidir; com o recorte
      antigo, "GEresized" fica ~30 px * GSD adiantada em relação a "GEraw".

Tudo é norte-acima de propósito: assim o offset medido sai direto em
metros de Leste/Norte, sem depender de rotação por proa.

Uso:
    python comparar_fontes_satelite.py \
        --base-dir "...\\CHAMPAIGN" --mach 1.0 \
        --fonte NAIP="...\\CHAMPAIGN\\map_v2.vrt" \
        --fonte ESRI="...\\CHAMPAIGN\\map_esri.tif" \
        --fonte S2="...\\CHAMPAIGN_ALT3500\\map_s2.vrt" \
        --n-frames 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2 as cv
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from affine import Affine
from pyproj import Transformer

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Frames brutos do Google Earth: 700 linhas x 640 colunas; o nadir (centro
# óptico do viewport, com tilt=0) fica na linha 350.
RAW_H, RAW_W = 700, 640


def _parse_fonte(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--fonte espera NOME=caminho, recebi '{s}'")
    nome, caminho = s.split("=", 1)
    return nome.strip(), caminho.strip()


def janela_norte_acima(ds, crs_dst, x0, y0, gsd, n_px):
    """Reamostra `ds` numa janela quadrada norte-acima de n_px, centrada em
    (x0, y0) no CRS de destino, com resolução `gsd` m/px."""
    meia = n_px / 2.0 * gsd
    T = Affine.translation(x0 - meia, y0 + meia) * Affine.scale(gsd, -gsd)
    n_bandas = min(3, ds.count)
    dst = np.zeros((n_bandas, n_px, n_px), dtype=np.uint8)
    for i in range(n_bandas):
        reproject(
            source=rasterio.band(ds, i + 1),
            destination=dst[i],
            dst_transform=T,
            dst_crs=crs_dst,
            resampling=Resampling.bilinear,
        )
    img = np.transpose(dst, (1, 2, 0))
    if n_bandas == 1:
        img = cv.cvtColor(img[:, :, 0], cv.COLOR_GRAY2BGR)
    else:  # bandas do raster são RGB; OpenCV trabalha em BGR
        img = img[:, :, ::-1].copy()
    return img


def frame_norte_acima(img_raw, proa, gsd_voo, centro_yx, gsd, n_px):
    """Leva o frame de voo (proa-acima) para norte-acima, no mesmo GSD e
    tamanho da janela satelital, colocando `centro_yx` no centro do destino.

    heading-up -> north-up = rotacionar o conteúdo no sentido horário pela
    proa, isto é, ângulo NEGATIVO na convenção do getRotationMatrix2D
    (inverso do que `_rotacionar_imagem` faz em map_matching.py, que leva
    o patch norte-acima para proa-acima com ângulo +proa).
    """
    cy, cx = centro_yx
    escala = gsd_voo / gsd
    M = cv.getRotationMatrix2D((float(cx), float(cy)), -float(proa), escala)
    M[0, 2] += n_px / 2.0 - cx
    M[1, 2] += n_px / 2.0 - cy
    return cv.warpAffine(img_raw, M, (n_px, n_px), flags=cv.INTER_LINEAR)


def _preproc(img):
    """Casa ESTRUTURA, não intensidade — mesma receita da sonda_desalinhamento."""
    g = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    gx = cv.Sobel(g, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(g, cv.CV_32F, 0, 1, ksize=3)
    return cv.magnitude(gx, gy)


def offset_m(img_a, img_b, gsd, han):
    """(dE, dN, resposta) do deslocamento de img_a em relação a img_b.
    Como as duas janelas são norte-acima, +x é Leste e +y é Sul."""
    (dx, dy), resp = cv.phaseCorrelate(_preproc(img_a), _preproc(img_b), han)
    return dx * gsd, -dy * gsd, resp


def _rotulo(img, texto, cor=(0, 255, 255)):
    img = img.copy()
    cv.rectangle(img, (0, img.shape[0] - 20), (img.shape[1], img.shape[0]), (20, 20, 20), -1)
    cv.putText(img, texto, (6, img.shape[0] - 6), cv.FONT_HERSHEY_SIMPLEX,
               0.42, cor, 1, cv.LINE_AA)
    h, w = img.shape[:2]
    cv.drawMarker(img, (w // 2, h // 2), (0, 0, 255), cv.MARKER_CROSS, 26, 2)
    cv.circle(img, (w // 2, h // 2), 12, (0, 0, 255), 2)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", required=True, help="Pasta da rota (contém mach_X/)")
    p.add_argument("--mach", type=float, default=1.0)
    p.add_argument("--fonte", action="append", type=_parse_fonte, required=True,
                   metavar="NOME=CAMINHO", help="Fonte de mapa base; repetir para várias")
    p.add_argument("--n-frames", type=int, default=5)
    p.add_argument("--janela-m", type=float, default=None,
                   help="Lado da janela em metros (default: cobertura da câmera)")
    p.add_argument("--h-fov", type=float, default=60.0)
    p.add_argument("--sensor-px", type=int, default=640)
    p.add_argument("--saida", default=None)
    args = p.parse_args()

    base_dir = Path(args.base_dir)
    route_dir = base_dir / f"mach_{args.mach}"
    raw_dir = route_dir / "frames"
    resized_dir = raw_dir / "Resized"
    csv_cand = list(route_dir.glob("Coord-Heading-Elev_*.csv"))
    if not csv_cand:
        sys.exit(f"[erro] Nenhum CSV em {route_dir}")
    df = pd.read_csv(csv_cand[0])

    fx = args.sensor_px / 2.0 / np.tan(np.deg2rad(args.h_fov) / 2.0)
    agl = float(df["Altura"].min())
    gsd_voo = agl / fx  # m/px no frame BRUTO (mesma escala do 640 cortado)
    janela_m = args.janela_m if args.janela_m else gsd_voo * args.sensor_px
    n_px = 480
    gsd = janela_m / n_px

    print(f"\n{'='*88}")
    print(f"[COMPARA FONTES] {base_dir.name} | mach {args.mach} | AGL min {agl:.0f} m")
    print(f"  fx={fx:.1f} px | GSD do voo {gsd_voo:.3f} m/px | janela {janela_m:.0f} m "
          f"| destino {n_px}px @ {gsd:.3f} m/px")
    amostra = next(iter(sorted(resized_dir.glob("*.png"))), None)
    if amostra is None:
        sys.exit(f"[erro] Nenhum frame em {resized_dir}")
    h_rs, w_rs = cv.imread(str(amostra)).shape[:2]
    desloc_px = (RAW_H / 2.0) - (h_rs / 2.0) if h_rs != RAW_H else 0.0
    print(f"  Resized/ atual: {w_rs}x{h_rs} | nadir do bruto na linha {RAW_H/2:.0f} | "
          f"se o recorte for do TOPO, erro = {desloc_px:.0f} px = {desloc_px*gsd_voo:.1f} m")
    print(f"{'='*88}")

    fontes = {}
    for nome, caminho in args.fonte:
        if not Path(caminho).exists():
            print(f"[aviso] fonte '{nome}' não encontrada: {caminho} — pulando")
            continue
        fontes[nome] = rasterio.open(caminho)
        print(f"  fonte {nome:<6} {fontes[nome].crs} {fontes[nome].res[0]:.2f} m/px  {caminho}")
    if not fontes:
        sys.exit("[erro] Nenhuma fonte válida.")

    crs_dst = next(iter(fontes.values())).crs
    to_dst = Transformer.from_crs(CRS.from_epsg(4326), crs_dst, always_xy=True)
    han = cv.createHanningWindow((n_px, n_px), cv.CV_32F)

    n = len(df)
    idxs = list(range(n)) if args.n_frames >= n else \
        sorted(set(np.linspace(0, n - 1, args.n_frames).round().astype(int).tolist()))

    linhas_csv, linhas_mosaico = [], []
    ref_nome = next(iter(fontes))

    for idx in idxs:
        row = df.iloc[idx]
        raw = cv.imread(str(raw_dir / row["File"]))
        if raw is None or raw.shape[:2] != (RAW_H, RAW_W):
            print(f"[aviso] frame {row['File']} ausente ou não é {RAW_H}x{RAW_W} — pulando")
            continue

        x0, y0 = to_dst.transform(float(row["Long"]), float(row["Lat"]))
        paineis, reg = [], {}
        for nome, ds in fontes.items():
            img = janela_norte_acima(ds, crs_dst, x0, y0, gsd, n_px)
            reg[nome] = img
            paineis.append(_rotulo(img, f"{nome} (satelite)"))

        proa = float(row["Proa"])
        rs = cv.imread(str(resized_dir / row["File"]))
        if rs is None:
            print(f"[aviso] {row['File']} ausente em Resized/ — pulando")
            continue
        # Tratado como o pipeline trata: o centro da imagem É a posição GPS.
        ge_resized = frame_norte_acima(rs, proa, gsd_voo,
                                       (rs.shape[0] / 2.0, rs.shape[1] / 2.0), gsd, n_px)
        ge_raw = frame_norte_acima(raw, proa, gsd_voo,
                                   (RAW_H / 2.0, RAW_W / 2.0), gsd, n_px)
        paineis.append(_rotulo(ge_resized, f"GE Resized/ {rs.shape[1]}x{rs.shape[0]} (pipeline)", (0, 200, 255)))
        paineis.append(_rotulo(ge_raw, "GE nadir real (bruto 700)", (0, 255, 0)))

        ref = reg[ref_nome]
        med = {"FRAME": idx, "LAT": row["Lat"], "LON": row["Long"], "PROA": proa}
        partes = []
        for nome, img in reg.items():
            if nome == ref_nome:
                continue
            dE, dN, r = offset_m(img, ref, gsd, han)
            med[f"{nome}_vs_{ref_nome}_dE"] = dE
            med[f"{nome}_vs_{ref_nome}_dN"] = dN
            med[f"{nome}_vs_{ref_nome}_m"] = float(np.hypot(dE, dN))
            med[f"{nome}_vs_{ref_nome}_resp"] = r
            partes.append(f"{nome}x{ref_nome} {np.hypot(dE, dN):6.1f}m(r={r:.2f})")
        for tag, img in (("GEresized", ge_resized), ("GEraw", ge_raw)):
            dE, dN, r = offset_m(img, ref, gsd, han)
            # componente ao longo da proa: positivo = frame "adiantado"
            along = dN * np.cos(np.deg2rad(proa)) + dE * np.sin(np.deg2rad(proa))
            med[f"{tag}_dE"], med[f"{tag}_dN"] = dE, dN
            med[f"{tag}_along"] = along
            med[f"{tag}_m"] = float(np.hypot(dE, dN))
            med[f"{tag}_resp"] = r
            partes.append(f"{tag} {np.hypot(dE, dN):6.1f}m along={along:7.1f}(r={r:.2f})")
        linhas_csv.append(med)
        print(f"  frame {idx:>3} | " + " | ".join(partes))

        sep = np.full((n_px, 5, 3), 90, dtype=np.uint8)
        par = []
        for i, pan in enumerate(paineis):
            if i:
                par.append(sep)
            par.append(pan)
        par = np.hstack(par)
        faixa = np.full((26, par.shape[1], 3), 30, dtype=np.uint8)
        cv.putText(faixa, f"Frame {idx} ({row['File']})  Lat={row['Lat']:.5f} "
                          f"Lon={row['Long']:.5f}  Proa={proa:.1f}deg  "
                          f"janela {janela_m:.0f}m  (norte p/ cima)",
                   (8, 18), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        linhas_mosaico.append(np.vstack([faixa, par]))

    if not linhas_mosaico:
        sys.exit("[erro] Nenhum frame processado.")

    sepl = np.full((5, linhas_mosaico[0].shape[1], 3), 60, dtype=np.uint8)
    partes = []
    for i, l in enumerate(linhas_mosaico):
        if i:
            partes.append(sepl)
        partes.append(l)
    mosaico = np.vstack(partes)

    saida = Path(args.saida) if args.saida else base_dir / "comparacao_fontes_satelite.png"
    cv.imwrite(str(saida), mosaico)
    out = pd.DataFrame(linhas_csv)
    csv_out = saida.with_suffix(".csv")
    out.to_csv(csv_out, index=False)

    print(f"\n{'='*88}\nRESUMO ({len(out)} frames) — referência: {ref_nome}\n{'='*88}")
    for nome in fontes:
        if nome == ref_nome:
            continue
        c = f"{nome}_vs_{ref_nome}_m"
        print(f"  satelite x satelite  {nome:>6} x {ref_nome:<6} "
              f"|offset| medio {out[c].mean():7.1f} m  (mediana {out[c].median():6.1f}, "
              f"resp {out[f'{nome}_vs_{ref_nome}_resp'].mean():.2f})")
    for tag in ("GEresized", "GEraw"):
        print(f"  voo x satelite       {tag:<15} |offset| medio {out[f'{tag}_m'].mean():7.1f} m  "
              f"| ao longo da proa medio {out[f'{tag}_along'].mean():7.1f} m "
              f"(desvio {out[f'{tag}_along'].std():6.1f}, resp {out[f'{tag}_resp'].mean():.2f})")
    # Só frames com correlação confiável nas DUAS colunas — a média cega é
    # dominada por frames onde a correlação de fase não achou pico.
    ok = (out["GEresized_resp"] > 0.05) & (out["GEraw_resp"] > 0.05)
    delta = (out.loc[ok, "GEresized_along"] - out.loc[ok, "GEraw_along"]).mean()
    print(f"\n  descentragem do Resized/ vs nadir real: {delta:.1f} m "
          f"({delta/gsd_voo:.1f} px, {ok.sum()}/{len(out)} frames com resp>0.05)")
    print("  -> proximo de 0 = recorte centrado no nadir, correto.")
    print(f"\n-> {saida}\n-> {csv_out}")


if __name__ == "__main__":
    main()
