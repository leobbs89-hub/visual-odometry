"""
Mosaico de diagnóstico visual: para alguns frames espalhados pela rota,
mostra lado a lado a foto de voo (Google Earth) e o patch satelital
georeferenciado já recortado/reamostrado/rotacionado do jeito que o
map matching realmente usa (_preparar_patch_satelital, mesma função de
map_matching.py) -- não uma aproximação separada. Marca o centro (posição
GPS do frame) em ambos com uma mira, pra checar visualmente se a rotação e
o recorte estão fazendo sentido ANTES de rodar o teste completo (que pode
levar dezenas de minutos).

Não faz nenhum matching/correção -- é só o patch preparado (recorte +
reamostragem pro GSD do voo + rotação pelo heading/Proa do CSV), o mesmo
primeiro passo que _corrigir_posicao_pelo_mapa faz internamente antes de
buscar correspondências.

Uso:
    python gerar_mosaico_diagnostico.py \
        --base-dir "...\CHAMPAIGN_ALT3500" --mach 1.0 \
        --map "...\CHAMPAIGN_ALT3500\map_s2.vrt" \
        --n-frames 5 --saida "...\CHAMPAIGN_ALT3500\diagnostico_mosaico.png"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2 as cv

sys.path.insert(0, str(Path(__file__).parent.parent))  # visual-odometry/ (map_matching.py)
from map_matching import MapMatchingMixin

_TARGET = 480


class _DetectorAbsolutoStub:
    """Só carrega o atributo que _recortar_roi_mapa lê via getattr(..., 'requires_color', False)."""
    def __init__(self, requires_color):
        self.requires_color = requires_color


class ConstrutorDePatch(MapMatchingMixin):
    """
    Instância mínima da MapMatchingMixin, só com os atributos que
    _inicializar_mapa/_inicializar_escala/_preparar_patch_satelital
    precisam (ver docstring da mixin em map_matching.py) -- não roda o
    resto do pipeline de odometria.
    """
    def __init__(self, map_tif_path, roi_margin_factor, fx, cx, cor=True):
        self.config = {"paths": {"map_tif_path": map_tif_path}}
        self.roi_margin_factor = roi_margin_factor
        self.fx = fx
        self.cx = cx
        self.detector_absoluto = _DetectorAbsolutoStub(requires_color=cor)


def _marcar_centro(img, cor=(0, 0, 255), raio=10):
    img = img.copy()
    if img.ndim == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    cv.drawMarker(img, (cx, cy), cor, markerType=cv.MARKER_CROSS,
                   markerSize=raio * 2, thickness=2)
    cv.circle(img, (cx, cy), raio, cor, 2)
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", required=True, help="Pasta da rota (contém mach_X/)")
    parser.add_argument("--mach", type=float, default=1.0)
    parser.add_argument("--map", required=True, help="Caminho do mapa base (.vrt/.tif)")
    parser.add_argument("--n-frames", type=int, default=5, help="Quantos frames espalhar pela rota (default: 5)")
    parser.add_argument("--roi-margin-factor", type=float, default=1.3)
    parser.add_argument("--h-fov", type=float, default=60.0)
    parser.add_argument("--sensor-px", type=int, default=640)
    parser.add_argument("--saida", default=None, help="Caminho do PNG de saída (default: <base-dir>/diagnostico_mosaico.png)")
    args = parser.parse_args()

    base_dir   = Path(args.base_dir)
    route_dir  = base_dir / f"mach_{args.mach}"
    resized_dir = route_dir / "frames" / "Resized"
    csv_candidatos = list(route_dir.glob("Coord-Heading-Elev_*.csv"))
    if not csv_candidatos:
        sys.exit(f"[erro] Nenhum CSV encontrado em {route_dir}")
    csv_path = csv_candidatos[0]

    df = pd.read_csv(csv_path)
    n = len(df)
    if args.n_frames >= n:
        indices = list(range(n))
    else:
        indices = sorted(set(np.linspace(0, n - 1, args.n_frames).round().astype(int).tolist()))

    fx = args.sensor_px / 2.0 / np.tan(np.deg2rad(args.h_fov) / 2.0)
    cx = args.sensor_px / 2.0

    construtor = ConstrutorDePatch(args.map, args.roi_margin_factor, fx, cx, cor=True)
    construtor._inicializar_mapa()
    construtor._inicializar_escala(float(df["Altura"].iloc[0]))

    linhas_mosaico = []
    for idx in indices:
        row = df.iloc[idx]
        fname = row["File"]
        img_path = resized_dir / fname
        img = cv.imread(str(img_path))
        if img is None:
            print(f"[aviso] Frame {fname} não encontrado em {resized_dir}, pulando.")
            continue

        img_aerea = cv.resize(img, (_TARGET, _TARGET)) if img.shape[:2] != (_TARGET, _TARGET) else img
        patch, T_final, M_rot_inv, shape_orig = construtor._preparar_patch_satelital(
            lat=float(row["Lat"]), lon=float(row["Long"]),
            angulo_graus=float(row["Proa"]), escala=1.0,
        )

        img_aerea_m = _marcar_centro(img_aerea)
        patch_m     = _marcar_centro(patch)

        sep = np.full((_TARGET, 6, 3), 128, dtype=np.uint8)
        par = np.hstack([img_aerea_m, sep, patch_m])

        faixa = np.full((30, par.shape[1], 3), 30, dtype=np.uint8)
        cv.putText(
            faixa, f"Frame {idx} ({fname})  |  Lat={row['Lat']:.5f} Lon={row['Long']:.5f}  Proa={row['Proa']:.1f}graus",
            (8, 21), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv.LINE_AA,
        )
        cv.putText(par, "Foto de voo (Google Earth)", (8, _TARGET - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)
        cv.putText(par, "Mapa base georeferenciado (recortado + rotacionado p/ Proa)",
                   (img_aerea_m.shape[1] + 14, _TARGET - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)

        linhas_mosaico.append(np.vstack([faixa, par]))

    if not linhas_mosaico:
        sys.exit("[erro] Nenhum frame processado -- confira --base-dir/--mach.")

    separador_linha = np.full((6, linhas_mosaico[0].shape[1], 3), 60, dtype=np.uint8)
    partes = []
    for i, linha in enumerate(linhas_mosaico):
        if i > 0:
            partes.append(separador_linha)
        partes.append(linha)
    mosaico = np.vstack(partes)

    saida = Path(args.saida) if args.saida else (base_dir / "diagnostico_mosaico.png")
    cv.imwrite(str(saida), mosaico)
    print(f"[ok] Mosaico salvo em: {saida}  ({mosaico.shape[1]}x{mosaico.shape[0]}px, {len(linhas_mosaico)} frames)")
    print(f"     GSD do voo (efetivo): {construtor.gsd_voo_efetivo:.3f} m/px | roi_size_m: {construtor.roi_size_m:.1f}m")


if __name__ == "__main__":
    main()
