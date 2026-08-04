"""
Download e mosaico do ESRI World Imagery (tiles XYZ) para uso como mapa base
ALTERNATIVO do map matching, na MESMA altitude/rota do teste NAIP (1500m,
CHAMPAIGN) — fonte independente do NAIP e do Google Earth, para medir se o
gap residual do map matching é específico do par NAIP↔Google Earth.

RESSALVA DE LICENÇA: o ESRI World Imagery é um mosaico Maxar/Vivid servido por
tiles XYZ (server.arcgisonline.com). O uso em pesquisa acadêmica costuma ser
tolerado, mas download em massa foge do ToS pretendido — este script baixa
apenas os poucos tiles que cobrem uma rota de ~2km + margem, para um teste
único de pesquisa, não para redistribuição. Ver tabela de fontes em
map_matching_naip/CLAUDE.md.

Sem STAC (diferente do NAIP/Sentinel-2): tiles Web Mercator (EPSG:3857) na
convenção {z}/{y}/{x}. Baixa os tiles que cobrem a bbox da rota, costura num
mosaico e grava um GeoTIFF georreferenciado em EPSG:3857 — que map_matching.py
abre normalmente (CRS projetado, reprojeção de bbox já suportada).

Reaproveita bbox_a_partir_do_csv() de baixar_mapa_naip.py.

Uso:
    python baixar_mapa_esri.py --csv "...\\CHAMPAIGN\\mach_1.0\\Coord-Heading-Elev_1500_1.0.csv" \
        --margem-m 1600 --zoom 17 --saida "...\\CHAMPAIGN\\map_esri.tif"
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import cv2 as cv
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).parent))
from baixar_mapa_naip import bbox_a_partir_do_csv

_R = 20037508.342789244  # meio-mundo em EPSG:3857 (m)
_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}")
_UA = {"User-Agent": "Mozilla/5.0 (research map-matching base map fetch)"}


def _lonlat_para_tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return x, y


def _baixar_tile(z, x, y, cache, sess):
    import requests
    destino = cache / f"{z}_{x}_{y}.jpg"
    if destino.exists() and destino.stat().st_size > 0:
        return destino
    url = _URL.format(z=z, x=x, y=y)
    for tentativa in range(4):
        try:
            r = sess.get(url, headers=_UA, timeout=30)
            if r.status_code == 200 and r.content:
                destino.write_bytes(r.content)
                return destino
        except Exception:
            pass
        time.sleep(1.0 + tentativa)
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="CSV de ground truth da rota (colunas Lat, Long)")
    p.add_argument("--margem-m", type=float, default=1600.0,
                    help="Margem ao redor da rota (m). Precisa >= raio de busca do map matching (~1431m p/ NAIP 1500m).")
    p.add_argument("--zoom", type=int, default=17,
                    help="Zoom XYZ (17 ~0.9m/px @lat40; 18 ~0.46m/px). Default 17.")
    p.add_argument("--saida", required=True, help="GeoTIFF de saída (EPSG:3857)")
    args = p.parse_args()

    import requests

    bbox = bbox_a_partir_do_csv(args.csv, args.margem_m)  # (min_lon, min_lat, max_lon, max_lat)
    z = args.zoom
    x0f, y0f = _lonlat_para_tile(bbox[0], bbox[3], z)  # canto sup-esq (min_lon, max_lat)
    x1f, y1f = _lonlat_para_tile(bbox[2], bbox[1], z)  # canto inf-dir (max_lon, min_lat)
    x_min, x_max = int(math.floor(x0f)), int(math.floor(x1f))
    y_min, y_max = int(math.floor(y0f)), int(math.floor(y1f))
    nx, ny = (x_max - x_min + 1), (y_max - y_min + 1)
    total = nx * ny
    tile_m = 2.0 * _R / (2 ** z)   # tamanho do tile em metros (EPSG:3857)
    res = tile_m / 256.0
    print(f"[info] bbox={bbox}")
    print(f"[info] zoom {z}: {nx}x{ny} = {total} tiles | {res:.3f} m/px | tile={tile_m:.1f}m")
    if total > 4000:
        sys.exit(f"[erro] {total} tiles é demais — reduza --margem-m ou --zoom.")

    cache = Path(args.saida).parent / "_esri_cache"
    cache.mkdir(parents=True, exist_ok=True)

    mosaico = np.zeros((ny * 256, nx * 256, 3), dtype=np.uint8)
    sess = requests.Session()
    faltando = 0
    for j, ty in enumerate(range(y_min, y_max + 1)):
        for i, tx in enumerate(range(x_min, x_max + 1)):
            path = _baixar_tile(z, tx, ty, cache, sess)
            if path is None:
                faltando += 1
                continue
            img = cv.imread(str(path))  # BGR
            if img is None:
                faltando += 1
                continue
            if img.shape[:2] != (256, 256):
                img = cv.resize(img, (256, 256))
            mosaico[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = img
        print(f"\r        linha {j+1}/{ny} baixada", end="", flush=True)
    print(f"\n[info] tiles faltando/falhos: {faltando}/{total}")

    # Geotransform em EPSG:3857: origem = canto sup-esq do tile (x_min, y_min)
    origin_x = -_R + x_min * tile_m
    origin_y = _R - y_min * tile_m
    transform = from_origin(origin_x, origin_y, res, res)

    # rasterio espera (bands, H, W) em ordem RGB; mosaico é BGR -> converte
    rgb = cv.cvtColor(mosaico, cv.COLOR_BGR2RGB)
    arr = np.transpose(rgb, (2, 0, 1))
    saida = Path(args.saida)
    with rasterio.open(
        saida, "w", driver="GTiff",
        height=arr.shape[1], width=arr.shape[2], count=3, dtype="uint8",
        crs="EPSG:3857", transform=transform,
        tiled=True, blockxsize=256, blockysize=256, compress="deflate",
    ) as dst:
        dst.write(arr)
        dst.build_overviews([2, 4, 8, 16, 32], Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")

    print(f"[ok] Mapa ESRI salvo: {saida}  ({arr.shape[2]}x{arr.shape[1]}px, {res:.3f} m/px, EPSG:3857)")


if __name__ == "__main__":
    main()
