"""
Download e mosaico de imagens Landsat 8/9 Collection 2 Level-2 (via Microsoft
Planetary Computer) para uso como mapa base do módulo de Map Matching, na rota
CHAMPAIGN_ALT7000 (altitude maior, GSD do voo mais grosseiro — ver
criar_rota_champaign_alt7000.py).

Diferente do NAIP/Sentinel-2 (asset 'image'/'visual' já RGB uint8 pronto), a
collection 'landsat-c2-l2' só oferece bandas de reflectância de superfície
separadas (red/green/blue, Int16 com fator de escala) — não existe um asset
"visual" TCI pronto. Este script baixa as 3 bandas, aplica o fator de escala
oficial da USGS Collection 2 Level 2 (refl = DN*0.0000275 - 0.2), empilha em
RGB e estica linearmente (reflectância 0-0.3 -> 0-255, faixa típica de "true
color" para cena terrestre não saturada) para uint8 -- um GeoTIFF composto por
cena, que depois entra no mesmo construir_vrt() (baixar_mapa_naip.py) usado
pelas outras fontes.

Requisitos (venv local): pystac-client, planetary-computer, rasterio, requests
(mesmos já usados por baixar_mapa_naip.py/baixar_mapa_sentinel2.py).

Uso:
    python baixar_mapa_landsat.py --csv "...\\CHAMPAIGN_ALT7000\\mach_1.0\\Coord-Heading-Elev_7000_1.0.csv" \
        --margem-m 6000 --saida "...\\CHAMPAIGN_ALT7000\\map_landsat.vrt"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, Resampling as WarpResampling
from pyproj import Transformer

from baixar_mapa_naip import bbox_a_partir_do_csv, construir_vrt

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Coeficientes oficiais USGS Collection 2 Level 2 (surface reflectance):
# reflectancia = DN * ESCALA + OFFSET (produz refletância adimensional ~0-1)
_ESCALA, _OFFSET = 0.0000275, -0.2
# Faixa de estica linear para "true color": reflectância 0..STRETCH_MAX -> 0..255
_STRETCH_MAX = 0.3


def buscar_itens_landsat(bbox, nuvem_max=20.0, data_filtro=None):
    """Busca cenas Landsat 8/9 C2L2 no STAC do Planetary Computer, cobrindo a
    bbox e filtradas por cobertura de nuvem -- mesmo padrão de
    buscar_itens_sentinel2 (baixar_mapa_sentinel2.py): escolhe a cena menos
    nublada como referência de data e reúne as cenas da MESMA data (caso a
    bbox cruze a borda de 2 cenas WRS vizinhas)."""
    import pystac_client
    import planetary_computer

    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bbox,
        datetime=data_filtro,
        query={"eo:cloud_cover": {"lt": nuvem_max}, "platform": {"in": ["landsat-8", "landsat-9"]}},
    )
    itens = list(search.items())
    if not itens:
        sys.exit(f"[erro] Nenhuma cena Landsat 8/9 encontrada para essa bbox com nuvem < {nuvem_max}%. "
                  "Tente aumentar --nuvem-max.")

    itens.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100.0))
    data_ref = itens[0].datetime.date()
    itens_mesma_data = [it for it in itens if it.datetime.date() == data_ref]

    print(f"[info] {len(itens_mesma_data)} cena(s) Landsat encontrada(s), data {data_ref} "
          f"(nuvem {itens[0].properties.get('eo:cloud_cover', '?'):.1f}%).")
    for it in itens_mesma_data:
        print(f"        - {it.id}  (nuvem {it.properties.get('eo:cloud_cover', '?'):.1f}%)")
    return itens_mesma_data


def _baixar_banda(item, banda, pasta_cache):
    import requests
    href = item.assets[banda].href
    destino = pasta_cache / f"{item.id}_{banda}.tif"
    if destino.exists() and destino.stat().st_size > 0:
        return destino
    print(f"[info] Baixando {item.id} banda {banda}...")
    with requests.get(href, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(destino, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    return destino


def _compor_rgb(item, pasta_cache):
    """Baixa red/green/blue, aplica escala oficial C2L2 e estica para RGB
    uint8. Retorna o caminho do GeoTIFF composto (mesma grade/CRS/transform
    das bandas originais -- as 3 bandas de uma cena Landsat já vêm co-
    registradas, sem reprojeção necessária aqui)."""
    destino_rgb = pasta_cache / f"{item.id}_rgb.tif"
    if destino_rgb.exists() and destino_rgb.stat().st_size > 0:
        print(f"[info] {item.id}_rgb.tif já composto, reaproveitando.")
        return destino_rgb

    caminhos = {b: _baixar_banda(item, b, pasta_cache) for b in ("red", "green", "blue")}
    bandas_u8 = []
    perfil = None
    for b in ("red", "green", "blue"):
        with rasterio.open(caminhos[b]) as ds:
            dn = ds.read(1).astype(np.float32)
            nodata_mask = dn == (ds.nodata if ds.nodata is not None else 0)
            refl = dn * _ESCALA + _OFFSET
            u8 = np.clip(refl / _STRETCH_MAX, 0.0, 1.0) * 255.0
            u8 = u8.astype(np.uint8)
            u8[nodata_mask] = 0
            bandas_u8.append(u8)
            if perfil is None:
                perfil = ds.profile.copy()

    perfil.update(count=3, dtype="uint8", nodata=0, compress="deflate",
                   tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(destino_rgb, "w", **perfil) as dst:
        for i, u8 in enumerate(bandas_u8, start=1):
            dst.write(u8, i)
        dst.build_overviews([2, 4, 8, 16], Resampling.average)

    print(f"[ok] {item.id}: RGB composto ({perfil['width']}x{perfil['height']}px) -> {destino_rgb.name}")
    return destino_rgb


def mosaicar_e_salvar(itens, bbox, saida_path):
    pasta_cache = Path(saida_path).parent / "_landsat_cache"
    pasta_cache.mkdir(parents=True, exist_ok=True)

    caminhos_rgb = [_compor_rgb(it, pasta_cache) for it in itens]

    with rasterio.open(str(caminhos_rgb[0])) as ds0:
        raster_crs = ds0.crs

    if raster_crs and not raster_crs.is_geographic:
        transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
        min_x, min_y = transformer.transform(bbox[0], bbox[1])
        max_x, max_y = transformer.transform(bbox[2], bbox[3])
        bounds_raster = (min(min_x, max_x), min(min_y, max_y), max(min_x, max_x), max(min_y, max_y))
        print(f"[info] CRS do raster: {raster_crs} — bbox reprojetada: {bounds_raster}")
    else:
        bounds_raster = bbox

    print("[info] Construindo mosaico virtual (VRT)...")
    construir_vrt(caminhos_rgb, saida_path, bounds_recorte=bounds_raster)

    with rasterio.open(saida_path, "r+") as vrt_ds:
        vrt_ds.build_overviews([2, 4, 8], Resampling.average)
        vrt_ds.update_tags(ns="rio_overview", resampling="average")
        largura, altura = vrt_ds.width, vrt_ds.height
        crs, transform = vrt_ds.crs, vrt_ds.transform

    print(f"[ok] Mapa (VRT) salvo em: {saida_path}")
    print(f"     Dimensões: {largura}x{altura} px")
    print(f"     CRS: {crs}")
    print(f"     Resolução: {abs(transform.a):.3f} (unidade do CRS)/px")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--csv", help="CSV de ground truth da rota (colunas Lat, Long)")
    grupo.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--margem-m", type=float, default=6000.0,
                         help="Margem ao redor da rota (m). Default 6000 -- GSD do voo bem maior exige roi_size_m maior.")
    parser.add_argument("--nuvem-max", type=float, default=20.0)
    parser.add_argument("--data", default=None)
    parser.add_argument("--saida", default="map_landsat.vrt")
    args = parser.parse_args()

    bbox = bbox_a_partir_do_csv(args.csv, args.margem_m) if args.csv else tuple(args.bbox)
    print(f"[info] Bounding box (WGS84): {bbox}")

    itens = buscar_itens_landsat(bbox, args.nuvem_max, args.data)
    mosaicar_e_salvar(itens, bbox, args.saida)


if __name__ == "__main__":
    main()
