"""
Download e mosaico de imagens Sentinel-2 (via Microsoft Planetary Computer)
para uso como mapa base do módulo de Map Matching (map_matching.py /
paths.map_tif), em vez do NAIP.

Motivação: teste de sensibilidade do map matching a uma fonte de resolução
mais grosseira (10 m/px, contra 0.3 m/px do NAIP), simulando uma rota de
altitude maior (ver CHAMPAIGN_ALT3500/, GSD do voo ~5.9 m/px). Cobertura
global (ao contrário do NAIP, restrito aos EUA), sem conta/chave necessária.

Reaproveita bbox_a_partir_do_csv() e construir_vrt() de baixar_mapa_naip.py
(mesma pasta) -- só a busca STAC e o download do asset mudam: collection
'sentinel-2-l2a' em vez de 'naip', asset 'visual' (True Color Image, RGB
uint8 já processado, 10 m/px) em vez de 'image', e seleção por menor
cobertura de nuvem em vez de "ano mais recente".

Requisitos (instalar no venv local, NÃO no sandbox):
    pip install pystac-client planetary-computer rasterio

Uso:
    # A partir do CSV de ground truth da rota (colunas Lat, Long)
    python baixar_mapa_sentinel2.py --csv "Coord-Heading-Elev_3500_1.0.csv" \
        --margem-m 3000 --saida "map_s2.vrt"

    # Ou informando a bounding box manualmente (min_lon min_lat max_lon max_lat)
    python baixar_mapa_sentinel2.py --bbox -88.20 40.07 -88.17 40.10 \
        --saida "map_s2.vrt"

    # Ajustar o limiar de cobertura de nuvem (default: 20%)
    python baixar_mapa_sentinel2.py --csv rota.csv --nuvem-max 10 --saida map_s2.vrt
"""

import argparse
import sys
from pathlib import Path

import rasterio
from rasterio.enums import Resampling
from pyproj import Transformer

from baixar_mapa_naip import bbox_a_partir_do_csv, construir_vrt

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def buscar_itens_sentinel2(bbox, nuvem_max=20.0, data_filtro=None):
    """
    Busca cenas Sentinel-2 L2A no catálogo STAC do Planetary Computer que
    cobrem a bbox, filtradas por cobertura de nuvem.

    Diferente do NAIP (onde "mais recente" é o critério e todas as cenas do
    ano vencedor entram no mosaico), aqui o critério é "menos nublado":
    escolhe a cena de menor eo:cloud_cover como referência de data, e reúne
    só as cenas da MESMA data (caso a bbox cruze a borda de 2 granulos MGRS
    vizinhos -- incomum numa rota de poucos km, mas tratado do mesmo jeito
    que o NAIP trata múltiplos tiles).
    """
    import pystac_client
    import planetary_computer

    catalog = pystac_client.Client.open(
        STAC_URL, modifier=planetary_computer.sign_inplace
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=data_filtro,
        query={"eo:cloud_cover": {"lt": nuvem_max}},
    )
    itens = list(search.items())
    if not itens:
        sys.exit(
            f"[erro] Nenhuma cena Sentinel-2 encontrada para essa bbox com cobertura de "
            f"nuvem < {nuvem_max}%. Tente aumentar --nuvem-max ou remover --data."
        )

    # O catálogo às vezes indexa a MESMA aquisição (mesmo s2:mgrs_tile +
    # datetime) duas vezes com sufixos de processamento diferentes (ex.:
    # reprocessamento posterior pela ESA) -- sem isso, o mosaico baixaria e
    # empilharia 2x o dado do mesmo tile à toa. Mantém só a maior
    # s2:processing_baseline por (tile, datetime).
    melhores_por_chave = {}
    for it in itens:
        chave = (it.properties.get("s2:mgrs_tile"), it.datetime)
        baseline_atual = float(it.properties.get("s2:processing_baseline", 0.0))
        anterior = melhores_por_chave.get(chave)
        if anterior is None or baseline_atual > float(anterior.properties.get("s2:processing_baseline", 0.0)):
            melhores_por_chave[chave] = it
    itens = list(melhores_por_chave.values())

    itens.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100.0))
    data_referencia = itens[0].datetime.date()
    itens_mesma_data = [it for it in itens if it.datetime.date() == data_referencia]

    print(f"[info] {len(itens_mesma_data)} cena(s) Sentinel-2 encontrada(s), "
          f"data {data_referencia} (nuvem {itens[0].properties.get('eo:cloud_cover', '?'):.1f}%).")
    for it in itens_mesma_data:
        print(f"        - {it.id}  (nuvem {it.properties.get('eo:cloud_cover', '?'):.1f}%)")

    return itens_mesma_data


def baixar_cena(item, pasta_cache):
    """
    Baixa o asset 'visual' (TCI, RGB uint8, 10m/px) de uma cena Sentinel-2
    para um arquivo local via requests -- mesmo motivo do baixar_mapa_naip.py:
    evitar abrir a URL remota direto com rasterio/GDAL (UnicodeDecodeError
    conhecido do driver HTTP do GDAL no Windows).
    """
    import requests

    href = item.assets["visual"].href
    destino = pasta_cache / f"{item.id}.tif"
    if destino.exists() and destino.stat().st_size > 0:
        print(f"[info] {item.id}.tif já baixado, reaproveitando.")
        return destino

    print(f"[info] Baixando {item.id}...")
    with requests.get(href, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        baixado = 0
        with open(destino, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                baixado += len(chunk)
                if total:
                    print(f"\r        {baixado / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
        print()
    return destino


def mosaicar_e_salvar(itens, bbox, saida_path):
    """Baixa as cenas localmente e monta a VRT recortada na bbox (ver construir_vrt em baixar_mapa_naip.py)."""
    pasta_cache = Path(saida_path).parent / "_sentinel2_cache"
    pasta_cache.mkdir(parents=True, exist_ok=True)

    caminhos_locais = [baixar_cena(it, pasta_cache) for it in itens]

    with rasterio.open(str(caminhos_locais[0])) as ds0:
        raster_crs = ds0.crs

    if raster_crs and not raster_crs.is_geographic:
        transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
        min_x, min_y = transformer.transform(bbox[0], bbox[1])
        max_x, max_y = transformer.transform(bbox[2], bbox[3])
        bounds_raster = (min(min_x, max_x), min(min_y, max_y), max(min_x, max_x), max(min_y, max_y))
        print(f"[info] CRS do raster: {raster_crs} — bbox reprojetada: {bounds_raster}")
    else:
        bounds_raster = bbox

    print("[info] Construindo mosaico virtual (VRT) sobre os tiles originais (COG)...")
    construir_vrt(caminhos_locais, saida_path, bounds_recorte=bounds_raster)

    print("[info] Adicionando overviews à VRT (facilita buscas de ROI grandes)...")
    with rasterio.open(saida_path, "r+") as vrt_ds:
        vrt_ds.build_overviews([2, 4, 8, 16], Resampling.average)
        vrt_ds.update_tags(ns="rio_overview", resampling="average")
        largura, altura = vrt_ds.width, vrt_ds.height
        crs, transform = vrt_ds.crs, vrt_ds.transform

    print(f"[ok] Mapa (VRT) salvo em: {saida_path}")
    print(f"     Dimensões: {largura}x{altura} px")
    print(f"     CRS: {crs}")
    print(f"     Resolução: {abs(transform.a):.3f} (unidade do CRS)/px")
    print(f"     Tiles referenciados (não copiados): {len(caminhos_locais)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--csv", help="CSV de ground truth da rota (colunas Lat, Long)")
    grupo.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                        help="Bounding box manual em graus decimais (WGS84)")
    parser.add_argument("--margem-m", type=float, default=3000.0,
                         help="Margem extra ao redor da rota em metros (default: 3000m -- GSD do voo maior exige roi_size_m maior)")
    parser.add_argument("--nuvem-max", type=float, default=20.0,
                         help="Cobertura de nuvem máxima aceita, em %% (default: 20)")
    parser.add_argument("--data", default=None,
                         help="Filtro de data STAC (ex.: '2023-01-01/2023-12-31'; default: sem filtro, pega a cena menos nublada disponível)")
    parser.add_argument("--saida", default="map_s2.vrt", help="Caminho da VRT de saída (mosaico virtual, não copia pixels)")
    args = parser.parse_args()

    if args.csv:
        bbox = bbox_a_partir_do_csv(args.csv, args.margem_m)
    else:
        bbox = tuple(args.bbox)

    print(f"[info] Bounding box (WGS84): {bbox}")

    itens = buscar_itens_sentinel2(bbox, args.nuvem_max, args.data)
    mosaicar_e_salvar(itens, bbox, args.saida)


if __name__ == "__main__":
    main()
