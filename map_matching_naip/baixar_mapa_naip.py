"""
Download e mosaico de imagens NAIP (via Microsoft Planetary Computer) para uso
como mapa base do módulo de Map Matching (map_matching.py / paths.map_tif).

Não requer conta nem chave de API — o Planetary Computer assina os links dos
tiles NAIP publicamente (`planetary_computer.sign_inplace`). Roda 100% local:
lê a bounding box (a partir de um CSV de rota ou de coordenadas manuais),
busca os tiles NAIP que cobrem a área via STAC, baixa as cenas necessárias
(cache em _naip_cache/ ao lado da saída, para não baixar de novo se rodar
outra vez) e mosaica recortando na bbox.

Requisitos (instalar no venv local, NÃO no sandbox):
    pip install pystac-client planetary-computer rasterio

Uso:
    # A partir do CSV de ground truth da rota (colunas Lat, Long)
    python baixar_mapa_naip.py --csv "Coord-Heading-Elev_1500_2.0.csv" \
        --margem-m 800 --saida "map.tif"

    # Ou informando a bounding box manualmente (min_lon min_lat max_lon max_lat)
    python baixar_mapa_naip.py --bbox -105.30 39.98 -105.26 40.02 \
        --saida "map.tif"

    # Filtrar por ano de aquisição (opcional; padrão = mais recente disponível)
    python baixar_mapa_naip.py --csv rota.csv --ano 2023 --saida map.tif
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
from pyproj import Geod, Transformer


STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def bbox_a_partir_do_csv(csv_path, margem_m):
    """Lê o CSV de ground truth (colunas Lat, Long) e calcula bbox com margem."""
    df = pd.read_csv(csv_path)
    if "Lat" not in df.columns or "Long" not in df.columns:
        sys.exit(f"[erro] CSV precisa ter colunas 'Lat' e 'Long'. Encontradas: {list(df.columns)}")

    lat_col = pd.to_numeric(df["Lat"], errors="coerce")
    lon_col = pd.to_numeric(df["Long"], errors="coerce")

    # Detecta o CSV corrompido pelo Excel/LibreOffice: ao abrir e salvar um CSV
    # com "." como separador decimal usando config. regional pt-BR (que usa "."
    # como separador de milhar), o app remove o ponto decimal original e
    # reformata o número como inteiro agrupado por milhar
    # (ex.: 40.08500000 -> "4.008.500.000"). Isso estoura o range geográfico
    # válido e é fácil de detectar.
    problema = (
        lat_col.isna().any() or lon_col.isna().any()
        or lat_col.abs().max() > 90 or lon_col.abs().max() > 180
    )
    if problema:
        sys.exit(
            "[erro] Coluna 'Lat'/'Long' do CSV não parecem coordenadas válidas "
            "(fora do range -90..90 / -180..180, ou não numéricas).\n"
            "Causa mais provável: o CSV foi aberto e salvo no Excel/LibreOffice "
            "com config. regional pt-BR, que troca o separador decimal e corrompe "
            "os números (ex.: 40.08500000 vira 4.008.500.000).\n"
            "Solução: rode criar_rota_champaign.py de novo para regerar o CSV "
            "(não abra/salve o CSV no Excel — se precisar inspecionar, use um "
            "editor de texto ou importe via 'Dados > De Texto/CSV' escolhendo "
            "'.' como separador decimal)."
        )

    min_lat, max_lat = lat_col.min(), lat_col.max()
    min_lon, max_lon = lon_col.min(), lon_col.max()

    # Expande a bbox pela margem (em metros) usando geodésia (WGS84)
    g = Geod(ellps="WGS84")
    center_lat = (min_lat + max_lat) / 2.0
    # graus de latitude por metro (aprox. constante)
    lat_margin_deg = margem_m / 111_320.0
    # graus de longitude por metro (varia com a latitude)
    lon_margin_deg = margem_m / (111_320.0 * np.cos(np.radians(center_lat)))

    return (
        min_lon - lon_margin_deg,
        min_lat - lat_margin_deg,
        max_lon + lon_margin_deg,
        max_lat + lat_margin_deg,
    )


def buscar_itens_naip(bbox, ano=None):
    """Busca cenas NAIP no catálogo STAC do Planetary Computer que cobrem a bbox."""
    import pystac_client
    import planetary_computer

    catalog = pystac_client.Client.open(
        STAC_URL, modifier=planetary_computer.sign_inplace
    )

    datetime_filter = None
    if ano is not None:
        datetime_filter = f"{ano}-01-01/{ano}-12-31"

    search = catalog.search(
        collections=["naip"],
        bbox=bbox,
        datetime=datetime_filter,
    )
    itens = list(search.items())
    if not itens:
        sys.exit(
            "[erro] Nenhuma cena NAIP encontrada para essa bbox/ano. "
            "Tente remover --ano ou verificar se a área está nos EUA (NAIP cobre só o território continental dos EUA)."
        )

    # Se houver mais de uma data disponível cobrindo a área, prioriza a mais recente
    itens.sort(key=lambda it: it.datetime, reverse=True)

    # Mantém só o ano mais recente entre os itens retornados (evita misturar anos)
    ano_mais_recente = itens[0].datetime.year
    itens_filtrados = [it for it in itens if it.datetime.year == ano_mais_recente]

    print(f"[info] {len(itens_filtrados)} cena(s) NAIP encontrada(s), ano {ano_mais_recente}.")
    for it in itens_filtrados:
        print(f"        - {it.id}  ({it.datetime.date()})")

    return itens_filtrados


def baixar_cena(item, pasta_cache):
    """
    Baixa o asset 'image' de uma cena NAIP para um arquivo local via requests
    (em vez de abrir a URL remota direto com rasterio/GDAL — no Windows, o
    driver HTTP do GDAL (vsicurl) pode travar com UnicodeDecodeError ao tentar
    decodificar mensagens de erro do sistema que não são UTF-8; baixar antes
    com requests evita esse caminho de código problemático).
    """
    import requests

    href = item.assets["image"].href
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
    """Baixa as cenas localmente e mosaica, recortando na bbox."""
    pasta_cache = Path(saida_path).parent / "_naip_cache"
    pasta_cache.mkdir(parents=True, exist_ok=True)

    caminhos_locais = [baixar_cena(it, pasta_cache) for it in itens]
    datasets = [rasterio.open(str(p)) for p in caminhos_locais]

    # bbox chega em WGS84 (graus); os tiles NAIP costumam vir em CRS projetado
    # (UTM, metros) -- reprojeta a bbox pro CRS do raster antes de recortar,
    # senão merge(bounds=...) não acha interseção nenhuma (0x0).
    raster_crs = datasets[0].crs
    if raster_crs and not raster_crs.is_geographic:
        transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
        min_x, min_y = transformer.transform(bbox[0], bbox[1])
        max_x, max_y = transformer.transform(bbox[2], bbox[3])
        bounds_raster = (min(min_x, max_x), min(min_y, max_y), max(min_x, max_x), max(min_y, max_y))
        print(f"[info] CRS do raster: {raster_crs} — bbox reprojetada: {bounds_raster}")
    else:
        bounds_raster = bbox

    print("[info] Mosaicando...")
    mosaico, transform = merge(
        datasets,
        bounds=bounds_raster,
        resampling=Resampling.bilinear,
    )

    if mosaico.shape[1] == 0 or mosaico.shape[2] == 0:
        sys.exit(
            "[erro] Recorte resultou em 0x0 -- a bbox não intersecta os tiles baixados.\n"
            f"       Bounds dos tiles (CRS nativo): {[ds.bounds for ds in datasets]}\n"
            f"       Bbox pedida (CRS nativo): {bounds_raster}"
        )

    perfil = datasets[0].profile.copy()
    perfil.update(
        driver="GTiff",
        height=mosaico.shape[1],
        width=mosaico.shape[2],
        count=mosaico.shape[0],
        transform=transform,
        compress="deflate",
    )

    with rasterio.open(saida_path, "w", **perfil) as dst:
        dst.write(mosaico)

    for ds in datasets:
        ds.close()

    print(f"[ok] Mapa salvo em: {saida_path}")
    print(f"     Dimensões: {mosaico.shape[2]}x{mosaico.shape[1]} px, {mosaico.shape[0]} banda(s)")
    print(f"     CRS: {perfil['crs']}")
    print(f"     Resolução: {abs(transform.a):.3f} (unidade do CRS)/px")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--csv", help="CSV de ground truth da rota (colunas Lat, Long)")
    grupo.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                        help="Bounding box manual em graus decimais (WGS84)")
    parser.add_argument("--margem-m", type=float, default=800.0,
                         help="Margem extra ao redor da rota em metros (default: 800m — cobre roi_size_m=1000 do map_matching com folga)")
    parser.add_argument("--ano", type=int, default=None, help="Filtrar por ano de aquisição NAIP (default: mais recente disponível)")
    parser.add_argument("--saida", default="map.tif", help="Caminho do GeoTIFF de saída")
    args = parser.parse_args()

    if args.csv:
        bbox = bbox_a_partir_do_csv(args.csv, args.margem_m)
    else:
        bbox = tuple(args.bbox)

    print(f"[info] Bounding box (WGS84): {bbox}")

    itens = buscar_itens_naip(bbox, args.ano)
    mosaicar_e_salvar(itens, bbox, args.saida)


if __name__ == "__main__":
    main()
