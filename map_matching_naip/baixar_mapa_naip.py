"""
Download e mosaico de imagens NAIP (via Microsoft Planetary Computer) para uso
como mapa base do módulo de Map Matching (map_matching.py / paths.map_tif).

Não requer conta nem chave de API — o Planetary Computer assina os links dos
tiles NAIP publicamente (`planetary_computer.sign_inplace`). Roda 100% local:
lê a bounding box (a partir de um CSV de rota ou de coordenadas manuais),
busca os tiles NAIP que cobrem a área via STAC, baixa as cenas necessárias
(cache em _naip_cache/ ao lado da saída, para não baixar de novo se rodar
outra vez) e monta um mosaico virtual (.vrt) recortado na bbox, referenciando
os tiles baixados sem copiar pixels (os tiles NAIP da Planetary Computer já
vêm como COG — tiled + overviews internas; um merge() monolítico como antes
jogava essa estrutura fora e deixava as leituras windowed do map matching
mais lentas do que precisava).

Requisitos (instalar no venv local, NÃO no sandbox):
    pip install pystac-client planetary-computer rasterio

Uso:
    # A partir do CSV de ground truth da rota (colunas Lat, Long)
    python baixar_mapa_naip.py --csv "Coord-Heading-Elev_1500_2.0.csv" \
        --margem-m 800 --saida "map.vrt"

    # Ou informando a bounding box manualmente (min_lon min_lat max_lon max_lat)
    python baixar_mapa_naip.py --bbox -105.30 39.98 -105.26 40.02 \
        --saida "map.vrt"

    # Filtrar por ano de aquisição (opcional; padrão = mais recente disponível)
    python baixar_mapa_naip.py --csv rota.csv --ano 2023 --saida map.vrt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from pyproj import Geod, Transformer

# Mapeamento numpy dtype -> tipo GDAL (usado na VRT escrita à mão abaixo)
_GDAL_DTYPES = {
    "uint8": "Byte", "int16": "Int16", "uint16": "UInt16",
    "int32": "Int32", "uint32": "UInt32", "float32": "Float32", "float64": "Float64",
}


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


def construir_vrt(caminhos_tiles, saida_vrt_path, bounds_recorte=None):
    """
    Constrói um mosaico virtual (.vrt) referenciando os tiles originais em
    vez de copiar/regravar os pixels (como rasterio.merge.merge() fazia).

    Motivação (otimização de carregamento para rotas maiores, ex. 10km):
    os tiles NAIP baixados via Planetary Computer já vêm como COG (tiled
    512x512 + overviews internas — confirmado por inspeção) — o antigo
    map.tif monolítico gerado por merge() jogava fora essa estrutura
    (virava um GeoTIFF em strips, sem overviews), tornando as leituras
    windowed de _recortar_roi_mapa() mais lentas que o necessário. A VRT
    preserva o tiling/overviews de cada fonte, sem duplicar dado nenhum
    (map.vrt tem alguns KB; os pixels continuam só em _naip_cache/).

    Assume todos os tiles no mesmo CRS/resolução/dtype/nº de bandas (válido
    aqui: mesmo lote NAIP, mesmo ano, filtrado por buscar_itens_naip). Não
    depende de gdalbuildvrt/osgeo (indisponíveis neste venv) — escreve o
    XML da VRT diretamente a partir dos metadados lidos via rasterio.
    """
    datasets = [rasterio.open(str(p)) for p in caminhos_tiles]
    ref = datasets[0]
    res_x = abs(ref.transform.a)
    res_y = abs(ref.transform.e)
    band_count = ref.count
    gdal_dtype = _GDAL_DTYPES.get(ref.dtypes[0], "Byte")

    minx = min(ds.bounds.left for ds in datasets)
    maxx = max(ds.bounds.right for ds in datasets)
    miny = min(ds.bounds.bottom for ds in datasets)
    maxy = max(ds.bounds.top for ds in datasets)
    if bounds_recorte is not None:
        minx = max(minx, bounds_recorte[0])
        miny = max(miny, bounds_recorte[1])
        maxx = min(maxx, bounds_recorte[2])
        maxy = min(maxy, bounds_recorte[3])
    if minx >= maxx or miny >= maxy:
        sys.exit(
            "[erro] Recorte resultou em 0x0 -- a bbox não intersecta os tiles baixados.\n"
            f"       Bounds dos tiles (CRS nativo): {[ds.bounds for ds in datasets]}\n"
            f"       Bbox pedida (CRS nativo): {bounds_recorte}"
        )

    width  = max(1, int(round((maxx - minx) / res_x)))
    height = max(1, int(round((maxy - miny) / res_y)))

    linhas = [
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">',
        f'  <SRS>{ref.crs.to_wkt()}</SRS>',
        f'  <GeoTransform>{minx}, {res_x}, 0.0, {maxy}, 0.0, {-res_y}</GeoTransform>',
    ]
    # Ordem de pintura: sources posteriores na lista sobrescrevem as
    # anteriores nas áreas de sobreposição (comportamento nativo da VRT).
    # rasterio.merge.merge() (usado antes) tem a semântica oposta --
    # method='first' faz o PRIMEIRO dataset da lista vencer sobreposições.
    # Iterar em ordem reversa aqui reproduz essa mesma prioridade (o tile
    # caminhos_tiles[0] é pintado por último, então "vence").
    fontes_em_ordem_de_pintura = list(reversed(list(zip(datasets, caminhos_tiles))))

    for b in range(1, band_count + 1):
        linhas.append(f'  <VRTRasterBand dataType="{gdal_dtype}" band="{b}">')
        for ds, path in fontes_em_ordem_de_pintura:
            tb = ds.bounds
            ov_minx, ov_maxx = max(tb.left, minx), min(tb.right, maxx)
            ov_miny, ov_maxy = max(tb.bottom, miny), min(tb.top, maxy)
            if ov_minx >= ov_maxx or ov_miny >= ov_maxy:
                continue  # este tile não intersecta a extensão de saída
            src_x_off = int(round((ov_minx - tb.left) / res_x))
            src_y_off = int(round((tb.top - ov_maxy) / res_y))
            src_w = max(1, int(round((ov_maxx - ov_minx) / res_x)))
            src_h = max(1, int(round((ov_maxy - ov_miny) / res_y)))
            dst_x_off = int(round((ov_minx - minx) / res_x))
            dst_y_off = int(round((maxy - ov_maxy) / res_y))
            linhas.append('    <SimpleSource>')
            linhas.append(f'      <SourceFilename relativeToVRT="0">{Path(path).resolve()}</SourceFilename>')
            linhas.append(f'      <SourceBand>{b}</SourceBand>')
            linhas.append(f'      <SrcRect xOff="{src_x_off}" yOff="{src_y_off}" xSize="{src_w}" ySize="{src_h}"/>')
            linhas.append(f'      <DstRect xOff="{dst_x_off}" yOff="{dst_y_off}" xSize="{src_w}" ySize="{src_h}"/>')
            linhas.append('    </SimpleSource>')
        linhas.append('  </VRTRasterBand>')
    linhas.append('</VRTDataset>')

    for ds in datasets:
        ds.close()

    Path(saida_vrt_path).write_text("\n".join(linhas), encoding="utf-8")


def mosaicar_e_salvar(itens, bbox, saida_path):
    """
    Baixa as cenas localmente e monta a VRT recortada na bbox (em vez de
    copiar os pixels num GeoTIFF monolítico — ver construir_vrt()).
    """
    pasta_cache = Path(saida_path).parent / "_naip_cache"
    pasta_cache.mkdir(parents=True, exist_ok=True)

    caminhos_locais = [baixar_cena(it, pasta_cache) for it in itens]

    with rasterio.open(str(caminhos_locais[0])) as ds0:
        raster_crs = ds0.crs

    # bbox chega em WGS84 (graus); os tiles NAIP costumam vir em CRS projetado
    # (UTM, metros) -- reprojeta a bbox pro CRS do raster antes de recortar,
    # senão a VRT não acha interseção nenhuma (0x0).
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
        vrt_ds.build_overviews([2, 4, 8, 16, 32], Resampling.average)
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
    parser.add_argument("--margem-m", type=float, default=800.0,
                         help="Margem extra ao redor da rota em metros (default: 800m — cobre roi_size_m=1000 do map_matching com folga)")
    parser.add_argument("--ano", type=int, default=None, help="Filtrar por ano de aquisição NAIP (default: mais recente disponível)")
    parser.add_argument("--saida", default="map.vrt", help="Caminho da VRT de saída (mosaico virtual, não copia pixels)")
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
