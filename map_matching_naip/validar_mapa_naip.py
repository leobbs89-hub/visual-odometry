"""
Sanity check do mapa base NAIP antes de rodar o experimento de map matching.

Confere:
    1. O GeoTIFF abre e tem CRS válido.
    2. Todos os pontos do CSV da rota caem dentro dos limites do raster,
       com folga suficiente para o roi_size_m usado no map matching
       (senão _recortar_roi_mapa em map_matching.py vai gerar patches vazios
       perto das bordas da rota).

Uso:
    python validar_mapa_naip.py --map map.tif --csv Coord-Heading-Elev_1500_1.0.csv --roi-size-m 1000
"""

import argparse
import sys

import pandas as pd
import rasterio
from pyproj import Transformer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", required=True, help="Caminho do GeoTIFF (map.tif)")
    parser.add_argument("--csv", required=True, help="CSV de ground truth da rota (colunas Lat, Long)")
    parser.add_argument("--roi-size-m", type=float, default=1000.0,
                         help="Mesmo valor de map_matching.roi_size_m no config (default: 1000)")
    args = parser.parse_args()

    ds = rasterio.open(args.map)
    print(f"[ok] Abriu {args.map}")
    print(f"     CRS: {ds.crs}")
    print(f"     Dimensões: {ds.width}x{ds.height} px")
    print(f"     Bounds (CRS nativo): {ds.bounds}")

    # Bounds em WGS84 para comparar com Lat/Long do CSV
    if ds.crs and not ds.crs.is_geographic:
        transformer = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
        lon_min, lat_min = transformer.transform(ds.bounds.left, ds.bounds.bottom)
        lon_max, lat_max = transformer.transform(ds.bounds.right, ds.bounds.top)
    else:
        lon_min, lat_min, lon_max, lat_max = ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top

    print(f"     Bounds (WGS84): lon [{lon_min:.6f}, {lon_max:.6f}]  lat [{lat_min:.6f}, {lat_max:.6f}]")

    df = pd.read_csv(args.csv)
    if "Lat" not in df.columns or "Long" not in df.columns:
        sys.exit(f"[erro] CSV precisa ter colunas 'Lat' e 'Long'. Encontradas: {list(df.columns)}")

    df["Lat"]  = pd.to_numeric(df["Lat"], errors="coerce")
    df["Long"] = pd.to_numeric(df["Long"], errors="coerce")
    if (df["Lat"].isna().any() or df["Long"].isna().any()
            or df["Lat"].abs().max() > 90 or df["Long"].abs().max() > 180):
        sys.exit(
            "[erro] CSV com Lat/Long fora do range geográfico válido — provável "
            "corrupção por abrir/salvar o CSV no Excel com config. regional pt-BR "
            "(troca separador decimal). Rode criar_rota_champaign.py de novo para "
            "regerar o CSV, sem abrir no Excel."
        )

    # Margem de segurança: metade do roi_size_m (com folga de 1.5x, igual ao
    # margem_extra usado em map_matching._recortar_roi_mapa) convertida em graus
    margem_deg_lat = (args.roi_size_m * 0.75) / 111_320.0
    margem_deg_lon = margem_deg_lat  # aproximação conservadora

    fora = df[
        (df["Long"] < lon_min + margem_deg_lon) | (df["Long"] > lon_max - margem_deg_lon) |
        (df["Lat"]  < lat_min + margem_deg_lat) | (df["Lat"]  > lat_max - margem_deg_lat)
    ]

    if len(fora) > 0:
        print(f"\n[AVISO] {len(fora)} de {len(df)} pontos da rota ficam a menos de "
              f"{args.roi_size_m * 0.75:.0f}m da borda do mapa (ou fora dele).")
        print("        Isso pode gerar patches vazios/recortados no map matching perto desses frames.")
        print("        Solução: rodar baixar_mapa_naip.py de novo com --margem-m maior.")
    else:
        print(f"\n[ok] Todos os {len(df)} pontos da rota têm folga >= {args.roi_size_m * 0.75:.0f}m até a borda do mapa.")

    ds.close()


if __name__ == "__main__":
    main()
