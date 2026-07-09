"""
Checagem de cobertura NAIP para uma rota candidata, SEM baixar nenhuma cena
completa -- só lista as cenas (id/data) que o STAC do Planetary Computer
retorna para a bbox da rota + margem. Serve para decidir o tamanho final da
rota (lado do quadrado) e a margem de download antes de comprometer
banda/disco com o mosaico completo (baixar_mapa_naip.py).

Reaproveita generate_square_route (criar_rota_champaign.py) para calcular a
bbox exata da rota (não uma aproximação), e a mesma lógica de busca STAC de
baixar_mapa_naip.py::buscar_itens_naip -- mas parando antes do download.

Uso:
    python verificar_cobertura_naip.py --side-km 5 6 --margem-m 3000 4000
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from criar_rota_champaign import (
    generate_square_route, SQUARE_START, SQUARE_INITIAL_HEADING,
    SQUARE_TURN_DIRECTION, SQUARE_N_CURVE_POINTS, ALTITUDE, TILT,
)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def bbox_da_rota(side_km, margem_m):
    """Gera os waypoints da rota candidata e devolve a bbox (WGS84) com margem."""
    waypoints = generate_square_route(
        start_lon=SQUARE_START[0], start_lat=SQUARE_START[1],
        initial_heading=SQUARE_INITIAL_HEADING, side_km=side_km,
        turn_direction=SQUARE_TURN_DIRECTION, speed_ms=100.0,
        n_curve_points=SQUARE_N_CURVE_POINTS, altitude=ALTITUDE, tilt=TILT,
    )
    lons = [w["longitude"] for w in waypoints]
    lats = [w["latitude"] for w in waypoints]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    center_lat = (min_lat + max_lat) / 2.0
    lat_margin_deg = margem_m / 111_320.0
    lon_margin_deg = margem_m / (111_320.0 * np.cos(np.radians(center_lat)))

    return (
        min_lon - lon_margin_deg, min_lat - lat_margin_deg,
        max_lon + lon_margin_deg, max_lat + lat_margin_deg,
    )


def listar_cenas(bbox):
    """Busca no STAC e lista TODAS as cenas retornadas (todos os anos),
    sem aplicar o filtro de 'só o ano mais recente' de baixar_mapa_naip.py --
    aqui queremos ver se existe mistura de anos antes de decidir."""
    import pystac_client
    import planetary_computer

    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(collections=["naip"], bbox=bbox)
    itens = list(search.items())
    itens.sort(key=lambda it: it.datetime, reverse=True)
    return itens


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side-km", type=float, nargs="+", default=[5.0, 6.0],
                     help="Lado(s) do quadrado candidato, em km (default: 5 6)")
    ap.add_argument("--margem-m", type=float, nargs="+", default=[3000.0, 4000.0],
                     help="Margem(ns) de download candidata(s), em metros (default: 3000 4000)")
    args = ap.parse_args()

    for side_km in args.side_km:
        for margem_m in args.margem_m:
            bbox = bbox_da_rota(side_km, margem_m)
            perimetro_km = 4 * side_km
            print(f"\n=== lado={side_km}km (perimetro={perimetro_km:.1f}km) margem={margem_m:.0f}m ===")
            print(f"    bbox (WGS84): {bbox}")

            itens = listar_cenas(bbox)
            if not itens:
                print("    [ALERTA] nenhuma cena NAIP encontrada para essa bbox.")
                continue

            anos = sorted({it.datetime.year for it in itens})
            print(f"    {len(itens)} cena(s) encontrada(s), ano(s): {anos}")
            for it in itens:
                print(f"        - {it.id}  ({it.datetime.date()})")

            if len(anos) > 1:
                print(f"    [ALERTA] bbox cruza mais de um ano de aquisicao ({anos}) -- "
                      f"baixar_mapa_naip.py hoje manteria SO {max(anos)} e descartaria "
                      f"cenas de {[a for a in anos if a != max(anos)]}, possivel buraco no mosaico.")


if __name__ == "__main__":
    main()
