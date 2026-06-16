"""
Geração de rota reta (dois pontos) simulada para testes de odometria visual.

A partir de um ponto inicial, uma proa e uma velocidade, gera um segmento
retilíneo até o ponto final definido pela distância configurada.

Produz: CSV com coordenadas interpoladas, KML Tour e KML de pontos.
"""

import os
import numpy as np
import requests
from pyproj import Geod
from ambiance import Atmosphere

from utils import gerar_tour_kml, gerar_pontos_kml, interpolar_waypoints

# ==========================================
# CONFIGURAÇÕES GERAIS
# ==========================================
ALTITUDE     = 1500   # metros AGL
TILT         = 0      # ângulo de inclinação da câmera (graus)
MACH         = 2.0    # número de Mach
FPS          = 1      # amostras por segundo no CSV

H_FOV        = 60     # FOV horizontal da câmera (graus)
SENSOR_PX    = 640    # largura do sensor em pixels

# ==========================================
# CONFIGURAÇÕES DA ROTA RETA
# ==========================================
ROUTE_START       = (-46.346074, -23.021269)  # (longitude, latitude)
ROUTE_HEADING     = 90.0    # proa em graus (0=Norte, 90=Leste, 180=Sul, 270=Oeste)
ROUTE_DISTANCE_KM = 20.0    # comprimento do segmento em km

# ==========================================
# PASTAS DE TRABALHO
# ==========================================
OUTPUT_DIR = r"C:\Users\bbs_l\OneDrive\Leandro\ITA\PMG\Google earth\1500_2_TIF"


# ==========================================
# GERAÇÃO DA ROTA RETA
# ==========================================

def generate_straight_route(start_lon, start_lat, heading, speed_ms,
                             distance_km, altitude=ALTITUDE, tilt=TILT):
    """
    Gera dois waypoints para uma rota reta.

    Args:
        start_lon, start_lat: ponto de partida.
        heading: proa em graus (0=Norte, sentido horário).
        speed_ms: velocidade em m/s.
        distance_km: comprimento do segmento em km.
        altitude: altitude AGL em metros.
        tilt: ângulo de inclinação da câmera.

    Returns:
        list[dict]: dois waypoints — partida e chegada.
    """
    g          = Geod(ellps='clrk66')
    distance_m = distance_km * 1000.0
    duration   = distance_m / speed_ms

    az_proj = heading - 360.0 * (heading > 180.0)
    end_lon, end_lat, _ = g.fwd(start_lon, start_lat, az_proj, distance_m)

    return [
        {
            "longitude":   start_lon,
            "latitude":    start_lat,
            "heading":     heading % 360,
            "duration":    0.0,
            "altitude":    altitude,
            "tilt":        tilt,
            "flight_mode": "bounce",
        },
        {
            "longitude":   end_lon,
            "latitude":    end_lat,
            "heading":     heading % 360,
            "duration":    duration,
            "altitude":    altitude,
            "tilt":        tilt,
            "flight_mode": "smooth",
        },
    ]


# ==========================================
# ELEVAÇÃO DO TERRENO (API)
# ==========================================

def get_elevation(lat_list, lon_list):
    """
    Consulta a API Open-Elevation.
    Retorna lista de elevações em metros; em caso de falha retorna zeros.
    """
    print("Consultando API de elevação...")
    url       = "https://api.open-elevation.com/api/v1/lookup"
    locations = [{"latitude": lat, "longitude": lon}
                 for lat, lon in zip(lat_list, lon_list)]
    try:
        resp = requests.post(url, json={"locations": locations}, timeout=60)
        resp.raise_for_status()
        return [r["elevation"] for r in resp.json()["results"]]
    except requests.exceptions.RequestException as e:
        print(f"  Erro na API de elevação: {e}")
        print("  Preenchendo elevação do terreno com 0 m.")
        return [0.0] * len(lat_list)


# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    atmos          = Atmosphere(ALTITUDE)
    speed_of_sound = float(atmos.speed_of_sound[0])
    speed_ms       = MACH * speed_of_sound

    print(f"Altitude: {ALTITUDE} m | Mach: {MACH} | v = {speed_ms:.1f} m/s "
          f"(som = {speed_of_sound:.1f} m/s)")
    print(f"Proa: {ROUTE_HEADING}° | Distância: {ROUTE_DISTANCE_KM} km")
    print(f"Saída: {OUTPUT_DIR}\n")

    # 1. Waypoints (dois pontos)
    print("Gerando rota reta...")
    waypoints = generate_straight_route(
        start_lon   = ROUTE_START[0],
        start_lat   = ROUTE_START[1],
        heading     = ROUTE_HEADING,
        speed_ms    = speed_ms,
        distance_km = ROUTE_DISTANCE_KM,
        altitude    = ALTITUDE,
        tilt        = TILT,
    )
    dur_total = waypoints[-1]["duration"]
    print(f"  Início    : ({waypoints[0]['latitude']:.6f}, {waypoints[0]['longitude']:.6f})")
    print(f"  Fim       : ({waypoints[1]['latitude']:.6f}, {waypoints[1]['longitude']:.6f})")
    print(f"  Duração   : {dur_total:.1f} s  ({dur_total/60:.2f} min)")

    # 2. Interpolação → DataFrame
    print("Interpolando coordenadas...")
    df = interpolar_waypoints(waypoints, speed_ms, fps=FPS, altitude=ALTITUDE)
    print(f"  Amostras  : {len(df)}")

    # 3. Elevação do terreno
    elevations   = get_elevation(df["Lat"].tolist(), df["Long"].tolist())
    df["Altura"] = ALTITUDE - np.array(elevations)

    # 4. KML
    tour_file = os.path.join(OUTPUT_DIR, f"KML_tour_reta_{ALTITUDE}_{MACH}_{int(ROUTE_HEADING)}graus.kml")
    gerar_tour_kml(df, tour_file, ALTITUDE, MACH)

    path_file = os.path.join(OUTPUT_DIR, f"KML_path_reta_{ALTITUDE}_{MACH}_{int(ROUTE_HEADING)}graus.kml")
    gerar_pontos_kml(df, path_file, ALTITUDE, MACH)

    # 5. CSV
    csv_file = os.path.join(
        OUTPUT_DIR,
        f"Coord-Heading-Elev_reta_{ALTITUDE}_{MACH}_{int(ROUTE_HEADING)}graus.csv",
    )
    df.to_csv(csv_file, index=False, float_format="%.8f")
    print(f"[CSV]       {csv_file}  ({len(df)} linhas)")

    print("\nProcesso concluído!")
