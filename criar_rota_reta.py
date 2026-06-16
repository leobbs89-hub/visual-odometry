"""
Geração de par de frames retos para testes de odometria visual.

A partir de um ponto inicial, uma proa e uma velocidade, calcula o ponto
exatamente 1 frame à frente (distância = speed_ms / FPS) e produz:
- 2 FlyTos no KML Tour (posição da câmera em frame N e frame N+1)
- CSV com 2 linhas (ground truth do par de frames)
- KML de pontos com os 2 Placemarks
"""

import os
import numpy as np
import pandas as pd
import requests
from pyproj import Geod
from ambiance import Atmosphere

from utils import gerar_tour_kml, gerar_pontos_kml

# ==========================================
# CONFIGURAÇÕES GERAIS
# ==========================================
ALTITUDE = 1500   # metros AGL
TILT     = 0      # ângulo de inclinação da câmera (graus)
MACH     = 2.0    # número de Mach
FPS      = 1      # frames por segundo — define a distância entre os dois pontos

H_FOV    = 60     # FOV horizontal da câmera (graus)
SENSOR_PX = 640   # largura do sensor em pixels

# ==========================================
# CONFIGURAÇÕES DA ROTA
# ==========================================
ROUTE_START   = (-46.346074, -23.021269)  # (longitude, latitude)
ROUTE_HEADING = 90.0  # proa em graus (0=Norte, 90=Leste, 180=Sul, 270=Oeste)

# ==========================================
# PASTAS DE TRABALHO
# ==========================================
OUTPUT_DIR = r"C:\Users\bbs_l\OneDrive\Leandro\ITA\PMG\Google earth\1500_2_TIF"


# ==========================================
# GERAÇÃO DO PAR DE FRAMES
# ==========================================

def generate_straight_route(start_lon, start_lat, heading, speed_ms,
                             fps, altitude=ALTITUDE, tilt=TILT):
    """
    Gera exatamente 2 waypoints separados pelo deslocamento de 1 frame.

    A distância entre os pontos é speed_ms / fps metros, correspondendo
    ao quanto a aeronave avança entre dois frames consecutivos.

    Args:
        start_lon, start_lat: posição da câmera no frame N.
        heading: proa em graus (0=Norte, sentido horário).
        speed_ms: velocidade em m/s.
        fps: frames por segundo.
        altitude: altitude AGL em metros.
        tilt: ângulo de inclinação da câmera.

    Returns:
        list[dict]: [frame N, frame N+1].
    """
    g          = Geod(ellps='clrk66')
    distance_m = speed_ms / fps          # deslocamento em 1 frame
    duration   = 1.0 / fps               # duração de 1 frame em segundos

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


def waypoints_to_dataframe(waypoints, altitude=ALTITUDE):
    """Converte lista de waypoints diretamente em DataFrame (sem interpolação)."""
    rows = []
    for i, wp in enumerate(waypoints):
        rows.append({
            "File":        f"-{i+1:06d}.png",
            "Lat":         wp["latitude"],
            "Long":        wp["longitude"],
            "Proa":        wp["heading"],
            "Altura":      float(altitude),
            "duration":    wp["duration"],
            "heading":     wp["heading"],
            "altitude":    float(wp["altitude"]),
            "tilt":        wp["tilt"],
            "flight_mode": wp["flight_mode"],
        })
    df = pd.DataFrame(rows)
    df["proa"] = df["heading"]
    return df


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
    dist_por_frame = speed_ms / FPS

    print(f"Altitude: {ALTITUDE} m | Mach: {MACH} | v = {speed_ms:.1f} m/s")
    print(f"Proa: {ROUTE_HEADING}° | FPS: {FPS} | Distância por frame: {dist_por_frame:.1f} m")
    print(f"Saída: {OUTPUT_DIR}\n")

    # 1. Par de waypoints (frame N e frame N+1)
    print("Gerando par de frames...")
    waypoints = generate_straight_route(
        start_lon = ROUTE_START[0],
        start_lat = ROUTE_START[1],
        heading   = ROUTE_HEADING,
        speed_ms  = speed_ms,
        fps       = FPS,
        altitude  = ALTITUDE,
        tilt      = TILT,
    )
    print(f"  Frame N   : ({waypoints[0]['latitude']:.6f}, {waypoints[0]['longitude']:.6f})")
    print(f"  Frame N+1 : ({waypoints[1]['latitude']:.6f}, {waypoints[1]['longitude']:.6f})")

    # 2. DataFrame direto dos 2 waypoints
    df = waypoints_to_dataframe(waypoints, altitude=ALTITUDE)

    # 3. Elevação do terreno
    elevations   = get_elevation(df["Lat"].tolist(), df["Long"].tolist())
    df["Altura"] = ALTITUDE - np.array(elevations)

    # 4. KML
    tag = f"{ALTITUDE}_{MACH}_{int(ROUTE_HEADING)}graus"
    tour_file = os.path.join(OUTPUT_DIR, f"KML_tour_reta_{tag}.kml")
    gerar_tour_kml(df, tour_file, ALTITUDE, MACH)

    path_file = os.path.join(OUTPUT_DIR, f"KML_path_reta_{tag}.kml")
    gerar_pontos_kml(df, path_file, ALTITUDE, MACH)

    # 5. CSV
    csv_file = os.path.join(OUTPUT_DIR, f"Coord-Heading-Elev_reta_{tag}.csv")
    df.to_csv(csv_file, index=False, float_format="%.8f")
    print(f"[CSV]       {csv_file}  ({len(df)} linhas)")

    print("\nProcesso concluído!")
