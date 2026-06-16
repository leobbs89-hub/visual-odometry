"""
Geração de rota quadrada simulada para testes de odometria visual.

Produz: waypoints geodésicos com curvas suaves de 90°, CSV com coordenadas
interpoladas, KML Tour e KML de pontos para visualização no Google Earth.
"""

import os
import math
import numpy as np
import pandas as pd
import requests
import cv2
from pyproj import Geod
from ambiance import Atmosphere

from utils import gerar_tour_kml, gerar_pontos_kml

# ==========================================
# CONFIGURAÇÕES GERAIS
# ==========================================
ALTITUDE   = 1500   # metros
TILT       = 0      # ângulo de inclinação da câmera
MACH       = 2.0    # número de Mach
FPS        = 1      # frames por segundo (pontos no CSV)

H_FOV      = 60     # FOV horizontal da câmera (graus)
SENSOR_PX  = 640    # largura do sensor em pixels

# ==========================================
# PASTAS DE TRABALHO
# ==========================================
IMAGE_DIR  = r"C:\Users\bbs_l\OneDrive\Leandro\ITA\PMG\Google earth\1500_2_TIF"
OUTPUT_DIR = r"C:\Users\bbs_l\OneDrive\Leandro\ITA\PMG\Google earth\1500_2_TIF"

# ==========================================
# CONFIGURAÇÕES DA ROTA QUADRADA
# ==========================================
SQUARE_START           = (-46.346074, -23.021269)  # (longitude, latitude)
SQUARE_INITIAL_HEADING = 0      # proa inicial em graus (0=Norte, 90=Leste)
SQUARE_SIDE_KM         = 10     # lado do quadrado em km
SQUARE_TURN_DIRECTION  = 1      # +1 = direita (horário), -1 = esquerda
SQUARE_N_CURVE_POINTS  = 30     # pontos por curva de 90° (mais = mais suave)


# ==========================================
# GERAÇÃO DA ROTA QUADRADA
# ==========================================

def generate_square_route(start_lon, start_lat, initial_heading, side_km,
                           turn_direction, speed_ms, n_curve_points=30,
                           altitude=ALTITUDE, tilt=TILT):
    """
    Gera waypoints para uma rota quadrada com 3 curvas de 90° suaves.

    Geometria:
      O ponto de partida é o canto C0 do quadrado ideal (vértice).
      Cada lado tem um canto virtual a side_km do anterior.
      Nas 3 primeiras curvas, a reta antes do arco é encurtada em r (raio),
      e o arco de 90° (ease in-out) conecta suavemente os lados adjacentes.
      O último lado encurta em r para que a rota retorne ao ponto C0.

      Comprimentos de reta entre pontos de tangência:
        Lado 0 (sem arco de entrada): reta = side - r
        Lados 1 e 2 (arco entrada + arco saída): reta = side - 2r
        Lado 3 (sem arco de saída): reta = side - r
      O raio r = 10% do lado.

    O fechamento fim→início é < 10 m (erro geodésico aceitável).
    """
    g = Geod(ellps='clrk66')
    side_m         = side_km * 1000.0
    radius         = side_m * 0.10
    turn_angle_deg = 90.0 * turn_direction
    hdgs = [(initial_heading + 90.0 * turn_direction * i) % 360.0 for i in range(4)]

    waypoints = []

    def add_wp(lon, lat, hdg, dur, flight_mode="smooth"):
        waypoints.append({
            "longitude":   lon,
            "latitude":    lat,
            "heading":     hdg % 360,
            "duration":    max(dur, 0.0),
            "altitude":    altitude,
            "tilt":        tilt,
            "flight_mode": flight_mode,
        })

    def move(lon, lat, az_deg, dist_m):
        az_proj = az_deg - 360.0 * (az_deg > 180.0)
        elo, ela, _ = g.fwd(lon, lat, az_proj, dist_m)
        return elo, ela

    cur_lon, cur_lat = start_lon, start_lat
    add_wp(cur_lon, cur_lat, hdgs[0], 0.0, "bounce")

    for side_idx in range(4):
        hdg = hdgs[side_idx]

        if side_idx == 3:
            reta = side_m - radius
            end_lon, end_lat = move(cur_lon, cur_lat, hdg, reta)
            add_wp(end_lon, end_lat, hdg, reta / speed_ms)
        else:
            reta = (side_m - radius) if side_idx == 0 else (side_m - 2.0 * radius)
            cs_lon, cs_lat = move(cur_lon, cur_lat, hdg, reta)
            add_wp(cs_lon, cs_lat, hdg, reta / speed_ms)

            perp_hdg  = hdg + 90.0 * turn_direction
            cen_lon, cen_lat = move(cs_lon, cs_lat, perp_hdg, radius)
            ang_start = (perp_hdg + 180.0) % 360.0

            arc_len = radius * math.radians(90.0)
            seg_dur = (arc_len / speed_ms) / n_curve_points

            for k in range(1, n_curve_points + 1):
                frac   = k / n_curve_points
                smooth = (1.0 - math.cos(frac * math.pi)) / 2.0
                ang_now = ang_start + smooth * turn_angle_deg
                pt_lon, pt_lat = move(cen_lon, cen_lat, ang_now, radius)
                new_hdg = (hdg + smooth * turn_angle_deg) % 360.0
                add_wp(pt_lon, pt_lat, new_hdg, seg_dur)

            cur_lon = waypoints[-1]["longitude"]
            cur_lat = waypoints[-1]["latitude"]

    for i in range(1, len(waypoints)):
        az_fwd, _, dist = g.inv(
            waypoints[i-1]["longitude"], waypoints[i-1]["latitude"],
            waypoints[i]["longitude"],   waypoints[i]["latitude"],
        )
        az_fwd += (az_fwd < 0) * 360
        waypoints[i]["duration"] = dist / speed_ms
        waypoints[i]["heading"]  = az_fwd

    return waypoints


# ==========================================
# INTERPOLAÇÃO → CSV
# ==========================================

def interpolate_to_dataframe(waypoints, speed_ms, fps=FPS, altitude=ALTITUDE):
    """
    Interpola os waypoints em amostras regulares de 1/fps segundos.

    Retorna DataFrame com colunas: File, Lat, Long, Proa, Altura.
    """
    g    = Geod(ellps='clrk66')
    step = 1.0 / fps

    total_time = sum(wp["duration"] for wp in waypoints)
    timestamps = np.arange(0.0, total_time + step, step)
    n          = len(timestamps)

    file_list = [f"-{i:06d}.png" for i in range(1, n + 1)]

    cum_times = np.cumsum([wp["duration"] for wp in waypoints])

    lats, lons, proas = [], [], []
    seg = 0

    for t in timestamps:
        while seg < len(waypoints) - 2 and t > cum_times[seg + 1]:
            seg += 1

        wp0 = waypoints[seg]
        wp1 = waypoints[seg + 1]

        seg_dur = wp1["duration"]
        frac = (t - cum_times[seg]) / seg_dur if seg_dur > 0 else 1.0
        frac = min(max(frac, 0.0), 1.0)

        if frac <= 0.0:
            lons.append(wp0["longitude"])
            lats.append(wp0["latitude"])
            proas.append(wp0["heading"])
        elif frac >= 1.0:
            lons.append(wp1["longitude"])
            lats.append(wp1["latitude"])
            proas.append(wp1["heading"])
        else:
            dist = frac * speed_ms * seg_dur
            az   = wp0["heading"] - 360.0 * (wp0["heading"] > 180.0)
            elo, ela, _ = g.fwd(wp0["longitude"], wp0["latitude"], az, dist)
            dh = wp1["heading"] - wp0["heading"]
            if dh >  180: dh -= 360
            if dh < -180: dh += 360
            lons.append(elo)
            lats.append(ela)
            proas.append((wp0["heading"] + frac * dh) % 360)

    df = pd.DataFrame({
        "File":        file_list[:n],
        "Lat":         lats,
        "Long":        lons,
        "Proa":        proas,
        "Altura":      float(altitude),
        "duration":    0.0,
        "heading":     proas,
        "altitude":    float(altitude),
        "tilt":        0,
        "flight_mode": "bounce",
    })

    for i in range(1, len(df)):
        az_fwd, _, dist = g.inv(
            df.iloc[i-1]["Long"], df.iloc[i-1]["Lat"],
            df.iloc[i]["Long"],   df.iloc[i]["Lat"],
        )
        az_fwd += (az_fwd < 0) * 360
        df.loc[i, "duration"]    = dist / speed_ms
        df.loc[i, "heading"]     = az_fwd
        df.loc[i, "flight_mode"] = "smooth"

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
# FOV
# ==========================================

def calculate_fov(df, altitude=ALTITUDE, h_fov=H_FOV, sensor_px=SENSOR_PX):
    """Calcula cobertura horizontal e GSD no solo."""
    min_agl      = float(df["Altura"].min())
    hor_coverage = 2.0 * np.tan(np.deg2rad(h_fov / 2.0)) * min_agl
    pixel_size_m = hor_coverage / sensor_px

    print(f"[FOV] Altura AGL mínima               : {min_agl:.1f} m")
    print(f"[FOV] Cobertura horizontal (pior caso) : {hor_coverage:.2f} m")
    print(f"[FOV] GSD (tamanho do pixel no solo)   : {pixel_size_m:.4f} m/px  ({pixel_size_m*100:.2f} cm/px)")
    return hor_coverage, pixel_size_m


# ==========================================
# REDIMENSIONAMENTO DE IMAGENS
# ==========================================

def resize_images(image_dir=IMAGE_DIR, output_dir=OUTPUT_DIR):
    """
    Lê imagens .png de image_dir, corta para SENSOR_PX linhas de altura
    e salva em output_dir/Resized.
    """
    if not os.path.exists(image_dir):
        print(f"[Resize] Pasta não encontrada: {image_dir}")
        return

    resized_dir = os.path.join(output_dir, "Resized")
    os.makedirs(resized_dir, exist_ok=True)

    arquivos = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(".png"))
    if not arquivos:
        print(f"[Resize] Nenhuma imagem .png em: {image_dir}")
        return

    print(f"[Resize] {len(arquivos)} imagens → {resized_dir}")
    for fname in arquivos:
        img = cv2.imread(os.path.join(image_dir, fname))
        if img is not None:
            cv2.imwrite(os.path.join(resized_dir, fname), img[:SENSOR_PX, :])
    print("[Resize] Concluído.")


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
    print(f"Saída: {OUTPUT_DIR}\n")

    # 1. Waypoints
    print("Gerando rota...")
    waypoints = generate_square_route(
        start_lon       = SQUARE_START[0],
        start_lat       = SQUARE_START[1],
        initial_heading = SQUARE_INITIAL_HEADING,
        side_km         = SQUARE_SIDE_KM,
        turn_direction  = SQUARE_TURN_DIRECTION,
        speed_ms        = speed_ms,
        n_curve_points  = SQUARE_N_CURVE_POINTS,
        altitude        = ALTITUDE,
        tilt            = TILT,
    )
    dur_total = sum(w["duration"] for w in waypoints)
    print(f"  Waypoints : {len(waypoints)}")
    print(f"  Duração   : {dur_total:.1f} s  ({dur_total/60:.2f} min)")

    # 2. Interpolação → DataFrame
    print("Interpolando coordenadas...")
    df = interpolate_to_dataframe(waypoints, speed_ms, fps=FPS, altitude=ALTITUDE)
    print(f"  Amostras  : {len(df)}")

    # 3. Elevação do terreno
    elevations   = get_elevation(df["Lat"].tolist(), df["Long"].tolist())
    df["Altura"] = ALTITUDE - np.array(elevations)

    # 4. KML
    tour_file = os.path.join(OUTPUT_DIR, f"KML_tour_{ALTITUDE}_{MACH}.kml")
    gerar_tour_kml(df, tour_file, ALTITUDE, MACH)

    path_file = os.path.join(OUTPUT_DIR, f"KML_path_{ALTITUDE}_{MACH}.kml")
    gerar_pontos_kml(df, path_file, ALTITUDE, MACH)

    # 5. CSV
    csv_file = os.path.join(OUTPUT_DIR, f"Coord-Heading-Elev_{ALTITUDE}_{MACH}.csv")
    df.to_csv(csv_file, index=False, float_format="%.8f")
    print(f"[CSV]       {csv_file}  ({len(df)} linhas)")

    # 6. FOV
    calculate_fov(df)

    # 7. Resize
    resize_images(IMAGE_DIR, OUTPUT_DIR)

    print("\nProcesso concluído!")
