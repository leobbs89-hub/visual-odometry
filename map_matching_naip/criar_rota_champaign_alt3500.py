"""
Geração da rota quadrada CHAMPAIGN_ALT3500 (teste de Map Matching com mapa
base Sentinel-2, simulando uma rota de altitude maior / resolução menor).

Cópia de criar_rota_champaign.py com ALTITUDE=3500 (era 1500) e BASE_ROTA
apontando para uma pasta separada, para não sobrescrever a rota original de
1500m (CHAMPAIGN/) nem a de 6km (CHAMPAIGN_GRANDE/). Mesma localização
(SQUARE_START idêntico) e mesmo lado de 2.2km -- só a altitude muda.

GSD do voo a 3500m: ~7.1 m/px (vs ~3.06 m/px a 1500m) -- mais grosseiro que
a resolução nativa do NAIP (0.3m/px) usado nos testes anteriores, mas ainda
mais fino que o Sentinel-2 (10m/px) que será usado como mapa base aqui
(ver baixar_mapa_sentinel2.py).

IMPORTANTE — Conferir no Google Earth antes de gravar o voo completo:
    1. Rode este script para gerar o KML_tour.
    2. Abra o KML_tour no Google Earth Pro e veja se o quadrado cai numa
       área com boa mistura de textura (ruas, quarteirões, talhões).

Fluxo depois deste script (mesmo processo manual já usado em CHAMPAIGN/CHAMPAIGN_GRANDE):
    1. Rodar este script -> gera CSV + KML em mach_X/.
    2. Abrir KML_tour_*.kml no Google Earth Pro, gravar o voo (frames em
       mach_X/frames/, mesmo nome de padrão dos outros PNGs).
    3. Rodar este script de novo (ou só a etapa de resize) para gerar
       mach_X/frames/Resized/.
    4. Baixar o mapa base Sentinel-2 separadamente com baixar_mapa_sentinel2.py
       (não depende deste script, roda em paralelo).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import cv2
from pyproj import Geod
from ambiance import Atmosphere

# utils.py mora na raiz de visual-odometry/, um nível acima desta pasta
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (gerar_tour_kml, gerar_pontos_kml, interpolar_waypoints,
                   recortar_frame_google_earth, recortar_para_dataset,
                   FRAME_RECORTE_PX, GE_PRIMEIRO_FRAME_VALIDO)
from criar_rota_champaign import gerar_resized, reconciliar_df_com_frames

# ==========================================
# CONFIGURAÇÕES GERAIS
# ==========================================
ALTITUDE   = 3500   # metros — rota de altitude maior, pra simular GSD mais grosseiro
TILT       = 0      # câmera nadir (reto para baixo)
MACH       = 1.0    # velocidade inicial de teste; editar e rerodar para outras
FPS        = 1      # frames por segundo (pontos no CSV)

H_FOV      = 60     # FOV horizontal da câmera (graus)
SENSOR_PX  = 640    # largura do sensor em pixels

# ==========================================
# PASTAS DE TRABALHO
# ==========================================
BASE_ROTA  = r"C:\Users\bbs_l\OneDrive\Leandro\ITA\MESTRADO\Tese\rotas_quadradas\CHAMPAIGN_ALT3500"
ROUTE_DIR  = os.path.join(BASE_ROTA, f"mach_{MACH}")          # CSV + KML
IMAGE_DIR  = os.path.join(ROUTE_DIR, "frames")                 # PNGs brutos do Google Earth
OUTPUT_DIR = ROUTE_DIR

# ==========================================
# CONFIGURAÇÕES DA ROTA QUADRADA
# ==========================================
SQUARE_START           = (-88.190, 40.085)  # (longitude, latitude) — borda SE de Champaign-Urbana, IL (mesma do CHAMPAIGN original)
SQUARE_INITIAL_HEADING = 0      # proa inicial em graus (0=Norte, 90=Leste)
SQUARE_SIDE_KM         = 2.2    # lado do quadrado em km -> perímetro ~8.8km (mesmo tamanho do CHAMPAIGN original)
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
    (idêntico a criar_rota_quadrado.py — ver docstring lá para detalhes
    da geometria).
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

            arc_len = radius * math_radians_90()
            seg_dur = (arc_len / speed_ms) / n_curve_points

            for k in range(1, n_curve_points + 1):
                frac   = k / n_curve_points
                smooth = (1.0 - np.cos(frac * np.pi)) / 2.0
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


def math_radians_90():
    return np.pi / 2.0


# ==========================================
# ELEVAÇÃO DO TERRENO (API)
# ==========================================

def get_elevation(lat_list, lon_list):
    """Consulta a API Open-Elevation. Em caso de falha, retorna zeros."""
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

# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================

if __name__ == "__main__":
    os.makedirs(ROUTE_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    atmos          = Atmosphere(ALTITUDE)
    speed_of_sound = float(atmos.speed_of_sound[0])
    speed_ms       = MACH * speed_of_sound

    print(f"Altitude: {ALTITUDE} m | Mach: {MACH} | v = {speed_ms:.1f} m/s "
          f"(som = {speed_of_sound:.1f} m/s)")
    print(f"Rota: {ROUTE_DIR}\n")

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
    print(f"  Perímetro : {4 * SQUARE_SIDE_KM:.1f} km")

    # 2. Interpolação → DataFrame
    print("Interpolando coordenadas...")
    df = interpolar_waypoints(waypoints, speed_ms, fps=FPS, altitude=ALTITUDE)
    print(f"  Amostras  : {len(df)}")

    # 3. Elevação do terreno
    elevations   = get_elevation(df["Lat"].tolist(), df["Long"].tolist())
    df["Altura"] = ALTITUDE - np.array(elevations)

    # 4. KML
    tour_file = os.path.join(ROUTE_DIR, f"KML_tour_{ALTITUDE}_{MACH}.kml")
    gerar_tour_kml(df, tour_file, ALTITUDE, MACH)

    path_file = os.path.join(ROUTE_DIR, f"KML_path_{ALTITUDE}_{MACH}.kml")
    gerar_pontos_kml(df, path_file, ALTITUDE, MACH)

    # 5. Dataset = amostras que têm imagem própria (ver GE_PRIMEIRO_FRAME_VALIDO)
    df_ds = recortar_para_dataset(df)
    print(f"  Dataset   : {len(df_ds)} amostras (descartadas as "
          f"{GE_PRIMEIRO_FRAME_VALIDO} primeiras: o Movie Maker grava o render "
          f"de t=0 duas vezes e nunca salva t=1)")

    # 6. Resize — o CSV manda em quais arquivos entram
    escritos, faltando = gerar_resized(IMAGE_DIR, IMAGE_DIR, list(df_ds["File"]))
    if escritos:
        df_ds = reconciliar_df_com_frames(df_ds, faltando)

    # 7. CSV (depois da reconciliação, para bater 1:1 com Resized/)
    csv_file = os.path.join(ROUTE_DIR, f"Coord-Heading-Elev_{ALTITUDE}_{MACH}.csv")
    df_ds.to_csv(csv_file, index=False, float_format="%.8f")
    print(f"[CSV]       {csv_file}  ({len(df_ds)} linhas)")

    # 8. FOV
    calculate_fov(df_ds)

    print("\nProcesso concluído!")
    print(f"\nPróximo passo: abra {tour_file} no Google Earth Pro, confira a área,")
    print(f"grave o voo e salve os frames PNG em: {IMAGE_DIR}")
    print(f"Depois rode este script de novo para gerar Resized/.")
