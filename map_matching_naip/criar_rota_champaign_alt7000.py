"""
Geração da rota quadrada CHAMPAIGN_ALT7000 (teste de Map Matching com mapa
base Landsat, simulando uma rota de altitude ainda maior que CHAMPAIGN_ALT3500).

Cópia de criar_rota_champaign_alt3500.py com ALTITUDE=7000 (era 3500) e
BASE_ROTA apontando para uma pasta separada. Mesma localização (SQUARE_START
idêntico) e mesmo lado de 2.2km -- só a altitude muda, para isolar o efeito
da fonte/resolução do mapa base do efeito de mudar a rota.

GSD do voo a 7000m: ~16.8 m/px efetivo (vs ~7,9 m/px a 3500m, ~3,06 m/px a
1500m) -- ainda mais fino que a resolução nativa do Landsat 8/9 usado como
mapa base aqui (15 m/px pancromática / 30 m/px multiespectral RGB, ver
baixar_mapa_landsat.py), mas a diferença relativa é bem menor que no caso
Sentinel-2 (câmera "vê" só ~1,1-1,9x mais fino que o pixel do mapa base, ante
~1,3x no caso Sentinel-2 e ~10x no caso NAIP) -- degrau seguinte de altitude
da tabela de fontes em CLAUDE.md.

PENDENTE: este script só gera CSV/KML -- o VOO precisa ser gravado
manualmente no Google Earth Pro (mesmo processo das rotas anteriores) antes
de existir frames/Resized/ para rodar o map matching. O mapa base Landsat
pode ser baixado e preparado ANTES disso (baixar_mapa_landsat.py só depende
do CSV, não dos frames).

Fluxo:
    1. Rodar este script -> gera CSV + KML em mach_X/.
    2. [PENDENTE - usuário] Abrir KML_tour_*.kml no Google Earth Pro, gravar
       o voo, salvar frames em mach_X/frames/.
    3. Rodar este script de novo (ou só a etapa de resize) para gerar
       mach_X/frames/Resized/.
    4. Mapa base Landsat: baixar_mapa_landsat.py (roda independente do voo).
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (gerar_tour_kml, gerar_pontos_kml, interpolar_waypoints,
                   recortar_frame_google_earth, recortar_para_dataset,
                   FRAME_RECORTE_PX, GE_PRIMEIRO_FRAME_VALIDO)
from criar_rota_champaign import gerar_resized, reconciliar_df_com_frames

# ==========================================
# CONFIGURAÇÕES GERAIS
# ==========================================
ALTITUDE   = 7000   # metros — próximo degrau de altitude (era 3500 no ALT3500)
TILT       = 0
MACH       = 1.0
FPS        = 1

H_FOV      = 60
SENSOR_PX  = 640

# ==========================================
# PASTAS DE TRABALHO
# ==========================================
BASE_ROTA  = r"C:\Users\bbs_l\OneDrive\Leandro\ITA\MESTRADO\Tese\rotas_quadradas\CHAMPAIGN_ALT7000"
ROUTE_DIR  = os.path.join(BASE_ROTA, f"mach_{MACH}")
IMAGE_DIR  = os.path.join(ROUTE_DIR, "frames")
OUTPUT_DIR = ROUTE_DIR

# ==========================================
# CONFIGURAÇÕES DA ROTA QUADRADA (idênticas ao CHAMPAIGN original)
# ==========================================
SQUARE_START           = (-88.190, 40.085)
SQUARE_INITIAL_HEADING = 0
SQUARE_SIDE_KM         = 2.2
SQUARE_TURN_DIRECTION  = 1
SQUARE_N_CURVE_POINTS  = 30


def generate_square_route(start_lon, start_lat, initial_heading, side_km,
                           turn_direction, speed_ms, n_curve_points=30,
                           altitude=ALTITUDE, tilt=TILT):
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

            arc_len = radius * (np.pi / 2.0)
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


def get_elevation(lat_list, lon_list):
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


def calculate_fov(df, altitude=ALTITUDE, h_fov=H_FOV, sensor_px=SENSOR_PX):
    min_agl      = float(df["Altura"].min())
    hor_coverage = 2.0 * np.tan(np.deg2rad(h_fov / 2.0)) * min_agl
    pixel_size_m = hor_coverage / sensor_px
    print(f"[FOV] Altura AGL mínima               : {min_agl:.1f} m")
    print(f"[FOV] Cobertura horizontal (pior caso) : {hor_coverage:.2f} m")
    print(f"[FOV] GSD (tamanho do pixel no solo)   : {pixel_size_m:.4f} m/px  ({pixel_size_m*100:.2f} cm/px)")
    return hor_coverage, pixel_size_m


if __name__ == "__main__":
    os.makedirs(ROUTE_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    atmos          = Atmosphere(ALTITUDE)
    speed_of_sound = float(atmos.speed_of_sound[0])
    speed_ms       = MACH * speed_of_sound

    print(f"Altitude: {ALTITUDE} m | Mach: {MACH} | v = {speed_ms:.1f} m/s (som = {speed_of_sound:.1f} m/s)")
    print(f"Rota: {ROUTE_DIR}\n")

    print("Gerando rota...")
    waypoints = generate_square_route(
        start_lon=SQUARE_START[0], start_lat=SQUARE_START[1],
        initial_heading=SQUARE_INITIAL_HEADING, side_km=SQUARE_SIDE_KM,
        turn_direction=SQUARE_TURN_DIRECTION, speed_ms=speed_ms,
        n_curve_points=SQUARE_N_CURVE_POINTS, altitude=ALTITUDE, tilt=TILT,
    )
    dur_total = sum(w["duration"] for w in waypoints)
    print(f"  Waypoints : {len(waypoints)}")
    print(f"  Duração   : {dur_total:.1f} s  ({dur_total/60:.2f} min)")
    print(f"  Perímetro : {4 * SQUARE_SIDE_KM:.1f} km")

    print("Interpolando coordenadas...")
    df = interpolar_waypoints(waypoints, speed_ms, fps=FPS, altitude=ALTITUDE)
    print(f"  Amostras  : {len(df)}")

    elevations   = get_elevation(df["Lat"].tolist(), df["Long"].tolist())
    df["Altura"] = ALTITUDE - np.array(elevations)

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
    print(f"\nPRÓXIMO PASSO (pendente do usuário): abra {tour_file} no Google Earth Pro,")
    print(f"grave o voo e salve os frames PNG em: {IMAGE_DIR}")
    print(f"Depois rode este script de novo para gerar Resized/.")
