"""
Geração da rota quadrada CHAMPAIGN_GRANDE (teste de Map Matching com ROI
maior) -- variante de criar_rota_champaign.py com lado ~3x maior, pra dar
mais espaço ao módulo de map matching se recuperar de deriva de
dead-reckoning entre correções (ver achado "Com MM pior que Sem MM" na rota
original de 8.8km, documentado em CLAUDE.md desta pasta).

Mesmo local (borda de Champaign-Urbana, IL), mesma altitude/câmera/mach --
só o lado do quadrado muda. Cobertura NAIP da bbox maior + margem já checada
com verificar_cobertura_naip.py (lado 6km, margem 4000m -> 9 cenas, todas do
mesmo ano 2023, 0.3m/px, sem gap espacial).

Saída em pasta separada (CHAMPAIGN_GRANDE) para não sobrescrever a rota de
8.8km já validada em CHAMPAIGN/.

Fluxo (mesmo processo manual de criar_rota_champaign.py):
    1. Rodar este script -> gera CSV + KML em mach_X/.
    2. Abrir KML_tour_*.kml no Google Earth Pro, gravar o voo (frames em
       mach_X/frames/). Rota ~3x maior -> proporcionalmente mais frames
       (mesmo FPS=1) -> captura manual mais longa que a rota original.
    3. Rodar este script de novo para gerar mach_X/frames/Resized/.
    4. Baixar o mapa base NAIP com baixar_mapa_naip.py --margem-m 4000
       (mapa bem maior que o da rota original -- checar espaço em disco).
"""

import os
import sys
from pathlib import Path

import numpy as np
import requests
import cv2
from pyproj import Geod
from ambiance import Atmosphere

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (gerar_tour_kml, gerar_pontos_kml, interpolar_waypoints,
                   recortar_para_dataset, GE_PRIMEIRO_FRAME_VALIDO)
from criar_rota_champaign import (
    generate_square_route, calculate_fov, get_elevation, gerar_resized,
    reconciliar_df_com_frames,
)

# ==========================================
# CONFIGURAÇÕES GERAIS (iguais à rota original)
# ==========================================
ALTITUDE   = 1500
TILT       = 0
MACH       = 1.0
FPS        = 1

H_FOV      = 60
SENSOR_PX  = 640

# ==========================================
# PASTAS DE TRABALHO (pasta separada da rota de 8.8km)
# ==========================================
BASE_ROTA  = r"C:\Users\bbs_l\OneDrive\Leandro\ITA\MESTRADO\Tese\rotas_quadradas\CHAMPAIGN_GRANDE"
ROUTE_DIR  = os.path.join(BASE_ROTA, f"mach_{MACH}")
IMAGE_DIR  = os.path.join(ROUTE_DIR, "frames")
OUTPUT_DIR = ROUTE_DIR

# ==========================================
# CONFIGURAÇÕES DA ROTA QUADRADA
# ==========================================
SQUARE_START           = (-88.190, 40.085)  # mesmo ponto de partida da rota original
SQUARE_INITIAL_HEADING = 0
SQUARE_SIDE_KM         = 6.0    # lado maior -> perimetro ~24km (vs 8.8km da rota original)
SQUARE_TURN_DIRECTION  = 1
SQUARE_N_CURVE_POINTS  = 30


if __name__ == "__main__":
    os.makedirs(ROUTE_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    atmos          = Atmosphere(ALTITUDE)
    speed_of_sound = float(atmos.speed_of_sound[0])
    speed_ms       = MACH * speed_of_sound

    print(f"Altitude: {ALTITUDE} m | Mach: {MACH} | v = {speed_ms:.1f} m/s "
          f"(som = {speed_of_sound:.1f} m/s)")
    print(f"Rota: {ROUTE_DIR}\n")

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

    print("Interpolando coordenadas...")
    df = interpolar_waypoints(waypoints, speed_ms, fps=FPS, altitude=ALTITUDE)
    print(f"  Amostras  : {len(df)}")

    elevations   = get_elevation(df["Lat"].tolist(), df["Long"].tolist())
    df["Altura"] = ALTITUDE - np.array(elevations)

    tour_file = os.path.join(ROUTE_DIR, f"KML_tour_{ALTITUDE}_{MACH}.kml")
    gerar_tour_kml(df, tour_file, ALTITUDE, MACH)

    path_file = os.path.join(ROUTE_DIR, f"KML_path_{ALTITUDE}_{MACH}.kml")
    gerar_pontos_kml(df, path_file, ALTITUDE, MACH)

    df_ds = recortar_para_dataset(df)
    print(f"  Dataset   : {len(df_ds)} amostras (descartadas as "
          f"{GE_PRIMEIRO_FRAME_VALIDO} primeiras: o Movie Maker grava o render "
          f"de t=0 duas vezes e nunca salva t=1)")

    escritos, faltando = gerar_resized(IMAGE_DIR, IMAGE_DIR, list(df_ds["File"]))
    if escritos:
        df_ds = reconciliar_df_com_frames(df_ds, faltando)

    csv_file = os.path.join(ROUTE_DIR, f"Coord-Heading-Elev_{ALTITUDE}_{MACH}.csv")
    df_ds.to_csv(csv_file, index=False, float_format="%.8f")
    print(f"[CSV]       {csv_file}  ({len(df_ds)} linhas)")

    calculate_fov(df_ds, altitude=ALTITUDE, h_fov=H_FOV, sensor_px=SENSOR_PX)

    print("\nProcesso concluído!")
    print(f"\nPróximo passo: abra {tour_file} no Google Earth Pro, confira a área,")
    print(f"grave o voo e salve os frames PNG em: {IMAGE_DIR}")
    print(f"Depois rode este script de novo para gerar Resized/.")
