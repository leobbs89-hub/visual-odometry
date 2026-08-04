"""
Geração da rota quadrada CHAMPAIGN (teste de Map Matching com mapa base NAIP).

Adaptado de criar_rota_quadrado.py — mesma lógica de geração de waypoints,
só muda o local, o tamanho do lado e a organização de pastas (CSV/KML ficam
em mach_X/, frames + Resized ficam em mach_X/frames/, batendo com a
convenção usada em Tese/rotas_quadradas/FLORESTA e URBANO).

Localização: borda de Champaign-Urbana, IL (EUA) — mistura de malha urbana
e terreno agrícola plano. Escolhida de propósito com relevo baixo: como o
mapa base do map matching vem do NAIP (nadir puro) e os frames de voo vêm
de captura no Google Earth (terreno 3D), relevo acentuado introduziria erro
de paralaxe que confundiria o teste. Ver CLAUDE.md / conversa sobre a
escolha da candidata.

IMPORTANTE — Conferir no Google Earth antes de gravar o voo completo:
    1. Rode este script para gerar o KML_tour.
    2. Abra o KML_tour no Google Earth Pro e veja se o quadrado cai numa
       área com boa mistura de textura (ruas, quarteirões, talhões).
       Ajuste SQUARE_START/SQUARE_INITIAL_HEADING abaixo se quiser deslocar
       a rota antes de gravar as imagens de verdade.

Fluxo depois deste script (mesmo processo manual já usado em FLORESTA/URBANO):
    1. Rodar este script -> gera CSV + KML em mach_X/.
    2. Abrir KML_tour_*.kml no Google Earth Pro, gravar o voo (frames em
       mach_X/frames/, mesmo nome de padrão dos outros PNGs).
    3. Rodar este script de novo (ou só a etapa de resize) para gerar
       mach_X/frames/Resized/.
    4. Baixar o mapa base NAIP separadamente com baixar_mapa_naip.py
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

# ==========================================
# CONFIGURAÇÕES GERAIS
# ==========================================
ALTITUDE   = 1500   # metros — mesmo valor do resto do dataset
TILT       = 0      # câmera nadir (reto para baixo)
MACH       = 1.0    # velocidade inicial de teste; editar e rerodar para outras
FPS        = 1      # frames por segundo (pontos no CSV)

H_FOV      = 60     # FOV horizontal da câmera (graus)
SENSOR_PX  = 640    # largura do sensor em pixels

# ==========================================
# PASTAS DE TRABALHO
# ==========================================
BASE_ROTA  = r"C:\Users\bbs_l\OneDrive\Leandro\ITA\MESTRADO\Tese\rotas_quadradas\CHAMPAIGN"
ROUTE_DIR  = os.path.join(BASE_ROTA, f"mach_{MACH}")          # CSV + KML
IMAGE_DIR  = os.path.join(ROUTE_DIR, "frames")                 # PNGs brutos do Google Earth
OUTPUT_DIR = ROUTE_DIR

# ==========================================
# CONFIGURAÇÕES DA ROTA QUADRADA
# ==========================================
SQUARE_START           = (-88.190, 40.085)  # (longitude, latitude) — borda SE de Champaign-Urbana, IL
SQUARE_INITIAL_HEADING = 0      # proa inicial em graus (0=Norte, 90=Leste)
SQUARE_SIDE_KM         = 2.2    # lado do quadrado em km -> perímetro ~8.8km (< 10km pedido)
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

def gerar_resized(image_dir, output_dir, arquivos_desejados):
    """
    Escreve em output_dir/Resized o recorte de EXATAMENTE os arquivos que o
    CSV de ground truth pede, na ordem em que ele pede.

    Antes isto varria a pasta e adivinhava quais frames descartar (duplicata
    inicial byte-a-byte, linha fantasma no fim, captura curta). A adivinhação
    deixou de ser necessária: `interpolar_waypoints` passou a nomear cada
    amostra pelo segundo de voo e `recortar_para_dataset` já remove as que não
    têm imagem própria (ver utils.GE_PRIMEIRO_FRAME_VALIDO). O CSV virou a
    autoridade sobre quais arquivos entram, e o que sobra na pasta de brutos
    (as duas escritas duplicadas do início, frames além do fim do voo) fica
    simplesmente sem ser referenciado, sem precisar ser movido nem apagado.

    Retorna (escritos, faltando) — nomes de arquivo, para o chamador
    reconciliar o CSV.
    """
    if not os.path.exists(image_dir):
        print(f"[Resize] Pasta não encontrada: {image_dir}")
        print("[Resize] Grave os PNGs no Google Earth antes de rodar de novo.")
        return [], list(arquivos_desejados)

    resized_dir = os.path.join(output_dir, "Resized")
    os.makedirs(resized_dir, exist_ok=True)

    escritos, faltando = [], []
    for fname in arquivos_desejados:
        origem = os.path.join(image_dir, fname)
        img = cv2.imread(origem) if os.path.exists(origem) else None
        if img is None:
            faltando.append(fname)
            continue
        cv2.imwrite(os.path.join(resized_dir, fname),
                    recortar_frame_google_earth(img))
        escritos.append(fname)

    if escritos:
        print(f"[Resize] {len(escritos)} imagens -> {resized_dir} "
              f"(recorte {FRAME_RECORTE_PX}x{FRAME_RECORTE_PX} centrado no nadir)")
    if faltando:
        print(f"[Resize] {len(faltando)} arquivos pedidos pelo CSV não existem "
              f"na pasta de brutos (ex.: {faltando[0]}"
              + (f" … {faltando[-1]}" if len(faltando) > 1 else "") + ")")
    return escritos, faltando


def reconciliar_df_com_frames(df, faltando):
    """
    Descarta do ground truth as amostras cujo frame não foi capturado.

    Só descarta no FIM da rota, que é o padrão observado (a captura manual
    segundo-a-segundo tende a parar antes do último instante, mais provável em
    voos longos). Buraco no meio é sintoma de outra coisa — avisa e mantém a
    linha, para não mascarar o problema.
    """
    if not faltando:
        return df

    faltantes = set(faltando)
    marcado = df["File"].isin(faltantes)
    # Índices ausentes que formam um sufixo contíguo do DataFrame
    idx_falt = set(df.index[marcado])
    corte = len(df)
    while corte - 1 in idx_falt:
        corte -= 1

    no_meio = sorted(i for i in idx_falt if i < corte)
    if no_meio:
        print(f"[Reconciliação] AVISO: {len(no_meio)} frame(s) faltando no MEIO "
              f"da rota (linhas {no_meio[:5]}{'…' if len(no_meio) > 5 else ''}) — "
              "não é o padrão de captura curta no fim. Confira manualmente.")

    if corte < len(df):
        print(f"[Reconciliação] Captura parou {len(df) - corte} frame(s) antes do "
              f"fim da rota; descartando essas linhas do CSV.")
        df = df.iloc[:corte].reset_index(drop=True)
    return df


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

    # 4. KML — sempre a rota COMPLETA: o voo tem de começar em t=0, mesmo que
    #    as primeiras amostras não entrem no dataset (passo 5).
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
