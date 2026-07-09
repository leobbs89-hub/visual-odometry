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
from utils import gerar_tour_kml, gerar_pontos_kml, interpolar_waypoints

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

def _remover_frame_extra_inicial(image_dir, arquivos):
    """
    Detecta e move pra fora do fluxo um frame "-000000.png" espúrio que o
    Google Earth às vezes grava a mais no início da captura.

    Causa: o primeiro waypoint da rota é um "bounce" de duração 0 (câmera
    salta pra posição inicial antes de qualquer interpolação de voo, ver
    generate_square_route). A gravação/extração de frames do Google Earth
    captura um frame estático nesse instante ALÉM do frame da primeira
    posição interpolada de verdade -- os dois são pixel-idênticos. Nosso
    CSV é indexado a partir de 1 ("-000001.png" = t=0), então esse frame
    extra sobra numerado "-000000.png", sem linha correspondente no CSV.

    Checagem é INCONDICIONAL sobre os dois primeiros arquivos (não depende
    de bater com nenhuma contagem esperada -- a contagem total de frames
    pode coincidir com o esperado por outro motivo, ex. o voo também
    terminou um frame curto no final, mascarando esse duplicado inicial se
    a checagem fosse só por contagem). Só age quando os dois primeiros
    frames são bit-a-bit idênticos. Move o extra para "_extra_<nome>.png"
    (mesma convenção usada manualmente antes) em vez de apagar.
    """
    if len(arquivos) < 2:
        return arquivos

    primeiro, segundo = arquivos[0], arquivos[1]
    caminho1 = os.path.join(image_dir, primeiro)
    caminho2 = os.path.join(image_dir, segundo)
    with open(caminho1, "rb") as f1, open(caminho2, "rb") as f2:
        identicos = f1.read() == f2.read()

    if not identicos:
        return arquivos

    destino = os.path.join(image_dir, f"_extra_{primeiro}")
    if os.path.exists(destino):
        print(f"[Resize] '{primeiro}' já tinha sido quarentenado antes (existe '{os.path.basename(destino)}'); "
              "ignorando de novo, sem sobrescrever.")
    else:
        os.rename(caminho1, destino)
        print(f"[Resize] '{primeiro}' é duplicata exata de '{segundo}' (bounce inicial do Google Earth) "
              f"e não tem linha no CSV -- movido para '{os.path.basename(destino)}', fora do pipeline.")
    return arquivos[1:]


def resize_images(image_dir=IMAGE_DIR, output_dir=IMAGE_DIR, expected_count=None):
    """
    Lê imagens .png de image_dir (frames brutos do Google Earth), corta para
    SENSOR_PX linhas de altura e salva em output_dir/Resized.
    Por padrão output_dir=image_dir, então o resultado fica em
    mach_X/frames/Resized/ (convenção usada em FLORESTA/URBANO).

    expected_count: número de linhas do CSV de ground truth (len(df)), só
    para o aviso final -- ver _remover_frame_extra_inicial pra remoção do
    duplicado inicial (incondicional, roda sempre que os 2 primeiros
    frames baterem).

    Retorna o número de imagens efetivamente escritas em Resized/ (depois
    de remover o duplicado inicial, se houver), para o chamador reconciliar
    contra o CSV se precisar (ver reconciliar_csv_com_frames).
    """
    if not os.path.exists(image_dir):
        print(f"[Resize] Pasta não encontrada: {image_dir}")
        print(f"[Resize] Crie a pasta e coloque lá os PNGs capturados no Google Earth antes de rodar de novo.")
        return 0

    resized_dir = os.path.join(output_dir, "Resized")
    os.makedirs(resized_dir, exist_ok=True)

    arquivos = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith(".png") and not f.startswith("_extra_")
    )
    if not arquivos:
        print(f"[Resize] Nenhuma imagem .png em: {image_dir}")
        return 0

    arquivos = _remover_frame_extra_inicial(image_dir, arquivos)

    if expected_count is not None and len(arquivos) != expected_count:
        print(f"[Resize] AVISO: {len(arquivos)} frames vão para Resized/, mas o CSV tem "
              f"{expected_count} linhas -- contagem não bate.")

    print(f"[Resize] {len(arquivos)} imagens -> {resized_dir}")
    for fname in arquivos:
        img = cv2.imread(os.path.join(image_dir, fname))
        if img is not None:
            cv2.imwrite(os.path.join(resized_dir, fname), img[:SENSOR_PX, :])
    print("[Resize] Concluído.")
    return len(arquivos)


def reconciliar_csv_com_frames(csv_path, frame_count):
    """
    Compara o número de frames efetivamente disponíveis em Resized/ (depois
    de remover o duplicado inicial) contra o número de linhas do CSV de
    ground truth, e corrige o CSV se a captura no Google Earth tiver
    parado exatamente 1 frame curta do fim da rota (visto acontecer em
    voos mais longos -- timing de captura manual/segundo-a-segundo tem
    mais chance de perder o último frame quanto mais longa a rota).

    Só age quando a diferença é EXATAMENTE 1 (frame_count == linhas - 1):
    descarta a última linha do CSV (reescreve o arquivo) e avisa. Qualquer
    outra diferença é reportada sem adivinhar.

    Retorna True se o CSV foi reescrito.
    """
    df = pd.read_csv(csv_path)
    n_csv = len(df)

    if frame_count == n_csv:
        return False

    if frame_count == n_csv - 1:
        print(f"[Reconciliação] Resized/ tem {frame_count} frames, CSV tinha {n_csv} linhas -- "
              f"a captura no Google Earth parou 1 frame curta do fim da rota (comum em voos "
              f"longos). Descartando a última linha do CSV ('{df.iloc[-1]['File']}', sem frame "
              "correspondente) e regravando.")
        df.iloc[:-1].to_csv(csv_path, index=False, float_format="%.8f")
        return True

    print(f"[Reconciliação] AVISO: Resized/ tem {frame_count} frames, CSV tem {n_csv} linhas "
          f"(diferença de {n_csv - frame_count}) -- não é o padrão conhecido de 1 frame curto, "
          "não vou adivinhar. Confira manualmente antes de rodar o pipeline.")
    return False


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

    # 5. CSV
    csv_file = os.path.join(ROUTE_DIR, f"Coord-Heading-Elev_{ALTITUDE}_{MACH}.csv")
    df.to_csv(csv_file, index=False, float_format="%.8f")
    print(f"[CSV]       {csv_file}  ({len(df)} linhas)")

    # 6. FOV
    calculate_fov(df)

    # 7. Resize (só faz algo depois que você gravar os PNGs em IMAGE_DIR)
    n_frames = resize_images(IMAGE_DIR, IMAGE_DIR, expected_count=len(df))
    if n_frames:
        reconciliar_csv_com_frames(csv_file, n_frames)

    print("\nProcesso concluído!")
    print(f"\nPróximo passo: abra {tour_file} no Google Earth Pro, confira a área,")
    print(f"grave o voo e salve os frames PNG em: {IMAGE_DIR}")
    print(f"Depois rode este script de novo para gerar Resized/.")
