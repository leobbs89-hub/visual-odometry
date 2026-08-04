# utils.py
"""Funções utilitárias puras (sem estado) usadas pelo pipeline de odometria."""

import os
import numpy as np
import pandas as pd
from pyproj import Geod


# ==========================================================================
# RECORTE DOS FRAMES GRAVADOS NO GOOGLE EARTH
# ==========================================================================
# Constantes medidas empiricamente (2026-08-03) — ver
# Tese/investigacao_offset_google_earth_causa_raiz.md

#: Dimensões (altura, largura) do PNG bruto gravado no Google Earth Pro.
#: Conferido em FLORESTA, URBANO, CHAMPAIGN e CHAMPAIGN_ALT3500 — todas 700x640.
FRAME_BRUTO_GE = (700, 640)

#: Primeira linha ocupada por legenda sobreposta. Há DUAS faixas de overlay,
#: medidas por realce do piso mínimo sobre 116 frames de FLORESTA e URBANO
#: (as duas rotas concordam linha a linha):
#:   - atribuição ("Image (c) 2026 Airbus"): linhas 641..653, colunas 259..628
#:   - marca d'água "Google Earth":          linhas 681..692, colunas 540..628
#: Acima da linha 641 o realce máximo é ~25 (ruído de terreno); dentro das
#: faixas passa de 240. Logo a última linha de imagem limpa é a 640.
LEGENDA_TOPO_Y = 641

#: Lado do recorte quadrado. 576 é o maior múltiplo de 32 que cabe centrado
#: no nadir sem tocar a legenda (o teto absoluto seria 582). Múltiplo de 32
#: importa: LoFTR/MatchFormer exigem dimensões divisíveis por 8, e o 640
#: anterior também era múltiplo de 32 — manter a propriedade evita padding
#: interno inesperado nos detectores neurais.
FRAME_RECORTE_PX = 576

#: FOV horizontal do viewport BRUTO do Google Earth (os 640 px de largura).
#: Validado indiretamente: como DIST_EST é proporcional a 1/fx, a razão
#: DIST_EST/DIST_REAL dos CSVs de resultado mede essa calibração — deu
#: mediana 1,003 em AKAZE/ORB/SIFT, descartando a hipótese de que os 60 graus
#: seriam verticais (que preveria 1,094). Ver a seção "C" de
#: Tese/investigacao_offset_google_earth_causa_raiz.md.
GE_VIEWPORT_H_FOV_DEG = 60.0

#: Índice do primeiro arquivo cujo número volta a bater com o segundo de voo.
#:
#: O Movie Maker do Google Earth grava o render de t=0 DUAS vezes
#: (`-000000.png` e `-000001.png` saem byte-idênticos) e **nunca salva o frame
#: de t=1 s**. De `-000002.png` em diante o índice do arquivo volta a ser o
#: segundo de voo: `-00000k.png` mostra t = k.
#:
#: Medido em CHAMPAIGN por correlação de fase contra o mapa satelital
#: georreferenciado (`map_matching_naip/identificar_linha_do_frame.py`):
#: resíduo de 2–5 m na linha certa contra 300–650 m nas vizinhas, isolamento
#: de 11× a 34×. Ver Tese/investigacao_offset_google_earth_causa_raiz.md,
#: seção B.
#:
#: Consequência: as duas primeiras amostras da rota não têm imagem própria
#: utilizável (a de t=0 tem, mas emendaria um passo de 2 s com a de t=2), então
#: o dataset começa em t=2 — ver recortar_para_dataset().
#:
#: ATENÇÃO: isto foi medido na captura 640×700. Ao mudar a janela do Movie
#: Maker, REMEÇA com identificar_linha_do_frame.py em vez de assumir que
#: continua 2.
GE_PRIMEIRO_FRAME_VALIDO = 2


def nome_frame(t_segundos):
    """Nome do PNG que o Google Earth grava para o instante t (em segundos)."""
    return f"-{int(t_segundos):06d}.png"


def recortar_para_dataset(df, primeiro=GE_PRIMEIRO_FRAME_VALIDO):
    """
    Recorta o DataFrame interpolado para as amostras que têm imagem própria.

    `interpolar_waypoints` devolve a rota COMPLETA (t = 0, 1, 2, … s), que é o
    que o tour KML precisa — o voo tem de começar em t=0. Mas o dataset de
    ground truth só pode conter amostras com um frame correspondente de
    verdade: por causa da escrita duplicada do Movie Maker (ver
    GE_PRIMEIRO_FRAME_VALIDO), o frame de t=1 não existe, e manter t=0 emendaria
    um passo de 2 s no meio de uma série de passos de 1 s.

    Por isso o dataset começa em t = `primeiro`. Custa as duas primeiras
    amostras da rota e garante espaçamento uniforme em todo o resto.
    """
    return df.iloc[primeiro:].reset_index(drop=True)


def camera_do_recorte(lado=FRAME_RECORTE_PX):
    """
    Bloco de câmera (width/height/h_fov/v_fov) coerente com o recorte vigente.

    fx NÃO depende do recorte — é propriedade da renderização do Google Earth
    (640 px de largura a GE_VIEWPORT_H_FOV_DEG). Recortar muda apenas quanto
    desse FOV sobra. Quem precisa do bloco de câmera deve chamar esta função
    em vez de repetir os números, para não dessincronizar de
    FRAME_RECORTE_PX (o config.yaml, que não executa Python, tem os valores
    escritos à mão com um comentário apontando para cá).
    """
    fx = FRAME_BRUTO_GE[1] / 2.0 / np.tan(np.deg2rad(GE_VIEWPORT_H_FOV_DEG) / 2.0)
    fov = float(2.0 * np.rad2deg(np.arctan((lado / 2.0) / fx)))
    return {"width": lado, "height": lado, "h_fov": fov, "v_fov": fov}


def recortar_frame_google_earth(img, lado=FRAME_RECORTE_PX):
    """
    Recorta o frame bruto do Google Earth num quadrado CENTRADO NO NADIR.

    Por que não basta cortar o rodapé (`img[:640, :]`, como era feito antes):
    o viewport gravado tem 700 linhas e o `<Camera>` do KML, com `tilt=0`,
    põe o nadir — o ponto que corresponde à coordenada do CSV de ground
    truth — no centro do viewport, isto é, na linha 350. Cortar as 60
    linhas de baixo mantém esse nadir na linha 350 de um frame de 640
    linhas, cujo centro é a 320: o pipeline inteiro (cx/cy do
    `recoverPose`, centro da homografia em `_estimar_posicao_pela_homografia`,
    centragem do patch em `_preparar_patch_satelital`) passa a assumir um
    ponto principal 30 px deslocado AO LONGO DA PROA. Medido: 67,7 m de erro
    a 1500 m de altitude e 174,8 m a 3500 m (previsto 68,7 e 177,0), somando
    com o lag de captura para explicar o offset de ~430 m que a campanha de
    map matching não conseguia mover.

    O recorte centrado resolve isso mantendo cx = largura/2 e cy = altura/2
    válidos, sem espalhar um ponto principal deslocado pelo código.

    O corte também precisa continuar excluindo as legendas sobrepostas (a
    razão original do `img[:640, :]`), que atrapalhariam as correspondências
    da odometria por serem features fixas na imagem, sem paralaxe. Ver
    LEGENDA_TOPO_Y para as duas faixas medidas. Com lado=576 o recorte vai
    até a linha 637, deixando 3 linhas de margem para a primeira legenda.

    Parâmetros
    ----------
    img : np.ndarray
        Frame bruto lido do PNG do Google Earth (700x640).
    lado : int
        Lado do recorte quadrado.

    Retorna
    -------
    np.ndarray
        Recorte `lado` x `lado` centrado no nadir.

    Levanta
    -------
    ValueError
        Se o frame não tiver as dimensões esperadas, ou se o recorte pedido
        invadir a faixa de legenda — falhar alto é proposital: um frame com
        dimensão diferente significa que a captura mudou e as constantes
        acima precisam ser remedidas antes de gerar dataset novo.
    """
    h, w = img.shape[:2]
    if (h, w) != FRAME_BRUTO_GE:
        raise ValueError(
            f"Frame {w}x{h} não bate com o formato esperado do Google Earth "
            f"{FRAME_BRUTO_GE[1]}x{FRAME_BRUTO_GE[0]}. As constantes de recorte "
            "(nadir e faixas de legenda) foram medidas para esse formato — "
            "remeça-as antes de usar uma captura diferente."
        )

    meio = lado // 2
    y0, x0 = h // 2 - meio, w // 2 - meio
    y1, x1 = y0 + lado, x0 + lado
    if y1 > LEGENDA_TOPO_Y:
        raise ValueError(
            f"Recorte de lado {lado} centrado no nadir chega até a linha {y1 - 1}, "
            f"invadindo a legenda que começa na linha {LEGENDA_TOPO_Y}. "
            f"Lado máximo possível: {2 * (LEGENDA_TOPO_Y - h // 2)}."
        )
    return img[y0:y1, x0:x1]


def gerar_tour_kml(df, file_path, altitude, mach):
    """
    Gera KML gx:Tour a partir de um DataFrame de waypoints.

    Colunas esperadas: Long, Lat, duration, altitude, heading, tilt, flight_mode.
    """
    pt = (
        "        <gx:FlyTo>\n"
        "            <gx:duration>{duration:.3f}</gx:duration>\n"
        "            <gx:flyToMode>{flight_mode}</gx:flyToMode>\n"
        "            <Camera>\n"
        "                <longitude>{longitude:.8f}</longitude>\n"
        "                <latitude>{latitude:.8f}</latitude>\n"
        "                <altitude>{altitude}</altitude>\n"
        "                <heading>{heading:.4f}</heading>\n"
        "                <tilt>{tilt}</tilt>\n"
        "                <roll>0</roll>\n"
        "                <altitudeMode>absolute</altitudeMode>\n"
        "            </Camera>\n"
        "        </gx:FlyTo>\n"
    )
    elements = ""
    for _, wp in df.iterrows():
        elements += pt.format(
            longitude=wp["Long"], latitude=wp["Lat"], duration=wp["duration"],
            altitude=wp["altitude"], heading=wp["heading"], tilt=wp["tilt"],
            flight_mode=wp["flight_mode"],
        )

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2" '
        'xmlns:gx="http://www.google.com/kml/ext/2.2">\n'
        "    <gx:Tour>\n"
        f"        <name>ROTA_QUADRADA_{altitude}m_Mach{mach}</name>\n"
        "        <gx:Playlist>\n"
        f"{elements}"
        "        </gx:Playlist>\n"
        "    </gx:Tour>\n"
        "</kml>"
    )
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"[KML Tour]  {file_path}  ({len(df)} FlyTos)")


def gerar_pontos_kml(df, file_path, alt, mach):
    """
    Gera KML com Placemarks individuais a partir de um DataFrame.

    Colunas esperadas: Long, Lat.
    """
    xml_template = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        "<Document>\n"
        f"    <name>Path_{alt}_{mach}.kml</name>\n"
        '    <Style id="s_ylw-pushpin"><IconStyle><scale>1.1</scale>'
        "<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>"
        "</Icon></IconStyle></Style>\n"
        "{point_elements}"
        "</Document>\n"
        "</kml>"
    )
    point_template = (
        "    <Placemark>\n"
        "        <name>Pto{i}</name>\n"
        "        <styleUrl>#s_ylw-pushpin</styleUrl>\n"
        "        <Point><coordinates>{lon},{lat},000</coordinates></Point>\n"
        "    </Placemark>\n"
    )
    points = "".join(
        point_template.format(i=i, lon=row["Long"], lat=row["Lat"])
        for i, row in df.iterrows()
    )
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(xml_template.format(point_elements=points))
    print(f"[KML Path]  {file_path}  ({len(df)} pontos)")


def interpolar_waypoints(waypoints, speed_ms, fps=1, altitude=1500):
    """
    Interpola uma lista de waypoints em amostras regulares de 1/fps segundos.

    Cada waypoint é um dict com: longitude, latitude, heading, duration,
    altitude, tilt, flight_mode.

    Retorna DataFrame com colunas: File, Lat, Long, Proa, Altura,
    duration, heading, altitude, tilt, flight_mode.
    """
    g    = Geod(ellps='clrk66')
    step = 1.0 / fps

    total_time = sum(wp["duration"] for wp in waypoints)
    # Exclusive de total_time: total_time quase nunca é múltiplo exato de
    # step (vem de distância/velocidade real), então um limite superior
    # inclusive (total_time + step) sempre gera uma amostra "fantasma" além
    # do fim real do voo -- o Google Earth nunca chega a capturar um frame
    # nesse instante (o tour já terminou), sobrando uma linha a mais no CSV
    # sem imagem correspondente.
    timestamps = np.arange(0.0, total_time, step)
    n          = len(timestamps)

    # O índice do arquivo é o SEGUNDO DE VOO: t=0 -> "-000000.png". A
    # convenção antiga começava em "-000001.png" para t=0, o que desalinhava
    # todo o resto da rota em +1 frame (ver GE_PRIMEIRO_FRAME_VALIDO).
    file_list = [nome_frame(i) for i in range(n)]
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


# Paleta de cores por velocidade Mach (formato KML: AABBGGRR)
# Gradiente frio → quente: Azul → Ciano → Verde → Verde-amarelo → Amarelo → Laranja → Vermelho
_MACH_KML_COLORS = {
    0.5: "ffff0000",   # Azul
    0.8: "ffffff00",   # Ciano
    1.0: "ff00ff00",   # Verde
    1.2: "ff00ff80",   # Verde-amarelo
    1.5: "ff00ffff",   # Amarelo
    1.8: "ff0080ff",   # Laranja
    2.0: "ff0000ff",   # Vermelho
}
_KML_COLOR_REAL      = "ff000000"   # Preto — trajetória real (GPS)
_KML_COLOR_DEFAULT   = "ff00ffff"   # Amarelo — fallback se mach desconhecido


def mach_to_kml_color(mach):
    """Retorna a cor KML (AABBGGRR) correspondente à velocidade Mach.

    Faz busca pela chave mais próxima para tolerar imprecisão de float.
    """
    if mach is None:
        return _KML_COLOR_DEFAULT
    closest = min(_MACH_KML_COLORS.keys(), key=lambda k: abs(k - mach))
    return _MACH_KML_COLORS[closest]


def criar_caminho_kml(lat_list, lon_list, file_path, color=None):
    """
    Cria um arquivo KML a partir de listas de coordenadas para visualização
    no Google Earth.

    Args:
        lat_list: lista de latitudes.
        lon_list: lista de longitudes.
        file_path: caminho de saída do arquivo .kml.
        color: cor da linha no formato KML AABBGGRR (str).
               Se None, usa o fallback amarelo.
    """
    if color is None:
        color = _KML_COLOR_DEFAULT

    kml_template = '''<?xml version="1.0" encoding="UTF-8"?>
    <kml xmlns="http://www.opengis.net/kml/2.2">
    <Document>
        <name>{route_name}</name>
        <Style id="trajLine"><LineStyle><color>{color}</color><width>4</width></LineStyle></Style>
        <Placemark>
            <name>Trajetoria</name>
            <styleUrl>#trajLine</styleUrl>
            <LineString><tessellate>1</tessellate><coordinates>{point_elements}</coordinates></LineString>
        </Placemark>
    </Document>
    </kml>'''

    point_elements = "".join(
        f'{lon},{lat},0 ' for lat, lon in zip(lat_list, lon_list)
    )
    xml_content = kml_template.format(
        route_name=os.path.basename(file_path),
        color=color,
        point_elements=point_elements.strip()
    )

    with open(file_path, 'w') as f:
        f.write(xml_content)
    print(f"Arquivo KML salvo em: {file_path}")
