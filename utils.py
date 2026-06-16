# utils.py
"""Funções utilitárias puras (sem estado) usadas pelo pipeline de odometria."""

import os
import numpy as np
import pandas as pd
from pyproj import Geod


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


def criar_caminho_kml(lat_list, lon_list, file_path):
    """
    Cria um arquivo KML a partir de listas de coordenadas para visualização
    no Google Earth.
    """
    kml_template = '''<?xml version="1.0" encoding="UTF-8"?>
    <kml xmlns="http://www.opengis.net/kml/2.2">
    <Document>
        <name>{route_name}</name>
        <Style id="yellowLine"><LineStyle><color>7f00ffff</color><width>4</width></LineStyle></Style>
        <Placemark>
            <name>Trajetoria</name>
            <styleUrl>#yellowLine</styleUrl>
            <LineString><tessellate>1</tessellate><coordinates>{point_elements}</coordinates></LineString>
        </Placemark>
    </Document>
    </kml>'''

    point_elements = "".join(
        f'{lon},{lat},0 ' for lat, lon in zip(lat_list, lon_list)
    )
    xml_content = kml_template.format(
        route_name=os.path.basename(file_path),
        point_elements=point_elements.strip()
    )

    with open(file_path, 'w') as f:
        f.write(xml_content)
    print(f"Arquivo KML salvo em: {file_path}")
