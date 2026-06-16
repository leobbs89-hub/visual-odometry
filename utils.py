# utils.py
"""Funções utilitárias puras (sem estado) usadas pelo pipeline de odometria."""

import os


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
