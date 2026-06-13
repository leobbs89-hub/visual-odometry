# utils.py
"""Funções utilitárias puras (sem estado) usadas pelo pipeline de odometria."""

import os


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
