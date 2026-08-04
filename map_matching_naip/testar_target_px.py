"""
Teste isolado: roda só o caso COM map matching (SIFT + SUPERPOINT abs,
CHAMPAIGN mach 1.0, map.vrt, roi_margin_factor 1.3, roi_center_mode
estimado) para medir o efeito de aumentar _MAP_MATCH_TARGET_PX
(map_matching.py) de 480 para 960 px.

Baseline conhecida (480px, mesmo cenario): sem MM 64.6m / com MM 221.7m.
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from rodar_teste_map_matching import montar_cfg, preparar_pasta_imagens
from main import montar_config
from odometria_visual import OdometriaVisual
import map_matching

BASE = Path(r"C:\Users\bbs_l\OneDrive\Leandro\ITA\MESTRADO\Tese\rotas_quadradas\CHAMPAIGN")
MAP_VRT = BASE / "map.vrt"
MACH = 1.0
DETECTOR = "SIFT"
ABS_DETECTOR = "SUPERPOINT"

print(f"[info] _MAP_MATCH_TARGET_PX = {map_matching._MAP_MATCH_TARGET_PX}")

frames_dir = preparar_pasta_imagens(BASE, MACH)
csv_name = f"Coord-Heading-Elev_1500_{MACH}.csv"
output_folder = f"{DETECTOR}_abs-{ABS_DETECTOR}_estimado_com_mapmatch_target{map_matching._MAP_MATCH_TARGET_PX}"

cfg_raw = montar_cfg(
    MACH, DETECTOR, True, frames_dir, csv_name, output_folder,
    BASE, MAP_VRT, ABS_DETECTOR, roi_margin_factor=1.3, interval=1,
    roi_center_mode="estimado",
)
config = montar_config(cfg_raw)

t0 = time.time()
OdometriaVisual(config).executar()
tempo_s = time.time() - t0

csv_path = BASE / f"mach_{MACH}" / "results" / output_folder / f"resultados_{DETECTOR}.csv"
df = pd.read_csv(csv_path)
erro_final = float(df["ERRO_ACUM(m)"].iloc[-1]) if len(df) else None

print(f"\n{'='*60}")
print(f"TARGET_PX={map_matching._MAP_MATCH_TARGET_PX}  ERRO_ACUM(com MM) = {erro_final:.1f} m  "
      f"(baseline 480px: 221.7m)  tempo={tempo_s:.1f}s")
print(f"{'='*60}")
