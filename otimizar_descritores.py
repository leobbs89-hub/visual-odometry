"""
Sweep de otimização de parâmetros dos descritores (estratégia OFAT).

Para cada rota × descritor × parâmetro × valor, executa o pipeline de odometria
(reusando montar_config de main.py e OdometriaVisual de odometria_visual.py),
mantendo todos os demais parâmetros nos defaults do config.yaml, e mede o número
de INLIERS (coluna do resultados_<DETECTOR>.csv).

Estratégia OFAT (One-Factor-At-A-Time): varia um parâmetro por vez para revelar
qual parâmetro vale a pena ajustar em cada cenário de terreno.

Saídas em OTIM_BASE:
    resumo_inliers.csv        — tabela consolidada
    relatorio_otimizacao.md   — relatório por rota/descritor + vencedor por rota
    resultados/<rota>/<descritor>/<param>=<valor>/resultados_<DET>.csv

Pré-requisitos:
    - gerar_rotas_experimento.py já rodado (CSV de ground truth por rota)
    - imagens capturadas no Google Earth e resize_img.py já rodado (Resized/)
"""

import os
import sys
import copy
import glob
import traceback

import matplotlib
matplotlib.use("Agg")  # evita backend interativo / plt.show bloqueante

import pandas as pd
import yaml

# MatchFormer fica no repo principal, não no worktree. Injetar antes do import
# de odometria_visual para que o `from model.matchformer import Matchformer`
# dentro de odometria_visual.py encontre o pacote.
_REPO_ROOT = r"C:\Users\bbs_l\OneDrive\Leandro\ITA\MESTRADO\CODIGOS\visual-odometry"
_mf_path = os.path.join(_REPO_ROOT, "MatchFormer")
if _mf_path not in sys.path:
    sys.path.insert(0, _mf_path)

from main import montar_config
from odometria_visual import OdometriaVisual

# ==========================================
# CAMINHOS
# ==========================================
REPO_DIR    = os.path.dirname(os.path.abspath(__file__))
CONFIG_BASE = os.path.join(REPO_DIR, "config.yaml")
OTIM_BASE   = r"C:\Users\bbs_l\OneDrive\Leandro\ITA\MESTRADO\Tese\Imagens\otimizacao_descritores"

ROTAS       = ["FLORESTA", "URBANO"]
DESCRITORES = ["ORB", "AKAZE", "SIFT", "SUPERPOINT", "LOFTR", "MATCHFORMER"]

# Se definido, roda apenas os descritores listados aqui (None = todos).
# Ao fundir resultados, os descritores ausentes são lidos do resumo_inliers.csv.
DESCRITORES_FILTRO = ["MATCHFORMER"]  # ex: ["SUPERPOINT", "MATCHFORMER"]


# ==========================================
# DEFINIÇÃO DOS SWEEPS (OFAT)
# Cada entrada: (caminho_no_cfg, lista_de_valores)
# caminho_no_cfg é uma tupla de chaves aninhadas a partir do dict do YAML.
# ==========================================
SWEEPS = {
    "ORB": [
        (("detector_params", "orb", "nfeatures"),     [500, 1000, 1500, 2000, 3000]),
        (("detector_params", "orb", "scaleFactor"),   [1.1, 1.2, 1.3]),
        (("detector_params", "orb", "nlevels"),       [4, 8, 12]),
        (("detector_params", "orb", "edgeThreshold"), [15, 31, 51]),
        (("matcher", "nn_match_ratio"),               [0.7, 0.8, 0.9]),
    ],
    "AKAZE": [
        (("detector_params", "akaze", "threshold"),           [0.0001, 0.0005, 0.001, 0.003]),
        (("detector_params", "akaze", "nOctaves"),            [2, 4, 6]),
        (("detector_params", "akaze", "nOctaveLayers"),       [2, 4, 6]),
        (("detector_params", "akaze", "descriptor_channels"), [1, 2, 3]),
        (("detector_params", "akaze", "diffusivity"),         ["PM_G1", "PM_G2", "WEICKERT", "CHARBONNIER"]),
    ],
    "SIFT": [
        (("detector_params", "sift", "nfeatures"),         [500, 1000, 1500, 2000, 3000]),
        (("detector_params", "sift", "contrastThreshold"), [0.01, 0.02, 0.04, 0.06]),
        (("detector_params", "sift", "edgeThreshold"),     [5, 10, 15, 20]),
        (("detector_params", "sift", "sigma"),             [1.2, 1.6, 2.0]),
        (("detector_params", "sift", "nOctaveLayers"),     [3, 4, 5]),
        (("matcher", "nn_match_ratio"),                    [0.7, 0.8, 0.9]),
    ],
    "SUPERPOINT": [
        (("detector_params", "superpoint", "max_num_keypoints"),  [512, 1024, 2048, 4096]),
        (("detector_params", "superpoint", "detection_threshold"), [0.0005, 0.005, 0.015]),
        (("detector_params", "superpoint", "nms_radius"),          [2, 4, 8]),
    ],
    "LOFTR": [
        (("detector_params", "loftr", "pretrained"), ["outdoor", "indoor"]),
    ],
    "MATCHFORMER": [
        # backbone_type não é varrido: só há checkpoint baixado para "largela"
        # (outdoor-large-LA.ckpt); os outros backbones exigiriam pesos próprios
        # e, sem eles, o load_state_dict cairia de volta em pesos aleatórios.
        (("detector_params", "matchformer", "match_coarse", "thr"),
         [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]),
    ],
}


# ==========================================
# HELPERS
# ==========================================

def set_nested(cfg, caminho, valor):
    """Define cfg[k1][k2]...[kn] = valor, criando dicts intermediários se preciso."""
    d = cfg
    for k in caminho[:-1]:
        d = d.setdefault(k, {})
    d[caminho[-1]] = valor


def encontrar_csv_gt(rota_dir):
    """Localiza o CSV de ground truth (Coord-Heading-Elev_reta_*.csv) da rota."""
    matches = glob.glob(os.path.join(rota_dir, "Coord-Heading-Elev_reta_*.csv"))
    return matches[0] if matches else None


def sanitizar(valor):
    """Converte um valor de parâmetro em fragmento de nome de pasta seguro."""
    return str(valor).replace(".", "p").replace(" ", "")


def construir_cfg(base_cfg, rota_dir, descritor, output_dir):
    """Monta o dict de configuração (formato YAML) para uma execução do sweep."""
    cfg = copy.deepcopy(base_cfg)
    cfg["detector"] = descritor
    cfg["device"]   = "auto"

    cfg.setdefault("map_matching", {})["enabled"] = False

    # Sem janelas / sem plot bloqueante / sem ruído no console
    cfg["display"] = {
        "show_plot":          False,
        "show_images":        False,
        "print_console":      False,
        "show_map_matching":  False,
        "debug_map_matching": False,
    }

    # Caminhos absolutos (base_path "." + caminho absoluto = caminho absoluto)
    cfg["paths"] = {
        "base_path":    ".",
        "images":       os.path.join(rota_dir, "Resized"),
        "ground_truth": encontrar_csv_gt(rota_dir),
        "output":       output_dir,
        "map_tif":      base_cfg["paths"].get("map_tif", "data/map/map.tif"),
    }
    return cfg


def ler_metricas(output_dir, descritor):
    """Lê resultados_<DET>.csv e retorna médias de INLIERS/MATCHES/KPT1/KPT2."""
    csv_path = os.path.join(output_dir, f"resultados_{descritor}.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty or "INLIERS" not in df.columns:
        return None
    return {
        "inliers": float(df["INLIERS"].mean()),
        "matches": float(df["MATCHES"].mean()),
        "kpt1":    float(df["KPT1"].mean()),
        "kpt2":    float(df["KPT2"].mean()),
    }


# ==========================================
# EXECUÇÃO DO SWEEP
# ==========================================

def rodar_sweep():
    with open(CONFIG_BASE, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    registros = []
    descritores_rodar = DESCRITORES_FILTRO if DESCRITORES_FILTRO else DESCRITORES

    for rota in ROTAS:
        rota_dir = os.path.join(OTIM_BASE, rota, "rota_reta")
        gt_csv   = encontrar_csv_gt(rota_dir)
        resized  = os.path.join(rota_dir, "Resized")

        if gt_csv is None or not os.path.isdir(resized) or not os.listdir(resized):
            print(f"[AVISO] Rota {rota}: faltam Resized/ ou CSV de ground truth — pulando.")
            continue

        for descritor in descritores_rodar:
            print(f"\n========== {rota} / {descritor} ==========")
            pulado = False

            for caminho, valores in SWEEPS[descritor]:
                if pulado:
                    break
                param_nome = ".".join(caminho)

                for valor in valores:
                    out_dir = os.path.join(
                        OTIM_BASE, "resultados", rota, descritor,
                        f"{caminho[-1]}={sanitizar(valor)}",
                    )
                    os.makedirs(out_dir, exist_ok=True)

                    cfg = construir_cfg(base_cfg, rota_dir, descritor, out_dir)
                    set_nested(cfg, caminho, valor)

                    try:
                        config = montar_config(cfg)
                        OdometriaVisual(config).executar()
                    except ImportError as e:
                        print(f"  [PULADO] {descritor} indisponível: {e}")
                        registros.append({
                            "rota": rota, "descritor": descritor,
                            "parametro": param_nome, "valor": valor,
                            "inliers": None, "matches": None,
                            "kpt1": None, "kpt2": None, "status": "pulado",
                        })
                        pulado = True
                        break
                    except Exception as e:
                        print(f"  [ERRO] {param_nome}={valor}: {e}")
                        traceback.print_exc()
                        registros.append({
                            "rota": rota, "descritor": descritor,
                            "parametro": param_nome, "valor": valor,
                            "inliers": None, "matches": None,
                            "kpt1": None, "kpt2": None, "status": "erro",
                        })
                        continue

                    metr = ler_metricas(out_dir, descritor)
                    if metr is None:
                        print(f"  [SEM RESULTADO] {param_nome}={valor}")
                        registros.append({
                            "rota": rota, "descritor": descritor,
                            "parametro": param_nome, "valor": valor,
                            "inliers": None, "matches": None,
                            "kpt1": None, "kpt2": None, "status": "sem_resultado",
                        })
                        continue

                    print(f"  {param_nome}={valor}: inliers={metr['inliers']:.0f} "
                          f"matches={metr['matches']:.0f}")
                    registros.append({
                        "rota": rota, "descritor": descritor,
                        "parametro": param_nome, "valor": valor,
                        "inliers": metr["inliers"], "matches": metr["matches"],
                        "kpt1": metr["kpt1"], "kpt2": metr["kpt2"],
                        "status": "ok",
                    })

    return pd.DataFrame(registros)


# ==========================================
# RELATÓRIO
# ==========================================

def gerar_relatorio(df):
    resumo_csv = os.path.join(OTIM_BASE, "resumo_inliers.csv")
    df.to_csv(resumo_csv, index=False)
    print(f"\n[CSV] {resumo_csv}")

    linhas = ["# Relatório de Otimização de Descritores", ""]
    ok = df[df["status"] == "ok"].copy()

    for rota in ROTAS:
        linhas.append(f"## Rota: {rota}")
        linhas.append("")
        rota_ok = ok[ok["rota"] == rota]

        for descritor in DESCRITORES:
            linhas.append(f"### {descritor}")
            sub = df[(df["rota"] == rota) & (df["descritor"] == descritor)]
            sub_ok = rota_ok[rota_ok["descritor"] == descritor]

            if sub.empty:
                linhas.append("_Sem execuções._\n")
                continue
            if sub_ok.empty:
                status = sub["status"].iloc[0]
                linhas.append(f"_Não avaliado (status: {status})._\n")
                continue

            linhas.append("| Parâmetro | Valor | Inliers | Matches |")
            linhas.append("|---|---|---|---|")
            for _, r in sub_ok.iterrows():
                linhas.append(
                    f"| {r['parametro']} | {r['valor']} | "
                    f"{r['inliers']:.0f} | {r['matches']:.0f} |"
                )
            linhas.append("")

            linhas.append("**Melhor valor por parâmetro:**")
            for param, g in sub_ok.groupby("parametro"):
                best = g.loc[g["inliers"].idxmax()]
                linhas.append(f"- `{param}` → **{best['valor']}** "
                              f"(inliers={best['inliers']:.0f})")
            linhas.append("")

        # Vencedor da rota
        if not rota_ok.empty:
            best = rota_ok.loc[rota_ok["inliers"].idxmax()]
            linhas.append(f"### 🏆 Vencedor da rota {rota}")
            linhas.append(
                f"- Descritor **{best['descritor']}**, parâmetro "
                f"`{best['parametro']}` = **{best['valor']}** → "
                f"**{best['inliers']:.0f} inliers** (matches={best['matches']:.0f})"
            )
            linhas.append("")

    relatorio_md = os.path.join(OTIM_BASE, "relatorio_otimizacao.md")
    with open(relatorio_md, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))
    print(f"[MD]  {relatorio_md}")


def main():
    df_novo = rodar_sweep()

    # Mescla com resultados anteriores, se houver e filtro ativo
    resumo_csv = os.path.join(OTIM_BASE, "resumo_inliers.csv")
    if DESCRITORES_FILTRO and os.path.exists(resumo_csv):
        df_ant = pd.read_csv(resumo_csv)
        # Remove entradas dos descritores que acabamos de (re)rodar
        mask = df_ant["descritor"].isin(DESCRITORES_FILTRO)
        df_ant = df_ant[~mask]
        df = pd.concat([df_ant, df_novo], ignore_index=True)
        print(f"\n[INFO] Mesclando {len(df_ant)} registros anteriores + {len(df_novo)} novos.")
    else:
        df = df_novo

    if df.empty:
        print("\nNenhum registro gerado. Verifique se as rotas e imagens existem.")
        return
    gerar_relatorio(df)
    print("\nSweep concluído.")


if __name__ == "__main__":
    main()
