import argparse
import random
import yaml
import copy
import pandas as pd
import os
import sys
from pathlib import Path
import traceback

# Adiciona o diretório atual ao path para poder importar módulos locais
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import carregar_yaml, montar_config
from odometria_visual import OdometriaVisual

def carregar_config_base(config_path):
    cfg = carregar_yaml(config_path)
    # Desabilita visualizações para acelerar execução
    if 'display' not in cfg:
        cfg['display'] = {}
    cfg['display']['show_plot'] = False
    cfg['display']['show_images'] = False

    # MUITO IMPORTANTE: precisamos que o base_path seja "." (raiz atual)
    # para que nossos caminhos relativos ao `image_calibration` funcionem corretamente,
    # caso contrário o script `main.py` vai tentar prepender o "C:/Users/...".
    if 'paths' in cfg:
        cfg['paths']['base_path'] = str(Path('.').resolve())

    return cfg

def get_random_params_orb():
    return {
        'nfeatures': random.choice([500, 1000, 1500, 2000, 3000, 4000]),
        'scaleFactor': round(random.uniform(1.1, 1.4), 2),
        'nlevels': random.choice([4, 6, 8, 10]),
        'edgeThreshold': random.choice([15, 31, 45]),
        'firstLevel': 0,
        'WTA_K': random.choice([2, 3, 4]),
        'scoreType': random.choice(['HARRIS', 'FAST']),
        'patchSize': random.choice([31, 45, 61]),
    }

def get_random_params_akaze():
    return {
        'descriptor_type': random.choice(['MLDB', 'MLDB_UPRIGHT']),
        'descriptor_size': 0,
        'descriptor_channels': 3,
        'threshold': random.choice([0.0001, 0.0005, 0.001, 0.003]),
        'nOctaves': random.choice([3, 4, 5]),
        'nOctaveLayers': random.choice([3, 4, 5]),
        'diffusivity': random.choice(['PM_G1', 'PM_G2', 'WEICKERT', 'CHARBONNIER']),
    }

def run_evaluation(base_cfg, detector, num_iterations=20):
    best_params = None
    best_avg_inliers = -1
    best_iter = -1

    results_log = []

    for i in range(num_iterations):
        print(f"\n[{i+1}/{num_iterations}] Otimizando {detector}...")

        # Cria uma cópia da configuração
        cfg = copy.deepcopy(base_cfg)
        cfg['detector'] = detector

        if 'detector_params' not in cfg:
            cfg['detector_params'] = {}

        if detector == 'ORB':
            params = get_random_params_orb()
            cfg['detector_params']['orb'] = params
        elif detector == 'AKAZE':
            params = get_random_params_akaze()
            cfg['detector_params']['akaze'] = params
        else:
            print(f"Detector {detector} não suportado para otimização simples.")
            return

        print(f"Parâmetros sorteados: {params}")

        try:
            # Monta a config para a classe OdometriaVisual
            cfg_mounted = montar_config(cfg)

            # Instancia o pipeline
            pipeline = OdometriaVisual(cfg_mounted)

            # Executa o pipeline
            pipeline.executar()

            # Lê o CSV gerado
            output_dir = cfg_mounted['paths']['output_dir']
            res_csv = os.path.join(output_dir, f"resultados_{detector}.csv")

            if os.path.exists(res_csv):
                df = pd.read_csv(res_csv)
                if 'INLIERS' in df.columns and len(df) > 0:
                    avg_inliers = df['INLIERS'].mean()
                    print(f"-> Média de inliers: {avg_inliers:.2f}")

                    results_log.append({
                        'iteration': i + 1,
                        'params': params,
                        'avg_inliers': avg_inliers
                    })

                    if avg_inliers > best_avg_inliers:
                        best_avg_inliers = avg_inliers
                        best_params = params
                        best_iter = i + 1
                        print("  >>> Novo melhor encontrado!")
                else:
                    print("-> Nenhuma coluna INLIERS ou CSV vazio.")
            else:
                print("-> Arquivo de resultados não encontrado.")

        except Exception as e:
            print(f"Erro na iteração {i+1}: {e}")
            traceback.print_exc()

    print("\n" + "="*50)
    print("RESUMO DA OTIMIZAÇÃO")
    print("="*50)
    if best_params:
        print(f"Melhor iteração: {best_iter}")
        print(f"Média de Inliers: {best_avg_inliers:.2f}")
        print("Melhores parâmetros:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
    else:
        print("Nenhuma iteração concluída com sucesso.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Otimização de hiperparâmetros (ORB/AKAZE)")
    parser.add_argument('--config', type=str, default='config.yaml', help='Caminho para o config.yaml base')
    parser.add_argument('--detector', type=str, choices=['ORB', 'AKAZE'], default='ORB', help='Detector a ser otimizado')
    parser.add_argument('--iter', type=int, default=20, help='Número de iterações do Random Search')
    parser.add_argument('--calib_path', type=str, default='image_calibration', help='Caminho relativo para calibração (imagens e GT)')
    parser.add_argument('--gt_file', type=str, default='ground_truth.csv', help='Nome do arquivo de ground truth dentro da pasta de calibração')

    args = parser.parse_args()

    base_cfg = carregar_config_base(args.config)

    # Atualiza as rotas para o diretório de calibração
    if 'paths' in base_cfg:
        base_cfg['paths']['images'] = args.calib_path
        base_cfg['paths']['ground_truth'] = os.path.join(args.calib_path, args.gt_file)
        base_cfg['paths']['output'] = os.path.join(args.calib_path, "output")

    print(f"Iniciando otimização para {args.detector} com {args.iter} iterações...")
    print(f"Lendo imagens de: {base_cfg['paths']['images']}")
    print(f"Lendo GT de: {base_cfg['paths']['ground_truth']}")

    run_evaluation(base_cfg, args.detector, args.iter)
