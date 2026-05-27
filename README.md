# 🛩️ Odometria Visual Monocular

Pipeline em Python para estimativa de trajetória de câmera a partir de sequências de imagens aéreas (drone/UAV), com comparação contra dados GPS de referência.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU%20%7C%20GPU-orange?logo=pytorch)
![Testes](https://img.shields.io/badge/Testes-36%20passando-brightgreen)
![Licença](https://img.shields.io/badge/Licença-MIT-lightgrey)

---

## Detectores suportados

| Detector | Tipo | Dependência |
|---|---|---|
| **ORB** | Clássico | OpenCV (incluso) |
| **AKAZE** | Clássico | OpenCV (incluso) |
| **SuperPoint + LightGlue** | Rede neural | torch + [LightGlue](https://github.com/cvg/LightGlue) |
| **LoFTR** | Rede neural | torch + kornia |
| **MatchFormer** | Rede neural | torch + [MatchFormer](https://github.com/gaopengcuhk/MatchFormer) |

> Os detectores neurais **funcionam em CPU** — GPU acelera mas não é obrigatória.

---

## Estrutura do projeto

```
├── main.py                  # Ponto de entrada + parsing do config.yaml
├── odometria_visual.py      # Classe OdometriaVisual — todo o pipeline
├── config.yaml              # Parâmetros editáveis (caminhos, câmera, detector)
├── requirements.txt         # Dependências base (ORB e AKAZE)
├── requirements-neural.txt  # Dependências extras para redes neurais
├── install.sh               # Instalação automática Linux/macOS
├── install.bat              # Instalação automática Windows
├── test_pipeline.py         # 36 testes unitários
└── CLAUDE.md                # Contexto técnico para uso com AI assistants
```

---

## Instalação

### Linux / macOS

```bash
# Apenas ORB e AKAZE
bash install.sh

# Com detectores neurais (CPU ou GPU automático)
bash install.sh --neural
```

### Windows

```bat
install.bat
install.bat --neural
```

O script cria automaticamente um `.venv`, instala as dependências e clona os repositórios necessários do GitHub.

### Instalação manual

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate.bat

pip install -r requirements.txt

# Opcional — para detectores neurais
pip install -r requirements-neural.txt
git clone https://github.com/cvg/LightGlue.git && pip install -e LightGlue/
git clone https://github.com/gaopengcuhk/MatchFormer.git
```

---

## Configuração

Edite o arquivo **`config.yaml`**:

```yaml
paths:
  base_path: "C:/Users/usuario/dados"   # use / mesmo no Windows
  images:       "sequencia/Resized"
  ground_truth: "Codigos/ground_truth.csv"
  output:       "Resultados"
  map_tif:      "Codigos/mapa.tif"

camera:
  width: 640
  height: 640
  h_fov: 71.56   # graus
  v_fov: 71.56

detector: ORB   # ORB | AKAZE | SUPERPOINT | LOFTR | MATCHFORMER
device: auto    # auto | cpu | cuda
```

---

## Uso

```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate.bat

python main.py                                   # config padrão
python main.py --detector AKAZE                  # troca detector sem editar YAML
python main.py --detector SUPERPOINT             # rede neural em CPU
python main.py --detector SUPERPOINT --device cpu
python main.py --config experimentos/config_loftr.yaml
```

---

## Formato do CSV de ground truth

| Coluna | Descrição |
|---|---|
| `Lat` | Latitude em graus decimais |
| `Long` | Longitude em graus decimais |
| `Altura` | Altitude AGL em metros (usada para calcular GSD) |
| `Proa` | Heading/yaw em graus (0–360) |

---

## Saídas

Salvas em `paths.output`:

| Arquivo | Conteúdo |
|---|---|
| `resultados_<DETECTOR>.csv` | Métricas por frame: keypoints, matches, erro acumulado, tempo |
| `trajetoria_real.kml` | Ground truth para Google Earth |
| `trajetoria_estimada_<DETECTOR>.kml` | Trajetória estimada para Google Earth |

---

## Testes

```bash
python -m pytest test_pipeline.py -v
# 36 testes — YAML, config, OpenCV, CLI, device, ORB, AKAZE, SuperPoint (mock)
```

---

## Dependências

### `requirements.txt`

| Pacote | Uso |
|---|---|
| `numpy` | Álgebra linear |
| `opencv-contrib-python` | Detecção de features e estimação de pose |
| `matplotlib` | Visualização da trajetória |
| `scipy` | Conversão de matrizes de rotação |
| `pandas` | Leitura do CSV de ground truth |
| `pyproj` | Projeções geodésicas e UTM |
| `geopy` | Distâncias geodésicas |
| `rasterio` | Leitura lazy de GeoTIFF |
| `pyyaml` | Leitura do `config.yaml` |

### `requirements-neural.txt`

| Pacote | Uso |
|---|---|
| `torch` + `torchvision` | Inferência para SuperPoint, LoFTR e MatchFormer |
| `kornia` | Implementação do LoFTR |

Para usar GPU NVIDIA, substitua a linha do `torch` pela versão compatível com seu CUDA — veja [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Desenvolvido no contexto

Pesquisa em navegação autônoma de VANTs — Instituto Tecnológico de Aeronáutica (ITA).
