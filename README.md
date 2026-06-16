# 🛩️ Odometria Visual Monocular

Pipeline em Python para estimativa de trajetória de câmera a partir de sequências de imagens aéreas (drone/UAV), com comparação contra dados GPS de referência.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU%20%7C%20GPU-orange?logo=pytorch)
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
├── map_matching.py          # MapMatchingMixin — localização absoluta via GeoTIFF
├── utils.py                 # Funções puras: KML, geração de rota
├── criar_rota_quadrado.py   # Geração de rota quadrada simulada + CSV/KML
├── config.yaml              # Parâmetros editáveis (caminhos, câmera, detector)
├── requirements.txt         # Dependências base (ORB e AKAZE)
├── requirements-neural.txt  # Dependências extras para redes neurais
├── install.sh               # Instalação automática Linux/macOS
└── install.bat              # Instalação automática Windows
```

---

## Instalação

### Linux / macOS

```bash
# 1. Clonar o repositório
git clone https://github.com/leobbs89-hub/visual-odometry.git
cd visual-odometry

# 2. Instalar — apenas ORB e AKAZE
bash install.sh

# 2. Ou com detectores neurais (CPU ou GPU automático)
bash install.sh --neural
```

### Windows

```bat
REM 1. Clonar o repositório
git clone https://github.com/leobbs89-hub/visual-odometry.git
cd visual-odometry

REM 2. Instalar — apenas ORB e AKAZE
install.bat

REM 2. Ou com detectores neurais
install.bat --neural
```

O script cria automaticamente um `.venv`, instala as dependências e clona os repositórios necessários do GitHub (LightGlue e MatchFormer).

### Instalação manual

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate.bat

pip install -r requirements.txt

# Opcional — para detectores neurais
pip install -r requirements-neural.txt
git clone https://github.com/cvg/LightGlue.git && pip install -e LightGlue/
git clone https://github.com/InSAI-Lab/MatchFormer.git
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

## Geração de rota simulada

O script `criar_rota_quadrado.py` gera uma sequência de imagens sintética para testes,
sem necessidade de voo real.

```powershell
.venv\Scripts\activate.bat
python criar_rota_quadrado.py
```

Saídas geradas em `OUTPUT_DIR` (configurável no topo do script):

| Arquivo | Conteúdo |
|---|---|
| `Coord-Heading-Elev_<ALT>_<MACH>.csv` | Ground truth simulado (Lat, Long, Proa, Altura) |
| `KML_tour_<ALT>_<MACH>.kml` | Tour animado para Google Earth (`gx:Tour`) |
| `KML_path_<ALT>_<MACH>.kml` | Pontos da rota como Placemarks |
| `Resized/` | Imagens cortadas para a largura do sensor (`SENSOR_PX`) |

Parâmetros principais no topo do arquivo:

| Variável | Descrição | Padrão |
|---|---|---|
| `ALTITUDE` | Altitude AGL em metros | 1500 |
| `MACH` | Número de Mach | 2.0 |
| `FPS` | Amostras por segundo no CSV | 1 |
| `SQUARE_SIDE_KM` | Lado do quadrado em km | 10 |
| `SQUARE_INITIAL_HEADING` | Proa inicial em graus (0=Norte) | 0 |
| `SQUARE_TURN_DIRECTION` | +1 = curvas à direita, -1 = esquerda | 1 |

A velocidade do som é calculada automaticamente pela biblioteca `ambiance` (modelo ICAO)
para a altitude configurada.

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
| `ambiance` | Modelo de atmosfera ICAO (velocidade do som por altitude) |
| `requests` | Consulta à API Open-Elevation para elevação do terreno |

### `requirements-neural.txt`

| Pacote | Uso |
|---|---|
| `torch` + `torchvision` | Inferência para SuperPoint, LoFTR e MatchFormer |
| `kornia` | Implementação do LoFTR |

Para usar GPU NVIDIA, substitua a linha do `torch` pela versão compatível com seu CUDA — veja [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Desenvolvido no contexto

Pesquisa em navegação autônoma de VANTs — Instituto Tecnológico de Aeronáutica (ITA).
