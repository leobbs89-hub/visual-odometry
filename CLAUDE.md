# CLAUDE.md — Contexto do Projeto: Odometria Visual Monocular

Pipeline em Python para estimativa de trajetória de câmera a partir de sequências de imagens aéreas (drone/UAV), comparada com GPS de referência.

---

## Arquivos e responsabilidades

| Arquivo | Papel |
|---|---|
| `main.py` | Ponto de entrada. Lê `config.yaml`, converte strings para constantes OpenCV, monta o dict `config` e passa para `OdometriaVisual`. |
| `odometria_visual.py` | Classe `OdometriaVisual`. Todo o pipeline: carregamento, detecção, matching, pose, GSD, KML, plot. |
| `config.yaml` | Único lugar para editar parâmetros. Caminhos relativos a `base_path`. |
| `requirements.txt` | Dependências base (funciona sem GPU). |
| `requirements-neural.txt` | Adiciona `torch`, `torchvision`, `kornia` para SUPERPOINT/LOFTR/MATCHFORMER. GPU **não é obrigatória**. |
| `install.sh` / `install.bat` | Criam `.venv`, instalam pip, clonam repos GitHub. Flag `--neural` ativa requirements-neural.txt. |
| `test_pipeline.py` | 36 testes unitários com `unittest`. Rodar: `python -m pytest test_pipeline.py -v`. |
| `LightGlue/` | Repo clonado: `github.com/cvg/LightGlue`. Fornece `SuperPoint` + `LightGlue`. |
| `MatchFormer/` | Repo clonado: `github.com/gaopengcuhk/MatchFormer`. |

---

## Fluxo de dados

```
config.yaml
    └─► main.py::montar_config()          # valida, converte constantes OpenCV, calcula fx/fy
            └─► OdometriaVisual(config)
                    ├─► _resolver_device()     # auto|cpu|cuda → torch.device
                    ├─► _inicializar_detector_matcher()
                    └─► executar()
                            ├─► _carregar_dados()          # imagens + CSV GPS + GSD + UTM
                            ├─► loop frame-a-frame
                            │       ├─► _obter_correspondencias()   # pts1, pts2
                            │       ├─► cv.findEssentialMat + recoverPose
                            │       ├─► _calcula_deslocamento_escala()  # metros via GSD
                            │       ├─► Rotation.from_matrix → yaw_delta
                            │       ├─► estima_latlon()
                            │       └─► _corrigir_posicao_pelo_mapa()  # opcional
                            └─► salva CSV + dois KML
```

---

## Detectores disponíveis

| Detector | Lib | Matcher | Precisa de torch? |
|---|---|---|---|
| `ORB` | OpenCV | BruteForce-Hamming | Não |
| `AKAZE` | OpenCV | BruteForce-Hamming | Não |
| `SUPERPOINT` | LightGlue/ | LightGlue | Sim |
| `LOFTR` | kornia | — (detector+matcher unificado) | Sim |
| `MATCHFORMER` | MatchFormer/ | — (detector+matcher unificado) | Sim |

Torch com CPU funciona para todos. Seleção de device: `config.yaml → device: auto|cpu|cuda` ou `--device cpu` na CLI.

---

## Dict `config` (formato interno após `montar_config`)

```python
{
    "detector_type":      str,            # 'ORB' | 'AKAZE' | 'SUPERPOINT' | 'LOFTR' | 'MATCHFORMER'
    "device":             str,            # 'auto' | 'cpu' | 'cuda'
    "paths": {
        "image_path":        str,         # diretório com imagens .jpg/.png (lidas em ordem alfabética)
        "ground_truth_path": str,         # CSV com colunas: Lat, Long, Altura, Proa
        "output_dir":        str,
        "map_tif_path":      str,         # GeoTIFF para map matching (opcional)
    },
    "camera_params": {
        "fx": float, "fy": float,        # calculados de width/h_fov no main.py
        "cx": float, "cy": float,
    },
    "matcher_params":     {"nn_match_ratio": float},   # Lowe ratio — só ORB/AKAZE
    "detector_params": {
        "orb":        dict,              # kwargs direto para cv.ORB_create()
        "akaze":      dict,              # kwargs direto para cv.AKAZE_create()
        "superpoint": dict,              # kwargs direto para SuperPoint()
        "loftr":      dict,              # kwargs direto para LoFTR()
        "matchformer": dict,
    },
    "display":            {"show_plot": bool, "show_images": bool},
    "use_map_matching":   bool,
    "map_match_interval": int,
    "roi_size_m":         int,
}
```

---

## Conversões feitas em `main.py`

Strings do YAML → constantes numéricas do OpenCV. Mapeamentos:

```python
scoreType:        "HARRIS" → cv.ORB_HARRIS_SCORE  |  "FAST" → cv.ORB_FAST_SCORE
descriptor_type:  "MLDB"   → cv.AKAZE_DESCRIPTOR_MLDB  |  "MLDB_UPRIGHT" → ...
diffusivity:      "PM_G1/G2" | "WEICKERT" | "CHARBONNIER" → cv.KAZE_DIFF_*
```

Constante desconhecida → default silencioso (HARRIS / MLDB / PM_G2).

---

## CSV de ground truth

Colunas obrigatórias: `Lat`, `Long`, `Altura`, `Proa`

- `Altura` (metros AGL) é usada para calcular GSD: `gsd = altura / fx`
- `Proa` (0–360°) é o yaw inicial acumulado
- Separador: vírgula

---

## Saídas geradas em `output_dir`

- `resultados_<DETECTOR>.csv` — métricas por frame: IMG, KPT1, KPT2, MATCHES, INLIERS, DIST_REAL, DIST_EST, ERRO_ACUM(m), TEMPO(s)
- `trajetoria_real.kml` — ground truth
- `trajetoria_estimada_<DETECTOR>.kml` — trajetória estimada

---

## Convenções e decisões de design

- **Imports neurais são opcionais**: flags `TORCH_AVAILABLE`, `KORNIA_AVAILABLE`, `MATCHFORMER_AVAILABLE` controlam o que está disponível. `ImportError` descritivo se o usuário escolher um detector sem o pacote instalado.
- **`device` resolvido uma única vez** em `__init__` via `_resolver_device()`. `device: cuda` sem GPU lança `RuntimeError` imediatamente.
- **Parâmetros neurais usam `.get(..., {})`** — seção ausente no YAML não quebra o código.
- **Projeção UTM local**: calculada na primeira posição GPS para converter lat/lon em XY metros (para plot e métricas).
- **Map matching opcional**: usa `rasterio` para recortar ROI do GeoTIFF e homografia para corrigir drift. Filtro complementar: 30% mapa + 70% odometria.
- **GSD**: escala da imagem em metros/pixel — único mecanismo de recuperação de escala monocular.
- **Yaw**: acumulado frame a frame a partir do Euler Z da matriz de rotação. Filtro nos primeiros 2 frames descarta deltas > 45°.

---

## CLI — argumentos de `main.py`

```
python main.py [--config CONFIG] [--detector {ORB,AKAZE,SUPERPOINT,LOFTR,MATCHFORMER}] [--device {auto,cpu,cuda}]
```

Argumentos CLI sobrescrevem os campos equivalentes do YAML.

---

## Instalar e testar

```bash
# Instalação (Linux/Mac)
bash install.sh            # sem redes neurais
bash install.sh --neural   # com torch (CPU ou GPU)

# Windows
install.bat --neural

# Testar tudo (36 testes)
python -m pytest test_pipeline.py -v

# Executar
python main.py
python main.py --detector SUPERPOINT --device cpu
```

---

## O que NÃO está implementado (pontos de expansão)

- Suporte a múltiplas câmeras (stereo)
- Otimização de pose em janela deslizante (bundle adjustment)
- Loop closure
- Exportação de nuvem de pontos
- Interface gráfica para configuração
