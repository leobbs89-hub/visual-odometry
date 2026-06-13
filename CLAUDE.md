# CLAUDE.md — Odometria Visual Monocular

## Visão geral
Pipeline de odometria visual monocular com localização absoluta por map matching.
Dissertação de mestrado (ITA). Linguagem: Python 3.x, Windows.

## Estrutura de módulos

| Arquivo | Responsabilidade |
|---|---|
| `main.py` | Ponto de entrada; parse de args e YAML; instancia `OdometriaVisual` |
| `odometria_visual.py` | Classe principal `OdometriaVisual(MapMatchingMixin)` — pipeline, detectores, matching, pose |
| `map_matching.py` | `MapMatchingMixin` — localização absoluta via GeoTIFF + homografia |
| `utils.py` | Funções puras (geração de KML) |
| `config.yaml` | Config padrão (não editar para testes pessoais) |
| `config_pessoal.yaml` | Config do usuário (paths reais, detectores preferidos) |

## Dependências externas (não são pacotes pip)

- **LightGlue/** — repositório clonado localmente (`git clone https://github.com/cvg/LightGlue.git`), instalado com `pip install -e LightGlue\`. Adicionado ao `sys.path` em runtime.
- **MatchFormer/** — repositório clonado localmente (`git clone https://github.com/InSAI-Lab/MatchFormer.git`). Adicionado ao `sys.path` em runtime. **Não tem `pip install`** — importado diretamente via path.

`install.bat --neural` cuida de ambos automaticamente.

## Detectores suportados

| Detector | Flag de disponibilidade | Deps extras |
|---|---|---|
| ORB | sempre disponível | — |
| AKAZE | sempre disponível | — |
| SUPERPOINT | `TORCH_AVAILABLE` | torch + LightGlue/ |
| LOFTR | `KORNIA_AVAILABLE` | kornia |
| MATCHFORMER | `MATCHFORMER_AVAILABLE` | timm, einops, MatchFormer/ |

Os três flags são definidos no topo de `odometria_visual.py` em blocos `try/except` independentes.

## Comportamento de cada detector no matching (`_obter_correspondencias`)

- **ORB/AKAZE**: retorna keypoints + descriptors; matcher BFMatcher/FLANN separado
- **SUPERPOINT**: `match_pair(extractor, matcher, img1, img2)` → chaves `keypoints0/1`, `matches0`
- **LOFTR**: `mat({'image0': t1, 'image1': t2})` → **retorna dict** com `keypoints0`, `keypoints1`
- **MATCHFORMER**: `mat(data)` → **modifica `data` in-place, não retorna nada** → ler `data['mkpts0_f']`, `data['mkpts1_f']`

## Instanciação dos detectores

- LOFTR: `LoFTR(**params)` — aceita kwargs
- MATCHFORMER: `Matchformer(params)` — aceita **um único dict**, NÃO `**params`
- Config YAML do MatchFormer precisa da estrutura completa (ver `config.yaml`, seção `matchformer`)

## Map Matching — pontos críticos

**CRS projetado (ex: EPSG:32723 UTM):** `patch_transform * (px, py)` retorna coordenadas UTM, não lat/lon. A conversão inversa é aplicada em `_estimar_posicao_pela_homografia` com `self._map_transformer.transform(..., direction='INVERSE')`. Sem isso, `est_lat`/`est_lon` ficam NaN → crash no pyproj.

**Validação de saída:** após a conversão geo, verificar `np.isfinite` e intervalo `-90≤lat≤90, -180≤lon≤180` antes de retornar. `cv.perspectiveTransform` pode retornar NaN para homografias degeneradas.

**Fluxo:** `_corrigir_posicao_pelo_mapa` → `_busca_escala` (testa 3 escalas) → `_estimar_posicao_pela_homografia` → retorna `(lat, lon, yaw, escala, n_inliers, map_ok)`. Se `map_ok=False`, mantém posição da odometria.

## Config — parâmetros que mais afetam resultado

```yaml
detector: MATCHFORMER   # ou ORB | AKAZE | SUPERPOINT | LOFTR
device: auto            # auto | cpu | cuda

detector_params:
  matchformer:
    backbone_type: largesea  # litela | largela | litesea | largesea
    match_coarse:
      thr: 0.1               # menor = mais matches (padrão 0.2)
      dsmax_temperature: 0.1  # maior = mais matches

map_matching:
  enabled: true
  absolute_detector: SUPERPOINT  # detector usado para comparar com mapa
  interval: 1                    # a cada quantos frames rodar
  roi_size_m: 1000               # tamanho da ROI no mapa (metros)
```

## Como executar

```powershell
# Ativar ambiente
.venv\Scripts\activate.bat

# Rodar com config pessoal
python main.py --config config_pessoal.yaml

# Sobrescrever detector sem editar YAML
python main.py --config config_pessoal.yaml --detector LOFTR

# Forçar CPU
python main.py --config config_pessoal.yaml --device cpu
```

## Instalação

```bat
install.bat            # apenas ORB/AKAZE
install.bat --neural   # inclui torch (auto-detecta GPU), kornia, timm, einops, LightGlue, MatchFormer
install.bat --neural --cpu       # força CPU mesmo com GPU presente
install.bat --neural --cuda 124  # força CUDA 12.4
```

## GPU / PyTorch

- PyTorch com CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`
- Verificar: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
- Se `TORCH_AVAILABLE=False`, verificar se `LightGlue/` não está vazio (pasta vazia existe mas clone falhou)

## Branch principal de desenvolvimento

`claude/odometria-modular-refactor` — branch ativa. `main` é a base.
