# Pesquisa: implementações que podem melhorar a acurácia do Map Matching

**Data:** 2026-07-10
**Motivação:** teste isolado NAIP (Champaign-Urbana) reduziu o erro "com MM" de
~3000–4600 m para 222 m após corrigir a busca de ângulo, mas ainda 3,4× pior
que a baseline sem correção (64,6 m). Esta pesquisa busca técnicas publicadas
(papers + repositórios) que possam fechar essa lacuna.

## Como o módulo atual funciona (baseline de comparação)

`map_matching.py`: recorte do mapa base → busca discreta de ângulo
(`_busca_angulo`, ±30°, 5 candidatos, ou variante Fourier-Mellin/log-polar) →
busca discreta de escala (`_busca_escala`, ±5%, 3 candidatos) → matching de
features (ORB/AKAZE/SuperPoint+LightGlue/LoFTR/MatchFormer, via
`_obter_correspondencias` compartilhado com a VO) → `cv.findHomography` RANSAC
→ projeção do centro da imagem pela homografia → conversão pixel→geo pela
afim inversa acumulada. Cada candidato de ângulo/escala roda o pipeline de
matching completo de novo (custo alto, busca grosseira).

## Candidatos avaliados

| # | Candidato | O que resolve | Esforço de integração | Risco |
|---|-----------|----------------|------------------------|-------|
| 1 | **RoMa / RoMa v2** — Parskatt/RoMa (CVPR 2024 + v2 2025) | Matcher denso (pixel-dense warp + certeza), estado da arte em pares aéreo↔satélite com grande diferença de escala/rotação; RoMa v2 reduz EPE em -84% vs RoMa v1 no benchmark AerialMegaDepth, superando LoFTR e DKM. Ataca diretamente o problema de poucos matches/matches ruidosos entre imagem de voo e ortofoto de fontes independentes (exatamente o cenário NAIP). | Baixo — mesmo ponto de plugue que LOFTR/MATCHFORMER hoje (`absolute_detector: ROMA`), sem mudar a lógica de busca de ângulo/escala nem a conversão pixel→geo. | Custo computacional maior por frame (matcher denso); pipeline já é offline, então aceitável. Licença MIT. |
| 2 | **OrthoLoC + AdHoP** — deepscenario.github.io/OrthoLoC (arXiv 2509.18350, código público) | Mesmo problema estrutural do usuário: localizar UAV contra ortofoto governamental (DOP) + DSM. A técnica central, **Adaptive Homography Preconditioning (AdHoP)**, substitui a busca discreta de ângulo/escala por um pré-warp iterativo/contínuo da ortofoto guiado por uma homografia estimada grosseiramente, seguido de re-matching — reportam até +95% de acurácia de matching e -63% de erro de translação vs matching direto. Também propõe lift 2D→3D via DSM + PnP em vez de homografia planar pura (relevante se parte do erro residual vier de relevo/parallax, não só de erro de rotação/escala). | Médio/alto — exige reestruturar `_busca_angulo`/`_busca_escala` para um esquema de refinamento iterativo em vez de grid discreto; a parte DSM+PnP é opcional e mais invasiva. | Maior esforço de port, mas é o candidato que ataca a mesma etapa (pré-alinhamento antes do matching fino) onde a lacuna de 222 m hoje se origina. |
| 3 | **Image Matching for UAV Geolocation: Classical vs Deep Learning** — MDPI Drones 2025, DOI 10.3390/drones11110409 | Comparação direta ORB/SIFT/AKAZE/template matching (NCC+voting) vs SuperPoint+SuperGlue/LoFTR em geolocalização por altitude. Achado: LoFTR/SIFT melhores <1000 m; SuperGlue/LoFTR dominam em 2500–3000 m (maior diferença de escala/aberração geométrica). | Nenhum (não é código, é referência de calibração) — dataset do usuário é 1500 m constante, mas serve para validar se `absolute_detector` ideal muda por altitude caso o mach/altitude varie no futuro. | — |
| 4 | **Cross-view geo-localization com retrieval** — benchmark University-1652, github.com/hmf21/UAVLocalization, séries de papers CVGL 2024–2025 (Game4Loc, OS-FPI, 3D Geometric Perception) | Usam retrieval tipo NetVLAD/CosPlace para localização grosseira antes do matching fino — não depende de dead-reckoning para centrar o ROI. Não ataca a lacuna de acurácia atual diretamente, mas é rede de segurança para os colapsos de tracking (⚠️ documentados na VO principal, ex. FLORESTA mach 1.8, URBANO mach ≥ 1.5) onde o ROI centrado por odometria já estaria muito errado. | Alto — pipeline de retrieval separado, banco de embeddings do mapa base pré-computado. | Fora do escopo imediato do gap NAIP; útil como item de trabalho futuro separado. |

## Recomendação de prioridade

1. **Testar RoMa como `absolute_detector`** no experimento NAIP existente
   (`rodar_teste_map_matching.py`), comparando erro final contra os detectores
   já testados (LOFTR/MATCHFORMER/SUPERPOINT). Troca de baixo esforço no mesmo
   ponto de integração — maior chance de ganho rápido de acurácia sem mexer na
   lógica de busca de ângulo/escala.
2. **Se o gap persistir após (1)**, prototipar uma versão simplificada de
   AdHoP: em vez de grid discreto de ângulo/escala, usar a homografia do frame
   anterior como estimativa inicial e refinar iterativamente (warp → re-match →
   nova homografia → repetir até convergência ou N iterações), substituindo
   `_busca_angulo`/`_busca_escala`. Esse é o candidato que ataca mais
   diretamente a causa estrutural do resíduo de 222 m, já que a busca discreta
   atual é uma versão grosseira da mesma ideia.
3. Manter (3) como referência de calibração ao expandir para outras altitudes,
   e (4) como item de backlog separado (robustez a colapso de tracking), não
   relacionado à lacuna de acurácia atual.

## Resultado do teste — RoMa (2026-07-10, branch `feature/roma-map-matching`)

Implementado `RomaDetector` em `detectors.py` (wrapper em torno do pacote
PyPI `romatch`) e testado como `absolute_detector` no mesmo cenário usado
para medir o gap original (rota CHAMPAIGN, mach 1.0, detector principal SIFT,
`map.vrt`, `roi_center_mode: estimado`, `angle_estimation_method: grid_search`,
`roi_margin_factor: 1.3`), comparando diretamente contra LOFTR na mesma
rodada:

| AbsDetector | Sem MM (m) | Com MM (m) | Delta   | Tempo (s) |
|-------------|-----------:|-----------:|--------:|----------:|
| LOFTR       |       64,6 |      221,0 |  +156,4 |     517,2 |
| ROMA        |       64,6 |      222,8 |  +158,2 |     717,3 |

**Resultado: ROMA não reduziu o gap** (222,8 m vs 221,0 m — diferença de
1,8 m, dentro do ruído esperado do RANSAC) e ficou ~39% mais lento (717s vs
517s) com os parâmetros usados (`num_matches=5000`, resolução default
560→864, sem `sample_thresh` customizado). O matcher denso não trouxe
vantagem sobre o LOFTR neste par de fontes (Google Earth × NAIP) nesta
rota/altitude.

Hipóteses não testadas para tentar melhorar o resultado do RoMa antes de
descartá-lo: (a) `sample_thresh` mais alto para descartar matches de baixa
confiança antes do RANSAC (hoje aceita tudo dentro dos 5000 amostrados);
(b) `num_matches` maior; (c) como o RoMa já é relativamente robusto a
diferença de rotação/escala, a busca discreta de ângulo/escala do
`map_matching.py` pode estar sendo redundante para ele — testar
`angle_estimation_method: odometria` (sem busca) para ver se o resultado
muda pouco (o que economizaria ~5-7x o custo por frame) ou se a busca
ainda ajuda (o que reforça a suspeita de que o gap residual não vem do
matcher, mas de outro lugar — precisão geométrica do RANSAC/homografia,
resolução do patch de 480px, ou gap de domínio real entre as duas fontes).

**Conclusão prática:** o candidato (1) da recomendação abaixo não resolveu
o gap sozinho. Item (2) — AdHoP — permanece como próximo candidato mais
provável, já que ataca uma causa estrutural diferente (pré-alinhamento
iterativo) em vez de só trocar o matcher.

## Fontes

- RoMa: https://github.com/Parskatt/RoMa
- RoMa v2 paper: https://arxiv.org/html/2511.15706v1
- OrthoLoC (dataset + código): https://deepscenario.github.io/OrthoLoC
- OrthoLoC paper: https://arxiv.org/html/2509.18350v2
- Image Matching for UAV Geolocation (Classical vs Deep Learning): https://www.mdpi.com/2313-433X/11/11/409
- UAVLocalization (retrieval + alinhamento contra satélite): https://github.com/hmf21/UAVLocalization
- University-1652 benchmark / CVGL surveys: https://arxiv.org/pdf/2409.16925 , https://arxiv.org/pdf/2604.01747
