#!/usr/bin/env bash
# install.sh — Configura o ambiente do pipeline de Odometria Visual
#
# Uso:
#   bash install.sh             — instala apenas ORB / AKAZE (sem torch)
#   bash install.sh --neural    — inclui detectores neurais (CPU ou GPU)

set -e

USE_NEURAL=false
for arg in "$@"; do
  [[ "$arg" == "--neural" ]] && USE_NEURAL=true
done

echo "========================================"
echo " Instalação — Odometria Visual"
echo "========================================"

# ---------- 1. Verificação do Python ----------
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
  echo "[ERRO] Python 3 não encontrado. Instale em https://python.org"
  exit 1
fi
echo "[OK] Python: $($PYTHON --version)"

# ---------- 2. Ambiente virtual ----------
if [ ! -d ".venv" ]; then
  echo "[INFO] Criando ambiente virtual em .venv ..."
  $PYTHON -m venv .venv
fi
source .venv/bin/activate
echo "[OK] Ambiente virtual ativado."

# ---------- 3. Pacotes principais ----------
echo "[INFO] Instalando dependências principais ..."
pip install --upgrade pip -q
pip install -r requirements.txt

# ---------- 4. Pacotes para detectores neurais (opcional) ----------
if [ "$USE_NEURAL" = true ]; then
  echo ""
  echo "[INFO] Instalando dependências para detectores neurais ..."
  echo "       (torch funciona em CPU — GPU não é obrigatória)"
  echo "       Para usar GPU NVIDIA, consulte https://pytorch.org e ajuste"
  echo "       a linha do torch em requirements-neural.txt antes de continuar."
  pip install -r requirements-neural.txt
fi

# ---------- 5. Repositórios do GitHub ----------
echo ""
echo "[INFO] Clonando repositórios do GitHub ..."

# LightGlue — necessário para SUPERPOINT
if [ ! -d "LightGlue" ]; then
  echo " -> Clonando LightGlue ..."
  git clone https://github.com/cvg/LightGlue.git
  pip install -e LightGlue/ -q
  echo "[OK] LightGlue instalado."
else
  echo "[OK] LightGlue já existe, pulando clone."
fi

# MatchFormer — necessário para MATCHFORMER
if [ ! -d "MatchFormer" ]; then
  echo " -> Clonando MatchFormer ..."
  git clone https://github.com/InSAI-Lab/MatchFormer.git
  echo "[OK] MatchFormer clonado."
else
  echo "[OK] MatchFormer já existe, pulando clone."
fi

echo ""
echo "========================================"
echo " Instalação concluída!"
echo ""
echo " Para ativar o ambiente na próxima vez:"
echo "   source .venv/bin/activate"
echo ""
echo " Para executar:"
echo "   python main.py                          # ORB (padrão do config.yaml)"
echo "   python main.py --detector AKAZE"
echo "   python main.py --detector SUPERPOINT    # rede neural em CPU"
echo "   python main.py --detector SUPERPOINT --device cpu"
echo "========================================"
