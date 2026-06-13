@echo off
REM install.bat — Configura o ambiente no Windows
REM
REM Uso:
REM   install.bat              instala apenas ORB / AKAZE (sem torch)
REM   install.bat --neural     inclui detectores neurais (auto-detecta GPU NVIDIA)
REM   install.bat --neural --cpu       forca instalacao CPU mesmo com GPU presente
REM   install.bat --neural --cuda 124  forca CUDA 12.4 (padrao auto: cu126)

setlocal enabledelayedexpansion

set USE_NEURAL=false
set FORCE_CPU=false
set CUDA_VER=cu126
set NEXT_IS_CUDA=false

for %%A in (%*) do (
    if "!NEXT_IS_CUDA!"=="true" (
        set CUDA_VER=cu%%A
        set NEXT_IS_CUDA=false
    ) else if "%%A"=="--neural"  set USE_NEURAL=true
    if "%%A"=="--cpu"     set FORCE_CPU=true
    if "%%A"=="--cuda"    set NEXT_IS_CUDA=true
)

echo ========================================
echo  Instalacao - Odometria Visual
echo ========================================

REM ---------- 1. Verificar Python ----------
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado. Instale em https://python.org
    pause & exit /b 1
)
echo [OK] Python encontrado.

REM ---------- 2. Ambiente virtual ----------
if not exist ".venv" (
    echo [INFO] Criando ambiente virtual em .venv ...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
echo [OK] Ambiente virtual ativado.

REM ---------- 3. Pacotes principais ----------
echo [INFO] Instalando dependencias principais ...
pip install --upgrade pip -q
pip install -r requirements.txt

REM ---------- 4. Pacotes para detectores neurais ----------
if "%USE_NEURAL%"=="true" (
    echo.
    echo [INFO] Instalando dependencias para detectores neurais ...

    REM --- Detectar GPU NVIDIA e instalar PyTorch correto ---
    if "%FORCE_CPU%"=="true" (
        echo [INFO] Modo CPU forcado ^(--cpu^). Instalando PyTorch CPU ...
        pip install torch torchvision torchaudio -q
    ) else (
        nvidia-smi >nul 2>&1
        if not errorlevel 1 (
            echo [INFO] GPU NVIDIA detectada. Instalando PyTorch com CUDA !CUDA_VER! ...
            echo        ^(use --cuda 124 para CUDA 12.4, --cpu para forcar CPU^)
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/!CUDA_VER! -q
        ) else (
            echo [INFO] GPU NVIDIA nao detectada. Instalando PyTorch CPU ...
            pip install torch torchvision torchaudio -q
        )
    )

    echo [INFO] Instalando kornia e outras dependencias neurais ...
    pip install "kornia>=0.7"

    echo.
    echo [INFO] Baixando checkpoint LoFTR ^(loftr_outdoor.ckpt^) ...
    set LOFTR_CKPT=%USERPROFILE%\.cache\torch\hub\checkpoints\loftr_outdoor.ckpt
    if not exist "!LOFTR_CKPT!" (
        python -c "import ssl, urllib.request, os; ssl._create_default_https_context = ssl._create_unverified_context; dest = os.path.expanduser(r'~\.cache\torch\hub\checkpoints\loftr_outdoor.ckpt'); os.makedirs(os.path.dirname(dest), exist_ok=True); urllib.request.urlretrieve('http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt', dest); print('[OK] Checkpoint salvo em', dest)"
    ) else (
        echo [OK] Checkpoint LoFTR ja existe, pulando download.
    )
)

REM ---------- 5. Repositórios do GitHub ----------
echo.
echo [INFO] Clonando repositorios do GitHub ...

where git >nul 2>&1
if errorlevel 1 (
    echo [AVISO] Git nao encontrado. Instale em https://git-scm.com e
    echo         re-execute este script, ou clone manualmente:
    echo           git clone https://github.com/cvg/LightGlue.git
    echo           git clone https://github.com/InSAI-Lab/MatchFormer.git
    goto :fim
)

if not exist "LightGlue" (
    echo  -^> Clonando LightGlue ...
    git clone https://github.com/cvg/LightGlue.git
    pip install -e LightGlue\ -q
    echo [OK] LightGlue instalado.
) else (
    echo [OK] LightGlue ja existe, pulando clone.
)

if not exist "MatchFormer" (
    echo  -^> Clonando MatchFormer ...
    git clone https://github.com/InSAI-Lab/MatchFormer.git
    echo [OK] MatchFormer clonado.
) else (
    echo [OK] MatchFormer ja existe, pulando clone.
)

:fim
echo.
echo ========================================
echo  Instalacao concluida!
echo.
echo  Para ativar o ambiente na proxima vez:
echo    .venv\Scripts\activate.bat
echo.
echo  Para executar:
echo    python main.py
echo    python main.py --detector AKAZE
echo    python main.py --detector SUPERPOINT         (usa GPU se instalado com --neural)
echo    python main.py --detector LOFTR
echo    python main.py --detector SUPERPOINT --device cpu   (forca CPU)
echo ========================================
pause
