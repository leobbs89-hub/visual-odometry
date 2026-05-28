@echo off
REM install.bat — Configura o ambiente no Windows
REM
REM Uso:
REM   install.bat              instala apenas ORB / AKAZE (sem torch)
REM   install.bat --neural     inclui detectores neurais (CPU ou GPU)

setlocal enabledelayedexpansion

set USE_NEURAL=false
for %%A in (%*) do (
    if "%%A"=="--neural" set USE_NEURAL=true
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
    echo        torch funciona em CPU - GPU NAO e obrigatoria
    echo        Para GPU NVIDIA, ajuste requirements-neural.txt antes de continuar.
    echo        Consulte: https://pytorch.org/get-started/locally/
    pip install -r requirements-neural.txt
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
echo    python main.py --detector SUPERPOINT         (rede neural em CPU)
echo    python main.py --detector SUPERPOINT --device cpu
echo ========================================
pause
