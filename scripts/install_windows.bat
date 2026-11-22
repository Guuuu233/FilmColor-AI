@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."
echo ======================================================
echo   FilmColor v1.0 Windows 一键环境配置与插件安装
echo ======================================================

where uv >nul 2>nul
if %errorlevel%==0 (
    echo [1/2] 检测到 uv，正在同步精简运行环境（~140 MB）...
    uv sync --no-group ml --no-dev
    if errorlevel 1 (
        echo [错误] uv sync 失败，请检查网络或 Python 3.12/3.13 环境。
        pause
        exit /b 1
    )
) else (
    echo [1/2] 未检测到 uv，尝试使用系统 Python 创建 .venv 并安装依赖...
    where python >nul 2>nul
    if errorlevel 1 (
        echo [错误] 未找到 python 或 uv，请先安装 Python 3.12/3.13 或 uv。
        pause
        exit /b 1
    )
    if not exist ".venv\Scripts\python.exe" (
        python -m venv .venv
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -e .
    if errorlevel 1 (
        echo [错误] pip 安装依赖失败。
        pause
        exit /b 1
    )
)

echo [2/2] 正在将 FilmColor v1.0 插件安装到 Lightroom Classic...
".venv\Scripts\python.exe" scripts\install_plugins.py --force
if errorlevel 1 (
    echo [错误] 插件安装失败。
    pause
    exit /b 1
)

echo.
echo [完成] FilmColor v1.0 已就绪！
echo  - Lightroom Classic：重启即可在「图库 → 增效工具额外信息 → FilmColor 负片转换…」使用
echo  - Photoshop UXP：先双击 scripts\start_filmcolor_bridge.bat，再在 UXP Developer Tool 加载 plugins\photoshop-uxp\manifest.json
pause
