@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."
echo ======================================================
echo   FilmColor v1.0 Windows 一键自动安装（免预装 Python）
echo ======================================================

set "UV_EXE="
where uv >nul 2>nul && set "UV_EXE=uv"
if not defined UV_EXE if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"

if not defined UV_EXE (
    echo [1/3] 未检测到 uv，正在自动下载轻量引导器 uv（无需预先安装 Python）...
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
)

if defined UV_EXE (
    echo [2/3] 正在自动配置独立 Python 3.13 与精简运行环境（~135 MB）...
    "%UV_EXE%" sync --no-group ml --no-dev
    if errorlevel 1 (
        echo [错误] uv sync 失败，请检查网络连接后重试。
        pause
        exit /b 1
    )
) else (
    echo [错误] 自动获取 uv 失败，请检查网络或手动安装 Python 3.12/3.13 后重试。
    pause
    exit /b 1
)

echo [3/3] 正在将 FilmColor v1.0 插件安装到 Lightroom Classic...
".venv\Scripts\python.exe" scripts\install_plugins.py --force
if errorlevel 1 (
    echo [错误] 插件安装失败。
    pause
    exit /b 1
)

echo.
echo [完成] FilmColor v1.0 已就绪！
echo  - Lightroom Classic：重启即可在「图库 → 增效工具额外信息 → FilmColor 负片转换…」使用
echo  - Photoshop UXP：先双击 scripts\start_filmcolor_bridge.bat，再加载 plugins\photoshop-uxp\manifest.json
pause
