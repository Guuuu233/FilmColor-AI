@echo off
chcp 65001 >nul
setlocal
set "DIR=%~dp0.."
set "PY=%DIR%\.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [错误] 找不到 Python 运行环境: %PY%
    echo 请先双击运行 scripts\install_windows.bat 完成初始化安装。
    pause
    exit /b 1
)
echo Starting FilmColor v2.0 Plugin Bridge (http://127.0.0.1:18765)...
"%PY%" -m filmcolor.plugin_bridge --port 18765
if errorlevel 1 pause
