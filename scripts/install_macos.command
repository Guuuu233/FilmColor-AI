#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"
echo "======================================================"
echo "  FilmColor v1.0 macOS 一键自动安装（免预装 Python）"
echo "======================================================"

UV_BIN="$(command -v uv || true)"
if [ -z "$UV_BIN" ] && [ -x "$HOME/.local/bin/uv" ]; then UV_BIN="$HOME/.local/bin/uv"; fi
if [ -z "$UV_BIN" ] && [ -x "/opt/homebrew/bin/uv" ]; then UV_BIN="/opt/homebrew/bin/uv"; fi

if [ -z "$UV_BIN" ]; then
  echo "[1/3] 正在自动下载轻量引导器 uv（无需预先安装 Python）..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  UV_BIN="$(command -v uv)"
fi

echo "[2/3] 正在自动配置独立 Python 3.13 与精简运行环境（~135 MB）..."
"$UV_BIN" sync --no-group ml --no-dev

echo "[3/3] 正在安装 FilmColor v1.0 Lightroom 插件..."
"$DIR/.venv/bin/python" scripts/install_plugins.py --force

echo ""
echo "[完成] FilmColor v1.0 已就绪！重启 Adobe Lightroom Classic 即可使用。"
