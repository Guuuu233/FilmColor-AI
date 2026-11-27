#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "Starting FilmColor v2.0 Plugin Bridge (http://127.0.0.1:18765)..."
exec "$DIR/.venv/bin/python" -m filmcolor.plugin_bridge --port 18765
