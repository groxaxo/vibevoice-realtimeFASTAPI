#!/usr/bin/env bash
# Bootstrap the PyTorch/VibeVoice runtime with uv.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Bootstrapping VibeVoice with uv..."

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not installed. Install it first:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

uv python install 3.11 >/dev/null 2>&1 || true
uv python pin 3.11

if [[ ! -d .venv ]]; then
    uv venv --python 3.11
fi

VIBEVOICE_DIR="third_party/VibeVoice"
if [[ ! -f "$VIBEVOICE_DIR/pyproject.toml" ]]; then
    mkdir -p third_party
    if [[ -f .gitmodules ]] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "Initializing the VibeVoice submodule..."
        git submodule update --init --recursive third_party/VibeVoice
    else
        echo "Cloning VibeVoice..."
        rm -rf "$VIBEVOICE_DIR"
        git clone --quiet --branch main --depth 1 \
            https://github.com/microsoft/VibeVoice.git "$VIBEVOICE_DIR"
    fi
fi

if [[ -f overrides/app.py ]]; then
    cp overrides/app.py "$VIBEVOICE_DIR/demo/web/app.py"
fi

echo "Syncing the VibeVoice dependency set..."
uv sync --extra vibevoice

echo "Installing the populated VibeVoice checkout editable..."
uv pip install -e "$VIBEVOICE_DIR"

echo "Bootstrap complete."
echo "Download: uv run python scripts/download_model.py --model realtime-0.5b"
echo "Run:      uv run vibevoice-server --model realtime-0.5b --port 8000"
