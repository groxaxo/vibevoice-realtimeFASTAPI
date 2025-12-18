#!/bin/bash
# Bootstrap script to set up VibeVoice locally using uv
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "🔧 Bootstrapping VibeVoice with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Ensure Python 3.11 is available + pinned for this repo
echo "📦 Ensuring Python 3.11..."
uv python install 3.11 >/dev/null 2>&1 || true
uv python pin 3.11

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "🐍 Creating virtual environment..."
    uv venv --python 3.11
else
    echo "✅ Virtual environment already exists"
fi

# Clone VibeVoice if it doesn't exist
VIBEVOICE_DIR="third_party/VibeVoice"
if [ ! -d "$VIBEVOICE_DIR" ]; then
    echo "📥 Cloning VibeVoice repository..."
    mkdir -p third_party
    git clone --quiet --branch main --depth 1 \
        https://github.com/microsoft/VibeVoice.git "$VIBEVOICE_DIR"
else
    echo "✅ VibeVoice repository already exists"
fi

# Apply overrides
if [ -f "overrides/app.py" ]; then
    echo "🔧 Applying custom app.py override..."
    cp overrides/app.py "$VIBEVOICE_DIR/demo/web/app.py"
fi

echo "📚 Syncing dependencies (this will install VibeVoice from third_party/ via pyproject.toml)..."
uv sync

echo "✅ Bootstrap complete!"
echo ""
echo "Next steps:"
echo "  1. Download the model: uv run python scripts/download_model.py"
echo "  2. Run the demo: uv run python scripts/run_realtime_demo.py --port 8000"

