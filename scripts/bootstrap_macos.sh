#!/usr/bin/env bash
# Native Apple-Silicon setup for the Qwen3-TTS MLX profiles.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "Error: this bootstrap is for macOS." >&2
    exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "Error: run a native arm64 shell on Apple Silicon; Rosetta/x86_64 is unsupported." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not installed. Install it with:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

# Speech Swift's Homebrew bottle is native-arm64 only.
if [[ ! -x /opt/homebrew/bin/brew ]]; then
    echo "Error: native ARM Homebrew was not found at /opt/homebrew/bin/brew." >&2
    echo "Install Homebrew natively, then run this script again." >&2
    exit 1
fi

export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:$PATH"

echo "Ensuring Python 3.12 and the MLX Python environment..."
uv python install 3.12
uv sync --python 3.12 --extra mac

if ! command -v speech >/dev/null 2>&1 && ! command -v audio >/dev/null 2>&1; then
    echo "Installing Speech Swift for the exact aufklarer 8-bit model..."
    brew install speech
else
    echo "Speech Swift is already installed."
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Installing ffmpeg for optional MP3/Opus responses..."
    brew install ffmpeg
fi

cat <<'EOF'

Apple-Silicon setup complete.

Exact aufklarer 8-bit backend (Speech Swift):
  uv run vibevoice-server --model qwen3-tts-mlx-8bit --host 127.0.0.1 --port 8000

1.7B 4-bit backend (mlx-audio):
  uv run vibevoice-server --model qwen3-tts-mlx-4bit --host 127.0.0.1 --port 8000

The first request downloads the selected model to the normal Hugging Face cache.
EOF
