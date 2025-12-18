# Podcast Creator - VibeVoice Realtime Local Runner

A local runner for the [VibeVoice Realtime](https://github.com/microsoft/VibeVoice) text-to-speech demo, adapted from the [Colab notebook](https://colab.research.google.com/github/microsoft/VibeVoice/blob/main/demo/vibevoice_realtime_colab.ipynb).

This project uses `uv` for Python environment and package management.

## Prerequisites

- **Ubuntu** (tested with CUDA GPU support) or **macOS** (tested on Apple Silicon)
- **uv** installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Python 3.11** (will be installed automatically by uv if needed)
- **Git** (for cloning VibeVoice repository)
- **Hugging Face account** (for downloading the model; may require authentication)
- **For Ubuntu with NVIDIA GPU**: CUDA-compatible PyTorch installation (see [PyTorch CUDA setup](#pytorch-cuda-setup))

## Quick Start

### 1. Bootstrap the environment

This will:

- Create a Python 3.11 virtual environment
- Install project dependencies
- Clone VibeVoice into `third_party/VibeVoice`
- Install VibeVoice as an editable package

```bash
./scripts/bootstrap_uv.sh
```

### 2. Download the model

Download the VibeVoice-Realtime-0.5B model from Hugging Face:

```bash
uv run python scripts/download_model.py
```

**Note:** If the model is gated, you'll need to authenticate:

- Run `huggingface-cli login` (or `uv run huggingface-cli login`), or
- Set the `HF_TOKEN` environment variable

The model will be downloaded to `models/VibeVoice-Realtime-0.5B/` (about ~2GB).

### 3. Run the demo

Start the realtime demo server:

```bash
uv run python scripts/run_realtime_demo.py --port 8000
```

Then open your browser to **<http://127.0.0.1:8000>** to use the interactive web UI.

Press `Ctrl+C` to stop the server.

## PyTorch CUDA Setup

For Ubuntu systems with NVIDIA GPUs, you need PyTorch with CUDA support:

### Option 1: Install PyTorch with CUDA in the existing environment

After running the bootstrap script, install PyTorch with CUDA support:

```bash
# For CUDA 11.8
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Option 2: Verify CUDA availability

Check if CUDA is available:

```bash
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

## Advanced Usage

### Custom model path

```bash
uv run python scripts/run_realtime_demo.py \
  --model-path /path/to/your/model \
  --port 8000
```

### Device selection

The script auto-detects the best device (prefers CUDA on Linux, MPS on macOS), but you can override:

```bash
# Use CUDA (NVIDIA GPU on Linux/Ubuntu)
uv run python scripts/run_realtime_demo.py --device cuda

# Use MPS (Apple Silicon GPU on macOS)
uv run python scripts/run_realtime_demo.py --device mps

# Use CPU
uv run python scripts/run_realtime_demo.py --device cpu
```

### Development mode (auto-reload)

```bash
uv run python scripts/run_realtime_demo.py --reload
```

## Configuration

The VibeVoice realtime demo supports several configuration parameters that affect speech generation quality and behavior. These can be adjusted in the web UI once the server is running.

### CFG Scale (Classifier-Free Guidance)

**Default:** `1.5`  
**Range:** `1.3` to `3.0`  
**What it does:** Controls how strongly the model follows the text condition during diffusion-based speech generation.

- **Lower values (1.3–1.5)**: More natural, varied prosody; may be less strict to text
- **Higher values (2.0–3.0)**: More strict adherence to text; may sound more constrained or less natural

The CFG mechanism blends conditioned (with text) and unconditioned (without text) predictions:

```python
output = uncond_prediction + cfg_scale × (cond_prediction - uncond_prediction)
```

**Recommendation:** Start with the default `1.5` for balanced quality and adherence. Adjust based on your needs:

- For more natural, conversational speech → lower (1.3–1.4)
- For more precise, controlled speech → higher (2.0–2.5)

### Inference Steps

**Default:** `5`  
**Range:** `5` to `20`  
**What it does:** Number of diffusion steps used during speech generation. More steps generally mean higher quality but slower generation.

- **Lower values (5–8)**: Faster generation, good quality for most use cases
- **Higher values (10–20)**: Slower but potentially higher quality, more detailed audio

**Recommendation:** The default `5` steps provides a good balance between speed and quality for real-time use. Increase if you need higher quality and can tolerate slower generation.

### Device Selection

Controls which hardware device is used for inference:

- **`cuda`** (default on Linux with NVIDIA GPU): NVIDIA GPU acceleration using CUDA
- **`mps`** (default on macOS with Apple Silicon): Uses Apple's Metal Performance Shaders for GPU acceleration
- **`cpu`**: CPU-only inference (slower but works everywhere)

The script auto-detects the best available device. See [Device selection](#device-selection) above for manual override.

### Voice Selection

The demo includes multiple voice presets that can be selected in the web UI. Available voices depend on the model and are loaded from `third_party/VibeVoice/demo/voices/streaming_model/`.

**Note:** Voice prompts are embedded in the model to mitigate deepfake risks and ensure low latency. Custom voice creation requires contacting the VibeVoice team.

## Project Structure

```text
vibevoice/
├── scripts/
│   ├── bootstrap_uv.sh          # Initial setup script
│   ├── download_model.py         # Model download script
│   └── run_realtime_demo.py     # Demo server launcher
├── third_party/
│   └── VibeVoice/                # Cloned VibeVoice repository
├── models/
│   └── VibeVoice-Realtime-0.5B/  # Downloaded model (created after step 2)
├── pyproject.toml                # Project configuration
└── README.md                     # This file
```

## Troubleshooting

### Model download fails with authentication error

If you see an authentication error, the model may be gated:

1. Log in to Hugging Face: `huggingface-cli login`
2. Or set `HF_TOKEN` environment variable
3. Or pass `--token <your_token>` to the download script

### MPS or CUDA device not available

If MPS is not available (older macOS or Intel Mac) or CUDA is not available (no NVIDIA GPU or missing CUDA drivers), the script will automatically fall back to CPU. You can also explicitly use CPU:

```bash
uv run python scripts/run_realtime_demo.py --device cpu
```

For Ubuntu systems, make sure you have:
- NVIDIA GPU drivers installed
- CUDA toolkit installed
- PyTorch with CUDA support (see [PyTorch CUDA setup](#pytorch-cuda-setup))

### Port already in use

If port 8000 is already in use, specify a different port:

```bash
uv run python scripts/run_realtime_demo.py --port 8080
```

### Bootstrap fails

Make sure:

- `uv` is installed and in your PATH
- You have internet connectivity
- Git is installed

## Differences from Colab

- **No cloudflared tunnel**: The demo runs purely locally (no public URL)
- **Multi-platform support**: Works on Ubuntu with CUDA GPUs and macOS with Apple Silicon
- **Auto-device detection**: Automatically detects and uses the best available device (CUDA, MPS, or CPU)
- **Local model storage**: Models are stored in `models/` directory instead of `/content/models/`

## Reference

- [VibeVoice GitHub](https://github.com/microsoft/VibeVoice)
- [VibeVoice Realtime Colab](https://colab.research.google.com/github/microsoft/VibeVoice/blob/main/demo/vibevoice_realtime_colab.ipynb)
- [VibeVoice Realtime Documentation](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md)
