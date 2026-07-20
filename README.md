# VibeVoice Realtime FASTAPI

A local, OpenAI-compatible text-to-speech service with two platform-specific runtime paths:

- **Linux/NVIDIA:** Microsoft VibeVoice through the existing PyTorch/CUDA realtime demo.
- **Apple Silicon MacBooks:** Qwen3-TTS through native MLX backends, including the exact linked 1.7B 8-bit model and a matching 1.7B 4-bit model.

The server remains local and does not require a hosted inference provider.

## Supported model profiles

| Model key | Model source | Runtime | Platform | Notes |
|---|---|---|---|---|
| `realtime-0.5b` | `microsoft/VibeVoice-Realtime-0.5B` | Existing VibeVoice subprocess | Linux/CUDA, PyTorch fallback devices | Realtime websocket and web UI |
| `tts-1.5b` | `microsoft/VibeVoice-1.5B` | Native long-form adapter | Backend-dependent | Non-streaming |
| `tts-7b` | `vibevoice/VibeVoice-7B` by default | Native long-form adapter | Backend-dependent | Non-streaming |
| `qwen3-tts-mlx-8bit` | `aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-8bit` | Speech Swift CLI | Apple Silicon macOS | Exact requested 8-bit/minmax bundle, 24 kHz |
| `qwen3-tts-mlx-4bit` | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit` | `mlx-audio` in process | Apple Silicon macOS | 4-bit/affine bundle, 24 kHz |

The two Qwen backends are deliberately separate. The aufklarer 8-bit bundle is packaged for Speech Swift's minmax loader, while the registered 4-bit bundle is packaged for `mlx-audio`'s affine loader. Treating them as interchangeable produces configuration or tensor-loading failures.

List every canonical key and alias:

```bash
uv run vibevoice-server --list-models
```

## Apple Silicon MacBook setup

### Requirements

- Apple Silicon (`arm64`) Mac; do not run the server under Rosetta.
- macOS 15 or newer for the current Speech Swift package.
- Native ARM Homebrew at `/opt/homebrew`.
- `uv`.
- Enough unified memory for the selected model and runtime. The 4-bit profile is the lower-memory option.

### Install

```bash
git clone https://github.com/groxaxo/vibevoice-realtimeFASTAPI.git
cd vibevoice-realtimeFASTAPI
bash scripts/bootstrap_macos.sh
```

The bootstrap script:

1. Uses a Python 3.12 `uv` environment.
2. Installs the mutually isolated Python `mac` extra, including the model-compatible `mlx-audio==0.3.0` loader.
3. Installs Speech Swift for the exact 8-bit model.
4. Installs `ffmpeg` for optional MP3 and Opus responses.

The Qwen-only path does **not** import PyTorch or require the VibeVoice submodule. The `mac` and `vibevoice` extras are declared mutually exclusive because their validated Transformers versions are incompatible.

### Run the exact 8-bit model

```bash
uv run vibevoice-server \
  --model qwen3-tts-mlx-8bit \
  --host 127.0.0.1 \
  --port 8000
```

This adapter invokes Speech Swift with the exact model ID:

```text
aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-8bit
```

### Run the 4-bit model

```bash
uv run vibevoice-server \
  --model qwen3-tts-mlx-4bit \
  --host 127.0.0.1 \
  --port 8000
```

On first use, `mlx-audio` downloads:

```text
mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit
```

The 4-bit loader resolves its source in this order:

1. Explicit `--model-path` value, which may be a local directory or Hugging Face repository ID.
2. A non-empty registry-local directory under `models/`.
3. The registry Hugging Face ID.

That fallback fixes the previous launcher behavior that converted every source into a local `Path`, even when no local model directory existed.

### Basic synthesis request

Use WAV while validating the installation because it requires no transcoding:

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-tts-mlx-8bit",
    "input": "Hello from a Qwen3 TTS model running locally on a MacBook.",
    "voice": "default",
    "language": "english",
    "response_format": "wav"
  }' \
  --output speech.wav
```

For the 4-bit server, change only the model value:

```json
"model": "qwen3-tts-mlx-4bit"
```

### Reference-voice conditioning

The Base profiles do not expose named speaker presets. Supply a local reference-audio path that is readable by the server process:

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-tts-mlx-4bit",
    "input": "This line uses the reference speaker characteristics.",
    "language": "english",
    "ref_audio": "/absolute/path/to/reference.wav",
    "response_format": "wav"
  }' \
  --output cloned.wav
```

For the `mlx-audio` 4-bit profile, adding the matching transcript enables its ICL voice-cloning path:

```json
{
  "ref_audio": "/absolute/path/to/reference.wav",
  "ref_text": "Transcript of the reference recording"
}
```

`ref_audio` is a server-local path, not an upload URL. Do not expose this service to untrusted clients without authentication and path controls.

## Linux/NVIDIA VibeVoice setup

Clone the submodule, resolve the VibeVoice extra, and install the populated checkout editable:

```bash
git clone --recurse-submodules https://github.com/groxaxo/vibevoice-realtimeFASTAPI.git
cd vibevoice-realtimeFASTAPI
./scripts/bootstrap_uv.sh
uv run python scripts/download_model.py --model realtime-0.5b
```

Start the existing realtime application:

```bash
CUDA_VISIBLE_DEVICES=0 uv run vibevoice-server \
  --model realtime-0.5b \
  --host 0.0.0.0 \
  --port 8000
```

The legacy launcher remains available:

```bash
uv run python scripts/run_realtime_demo.py --port 8000
```

For CUDA builds that need FlashAttention outside Docker:

```bash
uv sync --extra vibevoice
uv pip install --no-build-isolation flash-attn
```

## Server architecture

There are now two serving paths rather than one application pretending every adapter is the realtime engine:

1. **Realtime VibeVoice:** `RealtimeDemoAdapter` launches the existing vendored demo and preserves `/stream`, `/web`, and the optimized CUDA path.
2. **Native adapter API:** Qwen3-TTS and non-realtime adapters run through `runner.api`, which calls the selected adapter directly.

A process binds to one active model. Requests naming another registered model receive HTTP `409` with a command showing how to start the correct instance. This prevents accidental requests from being synthesized by the wrong loaded model.

## Native API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | Service and endpoint summary |
| `GET /health` | Backend availability, active source, and load state |
| `GET /config` | Complete registry plus active model capabilities |
| `GET /v1/models` | Active OpenAI-style model row |
| `GET /v1/audio/voices` | Voices/conditioning modes for the active adapter |
| `POST /v1/audio/speech` | OpenAI-style speech synthesis |
| `GET /docs` | FastAPI/OpenAPI documentation |

### Qwen request fields

| Field | Type | Description |
|---|---|---|
| `model` | string | Active model key or alias |
| `input` | string | Text to synthesize |
| `voice` | string | `default`, or a local reference WAV path for Base profiles |
| `language` | string | `english`, `chinese`, `auto`, and other model-supported language names |
| `response_format` | string | `wav`, `mp3`, or `opus`; WAV is the native output |
| `temp` | float | Sampling temperature |
| `top_k` | integer | Top-k sampling cutoff |
| `top_p` | float | Nucleus sampling threshold; used by the 4-bit `mlx-audio` backend |
| `max_tokens` | integer | Maximum generated codec tokens |
| `repetition_penalty` | float | Used by the 4-bit `mlx-audio` backend |
| `ref_audio` | string | Absolute/local reference-audio path |
| `ref_text` | string | Optional reference transcript for 4-bit ICL cloning |

Streaming is not yet exposed by the native Qwen HTTP adapter. A request with `stream: true` is rejected instead of silently returning a buffered response.

## Configuration

### Model source override

The existing argument name is retained for compatibility:

```bash
uv run vibevoice-server --model qwen3-tts-mlx-4bit \
  --model-path /absolute/path/to/local/model
```

It can also be an alternate Hugging Face ID for the `mlx-audio` profile:

```bash
uv run vibevoice-server --model qwen3-tts-mlx-4bit \
  --model-path organization/compatible-qwen3-tts-mlx-model
```

For the Speech Swift 8-bit profile, use `--model-path` only as a Hugging Face model-ID override.

### Concurrency

MLX generation is serialized internally because model instances are not thread-safe. The API also defaults to one concurrent synthesis request:

```bash
uv run vibevoice-server --model qwen3-tts-mlx-4bit \
  --max-concurrent-requests 1
```

Increasing this value does not create independent model replicas.

### Environment variables

| Variable | Purpose |
|---|---|
| `TTS_MODEL` | Active model used by the Uvicorn factory/reload mode |
| `MODEL_PATH` | Model source override |
| `MODEL_DEVICE` | Device override for PyTorch adapters |
| `MAX_CONCURRENT_REQUESTS` | Native API admission limit |
| `SPEECH_SWIFT_BIN` | Explicit path to `speech` or legacy `audio` executable |
| `QWEN3_TTS_MODEL_ID` | Exact 8-bit Speech Swift model-ID override |
| `QWEN3_TTS_TIMEOUT_SECONDS` | Speech Swift subprocess timeout; default 1800 |
| `HF_HOME` | Hugging Face cache root |

## Local validation

No GitHub Actions workflow is required. Run checks on the target machine:

```bash
uv sync --extra dev --extra mac        # Apple Silicon
# uv sync --extra dev --extra vibevoice    # Linux/VibeVoice
uv run ruff check runner scripts test_runner.py test_macos_mlx.py
uv run pytest test_macos_mlx.py             # hardware-free Mac integration tests
# uv run pytest                             # full suite in a VibeVoice environment
python -m compileall -q runner scripts main.py
bash -n scripts/bootstrap_uv.sh scripts/bootstrap_macos.sh
```

A real synthesis smoke test must run on Apple Silicon because Linux cannot execute MLX/Metal or Speech Swift.

## Docker

The Dockerfile remains CUDA-specific and installs the `vibevoice` dependency extra. MLX is an Apple framework and is not supported inside the NVIDIA Linux image.

```bash
docker build -t vibevoice-realtime .
docker run --gpus all -p 8000:8000 vibevoice-realtime
```

## License and upstream projects

Review the licenses and model cards for the software and model weights you use:

- Microsoft VibeVoice
- Qwen3-TTS
- Speech Swift
- MLX / `mlx-audio`
- The selected Hugging Face model repository
