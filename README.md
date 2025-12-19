# 🎙️ VibeVoice Realtime Runner

<div align="center">

![VibeVoice](https://img.shields.io/badge/VibeVoice-Realtime-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11-yellow?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green?style=for-the-badge&logo=fastapi)
![OpenAI API](https://img.shields.io/badge/OpenAI_API-Compatible-orange?style=for-the-badge&logo=openai)

**A high-performance local runner for Microsoft's VibeVoice Realtime text-to-speech model.**
*Now with OpenAI-compatible API endpoints!*

[Features](#features) • [Quick Start](#quick-start) • [API Documentation](#api-documentation) • [Credits](#credits)

</div>

---

## 🚀 Features

- **Local & Private**: Runs entirely on your machine (CUDA/MPS/CPU).
- **Realtime Streaming**: Low-latency text-to-speech generation.
- **OpenAI API Compatible**: Drop-in replacement for OpenAI's TTS API.
- **Web Interface**: Built-in interactive demo UI.
- **Multi-Platform**: Optimized for Ubuntu (CUDA) and macOS (Apple Silicon).
- **Easy Setup**: Powered by `uv` for fast, reliable dependency management.

## ⚡ Quick Start

### Prerequisites

- **uv** installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Git**
- **Hugging Face Account** (for model download)

### Installation

1.  **Bootstrap the environment**:
    ```bash
    ./scripts/bootstrap_uv.sh
    ```

2.  **Download the model**:
    ```bash
    uv run python scripts/download_model.py
    ```

3.  **Run the server**:
    ```bash
    uv run python scripts/run_realtime_demo.py --port 8000
    ```

    - **Web UI**: Open [http://127.0.0.1:8000](http://127.0.0.1:8000)
    - **API**: `http://127.0.0.1:8000/v1/audio/speech`

## 📖 API Documentation

This runner provides OpenAI-compatible endpoints for easy integration with existing tools and libraries.

### 🗣️ Speech Generation

**Endpoint**: `POST /v1/audio/speech`

Generates audio from text.

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello, this is VibeVoice running locally!",
    "voice": "en-Carter_man",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `model` | `string` | Model identifier (e.g., `tts-1`). Ignored but required for compatibility. |
| `input` | `string` | The text to generate audio for. |
| `voice` | `string` | The voice ID to use (see `/v1/audio/voices`). |
| `response_format` | `string` | Output format: `wav` (default) or `mp3`. |
| `speed` | `float` | Speed of generation (currently ignored). |

### 🎤 List Voices

**Endpoint**: `GET /v1/audio/voices`

Returns a list of available voices.

```bash
curl http://127.0.0.1:8000/v1/audio/voices
```

**Response:**
```json
{
  "voices": [
    {
      "id": "en-Carter_man",
      "name": "en-Carter_man",
      "object": "voice",
      "category": "vibe_voice",
      ...
    },
    ...
  ]
}
```

## ⚙️ Configuration

### Device Selection

The runner automatically detects the best available device:
- **CUDA**: NVIDIA GPUs (Linux)
- **MPS**: Apple Silicon (macOS)
- **CPU**: Fallback

To force a specific device:
```bash
uv run python scripts/run_realtime_demo.py --device cpu
```

### Inference Steps

Specify the number of DDPM inference steps. Higher values (e.g., 15-20) improve quality but increase latency. The default is **15**.

```bash
uv run python scripts/run_realtime_demo.py --inference-steps 15
```

### Custom Model Path

```bash
uv run python scripts/run_realtime_demo.py --model-path /path/to/model
```

## 🎧 Demos

All examples generated using **15 inference steps** with text in the voice's native language.

### English
| Voice | Audio Example (MP3) |
| :--- | :--- |
| **en-Carter_man** | <audio src="docs/demos/en-Carter_man.mp3" controls preload="none"></audio> |
| **en-Davis_man** | <audio src="docs/demos/en-Davis_man.mp3" controls preload="none"></audio> |
| **en-Emma_woman** | <audio src="docs/demos/en-Emma_woman.mp3" controls preload="none"></audio> |
| **en-Frank_man** | <audio src="docs/demos/en-Frank_man.mp3" controls preload="none"></audio> |
| **en-Grace_woman** | <audio src="docs/demos/en-Grace_woman.mp3" controls preload="none"></audio> |
| **en-Mike_man** | <audio src="docs/demos/en-Mike_man.mp3" controls preload="none"></audio> |
| **in-Samuel_man** | <audio src="docs/demos/in-Samuel_man.mp3" controls preload="none"></audio> |

### Other Languages
| Language | Voice | Audio Example (MP3) |
| :--- | :--- | :--- |
| **German** | de-Spk0_man | <audio src="docs/demos/de-Spk0_man.mp3" controls preload="none"></audio> |
| **German** | de-Spk1_woman | <audio src="docs/demos/de-Spk1_woman.mp3" controls preload="none"></audio> |
| **Spanish** | sp-Spk0_woman | <audio src="docs/demos/sp-Spk0_woman.mp3" controls preload="none"></audio> |
| **Spanish** | sp-Spk1_man | <audio src="docs/demos/sp-Spk1_man.mp3" controls preload="none"></audio> |
| **French** | fr-Spk0_man | <audio src="docs/demos/fr-Spk0_man.mp3" controls preload="none"></audio> |
| **French** | fr-Spk1_woman | <audio src="docs/demos/fr-Spk1_woman.mp3" controls preload="none"></audio> |
| **Italian** | it-Spk0_woman | <audio src="docs/demos/it-Spk0_woman.mp3" controls preload="none"></audio> |
| **Italian** | it-Spk1_man | <audio src="docs/demos/it-Spk1_man.mp3" controls preload="none"></audio> |
| **Japanese** | jp-Spk0_man | <audio src="docs/demos/jp-Spk0_man.mp3" controls preload="none"></audio> |
| **Japanese** | jp-Spk1_woman | <audio src="docs/demos/jp-Spk1_woman.mp3" controls preload="none"></audio> |
| **Korean** | kr-Spk0_woman | <audio src="docs/demos/kr-Spk0_woman.mp3" controls preload="none"></audio> |
| **Korean** | kr-Spk1_man | <audio src="docs/demos/kr-Spk1_man.mp3" controls preload="none"></audio> |
| **Dutch** | nl-Spk0_man | <audio src="docs/demos/nl-Spk0_man.mp3" controls preload="none"></audio> |
| **Dutch** | nl-Spk1_woman | <audio src="docs/demos/nl-Spk1_woman.mp3" controls preload="none"></audio> |
| **Polish** | pl-Spk0_man | <audio src="docs/demos/pl-Spk0_man.mp3" controls preload="none"></audio> |
| **Polish** | pl-Spk1_woman | <audio src="docs/demos/pl-Spk1_woman.mp3" controls preload="none"></audio> |
| **Portuguese** | pt-Spk0_woman | <audio src="docs/demos/pt-Spk0_woman.mp3" controls preload="none"></audio> |
| **Portuguese** | pt-Spk1_man | <audio src="docs/demos/pt-Spk1_man.mp3" controls preload="none"></audio> |


## 🏆 Credits & Acknowledgements

This project stands on the shoulders of giants. Huge thanks to:

- **[Microsoft](https://github.com/microsoft/VibeVoice)**: For releasing the incredible **VibeVoice** model and the original codebase.
- **[groxaxo](https://github.com/groxaxo)**: For the original repository and initial setup.
- **[Kokoro FastAPI Creators](https://github.com/remsky/Kokoro-FastAPI)**: For inspiration on the FastAPI implementation and structure.
- **Open Source Community**: For all the tools and libraries that make this possible.

---

<div align="center">
Made with ❤️ for the AI Community
</div>
