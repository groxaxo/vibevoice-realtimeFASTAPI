# 🎙️ VibeVoice Realtime Runner

<div align="center">

![VibeVoice](https://img.shields.io/badge/VibeVoice-Realtime-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11-yellow?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green?style=for-the-badge&logo=fastapi)
![OpenAI API](https://img.shields.io/badge/OpenAI_API-Compatible-orange?style=for-the-badge&logo=openai)
![Privacy](https://img.shields.io/badge/Privacy-100%25_Local-red?style=for-the-badge&logo=shield)
![Open-WebUI](https://img.shields.io/badge/Open--WebUI-Compatible-purple?style=for-the-badge)

**Run enterprise-grade text-to-speech completely offline on your own hardware.**  
*Privacy-first • Zero cloud costs • OpenAI API compatible • Open-WebUI ready*

Transform text into natural speech without sending data to the cloud. Perfect for privacy-conscious developers, local AI enthusiasts, and anyone who wants full control over their TTS pipeline.

[Features](#features) • [Why Local?](#-why-local-tts) • [Quick Start](#quick-start) • [Open-WebUI](#-open-webui-integration) • [API Docs](#api-documentation)

</div>

---

## 🚀 Features

### 🔒 Privacy & Control
- **100% Local**: All processing happens on your machine - no data leaves your system
- **No API Keys Required**: Free from cloud service limitations and costs
- **Offline Capable**: Works without internet after initial setup

### 🎯 Easy Integration
- **OpenAI API Compatible**: Drop-in replacement for OpenAI's TTS API - just change the endpoint URL
- **Open-WebUI Ready**: Plug and play with Open-WebUI for seamless integration
- **RESTful API**: Standard HTTP endpoints for easy integration with any language or framework

### ⚡ Performance & Quality
- **Realtime Streaming**: Low-latency text-to-speech generation
- **Hardware Accelerated**: Optimized for CUDA (NVIDIA), MPS (Apple Silicon), and CPU
- **Enterprise Quality**: Powered by Microsoft's VibeVoice model

### 🛠️ Developer Experience
- **Modern Setup**: Powered by `uv` - the fastest Python package manager
- **Web Interface**: Built-in interactive demo UI for testing
- **Multi-Platform**: Works on Ubuntu (CUDA) and macOS (Apple Silicon)

## 💡 Why Local TTS?

Running text-to-speech locally offers significant advantages over cloud services:

| Aspect | Local (VibeVoice) | Cloud TTS Services |
|--------|-------------------|-------------------|
| **Privacy** | ✅ Data never leaves your machine | ❌ Data sent to third-party servers |
| **Cost** | ✅ Free after setup (your hardware) | ❌ Pay per character/request |
| **Latency** | ✅ Direct local processing | ⚠️ Network + API overhead |
| **Offline** | ✅ Works without internet | ❌ Requires internet connection |
| **API Limits** | ✅ No rate limits | ❌ Subject to rate limiting |
| **Data Compliance** | ✅ Full control for GDPR/HIPAA | ⚠️ Depends on provider |

**Perfect for:**
- 🏥 Healthcare and legal applications requiring HIPAA/GDPR compliance
- 🎮 Gaming and interactive applications needing low latency
- 🤖 AI assistants and chatbots with privacy concerns
- 📚 Content creation and audiobook generation at scale
- 🔬 Research and development without API costs

## ⚡ Quick Start

Get up and running in under 5 minutes! No complex configuration needed.

### Prerequisites

- **uv** installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Git**
- **Hugging Face Account** (for one-time model download)

### Installation

1.  **Clone and bootstrap the environment**:
    ```bash
    git clone https://github.com/groxaxo/vibevoice-realtimeFASTAPI.git
    cd vibevoice-realtimeFASTAPI
    ./scripts/bootstrap_uv.sh
    ```

2.  **Download the model** (one-time, ~2GB):
    ```bash
    uv run python scripts/download_model.py
    ```

3.  **Run the server**:
    ```bash
    uv run python scripts/run_realtime_demo.py --port 8000
    ```

    ✨ **That's it!** You now have a local TTS server running!

    - **Web UI**: Open [http://127.0.0.1:8000](http://127.0.0.1:8000) to try the interactive demo
    - **API**: `http://127.0.0.1:8000/v1/audio/speech` - OpenAI compatible endpoint

### First API Call

Test your local TTS server:
```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello! I am running locally on your machine!",
    "voice": "en-Carter_man",
    "response_format": "mp3"
  }' \
  --output hello.mp3 && open hello.mp3
```

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

## 🌐 Open-WebUI Integration

VibeVoice Runner is fully compatible with [Open-WebUI](https://github.com/open-webui/open-webui), making it a perfect local TTS backend.

### Setup Steps

1. **Start VibeVoice Runner**:
   ```bash
   uv run python scripts/run_realtime_demo.py --port 8000
   ```

2. **Configure Open-WebUI**:
   - Go to **Settings** → **Audio**
   - Set **TTS Engine** to **OpenAI**
   - Set **API Base URL** to `http://127.0.0.1:8000/v1`
   - Leave **API Key** empty (not required for local)
   - Select a voice from the dropdown

3. **Enjoy Private TTS**: Your Open-WebUI instance now uses local TTS without sending data to external services!

### Benefits with Open-WebUI
- ✅ **Complete Privacy**: All AI conversations and TTS stay on your machine
- ✅ **No Additional Costs**: Free TTS alongside your local LLM
- ✅ **Seamless Experience**: Works exactly like cloud TTS but faster and private
- ✅ **Offline Capable**: Run your entire AI stack without internet

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

### Custom Model Path

```bash
uv run python scripts/run_realtime_demo.py --model-path /path/to/model
```

## 💬 FAQ

**Q: Do I need an internet connection to use this?**  
A: Only for the initial setup and model download (~2GB). After that, it works completely offline.

**Q: How much does this cost to run?**  
A: Zero recurring costs! You only use your own hardware. Cloud TTS services charge per character, which can add up quickly.

**Q: Is the quality as good as cloud TTS services?**  
A: Yes! VibeVoice is Microsoft's enterprise-grade model, offering quality comparable to leading cloud services.

**Q: Can I use this commercially?**  
A: Check Microsoft's VibeVoice license for commercial usage terms.

**Q: What hardware do I need?**  
A: Works on CPU, but GPU (NVIDIA CUDA or Apple Silicon) is recommended for real-time performance.

**Q: Can I add custom voices?**  
A: Currently, you can use the voices provided by the VibeVoice model. Check the model documentation for available voices.

## 🏆 Credits & Acknowledgements

This project stands on the shoulders of giants. Huge thanks to:

- **[Microsoft](https://github.com/microsoft/VibeVoice)**: For releasing the incredible **VibeVoice** model and the original codebase.
- **[groxaxo](https://github.com/groxaxo)**: For the original repository and initial setup.
- **[Kokoro FastAPI Creators](https://github.com/remsky/Kokoro-FastAPI)**: For inspiration on the FastAPI implementation and structure.
- **Open Source Community**: For all the tools and libraries that make this possible.

---

<div align="center">

**Made with ❤️ for Privacy-Conscious Developers and Local AI Enthusiasts**

⭐ Star us on GitHub if you believe in local-first AI! ⭐

</div>
