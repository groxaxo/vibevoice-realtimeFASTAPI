"""Apple-Silicon backends for Qwen3-TTS MLX model bundles.

The two registered bundles intentionally use different loaders:

* ``aufklarer/...-MLX-8bit`` is a Speech Swift/minmax bundle and is invoked
  through the ``speech`` CLI that officially supports a full Hugging Face ID.
* ``mlx-community/...-Base-4bit`` is an mlx-audio/affine bundle and is loaded
  in-process with ``mlx_audio.tts.utils.load_model``.

Keeping the adapters separate prevents silent tensor/config incompatibilities.
"""

from __future__ import annotations

import importlib.util
import io
import os
import platform
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy.io.wavfile

from runner.adapters.base import EngineAdapter
from runner.errors import (
    BackendUnavailableError,
    CapabilityError,
    InvalidRequestForModelError,
)
from runner.model_registry import ModelProfile
from runner.types import SpeechRequest

_OPENAI_DEFAULT_VOICES = {
    "alloy",
    "ash",
    "ballad",
    "cedar",
    "coral",
    "default",
    "echo",
    "fable",
    "juniper",
    "marin",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
}


def is_apple_silicon(
    system: str | None = None,
    machine: str | None = None,
) -> bool:
    """Return whether the current process is native Apple Silicon macOS."""
    resolved_system = (system or platform.system()).lower()
    resolved_machine = (machine or platform.machine()).lower()
    return resolved_system == "darwin" and resolved_machine in {"arm64", "aarch64"}


def _resolve_reference_audio(request: SpeechRequest) -> str | None:
    """Resolve explicit ``ref_audio`` or a file-valued OpenAI ``voice``."""
    raw = request.ref_audio
    if raw is None and request.voice:
        candidate = Path(request.voice).expanduser()
        if candidate.is_file():
            raw = str(candidate)
        elif request.voice.lower() not in _OPENAI_DEFAULT_VOICES:
            raise InvalidRequestForModelError(
                "This Qwen3-TTS Base profile has no named speaker presets. "
                "Use voice='default', omit voice, or pass a local reference-audio "
                "path in 'ref_audio' (or 'voice')."
            )

    if raw is None:
        return None

    path = Path(raw).expanduser()
    if not path.is_file():
        raise InvalidRequestForModelError(
            f"Reference audio does not exist or is not a file: {path}"
        )
    return str(path.resolve())


def _common_capabilities(profile: ModelProfile, available: bool, backend: str) -> dict[str, Any]:
    return {
        "model": profile.key,
        "family": profile.family,
        "backend": backend,
        "hf_model_id": profile.hf_model_id,
        "quantization": profile.quantization,
        "platforms": list(profile.platforms),
        "sample_rate": profile.sample_rate,
        "supports_stream": profile.supports_stream,
        "supports_multispeaker": profile.supports_multispeaker,
        "supports_voice_list": profile.supports_voice_list,
        "supports_reference_audio": profile.supports_reference_audio,
        "status": "available" if available else "backend_unavailable",
    }


class Qwen3SpeechSwiftAdapter(EngineAdapter):
    """Serve the exact aufklarer 8-bit minmax bundle through Speech Swift."""

    def __init__(self, profile: ModelProfile, **kwargs: Any) -> None:
        super().__init__(profile, **kwargs)
        self._system = kwargs.get("platform_system")
        self._machine = kwargs.get("platform_machine")
        self._run_command: Callable[..., Any] = kwargs.get("run_command", subprocess.run)
        self._lock = threading.Lock()

        model_override = kwargs.get("model_path") or os.environ.get("QWEN3_TTS_MODEL_ID")
        self.model_id = str(model_override) if model_override else profile.hf_model_id

        configured_binary = kwargs.get("speech_bin") or os.environ.get("SPEECH_SWIFT_BIN")
        self.binary = self._find_binary(configured_binary)
        self._backend_error: str | None = None

    @staticmethod
    def _find_binary(configured: str | os.PathLike[str] | None) -> str | None:
        if configured:
            configured_str = str(configured)
            found = shutil.which(configured_str)
            if found:
                return found
            path = Path(configured_str).expanduser()
            if path.is_file() and os.access(path, os.X_OK):
                return str(path.resolve())
            return None
        return shutil.which("speech") or shutil.which("audio")

    def _availability_error(self) -> str | None:
        if not is_apple_silicon(self._system, self._machine):
            return "Speech Swift requires native Apple Silicon macOS (arm64), not Rosetta/x86_64."
        if self.binary is None:
            return (
                "Speech Swift was not found. Install native ARM Homebrew and run "
                "'brew install speech', or set SPEECH_SWIFT_BIN."
            )
        return None

    def is_available(self) -> bool:
        self._backend_error = self._availability_error()
        return self._backend_error is None

    def capabilities(self) -> dict[str, Any]:
        available = self.is_available()
        result = _common_capabilities(self.profile, available, "speech_swift_cli")
        result["model_source"] = self.model_id
        if self.binary:
            result["binary"] = self.binary
        if self._backend_error:
            result["error"] = self._backend_error
        return result

    def list_voices(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "default",
                "name": "Default / unconditioned",
                "object": "voice",
                "category": "qwen3_base",
            },
            {
                "id": "reference_audio",
                "name": "Reference audio path",
                "object": "voice",
                "category": "voice_cloning",
            },
        ]

    def synthesize(self, request: SpeechRequest) -> tuple[bytes, str]:
        if not self.is_available():
            raise BackendUnavailableError(self.profile.key, self._backend_error)
        if not request.input:
            raise InvalidRequestForModelError("Field 'input' is required.")
        if request.speakers:
            raise InvalidRequestForModelError(
                "Qwen3-TTS Base does not accept multi-speaker 'speakers' turns."
            )
        if request.stream:
            raise CapabilityError(self.profile.key, "stream")
        if request.speed not in (None, 1.0):
            raise InvalidRequestForModelError(
                "Speech Swift's Qwen3 backend does not expose speed control; use speed=1.0."
            )
        if request.instruct:
            raise InvalidRequestForModelError(
                "The selected Base model does not support 'instruct'; use a CustomVoice model."
            )
        if request.ref_text:
            raise InvalidRequestForModelError(
                "The Speech Swift 8-bit CLI does not expose reference transcripts. "
                "Use the 4-bit mlx-audio profile for ref_audio + ref_text ICL cloning."
            )
        if request.top_p is not None or request.repetition_penalty is not None:
            raise InvalidRequestForModelError(
                "The Speech Swift 8-bit CLI exposes temperature, top_k, and max_tokens, "
                "but not top_p or repetition_penalty."
            )

        ref_audio = _resolve_reference_audio(request)
        language = request.language or "english"
        temperature = request.temp if request.temp is not None else 0.3
        top_k = request.top_k if request.top_k is not None else 50
        max_tokens = request.max_tokens if request.max_tokens is not None else 500

        assert self.binary is not None
        with tempfile.TemporaryDirectory(prefix="qwen3-tts-") as tmpdir:
            output_path = Path(tmpdir) / "speech.wav"
            cmd = [
                self.binary,
                "speak",
                request.input,
                "--engine",
                "qwen3",
                "--output",
                str(output_path),
                "--language",
                language,
                "--model",
                self.model_id,
                "--temperature",
                str(temperature),
                "--top-k",
                str(top_k),
                "--max-tokens",
                str(max_tokens),
            ]
            if ref_audio:
                cmd.extend(["--voice-sample", ref_audio])

            timeout = float(os.environ.get("QWEN3_TTS_TIMEOUT_SECONDS", "1800"))
            env = os.environ.copy()
            env.setdefault("NO_COLOR", "1")

            try:
                # Serialise calls to avoid loading multiple 1.7B models into unified
                # memory when the HTTP server is hit concurrently.
                with self._lock:
                    completed = self._run_command(
                        cmd,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        env=env,
                    )
            except subprocess.TimeoutExpired as exc:
                raise BackendUnavailableError(
                    self.profile.key,
                    f"Speech Swift synthesis exceeded {timeout:g} seconds.",
                ) from exc
            except OSError as exc:
                raise BackendUnavailableError(self.profile.key, str(exc)) from exc

            return_code = int(getattr(completed, "returncode", 1))
            if return_code != 0:
                stderr = str(getattr(completed, "stderr", "")).strip()
                stdout = str(getattr(completed, "stdout", "")).strip()
                detail = stderr or stdout or f"speech exited with status {return_code}"
                detail = detail[-2000:]
                raise BackendUnavailableError(self.profile.key, detail)
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise BackendUnavailableError(
                    self.profile.key,
                    "Speech Swift completed without producing a WAV file.",
                )
            return output_path.read_bytes(), "audio/wav"

    def stream(self, request: SpeechRequest) -> Any:
        raise CapabilityError(self.profile.key, "stream")

    def health(self) -> dict[str, Any]:
        available = self.is_available()
        result = {
            "adapter": "qwen3_speech_swift",
            "model": self.profile.key,
            "model_source": self.model_id,
            "available": available,
            "loaded": False,
            "platform": f"{platform.system()}-{platform.machine()}",
        }
        if self.binary:
            result["binary"] = self.binary
        if self._backend_error:
            result["error"] = self._backend_error
        return result


class Qwen3MLXAudioAdapter(EngineAdapter):
    """Serve mlx-community Qwen3-TTS bundles in-process with mlx-audio."""

    def __init__(self, profile: ModelProfile, **kwargs: Any) -> None:
        super().__init__(profile, **kwargs)
        self._system = kwargs.get("platform_system")
        self._machine = kwargs.get("platform_machine")
        self._load_model_fn: Callable[[Any], Any] | None = kwargs.get("load_model_fn")
        self._load_audio_fn: Callable[..., Any] | None = kwargs.get("load_audio_fn")
        self._model: Any | None = kwargs.get("model")
        self._runtime_loaded = self._model is not None
        self._backend_error: str | None = None
        self._lock = threading.Lock()
        self.model_source = self._resolve_model_source(kwargs.get("model_path"))

    def _resolve_model_source(self, override: Any | None) -> str:
        if override is not None:
            raw = str(override)
            path = Path(raw).expanduser()
            return str(path.resolve()) if path.exists() else raw

        local = Path(self.profile.default_local_dir).expanduser()
        if local.is_dir() and any(local.iterdir()):
            return str(local.resolve())
        # mlx-audio accepts either a local path or a Hugging Face repository ID,
        # so a fresh Mac can download into its normal HF cache on first load.
        return self.profile.hf_model_id

    def _availability_error(self) -> str | None:
        if not is_apple_silicon(self._system, self._machine):
            return "mlx-audio requires native Apple Silicon macOS (arm64)."
        if self._load_model_fn is not None or self._model is not None:
            return None
        if importlib.util.find_spec("mlx_audio") is None:
            return "mlx-audio is not installed. Run 'uv sync --extra mac'."
        if importlib.util.find_spec("mlx") is None:
            return "The MLX runtime is not installed. Run 'uv sync --extra mac'."
        return None

    def is_available(self) -> bool:
        self._backend_error = self._availability_error()
        return self._backend_error is None

    def _ensure_runtime_loaded(self) -> None:
        if self._runtime_loaded:
            return
        if not self.is_available():
            raise BackendUnavailableError(self.profile.key, self._backend_error)

        # Loading a 1.7B model twice can exhaust unified memory. Re-check state
        # under the same lock used for generation before constructing it.
        with self._lock:
            if self._runtime_loaded:
                return
            try:
                loader = self._load_model_fn
                if loader is None:
                    from mlx_audio.tts.utils import load_model

                    loader = load_model
                self._model = loader(self.model_source)
                self._runtime_loaded = True
                self._backend_error = None
            except Exception as exc:
                self._backend_error = str(exc)
                raise BackendUnavailableError(self.profile.key, self._backend_error) from exc

    def _load_reference_audio(self, path: str | None) -> Any | None:
        """Load and resample a reference file to the model's native sample rate."""
        if path is None:
            return None

        loader = self._load_audio_fn
        if loader is None:
            # mlx-audio's Qwen generate() API expects an MLX waveform, not a
            # filesystem path. Import lazily so registry/API inspection remains
            # usable on non-macOS hosts without MLX installed.
            from mlx_audio.tts.generate import load_audio

            loader = load_audio

        try:
            return loader(
                path,
                sample_rate=self.profile.sample_rate or 24_000,
            )
        except Exception as exc:
            raise InvalidRequestForModelError(
                f"Failed to load reference audio '{path}': {exc}"
            ) from exc

    def capabilities(self) -> dict[str, Any]:
        available = self.is_available()
        result = _common_capabilities(self.profile, available, "mlx_audio_native")
        result["model_source"] = self.model_source
        result["loaded"] = self._runtime_loaded
        if self._backend_error:
            result["error"] = self._backend_error
        return result

    def list_voices(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "default",
                "name": "Default / unconditioned",
                "object": "voice",
                "category": "qwen3_base",
            },
            {
                "id": "reference_audio",
                "name": "Reference audio path",
                "object": "voice",
                "category": "voice_cloning",
            },
        ]

    def synthesize(self, request: SpeechRequest) -> tuple[bytes, str]:
        if not request.input:
            raise InvalidRequestForModelError("Field 'input' is required.")
        if request.speakers:
            raise InvalidRequestForModelError(
                "Qwen3-TTS Base does not accept multi-speaker 'speakers' turns."
            )
        if request.stream:
            raise CapabilityError(self.profile.key, "stream")
        if request.speed not in (None, 1.0):
            raise InvalidRequestForModelError(
                "mlx-audio currently does not implement Qwen3-TTS speed control; use speed=1.0."
            )
        if request.instruct:
            raise InvalidRequestForModelError(
                "The selected Base model does not support 'instruct'; use a CustomVoice model."
            )

        ref_audio_path = _resolve_reference_audio(request)
        if request.ref_text and ref_audio_path is None:
            raise InvalidRequestForModelError(
                "Field 'ref_text' requires an actual reference-audio file, not voice='default'."
            )
        self._ensure_runtime_loaded()
        ref_audio = self._load_reference_audio(ref_audio_path)
        assert self._model is not None

        kwargs: dict[str, Any] = {
            "text": request.input,
            "voice": None,
            "temperature": request.temp if request.temp is not None else 0.9,
            "speed": 1.0,
            "lang_code": request.language or "auto",
            "ref_audio": ref_audio,
            "ref_text": request.ref_text,
            "max_tokens": request.max_tokens if request.max_tokens is not None else 4096,
            "verbose": False,
            "stream": False,
            "top_k": request.top_k if request.top_k is not None else 50,
            "top_p": request.top_p if request.top_p is not None else 1.0,
            "repetition_penalty": (
                request.repetition_penalty
                if request.repetition_penalty is not None
                else 1.05
            ),
        }

        try:
            audio_parts: list[np.ndarray] = []
            sample_rate = self.profile.sample_rate or 24000
            with self._lock:
                for result in self._model.generate(**kwargs):
                    part = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                    if part.size:
                        audio_parts.append(part.copy())
                    sample_rate = int(getattr(result, "sample_rate", sample_rate))
        except InvalidRequestForModelError:
            raise
        except Exception as exc:
            self._backend_error = str(exc)
            raise BackendUnavailableError(self.profile.key, self._backend_error) from exc

        if not audio_parts:
            raise BackendUnavailableError(
                self.profile.key,
                "mlx-audio returned no audio samples.",
            )

        audio = np.concatenate(audio_parts).astype(np.float32, copy=False)
        buffer = io.BytesIO()
        scipy.io.wavfile.write(buffer, sample_rate, audio)
        return buffer.getvalue(), "audio/wav"

    def stream(self, request: SpeechRequest) -> Any:
        raise CapabilityError(self.profile.key, "stream")

    def health(self) -> dict[str, Any]:
        available = self.is_available()
        result = {
            "adapter": "qwen3_mlx_audio",
            "model": self.profile.key,
            "model_source": self.model_source,
            "available": available,
            "loaded": self._runtime_loaded,
            "platform": f"{platform.system()}-{platform.machine()}",
        }
        if self._backend_error:
            result["error"] = self._backend_error
        return result
