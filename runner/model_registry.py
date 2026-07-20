"""Model registry – canonical profiles, aliases, and lookup helpers.

The vendored realtime VibeVoice application predates native adapter serving and
imports these helpers directly.  Its default registry view therefore remains
VibeVoice-only.  New launchers opt into ``include_native=True`` so Apple MLX
profiles cannot be advertised by, or accidentally routed through, the old
realtime service.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Literal

from runner.errors import UnknownModelError

ModelFamily = Literal["realtime", "longform", "qwen3_tts"]
LoaderMode = Literal[
    "subprocess_demo",
    "native_longform",
    "speech_swift_cli",
    "mlx_audio_native",
]


@dataclass(frozen=True)
class ModelProfile:
    """Describe one model and the backend required to serve it."""

    key: str
    hf_model_id: str
    default_local_dir: str
    family: ModelFamily
    loader_mode: LoaderMode
    supports_stream: bool
    supports_multispeaker: bool
    supports_voice_list: bool
    supports_reference_audio: bool = False
    sample_rate: int | None = None
    quantization: str | None = None
    platforms: tuple[str, ...] = ("linux", "darwin", "windows")
    description: str = ""

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        value = asdict(self)
        value["platforms"] = list(self.platforms)
        return value


# Profiles visible to the copied realtime demo.  Keep this surface stable so
# the demo cannot claim it loaded an adapter that it never calls.
_PROFILES: dict[str, ModelProfile] = {
    "realtime-0.5b": ModelProfile(
        key="realtime-0.5b",
        hf_model_id="microsoft/VibeVoice-Realtime-0.5B",
        default_local_dir="models/VibeVoice-Realtime-0.5B",
        family="realtime",
        loader_mode="subprocess_demo",
        supports_stream=True,
        supports_multispeaker=False,
        supports_voice_list=True,
        sample_rate=24000,
        description="VibeVoice realtime PyTorch demo backend.",
    ),
    "tts-1.5b": ModelProfile(
        key="tts-1.5b",
        hf_model_id="microsoft/VibeVoice-1.5B",
        default_local_dir="models/VibeVoice-1.5B",
        family="longform",
        loader_mode="native_longform",
        supports_stream=False,
        supports_multispeaker=True,
        supports_voice_list=False,
        supports_reference_audio=True,
        sample_rate=24000,
        description="VibeVoice 1.5B native long-form backend.",
    ),
    "tts-7b": ModelProfile(
        key="tts-7b",
        hf_model_id=os.environ.get("VIBEVOICE_7B_MODEL_ID", "vibevoice/VibeVoice-7B"),
        default_local_dir="models/VibeVoice-7B",
        family="longform",
        loader_mode="native_longform",
        supports_stream=False,
        supports_multispeaker=True,
        supports_voice_list=False,
        supports_reference_audio=True,
        sample_rate=24000,
        description="VibeVoice 7B native long-form backend.",
    ),
}

# Native adapter profiles are opt-in for registry lookups.  This isolates them
# from overrides/app.py, whose implementation always calls the realtime engine.
_NATIVE_PROFILES: dict[str, ModelProfile] = {
    "qwen3-tts-mlx-8bit": ModelProfile(
        key="qwen3-tts-mlx-8bit",
        hf_model_id="aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-8bit",
        default_local_dir="models/Qwen3-TTS-12Hz-1.7B-Base-MLX-8bit",
        family="qwen3_tts",
        loader_mode="speech_swift_cli",
        supports_stream=False,
        supports_multispeaker=False,
        supports_voice_list=True,
        supports_reference_audio=True,
        sample_rate=24000,
        quantization="8bit-minmax",
        platforms=("darwin-arm64",),
        description=(
            "Exact aufklarer 1.7B 8-bit MLX bundle served through Speech Swift "
            "on Apple Silicon."
        ),
    ),
    "qwen3-tts-mlx-4bit": ModelProfile(
        key="qwen3-tts-mlx-4bit",
        hf_model_id="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit",
        default_local_dir="models/Qwen3-TTS-12Hz-1.7B-Base-4bit",
        family="qwen3_tts",
        loader_mode="mlx_audio_native",
        supports_stream=False,
        supports_multispeaker=False,
        supports_voice_list=True,
        supports_reference_audio=True,
        sample_rate=24000,
        quantization="4bit-affine",
        platforms=("darwin-arm64",),
        description="Qwen3-TTS 1.7B Base 4-bit MLX bundle served with mlx-audio.",
    ),
}

_ALIASES: dict[str, str] = {
    "tts-1": "realtime-0.5b",
    "tts-1-hd": "realtime-0.5b",
    "vibevoice-realtime-0.5b": "realtime-0.5b",
    "vibevoice-1.5b": "tts-1.5b",
    "vibevoice-7b": "tts-7b",
}

_NATIVE_ALIASES: dict[str, str] = {
    "qwen3-tts-8bit": "qwen3-tts-mlx-8bit",
    "qwen3-tts-1.7b-8bit": "qwen3-tts-mlx-8bit",
    "qwen3-mlx-8bit": "qwen3-tts-mlx-8bit",
    "aufklarer/qwen3-tts-12hz-1.7b-base-mlx-8bit": "qwen3-tts-mlx-8bit",
    "qwen3-tts-4bit": "qwen3-tts-mlx-4bit",
    "qwen3-tts-1.7b-4bit": "qwen3-tts-mlx-4bit",
    "qwen3-mlx-4bit": "qwen3-tts-mlx-4bit",
    "mlx-community/qwen3-tts-12hz-1.7b-base-4bit": "qwen3-tts-mlx-4bit",
}

DEFAULT_MODEL_KEY = "realtime-0.5b"


def _profiles(include_native: bool) -> dict[str, ModelProfile]:
    if not include_native:
        return _PROFILES
    return {**_PROFILES, **_NATIVE_PROFILES}


def _aliases(include_native: bool) -> dict[str, str]:
    if not include_native:
        return _ALIASES
    return {**_ALIASES, **_NATIVE_ALIASES}


def list_model_keys(*, include_native: bool = False) -> list[str]:
    """Return canonical model keys for the requested registry surface."""
    return list(_profiles(include_native).keys())


def list_profiles(*, include_native: bool = False) -> list[ModelProfile]:
    """Return model profiles in registry order."""
    return list(_profiles(include_native).values())


def list_aliases(*, include_native: bool = False) -> dict[str, str]:
    """Return a copy of aliases for the requested registry surface."""
    return dict(_aliases(include_native))


def aliases_for_model(model_key: str, *, include_native: bool = True) -> list[str]:
    """Return aliases that resolve to *model_key*."""
    return sorted(
        alias for alias, target in _aliases(include_native).items() if target == model_key
    )


def resolve_model_key(requested: str | None, *, include_native: bool = False) -> str:
    """Resolve a model name, alias, or ``None`` to a canonical key.

    Native adapter profiles are excluded by default to preserve the contract of
    the copied realtime application.  The generic launcher and native API pass
    ``include_native=True``.
    """
    if requested is None or requested.strip() == "":
        return DEFAULT_MODEL_KEY

    normalised = requested.strip().lower()
    profiles = _profiles(include_native)
    aliases = _aliases(include_native)
    if normalised in profiles:
        return normalised
    if normalised in aliases:
        return aliases[normalised]
    raise UnknownModelError(normalised)


def get_model_profile(model_key: str, *, include_native: bool = False) -> ModelProfile:
    """Return a profile from the requested registry surface."""
    try:
        return _profiles(include_native)[model_key]
    except KeyError:
        raise UnknownModelError(model_key) from None
