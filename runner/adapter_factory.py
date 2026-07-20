"""Factory – instantiate the correct adapter for a model profile.

Backend imports are intentionally lazy.  A Qwen3/MLX-only installation on a Mac
must not import PyTorch or the vendored VibeVoice package merely to start the
server, and a CUDA installation must not require Apple's MLX runtime.
"""

from __future__ import annotations

from typing import Any

from runner.adapters.base import EngineAdapter
from runner.errors import UnknownModelError
from runner.model_registry import ModelProfile, get_model_profile


def make_adapter(model_key: str, **kwargs: Any) -> EngineAdapter:
    """Create an :class:`EngineAdapter` for canonical *model_key*."""
    profile: ModelProfile = get_model_profile(model_key, include_native=True)

    if profile.loader_mode == "subprocess_demo":
        from runner.adapters.realtime_demo import RealtimeDemoAdapter

        return RealtimeDemoAdapter(profile, **kwargs)
    if profile.loader_mode == "native_longform":
        from runner.adapters.longform_native import LongformNativeAdapter

        return LongformNativeAdapter(profile, **kwargs)
    if profile.loader_mode == "speech_swift_cli":
        from runner.adapters.qwen3_mlx import Qwen3SpeechSwiftAdapter

        return Qwen3SpeechSwiftAdapter(profile, **kwargs)
    if profile.loader_mode == "mlx_audio_native":
        from runner.adapters.qwen3_mlx import Qwen3MLXAudioAdapter

        return Qwen3MLXAudioAdapter(profile, **kwargs)

    raise UnknownModelError(
        f"No adapter registered for family={profile.family!r}, "
        f"loader_mode={profile.loader_mode!r}"
    )
