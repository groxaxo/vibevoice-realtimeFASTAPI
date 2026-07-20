"""Hardware-free coverage for the Apple-Silicon Qwen3-TTS integration."""

from __future__ import annotations

import sys
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from runner.adapters.base import EngineAdapter
from runner.adapters.qwen3_mlx import (
    Qwen3MLXAudioAdapter,
    Qwen3SpeechSwiftAdapter,
    is_apple_silicon,
)
from runner.api import create_app
from runner.model_registry import (
    get_model_profile,
    list_model_keys,
    resolve_model_key,
)
from runner.errors import InvalidRequestForModelError
from runner.types import SpeechRequest, validate_for_qwen3_tts


def _write_test_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24_000)
        wav.writeframes(b"\x00\x00" * 32)


def test_native_profiles_are_isolated_from_realtime_registry() -> None:
    assert "qwen3-tts-mlx-8bit" not in list_model_keys()
    assert "qwen3-tts-mlx-8bit" in list_model_keys(include_native=True)
    assert resolve_model_key("qwen3-tts-8bit", include_native=True) == "qwen3-tts-mlx-8bit"


def test_qwen_profiles_have_exact_sources() -> None:
    eight = get_model_profile("qwen3-tts-mlx-8bit", include_native=True)
    four = get_model_profile("qwen3-tts-mlx-4bit", include_native=True)
    assert eight.hf_model_id == "aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-8bit"
    assert eight.loader_mode == "speech_swift_cli"
    assert four.hf_model_id == "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit"
    assert four.loader_mode == "mlx_audio_native"


def test_apple_silicon_detection() -> None:
    assert is_apple_silicon("Darwin", "arm64") is True
    assert is_apple_silicon("Darwin", "x86_64") is False
    assert is_apple_silicon("Linux", "aarch64") is False


def test_qwen_validation_accepts_voice_path_as_reference(tmp_path: Path) -> None:
    reference = tmp_path / "ref.wav"
    _write_test_wav(reference)
    request = SpeechRequest(
        input="hello",
        voice=str(reference),
        ref_text="reference transcript",
    )
    assert validate_for_qwen3_tts(request) == []


def test_speech_swift_adapter_builds_exact_model_command(tmp_path: Path) -> None:
    profile = get_model_profile("qwen3-tts-mlx-8bit", include_native=True)
    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        captured["command"] = command
        captured["kwargs"] = kwargs
        output_path = Path(command[command.index("--output") + 1])
        _write_test_wav(output_path)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    adapter = Qwen3SpeechSwiftAdapter(
        profile,
        platform_system="Darwin",
        platform_machine="arm64",
        speech_bin="/bin/echo",
        run_command=fake_run,
    )
    audio, mime = adapter.synthesize(
        SpeechRequest(
            input="hello from the exact eight bit model",
            response_format="wav",
            language="english",
        )
    )

    command = captured["command"]
    assert command[0] == "/bin/echo"
    assert command[:2] == ["/bin/echo", "speak"]
    assert command[command.index("--model") + 1] == profile.hf_model_id
    assert audio.startswith(b"RIFF")
    assert mime == "audio/wav"


def test_speech_swift_adapter_rejects_unsupported_icl_transcript(tmp_path: Path) -> None:
    profile = get_model_profile("qwen3-tts-mlx-8bit", include_native=True)
    reference = tmp_path / "ref.wav"
    _write_test_wav(reference)
    adapter = Qwen3SpeechSwiftAdapter(
        profile,
        platform_system="Darwin",
        platform_machine="arm64",
        speech_bin="/bin/echo",
    )
    request = SpeechRequest(
        input="hello",
        ref_audio=str(reference),
        ref_text="reference transcript",
    )
    try:
        adapter.synthesize(request)
    except InvalidRequestForModelError as exc:
        assert "4-bit mlx-audio" in str(exc)
    else:
        raise AssertionError("expected an explicit unsupported-control error")


def test_speech_swift_adapter_rejects_non_apple_platform() -> None:
    profile = get_model_profile("qwen3-tts-mlx-8bit", include_native=True)
    adapter = Qwen3SpeechSwiftAdapter(
        profile,
        platform_system="Linux",
        platform_machine="x86_64",
        speech_bin="/bin/echo",
    )
    assert adapter.is_available() is False
    assert "Apple Silicon" in adapter.health()["error"]


def test_mlx_audio_source_falls_back_to_hf_id(tmp_path: Path) -> None:
    profile = replace(
        get_model_profile("qwen3-tts-mlx-4bit", include_native=True),
        default_local_dir=str(tmp_path / "not-downloaded"),
    )
    adapter = Qwen3MLXAudioAdapter(
        profile,
        platform_system="Darwin",
        platform_machine="arm64",
        model=object(),
    )
    assert adapter.model_source == profile.hf_model_id


def test_mlx_audio_loader_is_cached() -> None:
    profile = get_model_profile("qwen3-tts-mlx-4bit", include_native=True)
    calls: list[str] = []

    class FakeModel:
        def generate(self, **_kwargs: Any):
            yield SimpleNamespace(
                audio=np.zeros(32, dtype=np.float32),
                sample_rate=24_000,
            )

    def fake_loader(source: str) -> FakeModel:
        calls.append(source)
        return FakeModel()

    adapter = Qwen3MLXAudioAdapter(
        profile,
        platform_system="Darwin",
        platform_machine="arm64",
        load_model_fn=fake_loader,
    )
    request = SpeechRequest(input="hello", response_format="wav")
    adapter.synthesize(request)
    adapter.synthesize(request)
    assert calls == [profile.hf_model_id]


def test_mlx_audio_reference_path_is_loaded_as_waveform(tmp_path: Path) -> None:
    profile = get_model_profile("qwen3-tts-mlx-4bit", include_native=True)
    reference = tmp_path / "reference.wav"
    _write_test_wav(reference)
    captured: dict[str, Any] = {}
    waveform = np.linspace(-0.2, 0.2, 64, dtype=np.float32)

    def fake_load_audio(path: str, *, sample_rate: int) -> np.ndarray:
        captured["path"] = path
        captured["sample_rate"] = sample_rate
        return waveform

    class FakeModel:
        def generate(self, **kwargs: Any):
            captured["ref_audio"] = kwargs["ref_audio"]
            captured["ref_text"] = kwargs["ref_text"]
            yield SimpleNamespace(
                audio=np.zeros(32, dtype=np.float32),
                sample_rate=24_000,
            )

    adapter = Qwen3MLXAudioAdapter(
        profile,
        platform_system="Darwin",
        platform_machine="arm64",
        model=FakeModel(),
        load_audio_fn=fake_load_audio,
    )
    adapter.synthesize(
        SpeechRequest(
            input="voice clone",
            ref_audio=str(reference),
            ref_text="reference transcript",
            response_format="wav",
        )
    )

    assert captured["path"] == str(reference.resolve())
    assert captured["sample_rate"] == 24_000
    assert captured["ref_audio"] is waveform
    assert captured["ref_text"] == "reference transcript"


def test_mlx_audio_ref_text_rejects_default_voice_without_audio() -> None:
    profile = get_model_profile("qwen3-tts-mlx-4bit", include_native=True)

    class FakeModel:
        def generate(self, **_kwargs: Any):
            raise AssertionError("generation must not start")

    adapter = Qwen3MLXAudioAdapter(
        profile,
        platform_system="Darwin",
        platform_machine="arm64",
        model=FakeModel(),
    )

    try:
        adapter.synthesize(
            SpeechRequest(
                input="hello",
                voice="default",
                ref_text="orphan transcript",
            )
        )
    except InvalidRequestForModelError as exc:
        assert "actual reference-audio file" in str(exc)
    else:
        raise AssertionError("expected ref_text without reference audio to be rejected")


def test_mlx_audio_adapter_writes_wav() -> None:
    profile = get_model_profile("qwen3-tts-mlx-4bit", include_native=True)

    class FakeModel:
        def generate(self, **kwargs: Any):
            assert kwargs["text"] == "hello"
            assert kwargs["lang_code"] == "english"
            yield SimpleNamespace(
                audio=np.linspace(-0.1, 0.1, 240, dtype=np.float32),
                sample_rate=24_000,
            )

    adapter = Qwen3MLXAudioAdapter(
        profile,
        platform_system="Darwin",
        platform_machine="arm64",
        model=FakeModel(),
    )
    audio, mime = adapter.synthesize(
        SpeechRequest(input="hello", language="english", response_format="wav")
    )
    assert audio.startswith(b"RIFF")
    assert b"WAVE" in audio[:16]
    assert mime == "audio/wav"


def test_native_api_routes_to_active_adapter(monkeypatch) -> None:
    profile = get_model_profile("qwen3-tts-mlx-4bit", include_native=True)

    class FakeAdapter(EngineAdapter):
        def is_available(self) -> bool:
            return True

        def capabilities(self) -> dict[str, Any]:
            return {"available": True, "model": self.profile.key}

        def list_voices(self) -> list[dict[str, Any]]:
            return [{"id": "default", "object": "voice"}]

        def synthesize(self, request: SpeechRequest) -> tuple[bytes, str]:
            assert request.model == profile.key
            return b"RIFFfakeWAVE", "audio/wav"

        def stream(self, request: SpeechRequest) -> Any:
            raise AssertionError("not used")

        def health(self) -> dict[str, Any]:
            return {"available": True, "loaded": True}

    monkeypatch.setattr("runner.api.make_adapter", lambda *_args, **_kwargs: FakeAdapter(profile))
    client = TestClient(create_app(model_key=profile.key))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "qwen3-tts-4bit",
            "input": "hello",
            "response_format": "wav",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.headers["x-tts-model"] == profile.key

    wrong_model = client.post(
        "/v1/audio/speech",
        json={
            "model": "qwen3-tts-8bit",
            "input": "hello",
            "response_format": "wav",
        },
    )
    assert wrong_model.status_code == 409
    assert "Start another instance" in wrong_model.json()["error"]["message"]


def test_native_api_refuses_realtime_demo_profile() -> None:
    try:
        create_app(model_key="realtime-0.5b")
    except ValueError as exc:
        assert "vibevoice-server" in str(exc)
    else:
        raise AssertionError("native API must not wrap the realtime demo adapter")


def test_pyproject_keeps_mac_and_vibevoice_stacks_isolated() -> None:
    import tomllib

    data = tomllib.loads(Path("pyproject.toml").read_text())
    core = data["project"]["dependencies"]
    extras = data["project"]["optional-dependencies"]
    assert not any(item.startswith("huggingface-hub") for item in core)
    assert any(item.startswith("huggingface-hub>=0.30.0,<1.0") for item in extras["vibevoice"])
    assert any(item.startswith("huggingface-hub>=1.3.0,<2.0") for item in extras["mac"])
    assert any(item.startswith("mlx-audio==0.3.0") for item in extras["mac"])
    assert "sources" not in data.get("tool", {}).get("uv", {})
    assert [
        {"extra": "vibevoice"},
        {"extra": "mac"},
    ] in data["tool"]["uv"]["conflicts"]


def test_cli_import_does_not_eagerly_import_realtime_torch_backend() -> None:
    sys.modules.pop("runner.adapters.realtime_demo", None)
    sys.modules.pop("runner.cli", None)

    import runner.cli  # noqa: F401

    assert "runner.adapters.realtime_demo" not in sys.modules


def test_adapter_factory_does_not_eagerly_import_torch_backend() -> None:
    # This is the packaging invariant that lets a Qwen-only Mac install omit
    # PyTorch/VibeVoice. Importing the factory alone must not load longform code.
    sys.modules.pop("runner.adapters.longform_native", None)
    import runner.adapter_factory  # noqa: F401

    assert "runner.adapters.longform_native" not in sys.modules
