"""Shared request types for the runner package."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class SpeakerTurn(BaseModel):
    """A single speaker turn for multi-speaker dialogue."""

    speaker: str
    text: str


class SpeechRequest(BaseModel):
    """Unified speech-synthesis request accepted by all adapters."""

    model: str | None = None
    input: str | None = None
    voice: str | None = None
    response_format: str = "wav"
    temp: float | None = Field(default=None, ge=0.0, le=2.0)
    speed: float | None = Field(default=None, gt=0.0, le=4.0)
    stream: bool = False
    speakers: list[SpeakerTurn] | None = None

    # Qwen3-TTS / voice-cloning controls. Existing clients can ignore these.
    language: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    instruct: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=1000)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    repetition_penalty: float | None = Field(default=None, ge=0.1, le=5.0)

    @field_validator(
        "input",
        "voice",
        "language",
        "ref_audio",
        "ref_text",
        "instruct",
        mode="before",
    )
    @classmethod
    def _normalise_empty_string(cls, value: str | None) -> str | None:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("response_format", mode="before")
    @classmethod
    def _normalise_response_format(cls, value: str | None) -> str:
        return str(value or "wav").strip().lower()

    @model_validator(mode="after")
    def _normalise_empty_speakers(self) -> "SpeechRequest":
        if self.speakers is not None and len(self.speakers) == 0:
            self.speakers = None
        return self


def validate_for_realtime(req: SpeechRequest) -> list[str]:
    """Return validation errors for a realtime-family request."""
    errors: list[str] = []
    if not req.input:
        errors.append("Field 'input' is required for realtime models.")
    if req.speakers:
        errors.append("Field 'speakers' is not supported by realtime models.")
    return errors


def validate_for_longform(req: SpeechRequest) -> list[str]:
    """Return validation errors for a longform-family request."""
    errors: list[str] = []
    if not req.input and not req.speakers:
        errors.append("Either 'input' or 'speakers' is required for longform models.")
    if req.stream:
        errors.append("Streaming is not supported by longform models.")
    return errors


def validate_for_qwen3_tts(req: SpeechRequest) -> list[str]:
    """Return validation errors shared by Qwen3-TTS MLX backends."""
    errors: list[str] = []
    if not req.input:
        errors.append("Field 'input' is required for Qwen3-TTS models.")
    if req.speakers:
        errors.append("Field 'speakers' is not supported by Qwen3-TTS Base models.")
    if req.stream:
        errors.append("Streaming is not exposed by the Qwen3-TTS HTTP adapter yet.")
    if req.speed not in (None, 1.0):
        errors.append("The Qwen3-TTS backends currently require speed=1.0.")
    if req.ref_text and not (req.ref_audio or req.voice):
        errors.append("Field 'ref_text' requires reference audio in 'ref_audio' or 'voice'.")
    return errors
