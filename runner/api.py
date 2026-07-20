"""Adapter-driven FastAPI application for native/non-demo TTS backends."""

from __future__ import annotations

import asyncio
import io
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from runner.adapter_factory import make_adapter
from runner.errors import (
    BackendUnavailableError,
    CapabilityError,
    InvalidRequestForModelError,
    UnknownModelError,
)
from runner.model_registry import (
    aliases_for_model,
    get_model_profile,
    list_profiles,
    resolve_model_key,
)
from runner.types import (
    SpeechRequest,
    validate_for_longform,
    validate_for_qwen3_tts,
    validate_for_realtime,
)

logger = logging.getLogger(__name__)


class AudioEncodingError(RuntimeError):
    """Raised when WAV output cannot be converted to the requested format."""


def _error_response(message: str, status_code: int, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error" if status_code < 500 else "server_error",
                "param": None,
                "code": code,
            }
        },
    )


def _normalise_wav_mime(mime_type: str) -> bool:
    return mime_type.lower() in {"wav", "audio/wav", "audio/x-wav", "audio/wave"}


def _encode_response(audio: bytes, source_mime: str, response_format: str) -> tuple[bytes, str]:
    requested = response_format.lower()
    if requested not in {"wav", "mp3", "opus"}:
        raise InvalidRequestForModelError(
            "Unsupported response_format. Use one of: wav, mp3, opus."
        )

    if requested == "wav":
        if not _normalise_wav_mime(source_mime):
            raise AudioEncodingError(
                f"Adapter returned {source_mime!r}; WAV passthrough is unavailable."
            )
        return audio, "audio/wav"

    if not _normalise_wav_mime(source_mime):
        raise AudioEncodingError(
            f"Cannot transcode adapter output {source_mime!r}; expected WAV."
        )

    try:
        from pydub import AudioSegment
        from pydub.exceptions import CouldntDecodeError, CouldntEncodeError

        segment = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        output = io.BytesIO()
        if requested == "mp3":
            segment.export(output, format="mp3")
            return output.getvalue(), "audio/mpeg"
        segment.export(output, format="opus", codec="libopus")
        return output.getvalue(), "audio/opus"
    except (FileNotFoundError, CouldntDecodeError, CouldntEncodeError, OSError) as exc:
        raise AudioEncodingError(
            "Audio transcoding failed. Install ffmpeg or request response_format='wav'."
        ) from exc


def _validate_request(profile_family: str, request: SpeechRequest) -> None:
    if profile_family == "realtime":
        errors = validate_for_realtime(request)
    elif profile_family == "longform":
        errors = validate_for_longform(request)
    elif profile_family == "qwen3_tts":
        errors = validate_for_qwen3_tts(request)
    else:
        errors = [f"Unsupported model family: {profile_family}"]
    if errors:
        raise InvalidRequestForModelError(" ".join(errors))


def create_app(
    *,
    model_key: str,
    model_path: str | os.PathLike[str] | None = None,
    device: str | None = None,
    max_concurrent_requests: int = 1,
) -> FastAPI:
    """Create a FastAPI app backed by one canonical model adapter."""
    canonical_key = resolve_model_key(model_key, include_native=True)
    profile = get_model_profile(canonical_key, include_native=True)
    if profile.loader_mode == "subprocess_demo":
        raise ValueError(
            "Realtime demo profiles must be started through 'vibevoice-server' so "
            "their vendored FastAPI/WebSocket application is launched correctly."
        )

    adapter_kwargs: dict[str, Any] = {}
    if model_path is not None:
        adapter_kwargs["model_path"] = model_path
    if device is not None:
        adapter_kwargs["device"] = device
    adapter = make_adapter(canonical_key, **adapter_kwargs)

    concurrency = max(1, int(max_concurrent_requests))
    semaphore = asyncio.Semaphore(concurrency)

    app = FastAPI(
        title="VibeVoice / Qwen3 TTS API",
        version="0.2.0",
        description="OpenAI-compatible local TTS API backed by the runner adapter registry.",
    )
    app.state.model_key = canonical_key
    app.state.profile = profile
    app.state.adapter = adapter
    app.state.semaphore = semaphore

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(
        _request: Request, exc: HTTPException
    ) -> JSONResponse:
        return _error_response(str(exc.detail), exc.status_code, "http_error")

    @app.exception_handler(RequestValidationError)
    async def _request_validation_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(str(exc), 422, "request_validation_error")

    @app.exception_handler(UnknownModelError)
    async def _unknown_model_handler(
        _request: Request, exc: UnknownModelError
    ) -> JSONResponse:
        return _error_response(str(exc), 400, "unknown_model")

    @app.exception_handler(InvalidRequestForModelError)
    async def _invalid_request_handler(
        _request: Request, exc: InvalidRequestForModelError
    ) -> JSONResponse:
        return _error_response(str(exc), 422, "invalid_request_for_model")

    @app.exception_handler(CapabilityError)
    async def _capability_handler(
        _request: Request, exc: CapabilityError
    ) -> JSONResponse:
        return _error_response(str(exc), 400, "unsupported_capability")

    @app.exception_handler(BackendUnavailableError)
    async def _backend_handler(
        _request: Request, exc: BackendUnavailableError
    ) -> JSONResponse:
        return _error_response(str(exc), 503, "backend_unavailable")

    @app.exception_handler(AudioEncodingError)
    async def _encoding_handler(
        _request: Request, exc: AudioEncodingError
    ) -> JSONResponse:
        return _error_response(str(exc), 501, "audio_encoding_unavailable")

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "service": "vibevoice-realtimeFASTAPI",
            "active_model": canonical_key,
            "docs": "/docs",
            "models": "/v1/models",
            "speech": "/v1/audio/speech",
        }

    @app.get("/health")
    async def health() -> JSONResponse:
        details = adapter.health()
        available = bool(details["available"]) if "available" in details else adapter.is_available()
        return JSONResponse(
            status_code=200 if available else 503,
            content={
                "status": "ok" if available else "degraded",
                "active_model": canonical_key,
                **details,
            },
        )

    @app.get("/config")
    async def config() -> dict[str, Any]:
        models = []
        for item in list_profiles(include_native=True):
            row = item.as_dict()
            row["aliases"] = aliases_for_model(item.key, include_native=True)
            row["active"] = item.key == canonical_key
            models.append(row)
        return {
            "active_model": canonical_key,
            "active_capabilities": adapter.capabilities(),
            "max_concurrent_requests": concurrency,
            "available_models": models,
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": canonical_key,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                    "family": profile.family,
                    "backend": profile.loader_mode,
                    "hf_model_id": profile.hf_model_id,
                    "aliases": aliases_for_model(canonical_key, include_native=True),
                }
            ],
        }

    @app.get("/v1/audio/voices")
    async def voices() -> dict[str, Any]:
        return {
            "object": "list",
            "model": canonical_key,
            "data": adapter.list_voices(),
        }

    @app.post("/v1/audio/speech")
    async def speech(request: SpeechRequest) -> Response:
        requested_key = (
            resolve_model_key(request.model, include_native=True)
            if request.model
            else canonical_key
        )
        if requested_key != canonical_key:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"This server instance has '{canonical_key}' loaded, but the request "
                    f"selected '{requested_key}'. Start another instance with "
                    f"--model {requested_key}."
                ),
            )

        request.model = canonical_key
        _validate_request(profile.family, request)
        if not adapter.is_available():
            health_details = adapter.health()
            raise BackendUnavailableError(
                canonical_key,
                str(health_details.get("error") or "Backend readiness check failed."),
            )

        try:
            async with semaphore:
                audio, source_mime = await asyncio.to_thread(adapter.synthesize, request)
            encoded, media_type = await asyncio.to_thread(
                _encode_response,
                audio,
                source_mime,
                request.response_format,
            )
            return Response(
                content=encoded,
                media_type=media_type,
                headers={"X-TTS-Model": canonical_key},
            )
        except (BackendUnavailableError, CapabilityError, InvalidRequestForModelError):
            raise
        except Exception as exc:
            logger.exception("Speech synthesis failed for model %s", canonical_key)
            raise BackendUnavailableError(canonical_key, str(exc)) from exc

    return app


def create_app_from_env() -> FastAPI:
    """Uvicorn factory used by ``--reload`` and process managers."""
    model_key = os.environ.get("TTS_MODEL")
    if not model_key:
        raise RuntimeError(
            "TTS_MODEL is required when starting runner.api directly. "
            "Use 'vibevoice-server --model ...' for normal launches."
        )
    model_path = os.environ.get("MODEL_PATH") or None
    device = os.environ.get("MODEL_DEVICE") or None
    max_concurrency = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "1"))
    return create_app(
        model_key=model_key,
        model_path=model_path,
        device=device,
        max_concurrent_requests=max_concurrency,
    )
