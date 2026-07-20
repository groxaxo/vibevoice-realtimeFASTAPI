"""Command-line launcher for all registered TTS backends."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from runner.adapter_factory import make_adapter
from runner.api import create_app
from runner.errors import UnknownModelError
from runner.model_registry import (
    aliases_for_model,
    get_model_profile,
    list_profiles,
    resolve_model_key,
)


def _print_models() -> None:
    print("Registered models:\n")
    for profile in list_profiles(include_native=True):
        aliases = ", ".join(aliases_for_model(profile.key, include_native=True)) or "-"
        print(f"  {profile.key:24} {profile.loader_mode:20} {profile.hf_model_id}")
        print(f"    aliases: {aliases}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a registered VibeVoice or Qwen3-TTS backend"
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("TTS_MODEL", "realtime-0.5b"),
        help="Canonical model key or alias (default: realtime-0.5b)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Override the model source. VibeVoice expects a local path; the Speech Swift "
            "8-bit backend expects a Hugging Face model ID; mlx-audio accepts either a "
            "local directory or a Hugging Face model ID."
        ),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps", "mlx"],
        help="PyTorch device; Qwen3 MLX profiles always use Metal/MLX.",
    )
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--inference-steps", type=int, default=5)
    parser.add_argument("--lazy-load", action="store_true")
    parser.add_argument("--startup-warmup", dest="startup_warmup", action="store_true", default=None)
    parser.add_argument("--no-startup-warmup", dest="startup_warmup", action="store_false")
    parser.add_argument("--max-concurrent-requests", type=int, default=1)
    parser.add_argument("--list-models", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.list_models:
        _print_models()
        return

    try:
        model_key = resolve_model_key(args.model, include_native=True)
    except UnknownModelError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    profile = get_model_profile(model_key, include_native=True)
    device: str | None
    if profile.family == "qwen3_tts":
        device = "mlx"
    elif args.device == "auto":
        # Import the PyTorch-backed detector only for VibeVoice profiles.
        # A Qwen-only Mac installation deliberately has no torch dependency.
        from runner.adapters.realtime_demo import detect_device

        device = detect_device()
    else:
        device = args.device

    adapter_kwargs: dict[str, object] = {"device": device}
    if args.model_path:
        adapter_kwargs["model_path"] = (
            args.model_path
            if profile.family == "qwen3_tts"
            else Path(args.model_path).expanduser()
        )
    adapter = make_adapter(model_key, **adapter_kwargs)

    print(f"Model: {model_key}")
    print(f"Backend: {profile.loader_mode}")
    print(f"Source: {args.model_path or profile.hf_model_id}")
    print(f"Device: {device}")

    if profile.loader_mode == "subprocess_demo":
        adapter.launch(
            project_root=Path(__file__).resolve().parents[1],
            host=args.host,
            port=args.port,
            inference_steps=args.inference_steps,
            lazy_load=args.lazy_load,
            startup_warmup=args.startup_warmup,
            reload=args.reload,
        )
        return

    if not adapter.is_available():
        details = adapter.health()
        error = details.get("error", "backend readiness check failed")
        print(f"Backend unavailable: {error}", file=sys.stderr)
        raise SystemExit(1)

    import uvicorn

    if args.reload:
        os.environ["TTS_MODEL"] = model_key
        if args.model_path:
            os.environ["MODEL_PATH"] = args.model_path
        else:
            os.environ.pop("MODEL_PATH", None)
        if device:
            os.environ["MODEL_DEVICE"] = device
        os.environ["MAX_CONCURRENT_REQUESTS"] = str(
            max(1, args.max_concurrent_requests)
        )
        uvicorn.run(
            "runner.api:create_app_from_env",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
        )
        return

    app = create_app(
        model_key=model_key,
        model_path=args.model_path,
        device=device,
        max_concurrent_requests=max(1, args.max_concurrent_requests),
    )
    uvicorn.run(app, host=args.host, port=args.port)
