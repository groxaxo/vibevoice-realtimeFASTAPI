#!/usr/bin/env python3
"""
Run the VibeVoice realtime demo server locally.

This launches the same demo server that the Colab notebook runs,
but locally on your machine without cloudflared tunneling.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch


def detect_device():
    """Auto-detect the best device (prefers CUDA on Linux, MPS on macOS)."""
    # Prefer CUDA if available (common on Linux/Ubuntu with NVIDIA GPUs)
    if torch.cuda.is_available():
        return "cuda"
    # Fall back to MPS on macOS with Apple Silicon
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="Run VibeVoice realtime demo server locally"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/VibeVoice-Realtime-0.5B",
        help="Path to the model directory (default: models/VibeVoice-Realtime-0.5B)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to use (default: auto-detect, prefers cuda on Linux, mps on macOS)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (for development)",
    )
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=15,
        help="Number of inference steps (default: 15)",
    )
    args = parser.parse_args()

    # Auto-detect device if not specified
    device = args.device
    if device is None:
        device = detect_device()
        print(f"🔍 Auto-detected device: {device}")

    # Validate model path
    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        print(f"❌ Error: Model path does not exist: {model_path}")
        print("\n💡 Make sure you've downloaded the model first:")
        print("   uv run python scripts/download_model.py")
        sys.exit(1)

    # Find the VibeVoice demo script
    project_root = Path(__file__).parent.parent
    vibevoice_dir = project_root / "third_party" / "VibeVoice"
    demo_script = vibevoice_dir / "demo" / "vibevoice_realtime_demo.py"

    if not demo_script.exists():
        print(f"❌ Error: VibeVoice demo script not found: {demo_script}")
        print("\n💡 Make sure you've run the bootstrap script first:")
        print("   ./scripts/bootstrap_uv.sh")
        sys.exit(1)

    # Apply overrides if they exist
    overrides_dir = project_root / "overrides"
    if overrides_dir.exists():
        import shutil
        # Copy app.py
        override_app = overrides_dir / "app.py"
        target_app = vibevoice_dir / "demo" / "web" / "app.py"
        if override_app.exists():
            shutil.copy2(override_app, target_app)
        
        # Copy vibevoice_realtime_demo.py
        override_demo = overrides_dir / "vibevoice_realtime_demo.py"
        target_demo = vibevoice_dir / "demo" / "vibevoice_realtime_demo.py"
        if override_demo.exists():
            shutil.copy2(override_demo, target_demo)
            
        # Copy realtime_model_inference_from_file.py
        override_inf = overrides_dir / "realtime_model_inference_from_file.py"
        target_inf = vibevoice_dir / "demo" / "realtime_model_inference_from_file.py"
        if override_inf.exists():
            shutil.copy2(override_inf, target_inf)

        # Copy text_processing.py
        override_tp = overrides_dir / "text_processing.py"
        target_tp = vibevoice_dir / "demo" / "web" / "text_processing.py"
        if override_tp.exists():
            shutil.copy2(override_tp, target_tp)

    # Set environment variables (as the demo script expects)
    os.environ["MODEL_PATH"] = str(model_path)
    os.environ["MODEL_DEVICE"] = device

    print("🚀 Starting VibeVoice realtime demo server...")
    print(f"   Model: {model_path}")
    print(f"   Device: {device}")
    print(f"   Port: {args.port}")
    print(f"\n🌐 Server running at: http://0.0.0.0:{args.port}")
    print(f"   Local access: http://127.0.0.1:{args.port}")
    print("   Press Ctrl+C to stop the server\n")

    # Build command to run the demo script
    cmd = [
        sys.executable,
        str(demo_script),
        "--port",
        str(args.port),
        "--model_path",
        str(model_path),
        "--device",
        device,
        "--inference_steps",
        str(args.inference_steps),
    ]
    if args.reload:
        cmd.append("--reload")

    # Run the demo script
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running demo server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
