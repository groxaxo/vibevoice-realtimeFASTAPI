#!/usr/bin/env python3
"""Source-tree wrapper for :mod:`runner.cli`."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runner.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
