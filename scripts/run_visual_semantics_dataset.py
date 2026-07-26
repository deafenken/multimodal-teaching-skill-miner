#!/usr/bin/env python3
"""Compatibility entrypoint for batch visual-semantic extraction."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teaching_skill_miner.visual_semantics_dataset import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
