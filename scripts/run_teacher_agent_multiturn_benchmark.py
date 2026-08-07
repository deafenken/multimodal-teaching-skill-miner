#!/usr/bin/env python3
"""CLI wrapper for the adversarial multi-turn Teaching Agent benchmark."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teaching_skill_miner.teacher_agent_multiturn_benchmark import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
