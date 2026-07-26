#!/usr/bin/env python3
"""Compatibility entrypoint for hash-bound CLIP visual-semantic extraction."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teaching_skill_miner.visual_semantics import (  # noqa: E402,F401
    ONTOLOGY,
    SCHEMA_RESULT,
    SCHEMA_TASK,
    _projected_feature_tensor,
    main,
    run_inference,
    validate_task_manifest,
)


if __name__ == "__main__":
    raise SystemExit(main())
