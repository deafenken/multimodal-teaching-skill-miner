#!/usr/bin/env python3
"""Mine candidate episode Skills from a long-form dataset manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teaching_skill_miner.episode_skill import mine_episode_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-episode-seconds", type=float, default=180.0)
    parser.add_argument("--max-episode-seconds", type=float, default=900.0)
    parser.add_argument("--target-episode-seconds", type=float, default=480.0)
    parser.add_argument("--lexical-window", type=int, default=4)
    args = parser.parse_args()
    result = mine_episode_dataset(
        args.manifest,
        args.output,
        min_episode_seconds=args.min_episode_seconds,
        max_episode_seconds=args.max_episode_seconds,
        target_episode_seconds=args.target_episode_seconds,
        lexical_window=args.lexical_window,
    )
    print(
        json.dumps(
            {
                "manifest": str(Path(result["manifest_path"]).resolve()),
                "receipt": str(Path(result["receipt_path"]).resolve()),
                "video_count": result["manifest"]["video_count"],
                "episode_count": result["manifest"]["episode_count"],
                "internal_evaluation_is_accuracy": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["manifest"]["episode_count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
