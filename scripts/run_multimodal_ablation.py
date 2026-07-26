#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teaching_skill_miner.io_utils import (  # noqa: E402
    ensure_private_directory,
    read_json,
    write_json,
)
from teaching_skill_miner.multimodal_ablation import (  # noqa: E402
    ARM_ORDER,
    evaluate_multimodal_ablation,
)


def _resolve_transcript(manifest_path: Path, raw_path: str) -> Path:
    value = Path(raw_path)
    candidates = (
        value,
        manifest_path.parent / value,
        ROOT / value,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired transcript/audio/visual/full internal ablations. "
            "This command does not estimate recognition accuracy."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    transcripts = [
        read_json(_resolve_transcript(manifest_path, str(item["transcript_path"])))
        for item in manifest.get("videos", [])
    ]
    result = evaluate_multimodal_ablation(transcripts)
    payloads = result.pop("_payloads")

    output = ensure_private_directory(args.output_dir)
    skill_dir = ensure_private_directory(output / "skills")
    evaluation_dir = ensure_private_directory(output / "evaluations")
    for video_id, arms in payloads.items():
        for arm in ARM_ORDER:
            skill, evaluation = arms[arm]
            skill_path = skill_dir / f"{video_id}.{arm}.skill.json"
            evaluation_path = evaluation_dir / f"{video_id}.{arm}.evaluation.json"
            write_json(skill_path, skill)
            write_json(evaluation_path, evaluation)
            arm_row = next(
                item for item in result["per_lecture"] if item["video_id"] == video_id
            )["arms"][arm]
            arm_row["skill_artifact"] = str(skill_path.relative_to(output))
            arm_row["evaluation_artifact"] = str(evaluation_path.relative_to(output))

    write_json(output / "ablation_report.json", result)
    print(output / "ablation_report.json")
    print(
        "Claim boundary: internal paired metrics only; recognition accuracy and causal "
        "multimodal gain remain unestablished."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

