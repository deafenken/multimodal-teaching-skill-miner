#!/usr/bin/env python3
"""Generate a fail-closed TeachObs/new-site confirmatory preregistration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teaching_skill_miner.io_utils import write_json  # noqa: E402
from teaching_skill_miner.teachobs_lockbox import (  # noqa: E402
    ARM_ORDER,
    TeachObsLockboxError,
    build_teachobs_lockbox_preregistration,
    validate_teachobs_lockbox_preregistration,
)


def _arm_models(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        arm, separator, path = value.partition("=")
        if not separator or arm not in ARM_ORDER or not path.strip():
            raise TeachObsLockboxError(
                "--arm-model must use one of "
                f"{','.join(ARM_ORDER)}=PATH"
            )
        if arm in result:
            raise TeachObsLockboxError(f"duplicate --arm-model for {arm}")
        result[arm] = path
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a no-outcome four-arm protocol for a prospective, new-site, "
            "same-sample paired cluster lockbox. Public TeachObs 23/7 is always "
            "recorded as development-only and cannot become this lockbox."
        )
    )
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--system-artifact",
        help="serialized frozen four-arm system/bundle; omit only for a pending draft",
    )
    parser.add_argument(
        "--analysis-code",
        help="frozen evaluator/bootstrap source or archive; omit only for a pending draft",
    )
    parser.add_argument(
        "--arm-model",
        action="append",
        default=[],
        metavar="ARM=PATH",
        help=(
            "repeat for transcript_only, transcript_audio, transcript_visual, and "
            "full using each arm's manifest.json from the same frozen bundle; "
            "companion arrays.npz files are required and a partial set remains "
            "non-executable"
        ),
    )
    parser.add_argument(
        "--target-cluster-field",
        choices=("classroom_id", "session_id"),
        default="classroom_id",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        preregistration = build_teachobs_lockbox_preregistration(
            study_id=args.study_id,
            system_artifact_path=args.system_artifact,
            analysis_code_path=args.analysis_code,
            arm_model_artifact_paths=_arm_models(args.arm_model),
            target_cluster_field=args.target_cluster_field,
        )
        report = validate_teachobs_lockbox_preregistration(preregistration)
        output = write_json(args.output, preregistration)
    except (OSError, TeachObsLockboxError) as exc:
        raise SystemExit(f"TeachObs lockbox preregistration failed: {exc}") from exc
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "preregistration_fingerprint": report[
                    "preregistration_fingerprint"
                ],
                "frozen_artifact_set_complete": report[
                    "frozen_artifact_set_complete"
                ],
                "preregistration_execution_ready": False,
                "external_registration_signature_verified": False,
                "target_execution_evidence_complete": False,
                "confirmatory_multimodal_gain_established": False,
                "deployment_accuracy_established": False,
                "learner_effectiveness_established": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
