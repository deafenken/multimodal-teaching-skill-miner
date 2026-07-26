#!/usr/bin/env python3
"""Prepare or analyze the private token-only learner-effect study package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teaching_skill_miner.learner_effect_study import (  # noqa: E402
    analyze_learner_effect_study,
    generate_learner_effect_study_package,
    write_learner_effect_analysis,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Private cluster-randomized learner-study templates and fail-closed "
            "analysis; this never establishes learning effectiveness by itself."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--study-id", required=True)
    prepare.add_argument(
        "--cluster-unit", choices=("teacher", "classroom"), default="classroom"
    )
    prepare.add_argument("--cluster-count", type=int, default=6)
    prepare.add_argument("--participants-per-cluster", type=int, default=5)
    prepare.add_argument("--randomization-seed", type=int, default=20260723)
    prepare.add_argument("--bootstrap-seed", type=int, default=20260724)
    prepare.add_argument("--bootstrap-replicates", type=int, default=2000)
    prepare.add_argument(
        "--data-origin", choices=("template", "synthetic", "real"), default="template"
    )
    prepare.add_argument("--ethics-approval-id", default="")
    prepare.add_argument(
        "--informed-consent-or-approved-waiver", action="store_true"
    )
    prepare.add_argument(
        "--preregistration-frozen-before-allocation", action="store_true"
    )
    prepare.add_argument(
        "--allocation-concealment-procedure-declared", action="store_true"
    )
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--preregistration", required=True)
    analyze.add_argument("--allocation", required=True)
    analyze.add_argument("--outcomes", required=True)
    analyze.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        result = generate_learner_effect_study_package(
            args.output_dir,
            study_id=args.study_id,
            cluster_unit=args.cluster_unit,
            cluster_count=args.cluster_count,
            participants_per_cluster=args.participants_per_cluster,
            randomization_seed=args.randomization_seed,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_replicates=args.bootstrap_replicates,
            data_origin=args.data_origin,
            ethics_approval_id=args.ethics_approval_id,
            informed_consent_or_approved_waiver=(
                args.informed_consent_or_approved_waiver
            ),
            preregistration_frozen_before_allocation=(
                args.preregistration_frozen_before_allocation
            ),
            allocation_concealment_procedure_declared=(
                args.allocation_concealment_procedure_declared
            ),
        )
        summary = {
            "mode": result["mode"],
            "output_directory": result["output_directory"],
            "planned_cluster_count": result["planned_cluster_count"],
            "planned_participant_count": result["planned_participant_count"],
            "identity_fields_included": False,
            "learner_effectiveness_established": False,
        }
    else:
        result = analyze_learner_effect_study(
            args.preregistration, args.allocation, args.outcomes
        )
        output = write_learner_effect_analysis(args.output, result)
        summary = {
            "mode": "learner_effect_study_analyzed",
            "output": str(output.resolve()),
            "local_preregistered_positive_result_gate": result[
                "local_preregistered_positive_result_gate"
            ],
            "learner_effectiveness_established": False,
            "external_signature_still_required": True,
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
