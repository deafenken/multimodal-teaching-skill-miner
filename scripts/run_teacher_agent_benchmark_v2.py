#!/usr/bin/env python3
"""Validate or score the product-level Teaching Agent benchmark v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teaching_skill_miner.deepseek_client import DeepSeekClient, DeepSeekConfig  # noqa: E402
from teaching_skill_miner.io_utils import read_json, resolve_resource_path, write_json  # noqa: E402
from teaching_skill_miner.teacher_agent_benchmark_v2 import (  # noqa: E402
    INPUT_SCHEMA,
    TeachingAgentBenchmarkV2Error,
    _fingerprint,
    predictions_from_live_cases,
    score_benchmark_v2,
    validate_benchmark_gold,
    validate_benchmark_inputs,
)
from teaching_skill_miner.teacher_agent_live import LiveAgentOptions  # noqa: E402


DEFAULT_INPUTS = Path("data/teacher_agent_benchmark_v2_development.json")
DEFAULT_GOLD = Path("data/teacher_agent_benchmark_v2_development_gold.json")
DEFAULT_LIBRARY = Path("data/teacher_agent_skill_library_v2.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or score Teaching Agent benchmark v2. The bundled split "
            "is author-constructed development data, not a held-out or learning-effect study."
        )
    )
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--skill-library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--allow-remote-benchmark-data", action="store_true")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--no-rule-fallback", action="store_true")
    parser.add_argument("--acknowledge-held-out", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    inputs = read_json(resolve_resource_path(args.inputs))
    gold = read_json(resolve_resource_path(args.gold))
    library = read_json(resolve_resource_path(args.skill_library))
    try:
        validate_benchmark_inputs(inputs, library)
        validate_benchmark_gold(gold, inputs, library)
    except TeachingAgentBenchmarkV2Error as exc:
        parser.error(str(exc))
    if args.validate_only:
        receipt = {
            "schema": INPUT_SCHEMA,
            "benchmark_version": inputs["benchmark_version"],
            "benchmark_id": inputs["benchmark_id"],
            "split": inputs["split"],
            "case_count": len(inputs["cases"]),
            "turn_count": sum(len(case["turns"]) for case in inputs["cases"]),
            "input_fingerprint": _fingerprint(inputs),
            "gold_fingerprint": _fingerprint(gold),
            "gold_sent_to_executor": False,
            "held_out_after_prompt_development": bool(
                inputs["claim_boundary"]["held_out_after_prompt_development"]
            ),
            "deployment_accuracy_established": False,
            "real_learning_effect_established": False,
            "validated": True,
        }
        if args.output:
            write_json(args.output, receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.online and args.predictions:
        parser.error("choose either --online or --predictions, not both")
    if not args.online and args.predictions is None:
        parser.error("scoring requires --predictions or explicit --online execution")
    if args.online:
        if not args.allow_remote_benchmark_data:
            parser.error("--online requires --allow-remote-benchmark-data")
        config = DeepSeekConfig.from_environment(
            api_key_file=args.api_key_file,
            allow_remote_student_data=True,
            model="deepseek-v4-flash",
        )
        client = DeepSeekClient(config)
        predictions = predictions_from_live_cases(
            inputs,
            library,
            client,
            options=LiveAgentOptions(
                fallback_to_rules=not args.no_rule_fallback,
                action_only_repair_enabled=True,
                state_first_route_adjudication_enabled=True,
                agent_loop_enabled=True,
                agent_loop_post_assessment_enabled=True,
                maximum_agent_steps=4,
            ),
        )
        if args.predictions_output is not None:
            write_json(args.predictions_output, predictions)
    else:
        predictions = read_json(args.predictions)
    try:
        report = score_benchmark_v2(
            inputs,
            gold,
            predictions,
            library,
            acknowledge_held_out=args.acknowledge_held_out,
        )
    except TeachingAgentBenchmarkV2Error as exc:
        parser.error(str(exc))
    if args.output:
        write_json(args.output, report)
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "benchmark_id": report["benchmark_id"],
                "split": report["split"],
                "run_fingerprint": report["run_fingerprint"],
                "metrics": report["metrics"],
                "claim_boundary": report["claim_boundary"],
                "output": str(args.output.resolve()) if args.output else None,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
