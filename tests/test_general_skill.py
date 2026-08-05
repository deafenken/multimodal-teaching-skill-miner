"""Contract tests for cross-lecture general Teaching Skill distillation.

The fixtures in ``data/`` are the repository's public, deterministic demo
corpus: two MIT OCW courses with five transcript fixtures per course.  These
tests deliberately derive expected support from those ten mined Skills instead
of pinning statistics from the separate private paper corpus.

The central negative control is provenance semantics: a canonical scaffold or
other ``recommended_enrichment`` must remain executable, but must never be
reported as a teaching method observed across lectures.
"""

from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import re
import tempfile
import unittest
from collections import Counter, defaultdict
from typing import Any, Iterable

import jsonschema

from teaching_skill_miner.cli import main
from teaching_skill_miner.general_skill import (
    distill_general_skill,
    evaluate_general_skill,
    execute_general_skill,
    validate_general_skill,
)
from teaching_skill_miner.io_utils import project_root, read_json, write_json
from teaching_skill_miner.miner import mine_skill
from teaching_skill_miner.models import validate_skill
from teaching_skill_miner.teaching_phases import CANONICAL_PHASES


OVERALL_SUPPORT_FRACTION = 0.8
PER_COURSE_SUPPORT_FRACTION = 0.6


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def _course_sizes(skills: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(skill["source"]["course_id"]) for skill in skills)


def _observed_strategy_courses(
    skills: list[dict[str, Any]],
) -> dict[str, Counter[str]]:
    support: dict[str, Counter[str]] = defaultdict(Counter)
    for skill in skills:
        course_id = str(skill["source"]["course_id"])
        # Count at most once per Skill, even if a malformed producer repeats an
        # id.  Consensus means lecture support, not number of mentions.
        observed_ids = {
            str(strategy["id"])
            for strategy in skill["strategies"]
            if strategy.get("origin") == "observed_method"
        }
        for strategy_id in observed_ids:
            support[strategy_id][course_id] += 1
    return support


def _observed_phase_courses(
    skills: list[dict[str, Any]],
) -> dict[str, Counter[str]]:
    support: dict[str, Counter[str]] = defaultdict(Counter)
    for skill in skills:
        course_id = str(skill["source"]["course_id"])
        observed_ids = {
            str(step["teaching_phase"])
            for step in skill["procedure"]
            if step.get("origin") == "observed_method"
        }
        for phase_id in observed_ids:
            support[phase_id][course_id] += 1
    return support


def _passes_strict_consensus(
    per_course: Counter[str],
    course_sizes: Counter[str],
    total_skill_count: int,
) -> bool:
    overall_minimum = math.ceil(OVERALL_SUPPORT_FRACTION * total_skill_count)
    if sum(per_course.values()) < overall_minimum:
        return False
    return all(
        per_course[course_id]
        >= math.ceil(PER_COURSE_SUPPORT_FRACTION * course_skill_count)
        for course_id, course_skill_count in course_sizes.items()
    )


class GeneralSkillDistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = project_root()
        cls.manifest = read_json(cls.root / "data/dataset_manifest.json")
        cls.skills = [
            mine_skill(read_json(cls.root / item["transcript_path"]))
            for item in cls.manifest["videos"]
        ]
        cls.general_skill = distill_general_skill(cls.skills)

    def test_fixture_is_exactly_two_courses_by_five_valid_skills(self) -> None:
        self.assertEqual(len(self.skills), 10)
        self.assertEqual(_course_sizes(self.skills), Counter({"mit_1806": 5, "mit_60001": 5}))
        for skill in self.skills:
            with self.subTest(skill_id=skill["skill_id"]):
                self.assertTrue(validate_skill(skill).valid)

        distillation = self.general_skill["distillation"]
        self.assertEqual(distillation["input_skill_count"], 10)
        self.assertEqual(distillation["input_course_count"], 2)
        course_skill_counts = distillation["course_skill_counts"]
        self.assertEqual(len(course_skill_counts), 2)
        self.assertEqual(sorted(course_skill_counts.values()), [5, 5])
        self.assertTrue(
            all(re.fullmatch(r"crs_[0-9a-f]{16}", key) for key in course_skill_counts)
        )
        self.assertTrue(validate_general_skill(self.general_skill).valid)

    def test_distillation_is_deterministic_and_input_order_independent(self) -> None:
        repeated = distill_general_skill(copy.deepcopy(self.skills))
        reversed_input = distill_general_skill(list(reversed(copy.deepcopy(self.skills))))
        expected = _canonical_json(self.general_skill)
        self.assertEqual(_canonical_json(repeated), expected)
        self.assertEqual(_canonical_json(reversed_input), expected)

    def test_strict_consensus_and_nine_phase_origins_match_source_evidence(self) -> None:
        distillation = self.general_skill["distillation"]
        thresholds = distillation["thresholds"]
        self.assertEqual(
            thresholds["overall_support_fraction"], OVERALL_SUPPORT_FRACTION
        )
        self.assertEqual(
            thresholds["per_course_support_fraction"],
            PER_COURSE_SUPPORT_FRACTION,
        )

        course_sizes = _course_sizes(self.skills)
        strategy_support = _observed_strategy_courses(self.skills)
        expected_strategy_ids = {
            strategy_id
            for strategy_id, per_course in strategy_support.items()
            if _passes_strict_consensus(per_course, course_sizes, len(self.skills))
        }
        actual_strategies = {
            str(strategy["id"]): strategy
            for strategy in self.general_skill["skill"]["strategies"]
        }
        self.assertEqual(set(actual_strategies), expected_strategy_ids)
        for strategy_id, strategy in actual_strategies.items():
            with self.subTest(strategy_id=strategy_id):
                self.assertEqual(strategy["origin"], "cross_lecture_observed_consensus")
                self.assertEqual(
                    strategy["observed_skill_count"],
                    sum(strategy_support[strategy_id].values()),
                )
                self.assertEqual(
                    strategy["observed_course_count"],
                    sum(value > 0 for value in strategy_support[strategy_id].values()),
                )

        phase_support = _observed_phase_courses(self.skills)
        expected_phase_ids = {str(phase["id"]) for phase in CANONICAL_PHASES}
        procedure = self.general_skill["skill"]["procedure"]
        self.assertEqual(len(procedure), 9)
        self.assertEqual({str(step["teaching_phase"]) for step in procedure}, expected_phase_ids)

        observed_count = 0
        recommended_count = 0
        for step in procedure:
            phase_id = str(step["teaching_phase"])
            per_course = phase_support.get(phase_id, Counter())
            expected_observed = _passes_strict_consensus(
                per_course, course_sizes, len(self.skills)
            )
            expected_origin = (
                "cross_lecture_observed_consensus"
                if expected_observed
                else "recommended_enrichment"
            )
            with self.subTest(phase_id=phase_id):
                self.assertEqual(step["origin"], expected_origin)
                self.assertEqual(step["observed_skill_count"], sum(per_course.values()))
                self.assertEqual(
                    step["observed_course_count"],
                    sum(value > 0 for value in per_course.values()),
                )
            observed_count += expected_observed
            recommended_count += not expected_observed

        # The public fixture must exercise both paths.  This prevents a future
        # implementation from silently labelling all nine canonical scaffolds
        # as observations, or from discarding observed consensus entirely.
        self.assertGreater(observed_count, 0)
        self.assertGreater(recommended_count, 0)

    def test_recommended_strategy_never_becomes_observed_consensus(self) -> None:
        recommended_id = "invented_recommended_only"
        contaminated = copy.deepcopy(self.skills)
        for skill in contaminated:
            skill["strategies"].append(
                {
                    "id": recommended_id,
                    "name": "只用于负对照的推荐策略",
                    "evidence_count": 0,
                    "text_evidence_count": 0,
                    "multimodal_evidence_count": 0,
                    "confidence": 0.0,
                    "origin": "recommended_enrichment",
                }
            )
            self.assertTrue(validate_skill(skill).valid)

        distilled = distill_general_skill(contaminated)
        selected = {strategy["id"] for strategy in distilled["skill"]["strategies"]}
        self.assertNotIn(recommended_id, selected)
        self.assertTrue(validate_general_skill(distilled).valid)

    def test_executes_on_an_unseen_concept_without_source_topic_leakage(self) -> None:
        concept = "动态规划中的最优子结构"
        rendered = execute_general_skill(
            self.general_skill,
            concept=concept,
            learner_level="beginner",
        )
        self.assertIn(concept, rendered)
        self.assertIn("beginner", rendered)
        self.assertNotIn("{concept}", rendered)
        self.assertNotIn("{learner_level}", rendered)
        self.assertEqual(rendered.count("\n### "), 9)

        original_titles = {str(item["title"]) for item in self.manifest["videos"]}
        for title in original_titles:
            self.assertNotIn(title, rendered)

    def test_evaluation_is_structural_support_not_accuracy(self) -> None:
        report = evaluate_general_skill(self.general_skill)
        self.assertTrue(report["passed"])
        boundary = report["claim_boundary"]
        self.assertFalse(boundary["internal_score_is_accuracy"])
        self.assertFalse(boundary["cross_course_generality_established"])
        self.assertFalse(boundary["teaching_effectiveness_established"])

        normalized_keys = {key.lower().replace("-", "_") for key in _keys(report)}
        self.assertNotIn("accuracy", normalized_keys)
        self.assertNotIn("deployment_accuracy", normalized_keys)

    def test_tampered_collection_hash_is_rejected_fail_closed(self) -> None:
        tampered = copy.deepcopy(self.general_skill)
        original = str(tampered["distillation"]["source_collection_sha256"])
        tampered["distillation"]["source_collection_sha256"] = (
            "0" * 64 if original != "0" * 64 else "f" * 64
        )
        validation = validate_general_skill(tampered)
        self.assertFalse(validation.valid)
        self.assertTrue(
            any(
                token in error.lower()
                for error in validation.errors
                for token in ("hash", "sha256", "digest", "fingerprint")
            ),
            validation.errors,
        )

    def test_general_artifact_does_not_copy_quotes_urls_or_absolute_paths(self) -> None:
        serialized = _canonical_json(self.general_skill)
        forbidden_keys = {
            "quote",
            "source_url",
            "transcript_url",
            "transcript_path",
            "frame_path",
            "before_frame",
            "after_frame",
        }
        self.assertTrue(forbidden_keys.isdisjoint(set(_keys(self.general_skill))))

        for skill in self.skills:
            self.assertNotIn(str(skill["source"]["source_url"]), serialized)
            for evidence in skill["source"]["evidence"]:
                self.assertNotIn(str(evidence["quote"]), serialized)

        for value in _strings(self.general_skill):
            with self.subTest(value=value[:80]):
                self.assertFalse(value.startswith(("/", "~/")), value)
                self.assertIsNone(re.match(r"^[A-Za-z]:[\\/]", value), value)

    def test_public_json_schema_accepts_the_generated_artifact(self) -> None:
        schema = read_json(self.root / "schema/general_teaching_skill.schema.json")
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(self.general_skill)

    def test_cli_distils_and_applies_the_general_skill(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "skills"
            for index, skill in enumerate(self.skills):
                write_json(source / f"skill_{index:02d}.full.skill.json", skill)
            output = root / "output"
            with redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "distill-general-skill",
                        "--skill-root",
                        str(source),
                        "--output-dir",
                        str(output),
                        "--example-concept",
                        "动态规划",
                    ]
                )
            self.assertEqual(status, 0)
            for filename in (
                "general_skill.json",
                "general_skill_evaluation.json",
                "general_skill_receipt.json",
                "general_skill_summary.md",
                "example_teaching_process.md",
            ):
                self.assertTrue((output / filename).is_file(), filename)
            with redirect_stdout(io.StringIO()):
                validation_status = main(
                    [
                        "validate",
                        "general-skill",
                        str(output / "general_skill.json"),
                    ]
                )
            self.assertEqual(validation_status, 0)
            reevaluation = output / "reevaluation.json"
            with redirect_stdout(io.StringIO()):
                evaluation_status = main(
                    [
                        "evaluate-general-skill",
                        "--skill",
                        str(output / "general_skill.json"),
                        "--output",
                        str(reevaluation),
                    ]
                )
            self.assertEqual(evaluation_status, 0)
            self.assertTrue(read_json(reevaluation)["passed"])

            next_process = output / "next_process.md"
            with redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "apply-general-skill",
                        "--skill",
                        str(output / "general_skill.json"),
                        "--concept",
                        "牛顿法",
                        "--output",
                        str(next_process),
                    ]
                )
            self.assertEqual(status, 0)
            rendered = next_process.read_text(encoding="utf-8")
            self.assertIn("牛顿法", rendered)
            self.assertNotIn("{concept}", rendered)


if __name__ == "__main__":
    unittest.main()
