"""Negative controls for the ``method_fidelity`` dimension.

Every other rubric dimension checks a field the miner emits by construction, so
a skill cannot score badly on them.  ``method_fidelity`` is only meaningful if a
procedure that *claims* to reproduce the teacher's method, but cannot back the
claim, scores strictly lower than one that can.  These tests degrade synthetic
copies of a real skill and assert that ordering holds.

The real mined skills are never mutated in place; every case deep-copies.
"""

from __future__ import annotations

import copy
import unittest
from typing import Any, Callable

from teaching_skill_miner.evaluator import (
    DIMENSION_WEIGHTS,
    METHOD_FIDELITY_WEIGHTS,
    evaluate_skill,
)
from teaching_skill_miner.io_utils import project_root, read_json
from teaching_skill_miner.miner import mine_skill
from teaching_skill_miner.teaching_phases import CANONICAL_PHASES


def _observed(skill: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for step in skill["procedure"] if step["origin"] == "observed_method"]


def _strip_to_template(skill: dict[str, Any]) -> dict[str, Any]:
    """The pre-distillation failure mode: a generic template claiming nothing."""

    degraded = copy.deepcopy(skill)
    for step in degraded["procedure"]:
        step["origin"] = "recommended_enrichment"
        step["evidence_ids"] = []
        step["observed_span"] = None
        step["matched_cues"] = []
        step["provenance"] = {
            "origin": "recommended_enrichment",
            "strategy_id": None,
            "evidence_ids": [],
            "derivation": "canonical_phase_scaffold",
        }
    return degraded


def _fabricate_cues(skill: dict[str, Any]) -> dict[str, Any]:
    """Steps claim trigger cues that do not occur in the evidence they cite."""

    degraded = copy.deepcopy(skill)
    for step in _observed(degraded):
        step["matched_cues"] = ["this cue never occurs in any transcript segment"]
    return degraded


def _shuffle_spans(skill: dict[str, Any]) -> dict[str, Any]:
    """Timestamps are reversed, so spans no longer contain their own evidence."""

    degraded = copy.deepcopy(skill)
    steps = _observed(degraded)
    spans = [copy.deepcopy(step["observed_span"]) for step in steps][::-1]
    for step, span in zip(steps, spans):
        step["observed_span"] = span
    return degraded


def _drop_evidence(skill: dict[str, Any]) -> dict[str, Any]:
    """Each observed step keeps only one citation, thinning the grounding."""

    degraded = copy.deepcopy(skill)
    for step in _observed(degraded):
        step["evidence_ids"] = step["evidence_ids"][:1]
        step["provenance"]["evidence_ids"] = step["evidence_ids"]
    return degraded


def _collapse_phases(skill: dict[str, Any]) -> dict[str, Any]:
    """All observed steps claim the same phase, so the flow is not distilled."""

    degraded = copy.deepcopy(skill)
    for step in _observed(degraded):
        step["teaching_phase"] = CANONICAL_PHASES[0]["id"]
    return degraded


DEGRADATIONS: list[tuple[str, Callable[[dict[str, Any]], dict[str, Any]]]] = [
    ("template_only", _strip_to_template),
    ("fabricated_cues", _fabricate_cues),
    ("shuffled_spans", _shuffle_spans),
    ("dropped_evidence", _drop_evidence),
    ("collapsed_phases", _collapse_phases),
]


class MethodFidelityWeightTests(unittest.TestCase):
    def test_dimension_weights_sum_to_one(self) -> None:
        self.assertAlmostEqual(sum(DIMENSION_WEIGHTS.values()), 1.0, places=9)
        self.assertIn("method_fidelity", DIMENSION_WEIGHTS)

    def test_method_fidelity_component_weights_sum_to_one(self) -> None:
        self.assertAlmostEqual(sum(METHOD_FIDELITY_WEIGHTS.values()), 1.0, places=9)

    def test_reported_weights_match_module_constants(self) -> None:
        root = project_root()
        transcript = read_json(root / "data/transcripts/python_l03.json")
        report = evaluate_skill(mine_skill(transcript), transcript)
        self.assertEqual(report["weights"], DIMENSION_WEIGHTS)
        self.assertEqual(report["method_fidelity"]["weights"], METHOD_FIDELITY_WEIGHTS)
        self.assertEqual(
            set(report["dimensions"]), set(DIMENSION_WEIGHTS)
        )


class MethodFidelityDegradationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = project_root()
        cls.transcript = read_json(cls.root / "data/transcripts/python_l03.json")
        cls.skill = mine_skill(cls.transcript)
        cls.report = evaluate_skill(cls.skill, cls.transcript)

    def test_honest_skill_verifies_every_observed_step(self) -> None:
        components = self.report["method_fidelity"]["components"]
        self.assertEqual(components["cue_verification"], 1.0)
        self.assertEqual(components["span_consistency"], 1.0)
        self.assertEqual(components["temporal_monotonicity"], 1.0)
        self.assertTrue(self.report["gates"]["method_distilled_from_video"])

    def test_every_degradation_scores_strictly_lower(self) -> None:
        baseline = self.report["dimensions"]["method_fidelity"]
        for name, degrade in DEGRADATIONS:
            with self.subTest(degradation=name):
                report = evaluate_skill(degrade(self.skill), self.transcript)
                self.assertLess(
                    report["dimensions"]["method_fidelity"],
                    baseline,
                    f"{name} did not reduce method_fidelity",
                )
                self.assertLess(report["overall_score"], self.report["overall_score"])

    def test_each_degradation_is_caught_by_its_own_component(self) -> None:
        """A degradation must move the component that is supposed to detect it."""

        expected = {
            "fabricated_cues": "cue_verification",
            "shuffled_spans": "span_consistency",
            "dropped_evidence": "evidence_density",
            "collapsed_phases": "phase_coverage",
        }
        baseline = self.report["method_fidelity"]["components"]
        for name, degrade in DEGRADATIONS:
            component = expected.get(name)
            if component is None:
                continue
            with self.subTest(degradation=name, component=component):
                components = evaluate_skill(degrade(self.skill), self.transcript)[
                    "method_fidelity"
                ]["components"]
                self.assertLess(components[component], baseline[component])

    def test_template_only_skill_loses_the_distillation_gate(self) -> None:
        report = evaluate_skill(_strip_to_template(self.skill), self.transcript)
        fidelity = report["method_fidelity"]
        self.assertEqual(fidelity["observed_step_count"], 0)
        self.assertEqual(fidelity["score"], 0.0)
        self.assertFalse(report["gates"]["method_distilled_from_video"])
        self.assertFalse(report["passed"])

    def test_fabricated_evidence_ids_are_not_credited(self) -> None:
        degraded = copy.deepcopy(self.skill)
        for step in _observed(degraded):
            step["evidence_ids"] = ["evi_" + "0" * 16]
            step["provenance"]["evidence_ids"] = step["evidence_ids"]
        report = evaluate_skill(degraded, self.transcript)
        components = report["method_fidelity"]["components"]
        self.assertEqual(components["cue_verification"], 0.0)
        self.assertEqual(components["span_consistency"], 0.0)
        self.assertEqual(components["evidence_utilisation"], 0.0)
        self.assertEqual(components["evidence_density"], 0.0)
        self.assertLess(
            report["dimensions"]["method_fidelity"],
            self.report["dimensions"]["method_fidelity"],
        )


class MethodFidelityDiscriminationTests(unittest.TestCase):
    """The dimension must separate real skills from one another, not just from
    tampered ones — otherwise it is another constant."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = project_root()
        manifest = read_json(cls.root / "data/dataset_manifest.json")
        cls.reports = []
        for item in manifest["videos"]:
            transcript = read_json(cls.root / item["transcript_path"])
            cls.reports.append(evaluate_skill(mine_skill(transcript), transcript))

    def test_scores_are_not_constant_across_the_corpus(self) -> None:
        scores = {report["dimensions"]["method_fidelity"] for report in self.reports}
        self.assertGreater(
            len(scores), 1, "method_fidelity is constant and therefore measures nothing"
        )

    def test_overall_scores_are_not_constant_across_the_corpus(self) -> None:
        scores = {report["overall_score"] for report in self.reports}
        self.assertGreater(len(scores), 1)

    def test_all_real_skills_still_pass(self) -> None:
        for report in self.reports:
            with self.subTest(skill_id=report["skill_id"]):
                self.assertTrue(report["passed"])
                self.assertGreaterEqual(report["overall_score"], 75)

    def test_fidelity_tracks_observed_phase_coverage(self) -> None:
        """More distinct phases recovered must not score lower than fewer."""

        pairs = sorted(
            (
                report["method_fidelity"]["distinct_observed_phase_count"],
                report["method_fidelity"]["components"]["phase_coverage"],
            )
            for report in self.reports
        )
        for (count_a, cover_a), (count_b, cover_b) in zip(pairs, pairs[1:]):
            if count_a < count_b:
                self.assertLess(cover_a, cover_b)


if __name__ == "__main__":
    unittest.main()
