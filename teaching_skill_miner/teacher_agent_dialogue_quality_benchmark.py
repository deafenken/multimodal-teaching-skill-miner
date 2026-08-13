"""Gold-free, cross-disciplinary benchmark for observable teaching quality.

The benchmark intentionally does not accept a gold artifact.  Cases contain
only information available to the live teacher: learner utterances, the lesson
phase contract, and teacher-imported source chunks.  Prediction artifacts are
untrusted model output: their evidence IDs, hashes, mastery flags, terminal
booleans, and phase receipts are diagnostic claims, never runtime attestation.
Without an independently supplied runtime ledger, authenticity metrics fail
closed.  Every text/resource metric is computed from observable inputs, so
neither the planner nor this scorer can read a hidden answer key.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import hashlib
import math
import re
from typing import Any, Mapping


DIALOGUE_QUALITY_INPUT_SCHEMA = (
    "teaching_skill_miner.teacher_agent_dialogue_quality_inputs.v1"
)
DIALOGUE_QUALITY_PREDICTIONS_SCHEMA = (
    "teaching_skill_miner.teacher_agent_dialogue_quality_predictions.v1"
)
DIALOGUE_QUALITY_REPORT_SCHEMA = (
    "teaching_skill_miner.teacher_agent_dialogue_quality_report.v1"
)
BENCHMARK_VERSION = "1.0"

_IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]{2,79}")
_LATIN = re.compile(r"[a-zA-Z0-9_]+")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_PROMPT_FIRST = re.compile(
    r"^(?:你觉得|你认为|你能|请你|先说说|先回答|我们先.{0,12}(?:问题|想一想)|"
    r"还记得|what do you|can you|tell me|first[, ]+)",
    re.IGNORECASE,
)
_ASSERTIVE = re.compile(
    r"(?:是|指(?:的是)?|包括|分为|因为|意味着|也就是|可以|不能|答案|原因|"
    r"\bis\b|\bare\b|\bmeans\b|\bbecause\b|\bincludes\b)",
    re.IGNORECASE,
)
_EXPLANATION = re.compile(
    r"(?:因为|所以|例如|具体来说|换句话说|也就是|这一步|原因|意味着|"
    r"because|for example|that means|specifically)",
    re.IGNORECASE,
)
_FORBIDDEN_DATASET_KEYS = frozenset(
    {
        "gold",
        "gold_answer",
        "answer_key",
        "reference_answer",
        "expected_output",
        "acceptable_outputs",
    }
)


class DialogueQualityBenchmarkError(ValueError):
    """Raised when a benchmark artifact violates the observable-only contract."""


def _terms(value: str) -> list[str]:
    value = value.casefold()
    terms = [match.group(0) for match in _LATIN.finditer(value)]
    for match in _CJK.finditer(value):
        run = match.group(0)
        if len(run) == 1:
            terms.append(run)
        else:
            terms.extend(run[index : index + 2] for index in range(len(run) - 1))
    return terms


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _assert_no_hidden_answers(value: Any, *, path: str = "dataset") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold()
            if normalized in _FORBIDDEN_DATASET_KEYS or normalized.startswith("gold_"):
                raise DialogueQualityBenchmarkError(
                    f"{path} contains forbidden hidden-answer field: {key}"
                )
            _assert_no_hidden_answers(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_hidden_answers(item, path=f"{path}[{index}]")


def _require_id(value: Any, *, field: str) -> str:
    candidate = str(value or "")
    if _IDENTIFIER.fullmatch(candidate) is None:
        raise DialogueQualityBenchmarkError(f"{field} is invalid")
    return candidate


def validate_dialogue_quality_dataset(dataset: Any) -> None:
    if not isinstance(dataset, Mapping):
        raise DialogueQualityBenchmarkError("benchmark dataset must be an object")
    _assert_no_hidden_answers(dataset)
    if dataset.get("schema") != DIALOGUE_QUALITY_INPUT_SCHEMA:
        raise DialogueQualityBenchmarkError("benchmark dataset schema is invalid")
    if dataset.get("benchmark_version") != BENCHMARK_VERSION:
        raise DialogueQualityBenchmarkError("benchmark version is invalid")
    _require_id(dataset.get("benchmark_id"), field="benchmark_id")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or len(cases) < 6:
        raise DialogueQualityBenchmarkError("benchmark requires at least six cases")
    case_ids: set[str] = set()
    domains: set[str] = set()
    for case in cases:
        if not isinstance(case, Mapping):
            raise DialogueQualityBenchmarkError("benchmark case must be an object")
        case_id = _require_id(case.get("case_id"), field="case_id")
        if case_id in case_ids:
            raise DialogueQualityBenchmarkError("benchmark case IDs must be unique")
        case_ids.add(case_id)
        domain = _require_id(case.get("domain"), field=f"{case_id}.domain")
        domains.add(domain)
        lesson = case.get("lesson_contract")
        if not isinstance(lesson, Mapping):
            raise DialogueQualityBenchmarkError(f"{case_id}.lesson_contract is invalid")
        phases = lesson.get("required_phases")
        if (
            not isinstance(phases, list)
            or not phases
            or len(phases) > 12
            or len(set(map(str, phases))) != len(phases)
        ):
            raise DialogueQualityBenchmarkError(f"{case_id}.required_phases is invalid")
        if not isinstance(lesson.get("terminal_required_by_end"), bool):
            raise DialogueQualityBenchmarkError(
                f"{case_id}.terminal_required_by_end is invalid"
            )
        resources = case.get("resources", [])
        if not isinstance(resources, list) or len(resources) > 6:
            raise DialogueQualityBenchmarkError(f"{case_id}.resources is invalid")
        for resource in resources:
            if not isinstance(resource, Mapping):
                raise DialogueQualityBenchmarkError(f"{case_id}.resource is invalid")
            if not str(resource.get("resource_id", "")).startswith("res_"):
                raise DialogueQualityBenchmarkError(f"{case_id}.resource_id is invalid")
            chunks = resource.get("chunks", [])
            if not isinstance(chunks, list):
                raise DialogueQualityBenchmarkError(
                    f"{case_id}.resource.chunks is invalid"
                )
            for chunk in chunks:
                if (
                    not isinstance(chunk, Mapping)
                    or not str(chunk.get("text", "")).strip()
                ):
                    raise DialogueQualityBenchmarkError(
                        f"{case_id}.resource chunk is invalid"
                    )
                if chunk.get("content_sha256") != _sha256(str(chunk["text"])):
                    raise DialogueQualityBenchmarkError(
                        f"{case_id}.resource chunk hash is invalid"
                    )
        turns = case.get("turns")
        if not isinstance(turns, list) or len(turns) < 2:
            raise DialogueQualityBenchmarkError(
                f"{case_id} requires at least two learner turns"
            )
        turn_ids: set[str] = set()
        for turn in turns:
            if not isinstance(turn, Mapping):
                raise DialogueQualityBenchmarkError(f"{case_id}.turn is invalid")
            turn_id = _require_id(turn.get("turn_id"), field=f"{case_id}.turn_id")
            if turn_id in turn_ids:
                raise DialogueQualityBenchmarkError(
                    f"{case_id}.turn IDs must be unique"
                )
            turn_ids.add(turn_id)
            if not str(turn.get("learner_text", "")).strip():
                raise DialogueQualityBenchmarkError(
                    f"{case_id}.{turn_id}.learner_text is empty"
                )
            interaction = turn.get("interaction_contract")
            if not isinstance(interaction, Mapping) or any(
                not isinstance(interaction.get(field, False), bool)
                for field in (
                    "answer_first_required",
                    "explanation_gain_required",
                    "resource_grounding_required",
                )
            ):
                raise DialogueQualityBenchmarkError(
                    f"{case_id}.{turn_id}.interaction_contract is invalid"
                )
    if len(domains) < 6:
        raise DialogueQualityBenchmarkError("benchmark must cover six disciplines")


def validate_dialogue_quality_predictions(
    predictions: Any, dataset: Mapping[str, Any]
) -> None:
    if not isinstance(predictions, Mapping):
        raise DialogueQualityBenchmarkError("predictions must be an object")
    if predictions.get("schema") != DIALOGUE_QUALITY_PREDICTIONS_SCHEMA:
        raise DialogueQualityBenchmarkError("predictions schema is invalid")
    if predictions.get("benchmark_version") != BENCHMARK_VERSION:
        raise DialogueQualityBenchmarkError("prediction version is invalid")
    if predictions.get("benchmark_id") != dataset.get("benchmark_id"):
        raise DialogueQualityBenchmarkError("prediction benchmark identity is invalid")
    episodes = predictions.get("episodes")
    if not isinstance(episodes, list):
        raise DialogueQualityBenchmarkError("prediction episodes are invalid")
    expected_cases = {str(item["case_id"]): item for item in dataset["cases"]}
    observed_cases: set[str] = set()
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise DialogueQualityBenchmarkError("prediction episode is invalid")
        case_id = str(episode.get("case_id", ""))
        if case_id not in expected_cases or case_id in observed_cases:
            raise DialogueQualityBenchmarkError("prediction case coverage is invalid")
        observed_cases.add(case_id)
        expected_turn_ids = [
            str(item["turn_id"]) for item in expected_cases[case_id]["turns"]
        ]
        turns = episode.get("turns")
        if (
            not isinstance(turns, list)
            or [
                str(item.get("turn_id", ""))
                for item in turns
                if isinstance(item, Mapping)
            ]
            != expected_turn_ids
        ):
            raise DialogueQualityBenchmarkError(
                f"{case_id} prediction turns are invalid"
            )
        for turn in turns:
            if (
                not isinstance(turn, Mapping)
                or not str(turn.get("teacher_message", "")).strip()
            ):
                raise DialogueQualityBenchmarkError(
                    f"{case_id} teacher message is invalid"
                )
            evidence = turn.get("learner_evidence")
            if not isinstance(evidence, Mapping):
                raise DialogueQualityBenchmarkError(
                    f"{case_id} learner evidence is invalid"
                )
            if evidence.get("gold_accessed") is not False:
                raise DialogueQualityBenchmarkError(
                    f"{case_id} prediction does not prove gold isolation"
                )
            if evidence.get("source") != "learner_response":
                raise DialogueQualityBenchmarkError(
                    f"{case_id} mastery evidence source is invalid"
                )
            for field in ("mastery_before", "mastery_after"):
                values = evidence.get(field)
                if not isinstance(values, Mapping) or any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in values.values()
                ):
                    raise DialogueQualityBenchmarkError(f"{case_id} {field} is invalid")
            uses = turn.get("resource_uses", [])
            if not isinstance(uses, list) or len(uses) > 6:
                raise DialogueQualityBenchmarkError(
                    f"{case_id} resource uses are invalid"
                )
    if observed_cases != set(expected_cases):
        raise DialogueQualityBenchmarkError("predictions do not cover every case")


def _similarity(left: str, right: str) -> float:
    left_terms, right_terms = set(_terms(left)), set(_terms(right))
    jaccard = (
        len(left_terms & right_terms) / len(left_terms | right_terms)
        if left_terms or right_terms
        else 1.0
    )
    sequence = SequenceMatcher(
        None, re.sub(r"\s+", "", left), re.sub(r"\s+", "", right)
    ).ratio()
    return max(jaccard, sequence)


def _metric(passed: int, total: int, *, threshold: float) -> dict[str, Any]:
    score = 1.0 if total == 0 else passed / total
    return {
        "passed": passed,
        "total": total,
        "score": round(score, 6),
        "threshold": threshold,
        "meets_threshold": score >= threshold,
    }


def _resource_catalog(
    case: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    catalog: dict[tuple[str, str], Mapping[str, Any]] = {}
    for resource in case.get("resources", []) or []:
        if not isinstance(resource, Mapping):
            continue
        resource_id = str(resource.get("resource_id", ""))
        for chunk in resource.get("chunks", []) or []:
            if isinstance(chunk, Mapping):
                catalog[(resource_id, str(chunk.get("chunk_id", "")))] = chunk
    return catalog


def score_dialogue_quality_benchmark(
    dataset: Mapping[str, Any], predictions: Mapping[str, Any]
) -> dict[str, Any]:
    """Score seven observable metrics without accepting or loading gold."""

    validate_dialogue_quality_dataset(dataset)
    validate_dialogue_quality_predictions(predictions, dataset)
    case_map = {str(item["case_id"]): item for item in dataset["cases"]}

    repetition_passed = repetition_total = 0
    answer_passed = answer_total = 0
    gain_passed = gain_total = 0
    mastery_passed = mastery_total = 0
    termination_passed = termination_total = 0
    fidelity_passed = fidelity_total = 0
    closure_passed = closure_total = 0
    diagnostics: list[dict[str, Any]] = []

    for episode in predictions["episodes"]:
        case = case_map[str(episode["case_id"])]
        required_phases = set(map(str, case["lesson_contract"]["required_phases"]))
        catalog = _resource_catalog(case)
        previous_message = ""
        valid_closed_phases: set[str] = set()
        final_terminal = False
        for case_turn, predicted_turn in zip(
            case["turns"], episode["turns"], strict=True
        ):
            message = str(predicted_turn["teacher_message"]).strip()
            turn_diagnostic: dict[str, Any] = {
                "case_id": case["case_id"],
                "turn_id": case_turn["turn_id"],
            }
            if previous_message:
                repetition_total += 1
                similarity = _similarity(previous_message, message)
                repetition_ok = similarity < 0.72
                repetition_passed += int(repetition_ok)
                turn_diagnostic["semantic_similarity_to_previous"] = round(
                    similarity, 6
                )

            interaction = case_turn["interaction_contract"]
            if interaction.get("answer_first_required") is True:
                answer_total += 1
                first_clause = re.split(r"[。！？!?\n]", message, maxsplit=1)[0][:180]
                answer_ok = (
                    _PROMPT_FIRST.search(first_clause) is None
                    and _ASSERTIVE.search(first_clause) is not None
                )
                answer_passed += int(answer_ok)
                turn_diagnostic["answer_first"] = answer_ok

            if interaction.get("explanation_gain_required") is True:
                gain_total += 1
                old_terms = set(_terms(previous_message))
                new_terms = set(_terms(message))
                novelty = len(new_terms - old_terms) / max(1, len(new_terms))
                gain_ok = novelty >= 0.2 and (
                    _EXPLANATION.search(message) is not None or len(message) >= 80
                )
                gain_passed += int(gain_ok)
                turn_diagnostic["explanatory_novel_term_ratio"] = round(novelty, 6)

            evidence = predicted_turn["learner_evidence"]
            mastery_total += 1
            learner_text = str(case_turn.get("learner_text", "")).strip()
            claimed_positive = bool(
                str(evidence.get("label", "ambiguous"))
                in {"correct", "self_explanation_verified"}
                and str(evidence.get("evidence_id", "")).strip()
                and str(evidence.get("learner_text_sha256", ""))
                == _sha256(learner_text)
                and evidence.get("assessment_eligible") is True
                and evidence.get("applied_to_mastery") is True
            )
            before = evidence["mastery_before"]
            after = evidence["mastery_after"]
            dimensions = set(before) | set(after)
            increased = any(
                float(after.get(item, before.get(item, 0.0)))
                > float(before.get(item, 0.0)) + 1e-9
                for item in dimensions
            )
            # Everything in ``predictions`` is controlled by the system being
            # evaluated.  Even a perfectly shaped evidence ID/hash/eligibility
            # tuple is a self-assertion and therefore cannot authorize a
            # mastery increase.  A future runtime-ledger scorer must be a
            # separate trusted input path, not a boolean or nested prediction
            # field that an HTTP/model caller can enable.
            mastery_ok = not increased
            mastery_passed += int(mastery_ok)
            turn_diagnostic["false_mastery_avoided"] = mastery_ok
            turn_diagnostic["prediction_claimed_positive_evidence"] = claimed_positive
            turn_diagnostic["independent_runtime_evidence_attested"] = False

            status = str(predicted_turn.get("status", "active"))
            if status in {"completed", "terminated_success"}:
                termination_total += 1
                completed_phases = set(
                    map(str, predicted_turn.get("completed_phases", []) or [])
                )
                terminal = predicted_turn.get("terminal_evidence", {})
                claimed_termination_contract = bool(
                    required_phases.issubset(completed_phases)
                    and isinstance(terminal, Mapping)
                    and terminal.get("independent_recall_verified") is True
                    and terminal.get("gold_accessed") is False
                    and str(terminal.get("learner_evidence_id", ""))
                    == str(evidence.get("evidence_id", ""))
                    and str(terminal.get("learner_text_sha256", ""))
                    == _sha256(learner_text)
                    and claimed_positive
                )
                termination_ok = False
                termination_passed += int(termination_ok)
                turn_diagnostic["premature_termination_avoided"] = termination_ok
                turn_diagnostic["prediction_claimed_terminal_contract"] = (
                    claimed_termination_contract
                )

            resource_uses = predicted_turn.get("resource_uses", []) or []
            if (
                interaction.get("resource_grounding_required") is True
                and not resource_uses
            ):
                fidelity_total += 1
                turn_diagnostic["resource_fidelity"] = False
            for use in resource_uses:
                fidelity_total += 1
                if not isinstance(use, Mapping):
                    continue
                key = (str(use.get("resource_id", "")), str(use.get("chunk_id", "")))
                source = catalog.get(key)
                claim = str(use.get("response_claim", ""))
                excerpt = str(use.get("excerpt", ""))
                source_text = str(source.get("text", "")) if source else ""
                claim_terms = set(_terms(claim))
                support = (
                    len(claim_terms & set(_terms(excerpt))) / len(claim_terms)
                    if claim_terms
                    else 0.0
                )
                fidelity_ok = bool(
                    source
                    and excerpt == source_text
                    and use.get("chunk_content_sha256") == _sha256(excerpt)
                    and use.get("gold_accessed") is False
                    and support >= 0.65
                )
                fidelity_passed += int(fidelity_ok)
                turn_diagnostic["resource_fidelity"] = (
                    turn_diagnostic.get("resource_fidelity", True) and fidelity_ok
                )

            phase_before = str(predicted_turn.get("phase_before", ""))
            phase_after = str(predicted_turn.get("phase_after", phase_before))
            receipt = predicted_turn.get("phase_closure_receipt", {})
            claimed_closure = bool(
                phase_before
                and phase_after
                and phase_before != phase_after
                and isinstance(receipt, Mapping)
                and receipt.get("closed_phase") == phase_before
                and str(receipt.get("summary", "")).strip()
                and str(receipt.get("learner_evidence_id", "")).strip()
                and receipt.get("gold_accessed") is False
            )
            # Phase receipts are prediction-owned too.  Record whether the
            # claim is well formed, but do not score it as an observed closure.
            turn_diagnostic["phase_closure"] = False
            turn_diagnostic["prediction_claimed_phase_closure"] = claimed_closure
            diagnostics.append(turn_diagnostic)
            previous_message = message
            final_terminal = status in {"completed", "terminated_success"}
        if (
            case["lesson_contract"].get("terminal_required_by_end") is True
            and not final_terminal
        ):
            termination_total += 1
        closure_total += len(required_phases)
        closure_passed += len(valid_closed_phases)

    metrics = {
        "semantic_repetition_avoidance": _metric(
            repetition_passed, repetition_total, threshold=0.9
        ),
        "answer_first": _metric(answer_passed, answer_total, threshold=0.9),
        "explanatory_information_gain": _metric(
            gain_passed, gain_total, threshold=0.85
        ),
        "false_mastery_avoidance": _metric(
            mastery_passed, mastery_total, threshold=1.0
        ),
        "premature_termination_avoidance": _metric(
            termination_passed, termination_total, threshold=1.0
        ),
        "resource_fidelity": _metric(fidelity_passed, fidelity_total, threshold=1.0),
        "phase_closure": _metric(closure_passed, closure_total, threshold=0.95),
    }
    return {
        "schema": DIALOGUE_QUALITY_REPORT_SCHEMA,
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_id": dataset["benchmark_id"],
        "evaluation_mode": "observable_contract_no_gold",
        "gold_artifact_accepted": False,
        "prediction_self_assertions_are_evidence": False,
        "self_reported_authenticity_metrics_are_never_verified": True,
        "independent_runtime_evidence_ledger_accepted": False,
        "metrics": metrics,
        "passed": all(item["meets_threshold"] for item in metrics.values()),
        "diagnostics": diagnostics,
    }


__all__ = [
    "BENCHMARK_VERSION",
    "DIALOGUE_QUALITY_INPUT_SCHEMA",
    "DIALOGUE_QUALITY_PREDICTIONS_SCHEMA",
    "DIALOGUE_QUALITY_REPORT_SCHEMA",
    "DialogueQualityBenchmarkError",
    "score_dialogue_quality_benchmark",
    "validate_dialogue_quality_dataset",
    "validate_dialogue_quality_predictions",
]
