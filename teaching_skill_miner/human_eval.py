from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import mean
from typing import Any, Mapping


RATING_FIELDS = (
    "goal_clarity_1_5",
    "evidence_fidelity_1_5",
    "procedure_executability_1_5",
    "adaptivity_1_5",
    "transferability_1_5",
)


def skill_review_fingerprint(skill: Mapping[str, Any]) -> str:
    """Bind a human review to the exact canonical Teaching Skill contents."""

    if not isinstance(skill, Mapping):
        raise ValueError("skill review fingerprint requires one Skill object")
    try:
        payload = json.dumps(
            skill,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Skill is not canonical finite JSON") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _quadratic_weighted_kappa(left: list[int], right: list[int], minimum: int = 1, maximum: int = 5) -> float | None:
    if len(left) != len(right) or not left:
        return None
    categories = list(range(minimum, maximum + 1))
    denominator = float((maximum - minimum) ** 2)
    left_counts = Counter(left)
    right_counts = Counter(right)
    observed = sum(((a - b) ** 2 / denominator) for a, b in zip(left, right)) / len(left)
    expected = 0.0
    for a in categories:
        for b in categories:
            expected += (
                ((a - b) ** 2 / denominator)
                * (left_counts[a] / len(left))
                * (right_counts[b] / len(right))
            )
    if expected == 0:
        return 1.0 if observed == 0 else None
    return round(1 - observed / expected, 3)


def summarize_human_review(
    path: str | Path,
    expected_skill_ids: Iterable[str] | None = None,
    expected_skill_fingerprints: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    declared_skill_ids = {
        str(row.get("skill_id") or "").strip()
        for row in rows
        if str(row.get("skill_id") or "").strip()
    }
    if expected_skill_ids is None:
        expected_ids = set(declared_skill_ids)
        expected_skill_source = "csv_declared_skill_ids"
    else:
        expected_ids = {
            str(skill_id).strip()
            for skill_id in expected_skill_ids
            if str(skill_id).strip()
        }
        expected_skill_source = "caller_supplied"
    fingerprint_contract: dict[str, str] | None = None
    if expected_skill_fingerprints is not None:
        fingerprint_contract = {
            str(skill_id).strip(): str(fingerprint).strip().lower()
            for skill_id, fingerprint in expected_skill_fingerprints.items()
        }
        if set(fingerprint_contract) != expected_ids:
            raise ValueError(
                "expected Skill fingerprint keys must exactly match expected_skill_ids"
            )
        if any(
            len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
            for fingerprint in fingerprint_contract.values()
        ):
            raise ValueError("expected Skill fingerprints must be lowercase SHA-256 digests")

    completed_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    fingerprint_mismatches: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, 2):
        reviewer_id = str(row.get("reviewer_id") or "").strip()
        if not reviewer_id:
            continue
        parsed: dict[str, Any] = {
            "skill_id": str(row.get("skill_id") or "").strip(),
            "reviewer_id": reviewer_id,
        }
        if not parsed["skill_id"]:
            errors.append(f"row {row_number}: missing skill_id")
            continue
        valid = True
        if fingerprint_contract is not None:
            supplied_fingerprint = str(row.get("skill_fingerprint") or "").strip().lower()
            expected_fingerprint = fingerprint_contract.get(parsed["skill_id"])
            if expected_fingerprint is None:
                errors.append(
                    f"row {row_number}: unexpected skill_id has no fingerprint contract"
                )
                valid = False
            elif supplied_fingerprint != expected_fingerprint:
                errors.append(
                    f"row {row_number}: skill_fingerprint does not match current Skill contents"
                )
                fingerprint_mismatches.append(
                    {
                        "row_number": row_number,
                        "skill_id": parsed["skill_id"],
                        "supplied": supplied_fingerprint or None,
                        "expected": expected_fingerprint,
                    }
                )
                valid = False
            parsed["skill_fingerprint"] = supplied_fingerprint
        for field in RATING_FIELDS:
            try:
                value = int(row.get(field) or "")
            except (TypeError, ValueError):
                errors.append(f"row {row_number}: {field} must be an integer")
                valid = False
                continue
            if value < 1 or value > 5:
                errors.append(f"row {row_number}: {field} must be between 1 and 5")
                valid = False
            parsed[field] = value
        try:
            harm = int(row.get("harm_or_bias_flag_0_1") or "")
        except (TypeError, ValueError):
            errors.append(f"row {row_number}: harm_or_bias_flag_0_1 must be 0 or 1")
            valid = False
            harm = -1
        if harm not in {0, 1}:
            errors.append(f"row {row_number}: harm_or_bias_flag_0_1 must be 0 or 1")
            valid = False
        parsed["harm_or_bias_flag_0_1"] = harm
        if valid:
            parsed["row_number"] = row_number
            completed_rows.append(parsed)

    reviewer_assignment_counts = Counter(
        (row["skill_id"], row["reviewer_id"]) for row in completed_rows
    )
    duplicate_reviewer_coverage = [
        {
            "skill_id": skill_id,
            "reviewer_id": reviewer_id,
            "row_count": row_count,
        }
        for (skill_id, reviewer_id), row_count in sorted(
            reviewer_assignment_counts.items()
        )
        if row_count > 1
    ]

    # Duplicate rows never increase coverage or influence means. They remain an
    # explicit validation failure so conflicting duplicate ratings cannot be
    # silently resolved in favour of one result.
    grouped_by_reviewer: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in completed_rows:
        grouped_by_reviewer[row["skill_id"]].setdefault(row["reviewer_id"], row)
    grouped: dict[str, list[dict[str, Any]]] = {
        skill_id: list(by_reviewer.values())
        for skill_id, by_reviewer in grouped_by_reviewer.items()
    }

    reviewed_skill_ids = set(grouped)
    missing_skill_ids = sorted(expected_ids - reviewed_skill_ids)
    extra_skill_ids = sorted(declared_skill_ids - expected_ids)
    insufficient_reviewer_skill_ids = sorted(
        skill_id
        for skill_id in expected_ids
        if len(grouped_by_reviewer.get(skill_id, {})) < 2
    )

    skill_results: list[dict[str, Any]] = []
    for skill_id, reviews in sorted(grouped.items()):
        dimension_means = {
            field: round(mean(review[field] for review in reviews), 2)
            for field in RATING_FIELDS
        }
        reviewer_count = len({review["reviewer_id"] for review in reviews})
        harm_flags = sum(review["harm_or_bias_flag_0_1"] for review in reviews)
        passed = reviewer_count >= 2 and all(value >= 3.5 for value in dimension_means.values()) and harm_flags == 0
        skill_results.append(
            {
                "skill_id": skill_id,
                "reviewer_count": reviewer_count,
                "dimension_means": dimension_means,
                "overall_mean": round(mean(dimension_means.values()), 2),
                "harm_or_bias_flags": harm_flags,
                "passed": passed,
            }
        )

    pair_values: dict[tuple[str, str], tuple[list[int], list[int]]] = {}
    for reviews in grouped.values():
        by_reviewer: dict[str, dict[str, Any]] = {review["reviewer_id"]: review for review in reviews}
        reviewers = sorted(by_reviewer)
        for left_id, right_id in itertools.combinations(reviewers, 2):
            left_ratings, right_ratings = pair_values.setdefault((left_id, right_id), ([], []))
            left = by_reviewer[left_id]
            right = by_reviewer[right_id]
            for field in RATING_FIELDS:
                left_ratings.append(left[field])
                right_ratings.append(right[field])
    pair_kappas = {
        f"{left_id}__{right_id}": value
        for (left_id, right_id), ratings in sorted(pair_values.items())
        if (value := _quadratic_weighted_kappa(ratings[0], ratings[1])) is not None
    }
    kappa = round(mean(pair_kappas.values()), 3) if pair_kappas else None
    expected_coverage_complete = bool(expected_ids) and not missing_skill_ids
    distinct_reviewer_coverage_complete = (
        bool(expected_ids) and not insufficient_reviewer_skill_ids
    )
    coverage_valid = (
        expected_coverage_complete
        and distinct_reviewer_coverage_complete
        and not extra_skill_ids
        and not duplicate_reviewer_coverage
        and not fingerprint_mismatches
    )
    if errors or extra_skill_ids or duplicate_reviewer_coverage:
        validation_status = "invalid"
    elif not completed_rows or not coverage_valid:
        validation_status = "incomplete"
    else:
        validation_status = "complete"

    expected_skill_results = [
        item for item in skill_results if item["skill_id"] in expected_ids
    ]
    passed = (
        validation_status == "complete"
        and bool(expected_skill_results)
        and all(item["passed"] for item in expected_skill_results)
        and kappa is not None
        and kappa >= 0.67
    )
    return {
        "input_row_count": len(rows),
        "completed_review_count": len(completed_rows),
        "reviewed_skill_count": len(skill_results),
        "validation_status": validation_status,
        "expected_skill_id_source": expected_skill_source,
        "expected_skill_ids": sorted(expected_ids),
        "reviewed_skill_ids": sorted(reviewed_skill_ids),
        "missing_skill_ids": missing_skill_ids,
        "extra_skill_ids": extra_skill_ids,
        "insufficient_reviewer_skill_ids": insufficient_reviewer_skill_ids,
        "duplicate_reviewer_coverage": duplicate_reviewer_coverage,
        "skill_fingerprint_binding_required": fingerprint_contract is not None,
        "skill_fingerprint_mismatches": fingerprint_mismatches,
        "skill_fingerprint_binding_complete": bool(fingerprint_contract)
        and not fingerprint_mismatches
        and expected_coverage_complete,
        "coverage": {
            "expected_skill_count": len(expected_ids),
            "reviewed_expected_skill_count": len(expected_ids & reviewed_skill_ids),
            "expected_skill_coverage_complete": expected_coverage_complete,
            "two_distinct_reviewers_per_expected_skill": distinct_reviewer_coverage_complete,
            "no_unexpected_skill_ids": not extra_skill_ids,
            "no_duplicate_reviewer_assignments": not duplicate_reviewer_coverage,
            "reviews_match_current_skill_fingerprints": not fingerprint_mismatches
            if fingerprint_contract is not None
            else None,
            "valid": coverage_valid,
        },
        "quadratic_weighted_kappa": kappa,
        "reviewer_pair_kappas": pair_kappas,
        "agreement_threshold": 0.67,
        "pass_rule": (
            "预期 Skill 必须 100% 覆盖且无额外 Skill/重复 reviewer 行；每个预期 Skill "
            "至少两名不同复核者；正式验收时每行必须绑定当前 Skill SHA-256；"
            "五维均值均 >= 3.5；伤害标记为 0；kappa >= 0.67。"
        ),
        "passed": passed,
        "errors": errors,
        "skills": skill_results,
    }
