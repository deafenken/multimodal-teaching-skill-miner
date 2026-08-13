"""Durable learner metacognition and calibration records.

The assessment model's confidence is not a learner judgment-of-learning (JOL).
This module therefore records a learner self-report *before* an answer is
graded, then pairs it with an authoritative KC-v2 evidence row on the trusted
server boundary.  Callers cannot submit the actual outcome or a mastery delta.

Only opaque identifiers, hashes, controlled strategy codes, timestamps and
calibration aggregates are persisted.  Raw questions, answers and free-text
reflections are deliberately outside this store.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
from typing import Any, Callable, Iterator, Mapping, Sequence

from .io_utils import ensure_private_directory, ensure_private_file
from .teacher_agent_learning_records import learning_record_target

try:
    import fcntl
except ImportError:  # pragma: no cover - supported durable deployments are POSIX.
    fcntl = None  # type: ignore[assignment]


METACOGNITION_EVENT_SCHEMA = "teaching_skill_miner.metacognition_event.v1"
METACOGNITION_STORE_EVENT_SCHEMA = "teaching_skill_miner.metacognition_store_event.v1"
METACOGNITION_RECORD_SCHEMA = "teaching_skill_miner.metacognition_record.v1"
METACOGNITION_EXPORT_SCHEMA = "teaching_skill_miner.metacognition_export.v1"
METACOGNITION_ERASURE_SCHEMA = "teaching_skill_miner.metacognition_erasure.v1"

EXTERNAL_CALIBRATION_NOTE = (
    "This learner-facing calibration summary is descriptive and internally "
    "paired; external calibration and learning-effect validity are not established."
)

STRATEGY_CODES = frozenset(
    {
        "retrieval",
        "self_explanation",
        "decomposition",
        "worked_example",
        "analogy",
        "elimination",
        "diagram",
        "checking",
    }
)
ASSESSMENT_KINDS = frozenset({"verification", "transfer", "delayed_review"})
_OUTCOMES = {"correct": 100, "partial": 50, "incorrect": 0}
_SAFE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_LEARNER_KEY = re.compile(r"^learner_[0-9a-f]{64}$")
_CURRICULUM = re.compile(r"^curriculum_[0-9a-f]{64}$")
_KC = re.compile(r"^kc_[a-z0-9][a-z0-9_-]{2,80}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PREDICTION_ID = re.compile(r"^mcp_[0-9a-f]{64}$")
_PAIRING_ID = re.compile(r"^mco_[0-9a-f]{64}$")
_REVIEW_ID = re.compile(r"^review_[0-9a-f]{64}$")
_LEASE_ID = re.compile(r"^lease_[0-9a-f]{64}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
_TARGET_KEYS = frozenset(
    {
        "learner_key",
        "curriculum_namespace",
        "knowledge_component_id",
        "source_ref_sha256",
    }
)
_BINDING_KEYS = frozenset(
    {
        "attempt_id",
        "session_id_sha256",
        "assessment_kind",
        "item_id",
        "question_id",
        "rubric_id",
        "rubric_authority_sha256",
        "question_issued_at_utc",
        "review_id",
        "lease_id",
    }
)
_EVENT_KEYS = frozenset(
    {"schema", "event_id", "event_type", "target", "binding", "occurred_at_utc", "data"}
)


class MetacognitionError(ValueError):
    """Raised when an event violates the learner-metacognition contract."""


class MetacognitionStoreError(RuntimeError):
    """Raised when durable metacognition state cannot be trusted."""


class MetacognitionConflictError(MetacognitionStoreError):
    """Raised for a replay conflict, erased learner, or invalid pairing state."""


@dataclass(frozen=True, slots=True)
class MetacognitionApplyResult:
    event_id: str
    applied: bool
    event: dict[str, Any]
    calibration_record: dict[str, Any] | None


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MetacognitionError("metacognition values must be canonical JSON") from exc


def _digest(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _text(value: Any, *, field: str, maximum: int = 160) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
    ):
        raise MetacognitionError(f"{field} must be a trimmed string <= {maximum} chars")
    return value


def _safe_ref(value: Any, *, field: str) -> str:
    result = _text(value, field=field)
    if _SAFE_REF.fullmatch(result) is None:
        raise MetacognitionError(f"{field} must be an opaque safe identifier")
    return result


def _sha(value: Any, *, field: str) -> str:
    result = _text(value, field=field, maximum=64)
    if _SHA256.fullmatch(result) is None:
        raise MetacognitionError(f"{field} must be a SHA-256 digest")
    return result


def _parse_utc(value: Any, *, field: str) -> datetime:
    text = _text(value, field=field, maximum=40)
    if _UTC.fullmatch(text) is None:
        raise MetacognitionError(f"{field} must be an ISO-8601 UTC timestamp")
    try:
        return datetime.fromisoformat(text[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError as exc:
        raise MetacognitionError(f"{field} is not a real UTC timestamp") from exc


def _utc_text(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise MetacognitionError("clock must return a timezone-aware datetime")
    normalized = value.astimezone(timezone.utc)
    return normalized.isoformat(
        timespec="microseconds" if normalized.microsecond else "seconds"
    ).replace("+00:00", "Z")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _target(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _TARGET_KEYS:
        raise MetacognitionError("metacognition target must be a strict object")
    learner = _text(value.get("learner_key"), field="target.learner_key", maximum=72)
    curriculum = _text(
        value.get("curriculum_namespace"),
        field="target.curriculum_namespace",
        maximum=75,
    )
    kc_id = _text(
        value.get("knowledge_component_id"),
        field="target.knowledge_component_id",
        maximum=83,
    )
    source_hash = _sha(value.get("source_ref_sha256"), field="target.source_ref_sha256")
    if _LEARNER_KEY.fullmatch(learner) is None:
        raise MetacognitionError("target.learner_key must be server-minted")
    if _CURRICULUM.fullmatch(curriculum) is None or _KC.fullmatch(kc_id) is None:
        raise MetacognitionError("target curriculum or KC is invalid")
    return {
        "learner_key": learner,
        "curriculum_namespace": curriculum,
        "knowledge_component_id": kc_id,
        "source_ref_sha256": source_hash,
    }


def _binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _BINDING_KEYS:
        raise MetacognitionError("assessment binding must be a strict object")
    kind = value.get("assessment_kind")
    if kind not in ASSESSMENT_KINDS:
        raise MetacognitionError("assessment_kind is invalid")
    result: dict[str, Any] = {
        "attempt_id": _safe_ref(value.get("attempt_id"), field="binding.attempt_id"),
        "session_id_sha256": _sha(
            value.get("session_id_sha256"), field="binding.session_id_sha256"
        ),
        "assessment_kind": kind,
        "item_id": _safe_ref(value.get("item_id"), field="binding.item_id"),
        "question_id": _safe_ref(value.get("question_id"), field="binding.question_id"),
        "rubric_id": _safe_ref(value.get("rubric_id"), field="binding.rubric_id"),
        "rubric_authority_sha256": _sha(
            value.get("rubric_authority_sha256"),
            field="binding.rubric_authority_sha256",
        ),
        "question_issued_at_utc": _utc_text(
            _parse_utc(
                value.get("question_issued_at_utc"),
                field="binding.question_issued_at_utc",
            )
        ),
        "review_id": value.get("review_id"),
        "lease_id": value.get("lease_id"),
    }
    if kind == "delayed_review":
        if (
            not isinstance(result["review_id"], str)
            or _REVIEW_ID.fullmatch(result["review_id"]) is None
            or not isinstance(result["lease_id"], str)
            or _LEASE_ID.fullmatch(result["lease_id"]) is None
        ):
            raise MetacognitionError(
                "delayed review requires server review and lease IDs"
            )
    elif result["review_id"] is not None or result["lease_id"] is not None:
        raise MetacognitionError("only a delayed review may carry review lease IDs")
    return result


def _strategies(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MetacognitionError("strategy_codes must be a sequence")
    result = list(value)
    if not 1 <= len(result) <= 3 or len(set(result)) != len(result):
        raise MetacognitionError("choose one to three distinct strategy_codes")
    if any(not isinstance(item, str) or item not in STRATEGY_CODES for item in result):
        raise MetacognitionError("strategy_codes contains an unsupported strategy")
    return result


def _jol(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
        raise MetacognitionError("learner_jol_percent must be an integer from 0 to 100")
    return value


def _outcome_from_evidence(evidence: Mapping[str, Any]) -> tuple[str, int]:
    pair = (evidence.get("signal"), evidence.get("answer_alignment"))
    outcome = {
        ("correct", "aligned"): "correct",
        ("partial", "partially_aligned"): "partial",
        ("misconception", "contradicted"): "incorrect",
    }.get(pair)
    if outcome is None:
        raise MetacognitionError("authoritative evidence has no calibratable outcome")
    return outcome, _OUTCOMES[outcome]


def _classification(jol_percent: int, actual_percent: int) -> str:
    difference = jol_percent - actual_percent
    if difference >= 20:
        return "overconfident"
    if difference <= -20:
        return "underconfident"
    return "aligned"


def _feedback_message(classification: str) -> str:
    return {
        "overconfident": "这次自评把握高于权威结果。保留原评分标准，下一题先写出关键依据，再用所选策略检查一次。",
        "underconfident": "这次权威结果高于你的自评把握。保留原评分标准，可回看哪些策略真正帮助了你，并在相近题型中再次验证。",
        "aligned": "这次自评把握与权威结果较接近。保留原评分标准，并在迁移题或延迟复习中继续检验这种判断。",
    }[classification]


def build_metacognitive_prediction_event(
    *,
    learner_key: str,
    knowledge_component: Mapping[str, Any],
    attempt_id: str,
    session_id: str,
    assessment_kind: str,
    item_id: str,
    question_id: str,
    rubric_id: str,
    rubric_authority_sha256: str,
    question_issued_at_utc: str,
    captured_at_utc: str,
    learner_jol_percent: int,
    strategy_codes: Sequence[str],
    review_id: str | None = None,
    lease_id: str | None = None,
) -> dict[str, Any]:
    """Build a pre-answer self-report; no assessment/model confidence is accepted."""

    if knowledge_component.get("teacher_grading_authority_available") is not True:
        raise MetacognitionError("JOL requires a KC with teacher grading authority")
    try:
        target = _target(
            learning_record_target(
                learner_key=learner_key, knowledge_component=knowledge_component
            )
        )
    except (TypeError, ValueError) as exc:
        raise MetacognitionError("knowledge component target is invalid") from exc
    raw_session = _text(session_id, field="session_id", maximum=240)
    binding = _binding(
        {
            "attempt_id": attempt_id,
            "session_id_sha256": sha256(raw_session.encode("utf-8")).hexdigest(),
            "assessment_kind": assessment_kind,
            "item_id": item_id,
            "question_id": question_id,
            "rubric_id": rubric_id,
            "rubric_authority_sha256": rubric_authority_sha256,
            "question_issued_at_utc": question_issued_at_utc,
            "review_id": review_id,
            "lease_id": lease_id,
        }
    )
    captured = _utc_text(_parse_utc(captured_at_utc, field="captured_at_utc"))
    if _parse_utc(captured, field="captured_at_utc") < _parse_utc(
        binding["question_issued_at_utc"], field="question_issued_at_utc"
    ):
        raise MetacognitionError("JOL cannot predate question issuance")
    data = {
        "learner_jol_percent": _jol(learner_jol_percent),
        "strategy_codes": _strategies(strategy_codes),
        "confidence_source": "learner_self_report",
        "captured_before_answer": True,
        "external_calibration_established": False,
    }
    material = {"target": target, "binding": binding, "data": data}
    event = {
        "schema": METACOGNITION_EVENT_SCHEMA,
        "event_id": "mcp_" + _digest(material),
        "event_type": "prediction_recorded",
        "target": target,
        "binding": binding,
        "occurred_at_utc": captured,
        "data": data,
    }
    validate_metacognition_event(event)
    return event


def validate_metacognition_event(event: Mapping[str, Any]) -> None:
    if not isinstance(event, Mapping) or set(event) != _EVENT_KEYS:
        raise MetacognitionError("metacognition event must be a strict object")
    if event.get("schema") != METACOGNITION_EVENT_SCHEMA:
        raise MetacognitionError("metacognition event schema is unsupported")
    target = _target(event.get("target"))
    binding = _binding(event.get("binding"))
    occurred = _parse_utc(event.get("occurred_at_utc"), field="occurred_at_utc")
    data = event.get("data")
    if not isinstance(data, Mapping):
        raise MetacognitionError("metacognition event data must be an object")
    event_type = event.get("event_type")
    if event_type == "prediction_recorded":
        if set(data) != {
            "learner_jol_percent",
            "strategy_codes",
            "confidence_source",
            "captured_before_answer",
            "external_calibration_established",
        }:
            raise MetacognitionError("prediction data must be a strict object")
        jol = _jol(data.get("learner_jol_percent"))
        strategies = _strategies(data.get("strategy_codes"))
        if data.get("confidence_source") != "learner_self_report":
            raise MetacognitionError(
                "model assessment confidence is not learner confidence"
            )
        if data.get("captured_before_answer") is not True:
            raise MetacognitionError("prediction must be captured before the answer")
        if data.get("external_calibration_established") is not False:
            raise MetacognitionError("external calibration cannot be self-certified")
        expected = "mcp_" + _digest(
            {
                "target": target,
                "binding": binding,
                "data": {
                    "learner_jol_percent": jol,
                    "strategy_codes": strategies,
                    "confidence_source": "learner_self_report",
                    "captured_before_answer": True,
                    "external_calibration_established": False,
                },
            }
        )
        if event.get("event_id") != expected:
            raise MetacognitionError("prediction event_id does not bind its content")
        if occurred < _parse_utc(
            binding["question_issued_at_utc"], field="question_issued_at_utc"
        ):
            raise MetacognitionError("prediction predates question issuance")
        return
    if event_type == "outcome_paired":
        required = {
            "prediction_event_id",
            "prediction_event_sha256",
            "learner_jol_percent",
            "strategy_codes",
            "authoritative_evidence_id",
            "authoritative_evidence_sha256",
            "authoritative_outcome",
            "actual_score_percent",
            "calibration_classification",
            "commit_receipt_id",
            "scoring_standard_changed",
            "mastery_changed_by_metacognition",
            "external_calibration_established",
        }
        if set(data) != required:
            raise MetacognitionError("paired outcome data must be a strict object")
        prediction_id = _text(
            data.get("prediction_event_id"),
            field="data.prediction_event_id",
            maximum=68,
        )
        if _PREDICTION_ID.fullmatch(prediction_id) is None:
            raise MetacognitionError("prediction_event_id is invalid")
        _sha(data.get("prediction_event_sha256"), field="data.prediction_event_sha256")
        jol = _jol(data.get("learner_jol_percent"))
        _strategies(data.get("strategy_codes"))
        _safe_ref(
            data.get("authoritative_evidence_id"),
            field="data.authoritative_evidence_id",
        )
        _sha(
            data.get("authoritative_evidence_sha256"),
            field="data.authoritative_evidence_sha256",
        )
        outcome = data.get("authoritative_outcome")
        if (
            outcome not in _OUTCOMES
            or data.get("actual_score_percent") != _OUTCOMES[outcome]
        ):
            raise MetacognitionError(
                "actual score must be derived from the authoritative outcome"
            )
        if data.get("calibration_classification") != _classification(
            jol, _OUTCOMES[outcome]
        ):
            raise MetacognitionError("calibration classification is inconsistent")
        _safe_ref(data.get("commit_receipt_id"), field="data.commit_receipt_id")
        if data.get("scoring_standard_changed") is not False:
            raise MetacognitionError(
                "metacognition must not lower the scoring standard"
            )
        if data.get("mastery_changed_by_metacognition") is not False:
            raise MetacognitionError("metacognition is not mastery evidence")
        if data.get("external_calibration_established") is not False:
            raise MetacognitionError("external calibration cannot be self-certified")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or _PAIRING_ID.fullmatch(event_id) is None:
            raise MetacognitionError("paired outcome event_id is invalid")
        expected_event_id = "mco_" + _digest(
            {
                "prediction_event_sha256": data["prediction_event_sha256"],
                "authoritative_evidence_sha256": data["authoritative_evidence_sha256"],
                "commit_receipt_id": data["commit_receipt_id"],
            }
        )
        if event_id != expected_event_id:
            raise MetacognitionError(
                "paired outcome event_id does not bind its content"
            )
        return
    raise MetacognitionError("metacognition event type is unsupported")


def _component_key(target: Mapping[str, str]) -> str:
    return f"{target['curriculum_namespace']}:{target['knowledge_component_id']}"


def _empty_record(target: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema": METACOGNITION_RECORD_SCHEMA,
        **dict(target),
        "version": 0,
        "paired_count": 0,
        "jol_total_percent": 0,
        "actual_total_percent": 0,
        "squared_error_total": 0,
        "classification_counts": {
            "aligned": 0,
            "overconfident": 0,
            "underconfident": 0,
        },
        "strategy_statistics": {},
        "last_pairing": None,
        "external_calibration_established": False,
        "validity_note": EXTERNAL_CALIBRATION_NOTE,
    }


def _project_pairing(
    records: dict[str, dict[str, Any]], event: Mapping[str, Any]
) -> None:
    target = event["target"]
    learner = records.setdefault(str(target["learner_key"]), {})
    key = _component_key(target)
    record = learner.setdefault(key, _empty_record(target))
    data = event["data"]
    record["version"] += 1
    record["paired_count"] += 1
    jol = int(data["learner_jol_percent"])
    actual = int(data["actual_score_percent"])
    record["jol_total_percent"] += jol
    record["actual_total_percent"] += actual
    record["squared_error_total"] += (jol - actual) ** 2
    classification = str(data["calibration_classification"])
    record["classification_counts"][classification] += 1
    for strategy in data["strategy_codes"]:
        stats = record["strategy_statistics"].setdefault(
            strategy, {"paired_count": 0, "actual_total_percent": 0}
        )
        stats["paired_count"] += 1
        stats["actual_total_percent"] += actual
    record["last_pairing"] = {
        "event_id": event["event_id"],
        "paired_at_utc": event["occurred_at_utc"],
        "assessment_kind": event["binding"]["assessment_kind"],
        "authoritative_outcome": data["authoritative_outcome"],
        "calibration_classification": classification,
    }


def _public_record(record: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(record))
    count = int(result["paired_count"])
    result["mean_learner_jol_percent"] = (
        round(result["jol_total_percent"] / count, 2) if count else None
    )
    result["mean_authoritative_score_percent"] = (
        round(result["actual_total_percent"] / count, 2) if count else None
    )
    result["signed_calibration_bias_points"] = (
        round((result["jol_total_percent"] - result["actual_total_percent"]) / count, 2)
        if count
        else None
    )
    result["brier_score"] = (
        round(result["squared_error_total"] / count / 10_000, 6) if count else None
    )
    for stats in result["strategy_statistics"].values():
        stats["mean_authoritative_score_percent"] = round(
            stats["actual_total_percent"] / stats["paired_count"], 2
        )
    return result


class MetacognitionStore:
    """Private append-only hash-chain store with erasure fencing and replay."""

    def __init__(
        self,
        path: str | Path,
        *,
        authoritative_evidence_resolver: Callable[[str], Mapping[str, Any]]
        | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")
        self.tombstone_path = self.path.with_name(f".{self.path.name}.erased.json")
        self._clock = clock or _now
        self._authoritative_evidence_resolver = authoritative_evidence_resolver
        self._thread_lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._index: dict[str, dict[str, Any]] = {}
        self._records: dict[str, dict[str, Any]] = {}
        self._last_hash: str | None = None
        self._erased: set[str] = set()
        with self._thread_lock, self._process_lock():
            self._load_erasure_locked()
            self._load_locked(repair_tail=True)

    def _clock_text(self) -> str:
        return _utc_text(self._clock())

    @contextmanager
    def _process_lock(self) -> Iterator[None]:
        if fcntl is None:
            raise MetacognitionStoreError(
                "metacognition durability requires POSIX locking"
            )
        ensure_private_directory(self.path.parent)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.lock_path, flags, 0o600)
            os.fchmod(descriptor, 0o600)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise MetacognitionStoreError(
                    "metacognition lock must be a regular file"
                )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        except OSError as exc:
            raise MetacognitionStoreError("metacognition store lock failed") from exc
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except (OSError, UnboundLocalError):
                pass
            try:
                os.close(descriptor)
            except (OSError, UnboundLocalError):
                pass

    def _load_erasure_locked(self) -> None:
        if not self.tombstone_path.exists():
            self._erased = set()
            return
        try:
            info = self.tombstone_path.lstat()
            raw = self.tombstone_path.read_bytes()
            value = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MetacognitionStoreError(
                "metacognition erasure fence is unreadable"
            ) from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise MetacognitionStoreError("metacognition erasure fence must be regular")
        if raw != _canonical_bytes(value) + b"\n" or not isinstance(value, Mapping):
            raise MetacognitionStoreError(
                "metacognition erasure fence is non-canonical"
            )
        if (
            set(value) != {"schema", "learner_keys"}
            or value.get("schema") != METACOGNITION_ERASURE_SCHEMA
        ):
            raise MetacognitionStoreError(
                "metacognition erasure fence schema is invalid"
            )
        keys = value.get("learner_keys")
        if not isinstance(keys, list) or keys != sorted(set(keys)):
            raise MetacognitionStoreError("metacognition erasure keys are invalid")
        if any(
            not isinstance(key, str) or _LEARNER_KEY.fullmatch(key) is None
            for key in keys
        ):
            raise MetacognitionStoreError(
                "metacognition erasure learner key is invalid"
            )
        self._erased = set(keys)
        ensure_private_file(self.tombstone_path)

    def _load_locked(self, *, repair_tail: bool) -> None:
        if not self.path.exists():
            self._events, self._index, self._records, self._last_hash = [], {}, {}, None
            return
        info = self.path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise MetacognitionStoreError("metacognition store must be a regular file")
        try:
            with self.path.open("r+b") as stream:
                raw = stream.read()
                end = raw.rfind(b"\n") + 1
                if end != len(raw):
                    if not repair_tail:
                        raise MetacognitionStoreError(
                            "metacognition store has a truncated tail"
                        )
                    stream.seek(end)
                    stream.truncate(end)
                    stream.flush()
                    os.fsync(stream.fileno())
        except OSError as exc:
            raise MetacognitionStoreError("metacognition store cannot be read") from exc
        envelopes: list[dict[str, Any]] = []
        index: dict[str, dict[str, Any]] = {}
        records: dict[str, dict[str, Any]] = {}
        previous: str | None = None
        predictions_by_attempt: dict[str, str] = {}
        for seq, line in enumerate(raw[:end].splitlines(), 1):
            try:
                envelope = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise MetacognitionStoreError(
                    f"metacognition line {seq} is invalid JSON"
                ) from exc
            if line != _canonical_bytes(envelope) or not isinstance(envelope, Mapping):
                raise MetacognitionStoreError(
                    f"metacognition line {seq} is non-canonical"
                )
            if set(envelope) != {
                "schema",
                "seq",
                "recorded_at_utc",
                "previous_hash",
                "event",
                "event_sha256",
                "hash",
            }:
                raise MetacognitionStoreError(
                    f"metacognition line {seq} has invalid shape"
                )
            material = dict(envelope)
            claimed_hash = material.pop("hash", None)
            if (
                envelope.get("schema") != METACOGNITION_STORE_EVENT_SCHEMA
                or envelope.get("seq") != seq
                or envelope.get("previous_hash") != previous
                or claimed_hash != _digest(material)
            ):
                raise MetacognitionStoreError(
                    f"metacognition line {seq} hash chain failed"
                )
            event = envelope.get("event")
            try:
                validate_metacognition_event(event)
            except MetacognitionError as exc:
                raise MetacognitionStoreError(
                    f"metacognition line {seq} event failed"
                ) from exc
            if envelope.get("event_sha256") != _digest(event):
                raise MetacognitionStoreError(
                    f"metacognition line {seq} event hash failed"
                )
            if _parse_utc(
                event["occurred_at_utc"], field="occurred_at_utc"
            ) > _parse_utc(envelope["recorded_at_utc"], field="recorded_at_utc"):
                raise MetacognitionStoreError(
                    f"metacognition line {seq} is future-dated"
                )
            event_id = str(event["event_id"])
            if event_id in index:
                raise MetacognitionStoreError(
                    f"metacognition line {seq} duplicates an event"
                )
            attempt_key = _digest(
                {"target": event["target"], "binding": event["binding"]}
            )
            if event["event_type"] == "prediction_recorded":
                if attempt_key in predictions_by_attempt:
                    raise MetacognitionStoreError(
                        f"metacognition line {seq} duplicates an attempt"
                    )
                predictions_by_attempt[attempt_key] = event_id
            else:
                prediction_id = event["data"]["prediction_event_id"]
                prediction_envelope = index.get(prediction_id)
                if prediction_envelope is None:
                    raise MetacognitionStoreError(
                        f"metacognition line {seq} has no prediction"
                    )
                prediction = prediction_envelope["event"]
                if event["data"]["prediction_event_sha256"] != _digest(prediction):
                    raise MetacognitionStoreError(
                        f"metacognition line {seq} prediction hash failed"
                    )
                if (
                    event["target"] != prediction["target"]
                    or event["binding"] != prediction["binding"]
                ):
                    raise MetacognitionStoreError(
                        f"metacognition line {seq} crosses an attempt"
                    )
                if any(
                    prior["event"]["event_type"] == "outcome_paired"
                    and prior["event"]["data"]["prediction_event_id"] == prediction_id
                    for prior in envelopes
                ):
                    raise MetacognitionStoreError(
                        f"metacognition line {seq} pairs twice"
                    )
                if _parse_utc(
                    event["occurred_at_utc"], field="occurred_at_utc"
                ) < _parse_utc(
                    prediction["occurred_at_utc"], field="prediction.occurred_at_utc"
                ):
                    raise MetacognitionStoreError(
                        f"metacognition line {seq} predates prediction"
                    )
                _project_pairing(records, event)
            copied = deepcopy(dict(envelope))
            envelopes.append(copied)
            index[event_id] = copied
            previous = str(claimed_hash)
        self._events, self._index, self._records, self._last_hash = (
            envelopes,
            index,
            records,
            previous,
        )
        ensure_private_file(self.path)

    def _append_locked(self, event: Mapping[str, Any]) -> MetacognitionApplyResult:
        validate_metacognition_event(event)
        learner_key = str(event["target"]["learner_key"])
        if learner_key in self._erased:
            raise MetacognitionConflictError(
                "learner metacognition was permanently erased"
            )
        event_id = str(event["event_id"])
        existing = self._index.get(event_id)
        if existing is not None:
            if existing["event_sha256"] != _digest(event):
                raise MetacognitionConflictError(
                    "event_id replayed with different content"
                )
            return MetacognitionApplyResult(
                event_id, False, deepcopy(dict(event)), self._record_for_event(event)
            )
        if event["event_type"] == "prediction_recorded":
            for envelope in self._events:
                prior = envelope["event"]
                if (
                    prior["event_type"] == "prediction_recorded"
                    and prior["target"] == event["target"]
                    and prior["binding"] == event["binding"]
                ):
                    raise MetacognitionConflictError(
                        "attempt already has a different prediction"
                    )
        recorded = self._clock_text()
        if _parse_utc(event["occurred_at_utc"], field="occurred_at_utc") > _parse_utc(
            recorded, field="recorded_at_utc"
        ):
            raise MetacognitionConflictError(
                "metacognition event cannot occur in the future"
            )
        envelope: dict[str, Any] = {
            "schema": METACOGNITION_STORE_EVENT_SCHEMA,
            "seq": len(self._events) + 1,
            "recorded_at_utc": recorded,
            "previous_hash": self._last_hash,
            "event": deepcopy(dict(event)),
            "event_sha256": _digest(event),
        }
        envelope["hash"] = _digest(envelope)
        payload = _canonical_bytes(envelope) + b"\n"
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.path, flags, 0o600)
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "ab") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise MetacognitionStoreError(
                "metacognition event could not be committed"
            ) from exc
        self._load_locked(repair_tail=False)
        return MetacognitionApplyResult(
            event_id, True, deepcopy(dict(event)), self._record_for_event(event)
        )

    def _record_for_event(self, event: Mapping[str, Any]) -> dict[str, Any] | None:
        record = self._records.get(str(event["target"]["learner_key"]), {}).get(
            _component_key(event["target"])
        )
        return None if record is None else _public_record(record)

    def record_prediction(self, event: Mapping[str, Any]) -> MetacognitionApplyResult:
        if event.get("event_type") != "prediction_recorded":
            raise MetacognitionError("record_prediction accepts only prediction events")
        with self._thread_lock, self._process_lock():
            self._load_erasure_locked()
            self._load_locked(repair_tail=True)
            return self._append_locked(event)

    def pair_authoritative_outcome(
        self,
        *,
        prediction_event_id: str,
        knowledge_component: Mapping[str, Any],
        authoritative_evidence_id: str,
        authoritative_evidence_sha256: str,
        committed_at_utc: str,
        commit_receipt_id: str,
    ) -> MetacognitionApplyResult:
        """Pair one prediction; the actual result is derived, never supplied."""

        if (
            not isinstance(prediction_event_id, str)
            or _PREDICTION_ID.fullmatch(prediction_event_id) is None
        ):
            raise MetacognitionError("prediction_event_id is invalid")
        with self._thread_lock, self._process_lock():
            self._load_erasure_locked()
            self._load_locked(repair_tail=True)
            prediction_envelope = self._index.get(prediction_event_id)
            if prediction_envelope is None:
                raise MetacognitionConflictError("prediction does not exist")
            prediction = prediction_envelope["event"]
            if any(
                item["event"]["event_type"] == "outcome_paired"
                and item["event"]["data"]["prediction_event_id"] == prediction_event_id
                for item in self._events
            ):
                paired = next(
                    item["event"]
                    for item in self._events
                    if item["event"]["event_type"] == "outcome_paired"
                    and item["event"]["data"]["prediction_event_id"]
                    == prediction_event_id
                )
                existing_data = paired["data"]
                if (
                    existing_data["authoritative_evidence_id"]
                    != authoritative_evidence_id
                    or existing_data["authoritative_evidence_sha256"]
                    != authoritative_evidence_sha256
                    or existing_data["commit_receipt_id"] != commit_receipt_id
                    or paired["occurred_at_utc"] != committed_at_utc
                ):
                    raise MetacognitionConflictError(
                        "prediction replay requested a different authoritative pairing"
                    )
                return MetacognitionApplyResult(
                    str(paired["event_id"]),
                    False,
                    deepcopy(paired),
                    self._record_for_event(paired),
                )
            if (
                knowledge_component.get("teacher_grading_authority_available")
                is not True
            ):
                raise MetacognitionError(
                    "outcome pairing requires teacher grading authority"
                )
            target = _target(
                learning_record_target(
                    learner_key=prediction["target"]["learner_key"],
                    knowledge_component=knowledge_component,
                )
            )
            if target != prediction["target"]:
                raise MetacognitionConflictError(
                    "outcome crosses a KC or curriculum boundary"
                )
            evidence_id_request = _safe_ref(
                authoritative_evidence_id, field="authoritative_evidence_id"
            )
            evidence_hash_request = _sha(
                authoritative_evidence_sha256,
                field="authoritative_evidence_sha256",
            )
            if self._authoritative_evidence_resolver is None:
                raise MetacognitionStoreError(
                    "authoritative evidence resolver is required for outcome pairing"
                )
            try:
                resolved = self._authoritative_evidence_resolver(evidence_id_request)
            except Exception as exc:
                raise MetacognitionStoreError(
                    "authoritative evidence could not be resolved"
                ) from exc
            if not isinstance(resolved, Mapping):
                raise MetacognitionStoreError(
                    "authoritative evidence resolver returned an invalid object"
                )
            evidence = deepcopy(dict(resolved))
            binding = prediction["binding"]
            for evidence_field, binding_field in (
                ("knowledge_component_id", "knowledge_component_id"),
                ("item_id", "item_id"),
                ("question_id", "question_id"),
                ("rubric_id", "rubric_id"),
            ):
                expected = (
                    target[binding_field]
                    if binding_field == "knowledge_component_id"
                    else binding[binding_field]
                )
                if evidence.get(evidence_field) != expected:
                    raise MetacognitionConflictError(
                        f"evidence {evidence_field} crosses the prediction binding"
                    )
            resolved_rubric_authority = evidence.get("rubric_authority_sha256")
            if resolved_rubric_authority is not None and (
                resolved_rubric_authority != binding["rubric_authority_sha256"]
            ):
                raise MetacognitionConflictError(
                    "rubric authority changed after the prediction"
                )
            if (
                evidence.get("assessment_eligible") is not True
                or evidence.get("authoritative") is not True
            ):
                raise MetacognitionError(
                    "only authoritative assessment evidence can pair a JOL"
                )
            outcome, actual = _outcome_from_evidence(evidence)
            evidence_id = _safe_ref(
                evidence.get("evidence_id"), field="evidence.evidence_id"
            )
            evidence_hash = _sha(
                evidence.get("evidence_fingerprint"),
                field="evidence.evidence_fingerprint",
            )
            if (
                evidence_id != evidence_id_request
                or evidence_hash != evidence_hash_request
            ):
                raise MetacognitionConflictError(
                    "resolved evidence identity does not match the requested authoritative receipt"
                )
            paired_at = _utc_text(
                _parse_utc(committed_at_utc, field="committed_at_utc")
            )
            if _parse_utc(paired_at, field="committed_at_utc") < _parse_utc(
                prediction["occurred_at_utc"], field="prediction.occurred_at_utc"
            ):
                raise MetacognitionConflictError(
                    "authoritative outcome predates the prediction"
                )
            receipt = _safe_ref(commit_receipt_id, field="commit_receipt_id")
            jol = int(prediction["data"]["learner_jol_percent"])
            data = {
                "prediction_event_id": prediction_event_id,
                "prediction_event_sha256": _digest(prediction),
                "learner_jol_percent": jol,
                "strategy_codes": deepcopy(prediction["data"]["strategy_codes"]),
                "authoritative_evidence_id": evidence_id,
                "authoritative_evidence_sha256": evidence_hash,
                "authoritative_outcome": outcome,
                "actual_score_percent": actual,
                "calibration_classification": _classification(jol, actual),
                "commit_receipt_id": receipt,
                "scoring_standard_changed": False,
                "mastery_changed_by_metacognition": False,
                "external_calibration_established": False,
            }
            event: dict[str, Any] = {
                "schema": METACOGNITION_EVENT_SCHEMA,
                "event_id": "mco_"
                + _digest(
                    {
                        "prediction_event_sha256": data["prediction_event_sha256"],
                        "authoritative_evidence_sha256": evidence_hash,
                        "commit_receipt_id": receipt,
                    }
                ),
                "event_type": "outcome_paired",
                "target": deepcopy(target),
                "binding": deepcopy(binding),
                "occurred_at_utc": paired_at,
                "data": data,
            }
            return self._append_locked(event)

    def get_prediction_event(self, prediction_event_id: str) -> dict[str, Any] | None:
        """Return one content-bounded prediction for trusted server adapters."""

        if (
            not isinstance(prediction_event_id, str)
            or _PREDICTION_ID.fullmatch(prediction_event_id) is None
        ):
            raise MetacognitionError("prediction_event_id is invalid")
        with self._thread_lock, self._process_lock():
            self._load_erasure_locked()
            self._load_locked(repair_tail=True)
            envelope = self._index.get(prediction_event_id)
            if envelope is None:
                return None
            event = envelope["event"]
            if event["event_type"] != "prediction_recorded":
                return None
            if event["target"]["learner_key"] in self._erased:
                return None
            return deepcopy(dict(event))

    def list_session_predictions(
        self, *, learner_key: str, session_id: str
    ) -> tuple[dict[str, Any], ...]:
        """Return a content-bounded, restart-safe learner session projection.

        The projection deliberately omits the learner key, source reference,
        rubric authority material, raw answers, and assessor confidence.  A
        paired result is included only after it was derived from authoritative
        evidence by :meth:`pair_authoritative_outcome`.
        """

        learner = _text(learner_key, field="learner_key", maximum=72)
        if _LEARNER_KEY.fullmatch(learner) is None:
            raise MetacognitionError("learner_key is invalid")
        raw_session = _text(session_id, field="session_id", maximum=240)
        session_sha256 = sha256(raw_session.encode("utf-8")).hexdigest()
        with self._thread_lock, self._process_lock():
            self._load_erasure_locked()
            self._load_locked(repair_tail=True)
            if learner in self._erased:
                return ()
            pairings = {
                str(item["event"]["data"]["prediction_event_id"]): item["event"]
                for item in self._events
                if item["event"]["event_type"] == "outcome_paired"
                and item["event"]["target"]["learner_key"] == learner
                and item["event"]["binding"]["session_id_sha256"] == session_sha256
            }
            projected: list[dict[str, Any]] = []
            for item in self._events:
                event = item["event"]
                if (
                    event["event_type"] != "prediction_recorded"
                    or event["target"]["learner_key"] != learner
                    or event["binding"]["session_id_sha256"] != session_sha256
                ):
                    continue
                binding = event["binding"]
                data = event["data"]
                pairing = pairings.get(str(event["event_id"]))
                projection: dict[str, Any] = {
                    "prediction_event_id": event["event_id"],
                    "occurred_at_utc": event["occurred_at_utc"],
                    "knowledge_component_id": event["target"]["knowledge_component_id"],
                    "assessment_kind": binding["assessment_kind"],
                    "item_id": binding["item_id"],
                    "question_id": binding["question_id"],
                    "review_id": binding["review_id"],
                    "lease_id": binding["lease_id"],
                    "learner_jol_percent": data["learner_jol_percent"],
                    "strategy_codes": deepcopy(data["strategy_codes"]),
                    "status": "pending" if pairing is None else "paired",
                    "outcome_accepted_from_client": False,
                    "mastery_changed": False,
                }
                if pairing is not None:
                    paired = pairing["data"]
                    classification = str(paired["calibration_classification"])
                    projection["pairing_event_id"] = pairing["event_id"]
                    projection["feedback"] = {
                        "authoritative_outcome": paired["authoritative_outcome"],
                        "actual_score_percent": paired["actual_score_percent"],
                        "calibration_classification": classification,
                        "message_zh": _feedback_message(classification),
                        "scoring_standard_changed": False,
                        "mastery_changed_by_metacognition": False,
                        "external_calibration_established": False,
                    }
                projected.append(projection)
            return tuple(deepcopy(projected))

    def get_calibration_record(
        self,
        *,
        learner_key: str,
        curriculum_namespace: str,
        knowledge_component_id: str,
    ) -> dict[str, Any] | None:
        learner = _text(learner_key, field="learner_key", maximum=72)
        if _LEARNER_KEY.fullmatch(learner) is None:
            raise MetacognitionError("learner_key is invalid")
        stub = _target(
            {
                "learner_key": learner,
                "curriculum_namespace": curriculum_namespace,
                "knowledge_component_id": knowledge_component_id,
                "source_ref_sha256": "0" * 64,
            }
        )
        with self._thread_lock, self._process_lock():
            self._load_erasure_locked()
            self._load_locked(repair_tail=True)
            if learner in self._erased:
                return None
            record = self._records.get(learner, {}).get(_component_key(stub))
            return None if record is None else _public_record(record)

    def feedback_for_pairing(self, pairing_event_id: str) -> dict[str, Any]:
        if (
            not isinstance(pairing_event_id, str)
            or _PAIRING_ID.fullmatch(pairing_event_id) is None
        ):
            raise MetacognitionError("pairing_event_id is invalid")
        with self._thread_lock, self._process_lock():
            self._load_erasure_locked()
            self._load_locked(repair_tail=True)
            envelope = self._index.get(pairing_event_id)
            if envelope is None or envelope["event"]["event_type"] != "outcome_paired":
                raise MetacognitionConflictError("paired outcome does not exist")
            event = envelope["event"]
            if event["target"]["learner_key"] in self._erased:
                raise MetacognitionConflictError("learner metacognition was erased")
            data = event["data"]
            classification = data["calibration_classification"]
            return {
                "schema": "teaching_skill_miner.metacognition_feedback.v1",
                "pairing_event_id": pairing_event_id,
                "authoritative_outcome": data["authoritative_outcome"],
                "calibration_classification": classification,
                "strategy_codes": deepcopy(data["strategy_codes"]),
                "message_zh": _feedback_message(classification),
                "scoring_standard_changed": False,
                "mastery_changed_by_metacognition": False,
                "external_calibration_established": False,
                "validity_note": EXTERNAL_CALIBRATION_NOTE,
            }

    def export_learner(self, learner_key: str) -> dict[str, Any] | None:
        learner = _text(learner_key, field="learner_key", maximum=72)
        if _LEARNER_KEY.fullmatch(learner) is None:
            raise MetacognitionError("learner_key is invalid")
        with self._thread_lock, self._process_lock():
            self._load_erasure_locked()
            self._load_locked(repair_tail=True)
            if learner in self._erased:
                return None
            events = [
                deepcopy(item["event"])
                for item in self._events
                if item["event"]["target"]["learner_key"] == learner
            ]
            if not events:
                return None
            return {
                "schema": METACOGNITION_EXPORT_SCHEMA,
                "learner_key": learner,
                "events": events,
                "calibration_records": [
                    _public_record(record)
                    for _, record in sorted(self._records.get(learner, {}).items())
                ],
                "external_calibration_established": False,
                "validity_note": EXTERNAL_CALIBRATION_NOTE,
            }

    def purge_learner(self, learner_key: str) -> dict[str, int]:
        learner = _text(learner_key, field="learner_key", maximum=72)
        if _LEARNER_KEY.fullmatch(learner) is None:
            raise MetacognitionError("learner_key is invalid")
        with self._thread_lock, self._process_lock():
            self._load_erasure_locked()
            self._load_locked(repair_tail=True)
            was_erased = learner in self._erased
            erased = set(self._erased)
            erased.add(learner)
            self._atomic_replace(
                self.tombstone_path,
                _canonical_bytes(
                    {
                        "schema": METACOGNITION_ERASURE_SCHEMA,
                        "learner_keys": sorted(erased),
                    }
                )
                + b"\n",
            )
            retained = [
                deepcopy(item["event"])
                for item in self._events
                if item["event"]["target"]["learner_key"] != learner
            ]
            removed = len(self._events) - len(retained)
            payload = b""
            previous: str | None = None
            for seq, event in enumerate(retained, 1):
                envelope: dict[str, Any] = {
                    "schema": METACOGNITION_STORE_EVENT_SCHEMA,
                    "seq": seq,
                    "recorded_at_utc": event["occurred_at_utc"],
                    "previous_hash": previous,
                    "event": event,
                    "event_sha256": _digest(event),
                }
                envelope["hash"] = _digest(envelope)
                previous = envelope["hash"]
                payload += _canonical_bytes(envelope) + b"\n"
            self._atomic_replace(self.path, payload)
            self._load_erasure_locked()
            self._load_locked(repair_tail=False)
            return {
                "learners": 0 if was_erased and removed == 0 else 1,
                "events": removed,
                "erasure_fence": 1,
            }

    def _atomic_replace(self, target: Path, payload: bytes) -> None:
        ensure_private_directory(target.parent)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.replace-", suffix=".tmp", dir=target.parent
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            ensure_private_file(target)
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


__all__ = [
    "ASSESSMENT_KINDS",
    "EXTERNAL_CALIBRATION_NOTE",
    "METACOGNITION_EVENT_SCHEMA",
    "METACOGNITION_EXPORT_SCHEMA",
    "METACOGNITION_RECORD_SCHEMA",
    "METACOGNITION_STORE_EVENT_SCHEMA",
    "STRATEGY_CODES",
    "MetacognitionApplyResult",
    "MetacognitionConflictError",
    "MetacognitionError",
    "MetacognitionStore",
    "MetacognitionStoreError",
    "build_metacognitive_prediction_event",
    "validate_metacognition_event",
]
