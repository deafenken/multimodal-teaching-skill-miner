from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

import pytest

from teaching_skill_miner.teacher_agent_learning_records import mint_learner_key
from teaching_skill_miner.teacher_agent_data_rights import (
    build_project_export_archive,
    validate_project_export_archive,
)
from teaching_skill_miner.teacher_agent_metacognition import (
    EXTERNAL_CALIBRATION_NOTE,
    METACOGNITION_EVENT_SCHEMA,
    MetacognitionConflictError,
    MetacognitionError,
    MetacognitionStore,
    MetacognitionStoreError,
    build_metacognitive_prediction_event,
    validate_metacognition_event,
)


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 12, 6, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **kwargs: float) -> None:
        self.value += timedelta(**kwargs)

    def text(self) -> str:
        return self.value.isoformat().replace("+00:00", "Z")


@pytest.fixture()
def fixture() -> dict[str, object]:
    clock = Clock()
    learner = mint_learner_key(
        "student-meta-1",
        tenant_id="school-meta",
        secret=b"metacognition-test-secret-material-32bytes",
    )
    component = {
        "kc_id": "kc_state_definition",
        "source_ref": "syllabus-a:lesson-1#kc_state_definition",
        "teacher_grading_authority_available": True,
    }
    rubric_hash = "a" * 64
    evidence: dict[str, dict[str, object]] = {}

    def make_evidence(
        suffix: str,
        *,
        outcome: str = "correct",
        kc_id: str = "kc_state_definition",
        item_id: str = "item.dp.1",
        question_id: str = "question.dp.1",
        rubric_id: str = "rubric.dp.1",
        authority_hash: str = rubric_hash,
    ) -> dict[str, object]:
        signal, alignment = {
            "correct": ("correct", "aligned"),
            "partial": ("partial", "partially_aligned"),
            "incorrect": ("misconception", "contradicted"),
        }[outcome]
        material = {
            "suffix": suffix,
            "outcome": outcome,
            "kc_id": kc_id,
            "item_id": item_id,
        }
        row: dict[str, object] = {
            "evidence_id": f"evidence.{suffix}",
            "item_id": item_id,
            "question_id": question_id,
            "rubric_id": rubric_id,
            "rubric_authority_sha256": authority_hash,
            "knowledge_component_id": kc_id,
            "signal": signal,
            "answer_alignment": alignment,
            "assessment_eligible": True,
            "authoritative": True,
            "evidence_fingerprint": sha256(
                json.dumps(material, sort_keys=True).encode()
            ).hexdigest(),
            "learner_response": "raw answer must never enter metacognition records",
            "assessment_confidence": 0.99,
        }
        evidence[str(row["evidence_id"])] = row
        return row

    return {
        "clock": clock,
        "learner": learner,
        "component": component,
        "rubric_hash": rubric_hash,
        "evidence": evidence,
        "make_evidence": make_evidence,
    }


def prediction(
    fx: dict[str, object],
    *,
    attempt: str = "attempt.dp.1",
    session: str = "session-one",
    kind: str = "verification",
    jol: int = 80,
    strategies: tuple[str, ...] = ("self_explanation",),
    review_id: str | None = None,
    lease_id: str | None = None,
) -> dict[str, object]:
    clock = fx["clock"]
    assert isinstance(clock, Clock)
    return build_metacognitive_prediction_event(
        learner_key=str(fx["learner"]),
        knowledge_component=fx["component"],  # type: ignore[arg-type]
        attempt_id=attempt,
        session_id=session,
        assessment_kind=kind,
        item_id="item.dp.1",
        question_id="question.dp.1",
        rubric_id="rubric.dp.1",
        rubric_authority_sha256=str(fx["rubric_hash"]),
        question_issued_at_utc=clock.text(),
        captured_at_utc=clock.text(),
        learner_jol_percent=jol,
        strategy_codes=strategies,
        review_id=review_id,
        lease_id=lease_id,
    )


def store_for(path: Path, fx: dict[str, object]) -> MetacognitionStore:
    evidence = fx["evidence"]
    assert isinstance(evidence, dict)
    clock = fx["clock"]
    assert isinstance(clock, Clock)
    return MetacognitionStore(
        path,
        authoritative_evidence_resolver=lambda evidence_id: deepcopy(
            evidence[evidence_id]
        ),
        clock=clock,
    )


def pair(
    store: MetacognitionStore,
    fx: dict[str, object],
    prediction_event: dict[str, object],
    *,
    evidence_suffix: str,
    outcome: str,
):
    make_evidence = fx["make_evidence"]
    assert callable(make_evidence)
    row = make_evidence(evidence_suffix, outcome=outcome)
    clock = fx["clock"]
    assert isinstance(clock, Clock)
    clock.advance(seconds=1)
    return store.pair_authoritative_outcome(
        prediction_event_id=str(prediction_event["event_id"]),
        knowledge_component=fx["component"],  # type: ignore[arg-type]
        authoritative_evidence_id=str(row["evidence_id"]),
        authoritative_evidence_sha256=str(row["evidence_fingerprint"]),
        committed_at_utc=clock.text(),
        commit_receipt_id=f"turn_committed:{evidence_suffix}",
    )


def test_prediction_is_pre_answer_learner_report_not_model_confidence(
    fixture: dict[str, object],
) -> None:
    event = prediction(fixture, jol=73, strategies=("retrieval", "checking"))
    validate_metacognition_event(event)
    assert event["schema"] == METACOGNITION_EVENT_SCHEMA
    assert event["data"] == {
        "learner_jol_percent": 73,
        "strategy_codes": ["retrieval", "checking"],
        "confidence_source": "learner_self_report",
        "captured_before_answer": True,
        "external_calibration_established": False,
    }
    forged = deepcopy(event)
    forged["data"]["confidence_source"] = "assessment_model"
    with pytest.raises(MetacognitionError, match="model assessment confidence"):
        validate_metacognition_event(forged)
    forged = deepcopy(event)
    forged["data"]["outcome"] = "correct"
    with pytest.raises(MetacognitionError, match="strict object"):
        validate_metacognition_event(forged)


def test_prediction_requires_authority_controlled_strategy_and_real_pre_answer_time(
    fixture: dict[str, object],
) -> None:
    no_authority = {
        **fixture["component"],
        "teacher_grading_authority_available": False,
    }  # type: ignore[arg-type]
    fixture["component"] = no_authority
    with pytest.raises(MetacognitionError, match="grading authority"):
        prediction(fixture)
    fixture["component"] = {**no_authority, "teacher_grading_authority_available": True}
    with pytest.raises(MetacognitionError, match="unsupported strategy"):
        prediction(fixture, strategies=("free_text_strategy",))

    clock = fixture["clock"]
    assert isinstance(clock, Clock)
    issued = clock.text()
    earlier = (clock.value - timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    with pytest.raises(MetacognitionError, match="predate question"):
        build_metacognitive_prediction_event(
            learner_key=str(fixture["learner"]),
            knowledge_component=fixture["component"],  # type: ignore[arg-type]
            attempt_id="attempt.time",
            session_id="session-time",
            assessment_kind="verification",
            item_id="item.dp.1",
            question_id="question.dp.1",
            rubric_id="rubric.dp.1",
            rubric_authority_sha256=str(fixture["rubric_hash"]),
            question_issued_at_utc=issued,
            captured_at_utc=earlier,
            learner_jol_percent=50,
            strategy_codes=["retrieval"],
        )


def test_delayed_review_boundary_requires_server_review_and_lease(
    fixture: dict[str, object],
) -> None:
    with pytest.raises(MetacognitionError, match="requires server review"):
        prediction(fixture, kind="delayed_review")
    event = prediction(
        fixture,
        kind="delayed_review",
        review_id="review_" + "b" * 64,
        lease_id="lease_" + "c" * 64,
    )
    assert event["binding"]["assessment_kind"] == "delayed_review"
    with pytest.raises(MetacognitionError, match="only a delayed review"):
        prediction(
            fixture,
            kind="transfer",
            review_id="review_" + "b" * 64,
            lease_id="lease_" + "c" * 64,
        )


def test_durable_pairing_is_idempotent_and_ignores_assessment_confidence(
    fixture: dict[str, object],
) -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "metacognition.jsonl"
        store = store_for(path, fixture)
        pred = prediction(fixture, jol=90)
        first_prediction = store.record_prediction(pred)
        assert first_prediction.applied is True
        assert store.record_prediction(pred).applied is False
        result = pair(store, fixture, pred, evidence_suffix="one", outcome="incorrect")
        assert result.applied is True
        assert result.event["data"]["authoritative_outcome"] == "incorrect"
        assert result.event["data"]["actual_score_percent"] == 0
        assert result.event["data"]["calibration_classification"] == "overconfident"
        assert result.event["data"]["scoring_standard_changed"] is False
        assert result.event["data"]["mastery_changed_by_metacognition"] is False

        restarted = store_for(path, fixture)
        replay = restarted.pair_authoritative_outcome(
            prediction_event_id=str(pred["event_id"]),
            knowledge_component=fixture["component"],  # type: ignore[arg-type]
            authoritative_evidence_id="evidence.one",
            authoritative_evidence_sha256=str(
                result.event["data"]["authoritative_evidence_sha256"]
            ),
            committed_at_utc=fixture["clock"].text(),  # type: ignore[union-attr]
            commit_receipt_id="turn_committed:one",
        )
        assert replay.applied is False
        assert replay.event["event_id"] == result.event["event_id"]


def test_session_projection_survives_restart_and_exposes_only_bounded_feedback(
    fixture: dict[str, object],
) -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "metacognition.jsonl"
        store = store_for(path, fixture)
        current = prediction(
            fixture,
            attempt="attempt.session.current",
            session="session-current",
            jol=67,
            strategies=("retrieval", "checking"),
        )
        other = prediction(
            fixture,
            attempt="attempt.session.other",
            session="session-other",
            jol=12,
        )
        store.record_prediction(current)
        store.record_prediction(other)

        pending = store.list_session_predictions(
            learner_key=str(fixture["learner"]), session_id="session-current"
        )
        assert len(pending) == 1
        assert pending[0]["status"] == "pending"
        assert pending[0]["learner_jol_percent"] == 67
        assert pending[0]["strategy_codes"] == ["retrieval", "checking"]

        pair(store, fixture, current, evidence_suffix="projection", outcome="partial")
        restarted = store_for(path, fixture)
        paired = restarted.list_session_predictions(
            learner_key=str(fixture["learner"]), session_id="session-current"
        )
        assert len(paired) == 1
        assert paired[0]["status"] == "paired"
        assert paired[0]["feedback"]["authoritative_outcome"] == "partial"
        assert paired[0]["feedback"]["actual_score_percent"] == 50
        assert paired[0]["feedback"]["mastery_changed_by_metacognition"] is False
        serialized = json.dumps(paired, ensure_ascii=False)
        assert str(fixture["learner"]) not in serialized
        assert "raw answer" not in serialized
        assert "assessment_confidence" not in serialized
        assert "rubric_authority_sha256" not in serialized
        assert str(other["event_id"]) not in serialized


def test_outcome_cannot_be_client_self_reported_and_requires_resolved_evidence(
    fixture: dict[str, object],
) -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "metacognition.jsonl"
        no_resolver = MetacognitionStore(path, clock=fixture["clock"])  # type: ignore[arg-type]
        pred = prediction(fixture)
        no_resolver.record_prediction(pred)
        with pytest.raises(MetacognitionStoreError, match="resolver is required"):
            no_resolver.pair_authoritative_outcome(
                prediction_event_id=str(pred["event_id"]),
                knowledge_component=fixture["component"],  # type: ignore[arg-type]
                authoritative_evidence_id="evidence.forged",
                authoritative_evidence_sha256="f" * 64,
                committed_at_utc=fixture["clock"].text(),  # type: ignore[union-attr]
                commit_receipt_id="turn_committed:forged",
            )


def test_pairing_rejects_cross_kc_question_rubric_and_untrusted_evidence(
    fixture: dict[str, object],
) -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "metacognition.jsonl"
        store = store_for(path, fixture)
        pred = prediction(fixture)
        store.record_prediction(pred)
        make = fixture["make_evidence"]
        assert callable(make)
        row = make("cross", question_id="question.other")
        with pytest.raises(MetacognitionConflictError, match="question_id"):
            store.pair_authoritative_outcome(
                prediction_event_id=str(pred["event_id"]),
                knowledge_component=fixture["component"],  # type: ignore[arg-type]
                authoritative_evidence_id=str(row["evidence_id"]),
                authoritative_evidence_sha256=str(row["evidence_fingerprint"]),
                committed_at_utc=fixture["clock"].text(),  # type: ignore[union-attr]
                commit_receipt_id="turn_committed:cross",
            )
        row = make("untrusted")
        row["authoritative"] = False
        with pytest.raises(MetacognitionError, match="authoritative assessment"):
            store.pair_authoritative_outcome(
                prediction_event_id=str(pred["event_id"]),
                knowledge_component=fixture["component"],  # type: ignore[arg-type]
                authoritative_evidence_id=str(row["evidence_id"]),
                authoritative_evidence_sha256=str(row["evidence_fingerprint"]),
                committed_at_utc=fixture["clock"].text(),  # type: ignore[union-attr]
                commit_receipt_id="turn_committed:untrusted",
            )


def test_cross_session_and_delayed_review_pairs_accumulate_per_kc_calibration(
    fixture: dict[str, object],
) -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "metacognition.jsonl"
        store = store_for(path, fixture)
        first = prediction(
            fixture, attempt="attempt.one", session="session-one", jol=20
        )
        store.record_prediction(first)
        pair(store, fixture, first, evidence_suffix="session-one", outcome="correct")

        clock = fixture["clock"]
        assert isinstance(clock, Clock)
        clock.advance(days=7)
        second = prediction(
            fixture,
            attempt="attempt.review",
            session="session-two",
            kind="delayed_review",
            jol=80,
            strategies=("retrieval", "checking"),
            review_id="review_" + "d" * 64,
            lease_id="lease_" + "e" * 64,
        )
        store.record_prediction(second)
        pair(store, fixture, second, evidence_suffix="session-two", outcome="partial")

        record = store.get_calibration_record(
            learner_key=str(fixture["learner"]),
            curriculum_namespace=str(first["target"]["curriculum_namespace"]),
            knowledge_component_id="kc_state_definition",
        )
        assert record is not None
        assert record["paired_count"] == 2
        assert record["mean_learner_jol_percent"] == 50.0
        assert record["mean_authoritative_score_percent"] == 75.0
        assert record["signed_calibration_bias_points"] == -25.0
        assert record["classification_counts"] == {
            "aligned": 0,
            "overconfident": 1,
            "underconfident": 1,
        }
        assert record["last_pairing"]["assessment_kind"] == "delayed_review"
        assert record["external_calibration_established"] is False
        assert record["validity_note"] == EXTERNAL_CALIBRATION_NOTE


def test_feedback_is_descriptive_and_does_not_change_scoring_or_mastery(
    fixture: dict[str, object],
) -> None:
    with TemporaryDirectory() as directory:
        store = store_for(Path(directory) / "meta.jsonl", fixture)
        pred = prediction(fixture, jol=95, strategies=("decomposition",))
        store.record_prediction(pred)
        result = pair(
            store, fixture, pred, evidence_suffix="feedback", outcome="partial"
        )
        feedback = store.feedback_for_pairing(result.event_id)
        assert feedback["calibration_classification"] == "overconfident"
        assert "保留原评分标准" in feedback["message_zh"]
        assert feedback["scoring_standard_changed"] is False
        assert feedback["mastery_changed_by_metacognition"] is False
        assert feedback["external_calibration_established"] is False


def test_store_detects_tampering_and_repairs_only_truncated_tail(
    fixture: dict[str, object],
) -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "meta.jsonl"
        store = store_for(path, fixture)
        store.record_prediction(prediction(fixture))
        path.write_bytes(path.read_bytes() + b'{"partial":')
        restarted = store_for(path, fixture)
        assert restarted.export_learner(str(fixture["learner"])) is not None
        lines = path.read_bytes().splitlines()
        envelope = json.loads(lines[0])
        envelope["event"]["data"]["learner_jol_percent"] = 0
        lines[0] = json.dumps(
            envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        path.write_bytes(b"\n".join(lines) + b"\n")
        with pytest.raises(MetacognitionStoreError, match="hash chain"):
            store_for(path, fixture)


def test_export_contains_no_raw_answer_and_purge_is_fenced_and_idempotent(
    fixture: dict[str, object],
) -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "meta.jsonl"
        store = store_for(path, fixture)
        pred = prediction(fixture)
        store.record_prediction(pred)
        pair(store, fixture, pred, evidence_suffix="privacy", outcome="correct")
        exported = store.export_learner(str(fixture["learner"]))
        assert exported is not None
        serialized = json.dumps(exported, ensure_ascii=False)
        assert "raw answer" not in serialized
        assert "assessment_confidence" not in serialized
        assert exported["external_calibration_established"] is False
        purged = store.purge_learner(str(fixture["learner"]))
        assert purged == {"learners": 1, "events": 2, "erasure_fence": 1}
        assert store.export_learner(str(fixture["learner"])) is None
        assert store.purge_learner(str(fixture["learner"])) == {
            "learners": 0,
            "events": 0,
            "erasure_fence": 1,
        }
        with pytest.raises(MetacognitionConflictError, match="permanently erased"):
            store.record_prediction(pred)


def test_prediction_id_is_content_bound_and_attempt_cannot_be_repredicted(
    fixture: dict[str, object],
) -> None:
    with TemporaryDirectory() as directory:
        store = store_for(Path(directory) / "meta.jsonl", fixture)
        first = prediction(fixture, jol=30)
        store.record_prediction(first)
        forged = deepcopy(first)
        forged["data"]["learner_jol_percent"] = 99
        with pytest.raises(MetacognitionError, match="event_id does not bind"):
            store.record_prediction(forged)
        second = prediction(fixture, jol=99)
        with pytest.raises(MetacognitionConflictError, match="attempt already"):
            store.record_prediction(second)


def test_json_schemas_accept_runtime_events_records_envelopes_and_export(
    fixture: dict[str, object],
) -> None:
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    schema_root = Path(__file__).resolve().parent.parent / "schema"
    names = (
        "teacher_agent_metacognition_event.schema.json",
        "teacher_agent_metacognition_store_event.schema.json",
        "teacher_agent_metacognition_record.schema.json",
        "teacher_agent_metacognition_export.schema.json",
    )
    schemas = {
        name: json.loads((schema_root / name).read_text(encoding="utf-8"))
        for name in names
    }
    registry = Registry()
    for schema in schemas.values():
        Draft202012Validator.check_schema(schema)
        registry = registry.with_resource(
            str(schema["$id"]), Resource.from_contents(schema)
        )
    with TemporaryDirectory() as directory:
        store = store_for(Path(directory) / "meta.jsonl", fixture)
        pred = prediction(fixture)
        store.record_prediction(pred)
        result = pair(store, fixture, pred, evidence_suffix="schema", outcome="correct")
        exported = store.export_learner(str(fixture["learner"]))
        assert exported is not None
        Draft202012Validator(schemas[names[0]], registry=registry).validate(pred)
        Draft202012Validator(schemas[names[0]], registry=registry).validate(
            result.event
        )
        envelope = json.loads(store.path.read_text(encoding="utf-8").splitlines()[0])
        Draft202012Validator(schemas[names[1]], registry=registry).validate(envelope)
        record = exported["calibration_records"][0]
        Draft202012Validator(schemas[names[2]], registry=registry).validate(record)
        Draft202012Validator(schemas[names[3]], registry=registry).validate(exported)


def test_project_data_rights_export_includes_private_metacognition_record(
    fixture: dict[str, object],
) -> None:
    with TemporaryDirectory() as directory:
        store = store_for(Path(directory) / "meta.jsonl", fixture)
        pred = prediction(fixture)
        store.record_prediction(pred)
        pair(store, fixture, pred, evidence_suffix="data-rights", outcome="correct")
        learner = str(fixture["learner"])
        metacognition_export = store.export_learner(learner)
        assert metacognition_export is not None
        archive = build_project_export_archive(
            project={"project_id": "project_meta", "title": "private"},
            sessions={},
            syllabi={},
            resources={},
            metacognition_records={learner: metacognition_export},
        )
        manifest = validate_project_export_archive(archive.payload)
        entry = next(
            item
            for item in manifest["entries"]
            if item["path"].startswith("metacognition_records/")
        )
        assert entry["data_class"] == "learner_metacognition_calibration_private"
        with zipfile.ZipFile(io.BytesIO(archive.payload)) as bundle:
            restored = json.loads(bundle.read(entry["path"]))
        assert restored == metacognition_export
