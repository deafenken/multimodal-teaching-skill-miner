from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading
import unittest

from jsonschema import Draft202012Validator

from teaching_skill_miner.teacher_agent_adjudication import (
    ADJUDICATION_INSTRUCTION_SCHEMA,
    CLAIM_LEASE_SECONDS,
    LOCAL_OPERATOR_IDENTITY,
    AdjudicationConflictError,
    AdjudicationEvidenceError,
    TeacherAgentAdjudicationError,
    TeacherAgentAdjudicationQueue,
    authoritative_evidence_sha256,
    canonical_sha256,
    reduce_adjudication,
)


ROOT = Path(__file__).resolve().parents[1]


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 12, 4, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, *, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


def _sealed_evidence(
    *,
    evidence_id: str = "evidence.assessment.1",
    target_kc_ids: list[str] | None = None,
) -> dict:
    evidence = {
        "evidence_id": evidence_id,
        "source": {
            "session_id": "teach_session_1",
            "round_number": 3,
            "action_id": "action.assess.3",
            "question_id": "question.dp.3",
            "history_event_sha256": "1" * 64,
        },
        "target_kc_ids": target_kc_ids or ["kc_state_definition"],
        "original_assessment_id": "assessment.dp.3",
        "original_assessment_sha256": "2" * 64,
        "rubric_id": "rubric.dp.teacher.1",
        "rubric_authority_sha256": "3" * 64,
        "before_snapshot": {
            "schema": "teaching_skill_miner.student_state_estimate.v2",
            "version": 8,
            "learner_text": "这段原始学生文本不能进入公开复核项",
            "knowledge_components": {"kc_state_definition": {"p_mastery": 0.2}},
        },
        "after_snapshot": {
            "schema": "teaching_skill_miner.student_state_estimate.v2",
            "version": 9,
            "learner_text": "另一个不应公开的原始答案",
            "knowledge_components": {"kc_state_definition": {"p_mastery": 0.5}},
        },
        # Unknown evidence fields are deliberately included in the authoritative
        # hash but never copied into the queue projection.
        "learner_text": "不会，我想先听完整讲解",
        "private_trace": {"model_reasoning": "server-only"},
    }
    evidence["evidence_sha256"] = authoritative_evidence_sha256(evidence)
    return evidence


class TeacherAgentAdjudicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = MutableClock()
        self.registry = {"evidence.assessment.1": _sealed_evidence()}
        self.queue = TeacherAgentAdjudicationQueue(
            evidence_resolver=self.registry.get,
            clock=self.clock,
            token_factory=lambda: "local-claim-token-0001",
        )
        self.hash = self.registry["evidence.assessment.1"]["evidence_sha256"]

    def enqueue(self, *, key: str = "enqueue-key-0001") -> dict:
        return self.queue.enqueue(
            evidence_id="evidence.assessment.1",
            evidence_sha256=self.hash,
            idempotency_key=key,
        )

    def claim(self, item: dict, *, key: str = "claim-key-000001") -> dict:
        return self.queue.claim(
            item["item_id"], expected_version=item["version"], idempotency_key=key
        )

    def test_schema_accepts_items_receipts_and_pure_instructions(self) -> None:
        schema = json.loads(
            (ROOT / "schema" / "teacher_agent_adjudication.schema.json").read_text()
        )
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)

        pending = self.enqueue()
        claimed = self.claim(pending)
        result = self.queue.decide(
            pending["item_id"],
            expected_version=claimed["item"]["version"],
            claim_token=claimed["claim_token"],
            evidence_id="evidence.assessment.1",
            evidence_sha256=self.hash,
            idempotency_key="decision-key-0001",
            decision="approve",
            reason_code="assessment_confirmed",
        )
        validator.validate(result["item"])
        validator.validate(result["instruction"])
        for receipt in self.queue.audit_receipts:
            validator.validate(receipt)

        self.assertEqual(
            result["instruction"]["schema"], ADJUDICATION_INSTRUCTION_SCHEMA
        )
        before = deepcopy(result["item"])
        self.assertEqual(reduce_adjudication(result["item"]), result["instruction"])
        self.assertEqual(result["item"], before)
        self.assertFalse(result["instruction"]["mutates_student_model"])

    def test_forged_evidence_hash_and_forged_authoritative_record_fail_closed(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            AdjudicationEvidenceError, "does not match authoritative evidence"
        ):
            self.queue.enqueue(
                evidence_id="evidence.assessment.1",
                evidence_sha256="f" * 64,
                idempotency_key="forged-hash-0001",
            )

        self.registry["evidence.assessment.1"]["learner_text"] = "tampered after seal"
        with self.assertRaisesRegex(AdjudicationEvidenceError, "self-hash"):
            self.queue.enqueue(
                evidence_id="evidence.assessment.1",
                evidence_sha256=self.hash,
                idempotency_key="tampered-record-01",
            )

    def test_no_raw_client_evidence_api_and_no_raw_text_in_public_projection(
        self,
    ) -> None:
        with self.assertRaises(TypeError):
            self.queue.enqueue(  # type: ignore[call-arg]
                evidence_id="evidence.assessment.1",
                evidence_sha256=self.hash,
                idempotency_key="raw-client-00001",
                evidence={"learner_text": "forged"},
            )
        item = self.enqueue()
        serialized = json.dumps(item, ensure_ascii=False)
        self.assertNotIn("不会，我想先听完整讲解", serialized)
        self.assertNotIn("这段原始学生文本", serialized)
        self.assertNotIn("learner_text", serialized)
        self.assertEqual(item["before_snapshot"]["state_version"], 8)
        self.assertRegex(item["before_snapshot"]["content_sha256"], r"^[0-9a-f]{64}$")

    def test_concurrent_claims_are_serialized_by_expected_version_cas(self) -> None:
        item = self.enqueue()
        barrier = threading.Barrier(3)
        outcomes: list[tuple[str, object]] = []
        lock = threading.Lock()

        def worker(number: int) -> None:
            barrier.wait()
            try:
                result = self.queue.claim(
                    item["item_id"],
                    expected_version=1,
                    idempotency_key=f"concurrent-claim-{number:02d}",
                )
            except Exception as exc:  # noqa: BLE001 - concurrency outcome capture.
                outcome: tuple[str, object] = ("error", exc)
            else:
                outcome = ("ok", result)
            with lock:
                outcomes.append(outcome)

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())

        self.assertEqual(sum(kind == "ok" for kind, _value in outcomes), 1)
        errors = [value for kind, value in outcomes if kind == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], AdjudicationConflictError)
        claimed = self.queue.get(item["item_id"])
        self.assertEqual(claimed["version"], 2)
        self.assertEqual(claimed["lease"]["actor"]["identity"], LOCAL_OPERATOR_IDENTITY)
        claimed_at = datetime.fromisoformat(
            claimed["lease"]["claimed_at"].replace("Z", "+00:00")
        )
        expires_at = datetime.fromisoformat(
            claimed["lease"]["expires_at"].replace("Z", "+00:00")
        )
        self.assertEqual((expires_at - claimed_at).total_seconds(), CLAIM_LEASE_SECONDS)

    def test_expired_lease_can_be_reclaimed_only_at_current_cas_version(self) -> None:
        item = self.enqueue()
        first = self.claim(item)
        self.clock.advance(seconds=CLAIM_LEASE_SECONDS + 1)
        with self.assertRaises(AdjudicationConflictError):
            self.queue.claim(
                item["item_id"],
                expected_version=1,
                idempotency_key="stale-reclaim-0001",
            )
        second = self.queue.claim(
            item["item_id"],
            expected_version=first["item"]["version"],
            idempotency_key="valid-reclaim-0001",
        )
        self.assertEqual(second["item"]["version"], 3)

    def test_correct_decision_is_bounded_and_idempotent(self) -> None:
        item = self.enqueue()
        claim = self.claim(item)
        arguments = {
            "expected_version": claim["item"]["version"],
            "claim_token": claim["claim_token"],
            "evidence_id": "evidence.assessment.1",
            "evidence_sha256": self.hash,
            "idempotency_key": "correct-decision-01",
            "decision": "correct",
            "reason_code": "signal_misclassified",
            "correction": {
                "signal": "partial",
                "answer_alignment": "partially_aligned",
                "focus_dimension": "conceptual",
                "target_kc_ids": ["kc_state_definition"],
            },
        }
        first = self.queue.decide(item["item_id"], **arguments)
        replay = self.queue.decide(item["item_id"], **arguments)
        self.assertEqual(replay, first)
        self.assertEqual(len(self.queue.history(item["item_id"])), 3)
        self.assertEqual(first["instruction"]["operation"], "supersede_and_replay")
        self.assertTrue(first["instruction"]["requires_explicit_apply"])
        self.assertFalse(first["instruction"]["mastery_update_authorized"])
        self.assertFalse(
            first["instruction"]["authority"]["authoritative_for_mastery_update"]
        )
        self.assertEqual(
            first["instruction"]["authority"]["identity"],
            LOCAL_OPERATOR_IDENTITY,
        )
        with self.assertRaises(AdjudicationConflictError):
            self.queue.decide(
                item["item_id"],
                **{
                    **arguments,
                    "idempotency_key": "conflicting-decision-2",
                    "decision": "approve",
                    "reason_code": "assessment_confirmed",
                    "correction": None,
                },
            )
        with self.assertRaises(TeacherAgentAdjudicationError):
            self.queue.decide(
                item["item_id"],
                **{
                    **arguments,
                    "idempotency_key": "invalid-correction-1",
                    "correction": {"authoritative": True},
                },
            )

    def test_abstain_is_terminal_but_produces_no_replay(self) -> None:
        item = self.enqueue()
        claim = self.claim(item)
        result = self.queue.decide(
            item["item_id"],
            expected_version=claim["item"]["version"],
            claim_token=claim["claim_token"],
            evidence_id="evidence.assessment.1",
            evidence_sha256=self.hash,
            idempotency_key="abstain-decision-1",
            decision="abstain",
            reason_code="insufficient_evidence",
        )
        self.assertEqual(result["item"]["decision"]["kind"], "abstain")
        self.assertEqual(result["instruction"]["operation"], "do_not_replay")
        with self.assertRaises(AdjudicationConflictError):
            self.queue.decide(
                item["item_id"],
                expected_version=result["item"]["version"],
                claim_token=claim["claim_token"],
                evidence_id="evidence.assessment.1",
                evidence_sha256=self.hash,
                idempotency_key="conflicting-decision",
                decision="approve",
                reason_code="assessment_confirmed",
            )

    def test_evidence_purge_cancels_claimed_and_decided_items(self) -> None:
        item = self.enqueue()
        claim = self.claim(item)
        decided = self.queue.decide(
            item["item_id"],
            expected_version=claim["item"]["version"],
            claim_token=claim["claim_token"],
            evidence_id="evidence.assessment.1",
            evidence_sha256=self.hash,
            idempotency_key="approve-before-purge",
            decision="approve",
            reason_code="assessment_confirmed",
        )
        self.registry.pop("evidence.assessment.1")
        cancelled = self.queue.cancel_deleted_evidence("evidence.assessment.1")
        self.assertEqual(len(cancelled), 1)
        self.assertEqual(cancelled[0]["status"], "cancelled")
        self.assertEqual(cancelled[0]["version"], decided["item"]["version"] + 1)
        instruction = reduce_adjudication(cancelled[0])
        self.assertEqual(instruction["operation"], "cancel_adjudication")
        # A repeated purge callback is an idempotent projection, not a new event.
        repeated = self.queue.cancel_deleted_evidence("evidence.assessment.1")
        self.assertEqual(repeated, cancelled)
        self.assertEqual(len(self.queue.history(item["item_id"])), 4)

    def test_audit_and_item_versions_are_append_only_hash_chains(self) -> None:
        item = self.enqueue()
        self.claim(item)
        history = self.queue.history(item["item_id"])
        self.assertEqual(
            history[1]["previous_version_sha256"], history[0]["version_sha256"]
        )
        for version in history:
            material = deepcopy(version)
            declared = material.pop("version_sha256")
            self.assertEqual(declared, canonical_sha256(material))

        receipts = self.queue.audit_receipts
        self.assertIsNone(receipts[0]["prior_receipt_sha256"])
        self.assertEqual(
            receipts[1]["prior_receipt_sha256"], receipts[0]["receipt_sha256"]
        )
        for receipt in receipts:
            material = deepcopy(receipt)
            declared = material.pop("receipt_sha256")
            self.assertEqual(declared, canonical_sha256(material))


if __name__ == "__main__":
    unittest.main()
