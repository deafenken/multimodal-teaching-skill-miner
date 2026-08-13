from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from teaching_skill_miner.teacher_agent_adjudication import (
    AdjudicationConflictError,
    AdjudicationEvidenceDeletedError,
    authoritative_evidence_sha256,
    canonical_sha256,
)
from teaching_skill_miner.teacher_agent_adjudication_store import (
    STORE_TOMBSTONE_SCHEMA,
    DurableTeacherAgentAdjudicationQueue,
    TeacherAgentAdjudicationStore,
    TeacherAgentAdjudicationStoreError,
)


ROOT = Path(__file__).resolve().parents[1]


def _clock() -> datetime:
    return datetime(2026, 8, 12, 5, 0, tzinfo=timezone.utc)


def _evidence() -> dict:
    evidence = {
        "evidence_id": "evidence.assessment.1",
        "source": {
            "session_id": "teach_session_1",
            "round_number": 4,
            "action_id": "action.assess.4",
            "question_id": "question.dp.4",
            "history_event_sha256": "1" * 64,
        },
        "target_kc_ids": ["kc_state_definition"],
        "original_assessment_id": "assessment.dp.4",
        "original_assessment_sha256": "2" * 64,
        "rubric_id": "rubric.dp.teacher.1",
        "rubric_authority_sha256": "3" * 64,
        "before_snapshot": {
            "schema": "teaching_skill_miner.student_state_estimate.v2",
            "version": 10,
            "learner_text": "private-before",
        },
        "after_snapshot": {
            "schema": "teaching_skill_miner.student_state_estimate.v2",
            "version": 11,
            "learner_text": "private-after",
        },
        "learner_text": "不会，这段内容不能写入裁决日志",
    }
    evidence["evidence_sha256"] = authoritative_evidence_sha256(evidence)
    return evidence


def _claim_process(
    path: str,
    evidence: dict,
    item_id: str,
    number: int,
    barrier: multiprocessing.synchronize.Barrier,
    output: multiprocessing.queues.Queue,
) -> None:
    registry = {str(evidence["evidence_id"]): evidence}
    queue = DurableTeacherAgentAdjudicationQueue(
        path,
        evidence_resolver=registry.get,
        clock=_clock,
        token_factory=lambda: f"process-token-{number:02d}",
    )
    barrier.wait()
    try:
        result = queue.claim(
            item_id,
            expected_version=1,
            idempotency_key=f"process-claim-{number:02d}",
        )
    except Exception as exc:  # noqa: BLE001 - process outcome must cross IPC.
        output.put(("error", type(exc).__name__, str(exc)))
    else:
        output.put(("ok", result["item"]["version"], ""))


class DurableAdjudicationQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "adjudication.jsonl"
        self.evidence = _evidence()
        self.registry = {"evidence.assessment.1": self.evidence}
        self.queue = self.make_queue(token="claim-token-0001")

    def make_queue(
        self,
        *,
        token: str = "claim-token-0001",
        after_commit=None,
        resolver=None,
    ) -> DurableTeacherAgentAdjudicationQueue:
        return DurableTeacherAgentAdjudicationQueue(
            self.path,
            evidence_resolver=resolver or self.registry.get,
            clock=_clock,
            token_factory=lambda: token,
            after_commit=after_commit,
        )

    def enqueue(self) -> dict:
        return self.queue.enqueue(
            evidence_id="evidence.assessment.1",
            evidence_sha256=self.evidence["evidence_sha256"],
            idempotency_key="enqueue-key-0001",
        )

    def claim(self, item: dict) -> dict:
        return self.queue.claim(
            item["item_id"],
            expected_version=item["version"],
            idempotency_key="claim-key-000001",
        )

    def decide(self, item: dict, claim: dict, *, queue=None) -> dict:
        target = queue or self.queue
        return target.decide(
            item["item_id"],
            expected_version=claim["item"]["version"],
            claim_token=claim["claim_token"],
            evidence_id="evidence.assessment.1",
            evidence_sha256=self.evidence["evidence_sha256"],
            idempotency_key="decision-key-0001",
            decision="correct",
            reason_code="signal_misclassified",
            correction={
                "signal": "partial",
                "answer_alignment": "partially_aligned",
                "focus_dimension": "conceptual",
                "target_kc_ids": ["kc_state_definition"],
            },
        )

    def test_store_schema_validates_every_durable_transaction(self) -> None:
        item = self.enqueue()
        claim = self.claim(item)
        self.decide(item, claim)

        adjudication_schema = json.loads(
            (ROOT / "schema" / "teacher_agent_adjudication.schema.json").read_text()
        )
        store_schema = json.loads(
            (
                ROOT / "schema" / "teacher_agent_adjudication_store.schema.json"
            ).read_text()
        )
        registry = Registry().with_resources(
            [
                (
                    adjudication_schema["$id"],
                    Resource.from_contents(adjudication_schema),
                ),
                (store_schema["$id"], Resource.from_contents(store_schema)),
            ]
        )
        Draft202012Validator.check_schema(store_schema)
        validator = Draft202012Validator(store_schema, registry=registry)
        for event in self.queue.store.events:
            validator.validate(event)

    def test_crash_after_fsync_before_response_recovers_same_enqueue(self) -> None:
        callbacks: list[dict] = []

        def crash_after_commit(event: dict) -> None:
            callbacks.append(event)
            raise RuntimeError("simulated process crash after durable append")

        crashing = self.make_queue(after_commit=crash_after_commit)
        with self.assertRaisesRegex(RuntimeError, "after durable append"):
            crashing.enqueue(
                evidence_id="evidence.assessment.1",
                evidence_sha256=self.evidence["evidence_sha256"],
                idempotency_key="crash-enqueue-01",
            )
        self.assertEqual(len(callbacks), 1)
        restarted = self.make_queue()
        recovered = restarted.enqueue(
            evidence_id="evidence.assessment.1",
            evidence_sha256=self.evidence["evidence_sha256"],
            idempotency_key="crash-enqueue-01",
        )
        self.assertEqual(recovered["version"], 1)
        self.assertEqual(len(restarted.store.events), 1)

    def test_decision_idempotency_survives_restart_and_revalidates_evidence(
        self,
    ) -> None:
        item = self.enqueue()
        claim = self.claim(item)
        first = self.decide(item, claim)
        resolver_calls: list[str] = []

        def counting_resolver(evidence_id: str):
            resolver_calls.append(evidence_id)
            return self.registry.get(evidence_id)

        restarted = self.make_queue(
            token="unused-token-0002", resolver=counting_resolver
        )
        replay = self.decide(item, claim, queue=restarted)
        self.assertEqual(replay, first)
        self.assertEqual(resolver_calls, ["evidence.assessment.1"])
        self.assertEqual(len(restarted.store.events), 3)
        self.assertEqual(len(restarted.history(item["item_id"])), 3)

    @unittest.skipUnless(os.name == "posix", "durable queue requires POSIX flock")
    def test_multi_process_claims_use_one_disk_cas_boundary(self) -> None:
        item = self.enqueue()
        context = multiprocessing.get_context("fork")
        barrier = context.Barrier(3)
        output = context.Queue()
        processes = [
            context.Process(
                target=_claim_process,
                args=(
                    str(self.path),
                    self.evidence,
                    item["item_id"],
                    index,
                    barrier,
                    output,
                ),
            )
            for index in range(2)
        ]
        for process in processes:
            process.start()
        barrier.wait()
        for process in processes:
            process.join(timeout=5)
            self.assertFalse(process.is_alive())
            self.assertEqual(process.exitcode, 0)
        outcomes = [output.get(timeout=1) for _process in processes]
        self.assertEqual(sum(row[0] == "ok" for row in outcomes), 1)
        errors = [row for row in outcomes if row[0] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0][1], AdjudicationConflictError.__name__)
        self.assertEqual(self.queue.get(item["item_id"])["version"], 2)
        self.assertEqual(len(self.queue.store.events), 2)

    def test_torn_tail_is_repaired_but_complete_invalid_line_fails_closed(self) -> None:
        item = self.enqueue()
        clean = self.path.read_bytes()
        with self.path.open("ab") as stream:
            stream.write(b'{"schema":"torn')
            stream.flush()
            os.fsync(stream.fileno())
        restarted = self.make_queue()
        self.assertEqual(restarted.get(item["item_id"]), item)
        self.assertEqual(self.path.read_bytes(), clean)

        with self.path.open("ab") as stream:
            stream.write(b"{not-json}\n")
            stream.flush()
            os.fsync(stream.fileno())
        corrupted = self.path.read_bytes()
        with self.assertRaisesRegex(TeacherAgentAdjudicationStoreError, "invalid JSON"):
            TeacherAgentAdjudicationStore(self.path)
        self.assertEqual(self.path.read_bytes(), corrupted)

    def test_hash_tampering_fails_closed_without_repair(self) -> None:
        self.enqueue()
        original = self.path.read_bytes()
        event = json.loads(original)
        event["item_versions"][0]["review_reason"] = "learner_dispute"
        tampered = (
            json.dumps(event, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )
        self.path.write_bytes(tampered)
        os.chmod(self.path, 0o600)
        with self.assertRaisesRegex(
            TeacherAgentAdjudicationStoreError, "failed integrity"
        ):
            TeacherAgentAdjudicationStore(self.path)
        self.assertEqual(self.path.read_bytes(), tampered)

    def test_semantic_identity_tampering_fails_even_when_hashes_are_resealed(
        self,
    ) -> None:
        item = self.enqueue()
        self.claim(item)
        events = [json.loads(line) for line in self.path.read_text().splitlines()]
        event = events[1]
        version = event["item_versions"][0]
        version["lease"]["actor"]["authenticated"] = True
        version_material = deepcopy(version)
        version_material.pop("version_sha256")
        version["version_sha256"] = canonical_sha256(version_material)
        receipt = event["audit_receipts"][0]
        receipt["result_version_sha256"] = version["version_sha256"]
        receipt_material = deepcopy(receipt)
        receipt_material.pop("receipt_sha256")
        receipt["receipt_sha256"] = canonical_sha256(receipt_material)
        binding = event["idempotency_bindings"][0]
        binding["response"]["item"] = deepcopy(version)
        binding["response_sha256"] = canonical_sha256(binding["response"])
        event_material = deepcopy(event)
        event_material.pop("hash")
        event["hash"] = canonical_sha256(event_material)
        self.path.write_text(
            "\n".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) for row in events
            )
            + "\n"
        )
        os.chmod(self.path, 0o600)
        with self.assertRaisesRegex(
            TeacherAgentAdjudicationStoreError, "must not claim authentication"
        ):
            TeacherAgentAdjudicationStore(self.path)

    @unittest.skipUnless(os.name == "posix", "mode and symlink checks are POSIX")
    def test_store_requires_0600_regular_non_symlink_file(self) -> None:
        self.enqueue()
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        os.chmod(self.path, 0o644)
        with self.assertRaisesRegex(TeacherAgentAdjudicationStoreError, "0600"):
            TeacherAgentAdjudicationStore(self.path)
        os.chmod(self.path, 0o600)
        link = self.path.with_name("adjudication-link.jsonl")
        link.symlink_to(self.path)
        with self.assertRaisesRegex(TeacherAgentAdjudicationStoreError, "safely"):
            TeacherAgentAdjudicationStore(link)

    def test_purge_tombstone_blocks_old_and_new_enqueue_after_evidence_restore(
        self,
    ) -> None:
        item = self.enqueue()
        claim = self.claim(item)
        self.registry.pop("evidence.assessment.1")
        with self.assertRaises(AdjudicationEvidenceDeletedError):
            self.decide(item, claim)
        recovery = self.queue.store.recover()
        self.assertEqual(self.queue.get(item["item_id"])["status"], "cancelled")
        self.assertEqual(len(recovery.erasure_tombstones), 1)
        tombstone = next(iter(recovery.erasure_tombstones.values()))
        self.assertEqual(tombstone["schema"], STORE_TOMBSTONE_SCHEMA)
        self.assertEqual(
            set(tombstone),
            {"schema", "evidence_id_sha256", "erased_at_utc", "reason", "actor"},
        )
        serialized = json.dumps(tombstone, ensure_ascii=False)
        self.assertNotIn("evidence.assessment.1", serialized)
        self.assertNotIn("learner_text", serialized)

        # Even if a stale external replica restores the exact old evidence, the
        # local erasure fence takes precedence over both the old idempotency key
        # and a brand-new request.
        self.registry["evidence.assessment.1"] = self.evidence
        restarted = self.make_queue()
        for key in ("enqueue-key-0001", "enqueue-after-purge"):
            with self.assertRaises(AdjudicationEvidenceDeletedError):
                restarted.enqueue(
                    evidence_id="evidence.assessment.1",
                    evidence_sha256=self.evidence["evidence_sha256"],
                    idempotency_key=key,
                )
        self.assertEqual(len(restarted.store.events), 3)

    def test_restart_enqueue_retry_discovers_deletion_and_cancels_old_item(
        self,
    ) -> None:
        item = self.enqueue()
        self.registry.pop("evidence.assessment.1")
        restarted = self.make_queue()
        with self.assertRaises(AdjudicationEvidenceDeletedError):
            restarted.enqueue(
                evidence_id="evidence.assessment.1",
                evidence_sha256=self.evidence["evidence_sha256"],
                idempotency_key="enqueue-key-0001",
            )
        self.assertEqual(restarted.get(item["item_id"])["status"], "cancelled")
        recovery = restarted.store.recover()
        self.assertEqual(len(recovery.events), 2)
        self.assertEqual(len(recovery.erasure_tombstones), 1)

    def test_unknown_evidence_purge_creates_content_free_fence(self) -> None:
        empty_registry: dict[str, dict] = {}
        queue = self.make_queue(resolver=empty_registry.get)
        self.assertEqual(queue.cancel_deleted_evidence("evidence.unknown.1"), ())
        recovery = queue.store.recover()
        self.assertEqual(len(recovery.events), 1)
        self.assertEqual(len(recovery.erasure_tombstones), 1)
        empty_registry["evidence.unknown.1"] = _evidence()
        with self.assertRaises(AdjudicationEvidenceDeletedError):
            queue.enqueue(
                evidence_id="evidence.unknown.1",
                evidence_sha256=empty_registry["evidence.unknown.1"]["evidence_sha256"],
                idempotency_key="delayed-enqueue-1",
            )

    def test_raw_evidence_text_is_never_persisted(self) -> None:
        self.enqueue()
        serialized = self.path.read_text()
        self.assertNotIn("不会，这段内容不能写入裁决日志", serialized)
        self.assertNotIn("private-before", serialized)
        self.assertNotIn("private-after", serialized)
        self.assertNotIn("learner_text", serialized)


if __name__ == "__main__":
    unittest.main()
