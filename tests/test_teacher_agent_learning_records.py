from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from teaching_skill_miner.teacher_agent_learning_records import (
    LEARNING_OUTBOX_EVENT_SCHEMA,
    LEARNING_RECORD_SCHEMA,
    LEARNING_STORE_EVENT_SCHEMA,
    REVIEW_INTERVAL_DAYS,
    LearningRecordConflictError,
    LearningRecordError,
    LearningRecordStore,
    LearningRecordStoreError,
    build_learning_evidence_outbox_event,
    curriculum_namespace_for_source_ref,
    learning_record_target,
    mint_learner_key,
    validate_learning_outbox_event,
    validate_learning_record,
)


class _Clock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **kwargs: float) -> None:
        self.value += timedelta(**kwargs)


class LearningRecordTests(unittest.TestCase):
    secret = b"learning-record-test-secret-material-32bytes"

    def setUp(self) -> None:
        self.clock = _Clock(datetime(2026, 8, 12, 5, 0, tzinfo=timezone.utc))
        self.learner_key = mint_learner_key(
            "student-001", tenant_id="school-001", secret=self.secret
        )

    @staticmethod
    def _utc(value: datetime) -> str:
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _component(
        kc_id: str = "kc_state_definition",
        *,
        scope: str = "syl_0123456789abcdef01234567:module_01:lesson_01_01",
        authority: bool = True,
    ) -> dict[str, object]:
        return {
            "kc_id": kc_id,
            "label": "not persisted",
            "source": "syllabus_lesson_component",
            "source_ref": f"{scope}#{kc_id}",
            "teacher_grading_authority_available": authority,
        }

    def _evidence(
        self,
        suffix: str,
        *,
        kc_id: str = "kc_state_definition",
        outcome: str = "correct",
        observed_at: datetime | None = None,
    ) -> dict[str, object]:
        signal, alignment = {
            "correct": ("correct", "aligned"),
            "partial": ("partial", "partially_aligned"),
            "incorrect": ("misconception", "contradicted"),
        }[outcome]
        material = {
            "suffix": suffix,
            "kc_id": kc_id,
            "signal": signal,
            "alignment": alignment,
        }
        return {
            "evidence_id": f"evidence.{suffix}",
            "item_id": f"item.{suffix}",
            "question_id": f"question.{suffix}",
            "rubric_id": f"rubric.{suffix}",
            "knowledge_component_id": kc_id,
            "signal": signal,
            "answer_alignment": alignment,
            "assessment_eligible": True,
            "authoritative": True,
            "observed_at": self._utc(observed_at or self.clock.value),
            "time_basis": "session_logical",
            "evidence_fingerprint": sha256(
                json.dumps(material, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "learner_response": "这段原文不应进入学习记录",
        }

    def _event(
        self,
        suffix: str,
        *,
        component: dict[str, object] | None = None,
        outcome: str = "correct",
        expected_version: int = 0,
        review_id: str | None = None,
        lease_id: str | None = None,
        observed_at: datetime | None = None,
    ) -> dict[str, object]:
        selected = component or self._component()
        return build_learning_evidence_outbox_event(
            learner_key=self.learner_key,
            knowledge_component=selected,
            evidence=self._evidence(
                suffix,
                kc_id=str(selected["kc_id"]),
                outcome=outcome,
                observed_at=observed_at,
            ),
            expected_version=expected_version,
            committed_at_utc=self._utc(self.clock.value),
            commit_receipt_id=f"turn_committed:{suffix}",
            review_id=review_id,
            lease_id=lease_id,
        )

    def test_server_minted_learner_key_is_stable_tenant_bound_and_not_anonymous(
        self,
    ) -> None:
        self.assertEqual(
            self.learner_key,
            mint_learner_key(
                "STUDENT-001", tenant_id="SCHOOL-001", secret=self.secret
            ),
        )
        self.assertNotEqual(
            self.learner_key,
            mint_learner_key(
                "student-001", tenant_id="school-002", secret=self.secret
            ),
        )
        for value in (
            "anonymous",
            "profile-default",
            "访客",
            "游客",
            "未知",
            "默认用户",
            "未登录",
            "guest_42",
        ):
            with self.subTest(value=value), self.assertRaisesRegex(
                LearningRecordError, "anonymous/default"
            ):
                mint_learner_key(value, tenant_id="school-001", secret=self.secret)
        for real_name in ("游客邦", "未知数同学", "默认值研究员"):
            with self.subTest(real_name=real_name):
                self.assertRegex(
                    mint_learner_key(
                        real_name, tenant_id="school-001", secret=self.secret
                    ),
                    r"^learner_[0-9a-f]{64}$",
                )
        with self.assertRaisesRegex(LearningRecordError, "at least 32 bytes"):
            mint_learner_key(
                "student-001", tenant_id="school-001", secret=b"too-short"
            )

    def test_target_includes_source_ref_namespace_and_hash(self) -> None:
        first = learning_record_target(
            learner_key=self.learner_key,
            knowledge_component=self._component(scope="course-a:lesson-1"),
        )
        second = learning_record_target(
            learner_key=self.learner_key,
            knowledge_component=self._component(scope="course-b:lesson-1"),
        )
        self.assertEqual(first["knowledge_component_id"], second["knowledge_component_id"])
        self.assertNotEqual(first["curriculum_namespace"], second["curriculum_namespace"])
        self.assertNotEqual(first["source_ref_sha256"], second["source_ref_sha256"])
        self.assertEqual(
            first["curriculum_namespace"],
            curriculum_namespace_for_source_ref(
                "course-a:lesson-1#kc_state_definition", "kc_state_definition"
            ),
        )
        with self.assertRaisesRegex(LearningRecordError, "exact target"):
            curriculum_namespace_for_source_ref(
                "course-a:lesson-1#kc_other", "kc_state_definition"
            )

    def test_outbox_builder_derives_outcome_and_persists_no_raw_learner_text(
        self,
    ) -> None:
        event = self._event("initial")
        validate_learning_outbox_event(event)
        self.assertEqual(event["schema"], LEARNING_OUTBOX_EVENT_SCHEMA)
        self.assertEqual(event["data"]["outcome"], "correct")
        serialized = json.dumps(event, ensure_ascii=False)
        self.assertNotIn("learner_response", serialized)
        self.assertNotIn("这段原文", serialized)

        no_authority = self._component(authority=False)
        with self.assertRaisesRegex(LearningRecordError, "grading authority"):
            self._event("untrusted", component=no_authority)
        mismatched = self._evidence("mismatch", kc_id="kc_another_component")
        with self.assertRaisesRegex(LearningRecordError, "exactly"):
            build_learning_evidence_outbox_event(
                learner_key=self.learner_key,
                knowledge_component=self._component(),
                evidence=mismatched,
                expected_version=0,
                committed_at_utc=self._utc(self.clock.value),
                commit_receipt_id="turn_committed:mismatch",
            )

        forged = deepcopy(event)
        forged["data"]["mastery_delta"] = 1.0
        with self.assertRaisesRegex(LearningRecordError, "strict object"):
            validate_learning_outbox_event(forged)

    def test_first_authoritative_outcome_is_durable_and_idempotent(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "learning-records.jsonl"
            event = self._event("initial")
            store = LearningRecordStore(path, clock=self.clock)
            first = store.apply_outbox_event(event)
            self.assertTrue(first.applied)
            self.assertEqual(first.committed_target_version, 1)
            schedule = first.current_record["schedule"]
            self.assertEqual(schedule["state"], "scheduled")
            self.assertEqual(schedule["interval_days"], 1)
            self.assertEqual(schedule["repetition"], 1)
            self.assertEqual(schedule["due_at_utc"], "2026-08-13T05:00:00Z")

            replay = store.apply_outbox_event(event)
            self.assertFalse(replay.applied)
            self.assertEqual(len(store.events), 1)

            restarted = LearningRecordStore(path, clock=self.clock)
            record = restarted.get_learner_record(self.learner_key)
            self.assertIsNotNone(record)
            self.assertEqual(record["schema"], LEARNING_RECORD_SCHEMA)
            validate_learning_record(record)
            self.assertEqual(
                record["knowledge_components"]
                [event["target"]["curriculum_namespace"]]
                ["kc_state_definition"]["schedule"]["due_at_utc"],
                "2026-08-13T05:00:00Z",
            )
            self.assertNotIn("这段原文", path.read_text(encoding="utf-8"))
            if os.name == "posix":
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_session_logical_evidence_uses_trusted_turn_commit_wall_clock(self) -> None:
        """A live-session logical timestamp must never make a review overdue."""

        logical_time = datetime(2000, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
        evidence = self._evidence(
            "logical.clock",
            observed_at=logical_time,
        )
        first = build_learning_evidence_outbox_event(
            learner_key=self.learner_key,
            knowledge_component=self._component(),
            evidence=evidence,
            expected_version=0,
            committed_at_utc="2026-08-12T05:00:00Z",
            commit_receipt_id="turn_committed:session_001:r1",
        )
        replay = build_learning_evidence_outbox_event(
            learner_key=self.learner_key,
            knowledge_component=self._component(),
            evidence=evidence,
            expected_version=0,
            committed_at_utc="2026-08-12T05:00:00Z",
            commit_receipt_id="turn_committed:session_001:r1",
        )
        self.assertEqual(first["event_id"], replay["event_id"])
        self.assertEqual(first["occurred_at_utc"], "2026-08-12T05:00:00Z")
        self.assertEqual(first["data"]["evidence_observed_at"], "2000-01-01T00:00:01Z")
        self.assertEqual(first["data"]["evidence_time_basis"], "session_logical")
        with TemporaryDirectory() as directory:
            result = LearningRecordStore(
                Path(directory) / "learning-records.jsonl", clock=self.clock
            ).apply_outbox_event(first)
        self.assertEqual(
            result.current_record["schedule"]["due_at_utc"],
            "2026-08-13T05:00:00Z",
        )

        changed_receipt = build_learning_evidence_outbox_event(
            learner_key=self.learner_key,
            knowledge_component=self._component(),
            evidence=evidence,
            expected_version=0,
            committed_at_utc="2026-08-12T05:00:00Z",
            commit_receipt_id="turn_committed:session_001:r2",
        )
        self.assertNotEqual(first["event_id"], changed_receipt["event_id"])

    def test_due_projection_survives_restart_and_claim_is_single_kc_cas(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "learning-records.jsonl"
            store = LearningRecordStore(path, clock=self.clock)
            initial = store.apply_outbox_event(self._event("initial"))
            self.assertEqual(store.list_due_reviews(self.learner_key), ())

            self.clock.advance(days=1)
            restarted = LearningRecordStore(path, clock=self.clock)
            due = restarted.list_due_reviews(self.learner_key)
            self.assertEqual(len(due), 1)
            self.assertEqual(due[0]["knowledge_component_id"], "kc_state_definition")
            self.assertEqual(due[0]["expected_version"], 1)
            claim = restarted.claim_due_review(
                learner_key=self.learner_key,
                review_id=due[0]["review_id"],
                expected_version=due[0]["expected_version"],
                idempotency_key="claim.initial",
            )
            self.assertTrue(claim.applied)
            self.assertEqual(claim.current_record["schedule"]["state"], "in_progress")
            self.assertEqual(
                claim.current_record["schedule"]["lease_expires_at_utc"],
                "2026-08-13T05:15:00Z",
            )
            duplicate = restarted.claim_due_review(
                learner_key=self.learner_key,
                review_id=due[0]["review_id"],
                expected_version=due[0]["expected_version"],
                idempotency_key="claim.initial",
            )
            self.assertFalse(duplicate.applied)
            with self.assertRaisesRegex(LearningRecordConflictError, "currently due"):
                restarted.claim_due_review(
                    learner_key=self.learner_key,
                    review_id=due[0]["review_id"],
                    expected_version=claim.current_record["version"],
                    idempotency_key="claim.conflict",
                )
            self.assertEqual(initial.current_record["version"], 1)

    def test_correct_reviews_follow_fixed_ladder_then_cap_at_sixty_days(self) -> None:
        with TemporaryDirectory() as directory:
            store = LearningRecordStore(
                Path(directory) / "learning-records.jsonl", clock=self.clock
            )
            result = store.apply_outbox_event(self._event("initial"))
            observed_intervals = [result.current_record["schedule"]["interval_days"]]
            for index in range(1, 8):
                due_at = datetime.fromisoformat(
                    result.current_record["schedule"]["due_at_utc"].replace(
                        "Z", "+00:00"
                    )
                )
                self.clock.value = due_at
                due = store.list_due_reviews(self.learner_key)[0]
                claim = store.claim_due_review(
                    learner_key=self.learner_key,
                    review_id=due["review_id"],
                    expected_version=due["expected_version"],
                    idempotency_key=f"claim.{index}",
                )
                schedule = claim.current_record["schedule"]
                result = store.apply_outbox_event(
                    self._event(
                        f"review.{index}",
                        expected_version=claim.current_record["version"],
                        review_id=schedule["active_review_id"],
                        lease_id=schedule["active_lease_id"],
                    )
                )
                observed_intervals.append(
                    result.current_record["schedule"]["interval_days"]
                )
            self.assertEqual(observed_intervals, [1, 3, 7, 14, 30, 60, 60, 60])
            self.assertEqual(
                result.current_record["schedule"]["repetition"],
                len(REVIEW_INTERVAL_DAYS),
            )

    def test_partial_or_incorrect_review_resets_interval_and_counts_lapse(self) -> None:
        with TemporaryDirectory() as directory:
            store = LearningRecordStore(
                Path(directory) / "learning-records.jsonl", clock=self.clock
            )
            result = store.apply_outbox_event(self._event("initial"))
            for index, outcome in enumerate(("correct", "partial", "incorrect"), 1):
                self.clock.value = datetime.fromisoformat(
                    result.current_record["schedule"]["due_at_utc"].replace(
                        "Z", "+00:00"
                    )
                )
                due = store.list_due_reviews(self.learner_key)[0]
                claim = store.claim_due_review(
                    learner_key=self.learner_key,
                    review_id=due["review_id"],
                    expected_version=due["expected_version"],
                    idempotency_key=f"claim.reset.{index}",
                )
                schedule = claim.current_record["schedule"]
                result = store.apply_outbox_event(
                    self._event(
                        f"reset.{index}",
                        outcome=outcome,
                        expected_version=claim.current_record["version"],
                        review_id=schedule["active_review_id"],
                        lease_id=schedule["active_lease_id"],
                    )
                )
            schedule = result.current_record["schedule"]
            self.assertEqual(schedule["interval_days"], 1)
            self.assertEqual(schedule["repetition"], 0)
            self.assertEqual(schedule["lapse_count"], 2)

    def test_review_lease_expires_and_can_be_reclaimed_without_state_rewrite(self) -> None:
        with TemporaryDirectory() as directory:
            store = LearningRecordStore(
                Path(directory) / "learning-records.jsonl", clock=self.clock
            )
            store.apply_outbox_event(self._event("initial"))
            self.clock.advance(days=1)
            due = store.list_due_reviews(self.learner_key)[0]
            first = store.claim_due_review(
                learner_key=self.learner_key,
                review_id=due["review_id"],
                expected_version=due["expected_version"],
                idempotency_key="claim.expiring",
            )
            self.clock.advance(minutes=15)
            expired = store.list_due_reviews(self.learner_key)
            self.assertEqual(expired[0]["claim_state"], "expired")
            second = store.claim_due_review(
                learner_key=self.learner_key,
                review_id=expired[0]["review_id"],
                expected_version=expired[0]["expected_version"],
                idempotency_key="claim.reclaimed",
            )
            self.assertNotEqual(
                first.current_record["schedule"]["active_lease_id"],
                second.current_record["schedule"]["active_lease_id"],
            )
            self.assertEqual(second.current_record["version"], 3)

    def test_release_requires_active_lease_and_preserves_original_due_date(self) -> None:
        with TemporaryDirectory() as directory:
            store = LearningRecordStore(
                Path(directory) / "learning-records.jsonl", clock=self.clock
            )
            initial = store.apply_outbox_event(self._event("initial"))
            original_due = initial.current_record["schedule"]["due_at_utc"]
            self.clock.advance(days=1)
            due = store.list_due_reviews(self.learner_key)[0]
            claim = store.claim_due_review(
                learner_key=self.learner_key,
                review_id=due["review_id"],
                expected_version=1,
                idempotency_key="claim.release",
            )
            released = store.release_review_lease(
                learner_key=self.learner_key,
                review_id=due["review_id"],
                expected_version=claim.current_record["version"],
                idempotency_key="release.one",
            )
            self.assertEqual(released.current_record["schedule"]["state"], "scheduled")
            self.assertEqual(
                released.current_record["schedule"]["due_at_utc"], original_due
            )
            self.assertEqual(len(store.list_due_reviews(self.learner_key)), 1)

    def test_one_kc_event_does_not_mutate_another_kc(self) -> None:
        with TemporaryDirectory() as directory:
            store = LearningRecordStore(
                Path(directory) / "learning-records.jsonl", clock=self.clock
            )
            first_component = self._component("kc_state_definition")
            second_component = self._component("kc_transition_rule")
            first = store.apply_outbox_event(
                self._event("kc.one", component=first_component)
            )
            second = store.apply_outbox_event(
                self._event("kc.two", component=second_component)
            )
            before_second = deepcopy(second.current_record)
            first_current = store.get_knowledge_component_record(
                learner_key=self.learner_key,
                curriculum_namespace=first.current_record["curriculum_namespace"],
                knowledge_component_id="kc_state_definition",
            )
            self.clock.advance(hours=1)
            store.apply_outbox_event(
                self._event(
                    "kc.one.again",
                    component=first_component,
                    expected_version=first_current["version"],
                )
            )
            after_second = store.get_knowledge_component_record(
                learner_key=self.learner_key,
                curriculum_namespace=second.current_record["curriculum_namespace"],
                knowledge_component_id="kc_transition_rule",
            )
            self.assertEqual(after_second, before_second)

    def test_same_kc_id_across_curricula_has_independent_schedules(self) -> None:
        with TemporaryDirectory() as directory:
            store = LearningRecordStore(
                Path(directory) / "learning-records.jsonl", clock=self.clock
            )
            mathematics = self._component(scope="math:dynamic-programming")
            biology = self._component(scope="biology:population-recurrence")
            math_record = store.apply_committed_evidence_event(
                self._event("curriculum.math", component=mathematics)
            ).current_record
            bio_record = store.apply_committed_evidence_event(
                self._event("curriculum.bio", component=biology)
            ).current_record

            self.assertNotEqual(
                math_record["curriculum_namespace"],
                bio_record["curriculum_namespace"],
            )
            recovered = store.get_learner_record(self.learner_key)
            self.assertEqual(len(recovered["knowledge_components"]), 2)
            self.assertEqual(math_record["version"], 1)
            self.assertEqual(bio_record["version"], 1)

    def test_batch_is_atomic_when_one_target_version_is_stale(self) -> None:
        with TemporaryDirectory() as directory:
            store = LearningRecordStore(
                Path(directory) / "learning-records.jsonl", clock=self.clock
            )
            first = self._event("batch.first", expected_version=0)
            second = self._event("batch.stale", expected_version=0)
            with self.assertRaisesRegex(LearningRecordConflictError, "stale"):
                store.apply_outbox_batch([first, second])
            self.assertEqual(store.events, ())
            self.assertFalse(Path(directory, "learning-records.jsonl").exists())

    def test_source_cancellation_suspends_only_current_source_and_purge_compacts(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "learning-records.jsonl"
            store = LearningRecordStore(path, clock=self.clock)
            first = store.apply_outbox_event(self._event("cancel.source"))
            another_key = mint_learner_key(
                "student-002", tenant_id="school-001", secret=self.secret
            )
            other_event = build_learning_evidence_outbox_event(
                learner_key=another_key,
                knowledge_component=self._component(),
                evidence=self._evidence("other.source"),
                expected_version=0,
                committed_at_utc=self._utc(self.clock.value),
                commit_receipt_id="turn_committed:other.source",
            )
            store.apply_outbox_event(other_event)
            cancelled = store.cancel_source_observations(
                [first.current_record["schedule"]["source_observation_id"]]
            )
            self.assertEqual(len(cancelled), 1)
            self.assertEqual(cancelled[0].current_record["schedule"]["state"], "suspended")
            self.assertEqual(
                store.get_learner_record(another_key)["version"], 1
            )
            receipt = store.purge_learner(self.learner_key)
            self.assertEqual(receipt["learners"], 1)
            self.assertGreaterEqual(receipt["events"], 2)
            self.assertIsNone(store.get_learner_record(self.learner_key))
            self.assertIsNotNone(store.get_learner_record(another_key))
            restarted = LearningRecordStore(path, clock=self.clock)
            self.assertIsNone(restarted.get_learner_record(self.learner_key))
            self.assertIsNotNone(restarted.get_learner_record(another_key))
            with self.assertRaisesRegex(
                LearningRecordConflictError, "permanently erased"
            ):
                restarted.apply_outbox_event(self._event("cancel.source"))
            self.assertIsNone(restarted.get_learner_record(self.learner_key))
            self.assertEqual(restarted.recover().event_count, 1)
            tombstone_path = path.with_name(
                f".{path.name}.erasure_tombstones.json"
            )
            tombstone = json.loads(tombstone_path.read_text(encoding="utf-8"))
            self.assertEqual(
                tombstone["schema"],
                "teaching_skill_miner.teacher_agent_learning_erasure_tombstones.v1",
            )
            self.assertEqual(set(tombstone["tombstones"]), {self.learner_key})
            serialized = json.dumps(tombstone, ensure_ascii=False)
            self.assertNotIn("evidence", serialized)
            self.assertNotIn("source", serialized)
            self.assertNotIn("question", serialized)
            if os.name == "posix":
                self.assertEqual(tombstone_path.stat().st_mode & 0o777, 0o600)

    def test_purge_of_unknown_key_still_fences_delayed_outbox(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "learning-records.jsonl"
            store = LearningRecordStore(path, clock=self.clock)
            receipt = store.purge_learner(self.learner_key)
            self.assertEqual(receipt, {"learners": 1, "events": 0, "erasure_fence": 1})
            self.assertTrue(store.is_learner_erased(self.learner_key))
            old_outbox = self._event("delayed.after.purge")
            with self.assertRaisesRegex(
                LearningRecordConflictError, "permanently erased"
            ):
                store.apply_outbox_event(old_outbox)
            self.assertIsNone(store.get_learner_record(self.learner_key))
            self.assertEqual(store.purge_learner(self.learner_key)["learners"], 0)

    def test_erasure_fence_hides_record_if_compaction_crashes_and_retry_finishes(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "learning-records.jsonl"
            store = LearningRecordStore(path, clock=self.clock)
            old_outbox = self._event("purge.crash")
            store.apply_outbox_event(old_outbox)
            original_replace = store._atomic_replace
            with patch.object(
                store,
                "_atomic_replace",
                side_effect=LearningRecordStoreError("simulated compaction crash"),
            ):
                with self.assertRaisesRegex(
                    LearningRecordStoreError, "simulated compaction"
                ):
                    store.purge_learner(self.learner_key)

            fenced = LearningRecordStore(path, clock=self.clock)
            self.assertIsNone(fenced.get_learner_record(self.learner_key))
            self.assertEqual(fenced.recover().records, {})
            self.assertEqual(fenced.events, ())
            with self.assertRaisesRegex(
                LearningRecordConflictError, "permanently erased"
            ):
                fenced.apply_outbox_event(old_outbox)

            with patch.object(fenced, "_atomic_replace", wraps=original_replace):
                receipt = fenced.purge_learner(self.learner_key)
            self.assertEqual(receipt["events"], 1)
            self.assertEqual(path.read_bytes(), b"")

    def test_truncated_tail_is_repaired_but_complete_tamper_fails_closed(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "learning-records.jsonl"
            LearningRecordStore(path, clock=self.clock).apply_outbox_event(
                self._event("durability")
            )
            with path.open("ab") as stream:
                stream.write(b'{"truncated":true')
            repaired = LearningRecordStore(path, clock=self.clock)
            self.assertEqual(repaired.recover().event_count, 1)
            self.assertTrue(path.read_bytes().endswith(b"\n"))
            lines = path.read_text(encoding="utf-8").splitlines()
            envelope = json.loads(lines[0])
            envelope["target_version_after"] = 99
            path.write_text(
                json.dumps(
                    envelope,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(LearningRecordStoreError, "replay validation"):
                LearningRecordStore(path, clock=self.clock)

    @unittest.skipIf(os.name != "posix", "symlink safety is POSIX-specific")
    def test_store_and_parent_symlinks_fail_closed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.jsonl"
            target.write_text("", encoding="utf-8")
            store_link = root / "store-link.jsonl"
            store_link.symlink_to(target)
            with self.assertRaisesRegex(LearningRecordStoreError, "non-symlink"):
                LearningRecordStore(store_link, clock=self.clock)

            real_parent = root / "real-parent"
            real_parent.mkdir()
            parent_link = root / "parent-link"
            parent_link.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(LearningRecordStoreError, "must not be a symlink"):
                LearningRecordStore(parent_link / "records.jsonl", clock=self.clock)

    def test_two_store_instances_are_process_locked_and_one_stale_cas_loses(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "learning-records.jsonl"
            first_store = LearningRecordStore(path, clock=self.clock)
            second_store = LearningRecordStore(path, clock=self.clock)
            first_event = self._event("race.first", expected_version=0)
            second_event = self._event("race.second", expected_version=0)

            def apply(store: LearningRecordStore, event: dict[str, object]) -> str:
                try:
                    store.apply_outbox_event(event)
                    return "applied"
                except LearningRecordConflictError:
                    return "conflict"

            with ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = list(
                    executor.map(
                        lambda pair: apply(*pair),
                        ((first_store, first_event), (second_store, second_event)),
                    )
                )
            self.assertCountEqual(outcomes, ["applied", "conflict"])
            self.assertEqual(
                LearningRecordStore(path, clock=self.clock).recover().event_count, 1
            )

    def test_two_sessions_committed_evidence_are_atomically_sequenced_exactly_once(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "learning-records.jsonl"
            first_store = LearningRecordStore(path, clock=self.clock)
            second_store = LearningRecordStore(path, clock=self.clock)
            first_event = self._event("session.one", expected_version=0)
            second_event = self._event("session.two", expected_version=0)

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(
                    executor.map(
                        lambda pair: pair[0].apply_committed_evidence_event(pair[1]),
                        (
                            (first_store, first_event),
                            (second_store, second_event),
                        ),
                    )
                )

            self.assertTrue(all(result.applied for result in results))
            restarted = LearningRecordStore(path, clock=self.clock)
            self.assertEqual(restarted.recover().event_count, 2)
            record = restarted.get_learner_record(self.learner_key)
            component = next(
                iter(next(iter(record["knowledge_components"].values())).values())
            )
            self.assertEqual(component["version"], 2)
            replay = restarted.apply_committed_evidence_event(first_event)
            self.assertFalse(replay.applied)
            self.assertEqual(restarted.recover().event_count, 2)

    def test_delayed_committed_evidence_is_audited_without_regressing_due_date(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "learning-records.jsonl"
            store = LearningRecordStore(path, clock=self.clock)
            older = self._event("older.commit", expected_version=0)
            self.clock.advance(hours=2)
            newer = self._event("newer.commit", expected_version=0)
            applied_newer = store.apply_committed_evidence_event(newer)
            newer_due = applied_newer.current_record["schedule"]["due_at_utc"]

            applied_older = store.apply_committed_evidence_event(older)

            self.assertTrue(applied_older.applied)
            self.assertEqual(applied_older.current_record["version"], 2)
            self.assertEqual(
                applied_older.current_record["schedule"]["due_at_utc"], newer_due
            )
            self.assertEqual(store.recover().event_count, 2)

    def test_json_schemas_accept_python_contract_outputs(self) -> None:
        try:
            from jsonschema import Draft202012Validator
            from referencing import Registry, Resource
        except ImportError:  # pragma: no cover - jsonschema is a dev dependency.
            self.skipTest("jsonschema is unavailable")
        root = Path(__file__).resolve().parent.parent / "schema"
        schemas = {
            path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in (
                root / "teacher_agent_learning_outbox_event.schema.json",
                root / "teacher_agent_learning_record.schema.json",
                root / "teacher_agent_learning_store_event.schema.json",
                root / "teacher_agent_learning_erasure_tombstones.schema.json",
            )
        }
        for schema in schemas.values():
            Draft202012Validator.check_schema(schema)

        with TemporaryDirectory() as directory:
            store = LearningRecordStore(
                Path(directory) / "learning-records.jsonl", clock=self.clock
            )
            event = self._event("schema")
            result = store.apply_outbox_event(event)
            Draft202012Validator(
                schemas["teacher_agent_learning_outbox_event.schema.json"]
            ).validate(event)
            record = store.get_learner_record(self.learner_key)
            Draft202012Validator(
                schemas["teacher_agent_learning_record.schema.json"]
            ).validate(record)
            self.assertEqual(result.current_record["version"], 1)
            self.assertEqual(store.events[0]["schema"], LEARNING_STORE_EVENT_SCHEMA)
            outbox_schema = schemas[
                "teacher_agent_learning_outbox_event.schema.json"
            ]
            registry = Registry().with_resource(
                str(outbox_schema["$id"]), Resource.from_contents(outbox_schema)
            )
            Draft202012Validator(
                schemas["teacher_agent_learning_store_event.schema.json"],
                registry=registry,
            ).validate(store.events[0])
            store.purge_learner(self.learner_key)
            tombstones = json.loads(store.tombstone_path.read_text(encoding="utf-8"))
            Draft202012Validator(
                schemas[
                    "teacher_agent_learning_erasure_tombstones.schema.json"
                ]
            ).validate(tombstones)

    def test_learning_contract_schemas_are_explicit_release_resources(self) -> None:
        root = Path(__file__).resolve().parent.parent
        names = (
            "teacher_agent_learning_erasure_tombstones.schema.json",
            "teacher_agent_learning_outbox_event.schema.json",
            "teacher_agent_learning_record.schema.json",
            "teacher_agent_learning_store_event.schema.json",
        )
        pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
        reviewed = set(
            (root / "release/public_json_resources.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        for name in names:
            with self.subTest(name=name):
                resource = f"schema/{name}"
                self.assertIn(f'"{resource}"', pyproject)
                self.assertIn(resource, reviewed)


if __name__ == "__main__":
    unittest.main()
