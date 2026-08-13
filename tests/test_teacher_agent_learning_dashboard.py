from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import base64
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import hmac
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from teaching_skill_miner.io_utils import project_root
from teaching_skill_miner.student_model import (
    initialize_student_model,
    update_student_model,
)
from teaching_skill_miner.teacher_agent import (
    _refresh_integrity,
    advance_teacher_agent_session as real_advance_teacher_agent_session,
)
from teaching_skill_miner.teacher_agent_authority import (
    AUTHENTICATED_TEACHER_ACTOR,
    AUTHORITY_ASSURANCE,
    TEACHER_AUTHORITY_SCHEMA,
    TeacherAuthorityVerifier,
    canonical_bytes,
    canonical_sha256,
)
from teaching_skill_miner.teacher_agent_curriculum import (
    build_teacher_owned_curriculum_spec,
)
from teaching_skill_miner.teacher_agent_curriculum_authority_store import (
    TeachingCurriculumAuthorityStore,
)
from teaching_skill_miner.teacher_agent_curriculum_signing import (
    CurriculumSigningKeyring,
)
from teaching_skill_miner.teacher_agent_dashboard import (
    TeacherAgentDashboardError,
    build_teacher_agent_dashboard_snapshot,
)
from teaching_skill_miner.teacher_agent_learning_records import (
    LearningRecordStoreError,
)
from teaching_skill_miner.teacher_agent_syllabus import _seal_generated_syllabus


AUTHORITY_NOW = datetime(2026, 8, 12, 4, 0, tzinfo=timezone.utc)
AUTHORITY_SCOPE = "scope_" + "a" * 48
AUTHORITY_GATEWAY_KEY = b"g" * 32
AUTHORITY_KEYRING_KEY = b"k" * 32


class TeacherAgentLearningDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = project_root()
        cls.library_path = root / "data/teacher_agent_skill_library.json"
        cls.input_path = root / "data/teacher_agent_demo_input.json"
        cls.cases_path = root / "data/teacher_agent_evaluation_cases.json"
        cls.secret = b"dashboard-learning-integration-secret-32-bytes"

    def _snapshot(self, directory: str):
        return self._build_snapshot(Path(directory))

    def _build_snapshot(
        self,
        root: Path,
        *,
        learner_tenant_id: str = "school-integration",
        trusted_learner_profile_ref: str | None = None,
    ):
        verifier = TeacherAuthorityVerifier(
            key=AUTHORITY_GATEWAY_KEY,
            scope_id=AUTHORITY_SCOPE,
            scope_key_version="k1",
            replay_store_path=root / "teacher-authority-replay.jsonl",
            clock=lambda: AUTHORITY_NOW,
        )
        keyring = CurriculumSigningKeyring(
            root / "syllabi" / ".curriculum_signing_keyring.json",
            integrity_key=AUTHORITY_KEYRING_KEY,
        )
        authority_store = TeachingCurriculumAuthorityStore(
            root / "syllabi" / ".curriculum_authority.json",
            scope_id=AUTHORITY_SCOPE,
            trusted_teacher_public_keys=keyring.trusted_public_keys,
            gateway_receipt_validator=verifier.verify_verification_receipt,
        )
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.library_path,
            self.input_path,
            self.cases_path,
            store_path=root / "sessions.jsonl",
            learning_record_store_path=root / "learning.jsonl",
            learner_key_secret=self.secret,
            learner_tenant_id=learner_tenant_id,
            trusted_learner_profile_ref=trusted_learner_profile_ref,
            syllabus_store_path=root / "syllabi",
            teacher_authority_verifier=verifier,
            curriculum_authority_store=authority_store,
            curriculum_signing_keyring=keyring,
        )
        syllabus = self._demo_syllabus(snapshot)
        family = snapshot.import_syllabus({"syllabus": syllabus})["version_family"]
        if authority_store.read_family(family["family_id"]) is None:
            spec = self._demo_curriculum_spec(snapshot)
            review_body = {
                "syllabus_id": syllabus["syllabus_id"],
                "teacher_spec": spec,
                "expected_syllabus_version": family["version"],
                "expected_authority_version": 0,
                "curriculum_authority_idempotency_key": "learning-review-fixture-v1",
            }
            reviewed = snapshot.review_curriculum(
                self._gateway_request(
                    verifier,
                    path="api/curriculum/review",
                    body=review_body,
                    nonce="a",
                )
            )
            seal_body = {
                "syllabus_id": syllabus["syllabus_id"],
                "review_id": reviewed["curriculum_authority"]["review"]["review_id"],
                "teacher_confirmed_authority": True,
                "expected_syllabus_version": family["version"],
                "expected_authority_version": 1,
                "curriculum_authority_idempotency_key": "learning-seal-fixture-v1",
            }
            snapshot.seal_curriculum(
                self._gateway_request(
                    verifier,
                    path="api/curriculum/seal",
                    body=seal_body,
                    nonce="b",
                )
            )
        return snapshot

    @staticmethod
    def _gateway_request(
        verifier: TeacherAuthorityVerifier,
        *,
        path: str,
        body: dict,
        nonce: str,
    ) -> dict:
        idempotency_key = body["curriculum_authority_idempotency_key"]
        envelope = {
            "schema": TEACHER_AUTHORITY_SCHEMA,
            "authority_kind": AUTHENTICATED_TEACHER_ACTOR,
            "assurance": AUTHORITY_ASSURANCE,
            "scope_id": verifier.scope_id,
            "scope_key_version": verifier.scope_key_version,
            "actor_principal_sha256": "5" * 64,
            "roles_sha256": "6" * 64,
            "role_policy_sha256": "7" * 64,
            "method": "POST",
            "path": path,
            "body_sha256": canonical_sha256(body),
            "idempotency_key_sha256": sha256(idempotency_key.encode()).hexdigest(),
            "issued_at": AUTHORITY_NOW.isoformat(timespec="seconds").replace(
                "+00:00", "Z"
            ),
            "expires_at": (AUTHORITY_NOW + timedelta(minutes=2))
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "nonce": "tan_" + nonce * 48,
        }
        envelope["authority_id"] = "tauth_" + canonical_sha256(envelope)[:24]
        envelope["signature"] = (
            base64.urlsafe_b64encode(
                hmac.new(
                    AUTHORITY_GATEWAY_KEY,
                    canonical_bytes(envelope),
                    sha256,
                ).digest()
            )
            .decode()
            .rstrip("=")
        )
        return {**deepcopy(body), "_teacher_authority": envelope}

    @staticmethod
    def _demo_syllabus(snapshot) -> dict:
        goal = snapshot.demo_input["goal"]
        return _seal_generated_syllabus(
            {
                "title": goal["concept"],
                "description": goal["objective"],
                "audience": "学习记录集成测试学习者",
                "estimated_duration_minutes": 45,
                "learning_objectives": [goal["objective"]],
                "prerequisites": [],
                "modules": [
                    {
                        "title": goal["concept"],
                        "description": goal["objective"],
                        "lessons": [
                            {
                                "title": goal["concept"],
                                "objective": goal["objective"],
                                "summary": goal["objective"],
                                "duration_minutes": 45,
                                "knowledge_components": list(
                                    goal["knowledge_components"]
                                ),
                                "materials": deepcopy(goal["materials"]),
                            }
                        ],
                    }
                ],
            },
            model="fixture-teacher",
            source_resource_ids=["source_demo_teacher"],
            created_at="2026-08-12T04:00:00Z",
        )

    @staticmethod
    def _demo_curriculum_spec(snapshot) -> dict:
        goal = snapshot.demo_input["goal"]
        labels = list(goal["knowledge_components"])
        source_id = "source_demo_teacher"
        return build_teacher_owned_curriculum_spec(
            title=goal["concept"],
            lessons=[
                {
                    "legacy_lesson_id": "lesson_01_01",
                    "title": goal["concept"],
                    "objective": goal["objective"],
                    "knowledge_components": [
                        {
                            "label": label,
                            "prerequisites": [],
                            "source_resource_ids": [source_id],
                        }
                        for label in labels
                    ],
                }
            ],
            source_spans=[
                {
                    "resource_id": source_id,
                    "content_sha256": sha256(b"demo teacher source").hexdigest(),
                    "excerpt_sha256": sha256(b"demo teacher excerpt").hexdigest(),
                    "locator": {"kind": "line", "start": 1, "end": 1},
                }
            ],
            factual_claims=[
                {
                    "statement": claim["statement"],
                    "knowledge_components": list(claim["knowledge_components"]),
                    "source_resource_ids": [source_id],
                }
                for claim in goal["knowledge_spec"]["canonical_claims"]
            ],
        )

    @staticmethod
    def _sealed_lesson_payload(snapshot) -> dict:
        family = snapshot.syllabus_version_store.list_families()[0]
        published = next(
            row
            for row in family["revisions"]
            if row["revision_id"] == family["published_revision_id"]
        )
        return snapshot.syllabus_lesson_payload(
            published["syllabus_id"], "lesson_01_01"
        )

    def test_authenticated_scope_owns_learner_identity_across_sessions_and_restart(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trusted_ref = "profile_" + "a" * 64

            def build():
                return self._build_snapshot(
                    root,
                    learner_tenant_id="authenticated-scope-1",
                    trusted_learner_profile_ref=trusted_ref,
                )

            snapshot = build()
            first = self._start(snapshot, "browser-chosen-a", "start.trusted.a")
            second = self._start(snapshot, "browser-chosen-b", "start.trusted.b")
            first_record = snapshot.sessions[first["session_id"]]
            second_record = snapshot.sessions[second["session_id"]]
            self.assertEqual(
                first_record.session["student_profile"]["profile_ref"], trusted_ref
            )
            self.assertEqual(
                second_record.session["student_profile"]["profile_ref"], trusted_ref
            )
            self.assertEqual(
                snapshot._learner_key_for_record(first_record),
                snapshot._learner_key_for_record(second_record),
            )
            with patch(
                "teaching_skill_miner.teacher_agent_dashboard.advance_teacher_agent_session",
                side_effect=self._authoritative_advance,
            ):
                snapshot.step(
                    self._step_body(
                        first,
                        key="step.trusted.evidence",
                        answer="authoritative cross-session answer",
                    )
                )
            learner = next(
                iter(snapshot.learning_record_store.recover().records.values())
            )
            component = next(
                iter(next(iter(learner["knowledge_components"].values())).values())
            )
            due_at = datetime.fromisoformat(
                component["schedule"]["due_at_utc"].replace("Z", "+00:00")
            )
            snapshot.learning_record_store._clock = lambda: due_at
            due = snapshot.list_due_learning_reviews(self._guards(second))
            self.assertEqual(len(due["reviews"]), 1)
            capabilities = snapshot.bootstrap()["interaction_contract"]
            self.assertTrue(capabilities["learning_record_identity_authenticated"])
            self.assertTrue(
                capabilities["learning_record_cross_user_authorization_established"]
            )
            restarted = build()
            restored = restarted._load_session_record(first["session_id"])
            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual(
                restarted._learner_key_for_record(restored),
                snapshot._learner_key_for_record(first_record),
            )

    @staticmethod
    def _profile(snapshot, profile_ref: str) -> dict:
        profile = deepcopy(snapshot.demo_input["student_profile"])
        profile["profile_ref"] = profile_ref
        return profile

    def _start(self, snapshot, profile_ref: str, key: str) -> dict:
        payload = self._sealed_lesson_payload(snapshot)
        return snapshot.start(
            {
                "goal": payload["goal"],
                "syllabus_ref": payload["syllabus_ref"],
                "student_profile": self._profile(snapshot, profile_ref),
                "start_idempotency_key": key,
                "profile_revision": f"{profile_ref}-v1",
                "profile_display_name": "持久学习者",
            }
        )

    @staticmethod
    def _step_body(started: dict, *, key: str, answer: str) -> dict:
        return {
            "session_id": started["session_id"],
            "expected_round": started["rounds_completed"],
            "expected_question_id": started["expected_question_id"],
            "expected_context_version": started["context_version"],
            "profile_revision": started["profile_summary"]["profile_revision"],
            "idempotency_key": key,
            "learner_response": answer,
            "signal": "correct",
            "signal_confidence": 0.9,
        }

    @staticmethod
    def _guards(session: dict) -> dict:
        return {
            "session_id": session["session_id"],
            "expected_round": session["rounds_completed"],
            "expected_question_id": session["expected_question_id"],
            "expected_context_version": session["context_version"],
            "profile_revision": session["profile_summary"]["profile_revision"],
        }

    @staticmethod
    def _authoritative_advance(session: dict, **kwargs) -> dict:
        answer = str(kwargs.get("learner_response", "answer"))
        consumed_action = deepcopy(session["current_action"])
        signal = (
            "partial"
            if answer.startswith("partial:")
            else "misconception"
            if answer.startswith("incorrect:")
            else "correct"
        )
        alignment = {
            "correct": "aligned",
            "partial": "partially_aligned",
            "misconception": "contradicted",
        }[signal]
        advance_kwargs = dict(kwargs)
        advance_kwargs["signal"] = signal
        if signal == "misconception":
            advance_kwargs["misconception_tag"] = "review_incorrect_claim"
        advanced = real_advance_teacher_agent_session(session, **advance_kwargs)
        model = initialize_student_model(
            advanced["student_state"].get("knowledge_mastery", {}),
            goal=advanced["goal"],
        )
        active_labels = consumed_action.get("knowledge_components", [])
        kc_id, component = next(
            (kc_id, component)
            for kc_id, component in model["knowledge_components"].items()
            if component["label"] in active_labels
        )
        label = component["label"]
        spec = advanced["goal"]["knowledge_spec"]
        rubric_id = next(
            (
                f"teacher_rubric:{row['criterion_id']}"
                for row in spec["rubric_criteria"]
                if row.get("knowledge_component") == label
            ),
            None,
        )
        if rubric_id is None:
            rubric_id = next(
                f"teacher_claim:{row['claim_id']}"
                for row in spec["canonical_claims"]
                if label in row.get("knowledge_components", [])
            )
        action_id = consumed_action["action_id"]
        contract = consumed_action.get("teacher_action", {}).get(
            "question_contract", {}
        )
        question_id = contract.get("question_id") or action_id
        suffix = sha256(answer.encode("utf-8")).hexdigest()[:16]
        updated = update_student_model(
            model,
            signal=signal,
            confidence=0.9,
            focus_dimension="conceptual",
            knowledge_component_ids=[kc_id],
            answer_alignment=alignment,
            assessment_eligible=True,
            authoritative=True,
            round_number=int(advanced["round"]),
            evidence_id=f"evidence_{suffix}",
            item_id=action_id,
            question_id=question_id,
            rubric_id=rubric_id,
            observed_at=f"2000-01-01T00:00:{int(advanced['round']):02d}Z",
            time_basis="session_logical",
            source="validated_teacher_rubric",
        )
        advanced["student_state"]["student_model"] = updated
        return _refresh_integrity(advanced)

    def _claim_first_due_review(
        self,
        snapshot,
        session: dict,
        *,
        idempotency_key: str,
    ) -> tuple[dict, dict]:
        learner = next(iter(snapshot.learning_record_store.recover().records.values()))
        component = next(
            iter(next(iter(learner["knowledge_components"].values())).values())
        )
        due_at = datetime.fromisoformat(
            component["schedule"]["due_at_utc"].replace("Z", "+00:00")
        )
        snapshot.learning_record_store._clock = lambda: due_at
        due = snapshot.list_due_learning_reviews(self._guards(session))
        self.assertEqual(len(due["reviews"]), 1)
        review = due["reviews"][0]
        claim = snapshot.claim_due_learning_review(
            {
                **self._guards(session),
                "review_id": review["review_id"],
                "expected_version": review["expected_version"],
                "review_idempotency_key": idempotency_key,
            }
        )
        return review, claim

    def test_turn_outbox_drains_exactly_once_and_cached_retry_does_not_duplicate(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            snapshot = self._snapshot(directory)
            started = self._start(snapshot, "student-outbox-1", "start.outbox")
            body = self._step_body(
                started, key="step.outbox", answer="authoritative answer one"
            )
            with patch(
                "teaching_skill_miner.teacher_agent_dashboard.advance_teacher_agent_session",
                side_effect=self._authoritative_advance,
            ):
                response = snapshot.step(body)

            self.assertEqual(snapshot.learning_record_store.recover().event_count, 1)
            self.assertEqual(
                snapshot.store.recover_session(started["session_id"])[
                    "learning_outbox"
                ],
                {},
            )
            self.assertEqual(snapshot.step(body), response)
            self.assertEqual(snapshot.learning_record_store.recover().event_count, 1)

            restarted = self._snapshot(directory)
            self.assertEqual(restarted.learning_record_store.recover().event_count, 1)
            self.assertEqual(
                restarted.store.recover_session(started["session_id"])[
                    "learning_outbox"
                ],
                {},
            )

    def test_crash_windows_recover_after_session_commit_and_after_learning_apply(
        self,
    ) -> None:
        for failure_point in ("before_apply", "after_apply_before_clear"):
            with (
                self.subTest(failure_point=failure_point),
                TemporaryDirectory() as directory,
            ):
                snapshot = self._snapshot(directory)
                started = self._start(
                    snapshot,
                    f"student-{failure_point}",
                    f"start.{failure_point}",
                )
                body = self._step_body(
                    started,
                    key=f"step.{failure_point}",
                    answer=f"answer {failure_point}",
                )
                patches = [
                    patch(
                        "teaching_skill_miner.teacher_agent_dashboard.advance_teacher_agent_session",
                        side_effect=self._authoritative_advance,
                    )
                ]
                if failure_point == "before_apply":
                    patches.append(
                        patch.object(
                            type(snapshot),
                            "_drain_learning_outbox_locked",
                            autospec=True,
                            return_value=0,
                        )
                    )
                else:
                    original = (
                        snapshot.learning_record_store.apply_committed_evidence_event
                    )

                    def apply_then_interrupt(event):
                        original(event)
                        raise LearningRecordStoreError("simulated crash after apply")

                    patches.append(
                        patch.object(
                            snapshot.learning_record_store,
                            "apply_committed_evidence_event",
                            side_effect=apply_then_interrupt,
                        )
                    )
                with patches[0], patches[1]:
                    snapshot.step(body)

                stored = snapshot.store.recover_session(started["session_id"])
                self.assertEqual(len(stored["learning_outbox"]), 1)
                expected_before_restart = 0 if failure_point == "before_apply" else 1
                self.assertEqual(
                    snapshot.learning_record_store.recover().event_count,
                    expected_before_restart,
                )

                restarted = self._snapshot(directory)
                self.assertEqual(
                    restarted.learning_record_store.recover().event_count, 1
                )
                self.assertEqual(
                    restarted.store.recover_session(started["session_id"])[
                        "learning_outbox"
                    ],
                    {},
                )

    def test_session_commit_failure_never_writes_learning_store(self) -> None:
        with TemporaryDirectory() as directory:
            snapshot = self._snapshot(directory)
            started = self._start(snapshot, "student-precommit", "start.precommit")
            body = self._step_body(
                started, key="step.precommit", answer="precommit answer"
            )
            original_persist = snapshot._persist_events

            def fail_turn_commit(_snapshot, events):
                if any(event.get("event_type") == "turn_committed" for event in events):
                    raise TeacherAgentDashboardError("simulated session commit failure")
                return original_persist(events)

            with (
                patch(
                    "teaching_skill_miner.teacher_agent_dashboard.advance_teacher_agent_session",
                    side_effect=self._authoritative_advance,
                ),
                patch.object(
                    type(snapshot),
                    "_persist_events",
                    autospec=True,
                    side_effect=fail_turn_commit,
                ),
            ):
                with self.assertRaisesRegex(
                    TeacherAgentDashboardError, "session commit failure"
                ):
                    snapshot.step(body)
            self.assertEqual(snapshot.learning_record_store.recover().event_count, 0)

    def test_learning_store_read_failure_does_not_block_teaching_commit(self) -> None:
        with TemporaryDirectory() as directory:
            snapshot = self._snapshot(directory)
            started = self._start(snapshot, "student-store-read", "start.store-read")
            body = self._step_body(
                started, key="step.store-read", answer="store read outage answer"
            )
            with (
                patch(
                    "teaching_skill_miner.teacher_agent_dashboard.advance_teacher_agent_session",
                    side_effect=self._authoritative_advance,
                ),
                patch.object(
                    snapshot.learning_record_store,
                    "is_learner_erased",
                    side_effect=LearningRecordStoreError("synthetic read outage"),
                ),
            ):
                response = snapshot.step(body)

            self.assertEqual(response["rounds_completed"], 1)
            self.assertEqual(snapshot.learning_record_store.recover().event_count, 1)
            self.assertEqual(
                snapshot.store.recover_session(started["session_id"])[
                    "learning_outbox"
                ],
                {},
            )

    def test_same_learner_kc_two_sessions_never_leave_stale_outbox(self) -> None:
        with TemporaryDirectory() as directory:
            snapshot = self._snapshot(directory)
            first = self._start(snapshot, "student-shared", "start.shared.one")
            second = self._start(snapshot, "student-shared", "start.shared.two")
            with (
                patch(
                    "teaching_skill_miner.teacher_agent_dashboard.advance_teacher_agent_session",
                    side_effect=self._authoritative_advance,
                ),
                patch.object(
                    type(snapshot),
                    "_drain_learning_outbox_locked",
                    autospec=True,
                    return_value=0,
                ),
            ):
                snapshot.step(
                    self._step_body(
                        first, key="step.shared.one", answer="shared answer one"
                    )
                )
                snapshot.step(
                    self._step_body(
                        second, key="step.shared.two", answer="shared answer two"
                    )
                )

            records = [
                snapshot.sessions[first["session_id"]],
                snapshot.sessions[second["session_id"]],
            ]
            self.assertEqual(
                [len(record.learning_outbox) for record in records], [1, 1]
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                list(
                    executor.map(
                        lambda pair: snapshot._drain_learning_outbox(*pair),
                        (
                            (first["session_id"], records[0]),
                            (second["session_id"], records[1]),
                        ),
                    )
                )
            self.assertEqual(
                [len(record.learning_outbox) for record in records], [0, 0]
            )
            recovery = snapshot.learning_record_store.recover()
            self.assertEqual(recovery.event_count, 2)
            learner = next(iter(recovery.records.values()))
            component = next(
                iter(next(iter(learner["knowledge_components"].values())).values())
            )
            self.assertEqual(component["version"], 2)

    def test_anonymous_default_profile_never_persists_cross_session_learning(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            snapshot = self._snapshot(directory)
            contract = snapshot.bootstrap()["interaction_contract"]
            self.assertEqual(
                contract["learning_record_identity_source"],
                "client_supplied_local_profile_ref_hmac_untrusted_non_production",
            )
            self.assertFalse(contract["learning_record_identity_authenticated"])
            self.assertFalse(
                contract["learning_record_cross_user_authorization_established"]
            )
            self.assertNotIn(self.secret.decode("utf-8"), str(snapshot.bootstrap()))
            started = snapshot.start(
                {
                    "goal": self._sealed_lesson_payload(snapshot)["goal"],
                    "syllabus_ref": self._sealed_lesson_payload(snapshot)[
                        "syllabus_ref"
                    ],
                    "student_profile": snapshot.demo_input["student_profile"],
                    "start_idempotency_key": "start.default-profile",
                }
            )
            with patch(
                "teaching_skill_miner.teacher_agent_dashboard.advance_teacher_agent_session",
                side_effect=self._authoritative_advance,
            ):
                snapshot.step(
                    self._step_body(
                        started,
                        key="step.default-profile",
                        answer="default profile answer",
                    )
                )
            self.assertEqual(snapshot.learning_record_store.recover().event_count, 0)

    def test_due_claim_release_uses_session_identity_and_rejects_client_authority(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            snapshot = self._snapshot(directory)
            started = self._start(snapshot, "student-review-api", "start.review")
            with patch(
                "teaching_skill_miner.teacher_agent_dashboard.advance_teacher_agent_session",
                side_effect=self._authoritative_advance,
            ):
                stepped = snapshot.step(
                    self._step_body(
                        started,
                        key="step.review",
                        answer="SECRET_SOURCE_ANSWER review source answer",
                    )
                )
            learner = next(
                iter(snapshot.learning_record_store.recover().records.values())
            )
            component = next(
                iter(next(iter(learner["knowledge_components"].values())).values())
            )
            due_at = datetime.fromisoformat(
                component["schedule"]["due_at_utc"].replace("Z", "+00:00")
            )
            snapshot.learning_record_store._clock = lambda: due_at
            guards = self._guards(stepped)
            due = snapshot.list_due_learning_reviews(guards)
            self.assertEqual(len(due["reviews"]), 1)
            self.assertNotIn("learner_key", str(due))
            review = due["reviews"][0]
            claim = snapshot.claim_due_learning_review(
                {
                    **guards,
                    "review_id": review["review_id"],
                    "expected_version": review["expected_version"],
                    "review_idempotency_key": "claim.review.api",
                }
            )
            self.assertEqual(
                claim["schema"], "teaching_skill_miner.learning_review_claim.v2"
            )
            self.assertNotEqual(claim["session_id"], started["session_id"])
            self.assertEqual(claim["review_session"]["session_id"], claim["session_id"])
            self.assertTrue(
                claim["review_session"]["learning_review"][
                    "server_owned_review_session"
                ]
            )
            self.assertFalse(
                claim["review_session"]["learning_review"]["answer_key_exposed"]
            )
            self.assertNotIn("canonical_claims", str(claim["review_session"]["goal"]))
            self.assertEqual(claim["review_session"]["history"], [])
            self.assertNotIn("SECRET_SOURCE_ANSWER", str(claim["review_session"]))
            self.assertNotIn(
                "understanding_signal", claim["review_session"]["student_state"]
            )
            self.assertNotIn("student_model", claim["review_session"]["student_state"])
            self.assertEqual(
                claim["review_session"]["setup_snapshot"]["student_profile"][
                    "known_misconceptions"
                ],
                [],
            )
            self.assertNotIn("learner_key", str(claim))
            after_claim = snapshot.list_due_learning_reviews(guards)
            self.assertEqual(after_claim["reviews"], [])
            self.assertEqual(len(after_claim["active_reviews"]), 1)
            self.assertEqual(
                after_claim["active_reviews"][0]["active_lease_id"],
                claim["component"]["active_lease_id"],
            )
            self.assertEqual(
                after_claim["active_reviews"][0]["review_session_id"],
                claim["session_id"],
            )
            self.assertNotIn("learner_key", str(after_claim))
            # A normal teaching answer cannot silently impersonate the claimed
            # review outcome.  Even authoritative KC evidence is left out of
            # the scheduler unless the assessment path retains the server
            # review/lease binding.
            with patch(
                "teaching_skill_miner.teacher_agent_dashboard.advance_teacher_agent_session",
                side_effect=self._authoritative_advance,
            ):
                after_unbound_answer = snapshot.step(
                    self._step_body(
                        stepped,
                        key="step.review.unbound",
                        answer="unbound review answer",
                    )
                )
            claimed_target = (
                snapshot.learning_record_store.get_knowledge_component_record(
                    learner_key=learner["learner_key"],
                    curriculum_namespace=claim["component"]["curriculum_namespace"],
                    knowledge_component_id=claim["component"]["knowledge_component_id"],
                )
            )
            self.assertEqual(claimed_target["version"], claim["component"]["version"])
            self.assertEqual(claimed_target["schedule"]["state"], "in_progress")
            self.assertEqual(
                snapshot.store.recover_session(started["session_id"])[
                    "learning_outbox"
                ],
                {},
            )
            release_body = {
                **self._guards(claim["review_session"]),
                "review_id": review["review_id"],
                "lease_id": claim["component"]["active_lease_id"],
                "expected_version": claim["component"]["version"],
                "review_idempotency_key": "release.review.api",
            }
            with self.assertRaisesRegex(
                TeacherAgentDashboardError, "server-owned review session"
            ):
                snapshot.release_learning_review(
                    {
                        **self._guards(after_unbound_answer),
                        "review_id": review["review_id"],
                        "lease_id": claim["component"]["active_lease_id"],
                        "expected_version": claim["component"]["version"],
                        "review_idempotency_key": "release.review.wrong-session",
                    }
                )
            release = snapshot.release_learning_review(release_body)
            self.assertEqual(release["component"]["state"], "scheduled")
            replay = snapshot.release_learning_review(release_body)
            self.assertFalse(replay["applied"])
            with self.assertRaisesRegex(
                TeacherAgentDashboardError, "server-authoritative"
            ):
                snapshot.list_due_learning_reviews({**guards, "outcome": "correct"})

    def test_review_answer_turn_atomically_schedules_correct_partial_and_incorrect(
        self,
    ) -> None:
        expectations = {
            "correct: independent retrieval": ("correct", 2, 3),
            "partial: incomplete retrieval": ("partial", 0, 1),
            "incorrect: contradicted retrieval": ("incorrect", 0, 1),
        }
        for index, (answer, expected) in enumerate(expectations.items(), 1):
            with (
                self.subTest(answer=answer),
                TemporaryDirectory() as directory,
            ):
                snapshot = self._snapshot(directory)
                started = self._start(
                    snapshot,
                    f"student-review-outcome-{index}",
                    f"start.outcome.{index}",
                )
                with patch(
                    "teaching_skill_miner.teacher_agent_dashboard.advance_teacher_agent_session",
                    side_effect=self._authoritative_advance,
                ):
                    source = snapshot.step(
                        self._step_body(
                            started,
                            key=f"step.source.{index}",
                            answer=f"source answer {index}",
                        )
                    )
                    review, claim = self._claim_first_due_review(
                        snapshot,
                        source,
                        idempotency_key=f"claim.outcome.{index}",
                    )
                    review_body = self._step_body(
                        claim["review_session"],
                        key=f"step.review.outcome.{index}",
                        answer=answer,
                    )
                    result = snapshot.step(review_body)

                learner = next(
                    iter(snapshot.learning_record_store.recover().records.values())
                )
                component = next(
                    iter(next(iter(learner["knowledge_components"].values())).values())
                )
                expected_outcome, expected_repetition, expected_interval = expected
                self.assertEqual(component["schedule"]["state"], "scheduled")
                self.assertEqual(
                    component["schedule"]["last_outcome"], expected_outcome
                )
                self.assertEqual(
                    component["schedule"]["repetition"], expected_repetition
                )
                self.assertEqual(
                    component["schedule"]["interval_days"], expected_interval
                )
                self.assertIsNone(component["schedule"]["active_review_id"])
                self.assertIsNone(component["schedule"]["active_lease_id"])
                self.assertEqual(
                    result["learning_review"]["status"], "outcome_committed"
                )
                stored = snapshot.store.recover_session(claim["session_id"])
                self.assertEqual(stored["learning_outbox"], {})
                self.assertEqual(
                    stored["learning_review_binding"]["status"],
                    "outcome_committed",
                )
                self.assertEqual(snapshot.step(review_body), result)
                self.assertEqual(
                    snapshot.learning_record_store.recover().event_count, 3
                )
                self.assertEqual(review["review_id"], claim["review_id"])

    def test_review_outbox_recovers_claimed_lease_after_crash_window(self) -> None:
        with TemporaryDirectory() as directory:
            snapshot = self._snapshot(directory)
            started = self._start(
                snapshot, "student-review-crash", "start.crash.review"
            )
            with patch(
                "teaching_skill_miner.teacher_agent_dashboard.advance_teacher_agent_session",
                side_effect=self._authoritative_advance,
            ):
                source = snapshot.step(
                    self._step_body(
                        started,
                        key="step.crash.source",
                        answer="source before crash",
                    )
                )
                _review, claim = self._claim_first_due_review(
                    snapshot,
                    source,
                    idempotency_key="claim.crash.review",
                )
                with patch.object(
                    type(snapshot),
                    "_drain_learning_outbox_locked",
                    autospec=True,
                    return_value=0,
                ):
                    snapshot.step(
                        self._step_body(
                            claim["review_session"],
                            key="step.crash.review",
                            answer="correct: crash-safe retrieval",
                        )
                    )
            stored = snapshot.store.recover_session(claim["session_id"])
            self.assertEqual(len(stored["learning_outbox"]), 1)
            self.assertEqual(
                next(iter(stored["learning_outbox"].values()))["event_type"],
                "review_outcome_recorded",
            )
            component_before = (
                snapshot.learning_record_store.get_knowledge_component_record(
                    learner_key=next(
                        iter(snapshot.learning_record_store.recover().records)
                    ),
                    curriculum_namespace=claim["component"]["curriculum_namespace"],
                    knowledge_component_id=claim["component"]["knowledge_component_id"],
                )
            )
            self.assertEqual(component_before["schedule"]["state"], "in_progress")

            restarted = self._snapshot(directory)
            component_after = (
                restarted.learning_record_store.get_knowledge_component_record(
                    learner_key=next(
                        iter(restarted.learning_record_store.recover().records)
                    ),
                    curriculum_namespace=claim["component"]["curriculum_namespace"],
                    knowledge_component_id=claim["component"]["knowledge_component_id"],
                )
            )
            self.assertEqual(component_after["schedule"]["state"], "scheduled")
            self.assertEqual(
                restarted.store.recover_session(claim["session_id"])["learning_outbox"],
                {},
            )

    def test_review_claim_replays_after_restart_and_wrong_learner_fails_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            snapshot = self._snapshot(directory)
            started = self._start(snapshot, "student-review-replay", "start.replay")
            other = self._start(snapshot, "student-other", "start.other")
            with patch(
                "teaching_skill_miner.teacher_agent_dashboard.advance_teacher_agent_session",
                side_effect=self._authoritative_advance,
            ):
                source = snapshot.step(
                    self._step_body(
                        started,
                        key="step.replay.source",
                        answer="source replay answer",
                    )
                )
            review, claim = self._claim_first_due_review(
                snapshot,
                source,
                idempotency_key="claim.replay.review",
            )
            claim_body = {
                **self._guards(source),
                "review_id": review["review_id"],
                "expected_version": review["expected_version"],
                "review_idempotency_key": "claim.replay.review",
            }
            replay = snapshot.claim_due_learning_review(claim_body)
            self.assertFalse(replay["applied"])
            self.assertEqual(replay["session_id"], claim["session_id"])
            with self.assertRaisesRegex(
                TeacherAgentDashboardError, "not currently due"
            ):
                snapshot.claim_due_learning_review(
                    {
                        **self._guards(other),
                        "review_id": review["review_id"],
                        "expected_version": review["expected_version"],
                        "review_idempotency_key": "claim.cross-learner",
                    }
                )

            restarted = self._snapshot(directory)
            active_after_restart = restarted.list_due_learning_reviews(
                self._guards(source)
            )
            self.assertEqual(
                active_after_restart["active_reviews"][0]["review_session_id"],
                claim["session_id"],
            )
            replay_after_restart = restarted.claim_due_learning_review(claim_body)
            self.assertFalse(replay_after_restart["applied"])
            self.assertEqual(replay_after_restart["session_id"], claim["session_id"])
            self.assertEqual(
                replay_after_restart["review_session"]["expected_question_id"],
                claim["review_session"]["expected_question_id"],
            )

    def test_review_claim_without_teacher_authority_never_acquires_a_lease(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            snapshot = self._snapshot(directory)
            started = self._start(
                snapshot,
                "student-review-no-authority",
                "start.review.no-authority",
            )
            with patch(
                "teaching_skill_miner.teacher_agent_dashboard.advance_teacher_agent_session",
                side_effect=self._authoritative_advance,
            ):
                source = snapshot.step(
                    self._step_body(
                        started,
                        key="step.review.no-authority.source",
                        answer="source answer before authority revocation",
                    )
                )

            recovery = snapshot.learning_record_store.recover()
            learner_key, learner = next(iter(recovery.records.items()))
            component = next(
                iter(next(iter(learner["knowledge_components"].values())).values())
            )
            due_at = datetime.fromisoformat(
                component["schedule"]["due_at_utc"].replace("Z", "+00:00")
            )
            snapshot.learning_record_store._clock = lambda: due_at
            review = snapshot.list_due_learning_reviews(self._guards(source))[
                "reviews"
            ][0]
            events_before = snapshot.learning_record_store.recover().event_count

            with (
                patch.object(
                    type(snapshot),
                    "_authoritative_teacher_rubric",
                    return_value=None,
                ),
                self.assertRaisesRegex(
                    TeacherAgentDashboardError,
                    "no unique authoritative durable source",
                ),
            ):
                snapshot.claim_due_learning_review(
                    {
                        **self._guards(source),
                        "review_id": review["review_id"],
                        "expected_version": review["expected_version"],
                        "review_idempotency_key": "claim.review.no-authority",
                    }
                )

            unchanged = snapshot.learning_record_store.get_knowledge_component_record(
                learner_key=learner_key,
                curriculum_namespace=review["curriculum_namespace"],
                knowledge_component_id=review["knowledge_component_id"],
            )
            self.assertEqual(
                snapshot.learning_record_store.recover().event_count,
                events_before,
            )
            self.assertEqual(unchanged["version"], review["expected_version"])
            self.assertEqual(unchanged["schedule"]["state"], "scheduled")
            self.assertIsNone(unchanged["schedule"]["active_review_id"])
            self.assertIsNone(unchanged["schedule"]["active_lease_id"])


if __name__ == "__main__":
    unittest.main()
