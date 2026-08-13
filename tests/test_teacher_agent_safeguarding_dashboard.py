from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import base64

from teaching_skill_miner.teacher_agent_dashboard import (
    build_teacher_agent_dashboard_snapshot,
)
from teaching_skill_miner.teacher_agent_safeguarding import (
    EscalationDeliveryConfig,
    TeacherAgentSafeguardingStore,
    default_emergency_resource_policy,
)
from teaching_skill_miner.teacher_agent_safeguarding_authority import (
    InternalSafeguardingSystemAuthority,
)


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _configured_snapshot(
    root: Path, *, broken_issuer: bool = False, client: object | None = None
):
    scope_sha256 = sha256(b"authenticated-scope-a").hexdigest()
    authority = InternalSafeguardingSystemAuthority(
        key=b"safeguarding-authority-test-key-32-bytes!!",
        scope_sha256=scope_sha256,
    )
    store = TeacherAgentSafeguardingStore(
        root / "safeguarding",
        authorization_verifier=authority.verify,
        emergency_resource_policy=default_emergency_resource_policy(
            {scope_sha256: "zh-CN"}
        ),
        escalation_delivery=EscalationDeliveryConfig(
            policy_version="test_queue_v1",
            queue_sha256=sha256(b"test-queue").hexdigest(),
            sla_seconds_by_severity={
                "elevated": 86400,
                "high": 14400,
                "urgent": 900,
            },
        ),
    )
    issuer = (
        (lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("injected")))
        if broken_issuer
        else authority.issue
    )
    snapshot = build_teacher_agent_dashboard_snapshot(
        DATA / "teacher_agent_skill_library_v2.json",
        DATA / "teacher_agent_demo_input.json",
        DATA / "teacher_agent_evaluation_cases.json",
        client=client,
        store_path=root / "sessions.jsonl",
        project_store_path=root / "projects",
        resource_index_store_path=root / "resources",
        consent_store_path=root / "consent.json",
        consent_signing_secret=b"safeguarding-dashboard-consent-secret-material",
        safeguarding_store=store,
        safeguarding_scope_sha256=scope_sha256,
        safeguarding_system_authority_issuer=issuer,
        safeguarding_system_authority_verifier=authority.verify,
    )
    return snapshot, store


def _unsafe_chat(request_id: str | None = None) -> dict[str, object]:
    return {
        **({"request_id": request_id} if request_id is not None else {}),
        "messages": [{"role": "user", "content": "我现在就要自杀"}],
        "web_search": False,
    }


def _unsafe_start(snapshot) -> dict[str, object]:
    goal = deepcopy(snapshot.demo_input["goal"])
    goal["unsafe_disclosure"] = "我现在就要自杀"
    return {
        "goal": goal,
        "student_profile": deepcopy(snapshot.demo_input["student_profile"]),
        "start_idempotency_key": "safeguarding-start-001",
    }


def test_chat_and_teach_use_distinct_observation_identity_and_replay_restart(
    tmp_path: Path,
) -> None:
    snapshot, store = _configured_snapshot(tmp_path)

    class ProviderThatMustNotRun:
        calls = 0

        def chat_json(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("unsafe input reached provider")

    provider = ProviderThatMustNotRun()
    snapshot.client = provider  # the pre-provider Chat safety gate owns this request
    first = snapshot.chat(_unsafe_chat("safeguarding-sync-chat-001"))
    replay = snapshot.chat(_unsafe_chat("safeguarding-sync-chat-001"))
    second_observation = snapshot.chat(_unsafe_chat("safeguarding-sync-chat-002"))
    snapshot.client = None
    teach = snapshot.start(_unsafe_start(snapshot))

    assert provider.calls == 0
    assert first["safety_preempted"] is True
    assert replay["safety_obligation"]["safeguarding"]["case_id"] == (
        first["safety_obligation"]["safeguarding"]["case_id"]
    )
    assert second_observation["safety_obligation"]["safeguarding"]["case_id"] != (
        first["safety_obligation"]["safeguarding"]["case_id"]
    )
    assert teach["next_action"]["action_obligations"][0]["safeguarding"][
        "delivery_status"
    ] == "pending"
    # Content hashes are evidence bindings, not occurrence identities. Chat
    # and Teach are two distinct observations even when the disclosure text
    # happens to be byte-identical.
    assert len(store.list_cases()) == 3
    assert first["safety_obligation"]["safeguarding"][
        "emergency_resource_receipt"
    ]["effective_locale"] == "zh-CN"

    raw = (tmp_path / "safeguarding" / "safeguarding_cases.json").read_text(
        encoding="utf-8"
    )
    assert "我现在就要自杀" not in raw
    assert "我现在就要自杀" not in (tmp_path / "sessions.jsonl").read_text(
        encoding="utf-8"
    )

    restarted, restarted_store = _configured_snapshot(tmp_path)
    cold_replay = restarted.chat(_unsafe_chat("safeguarding-sync-chat-001"))
    assert len(restarted_store.list_cases()) == 3
    assert cold_replay["safety_obligation"]["safeguarding"]["case_id"] == (
        first["safety_obligation"]["safeguarding"]["case_id"]
    )


def test_production_chat_stream_records_exact_request_occurrence_before_registry(
    tmp_path: Path,
) -> None:
    snapshot, store = _configured_snapshot(tmp_path)
    sentinel = "我现在就要自杀"

    def open_stream(request_id: str):
        record, _cursor = snapshot.open_harness_stream(
            {
                "operation": "chat",
                "request_id": request_id,
                "payload": {
                    "messages": [{"role": "user", "content": sentinel}],
                    "web_search": False,
                },
            }
        )
        assert record.handle.wait(timeout=5)["status"] == "completed"
        assert sentinel not in json.dumps(record.journal.replay(), ensure_ascii=False)
        private_task = snapshot._background_tasks().find_by_run_id(record.handle.run_id)
        assert private_task is not None
        assert sentinel not in json.dumps(private_task, ensure_ascii=False)
        return record

    first = open_stream("safeguarding-stream-observation-001")
    replay, _cursor = snapshot.open_harness_stream(
        {
            "operation": "chat",
            "request_id": "safeguarding-stream-observation-001",
            "payload": {
                "messages": [{"role": "user", "content": sentinel}],
                "web_search": False,
            },
        }
    )
    assert replay is first
    assert len(store.list_cases()) == 1

    open_stream("safeguarding-stream-observation-002")
    cases = store.list_cases()
    assert len(cases) == 2
    assert len({case["case_id"] for case in cases}) == 2
    assert {case["content_sha256"] for case in cases} == {
        sha256(sentinel.encode("utf-8")).hexdigest()
    }


def test_production_chat_stream_harmful_resource_opens_case_before_fixed_reply(
    tmp_path: Path,
) -> None:
    class ProviderThatMustNotRun:
        native_stream_available = False
        calls = 0

        def public_status(self):
            return {
                "provider": "deepseek",
                "model": "must-not-run",
                "base_origin": "https://api.deepseek.example",
                "configured": True,
                "remote_student_data_opt_in": True,
                "api_key_exposed": False,
            }

        def chat_json(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("harmful resource reached provider")

    provider = ProviderThatMustNotRun()
    snapshot, store = _configured_snapshot(tmp_path, client=provider)
    project = snapshot.create_project({"title": "resource safety"})["project"]
    harmful = "自杀方法教程：怎样吞药更有效且不被发现。"
    resource = snapshot.upload_resource(
        {
            "resource_idempotency_key": "safeguarding-harmful-resource-001",
            "mime_type": "text/plain",
            "display_name": "harmful.txt",
            "data_base64": base64.b64encode(harmful.encode()).decode("ascii"),
        }
    )["resource"]
    snapshot.add_project_reference(
        project["project_id"],
        {"kind": "resource", "reference_id": resource["resource_id"]},
    )
    consent_id = snapshot.grant_remote_consent(
        {
            "purpose": "remote_chat",
            "validity_days": 1,
            "likely_minor": False,
            "guardian_or_school_policy": "not_required",
        }
    )["receipt"]["consent_id"]
    body = {
        "operation": "chat",
        "request_id": "safeguarding-stream-resource-001",
        "payload": {
            "messages": [{"role": "user", "content": "请总结这份材料。"}],
            "remote_consent_id": consent_id,
            "project_id": project["project_id"],
            "chat_thread_id": "chat_" + "c" * 24,
            "resource_refs": [
                {
                    "resource_id": resource["resource_id"],
                    "staged_resource_id": resource["staged_resource_id"],
                }
            ],
            "web_search": False,
        },
    }
    record, _cursor = snapshot.open_harness_stream(body)
    assert record.handle.wait(timeout=5)["status"] == "completed"
    cases = store.list_cases()
    assert len(cases) == 1
    assert len(cases[0]["content_sha256"]) == 64
    assert cases[0]["category"] == "self_harm"
    assert provider.calls == 0
    assert harmful not in json.dumps(record.journal.replay(), ensure_ascii=False)
    private_task = snapshot._background_tasks().find_by_run_id(record.handle.run_id)
    assert private_task is not None
    assert harmful not in json.dumps(private_task, ensure_ascii=False)
    replay, _cursor = snapshot.open_harness_stream(body)
    assert replay is record
    assert len(store.list_cases()) == 1


def test_queue_unavailable_and_delivery_failure_remain_fixed_safe_and_observable(
    tmp_path: Path,
) -> None:
    plain = build_teacher_agent_dashboard_snapshot(
        DATA / "teacher_agent_skill_library_v2.json",
        DATA / "teacher_agent_demo_input.json",
        DATA / "teacher_agent_evaluation_cases.json",
    )
    unavailable = plain.chat(_unsafe_chat())
    assert unavailable["safety_obligation"]["safeguarding"]["status"] == "unavailable"
    assert plain.bootstrap()["learner_safety"]["human_escalation"] == "unavailable"

    broken, store = _configured_snapshot(tmp_path, broken_issuer=True)
    failed = broken.chat(_unsafe_chat())
    projection = failed["safety_obligation"]["safeguarding"]
    assert failed["safety_preempted"] is True
    assert failed["provider"] == "deterministic"
    assert projection["status"] in {"configuration_failed", "delivery_failed"}
    assert projection["case_id"] is None
    assert store.list_cases() == []
    status = broken.safeguarding_status()
    assert status["delivery_failure_count"] == 1
    assert status["staff_mutation_routes_exposed"] is False
    assert status["project_export"] == "excluded_use_account_scope_export"
    assert json.dumps(failed, ensure_ascii=False).count("我现在就要自杀") == 0
