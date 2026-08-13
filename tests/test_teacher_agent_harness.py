from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import threading
import tempfile
import unittest
from typing import Any, Mapping

from teaching_skill_miner.deepseek_client import DeepSeekClient, DeepSeekConfig
from teaching_skill_miner.harness import (
    CancellationToken,
    HarnessModelRequest,
    HarnessModelResponse,
    ToolCall,
    ToolRegistry,
    run_agent_harness,
)
from teaching_skill_miner.teacher_agent import start_teacher_agent_session
from teaching_skill_miner.teacher_agent_harness import (
    TEACHER_AGENT_HARNESS_PERMISSIONS,
    TEACHER_AGENT_HARNESS_PERMISSION_ROUTE,
    TEACHER_AGENT_HARNESS_PERMISSION_RESOURCES,
    TEACHER_AGENT_HARNESS_PERMISSION_STUDENT,
    TeachingAgentHarnessModelAdapter,
    build_teacher_agent_tool_registry,
    run_teaching_agent_harness,
)
from teaching_skill_miner.teacher_agent_resources import extract_teaching_resource
from teaching_skill_miner.teacher_agent_resource_retrieval import (
    TeachingResourceIndexStore,
)
from teaching_skill_miner.teacher_agent_loop import (
    LOOP_SCHEMA,
    PLAN_SCHEMA,
    TeachingAgentLoopOptions,
    _session_view,
    public_agent_loop_trace,
)


ROOT = Path(__file__).resolve().parents[1]


class _ScriptedModel:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.messages: list[list[dict[str, str]]] = []

    def __call__(self, messages: object) -> object:
        self.calls += 1
        self.messages.append(deepcopy(list(messages)))  # type: ignore[arg-type]
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _UnknownToolThenFinalModel:
    def __init__(self) -> None:
        self.calls = 0

    def plan(
        self,
        _request: HarnessModelRequest,
        *,
        cancellation_token: CancellationToken,
        deadline_monotonic: float,
    ) -> HarnessModelResponse:
        del deadline_monotonic
        cancellation_token.raise_if_cancelled()
        self.calls += 1
        if self.calls == 1:
            return HarnessModelResponse(
                kind="tool_calls",
                tool_calls=(
                    ToolCall(
                        call_id="unknown-1",
                        name="not_registered",
                        arguments={},
                    ),
                ),
            )
        return HarnessModelResponse(kind="final", output={"message": "done"})


def _call(
    name: str,
    arguments: Mapping[str, Any] | None = None,
    *,
    call_id: str | None = None,
) -> dict[str, Any]:
    return {
        "call_id": call_id or f"call_{name}",
        "name": name,
        "arguments": dict(arguments or {}),
    }


def _tools(*calls: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": PLAN_SCHEMA,
        "kind": "tool_calls",
        "tool_calls": [dict(call) for call in calls],
    }


def _route_ready(reason: str = "route complete") -> dict[str, Any]:
    return {"schema": PLAN_SCHEMA, "kind": "route_ready", "reason": reason}


def _selection_calls() -> dict[str, Any]:
    return _tools(
        _call(
            "select_skills",
            {
                "primary_skill_id": "skill_diagnostic_questioning",
                "supporting_skill_ids": ["skill_wait_and_elicit"],
                "reason": "先诊断前置知识",
            },
            call_id="select-1",
        ),
        _call(
            "set_next_focus",
            {"next_focus": "prerequisite"},
            call_id="focus-1",
        ),
    )


def _action() -> dict[str, Any]:
    return {
        "schema": PLAN_SCHEMA,
        "kind": "teaching_action",
        "selected_skill_id": "skill_diagnostic_questioning",
        "supporting_skill_ids": ["skill_wait_and_elicit"],
        "next_focus": "prerequisite",
        "action_type": "probe_prior_knowledge",
        "message": "请用自己的话说出一个必要的前置概念，并举一个最小例子。",
        "expected_signal": "学生能说明前置概念及其作用。",
        "reason": "先确认已有知识。",
    }


class TeacherAgentHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.demo = json.loads((ROOT / "data/teacher_agent_demo_input.json").read_text())
        cls.library = json.loads(
            (ROOT / "data/teacher_agent_skill_library_v2.json").read_text()
        )

    def _session(self) -> dict[str, Any]:
        return start_teacher_agent_session(
            deepcopy(self.demo["goal"]),
            deepcopy(self.demo["student_profile"]),
            deepcopy(self.library),
        )

    def test_registry_contains_versioned_permissioned_tools(self) -> None:
        session = self._session()
        context = _session_view(session, history_limit=6)
        state = {
            "selected_skill_id": None,
            "supporting_skill_ids": [],
            "next_focus": context["student"].get("next_focus"),
        }

        registry = build_teacher_agent_tool_registry(
            context, session["skill_library"], state
        )
        specs = {spec.name: spec for spec in registry.specs()}

        self.assertTrue(
            {
                "inspect_student_state",
                "inspect_recent_history",
                "search_skills",
                "select_skills",
                "set_next_focus",
                "evaluate_termination",
                "retrieve_resources",
            }.issubset(specs),
        )
        self.assertTrue(all(spec.version == "1.0.0" for spec in specs.values()))
        self.assertTrue(
            {spec.permission for spec in specs.values()}.issubset(
                TEACHER_AGENT_HARNESS_PERMISSIONS
            )
        )
        self.assertEqual(
            specs["select_skills"].permission,
            TEACHER_AGENT_HARNESS_PERMISSION_ROUTE,
        )
        self.assertEqual(
            specs["inspect_student_state"].permission,
            TEACHER_AGENT_HARNESS_PERMISSION_STUDENT,
        )
        self.assertEqual(
            specs["retrieve_resources"].permission,
            TEACHER_AGENT_HARNESS_PERMISSION_RESOURCES,
        )
        self.assertEqual(specs["retrieve_resources"].data_scope, "teacher_resource")
        self.assertFalse(specs["retrieve_resources"].requires_user_consent)
        self.assertEqual(
            specs["set_next_focus"].input_schema["properties"]["next_focus"]["enum"],
            ["conceptual", "prerequisite", "procedural", "transfer"],
        )

    def test_retrieve_resources_tool_returns_bounded_provenance_not_student_evidence(
        self,
    ) -> None:
        session = self._session()
        session["teaching_resources"] = [
            extract_teaching_resource(
                (
                    "第一段：动态规划需要明确状态。\n\n"
                    "第二段：状态转移描述当前状态怎样由更小子问题得到。"
                ).encode("utf-8"),
                "text/plain",
                display_name="动态规划讲义.txt",
            )
        ]
        client = _ScriptedModel(
            [
                _tools(
                    _call(
                        "retrieve_resources",
                        {"query": "状态转移 更小子问题", "max_results": 2},
                        call_id="resource-1",
                    )
                ),
                _selection_calls(),
                _route_ready(),
            ]
        )

        result = run_teaching_agent_harness(session, client)

        resource_events = [
            event
            for event in result["events"]
            if event.get("type") == "tool_result"
            and event.get("tool") == "retrieve_resources"
        ]
        self.assertEqual(len(resource_events), 1)
        receipt = resource_events[0]["result"]
        self.assertEqual(
            receipt["schema"], "teaching_skill_miner.resource_retrieval.v1"
        )
        self.assertGreaterEqual(receipt["result_count"], 1)
        self.assertFalse(receipt["student_evidence_used"])
        self.assertTrue(receipt["vector_scores_used"])
        self.assertFalse(receipt["grading_evidence_allowed"])
        self.assertFalse(receipt["safe_to_synthesize"])
        self.assertEqual(receipt["claim_consistency_status"], "not_verified")
        self.assertLessEqual(receipt["returned_char_count"], 3_600)
        provenance = receipt["results"][0]["provenance"]
        self.assertEqual(provenance["display_name"], "动态规划讲义.txt")
        self.assertIn("chunk_content_sha256", provenance)
        self.assertIn("location", provenance)

    def test_harness_accepts_local_long_retrieval_projection_without_expanding_session(
        self,
    ) -> None:
        session = self._session()
        with tempfile.TemporaryDirectory() as directory:
            store = TeachingResourceIndexStore(Path(directory) / "resources")
            resource = extract_teaching_resource(
                ("开头\n" + "中段" * 7_000 + "\n末尾教学锚点").encode("utf-8"),
                "text/plain",
                display_name="长讲义.txt",
                index_store=store,
            )
            session["teaching_resources"] = [resource]
            retrieval_resource = store.get_retrieval_resource(
                resource["content_sha256"]
            )
            self.assertIsNotNone(retrieval_resource)
            client = _ScriptedModel(
                [
                    _tools(
                        _call(
                            "retrieve_resources",
                            {"query": "末尾教学锚点"},
                            call_id="resource-tail",
                        )
                    ),
                    _selection_calls(),
                    _route_ready(),
                ]
            )

            result = run_teaching_agent_harness(
                session,
                client,
                retrieval_resources=[retrieval_resource],
            )

        resource_event = next(
            event
            for event in result["events"]
            if event.get("type") == "tool_result"
            and event.get("tool") == "retrieve_resources"
        )
        self.assertIn("末尾教学锚点", resource_event["result"]["results"][0]["excerpt"])
        self.assertEqual(
            len(session["teaching_resources"][0]["extracted_text"]), 12_000
        )

    def test_adapter_prompt_exposes_only_registry_request_tool_subset(self) -> None:
        session = self._session()
        context = _session_view(session, history_limit=6)
        state = {
            "selected_skill_id": None,
            "supporting_skill_ids": [],
            "next_focus": context["student"].get("next_focus"),
        }
        registry = build_teacher_agent_tool_registry(
            context, session["skill_library"], state
        )
        visible_tools = registry.definitions({TEACHER_AGENT_HARNESS_PERMISSION_STUDENT})
        client = _ScriptedModel(
            [
                {
                    "schema": PLAN_SCHEMA,
                    "kind": "terminate",
                    "outcome": "handoff",
                    "reason": "需要人工接管",
                }
            ]
        )
        adapter = TeachingAgentHarnessModelAdapter(
            client,
            context=context,
            library=session["skill_library"],
            state=state,
        )

        response = adapter.plan(
            HarnessModelRequest(
                run_id="registry-prompt-run",
                turn_id="registry-prompt-turn",
                step=1,
                context=context,
                observations=(),
                tools=visible_tools,
                state={},
            ),
            cancellation_token=CancellationToken(),
            deadline_monotonic=float("inf"),
        )

        self.assertEqual(response.kind, "handoff")
        self.assertEqual(len(visible_tools), 1)
        prompt = "\n".join(item["content"] for item in client.messages[0])
        self.assertIn("inspect_student_state", prompt)
        for hidden_name in {
            "inspect_recent_history",
            "search_skills",
            "select_skills",
            "set_next_focus",
            "evaluate_termination",
            "retrieve_resources",
        }:
            self.assertNotIn(hidden_name, prompt)
        user_payload = json.loads(client.messages[0][1]["content"])
        self.assertEqual(user_payload["available_tools"], list(visible_tools))
        self.assertEqual(
            user_payload["available_tools"][0]["permission"],
            TEACHER_AGENT_HARNESS_PERMISSION_STUDENT,
        )
        self.assertEqual(
            user_payload["available_tools"][0]["input_schema"],
            {"type": "object", "properties": {}, "additionalProperties": False},
        )

    def test_registry_executor_rejects_unknown_tool_even_if_model_requests_it(
        self,
    ) -> None:
        result = run_agent_harness(
            _UnknownToolThenFinalModel(),
            ToolRegistry(),
            {"scope": "unknown-tool-test"},
            allowed_permissions={"tool.read"},
            run_id="unknown-tool-run",
            turn_id="unknown-tool-turn",
        )

        self.assertEqual(result["status"], "completed")
        rejection = next(
            event for event in result["events"] if event["type"] == "tool.rejected"
        )
        self.assertEqual(rejection["payload"]["tool_name"], "not_registered")
        self.assertEqual(rejection["payload"]["error_code"], "unknown_tool")

    def test_route_ready_is_projected_to_legacy_loop_receipt_with_public_trace(
        self,
    ) -> None:
        session = self._session()
        original = deepcopy(session)
        model = _ScriptedModel([_selection_calls(), _route_ready("路由完成")])

        result = run_teaching_agent_harness(
            session,
            model,
            run_id="teacher_harness_route",
            turn_id="turn_route",
        )

        self.assertEqual(session, original)
        self.assertEqual(result["schema"], LOOP_SCHEMA)
        self.assertEqual(result["status"], "route_ready")
        self.assertFalse(result["deterministic_fallback"])
        self.assertEqual(
            result["loop_state"]["selected_skill_id"],
            "skill_diagnostic_questioning",
        )
        self.assertEqual(result["loop_state"]["next_focus"], "prerequisite")
        self.assertEqual(result["harness_trace"]["status"], "completed")
        self.assertEqual(result["harness_trace"]["tool_call_count"], 2)
        self.assertEqual(result["harness_trace"]["run_id"], "teacher_harness_route")
        legacy_public = public_agent_loop_trace(result)
        self.assertEqual(legacy_public["status"], "route_ready")
        self.assertEqual(legacy_public["model_call_count"], 2)
        self.assertEqual(legacy_public["tool_call_count"], 2)
        self.assertTrue(legacy_public["bounded_route_completion"])
        self.assertNotIn(
            "先诊断前置知识",
            json.dumps(result["harness_trace"], ensure_ascii=False),
        )

    def test_valid_teaching_action_keeps_legacy_skill_projection(self) -> None:
        result = run_teaching_agent_harness(
            self._session(), _ScriptedModel([_selection_calls(), _action()])
        )

        self.assertEqual(result["status"], "action_ready")
        self.assertFalse(result["deterministic_fallback"])
        self.assertEqual(
            result["action"]["skill"]["skill_id"],
            "skill_diagnostic_questioning",
        )
        self.assertEqual(result["action"]["action_type"], "probe_prior_knowledge")
        self.assertTrue(
            any(event["type"] == "teaching_action" for event in result["events"])
        )

    def test_model_failure_uses_deterministic_domain_fallback(self) -> None:
        result = run_teaching_agent_harness(
            self._session(),
            _ScriptedModel([TimeoutError("provider unavailable")]),
            options=TeachingAgentLoopOptions(model_retries=0),
        )

        self.assertEqual(result["status"], "action_ready")
        self.assertTrue(result["deterministic_fallback"])
        self.assertIsInstance(result["action"], dict)
        self.assertEqual(result["harness_trace"]["status"], "failed")
        self.assertTrue(
            any(event["type"] == "model_error" for event in result["events"])
        )
        self.assertTrue(any(event["type"] == "fallback" for event in result["events"]))

    def test_repeated_tool_guard_fails_closed_without_reexecuting_tool(self) -> None:
        model = _ScriptedModel(
            [
                _tools(_call("inspect_student_state", call_id="inspect-1")),
                _tools(_call("inspect_student_state", call_id="inspect-2")),
            ]
        )

        result = run_teaching_agent_harness(
            self._session(),
            model,
            options=TeachingAgentLoopOptions(max_repeated_tool_calls=1),
        )

        self.assertTrue(result["deterministic_fallback"])
        self.assertEqual(result["harness_trace"]["tool_call_count"], 1)
        guards = [
            event
            for event in result["harness_trace"]["events"]
            if event["type"] == "guard.triggered"
        ]
        self.assertEqual(guards[0]["payload"]["reason_code"], "repeated_tool_call")

    def test_invalid_arguments_and_permission_denial_do_not_commit_route(self) -> None:
        invalid_model = _ScriptedModel(
            [
                _tools(_call("select_skills", {}, call_id="invalid-select")),
                _route_ready(),
            ]
        )
        invalid = run_teaching_agent_harness(self._session(), invalid_model)

        self.assertIsNone(invalid["loop_state"]["selected_skill_id"])
        self.assertTrue(invalid["deterministic_fallback"])
        invalid_rejection = next(
            event
            for event in invalid["harness_trace"]["events"]
            if event["type"] == "tool.rejected"
        )
        self.assertEqual(
            invalid_rejection["payload"]["error_code"], "invalid_arguments"
        )
        second_payload = json.loads(invalid_model.messages[1][1]["content"])
        self.assertEqual(
            second_payload["previous_tool_results"][0]["tool"], "select_skills"
        )
        self.assertFalse(second_payload["previous_tool_results"][0]["ok"])

        denied_model = _ScriptedModel([_selection_calls(), _route_ready()])
        denied = run_teaching_agent_harness(
            self._session(),
            denied_model,
            allowed_permissions={TEACHER_AGENT_HARNESS_PERMISSION_STUDENT},
        )
        self.assertIsNone(denied["loop_state"]["selected_skill_id"])
        self.assertTrue(denied["deterministic_fallback"])
        rejected = next(
            event
            for event in denied["harness_trace"]["events"]
            if event["type"] == "tool.rejected"
        )
        self.assertEqual(rejected["payload"]["error_code"], "permission_denied")

    def test_deepseek_client_adapter_receives_tool_observations(self) -> None:
        responses = [
            _selection_calls(),
            _route_ready("DeepSeek route complete"),
        ]
        request_bodies: list[dict[str, Any]] = []

        def stream_transport(
            _url: str,
            _headers: Mapping[str, str],
            payload: bytes,
            _timeout: float,
            _cancellation_token: object,
        ) -> tuple[int, list[bytes]]:
            request_bodies.append(json.loads(payload))
            plan = responses.pop(0)
            envelope = {
                "id": f"deepseek-{len(request_bodies)}",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "delta": {"content": json.dumps(plan, ensure_ascii=False)},
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            }
            return 200, [
                b"data: " + json.dumps(envelope, ensure_ascii=False).encode() + b"\n",
                b"data: [DONE]\n",
            ]

        client = DeepSeekClient(
            DeepSeekConfig(
                allow_remote_student_data=True,
                max_retries=0,
                temperature=0.0,
            ),
            api_key="unit-test-secret",
            stream_transport=stream_transport,
        )

        result = run_teaching_agent_harness(self._session(), client)

        self.assertEqual(result["status"], "route_ready")
        self.assertEqual(len(request_bodies), 2)
        second_user = json.loads(request_bodies[1]["messages"][1]["content"])
        self.assertEqual(
            {item["call_id"] for item in second_user["previous_tool_results"]},
            {"select-1", "focus-1"},
        )
        completed = [
            event
            for event in result["harness_trace"]["events"]
            if event["type"] == "model.completed"
        ]
        self.assertEqual(
            [event["payload"]["provider_request_id"] for event in completed],
            ["deepseek-1", "deepseek-2"],
        )
        self.assertNotIn("unit-test-secret", json.dumps(result, ensure_ascii=False))

    def test_deepseek_structured_stream_cancellation_closes_planner_boundary(
        self,
    ) -> None:
        entered = threading.Event()
        transport_observed_cancel = threading.Event()

        def stream_transport(
            _url: str,
            _headers: Mapping[str, str],
            _payload: bytes,
            _timeout: float,
            cancellation_token: CancellationToken,
        ) -> tuple[int, object]:
            def lines():
                entered.set()
                while not cancellation_token.cancelled:
                    cancellation_token.wait(0.01)
                transport_observed_cancel.set()
                cancellation_token.raise_if_cancelled()
                yield b"data: [DONE]\n"

            return 200, lines()

        client = DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=True, max_retries=0),
            api_key="unit-test-secret",
            stream_transport=stream_transport,
        )
        token = CancellationToken()
        results: list[dict[str, Any]] = []

        worker = threading.Thread(
            target=lambda: results.append(
                run_teaching_agent_harness(
                    self._session(), client, cancellation_token=token
                )
            ),
            daemon=True,
        )
        worker.start()
        self.assertTrue(entered.wait(timeout=2))
        self.assertTrue(token.cancel("test_stop"))
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertTrue(transport_observed_cancel.is_set())
        self.assertEqual(results[0]["harness_trace"]["status"], "cancelled")
        self.assertEqual(
            results[0]["harness_trace"]["events"][-1]["type"],
            "run.cancelled",
        )


if __name__ == "__main__":
    unittest.main()
