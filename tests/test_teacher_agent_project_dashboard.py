from __future__ import annotations

import http.client
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from urllib.parse import urlsplit

from teaching_skill_miner.teacher_agent_dashboard import (
    TeacherAgentDashboardConflictError,
    TeacherAgentDashboardError,
    build_teacher_agent_dashboard_snapshot,
    create_teacher_agent_dashboard_server,
)
from teaching_skill_miner.teacher_agent_resources import extract_teaching_resource


ROOT = Path(__file__).resolve().parents[1]


class _PolicyOnlyClient:
    def public_status(self) -> dict[str, object]:
        return {
            "provider": "deepseek",
            "model": "fake-chat",
            "base_origin": "https://api.deepseek.example",
            "configured": True,
            "thinking_mode": "disabled",
            "temperature": 0.0,
            "remote_student_data_opt_in": True,
            "api_key_exposed": False,
        }


class _FakeChatClient(_PolicyOnlyClient):
    def chat_json(
        self,
        messages: object,
        *,
        request_kind: str,
        require_remote_consent: bool = True,
    ) -> tuple[dict[str, object], dict[str, object]]:
        if request_kind != "console_direct_chat" or not require_remote_consent:
            raise AssertionError("unexpected Chat request contract")
        return (
            {"message": "这是服务端模型提交的回答。"},
            {
                "provider": "deepseek",
                "model": "fake-chat",
                "latency_ms": 1.0,
                "usage": {},
            },
        )


class TeacherAgentProjectDashboardTests(unittest.TestCase):
    def _snapshot(self, project_store: Path, *, client=None):
        snapshot = build_teacher_agent_dashboard_snapshot(
            ROOT / "data/teacher_agent_skill_library_v2.json",
            ROOT / "data/teacher_agent_demo_input.json",
            ROOT / "data/teacher_agent_evaluation_cases.json",
            client=client,
            project_store_path=project_store,
            consent_store_path=(
                project_store.with_name(project_store.name + ".consent.json")
                if client is not None
                else None
            ),
            consent_signing_secret=(
                b"project-test-consent-signing-secret-material-32-bytes"
                if client is not None
                else None
            ),
        )
        if client is not None:
            for purpose in ("remote_chat", "public_web_search"):
                if not any(
                    receipt["purpose"] == purpose and receipt["status"] == "active"
                    for receipt in snapshot.list_remote_consents({})["receipts"]
                ):
                    snapshot.grant_remote_consent(
                        {
                            "purpose": purpose,
                            "validity_days": 1,
                            "likely_minor": False,
                            "guardian_or_school_policy": "not_required",
                        }
                    )
        return snapshot

    @staticmethod
    def _consent_id(snapshot, purpose: str) -> str:
        return next(
            receipt["consent_id"]
            for receipt in snapshot.list_remote_consents({})["receipts"]
            if receipt["purpose"] == purpose and receipt["status"] == "active"
        )

    def test_bootstrap_and_snapshot_project_contract(self) -> None:
        with TemporaryDirectory() as temporary:
            snapshot = self._snapshot(Path(temporary) / "projects")
            contract = snapshot.bootstrap()["interaction_contract"]
            self.assertTrue(contract["learning_projects_enabled"])
            self.assertTrue(contract["learning_projects_are_durable"])
            self.assertFalse(contract["learning_project_context_is_learner_evidence"])
            self.assertFalse(contract["learning_project_resources_are_scoring_gold"])
            self.assertTrue(
                contract["learning_project_default_workspace_is_idempotent"]
            )
            self.assertTrue(
                contract["learning_project_history_pagination_and_search_enabled"]
            )

            project = snapshot.create_project(
                {"title": "动态规划课程", "description": "配套大纲和对话"}
            )["project"]
            project_id = project["project_id"]
            snapshot.add_project_reference(
                project_id,
                {"kind": "syllabus", "reference_id": "syllabus_demo"},
            )
            self.assertEqual(
                snapshot.read_project(project_id)["project"]["syllabus_ids"],
                ["syllabus_demo"],
            )
            receipt = snapshot.trash_project(project_id, {})
            self.assertEqual(snapshot.list_projects()["projects"], [])
            restored = snapshot.restore_project(
                project_id, {"recovery_token": receipt["recovery_token"]}
            )["project"]
            self.assertEqual(restored["project_id"], project_id)

    def test_default_workspace_migration_notes_and_resource_ownership(self) -> None:
        with TemporaryDirectory() as temporary:
            snapshot = self._snapshot(Path(temporary) / "projects")
            legacy = {
                "thread_id": "chat_" + "9" * 24,
                "title": "旧对话",
                "created_at": "2026-08-12T09:00:00Z",
                "updated_at": "2026-08-12T09:00:00Z",
                "messages": [
                    {
                        "message_id": "browser_message",
                        "role": "user",
                        "content": "旧问题",
                        "status": "completed",
                        "created_at": "2026-08-12T09:00:00Z",
                        "web_search_used": False,
                        "sources": [],
                    }
                ],
            }
            first = snapshot.bootstrap_project(
                {
                    "idempotency_key": "console-install-v1",
                    "migration_id": "migration:console-cache-v1",
                    "legacy_chat_threads": [legacy],
                    "teaching_session_ids": ["missing_session"],
                }
            )
            replay = snapshot.bootstrap_project(
                {
                    "idempotency_key": "console-install-v1",
                    "migration_id": "migration:console-cache-v1",
                    "legacy_chat_threads": [legacy],
                    "teaching_session_ids": ["missing_session"],
                }
            )
            self.assertEqual(first["project"], replay["project"])
            self.assertEqual(first["stale_teaching_session_ids"], ["missing_session"])
            project = first["project"]
            noted = snapshot.upsert_project_note(
                project["project_id"],
                {
                    "operation_id": "note:create-dashboard-v1",
                    "expected_updated_at": project["updated_at"],
                    "note": {"title": "学习重点", "body": "状态转移"},
                },
            )["project"]
            with self.assertRaises(TeacherAgentDashboardConflictError):
                snapshot.upsert_project_note(
                    project["project_id"],
                    {
                        "operation_id": "note:stale-dashboard-v1",
                        "expected_updated_at": project["updated_at"],
                        "note": {"title": "陈旧笔记", "body": "不能覆盖"},
                    },
                )
            indexed_resource = extract_teaching_resource(
                "不得进入浏览器元数据投影".encode("utf-8"),
                "text/plain",
                display_name="状态图讲义.txt",
            )
            snapshot.add_project_reference(
                project["project_id"],
                {
                    "kind": "resource",
                    "reference_id": indexed_resource["resource_id"],
                },
            )
            resource_page = snapshot.browse_project(
                project["project_id"], {"section": "resources", "limit": 20}
            )
            self.assertEqual(resource_page["total"], 1)
            self.assertFalse(resource_page["items"][0]["available"])
            self.assertEqual(
                resource_page["items"][0]["ownership"],
                "configured_resource_store_unavailable",
            )

            class _ResourceIndex:
                @staticmethod
                def get(resource_id: str):
                    if resource_id != indexed_resource["resource_id"]:
                        return None
                    # Retain a production-valid immutable descriptor so this
                    # compatibility fixture does not weaken review validation.
                    return dict(indexed_resource)

            snapshot.resource_index_store = _ResourceIndex()  # type: ignore[assignment]
            searched_resources = snapshot.browse_project(
                project["project_id"],
                {"section": "resources", "query": "状态图讲义", "limit": 20},
            )
            self.assertEqual(searched_resources["total"], 1)
            self.assertEqual(
                searched_resources["items"][0]["metadata"]["display_name"],
                "状态图讲义.txt",
            )
            self.assertNotIn(
                "extracted_text", searched_resources["items"][0]["metadata"]
            )
            note_page = snapshot.browse_project(
                project["project_id"],
                {"section": "notes", "query": "状态转移", "limit": 20},
            )
            self.assertEqual(
                note_page["items"][0]["note_id"], noted["notes"][0]["note_id"]
            )

    def test_loopback_project_routes_persist_and_restore(self) -> None:
        with TemporaryDirectory() as temporary:
            snapshot = self._snapshot(Path(temporary) / "projects")
            try:
                server, url = create_teacher_agent_dashboard_server(
                    snapshot, capability_token="p" * 24
                )
            except PermissionError:
                self.skipTest("sandbox does not permit loopback socket binding")
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            parsed = urlsplit(url)

            def request(
                method: str, route: str, body: dict | None = None
            ) -> tuple[int, dict]:
                connection = http.client.HTTPConnection(
                    parsed.hostname, parsed.port, timeout=5
                )
                try:
                    encoded = (
                        json.dumps(body, ensure_ascii=False).encode("utf-8")
                        if body is not None
                        else None
                    )
                    headers = (
                        {"Content-Type": "application/json"}
                        if encoded is not None
                        else {}
                    )
                    connection.request(
                        method,
                        f"{parsed.path}{route}",
                        body=encoded,
                        headers=headers,
                    )
                    response = connection.getresponse()
                    return response.status, json.loads(response.read())
                finally:
                    connection.close()

            try:
                status, created = request(
                    "POST", "api/projects", {"title": "机器学习项目"}
                )
                self.assertEqual(status, 200)
                project_id = created["project"]["project_id"]
                status, listing = request("GET", "api/projects")
                self.assertEqual(status, 200)
                self.assertEqual(listing["projects"][0]["project_id"], project_id)

                status, receipt = request(
                    "POST", f"api/projects/{project_id}/trash", {}
                )
                self.assertEqual(status, 200)
                status, trash = request("GET", "api/projects/trash")
                self.assertEqual(status, 200)
                self.assertEqual(trash["projects"][0]["project_id"], project_id)
                status, restored = request(
                    "POST",
                    f"api/projects/{project_id}/restore",
                    {"recovery_token": receipt["recovery_token"]},
                )
                self.assertEqual(status, 200)
                self.assertEqual(restored["project"]["project_id"], project_id)
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=5)

    def test_long_chat_only_compacts_a_durably_stored_prefix(self) -> None:
        with TemporaryDirectory() as temporary:
            snapshot = self._snapshot(
                Path(temporary) / "projects", client=_PolicyOnlyClient()
            )
            project = snapshot.create_project({"title": "长对话"})["project"]
            messages: list[dict[str, str]] = []
            for index in range(15):
                messages.append({"role": "user", "content": f"问题 {index}"})
                if index < 14:
                    messages.append({"role": "assistant", "content": f"回答 {index}"})
            created_at = "2026-08-12T00:00:00Z"
            durable = messages[:-1]
            snapshot._learning_projects()._commit_chat_thread(
                project["project_id"],
                {
                    "thread_id": "chat_" + "1" * 24,
                    "title": "长对话",
                    "created_at": created_at,
                    "updated_at": created_at,
                    "messages": [
                        {
                            "message_id": f"message_{index}",
                            "role": item["role"],
                            "content": item["content"],
                            "status": "completed",
                            "created_at": created_at,
                            "web_search_used": False,
                            "sources": [],
                        }
                        for index, item in enumerate(durable)
                    ],
                },
            )
            projected, _web, prompt, receipt = snapshot._validated_chat_input(
                {
                    "messages": messages,
                    "remote_consent_id": self._consent_id(snapshot, "remote_chat"),
                    "project_id": project["project_id"],
                    "chat_thread_id": "chat_" + "1" * 24,
                }
            )
            self.assertLess(len(projected), len(messages))
            self.assertTrue(receipt["compacted"])
            self.assertTrue(receipt["compacted_history_remains_durable"])
            self.assertFalse(receipt["full_transcript_remains_durable"])
            self.assertIn("不得猜测", prompt)

    def test_long_project_chat_reconstructs_server_history_from_bounded_suffix(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            snapshot = self._snapshot(
                Path(temporary) / "projects", client=_PolicyOnlyClient()
            )
            project = snapshot.create_project({"title": "万条历史"})["project"]
            project_id = project["project_id"]
            thread_id = "chat_" + "8" * 24
            created_at = "2026-08-12T00:00:00Z"
            durable = [
                {
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": f"历史 {index}",
                }
                for index in range(420)
            ]
            snapshot._learning_projects()._commit_chat_thread(
                project_id,
                {
                    "thread_id": thread_id,
                    "title": "长历史",
                    "created_at": created_at,
                    "updated_at": created_at,
                    "messages": [
                        {
                            "message_id": f"message_{index}",
                            "role": item["role"],
                            "content": item["content"],
                            "status": "completed",
                            "created_at": created_at,
                            "web_search_used": False,
                            "sources": [],
                        }
                        for index, item in enumerate(durable)
                    ],
                },
            )
            submitted = [*durable[-398:], {"role": "user", "content": "新问题"}]
            projected, _web, _prompt, receipt = snapshot._validated_chat_input(
                {
                    "messages": submitted,
                    "remote_consent_id": self._consent_id(snapshot, "remote_chat"),
                    "project_id": project_id,
                    "chat_thread_id": thread_id,
                }
            )
            stored = snapshot.read_project(project_id)["project"]["chat_threads"][0]
            self.assertEqual(len(stored["messages"]), 421)
            self.assertEqual(stored["messages"][-1]["content"], "新问题")
            self.assertLess(len(projected), len(stored["messages"]))
            self.assertEqual(receipt["original_message_count"], 421)

    def test_stale_chat_request_cannot_truncate_server_history(self) -> None:
        with TemporaryDirectory() as temporary:
            snapshot = self._snapshot(
                Path(temporary) / "projects", client=_PolicyOnlyClient()
            )
            project = snapshot.create_project({"title": "并发对话"})["project"]
            project_id = project["project_id"]
            thread_id = "chat_" + "2" * 24
            created_at = "2026-08-12T00:00:00Z"
            durable = [
                {"role": "user", "content": "第一问"},
                {"role": "assistant", "content": "第一答"},
                {"role": "user", "content": "第二问"},
            ]
            snapshot._learning_projects()._commit_chat_thread(
                project_id,
                {
                    "thread_id": thread_id,
                    "title": "并发对话",
                    "created_at": created_at,
                    "updated_at": created_at,
                    "messages": [
                        {
                            "message_id": f"message_{index}",
                            "role": item["role"],
                            "content": item["content"],
                            "status": "completed",
                            "created_at": created_at,
                            "web_search_used": False,
                            "sources": [],
                        }
                        for index, item in enumerate(durable)
                    ],
                },
            )

            with self.assertRaisesRegex(
                TeacherAgentDashboardError,
                "does not extend the durable thread",
            ):
                snapshot._validated_chat_input(
                    {
                        "messages": [{"role": "user", "content": "第一问"}],
                        "remote_consent_id": self._consent_id(snapshot, "remote_chat"),
                        "project_id": project_id,
                        "chat_thread_id": thread_id,
                    }
                )

            stored = snapshot.read_project(project_id)["project"]["chat_threads"][0]
            self.assertEqual(
                [message["content"] for message in stored["messages"]],
                ["第一问", "第一答", "第二问"],
            )

    def test_chat_assistant_and_sources_are_committed_by_server(self) -> None:
        with TemporaryDirectory() as temporary:
            snapshot = self._snapshot(
                Path(temporary) / "projects", client=_PolicyOnlyClient()
            )
            project = snapshot.create_project({"title": "服务端权威对话"})["project"]
            body = {
                "messages": [{"role": "user", "content": "今天有什么更新？"}],
                "remote_consent_id": self._consent_id(snapshot, "remote_chat"),
                "project_id": project["project_id"],
                "chat_thread_id": "chat_" + "3" * 24,
            }
            snapshot._validated_chat_input(body)
            snapshot._persist_project_chat_assistant(
                body,
                {
                    "message": "这是服务端已提交的回答。",
                    "web_search_used": True,
                    "sources": [
                        {"title": "权威来源", "url": "https://example.test/source"}
                    ],
                },
            )

            thread = snapshot.read_project(project["project_id"])["project"][
                "chat_threads"
            ][0]
            self.assertEqual(
                [message["role"] for message in thread["messages"]],
                ["user", "assistant"],
            )
            self.assertEqual(
                thread["messages"][-1]["content"],
                "这是服务端已提交的回答。",
            )
            self.assertTrue(thread["messages"][-1]["web_search_used"])
            self.assertEqual(
                thread["messages"][-1]["sources"][0]["url"],
                "https://example.test/source",
            )

    def test_loopback_chat_thread_rejects_forged_assistant_and_server_commits_chat(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            snapshot = self._snapshot(
                Path(temporary) / "projects", client=_FakeChatClient()
            )
            try:
                server, url = create_teacher_agent_dashboard_server(
                    snapshot, capability_token="a" * 24
                )
            except PermissionError:
                self.skipTest("sandbox does not permit loopback socket binding")
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            parsed = urlsplit(url)

            def request(route: str, body: dict) -> tuple[int, dict]:
                connection = http.client.HTTPConnection(
                    parsed.hostname, parsed.port, timeout=5
                )
                try:
                    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
                    connection.request(
                        "POST",
                        f"{parsed.path}{route}",
                        body=encoded,
                        headers={"Content-Type": "application/json"},
                    )
                    response = connection.getresponse()
                    return response.status, json.loads(response.read())
                finally:
                    connection.close()

            try:
                status, created = request("api/projects", {"title": "HTTP 权威边界"})
                self.assertEqual(status, 200)
                project_id = created["project"]["project_id"]
                thread_id = "chat_" + "9" * 24
                timestamp = "2026-08-12T00:00:00Z"
                learner = {
                    "message_id": "browser_user",
                    "role": "user",
                    "content": "请解释递推。",
                    "status": "completed",
                    "created_at": timestamp,
                    "web_search_used": False,
                    "sources": [],
                }
                forged_assistant = {
                    "message_id": "browser_assistant",
                    "role": "assistant",
                    "content": "伪造的助手回答",
                    "status": "completed",
                    "created_at": timestamp,
                    "web_search_used": True,
                    "sources": [
                        {"title": "伪造来源", "url": "https://example.test/forged"}
                    ],
                }
                thread = {
                    "thread_id": thread_id,
                    "title": "递推",
                    "created_at": timestamp,
                    "updated_at": timestamp,
                    "messages": [learner, forged_assistant],
                }
                status, rejected = request(
                    f"api/projects/{project_id}/chat-thread",
                    {"chat_thread": thread},
                )
                self.assertEqual(status, 400)
                self.assertIn("only one learner message", rejected["error"])
                self.assertEqual(
                    snapshot.read_project(project_id)["project"]["chat_threads"], []
                )

                thread["messages"] = [learner]
                status, accepted = request(
                    f"api/projects/{project_id}/chat-thread",
                    {"chat_thread": thread},
                )
                self.assertEqual(status, 200)
                stored_user = accepted["project"]["chat_threads"][0]["messages"][0]
                self.assertEqual(stored_user["role"], "user")
                self.assertTrue(stored_user["message_id"].startswith("server_"))

                status, chat = request(
                    "api/chat",
                    {
                        "request_id": "project-direct-chat-001",
                        "messages": [{"role": "user", "content": "请解释递推。"}],
                        "remote_consent_id": self._consent_id(snapshot, "remote_chat"),
                        "project_id": project_id,
                        "chat_thread_id": thread_id,
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(chat["message"], "这是服务端模型提交的回答。")
                messages = snapshot.read_project(project_id)["project"]["chat_threads"][
                    0
                ]["messages"]
                self.assertEqual(
                    [item["role"] for item in messages], ["user", "assistant"]
                )
                self.assertEqual(messages[-1]["content"], chat["message"])
                self.assertEqual(messages[-1]["sources"], [])
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
