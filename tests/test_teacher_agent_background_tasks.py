from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import http.client
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job
    import tomli as tomllib
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator, FormatChecker

from teaching_skill_miner.harness import HarnessJournal, HarnessRunHandle
from teaching_skill_miner.harness.events import HarnessEventEmitter
from teaching_skill_miner.io_utils import project_root
from teaching_skill_miner.teacher_agent_dashboard import (
    TeacherAgentDashboardError,
    TeacherAgentDashboardSnapshot,
    _request_fingerprint,
    build_teacher_agent_dashboard_snapshot,
    create_teacher_agent_dashboard_server,
)
from teaching_skill_miner.teacher_agent_task_registry import (
    BackgroundTaskConflictError,
    BackgroundTaskRegistryError,
    DurableBackgroundTaskRegistry,
    public_task_projection,
    task_id_for_run,
)


class BackgroundTaskRegistryTests(unittest.TestCase):
    def _register(
        self, store: DurableBackgroundTaskRegistry, *, suffix: str = "1"
    ) -> tuple[str, dict[str, object]]:
        run_id = "stream_" + suffix.zfill(40)[-40:]
        task_id = task_id_for_run(run_id)
        record, created = store.register(
            run_id=run_id,
            turn_id="turn_" + suffix.zfill(40)[-40:],
            operation="chat",
            request_fingerprint=(suffix[-1:] or "1") * 64,
            private_request={
                "operation": "chat",
                "payload": {
                    "messages": [
                        {"role": "user", "content": "learner secret answer"}
                    ],
                    "provider_token": "must-never-be-public",
                },
                "request_id": f"request-{suffix}",
                "run_id": run_id,
                "turn_id": "turn_" + suffix.zfill(40)[-40:],
                "after_sequence": 0,
            },
            scope={"project_id": "project_0123456789abcdef01234567"},
            session_id=None,
        )
        self.assertTrue(created)
        return task_id, record

    def test_private_request_is_durable_but_public_projection_is_content_free(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            first = DurableBackgroundTaskRegistry(directory)
            task_id, _record = self._register(first)
            restarted = DurableBackgroundTaskRegistry(directory)

            private = restarted.get_private(task_id)
            self.assertIn("learner secret answer", str(private))
            public = public_task_projection(private)
            rendered = str(public)
            self.assertNotIn("learner secret answer", rendered)
            self.assertNotIn("must-never-be-public", rendered)
            self.assertNotIn("messages", rendered)
            self.assertFalse(public["content_included"])
            self.assertEqual(public["status"], "queued")

    def test_command_outbox_survives_restart_and_clears_only_after_ack(self) -> None:
        with TemporaryDirectory() as directory:
            store = DurableBackgroundTaskRegistry(directory)
            task_id, _record = self._register(store)
            running = store.mark_running(task_id)
            requested, applied = store.request_control(
                task_id,
                command_type="cancel",
                expected_version=int(running["version"]),
                idempotency_key="cancel-once",
                reason_code="user_requested",
            )
            self.assertTrue(applied)
            command_id = str(requested["command_outbox"][0]["command_id"])

            restarted = DurableBackgroundTaskRegistry(directory)
            self.assertEqual(
                restarted.get_private(task_id)["command_outbox"][0]["command_id"],
                command_id,
            )
            acknowledged = restarted.acknowledge_command(task_id, command_id)
            self.assertEqual(acknowledged["command_outbox"], [])
            replay, replay_applied = restarted.request_control(
                task_id,
                command_type="cancel",
                expected_version=int(running["version"]),
                idempotency_key="cancel-once",
                reason_code="user_requested",
            )
            self.assertTrue(replay_applied)
            self.assertEqual(replay["command_outbox"], [])

    def test_control_cas_allows_only_one_concurrent_writer(self) -> None:
        with TemporaryDirectory() as directory:
            store = DurableBackgroundTaskRegistry(directory)
            task_id, _record = self._register(store)
            running = store.mark_running(task_id)
            expected_version = int(running["version"])
            barrier = threading.Barrier(2)

            def cancel(index: int) -> str:
                barrier.wait(timeout=2)
                try:
                    store.request_control(
                        task_id,
                        command_type="cancel",
                        expected_version=expected_version,
                        idempotency_key=f"cancel-{index}",
                        reason_code="user_requested",
                    )
                except BackgroundTaskConflictError:
                    return "conflict"
                return "applied"

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = sorted(executor.map(cancel, (1, 2)))
            self.assertEqual(results, ["applied", "conflict"])
            self.assertEqual(
                len(store.get_private(task_id)["command_outbox"]), 1
            )

    def test_purge_removes_private_content_and_blocks_identity_reuse(self) -> None:
        with TemporaryDirectory() as directory:
            store = DurableBackgroundTaskRegistry(directory)
            task_id, record = self._register(store)
            running = store.mark_running(task_id)
            terminal = store.mark_terminal(
                task_id,
                status="completed",
                terminal_type="run.completed",
                last_sequence=4,
            )
            self.assertGreater(int(terminal["version"]), int(running["version"]))
            self.assertEqual(store.purge_tasks([task_id])["tasks"], 1)
            self.assertNotIn("learner secret answer", store.path.read_text("utf-8"))
            with self.assertRaises(BackgroundTaskConflictError):
                store.register(
                    run_id=str(record["run_id"]),
                    turn_id=str(record["turn_id"]),
                    operation="chat",
                    request_fingerprint=str(record["request_fingerprint"]),
                    private_request=record["private_request"],
                    scope=record["scope"],
                    session_id=None,
                )

    def test_control_reason_is_a_code_not_free_text(self) -> None:
        with TemporaryDirectory() as directory:
            store = DurableBackgroundTaskRegistry(directory)
            task_id, _record = self._register(store)
            running = store.mark_running(task_id)
            with self.assertRaises(BackgroundTaskRegistryError):
                store.request_control(
                    task_id,
                    command_type="cancel",
                    expected_version=int(running["version"]),
                    idempotency_key="unsafe-reason",
                    reason_code="student said a private sentence",
                )

    def test_public_and_private_schemas_are_packaged_and_validate(self) -> None:
        root = project_root()
        schema_names = {
            "schema/teacher_agent_background_task.schema.json",
            "schema/teacher_agent_background_task_registry.schema.json",
        }
        release_names = set(
            (root / "release/public_json_resources.txt")
            .read_text("utf-8")
            .splitlines()
        )
        pyproject = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))
        packaged = set(
            pyproject["tool"]["setuptools"]["data-files"][
                "share/teaching-skill-miner/schema"
            ]
        )
        self.assertTrue(schema_names.issubset(release_names))
        self.assertTrue(schema_names.issubset(packaged))

        with TemporaryDirectory() as directory:
            store = DurableBackgroundTaskRegistry(directory)
            _task_id, record = self._register(store)
            store.request_cancel_by_request_id(
                "schema-pending-request", reason_code="user_requested"
            )
            values = {
                "teacher_agent_background_task.schema.json": public_task_projection(
                    record
                ),
                "teacher_agent_background_task_registry.schema.json": json.loads(
                    store.path.read_text("utf-8")
                ),
            }
            for name, value in values.items():
                schema = json.loads((root / "schema" / name).read_text("utf-8"))
                Draft202012Validator.check_schema(schema)
                Draft202012Validator(
                    schema, format_checker=FormatChecker()
                ).validate(value)

    def test_registry_rejects_symlink_lock_and_checksum_valid_extra_fields(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = DurableBackgroundTaskRegistry(root)
            _task_id, _record = self._register(store)
            state = json.loads(store.path.read_text(encoding="utf-8"))
            material = dict(state)
            material.pop("registry_sha256")
            task_id = next(iter(material["tasks"]))
            material["tasks"][task_id]["unexpected_private_field"] = (
                "must-fail-closed"
            )
            material["registry_sha256"] = sha256(
                json.dumps(
                    material,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            store.path.write_text(
                json.dumps(material, ensure_ascii=False), encoding="utf-8"
            )
            with self.assertRaises(BackgroundTaskRegistryError):
                DurableBackgroundTaskRegistry(root)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "attacker-controlled"
            target.write_text("unchanged", encoding="utf-8")
            (root / ".task_registry.lock").symlink_to(target)
            with self.assertRaises(BackgroundTaskRegistryError):
                DurableBackgroundTaskRegistry(root)
            self.assertEqual(target.read_text(encoding="utf-8"), "unchanged")


class BackgroundTaskDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = project_root()
        cls.library = root / "data/teacher_agent_skill_library.json"
        cls.demo = root / "data/teacher_agent_demo_input.json"
        cls.cases = root / "data/teacher_agent_evaluation_cases.json"

    def _snapshot(self, store_path: Path):
        return build_teacher_agent_dashboard_snapshot(
            self.library,
            self.demo,
            self.cases,
            store_path=store_path,
        )

    @staticmethod
    def _body(snapshot, suffix: str) -> dict[str, object]:
        return {
            "operation": "start",
            "request_id": f"background-task-{suffix}",
            "payload": {
                "goal": snapshot.demo_input["goal"],
                "student_profile": snapshot.demo_input["student_profile"],
                "start_idempotency_key": f"background-task-start-{suffix}",
            },
        }

    def test_crash_after_registration_before_worker_start_recovers_once(self) -> None:
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "sessions.jsonl"
            first = self._snapshot(store_path)
            body = self._body(first, "register-crash")
            with patch.object(
                HarnessRunHandle,
                "start",
                side_effect=KeyboardInterrupt("injected process death"),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    first.open_harness_stream(body)
            queued = first._background_tasks().list_private()
            self.assertEqual(len(queued), 1)
            self.assertEqual(queued[0]["status"], "running")

            restarted = self._snapshot(store_path)
            task_id = str(queued[0]["task_id"])
            run_id = str(queued[0]["run_id"])
            live = restarted.stream_runs[run_id]
            self.assertEqual(live.handle.wait(timeout=5)["status"], "completed")
            status = restarted.background_task_status({"task_id": task_id})["task"]
            self.assertEqual(status["status"], "completed")
            self.assertEqual(len(restarted.sessions), 1)

    def test_stop_before_registration_survives_server_restart(self) -> None:
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "sessions.jsonl"
            first = self._snapshot(store_path)
            body = self._body(first, "cancel-before-register")
            receipt = first.cancel_harness_stream(
                {
                    "request_id": body["request_id"],
                    "reason": "user_requested",
                }
            )
            self.assertTrue(receipt["pending_registration"])
            self.assertTrue(receipt["durable"])

            restarted = self._snapshot(store_path)
            record, _cursor = restarted.open_harness_stream(body)
            self.assertEqual(record.handle.wait(timeout=5)["status"], "cancelled")
            self.assertEqual(len(restarted.sessions), 0)
            task = restarted.background_task_status(
                {"task_id": str(record.task_id)}
            )["task"]
            self.assertEqual(task["status"], "cancelled")

    def test_safe_suspended_task_requires_cas_resume_and_then_completes(self) -> None:
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "sessions.jsonl"
            snapshot = self._snapshot(store_path)
            body = self._body(snapshot, "resume")
            with patch.object(
                HarnessRunHandle,
                "start",
                side_effect=KeyboardInterrupt("injected process death"),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    snapshot.open_harness_stream(body)
            registry = snapshot._background_tasks()
            private = registry.list_private()[0]
            suspended = registry.mark_suspended(
                str(private["task_id"]),
                error_code="worker_interrupted",
                last_sequence=0,
                unsafe_handoff=False,
            )
            response = snapshot.resume_background_task(
                {
                    "task_id": suspended["task_id"],
                    "expected_version": suspended["version"],
                    "task_idempotency_key": "resume-once",
                }
            )
            run_id = str(private["run_id"])
            self.assertTrue(response["resume_requested"])
            self.assertEqual(
                snapshot.stream_runs[run_id].handle.wait(timeout=5)["status"],
                "completed",
            )
            self.assertEqual(
                snapshot.background_task_status(
                    {"task_id": str(private["task_id"])}
                )["task"]["status"],
                "completed",
            )

    def test_restart_after_cancel_command_ack_still_cancels_before_effect(self) -> None:
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "sessions.jsonl"
            first = self._snapshot(store_path)
            with patch.object(
                HarnessRunHandle,
                "start",
                side_effect=KeyboardInterrupt("crash before worker start"),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    first.open_harness_stream(self._body(first, "cancel-ack-crash"))
            registry = first._background_tasks()
            private = registry.list_private()[0]
            requested, applied = registry.request_control(
                str(private["task_id"]),
                command_type="cancel",
                expected_version=int(private["version"]),
                idempotency_key="cancel-before-restart",
                reason_code="user_requested",
            )
            self.assertTrue(applied)
            registry.acknowledge_command(
                str(private["task_id"]),
                str(requested["command_outbox"][0]["command_id"]),
            )

            restarted = self._snapshot(store_path)
            run_id = str(private["run_id"])
            self.assertEqual(
                restarted.stream_runs[run_id].handle.wait(timeout=5)["status"],
                "cancelled",
            )
            self.assertEqual(len(restarted.sessions), 0)
            self.assertEqual(
                restarted.background_task_status(
                    {"task_id": str(private["task_id"])}
                )["task"]["status"],
                "cancelled",
            )

    def test_no_sse_subscriber_is_not_task_cancellation(self) -> None:
        with TemporaryDirectory() as directory:
            snapshot = self._snapshot(Path(directory) / "sessions.jsonl")
            record, _cursor = snapshot.open_harness_stream(
                self._body(snapshot, "detached")
            )
            # No call to the HTTP `_stream` subscriber is made. The worker is
            # server-owned and must still reach its durable terminal state.
            self.assertEqual(record.handle.wait(timeout=5)["status"], "completed")
            task = snapshot.background_task_status(
                {"task_id": str(record.task_id)}
            )["task"]
            self.assertEqual(task["status"], "completed")
            self.assertFalse(task["cancel_command_pending"])

    def test_http_reader_disconnect_detaches_but_task_reaches_terminal(self) -> None:
        with TemporaryDirectory() as directory:
            snapshot = self._snapshot(Path(directory) / "sessions.jsonl")
            entered = threading.Event()
            release = threading.Event()
            original_start = TeacherAgentDashboardSnapshot.start

            def delayed_start(current, *args, **kwargs):
                entered.set()
                self.assertTrue(release.wait(timeout=5))
                return original_start(current, *args, **kwargs)

            try:
                server, url = create_teacher_agent_dashboard_server(
                    snapshot, capability_token="d" * 24
                )
            except PermissionError:
                self.skipTest("sandbox does not permit loopback socket binding")
            server_thread = threading.Thread(
                target=server.serve_forever, daemon=True
            )
            server_thread.start()
            parsed = urlsplit(url)
            connection = http.client.HTTPConnection(
                parsed.hostname, parsed.port, timeout=5
            )
            body = self._body(snapshot, "http-detach")
            try:
                with patch.object(
                    TeacherAgentDashboardSnapshot, "start", new=delayed_start
                ):
                    connection.request(
                        "POST",
                        f"{parsed.path}api/stream",
                        body=json.dumps(body).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                    )
                    response = connection.getresponse()
                    self.assertEqual(response.status, 200)
                    task_id = str(response.getheader("X-Background-Task-ID"))
                    self.assertTrue(entered.wait(timeout=5))
                    # Close the only SSE reader without sending /api/cancel.
                    connection.close()
                    release.set()
                    task = snapshot._background_tasks().get_private(task_id)
                    live = snapshot.stream_runs[str(task["run_id"])]
                    self.assertEqual(
                        live.handle.wait(timeout=5)["status"], "completed"
                    )
                status = snapshot.background_task_status({"task_id": task_id})[
                    "task"
                ]
                self.assertEqual(status["status"], "completed")
                self.assertFalse(status["cancel_command_pending"])
                api_connection = http.client.HTTPConnection(
                    parsed.hostname, parsed.port, timeout=5
                )
                api_connection.request(
                    "POST",
                    f"{parsed.path}api/tasks/list",
                    body=b"{}",
                    headers={"Content-Type": "application/json"},
                )
                listed_response = api_connection.getresponse()
                listed_bytes = listed_response.read()
                self.assertEqual(listed_response.status, 200)
                self.assertNotIn(b"http-detach", listed_bytes)
                listed = json.loads(listed_bytes)
                self.assertFalse(listed["content_included"])
                self.assertEqual(listed["tasks"][0]["task_id"], task_id)
                self.assertFalse(listed["tasks"][0]["content_included"])
                api_connection.close()
            finally:
                release.set()
                connection.close()
                server.shutdown()
                server.server_close()
                server_thread.join(timeout=5)

    def test_restart_with_unknown_chat_effect_handoffs_without_replay(self) -> None:
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "sessions.jsonl"
            first = self._snapshot(store_path)
            request_id = "unknown-chat-effect"
            payload = {
                "messages": [{"role": "user", "content": "private question"}],
            }
            fingerprint = _request_fingerprint(
                {"operation": "chat", "payload": payload}
            )
            run_id = "stream_" + sha256(
                f"request\x00{request_id}".encode("utf-8")
            ).hexdigest()[:40]
            turn_id = "turn_" + sha256(
                f"{run_id}\x00{fingerprint}".encode("utf-8")
            ).hexdigest()[:40]
            private_request = {
                "operation": "chat",
                "payload": payload,
                "request_id": request_id,
                "run_id": run_id,
                "turn_id": turn_id,
                "after_sequence": 0,
            }
            task, _created = first._background_tasks().register(
                run_id=run_id,
                turn_id=turn_id,
                operation="chat",
                request_fingerprint=fingerprint,
                private_request=private_request,
                scope={},
                session_id=None,
            )
            journal = HarnessJournal(
                first._stream_journal_root() / f"{run_id}.jsonl",
                run_id=run_id,
                turn_id=turn_id,
            )
            emitter = HarnessEventEmitter(
                run_id=run_id,
                turn_id=turn_id,
                next_sequence=journal.next_sequence,
                durable_sink=journal.append,
            )
            emitter.emit(
                "run.started",
                {
                    "channel": "internal",
                    "operation": "chat",
                    "durable": True,
                    "resumed": False,
                    "restart_recoverable": False,
                },
            )

            restarted = self._snapshot(store_path)
            status = restarted.background_task_status(
                {"task_id": str(task["task_id"])}
            )["task"]
            self.assertEqual(status["status"], "handoff")
            self.assertEqual(status["terminal_type"], "run.handoff")
            self.assertEqual(
                HarnessJournal(
                    restarted._stream_journal_root() / f"{run_id}.jsonl",
                    run_id=run_id,
                    turn_id=turn_id,
                ).terminal_type,
                "run.handoff",
            )

    def test_poisoned_teach_journal_handoffs_without_domain_replay(self) -> None:
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "sessions.jsonl"
            first = self._snapshot(store_path)
            body = self._body(first, "poisoned-journal")
            operation = str(body["operation"])
            payload = dict(body["payload"])
            request_id = str(body["request_id"])
            fingerprint = _request_fingerprint(
                {"operation": operation, "payload": payload}
            )
            run_id = "stream_" + sha256(
                f"request\x00{request_id}".encode("utf-8")
            ).hexdigest()[:40]
            turn_id = "turn_" + sha256(
                f"{run_id}\x00{fingerprint}".encode("utf-8")
            ).hexdigest()[:40]
            task, _created = first._background_tasks().register(
                run_id=run_id,
                turn_id=turn_id,
                operation=operation,
                request_fingerprint=fingerprint,
                private_request={
                    **body,
                    "run_id": run_id,
                    "turn_id": turn_id,
                    "after_sequence": 0,
                },
                scope={},
                session_id=None,
            )
            journal_path = first._stream_journal_root() / f"{run_id}.jsonl"
            journal_path.write_text("not-json\n", encoding="utf-8")

            restarted = self._snapshot(store_path)
            status = restarted.background_task_status(
                {"task_id": str(task["task_id"])}
            )["task"]
            self.assertEqual(status["status"], "handoff")
            self.assertEqual(status["terminal_type"], "run.handoff")
            self.assertEqual(status["error_code"], "journal_integrity_failed")
            self.assertEqual(len(restarted.sessions), 0)

    def test_restart_before_teach_checkpoint_reconstructs_safe_prefix(self) -> None:
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "sessions.jsonl"
            first = self._snapshot(store_path)
            request_id = "teach-before-checkpoint"
            payload = {
                "goal": first.demo_input["goal"],
                "student_profile": first.demo_input["student_profile"],
                "start_idempotency_key": "teach-before-checkpoint-key",
            }
            fingerprint = _request_fingerprint(
                {"operation": "start", "payload": payload}
            )
            run_id = "stream_" + sha256(
                f"request\x00{request_id}".encode("utf-8")
            ).hexdigest()[:40]
            turn_id = "turn_" + sha256(
                f"{run_id}\x00{fingerprint}".encode("utf-8")
            ).hexdigest()[:40]
            private_request = {
                "operation": "start",
                "payload": payload,
                "request_id": request_id,
                "run_id": run_id,
                "turn_id": turn_id,
                "after_sequence": 0,
            }
            task, _created = first._background_tasks().register(
                run_id=run_id,
                turn_id=turn_id,
                operation="start",
                request_fingerprint=fingerprint,
                private_request=private_request,
                scope={},
                session_id=None,
            )
            journal = HarnessJournal(
                first._stream_journal_root() / f"{run_id}.jsonl",
                run_id=run_id,
                turn_id=turn_id,
            )
            emitter = HarnessEventEmitter(
                run_id=run_id,
                turn_id=turn_id,
                next_sequence=journal.next_sequence,
                durable_sink=journal.append,
            )
            emitter.emit(
                "run.started",
                {
                    "channel": "internal",
                    "operation": "start",
                    "durable": True,
                    "resumed": False,
                    "restart_recoverable": True,
                },
            )
            emitter.emit(
                "action.started", {"channel": "internal", "operation": "start"}
            )

            restarted = self._snapshot(store_path)
            self.assertEqual(
                restarted.stream_runs[run_id].handle.wait(timeout=5)["status"],
                "completed",
            )
            self.assertEqual(len(restarted.sessions), 1)
            self.assertEqual(
                restarted.background_task_status(
                    {"task_id": str(task["task_id"])}
                )["task"]["status"],
                "completed",
            )

    def test_two_server_instances_dispatch_one_run_only_once(self) -> None:
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "sessions.jsonl"
            first = self._snapshot(store_path)
            second = self._snapshot(store_path)
            body = self._body(first, "two-instances")
            entered = threading.Event()
            release = threading.Event()
            original_start = TeacherAgentDashboardSnapshot.start

            def slow_start(snapshot, *args, **kwargs):
                if snapshot is first:
                    entered.set()
                    self.assertTrue(release.wait(timeout=3))
                return original_start(snapshot, *args, **kwargs)

            with patch.object(TeacherAgentDashboardSnapshot, "start", new=slow_start):
                first_record, _ = first.open_harness_stream(body)
                self.assertTrue(entered.wait(timeout=3))
                with self.assertRaises(TeacherAgentDashboardError):
                    second.open_harness_stream(body)
                release.set()
                self.assertEqual(
                    first_record.handle.wait(timeout=5)["status"], "completed"
                )

            restarted = self._snapshot(store_path)
            self.assertEqual(len(restarted.sessions), 1)
            tasks = restarted.list_background_tasks({})["tasks"]
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["status"], "completed")

    def test_capacity_queue_drains_fifo_after_worker_settles(self) -> None:
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "sessions.jsonl"
            snapshot = self._snapshot(store_path)
            first_body = self._body(snapshot, "fifo-first")
            second_body = self._body(snapshot, "fifo-second")
            first_entered = threading.Event()
            release_first = threading.Event()
            second_entered = threading.Event()
            original_start = TeacherAgentDashboardSnapshot.start

            def ordered_start(current, body, **kwargs):
                key = body.get("start_idempotency_key")
                if key == "background-task-start-fifo-first":
                    first_entered.set()
                    self.assertTrue(release_first.wait(timeout=5))
                elif key == "background-task-start-fifo-second":
                    second_entered.set()
                return original_start(current, body, **kwargs)

            with (
                patch(
                    "teaching_skill_miner.teacher_agent_dashboard._MAX_STREAM_RUNS",
                    1,
                ),
                patch.object(
                    TeacherAgentDashboardSnapshot, "start", new=ordered_start
                ),
            ):
                first, _cursor = snapshot.open_harness_stream(first_body)
                self.assertTrue(first_entered.wait(timeout=5))
                with self.assertRaisesRegex(
                    TeacherAgentDashboardError, "stream run capacity is busy"
                ):
                    snapshot.open_harness_stream(second_body)
                queued = [
                    item
                    for item in snapshot._background_tasks().list_private()
                    if item["private_request"].get("request_id")
                    == second_body["request_id"]
                ]
                self.assertEqual(len(queued), 1)
                self.assertEqual(queued[0]["status"], "queued")

                release_first.set()
                self.assertEqual(first.handle.wait(timeout=5)["status"], "completed")
                self.assertTrue(second_entered.wait(timeout=5))
                second_run_id = str(queued[0]["run_id"])
                self.assertEqual(
                    snapshot.stream_runs[second_run_id].handle.wait(timeout=5)[
                        "status"
                    ],
                    "completed",
                )
                snapshot._wait_for_background_task_queue_drains(timeout=5)
            self.assertEqual(len(snapshot.sessions), 2)


if __name__ == "__main__":
    unittest.main()
