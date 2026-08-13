from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import tempfile
from threading import Event, Thread
import time
import unittest
from unittest import mock

from teaching_skill_miner.deepseek_client import (
    DeepSeekClient,
    DeepSeekClientError,
    DeepSeekConfig,
    DeepSeekConfigurationError,
    _default_stream_transport,
)
from teaching_skill_miner.harness import CancellationToken, HarnessCancelled


def _envelope(content: dict, *, response_id: str = "response_test") -> bytes:
    return json.dumps(
        {
            "id": response_id,
            "choices": [{"message": {"content": json.dumps(content)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
    ).encode()


def _web_envelope() -> bytes:
    return json.dumps(
        {
            "id": "msg_web_test",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_test",
                    "name": "web_search",
                    "input": {"query": "DeepSeek current release"},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_test",
                    "content": [
                        {
                            "type": "web_search_result",
                            "title": "DeepSeek API Docs",
                            "url": (
                                "https://api-docs.deepseek.com/updates/"
                                "?utm_source=test&from=search&lang=zh"
                            ),
                        },
                        {
                            "type": "web_search_result",
                            "title": "Unsafe source",
                            "url": "javascript:alert(1)",
                        },
                    ],
                },
                {
                    "type": "text",
                    "text": "DeepSeek 当前版本信息已经核验。",
                    "citations": [
                        {
                            "title": "DeepSeek API Docs",
                            "url": (
                                "https://api-docs.deepseek.com/updates/"
                                "?utm_source=test&from=search&lang=zh"
                            ),
                        }
                    ],
                },
            ],
            "usage": {
                "input_tokens": 30,
                "output_tokens": 20,
                "server_tool_use": {"web_search_requests": 1},
            },
            "stop_reason": "end_turn",
        }
    ).encode()


def _anthropic_sse(events: list[dict], *, crlf: bool = False) -> bytes:
    newline = b"\r\n" if crlf else b"\n"
    frames: list[bytes] = []
    for item in events:
        data_lines = (
            json.dumps(item, ensure_ascii=False, indent=2).encode("utf-8").splitlines()
        )
        frames.append(
            b"event: "
            + str(item["type"]).encode("utf-8")
            + newline
            + b"".join(b"data: " + line + newline for line in data_lines)
            + newline
        )
    return b"".join(frames)


def _fragment_bytes(value: bytes, sizes: tuple[int, ...]) -> list[bytes]:
    chunks: list[bytes] = []
    cursor = 0
    size_index = 0
    while cursor < len(value):
        size = sizes[size_index % len(sizes)]
        chunks.append(value[cursor : cursor + size])
        cursor += size
        size_index += 1
    return chunks


def _minimal_web_events(*, stop_reason: str | None = "end_turn") -> list[dict]:
    events: list[dict] = [
        {
            "type": "message_start",
            "message": {
                "id": "msg-minimal-web",
                "stop_reason": None,
                "usage": {"input_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "answer"},
        },
        {"type": "content_block_stop", "index": 0},
    ]
    delta: dict[str, object] = {"stop_reason": stop_reason}
    events.extend(
        [
            {
                "type": "message_delta",
                "delta": delta,
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
    )
    return events


class DeepSeekClientTests(unittest.TestCase):
    def test_content_free_model_probe_validates_credential_path_and_model(self) -> None:
        captured: dict[str, object] = {}

        def probe(url, headers, timeout):
            captured.update(url=url, headers=dict(headers), timeout=timeout)
            return 200, json.dumps(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "deepseek-v4-flash",
                            "object": "model",
                            "owned_by": "deepseek",
                        }
                    ],
                }
            ).encode()

        client = DeepSeekClient(
            DeepSeekConfig(max_retries=0),
            api_key="provider-readiness-secret",
            probe_transport=probe,
        )
        result = client.probe_model_availability(timeout_seconds=3)
        self.assertEqual(captured["url"], "https://api.deepseek.com/models")
        self.assertEqual(captured["timeout"], 3.0)
        self.assertEqual(
            captured["headers"]["Authorization"],  # type: ignore[index]
            "Bearer provider-readiness-secret",
        )
        self.assertEqual(
            result,
            {
                "schema": "teaching_skill_miner.deepseek_readiness.v1",
                "credential_validated": True,
                "provider_network_validated": True,
                "configured_model_available": True,
                "learner_content_sent": False,
                "generation_created": False,
            },
        )
        self.assertNotIn("provider-readiness-secret", json.dumps(result))

    def test_model_probe_fails_closed_for_auth_outage_schema_and_missing_model(
        self,
    ) -> None:
        failures = (
            (401, b'{"error":"credential leaked detail"}'),
            (503, b"upstream private detail"),
            (200, b"not-json"),
            (200, b'{"object":"list","data":[]}'),
            (
                200,
                b'{"object":"list","data":[{"id":"other-model","object":"model","owned_by":"deepseek"}]}',
            ),
        )
        for status, body in failures:
            with self.subTest(status=status, body=body[:16]):
                client = DeepSeekClient(
                    DeepSeekConfig(max_retries=0),
                    api_key="never-log-this-secret",
                    probe_transport=lambda *_args, pair=(status, body): pair,
                )
                with self.assertRaisesRegex(
                    DeepSeekClientError, "^DeepSeek readiness probe failed$"
                ) as caught:
                    client.probe_model_availability()
                self.assertNotIn("never-log-this-secret", str(caught.exception))
                self.assertNotIn("credential leaked detail", str(caught.exception))

    def test_anthropic_web_search_returns_safe_sources_without_exposing_key(
        self,
    ) -> None:
        captured: dict[str, object] = {}

        def transport(url: str, headers: dict, payload: bytes, timeout: float):
            captured.update(
                {
                    "url": url,
                    "api_key": headers["x-api-key"],
                    "payload": payload,
                    "timeout": timeout,
                }
            )
            return 200, _web_envelope()

        key = "secret-web-key"
        client = DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=True),
            api_key=key,
            transport=transport,
        )
        content, trace = client.chat_web(
            [{"role": "user", "content": "搜索最新版本"}],
            system="需要最新信息时使用联网搜索。",
            request_kind="web_unit_test",
        )
        self.assertEqual(content["message"], "DeepSeek 当前版本信息已经核验。")
        self.assertTrue(content["web_search_used"])
        self.assertEqual(
            content["sources"],
            [
                {
                    "title": "DeepSeek API Docs",
                    "url": "https://api-docs.deepseek.com/updates/?lang=zh",
                }
            ],
        )
        self.assertEqual(
            captured["url"], "https://api.deepseek.com/anthropic/v1/messages"
        )
        self.assertEqual(captured["api_key"], key)
        request_body = json.loads(captured["payload"])
        self.assertEqual(
            request_body["tools"],
            [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 3,
                }
            ],
        )
        self.assertNotIn(key, json.dumps(trace))
        self.assertEqual(trace["web_search_source_count"], 1)

    def test_remote_consent_is_fail_closed(self) -> None:
        client = DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=False),
            api_key="secret-test-key",
            transport=lambda *_args: (200, _envelope({"ok": True})),
        )
        with self.assertRaisesRegex(DeepSeekConfigurationError, "disabled"):
            client.chat_json(
                [{"role": "user", "content": "json please"}],
                request_kind="test",
            )

    def test_structured_response_and_trace_never_expose_key(self) -> None:
        captured: dict[str, object] = {}

        def transport(url: str, headers: dict, payload: bytes, timeout: float):
            captured.update(
                {
                    "url": url,
                    "authorization": headers["Authorization"],
                    "payload": payload,
                    "timeout": timeout,
                }
            )
            return 200, _envelope({"signal": "partial"})

        key = "secret-test-key"
        client = DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=True),
            api_key=key,
            transport=transport,
        )
        content, trace = client.chat_json(
            [{"role": "user", "content": "return json"}], request_kind="unit_test"
        )
        self.assertEqual(content, {"signal": "partial"})
        self.assertEqual(captured["url"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(captured["authorization"], f"Bearer {key}")
        self.assertNotIn(key, json.dumps(trace))
        self.assertFalse(trace["credential_logged"])
        body = json.loads(captured["payload"])
        self.assertEqual(body["model"], "deepseek-v4-flash")
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["response_format"], {"type": "json_object"})

    def test_structured_request_can_use_a_bounded_per_call_token_budget(self) -> None:
        captured: dict[str, object] = {}

        def transport(_url: str, _headers: dict, payload: bytes, _timeout: float):
            captured["payload"] = payload
            return 200, _envelope({"ok": True})

        client = DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=True),
            api_key="secret-test-key",
            transport=transport,
        )
        content, _ = client.chat_json(
            [{"role": "user", "content": "return json"}],
            request_kind="larger_bounded_output",
            max_tokens=8_192,
        )
        self.assertTrue(content["ok"])
        request_body = json.loads(captured["payload"])
        self.assertEqual(request_body["max_tokens"], 8_192)
        self.assertEqual(client.config.max_tokens, 1_800)

    def test_length_finish_reason_is_reported_as_truncated_output(self) -> None:
        envelope = json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": '{"incomplete":'},
                    }
                ]
            }
        ).encode()
        client = DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=True),
            api_key="secret-test-key",
            transport=lambda *_args: (200, envelope),
        )
        with self.assertRaisesRegex(DeepSeekClientError, "truncated"):
            client.chat_json(
                [{"role": "user", "content": "return json"}],
                request_kind="truncated_output",
                max_tokens=8_192,
            )

    def test_api_key_file_is_read_only_at_request_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deepseek_api.txt"
            path.write_text("file-secret-key\n", encoding="utf-8")
            client = DeepSeekClient(
                DeepSeekConfig(
                    allow_remote_student_data=True,
                    api_key_file=path,
                ),
                transport=lambda _url, headers, _payload, _timeout: (
                    200,
                    _envelope(
                        {
                            "authorized": headers["Authorization"]
                            == "Bearer file-secret-key"
                        }
                    ),
                ),
            )
            content, trace = client.chat_json(
                [{"role": "system", "content": "json"}], request_kind="key_file_test"
            )
            self.assertTrue(content["authorized"])
            self.assertNotIn("file-secret-key", json.dumps(trace))

    def test_retryable_status_is_retried(self) -> None:
        responses = deque(
            [
                (429, b'{"error":"rate limited"}'),
                (200, _envelope({"ok": True})),
            ]
        )
        client = DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=True, max_retries=1),
            api_key="secret-test-key",
            transport=lambda *_args: responses.popleft(),
        )
        content, trace = client.chat_json(
            [{"role": "user", "content": "json"}], request_kind="retry_test"
        )
        self.assertTrue(content["ok"])
        self.assertEqual(trace["attempt_count"], 2)

    def test_malformed_model_json_fails_without_leaking_response(self) -> None:
        malformed = json.dumps(
            {"choices": [{"message": {"content": "not-json"}}]}
        ).encode()
        client = DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=True),
            api_key="secret-test-key",
            transport=lambda *_args: (200, malformed),
        )
        with self.assertRaisesRegex(DeepSeekClientError, "malformed") as caught:
            client.chat_json(
                [{"role": "user", "content": "json"}], request_kind="malformed_test"
            )
        self.assertNotIn("secret-test-key", str(caught.exception))
        self.assertNotIn("not-json", str(caught.exception))

    def test_legacy_model_aliases_are_rejected(self) -> None:
        for model in ("deepseek-chat", "deepseek-reasoner", "deepseek-v4-pro"):
            with self.subTest(model=model):
                with self.assertRaisesRegex(DeepSeekConfigurationError, "model"):
                    DeepSeekConfig(model=model).validated()

    def test_direct_api_key_rejects_embedded_whitespace(self) -> None:
        client = DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=True),
            api_key="secret\r\nInjected: header",
            transport=lambda *_args: (200, _envelope({"ok": True})),
        )
        with self.assertRaisesRegex(DeepSeekConfigurationError, "malformed"):
            client.chat_json(
                [{"role": "user", "content": "json"}], request_kind="bad_key"
            )

    def test_native_text_stream_emits_deltas_and_secret_free_trace(self) -> None:
        captured: dict[str, object] = {}
        lines = [
            b'data: {"id":"stream_1","choices":[{"delta":{"content":"hello "}}]}\n',
            b'data: {"id":"stream_1","choices":[{"delta":{"content":"world"},"finish_reason":"stop"}]}\n',
            b'data: {"id":"stream_1","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":2}}\n',
            b"data: [DONE]\n",
        ]

        def stream_transport(url, headers, payload, timeout, cancellation_token):
            captured.update(
                {
                    "url": url,
                    "headers": dict(headers),
                    "body": json.loads(payload),
                    "timeout": timeout,
                    "token": cancellation_token,
                }
            )
            return 200, list(lines)

        token = CancellationToken()
        client = DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=True, max_retries=0),
            api_key="stream-secret",
            stream_transport=stream_transport,
        )
        chunks = list(
            client.chat_text_stream(
                [{"role": "user", "content": "say hello"}],
                request_kind="native_stream_test",
                cancellation_token=token,
                deadline_monotonic=time.monotonic() + 10,
            )
        )

        self.assertEqual(
            "".join(
                str(item.get("text", ""))
                for item in chunks
                if item["type"] == "text_delta"
            ),
            "hello world",
        )
        trace = next(item["trace"] for item in chunks if item["type"] == "completed")
        self.assertTrue(trace["native_stream"])
        self.assertTrue(trace["transport_cancellation_supported"])
        self.assertEqual(trace["response_id"], "stream_1")
        self.assertNotIn("stream-secret", json.dumps(trace))
        self.assertTrue(captured["body"]["stream"])
        self.assertEqual(captured["headers"]["Accept"], "text/event-stream")

    def test_native_json_stream_buffers_planner_tokens_and_requests_json(self) -> None:
        captured: dict[str, object] = {}

        def stream_transport(url, headers, payload, timeout, cancellation_token):
            captured.update(
                {
                    "url": url,
                    "headers": dict(headers),
                    "body": json.loads(payload),
                    "timeout": timeout,
                    "token": cancellation_token,
                }
            )
            return 200, [
                b'data: {"id":"json-stream-1","choices":[{"delta":{"content":"{\\"kind\\": "}}]}\n',
                b'data: {"id":"json-stream-1","choices":[{"delta":{"content":"\\"final\\"}"},"finish_reason":"stop"}]}\n',
                b"data: [DONE]\n",
            ]

        client = DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=True, max_retries=0),
            api_key="stream-secret",
            stream_transport=stream_transport,
        )
        result, trace = client.chat_json_stream(
            [{"role": "user", "content": "return json"}],
            request_kind="native_json_stream_test",
            cancellation_token=CancellationToken(),
            deadline_monotonic=time.monotonic() + 10,
        )
        self.assertEqual(result, {"kind": "final"})
        self.assertEqual(trace["response_id"], "json-stream-1")
        self.assertEqual(captured["body"]["response_format"], {"type": "json_object"})

    def test_native_web_search_stream_emits_text_tool_lifecycle_and_sources(
        self,
    ) -> None:
        captured: dict[str, object] = {}

        def stream_transport(url, headers, payload, timeout, cancellation_token):
            captured.update(
                {
                    "url": url,
                    "headers": dict(headers),
                    "body": json.loads(payload),
                    "timeout": timeout,
                    "token": cancellation_token,
                }
            )
            events = [
                {
                    "type": "message_start",
                    "message": {
                        "id": "web-stream-1",
                        "stop_reason": None,
                        "usage": {"input_tokens": 3},
                    },
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "server_tool_use",
                        "id": "srvtoolu-stream-1",
                        "name": "web_search",
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": '{"query":"private search body"}',
                    },
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu-stream-1",
                        "content": [
                            {
                                "type": "web_search_result",
                                "title": "Official update",
                                "url": "https://example.com/update?utm_source=test&v=1",
                            }
                        ],
                    },
                },
                {"type": "content_block_stop", "index": 1},
                {
                    "type": "content_block_start",
                    "index": 2,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 2,
                    "delta": {
                        "type": "thinking_delta",
                        "thinking": "private chain of thought",
                    },
                },
                {"type": "content_block_stop", "index": 2},
                {
                    "type": "content_block_start",
                    "index": 3,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 3,
                    "delta": {"type": "text_delta", "text": "fresh answer"},
                },
                {"type": "content_block_stop", "index": 3},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 2},
                },
                {"type": "message_stop"},
            ]
            wire = b": keep-alive\r\n\r\n" + _anthropic_sse(events, crlf=True)
            return 200, _fragment_bytes(wire, (1, 2, 7, 3, 19))

        client = DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=True, max_retries=0),
            api_key="stream-secret",
            stream_transport=stream_transport,
        )
        chunks = list(
            client.chat_web_stream(
                [{"role": "user", "content": "latest?"}],
                system="Use web search only when current information is required.",
                request_kind="native_web_stream_test",
                cancellation_token=CancellationToken(),
                deadline_monotonic=time.monotonic() + 10,
            )
        )
        self.assertEqual(
            captured["url"], "https://api.deepseek.com/anthropic/v1/messages"
        )
        self.assertTrue(captured["body"]["stream"])
        self.assertEqual(captured["headers"]["Accept"], "text/event-stream")
        self.assertEqual(
            [item["type"] for item in chunks if item["type"].startswith("tool_")],
            ["tool_started", "tool_progress", "tool_completed"],
        )
        completed = next(item for item in chunks if item["type"] == "completed")
        self.assertEqual(completed["result"]["message"], "fresh answer")
        self.assertTrue(completed["result"]["web_search_used"])
        self.assertEqual(
            completed["result"]["sources"],
            [{"title": "Official update", "url": "https://example.com/update?v=1"}],
        )
        self.assertNotIn("private search body", json.dumps(chunks))
        self.assertNotIn("private chain of thought", json.dumps(chunks))
        self.assertEqual(completed["trace"]["stop_reason"], "end_turn")

    def test_web_stream_validates_terminal_marker_and_stop_reason(self) -> None:
        cases = {
            "missing": (
                _anthropic_sse(_minimal_web_events(stop_reason=None)),
                "stop_reason",
            ),
            "truncated": (
                _anthropic_sse(_minimal_web_events(stop_reason="max_tokens")),
                "truncated",
            ),
            "pause": (
                _anthropic_sse(_minimal_web_events(stop_reason="pause_turn")),
                "pause_turn",
            ),
            "wrong_protocol": (
                _anthropic_sse(_minimal_web_events()[:-2]) + b"data: [DONE]\n\n",
                "invalid terminal marker",
            ),
        }
        for label, (wire, pattern) in cases.items():
            with self.subTest(label=label):
                client = DeepSeekClient(
                    DeepSeekConfig(
                        allow_remote_student_data=True,
                        max_retries=0,
                    ),
                    api_key="stream-secret",
                    stream_transport=lambda *_args, wire=wire: (200, [wire]),
                )
                with self.assertRaisesRegex(DeepSeekClientError, pattern):
                    list(
                        client.chat_web_stream(
                            [{"role": "user", "content": "latest?"}],
                            system="Search when current information is required.",
                            request_kind=f"terminal_{label}",
                            cancellation_token=CancellationToken(),
                            deadline_monotonic=time.monotonic() + 10,
                        )
                    )

    def test_web_stream_never_retries_after_server_tool_effect(self) -> None:
        calls = 0
        prefix = _anthropic_sse(
            [
                {
                    "type": "message_start",
                    "message": {
                        "id": "web-effect",
                        "stop_reason": None,
                        "usage": {},
                    },
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "server_tool_use",
                        "id": "srvtoolu-effect",
                        "name": "web_search",
                    },
                },
            ]
        )

        def stream_transport(*_args):
            nonlocal calls
            calls += 1

            def broken_stream():
                yield prefix
                raise OSError("private transport detail")

            return 200, broken_stream()

        client = DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=True, max_retries=3),
            api_key="stream-secret",
            stream_transport=stream_transport,
        )
        with self.assertRaisesRegex(DeepSeekClientError, "transport failed") as caught:
            list(
                client.chat_web_stream(
                    [{"role": "user", "content": "latest?"}],
                    system="Search when current information is required.",
                    request_kind="no_search_replay",
                    cancellation_token=CancellationToken(),
                    deadline_monotonic=time.monotonic() + 10,
                )
            )
        self.assertEqual(calls, 1)
        self.assertNotIn("private transport detail", str(caught.exception))

    def test_web_stream_maps_provider_search_error_without_leaking_body(self) -> None:
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "web-error",
                    "stop_reason": None,
                    "usage": {},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "server_tool_use",
                    "id": "srvtoolu-error",
                    "name": "web_search",
                },
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu-error",
                    "content": {
                        "type": "web_search_tool_result_error",
                        "error_code": "unavailable",
                        "private_body": "provider-internal-search-body",
                    },
                },
            },
            {"type": "content_block_stop", "index": 1},
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {
                    "type": "text_delta",
                    "text": "Search is temporarily unavailable.",
                },
            },
            {"type": "content_block_stop", "index": 2},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {},
            },
            {"type": "message_stop"},
        ]
        client = DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=True, max_retries=0),
            api_key="stream-secret",
            stream_transport=lambda *_args: (200, [_anthropic_sse(events)]),
        )
        chunks = list(
            client.chat_web_stream(
                [{"role": "user", "content": "latest?"}],
                system="Search when current information is required.",
                request_kind="search_error_lifecycle",
                cancellation_token=CancellationToken(),
                deadline_monotonic=time.monotonic() + 10,
            )
        )
        failed = next(item for item in chunks if item["type"] == "tool_failed")
        self.assertEqual(failed["error_code"], "unavailable")
        self.assertNotIn("provider-internal-search-body", json.dumps(chunks))
        completed = next(item for item in chunks if item["type"] == "completed")
        self.assertEqual(completed["result"]["sources"], [])

    def test_stream_and_nonstream_web_sources_use_the_same_sanitizer(self) -> None:
        result_source = {
            "type": "web_search_result",
            "title": "Result source",
            "url": "https://example.com/result?utm_campaign=x&keep=1",
            "encrypted_content": "private-result-body",
        }
        citation_source = {
            "title": "Citation source",
            "url": "https://example.org/citation#private-fragment",
            "cited_text": "private cited body",
        }
        content_blocks = [
            {
                "type": "server_tool_use",
                "id": "srvtoolu-parity",
                "name": "web_search",
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu-parity",
                "content": [result_source],
            },
            {
                "type": "text",
                "text": "answer",
                "citations": [citation_source],
            },
        ]
        nonstream_body = json.dumps(
            {
                "id": "nonstream-parity",
                "content": content_blocks,
                "usage": {},
                "stop_reason": "end_turn",
            }
        ).encode("utf-8")
        nonstream_client = DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=True, max_retries=0),
            api_key="stream-secret",
            transport=lambda *_args: (200, nonstream_body),
        )
        nonstream_result, _ = nonstream_client.chat_web(
            [{"role": "user", "content": "latest?"}],
            system="Search when current information is required.",
            request_kind="source_parity_nonstream",
        )

        stream_events = [
            {
                "type": "message_start",
                "message": {
                    "id": "stream-parity",
                    "stop_reason": None,
                    "usage": {},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": content_blocks[0],
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": content_blocks[1],
            },
            {"type": "content_block_stop", "index": 1},
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {
                    "type": "citations_delta",
                    "citation": citation_source,
                },
            },
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "text_delta", "text": "answer"},
            },
            {"type": "content_block_stop", "index": 2},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {},
            },
            {"type": "message_stop"},
        ]
        stream_client = DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=True, max_retries=0),
            api_key="stream-secret",
            stream_transport=lambda *_args: (
                200,
                [_anthropic_sse(stream_events)],
            ),
        )
        stream_chunks = list(
            stream_client.chat_web_stream(
                [{"role": "user", "content": "latest?"}],
                system="Search when current information is required.",
                request_kind="source_parity_stream",
                cancellation_token=CancellationToken(),
                deadline_monotonic=time.monotonic() + 10,
            )
        )
        stream_result = next(
            item["result"] for item in stream_chunks if item["type"] == "completed"
        )
        self.assertEqual(stream_result["sources"], nonstream_result["sources"])
        serialized = json.dumps(stream_result)
        self.assertNotIn("private-result-body", serialized)
        self.assertNotIn("private cited body", serialized)

    def test_default_stream_transport_closes_response_on_cancellation(self) -> None:
        class BlockingResponse:
            status = 200

            def __init__(self) -> None:
                self.iterating = Event()
                self.closed = Event()

            def __iter__(self):
                return self

            def __next__(self):
                self.iterating.set()
                self.closed.wait(2)
                raise StopIteration

            def close(self) -> None:
                self.closed.set()

        response = BlockingResponse()
        token = CancellationToken()
        with mock.patch(
            "teaching_skill_miner.deepseek_client.request.urlopen",
            return_value=response,
        ):
            status, lines = _default_stream_transport(
                "https://api.deepseek.com/anthropic/v1/messages",
                {"Content-Type": "application/json"},
                b"{}",
                10.0,
                token,
            )
        worker = Thread(target=lambda: list(lines), daemon=True)
        worker.start()
        self.assertTrue(response.iterating.wait(1))
        token.cancel("test_cancel")
        worker.join(1)
        self.assertEqual(status, 200)
        self.assertTrue(response.closed.is_set())
        self.assertFalse(worker.is_alive())

    def test_web_client_propagates_cancellation_after_socket_close(self) -> None:
        class BlockingResponse:
            status = 200

            def __init__(self) -> None:
                self.iterating = Event()
                self.closed = Event()

            def __iter__(self):
                return self

            def __next__(self):
                self.iterating.set()
                self.closed.wait(2)
                raise StopIteration

            def close(self) -> None:
                self.closed.set()

        response = BlockingResponse()
        token = CancellationToken()
        failures: list[BaseException] = []
        client = DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=True, max_retries=0),
            api_key="stream-secret",
        )

        def consume() -> None:
            try:
                list(
                    client.chat_web_stream(
                        [{"role": "user", "content": "latest?"}],
                        system="Search when current information is required.",
                        request_kind="cancel_active_web_socket",
                        cancellation_token=token,
                        deadline_monotonic=time.monotonic() + 10,
                    )
                )
            except BaseException as exc:
                failures.append(exc)

        with mock.patch(
            "teaching_skill_miner.deepseek_client.request.urlopen",
            return_value=response,
        ):
            worker = Thread(target=consume, daemon=True)
            worker.start()
            self.assertTrue(response.iterating.wait(1))
            token.cancel("user_cancelled")
            worker.join(1)
        self.assertTrue(response.closed.is_set())
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], HarnessCancelled)

    def test_stream_does_not_retry_after_content_was_emitted(self) -> None:
        calls = 0

        def truncated_stream(*_args):
            nonlocal calls
            calls += 1
            return 200, [b'data: {"choices":[{"delta":{"content":"partial"}}]}\n']

        client = DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=True, max_retries=3),
            api_key="stream-secret",
            stream_transport=truncated_stream,
        )
        with self.assertRaisesRegex(DeepSeekClientError, "terminal marker"):
            list(
                client.chat_text_stream(
                    [{"role": "user", "content": "stream"}],
                    request_kind="no_duplicate_prefix",
                    cancellation_token=CancellationToken(),
                    deadline_monotonic=time.monotonic() + 10,
                )
            )
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
