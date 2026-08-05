from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import tempfile
import unittest

from teaching_skill_miner.deepseek_client import (
    DeepSeekClient,
    DeepSeekClientError,
    DeepSeekConfig,
    DeepSeekConfigurationError,
)


def _envelope(content: dict, *, response_id: str = "response_test") -> bytes:
    return json.dumps(
        {
            "id": response_id,
            "choices": [{"message": {"content": json.dumps(content)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
    ).encode()


class DeepSeekClientTests(unittest.TestCase):
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
                {"url": url, "authorization": headers["Authorization"], "payload": payload, "timeout": timeout}
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
                    _envelope({"authorized": headers["Authorization"] == "Bearer file-secret-key"}),
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


if __name__ == "__main__":
    unittest.main()
