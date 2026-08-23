from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from agent_harness.core import HarnessCheckpoint
from agent_harness.core.contracts import HarnessContractError
from agent_harness.core.events import HarnessEventEmitter, canonical_sha256
from agent_harness.core.journal import HarnessJournal


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = ROOT / "schema"


def _schema(name: str) -> dict[str, object]:
    return json.loads((SCHEMA_ROOT / name).read_text(encoding="utf-8"))


class HarnessSchemaTests(unittest.TestCase):
    def _validate(self, name: str, value: object) -> None:
        schema = _schema(name)
        Draft202012Validator.check_schema(schema)
        resources = []
        for path in SCHEMA_ROOT.glob("*.schema.json"):
            candidate = json.loads(path.read_text(encoding="utf-8"))
            identifier = candidate.get("$id")
            if isinstance(identifier, str):
                resources.append((identifier, Resource.from_contents(candidate)))
        registry = Registry().with_resources(resources)
        Draft202012Validator(schema, registry=registry).validate(value)

    def test_authoritative_event_and_checkpoint_match_public_schemas(self) -> None:
        emitter = HarnessEventEmitter(run_id="run_schema", turn_id="turn_schema")
        event = emitter.emit("run.started", {"resumed": False})
        self._validate("agent_harness_event.schema.json", event)

        checkpoint = HarnessCheckpoint.create(
            run_id="run_schema",
            turn_id="turn_schema",
            status="running",
            next_sequence=2,
            next_step=1,
            model_calls=0,
            tool_calls=0,
            context_sha256=canonical_sha256({"task": "schema"}),
            policy_sha256=canonical_sha256({"policy": "schema"}),
            external_effect_started=False,
            repeated_calls={},
            observations=(),
            idempotency_receipts={},
            completed_call_receipts={},
            pending_effect=None,
        )
        self._validate("agent_harness_checkpoint.schema.json", checkpoint.to_dict())

    def test_journal_record_and_checkpoint_envelope_match_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.jsonl"
            journal = HarnessJournal(path, run_id="run_schema", turn_id="turn_schema")
            emitter = HarnessEventEmitter(run_id="run_schema", turn_id="turn_schema")
            journal.append(emitter.emit("run.started", {"resumed": False}))
            snapshot = {
                "run_id": "run_schema",
                "turn_id": "turn_schema",
                "next_sequence": 2,
                "status": "running",
            }
            journal.write_checkpoint(snapshot)
            record = json.loads(path.read_text(encoding="utf-8"))
            envelope = json.loads(journal.checkpoint_path.read_text(encoding="utf-8"))
            self._validate("harness_journal_record.schema.json", record)
            self._validate("harness_journal_checkpoint.schema.json", envelope)

    def test_event_payload_contract_fails_before_sequence_or_journal_advance(
        self,
    ) -> None:
        emitter = HarnessEventEmitter(run_id="run_invalid", turn_id="turn_invalid")
        invalid_events = (
            ("message.delta", {}),
            ("tool.started", {"tool_name": "retrieve_resources", "attempt": 1}),
            (
                "tool.completed",
                {
                    "call_id": "call_1",
                    "tool_name": "retrieve_resources",
                    "attempt": 1,
                    "result": {"ok": True},
                    "result_sha256": "0" * 64,
                },
            ),
            ("state.committed", {"session_id": "teach_1"}),
        )

        for event_type, payload in invalid_events:
            with self.subTest(event_type=event_type):
                with self.assertRaises(HarnessContractError):
                    emitter.emit(event_type, payload)
                self.assertEqual(emitter.next_sequence, 1)
                self.assertEqual(emitter.events, [])


if __name__ == "__main__":
    unittest.main()
