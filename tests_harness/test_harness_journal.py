from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent_harness.core.checkpoint import HarnessCheckpoint
from agent_harness.core.events import HarnessEventEmitter, canonical_json
from agent_harness.core.journal import (
    GENESIS_RECORD_HASH,
    HARNESS_JOURNAL_CHECKPOINT_SCHEMA,
    HARNESS_JOURNAL_RECORD_SCHEMA,
    HarnessJournal,
    HarnessJournalError,
    JournalCorruptionError,
    JournalLifecycleError,
)


class HarnessJournalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.path = self.root / "events.jsonl"
        self.run_id = "run-journal-test"
        self.turn_id = "turn-journal-test"
        self.emitter = HarnessEventEmitter(
            run_id=self.run_id,
            turn_id=self.turn_id,
        )

    def journal(self, path: Path | None = None) -> HarnessJournal:
        return HarnessJournal(
            path or self.path,
            run_id=self.run_id,
            turn_id=self.turn_id,
        )

    def start_event(self) -> dict[str, object]:
        return self.emitter.emit(
            "run.started", {"resumed": False, "context_sha256": "a" * 64}
        )

    def test_append_hash_chain_replay_and_terminal_durable_ack(self) -> None:
        journal = self.journal()
        start = self.start_event()
        middle = self.emitter.emit("model.started", {"attempt": 1, "step": 1})
        terminal = self.emitter.emit(
            "run.completed", {"reason_code": "done", "duration_ms": 4}
        )

        first_ack = journal.append(start)
        journal.append(middle)
        terminal_ack = journal.append(terminal)

        self.assertTrue(first_ack.durable)
        self.assertFalse(first_ack.terminal)
        self.assertTrue(terminal_ack.durable)
        self.assertTrue(terminal_ack.is_terminal)
        self.assertEqual(terminal_ack.event_type, "run.completed")
        self.assertEqual(journal.terminal_ack, terminal_ack)
        self.assertEqual(journal.terminal_type, "run.completed")
        self.assertEqual(journal.next_sequence, 4)

        records = journal.replay_records()
        self.assertEqual([item["sequence"] for item in records], [1, 2, 3])
        self.assertEqual(records[0]["schema"], HARNESS_JOURNAL_RECORD_SCHEMA)
        self.assertEqual(records[0]["previous_hash"], GENESIS_RECORD_HASH)
        self.assertEqual(records[1]["previous_hash"], records[0]["record_hash"])
        self.assertEqual(records[2]["previous_hash"], records[1]["record_hash"])
        self.assertEqual(journal.replay(after_sequence=1), [middle, terminal])
        self.assertEqual(journal.replay(after_sequence=3), [])

        extra_emitter = HarnessEventEmitter(
            run_id=self.run_id,
            turn_id=self.turn_id,
            next_sequence=4,
        )
        with self.assertRaises(JournalLifecycleError):
            journal.append(
                extra_emitter.emit("model.started", {"attempt": 1, "step": 2})
            )

    def test_journal_is_an_event_sink_and_survives_reopen(self) -> None:
        journal = self.journal()
        emitter = HarnessEventEmitter(
            run_id=self.run_id,
            turn_id=self.turn_id,
            event_sink=journal,
        )
        start = emitter.emit("run.started", {"resumed": False})
        model = emitter.emit("model.started", {"attempt": 1, "step": 1})

        reopened = self.journal()
        self.assertEqual(reopened.replay(), [start, model])
        self.assertEqual(reopened.last_sequence, 2)
        self.assertEqual(reopened.last_ack.sequence, 2)  # type: ignore[union-attr]

    def test_unchanged_journal_does_not_rescan_the_full_hash_chain(self) -> None:
        journal = self.journal()
        with patch.object(journal, "_scan", wraps=journal._scan) as scan:
            journal.append(self.start_event())
            for step in range(1, 20):
                journal.append(
                    self.emitter.emit("model.started", {"attempt": 1, "step": step})
                )
            self.assertEqual(journal.last_sequence, 20)
            self.assertEqual(len(journal.replay()), 20)
            self.assertEqual(scan.call_count, 0)

    def test_external_append_invalidates_cached_journal_state(self) -> None:
        first = self.journal()
        second = self.journal()
        first.append(self.start_event())
        second_emitter = HarnessEventEmitter(
            run_id=self.run_id,
            turn_id=self.turn_id,
            next_sequence=2,
        )
        second.append(second_emitter.emit("model.started", {"attempt": 1, "step": 1}))

        with patch.object(first, "_scan", wraps=first._scan) as scan:
            self.assertEqual(first.last_sequence, 2)
            self.assertEqual(first.last_ack.sequence, 2)  # type: ignore[union-attr]
            self.assertEqual(scan.call_count, 1)
            self.assertEqual(first.last_sequence, 2)
            self.assertEqual(scan.call_count, 1)

    def test_strict_run_turn_sequence_and_start_lifecycle(self) -> None:
        journal = self.journal()
        wrong_first = HarnessEventEmitter(
            run_id=self.run_id,
            turn_id=self.turn_id,
        ).emit("model.started", {"attempt": 1, "step": 1})
        with self.assertRaises(JournalLifecycleError):
            journal.append(wrong_first)

        start = self.start_event()
        journal.append(start)

        wrong_run = deepcopy(
            HarnessEventEmitter(
                run_id="other-run", turn_id=self.turn_id, next_sequence=2
            ).emit("model.started", {"attempt": 1, "step": 1})
        )
        with self.assertRaises(JournalLifecycleError):
            journal.append(wrong_run)

        skipped = HarnessEventEmitter(
            run_id=self.run_id, turn_id=self.turn_id, next_sequence=3
        ).emit("model.started", {"attempt": 1, "step": 1})
        with self.assertRaises(JournalLifecycleError):
            journal.append(skipped)

        non_resumed = self.emitter.emit("run.started", {"resumed": False})
        with self.assertRaises(JournalLifecycleError):
            journal.append(non_resumed)

        resumed_emitter = HarnessEventEmitter(
            run_id=self.run_id, turn_id=self.turn_id, next_sequence=2
        )
        resumed = resumed_emitter.emit("run.started", {"resumed": True})
        journal.append(resumed)
        self.assertEqual(journal.last_sequence, 2)

    def test_torn_final_line_is_truncated_and_append_can_continue(self) -> None:
        journal = self.journal()
        start = self.start_event()
        journal.append(start)
        good_size = self.path.stat().st_size
        with self.path.open("ab") as stream:
            stream.write(b'{"schema":"torn"')
            stream.flush()
            os.fsync(stream.fileno())

        recovered = self.journal()
        self.assertEqual(recovered.replay(), [start])
        self.assertEqual(self.path.stat().st_size, good_size)
        middle = self.emitter.emit("model.started", {"attempt": 1, "step": 1})
        recovered.append(middle)
        self.assertEqual(recovered.replay(), [start, middle])

    def test_only_the_last_malformed_line_may_be_recovered(self) -> None:
        journal = self.journal()
        journal.append(self.start_event())
        original = self.path.read_bytes()
        self.path.write_bytes(original + b"not-json\n" + original)
        corrupt_size = self.path.stat().st_size

        with self.assertRaises(JournalCorruptionError):
            self.journal()
        self.assertEqual(self.path.stat().st_size, corrupt_size)

    def test_newline_terminated_malformed_tail_fails_closed(self) -> None:
        journal = self.journal()
        journal.append(self.start_event())
        with self.path.open("ab") as stream:
            stream.write(b'{"schema":"not-a-complete-record"\n')
            stream.flush()
            os.fsync(stream.fileno())
        corrupt_size = self.path.stat().st_size

        with self.assertRaises(JournalCorruptionError):
            self.journal()
        self.assertEqual(self.path.stat().st_size, corrupt_size)

    def test_valid_json_with_bad_hash_fails_closed_even_at_tail(self) -> None:
        journal = self.journal()
        journal.append(self.start_event())
        records = journal.replay_records()
        records[0]["event"]["payload"]["resumed"] = True
        tampered = canonical_json(records[0]).encode("utf-8") + b"\n"
        self.path.write_bytes(tampered)
        corrupt_size = self.path.stat().st_size

        with self.assertRaises(JournalCorruptionError):
            self.journal()
        self.assertEqual(self.path.stat().st_size, corrupt_size)

    def test_valid_but_noncanonical_tail_fails_closed(self) -> None:
        journal = self.journal()
        journal.append(self.start_event())
        record = journal.replay_records()[0]
        noncanonical = json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n"
        self.path.write_bytes(noncanonical)
        corrupt_size = self.path.stat().st_size

        with self.assertRaises(JournalCorruptionError):
            self.journal()
        self.assertEqual(self.path.stat().st_size, corrupt_size)

    def test_complete_final_record_without_newline_is_preserved(self) -> None:
        journal = self.journal()
        start = self.start_event()
        journal.append(start)
        self.path.write_bytes(self.path.read_bytes().removesuffix(b"\n"))

        recovered = self.journal()
        self.assertEqual(recovered.replay(), [start])
        self.assertTrue(self.path.read_bytes().endswith(b"\n"))

    def test_atomic_checkpoint_round_trip_and_old_anchor_replay(self) -> None:
        journal = self.journal()
        start = self.start_event()
        journal.append(start)
        snapshot = {
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "status": "running",
            "next_sequence": 2,
            "next_step": 1,
        }
        acknowledgement = journal.write_checkpoint(snapshot)
        self.assertTrue(acknowledgement.durable)
        self.assertEqual(acknowledgement.sequence, 1)
        self.assertEqual(journal.load_checkpoint(), snapshot)

        envelope = json.loads(journal.checkpoint_path.read_text("utf-8"))
        self.assertEqual(envelope["schema"], HARNESS_JOURNAL_CHECKPOINT_SCHEMA)
        self.assertEqual(
            envelope["record_hash"], journal.replay_records()[0]["record_hash"]
        )
        self.assertFalse(list(self.root.glob(f".{journal.checkpoint_path.name}.*.tmp")))

        later = self.emitter.emit("model.started", {"attempt": 1, "step": 1})
        journal.append(later)
        self.assertEqual(journal.load_checkpoint(), snapshot)
        self.assertEqual(journal.replay(after_sequence=1), [later])

    def test_checkpoint_accepts_harness_checkpoint_object(self) -> None:
        journal = self.journal()
        journal.append(self.start_event())
        checkpoint = HarnessCheckpoint.create(
            run_id=self.run_id,
            turn_id=self.turn_id,
            status="running",
            next_sequence=2,
            next_step=1,
            model_calls=0,
            tool_calls=0,
            context_sha256="a" * 64,
            policy_sha256="b" * 64,
            external_effect_started=False,
            repeated_calls={},
            observations=(),
            idempotency_receipts={},
            completed_call_receipts={},
            pending_effect=None,
        )
        journal.save_checkpoint(checkpoint)
        self.assertEqual(journal.load_checkpoint(), checkpoint.to_dict())

    def test_checkpoint_identity_and_cursor_are_strict(self) -> None:
        journal = self.journal()
        with self.assertRaises(HarnessJournalError):
            journal.write_checkpoint({"next_sequence": 1})
        journal.append(self.start_event())
        with self.assertRaises(HarnessJournalError):
            journal.write_checkpoint(
                {"run_id": "other", "turn_id": self.turn_id, "next_sequence": 2}
            )
        with self.assertRaises(HarnessJournalError):
            journal.write_checkpoint(
                {"run_id": self.run_id, "turn_id": self.turn_id, "next_sequence": 3}
            )

    def test_checkpoint_tampering_fails_closed(self) -> None:
        journal = self.journal()
        journal.append(self.start_event())
        journal.write_checkpoint({"next_sequence": 2, "value": "safe"})
        envelope = json.loads(journal.checkpoint_path.read_text("utf-8"))
        envelope["snapshot"]["value"] = "tampered"
        journal.checkpoint_path.write_text(
            canonical_json(envelope) + "\n", encoding="utf-8"
        )
        with self.assertRaises(JournalCorruptionError):
            journal.load_checkpoint()

    def test_directory_and_symlink_targets_are_rejected(self) -> None:
        directory = self.root / "journal-directory"
        directory.mkdir()
        with self.assertRaises(HarnessJournalError):
            self.journal(directory)

        real_file = self.root / "real.jsonl"
        real_file.write_text("", encoding="utf-8")
        symlink = self.root / "journal-link.jsonl"
        try:
            symlink.symlink_to(real_file)
        except (NotImplementedError, OSError):
            self.skipTest("symbolic links are not supported on this platform")
        with self.assertRaises(HarnessJournalError):
            self.journal(symlink)

        checkpoint_directory = self.root / "checkpoint-directory"
        checkpoint_directory.mkdir()
        with self.assertRaises(HarnessJournalError):
            HarnessJournal(
                self.root / "separate.jsonl",
                run_id=self.run_id,
                turn_id=self.turn_id,
                checkpoint_path=checkpoint_directory,
            )

    def test_terminal_ack_is_not_returned_when_fsync_fails(self) -> None:
        journal = self.journal()
        start = self.start_event()
        first_ack = journal.append(start)
        terminal = self.emitter.emit("run.completed", {"reason_code": "done"})

        with patch(
            "agent_harness.core.journal.os.fsync",
            side_effect=OSError("injected fsync failure"),
        ):
            with self.assertRaises(OSError):
                journal.append(terminal)

        self.assertEqual(journal.last_ack, first_ack)
        self.assertIsNone(journal.terminal_ack)
        self.assertTrue(journal.durability_uncertain)
        with self.assertRaisesRegex(HarnessJournalError, "durability is uncertain"):
            journal.replay()
        with self.assertRaisesRegex(JournalLifecycleError, "failed durability"):
            journal.append(terminal)

        # A fresh instance validates and fsyncs the truncated acknowledged
        # prefix before making it replayable again.
        recovered = self.journal()
        self.assertEqual(recovered.replay(), [start])
        self.assertEqual(recovered.next_sequence, 2)

    def test_transient_fsync_failure_rolls_back_then_retries_exact_event(self) -> None:
        journal = self.journal()
        start = self.start_event()
        journal.append(start)
        acknowledged_bytes = self.path.read_bytes()
        terminal = self.emitter.emit("run.completed", {"reason_code": "done"})
        real_fsync = os.fsync
        failed_once = False

        def fail_append_then_fence_rollback(file_descriptor: int) -> None:
            nonlocal failed_once
            if not failed_once:
                failed_once = True
                raise OSError("injected append fsync failure")
            real_fsync(file_descriptor)

        with patch(
            "agent_harness.core.journal.os.fsync",
            side_effect=fail_append_then_fence_rollback,
        ):
            acknowledgement = journal.append(terminal)

        self.assertFalse(journal.durability_uncertain)
        self.assertNotEqual(self.path.read_bytes(), acknowledged_bytes)
        self.assertTrue(acknowledgement.terminal)
        self.assertEqual(journal.replay(), [start, terminal])
        self.assertEqual(journal.next_sequence, 3)

    def test_after_sequence_rejects_ambiguous_values(self) -> None:
        journal = self.journal()
        for value in (-1, True, 1.5, "1"):
            with self.subTest(value=value):
                with self.assertRaises(HarnessJournalError):
                    journal.replay(after_sequence=value)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
