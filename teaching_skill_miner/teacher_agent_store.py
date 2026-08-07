"""Integrity-protected opt-in rollout storage for Teaching Agent sessions.

The dashboard remains in-memory by default.  When a caller explicitly provides a
store path, this module appends canonical JSON events to one local JSONL file and
flushes them before the caller publishes the corresponding in-memory state.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import threading
from typing import Any, Mapping, Sequence

try:  # ``flock`` gives the opt-in store a process, not merely thread, fence.
    import fcntl
except ImportError:  # pragma: no cover - Teaching Agent durable mode is POSIX-only.
    fcntl = None  # type: ignore[assignment]


EVENT_SCHEMA = "teaching_skill_miner.teacher_agent_rollout_event.v1"
ALLOWED_EVENT_TYPES = frozenset(
    {
        "session_started",
        "turn_started",
        "turn_committed",
        "turn_aborted",
        "context_checkpoint",
        "session_stopped",
    }
)
_HASH_HEX_LENGTH = 64


class TeacherAgentStoreError(RuntimeError):
    """Raised when durable Teaching Agent state cannot be trusted or flushed."""


@dataclass(frozen=True, slots=True)
class TeacherAgentStoreRecovery:
    """Validated state reconstructed from the latest per-session checkpoints."""

    session_records: dict[str, dict[str, Any]]
    start_idempotency_cache: dict[str, dict[str, Any]]
    dangling_turns: tuple[dict[str, Any], ...]
    last_touched_session_id: str | None


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TeacherAgentStoreError(
            "teacher Agent rollout events must contain canonical JSON"
        ) from exc


def _event_hash(event_without_hash: Mapping[str, Any]) -> str:
    return sha256(_canonical_bytes(event_without_hash)).hexdigest()


def _nonnegative_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TeacherAgentStoreError(
            f"teacher Agent rollout {field_name} must be a non-negative integer"
        )
    return value


def _required_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TeacherAgentStoreError(
            f"teacher Agent rollout {field_name} must be a non-empty string"
        )
    return value


class TeacherAgentStore:
    """Single-writer append-only JSONL store with a SHA-256 hash chain."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._last_hash: str | None = None
        self._next_seq = 1
        self._file_size = 0
        self._poisoned = False
        self._load_and_repair_truncated_tail()

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        """Return a defensive snapshot of all verified events."""

        with self._lock:
            return tuple(deepcopy(self._events))

    def _load_and_repair_truncated_tail(self) -> None:
        if not self.path.exists():
            return
        try:
            file_stat = self.path.lstat()
        except OSError as exc:
            raise TeacherAgentStoreError(
                "teacher Agent rollout store metadata cannot be read"
            ) from exc
        if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
            raise TeacherAgentStoreError(
                "teacher Agent rollout store must be a regular non-symlink file"
            )
        try:
            stream = self.path.open("r+b")
        except OSError as exc:
            raise TeacherAgentStoreError(
                "teacher Agent rollout store cannot be read"
            ) from exc
        try:
            self._lock_stream(stream)
            raw = stream.read()
            final_newline = raw.rfind(b"\n")
            valid_size = final_newline + 1 if final_newline >= 0 else 0
            complete = raw[:valid_size]
            previous_hash: str | None = None
            expected_seq = 1
            events: list[dict[str, Any]] = []
            for line_number, raw_line in enumerate(complete.splitlines(), 1):
                if not raw_line:
                    raise TeacherAgentStoreError(
                        f"teacher Agent rollout line {line_number} is empty"
                    )
                try:
                    event = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise TeacherAgentStoreError(
                        f"teacher Agent rollout line {line_number} is invalid JSON"
                    ) from exc
                if not isinstance(event, dict):
                    raise TeacherAgentStoreError(
                        f"teacher Agent rollout line {line_number} is not an object"
                    )
                self._validate_loaded_event(
                    event,
                    expected_seq=expected_seq,
                    previous_hash=previous_hash,
                    line_number=line_number,
                )
                events.append(event)
                previous_hash = event["hash"]
                expected_seq += 1
            self._validate_event_lifecycle(events)
            if valid_size != len(raw):
                stream.seek(valid_size)
                stream.truncate(valid_size)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise TeacherAgentStoreError(
                "teacher Agent rollout truncated tail cannot be repaired"
            ) from exc
        finally:
            self._unlock_stream(stream)
            stream.close()
        self._events = events
        self._last_hash = previous_hash
        self._next_seq = expected_seq
        self._file_size = valid_size

    @staticmethod
    def _require_process_locking() -> None:
        """Fail closed if this host cannot provide a process-level store fence.

        The store is deliberately an opt-in durability feature.  A Python
        ``threading.Lock`` protects only one dashboard process; without an OS
        lock two dashboard processes can both append a valid next sequence
        number and corrupt the event log.  Do not silently downgrade that
        guarantee on unsupported hosts.
        """

        if fcntl is None:
            raise TeacherAgentStoreError(
                "teacher Agent durable store requires POSIX process locking"
            )

    @classmethod
    def _lock_stream(cls, stream: Any) -> None:
        cls._require_process_locking()
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            raise TeacherAgentStoreError(
                "teacher Agent rollout store process lock cannot be acquired"
            ) from exc

    @staticmethod
    def _unlock_stream(stream: Any) -> None:
        if fcntl is None:
            return
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OSError:
            # The descriptor is about to close.  A failed unlock cannot make a
            # successfully flushed append less durable and should not mask its
            # original outcome.
            pass

    @staticmethod
    def _validate_event_lifecycle(events: Sequence[Mapping[str, Any]]) -> None:
        """Reject hash-valid but semantically impossible event histories.

        A hash chain detects accidental edits, but it cannot by itself express
        the Agent's state machine.  In particular, an old ``turn_committed``
        must never resurrect a session after profile replacement.  This
        lightweight replay keeps the event log authoritative for recovery
        without persisting any model secrets or raw remote request payloads.
        """

        states: dict[str, dict[str, Any]] = {}
        for event in events:
            session_id = str(event["session_id"])
            event_type = str(event["event_type"])
            state = states.setdefault(
                session_id,
                {
                    "started": False,
                    "retired": False,
                    "terminal": False,
                    "active_turn_ids": set(),
                    "seen_turn_ids": set(),
                },
            )
            turn_id = event.get("turn_id")

            if event_type == "session_started":
                if state["started"]:
                    raise TeacherAgentStoreError(
                        "teacher Agent rollout session_started is duplicated"
                    )
                state["started"] = True
                continue

            if not state["started"]:
                raise TeacherAgentStoreError(
                    "teacher Agent rollout event precedes session_started"
                )

            # A replacement/eviction is final.  The sole permitted trailing
            # event is an abort of a turn which began *before* replacement;
            # this is how an in-flight remote model call is safely receipted.
            if state["retired"]:
                if (
                    event_type == "turn_aborted"
                    and isinstance(turn_id, str)
                    and turn_id in state["active_turn_ids"]
                ):
                    state["active_turn_ids"].remove(turn_id)
                    continue
                raise TeacherAgentStoreError(
                    "teacher Agent rollout event follows a removed session"
                )

            if event_type == "turn_started":
                if state["terminal"]:
                    raise TeacherAgentStoreError(
                        "teacher Agent rollout turn starts after session stop"
                    )
                if not isinstance(turn_id, str):
                    raise TeacherAgentStoreError(
                        "teacher Agent rollout turn_started lacks turn_id"
                    )
                if turn_id in state["seen_turn_ids"]:
                    raise TeacherAgentStoreError(
                        "teacher Agent rollout turn_id is reused"
                    )
                state["seen_turn_ids"].add(turn_id)
                state["active_turn_ids"].add(turn_id)
                continue

            if event_type in {"turn_committed", "turn_aborted"}:
                if not isinstance(turn_id, str) or turn_id not in state[
                    "active_turn_ids"
                ]:
                    raise TeacherAgentStoreError(
                        "teacher Agent rollout terminal turn event has no active turn"
                    )
                if event_type == "turn_committed" and state["terminal"]:
                    raise TeacherAgentStoreError(
                        "teacher Agent rollout turn commits after session stop"
                    )
                state["active_turn_ids"].remove(turn_id)
                continue

            if event_type == "session_stopped":
                state["terminal"] = True
                if event.get("data", {}).get("remove_session") is True:
                    state["retired"] = True
                continue

            if event_type == "context_checkpoint":
                # Checkpoints may follow a normal commit, a stop receipt, or
                # an interrupted-turn recovery.  They never change liveness.
                continue

            raise TeacherAgentStoreError(
                "teacher Agent rollout event lifecycle is unsupported"
            )

    def _validate_loaded_event(
        self,
        event: Mapping[str, Any],
        *,
        expected_seq: int,
        previous_hash: str | None,
        line_number: int,
    ) -> None:
        if event.get("schema") != EVENT_SCHEMA:
            raise TeacherAgentStoreError(
                f"teacher Agent rollout line {line_number} has an unsupported schema"
            )
        if event.get("event_type") not in ALLOWED_EVENT_TYPES:
            raise TeacherAgentStoreError(
                f"teacher Agent rollout line {line_number} has an unsupported event type"
            )
        if event.get("seq") != expected_seq:
            raise TeacherAgentStoreError(
                f"teacher Agent rollout line {line_number} has a non-contiguous seq"
            )
        if event.get("previous_hash") != previous_hash:
            raise TeacherAgentStoreError(
                f"teacher Agent rollout line {line_number} breaks the hash chain"
            )
        claimed_hash = event.get("hash")
        if not isinstance(claimed_hash, str) or len(claimed_hash) != _HASH_HEX_LENGTH:
            raise TeacherAgentStoreError(
                f"teacher Agent rollout line {line_number} has an invalid hash"
            )
        material = dict(event)
        material.pop("hash", None)
        if claimed_hash != _event_hash(material):
            raise TeacherAgentStoreError(
                f"teacher Agent rollout line {line_number} failed integrity validation"
            )
        _required_string(event.get("session_id"), field_name="session_id")
        _nonnegative_integer(event.get("round"), field_name="round")
        _nonnegative_integer(event.get("context_version"), field_name="context_version")
        _required_string(event.get("profile_revision"), field_name="profile_revision")
        question_id = event.get("question_id")
        if question_id is not None:
            _required_string(question_id, field_name="question_id")
        for field_name in (
            "idempotency_key",
            "request_fingerprint",
            "turn_id",
        ):
            value = event.get(field_name)
            if value is not None:
                _required_string(value, field_name=field_name)
        if not isinstance(event.get("data"), Mapping):
            raise TeacherAgentStoreError(
                f"teacher Agent rollout line {line_number} data must be an object"
            )

    def _build_event(
        self,
        specification: Mapping[str, Any],
        *,
        seq: int,
        previous_hash: str | None,
    ) -> dict[str, Any]:
        event_type = specification.get("event_type")
        if event_type not in ALLOWED_EVENT_TYPES:
            raise TeacherAgentStoreError(
                "teacher Agent rollout event type is unsupported"
            )
        session_id = _required_string(
            specification.get("session_id"), field_name="session_id"
        )
        round_number = _nonnegative_integer(
            specification.get("round"), field_name="round"
        )
        context_version = _nonnegative_integer(
            specification.get("context_version"), field_name="context_version"
        )
        profile_revision = _required_string(
            specification.get("profile_revision"), field_name="profile_revision"
        )
        question_id = specification.get("question_id")
        if question_id is not None:
            question_id = _required_string(question_id, field_name="question_id")
        optional_strings: dict[str, str | None] = {}
        for field_name in (
            "idempotency_key",
            "request_fingerprint",
            "turn_id",
        ):
            value = specification.get(field_name)
            optional_strings[field_name] = (
                None
                if value is None
                else _required_string(value, field_name=field_name)
            )
        data = specification.get("data", {})
        if not isinstance(data, Mapping):
            raise TeacherAgentStoreError(
                "teacher Agent rollout event data must be an object"
            )
        event: dict[str, Any] = {
            "schema": EVENT_SCHEMA,
            "seq": seq,
            "event_type": event_type,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "previous_hash": previous_hash,
            "session_id": session_id,
            "round": round_number,
            "question_id": question_id,
            "context_version": context_version,
            "profile_revision": profile_revision,
            **optional_strings,
            "data": deepcopy(dict(data)),
        }
        event["hash"] = _event_hash(event)
        return event

    def _flush(self, stream: Any) -> None:
        """Flush one append barrier; split out so failure paths are testable."""

        stream.flush()
        os.fsync(stream.fileno())

    @staticmethod
    def _rollback(stream: Any, original_size: int) -> None:
        stream.seek(original_size)
        stream.truncate(original_size)
        stream.flush()
        os.fsync(stream.fileno())

    def append_batch(
        self, specifications: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[str, Any], ...]:
        """Append and fsync one event batch before publishing its sequence state."""

        if not specifications:
            return ()
        with self._lock:
            if self._poisoned:
                raise TeacherAgentStoreError(
                    "teacher Agent rollout store is unavailable after a failed barrier"
                )
            events: list[dict[str, Any]] = []
            previous_hash = self._last_hash
            seq = self._next_seq
            for specification in specifications:
                event = self._build_event(
                    specification, seq=seq, previous_hash=previous_hash
                )
                events.append(event)
                previous_hash = event["hash"]
                seq += 1
            payload = b"".join(_canonical_bytes(event) + b"\n" for event in events)
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise TeacherAgentStoreError(
                    "teacher Agent rollout store directory cannot be created"
                ) from exc
            last_error: OSError | None = None
            for _attempt in range(2):
                try:
                    descriptor = os.open(
                        self.path,
                        os.O_RDWR | os.O_CREAT,
                        0o600,
                    )
                    with os.fdopen(descriptor, "r+b") as stream:
                        self._lock_stream(stream)
                        try:
                            stream.seek(0, os.SEEK_END)
                            original_size = stream.tell()
                            if original_size != self._file_size:
                                raise TeacherAgentStoreError(
                                    "teacher Agent rollout store has another active writer"
                                )
                            self._validate_event_lifecycle([*self._events, *events])
                            try:
                                stream.write(payload)
                                self._flush(stream)
                            except OSError as exc:
                                last_error = exc
                                try:
                                    self._rollback(stream, original_size)
                                except OSError as rollback_exc:
                                    self._poisoned = True
                                    raise TeacherAgentStoreError(
                                        "teacher Agent rollout barrier and rollback both failed"
                                    ) from rollback_exc
                                continue
                        finally:
                            self._unlock_stream(stream)
                    self._events.extend(events)
                    self._last_hash = previous_hash
                    self._next_seq = seq
                    self._file_size += len(payload)
                    return tuple(deepcopy(events))
                except TeacherAgentStoreError:
                    raise
                except OSError as exc:
                    last_error = exc
                    continue
            raise TeacherAgentStoreError(
                "teacher Agent rollout flush failed after one reopen retry"
            ) from last_error

    def recover(self) -> TeacherAgentStoreRecovery:
        """Recover each session from its latest checkpoint and replay its tail."""

        with self._lock:
            events = deepcopy(self._events)
        by_session: dict[str, list[dict[str, Any]]] = {}
        start_cache: dict[str, dict[str, Any]] = {}
        pending_turns: dict[tuple[str, str], dict[str, Any]] = {}
        last_active_order: dict[str, int] = {}
        for index, event in enumerate(events):
            session_id = event["session_id"]
            by_session.setdefault(session_id, []).append(event)
            if event["event_type"] == "session_started":
                cache_entry = event["data"].get("start_cache_entry")
                key = event.get("idempotency_key")
                if isinstance(cache_entry, Mapping) and isinstance(key, str):
                    start_cache.pop(key, None)
                    start_cache[key] = deepcopy(dict(cache_entry))
            turn_id = event.get("turn_id")
            if isinstance(turn_id, str):
                turn_key = (session_id, turn_id)
                if event["event_type"] == "turn_started":
                    pending_turns[turn_key] = deepcopy(event)
                elif event["event_type"] in {"turn_committed", "turn_aborted"}:
                    pending_turns.pop(turn_key, None)
            if event["event_type"] != "session_stopped" or not event["data"].get(
                "remove_session"
            ):
                last_active_order[session_id] = index

        recovered: dict[str, dict[str, Any]] = {}
        for session_id, session_events in by_session.items():
            checkpoint_index = -1
            for index, event in enumerate(session_events):
                if event["event_type"] == "context_checkpoint" and isinstance(
                    event["data"].get("record"), Mapping
                ):
                    checkpoint_index = index
            record: dict[str, Any] | None = None
            replay_from = 0
            if checkpoint_index >= 0:
                record = deepcopy(
                    dict(session_events[checkpoint_index]["data"]["record"])
                )
                replay_from = checkpoint_index + 1
            for event in session_events[replay_from:]:
                event_type = event["event_type"]
                event_record = event["data"].get("record")
                if event_type in {
                    "session_started",
                    "turn_committed",
                    "context_checkpoint",
                } and isinstance(event_record, Mapping):
                    record = deepcopy(dict(event_record))
                elif event_type == "session_stopped":
                    if event["data"].get("remove_session") is True:
                        record = None
                    elif isinstance(event_record, Mapping):
                        record = deepcopy(dict(event_record))
            if record is not None:
                recovered[session_id] = record

        last_touched_session_id = None
        if recovered:
            last_touched_session_id = max(
                recovered,
                key=lambda session_id: last_active_order.get(session_id, -1),
            )
        return TeacherAgentStoreRecovery(
            session_records=recovered,
            start_idempotency_cache=start_cache,
            dangling_turns=tuple(
                deepcopy(event)
                for _key, event in sorted(
                    pending_turns.items(), key=lambda item: item[1]["seq"]
                )
            ),
            last_touched_session_id=last_touched_session_id,
        )
