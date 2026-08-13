"""Crash-safe, tamper-evident persistence for harness events.

The journal is deliberately independent from the teacher-agent store.  It is
an append-only JSONL log whose records form a SHA-256 chain.  An append is
acknowledged only after the record has been flushed and ``fsync``-ed.

Recovery is intentionally conservative: a syntactically torn *last* line is
discarded, while corruption anywhere else (including a well-formed record
with a bad hash) fails closed.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
from hashlib import sha256
from hmac import compare_digest
import json
import os
from pathlib import Path
import stat
import tempfile
from threading import RLock
from typing import Any, Iterator, Mapping

from .contracts import (
    HARNESS_EVENT_SCHEMA,
    HarnessContractError,
    TERMINAL_EVENT_TYPES,
)
from .events import canonical_json, canonical_sha256

try:  # pragma: no cover - exercised on POSIX, absent on Windows.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


HARNESS_JOURNAL_RECORD_SCHEMA = "teaching_skill_miner.harness_journal_record.v1"
HARNESS_JOURNAL_CHECKPOINT_SCHEMA = "teaching_skill_miner.harness_journal_checkpoint.v1"
GENESIS_RECORD_HASH = "0" * 64

_RECORD_FIELDS = frozenset(
    {
        "schema",
        "run_id",
        "turn_id",
        "sequence",
        "previous_hash",
        "event",
        "record_hash",
    }
)
_EVENT_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "event_id",
        "run_id",
        "turn_id",
        "sequence",
        "type",
        "timestamp",
        "payload",
    }
)
_EVENT_OPTIONAL_FIELDS = frozenset({"causation_id"})
_CHECKPOINT_FIELDS = frozenset(
    {
        "schema",
        "run_id",
        "turn_id",
        "sequence",
        "record_hash",
        "snapshot",
        "checkpoint_sha256",
    }
)


class HarnessJournalError(HarnessContractError):
    """Base class for journal persistence and validation failures."""


class JournalCorruptionError(HarnessJournalError):
    """Raised when durable journal or checkpoint data cannot be trusted."""


class JournalLifecycleError(HarnessJournalError):
    """Raised when a new event violates the run/turn lifecycle."""


class _MalformedJSONError(Exception):
    """Distinguish a torn JSON encoding from valid-but-corrupt data."""


@dataclass(frozen=True, slots=True)
class DurableEventAck:
    """Proof returned only after an appended event crosses the fsync boundary."""

    run_id: str
    turn_id: str
    sequence: int
    event_id: str
    event_type: str
    record_hash: str
    terminal: bool
    durable: bool = True

    @property
    def is_terminal(self) -> bool:
        return self.terminal


@dataclass(frozen=True, slots=True)
class CheckpointWriteAck:
    """Proof returned after an atomic checkpoint replacement is durable."""

    run_id: str
    turn_id: str
    sequence: int
    record_hash: str
    checkpoint_sha256: str
    durable: bool = True


@dataclass(slots=True)
class _JournalState:
    records: list[dict[str, Any]]
    head_hash: str = GENESIS_RECORD_HASH
    terminal_type: str | None = None

    @property
    def last_sequence(self) -> int:
        return len(self.records)

    @property
    def next_sequence(self) -> int:
        return self.last_sequence + 1


@dataclass(frozen=True, slots=True)
class _FileSignature:
    """Metadata used to detect writes made by another journal instance.

    The journal still takes an advisory file lock before trusting this value.
    This is only a fast path that avoids validating the whole hash chain again
    when the same instance has exclusive ownership of an unchanged file.
    """

    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


def _file_signature(file_descriptor: int) -> _FileSignature:
    file_stat = os.fstat(file_descriptor)
    return _FileSignature(
        device=file_stat.st_dev,
        inode=file_stat.st_ino,
        size=file_stat.st_size,
        modified_ns=file_stat.st_mtime_ns,
        changed_ns=file_stat.st_ctime_ns,
    )


def _json_constant_is_invalid(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _decode_json_object(raw: bytes, *, description: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
        value = json.loads(text, parse_constant=_json_constant_is_invalid)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise _MalformedJSONError(f"{description} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise JournalCorruptionError(f"{description} must be a JSON object")
    try:
        rendered = canonical_json(value).encode("utf-8")
    except HarnessContractError as exc:
        raise JournalCorruptionError(f"{description} is not canonical JSON") from exc
    if rendered != raw:
        raise JournalCorruptionError(f"{description} is not canonically encoded")
    return value


def _path_from_value(value: os.PathLike[str] | str, *, label: str) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise HarnessJournalError(f"{label} must be a filesystem path") from exc
    if isinstance(raw, bytes):
        raw = os.fsdecode(raw)
    if not raw or "\x00" in raw:
        raise HarnessJournalError(f"{label} is invalid")
    return Path(os.path.abspath(raw))


def _prepare_parent(path: Path, *, label: str) -> None:
    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HarnessJournalError(f"cannot create {label} parent directory") from exc
    try:
        parent_stat = parent.stat()
    except OSError as exc:
        raise HarnessJournalError(f"cannot inspect {label} parent directory") from exc
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise HarnessJournalError(f"{label} parent is not a directory")


def _reject_non_regular_target(path: Path, *, label: str) -> bool:
    """Validate an existing target and return whether it already exists."""

    try:
        target_stat = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise HarnessJournalError(f"cannot inspect {label}") from exc
    if stat.S_ISLNK(target_stat.st_mode):
        raise HarnessJournalError(f"{label} cannot be a symbolic link")
    if not stat.S_ISREG(target_stat.st_mode):
        raise HarnessJournalError(f"{label} must be a regular file")
    return True


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    directory_fd = os.open(directory, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_all(file_descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    written = 0
    while written < len(view):
        count = os.write(file_descriptor, view[written:])
        if count <= 0:
            raise OSError(errno.EIO, "journal write made no progress")
        written += count


def _read_all(file_descriptor: int) -> bytes:
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(file_descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _validate_timestamp(value: Any, *, error_type: type[HarnessJournalError]) -> None:
    if not isinstance(value, str) or not value or len(value) > 80:
        raise error_type("event timestamp is invalid")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise error_type("event timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise error_type("event timestamp must include a UTC offset")


def _validate_event(
    event: Mapping[str, Any],
    *,
    run_id: str,
    turn_id: str,
    expected_sequence: int,
    terminal_type: str | None,
    error_type: type[HarnessJournalError],
) -> str:
    keys = frozenset(event)
    if not _EVENT_REQUIRED_FIELDS.issubset(keys) or not keys.issubset(
        _EVENT_REQUIRED_FIELDS | _EVENT_OPTIONAL_FIELDS
    ):
        raise error_type("event envelope fields are invalid")
    if event.get("schema") != HARNESS_EVENT_SCHEMA:
        raise error_type("event schema is invalid")
    if event.get("run_id") != run_id or event.get("turn_id") != turn_id:
        raise error_type("event run_id or turn_id does not match the journal")
    sequence = event.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise error_type("event sequence must be an integer")
    if sequence != expected_sequence:
        raise error_type(
            f"event sequence must be {expected_sequence}, received {sequence}"
        )
    event_type = event.get("type")
    if (
        not isinstance(event_type, str)
        or not event_type
        or event_type != event_type.strip()
        or len(event_type) > 100
    ):
        raise error_type("event type is invalid")
    if terminal_type is not None:
        raise error_type(f"event follows terminal event {terminal_type}")
    payload = event.get("payload")
    if not isinstance(payload, dict):
        raise error_type("event payload must be an object")
    _validate_timestamp(event.get("timestamp"), error_type=error_type)
    if "causation_id" in event:
        causation_id = event["causation_id"]
        if (
            not isinstance(causation_id, str)
            or not causation_id
            or len(causation_id) > 160
        ):
            raise error_type("event causation_id is invalid")

    try:
        payload_rendered = canonical_json(payload)
        expected_event_id = canonical_sha256(
            {
                "run_id": run_id,
                "turn_id": turn_id,
                "sequence": sequence,
                "type": event_type,
                "payload_sha256": sha256(payload_rendered.encode("utf-8")).hexdigest(),
            }
        )[:32]
    except HarnessContractError as exc:
        raise error_type("event payload is not canonical JSON") from exc
    if not isinstance(event.get("event_id"), str) or not compare_digest(
        event["event_id"], expected_event_id
    ):
        raise error_type("event_id does not match the event contents")

    resumed = payload.get("resumed")
    if expected_sequence == 1:
        if event_type != "run.started" or resumed is not False:
            raise error_type("the first event must be run.started with resumed=false")
    elif event_type == "run.started" and resumed is not True:
        raise error_type("a later run.started event must have resumed=true")
    return event_type


def _record_material(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(record[key]) for key in _RECORD_FIELDS if key != "record_hash"
    }


def _validate_record(
    record: Mapping[str, Any],
    *,
    run_id: str,
    turn_id: str,
    expected_sequence: int,
    expected_previous_hash: str,
    terminal_type: str | None,
) -> str:
    if frozenset(record) != _RECORD_FIELDS:
        raise JournalCorruptionError("journal record fields are invalid")
    if record.get("schema") != HARNESS_JOURNAL_RECORD_SCHEMA:
        raise JournalCorruptionError("journal record schema is invalid")
    if record.get("run_id") != run_id or record.get("turn_id") != turn_id:
        raise JournalCorruptionError("journal record identifiers do not match")
    sequence = record.get("sequence")
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence != expected_sequence
    ):
        raise JournalCorruptionError("journal record sequence is not contiguous")
    if record.get("previous_hash") != expected_previous_hash:
        raise JournalCorruptionError("journal hash chain predecessor mismatch")
    event = record.get("event")
    if not isinstance(event, dict):
        raise JournalCorruptionError("journal record event must be an object")
    if event.get("sequence") != sequence:
        raise JournalCorruptionError("record and event sequences do not match")
    record_hash = record.get("record_hash")
    if not isinstance(record_hash, str) or len(record_hash) != 64:
        raise JournalCorruptionError("journal record hash is invalid")
    try:
        expected_hash = canonical_sha256(_record_material(record))
    except HarnessContractError as exc:
        raise JournalCorruptionError("journal record is not canonical JSON") from exc
    if not compare_digest(record_hash, expected_hash):
        raise JournalCorruptionError("journal record hash mismatch")
    return _validate_event(
        event,
        run_id=run_id,
        turn_id=turn_id,
        expected_sequence=expected_sequence,
        terminal_type=terminal_type,
        error_type=JournalCorruptionError,
    )


def _ack_from_record(record: Mapping[str, Any]) -> DurableEventAck:
    event = record["event"]
    event_type = event["type"]
    return DurableEventAck(
        run_id=record["run_id"],
        turn_id=record["turn_id"],
        sequence=record["sequence"],
        event_id=event["event_id"],
        event_type=event_type,
        record_hash=record["record_hash"],
        terminal=event_type in TERMINAL_EVENT_TYPES,
    )


class HarnessJournal:
    """A single-run append-only event journal with atomic checkpoints."""

    def __init__(
        self,
        path: os.PathLike[str] | str,
        *,
        run_id: str,
        turn_id: str,
        checkpoint_path: os.PathLike[str] | str | None = None,
    ) -> None:
        if (
            not isinstance(run_id, str)
            or not run_id.strip()
            or len(run_id.strip()) > 160
        ):
            raise HarnessJournalError("run_id is required")
        if (
            not isinstance(turn_id, str)
            or not turn_id.strip()
            or len(turn_id.strip()) > 160
        ):
            raise HarnessJournalError("turn_id is required")
        self.run_id = run_id.strip()
        self.turn_id = turn_id.strip()
        self.path = _path_from_value(path, label="journal path")
        if checkpoint_path is None:
            checkpoint_path = self.path.with_name(self.path.name + ".checkpoint.json")
        self.checkpoint_path = _path_from_value(
            checkpoint_path, label="checkpoint path"
        )
        try:
            paths_alias = self.path.resolve(
                strict=False
            ) == self.checkpoint_path.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise HarnessJournalError("cannot resolve journal paths safely") from exc
        if paths_alias:
            raise HarnessJournalError("journal and checkpoint paths must differ")
        _prepare_parent(self.path, label="journal")
        _prepare_parent(self.checkpoint_path, label="checkpoint")
        _reject_non_regular_target(self.path, label="journal path")
        _reject_non_regular_target(self.checkpoint_path, label="checkpoint path")
        self._lock = RLock()
        self._state = _JournalState(records=[])
        self._signature: _FileSignature | None = None
        self._uncertain_after_write_failure = False
        self._last_ack: DurableEventAck | None = None
        self._terminal_ack: DurableEventAck | None = None
        with self._lock, self._open_journal() as file_descriptor:
            recovered = self._scan(file_descriptor)
            # Reopening is the explicit recovery boundary for a process that
            # may have died after a write but before receiving its fsync ack.
            # Validate first, then fence the exact observed prefix before any
            # record is published as a durable acknowledgement.
            os.fsync(file_descriptor)
            self._adopt_state(recovered, file_descriptor)

    @contextmanager
    def _open_journal(self) -> Iterator[int]:
        existed = _reject_non_regular_target(self.path, label="journal path")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            file_descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise HarnessJournalError("cannot open journal path") from exc
        try:
            file_stat = os.fstat(file_descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise HarnessJournalError("journal path must be a regular file")
            if fcntl is not None:
                fcntl.flock(file_descriptor, fcntl.LOCK_EX)
            if not existed:
                os.fsync(file_descriptor)
                _fsync_directory(self.path.parent)
            yield file_descriptor
        finally:
            if fcntl is not None:
                try:
                    fcntl.flock(file_descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(file_descriptor)

    def _truncate_torn_tail(self, file_descriptor: int, offset: int) -> None:
        os.ftruncate(file_descriptor, offset)
        os.fsync(file_descriptor)
        _fsync_directory(self.path.parent)

    def _scan(self, file_descriptor: int) -> _JournalState:
        data = _read_all(file_descriptor)
        records: list[dict[str, Any]] = []
        previous_hash = GENESIS_RECORD_HASH
        terminal_type: str | None = None
        cursor = 0
        while cursor < len(data):
            start = cursor
            newline = data.find(b"\n", cursor)
            has_newline = newline >= 0
            if has_newline:
                raw = data[cursor:newline]
                cursor = newline + 1
            else:
                raw = data[cursor:]
                cursor = len(data)
            is_final_line = cursor == len(data)
            try:
                record = _decode_json_object(
                    raw,
                    description=f"journal record at byte {start}",
                )
            except _MalformedJSONError as exc:
                # Only a non-newline-terminated tail can be a torn write. A
                # malformed complete JSONL record is durable corruption and
                # must never be silently discarded.
                if is_final_line and not has_newline:
                    self._truncate_torn_tail(file_descriptor, start)
                    data = data[:start]
                    break
                raise JournalCorruptionError(
                    f"malformed internal journal record at byte {start}"
                ) from exc
            event_type = _validate_record(
                record,
                run_id=self.run_id,
                turn_id=self.turn_id,
                expected_sequence=len(records) + 1,
                expected_previous_hash=previous_hash,
                terminal_type=terminal_type,
            )
            records.append(record)
            previous_hash = record["record_hash"]
            if event_type in TERMINAL_EVENT_TYPES:
                terminal_type = event_type
            if not has_newline:
                os.lseek(file_descriptor, 0, os.SEEK_END)
                _write_all(file_descriptor, b"\n")
                os.fsync(file_descriptor)
        return _JournalState(
            records=records,
            head_hash=previous_hash,
            terminal_type=terminal_type,
        )

    def _adopt_state(
        self,
        state: _JournalState,
        file_descriptor: int,
        *,
        update_ack: bool = True,
    ) -> _JournalState:
        self._state = state
        self._signature = _file_signature(file_descriptor)
        if update_ack:
            if state.records:
                self._last_ack = _ack_from_record(state.records[-1])
                self._terminal_ack = self._last_ack if self._last_ack.terminal else None
            else:
                self._last_ack = None
                self._terminal_ack = None
        return state

    def _refresh_locked(self, file_descriptor: int) -> _JournalState:
        if self._uncertain_after_write_failure:
            raise HarnessJournalError(
                "journal durability is uncertain after a failed write fence"
            )
        signature = _file_signature(file_descriptor)
        if self._signature == signature:
            return self._state
        return self._adopt_state(
            self._scan(file_descriptor),
            file_descriptor,
            update_ack=not self._uncertain_after_write_failure,
        )

    def _refresh(self) -> _JournalState:
        with self._lock, self._open_journal() as file_descriptor:
            return self._refresh_locked(file_descriptor)

    @property
    def next_sequence(self) -> int:
        return self._refresh().next_sequence

    @property
    def last_sequence(self) -> int:
        return self._refresh().last_sequence

    @property
    def head_hash(self) -> str:
        return self._refresh().head_hash

    @property
    def terminal_type(self) -> str | None:
        return self._refresh().terminal_type

    @property
    def last_ack(self) -> DurableEventAck | None:
        with self._lock:
            return self._last_ack

    @property
    def terminal_ack(self) -> DurableEventAck | None:
        with self._lock:
            return self._terminal_ack

    @property
    def durability_uncertain(self) -> bool:
        """Whether a failed fence left this journal unsafe to replay."""

        with self._lock:
            return self._uncertain_after_write_failure

    def append(self, event: Mapping[str, Any]) -> DurableEventAck:
        """Durably append ``event`` and return an acknowledgement after fsync."""

        if not isinstance(event, Mapping):
            raise JournalLifecycleError("event must be an object")
        try:
            normalized_event = json.loads(
                canonical_json(dict(event)),
                parse_constant=_json_constant_is_invalid,
            )
        except (HarnessContractError, ValueError, json.JSONDecodeError) as exc:
            raise JournalLifecycleError("event is not canonical JSON") from exc
        if not isinstance(normalized_event, dict):
            raise JournalLifecycleError("event must be an object")

        with self._lock, self._open_journal() as file_descriptor:
            if self._uncertain_after_write_failure:
                raise JournalLifecycleError(
                    "journal cannot append after a failed durability fence"
                )
            state = self._refresh_locked(file_descriptor)
            event_type = _validate_event(
                normalized_event,
                run_id=self.run_id,
                turn_id=self.turn_id,
                expected_sequence=state.next_sequence,
                terminal_type=state.terminal_type,
                error_type=JournalLifecycleError,
            )
            material: dict[str, Any] = {
                "schema": HARNESS_JOURNAL_RECORD_SCHEMA,
                "run_id": self.run_id,
                "turn_id": self.turn_id,
                "sequence": state.next_sequence,
                "previous_hash": state.head_hash,
                "event": normalized_event,
            }
            record = {**material, "record_hash": canonical_sha256(material)}
            encoded = canonical_json(record).encode("utf-8") + b"\n"
            original_size = os.lseek(file_descriptor, 0, os.SEEK_END)
            # ``os.write`` is unbuffered, so there is no userspace buffer to
            # flush between the completed write loop and this durability fence.
            for write_attempt in range(2):
                try:
                    os.lseek(file_descriptor, original_size, os.SEEK_SET)
                    _write_all(file_descriptor, encoded)
                    os.fsync(file_descriptor)
                    break
                except BaseException as append_error:
                    # Bytes may have reached the kernel page cache without
                    # crossing the durability fence. Roll the append back to
                    # the last acknowledged boundary and fence that truncation.
                    # If even the rollback fence fails, poison this journal
                    # instance: no replay, cursor read, or later append may
                    # adopt the visible page-cache bytes as durable state.
                    try:
                        os.ftruncate(file_descriptor, original_size)
                        os.fsync(file_descriptor)
                    except BaseException:
                        self._uncertain_after_write_failure = True
                        raise append_error
                    self._signature = _file_signature(file_descriptor)
                    self._uncertain_after_write_failure = False
                    if write_attempt == 0 and isinstance(append_error, OSError):
                        # This retries only the same local event append after a
                        # durable rollback. Model/tool effects are never
                        # repeated by the journal.
                        continue
                    raise

            state.records.append(record)
            state.head_hash = record["record_hash"]
            if event_type in TERMINAL_EVENT_TYPES:
                state.terminal_type = event_type
            self._state = state
            self._signature = _file_signature(file_descriptor)
            self._uncertain_after_write_failure = False
            acknowledgement = _ack_from_record(record)
            self._last_ack = acknowledgement
            if acknowledgement.terminal:
                self._terminal_ack = acknowledgement
            return acknowledgement

    append_event = append

    def __call__(self, event: Mapping[str, Any]) -> DurableEventAck:
        """Allow a journal instance to be passed directly as ``event_sink``."""

        return self.append(event)

    @staticmethod
    def _validate_after_sequence(after_sequence: int) -> None:
        if (
            isinstance(after_sequence, bool)
            or not isinstance(after_sequence, int)
            or after_sequence < 0
        ):
            raise HarnessJournalError("after_sequence must be a non-negative integer")

    def replay_records(self, *, after_sequence: int = 0) -> list[dict[str, Any]]:
        """Return verified records whose sequence is greater than the cursor."""

        self._validate_after_sequence(after_sequence)
        state = self._refresh()
        # Sequence numbers are one-based and contiguous, so the cursor maps
        # directly to a list offset.  Avoid walking the entire durable history
        # for every live subscriber poll.
        return [deepcopy(record) for record in state.records[after_sequence:]]

    def replay(self, *, after_sequence: int = 0) -> list[dict[str, Any]]:
        """Return verified event envelopes whose sequence is after the cursor."""

        return [
            deepcopy(record["event"])
            for record in self.replay_records(after_sequence=after_sequence)
        ]

    iter_events = replay

    @staticmethod
    def _snapshot_value(snapshot: Any) -> dict[str, Any]:
        if isinstance(snapshot, Mapping):
            value = dict(snapshot)
        else:
            to_dict = getattr(snapshot, "to_dict", None)
            if not callable(to_dict):
                raise HarnessJournalError("checkpoint snapshot must be an object")
            value = to_dict()
            if not isinstance(value, Mapping):
                raise HarnessJournalError("checkpoint to_dict() must return an object")
            value = dict(value)
        try:
            normalized = json.loads(
                canonical_json(value), parse_constant=_json_constant_is_invalid
            )
        except (HarnessContractError, ValueError, json.JSONDecodeError) as exc:
            raise HarnessJournalError(
                "checkpoint snapshot is not canonical JSON"
            ) from exc
        if not isinstance(normalized, dict):
            raise HarnessJournalError("checkpoint snapshot must be an object")
        return normalized

    def _validate_snapshot_identity(
        self, snapshot: Mapping[str, Any], *, sequence: int
    ) -> None:
        if "run_id" in snapshot and snapshot["run_id"] != self.run_id:
            raise HarnessJournalError("checkpoint run_id does not match the journal")
        if "turn_id" in snapshot and snapshot["turn_id"] != self.turn_id:
            raise HarnessJournalError("checkpoint turn_id does not match the journal")
        if "next_sequence" in snapshot:
            next_sequence = snapshot["next_sequence"]
            if (
                isinstance(next_sequence, bool)
                or not isinstance(next_sequence, int)
                or next_sequence != sequence + 1
            ):
                raise HarnessJournalError(
                    "checkpoint next_sequence does not match its journal anchor"
                )

    def _atomic_write_checkpoint(self, encoded: bytes) -> None:
        _prepare_parent(self.checkpoint_path, label="checkpoint")
        _reject_non_regular_target(self.checkpoint_path, label="checkpoint path")
        temporary_fd = -1
        temporary_path: str | None = None
        try:
            temporary_fd, temporary_path = tempfile.mkstemp(
                dir=self.checkpoint_path.parent,
                prefix=f".{self.checkpoint_path.name}.",
                suffix=".tmp",
            )
            os.fchmod(temporary_fd, 0o600)
            # The temporary file is written through an unbuffered descriptor;
            # fsync therefore follows the complete write without a hidden buffer.
            _write_all(temporary_fd, encoded)
            os.fsync(temporary_fd)
            os.close(temporary_fd)
            temporary_fd = -1
            _reject_non_regular_target(self.checkpoint_path, label="checkpoint path")
            os.replace(temporary_path, self.checkpoint_path)
            temporary_path = None
            _fsync_directory(self.checkpoint_path.parent)
        finally:
            if temporary_fd >= 0:
                os.close(temporary_fd)
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass

    def write_checkpoint(self, snapshot: Any) -> CheckpointWriteAck:
        """Atomically persist a snapshot anchored to the current journal head."""

        normalized = self._snapshot_value(snapshot)
        with self._lock, self._open_journal() as file_descriptor:
            state = self._refresh_locked(file_descriptor)
            if not state.records:
                raise HarnessJournalError(
                    "cannot checkpoint a journal before run.started is durable"
                )
            self._validate_snapshot_identity(normalized, sequence=state.last_sequence)
            material: dict[str, Any] = {
                "schema": HARNESS_JOURNAL_CHECKPOINT_SCHEMA,
                "run_id": self.run_id,
                "turn_id": self.turn_id,
                "sequence": state.last_sequence,
                "record_hash": state.head_hash,
                "snapshot": normalized,
            }
            checkpoint_sha256 = canonical_sha256(material)
            envelope = {
                **material,
                "checkpoint_sha256": checkpoint_sha256,
            }
            self._atomic_write_checkpoint(
                canonical_json(envelope).encode("utf-8") + b"\n"
            )
            self._state = state
            return CheckpointWriteAck(
                run_id=self.run_id,
                turn_id=self.turn_id,
                sequence=state.last_sequence,
                record_hash=state.head_hash,
                checkpoint_sha256=checkpoint_sha256,
            )

    save_checkpoint = write_checkpoint

    def _read_checkpoint_envelope(self) -> dict[str, Any] | None:
        exists = _reject_non_regular_target(
            self.checkpoint_path, label="checkpoint path"
        )
        if not exists:
            return None
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            file_descriptor = os.open(self.checkpoint_path, flags)
        except OSError as exc:
            raise HarnessJournalError("cannot open checkpoint path") from exc
        try:
            file_stat = os.fstat(file_descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise HarnessJournalError("checkpoint path must be a regular file")
            data = _read_all(file_descriptor)
        finally:
            os.close(file_descriptor)
        if not data.endswith(b"\n") or data.count(b"\n") != 1:
            raise JournalCorruptionError("checkpoint file framing is invalid")
        try:
            return _decode_json_object(data[:-1], description="checkpoint file")
        except _MalformedJSONError as exc:
            raise JournalCorruptionError("checkpoint file is malformed") from exc

    def load_checkpoint(self) -> dict[str, Any] | None:
        """Load and verify the atomic checkpoint, returning its snapshot."""

        with self._lock, self._open_journal() as file_descriptor:
            state = self._refresh_locked(file_descriptor)
            envelope = self._read_checkpoint_envelope()
            if envelope is None:
                return None
            if frozenset(envelope) != _CHECKPOINT_FIELDS:
                raise JournalCorruptionError("checkpoint envelope fields are invalid")
            if envelope.get("schema") != HARNESS_JOURNAL_CHECKPOINT_SCHEMA:
                raise JournalCorruptionError("checkpoint schema is invalid")
            if (
                envelope.get("run_id") != self.run_id
                or envelope.get("turn_id") != self.turn_id
            ):
                raise JournalCorruptionError("checkpoint identifiers do not match")
            sequence = envelope.get("sequence")
            if (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence < 1
                or sequence > state.last_sequence
            ):
                raise JournalCorruptionError("checkpoint sequence is invalid")
            anchor = state.records[sequence - 1]
            if envelope.get("record_hash") != anchor["record_hash"]:
                raise JournalCorruptionError("checkpoint journal anchor mismatch")
            snapshot = envelope.get("snapshot")
            if not isinstance(snapshot, dict):
                raise JournalCorruptionError("checkpoint snapshot must be an object")
            checkpoint_sha256 = envelope.get("checkpoint_sha256")
            if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
                raise JournalCorruptionError("checkpoint hash is invalid")
            material = {
                key: deepcopy(envelope[key])
                for key in _CHECKPOINT_FIELDS
                if key != "checkpoint_sha256"
            }
            try:
                expected_hash = canonical_sha256(material)
            except HarnessContractError as exc:
                raise JournalCorruptionError(
                    "checkpoint envelope is not canonical JSON"
                ) from exc
            if not compare_digest(checkpoint_sha256, expected_hash):
                raise JournalCorruptionError("checkpoint hash mismatch")
            try:
                self._validate_snapshot_identity(snapshot, sequence=sequence)
            except HarnessJournalError as exc:
                raise JournalCorruptionError(str(exc)) from exc
            self._state = state
            return deepcopy(snapshot)


# Explicit aliases keep integration code readable without creating a second
# implementation or a dependency on the teacher-agent persistence layer.
HarnessEventJournal = HarnessJournal
JournalAppendAck = DurableEventAck
HarnessJournalCorruptionError = JournalCorruptionError


__all__ = [
    "GENESIS_RECORD_HASH",
    "HARNESS_JOURNAL_CHECKPOINT_SCHEMA",
    "HARNESS_JOURNAL_RECORD_SCHEMA",
    "CheckpointWriteAck",
    "DurableEventAck",
    "HarnessEventJournal",
    "HarnessJournal",
    "HarnessJournalCorruptionError",
    "HarnessJournalError",
    "JournalAppendAck",
    "JournalCorruptionError",
    "JournalLifecycleError",
]
