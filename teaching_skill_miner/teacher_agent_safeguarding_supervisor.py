"""Cross-scope, content-free safeguarding delivery supervisor.

The learner-facing worker is deliberately not the lifecycle owner of urgent
delivery.  This process scans only opaque scope directories and hash-only
safeguarding ledgers, so pending delivery continues after learner workers are
idle-evicted and after an API process restart.  Receiver calls remain
idempotent by the durable ``delivery_id``; a separate fresh staff mutation is
still required to acknowledge ownership of a case.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import tempfile
import threading
import time
from typing import Any

try:  # pragma: no cover - production is POSIX/Linux.
    import fcntl
except ImportError:  # pragma: no cover - process-local fallback is explicit.
    fcntl = None  # type: ignore[assignment]

from .teacher_agent_safeguarding import (
    SafeguardingCapacityLimits,
    SafeguardingRetentionPolicy,
    TeacherAgentSafeguardingStore,
    default_emergency_resource_policy,
)
from .teacher_agent_safeguarding_dispatch import HttpsSafeguardingDispatcher
from .teacher_agent_safeguarding_pump import SafeguardingDispatchPump
from .teacher_agent_safeguarding_retention_authority import (
    InternalSafeguardingRetentionAuthority,
)


SUPERVISOR_BOOTSTRAP_SCHEMA = (
    "teaching_skill_miner.safeguarding_supervisor_bootstrap.v2"
)
SUPERVISOR_STATUS_SCHEMA = "teaching_skill_miner.safeguarding_supervisor_status.v2"
ROUTE_RECORD_SCHEMA = "teaching_skill_miner.safeguarding_route_record.v1"

_KEY_VERSION = re.compile(r"^k[1-9][0-9]{0,8}$")
_SCOPE_ID = re.compile(r"^scope_[0-9a-f]{48}$")
_SAFE_ROUTE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,2047}$")
_MAX_SCOPES = 100_000
_ROUTE_FILE = ".safeguarding_dispatch_route_v1.json"
_LEASE_FILE = ".safeguarding_dispatch_supervisor.lock"
_LOCAL_LOCKS_GUARD = threading.Lock()
_LOCAL_LOCKS: dict[str, threading.Lock] = {}


class SafeguardingSupervisorError(RuntimeError):
    """Stable supervisor configuration/integrity failure."""


def _local_lock(path: Path) -> threading.Lock:
    identity = str(path)
    with _LOCAL_LOCKS_GUARD:
        lock = _LOCAL_LOCKS.get(identity)
        if lock is None:
            lock = threading.Lock()
            _LOCAL_LOCKS[identity] = lock
        return lock


def _private_regular(path: Path, *, maximum_bytes: int) -> bytes:
    if path.is_symlink():
        raise SafeguardingSupervisorError("safeguarding supervisor file is unsafe")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SafeguardingSupervisorError(
            "safeguarding supervisor file is unavailable"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or (os.name == "posix" and metadata.st_uid != os.getuid())
            or metadata.st_size < 1
            or metadata.st_size > maximum_bytes
            or (metadata.st_mode & 0o077) != 0
        ):
            raise SafeguardingSupervisorError("safeguarding supervisor file is unsafe")
        value = os.read(descriptor, maximum_bytes + 1)
        if len(value) != metadata.st_size:
            raise SafeguardingSupervisorError(
                "safeguarding supervisor file changed while reading"
            )
        return value
    finally:
        os.close(descriptor)


def write_safeguarding_route_record(root: str | Path, route_locator: str) -> Path:
    """Atomically persist the opaque server route needed by the supervisor."""

    if (
        not isinstance(route_locator, str)
        or _SAFE_ROUTE.fullmatch(route_locator) is None
    ):
        raise SafeguardingSupervisorError("safeguarding route locator is invalid")
    candidate = Path(root).expanduser()
    if candidate.is_symlink():
        raise SafeguardingSupervisorError("safeguarding route root is unsafe")
    directory = candidate.resolve()
    if not directory.is_dir():
        raise SafeguardingSupervisorError("safeguarding route root is unsafe")
    payload = (
        json.dumps(
            {"schema": ROUTE_RECORD_SCHEMA, "route_locator": route_locator},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".safeguarding_route.", suffix=".tmp", dir=directory
    )
    temporary = Path(temporary_name)
    target = directory / _ROUTE_FILE
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written < 1:
                raise SafeguardingSupervisorError(
                    "safeguarding route record could not be persisted"
                )
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, target)
        parent = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    return target


def _read_route_record(root: Path) -> str:
    raw = _private_regular(root / _ROUTE_FILE, maximum_bytes=4096)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafeguardingSupervisorError(
            "safeguarding route record is invalid"
        ) from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema", "route_locator"}
        or value.get("schema") != ROUTE_RECORD_SCHEMA
        or not isinstance(value.get("route_locator"), str)
        or _SAFE_ROUTE.fullmatch(str(value["route_locator"])) is None
    ):
        raise SafeguardingSupervisorError("safeguarding route record is invalid")
    return str(value["route_locator"])


class _StoreLease:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._local = _local_lock(root)
        self._descriptor: int | None = None
        self._local_acquired = False

    def __enter__(self) -> bool:
        if not self._local.acquire(blocking=False):
            return False
        self._local_acquired = True
        try:
            descriptor = os.open(
                self.root / _LEASE_FILE,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            os.fchmod(descriptor, 0o600)
            if fcntl is not None:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(descriptor)
                    self._local.release()
                    self._local_acquired = False
                    return False
            self._descriptor = descriptor
            return True
        except BaseException:
            self._local.release()
            self._local_acquired = False
            raise

    def __exit__(self, *_args: object) -> None:
        if self._descriptor is not None:
            if fcntl is not None:
                fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            os.close(self._descriptor)
            self._descriptor = None
        if self._local_acquired:
            self._local.release()
            self._local_acquired = False


class SafeguardingDispatchSupervisor:
    """Scan opaque roots and deliver every durable pending escalation."""

    def __init__(
        self,
        root: str | Path,
        *,
        dispatcher_factory: Callable[[str], Any],
        readiness_probe: Callable[[], Mapping[str, Any]] | None = None,
        retention_policy: SafeguardingRetentionPolicy | None = None,
        retention_authority: Any | None = None,
        capacity_limits: SafeguardingCapacityLimits | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        candidate = Path(root).resolve()
        if candidate.is_symlink() or not candidate.is_dir():
            raise SafeguardingSupervisorError("safeguarding supervisor root is unsafe")
        if (
            not callable(dispatcher_factory)
            or (readiness_probe is not None and not callable(readiness_probe))
            or (
                retention_policy is not None
                and not isinstance(retention_policy, SafeguardingRetentionPolicy)
            )
            or (
                capacity_limits is not None
                and not isinstance(capacity_limits, SafeguardingCapacityLimits)
            )
            or not callable(monotonic_clock)
            or not callable(wall_clock)
        ):
            raise SafeguardingSupervisorError(
                "safeguarding supervisor configuration is invalid"
            )
        if (retention_policy is None) != (retention_authority is None):
            raise SafeguardingSupervisorError(
                "retention policy and dedicated authority must be configured together"
            )
        if retention_authority is not None and (
            not callable(getattr(retention_authority, "issue", None))
            or not callable(getattr(retention_authority, "verify", None))
        ):
            raise SafeguardingSupervisorError(
                "safeguarding retention authority is invalid"
            )
        self.root = candidate
        self._dispatcher_factory = dispatcher_factory
        self._readiness_probe = readiness_probe
        self._retention_policy = retention_policy
        self._retention_authority = retention_authority
        self._capacity_limits = capacity_limits
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self._pumps: dict[
            str,
            tuple[str, TeacherAgentSafeguardingStore, SafeguardingDispatchPump],
        ] = {}
        self._attempted_total = 0
        self._accepted_total = 0
        self._failed_total = 0
        self._last_success_at_utc: str | None = None
        self._receiver_readiness_result = (
            "never" if readiness_probe is not None else "not_required"
        )
        self._receiver_readiness_expires_at = 0.0
        self._receiver_readiness_attempts = 0
        self._receiver_readiness_successes = 0
        self._receiver_readiness_failures = 0
        self._last_receiver_success_at_utc: str | None = None

    def _probe_receiver(self, *, now: datetime) -> None:
        if self._readiness_probe is None:
            return
        monotonic = float(self._monotonic_clock())
        if not 0 <= monotonic < float("inf"):
            raise SafeguardingSupervisorError(
                "safeguarding supervisor clock is invalid"
            )
        if monotonic < self._receiver_readiness_expires_at:
            return
        self._receiver_readiness_attempts += 1
        try:
            result = self._readiness_probe()
            if (
                not isinstance(result, Mapping)
                or result.get("status") != "ready"
                or result.get("policy_version_validated") is not True
                or result.get("receiver_network_validated") is not True
                or result.get("credential_validated") is not True
                or result.get("learner_content_sent") is not False
                or result.get("case_created") is not False
            ):
                raise SafeguardingSupervisorError(
                    "safeguarding receiver readiness is invalid"
                )
        except Exception:
            self._receiver_readiness_result = "failed"
            self._receiver_readiness_failures += 1
            self._receiver_readiness_expires_at = monotonic + 1.0
        else:
            self._receiver_readiness_result = "ready"
            self._receiver_readiness_successes += 1
            self._receiver_readiness_expires_at = monotonic + 30.0
            self._last_receiver_success_at_utc = now.isoformat(
                timespec="seconds"
            ).replace("+00:00", "Z")

    def _roots(self) -> list[Path]:
        result: list[Path] = []
        for version in sorted(self.root.iterdir(), key=lambda item: item.name):
            if version.name.startswith("."):
                continue
            if (
                version.is_symlink()
                or not version.is_dir()
                or _KEY_VERSION.fullmatch(version.name) is None
            ):
                continue
            for scope in sorted(version.iterdir(), key=lambda item: item.name):
                if (
                    scope.is_symlink()
                    or not scope.is_dir()
                    or _SCOPE_ID.fullmatch(scope.name) is None
                ):
                    continue
                safeguarding = scope / "safeguarding"
                if safeguarding.is_symlink() or not safeguarding.is_dir():
                    continue
                if (safeguarding / "safeguarding_cases.json").is_file():
                    result.append(safeguarding)
                    if len(result) > _MAX_SCOPES:
                        raise SafeguardingSupervisorError(
                            "safeguarding supervisor scope capacity is exhausted"
                        )
        return result

    def _pump(
        self, root: Path, route_locator: str
    ) -> tuple[TeacherAgentSafeguardingStore, SafeguardingDispatchPump]:
        route_identity = sha256(route_locator.encode("utf-8")).hexdigest()
        cached = self._pumps.get(str(root))
        if cached is not None and cached[0] == route_identity:
            return cached[1], cached[2]
        dispatcher = self._dispatcher_factory(route_locator)
        store = TeacherAgentSafeguardingStore(
            root,
            authorization_verifier=lambda _value: (_ for _ in ()).throw(
                PermissionError("supervisor is read-only")
            ),
            emergency_resource_policy=default_emergency_resource_policy({}),
            retention_authorization_verifier=(
                self._retention_authority.verify
                if self._retention_authority is not None
                else None
            ),
            capacity_limits=self._capacity_limits,
            clock=self._wall_clock,
        )
        pump = SafeguardingDispatchPump(
            store=store,
            dispatcher=dispatcher,
            clock=self._monotonic_clock,
        )
        self._pumps[str(root)] = (route_identity, store, pump)
        return store, pump

    def run_once(self) -> dict[str, Any]:
        now = self._wall_clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise SafeguardingSupervisorError(
                "safeguarding supervisor clock is invalid"
            )
        now = now.astimezone(timezone.utc)
        self._probe_receiver(now=now)
        roots = self._roots()
        pending = overdue = attempted = accepted = failed = unavailable = 0
        retention_eligible = retention_compacted = retention_events_compacted = 0
        retention_failures = retention_blocked = capacity_near = 0
        events_total = event_capacity_total = store_bytes_total = 0
        store_byte_capacity_total = tombstones_total = tombstone_capacity_total = 0
        cases_compacted_total = events_compacted_total = fence_inserted_total = 0
        event_headroom_min: int | None = None
        store_byte_headroom_min: int | None = None
        tombstone_headroom_min: int | None = None
        fence_false_positive_upper_bound_max = 0.0
        fence_false_positive_within_target = True
        oldest_seconds = 0
        active_paths: set[str] = set()
        for root in roots:
            active_paths.add(str(root))
            try:
                route = _read_route_record(root)
                store, pump = self._pump(root, route)
                with _StoreLease(root) as acquired:
                    if not acquired:
                        continue
                    rows = store.pending_escalations()
                    pending += len(rows)
                    for row in rows:
                        observed = datetime.fromisoformat(
                            str(row["observed_at_utc"]).replace("Z", "+00:00")
                        ).astimezone(timezone.utc)
                        due = datetime.fromisoformat(
                            str(row["sla_due_at_utc"]).replace("Z", "+00:00")
                        ).astimezone(timezone.utc)
                        oldest_seconds = max(
                            oldest_seconds,
                            max(0, int((now - observed).total_seconds())),
                        )
                        overdue += int(now >= due)
                    result = pump.run_once()
                    plan: Mapping[str, Any] | None = None
                    if self._retention_policy is not None:
                        try:
                            plan = store.retention_plan(
                                self._retention_policy,
                                as_of_utc=now.isoformat(timespec="seconds").replace(
                                    "+00:00", "Z"
                                ),
                            )
                            retention_eligible += int(plan["eligible_case_count"])
                            if int(plan["selected_case_count"]):
                                receipt = self._retention_authority.issue(
                                    operation="case.retention_compacted",
                                    body_sha256=str(plan["body_sha256"]),
                                )
                                compacted = store.compact_retained_cases(
                                    self._retention_policy,
                                    as_of_utc=str(plan["as_of_utc"]),
                                    authorization_receipt=receipt,
                                )
                                retention_compacted += int(compacted["cases_compacted"])
                                retention_events_compacted += int(
                                    compacted["events_compacted"]
                                )
                        except Exception:
                            retention_failures += 1
                    capacity = store.capacity_status(
                        self._retention_policy,
                        as_of_utc=now.isoformat(timespec="seconds").replace(
                            "+00:00", "Z"
                        ),
                    )
                attempted += int(result["attempted"])
                accepted += int(result["accepted"])
                failed += int(result["failed"])
                if result.get("store_available") is not True:
                    unavailable += 1
                events_total += int(capacity["events"])
                event_capacity_total += int(capacity["event_capacity"])
                store_bytes_total += int(capacity["store_bytes"])
                store_byte_capacity_total += int(capacity["store_byte_capacity"])
                tombstones_total += int(capacity["recent_erasure_tombstones"])
                tombstone_capacity_total += int(
                    capacity["recent_erasure_tombstone_capacity"]
                )
                cases_compacted_total += int(capacity["cases_compacted_total"])
                events_compacted_total += int(capacity["events_compacted_total"])
                fence_inserted_total += int(capacity["erasure_fence_inserted_count"])
                capacity_near += int(capacity["near_capacity"] is True)
                retention_blocked += int(
                    capacity["retention_compaction_blocked"] is True
                )
                event_headroom = int(capacity["event_headroom"])
                byte_headroom = int(capacity["store_byte_headroom"])
                tombstone_headroom = int(capacity["recent_erasure_tombstone_headroom"])
                event_headroom_min = (
                    event_headroom
                    if event_headroom_min is None
                    else min(event_headroom_min, event_headroom)
                )
                store_byte_headroom_min = (
                    byte_headroom
                    if store_byte_headroom_min is None
                    else min(store_byte_headroom_min, byte_headroom)
                )
                tombstone_headroom_min = (
                    tombstone_headroom
                    if tombstone_headroom_min is None
                    else min(tombstone_headroom_min, tombstone_headroom)
                )
                fence_false_positive_upper_bound_max = max(
                    fence_false_positive_upper_bound_max,
                    float(
                        capacity["erasure_fence_estimated_false_positive_upper_bound"]
                    ),
                )
                fence_false_positive_within_target = (
                    fence_false_positive_within_target
                    and capacity["erasure_fence_false_positive_within_target"] is True
                )
            except Exception:
                unavailable += 1
        self._pumps = {
            path: value for path, value in self._pumps.items() if path in active_paths
        }
        self._attempted_total += attempted
        self._accepted_total += accepted
        self._failed_total += failed
        if accepted:
            self._last_success_at_utc = now.isoformat(timespec="seconds").replace(
                "+00:00", "Z"
            )
        if self._retention_policy is None:
            retention_status = "disabled"
        elif retention_failures:
            retention_status = "failed"
        elif retention_blocked:
            retention_status = "blocked"
        elif retention_compacted:
            retention_status = "compacted"
        else:
            retention_status = "idle"
        return {
            "schema": SUPERVISOR_STATUS_SCHEMA,
            "status": (
                "ready"
                if unavailable == 0
                and failed == 0
                and retention_failures == 0
                and retention_blocked == 0
                and capacity_near == 0
                and fence_false_positive_within_target
                and self._receiver_readiness_result in {"not_required", "ready"}
                else "degraded"
            ),
            "scopes_scanned": len(roots),
            "stores_unavailable": unavailable,
            "pending": pending,
            "overdue": overdue,
            "oldest_pending_age_seconds": oldest_seconds,
            "attempted": attempted,
            "accepted": accepted,
            "failed": failed,
            "attempted_total": self._attempted_total,
            "accepted_total": self._accepted_total,
            "failed_total": self._failed_total,
            "last_success_at_utc": self._last_success_at_utc,
            "receiver_readiness_required": self._readiness_probe is not None,
            "receiver_readiness_status": self._receiver_readiness_result,
            "receiver_network_validated": self._receiver_readiness_result == "ready",
            "receiver_credential_validated": self._receiver_readiness_result == "ready",
            "receiver_readiness_attempts_total": self._receiver_readiness_attempts,
            "receiver_readiness_successes_total": self._receiver_readiness_successes,
            "receiver_readiness_failures_total": self._receiver_readiness_failures,
            "last_receiver_success_at_utc": self._last_receiver_success_at_utc,
            "retention_enabled": self._retention_policy is not None,
            "retention_status": retention_status,
            "retention_policy_version_sha256": (
                self._retention_policy.policy_version_sha256
                if self._retention_policy is not None
                else None
            ),
            "retention_minimum_closed_age_seconds": (
                self._retention_policy.minimum_closed_age_seconds
                if self._retention_policy is not None
                else None
            ),
            "retention_maximum_cases_per_run": (
                self._retention_policy.maximum_cases_per_run
                if self._retention_policy is not None
                else None
            ),
            "retention_eligible_cases": retention_eligible,
            "retention_cases_compacted": retention_compacted,
            "retention_events_compacted": retention_events_compacted,
            "retention_failures": retention_failures,
            "retention_blocked_stores": retention_blocked,
            "capacity_near_limit_stores": capacity_near,
            "capacity_events": events_total,
            "capacity_event_limit": event_capacity_total,
            "capacity_event_headroom_min": event_headroom_min,
            "capacity_store_bytes": store_bytes_total,
            "capacity_store_byte_limit": store_byte_capacity_total,
            "capacity_store_byte_headroom_min": store_byte_headroom_min,
            "capacity_recent_erasure_tombstones": tombstones_total,
            "capacity_recent_erasure_tombstone_limit": tombstone_capacity_total,
            "capacity_recent_erasure_tombstone_headroom_min": tombstone_headroom_min,
            "retention_cases_compacted_total": cases_compacted_total,
            "retention_events_compacted_total": events_compacted_total,
            "erasure_fence_inserted_count": fence_inserted_total,
            "erasure_fence_estimated_false_positive_upper_bound": (
                fence_false_positive_upper_bound_max
            ),
            "erasure_fence_false_positive_target_upper_bound": 1e-6,
            "erasure_fence_false_positive_within_target": (
                fence_false_positive_within_target
            ),
            "erasure_fence_false_negative_possible": False,
            "erasure_fence_false_positive_policy": "fail_closed_as_erased",
            "raw_learner_text_read_or_sent": False,
            "scope_identity_labels_exposed": False,
            "updated_at_utc": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        }

    def run_forever(self, stopping: threading.Event, *, poll_seconds: float) -> None:
        if not 0.25 <= poll_seconds <= 60:
            raise SafeguardingSupervisorError("safeguarding supervisor poll is invalid")
        while not stopping.is_set():
            status = self.run_once()
            sys.stdout.write(json.dumps(status, separators=(",", ":")) + "\n")
            sys.stdout.flush()
            stopping.wait(poll_seconds)


def _main() -> int:
    if sys.argv[1:] in (["--help"], ["-h"]):
        sys.stdout.write(
            "usage: python -m teaching_skill_miner.teacher_agent_safeguarding_supervisor "
            "[--self-check]\n"
        )
        return 0
    if sys.argv[1:] == ["--self-check"]:
        if not SUPERVISOR_STATUS_SCHEMA or not callable(SafeguardingDispatchSupervisor):
            return 2
        sys.stdout.write("teacher_agent_safeguarding_supervisor self-check passed\n")
        return 0
    if len(sys.argv) != 1:
        return 2
    raw = sys.stdin.buffer.readline(32 * 1024 + 1)
    if not raw or len(raw) > 32 * 1024 or sys.stdin.buffer.read(1):
        return 2
    try:
        config = json.loads(raw.decode("utf-8"))
        if (
            not isinstance(config, Mapping)
            or set(config)
            != {
                "schema",
                "root",
                "endpoint",
                "bearer_secret",
                "policy_version",
                "timeout_ms",
                "maximum_response_bytes",
                "poll_seconds",
                "retention_policy_version",
                "retention_minimum_closed_age_seconds",
                "retention_maximum_cases_per_run",
                "retention_authority_secret",
                "retention_deployment_context_sha256",
            }
            or config.get("schema") != SUPERVISOR_BOOTSTRAP_SCHEMA
        ):
            raise SafeguardingSupervisorError("supervisor bootstrap is invalid")
        retention_secret = config.get("retention_authority_secret")
        if (
            not isinstance(retention_secret, str)
            or retention_secret != retention_secret.strip()
            or any(character in retention_secret for character in "\x00\r\n")
            or len(retention_secret.encode("utf-8")) < 32
        ):
            raise SafeguardingSupervisorError(
                "retention authority bootstrap is invalid"
            )
        retention_policy = SafeguardingRetentionPolicy(
            policy_version=config.get("retention_policy_version"),
            minimum_closed_age_seconds=config.get(
                "retention_minimum_closed_age_seconds"
            ),
            maximum_cases_per_run=config.get("retention_maximum_cases_per_run"),
        )
        retention_authority = InternalSafeguardingRetentionAuthority(
            key=retention_secret.encode("utf-8"),
            deployment_context_sha256=config.get("retention_deployment_context_sha256"),
        )

        def factory(route: str) -> HttpsSafeguardingDispatcher:
            return HttpsSafeguardingDispatcher(
                endpoint=str(config["endpoint"]),
                bearer_secret=str(config["bearer_secret"]),
                route_locator=route,
                policy_version=str(config["policy_version"]),
                timeout_seconds=float(config["timeout_ms"]) / 1000.0,
                maximum_response_bytes=int(config["maximum_response_bytes"]),
            )

        readiness_dispatcher = factory("supervisor_readiness")
        supervisor = SafeguardingDispatchSupervisor(
            str(config["root"]),
            dispatcher_factory=factory,
            readiness_probe=readiness_dispatcher.probe_readiness,
            retention_policy=retention_policy,
            retention_authority=retention_authority,
        )
        poll_seconds = float(config["poll_seconds"])
    except Exception:
        return 2
    stopping = threading.Event()

    def stop(_signal: int, _frame: Any) -> None:
        stopping.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    supervisor.run_forever(stopping, poll_seconds=poll_seconds)
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess exercised by integration.
    raise SystemExit(_main())


__all__ = [
    "ROUTE_RECORD_SCHEMA",
    "SUPERVISOR_BOOTSTRAP_SCHEMA",
    "SUPERVISOR_STATUS_SCHEMA",
    "SafeguardingDispatchSupervisor",
    "SafeguardingSupervisorError",
    "write_safeguarding_route_record",
]
