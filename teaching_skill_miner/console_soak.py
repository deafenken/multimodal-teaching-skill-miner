"""Resource-budget and soak acceptance for a loopback Teaching Console/Harness.

The runner deliberately has no dependency on the Console or Harness internals.  It
can exercise either a runner-owned deterministic HTTP fixture or an operator-owned
loopback target.  Resource samples come from the host process table and FD inventory;
the receipt never substitutes fixture constants for OS measurements.

Private target URLs, process IDs, commands, filesystem paths, response bodies, and
raw exceptions are excluded from the receipt.  Every safe configuration projection,
the runtime identity projection, and the aggregate sample stream are SHA-256 bound.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


RECEIPT_SCHEMA = "teachlab.console.resource_soak.v1"
FIXTURE_READY_SCHEMA = "teachlab.console.resource_soak.fixture_ready.v1"
BROWSER_CONTROL_SCHEMA = "teachlab.console.resource_soak.browser_control.v1"
BROWSER_STATE_SCHEMA = "teachlab.console.resource_soak.browser_state.v1"
LONG_SOAK_SECONDS = 8 * 60 * 60
RESERVED_CONSOLE_PORT = 3030
SUPPORTED_BROWSERS = ("chromium", "firefox", "webkit")
MIB = 1024 * 1024


class SoakFailure(RuntimeError):
    """A fixed-code failure which is safe to include in a public receipt."""

    def __init__(self, stage: str, code: str) -> None:
        super().__init__(code)
        self.stage = stage
        self.code = code


def canonical_json(value: Any) -> str:
    """Return the one JSON representation used by every receipt hash."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SoakFailure("receipt", "non_canonical_receipt_value") from exc


def canonical_sha256(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def bind_receipt(content: Mapping[str, Any]) -> dict[str, Any]:
    """Attach a hash over the complete receipt except the integrity member."""

    material = dict(content)
    material.pop("integrity", None)
    return {
        **material,
        "integrity": {
            "algorithm": "sha256_canonical_json_without_integrity",
            "content_sha256": canonical_sha256(material),
        },
    }


def verify_receipt(receipt: Mapping[str, Any]) -> bool:
    """Verify the outer hash and all explicitly hash-bound safe projections."""

    try:
        material = dict(receipt)
        integrity = material.pop("integrity")
        if not isinstance(integrity, Mapping):
            return False
        if integrity.get("algorithm") != "sha256_canonical_json_without_integrity":
            return False
        if integrity.get("content_sha256") != canonical_sha256(material):
            return False
        configuration = receipt["configuration"]
        budgets = receipt["budgets"]
        sampling = receipt["sampling"]
        runtime = receipt["runtime"]
        return bool(
            isinstance(configuration, Mapping)
            and configuration.get("content_sha256")
            == canonical_sha256(configuration.get("material"))
            and isinstance(budgets, Mapping)
            and budgets.get("content_sha256") == canonical_sha256(budgets.get("limits"))
            and isinstance(sampling, Mapping)
            and sampling.get("configuration_sha256")
            == canonical_sha256(sampling.get("configuration"))
            and isinstance(runtime, Mapping)
            and runtime.get("identity_sha256")
            == canonical_sha256(runtime.get("identity_material"))
        )
    except (KeyError, TypeError, SoakFailure):
        return False


def _sha256_file(path: Path) -> str | None:
    try:
        digest = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _round(value: float, digits: int = 3) -> float:
    return round(max(0.0, float(value)), digits)


def percentile_95(values: Sequence[float]) -> float | None:
    """Nearest-rank p95, which is deterministic for both short and long runs."""

    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def long_soak_threshold_reached(actual_duration_seconds: float) -> bool:
    """Return true only after the measured window really reaches eight hours."""

    return bool(
        math.isfinite(float(actual_duration_seconds))
        and float(actual_duration_seconds) >= LONG_SOAK_SECONDS
    )


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    request_p95_ms: float = 1_000.0
    failure_rate: float = 0.01
    target_peak_rss_bytes: int = 512 * MIB
    target_rss_growth_bytes: int = 64 * MIB
    target_peak_fd_count: int = 512
    target_fd_growth: int = 64
    target_subprocesses: int = 16
    disk_growth_bytes: int = 64 * MIB
    browser_peak_rss_bytes: int = 1_024 * MIB
    browser_rss_growth_bytes: int = 256 * MIB
    browser_peak_fd_count: int = 2_048
    browser_fd_growth: int = 256
    browser_subprocesses: int = 32
    minimum_fd_sample_coverage: float = 0.90

    def validate(self) -> None:
        positive = {
            "request_p95_ms": self.request_p95_ms,
            "target_peak_rss_bytes": self.target_peak_rss_bytes,
            "target_peak_fd_count": self.target_peak_fd_count,
            "disk_growth_bytes": self.disk_growth_bytes,
            "browser_peak_rss_bytes": self.browser_peak_rss_bytes,
            "browser_peak_fd_count": self.browser_peak_fd_count,
        }
        if any(
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0
            for value in positive.values()
        ):
            raise SoakFailure("input", "invalid_positive_budget")
        nonnegative = {
            "target_rss_growth_bytes": self.target_rss_growth_bytes,
            "target_fd_growth": self.target_fd_growth,
            "target_subprocesses": self.target_subprocesses,
            "browser_rss_growth_bytes": self.browser_rss_growth_bytes,
            "browser_fd_growth": self.browser_fd_growth,
            "browser_subprocesses": self.browser_subprocesses,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in nonnegative.values()
        ):
            raise SoakFailure("input", "invalid_nonnegative_budget")
        if not 0.0 <= float(self.failure_rate) <= 1.0:
            raise SoakFailure("input", "invalid_failure_rate_budget")
        if not 0.0 < float(self.minimum_fd_sample_coverage) <= 1.0:
            raise SoakFailure("input", "invalid_fd_coverage_budget")

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "request_p95_ms": float(self.request_p95_ms),
            "failure_rate": float(self.failure_rate),
            "target_peak_rss_bytes": int(self.target_peak_rss_bytes),
            "target_rss_growth_bytes": int(self.target_rss_growth_bytes),
            "target_peak_fd_count": int(self.target_peak_fd_count),
            "target_fd_growth": int(self.target_fd_growth),
            "target_subprocesses": int(self.target_subprocesses),
            "disk_growth_bytes": int(self.disk_growth_bytes),
            "browser_peak_rss_bytes": int(self.browser_peak_rss_bytes),
            "browser_rss_growth_bytes": int(self.browser_rss_growth_bytes),
            "browser_peak_fd_count": int(self.browser_peak_fd_count),
            "browser_fd_growth": int(self.browser_fd_growth),
            "browser_subprocesses": int(self.browser_subprocesses),
            "minimum_fd_sample_coverage": float(self.minimum_fd_sample_coverage),
        }


@dataclass(frozen=True, slots=True)
class SoakConfig:
    mode: str = "smoke"
    duration_seconds: float = 60.0
    request_interval_seconds: float = 1.0
    sample_interval_seconds: float = 5.0
    request_timeout_seconds: float = 5.0
    target_url: str | None = None
    target_pid: int | None = None
    runtime_id: str | None = None
    request_path: str = "work"
    disk_paths: tuple[Path, ...] = ()
    browser: str | None = None
    browser_tabs: int = 1
    browser_refresh_seconds: float = 30.0
    self_test_request_limit: int | None = None
    budgets: BudgetConfig = field(default_factory=BudgetConfig)

    @property
    def uses_fixture(self) -> bool:
        return self.target_url is None

    def validate(self) -> None:
        self.budgets.validate()
        if self.mode not in {"smoke", "8h", "self-test"}:
            raise SoakFailure("input", "invalid_mode")
        if self.mode == "8h" and self.duration_seconds != LONG_SOAK_SECONDS:
            raise SoakFailure("input", "long_mode_duration_must_be_eight_hours")
        if not 0 < float(self.duration_seconds) <= LONG_SOAK_SECONDS:
            raise SoakFailure("input", "invalid_duration")
        for value in (
            self.request_interval_seconds,
            self.sample_interval_seconds,
            self.request_timeout_seconds,
            self.browser_refresh_seconds,
        ):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise SoakFailure("input", "invalid_interval")
        if self.request_timeout_seconds > 300:
            raise SoakFailure("input", "request_timeout_too_large")
        if self.browser not in {None, *SUPPORTED_BROWSERS}:
            raise SoakFailure("input", "unsupported_browser")
        if (
            isinstance(self.browser_tabs, bool)
            or not isinstance(self.browser_tabs, int)
            or not 1 <= self.browser_tabs <= 8
        ):
            raise SoakFailure("input", "invalid_browser_tab_count")
        if self.self_test_request_limit is not None and (
            self.mode != "self-test"
            or isinstance(self.self_test_request_limit, bool)
            or not 1 <= self.self_test_request_limit <= 100
        ):
            raise SoakFailure("input", "invalid_self_test_request_limit")
        _validated_request_path(self.request_path)
        if self.uses_fixture:
            if (
                self.target_pid is not None
                or self.runtime_id is not None
                or self.disk_paths
            ):
                raise SoakFailure("input", "fixture_rejects_external_target_options")
        else:
            _validated_loopback_url(str(self.target_url))
            if (
                isinstance(self.target_pid, bool)
                or not isinstance(self.target_pid, int)
                or self.target_pid < 2
            ):
                raise SoakFailure("input", "external_target_pid_required")
            if (
                not isinstance(self.runtime_id, str)
                or not self.runtime_id.strip()
                or len(self.runtime_id) > 256
            ):
                raise SoakFailure("input", "external_runtime_id_required")
            if not self.disk_paths:
                raise SoakFailure("input", "external_disk_measurement_root_required")
            for path in self.disk_paths:
                _validated_disk_path(path)

    def safe_material(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "requested_duration_seconds": float(self.duration_seconds),
            "long_soak_threshold_seconds": LONG_SOAK_SECONDS,
            "request_interval_seconds": float(self.request_interval_seconds),
            "sample_interval_seconds": float(self.sample_interval_seconds),
            "request_timeout_seconds": float(self.request_timeout_seconds),
            "request_path_sha256": sha256(
                self.request_path.encode("utf-8")
            ).hexdigest(),
            "target_kind": "isolated_fixture"
            if self.uses_fixture
            else "external_loopback",
            "target_process_scope": (
                "runner_owned_process_group"
                if self.uses_fixture
                else "operator_owned_descendant_tree"
            ),
            "disk_root_count": 1 if self.uses_fixture else len(self.disk_paths),
            "browser_engine": self.browser or "disabled",
            "browser_tabs": int(self.browser_tabs) if self.browser else 0,
            "browser_refresh_seconds": float(self.browser_refresh_seconds),
            "self_test_request_limit": self.self_test_request_limit,
        }


def _validated_loopback_url(raw: str) -> str:
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise SoakFailure("input", "invalid_target_url") from exc
    decoded = unquote(parsed.path)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1"}
        or port is None
        or not 1 <= port <= 65535
        or port == RESERVED_CONSOLE_PORT
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
        or any(part in {".", ".."} for part in decoded.split("/"))
    ):
        raise SoakFailure("input", "non_loopback_reserved_or_invalid_target")
    return raw


def _validated_request_path(raw: str) -> str:
    if not isinstance(raw, str) or len(raw) > 512:
        raise SoakFailure("input", "invalid_request_path")
    if raw.startswith(("/", "//")) or "?" in raw or "#" in raw or "\\" in raw:
        raise SoakFailure("input", "invalid_request_path")
    decoded = unquote(raw)
    if any(part in {".", ".."} for part in decoded.split("/")):
        raise SoakFailure("input", "invalid_request_path")
    return raw


def _request_url(base_url: str, request_path: str) -> str:
    base = _validated_loopback_url(base_url)
    relative = _validated_request_path(request_path)
    return f"{base.rstrip('/')}/{relative}" if relative else base


def _validated_disk_path(path: Path) -> Path:
    try:
        expanded = path.expanduser()
        source_info = expanded.lstat()
        if stat.S_ISLNK(source_info.st_mode):
            raise OSError
        resolved = expanded.resolve(strict=True)
        info = resolved.lstat()
    except OSError as exc:
        raise SoakFailure("input", "invalid_disk_measurement_root") from exc
    if stat.S_ISLNK(info.st_mode) or not (
        stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)
    ):
        raise SoakFailure("input", "invalid_disk_measurement_root")
    return resolved


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(  # type: ignore[override]
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


_HTTP_OPENER = build_opener(ProxyHandler({}), _NoRedirect())


def _http_probe(url: str, timeout_seconds: float) -> tuple[bool, float]:
    """Issue one bounded GET without redirects and discard all response content."""

    _validated_loopback_url(url)
    started = time.monotonic()
    success = False
    try:
        request = Request(
            url,
            method="GET",
            headers={
                "Accept": "application/json,text/html;q=0.5",
                "Cache-Control": "no-store",
            },
        )
        with _HTTP_OPENER.open(request, timeout=timeout_seconds) as response:
            response.read(64 * 1024)
            success = 200 <= int(response.status) < 300
    except HTTPError as exc:
        with suppress(OSError):
            exc.read(64 * 1024)
    except (OSError, URLError, ValueError):
        pass
    return success, (time.monotonic() - started) * 1000.0


@dataclass(frozen=True, slots=True)
class _ProcessRow:
    pid: int
    ppid: int
    pgid: int
    rss_bytes: int
    start_marker: str


@dataclass(frozen=True, slots=True)
class _ProcessMetric:
    pid: int
    ppid: int
    rss_bytes: int
    fd_count: int | None
    direct_children: int
    start_marker: str


@dataclass(frozen=True, slots=True)
class _ProcessScope:
    role: str
    root_pid: int
    selection: str
    process_group_id: int | None = None
    root_start_marker: str | None = None


@dataclass(frozen=True, slots=True)
class _LiveGroupSample:
    processes: tuple[_ProcessMetric, ...]
    fd_inventory_source: str

    @property
    def rss_bytes(self) -> int:
        return sum(item.rss_bytes for item in self.processes)

    @property
    def fd_count(self) -> int | None:
        if any(item.fd_count is None for item in self.processes):
            return None
        return sum(int(item.fd_count or 0) for item in self.processes)

    @property
    def subprocesses(self) -> int:
        return max(0, len(self.processes) - 1)


def _read_process_table() -> dict[int, _ProcessRow]:
    """Read one live process table without exposing command lines."""

    if os.name != "posix":
        raise SoakFailure("sampling", "process_sampling_unsupported_platform")
    try:
        completed = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,pgid=,rss=,lstart="],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SoakFailure("sampling", "process_table_unavailable") from exc
    if completed.returncode != 0:
        raise SoakFailure("sampling", "process_table_unavailable")
    rows: dict[int, _ProcessRow] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            pid, ppid, pgid, rss_kib = (int(field) for field in fields[:4])
        except ValueError:
            continue
        if pid >= 2 and ppid >= 0 and pgid >= 0 and rss_kib >= 0:
            rows[pid] = _ProcessRow(
                pid=pid,
                ppid=ppid,
                pgid=pgid,
                rss_bytes=rss_kib * 1024,
                start_marker=" ".join(fields[4:]),
            )
    if not rows:
        raise SoakFailure("sampling", "empty_process_table")
    return rows


def _selected_rows(
    rows: Mapping[int, _ProcessRow], scope: _ProcessScope
) -> tuple[_ProcessRow, ...]:
    root = rows.get(scope.root_pid)
    if root is None or (
        scope.root_start_marker is not None
        and root.start_marker != scope.root_start_marker
    ):
        return ()
    if scope.selection == "process_group":
        if scope.process_group_id is None:
            raise SoakFailure("sampling", "missing_process_group_identity")
        selected = [row for row in rows.values() if row.pgid == scope.process_group_id]
    elif scope.selection == "descendant_tree":
        selected_pids = {scope.root_pid}
        changed = True
        while changed:
            changed = False
            for row in rows.values():
                if row.ppid in selected_pids and row.pid not in selected_pids:
                    selected_pids.add(row.pid)
                    changed = True
        selected = [row for row in rows.values() if row.pid in selected_pids]
    else:
        raise SoakFailure("sampling", "invalid_process_scope")
    return tuple(
        sorted(selected, key=lambda item: (item.pid != scope.root_pid, item.pid))
    )


def _procfs_fd_counts(pids: Sequence[int]) -> dict[int, int | None] | None:
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return None
    counts: dict[int, int | None] = {}
    for pid in pids:
        try:
            counts[pid] = sum(1 for _ in (proc_root / str(pid) / "fd").iterdir())
        except OSError:
            counts[pid] = None
    return counts


def _lsof_fd_counts(pids: Sequence[int]) -> dict[int, int | None] | None:
    executable = shutil.which("lsof")
    if executable is None or not pids:
        return None
    try:
        completed = subprocess.run(
            [executable, "-nP", "-Fpf", "-p", ",".join(str(pid) for pid in pids)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {pid: None for pid in pids}
    counts = {pid: 0 for pid in pids}
    current: int | None = None
    for line in completed.stdout.splitlines():
        if line.startswith("p"):
            try:
                candidate = int(line[1:])
            except ValueError:
                current = None
            else:
                current = candidate if candidate in counts else None
        elif line.startswith("f") and current is not None:
            counts[current] += 1
    for pid in pids:
        if counts[pid] == 0:
            counts[pid] = None  # type: ignore[assignment]
    return counts


def _fd_counts(pids: Sequence[int]) -> tuple[dict[int, int | None], str]:
    counts = _procfs_fd_counts(pids)
    if counts is not None:
        return counts, "procfs_fd_inventory"
    counts = _lsof_fd_counts(pids)
    if counts is not None:
        return counts, "lsof_fd_inventory"
    return {pid: None for pid in pids}, "fd_inventory_unavailable"


def _collect_group(
    rows: Mapping[int, _ProcessRow], scope: _ProcessScope
) -> _LiveGroupSample:
    selected = _selected_rows(rows, scope)
    if not selected or not any(item.pid == scope.root_pid for item in selected):
        raise SoakFailure("sampling", f"{scope.role}_root_process_missing")
    child_counts = {row.pid: 0 for row in selected}
    for row in selected:
        if row.ppid in child_counts:
            child_counts[row.ppid] += 1
    counts, source = _fd_counts([row.pid for row in selected])
    return _LiveGroupSample(
        processes=tuple(
            _ProcessMetric(
                pid=row.pid,
                ppid=row.ppid,
                rss_bytes=row.rss_bytes,
                fd_count=counts.get(row.pid),
                direct_children=child_counts[row.pid],
                start_marker=row.start_marker,
            )
            for row in selected
        ),
        fd_inventory_source=source,
    )


def _root_start_marker(pid: int) -> str:
    row = _read_process_table().get(pid)
    if row is None or not row.start_marker:
        raise SoakFailure("runtime", "root_process_identity_unavailable")
    return row.start_marker


@dataclass(slots=True)
class _PerProcessAggregate:
    ref: str
    sample_count: int = 0
    first_sample: int = 0
    last_sample: int = 0
    baseline_rss_bytes: int = 0
    ending_rss_bytes: int = 0
    peak_rss_bytes: int = 0
    baseline_fd_count: int | None = None
    ending_fd_count: int | None = None
    peak_fd_count: int | None = None
    fd_samples_known: int = 0
    peak_direct_children: int = 0

    def add(self, metric: _ProcessMetric, sample_index: int) -> None:
        if self.sample_count == 0:
            self.first_sample = sample_index
            self.baseline_rss_bytes = metric.rss_bytes
        self.sample_count += 1
        self.last_sample = sample_index
        self.ending_rss_bytes = metric.rss_bytes
        self.peak_rss_bytes = max(self.peak_rss_bytes, metric.rss_bytes)
        self.peak_direct_children = max(
            self.peak_direct_children, metric.direct_children
        )
        if metric.fd_count is not None:
            if self.baseline_fd_count is None:
                self.baseline_fd_count = metric.fd_count
            self.ending_fd_count = metric.fd_count
            self.peak_fd_count = max(self.peak_fd_count or 0, metric.fd_count)
            self.fd_samples_known += 1

    def to_safe_dict(self) -> dict[str, Any]:
        rss_growth = self.ending_rss_bytes - self.baseline_rss_bytes
        fd_growth = (
            None
            if self.baseline_fd_count is None or self.ending_fd_count is None
            else self.ending_fd_count - self.baseline_fd_count
        )
        return {
            "process_ref": self.ref,
            "sample_count": self.sample_count,
            "first_seen_sample": self.first_sample,
            "last_seen_sample": self.last_sample,
            "baseline_rss_bytes": self.baseline_rss_bytes,
            "ending_rss_bytes": self.ending_rss_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "ending_minus_baseline_rss_bytes": rss_growth,
            "peak_fd_count": self.peak_fd_count,
            "ending_minus_baseline_fd_count": fd_growth,
            "fd_samples_known": self.fd_samples_known,
            "peak_direct_children": self.peak_direct_children,
        }


@dataclass(slots=True)
class _GroupAccumulator:
    role: str
    scope: str
    samples: int = 0
    baseline_rss_bytes: int = 0
    ending_rss_bytes: int = 0
    peak_rss_bytes: int = 0
    baseline_fd_count: int | None = None
    ending_fd_count: int | None = None
    peak_fd_count: int | None = None
    fd_samples_known: int = 0
    peak_processes: int = 0
    peak_subprocesses: int = 0
    fd_sources: set[str] = field(default_factory=set)
    _process_refs: dict[tuple[int, str], str] = field(default_factory=dict)
    _per_process: dict[tuple[int, str], _PerProcessAggregate] = field(
        default_factory=dict
    )

    def add(self, sample: _LiveGroupSample, sample_index: int) -> dict[str, Any]:
        if self.samples == 0:
            self.baseline_rss_bytes = sample.rss_bytes
        self.samples += 1
        self.ending_rss_bytes = sample.rss_bytes
        self.peak_rss_bytes = max(self.peak_rss_bytes, sample.rss_bytes)
        self.peak_processes = max(self.peak_processes, len(sample.processes))
        self.peak_subprocesses = max(self.peak_subprocesses, sample.subprocesses)
        self.fd_sources.add(sample.fd_inventory_source)
        group_fd = sample.fd_count
        if group_fd is not None:
            if self.baseline_fd_count is None:
                self.baseline_fd_count = group_fd
            self.ending_fd_count = group_fd
            self.peak_fd_count = max(self.peak_fd_count or 0, group_fd)
            self.fd_samples_known += 1

        safe_processes: list[dict[str, Any]] = []
        child_counter = sum(
            1 for reference in self._process_refs.values() if ":child-" in reference
        )
        for metric in sample.processes:
            identity = (metric.pid, metric.start_marker)
            if identity not in self._process_refs:
                if metric.pid == sample.processes[0].pid and not self._process_refs:
                    ref = f"{self.role}:root"
                else:
                    child_counter += 1
                    ref = f"{self.role}:child-{child_counter:03d}"
                self._process_refs[identity] = ref
                self._per_process[identity] = _PerProcessAggregate(ref=ref)
            aggregate = self._per_process[identity]
            aggregate.add(metric, sample_index)
            safe_processes.append(
                {
                    "process_ref": aggregate.ref,
                    "rss_bytes": metric.rss_bytes,
                    "fd_count": metric.fd_count,
                    "direct_children": metric.direct_children,
                }
            )
        return {
            "rss_bytes": sample.rss_bytes,
            "fd_count": group_fd,
            "process_count": len(sample.processes),
            "subprocess_count": sample.subprocesses,
            "fd_inventory_source": sample.fd_inventory_source,
            "processes": safe_processes,
        }

    def to_safe_dict(self) -> dict[str, Any]:
        if self.samples == 0:
            return {
                "measurement_source": "live_os_process_table_and_fd_inventory",
                "synthetic_process_samples": False,
                "scope": self.scope,
                "sample_count": 0,
                "baseline_rss_bytes": None,
                "ending_rss_bytes": None,
                "peak_rss_bytes": None,
                "ending_minus_baseline_rss_bytes": None,
                "peak_minus_baseline_rss_bytes": None,
                "baseline_fd_count": None,
                "ending_fd_count": None,
                "peak_fd_count": None,
                "ending_minus_baseline_fd_count": None,
                "peak_minus_baseline_fd_count": None,
                "fd_sample_coverage": 0.0,
                "fd_inventory_sources": [],
                "peak_process_count": None,
                "peak_subprocess_count": None,
                "processes": [],
            }
        rss_ending_growth = self.ending_rss_bytes - self.baseline_rss_bytes
        rss_peak_growth = self.peak_rss_bytes - self.baseline_rss_bytes
        fd_ending_growth = (
            None
            if self.baseline_fd_count is None or self.ending_fd_count is None
            else self.ending_fd_count - self.baseline_fd_count
        )
        fd_peak_growth = (
            None
            if self.baseline_fd_count is None or self.peak_fd_count is None
            else self.peak_fd_count - self.baseline_fd_count
        )
        coverage = self.fd_samples_known / self.samples if self.samples else 0.0
        return {
            "measurement_source": "live_os_process_table_and_fd_inventory",
            "synthetic_process_samples": False,
            "scope": self.scope,
            "sample_count": self.samples,
            "baseline_rss_bytes": self.baseline_rss_bytes,
            "ending_rss_bytes": self.ending_rss_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "ending_minus_baseline_rss_bytes": rss_ending_growth,
            "peak_minus_baseline_rss_bytes": rss_peak_growth,
            "baseline_fd_count": self.baseline_fd_count,
            "ending_fd_count": self.ending_fd_count,
            "peak_fd_count": self.peak_fd_count,
            "ending_minus_baseline_fd_count": fd_ending_growth,
            "peak_minus_baseline_fd_count": fd_peak_growth,
            "fd_sample_coverage": _round(coverage, 6),
            "fd_inventory_sources": sorted(self.fd_sources),
            "peak_process_count": self.peak_processes,
            "peak_subprocess_count": self.peak_subprocesses,
            "processes": [
                aggregate.to_safe_dict()
                for aggregate in sorted(
                    self._per_process.values(), key=lambda item: item.ref
                )
            ],
        }


def _tree_size(paths: Sequence[Path]) -> tuple[int, int]:
    """Return logical bytes and regular-file count without following symlinks."""

    total = 0
    files = 0
    for root in paths:
        try:
            root_info = root.lstat()
        except OSError as exc:
            raise SoakFailure("sampling", "disk_measurement_root_unavailable") from exc
        if stat.S_ISLNK(root_info.st_mode):
            raise SoakFailure("sampling", "disk_measurement_symlink_rejected")
        if stat.S_ISREG(root_info.st_mode):
            total += root_info.st_size
            files += 1
            continue
        if not stat.S_ISDIR(root_info.st_mode):
            raise SoakFailure("sampling", "disk_measurement_root_invalid")
        for directory, directory_names, file_names in os.walk(
            root, topdown=True, followlinks=False
        ):
            base = Path(directory)
            kept_directories: list[str] = []
            for name in directory_names:
                candidate = base / name
                with suppress(OSError):
                    if not candidate.is_symlink():
                        kept_directories.append(name)
            directory_names[:] = kept_directories
            for name in file_names:
                candidate = base / name
                try:
                    info = candidate.lstat()
                except OSError:
                    continue
                if stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                    total += info.st_size
                    files += 1
    return total, files


@dataclass(slots=True)
class _DiskAccumulator:
    root_count: int
    samples: int = 0
    baseline_bytes: int = 0
    ending_bytes: int = 0
    peak_bytes: int = 0
    baseline_files: int = 0
    ending_files: int = 0
    peak_files: int = 0

    def add(self, byte_count: int, file_count: int) -> dict[str, int]:
        if self.samples == 0:
            self.baseline_bytes = byte_count
            self.baseline_files = file_count
        self.samples += 1
        self.ending_bytes = byte_count
        self.ending_files = file_count
        self.peak_bytes = max(self.peak_bytes, byte_count)
        self.peak_files = max(self.peak_files, file_count)
        return {"bytes": byte_count, "regular_files": file_count}

    def to_safe_dict(self) -> dict[str, Any]:
        if self.samples == 0:
            return {
                "measurement_source": "logical_regular_file_size_without_symlinks",
                "root_count": self.root_count,
                "sample_count": 0,
                "baseline_bytes": None,
                "ending_bytes": None,
                "peak_bytes": None,
                "ending_minus_baseline_bytes": None,
                "peak_minus_baseline_bytes": None,
                "baseline_regular_files": None,
                "ending_regular_files": None,
                "peak_regular_files": None,
            }
        return {
            "measurement_source": "logical_regular_file_size_without_symlinks",
            "root_count": self.root_count,
            "sample_count": self.samples,
            "baseline_bytes": self.baseline_bytes,
            "ending_bytes": self.ending_bytes,
            "peak_bytes": self.peak_bytes,
            "ending_minus_baseline_bytes": self.ending_bytes - self.baseline_bytes,
            "peak_minus_baseline_bytes": self.peak_bytes - self.baseline_bytes,
            "baseline_regular_files": self.baseline_files,
            "ending_regular_files": self.ending_files,
            "peak_regular_files": self.peak_files,
        }


@dataclass(slots=True)
class _RequestAccumulator:
    attempts: int = 0
    successes: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def add(self, success: bool, latency_ms: float) -> None:
        self.attempts += 1
        self.successes += int(success)
        self.latencies_ms.append(float(latency_ms))

    def to_safe_dict(self) -> dict[str, Any]:
        failures = self.attempts - self.successes
        p95 = percentile_95(self.latencies_ms)
        return {
            "method": "GET",
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": failures,
            "failure_rate": (
                None if not self.attempts else _round(failures / self.attempts, 8)
            ),
            "latency_ms": {
                "minimum": (
                    None if not self.latencies_ms else _round(min(self.latencies_ms))
                ),
                "median": (
                    None
                    if not self.latencies_ms
                    else _round(sorted(self.latencies_ms)[len(self.latencies_ms) // 2])
                ),
                "p95_nearest_rank": None if p95 is None else _round(p95),
                "maximum": (
                    None if not self.latencies_ms else _round(max(self.latencies_ms))
                ),
            },
        }


@dataclass(slots=True)
class _SampleStream:
    count: int = 0
    process_table_reads: int = 0
    _digest: Any = field(default_factory=sha256)

    def add(self, safe_sample: Mapping[str, Any]) -> None:
        self._digest.update(canonical_json(safe_sample).encode("utf-8"))
        self._digest.update(b"\n")
        self.count += 1
        self.process_table_reads += 1

    @property
    def content_sha256(self) -> str:
        return self._digest.hexdigest()


@dataclass(slots=True)
class _BrowserAccumulator:
    engine: str | None
    configured_tabs: int
    state_samples: int = 0
    peak_active_tabs: int = 0
    final_state: dict[str, Any] = field(default_factory=dict)

    def add(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != BROWSER_STATE_SCHEMA:
            return
        self.state_samples += 1
        active = state.get("active_tabs")
        if isinstance(active, int) and not isinstance(active, bool) and active >= 0:
            self.peak_active_tabs = max(self.peak_active_tabs, active)
        safe_keys = {
            "status",
            "failure_code",
            "configured_tabs",
            "active_tabs",
            "navigation_attempts",
            "navigation_successes",
            "navigation_failures",
            "navigation_p95_ms",
            "blocked_external_requests",
        }
        self.final_state = {key: state.get(key) for key in safe_keys if key in state}

    def to_safe_dict(self, *, process_group_measured: bool = False) -> dict[str, Any]:
        if self.engine is None:
            return {
                "mode": "disabled",
                "engine": None,
                "configured_tabs": 0,
                "measurement_boundary": (
                    "no_browser_launched; HTTP target process resources only"
                ),
                "process_group_measured": False,
                "tab_count_measured": False,
            }
        return {
            "mode": "runner_owned_headless_browser",
            "engine": self.engine,
            "configured_tabs": self.configured_tabs,
            "state_samples": self.state_samples,
            "peak_active_tabs": self.peak_active_tabs,
            "final_state": dict(sorted(self.final_state.items())),
            "measurement_boundary": (
                "browser worker and engine descendants are a separate process group; "
                + (
                    "group resources were sampled, but renderer attribution to a "
                    "single tab is not inferred"
                    if process_group_measured
                    else "no live group resource sample was collected"
                )
            ),
            "process_group_measured": process_group_measured,
            "tab_count_measured": True,
        }


@dataclass(slots=True)
class _OwnedProcess:
    role: str
    process: subprocess.Popen[bytes]
    process_group_id: int


class _OwnedProcessRegistry:
    """Own and reap only process groups started by this runner."""

    def __init__(self) -> None:
        self._processes: list[_OwnedProcess] = []

    def spawn(self, role: str, command: Sequence[str]) -> _OwnedProcess:
        if os.name != "posix" or not hasattr(os, "killpg"):
            raise SoakFailure("runtime", "owned_process_groups_unsupported")
        try:
            process = subprocess.Popen(
                list(command),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            process_group_id = os.getpgid(process.pid)
        except OSError as exc:
            raise SoakFailure("runtime", f"{role}_process_start_failed") from exc
        if process_group_id != process.pid:
            with suppress(ProcessLookupError):
                process.terminate()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=2)
            raise SoakFailure("runtime", f"{role}_process_group_not_isolated")
        owned = _OwnedProcess(role, process, process_group_id)
        self._processes.append(owned)
        return owned

    def cleanup(self) -> dict[str, Any]:
        term_signals = 0
        kill_signals = 0
        failures = 0
        for owned in reversed(self._processes):
            if not _process_group_has_live_members(owned.process_group_id):
                continue
            try:
                if (
                    owned.process.poll() is None
                    and os.getpgid(owned.process.pid) != owned.process_group_id
                ):
                    failures += 1
                    continue
                os.killpg(owned.process_group_id, signal.SIGTERM)
                term_signals += 1
            except ProcessLookupError:
                continue
            except OSError:
                failures += 1
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and any(
            _process_group_has_live_members(item.process_group_id)
            for item in self._processes
        ):
            for owned in self._processes:
                owned.process.poll()
            time.sleep(0.05)
        for owned in reversed(self._processes):
            if _process_group_has_live_members(owned.process_group_id):
                try:
                    if (
                        owned.process.poll() is None
                        and os.getpgid(owned.process.pid) != owned.process_group_id
                    ):
                        failures += 1
                        continue
                    os.killpg(owned.process_group_id, signal.SIGKILL)
                    kill_signals += 1
                except OSError:
                    failures += 1
        kill_deadline = time.monotonic() + 3.0
        while time.monotonic() < kill_deadline and any(
            _process_group_has_live_members(item.process_group_id)
            for item in self._processes
        ):
            time.sleep(0.05)
        for owned in self._processes:
            with suppress(subprocess.TimeoutExpired):
                owned.process.wait(timeout=0.2)
        all_reaped = bool(
            all(item.process.poll() is not None for item in self._processes)
            and not any(
                _process_group_has_live_members(item.process_group_id)
                for item in self._processes
            )
        )
        return {
            "owned_process_groups": len(self._processes),
            "term_signals_sent": term_signals,
            "kill_signals_sent": kill_signals,
            "cleanup_failures": failures,
            "all_owned_processes_reaped": bool(all_reaped and failures == 0),
            "external_target_signalled": False,
        }


def _process_group_has_live_members(process_group_id: int) -> bool:
    """Ignore already-dead zombies while checking an owned process group."""

    try:
        completed = subprocess.run(
            ["ps", "-axo", "pgid=,stat="],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return False
        except OSError:
            return True
        return True
    if completed.returncode != 0:
        return True
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            candidate_group = int(fields[0])
        except ValueError:
            continue
        if candidate_group == process_group_id and not fields[1].startswith("Z"):
            return True
    return False


@dataclass(slots=True)
class _StopController:
    requested_signal: str | None = None

    @property
    def requested(self) -> bool:
        return self.requested_signal is not None

    def request(self, signal_number: int) -> None:
        if self.requested_signal is None:
            self.requested_signal = signal.Signals(signal_number).name


def _private_json_write(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _private_json_read(path: Path) -> dict[str, Any] | None:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


class _FixtureHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, capability: str, activity_path: Path, port: int) -> None:
        super().__init__(("127.0.0.1", port), _FixtureHandler)
        self.capability = capability
        self.activity_path = activity_path
        self.request_count = 0
        self._activity_lock = threading.Lock()

    def record_work(self) -> None:
        with self._activity_lock:
            self.request_count += 1
            with self.activity_path.open("ab", buffering=0) as stream:
                stream.write(b"teachlab-soak-fixture-record\n")


class _FixtureHandler(BaseHTTPRequestHandler):
    server: _FixtureHttpServer

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _route(self) -> str | None:
        path = urlsplit(self.path).path
        prefix = f"/{self.server.capability}/"
        if path == prefix[:-1]:
            return ""
        if not path.startswith(prefix):
            return None
        return path.removeprefix(prefix).strip("/")

    def _send(self, status_code: int, content_type: str, body: bytes) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        route = self._route()
        if route == "":
            self._send(
                200,
                "text/html; charset=utf-8",
                b"<!doctype html><html><title>soak fixture</title><body>ready</body></html>",
            )
        elif route == "health":
            self._send(
                200,
                "application/json; charset=utf-8",
                b'{"status":"healthy"}',
            )
        elif route == "work":
            self.server.record_work()
            self._send(
                200,
                "application/json; charset=utf-8",
                b'{"status":"accepted"}',
            )
        else:
            self._send(
                404,
                "application/json; charset=utf-8",
                b'{"status":"not_found"}',
            )


def _fixture_worker(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--data-directory", type=Path, required=True)
    arguments = parser.parse_args(list(argv))
    ready_file = arguments.ready_file.resolve()
    data_directory = arguments.data_directory.resolve()
    try:
        ready_parent = ready_file.parent.lstat()
        data_info = data_directory.lstat()
    except OSError:
        return 2
    if (
        not stat.S_ISDIR(ready_parent.st_mode)
        or stat.S_ISLNK(ready_parent.st_mode)
        or not stat.S_ISDIR(data_info.st_mode)
        or stat.S_ISLNK(data_info.st_mode)
    ):
        return 2
    capability = secrets.token_urlsafe(36)
    server: _FixtureHttpServer | None = None
    for _ in range(20):
        port = 49_152 + secrets.randbelow(65_536 - 49_152)
        try:
            server = _FixtureHttpServer(
                capability, data_directory / "fixture-activity.log", port
            )
            break
        except OSError:
            continue
    if server is None:
        return 3
    base_url = f"http://127.0.0.1:{int(server.server_address[1])}/{capability}/"
    try:
        _private_json_write(
            ready_file,
            {
                "schema": FIXTURE_READY_SCHEMA,
                "base_url": base_url,
                "root_process_id": os.getpid(),
            },
        )
        server.serve_forever(poll_interval=0.2)
    except (OSError, SoakFailure):
        return 4
    finally:
        server.server_close()
    return 0


def _browser_state(
    *,
    status: str,
    configured_tabs: int,
    active_tabs: int,
    attempts: int,
    successes: int,
    latencies: Sequence[float],
    blocked_external: int,
    failure_code: str | None = None,
) -> dict[str, Any]:
    p95 = percentile_95(latencies)
    state: dict[str, Any] = {
        "schema": BROWSER_STATE_SCHEMA,
        "status": status,
        "configured_tabs": configured_tabs,
        "active_tabs": active_tabs,
        "navigation_attempts": attempts,
        "navigation_successes": successes,
        "navigation_failures": attempts - successes,
        "navigation_p95_ms": None if p95 is None else _round(p95),
        "blocked_external_requests": blocked_external,
    }
    if failure_code:
        state["failure_code"] = failure_code
    return state


def _browser_worker(argv: Sequence[str]) -> int:
    """Run an optional headless browser in its own measured process group."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--control-file", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    arguments = parser.parse_args(list(argv))
    control = _private_json_read(arguments.control_file.resolve())
    if control is None or control.get("schema") != BROWSER_CONTROL_SCHEMA:
        return 2
    engine = control.get("engine")
    target_url = control.get("target_url")
    configured_tabs = control.get("tabs")
    refresh_seconds = control.get("refresh_seconds")
    try:
        _validated_loopback_url(str(target_url))
    except SoakFailure:
        return 2
    if (
        engine not in SUPPORTED_BROWSERS
        or isinstance(configured_tabs, bool)
        or not isinstance(configured_tabs, int)
        or not 1 <= configured_tabs <= 8
        or isinstance(refresh_seconds, bool)
        or not isinstance(refresh_seconds, (int, float))
        or refresh_seconds <= 0
    ):
        return 2
    state_file = arguments.state_file.resolve()
    attempts = 0
    successes = 0
    latencies: list[float] = []
    blocked_external = 0
    try:
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError:
        _private_json_write(
            state_file,
            _browser_state(
                status="failed",
                configured_tabs=configured_tabs,
                active_tabs=0,
                attempts=0,
                successes=0,
                latencies=(),
                blocked_external=0,
                failure_code="python_playwright_not_installed",
            ),
        )
        return 3

    base = urlsplit(str(target_url))
    try:
        with sync_playwright() as playwright:
            try:
                browser = getattr(playwright, str(engine)).launch(headless=True)
            except Exception:
                _private_json_write(
                    state_file,
                    _browser_state(
                        status="failed",
                        configured_tabs=configured_tabs,
                        active_tabs=0,
                        attempts=0,
                        successes=0,
                        latencies=(),
                        blocked_external=0,
                        failure_code="browser_executable_unavailable",
                    ),
                )
                return 4
            context = browser.new_context(service_workers="block")

            def guard(route: Any) -> None:
                nonlocal blocked_external
                try:
                    target = urlsplit(route.request.url)
                    allowed = bool(
                        target.scheme == base.scheme == "http"
                        and target.hostname == base.hostname
                        and target.port == base.port
                        and target.username is None
                        and target.password is None
                    )
                except (TypeError, ValueError):
                    allowed = False
                if allowed:
                    route.continue_()
                else:
                    blocked_external += 1
                    route.abort()

            context.route("**/*", guard)
            pages = [context.new_page() for _ in range(configured_tabs)]

            def navigate_all() -> None:
                nonlocal attempts, successes
                for page in pages:
                    started = time.monotonic()
                    attempts += 1
                    try:
                        response = page.goto(
                            str(target_url),
                            wait_until="domcontentloaded",
                            timeout=min(
                                60_000, max(1_000, int(refresh_seconds * 1_000))
                            ),
                        )
                        if response is not None and 200 <= response.status < 300:
                            successes += 1
                    except Exception:
                        pass
                    latencies.append((time.monotonic() - started) * 1000.0)

            navigate_all()
            _private_json_write(
                state_file,
                _browser_state(
                    status="running",
                    configured_tabs=configured_tabs,
                    active_tabs=len(
                        [page for page in context.pages if not page.is_closed()]
                    ),
                    attempts=attempts,
                    successes=successes,
                    latencies=latencies,
                    blocked_external=blocked_external,
                ),
            )
            while True:
                time.sleep(float(refresh_seconds))
                navigate_all()
                _private_json_write(
                    state_file,
                    _browser_state(
                        status="running",
                        configured_tabs=configured_tabs,
                        active_tabs=len(
                            [page for page in context.pages if not page.is_closed()]
                        ),
                        attempts=attempts,
                        successes=successes,
                        latencies=latencies,
                        blocked_external=blocked_external,
                    ),
                )
    except (OSError, SoakFailure):
        return 5


def _wait_for_private_state(
    path: Path,
    *,
    process: subprocess.Popen[bytes],
    schema: str,
    timeout_seconds: float,
    failure_stage: str,
    stop_controller: _StopController | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if stop_controller is not None and stop_controller.requested:
            raise SoakFailure("signal", "run_interrupted")
        state = _private_json_read(path)
        if state is not None and state.get("schema") == schema:
            return state
        if process.poll() is not None:
            raise SoakFailure(failure_stage, f"{failure_stage}_worker_exited")
        time.sleep(0.05)
    raise SoakFailure(failure_stage, f"{failure_stage}_worker_not_ready")


def _executable_sha256(pid: int) -> str | None:
    proc_executable = Path("/proc") / str(pid) / "exe"
    if proc_executable.exists():
        try:
            return _sha256_file(proc_executable.resolve(strict=True))
        except OSError:
            return None
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "comm="],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    candidate = Path(result.stdout.strip())
    return _sha256_file(candidate) if candidate.is_absolute() else None


def _budget_check(
    name: str,
    actual: int | float | None,
    limit: int | float,
    unit: str,
    *,
    minimum: bool = False,
) -> dict[str, Any]:
    evaluable = isinstance(actual, (int, float)) and not isinstance(actual, bool)
    passed = bool(
        evaluable
        and (
            float(actual) >= float(limit) if minimum else float(actual) <= float(limit)
        )
    )
    return {
        "name": name,
        "actual": actual,
        "limit": limit,
        "comparison": ">=" if minimum else "<=",
        "unit": unit,
        "evaluable": evaluable,
        "passed": passed,
    }


def _evaluate_budgets(
    *,
    budgets: BudgetConfig,
    request_summary: Mapping[str, Any],
    target_summary: Mapping[str, Any],
    disk_summary: Mapping[str, Any],
    browser_summary: Mapping[str, Any] | None,
    browser_state: Mapping[str, Any],
    configured_browser_tabs: int,
) -> dict[str, Any]:
    latency = request_summary.get("latency_ms", {})
    if not isinstance(latency, Mapping):
        latency = {}
    checks = [
        _budget_check(
            "request_latency_p95",
            latency.get("p95_nearest_rank"),
            budgets.request_p95_ms,
            "milliseconds",
        ),
        _budget_check(
            "request_failure_rate",
            request_summary.get("failure_rate"),
            budgets.failure_rate,
            "ratio",
        ),
        _budget_check(
            "target_peak_rss",
            target_summary.get("peak_rss_bytes"),
            budgets.target_peak_rss_bytes,
            "bytes",
        ),
        _budget_check(
            "target_rss_growth",
            target_summary.get("peak_minus_baseline_rss_bytes"),
            budgets.target_rss_growth_bytes,
            "bytes",
        ),
        _budget_check(
            "target_peak_fd",
            target_summary.get("peak_fd_count"),
            budgets.target_peak_fd_count,
            "descriptors",
        ),
        _budget_check(
            "target_fd_growth",
            target_summary.get("peak_minus_baseline_fd_count"),
            budgets.target_fd_growth,
            "descriptors",
        ),
        _budget_check(
            "target_fd_sample_coverage",
            target_summary.get("fd_sample_coverage"),
            budgets.minimum_fd_sample_coverage,
            "ratio",
            minimum=True,
        ),
        _budget_check(
            "target_subprocesses",
            target_summary.get("peak_subprocess_count"),
            budgets.target_subprocesses,
            "processes",
        ),
        _budget_check(
            "disk_growth",
            disk_summary.get("peak_minus_baseline_bytes"),
            budgets.disk_growth_bytes,
            "bytes",
        ),
    ]
    if browser_summary is not None:
        checks.extend(
            [
                _budget_check(
                    "browser_peak_rss",
                    browser_summary.get("peak_rss_bytes"),
                    budgets.browser_peak_rss_bytes,
                    "bytes",
                ),
                _budget_check(
                    "browser_rss_growth",
                    browser_summary.get("peak_minus_baseline_rss_bytes"),
                    budgets.browser_rss_growth_bytes,
                    "bytes",
                ),
                _budget_check(
                    "browser_peak_fd",
                    browser_summary.get("peak_fd_count"),
                    budgets.browser_peak_fd_count,
                    "descriptors",
                ),
                _budget_check(
                    "browser_fd_growth",
                    browser_summary.get("peak_minus_baseline_fd_count"),
                    budgets.browser_fd_growth,
                    "descriptors",
                ),
                _budget_check(
                    "browser_fd_sample_coverage",
                    browser_summary.get("fd_sample_coverage"),
                    budgets.minimum_fd_sample_coverage,
                    "ratio",
                    minimum=True,
                ),
                _budget_check(
                    "browser_subprocesses",
                    browser_summary.get("peak_subprocess_count"),
                    budgets.browser_subprocesses,
                    "processes",
                ),
                _budget_check(
                    "browser_active_tabs_minimum",
                    browser_state.get("active_tabs"),
                    configured_browser_tabs,
                    "tabs",
                    minimum=True,
                ),
                _budget_check(
                    "browser_active_tabs_maximum",
                    browser_state.get("active_tabs"),
                    configured_browser_tabs,
                    "tabs",
                ),
                _budget_check(
                    "browser_navigation_latency_p95",
                    browser_state.get("navigation_p95_ms"),
                    budgets.request_p95_ms,
                    "milliseconds",
                ),
                _budget_check(
                    "browser_blocked_external_requests",
                    browser_state.get("blocked_external_requests"),
                    0,
                    "requests",
                ),
            ]
        )
        attempts = browser_state.get("navigation_attempts")
        failures = browser_state.get("navigation_failures")
        browser_failure_rate: float | None = None
        if (
            isinstance(attempts, int)
            and not isinstance(attempts, bool)
            and attempts > 0
            and isinstance(failures, int)
            and not isinstance(failures, bool)
            and failures >= 0
        ):
            browser_failure_rate = failures / attempts
        checks.append(
            _budget_check(
                "browser_navigation_failure_rate",
                browser_failure_rate,
                budgets.failure_rate,
                "ratio",
            )
        )
    return {
        "method": "all_evaluable_metrics_must_satisfy_configured_limits",
        "checks": checks,
        "all_passed": all(check["passed"] for check in checks),
    }


def _wait_for_target_ready(
    url: str,
    timeout_seconds: float,
    *,
    stop_controller: _StopController | None = None,
) -> None:
    deadline = time.monotonic() + min(30.0, max(2.0, timeout_seconds * 3))
    while time.monotonic() < deadline:
        if stop_controller is not None and stop_controller.requested:
            raise SoakFailure("signal", "run_interrupted")
        success, _ = _http_probe(url, min(timeout_seconds, 2.0))
        if success:
            return
        time.sleep(0.1)
    raise SoakFailure("runtime", "target_http_not_ready")


def _runtime_identity(
    config: SoakConfig,
    *,
    target_pid: int,
    target_start_marker: str,
) -> dict[str, Any]:
    module_hash = _sha256_file(Path(__file__).resolve())
    executable_hash = _executable_sha256(target_pid)
    if config.uses_fixture and module_hash is None:
        raise SoakFailure("runtime", "fixture_source_identity_unavailable")
    material = {
        "target_kind": "isolated_fixture"
        if config.uses_fixture
        else "external_loopback",
        "process_scope": (
            "runner_owned_process_group"
            if config.uses_fixture
            else "operator_owned_descendant_tree"
        ),
        "identity_source": (
            "fixture_source_and_root_executable_content"
            if config.uses_fixture
            else "operator_assertion_and_root_executable_content"
        ),
        "fixture_source_sha256": module_hash if config.uses_fixture else None,
        "operator_runtime_id_sha256": (
            None
            if config.uses_fixture
            else sha256(str(config.runtime_id).encode("utf-8")).hexdigest()
        ),
        "root_executable_sha256": executable_hash,
        "root_process_start_sha256": sha256(
            target_start_marker.encode("utf-8")
        ).hexdigest(),
        "runtime_identity_verified": config.uses_fixture
        and executable_hash is not None,
    }
    return {
        "identity_material": material,
        "identity_sha256": canonical_sha256(material),
    }


def run_soak(
    config: SoakConfig,
    *,
    stop_controller: _StopController | None = None,
) -> dict[str, Any]:
    """Run one isolated acceptance and return a content-safe bound receipt."""

    config.validate()
    controller = stop_controller or _StopController()
    registry = _OwnedProcessRegistry()
    run_started_at = _utc_now()
    measurement_started_at: str | None = None
    measurement_ended_at: str | None = None
    measurement_started_monotonic: float | None = None
    measurement_elapsed = 0.0
    failure: SoakFailure | None = None
    cleanup: dict[str, Any] = {
        "owned_process_groups": 0,
        "term_signals_sent": 0,
        "kill_signals_sent": 0,
        "cleanup_failures": 0,
        "all_owned_processes_reaped": True,
        "external_target_signalled": False,
    }
    runtime: dict[str, Any] = {
        "identity_material": {
            "target_kind": "isolated_fixture"
            if config.uses_fixture
            else "external_loopback",
            "identity_source": "not_available_before_runtime_setup",
        }
    }
    runtime["identity_sha256"] = canonical_sha256(runtime["identity_material"])
    sampling_configuration = {
        "process_source": "host_ps_process_table",
        "process_identity": "pid_plus_ps_lstart",
        "fd_sources": ["procfs_fd_inventory", "lsof_fd_inventory"],
        "sample_interval_seconds": float(config.sample_interval_seconds),
        "disk_source": "logical_regular_file_size_without_symlinks",
        "percentile_method": "nearest_rank",
        "synthetic_process_samples_allowed": False,
    }
    target_accumulator = _GroupAccumulator(
        role="target",
        scope=(
            "runner_owned_process_group"
            if config.uses_fixture
            else "operator_owned_descendant_tree"
        ),
    )
    browser_group_accumulator: _GroupAccumulator | None = None
    browser_accumulator = _BrowserAccumulator(config.browser, config.browser_tabs)
    disk_accumulator = _DiskAccumulator(
        root_count=1 if config.uses_fixture else len(config.disk_paths)
    )
    requests = _RequestAccumulator()
    samples = _SampleStream()

    with tempfile.TemporaryDirectory(prefix="teachlab-console-soak-") as temporary:
        control_directory = Path(temporary).resolve()
        os.chmod(control_directory, 0o700)
        target_owned: _OwnedProcess | None = None
        browser_owned: _OwnedProcess | None = None
        browser_state_file = control_directory / "browser-state.json"
        disk_roots: tuple[Path, ...] = ()
        target_url = ""
        workload_url = ""
        target_scope: _ProcessScope | None = None
        browser_scope: _ProcessScope | None = None

        try:
            if config.uses_fixture:
                ready_file = control_directory / "fixture-ready.json"
                fixture_data = control_directory / "fixture-data"
                fixture_data.mkdir(mode=0o700)
                target_owned = registry.spawn(
                    "target",
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--fixture-worker",
                        "--ready-file",
                        str(ready_file),
                        "--data-directory",
                        str(fixture_data),
                    ],
                )
                ready = _wait_for_private_state(
                    ready_file,
                    process=target_owned.process,
                    schema=FIXTURE_READY_SCHEMA,
                    timeout_seconds=10,
                    failure_stage="fixture",
                    stop_controller=controller,
                )
                target_url = _validated_loopback_url(str(ready.get("base_url")))
                if ready.get("root_process_id") != target_owned.process.pid:
                    raise SoakFailure("fixture", "fixture_process_identity_mismatch")
                target_scope = _ProcessScope(
                    "target",
                    target_owned.process.pid,
                    "process_group",
                    target_owned.process_group_id,
                    _root_start_marker(target_owned.process.pid),
                )
                disk_roots = (fixture_data,)
                health_url = _request_url(target_url, "health")
                workload_url = _request_url(target_url, config.request_path)
                _wait_for_target_ready(
                    health_url,
                    config.request_timeout_seconds,
                    stop_controller=controller,
                )
            else:
                target_url = _validated_loopback_url(str(config.target_url))
                workload_url = _request_url(target_url, config.request_path)
                target_scope = _ProcessScope(
                    "target",
                    int(config.target_pid or 0),
                    "descendant_tree",
                    None,
                    _root_start_marker(int(config.target_pid or 0)),
                )
                disk_roots = tuple(
                    _validated_disk_path(path) for path in config.disk_paths
                )
                _wait_for_target_ready(
                    workload_url,
                    config.request_timeout_seconds,
                    stop_controller=controller,
                )

            if target_scope is None:
                raise SoakFailure("runtime", "target_scope_unavailable")
            if target_scope.root_start_marker is None:
                raise SoakFailure("runtime", "root_process_identity_unavailable")
            runtime = _runtime_identity(
                config,
                target_pid=target_scope.root_pid,
                target_start_marker=target_scope.root_start_marker,
            )

            if config.browser is not None:
                browser_control_file = control_directory / "browser-control.json"
                _private_json_write(
                    browser_control_file,
                    {
                        "schema": BROWSER_CONTROL_SCHEMA,
                        "target_url": target_url,
                        "engine": config.browser,
                        "tabs": config.browser_tabs,
                        "refresh_seconds": config.browser_refresh_seconds,
                    },
                )
                browser_owned = registry.spawn(
                    "browser",
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--browser-worker",
                        "--control-file",
                        str(browser_control_file),
                        "--state-file",
                        str(browser_state_file),
                    ],
                )
                initial_browser_state = _wait_for_private_state(
                    browser_state_file,
                    process=browser_owned.process,
                    schema=BROWSER_STATE_SCHEMA,
                    timeout_seconds=60,
                    failure_stage="browser",
                    stop_controller=controller,
                )
                browser_accumulator.add(initial_browser_state)
                if initial_browser_state.get("status") != "running":
                    code = initial_browser_state.get("failure_code")
                    if code not in {
                        "python_playwright_not_installed",
                        "browser_executable_unavailable",
                    }:
                        code = "browser_worker_initialization_failed"
                    raise SoakFailure("browser", str(code))
                browser_scope = _ProcessScope(
                    "browser",
                    browser_owned.process.pid,
                    "process_group",
                    browser_owned.process_group_id,
                    _root_start_marker(browser_owned.process.pid),
                )
                browser_group_accumulator = _GroupAccumulator(
                    role="browser", scope="runner_owned_browser_process_group"
                )

            measurement_started_at = _utc_now()
            measurement_started_monotonic = time.monotonic()

            def take_sample() -> None:
                if target_scope is None or measurement_started_monotonic is None:
                    raise SoakFailure("sampling", "sampling_started_without_target")
                rows = _read_process_table()
                target_sample = _collect_group(rows, target_scope)
                sample_index = samples.count
                safe_groups: dict[str, Any] = {
                    "target": target_accumulator.add(target_sample, sample_index)
                }
                if browser_scope is not None:
                    if browser_group_accumulator is None:
                        raise SoakFailure("sampling", "browser_accumulator_missing")
                    browser_sample = _collect_group(rows, browser_scope)
                    safe_groups["browser"] = browser_group_accumulator.add(
                        browser_sample, sample_index
                    )
                    browser_state = _private_json_read(browser_state_file)
                    if browser_state is not None:
                        browser_accumulator.add(browser_state)
                disk_bytes, disk_files = _tree_size(disk_roots)
                safe_disk = disk_accumulator.add(disk_bytes, disk_files)
                samples.add(
                    {
                        "sample_index": sample_index,
                        "elapsed_ms": round(
                            (time.monotonic() - measurement_started_monotonic) * 1000
                        ),
                        "groups": safe_groups,
                        "disk": safe_disk,
                    }
                )

            take_sample()
            if config.self_test_request_limit is not None:
                for _ in range(config.self_test_request_limit):
                    if controller.requested:
                        break
                    success, latency = _http_probe(
                        workload_url, config.request_timeout_seconds
                    )
                    requests.add(success, latency)
                    take_sample()
            else:
                deadline = measurement_started_monotonic + config.duration_seconds
                next_request = measurement_started_monotonic
                next_sample = (
                    measurement_started_monotonic + config.sample_interval_seconds
                )
                while time.monotonic() < deadline and not controller.requested:
                    if (
                        target_owned is not None
                        and target_owned.process.poll() is not None
                    ):
                        raise SoakFailure("runtime", "target_process_exited_during_run")
                    if (
                        browser_owned is not None
                        and browser_owned.process.poll() is not None
                    ):
                        raise SoakFailure(
                            "browser", "browser_process_exited_during_run"
                        )
                    now = time.monotonic()
                    if now >= next_request:
                        success, latency = _http_probe(
                            workload_url, config.request_timeout_seconds
                        )
                        requests.add(success, latency)
                        next_request = max(
                            next_request + config.request_interval_seconds,
                            time.monotonic(),
                        )
                    now = time.monotonic()
                    if now >= next_sample:
                        take_sample()
                        next_sample = max(
                            next_sample + config.sample_interval_seconds,
                            time.monotonic(),
                        )
                    time.sleep(
                        max(
                            0.0,
                            min(
                                0.05,
                                deadline - time.monotonic(),
                                next_request - time.monotonic(),
                                next_sample - time.monotonic(),
                            ),
                        )
                    )
                if not controller.requested:
                    take_sample()
            if requests.attempts == 0 and not controller.requested:
                raise SoakFailure("workload", "no_requests_executed")
        except SoakFailure as exc:
            failure = exc
        except Exception:
            failure = SoakFailure("runner", "unexpected_runner_failure")
        finally:
            if measurement_started_monotonic is not None:
                measurement_elapsed = time.monotonic() - measurement_started_monotonic
                measurement_ended_at = _utc_now()
            final_browser_state = _private_json_read(browser_state_file)
            if final_browser_state is not None:
                browser_accumulator.add(final_browser_state)
            cleanup = registry.cleanup()

    target_summary = target_accumulator.to_safe_dict()
    browser_group_summary = (
        None
        if browser_group_accumulator is None
        else browser_group_accumulator.to_safe_dict()
    )
    disk_summary = disk_accumulator.to_safe_dict()
    request_summary = requests.to_safe_dict()
    browser_summary = browser_accumulator.to_safe_dict(
        process_group_measured=bool(
            browser_group_summary is not None
            and browser_group_summary.get("sample_count", 0) > 0
        )
    )
    browser_final_state = (
        browser_summary.get("final_state", {})
        if isinstance(browser_summary.get("final_state"), Mapping)
        else {}
    )
    evaluation = _evaluate_budgets(
        budgets=config.budgets,
        request_summary=request_summary,
        target_summary=target_summary,
        disk_summary=disk_summary,
        browser_summary=browser_group_summary,
        browser_state=browser_final_state,
        configured_browser_tabs=config.browser_tabs,
    )
    cleanup_passed = cleanup.get("all_owned_processes_reaped") is True
    interrupted = controller.requested
    passed = bool(
        not interrupted
        and failure is None
        and cleanup_passed
        and evaluation["all_passed"]
        and samples.count > 0
    )
    if interrupted:
        status = "interrupted"
        failure_projection: dict[str, Any] | None = {
            "stage": "signal",
            "code": "run_interrupted",
            "signal": controller.requested_signal,
        }
    elif failure is not None:
        status = "failed"
        failure_projection = {"stage": failure.stage, "code": failure.code}
    elif not cleanup_passed:
        status = "failed"
        failure_projection = {
            "stage": "cleanup",
            "code": "owned_process_cleanup_failed",
        }
    elif not evaluation["all_passed"]:
        status = "failed"
        failure_projection = {"stage": "budget", "code": "resource_budget_exceeded"}
    else:
        status = "passed"
        failure_projection = None

    safe_configuration = config.safe_material()
    budget_limits = config.budgets.to_safe_dict()
    long_soak_executed = long_soak_threshold_reached(measurement_elapsed)
    content: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": status,
        "passed": passed,
        "mode": config.mode,
        "long_soak_threshold_seconds": LONG_SOAK_SECONDS,
        "long_soak_executed": long_soak_executed,
        "long_soak_completed": bool(long_soak_executed and passed),
        "timing": {
            "run_started_at": run_started_at,
            "measurement_started_at": measurement_started_at,
            "measurement_ended_at": measurement_ended_at,
            "actual_measurement_duration_seconds": _round(measurement_elapsed, 6),
            "requested_measurement_duration_seconds": float(config.duration_seconds),
        },
        "configuration": {
            "material": safe_configuration,
            "content_sha256": canonical_sha256(safe_configuration),
        },
        "budgets": {
            "limits": budget_limits,
            "content_sha256": canonical_sha256(budget_limits),
        },
        "runtime": runtime,
        "sampling": {
            "configuration": sampling_configuration,
            "configuration_sha256": canonical_sha256(sampling_configuration),
            "sample_count": samples.count,
            "process_table_reads": samples.process_table_reads,
            "sample_series_sha256": samples.content_sha256,
            "actual_os_process_samples_collected": samples.count > 0,
            "synthetic_process_samples": False,
        },
        "workload": request_summary,
        "resources": {
            "target": target_summary,
            "browser": browser_group_summary,
            "disk": disk_summary,
        },
        "browser": browser_summary,
        "budget_evaluation": evaluation,
        "cleanup": cleanup,
        "privacy": {
            "target_url_emitted": False,
            "process_ids_emitted": False,
            "commands_emitted": False,
            "filesystem_paths_emitted": False,
            "response_bodies_emitted": False,
            "raw_exceptions_emitted": False,
        },
        "claim_boundaries": {
            "reserved_port_3030_targeted": False,
            "fixture_is_real_console_or_harness": False
            if config.uses_fixture
            else None,
            "external_runtime_identity_operator_asserted": not config.uses_fixture,
            "browser_boundary": browser_summary["measurement_boundary"],
            "eight_hour_claim_requires_actual_duration_threshold": True,
        },
    }
    if failure_projection is not None:
        content["failure"] = failure_projection
    return bind_receipt(content)


def _read_target_url_file(path: Path) -> str:
    try:
        expanded = path.expanduser()
        source_info = expanded.lstat()
        if stat.S_ISLNK(source_info.st_mode):
            raise OSError
        resolved = expanded.resolve(strict=True)
        info = resolved.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or (info.st_mode & 0o077) != 0
            or info.st_size > 4096
        ):
            raise OSError
        value = resolved.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise SoakFailure("input", "invalid_target_url_file") from exc
    return _validated_loopback_url(value)


def _write_receipt_file(path: Path, receipt: Mapping[str, Any]) -> None:
    try:
        expanded = path.expanduser()
        if expanded.is_symlink():
            raise OSError
        target = expanded.parent.resolve(strict=False) / expanded.name
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        parent = target.parent.lstat()
        if not stat.S_ISDIR(parent.st_mode) or stat.S_ISLNK(parent.st_mode):
            raise OSError
        if target.exists() or target.is_symlink():
            existing = target.lstat()
            if not stat.S_ISREG(existing.st_mode) or stat.S_ISLNK(existing.st_mode):
                raise OSError
        _private_json_write(target, receipt)
        os.chmod(target, 0o600)
    except OSError as exc:
        raise SoakFailure("receipt", "receipt_write_failed") from exc


def _failure_receipt(stage: str, code: str) -> dict[str, Any]:
    return bind_receipt(
        {
            "schema": RECEIPT_SCHEMA,
            "status": "failed",
            "passed": False,
            "mode": "not_started",
            "long_soak_threshold_seconds": LONG_SOAK_SECONDS,
            "long_soak_executed": False,
            "long_soak_completed": False,
            "failure": {"stage": stage, "code": code},
            "privacy": {
                "target_url_emitted": False,
                "process_ids_emitted": False,
                "commands_emitted": False,
                "filesystem_paths_emitted": False,
                "response_bodies_emitted": False,
                "raw_exceptions_emitted": False,
            },
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a resource-budget smoke or eight-hour soak against an isolated "
            "fixture or an explicit non-3030 loopback Console/Harness target."
        )
    )
    parser.add_argument("--mode", choices=("smoke", "8h"), default="smoke")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--duration-seconds", type=float)
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--target-url",
        help="operator-owned loopback URL; never included in the receipt",
    )
    target.add_argument(
        "--target-url-file",
        type=Path,
        help="private file containing the operator-owned loopback URL",
    )
    parser.add_argument("--target-pid", type=int)
    parser.add_argument(
        "--runtime-id",
        help="non-sensitive release/runtime identity; only its SHA-256 is emitted",
    )
    parser.add_argument(
        "--request-path",
        help="relative GET path (fixture default: work; external default: base URL)",
    )
    parser.add_argument("--disk-path", type=Path, action="append", default=[])
    parser.add_argument("--request-interval-seconds", type=float, default=1.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=5.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--browser", choices=SUPPORTED_BROWSERS)
    parser.add_argument("--browser-tabs", type=int, default=1)
    parser.add_argument("--browser-refresh-seconds", type=float, default=30.0)
    parser.add_argument("--max-request-p95-ms", type=float, default=1_000.0)
    parser.add_argument("--max-failure-rate", type=float, default=0.01)
    parser.add_argument("--max-target-peak-rss-mib", type=float, default=512.0)
    parser.add_argument("--max-target-rss-growth-mib", type=float, default=64.0)
    parser.add_argument("--max-target-peak-fd", type=int, default=512)
    parser.add_argument("--max-target-fd-growth", type=int, default=64)
    parser.add_argument("--max-target-subprocesses", type=int, default=16)
    parser.add_argument("--max-disk-growth-mib", type=float, default=64.0)
    parser.add_argument("--max-browser-peak-rss-mib", type=float, default=1_024.0)
    parser.add_argument("--max-browser-rss-growth-mib", type=float, default=256.0)
    parser.add_argument("--max-browser-peak-fd", type=int, default=2_048)
    parser.add_argument("--max-browser-fd-growth", type=int, default=256)
    parser.add_argument("--max-browser-subprocesses", type=int, default=32)
    parser.add_argument("--minimum-fd-sample-coverage", type=float, default=0.90)
    parser.add_argument("--receipt", type=Path)
    return parser


def _bytes_from_mib(value: float) -> int:
    if not math.isfinite(value) or value < 0:
        raise SoakFailure("input", "invalid_mib_budget")
    return round(value * MIB)


def _config_from_arguments(arguments: argparse.Namespace) -> SoakConfig:
    if arguments.self_test and (
        arguments.target_url is not None
        or arguments.target_url_file is not None
        or arguments.target_pid is not None
        or arguments.runtime_id is not None
        or arguments.disk_path
        or arguments.browser is not None
        or arguments.duration_seconds is not None
        or arguments.mode != "smoke"
    ):
        raise SoakFailure("input", "self_test_rejects_runtime_overrides")
    target_url = arguments.target_url
    if arguments.target_url_file is not None:
        target_url = _read_target_url_file(arguments.target_url_file)
    uses_fixture = target_url is None
    if arguments.self_test:
        mode = "self-test"
        duration = 1.0
        request_interval = 0.01
        sample_interval = 0.01
        timeout = 2.0
        request_limit = 6
        request_path = "work"
    else:
        mode = arguments.mode
        if mode == "8h":
            if arguments.duration_seconds is not None:
                raise SoakFailure("input", "long_mode_rejects_duration_override")
            duration = float(LONG_SOAK_SECONDS)
        else:
            duration = float(
                60.0
                if arguments.duration_seconds is None
                else arguments.duration_seconds
            )
        request_interval = arguments.request_interval_seconds
        sample_interval = arguments.sample_interval_seconds
        timeout = arguments.request_timeout_seconds
        request_limit = None
        request_path = (
            ("work" if uses_fixture else "")
            if arguments.request_path is None
            else arguments.request_path
        )
    budgets = BudgetConfig(
        request_p95_ms=arguments.max_request_p95_ms,
        failure_rate=arguments.max_failure_rate,
        target_peak_rss_bytes=_bytes_from_mib(arguments.max_target_peak_rss_mib),
        target_rss_growth_bytes=_bytes_from_mib(arguments.max_target_rss_growth_mib),
        target_peak_fd_count=arguments.max_target_peak_fd,
        target_fd_growth=arguments.max_target_fd_growth,
        target_subprocesses=arguments.max_target_subprocesses,
        disk_growth_bytes=_bytes_from_mib(arguments.max_disk_growth_mib),
        browser_peak_rss_bytes=_bytes_from_mib(arguments.max_browser_peak_rss_mib),
        browser_rss_growth_bytes=_bytes_from_mib(arguments.max_browser_rss_growth_mib),
        browser_peak_fd_count=arguments.max_browser_peak_fd,
        browser_fd_growth=arguments.max_browser_fd_growth,
        browser_subprocesses=arguments.max_browser_subprocesses,
        minimum_fd_sample_coverage=arguments.minimum_fd_sample_coverage,
    )
    return SoakConfig(
        mode=mode,
        duration_seconds=duration,
        request_interval_seconds=request_interval,
        sample_interval_seconds=sample_interval,
        request_timeout_seconds=timeout,
        target_url=target_url,
        target_pid=arguments.target_pid,
        runtime_id=arguments.runtime_id,
        request_path=request_path,
        disk_paths=tuple(arguments.disk_path),
        browser=arguments.browser,
        browser_tabs=arguments.browser_tabs,
        browser_refresh_seconds=arguments.browser_refresh_seconds,
        self_test_request_limit=request_limit,
        budgets=budgets,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    controller = _StopController()
    previous_handlers: dict[int, Any] = {}
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signal_number] = signal.getsignal(signal_number)
        signal.signal(
            signal_number,
            lambda received, _frame, control=controller: control.request(received),
        )
    try:
        try:
            config = _config_from_arguments(arguments)
            receipt = run_soak(config, stop_controller=controller)
        except SoakFailure as exc:
            receipt = _failure_receipt(exc.stage, exc.code)
        if arguments.receipt is not None:
            try:
                _write_receipt_file(arguments.receipt, receipt)
            except SoakFailure as exc:
                receipt = _failure_receipt(exc.stage, exc.code)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    finally:
        for signal_number, handler in previous_handlers.items():
            signal.signal(signal_number, handler)
    if controller.requested_signal == "SIGINT":
        return 130
    if controller.requested_signal == "SIGTERM":
        return 143
    if receipt.get("passed") is True:
        return 0
    return 2 if receipt.get("failure", {}).get("stage") == "input" else 1


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--fixture-worker":
        raise SystemExit(_fixture_worker(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "--browser-worker":
        raise SystemExit(_browser_worker(sys.argv[2:]))
    raise SystemExit(main())
