#!/usr/bin/env python3
"""Smoke-test a sealed Teaching Console production runtime in real browsers.

This runner starts only its own random-port loopback processes: a deterministic
fake Teaching Agent backend and the already-built Next standalone server.  It
never uses the developer server, never opens a visible browser window, and
never targets the normal Console port (3030).

The Playwright browsers exercise the web runtime on the current host.  They do
not certify the native macOS .app/DMG, codesigning, or notarization layers.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import select
import signal
import socket
import socketserver
import struct
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_ROOT = ROOT / ".private" / "console-runtime"
AXE_CORE_PATH = ROOT / "apps" / "console" / "node_modules" / "axe-core" / "axe.min.js"
RECEIPT_SCHEMA = "teachlab.console.cross_browser_smoke.v1"
SUPPORTED_BROWSERS = ("chromium", "firefox", "webkit")
PROJECT_ID = "project_000000000000000000000001"
TIMESTAMP = "2026-01-01T00:00:00Z"
_MAX_FIREFOX_OFFLINE_NETWORK_DIAGNOSTICS = 64


class SmokeFailure(RuntimeError):
    """A fixed-code failure that is safe to print in a CI receipt."""

    def __init__(self, stage: str, code: str, browser: str | None = None) -> None:
        super().__init__(code)
        self.stage = stage
        self.code = code
        self.browser = browser


@dataclass(frozen=True, slots=True)
class RuntimeRelease:
    release_id: str
    version: str
    directory: Path


@dataclass(slots=True)
class _BrowserErrorGate:
    """Keep online failures distinct from one intentional Firefox network cut."""

    browser: str
    production_origin: str
    phase: str = "online"
    online_page_errors: int = 0
    online_console_errors: int = 0
    offline_page_errors: int = 0
    offline_console_errors: int = 0
    accepted_firefox_offline_diagnostics: int = 0
    first_unexpected_offline_console_source: str = "none"

    def on_page_error(self, _error: object) -> None:
        if self.phase == "online":
            self.online_page_errors += 1
        else:
            self.offline_page_errors += 1

    def on_console(self, message: Any) -> None:
        if getattr(message, "type", "") != "error":
            return
        if self.phase == "online":
            self.online_console_errors += 1
            return
        source = self._console_error_source(message)
        # The only tolerated browser diagnostic is attributed to the sealed
        # worker while the test itself has deliberately severed its transport.
        # Raw/localized console text is never retained or emitted.
        if (
            self.accepted_firefox_offline_diagnostics
            < _MAX_FIREFOX_OFFLINE_NETWORK_DIAGNOSTICS
            and self.browser == "firefox"
            and self.phase == "intentional_offline_navigation"
            and source == "sealed_worker"
        ):
            self.accepted_firefox_offline_diagnostics += 1
            return
        if self.offline_console_errors == 0:
            self.first_unexpected_offline_console_source = (
                "diagnostic_limit"
                if source == "sealed_worker"
                and self.accepted_firefox_offline_diagnostics
                >= _MAX_FIREFOX_OFFLINE_NETWORK_DIAGNOSTICS
                else source
            )
        self.offline_console_errors += 1

    def require_online_clean(self) -> None:
        _require(
            self.online_page_errors == 0,
            stage="browser_online",
            code="page_error",
            browser=self.browser,
        )
        _require(
            self.online_console_errors == 0,
            stage="browser_online",
            code="console_error",
            browser=self.browser,
        )

    def begin_intentional_offline_navigation(self) -> None:
        _require(
            self.phase == "online",
            stage="browser_offline",
            code="invalid_error_gate_phase",
            browser=self.browser,
        )
        # The caller opens this phase immediately before the deliberate
        # transport cut, after proving the online phase is clean.
        self.require_online_clean()
        self.phase = "intentional_offline_navigation"

    def complete_intentional_offline_navigation(self) -> None:
        _require(
            self.phase == "intentional_offline_navigation",
            stage="browser_offline",
            code="invalid_error_gate_phase",
            browser=self.browser,
        )
        _require(
            self.offline_page_errors == 0,
            stage="browser_offline",
            code="page_error",
            browser=self.browser,
        )
        _require(
            self.offline_console_errors == 0,
            stage="browser_offline",
            code=(
                "navigation_console_error_"
                f"{self.first_unexpected_offline_console_source}"
            ),
            browser=self.browser,
        )
        # The caller invokes this immediately after proving the fixed offline
        # document and account-scoped snapshot rendered. From this point on,
        # even Firefox diagnostics attributed to the worker are unexpected.
        self.phase = "offline_shell_verified"

    def require_offline_clean(self) -> None:
        _require(
            self.phase == "offline_shell_verified",
            stage="browser_offline",
            code="invalid_error_gate_phase",
            browser=self.browser,
        )
        _require(
            self.offline_page_errors == 0,
            stage="browser_offline",
            code="page_error",
            browser=self.browser,
        )
        _require(
            self.offline_console_errors == 0,
            stage="browser_offline",
            code=(
                "verified_shell_console_error_"
                f"{self.first_unexpected_offline_console_source}"
            ),
            browser=self.browser,
        )

    def _console_error_source(self, message: Any) -> str:
        """Return a fixed, non-sensitive source class for CI diagnosis."""

        try:
            location = message.location
            location_url = str(location.get("url", ""))
            if not location_url:
                return "missing_location"
            expected_origin = urlsplit(self.production_origin)
            observed = urlsplit(location_url)
            same_origin = (
                observed.scheme == expected_origin.scheme == "http"
                and observed.hostname == expected_origin.hostname == "127.0.0.1"
                and observed.port == expected_origin.port
            )
        except (AttributeError, TypeError, ValueError):
            return "invalid_location"
        if not same_origin:
            return "other_origin"
        if (
            observed.path == "/teachlab-sw-v1.js"
            and not observed.query
            and not observed.fragment
        ):
            return "sealed_worker"
        if observed.path == "/teachlab-offline-shell-v1.js":
            return "offline_shell_script"
        if observed.path.startswith("/_next/static/"):
            return "next_static_asset"
        if observed.path in {"", "/"}:
            return "navigation_document"
        return "same_origin_other"


def _require(
    condition: bool,
    *,
    stage: str,
    code: str,
    browser: str | None = None,
) -> None:
    if not condition:
        raise SmokeFailure(stage, code, browser)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SmokeFailure("runtime", "invalid_runtime_metadata") from exc
    if not isinstance(value, dict):
        raise SmokeFailure("runtime", "invalid_runtime_metadata")
    return value


def _sealed_runtime(runtime_root: Path) -> RuntimeRelease:
    channel_path = runtime_root / "channel.json"
    _require(channel_path.is_file() and not channel_path.is_symlink(), stage="runtime", code="missing_channel")
    channel = _read_json(channel_path)
    release_id = channel.get("current")
    _require(
        channel.get("schema") == "teachlab.console.channel.v1"
        and isinstance(release_id, str)
        and release_id
        and Path(release_id).name == release_id,
        stage="runtime",
        code="invalid_channel",
    )
    directory = runtime_root / "releases" / release_id
    manifest_path = directory / "runtime-manifest.json"
    server_path = directory / "server.js"
    _require(
        directory.is_dir()
        and not directory.is_symlink()
        and manifest_path.is_file()
        and not manifest_path.is_symlink()
        and server_path.is_file()
        and not server_path.is_symlink(),
        stage="runtime",
        code="missing_sealed_release",
    )
    manifest = _read_json(manifest_path)
    _require(
        manifest.get("schema") == "teachlab.console.runtime.v1"
        and manifest.get("release_id") == release_id
        and channel.get("current_manifest_sha256") == _sha256(manifest_path),
        stage="runtime",
        code="runtime_manifest_mismatch",
    )
    version = manifest.get("version")
    _require(isinstance(version, str) and bool(version), stage="runtime", code="invalid_runtime_version")
    return RuntimeRelease(release_id=release_id, version=version, directory=directory)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _free_loopback_port() -> int:
    for _ in range(20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            candidate.bind(("127.0.0.1", 0))
            port = int(candidate.getsockname()[1])
        if port != 3030:
            return port
    raise SmokeFailure("runtime", "ephemeral_port_unavailable")


def _abort_socket(connection: socket.socket) -> None:
    """Close a TCP stream with a reset so browser fetch() observes failure."""

    try:
        connection.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_LINGER,
            struct.pack("ii", 1, 0),
        )
    except OSError:
        pass
    try:
        connection.close()
    except OSError:
        pass


class _OriginGateHandler(socketserver.BaseRequestHandler):
    """Relay one browser connection until the smoke test cuts the origin."""

    def handle(self) -> None:
        gate = self.server
        if not isinstance(gate, _OriginGateServer):
            return
        client = self.request
        if not isinstance(client, socket.socket):
            return
        if not gate.track(client):
            gate.reject_after_accept(client)
            return
        upstream: socket.socket | None = None
        try:
            try:
                upstream = socket.create_connection(gate.upstream_address, timeout=2)
                upstream.settimeout(None)
            except OSError:
                return
            if not gate.track(upstream):
                return
            streams = (client, upstream)
            while not gate.disconnected.is_set():
                try:
                    readable, _, exceptional = select.select(streams, (), streams, 0.1)
                except (OSError, ValueError):
                    return
                if gate.disconnected.is_set() or exceptional:
                    return
                for source in readable:
                    if gate.disconnected.is_set():
                        return
                    target = upstream if source is client else client
                    try:
                        chunk = source.recv(64 * 1024)
                        if not chunk:
                            return
                        target.sendall(chunk)
                    except OSError:
                        return
        finally:
            if upstream is not None:
                gate.untrack(upstream)
                try:
                    upstream.close()
                except OSError:
                    pass
            gate.untrack(client)
            if gate.disconnected.is_set():
                _abort_socket(client)


class _OriginGateServer(socketserver.ThreadingTCPServer):
    """Same-origin TCP relay that can create a deterministic network cut."""

    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 64

    def __init__(self, server_address: tuple[str, int]) -> None:
        super().__init__(server_address, _OriginGateHandler)
        self.upstream_address = ("127.0.0.1", 0)
        self.disconnected = threading.Event()
        self._active_lock = threading.Lock()
        self._active: set[socket.socket] = set()

    def track(self, connection: socket.socket) -> bool:
        with self._active_lock:
            if self.disconnected.is_set():
                should_abort = True
            else:
                self._active.add(connection)
                should_abort = False
        if should_abort:
            return False
        return True

    def untrack(self, connection: socket.socket) -> None:
        with self._active_lock:
            self._active.discard(connection)

    @staticmethod
    def reject_after_accept(connection: socket.socket) -> None:
        """Let connect() succeed, then fail the first offline HTTP exchange."""

        try:
            connection.settimeout(0.25)
            connection.recv(64 * 1024)
        except OSError:
            pass
        _abort_socket(connection)

    def disconnect(self) -> None:
        with self._active_lock:
            if self.disconnected.is_set():
                return
            self.disconnected.set()
            active = tuple(self._active)
            self._active.clear()
        for connection in active:
            _abort_socket(connection)


def _project() -> dict[str, Any]:
    return {
        "schema": "teaching_skill_miner.learning_project.v1",
        "project_id": PROJECT_ID,
        "title": "Cross-browser smoke project",
        "description": "Deterministic production-runtime fixture",
        "status": "active",
        "pinned": True,
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
        "syllabus_ids": [],
        "teaching_session_ids": [],
        "resource_ids": [],
        "chat_threads": [],
        "notes": [],
        "claim_boundary": {},
    }


def _project_summary() -> dict[str, Any]:
    return {
        "project_id": PROJECT_ID,
        "title": "Cross-browser smoke project",
        "description": "Deterministic production-runtime fixture",
        "status": "active",
        "pinned": True,
        "updated_at": TIMESTAMP,
        "syllabus_count": 0,
        "teaching_session_count": 0,
        "resource_count": 0,
        "chat_thread_count": 0,
        "note_count": 0,
    }


def _trashed_project() -> dict[str, Any]:
    return {
        "project_id": "project_000000000000000000000099",
        "title": "Deletion focus fixture",
        "updated_at": TIMESTAMP,
        "recovery_token": "recovery_fixture_token",
    }


def _bootstrap() -> dict[str, Any]:
    return {
        "schema_version": "1.1",
        "dashboard_kind": "loopback_interactive_teacher_agent",
        "mode": "cross_browser_smoke",
        "provider_status": {
            "provider": "deterministic_smoke",
            "model": None,
            "configured": False,
            "web_search_supported": False,
        },
        "remote_consent": {
            "configured": False,
            "server_minted_receipts_required": True,
            "legacy_browser_receipts_authoritative": False,
            "grant_list_revoke_enabled": False,
            "purposes": [],
            "policies": [],
        },
        "account_data_rights": {
            "mode": "local_only_no_account_authority",
            "recent_auth_required": False,
            "remote_provider_copies_deleted": False,
        },
        "visual_semantics": {
            "available": False,
            "provider_id": None,
            "processing_region": None,
            "sends_raw_media_remotely": False,
        },
        "interaction_contract": {"remote_processing_server_consent_required": False},
        "skills": [],
        "agent_runtime_policy": {
            "agent_loop_enabled": True,
            "maximum_agent_steps": 4,
            "maximum_agent_tool_calls_per_step": 2,
            "recoverable_context": True,
            "structured_tool_allowlist": True,
        },
        "default_goal": {"concept": "smoke", "max_rounds": 4},
        "default_student_profile": {},
    }


class _FixtureServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], capability: str) -> None:
        super().__init__(address, _FixtureHandler)
        self.capability = capability
        self.routes: list[str] = []
        self.routes_lock = threading.Lock()

    def record(self, route: str) -> None:
        with self.routes_lock:
            self.routes.append(route)


class _FixtureHandler(BaseHTTPRequestHandler):
    server: _FixtureServer

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _route(self) -> str | None:
        prefix = f"/{self.server.capability}/"
        path = urlsplit(self.path).path
        if not path.startswith(prefix):
            return None
        route = path.removeprefix(prefix).strip("/")
        self.server.record(route)
        return route

    def _json(self, status: int, payload: object) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        route = self._route()
        if route == "api/bootstrap":
            self._json(200, _bootstrap())
        elif route == "api/projects":
            self._json(200, {"projects": [_project_summary()]})
        elif route == "api/projects/trash":
            self._json(200, {"projects": [_trashed_project()]})
        elif route == f"api/projects/{PROJECT_ID}":
            self._json(200, {"project": _project()})
        else:
            self._json(404, {"error": "resource not found"})

    def do_POST(self) -> None:  # noqa: N802
        route = self._route()
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length:
            self.rfile.read(min(length, 1024 * 1024))
        if route == "api/projects/bootstrap":
            self._json(
                200,
                {
                    "project": _project(),
                    "created_or_replayed": True,
                    "legacy_migration_applied_or_replayed": False,
                    "stale_teaching_session_ids": [],
                },
            )
        elif route == "api/tasks/list":
            # Deliberately invalid confidentiality marker: the UI must surface
            # an in-dialog error without relying on a transport-level 404.
            self._json(200, {"schema": "fixture.invalid", "tasks": [], "content_included": True})
        else:
            self._json(404, {"error": "resource not found"})


@contextmanager
def _fixture_backend() -> Iterator[tuple[_FixtureServer, str]]:
    capability = f"capability_{secrets.token_urlsafe(30)}"
    server = _FixtureServer(("127.0.0.1", _free_loopback_port()), capability)
    thread = threading.Thread(target=server.serve_forever, name="console-smoke-backend", daemon=True)
    thread.start()
    try:
        port = int(server.server_address[1])
        yield server, f"http://127.0.0.1:{port}/{capability}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _http_json(url: str, *, expected_status: int) -> dict[str, Any]:
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(url, timeout=2) as response:
            status = int(response.status)
            content_type = response.headers.get_content_type()
            payload = json.loads(response.read(64 * 1024))
    except HTTPError as exc:
        status = exc.code
        content_type = exc.headers.get_content_type()
        try:
            payload = json.loads(exc.read(64 * 1024))
        except ValueError:
            payload = {}
    except (OSError, URLError, ValueError) as exc:
        raise SmokeFailure("runtime", "runtime_http_unavailable") from exc
    _require(status == expected_status, stage="runtime", code="unexpected_http_status")
    _require(content_type == "application/json", stage="runtime", code="unexpected_http_content_type")
    _require(isinstance(payload, dict), stage="runtime", code="unexpected_http_payload")
    return payload


@contextmanager
def _production_server(
    runtime: RuntimeRelease,
    capability_url: str,
) -> Iterator[tuple[str, Callable[[], None]]]:
    # The public port remains bound for the full browser run.  A separate
    # internal Next port lets the test cut network transport while the service
    # worker's public origin address remains bound for the navigation attempt.
    origin_gate = _OriginGateServer(("127.0.0.1", 0))
    public_port = int(origin_gate.server_address[1])
    internal_port = _free_loopback_port()
    origin_gate.upstream_address = ("127.0.0.1", internal_port)
    gate_thread = threading.Thread(
        target=origin_gate.serve_forever,
        name="console-smoke-origin-gate",
        daemon=True,
    )
    base_url = f"http://127.0.0.1:{public_port}"
    environment = {
        **os.environ,
        "NODE_ENV": "production",
        "HOSTNAME": "127.0.0.1",
        "PORT": str(internal_port),
        "TEACHLAB_HARNESS_MODE": "local_python",
        "TEACHER_AGENT_CAPABILITY_URL": capability_url,
        "TEACHLAB_LOCAL_SECURITY_SECRET": secrets.token_hex(32),
        "TEACHLAB_RELEASE_ID": runtime.release_id,
        "TEACHLAB_RELEASE_VERSION": runtime.version,
    }
    process: subprocess.Popen[bytes] | None = None

    def disconnect_origin() -> None:
        origin_gate.disconnect()

    def stop_process() -> None:
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)

    try:
        process = subprocess.Popen(
            ["node", str(runtime.directory / "server.js")],
            cwd=runtime.directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        gate_thread.start()
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise SmokeFailure("runtime", "production_server_exited")
            try:
                health = _http_json(f"{base_url}/health", expected_status=200)
                ready = _http_json(f"{base_url}/ready", expected_status=200)
                if (
                    health.get("status") == "healthy"
                    and health.get("release_id") == runtime.release_id
                    and ready.get("status") == "ready"
                    and ready.get("backend") == "ready"
                ):
                    break
            except SmokeFailure:
                time.sleep(0.2)
        else:
            raise SmokeFailure("runtime", "production_server_not_ready")
        yield base_url, disconnect_origin
    finally:
        disconnect_origin()
        if gate_thread.is_alive():
            origin_gate.shutdown()
        origin_gate.server_close()
        if gate_thread.ident is not None:
            gate_thread.join(timeout=5)
        stop_process()


def _axe_audit(page: Any, axe_source: str, state: str, browser_name: str) -> dict[str, Any]:
    """Run the real axe engine in the current document and fail on P1 impact."""

    page.evaluate(axe_source)
    result = page.evaluate(
        """async () => {
          if (!globalThis.axe || typeof globalThis.axe.run !== 'function') {
            throw new Error('axe_not_loaded');
          }
          const report = await globalThis.axe.run(document, {
            runOnly: {type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa']},
            resultTypes: ['violations']
          });
          const blocking = report.violations.filter((item) =>
            item.impact === 'serious' || item.impact === 'critical'
          );
          return {
            version: globalThis.axe.version,
            blocking: blocking.map((item) => ({
              id: item.id,
              impact: item.impact,
              nodes: item.nodes.length
            }))
          };
        }"""
    )
    _require(
        isinstance(result, dict)
        and isinstance(result.get("version"), str)
        and result.get("blocking") == [],
        stage="axe",
        code=f"axe_serious_or_critical_{state}",
        browser=browser_name,
    )
    return {"state": state, "axe_version": result["version"], "serious_critical": 0}


def _reflow_check(page: Any, width: int, label: str, browser_name: str) -> dict[str, Any]:
    page.set_viewport_size({"width": width, "height": 844})
    page.wait_for_timeout(100)
    measurement = page.evaluate(
        """() => ({
          viewport: innerWidth,
          documentWidth: document.documentElement.scrollWidth,
          bodyWidth: document.body.scrollWidth
        })"""
    )
    _require(
        measurement["documentWidth"] <= measurement["viewport"] + 1
        and measurement["bodyWidth"] <= measurement["viewport"] + 1,
        stage="reflow",
        code=f"horizontal_overflow_{label}",
        browser=browser_name,
    )
    composer = page.get_by_role("textbox", name="输入消息")
    box = composer.bounding_box()
    _require(
        box is not None and box["x"] >= 0 and box["x"] + box["width"] <= width + 1,
        stage="reflow",
        code=f"composer_outside_viewport_{label}",
        browser=browser_name,
    )
    return {"state": label, "viewport_width": width, "horizontal_overflow": False}


def _seed_offline_snapshot(page: Any) -> None:
    scope = f"acs1_{'s' * 43}"
    project = _project()
    project["chat_threads"] = [{
        "thread_id": "chat_000000000000000000000001",
        "title": "Offline fixture thread",
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
        "messages": [{
            "message_id": "message_000000000000000000000001",
            "role": "assistant",
            "content": "Scope-bound offline snapshot sentinel",
            "status": "completed",
            "created_at": TIMESTAMP,
            "web_search_used": False,
            "sources": [],
        }],
    }]
    page.evaluate(
        """async ({scope, project}) => {
          localStorage.setItem('teachlab.account-cache-scope.v1', scope);
          localStorage.setItem('teachlab.learning-project.active-id', project.project_id);
          const database = await new Promise((resolve, reject) => {
            const request = indexedDB.open('teachlab-console-runtime', 2);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
          });
          const serializedChars = JSON.stringify(project).length;
          const savedAt = Date.now();
          await new Promise((resolve, reject) => {
            const transaction = database.transaction('workspaces', 'readwrite');
            transaction.objectStore('workspaces').put({
              schema: 'teachlab.offline_workspace_snapshot.v2',
              accountCacheScope: scope,
              projectId: project.project_id,
              projectUpdatedAt: project.updated_at,
              project,
              savedAt,
              expiresAt: savedAt + 7 * 24 * 60 * 60 * 1000,
              serializedChars
            });
            transaction.oncomplete = resolve;
            transaction.onerror = () => reject(transaction.error);
          });
          database.close();
        }""",
        {"scope": scope, "project": project},
    )


def _browser_smoke(
    playwright: Any,
    browser_name: str,
    base_url: str,
    disconnect_production_origin: Callable[[], None],
) -> dict[str, Any]:
    try:
        axe_source = AXE_CORE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise SmokeFailure("dependency", "axe_core_not_installed", browser_name) from exc
    browser_type = getattr(playwright, browser_name)
    try:
        browser = browser_type.launch(headless=True)
    except Exception as exc:
        raise SmokeFailure("browser_launch", "browser_unavailable", browser_name) from exc
    try:
        context = browser.new_context(viewport={"width": 1280, "height": 800}, locale="zh-CN")
        page = context.new_page()
    except Exception as exc:
        browser.close()
        raise SmokeFailure("browser_launch", "browser_context_unavailable", browser_name) from exc
    error_gate = _BrowserErrorGate(browser_name, base_url)
    external_requests = 0

    def on_page_error(error: object) -> None:
        error_gate.on_page_error(error)

    def on_console(message: Any) -> None:
        error_gate.on_console(message)

    def guard_route(route: Any) -> None:
        target = urlsplit(route.request.url)
        if target.scheme not in {"http", "https"} or target.hostname != "127.0.0.1" or target.port != urlsplit(base_url).port:
            route.abort()
        else:
            route.continue_()

    def observe_request(request: Any) -> None:
        nonlocal external_requests
        target = urlsplit(request.url)
        if target.scheme not in {"http", "https"} or target.hostname != "127.0.0.1" or target.port != urlsplit(base_url).port:
            external_requests += 1

    page.on("pageerror", on_page_error)
    page.on("console", on_console)
    page.on("request", observe_request)
    page.route("**/*", guard_route)
    try:
        response = page.goto(base_url, wait_until="domcontentloaded", timeout=30_000)
        _require(response is not None and response.status == 200, stage="root", code="root_not_200", browser=browser_name)
        _require(page.locator("html").get_attribute("lang") == "zh-CN", stage="a11y", code="document_language_missing", browser=browser_name)
        _require(page.get_by_role("main").count() == 1, stage="a11y", code="main_landmark_missing", browser=browser_name)

        chat_log = page.get_by_role("log", name="对话记录")
        chat_log.wait_for(state="visible", timeout=20_000)
        work_modes = page.get_by_role("group", name="工作模式")
        work_modes.wait_for(state="visible", timeout=20_000)
        chat_button = work_modes.get_by_role("button", name="Chat", exact=True)
        teach_button = work_modes.get_by_role("button", name="Teach", exact=True)
        _require(chat_button.get_attribute("aria-pressed") == "true", stage="controls", code="chat_not_selected", browser=browser_name)
        composer = page.get_by_role("textbox", name="输入消息")
        composer.wait_for(state="visible", timeout=20_000)
        deadline = time.monotonic() + 20
        while composer.is_disabled() and time.monotonic() < deadline:
            page.wait_for_timeout(100)
        _require(not composer.is_disabled(), stage="controls", code="composer_not_ready", browser=browser_name)
        security = page.evaluate(
            """async () => {
              const session = await fetch('/api/teacher-agent/security/session', {
                credentials: 'include', cache: 'no-store'
              });
              const sessionBody = await session.json();
              const bootstrap = await fetch('/api/teacher-agent/api/bootstrap', {
                credentials: 'include', cache: 'no-store'
              });
              const bootstrapBody = await bootstrap.json();
              return {
                sessionStatus: session.status,
                csrfShape: typeof sessionBody.csrf_token === 'string'
                  && /^[A-Za-z0-9_-]{43}$/.test(sessionBody.csrf_token),
                bootstrapStatus: bootstrap.status,
                schemaVersion: bootstrapBody.schema_version,
                dashboardKind: bootstrapBody.dashboard_kind,
                cookieVisible: document.cookie.includes('teachlab_local_session=')
              };
            }"""
        )
        _require(
            security == {
                "sessionStatus": 200,
                "csrfShape": True,
                "bootstrapStatus": 200,
                "schemaVersion": "1.1",
                "dashboardKind": "loopback_interactive_teacher_agent",
                "cookieVisible": False,
            },
            stage="security",
            code="session_or_bootstrap_contract_failed",
            browser=browser_name,
        )

        cookies = context.cookies([f"{base_url}/api/teacher-agent/security/session"])
        local_session = next((item for item in cookies if item.get("name") == "teachlab_local_session"), None)
        _require(
            local_session is not None
            and local_session.get("httpOnly") is True
            and local_session.get("sameSite") == "Strict",
            stage="security",
            code="session_cookie_attributes_failed",
            browser=browser_name,
        )

        teach_button.click()
        _require(teach_button.get_attribute("aria-pressed") == "true", stage="controls", code="teach_not_selected", browser=browser_name)
        page.get_by_role("log", name="教学对话记录").wait_for(state="visible", timeout=10_000)
        page.get_by_role("textbox", name="输入学习目标或教学命令").wait_for(state="visible", timeout=10_000)
        chat_button.click()
        _require(chat_button.get_attribute("aria-pressed") == "true", stage="controls", code="chat_restore_failed", browser=browser_name)

        axe_results = [_axe_audit(page, axe_source, "workspace", browser_name)]

        # Roving project menu: keyboard entry, movement, Escape and deterministic
        # restoration to the owning trigger.
        project_trigger = page.get_by_role("button", name="选择学习项目")
        project_trigger.focus()
        page.keyboard.press("Enter")
        project_menu = page.get_by_role("menu", name="学习项目")
        project_menu.wait_for(state="visible", timeout=5_000)
        first_project = project_menu.locator('[data-roving-menuitem="true"]').first
        first_project.focus()
        page.keyboard.press("ArrowDown")
        _require(
            page.evaluate("document.activeElement?.getAttribute('role')") == "menuitem",
            stage="focus",
            code="project_menu_roving_focus_failed",
            browser=browser_name,
        )
        page.keyboard.press("Escape")
        _require(
            project_trigger.evaluate("element => document.activeElement === element"),
            stage="focus",
            code="project_menu_focus_restore_failed",
            browser=browser_name,
        )

        # A real destructive prompt is dismissed, proving cancellation performs
        # no mutation and returns focus to the permanent-delete control.
        project_trigger.press("Enter")
        permanent_delete = project_menu.locator(
            '[aria-label="永久删除 Deletion focus fixture"]'
        )
        prompt_seen = {"value": False}

        def dismiss_delete_prompt(dialog: Any) -> None:
            _require(
                dialog.type == "prompt"
                and "PERMANENTLY DELETE project_000000000000000000000099" in dialog.message,
                stage="focus",
                code="deletion_confirmation_contract_failed",
                browser=browser_name,
            )
            prompt_seen["value"] = True
            dialog.dismiss()

        page.once("dialog", dismiss_delete_prompt)
        permanent_delete.focus()
        permanent_delete.press("Enter")
        focus_restore_deadline = time.monotonic() + 2
        while (
            not permanent_delete.evaluate("element => document.activeElement === element")
            and time.monotonic() < focus_restore_deadline
        ):
            page.wait_for_timeout(25)
        _require(
            prompt_seen["value"]
            and permanent_delete.evaluate("element => document.activeElement === element"),
            stage="focus",
            code="deletion_confirmation_focus_restore_failed",
            browser=browser_name,
        )
        page.keyboard.press("Escape")

        # Radix command dialog: keyboard invocation, trapped initial focus,
        # surfaced error alert and focus restoration to the composer.
        composer.focus()
        page.keyboard.press("Control+K")
        command_dialog = page.get_by_role("dialog", name="命令与后台任务中心")
        command_dialog.wait_for(state="visible", timeout=5_000)
        command_search = page.get_by_role("textbox", name="搜索工作区命令")
        _require(
            command_search.evaluate("element => document.activeElement === element"),
            stage="focus",
            code="command_dialog_initial_focus_failed",
            browser=browser_name,
        )
        page.get_by_role("alert").wait_for(state="visible", timeout=5_000)
        _require(
            command_search.evaluate("element => document.activeElement === element"),
            stage="focus",
            code="error_alert_stole_focus",
            browser=browser_name,
        )
        axe_results.append(_axe_audit(page, axe_source, "command_error_dialog", browser_name))
        page.keyboard.press("Escape")
        page.wait_for_timeout(50)
        _require(
            composer.evaluate("element => document.activeElement === element"),
            stage="focus",
            code="command_dialog_focus_restore_failed",
            browser=browser_name,
        )

        # At 390px the Inspector is a modal dialog. It must focus its close
        # button, expose Consent by keyboard, trap Tab, and restore its trigger.
        page.set_viewport_size({"width": 390, "height": 844})
        # Let the Workbench and Inspector matchMedia listeners commit their
        # compact-layout state before opening the drawer. Otherwise the resize
        # handler can race the keyboard activation and close the drawer again.
        page.wait_for_timeout(100)
        inspector_trigger = page.get_by_role("button", name="展开右侧检查器")
        inspector_trigger.focus()
        inspector_trigger.press("Enter")
        inspector = page.get_by_role("dialog", name="会话检查器")
        inspector.wait_for(state="visible", timeout=5_000)
        inspector_close = page.get_by_role("button", name="收起右侧栏")
        _require(
            inspector_close.evaluate("element => document.activeElement === element"),
            stage="focus",
            code="inspector_initial_focus_failed",
            browser=browser_name,
        )
        consent_tab = inspector.get_by_role("button", name="同意中心")
        consent_tab.focus()
        consent_tab.press("Enter")
        inspector.get_by_role("heading", name="同意中心").wait_for(state="visible", timeout=5_000)
        # Exercise the actual wrap boundary. WebKit may omit intermediate
        # buttons from native tab traversal unless the host enables full
        # keyboard access, but the modal must still intercept Shift+Tab from
        # its first focusable control and wrap inside the drawer.
        first_inspector_control = inspector.locator(
            'button:not([disabled]), summary, [href], input:not([disabled]), '
            'textarea:not([disabled]), select:not([disabled]), '
            '[tabindex]:not([tabindex="-1"])'
        ).first
        first_inspector_control.focus()
        first_inspector_control.press("Shift+Tab")
        _require(
            inspector.evaluate("element => element.contains(document.activeElement)"),
            stage="focus",
            code="inspector_focus_trap_failed",
            browser=browser_name,
        )
        axe_results.append(_axe_audit(page, axe_source, "mobile_inspector_consent", browser_name))
        page.keyboard.press("Escape")
        page.wait_for_timeout(50)
        restored_inspector_trigger = page.get_by_role("button", name="展开右侧检查器")
        _require(
            restored_inspector_trigger.evaluate("element => document.activeElement === element"),
            stage="focus",
            code="inspector_focus_restore_failed",
            browser=browser_name,
        )

        reflow_results = [
            _reflow_check(page, 390, "mobile_390px", browser_name),
            _reflow_check(page, 640, "equivalent_200_percent", browser_name),
            _reflow_check(page, 320, "equivalent_400_percent", browser_name),
        ]
        page.set_viewport_size({"width": 1280, "height": 800})

        # The installed worker must contain only immutable static entries. Then
        # a true network-off hard reload renders the account-scope-bound shell.
        worker_deadline = time.monotonic() + 15
        while time.monotonic() < worker_deadline:
            if page.evaluate("Boolean(navigator.serviceWorker?.controller)"):
                break
            page.wait_for_timeout(100)
        else:
            raise SmokeFailure("offline", "service_worker_not_controlling", browser_name)
        cached_urls: list[str] = []
        cached_paths: list[str] = []
        cache_deadline = time.monotonic() + 5
        while time.monotonic() < cache_deadline:
            cached_urls = page.evaluate(
                """async () => {
                  const names = (await caches.keys()).filter((name) =>
                    name.startsWith('teachlab-console-static-'));
                  const urls = [];
                  for (const name of names) {
                    const cache = await caches.open(name);
                    for (const request of await cache.keys()) urls.push(request.url);
                  }
                  return urls;
                }"""
            )
            cached_paths = [urlsplit(item).path for item in cached_urls]
            _require(
                all(
                    path.startswith("/_next/static/")
                    or path == "/teachlab-offline-shell-v1.js"
                    for path in cached_paths
                ),
                stage="offline",
                code="cache_contains_non_immutable_surface",
                browser=browser_name,
            )
            if "/teachlab-offline-shell-v1.js" in cached_paths:
                break
            # Firefox can expose the claimed controller one task before Cache
            # API entries from install/prewarm become visible to the client.
            page.wait_for_timeout(100)
        _require(
            bool(cached_urls)
            and bool(cached_paths),
            stage="offline",
            code="static_cache_empty",
            browser=browser_name,
        )
        _require(
            "/teachlab-offline-shell-v1.js" in cached_paths,
            stage="offline",
            code="offline_shell_not_cached",
            browser=browser_name,
        )
        _seed_offline_snapshot(page)
        # A Playwright route shim can sit ahead of browser-native service-worker
        # navigation handling. Online requests have already been actively
        # guarded; keep the passive request observer and remove only that shim
        # before exercising the CSP-locked shell.
        page.unroute("**/*", guard_route)
        # Cut the transport behind a same-origin TCP gate instead of relying on
        # Playwright's engine-specific offline emulation. The public listener
        # stays bound, while the worker's internal fetch receives a real
        # connection failure and falls back to the scope-bound offline document.
        error_gate.require_online_clean()
        error_gate.begin_intentional_offline_navigation()
        disconnect_production_origin()
        page.evaluate(
            "url => window.location.assign(url)",
            f"{base_url}/?offline-smoke={secrets.token_hex(8)}",
        )
        page.get_by_role("heading", name="TeachLab 离线只读模式").wait_for(
            state="visible", timeout=10_000
        )
        page.get_by_role("heading", name="Cross-browser smoke project").wait_for(
            state="visible", timeout=10_000
        )
        page.get_by_text("Scope-bound offline snapshot sentinel", exact=True).wait_for(
            state="visible", timeout=10_000
        )
        error_gate.complete_intentional_offline_navigation()
        offline_main = page.locator("#offline-main")
        _require(
            offline_main.evaluate("element => document.activeElement === element"),
            stage="focus",
            code="offline_shell_initial_focus_failed",
            browser=browser_name,
        )
        page.keyboard.press("Tab")
        _require(
            page.get_by_role("button", name="重新连接").evaluate(
                "element => document.activeElement === element"
            ),
            stage="focus",
            code="offline_shell_focus_order_failed",
            browser=browser_name,
        )
        axe_results.append(_axe_audit(page, axe_source, "offline_scope_bound_shell", browser_name))
        error_gate.require_offline_clean()
        _require(external_requests == 0, stage="browser", code="external_request", browser=browser_name)
        return {
            "browser": browser_name,
            "root": "passed",
            "security_session_bootstrap": "passed",
            "chat_teach_controls": "passed",
            "critical_a11y_structure": "passed",
            "axe": axe_results,
            "keyboard_focus": "passed",
            "deletion_confirmation_cancel": "passed",
            "reflow": reflow_results,
            "offline_hard_reload": "scope_bound_read_only_shell_passed",
            "cache_policy": "same_origin_immutable_static_only",
            "error_gate": {
                "online": "zero_page_or_console_errors",
                "offline": "zero_unexpected_page_or_console_errors",
                "accepted_firefox_offline_network_diagnostics": (
                    error_gate.accepted_firefox_offline_diagnostics
                ),
            },
        }
    except SmokeFailure:
        raise
    except Exception as exc:
        raise SmokeFailure("browser", "browser_interaction_failed", browser_name) from exc
    finally:
        try:
            context.set_offline(False)
        except Exception:
            pass
        try:
            context.close()
        finally:
            browser.close()


def _self_test() -> None:
    _require(_project()["project_id"] == PROJECT_ID, stage="self_test", code="project_fixture")
    _require(_bootstrap()["schema_version"] == "1.1", stage="self_test", code="bootstrap_fixture")
    ports = {_free_loopback_port() for _ in range(3)}
    _require(3030 not in ports, stage="self_test", code="reserved_port_selected")
    with _fixture_backend() as (server, capability_url):
        payload = _http_json(f"{capability_url}api/bootstrap", expected_status=200)
        _require(payload.get("dashboard_kind") == "loopback_interactive_teacher_agent", stage="self_test", code="fixture_server")
        _require("api/bootstrap" in server.routes, stage="self_test", code="fixture_telemetry")


def _arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--browser", action="append", choices=SUPPORTED_BROWSERS)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = _arguments(list(sys.argv[1:] if argv is None else argv))
    if arguments.self_test:
        try:
            _self_test()
        except SmokeFailure as exc:
            print(json.dumps({"schema": RECEIPT_SCHEMA, "status": "failed", "stage": exc.stage, "code": exc.code}, sort_keys=True))
            return 1
        print(json.dumps({"schema": RECEIPT_SCHEMA, "status": "passed", "scope": "dependency_free_self_test"}, sort_keys=True))
        return 0

    selected = tuple(dict.fromkeys(arguments.browser or SUPPORTED_BROWSERS))
    try:
        _require(AXE_CORE_PATH.is_file(), stage="dependency", code="axe_core_not_installed")
        runtime = _sealed_runtime(arguments.runtime_root.resolve())
        try:
            from playwright.sync_api import sync_playwright
        except ModuleNotFoundError as exc:
            raise SmokeFailure("dependency", "python_playwright_not_installed") from exc
        with _fixture_backend() as (backend, capability_url):
            with sync_playwright() as playwright:
                results = []
                for name in selected:
                    with _production_server(runtime, capability_url) as (
                        base_url,
                        disconnect_production_origin,
                    ):
                        results.append(
                            _browser_smoke(
                                playwright,
                                name,
                                base_url,
                                disconnect_production_origin,
                            )
                        )
            routes = tuple(backend.routes)
        _require(routes.count("api/bootstrap") >= len(selected) + 1, stage="backend", code="bootstrap_not_exercised")
        _require("api/projects" in routes, stage="backend", code="project_list_not_exercised")
        _require(f"api/projects/{PROJECT_ID}" in routes, stage="backend", code="project_read_not_exercised")
    except SmokeFailure as exc:
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "failed",
            "stage": exc.stage,
            "code": exc.code,
            **({"browser": exc.browser} if exc.browser else {}),
        }
        print(json.dumps(receipt, sort_keys=True))
        return 1
    except Exception:
        print(
            json.dumps(
                {
                    "schema": RECEIPT_SCHEMA,
                    "status": "failed",
                    "stage": "runner",
                    "code": "unexpected_runner_failure",
                },
                sort_keys=True,
            )
        )
        return 1
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "passed",
        "runtime": "sealed_next_standalone_production",
        "release_id": runtime.release_id,
        "browsers": results,
        "platform_boundary": "web_runtime_only_not_macos_app_dmg_codesign_or_notarization",
    }
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
