from __future__ import annotations

import base64
from hashlib import sha256
import hmac
import json
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Thread
from types import SimpleNamespace

import pytest

from teaching_skill_miner import teacher_agent_gateway_worker as gateway_worker


def _bootstrap(root: Path, *, capability: str) -> dict[str, object]:
    return {
        "schema": "teaching_skill_miner.gateway_worker_bootstrap.v1",
        "scope_id": "scope_" + "a" * 48,
        "scope_key_version": "k2",
        "worker_id": "worker_" + "b" * 32,
        "capability_token": capability,
        "private_root": str(root),
        "scope_key_material": base64.b64encode(b"x" * 32).decode("ascii"),
        "learner_scope_id": "scope_" + "a" * 48,
        "authority_scope_bindings": [
            {"scope_id": "scope_" + "a" * 48, "key_version": "k1"}
        ],
        "agent_backend": "deterministic",
        "api_key_file": None,
        "filesystem_isolation_required": False,
        "process_resource_limits": None,
        "remote_provider_policy": None,
        "remote_subject_policy": None,
        "safeguarding_locale": "zh-CN",
        "safeguarding_dispatcher": None,
    }


def _production_process_resource_limits() -> dict[str, object]:
    return {
        "schema": "teaching_skill_miner.worker_process_resource_limits.v1",
        "address_space_bytes": 1536 * 1024 * 1024,
        "file_size_bytes": 512 * 1024 * 1024,
        "open_files": 256,
        "core_dump_bytes": 0,
    }


def _capability() -> str:
    binding = (
        "capability-v1\x00k2\x00scope_" + "a" * 48 + "\x00worker_" + "b" * 32
    ).encode("ascii")
    return (
        base64.urlsafe_b64encode(hmac.new(b"x" * 32, binding, sha256).digest())
        .decode("ascii")
        .rstrip("=")
    )


def _run(
    config: dict[str, object], repository_root: Path, *, runtime_canary: bool = False
) -> subprocess.CompletedProcess[str]:
    argv = [sys.executable, "-m", "teaching_skill_miner.teacher_agent_gateway_worker"]
    if runtime_canary:
        argv.append("--runtime-canary")
    # Capability and private state are sent only on the closed stdin pipe. They
    # are deliberately absent from both argv and the child environment.
    assert str(config["private_root"]) not in argv
    assert str(config["capability_token"]) not in argv
    return subprocess.run(
        argv,
        cwd=repository_root,
        input=json.dumps(config, separators=(",", ":")) + "\n",
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
        env={
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        },
    )


def _run_canary(
    config: dict[str, object], repository_root: Path
) -> tuple[dict[str, object], str, int]:
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "teaching_skill_miner.teacher_agent_gateway_worker",
            "--runtime-canary",
        ],
        cwd=repository_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        },
    )
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(json.dumps(config, separators=(",", ":")) + "\n")
    process.stdin.close()
    process.stdin = None
    try:
        lines: Queue[str] = Queue(maxsize=1)
        Thread(target=lambda: lines.put(process.stdout.readline()), daemon=True).start()
        try:
            status_line = lines.get(timeout=10)
        except Empty as exc:
            raise AssertionError(
                "runtime canary did not report bounded status"
            ) from exc
        status = json.loads(status_line)
        process.terminate()
        _stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
    return status, stderr, int(process.returncode)


def test_worker_self_check_is_side_effect_free_and_normal_start_still_requires_bootstrap(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parent.parent
    environment = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    self_check = subprocess.run(
        [
            sys.executable,
            "-m",
            "teaching_skill_miner.teacher_agent_gateway_worker",
            "--self-check",
        ],
        cwd=repository_root,
        input="",
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
        env=environment,
    )
    assert self_check.returncode == 0
    assert self_check.stdout == "teacher_agent_gateway_worker self-check passed\n"
    assert self_check.stderr == ""
    assert list(tmp_path.iterdir()) == []

    missing = subprocess.run(
        [sys.executable, "-m", "teaching_skill_miner.teacher_agent_gateway_worker"],
        cwd=repository_root,
        input="",
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
        env=environment,
    )
    assert missing.returncode == 2
    assert json.loads(missing.stdout)["code"] == "worker_configuration_invalid"
    assert missing.stderr == ""


def test_worker_rejects_unbound_capability_without_reflecting_secrets_or_paths(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parent.parent
    private_root = tmp_path / "k2" / ("scope_" + "a" * 48)
    private_root.mkdir(parents=True, mode=0o700)
    private_root.chmod(0o700)
    sentinel = "TOKEN_SENTINEL_" + "z" * 30
    result = _run(_bootstrap(private_root, capability=sentinel), repository_root)
    assert result.returncode == 2
    assert result.stderr == ""
    status = json.loads(result.stdout)
    assert status == {
        "schema": "teaching_skill_miner.gateway_worker_status.v1",
        "status": "failed",
        "code": "worker_configuration_invalid",
    }
    assert sentinel not in result.stdout
    assert str(private_root) not in result.stdout


def test_worker_rejects_non_private_scope_root_with_content_free_status(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parent.parent
    private_root = tmp_path / "k2" / ("scope_" + "a" * 48)
    private_root.mkdir(parents=True, mode=0o755)
    private_root.chmod(0o755)
    capability = _capability()
    result = _run(_bootstrap(private_root, capability=capability), repository_root)
    assert result.returncode == 2
    assert result.stderr == ""
    assert json.loads(result.stdout)["code"] == "worker_configuration_invalid"
    assert capability not in result.stdout
    assert str(private_root) not in result.stdout


def test_worker_bootstrap_requires_server_owned_process_limit_field(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parent.parent
    private_root = tmp_path / "k2" / ("scope_" + "a" * 48)
    private_root.mkdir(parents=True, mode=0o700)
    private_root.chmod(0o700)
    config = _bootstrap(private_root, capability=_capability())
    config.pop("process_resource_limits")
    result = _run(config, repository_root)
    assert result.returncode == 2
    assert result.stderr == ""
    assert json.loads(result.stdout) == {
        "schema": "teaching_skill_miner.gateway_worker_status.v1",
        "status": "failed",
        "code": "worker_configuration_invalid",
    }


def test_local_worker_rejects_an_unrequired_or_browser_shaped_limit_override(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parent.parent
    private_root = tmp_path / "k2" / ("scope_" + "a" * 48)
    private_root.mkdir(parents=True, mode=0o700)
    private_root.chmod(0o700)
    config = _bootstrap(private_root, capability=_capability())
    config["process_resource_limits"] = {
        **_production_process_resource_limits(),
        "browser_override": True,
    }
    result = _run(config, repository_root)
    assert result.returncode == 2
    assert result.stderr == ""
    assert json.loads(result.stdout)["code"] == "worker_configuration_invalid"


def test_runtime_canary_rejects_an_error_path_without_reflecting_it(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parent.parent
    private_root = tmp_path / "missing" / "k2" / ("scope_" + "a" * 48)
    config = _bootstrap(private_root, capability=_capability())
    result = _run(config, repository_root, runtime_canary=True)
    assert result.returncode == 2
    assert result.stderr == ""
    assert json.loads(result.stdout) == {
        "schema": "teaching_skill_miner.gateway_worker_status.v1",
        "status": "failed",
        "code": "worker_configuration_invalid",
    }
    assert str(private_root) not in result.stdout


def test_runtime_canary_starts_and_shuts_down_without_creating_durable_stores(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parent.parent
    private_root = tmp_path / "k2" / ("scope_" + "a" * 48)
    private_root.mkdir(parents=True, mode=0o700)
    private_root.chmod(0o700)
    status, stderr, returncode = _run_canary(
        _bootstrap(private_root, capability=_capability()), repository_root
    )
    assert returncode == 0
    assert stderr == ""
    assert status == {
        "schema": "teaching_skill_miner.gateway_worker_status.v1",
        "status": "canary_ready",
        "worker_id": "worker_" + "b" * 32,
        "scope_key_version": "k2",
        "backend": "deterministic",
        "filesystem_isolation": "not_required_local_or_test",
        "process_resource_limits": {
            "schema": "teaching_skill_miner.worker_process_resource_limits_status.v1",
            "enforcement": "not_required_local_or_test",
            "address_space_bytes": None,
            "file_size_bytes": None,
            "open_files": None,
            "core_dump_bytes": None,
            "cpu_time_limit": "not_required_local_or_test",
            "process_count_limit": "not_required_local_or_test",
            "scope": "none",
        },
        "remote_provider_network": "not_contacted",
        "provider_credential": "not_applicable",
        "persistent_tenant_data_created": False,
    }
    assert list(private_root.iterdir()) == []


def test_runtime_data_root_supports_installed_wheel_data_files(
    tmp_path: Path, monkeypatch
) -> None:
    missing_source = tmp_path / "site-packages" / "teaching_skill_miner" / "worker.py"
    installed = tmp_path / "prefix" / "share" / "teaching-skill-miner" / "data"
    installed.mkdir(parents=True)
    for name in gateway_worker._REQUIRED_RUNTIME_DATA:
        (installed / name).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(gateway_worker, "__file__", str(missing_source))
    monkeypatch.setattr(
        gateway_worker.sysconfig,
        "get_path",
        lambda name: str(tmp_path / "prefix") if name == "data" else None,
    )
    assert gateway_worker._runtime_data_root() == installed.resolve()


def test_authenticated_worker_mounts_private_resource_review_store(
    tmp_path: Path, monkeypatch
) -> None:
    private_root = tmp_path / "k2" / ("scope_" + "a" * 48)
    private_root.mkdir(parents=True, mode=0o700)
    private_root.chmod(0o700)
    captured: dict[str, object] = {}

    def build(*_args, **kwargs):
        captured.update(kwargs)
        return object()

    class _Server:
        server_port = 43210

        def serve_forever(self, *, poll_interval: float) -> None:
            assert poll_interval == 0.1

        def shutdown(self) -> None:
            pass

        def server_close(self) -> None:
            pass

    monkeypatch.setattr(gateway_worker, "build_teacher_agent_dashboard_snapshot", build)
    monkeypatch.setattr(
        gateway_worker,
        "create_teacher_agent_dashboard_server",
        lambda *_args, **_kwargs: (_Server(), "private"),
    )
    monkeypatch.setattr(gateway_worker.signal, "signal", lambda *_args: None)

    config = _bootstrap(private_root, capability=_capability())
    config["remote_provider_policy"] = {
        "policy_id": "deepseek-deployment-terms",
        "policy_version": "2026-08-12",
        "policy_source": "deployment_operator_asserted_external_terms_not_repository_verified",
        "processing_region": "cn_north",
        "provider_retention_days": 7,
        "deletion_status": "outside_service_control_subject_to_provider_policy",
        "documentation_url": "https://provider.example/privacy",
    }
    config["remote_subject_policy"] = {
        "policy_id": "organization-adult-policy",
        "policy_version": "2026-08-12",
        "policy_source": "organization_oidc_or_roster_policy",
        "likely_minor": False,
        "guardian_or_school_policy": "not_required",
        "remote_processing_eligible": True,
    }
    config["safeguarding_dispatcher"] = {
        "endpoint": "https://safeguarding.school.example/v1/cases",
        "bearer_secret": "school-safeguarding-dispatch-secret-at-least-32-bytes",
        "route_locator": "sgr1_k1_" + "A" * 100,
        "policy_version": "school-safeguarding-v1",
        "timeout_ms": 1000,
        "maximum_response_bytes": 8192,
    }
    assert gateway_worker._serve(config) == 0
    assert captured["resource_review_store_path"] == private_root / "resource_reviews"
    assert captured["resource_index_store_path"] == private_root / "resource_index"
    assert captured["teacher_authority_verifier"] is not None
    assert str(captured["trusted_learner_profile_ref"]).startswith("profile_")
    assert len(str(captured["trusted_learner_profile_ref"])) == 72
    assert captured["remote_processing_region"] == "cn_north"
    assert captured["remote_provider_retention_days"] == 7
    assert captured["remote_provider_policy"] == config["remote_provider_policy"]
    assert captured["remote_subject_policy"] == config["remote_subject_policy"]
    assert captured["safeguarding_store"] is not None
    assert len(str(captured["safeguarding_scope_sha256"])) == 64
    assert callable(captured["safeguarding_system_authority_issuer"])
    assert callable(captured["safeguarding_system_authority_verifier"])
    assert callable(captured["safeguarding_staff_authority_issuer"])
    assert captured["safeguarding_dispatcher"] is not None
    safeguarding_store = captured["safeguarding_store"]
    assert getattr(safeguarding_store, "_escalation_delivery") is not None
    route_record = json.loads(
        (
            private_root / "safeguarding" / ".safeguarding_dispatch_route_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert route_record == {
        "schema": "teaching_skill_miner.safeguarding_route_record.v1",
        "route_locator": config["safeguarding_dispatcher"]["route_locator"],
    }


def test_remote_worker_fails_before_start_without_deployment_processing_policies(
    tmp_path: Path, monkeypatch
) -> None:
    private_root = tmp_path / "k2" / ("scope_" + "a" * 48)
    private_root.mkdir(parents=True, mode=0o700)
    private_root.chmod(0o700)
    config = _bootstrap(private_root, capability=_capability())
    monkeypatch.setattr(gateway_worker, "_client", lambda _config: object())

    with pytest.raises(
        gateway_worker.GatewayWorkerConfigurationError,
        match="deployment policies",
    ):
        gateway_worker._serve(config)


def test_required_worker_isolation_fails_closed_off_linux(
    tmp_path: Path, monkeypatch
) -> None:
    private_root = tmp_path / "k2" / ("scope_" + "a" * 48)
    private_root.mkdir(parents=True, mode=0o700)
    private_root.chmod(0o700)
    config = _bootstrap(private_root, capability=_capability())
    config["filesystem_isolation_required"] = True
    config["process_resource_limits"] = _production_process_resource_limits()
    monkeypatch.setattr(gateway_worker.sys, "platform", "darwin")
    with pytest.raises(
        gateway_worker.GatewayWorkerConfigurationError,
        match="filesystem isolation",
    ):
        gateway_worker._serve(config)


def test_runtime_canary_fails_closed_when_landlock_installation_fails(
    tmp_path: Path, monkeypatch
) -> None:
    private_root = tmp_path / "k2" / ("scope_" + "a" * 48)
    private_root.mkdir(parents=True, mode=0o700)
    private_root.chmod(0o700)
    config = _bootstrap(private_root, capability=_capability())
    config["filesystem_isolation_required"] = True
    config["process_resource_limits"] = _production_process_resource_limits()

    def fail_isolation(**_kwargs):
        raise gateway_worker.WorkerIsolationError("private kernel detail")

    monkeypatch.setattr(
        gateway_worker, "install_worker_filesystem_isolation", fail_isolation
    )
    with pytest.raises(
        gateway_worker.GatewayWorkerConfigurationError,
        match="required worker filesystem isolation is unavailable",
    ) as captured:
        gateway_worker._prepare_runtime(config)
    assert "private kernel detail" not in str(captured.value)


def test_runtime_canary_fails_closed_when_process_limits_are_missing(
    tmp_path: Path, monkeypatch
) -> None:
    private_root = tmp_path / "k2" / ("scope_" + "a" * 48)
    private_root.mkdir(parents=True, mode=0o700)
    private_root.chmod(0o700)
    config = _bootstrap(private_root, capability=_capability())
    config["filesystem_isolation_required"] = True
    monkeypatch.setattr(
        gateway_worker,
        "install_worker_filesystem_isolation",
        lambda **_kwargs: "linux_landlock_scope_allowlist_abi_6",
    )
    with pytest.raises(
        gateway_worker.GatewayWorkerConfigurationError,
        match="process resource limits",
    ):
        gateway_worker._prepare_runtime(config)


def test_runtime_installs_server_process_policy_and_projects_only_bounded_status(
    tmp_path: Path, monkeypatch
) -> None:
    # _prepare_runtime intentionally enables the parser child network sandbox
    # for the lifetime of a real worker process. Register the variable with
    # pytest's environment undo stack so this in-process contract test cannot
    # leak production policy into later resource/OCR tests.
    monkeypatch.setenv("TSM_REQUIRE_PARSER_NETWORK_ISOLATION", "0")
    private_root = tmp_path / "k2" / ("scope_" + "a" * 48)
    private_root.mkdir(parents=True, mode=0o700)
    private_root.chmod(0o700)
    config = _bootstrap(private_root, capability=_capability())
    policy = _production_process_resource_limits()
    config["filesystem_isolation_required"] = True
    config["process_resource_limits"] = policy
    captured: dict[str, object] = {}
    projected = {
        "schema": "teaching_skill_miner.worker_process_resource_limits_status.v1",
        "enforcement": "linux_rlimit_v1",
        "address_space_bytes": policy["address_space_bytes"],
        "file_size_bytes": policy["file_size_bytes"],
        "open_files": policy["open_files"],
        "core_dump_bytes": 0,
        "cpu_time_limit": "not_set_long_lived_worker_uses_request_wall_clock",
        "process_count_limit": "not_set_shared_uid_unsafe",
        "scope": "per_process_individual_inherited_not_process_tree_aggregate",
    }

    monkeypatch.setattr(
        gateway_worker,
        "install_worker_filesystem_isolation",
        lambda **_kwargs: "linux_landlock_scope_allowlist_abi_6",
    )

    def install_limits(*, required, policy):
        captured.update(required=required, policy=policy)
        return projected

    monkeypatch.setattr(
        gateway_worker,
        "install_worker_process_resource_limits",
        install_limits,
    )
    prepared = gateway_worker._prepare_runtime(config)
    assert captured == {"required": True, "policy": policy}
    assert prepared.process_resource_limits == projected
    assert "private_root" not in prepared.process_resource_limits
    assert "capability_token" not in prepared.process_resource_limits


@pytest.mark.skipif(sys.platform != "linux", reason="Landlock is Linux-only")
def test_runtime_canary_reports_real_linux_landlock_and_exits_cleanly(
    tmp_path: Path,
) -> None:
    from teaching_skill_miner.teacher_agent_worker_isolation import (
        landlock_abi_version,
    )

    if landlock_abi_version() < 1:
        pytest.skip("Landlock filesystem rules are unavailable")
    repository_root = Path(__file__).resolve().parent.parent
    private_root = tmp_path / "k2" / ("scope_" + "a" * 48)
    private_root.mkdir(parents=True, mode=0o700)
    private_root.chmod(0o700)
    config = _bootstrap(private_root, capability=_capability())
    config["filesystem_isolation_required"] = True
    config["process_resource_limits"] = _production_process_resource_limits()
    status, stderr, returncode = _run_canary(config, repository_root)
    assert returncode == 0
    assert stderr == ""
    assert str(status["filesystem_isolation"]).startswith(
        "linux_landlock_scope_allowlist_abi_"
    )
    assert status["remote_provider_network"] == "not_contacted"
    assert status["persistent_tenant_data_created"] is False


@pytest.mark.skipif(sys.platform != "linux", reason="Landlock is Linux-only")
def test_landlock_worker_cannot_stat_or_read_a_sibling_scope(tmp_path: Path) -> None:
    parent = tmp_path / "scopes" / "k1"
    own = parent / ("scope_" + "a" * 48)
    sibling = parent / ("scope_" + "b" * 48)
    runtime = own / "runtime"
    runtime.mkdir(parents=True, mode=0o700)
    sibling.mkdir(mode=0o700)
    secret = sibling / "private.txt"
    secret.write_text("sibling-private-sentinel", encoding="utf-8")
    script = """
import os
from pathlib import Path
from teaching_skill_miner.teacher_agent_worker_isolation import install_worker_filesystem_isolation
own, sibling, runtime = map(Path, __import__('sys').argv[1:])
result = install_worker_filesystem_isolation(required=True, private_root=own, runtime_data_root=runtime, provider_key_file=None)
assert result.startswith('linux_landlock_scope_allowlist_abi_')
for operation in (lambda: os.listdir(own.parent), lambda: (sibling / 'private.txt').read_text()):
    try:
        operation()
    except PermissionError:
        pass
    else:
        raise SystemExit('sibling scope remained readable')
print('isolated')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(own), str(sibling), str(runtime)],
        cwd=Path(__file__).resolve().parent.parent,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert completed.stdout.strip() == "isolated"


@pytest.mark.skipif(sys.platform != "linux", reason="Landlock is Linux-only")
def test_landlock_parser_child_cannot_bind_tcp_socket() -> None:
    from teaching_skill_miner.teacher_agent_worker_isolation import (
        landlock_abi_version,
    )

    if landlock_abi_version() < 4:
        pytest.skip("Landlock network rules require ABI 4")
    script = """
import socket
from teaching_skill_miner.teacher_agent_worker_isolation import install_parser_network_isolation
result = install_parser_network_isolation()
assert result.startswith('linux_landlock_no_tcp_abi_')
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(('127.0.0.1', 0))
except PermissionError:
    print('network-isolated')
else:
    raise SystemExit('parser retained TCP bind capability')
finally:
    sock.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parent.parent,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert completed.stdout.strip() == "network-isolated"


def test_provider_readiness_uses_content_free_probe_and_safe_status(
    monkeypatch, capsys
) -> None:
    class Client:
        def probe_model_availability(self, *, timeout_seconds: float):
            assert timeout_seconds == 5.0
            return {
                "schema": "teaching_skill_miner.deepseek_readiness.v1",
                "credential_validated": True,
                "provider_network_validated": True,
                "configured_model_available": True,
                "learner_content_sent": False,
                "generation_created": False,
            }

    monkeypatch.setattr(
        gateway_worker,
        "_prepare_runtime",
        lambda _config: SimpleNamespace(client=Client()),
    )
    assert gateway_worker._provider_readiness({"agent_backend": "deepseek"}) == 0
    status = json.loads(capsys.readouterr().out)
    assert status == {
        "schema": "teaching_skill_miner.gateway_worker_status.v1",
        "status": "provider_ready",
        "backend": "deepseek",
        "credential_validated": True,
        "provider_network_validated": True,
        "configured_model_available": True,
        "learner_content_sent": False,
        "generation_created": False,
        "persistent_tenant_data_created": False,
    }


def test_deterministic_provider_readiness_is_explicitly_not_required(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        gateway_worker,
        "_prepare_runtime",
        lambda _config: SimpleNamespace(client=None),
    )
    assert gateway_worker._provider_readiness({"agent_backend": "deterministic"}) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "provider_not_required"
    assert status["credential_validated"] is False
    assert status["provider_network_validated"] is False
    assert status["learner_content_sent"] is False
