from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from teaching_skill_miner import teacher_agent_worker_isolation as isolation


def _policy() -> dict[str, int | str]:
    return isolation.default_worker_process_resource_limit_policy()


def _fake_resource_kernel(monkeypatch, hard_limits: dict[int, int] | None = None):
    infinity = int(isolation.resource.RLIM_INFINITY)
    identifiers = {
        isolation.resource.RLIMIT_CORE,
        isolation.resource.RLIMIT_NOFILE,
        isolation.resource.RLIMIT_FSIZE,
        isolation.resource.RLIMIT_AS,
    }
    state = {
        identifier: (infinity, (hard_limits or {}).get(identifier, infinity))
        for identifier in identifiers
    }
    calls: list[tuple[int, tuple[int, int]]] = []

    def getrlimit(identifier: int) -> tuple[int, int]:
        return state[identifier]

    def setrlimit(identifier: int, value: tuple[int, int]) -> None:
        calls.append((identifier, value))
        state[identifier] = value

    monkeypatch.setattr(isolation.sys, "platform", "linux")
    monkeypatch.setattr(isolation.resource, "getrlimit", getrlimit)
    monkeypatch.setattr(isolation.resource, "setrlimit", setrlimit)
    return state, calls


def test_linux_worker_installs_bounded_irreversible_limits_without_cpu_or_nproc(
    monkeypatch,
) -> None:
    state, calls = _fake_resource_kernel(monkeypatch)
    status = isolation.install_worker_process_resource_limits(
        required=True,
        policy=_policy(),
    )

    expected = {
        isolation.resource.RLIMIT_CORE: 0,
        isolation.resource.RLIMIT_FSIZE: 512 * 1024 * 1024,
        isolation.resource.RLIMIT_NOFILE: 256,
        isolation.resource.RLIMIT_AS: 1536 * 1024 * 1024,
    }
    assert calls == [
        (identifier, (target, target)) for identifier, target in expected.items()
    ]
    assert state == {
        identifier: (target, target) for identifier, target in expected.items()
    }
    assert {identifier for identifier, _value in calls}.isdisjoint(
        {
            getattr(isolation.resource, "RLIMIT_CPU", -100),
            getattr(isolation.resource, "RLIMIT_NPROC", -101),
        }
    )
    assert status == {
        "schema": "teaching_skill_miner.worker_process_resource_limits_status.v1",
        "enforcement": "linux_rlimit_v1",
        "address_space_bytes": 1536 * 1024 * 1024,
        "file_size_bytes": 512 * 1024 * 1024,
        "open_files": 256,
        "core_dump_bytes": 0,
        "cpu_time_limit": "not_set_long_lived_worker_uses_request_wall_clock",
        "process_count_limit": "not_set_shared_uid_unsafe",
        "scope": "per_process_individual_inherited_not_process_tree_aggregate",
    }


def test_existing_stronger_but_operational_hard_limits_are_preserved(
    monkeypatch,
) -> None:
    hard = {
        isolation.resource.RLIMIT_AS: 1280 * 1024 * 1024,
        isolation.resource.RLIMIT_FSIZE: 384 * 1024 * 1024,
        isolation.resource.RLIMIT_NOFILE: 192,
    }
    _state, calls = _fake_resource_kernel(monkeypatch, hard)
    status = isolation.install_worker_process_resource_limits(
        required=True,
        policy=_policy(),
    )
    assert status["address_space_bytes"] == hard[isolation.resource.RLIMIT_AS]
    assert status["file_size_bytes"] == hard[isolation.resource.RLIMIT_FSIZE]
    assert status["open_files"] == hard[isolation.resource.RLIMIT_NOFILE]
    assert dict(calls)[isolation.resource.RLIMIT_AS] == (
        hard[isolation.resource.RLIMIT_AS],
        hard[isolation.resource.RLIMIT_AS],
    )


def test_too_small_existing_hard_limit_fails_before_irreversible_changes(
    monkeypatch,
) -> None:
    _state, calls = _fake_resource_kernel(
        monkeypatch,
        {isolation.resource.RLIMIT_NOFILE: 127},
    )
    with pytest.raises(
        isolation.WorkerIsolationError,
        match="below safe minimum",
    ):
        isolation.install_worker_process_resource_limits(
            required=True,
            policy=_policy(),
        )
    assert calls == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda policy: policy.pop("open_files"),
        lambda policy: policy.update(open_files=True),
        lambda policy: policy.update(open_files=513),
        lambda policy: policy.update(file_size_bytes=256 * 1024 * 1024),
        lambda policy: policy.update(address_space_bytes=3 * 1024 * 1024 * 1024),
        lambda policy: policy.update(core_dump_bytes=1),
        lambda policy: policy.update(core_dump_bytes=0.0),
        lambda policy: policy.update(browser_override=True),
    ],
)
def test_required_policy_rejects_missing_weak_or_extra_fields(
    monkeypatch, mutate
) -> None:
    _state, calls = _fake_resource_kernel(monkeypatch)
    policy = _policy()
    mutate(policy)
    with pytest.raises(isolation.WorkerIsolationError):
        isolation.install_worker_process_resource_limits(
            required=True,
            policy=policy,
        )
    assert calls == []


def test_local_or_test_mode_explicitly_has_no_os_limit_claim() -> None:
    assert isolation.install_worker_process_resource_limits(
        required=False,
        policy=None,
    ) == {
        "schema": "teaching_skill_miner.worker_process_resource_limits_status.v1",
        "enforcement": "not_required_local_or_test",
        "address_space_bytes": None,
        "file_size_bytes": None,
        "open_files": None,
        "core_dump_bytes": None,
        "cpu_time_limit": "not_required_local_or_test",
        "process_count_limit": "not_required_local_or_test",
        "scope": "none",
    }
    with pytest.raises(isolation.WorkerIsolationError, match="forbidden"):
        isolation.install_worker_process_resource_limits(
            required=False,
            policy=_policy(),
        )


def test_required_process_limits_fail_closed_off_linux(monkeypatch) -> None:
    monkeypatch.setattr(isolation.sys, "platform", "darwin")
    with pytest.raises(isolation.WorkerIsolationError, match="need Linux"):
        isolation.install_worker_process_resource_limits(
            required=True,
            policy=_policy(),
        )


def test_worker_allowlist_resolves_usrmerge_system_symlinks(
    tmp_path: Path, monkeypatch
) -> None:
    system_root = tmp_path / "system"
    usr = system_root / "usr"
    for name in ("bin", "lib", "lib64"):
        (usr / name).mkdir(parents=True)
        (system_root / name).symlink_to(Path("usr") / name, target_is_directory=True)
    monkeypatch.setattr(
        isolation,
        "_TRUSTED_READ_ONLY_SYSTEM_PATHS",
        (usr, system_root / "bin", system_root / "lib", system_root / "lib64"),
    )
    private_root = tmp_path / "private"
    runtime_root = tmp_path / "runtime"
    application_root = tmp_path / "application"
    for path in (private_root, runtime_root, application_root):
        path.mkdir()
    provider_key = tmp_path / "provider-key"
    provider_key.write_text("secret", encoding="utf-8")

    resolved_private_root, read_paths = isolation._worker_allowlist_paths(
        private_root=private_root,
        runtime_data_root=runtime_root,
        provider_key_file=provider_key,
        application_root=application_root,
    )

    assert resolved_private_root == private_root.resolve(strict=True)
    assert read_paths == [
        usr.resolve(strict=True),
        (usr / "bin").resolve(strict=True),
        (usr / "lib").resolve(strict=True),
        (usr / "lib64").resolve(strict=True),
        runtime_root.resolve(strict=True),
        application_root.resolve(strict=True),
        provider_key.resolve(strict=True),
    ]


@pytest.mark.parametrize(
    "linked_path",
    ["private_root", "runtime_data_root", "application_root", "provider_key_file"],
)
def test_worker_allowlist_rejects_symlinks_for_private_deployment_paths(
    tmp_path: Path, monkeypatch, linked_path: str
) -> None:
    monkeypatch.setattr(isolation, "_TRUSTED_READ_ONLY_SYSTEM_PATHS", ())
    private_root = tmp_path / "private"
    runtime_root = tmp_path / "runtime"
    application_root = tmp_path / "application"
    provider_key = tmp_path / "provider-key"
    for path in (private_root, runtime_root, application_root):
        path.mkdir()
    provider_key.write_text("secret", encoding="utf-8")
    values = {
        "private_root": private_root,
        "runtime_data_root": runtime_root,
        "application_root": application_root,
        "provider_key_file": provider_key,
    }
    target = values[linked_path]
    alias = tmp_path / f"{linked_path}-alias"
    alias.symlink_to(target, target_is_directory=target.is_dir())
    values[linked_path] = alias

    with pytest.raises(isolation.WorkerIsolationError, match="must not be a symlink"):
        isolation._worker_allowlist_paths(**values)


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT inheritance is Linux-only")
def test_real_linux_limits_are_inherited_by_parser_children() -> None:
    script = r"""
import json
import resource
import subprocess
import sys
from teaching_skill_miner.teacher_agent_worker_isolation import (
    default_worker_process_resource_limit_policy,
    install_worker_process_resource_limits,
)
before_cpu = resource.getrlimit(resource.RLIMIT_CPU)
before_nproc = resource.getrlimit(resource.RLIMIT_NPROC)
status = install_worker_process_resource_limits(
    required=True,
    policy=default_worker_process_resource_limit_policy(),
)
child = subprocess.run(
    [sys.executable, '-c', (
        'import json,resource; print(json.dumps({'
        '"core":resource.getrlimit(resource.RLIMIT_CORE),'
        '"nofile":resource.getrlimit(resource.RLIMIT_NOFILE),'
        '"fsize":resource.getrlimit(resource.RLIMIT_FSIZE),'
        '"as":resource.getrlimit(resource.RLIMIT_AS)}))'
    )],
    text=True,
    capture_output=True,
    check=True,
)
assert resource.getrlimit(resource.RLIMIT_CPU) == before_cpu
assert resource.getrlimit(resource.RLIMIT_NPROC) == before_nproc
print(json.dumps({"status": status, "child": json.loads(child.stdout)}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parent.parent,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    result = json.loads(completed.stdout)
    status = result["status"]
    child = result["child"]
    assert child == {
        "core": [0, 0],
        "nofile": [status["open_files"], status["open_files"]],
        "fsize": [status["file_size_bytes"], status["file_size_bytes"]],
        "as": [status["address_space_bytes"], status["address_space_bytes"]],
    }
