from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import signal
import stat

import pytest

from teaching_skill_miner.console_soak import (
    LONG_SOAK_SECONDS,
    SoakConfig,
    SoakFailure,
    _StopController,
    _validated_loopback_url,
    canonical_sha256,
    long_soak_threshold_reached,
    main,
    percentile_95,
    run_soak,
    verify_receipt,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def self_test_receipt(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    receipt_path = tmp_path_factory.mktemp("console-soak") / "receipt.json"
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["--self-test", "--receipt", str(receipt_path)])
    assert exit_code == 0
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    receipt = json.loads(stdout.getvalue())
    assert receipt == json.loads(receipt_path.read_text(encoding="utf-8"))
    return receipt


def test_dependency_free_self_test_uses_live_os_samples_and_reaps_its_group(
    self_test_receipt: dict[str, object],
) -> None:
    receipt = self_test_receipt
    assert receipt["schema"] == "teachlab.console.resource_soak.v1"
    assert receipt["status"] == "passed"
    assert receipt["passed"] is True
    assert receipt["mode"] == "self-test"
    assert receipt["long_soak_executed"] is False
    assert receipt["long_soak_completed"] is False
    assert receipt["sampling"]["actual_os_process_samples_collected"] is True
    assert receipt["sampling"]["synthetic_process_samples"] is False
    assert receipt["sampling"]["sample_count"] == 7
    assert receipt["workload"]["attempts"] == 6
    assert receipt["workload"]["failures"] == 0

    target = receipt["resources"]["target"]
    assert target["measurement_source"] == "live_os_process_table_and_fd_inventory"
    assert target["synthetic_process_samples"] is False
    assert target["peak_rss_bytes"] > 0
    assert target["peak_fd_count"] > 0
    assert target["fd_sample_coverage"] >= 0.9
    assert target["processes"][0]["process_ref"] == "target:root"
    assert target["processes"][0]["sample_count"] == 7
    assert receipt["resources"]["disk"]["ending_minus_baseline_bytes"] > 0

    assert receipt["runtime"]["identity_material"]["target_kind"] == "isolated_fixture"
    assert receipt["runtime"]["identity_material"]["runtime_identity_verified"] is True
    assert (
        len(receipt["runtime"]["identity_material"]["root_process_start_sha256"]) == 64
    )
    assert receipt["claim_boundaries"]["fixture_is_real_console_or_harness"] is False
    assert receipt["claim_boundaries"]["reserved_port_3030_targeted"] is False
    assert receipt["browser"]["mode"] == "disabled"
    assert receipt["browser"]["process_group_measured"] is False
    assert receipt["cleanup"]["all_owned_processes_reaped"] is True
    assert receipt["cleanup"]["external_target_signalled"] is False
    assert verify_receipt(receipt)

    serialized = json.dumps(receipt, sort_keys=True)
    assert "http://" not in serialized
    assert str(ROOT) not in serialized
    assert "fixture-ready.json" not in serialized


def test_receipt_hashes_cover_config_budgets_runtime_and_full_content(
    self_test_receipt: dict[str, object],
) -> None:
    receipt = deepcopy(self_test_receipt)
    assert verify_receipt(receipt)
    assert receipt["configuration"]["content_sha256"] == canonical_sha256(
        receipt["configuration"]["material"]
    )
    assert receipt["budgets"]["content_sha256"] == canonical_sha256(
        receipt["budgets"]["limits"]
    )
    assert receipt["runtime"]["identity_sha256"] == canonical_sha256(
        receipt["runtime"]["identity_material"]
    )

    receipt["budgets"]["limits"]["target_peak_fd_count"] += 1
    assert not verify_receipt(receipt)


def test_reserved_port_and_non_loopback_targets_fail_before_network_access() -> None:
    for target in (
        "http://127.0.0.1:3030/private/",
        "http://localhost:43030/private/",
        "https://127.0.0.1:43030/private/",
        "http://127.0.0.1:43030/a/../private/",
        "http://127.0.0.1:43030/private/?secret=value",
    ):
        with pytest.raises(SoakFailure):
            _validated_loopback_url(target)
    assert _validated_loopback_url("http://127.0.0.1:43030/private/")


def test_external_target_requires_process_identity_and_disk_boundary(
    tmp_path: Path,
) -> None:
    base = dict(
        mode="smoke",
        duration_seconds=1.0,
        target_url="http://127.0.0.1:43030/private/",
        request_path="",
    )
    with pytest.raises(SoakFailure, match="external_target_pid_required"):
        SoakConfig(**base).validate()
    with pytest.raises(SoakFailure, match="external_runtime_id_required"):
        SoakConfig(**base, target_pid=os.getpid()).validate()
    with pytest.raises(SoakFailure, match="external_disk_measurement_root_required"):
        SoakConfig(**base, target_pid=os.getpid(), runtime_id="release-test").validate()
    SoakConfig(
        **base,
        target_pid=os.getpid(),
        runtime_id="release-test",
        disk_paths=(tmp_path,),
    ).validate()


def test_p95_and_long_soak_claim_have_exact_non_simulated_thresholds() -> None:
    assert percentile_95(list(range(1, 101))) == 95
    assert percentile_95([7.0]) == 7.0
    assert percentile_95([]) is None
    assert not long_soak_threshold_reached(LONG_SOAK_SECONDS - 0.001)
    assert long_soak_threshold_reached(LONG_SOAK_SECONDS)
    assert long_soak_threshold_reached(LONG_SOAK_SECONDS + 1)


@pytest.mark.skipif(
    os.name != "posix", reason="process-group cleanup contract is POSIX"
)
def test_interrupted_self_test_reaps_only_owned_processes() -> None:
    controller = _StopController()
    controller.request(signal.SIGINT)
    receipt = run_soak(
        SoakConfig(
            mode="self-test",
            duration_seconds=1.0,
            request_interval_seconds=0.01,
            sample_interval_seconds=0.01,
            request_timeout_seconds=2.0,
            request_path="work",
            self_test_request_limit=6,
        ),
        stop_controller=controller,
    )
    assert receipt["status"] == "interrupted"
    assert receipt["passed"] is False
    assert receipt["failure"] == {
        "stage": "signal",
        "code": "run_interrupted",
        "signal": "SIGINT",
    }
    assert receipt["long_soak_executed"] is False
    assert receipt["cleanup"]["all_owned_processes_reaped"] is True
    assert receipt["cleanup"]["external_target_signalled"] is False
    assert verify_receipt(receipt)
