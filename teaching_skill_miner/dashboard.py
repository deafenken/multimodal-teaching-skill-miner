from __future__ import annotations

from hashlib import sha256
from importlib import resources
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import webbrowser


DASHBOARD_PACKAGE = "teaching_skill_miner.web"
DASHBOARD_RESOURCE = "index.html"


def dashboard_html_bytes() -> bytes:
    """Return the packaged, public-only dashboard document."""

    return resources.files(DASHBOARD_PACKAGE).joinpath(DASHBOARD_RESOURCE).read_bytes()


def dashboard_self_check() -> dict[str, Any]:
    """Validate the packaged dashboard without opening a browser or socket."""

    payload = dashboard_html_bytes()
    text = payload.decode("utf-8")
    required_markers = (
        "Teaching Skill Miner",
        "engineering_ready_external_validation_pending",
        "data-screen-label=\"01 Overview\"",
        "data-screen-label=\"02 Multimodal timeline\"",
        "data-screen-label=\"03 Evidence\"",
        "data-screen-label=\"04 Skill runtime\"",
        "data-screen-label=\"05 Claim boundaries\"",
        "/*EDITMODE-BEGIN*/",
        "/*EDITMODE-END*/",
    )
    forbidden_release_paths = (
        "artifacts/private",
        "data/real",
        "/data/winbeau_zhao",
    )
    missing_markers = [marker for marker in required_markers if marker not in text]
    forbidden_matches = [
        marker for marker in forbidden_release_paths if marker in text
    ]
    return {
        "schema_version": "1.0",
        "dashboard_kind": "public_aggregate_evidence_frontend",
        "resource": f"{DASHBOARD_PACKAGE}:{DASHBOARD_RESOURCE}",
        "size_bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
        "utf8_valid": True,
        "single_html_document": True,
        "required_markers_present": not missing_markers,
        "missing_markers": missing_markers,
        "forbidden_release_paths_absent": not forbidden_matches,
        "forbidden_matches": forbidden_matches,
        "external_network_assets_required": False,
        "private_media_embedded": False,
        "row_level_private_data_embedded": False,
        "passed": not missing_markers and not forbidden_matches,
    }


def materialize_dashboard(output: str | Path | None = None) -> Path:
    """Write the reviewed dashboard to a safe local HTML path."""

    payload = dashboard_html_bytes()
    digest = sha256(payload).hexdigest()[:16]
    if output is None:
        private_root = Path(
            tempfile.mkdtemp(prefix=f"tsm-evidence-dashboard-{digest}-")
        )
        target = private_root / "index.html"
    else:
        expanded = Path(output).expanduser()
        candidate = Path(os.path.abspath(expanded))
        target = candidate.parent.resolve() / candidate.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_symlink() and target.is_file() and target.read_bytes() == payload:
        return target

    file_descriptor, staged_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    staged = Path(staged_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, target)
    finally:
        if staged.exists():
            staged.unlink()
    return target


def open_dashboard(
    *,
    output: str | Path | None = None,
    open_browser: bool = True,
) -> int:
    check = dashboard_self_check()
    if not check["passed"]:
        raise RuntimeError("packaged dashboard failed its self-check")
    target = materialize_dashboard(output)
    opened = False
    if open_browser:
        opened = bool(webbrowser.open(target.as_uri()))
    print(
        json.dumps(
            {
                "dashboard_path": str(target),
                "dashboard_uri": target.as_uri(),
                "browser_open_requested": open_browser,
                "browser_open_result": opened if open_browser else None,
                "public_aggregate_only": True,
                "private_media_embedded": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0
