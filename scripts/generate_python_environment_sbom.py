#!/usr/bin/env python3
"""Generate a deterministic CycloneDX inventory without installing an SBOM tool.

This describes the already-installed verification environment.  It deliberately
does not claim wheel provenance or vulnerability status; release provenance is
attested separately by CI.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import quote


_SAFE_VALUE = re.compile(r"^[^\x00-\x1f\x7f]{1,512}$")


def _component(distribution: metadata.Distribution) -> dict[str, object] | None:
    name = str(distribution.metadata.get("Name") or "").strip()
    version = str(distribution.version or "").strip()
    if _SAFE_VALUE.fullmatch(name) is None or _SAFE_VALUE.fullmatch(version) is None:
        return None
    normalized = re.sub(r"[-_.]+", "-", name).lower()
    reference = f"pkg:pypi/{quote(normalized, safe='')}@{quote(version, safe='') }"
    return {
        "type": "library",
        "bom-ref": reference,
        "name": name,
        "version": version,
        "purl": reference,
        "properties": [
            {
                "name": "teachlab:inventory_boundary",
                "value": "installed_environment_not_wheel_provenance",
            }
        ],
    }


def build_sbom() -> dict[str, object]:
    rows = [row for item in metadata.distributions() if (row := _component(item))]
    unique = {str(row["bom-ref"]): row for row in rows}
    components = [unique[key] for key in sorted(unique)]
    root_ref = "pkg:pypi/teaching-skill-miner@1.2.0?type=application"
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "bom-ref": root_ref,
                "name": "teaching-skill-miner-verification-environment",
                "version": "1.2.0",
            },
            "properties": [
                {
                    "name": "teachlab:claim_boundary",
                    "value": "inventory_only_not_vulnerability_or_provenance_attestation",
                }
            ],
        },
        "components": components,
        "dependencies": [
            {"ref": root_ref, "dependsOn": sorted(unique)},
            *({"ref": key, "dependsOn": []} for key in sorted(unique)),
        ],
    }


def write_sbom(path: Path) -> None:
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise ValueError("SBOM output must be a regular file or absent")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(build_sbom(), ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_sbom(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
