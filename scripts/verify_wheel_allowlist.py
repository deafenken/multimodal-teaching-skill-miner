#!/usr/bin/env python3
"""Verify that a release wheel is an exact projection of public source files."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any
import zipfile


BUNDLED_TRANSCRIPTS = (
    "linear_algebra_l01.json",
    "linear_algebra_l02.json",
    "linear_algebra_l03.json",
    "linear_algebra_l04.json",
    "linear_algebra_l05.json",
    "python_l01.json",
    "python_l02.json",
    "python_l03.json",
    "python_l04.json",
    "python_l05.json",
)
PUBLIC_DATA_FILES = (
    "dataset_manifest.json",
    "evaluation_cases.json",
    "formal_caption_sources.json",
    "neural_v1_runtime_manifest.json",
    "teacher_agent_benchmark_v2_development.json",
    "teacher_agent_benchmark_v2_development_gold.json",
    "teacher_agent_demo_input.json",
    "teacher_agent_evaluation_cases.json",
    "teacher_agent_free_text_benchmark.json",
    "teacher_agent_free_text_benchmark_receipt.json",
    "teacher_agent_learning_outcome_demo.json",
    "teacher_agent_multiturn_benchmark_v1.json",
    "teacher_agent_skill_library.json",
    "teacher_agent_skill_library_v2.json",
)
GOVERNANCE_FILES = (
    "CHANGELOG.md",
    "PRIVACY.md",
    "SECURITY.md",
    "THIRD_PARTY_DATA.md",
)
PACKAGE_RESOURCE_FILES = (
    "teaching_skill_miner/web/index.html",
    "teaching_skill_miner/web/private_demo.html",
    "teaching_skill_miner/web/private_skill_demo.css",
    "teaching_skill_miner/web/private_skill_demo.js",
    "teaching_skill_miner/web/teacher_agent_demo.css",
    "teaching_skill_miner/web/teacher_agent_demo.html",
    "teaching_skill_miner/web/teacher_agent_demo.js",
    "teaching_skill_miner/web/assets/student-xiaoyu.png",
    "teaching_skill_miner/web/assets/student-zimo.png",
    "teaching_skill_miner/web/assets/student-zhixing.png",
)
DIST_INFO_FILES = (
    "METADATA",
    "WHEEL",
    "entry_points.txt",
    "top_level.txt",
    "RECORD",
)
PUBLIC_JSON_ALLOWLIST = Path("release/public_json_resources.txt")
PUBLIC_JSON_DIRECTORIES = ("configs", "schema")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _public_json_resource_state(
    repository_root: Path,
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """Return reviewed, discovered, unlisted, missing, and validation errors."""

    allowlist_path = repository_root / PUBLIC_JSON_ALLOWLIST
    reviewed: list[str] = []
    errors: list[str] = []
    if allowlist_path.is_symlink() or not allowlist_path.is_file():
        errors.append(
            "public JSON resource allowlist is missing or unsafe: "
            f"{PUBLIC_JSON_ALLOWLIST.as_posix()}"
        )
    else:
        try:
            raw_lines = allowlist_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            errors.append(
                "public JSON resource allowlist could not be read: "
                f"{type(exc).__name__}"
            )
            raw_lines = []
        if any(not line or line != line.strip() for line in raw_lines):
            errors.append(
                "public JSON resource allowlist must contain one nonblank, "
                "unindented path per line"
            )
        reviewed = [line for line in raw_lines if line and line == line.strip()]
        if reviewed != sorted(set(reviewed)):
            errors.append(
                "public JSON resource allowlist must be sorted and contain no duplicates"
            )

    valid_reviewed: list[str] = []
    for relative in reviewed:
        path = PurePosixPath(relative)
        if (
            len(path.parts) != 2
            or path.parts[0] not in PUBLIC_JSON_DIRECTORIES
            or path.suffix != ".json"
            or path.name in {".", ".."}
        ):
            errors.append(f"invalid public JSON resource allowlist entry: {relative}")
            continue
        valid_reviewed.append(relative)

    discovered: list[str] = []
    for directory in PUBLIC_JSON_DIRECTORIES:
        root = repository_root / directory
        if root.is_symlink() or not root.is_dir():
            errors.append(f"public JSON resource directory is missing or unsafe: {directory}")
            continue
        for source in root.rglob("*.json"):
            if source.is_file() or source.is_symlink():
                discovered.append(source.relative_to(repository_root).as_posix())
    discovered = sorted(set(discovered))
    reviewed_set = set(valid_reviewed)
    discovered_set = set(discovered)
    unlisted = sorted(discovered_set - reviewed_set)
    missing = sorted(reviewed_set - discovered_set)
    if unlisted:
        errors.append(
            "unreviewed public JSON resources are present: " + ", ".join(unlisted)
        )
    if missing:
        errors.append(
            "reviewed public JSON resources are missing: " + ", ".join(missing)
        )
    for relative in sorted(reviewed_set & discovered_set):
        source = repository_root / relative
        if source.is_symlink() or not source.is_file():
            errors.append(f"reviewed public JSON resource is unsafe: {relative}")
    return valid_reviewed, discovered, unlisted, missing, errors


def verify_public_json_resource_allowlist(
    repository_root: str | Path,
) -> dict[str, Any]:
    """Fail closed unless schema/config JSONs exactly match the reviewed list."""

    root = Path(repository_root).resolve()
    reviewed, discovered, unlisted, missing, errors = _public_json_resource_state(root)
    return {
        "schema_version": "1.0",
        "audit_kind": "public_json_resource_source_allowlist",
        "repository_root": str(root),
        "allowlist": PUBLIC_JSON_ALLOWLIST.as_posix(),
        "reviewed_resources": reviewed,
        "discovered_resources": discovered,
        "unlisted_resources": unlisted,
        "missing_resources": missing,
        "errors": errors,
        "passed": not errors,
    }


def _expected_payloads(
    repository_root: Path,
    distribution_stem: str,
    dist_info_root: str,
) -> tuple[dict[str, Path], list[str]]:
    expected: dict[str, Path] = {}
    errors: list[str] = []

    def include(member: str, relative_source: str) -> None:
        source = repository_root / relative_source
        if not source.is_file():
            errors.append(f"required release source is missing: {relative_source}")
            return
        expected[member] = source

    package_root = repository_root / "teaching_skill_miner"
    package_sources = sorted(package_root.rglob("*.py"))
    if not package_sources:
        errors.append("no teaching_skill_miner Python package files found")
    for source in package_sources:
        member = source.relative_to(repository_root).as_posix()
        expected[member] = source
    for relative in PACKAGE_RESOURCE_FILES:
        include(relative, relative)

    share_root = (
        f"{distribution_stem}.data/data/share/teaching-skill-miner"
    )
    for filename in PUBLIC_DATA_FILES:
        include(
            f"{share_root}/data/{filename}",
            f"data/{filename}",
        )
    for filename in BUNDLED_TRANSCRIPTS:
        include(
            f"{share_root}/data/transcripts/{filename}",
            f"data/transcripts/{filename}",
        )
    reviewed_json, _, _, _, json_errors = _public_json_resource_state(
        repository_root
    )
    errors.extend(json_errors)
    for relative in reviewed_json:
        source = repository_root / relative
        if source.is_symlink() or not source.is_file():
            continue
        directory, filename = PurePosixPath(relative).parts
        include(
            f"{share_root}/{directory}/{filename}",
            relative,
        )
    for filename in GOVERNANCE_FILES:
        include(
            f"{share_root}/governance/{filename}",
            filename,
        )
    include(f"{dist_info_root}/licenses/LICENSE", "LICENSE")
    return expected, errors


def verify_wheel_allowlist(
    wheel_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Compare every non-metadata wheel member with its public source file."""

    wheel = Path(wheel_path).resolve()
    root = Path(repository_root).resolve()
    errors: list[str] = []
    missing: list[str] = []
    unexpected: list[str] = []
    content_mismatches: list[dict[str, str]] = []
    duplicate_members: list[str] = []
    dist_info_root: str | None = None
    distribution_stem: str | None = None
    member_count = 0
    wheel_size_bytes: int | None = None
    wheel_sha256: str | None = None

    try:
        wheel_size_bytes = wheel.stat().st_size
        wheel_sha256 = _file_sha256(wheel)
        with zipfile.ZipFile(wheel) as archive:
            infos = [item for item in archive.infolist() if not item.is_dir()]
            names = [item.filename for item in infos]
            member_count = len(names)
            duplicate_members = sorted(
                name for name, count in Counter(names).items() if count > 1
            )
            roots = {
                PurePosixPath(name).parts[0]
                for name in names
                if PurePosixPath(name).parts
                and PurePosixPath(name).parts[0].endswith(".dist-info")
            }
            if len(roots) != 1:
                errors.append(
                    "wheel must contain exactly one top-level .dist-info directory"
                )
            else:
                dist_info_root = next(iter(roots))
                distribution_stem = dist_info_root.removesuffix(".dist-info")
                if not distribution_stem.startswith("teaching_skill_miner-"):
                    errors.append(
                        "unexpected wheel distribution prefix: "
                        f"{distribution_stem}"
                    )
                expected_payloads, source_errors = _expected_payloads(
                    root,
                    distribution_stem,
                    dist_info_root,
                )
                errors.extend(source_errors)
                generated_metadata = {
                    f"{dist_info_root}/{filename}"
                    for filename in DIST_INFO_FILES
                }
                expected_names = set(expected_payloads) | generated_metadata
                actual_names = set(names)
                missing = sorted(expected_names - actual_names)
                unexpected = sorted(actual_names - expected_names)
                for member, source in sorted(expected_payloads.items()):
                    if member not in actual_names:
                        continue
                    actual_payload = archive.read(member)
                    expected_payload = source.read_bytes()
                    if actual_payload != expected_payload:
                        content_mismatches.append(
                            {
                                "member": member,
                                "expected_sha256": _sha256(expected_payload),
                                "actual_sha256": _sha256(actual_payload),
                            }
                        )
    except (OSError, zipfile.BadZipFile) as exc:
        errors.append(f"wheel could not be read: {type(exc).__name__}")

    return {
        "schema_version": "1.0",
        "audit_kind": "exact_release_wheel_allowlist",
        "wheel": str(wheel),
        "wheel_size_bytes": wheel_size_bytes,
        "wheel_sha256": wheel_sha256,
        "distribution_stem": distribution_stem,
        "dist_info_root": dist_info_root,
        "member_count": member_count,
        "passed": not (
            errors
            or missing
            or unexpected
            or content_mismatches
            or duplicate_members
        ),
        "errors": errors,
        "missing_members": missing,
        "unexpected_members": unexpected,
        "content_mismatches": content_mismatches,
        "duplicate_members": duplicate_members,
        "policy": {
            "package_members": (
                "exact repository teaching_skill_miner/**/*.py plus explicit "
                "package resources"
            ),
            "package_resource_files": list(PACKAGE_RESOURCE_FILES),
            "bundled_transcripts": list(BUNDLED_TRANSCRIPTS),
            "public_json_allowlist": PUBLIC_JSON_ALLOWLIST.as_posix(),
            "private_artifacts_allowed": False,
            "media_frames_captions_or_model_weights_allowed": False,
            "source_payload_hash_match_required": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, nargs="?")
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the JSON report; validation errors are still written to stderr",
    )
    parser.add_argument(
        "--check-public-json-source-only",
        action="store_true",
        help="validate the reviewed schema/config JSON source allowlist, then exit",
    )
    args = parser.parse_args(argv)
    if args.check_public_json_source_only:
        if args.wheel is not None:
            parser.error("wheel must be omitted for --check-public-json-source-only")
        report = verify_public_json_resource_allowlist(args.repository_root)
    else:
        if args.wheel is None:
            parser.error("wheel is required unless --check-public-json-source-only is used")
        report = verify_wheel_allowlist(args.wheel, args.repository_root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.quiet:
        if not report["passed"]:
            for error in report["errors"]:
                print(error, file=sys.stderr)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
