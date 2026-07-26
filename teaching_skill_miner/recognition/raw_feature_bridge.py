"""Content-addressed raw-to-feature bridge for strict recognition workflows.

The bridge performs feature extraction only.  It deliberately does not load a
classifier, emit predictions, or convert successful extraction into an accuracy
claim.  Every output row is tied to verified raw bytes, an optional fixed media
window, the exact manifest order, and a frozen extractor provenance record.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from .datasets import file_sha256, probe_video
from .features import extract_video_audio_features, extract_video_visual_features
from .strict_evaluation import (
    FrozenDeploymentModel,
    strict_feature_bundle_fingerprint,
    validate_feature_provenance,
    validate_strict_manifest,
)


RAW_BINDING_ALGORITHM = "strict_raw_source_binding_v1"
RAW_CONTENT_ALGORITHM = "strict_synchronized_raw_content_v1"
BRIDGE_PROTOCOL = "strict_raw_feature_bridge_v1"
SUPPORTED_EXTRACTORS = {
    "builtin_classroom_video_visual_v1",
    "builtin_classroom_video_audio_v1",
    "builtin_numeric_sensor_summary_v1",
}
SENSOR_STATISTICS = (
    "mean",
    "std",
    "min",
    "p25",
    "p50",
    "p75",
    "max",
    "delta",
    "slope",
)
_SAFE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_EXTRACTION_CLAIM_SCOPE = {
    "raw_features_extracted": True,
    "automatic_recognition_performed": False,
    "recognition_accuracy_established": False,
    "deployment_accuracy_established": False,
    "teaching_effectiveness_established": False,
}


class RawFeatureBridgeError(ValueError):
    """Raised when raw inputs cannot be bound to a strict feature bundle."""


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RawFeatureBridgeError("evidence is not canonical JSON") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _finite_number(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RawFeatureBridgeError(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise RawFeatureBridgeError(f"{field} must be finite")
    if strictly_positive and normalized <= 0:
        raise RawFeatureBridgeError(f"{field} must be positive")
    if minimum is not None and normalized < minimum:
        raise RawFeatureBridgeError(f"{field} must be at least {minimum}")
    return normalized


def _positive_integer(value: Any, *, field: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RawFeatureBridgeError(f"{field} must be an integer >= {minimum}")
    return int(value)


def _normalized_selection(value: Mapping[str, Any]) -> dict[str, float]:
    has_start = "start_offset_seconds" in value
    has_duration = "duration_seconds" in value
    if not has_start and not has_duration:
        return {}
    start = _finite_number(
        value.get("start_offset_seconds", 0.0),
        field="raw input start_offset_seconds",
        minimum=0.0,
    )
    normalized = {"start_offset_seconds": start}
    if has_duration:
        normalized["duration_seconds"] = _finite_number(
            value["duration_seconds"],
            field="raw input duration_seconds",
            strictly_positive=True,
        )
    return normalized


def synchronized_content_sha256(raw_inputs: Mapping[str, Mapping[str, Any]]) -> str:
    """Hash the exact input bytes and optional windows that form one sample."""

    if not isinstance(raw_inputs, Mapping) or not raw_inputs:
        raise RawFeatureBridgeError("raw_inputs must be a non-empty object")
    normalized: list[dict[str, Any]] = []
    for input_key in sorted(raw_inputs):
        if not isinstance(input_key, str) or not _SAFE_NAME.fullmatch(input_key):
            raise RawFeatureBridgeError("raw input keys must be stable non-empty names")
        evidence = raw_inputs[input_key]
        if not isinstance(evidence, Mapping):
            raise RawFeatureBridgeError(f"raw input {input_key} must be an object")
        raw_hash = evidence.get("file_sha256", evidence.get("sha256"))
        if not _valid_sha256(raw_hash):
            raise RawFeatureBridgeError(f"raw input {input_key} lacks a valid SHA-256")
        item: dict[str, Any] = {
            "input_key": input_key,
            "file_sha256": str(raw_hash).lower(),
        }
        item.update(_normalized_selection(evidence))
        normalized.append(item)
    return _canonical_sha256(
        {"algorithm": RAW_CONTENT_ALGORITHM, "inputs": normalized}
    )


def raw_source_binding_fingerprint(binding: Mapping[str, Any]) -> str:
    """Fingerprint raw evidence, excluding only its derived fingerprint field."""

    if not isinstance(binding, Mapping):
        raise RawFeatureBridgeError("raw source binding must be an object")
    return _canonical_sha256(
        {key: value for key, value in binding.items() if key != "binding_fingerprint"}
    )


def validate_raw_source_binding(
    binding: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Validate stored raw evidence against manifest content hashes and row order."""

    if not isinstance(binding, Mapping):
        raise RawFeatureBridgeError("raw source binding must be an object")
    expected_binding_fields = {
        "schema_version",
        "algorithm",
        "content_hash_algorithm",
        "sample_ids",
        "samples",
        "claim_scope",
        "binding_fingerprint",
    }
    if set(binding) != expected_binding_fields:
        raise RawFeatureBridgeError("raw source binding fields are not canonical")
    if binding.get("schema_version") != "1.0":
        raise RawFeatureBridgeError("raw source binding schema_version must be 1.0")
    if binding.get("algorithm") != RAW_BINDING_ALGORITHM:
        raise RawFeatureBridgeError("unsupported raw source binding algorithm")
    if binding.get("content_hash_algorithm") != RAW_CONTENT_ALGORITHM:
        raise RawFeatureBridgeError("unsupported synchronized content hash algorithm")
    if binding.get("claim_scope") != _EXTRACTION_CLAIM_SCOPE:
        raise RawFeatureBridgeError(
            "raw extraction evidence must not claim recognition accuracy or effectiveness"
        )
    expected_ids = [str(record.get("sample_id", "")) for record in records]
    if binding.get("sample_ids") != expected_ids:
        raise RawFeatureBridgeError("raw source binding sample order does not match manifest")
    samples = binding.get("samples")
    if not isinstance(samples, list) or len(samples) != len(records):
        raise RawFeatureBridgeError("raw source binding sample evidence is incomplete")
    input_file_hashes: set[str] = set()
    for index, (sample, record) in enumerate(zip(samples, records)):
        if not isinstance(sample, Mapping):
            raise RawFeatureBridgeError(f"raw source sample {index} is not an object")
        if set(sample) != {"sample_id", "content_sha256", "inputs"}:
            raise RawFeatureBridgeError(
                f"raw source sample {index} fields are not canonical"
            )
        sample_id = str(record.get("sample_id", ""))
        if sample.get("sample_id") != sample_id:
            raise RawFeatureBridgeError(
                f"raw source binding row {index} does not match sample {sample_id}"
            )
        inputs = sample.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise RawFeatureBridgeError(f"raw source sample {sample_id} has no inputs")
        input_map: dict[str, Mapping[str, Any]] = {}
        for evidence in inputs:
            if not isinstance(evidence, Mapping):
                raise RawFeatureBridgeError(
                    f"raw source sample {sample_id} contains malformed input evidence"
                )
            key = evidence.get("input_key")
            allowed_evidence_fields = {
                "input_key",
                "file_sha256",
                "start_offset_seconds",
                "duration_seconds",
            }
            if not set(evidence).issubset(allowed_evidence_fields):
                raise RawFeatureBridgeError(
                    f"raw source sample {sample_id} contains non-canonical input evidence"
                )
            if (
                not isinstance(key, str)
                or not _SAFE_NAME.fullmatch(key)
                or key in input_map
            ):
                raise RawFeatureBridgeError(
                    f"raw source sample {sample_id} has invalid or duplicate input keys"
                )
            raw_hash = evidence.get("file_sha256")
            if not _valid_sha256(raw_hash):
                raise RawFeatureBridgeError(
                    f"raw source sample {sample_id}/{key} has an invalid file hash"
                )
            _normalized_selection(evidence)
            input_file_hashes.add(str(raw_hash).lower())
            input_map[key] = evidence
        if [str(item.get("input_key")) for item in inputs] != sorted(input_map):
            raise RawFeatureBridgeError(
                f"raw source inputs for sample {sample_id} are not canonically ordered"
            )
        computed_content_hash = synchronized_content_sha256(input_map)
        if (
            sample.get("content_sha256") != computed_content_hash
            or record.get("content_sha256") != computed_content_hash
        ):
            raise RawFeatureBridgeError(
                f"raw source bytes/windows do not match content_sha256 for {sample_id}"
            )
    expected_fingerprint = raw_source_binding_fingerprint(binding)
    if binding.get("binding_fingerprint") != expected_fingerprint:
        raise RawFeatureBridgeError("raw source binding fingerprint mismatch")
    return {
        "verified": True,
        "binding_fingerprint": expected_fingerprint,
        "sample_count": len(records),
        "unique_raw_file_count": len(input_file_hashes),
        "claim_scope": dict(_EXTRACTION_CLAIM_SCOPE),
    }


def _normalize_extractor_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping) or config.get("schema_version") != "1.0":
        raise RawFeatureBridgeError("extractor config schema_version must be 1.0")
    suite_id = config.get("extractor_suite_id")
    if not isinstance(suite_id, str) or not suite_id.strip():
        raise RawFeatureBridgeError("extractor_suite_id must be a non-empty string")
    modalities = config.get("modalities")
    if not isinstance(modalities, Mapping) or not modalities:
        raise RawFeatureBridgeError("extractor config requires modalities")
    normalized_modalities: dict[str, dict[str, Any]] = {}
    for modality in sorted(modalities):
        if (
            not isinstance(modality, str)
            or not _SAFE_NAME.fullmatch(modality)
            or modality in {"fusion", "best_unimodal_nested"}
        ):
            raise RawFeatureBridgeError(f"invalid modality name: {modality}")
        raw = modalities[modality]
        if not isinstance(raw, Mapping):
            raise RawFeatureBridgeError(f"extractor config for {modality} must be an object")
        kind = raw.get("kind")
        if kind not in SUPPORTED_EXTRACTORS:
            raise RawFeatureBridgeError(f"unsupported extractor kind for {modality}: {kind}")
        input_key = raw.get("input_key")
        if not isinstance(input_key, str) or not _SAFE_NAME.fullmatch(input_key):
            raise RawFeatureBridgeError(f"invalid input_key for modality {modality}")
        if kind == "builtin_classroom_video_visual_v1":
            allowed = {
                "kind",
                "input_key",
                "frame_count",
                "width",
                "height",
                "clip_seconds",
            }
            unknown = set(raw) - allowed
            if unknown:
                raise RawFeatureBridgeError(
                    f"unknown visual extractor fields for {modality}: {sorted(unknown)}"
                )
            normalized = {
                "kind": kind,
                "input_key": input_key,
                "frame_count": _positive_integer(
                    raw.get("frame_count", 12), field=f"{modality}.frame_count"
                ),
                "width": _positive_integer(
                    raw.get("width", 64), field=f"{modality}.width", minimum=16
                ),
                "height": _positive_integer(
                    raw.get("height", 36), field=f"{modality}.height", minimum=9
                ),
                "clip_seconds": _finite_number(
                    raw.get("clip_seconds", 10.0),
                    field=f"{modality}.clip_seconds",
                    minimum=0.2,
                ),
            }
        elif kind == "builtin_classroom_video_audio_v1":
            allowed = {"kind", "input_key", "sample_rate", "clip_seconds"}
            unknown = set(raw) - allowed
            if unknown:
                raise RawFeatureBridgeError(
                    f"unknown audio extractor fields for {modality}: {sorted(unknown)}"
                )
            normalized = {
                "kind": kind,
                "input_key": input_key,
                "sample_rate": _positive_integer(
                    raw.get("sample_rate", 8000),
                    field=f"{modality}.sample_rate",
                    minimum=8000,
                ),
                "clip_seconds": _finite_number(
                    raw.get("clip_seconds", 10.0),
                    field=f"{modality}.clip_seconds",
                    minimum=0.2,
                ),
            }
        else:
            allowed = {
                "kind",
                "input_key",
                "value_fields",
                "timestamp_field",
                "minimum_rows",
            }
            unknown = set(raw) - allowed
            if unknown:
                raise RawFeatureBridgeError(
                    f"unknown sensor extractor fields for {modality}: {sorted(unknown)}"
                )
            value_fields = raw.get("value_fields")
            if (
                not isinstance(value_fields, list)
                or not value_fields
                or any(
                    not isinstance(field, str) or not _SAFE_NAME.fullmatch(field)
                    for field in value_fields
                )
                or len(set(value_fields)) != len(value_fields)
            ):
                raise RawFeatureBridgeError(
                    f"{modality}.value_fields must be unique stable field names"
                )
            timestamp_field = raw.get("timestamp_field")
            if timestamp_field is not None and (
                not isinstance(timestamp_field, str)
                or not _SAFE_NAME.fullmatch(timestamp_field)
                or timestamp_field in value_fields
            ):
                raise RawFeatureBridgeError(f"invalid {modality}.timestamp_field")
            normalized = {
                "kind": kind,
                "input_key": input_key,
                "value_fields": list(value_fields),
                "timestamp_field": timestamp_field,
                "minimum_rows": _positive_integer(
                    raw.get("minimum_rows", 1), field=f"{modality}.minimum_rows"
                ),
            }
        normalized_modalities[modality] = normalized
    return {
        "schema_version": "1.0",
        "extractor_suite_id": suite_id.strip(),
        "modalities": normalized_modalities,
    }


def _implementation_artifacts(kind: str) -> list[dict[str, Any]]:
    paths = [(Path(__file__), "teaching_skill_miner/recognition/raw_feature_bridge.py")]
    if kind.startswith("builtin_classroom_video_"):
        paths.extend(
            [
                (
                    Path(extract_video_visual_features.__code__.co_filename),
                    "teaching_skill_miner/recognition/features.py",
                ),
                (
                    Path(probe_video.__code__.co_filename),
                    "teaching_skill_miner/recognition/datasets.py",
                ),
            ]
        )
    artifacts = []
    observed: set[str] = set()
    for path, logical_name in paths:
        if logical_name in observed:
            continue
        observed.add(logical_name)
        artifacts.append(
            {
                "logical_name": logical_name,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return artifacts


def _tool_artifact(name: str) -> dict[str, Any]:
    executable = shutil.which(name)
    if not executable:
        raise RawFeatureBridgeError(f"{name} is required for video feature extraction")
    resolved = Path(executable).resolve()
    result = subprocess.run(
        [str(resolved), "-version"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RawFeatureBridgeError(f"cannot fingerprint {name} runtime")
    first_line = (result.stdout or result.stderr).splitlines()
    return {
        "name": name,
        "binary_sha256": file_sha256(resolved),
        "version_line": first_line[0].strip() if first_line else "unknown",
    }


def _feature_provenance(
    suite_id: str, modality: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "raw feature extraction requires `pip install -e '.[recognition]'`"
        ) from exc
    kind = str(config["kind"])
    execution_entrypoint = {
        "builtin_classroom_video_visual_v1": (
            "teaching_skill_miner.recognition.features:"
            "extract_video_visual_features"
        ),
        "builtin_classroom_video_audio_v1": (
            "teaching_skill_miner.recognition.features:"
            "extract_video_audio_features"
        ),
        "builtin_numeric_sensor_summary_v1": (
            "teaching_skill_miner.recognition.raw_feature_bridge:"
            "_extract_sensor_features"
        ),
    }[kind]
    implementation_artifacts = _implementation_artifacts(kind)
    runtime: dict[str, Any] = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
    }
    if kind.startswith("builtin_classroom_video_"):
        runtime["tools"] = [_tool_artifact("ffmpeg"), _tool_artifact("ffprobe")]
    configuration = dict(config)
    configuration_fingerprint = _canonical_sha256(configuration)
    implementation_fingerprint = _canonical_sha256(implementation_artifacts)
    runtime_fingerprint = _canonical_sha256(runtime)
    weight_artifacts: list[dict[str, Any]] = []
    weight_bundle_fingerprint = _canonical_sha256(weight_artifacts)
    extractor_id = f"{suite_id}:{modality}:{kind}"
    fingerprint_payload = {
        "protocol": BRIDGE_PROTOCOL,
        "extractor_id": extractor_id,
        "extractor_kind": kind,
        "execution_entrypoint": execution_entrypoint,
        "execution_protocol": "python_api_and_argv_subprocess_without_shell_v1",
        "configuration_fingerprint": configuration_fingerprint,
        "implementation_fingerprint": implementation_fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
        "weight_bundle_fingerprint": weight_bundle_fingerprint,
    }
    return {
        "extractor_id": extractor_id,
        "extractor_fingerprint": _canonical_sha256(fingerprint_payload),
        "frozen_before_evaluation": True,
        "uses_ground_truth_labels": False,
        "fitted_on_evaluation_records": False,
        "bridge_protocol": BRIDGE_PROTOCOL,
        "extractor_kind": kind,
        "execution_entrypoint": execution_entrypoint,
        "execution_protocol": "python_api_and_argv_subprocess_without_shell_v1",
        "configuration": configuration,
        "configuration_fingerprint": configuration_fingerprint,
        "implementation_artifacts": implementation_artifacts,
        "implementation_fingerprint": implementation_fingerprint,
        "learned_weights_used": False,
        "weight_artifacts": weight_artifacts,
        "weight_bundle_fingerprint": weight_bundle_fingerprint,
        "runtime": runtime,
        "runtime_fingerprint": runtime_fingerprint,
    }


def _resolve_raw_inputs(
    record: Mapping[str, Any],
    raw_root: Path,
    *,
    hash_cache: dict[Path, str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    sample_id = str(record.get("sample_id", ""))
    raw_inputs = record.get("raw_inputs")
    if not isinstance(raw_inputs, Mapping) or not raw_inputs:
        raise RawFeatureBridgeError(f"record {sample_id} requires raw_inputs")
    resolved: dict[str, dict[str, Any]] = {}
    evidence_rows: list[dict[str, Any]] = []
    for input_key in sorted(raw_inputs):
        if not isinstance(input_key, str) or not _SAFE_NAME.fullmatch(input_key):
            raise RawFeatureBridgeError(f"record {sample_id} has an invalid raw input key")
        declared = raw_inputs[input_key]
        if not isinstance(declared, Mapping):
            raise RawFeatureBridgeError(
                f"record {sample_id} raw input {input_key} must be an object"
            )
        unknown = set(declared) - {
            "path",
            "sha256",
            "start_offset_seconds",
            "duration_seconds",
        }
        if unknown:
            raise RawFeatureBridgeError(
                f"record {sample_id}/{input_key} has unknown raw input fields: {sorted(unknown)}"
            )
        raw_path = declared.get("path")
        declared_hash = declared.get("sha256")
        if not isinstance(raw_path, str) or not raw_path.strip() or Path(raw_path).is_absolute():
            raise RawFeatureBridgeError(
                f"record {sample_id}/{input_key} path must be relative to raw_root"
            )
        if not _valid_sha256(declared_hash):
            raise RawFeatureBridgeError(
                f"record {sample_id}/{input_key} requires a declared SHA-256"
            )
        candidate = (raw_root / raw_path).resolve()
        if candidate != raw_root and raw_root not in candidate.parents:
            raise RawFeatureBridgeError(
                f"record {sample_id}/{input_key} path escapes raw_root"
            )
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        size = candidate.stat().st_size
        if size < 1:
            raise RawFeatureBridgeError(f"record {sample_id}/{input_key} is empty")
        if candidate not in hash_cache:
            hash_cache[candidate] = file_sha256(candidate)
        actual_hash = hash_cache[candidate]
        if actual_hash != str(declared_hash).lower():
            raise RawFeatureBridgeError(
                f"raw input SHA-256 mismatch for {sample_id}/{input_key}"
            )
        selection = _normalized_selection(declared)
        resolved[input_key] = {
            "path": candidate,
            "file_sha256": actual_hash,
            **selection,
        }
        evidence_rows.append(
            {
                "input_key": input_key,
                "file_sha256": actual_hash,
                **selection,
            }
        )
    computed_content_hash = synchronized_content_sha256(resolved)
    if record.get("content_sha256") != computed_content_hash:
        raise RawFeatureBridgeError(
            f"record {sample_id} content_sha256 does not bind its raw inputs/windows"
        )
    return resolved, evidence_rows


def _media_window(
    raw_input: Mapping[str, Any], probe: Mapping[str, Any], *, sample_id: str
) -> tuple[float, float]:
    if not probe.get("valid"):
        raise RawFeatureBridgeError(f"raw media for {sample_id} is not decodable")
    total_duration = float(probe.get("duration_seconds", 0.0))
    offset = float(raw_input.get("start_offset_seconds", 0.0))
    duration = float(raw_input.get("duration_seconds", total_duration - offset))
    if duration < 0.2 or offset < 0 or offset + duration > total_duration + 1e-6:
        raise RawFeatureBridgeError(
            f"raw media window for {sample_id} falls outside the decoded video timeline"
        )
    return float(probe.get("video_start_time_seconds", 0.0)) + offset, duration


def _load_sensor_rows(path: Path) -> list[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows: Any = list(csv.DictReader(handle))
    elif suffix in {".jsonl", ".ndjson"}:
        rows = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RawFeatureBridgeError(
                        f"invalid sensor JSON on line {line_number} of {path.name}"
                    ) from exc
    elif suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, Mapping):
            rows = payload.get("samples", payload.get("records"))
        else:
            rows = None
    else:
        raise RawFeatureBridgeError(
            f"unsupported sensor format for {path.name}; use CSV, JSON, or JSONL"
        )
    if not isinstance(rows, list) or not rows or any(
        not isinstance(row, Mapping) for row in rows
    ):
        raise RawFeatureBridgeError(f"sensor input {path.name} has no object rows")
    return rows


def _numeric_column(
    rows: Sequence[Mapping[str, Any]], field: str, *, source_name: str
) -> list[float]:
    values: list[float] = []
    for row_index, row in enumerate(rows):
        raw = row.get(field)
        if isinstance(raw, bool):
            raise RawFeatureBridgeError(
                f"sensor field {field} row {row_index} in {source_name} is boolean"
            )
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RawFeatureBridgeError(
                f"sensor field {field} row {row_index} in {source_name} is not numeric"
            ) from exc
        if not math.isfinite(value):
            raise RawFeatureBridgeError(
                f"sensor field {field} row {row_index} in {source_name} is non-finite"
            )
        values.append(value)
    return values


def _extract_sensor_features(
    path: Path, modality: str, config: Mapping[str, Any]
) -> tuple[list[float], list[str]]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "sensor feature extraction requires `pip install -e '.[recognition]'`"
        ) from exc
    rows = _load_sensor_rows(path)
    if len(rows) < int(config["minimum_rows"]):
        raise RawFeatureBridgeError(
            f"sensor input {path.name} has fewer than {config['minimum_rows']} rows"
        )
    timestamp_field = config.get("timestamp_field")
    if timestamp_field:
        times = np.asarray(
            _numeric_column(rows, str(timestamp_field), source_name=path.name),
            dtype=float,
        )
        if len(times) > 1 and bool((np.diff(times) < 0).any()):
            raise RawFeatureBridgeError(
                f"sensor timestamps in {path.name} are not monotonically ordered"
            )
    else:
        times = np.arange(len(rows), dtype=float)
    values: list[float] = []
    names: list[str] = []
    statistics = SENSOR_STATISTICS
    for field in config["value_fields"]:
        column = np.asarray(
            _numeric_column(rows, str(field), source_name=path.name), dtype=float
        )
        centered_time = times - times.mean()
        denominator = float((centered_time**2).sum())
        slope = (
            float((centered_time * (column - column.mean())).sum() / denominator)
            if denominator > 0
            else 0.0
        )
        computed = {
            "mean": float(column.mean()),
            "std": float(column.std()),
            "min": float(column.min()),
            "p25": float(np.quantile(column, 0.25)),
            "p50": float(np.quantile(column, 0.50)),
            "p75": float(np.quantile(column, 0.75)),
            "max": float(column.max()),
            "delta": float(column[-1] - column[0]),
            "slope": slope,
        }
        for statistic in statistics:
            values.append(computed[statistic])
            names.append(f"{modality}_{field}_{statistic}")
    return values, names


def _extract_one(
    modality: str,
    config: Mapping[str, Any],
    raw_inputs: Mapping[str, Mapping[str, Any]],
    *,
    sample_id: str,
    probe_cache: dict[Path, dict[str, Any]],
) -> tuple[list[float], list[str]]:
    input_key = str(config["input_key"])
    if input_key not in raw_inputs:
        raise RawFeatureBridgeError(
            f"record {sample_id} lacks raw input {input_key} for modality {modality}"
        )
    raw_input = raw_inputs[input_key]
    path = Path(raw_input["path"])
    kind = str(config["kind"])
    if kind == "builtin_numeric_sensor_summary_v1":
        if "start_offset_seconds" in raw_input or "duration_seconds" in raw_input:
            raise RawFeatureBridgeError(
                f"sensor modality {modality} does not support media-window selection"
            )
        return _extract_sensor_features(path, modality, config)
    if path not in probe_cache:
        probe_cache[path] = probe_video(path)
    probe = probe_cache[path]
    start_time, duration = _media_window(raw_input, probe, sample_id=sample_id)
    if kind == "builtin_classroom_video_visual_v1":
        result = extract_video_visual_features(
            path,
            duration=duration,
            start_time=start_time,
            frame_count=int(config["frame_count"]),
            width=int(config["width"]),
            height=int(config["height"]),
            clip_seconds=float(config["clip_seconds"]),
        )
    else:
        if not probe.get("has_audio"):
            raise RawFeatureBridgeError(
                f"audio modality {modality} requires decoded audio overlapping {sample_id}"
            )
        result = extract_video_audio_features(
            path,
            duration=duration,
            start_time=start_time,
            clip_seconds=float(config["clip_seconds"]),
            sample_rate=int(config["sample_rate"]),
            audio_stream_index=probe.get("selected_audio_stream_index"),
        )
        if result.get("audio_present") is not True:
            raise RawFeatureBridgeError(
                f"audio decoding produced no samples for {sample_id}/{modality}"
            )
    vector = result["features"]
    return [float(value) for value in vector], list(result["feature_names"])


def validate_bundle_against_frozen_model(
    bundle: Mapping[str, Any], model: FrozenDeploymentModel
) -> dict[str, Any]:
    """Confirm that extraction output matches a frozen model without predicting."""

    matrices = bundle.get("matrices")
    names = bundle.get("feature_names_by_modality")
    provenance = bundle.get("feature_provenance")
    if not isinstance(matrices, Mapping) or not isinstance(names, Mapping) or not isinstance(
        provenance, Mapping
    ):
        raise RawFeatureBridgeError("malformed feature bundle")
    expected = set(model.component_modalities)
    if set(matrices) != expected or set(names) != expected or set(provenance) != expected:
        raise RawFeatureBridgeError(
            "extractor config modalities do not exactly match the frozen model components"
        )
    normalized_provenance = validate_feature_provenance(
        provenance, model.component_modalities
    )
    if normalized_provenance != model.feature_provenance_by_modality:
        raise RawFeatureBridgeError(
            "raw extractor provenance does not match the frozen model"
        )
    for modality in model.component_modalities:
        if tuple(names[modality]) != model.feature_names_by_modality[modality]:
            raise RawFeatureBridgeError(
                f"raw extractor feature schema does not match frozen modality {modality}"
            )
    return {
        "verified": True,
        "model_fingerprint": model.model_fingerprint,
        "prediction_performed": False,
    }


def validate_extractor_suite_binding(
    suite: Mapping[str, Any],
    feature_names_by_modality: Mapping[str, Sequence[str]],
    feature_provenance: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the human-readable suite summary against fingerprinted provenance."""

    if not isinstance(suite, Mapping) or set(suite) != {
        "schema_version",
        "extractor_suite_id",
        "configuration_fingerprint",
        "configuration",
    }:
        raise RawFeatureBridgeError("extractor suite binding is malformed")
    if suite.get("schema_version") != "1.0":
        raise RawFeatureBridgeError("extractor suite schema_version must be 1.0")
    configuration = suite.get("configuration")
    if not isinstance(configuration, Mapping):
        raise RawFeatureBridgeError("extractor suite configuration is missing")
    normalized = _normalize_extractor_config(configuration)
    if dict(configuration) != normalized:
        raise RawFeatureBridgeError("extractor suite configuration is not normalized")
    expected_fingerprint = _canonical_sha256(normalized)
    if suite.get("configuration_fingerprint") != expected_fingerprint:
        raise RawFeatureBridgeError("extractor suite configuration fingerprint mismatch")
    suite_id = normalized["extractor_suite_id"]
    modalities = normalized["modalities"]
    if set(modalities) != set(feature_names_by_modality) or set(modalities) != set(
        feature_provenance
    ):
        raise RawFeatureBridgeError("extractor suite modalities do not match the bundle")
    for modality, config in modalities.items():
        evidence = feature_provenance[modality]
        expected_id = f"{suite_id}:{modality}:{config['kind']}"
        if (
            evidence.get("extractor_id") != expected_id
            or evidence.get("configuration") != config
            or evidence.get("configuration_fingerprint") != _canonical_sha256(config)
        ):
            raise RawFeatureBridgeError(
                f"extractor suite summary disagrees with provenance for {modality}"
            )
    return {
        "verified": True,
        "extractor_suite_id": suite_id,
        "configuration_fingerprint": expected_fingerprint,
    }


def extract_strict_feature_bundle(
    manifest: Mapping[str, Any],
    raw_root: str | Path,
    extractor_config: Mapping[str, Any],
    *,
    required_identity_fields: Sequence[str],
) -> dict[str, Any]:
    """Generate an ordered strict bundle from real raw media and sensor files."""

    normalized_config = _normalize_extractor_config(extractor_config)
    modalities = normalized_config["modalities"]
    records = manifest.get("records", [])
    validate_strict_manifest(
        manifest,
        required_modalities=list(modalities),
        required_identity_fields=required_identity_fields,
    )
    root = Path(raw_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    suite_id = str(normalized_config["extractor_suite_id"])
    provenance = {
        modality: _feature_provenance(suite_id, modality, config)
        for modality, config in modalities.items()
    }
    validate_feature_provenance(provenance, list(modalities))
    matrices: dict[str, list[list[float]]] = {
        modality: [] for modality in modalities
    }
    feature_names: dict[str, list[str]] = {}
    raw_samples: list[dict[str, Any]] = []
    hash_cache: dict[Path, str] = {}
    probe_cache: dict[Path, dict[str, Any]] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        raw_inputs, evidence_rows = _resolve_raw_inputs(
            record, root, hash_cache=hash_cache
        )
        raw_samples.append(
            {
                "sample_id": sample_id,
                "content_sha256": record["content_sha256"],
                "inputs": evidence_rows,
            }
        )
        for modality, config in modalities.items():
            vector, names = _extract_one(
                modality,
                config,
                raw_inputs,
                sample_id=sample_id,
                probe_cache=probe_cache,
            )
            if not vector or any(not math.isfinite(value) for value in vector):
                raise RawFeatureBridgeError(
                    f"extractor produced an invalid vector for {sample_id}/{modality}"
                )
            if modality not in feature_names:
                if len(set(names)) != len(names) or len(names) != len(vector):
                    raise RawFeatureBridgeError(
                        f"extractor produced an invalid schema for modality {modality}"
                    )
                feature_names[modality] = names
            elif feature_names[modality] != names:
                raise RawFeatureBridgeError(
                    f"feature schema changed between samples for modality {modality}"
                )
            matrices[modality].append(vector)
    raw_binding: dict[str, Any] = {
        "schema_version": "1.0",
        "algorithm": RAW_BINDING_ALGORITHM,
        "content_hash_algorithm": RAW_CONTENT_ALGORITHM,
        "sample_ids": [str(record["sample_id"]) for record in records],
        "samples": raw_samples,
        "claim_scope": dict(_EXTRACTION_CLAIM_SCOPE),
    }
    raw_binding["binding_fingerprint"] = raw_source_binding_fingerprint(raw_binding)
    validate_raw_source_binding(raw_binding, records)
    feature_fingerprint = strict_feature_bundle_fingerprint(
        records, feature_names, matrices, provenance
    )
    return {
        "schema_version": "1.0",
        "dataset_fingerprint": manifest["audit"]["dataset_fingerprint"],
        "feature_bundle_fingerprint": feature_fingerprint,
        "fingerprint_algorithm": "strict_feature_bundle_v1",
        "sample_ids": [str(record["sample_id"]) for record in records],
        "feature_names_by_modality": feature_names,
        "feature_provenance": provenance,
        "matrices": matrices,
        "raw_source_binding": raw_binding,
        "extractor_suite": {
            "schema_version": "1.0",
            "extractor_suite_id": suite_id,
            "configuration_fingerprint": _canonical_sha256(normalized_config),
            "configuration": normalized_config,
        },
    }
