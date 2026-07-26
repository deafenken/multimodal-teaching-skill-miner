"""Safe, deterministic frozen-model artifacts for the TeachObs four-arm study.

The artifacts produced here are private research checkpoints.  They make the
train-only models reproducible and usable for a later external lockbox, but do
not themselves establish deployment accuracy or confirmatory validity.

The format deliberately avoids pickle.  Human-readable contracts and TF-IDF
vocabularies live in canonical JSON; numeric state lives in deterministic NPZ
files whose ZIP members, dtypes, shapes, and hashes are all verified before
``numpy.load(..., allow_pickle=False)`` is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from typing import Any, Mapping, Sequence
import zipfile


FROZEN_BUNDLE_SCHEMA = "teaching_skill_miner.teachobs_frozen_four_arm_bundle.v2"
FROZEN_ARM_SCHEMA = "teaching_skill_miner.teachobs_frozen_arm.v2"
FROZEN_LOCKBOX_BINDING_SCHEMA = (
    "teaching_skill_miner.teachobs_frozen_lockbox_artifact_set.v1"
)
_ARMS = ("transcript_only", "transcript_audio", "transcript_visual", "full")
_EXPECTED_BLOCKS = {
    "transcript_only": ("transcript_tfidf",),
    "transcript_audio": ("transcript_tfidf", "audio_numeric"),
    "transcript_visual": (
        "transcript_tfidf",
        "ocr_char_tfidf",
        "ocr_word_tfidf",
        "visual_numeric",
        "clip_embedding",
    ),
    "full": (
        "transcript_tfidf",
        "audio_numeric",
        "ocr_char_tfidf",
        "ocr_word_tfidf",
        "visual_numeric",
        "clip_embedding",
    ),
}
_TEXT_BLOCKS = ("transcript_tfidf", "ocr_char_tfidf", "ocr_word_tfidf")
_NUMERIC_BLOCKS = ("audio_numeric", "visual_numeric", "clip_embedding")
_MAX_JSON_BYTES = 32 * 1024 * 1024
# The current maximum configured feature layout is far below this bound.  Keep
# the cap large enough for the frozen 39-label model but small enough that a
# self-consistent hostile manifest cannot request a gigabyte allocation.
_MAX_NPZ_BYTES = 64 * 1024 * 1024
_MAX_ARRAY_ELEMENTS = _MAX_NPZ_BYTES // 8
_MAX_NPZ_MEMBERS = 16
_SHA256_LENGTH = 64


class TeachObsFrozenModelError(ValueError):
    """Raised when a frozen artifact or prediction input fails closed."""


@dataclass(frozen=True, slots=True)
class FrozenTeachObsArm:
    """One fully validated frozen arm held in memory."""

    arm: str
    label_names: tuple[str, ...]
    label_groups: tuple[str, ...]
    ordered_blocks: tuple[str, ...]
    block_dimensions: dict[str, int]
    text_contracts: dict[str, dict[str, Any]]
    numeric_contracts: dict[str, dict[str, Any]]
    arrays: dict[str, Any]
    model_rows: tuple[dict[str, Any], ...]
    training_provenance: dict[str, Any]
    software_provenance: dict[str, Any]
    manifest_sha256: str
    manifest_file_sha256: str
    arrays_file_sha256: str


@dataclass(frozen=True, slots=True)
class FrozenTeachObsBundle:
    """All four validated frozen arms and their common integrity contract."""

    root: Path
    label_names: tuple[str, ...]
    label_groups: tuple[str, ...]
    arms: dict[str, FrozenTeachObsArm]
    training_provenance: dict[str, Any]
    software_provenance: dict[str, Any]
    shared_block_fingerprints: dict[str, str]
    bundle_fingerprint: str
    bundle_manifest_file_sha256: str
    benchmark_profile: str
    dataset_profile_fingerprint: str
    benchmark_input_fingerprint: str
    transcript_materialization_manifest_file_sha256: str
    transcript_materialization_fingerprint_sha256: str
    selected_train_lesson_ids: tuple[str, ...]
    selected_test_lesson_ids: tuple[str, ...]
    selected_train_sample_ids: tuple[str, ...]
    selected_test_sample_ids: tuple[str, ...]


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TeachObsFrozenModelError("frozen model contains non-canonical JSON") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_exact_keys(value: Any, expected: set[str], *, purpose: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise TeachObsFrozenModelError(f"frozen {purpose} fields differ from schema")
    return value


def _expected_array_names(arm: str) -> set[str]:
    names = {"model_coefficient", "model_intercept", "model_thresholds"}
    for block in _EXPECTED_BLOCKS[arm]:
        if block in _TEXT_BLOCKS:
            names.add(f"idf__{block}")
        else:
            names.add(f"scaler_mean__{block}")
            names.add(f"scaler_scale__{block}")
    return names


def _validate_array_specs(specs: Any, *, arm: str) -> Mapping[str, Any]:
    expected_names = _expected_array_names(arm)
    if not isinstance(specs, Mapping) or set(specs) != expected_names:
        raise TeachObsFrozenModelError(f"frozen array member set differs for {arm}")
    total_payload_bytes = 0
    for name, raw in specs.items():
        _require_exact_keys(
            raw, {"dtype", "shape", "sha256"}, purpose=f"array specification {name}"
        )
        shape = raw["shape"]
        if (
            raw["dtype"] != "float64"
            or not isinstance(shape, list)
            or len(shape) not in {1, 2}
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape)
            or not _is_sha256(raw["sha256"])
        ):
            raise TeachObsFrozenModelError(f"invalid frozen array specification: {name}")
        element_count = math.prod(shape)
        if element_count > _MAX_ARRAY_ELEMENTS:
            raise TeachObsFrozenModelError(f"frozen array is too large: {name}")
        total_payload_bytes += element_count * 8
    if total_payload_bytes > _MAX_NPZ_BYTES:
        raise TeachObsFrozenModelError("frozen numeric state exceeds the safe size bound")
    return specs


def _runtime_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise TeachObsFrozenModelError(
            f"required frozen-model runtime is missing: {distribution}"
        ) from exc


def _validate_runtime_and_source_provenance(
    training_provenance: Any, software_provenance: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    from . import teachobs_multimodal_benchmark as benchmark

    training = dict(
        _require_exact_keys(
            training_provenance,
            {
                "benchmark_profile",
                "profile_source_alignment",
                "published_six_lesson_intersection",
                "excluded_official_test_lesson_ids",
                "selected_train_lesson_ids",
                "selected_test_lesson_ids",
                "selected_train_sample_ids",
                "selected_test_sample_ids",
                "selected_train_sample_order_sha256",
                "selected_test_sample_order_sha256",
                "dataset_profile_fingerprint",
                "training_repository_dataset_sha256",
                "feature_manifest_sha256",
                "transcript_materialization",
                "benchmark_input_fingerprint",
                "visual_evidence_set_sha256",
                "visual_evidence_configuration_set_sha256",
                "label_order_sha256",
                "benchmark_configuration_sha256",
                "benchmark_source_sha256",
            },
            purpose="training provenance",
        )
    )
    digest_fields = {
        "selected_train_sample_order_sha256",
        "selected_test_sample_order_sha256",
        "dataset_profile_fingerprint",
        "training_repository_dataset_sha256",
        "feature_manifest_sha256",
        "benchmark_input_fingerprint",
        "visual_evidence_set_sha256",
        "visual_evidence_configuration_set_sha256",
        "label_order_sha256",
        "benchmark_configuration_sha256",
        "benchmark_source_sha256",
    }
    if not all(_is_sha256(training[field]) for field in digest_fields):
        raise TeachObsFrozenModelError("frozen training provenance is malformed")
    try:
        benchmark.validate_frozen_profile_training_binding(training)
    except benchmark.TeachObsMultimodalBenchmarkError as exc:
        raise TeachObsFrozenModelError(
            f"frozen benchmark profile binding is invalid: {exc}"
        ) from exc
    software = dict(
        _require_exact_keys(
            software_provenance,
            {"python", "numpy", "scipy", "scikit_learn", "frozen_model_module_sha256"},
            purpose="software provenance",
        )
    )
    expected_runtime = {
        "python": sys.version.split()[0],
        "numpy": _runtime_version("numpy"),
        "scipy": _runtime_version("scipy"),
        "scikit_learn": _runtime_version("scikit-learn"),
    }
    if any(software.get(key) != value for key, value in expected_runtime.items()):
        raise TeachObsFrozenModelError(
            "frozen-model runtime versions differ from the training contract"
        )
    if software.get("frozen_model_module_sha256") != _file_sha256(Path(__file__)):
        raise TeachObsFrozenModelError("frozen inference module hash differs")
    if training["benchmark_source_sha256"] != _file_sha256(Path(benchmark.__file__)):
        raise TeachObsFrozenModelError("frozen benchmark source hash differs")
    return training, software


def _write_private_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_bytes(value) + b"\n"
    path.write_bytes(payload)
    path.chmod(0o600)


def _write_deterministic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    """Write sorted NPY members with fixed ZIP metadata for byte reproducibility."""

    import numpy as np

    if not arrays or len(arrays) > _MAX_NPZ_MEMBERS:
        raise TeachObsFrozenModelError("invalid frozen numeric member count")
    total_payload_bytes = 0
    checked_arrays: dict[str, Any] = {}
    for name, raw in arrays.items():
        if not name or not all(
            character.isalnum() or character == "_" for character in name
        ):
            raise TeachObsFrozenModelError("unsafe frozen numeric member name")
        array = np.ascontiguousarray(raw, dtype=np.float64)
        total_payload_bytes += int(array.size) * int(array.dtype.itemsize)
        if (
            array.ndim not in {1, 2}
            or array.size > _MAX_ARRAY_ELEMENTS
            or total_payload_bytes > _MAX_NPZ_BYTES
            or array.dtype.hasobject
            or not np.isfinite(array).all()
        ):
            raise TeachObsFrozenModelError("frozen numeric state is unsafe")
        checked_arrays[name] = array
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(checked_arrays):
            array = checked_arrays[name]
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    path.chmod(0o600)


def _array_spec(arrays: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    import numpy as np

    result: dict[str, dict[str, Any]] = {}
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name], dtype=np.float64)
        result[name] = {
            "dtype": "float64",
            "shape": list(array.shape),
            "sha256": _array_sha256(array),
        }
    return result


def _manifest_with_fingerprint(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    checked = dict(payload)
    if field in checked:
        raise TeachObsFrozenModelError("fingerprint field already exists")
    checked[field] = _canonical_sha256(checked)
    return checked


def _validate_labels(
    label_names: Sequence[str], label_groups: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    names = tuple(label_names)
    groups = tuple(label_groups)
    if (
        len(names) != 39
        or len(groups) != len(names)
        or len(set(names)) != len(names)
        or any(not isinstance(value, str) or not value.strip() for value in names)
        or any(value not in {"visual", "nonvisual"} for value in groups)
    ):
        raise TeachObsFrozenModelError("invalid frozen TeachObs label contract")
    return names, groups


def _normalize_vocabulary(value: Any, *, dimension: int) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TeachObsFrozenModelError("TF-IDF vocabulary must be an object")
    vocabulary: dict[str, int] = {}
    for token, index in value.items():
        if (
            not isinstance(token, str)
            or not token
            or "\x00" in token
            or isinstance(index, bool)
            or not isinstance(index, int)
        ):
            raise TeachObsFrozenModelError("invalid TF-IDF vocabulary entry")
        vocabulary[token] = index
    if (
        len(vocabulary) != dimension
        or sorted(vocabulary.values()) != list(range(dimension))
    ):
        raise TeachObsFrozenModelError("TF-IDF vocabulary indices are not contiguous")
    return vocabulary


def _text_contract_for_export(
    block: str,
    state: Mapping[str, Any],
    *,
    dimension: int,
) -> tuple[dict[str, Any], Any]:
    import numpy as np

    configuration = state.get("configuration")
    vocabulary = _normalize_vocabulary(state.get("vocabulary"), dimension=dimension)
    idf = np.ascontiguousarray(state.get("idf"), dtype=np.float64)
    if (
        not isinstance(configuration, Mapping)
        or idf.shape != (dimension,)
        or not np.isfinite(idf).all()
        or (dimension and np.any(idf < 1.0))
    ):
        raise TeachObsFrozenModelError(f"invalid fitted text state for {block}")
    contract = {
        "configuration": dict(configuration),
        "vocabulary": vocabulary,
        "vocabulary_sha256": _canonical_sha256(sorted(vocabulary.items())),
        "idf_array": f"idf__{block}",
        "feature_count": dimension,
    }
    return contract, idf


def _numeric_contract_for_export(
    block: str,
    state: Mapping[str, Any],
    *,
    dimension: int,
    feature_names: Sequence[str] | None,
    provenance: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    mean = np.ascontiguousarray(state.get("mean"), dtype=np.float64)
    scale = np.ascontiguousarray(state.get("scale"), dtype=np.float64)
    if (
        mean.shape != (dimension,)
        or scale.shape != (dimension,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise TeachObsFrozenModelError(f"invalid fitted numeric state for {block}")
    names = tuple(feature_names or ())
    if block != "clip_embedding" and (
        len(names) != dimension or len(set(names)) != dimension
    ):
        raise TeachObsFrozenModelError(f"invalid numeric feature names for {block}")
    contract = {
        "feature_count": dimension,
        "feature_names": list(names) if names else None,
        "scaler_mean_array": f"scaler_mean__{block}",
        "scaler_scale_array": f"scaler_scale__{block}",
        "provenance": dict(provenance or {}),
    }
    return contract, {
        f"scaler_mean__{block}": mean,
        f"scaler_scale__{block}": scale,
    }


def export_teachobs_frozen_bundle(
    output_directory: str | Path,
    *,
    label_names: Sequence[str],
    label_groups: Sequence[str],
    feature_blocks_by_arm: Mapping[str, Mapping[str, int]],
    text_states: Mapping[str, Mapping[str, Any]],
    numeric_states: Mapping[str, Mapping[str, Any]],
    numeric_feature_names: Mapping[str, Sequence[str]],
    numeric_provenance: Mapping[str, Mapping[str, Any]],
    fitted_arm_states: Mapping[str, Mapping[str, Any]],
    training_provenance: Mapping[str, Any],
    software_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Export four standalone, deterministic, non-pickle arm artifacts."""

    import numpy as np
    from . import teachobs_multimodal_benchmark as benchmark

    names, groups = _validate_labels(label_names, label_groups)
    if tuple(feature_blocks_by_arm) != _ARMS or tuple(fitted_arm_states) != _ARMS:
        raise TeachObsFrozenModelError("frozen export does not contain exactly four arms")
    checked_software_provenance = dict(software_provenance)
    checked_software_provenance["frozen_model_module_sha256"] = _file_sha256(
        Path(__file__)
    )
    checked_training_provenance, checked_software_provenance = (
        _validate_runtime_and_source_provenance(
            training_provenance, checked_software_provenance
        )
    )
    if checked_training_provenance["label_order_sha256"] != _canonical_sha256(
        list(names)
    ):
        raise TeachObsFrozenModelError(
            "frozen training provenance label order differs from model labels"
        )

    destination = Path(output_directory).expanduser()
    if destination.exists() or destination.is_symlink():
        raise TeachObsFrozenModelError("frozen model output already exists")
    destination_parent = destination.parent.resolve()
    destination_parent.mkdir(parents=True, exist_ok=True)
    destination_parent.chmod(0o700)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination_parent)
    )
    staging.chmod(0o700)
    committed = False
    arm_entries: dict[str, dict[str, Any]] = {}
    shared_fingerprints: dict[str, str] = {}
    try:
        for arm in _ARMS:
            block_dimensions = {
                str(key): int(value)
                for key, value in feature_blocks_by_arm[arm].items()
            }
            if tuple(block_dimensions) != _EXPECTED_BLOCKS[arm] or any(
                value < 0 or (block in _NUMERIC_BLOCKS and value == 0)
                for block, value in block_dimensions.items()
            ):
                raise TeachObsFrozenModelError(f"invalid feature contract for {arm}")
            arm_directory = staging / arm
            arm_directory.mkdir(mode=0o700)
            arrays: dict[str, Any] = {}
            text_contracts: dict[str, Any] = {}
            numeric_contracts: dict[str, Any] = {}
            for block in _EXPECTED_BLOCKS[arm]:
                dimension = block_dimensions[block]
                if block in _TEXT_BLOCKS:
                    contract, idf = _text_contract_for_export(
                        block, text_states[block], dimension=dimension
                    )
                    text_contracts[block] = contract
                    arrays[contract["idf_array"]] = idf
                elif block in _NUMERIC_BLOCKS:
                    contract, block_arrays = _numeric_contract_for_export(
                        block,
                        numeric_states[block],
                        dimension=dimension,
                        feature_names=numeric_feature_names.get(block),
                        provenance=numeric_provenance.get(block),
                    )
                    numeric_contracts[block] = contract
                    arrays.update(block_arrays)
                else:  # pragma: no cover - fixed constant defense
                    raise TeachObsFrozenModelError("unknown frozen feature block")

            fitted = fitted_arm_states[arm]
            coefficient = np.ascontiguousarray(
                fitted.get("coefficient"), dtype=np.float64
            )
            intercept = np.ascontiguousarray(fitted.get("intercept"), dtype=np.float64)
            thresholds = np.ascontiguousarray(
                fitted.get("thresholds"), dtype=np.float64
            )
            total_dimension = sum(block_dimensions.values())
            if (
                coefficient.shape != (len(names), total_dimension)
                or intercept.shape != (len(names),)
                or thresholds.shape != (len(names),)
                or not np.isfinite(coefficient).all()
                or not np.isfinite(intercept).all()
                or not np.isfinite(thresholds).all()
                or np.any(thresholds <= 0)
                or np.any(thresholds >= 1)
            ):
                raise TeachObsFrozenModelError(f"invalid fitted classifier state for {arm}")
            model_kinds = fitted.get("model_kinds")
            model_classes = fitted.get("model_classes")
            regularization_c = fitted.get("regularization_c")
            class_weight = fitted.get("class_weight")
            selection_provenance_sha256 = fitted.get(
                "selection_provenance_sha256"
            )
            if (
                not isinstance(model_kinds, Sequence)
                or isinstance(model_kinds, (str, bytes))
                or not isinstance(model_classes, Sequence)
                or len(model_kinds) != len(names)
                or len(model_classes) != len(names)
                or isinstance(regularization_c, bool)
                or not isinstance(regularization_c, (int, float))
                or not math.isfinite(float(regularization_c))
                or float(regularization_c) <= 0.0
                or float(regularization_c)
                not in {
                    float(row["regularization_c"])
                    for row in benchmark._ADVANCED_CLASSIFIER_CANDIDATES
                    if row["class_weight"] == "balanced"
                }
                or class_weight != "balanced"
                or not _is_sha256(selection_provenance_sha256)
            ):
                raise TeachObsFrozenModelError(f"invalid model-kind state for {arm}")
            model_rows: list[dict[str, Any]] = []
            for index, (kind, classes) in enumerate(
                zip(model_kinds, model_classes, strict=True)
            ):
                checked_classes = list(classes) if isinstance(classes, Sequence) else []
                valid = (
                    kind == "logistic_regression" and checked_classes == [0, 1]
                ) or (
                    kind in {"constant_0", "constant_1"}
                    and checked_classes == [int(str(kind)[-1])]
                    and np.count_nonzero(coefficient[index]) == 0
                    and intercept[index] == 0
                )
                if not valid:
                    raise TeachObsFrozenModelError(f"invalid per-label state for {arm}")
                model_rows.append(
                    {
                        "label_index": index,
                        "label_name": names[index],
                        "kind": str(kind),
                        "classes": checked_classes,
                    }
                )
            arrays.update(
                {
                    "model_coefficient": coefficient,
                    "model_intercept": intercept,
                    "model_thresholds": thresholds,
                }
            )
            array_specs = _array_spec(arrays)
            block_fingerprints: dict[str, str] = {}
            for block in _EXPECTED_BLOCKS[arm]:
                contract = (
                    text_contracts[block]
                    if block in text_contracts
                    else numeric_contracts[block]
                )
                relevant_specs = {
                    key: value
                    for key, value in array_specs.items()
                    if key.endswith(f"__{block}")
                }
                fingerprint = _canonical_sha256(
                    {"contract": contract, "arrays": relevant_specs}
                )
                previous = shared_fingerprints.setdefault(block, fingerprint)
                if previous != fingerprint:
                    raise TeachObsFrozenModelError(
                        f"shared fitted block differs across arms: {block}"
                    )
                block_fingerprints[block] = fingerprint

            arrays_path = arm_directory / "arrays.npz"
            _write_deterministic_npz(arrays_path, arrays)
            arrays_file_digest = _file_sha256(arrays_path)
            algorithm = {
                "classifier": "independent_binary_logistic_regression",
                "class_weight": class_weight,
                "regularization_c": float(regularization_c),
                "solver": "liblinear",
                "maximum_iterations": 2000,
                "random_state": 0,
                "constant_train_label_handling": "predict_that_constant",
                "threshold_scope": (
                    "per_label_training_lesson_grouped_oof_frozen_before_"
                    "current_revised_test_scoring"
                ),
                "threshold_candidate_grid": list(
                    benchmark._ADVANCED_THRESHOLDS
                ),
                "selection_provenance_sha256": selection_provenance_sha256,
            }
            feature_contract = {
                "ordered_blocks": list(_EXPECTED_BLOCKS[arm]),
                "block_dimensions": block_dimensions,
                "total_feature_count": total_dimension,
                "text_blocks": text_contracts,
                "numeric_blocks": numeric_contracts,
                "shared_block_fingerprints": block_fingerprints,
            }
            manifest_payload = {
                "schema": FROZEN_ARM_SCHEMA,
                "private_artifact": True,
                "public_release_authorized": False,
                "arm": arm,
                "label_names": list(names),
                "label_groups": list(groups),
                "label_order_sha256": _canonical_sha256(list(names)),
                "feature_contract": feature_contract,
                "feature_contract_sha256": _canonical_sha256(feature_contract),
                "algorithm": algorithm,
                "algorithm_sha256": _canonical_sha256(algorithm),
                "models": model_rows,
                "arrays_file": "arrays.npz",
                "arrays_file_sha256": arrays_file_digest,
                "arrays": array_specs,
                "training_provenance": checked_training_provenance,
                "software_provenance": checked_software_provenance,
                "claim_boundary": {
                    "train_only_model_state_frozen": True,
                    "public_test_labels_were_accessible_before_freeze": True,
                    "public_test_outcomes_previously_observed_before_revision": True,
                    "exploratory_secondary_artifact": True,
                    "confirmatory_lockbox_result_established": False,
                    "deployment_accuracy_established": False,
                    "learning_effectiveness_established": False,
                },
            }
            manifest = _manifest_with_fingerprint(
                manifest_payload, "manifest_sha256"
            )
            manifest_path = arm_directory / "manifest.json"
            _write_private_json(manifest_path, manifest)
            arm_entries[arm] = {
                "directory": arm,
                "manifest_file": f"{arm}/manifest.json",
                "manifest_file_sha256": _file_sha256(manifest_path),
                "manifest_sha256": manifest["manifest_sha256"],
                "arrays_file": f"{arm}/arrays.npz",
                "arrays_file_sha256": arrays_file_digest,
            }

        bundle_payload = {
            "schema": FROZEN_BUNDLE_SCHEMA,
            "private_artifact": True,
            "public_release_authorized": False,
            "arms": list(_ARMS),
            "label_names": list(names),
            "label_groups": list(groups),
            "label_order_sha256": _canonical_sha256(list(names)),
            "training_provenance": checked_training_provenance,
            "software_provenance": checked_software_provenance,
            "shared_block_fingerprints": shared_fingerprints,
            "arm_artifacts": arm_entries,
            "claim_boundary": {
                "four_train_only_arms_frozen": True,
                "pickle_used": False,
                "safe_numeric_loading_requires_allow_pickle_false": True,
                "public_test_labels_were_accessible_before_freeze": True,
                "public_test_outcomes_previously_observed_before_revision": True,
                "confirmatory_lockbox_result_established": False,
                "deployment_accuracy_established": False,
            },
        }
        bundle = _manifest_with_fingerprint(bundle_payload, "bundle_fingerprint")
        bundle_path = staging / "bundle_manifest.json"
        _write_private_json(bundle_path, bundle)
        os.replace(staging, destination)
        committed = True
        destination.chmod(0o700)
        bundle_path = destination / "bundle_manifest.json"
        return {
            "output_directory": str(destination.resolve()),
            "bundle_fingerprint": bundle["bundle_fingerprint"],
            "bundle_manifest_file_sha256": _file_sha256(bundle_path),
            "benchmark_profile": checked_training_provenance[
                "benchmark_profile"
            ],
            "dataset_profile_fingerprint": checked_training_provenance[
                "dataset_profile_fingerprint"
            ],
            "benchmark_input_fingerprint": checked_training_provenance[
                "benchmark_input_fingerprint"
            ],
            "transcript_materialization_manifest_file_sha256": (
                checked_training_provenance["transcript_materialization"][
                    "manifest_file_sha256"
                ]
            ),
            "transcript_materialization_fingerprint_sha256": (
                checked_training_provenance["transcript_materialization"][
                    "materialization_fingerprint_sha256"
                ]
            ),
            "arm_count": len(_ARMS),
            "arms": list(_ARMS),
            "arm_model_artifacts": {
                arm: {
                    "model_manifest_path": str(
                        (destination / arm / "manifest.json").resolve()
                    ),
                    "model_manifest_file_sha256": arm_entries[arm][
                        "manifest_file_sha256"
                    ],
                    "numeric_state_file_sha256": arm_entries[arm][
                        "arrays_file_sha256"
                    ],
                }
                for arm in _ARMS
            },
            "pickle_used": False,
            "deterministic_npz": True,
            "deployment_accuracy_established": False,
        }
    finally:
        if not committed and staging.exists():
            import shutil

            shutil.rmtree(staging)


def _safe_json(path: Path, *, root: Path, purpose: str) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise TeachObsFrozenModelError(f"missing frozen {purpose}") from exc
    if (
        _path_uses_symlink(path, root=root)
        or not resolved.is_relative_to(root)
        or not resolved.is_file()
        or resolved.stat().st_size > _MAX_JSON_BYTES
    ):
        raise TeachObsFrozenModelError(f"unsafe frozen {purpose}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TeachObsFrozenModelError(f"invalid frozen {purpose} JSON") from exc
    if not isinstance(value, dict):
        raise TeachObsFrozenModelError(f"frozen {purpose} must be an object")
    return value


def _path_uses_symlink(path: Path, *, root: Path) -> bool:
    current = path
    while current != root:
        if current.is_symlink():
            return True
        parent = current.parent
        if parent == current:
            return True
        current = parent
    return root.is_symlink()


def _safe_relative(root: Path, value: Any, *, expected: str) -> Path:
    if value != expected:
        raise TeachObsFrozenModelError("frozen artifact path contract mismatch")
    relative = PurePosixPath(str(value))
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise TeachObsFrozenModelError("unsafe frozen artifact path")
    return root.joinpath(*relative.parts)


def _load_npz(
    path: Path,
    *,
    arm: str,
    root: Path,
    expected_file_sha256: str,
    specs: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    specs = _validate_array_specs(specs, arm=arm)

    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise TeachObsFrozenModelError("missing frozen numeric state") from exc
    if (
        _path_uses_symlink(path, root=root)
        or not resolved.is_relative_to(root)
        or not resolved.is_file()
        or resolved.stat().st_size > _MAX_NPZ_BYTES
        or _file_sha256(resolved) != expected_file_sha256
        or not _is_sha256(expected_file_sha256)
    ):
        raise TeachObsFrozenModelError("frozen numeric file integrity mismatch")
    expected_members = {f"{name}.npy" for name in specs}
    try:
        with zipfile.ZipFile(resolved) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if (
                len(names) != len(set(names))
                or set(names) != expected_members
                or any(
                    info.is_dir()
                    or info.file_size > _MAX_NPZ_BYTES
                    or PurePosixPath(info.filename).name != info.filename
                    or info.compress_type != zipfile.ZIP_DEFLATED
                    or bool(info.flag_bits & 0x1)
                    for info in infos
                )
                or sum(info.file_size for info in infos) > _MAX_NPZ_BYTES
            ):
                raise TeachObsFrozenModelError("unsafe frozen NPZ member layout")
            for info in infos:
                key = info.filename.removesuffix(".npy")
                raw_spec = specs.get(key)
                if not isinstance(raw_spec, Mapping):
                    raise TeachObsFrozenModelError(
                        "frozen NPZ member lacks an array specification"
                    )
                with archive.open(info) as member:
                    version = np.lib.format.read_magic(member)
                    if version == (1, 0):
                        shape, fortran_order, dtype = (
                            np.lib.format.read_array_header_1_0(member)
                        )
                    elif version == (2, 0):
                        shape, fortran_order, dtype = (
                            np.lib.format.read_array_header_2_0(member)
                        )
                    else:
                        raise TeachObsFrozenModelError(
                            "unsupported frozen NPY format version"
                        )
                    element_count = math.prod(shape)
                    expected_payload_bytes = element_count * dtype.itemsize
                    if (
                        fortran_order
                        or str(dtype) != "float64"
                        or dtype.hasobject
                        or len(shape) not in {1, 2}
                        or element_count > _MAX_ARRAY_ELEMENTS
                        or list(shape) != raw_spec.get("shape")
                        or expected_payload_bytes < 0
                        or member.tell() + expected_payload_bytes != info.file_size
                    ):
                        raise TeachObsFrozenModelError(
                            f"unsafe frozen NPY header: {key}"
                        )
    except TeachObsFrozenModelError:
        raise
    except (OSError, EOFError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise TeachObsFrozenModelError("invalid frozen NPZ archive") from exc
    arrays: dict[str, Any] = {}
    try:
        with np.load(resolved, allow_pickle=False) as archive:
            if set(archive.files) != set(specs):
                raise TeachObsFrozenModelError("frozen NPZ keys differ from manifest")
            for name, raw_spec in specs.items():
                if not isinstance(raw_spec, Mapping):
                    raise TeachObsFrozenModelError("invalid frozen array specification")
                value = np.ascontiguousarray(archive[name])
                expected_shape = raw_spec.get("shape")
                if (
                    str(value.dtype) != "float64"
                    or value.dtype.hasobject
                    or list(value.shape) != expected_shape
                    or not np.isfinite(value).all()
                    or raw_spec.get("dtype") != "float64"
                    or not _is_sha256(raw_spec.get("sha256"))
                    or _array_sha256(value) != raw_spec.get("sha256")
                ):
                    raise TeachObsFrozenModelError(
                        f"frozen array integrity mismatch: {name}"
                    )
                arrays[str(name)] = value.copy()
    except TeachObsFrozenModelError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise TeachObsFrozenModelError("frozen numeric state cannot be loaded safely") from exc
    return arrays


def _validate_text_configuration(block: str, value: Any) -> dict[str, Any]:
    from . import teachobs_multimodal_benchmark as benchmark

    expected = {
        "transcript_tfidf": {
            "analyzer": "char_wb",
            "ngram_range": list(benchmark._TFIDF_NGRAM_RANGE),
            "lowercase": True,
            "min_df": 1,
            "max_features": benchmark._TFIDF_MAX_FEATURES,
            "sublinear_tf": True,
            "norm": "l2",
            "use_idf": True,
            "smooth_idf": True,
        },
        "ocr_char_tfidf": {
            "analyzer": "char_wb",
            "ngram_range": list(benchmark._OCR_CHAR_NGRAM_RANGE),
            "lowercase": True,
            "min_df": 1,
            "max_features": benchmark._OCR_CHAR_MAX_FEATURES,
            "sublinear_tf": True,
            "norm": "l2",
            "use_idf": True,
            "smooth_idf": True,
        },
        "ocr_word_tfidf": {
            "analyzer": "word",
            "ngram_range": list(benchmark._OCR_WORD_NGRAM_RANGE),
            "lowercase": True,
            "min_df": 1,
            "max_features": benchmark._OCR_WORD_MAX_FEATURES,
            "sublinear_tf": True,
            "norm": "l2",
            "use_idf": True,
            "smooth_idf": True,
        },
    }[block]
    if value != expected:
        raise TeachObsFrozenModelError(f"frozen text configuration differs for {block}")
    return dict(expected)


def _validate_arm_manifest(
    arm: str,
    manifest: dict[str, Any],
    arrays: dict[str, Any],
    *,
    bundle_labels: tuple[str, ...],
    bundle_groups: tuple[str, ...],
    training_provenance: dict[str, Any],
    software_provenance: dict[str, Any],
    shared_fingerprints: dict[str, str],
    manifest_file_sha256: str,
) -> FrozenTeachObsArm:
    from . import teachobs_multimodal_benchmark as benchmark

    _require_exact_keys(
        manifest,
        {
            "schema",
            "private_artifact",
            "public_release_authorized",
            "arm",
            "label_names",
            "label_groups",
            "label_order_sha256",
            "feature_contract",
            "feature_contract_sha256",
            "algorithm",
            "algorithm_sha256",
            "models",
            "arrays_file",
            "arrays_file_sha256",
            "arrays",
            "training_provenance",
            "software_provenance",
            "claim_boundary",
            "manifest_sha256",
        },
        purpose=f"{arm} arm manifest",
    )
    fingerprint = manifest.get("manifest_sha256")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        not _is_sha256(fingerprint)
        or _canonical_sha256(payload) != fingerprint
        or manifest.get("schema") != FROZEN_ARM_SCHEMA
        or manifest.get("private_artifact") is not True
        or manifest.get("public_release_authorized") is not False
        or manifest.get("arm") != arm
        or tuple(manifest.get("label_names", ())) != bundle_labels
        or tuple(manifest.get("label_groups", ())) != bundle_groups
        or manifest.get("label_order_sha256") != _canonical_sha256(list(bundle_labels))
        or manifest.get("training_provenance") != training_provenance
        or manifest.get("software_provenance") != software_provenance
    ):
        raise TeachObsFrozenModelError(f"frozen arm manifest binding mismatch: {arm}")
    contract = manifest.get("feature_contract")
    if not isinstance(contract, Mapping) or set(contract) != {
        "ordered_blocks",
        "block_dimensions",
        "total_feature_count",
        "text_blocks",
        "numeric_blocks",
        "shared_block_fingerprints",
    } or manifest.get(
        "feature_contract_sha256"
    ) != _canonical_sha256(contract):
        raise TeachObsFrozenModelError(f"frozen feature contract hash mismatch: {arm}")
    ordered_blocks = tuple(contract.get("ordered_blocks", ()))
    dimensions = contract.get("block_dimensions")
    if ordered_blocks != _EXPECTED_BLOCKS[arm] or not isinstance(dimensions, Mapping):
        raise TeachObsFrozenModelError(f"frozen feature order mismatch: {arm}")
    block_dimensions: dict[str, int] = {}
    for block in ordered_blocks:
        dimension = dimensions.get(block)
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 0
            or (block in _NUMERIC_BLOCKS and dimension == 0)
        ):
            raise TeachObsFrozenModelError(f"invalid frozen feature dimension: {block}")
        block_dimensions[block] = dimension
    if set(dimensions) != set(ordered_blocks) or contract.get(
        "total_feature_count"
    ) != sum(block_dimensions.values()):
        raise TeachObsFrozenModelError(f"frozen total dimension mismatch: {arm}")
    if block_dimensions.get("audio_numeric", len(benchmark._AUDIO_FEATURE_NAMES)) != len(
        benchmark._AUDIO_FEATURE_NAMES
    ) or block_dimensions.get(
        "visual_numeric", len(benchmark._VISUAL_NUMERIC_FEATURE_NAMES)
    ) != len(benchmark._VISUAL_NUMERIC_FEATURE_NAMES):
        raise TeachObsFrozenModelError("fixed numeric feature dimension mismatch")

    raw_text_contracts = contract.get("text_blocks")
    raw_numeric_contracts = contract.get("numeric_blocks")
    if not isinstance(raw_text_contracts, Mapping) or not isinstance(
        raw_numeric_contracts, Mapping
    ):
        raise TeachObsFrozenModelError("frozen block contracts are malformed")
    text_contracts: dict[str, dict[str, Any]] = {}
    numeric_contracts: dict[str, dict[str, Any]] = {}
    calculated_fingerprints: dict[str, str] = {}
    for block in ordered_blocks:
        dimension = block_dimensions[block]
        if block in _TEXT_BLOCKS:
            raw = raw_text_contracts.get(block)
            if (
                not isinstance(raw, Mapping)
                or set(raw)
                != {
                    "configuration",
                    "vocabulary",
                    "vocabulary_sha256",
                    "idf_array",
                    "feature_count",
                }
                or raw.get("feature_count") != dimension
            ):
                raise TeachObsFrozenModelError(f"invalid frozen text contract: {block}")
            configuration = _validate_text_configuration(
                block, raw.get("configuration")
            )
            vocabulary = _normalize_vocabulary(
                raw.get("vocabulary"), dimension=dimension
            )
            array_name = f"idf__{block}"
            idf = arrays.get(array_name)
            if (
                raw.get("idf_array") != array_name
                or raw.get("vocabulary_sha256")
                != _canonical_sha256(sorted(vocabulary.items()))
                or idf is None
                or idf.shape != (dimension,)
                or (dimension and (idf < 1.0).any())
            ):
                raise TeachObsFrozenModelError(f"frozen TF-IDF state mismatch: {block}")
            text_contracts[block] = {
                **dict(raw),
                "configuration": configuration,
                "vocabulary": vocabulary,
            }
        else:
            raw = raw_numeric_contracts.get(block)
            if (
                not isinstance(raw, Mapping)
                or set(raw)
                != {
                    "feature_count",
                    "feature_names",
                    "scaler_mean_array",
                    "scaler_scale_array",
                    "provenance",
                }
                or raw.get("feature_count") != dimension
            ):
                raise TeachObsFrozenModelError(f"invalid frozen numeric contract: {block}")
            mean_name = f"scaler_mean__{block}"
            scale_name = f"scaler_scale__{block}"
            mean = arrays.get(mean_name)
            scale = arrays.get(scale_name)
            if (
                raw.get("scaler_mean_array") != mean_name
                or raw.get("scaler_scale_array") != scale_name
                or mean is None
                or scale is None
                or mean.shape != (dimension,)
                or scale.shape != (dimension,)
                or (scale <= 0).any()
            ):
                raise TeachObsFrozenModelError(f"frozen scaler state mismatch: {block}")
            expected_names: tuple[str, ...] | None
            if block == "audio_numeric":
                expected_names = benchmark._AUDIO_FEATURE_NAMES
            elif block == "visual_numeric":
                expected_names = benchmark._VISUAL_NUMERIC_FEATURE_NAMES
            else:
                expected_names = None
                provenance = raw.get("provenance")
                if (
                    not isinstance(provenance, Mapping)
                    or not str(provenance.get("clip_source_revision", "")).strip()
                    or not _is_sha256(
                        provenance.get("clip_weight_manifest_sha256")
                    )
                ):
                    raise TeachObsFrozenModelError("invalid frozen CLIP provenance")
            if expected_names is not None and tuple(raw.get("feature_names") or ()) != tuple(
                expected_names
            ):
                raise TeachObsFrozenModelError(f"frozen feature names mismatch: {block}")
            if expected_names is None and raw.get("feature_names") is not None:
                raise TeachObsFrozenModelError("CLIP must be dimension-bound, not name-invented")
            provenance = raw.get("provenance")
            expected_provenance_fields = {
                "audio_numeric": {
                    "feature_schema_sha256",
                    "audio_feature_schema",
                },
                "visual_numeric": {
                    "feature_schema_sha256",
                    "visual_evidence_configuration_set_sha256",
                    "visual_evidence_schema",
                    "configuration_sha256_values",
                },
                "clip_embedding": {
                    "clip_source_revision",
                    "clip_weight_manifest_sha256",
                },
            }[block]
            if (
                not isinstance(provenance, Mapping)
                or set(provenance) != expected_provenance_fields
                or any(
                    not _is_sha256(provenance[key])
                    for key in provenance
                    if key
                    in {
                        "feature_schema_sha256",
                        "visual_evidence_configuration_set_sha256",
                        "clip_weight_manifest_sha256",
                    }
                )
            ):
                raise TeachObsFrozenModelError(
                    f"invalid frozen numeric provenance: {block}"
                )
            if block in {"audio_numeric", "visual_numeric"} and provenance[
                "feature_schema_sha256"
            ] != _canonical_sha256(list(expected_names or ())):
                raise TeachObsFrozenModelError(
                    f"frozen feature schema hash differs from names: {block}"
                )
            if block == "audio_numeric" and provenance[
                "audio_feature_schema"
            ] != benchmark.AUDIO_SCHEMA:
                raise TeachObsFrozenModelError("frozen audio feature schema differs")
            if block == "visual_numeric":
                configuration_values = provenance["configuration_sha256_values"]
                if (
                    provenance["visual_evidence_schema"]
                    != benchmark.VISUAL_EVIDENCE_SCHEMA
                    or not isinstance(configuration_values, list)
                    or len(configuration_values) != 1
                    or not _is_sha256(configuration_values[0])
                ):
                    raise TeachObsFrozenModelError(
                        "frozen visual extractor configuration is ambiguous"
                    )
            numeric_contracts[block] = dict(raw)
        block_contract = (
            text_contracts[block]
            if block in text_contracts
            else numeric_contracts[block]
        )
        relevant_specs = {
            key: value
            for key, value in manifest["arrays"].items()
            if key.endswith(f"__{block}")
        }
        calculated_fingerprints[block] = _canonical_sha256(
            {"contract": block_contract, "arrays": relevant_specs}
        )
    if (
        set(raw_text_contracts) != set(text_contracts)
        or set(raw_numeric_contracts) != set(numeric_contracts)
        or contract.get("shared_block_fingerprints") != calculated_fingerprints
        or any(shared_fingerprints.get(key) != value for key, value in calculated_fingerprints.items())
    ):
        raise TeachObsFrozenModelError("shared frozen block fingerprint mismatch")

    coefficient = arrays.get("model_coefficient")
    intercept = arrays.get("model_intercept")
    thresholds = arrays.get("model_thresholds")
    total = sum(block_dimensions.values())
    if (
        coefficient is None
        or intercept is None
        or thresholds is None
        or coefficient.shape != (len(bundle_labels), total)
        or intercept.shape != (len(bundle_labels),)
        or thresholds.shape != (len(bundle_labels),)
        or (thresholds <= 0).any()
        or (thresholds >= 1).any()
        or any(
            float(value) not in benchmark._ADVANCED_THRESHOLDS
            for value in thresholds
        )
    ):
        raise TeachObsFrozenModelError(f"frozen classifier dimension mismatch: {arm}")
    model_rows = manifest.get("models")
    if not isinstance(model_rows, list) or len(model_rows) != len(bundle_labels):
        raise TeachObsFrozenModelError(f"frozen model rows are malformed: {arm}")
    checked_rows: list[dict[str, Any]] = []
    for index, row in enumerate(model_rows):
        if not isinstance(row, dict) or set(row) != {
            "label_index",
            "label_name",
            "kind",
            "classes",
        }:
            raise TeachObsFrozenModelError("frozen model row is not an object")
        kind = row.get("kind")
        classes = row.get("classes")
        valid = kind == "logistic_regression" and classes == [0, 1]
        if kind in {"constant_0", "constant_1"}:
            constant = int(str(kind)[-1])
            valid = (
                classes == [constant]
                and not coefficient[index].any()
                and intercept[index] == 0
            )
        if (
            not valid
            or row.get("label_index") != index
            or row.get("label_name") != bundle_labels[index]
        ):
            raise TeachObsFrozenModelError(f"frozen label model mismatch: {arm}")
        checked_rows.append(dict(row))
    algorithm = manifest.get("algorithm")
    expected_algorithm_fields = {
        "classifier",
        "class_weight",
        "regularization_c",
        "solver",
        "maximum_iterations",
        "random_state",
        "constant_train_label_handling",
        "threshold_scope",
        "threshold_candidate_grid",
        "selection_provenance_sha256",
    }
    allowed_regularization = {
        float(row["regularization_c"])
        for row in benchmark._ADVANCED_CLASSIFIER_CANDIDATES
        if row["class_weight"] == "balanced"
    }
    valid_algorithm = (
        isinstance(algorithm, Mapping)
        and set(algorithm) == expected_algorithm_fields
        and algorithm["classifier"]
        == "independent_binary_logistic_regression"
        and algorithm["class_weight"] == "balanced"
        and algorithm["regularization_c"] in allowed_regularization
        and algorithm["solver"] == "liblinear"
        and algorithm["maximum_iterations"] == 2000
        and algorithm["random_state"] == 0
        and algorithm["constant_train_label_handling"]
        == "predict_that_constant"
        and algorithm["threshold_scope"]
        == (
            "per_label_training_lesson_grouped_oof_frozen_before_"
            "current_revised_test_scoring"
        )
        and algorithm["threshold_candidate_grid"]
        == list(benchmark._ADVANCED_THRESHOLDS)
        and _is_sha256(algorithm["selection_provenance_sha256"])
    )
    if (
        not valid_algorithm
        or manifest.get("algorithm_sha256") != _canonical_sha256(algorithm)
    ):
        raise TeachObsFrozenModelError("frozen classifier configuration mismatch")
    expected_claim_boundary = {
        "train_only_model_state_frozen": True,
        "public_test_labels_were_accessible_before_freeze": True,
        "public_test_outcomes_previously_observed_before_revision": True,
        "exploratory_secondary_artifact": True,
        "confirmatory_lockbox_result_established": False,
        "deployment_accuracy_established": False,
        "learning_effectiveness_established": False,
    }
    if manifest.get("claim_boundary") != expected_claim_boundary:
        raise TeachObsFrozenModelError("frozen arm claim boundary is weakened")
    return FrozenTeachObsArm(
        arm=arm,
        label_names=bundle_labels,
        label_groups=bundle_groups,
        ordered_blocks=ordered_blocks,
        block_dimensions=block_dimensions,
        text_contracts=text_contracts,
        numeric_contracts=numeric_contracts,
        arrays=arrays,
        model_rows=tuple(checked_rows),
        training_provenance=dict(training_provenance),
        software_provenance=dict(software_provenance),
        manifest_sha256=str(fingerprint),
        manifest_file_sha256=manifest_file_sha256,
        arrays_file_sha256=str(manifest["arrays_file_sha256"]),
    )


def load_teachobs_frozen_bundle(
    directory: str | Path,
    *,
    expected_bundle_manifest_file_sha256: str | None = None,
    expected_benchmark_profile: str | None = None,
    expected_dataset_profile_fingerprint: str | None = None,
    expected_transcript_materialization_fingerprint: str | None = None,
    expected_benchmark_input_fingerprint: str | None = None,
) -> FrozenTeachObsBundle:
    """Load and validate a four-arm bundle without executing serialized code."""

    root_value = Path(directory).expanduser()
    try:
        root = root_value.resolve(strict=True)
    except FileNotFoundError as exc:
        raise TeachObsFrozenModelError("frozen model directory is missing") from exc
    if root_value.is_symlink() or not root.is_dir():
        raise TeachObsFrozenModelError("frozen model directory is unsafe")
    bundle_path = root / "bundle_manifest.json"
    bundle = _safe_json(bundle_path, root=root, purpose="bundle manifest")
    bundle_file_digest = _file_sha256(bundle_path)
    if expected_bundle_manifest_file_sha256 is not None and (
        not _is_sha256(expected_bundle_manifest_file_sha256)
        or bundle_file_digest != expected_bundle_manifest_file_sha256
    ):
        raise TeachObsFrozenModelError("pinned frozen bundle hash mismatch")
    fingerprint = bundle.get("bundle_fingerprint")
    payload = {key: value for key, value in bundle.items() if key != "bundle_fingerprint"}
    _require_exact_keys(
        bundle,
        {
            "schema",
            "private_artifact",
            "public_release_authorized",
            "arms",
            "label_names",
            "label_groups",
            "label_order_sha256",
            "training_provenance",
            "software_provenance",
            "shared_block_fingerprints",
            "arm_artifacts",
            "claim_boundary",
            "bundle_fingerprint",
        },
        purpose="bundle manifest",
    )
    names, groups = _validate_labels(
        bundle.get("label_names", ()), bundle.get("label_groups", ())
    )
    training_provenance, software_provenance = (
        _validate_runtime_and_source_provenance(
            bundle.get("training_provenance"), bundle.get("software_provenance")
        )
    )
    if (
        expected_benchmark_profile is not None
        and training_provenance["benchmark_profile"]
        != expected_benchmark_profile
    ) or (
        expected_dataset_profile_fingerprint is not None
        and training_provenance["dataset_profile_fingerprint"]
        != expected_dataset_profile_fingerprint
    ):
        raise TeachObsFrozenModelError(
            "loaded frozen benchmark profile differs from the pinned profile"
        )
    transcript_materialization = training_provenance[
        "transcript_materialization"
    ]
    if (
        expected_transcript_materialization_fingerprint is not None
        and (
            not _is_sha256(expected_transcript_materialization_fingerprint)
            or transcript_materialization[
                "materialization_fingerprint_sha256"
            ]
            != expected_transcript_materialization_fingerprint
        )
    ):
        raise TeachObsFrozenModelError(
            "loaded frozen transcript materialization differs from the pinned input"
        )
    if (
        expected_benchmark_input_fingerprint is not None
        and (
            not _is_sha256(expected_benchmark_input_fingerprint)
            or training_provenance["benchmark_input_fingerprint"]
            != expected_benchmark_input_fingerprint
        )
    ):
        raise TeachObsFrozenModelError(
            "loaded frozen benchmark input fingerprint differs"
        )
    shared_fingerprints = bundle.get("shared_block_fingerprints")
    entries = bundle.get("arm_artifacts")
    if (
        bundle.get("schema") != FROZEN_BUNDLE_SCHEMA
        or bundle.get("private_artifact") is not True
        or bundle.get("public_release_authorized") is not False
        or tuple(bundle.get("arms", ())) != _ARMS
        or not _is_sha256(fingerprint)
        or _canonical_sha256(payload) != fingerprint
        or bundle.get("label_order_sha256") != _canonical_sha256(list(names))
        or training_provenance["label_order_sha256"]
        != _canonical_sha256(list(names))
        or not isinstance(shared_fingerprints, dict)
        or set(shared_fingerprints) != set(_TEXT_BLOCKS + _NUMERIC_BLOCKS)
        or not all(_is_sha256(value) for value in shared_fingerprints.values())
        or not isinstance(entries, dict)
        or set(entries) != set(_ARMS)
        or bundle.get("claim_boundary")
        != {
            "four_train_only_arms_frozen": True,
            "pickle_used": False,
            "safe_numeric_loading_requires_allow_pickle_false": True,
            "public_test_labels_were_accessible_before_freeze": True,
            "public_test_outcomes_previously_observed_before_revision": True,
            "confirmatory_lockbox_result_established": False,
            "deployment_accuracy_established": False,
        }
    ):
        raise TeachObsFrozenModelError("frozen bundle manifest binding mismatch")
    arms: dict[str, FrozenTeachObsArm] = {}
    for arm in _ARMS:
        entry = entries[arm]
        if (
            not isinstance(entry, Mapping)
            or set(entry)
            != {
                "directory",
                "manifest_file",
                "manifest_file_sha256",
                "manifest_sha256",
                "arrays_file",
                "arrays_file_sha256",
            }
            or entry.get("directory") != arm
            or not _is_sha256(entry.get("manifest_sha256"))
            or not _is_sha256(entry.get("arrays_file_sha256"))
        ):
            raise TeachObsFrozenModelError("frozen arm bundle entry is malformed")
        manifest_path = _safe_relative(
            root, entry.get("manifest_file"), expected=f"{arm}/manifest.json"
        )
        arrays_path = _safe_relative(
            root, entry.get("arrays_file"), expected=f"{arm}/arrays.npz"
        )
        manifest = _safe_json(manifest_path, root=root, purpose=f"{arm} manifest")
        if (
            not _is_sha256(entry.get("manifest_file_sha256"))
            or _file_sha256(manifest_path) != entry.get("manifest_file_sha256")
        ):
            raise TeachObsFrozenModelError("frozen arm manifest file hash mismatch")
        if (
            manifest.get("manifest_sha256") != entry.get("manifest_sha256")
            or manifest.get("arrays_file_sha256")
            != entry.get("arrays_file_sha256")
            or manifest.get("arrays_file") != "arrays.npz"
        ):
            raise TeachObsFrozenModelError("frozen arm entry/manifest hash mismatch")
        arrays = _load_npz(
            arrays_path,
            arm=arm,
            root=root,
            expected_file_sha256=str(entry.get("arrays_file_sha256")),
            specs=manifest.get("arrays"),
        )
        arms[arm] = _validate_arm_manifest(
            arm,
            manifest,
            arrays,
            bundle_labels=names,
            bundle_groups=groups,
            training_provenance=training_provenance,
            software_provenance=software_provenance,
            shared_fingerprints={str(key): str(value) for key, value in shared_fingerprints.items()},
            manifest_file_sha256=str(entry.get("manifest_file_sha256")),
        )
    return FrozenTeachObsBundle(
        root=root,
        label_names=names,
        label_groups=groups,
        arms=arms,
        training_provenance=dict(training_provenance),
        software_provenance=dict(software_provenance),
        shared_block_fingerprints={
            str(key): str(value) for key, value in shared_fingerprints.items()
        },
        bundle_fingerprint=str(fingerprint),
        bundle_manifest_file_sha256=bundle_file_digest,
        benchmark_profile=str(training_provenance["benchmark_profile"]),
        dataset_profile_fingerprint=str(
            training_provenance["dataset_profile_fingerprint"]
        ),
        benchmark_input_fingerprint=str(
            training_provenance["benchmark_input_fingerprint"]
        ),
        transcript_materialization_manifest_file_sha256=str(
            transcript_materialization["manifest_file_sha256"]
        ),
        transcript_materialization_fingerprint_sha256=str(
            transcript_materialization["materialization_fingerprint_sha256"]
        ),
        selected_train_lesson_ids=tuple(
            training_provenance["selected_train_lesson_ids"]
        ),
        selected_test_lesson_ids=tuple(
            training_provenance["selected_test_lesson_ids"]
        ),
        selected_train_sample_ids=tuple(
            training_provenance["selected_train_sample_ids"]
        ),
        selected_test_sample_ids=tuple(
            training_provenance["selected_test_sample_ids"]
        ),
    )


def load_teachobs_frozen_arm(
    manifest_path: str | Path,
    *,
    expected_manifest_file_sha256: str | None = None,
    expected_benchmark_profile: str | None = None,
    expected_dataset_profile_fingerprint: str | None = None,
    expected_transcript_materialization_fingerprint: str | None = None,
    expected_benchmark_input_fingerprint: str | None = None,
) -> FrozenTeachObsArm:
    """Independently load one arm from its manifest-anchored artifact pair."""

    manifest_value = Path(manifest_path).expanduser()
    try:
        resolved_manifest = manifest_value.resolve(strict=True)
    except FileNotFoundError as exc:
        raise TeachObsFrozenModelError("frozen arm manifest is missing") from exc
    arm_root = resolved_manifest.parent
    if manifest_value.is_symlink() or not resolved_manifest.is_file():
        raise TeachObsFrozenModelError("frozen arm manifest is unsafe")
    manifest = _safe_json(
        resolved_manifest, root=arm_root, purpose="standalone arm manifest"
    )
    manifest_file_digest = _file_sha256(resolved_manifest)
    if expected_manifest_file_sha256 is not None and (
        not _is_sha256(expected_manifest_file_sha256)
        or manifest_file_digest != expected_manifest_file_sha256
    ):
        raise TeachObsFrozenModelError("pinned frozen arm manifest hash mismatch")
    arm = str(manifest.get("arm", ""))
    if arm not in _ARMS:
        raise TeachObsFrozenModelError("unknown standalone frozen arm")
    names, groups = _validate_labels(
        manifest.get("label_names", ()), manifest.get("label_groups", ())
    )
    training_provenance, software_provenance = (
        _validate_runtime_and_source_provenance(
            manifest.get("training_provenance"), manifest.get("software_provenance")
        )
    )
    if (
        expected_benchmark_profile is not None
        and training_provenance["benchmark_profile"]
        != expected_benchmark_profile
    ) or (
        expected_dataset_profile_fingerprint is not None
        and training_provenance["dataset_profile_fingerprint"]
        != expected_dataset_profile_fingerprint
    ):
        raise TeachObsFrozenModelError(
            "loaded frozen benchmark profile differs from the pinned profile"
        )
    transcript_materialization = training_provenance[
        "transcript_materialization"
    ]
    if (
        expected_transcript_materialization_fingerprint is not None
        and (
            not _is_sha256(expected_transcript_materialization_fingerprint)
            or transcript_materialization[
                "materialization_fingerprint_sha256"
            ]
            != expected_transcript_materialization_fingerprint
        )
    ):
        raise TeachObsFrozenModelError(
            "loaded frozen transcript materialization differs from the pinned input"
        )
    if (
        expected_benchmark_input_fingerprint is not None
        and (
            not _is_sha256(expected_benchmark_input_fingerprint)
            or training_provenance["benchmark_input_fingerprint"]
            != expected_benchmark_input_fingerprint
        )
    ):
        raise TeachObsFrozenModelError(
            "loaded frozen benchmark input fingerprint differs"
        )
    contract = manifest.get("feature_contract")
    shared = contract.get("shared_block_fingerprints") if isinstance(contract, Mapping) else None
    specs = manifest.get("arrays")
    if (
        not isinstance(shared, dict)
        or set(shared) != set(_EXPECTED_BLOCKS[arm])
        or not all(_is_sha256(value) for value in shared.values())
        or not isinstance(specs, Mapping)
        or manifest.get("arrays_file") != "arrays.npz"
    ):
        raise TeachObsFrozenModelError("standalone frozen arm contract is malformed")
    arrays_path = arm_root / "arrays.npz"
    arrays = _load_npz(
        arrays_path,
        arm=arm,
        root=arm_root,
        expected_file_sha256=str(manifest.get("arrays_file_sha256")),
        specs=specs,
    )
    return _validate_arm_manifest(
        arm,
        manifest,
        arrays,
        bundle_labels=names,
        bundle_groups=groups,
        training_provenance=training_provenance,
        software_provenance=software_provenance,
        shared_fingerprints={str(key): str(value) for key, value in shared.items()},
        manifest_file_sha256=manifest_file_digest,
    )


def verify_teachobs_frozen_artifact_set(
    bundle_manifest_path: str | Path,
    arm_manifest_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Validate and summarize one exact bundle/manifest/NPZ tree for lockbox use.

    The returned path-free binding is safe to embed in a preregistration.  A
    bundle-manifest digest alone is transitive only when its four declared arm
    manifests and their companion arrays are actually present and reloaded;
    this function performs that full check and rejects copied or cross-bundle
    arm manifests.
    """

    supplied = dict(arm_manifest_paths)
    if set(supplied) != set(_ARMS):
        raise TeachObsFrozenModelError(
            "frozen artifact verification requires exactly four arm manifests"
        )
    bundle_value = Path(bundle_manifest_path).expanduser()
    try:
        resolved_bundle_manifest = bundle_value.resolve(strict=True)
    except FileNotFoundError as exc:
        raise TeachObsFrozenModelError("frozen bundle manifest is missing") from exc
    if (
        bundle_value.is_symlink()
        or not resolved_bundle_manifest.is_file()
        or resolved_bundle_manifest.name != "bundle_manifest.json"
    ):
        raise TeachObsFrozenModelError("frozen system artifact must be bundle_manifest.json")
    bundle_file_sha256 = _file_sha256(resolved_bundle_manifest)
    bundle = load_teachobs_frozen_bundle(
        resolved_bundle_manifest.parent,
        expected_bundle_manifest_file_sha256=bundle_file_sha256,
    )
    arm_bindings: dict[str, dict[str, str]] = {}
    for arm in _ARMS:
        supplied_value = Path(supplied[arm]).expanduser()
        try:
            resolved_supplied = supplied_value.resolve(strict=True)
        except FileNotFoundError as exc:
            raise TeachObsFrozenModelError(
                f"frozen arm manifest is missing: {arm}"
            ) from exc
        expected = (bundle.root / arm / "manifest.json").resolve(strict=True)
        if (
            supplied_value.is_symlink()
            or not resolved_supplied.is_file()
            or resolved_supplied != expected
        ):
            raise TeachObsFrozenModelError(
                f"{arm} manifest is not the arm declared by this bundle"
            )
        loaded_arm = load_teachobs_frozen_arm(
            resolved_supplied,
            expected_manifest_file_sha256=bundle.arms[arm].manifest_file_sha256,
        )
        if (
            loaded_arm.manifest_sha256 != bundle.arms[arm].manifest_sha256
            or loaded_arm.arrays_file_sha256 != bundle.arms[arm].arrays_file_sha256
        ):
            raise TeachObsFrozenModelError(
                f"{arm} standalone and bundle bindings differ"
            )
        arm_bindings[arm] = {
            "manifest_file_sha256": loaded_arm.manifest_file_sha256,
            "manifest_sha256": loaded_arm.manifest_sha256,
            "arrays_file_sha256": loaded_arm.arrays_file_sha256,
        }
    binding: dict[str, Any] = {
        "schema": FROZEN_LOCKBOX_BINDING_SCHEMA,
        "bound": True,
        "bundle_manifest_file_sha256": bundle_file_sha256,
        "bundle_fingerprint": bundle.bundle_fingerprint,
        "label_order_sha256": _canonical_sha256(list(bundle.label_names)),
        "training_provenance_sha256": _canonical_sha256(bundle.training_provenance),
        "software_provenance_sha256": _canonical_sha256(bundle.software_provenance),
        "arms": arm_bindings,
        "companion_arrays_loaded_with_allow_pickle_false": True,
        "paths_included": False,
    }
    binding["artifact_set_fingerprint"] = _canonical_sha256(binding)
    return binding


def _transform_text(
    contract: Mapping[str, Any], idf: Any, documents: Sequence[str]
) -> Any:
    import numpy as np
    from scipy.sparse import csr_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer

    dimension = int(contract["feature_count"])
    if not all(isinstance(value, str) for value in documents):
        raise TeachObsFrozenModelError("frozen text prediction input must be strings")
    if dimension == 0:
        return csr_matrix((len(documents), 0), dtype=np.float64)
    configuration = contract["configuration"]
    vectorizer = TfidfVectorizer(
        analyzer=configuration["analyzer"],
        ngram_range=tuple(configuration["ngram_range"]),
        lowercase=configuration["lowercase"],
        min_df=configuration["min_df"],
        max_features=configuration["max_features"],
        sublinear_tf=configuration["sublinear_tf"],
        norm=configuration["norm"],
        use_idf=configuration["use_idf"],
        smooth_idf=configuration["smooth_idf"],
        vocabulary=contract["vocabulary"],
        dtype=np.float64,
    )
    vectorizer.idf_ = np.asarray(idf, dtype=np.float64)
    result = vectorizer.transform(documents)
    if result.shape != (len(documents), dimension):
        raise TeachObsFrozenModelError("frozen TF-IDF output shape mismatch")
    return result


def predict_teachobs_frozen_bundle(
    bundle: FrozenTeachObsBundle,
    *,
    transcripts: Sequence[str],
    transcript_input_materialization_fingerprint_sha256: str,
    ocr_texts: Sequence[str],
    audio: Any,
    visual_numeric: Any,
    clip: Any,
    label_names: Sequence[str],
    audio_feature_names: Sequence[str],
    visual_numeric_feature_names: Sequence[str],
    clip_source_revision: str,
    clip_weight_manifest_sha256: str,
    audio_feature_schema: str,
    visual_evidence_schema: str,
    visual_evidence_configuration_sha256: str,
    sample_ids: Sequence[str] | None = None,
    benchmark_profile: str | None = None,
    dataset_profile_fingerprint: str | None = None,
    expected_prediction_input_fingerprint: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Predict all four arms after validating the external feature contract."""

    import numpy as np
    from scipy.sparse import hstack

    if tuple(label_names) != bundle.label_names:
        raise TeachObsFrozenModelError("prediction label order differs from frozen model")
    if not _is_sha256(transcript_input_materialization_fingerprint_sha256):
        raise TeachObsFrozenModelError(
            "prediction transcript materialization fingerprint is invalid"
        )
    checked_profile = benchmark_profile or bundle.benchmark_profile
    checked_profile_fingerprint = (
        dataset_profile_fingerprint or bundle.dataset_profile_fingerprint
    )
    if (
        checked_profile != bundle.benchmark_profile
        or checked_profile_fingerprint != bundle.dataset_profile_fingerprint
    ):
        raise TeachObsFrozenModelError(
            "prediction benchmark profile differs from frozen model"
        )
    sample_count = len(transcripts)
    if (
        len(ocr_texts) != sample_count
        or not all(isinstance(value, str) for value in transcripts)
        or not all(isinstance(value, str) for value in ocr_texts)
    ):
        raise TeachObsFrozenModelError("OCR/transcript prediction row count differs")
    checked_sample_ids: tuple[str, ...] | None = None
    if sample_ids is not None:
        checked_sample_ids = tuple(sample_ids)
        if (
            len(checked_sample_ids) != sample_count
            or len(set(checked_sample_ids)) != sample_count
            or any(not isinstance(value, str) or not value.strip() for value in checked_sample_ids)
        ):
            raise TeachObsFrozenModelError("invalid frozen prediction sample identity order")
    try:
        raw_numeric = {
            "audio_numeric": np.asarray(audio, dtype=np.float64),
            "visual_numeric": np.asarray(visual_numeric, dtype=np.float64),
            "clip_embedding": np.asarray(clip, dtype=np.float64),
        }
    except (TypeError, ValueError, OverflowError) as exc:
        raise TeachObsFrozenModelError("invalid frozen prediction numeric input") from exc
    full_arm = bundle.arms["full"]
    expected_names = {
        "audio_numeric": tuple(audio_feature_names),
        "visual_numeric": tuple(visual_numeric_feature_names),
    }
    for block, matrix in raw_numeric.items():
        dimension = full_arm.block_dimensions[block]
        if matrix.shape != (sample_count, dimension) or not np.isfinite(matrix).all():
            raise TeachObsFrozenModelError(f"invalid frozen prediction matrix: {block}")
        contract = full_arm.numeric_contracts[block]
        if block in expected_names and tuple(contract["feature_names"]) != expected_names[block]:
            raise TeachObsFrozenModelError(f"prediction feature order differs: {block}")
    clip_provenance = full_arm.numeric_contracts["clip_embedding"]["provenance"]
    audio_provenance = full_arm.numeric_contracts["audio_numeric"]["provenance"]
    visual_provenance = full_arm.numeric_contracts["visual_numeric"]["provenance"]
    if (
        clip_provenance.get("clip_source_revision") != clip_source_revision
        or clip_provenance.get("clip_weight_manifest_sha256")
        != clip_weight_manifest_sha256
    ):
        raise TeachObsFrozenModelError("prediction CLIP provenance differs from frozen model")
    if (
        audio_provenance.get("feature_schema_sha256")
        != _canonical_sha256(list(audio_feature_names))
        or audio_provenance.get("audio_feature_schema") != audio_feature_schema
    ):
        raise TeachObsFrozenModelError(
            "prediction audio feature provenance differs from frozen model"
        )
    if (
        visual_provenance.get("feature_schema_sha256")
        != _canonical_sha256(list(visual_numeric_feature_names))
        or visual_provenance.get("visual_evidence_schema")
        != visual_evidence_schema
        or visual_evidence_configuration_sha256
        not in visual_provenance.get("configuration_sha256_values", ())
    ):
        raise TeachObsFrozenModelError(
            "prediction visual feature provenance differs from frozen model"
        )

    input_binding = {
        "schema": "teaching_skill_miner.teachobs_frozen_prediction_input.v2",
        "bundle_fingerprint": bundle.bundle_fingerprint,
        "benchmark_profile": checked_profile,
        "dataset_profile_fingerprint": checked_profile_fingerprint,
        "sample_count": sample_count,
        "sample_order_sha256": (
            _canonical_sha256(list(checked_sample_ids))
            if checked_sample_ids is not None
            else None
        ),
        "sample_identity_bound": checked_sample_ids is not None,
        "transcript_order_sha256": _canonical_sha256(list(transcripts)),
        "transcript_input_materialization_fingerprint_sha256": (
            transcript_input_materialization_fingerprint_sha256
        ),
        "ocr_text_order_sha256": _canonical_sha256(list(ocr_texts)),
        "audio_matrix_sha256": _array_sha256(raw_numeric["audio_numeric"]),
        "visual_numeric_matrix_sha256": _array_sha256(
            raw_numeric["visual_numeric"]
        ),
        "clip_matrix_sha256": _array_sha256(raw_numeric["clip_embedding"]),
        "label_order_sha256": _canonical_sha256(list(label_names)),
        "audio_feature_order_sha256": _canonical_sha256(
            list(audio_feature_names)
        ),
        "visual_numeric_feature_order_sha256": _canonical_sha256(
            list(visual_numeric_feature_names)
        ),
        "clip_source_revision": clip_source_revision,
        "clip_weight_manifest_sha256": clip_weight_manifest_sha256,
        "audio_feature_schema": audio_feature_schema,
        "visual_evidence_schema": visual_evidence_schema,
        "visual_evidence_configuration_sha256": (
            visual_evidence_configuration_sha256
        ),
    }
    prediction_input_fingerprint = _canonical_sha256(input_binding)
    if expected_prediction_input_fingerprint is not None and (
        not _is_sha256(expected_prediction_input_fingerprint)
        or prediction_input_fingerprint != expected_prediction_input_fingerprint
    ):
        raise TeachObsFrozenModelError("frozen prediction input fingerprint differs")

    documents = {
        "transcript_tfidf": transcripts,
        "ocr_char_tfidf": ocr_texts,
        "ocr_word_tfidf": ocr_texts,
    }
    outputs: dict[str, dict[str, Any]] = {}
    for arm_name in _ARMS:
        arm = bundle.arms[arm_name]
        blocks: list[Any] = []
        for block in arm.ordered_blocks:
            if block in _TEXT_BLOCKS:
                contract = arm.text_contracts[block]
                blocks.append(
                    _transform_text(
                        contract,
                        arm.arrays[str(contract["idf_array"])],
                        documents[block],
                    )
                )
            else:
                contract = arm.numeric_contracts[block]
                mean = arm.arrays[str(contract["scaler_mean_array"])]
                scale = arm.arrays[str(contract["scaler_scale_array"])]
                blocks.append((raw_numeric[block] - mean) / scale)
        matrix = blocks[0] if len(blocks) == 1 else hstack(blocks, format="csr")
        if matrix.shape[1] != sum(arm.block_dimensions.values()):
            raise TeachObsFrozenModelError("frozen arm matrix dimension drift")
        coefficient = arm.arrays["model_coefficient"]
        intercept = arm.arrays["model_intercept"]
        scores = np.asarray(matrix @ coefficient.T, dtype=np.float64) + intercept
        probabilities = np.empty_like(scores)
        predictions = np.empty(scores.shape, dtype=np.uint8)
        thresholds = arm.arrays["model_thresholds"]
        for index, row in enumerate(arm.model_rows):
            kind = row["kind"]
            if kind == "constant_0":
                probabilities[:, index] = 0.0
                predictions[:, index] = 0
            elif kind == "constant_1":
                probabilities[:, index] = 1.0
                predictions[:, index] = 1
            else:
                column = scores[:, index]
                positive = np.empty(column.shape, dtype=np.float64)
                nonnegative = column >= 0
                positive[nonnegative] = 1.0 / (1.0 + np.exp(-column[nonnegative]))
                exponent = np.exp(column[~nonnegative])
                positive[~nonnegative] = exponent / (1.0 + exponent)
                probabilities[:, index] = positive
                predictions[:, index] = (
                    positive >= thresholds[index]
                ).astype(np.uint8)
        outputs[arm_name] = {
            "predictions": predictions,
            "probabilities": probabilities,
            "prediction_matrix_sha256": _array_sha256(predictions),
            "probability_matrix_sha256": _array_sha256(probabilities),
            "prediction_input_fingerprint": prediction_input_fingerprint,
            "sample_order_sha256": input_binding["sample_order_sha256"],
            "sample_identity_bound": input_binding["sample_identity_bound"],
        }
    return outputs
