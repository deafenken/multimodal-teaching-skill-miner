"""Four-arm multimodal benchmark on pinned TeachObs evaluation profiles.

This module joins the released TeachObs transcripts and consensus labels with
the private, hash-bound features produced by :mod:`teachobs_media`.  Every
profile keeps its selected test lessons out of every fitted transform and
classifier.  The default remains the complete official 23/7 split; the
paper-aligned Track 1 profile uses the six-lesson text/frame intersection from
arXiv:2605.30673v2, section 4.1, and excludes S4 before model fitting.

The released test labels and the first-pass test outcomes were observed before
this revised train-only OOF procedure was implemented, so the result remains
an iterative exploratory, provisional comparison.  It is not a confirmatory
lockbox result, deployment accuracy, or evidence of learning effectiveness.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

from .teachobs_benchmark import (
    TeachObsBenchmarkDependencyError,
    TeachObsBenchmarkError,
    TeachObsScene,
    TeachObsTextDataset,
    load_teachobs_text_dataset,
)
from .teachobs_media import (
    AUDIO_SCHEMA,
    FEATURE_FAILURE_SCHEMA,
    FEATURE_SCHEMA,
    FRAME_TASK_SCHEMA,
    VISUAL_EVIDENCE_EVENT_COUNTING_POLICY,
    VISUAL_EVIDENCE_EVENT_TYPES,
    VISUAL_EVIDENCE_SCHEMA,
)
from .visual_semantics import SCHEMA_RESULT
from .teachobs_transcript_materialization import (
    MANIFEST_SCHEMA as TRANSCRIPT_MATERIALIZATION_SCHEMA,
    TeachObsTranscriptMaterializationError,
    validate_teachobs_transcript_materialization,
)


BENCHMARK_KIND = "teachobs_official_held_out_four_arm_multimodal_benchmark"
FULL_23_TRAIN_7_TEST_PROFILE = "full_23_train_7_test"
PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE = "paper_track1_23_train_6_test"
DEFAULT_BENCHMARK_PROFILE = FULL_23_TRAIN_7_TEST_PROFILE
PAPER_TRACK1_TEST_LESSON_IDS = ("S2", "S5", "S19", "S24", "S28", "S30")
PAPER_TRACK1_EXPECTED_TEST_SCENE_COUNT = 1_099
_FULL_TEST_LESSON_IDS = ("S2", "S4", "S5", "S19", "S24", "S28", "S30")
_FULL_TRAIN_LESSON_IDS = tuple(
    f"S{index}"
    for index in range(1, 31)
    if f"S{index}" not in _FULL_TEST_LESSON_IDS
)
_FULL_EXPECTED_TRAIN_SCENE_COUNT = 3_846
_FULL_EXPECTED_TEST_SCENE_COUNT = 1_312
_PINNED_RELEASE_SCENE_COUNT = 5_158
_PROFILE_SCHEMA = "teaching_skill_miner.teachobs_multimodal_profile.v1"
_DECISION_THRESHOLD = 0.5
_TFIDF_MAX_FEATURES = 30_000
_TFIDF_NGRAM_RANGE = (3, 5)
_OCR_CHAR_MAX_FEATURES = 20_000
_OCR_CHAR_NGRAM_RANGE = (3, 5)
_OCR_WORD_MAX_FEATURES = 10_000
_OCR_WORD_NGRAM_RANGE = (1, 2)
_LOGISTIC_MAX_ITER = 2_000
_LOGISTIC_SOLVER = "liblinear"
_RANDOM_STATE = 0
_BOOTSTRAP_REPLICATES = 2_000
_BOOTSTRAP_SEED = 20_260_723
_ADVANCED_GROUP_FOLDS = 5
_ADVANCED_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
_ADVANCED_CLASSIFIER_CANDIDATES = (
    {
        "candidate_id": "balanced_c0.25",
        "regularization_c": 0.25,
        "class_weight": "balanced",
    },
    {
        "candidate_id": "balanced_c1",
        "regularization_c": 1.0,
        "class_weight": "balanced",
    },
)
_ADVANCED_THRESHOLD_PASSES = 3
_ADVANCED_SELECTION_SCHEMA = (
    "teaching_skill_miner.teachobs_train_lesson_grouped_oof_selection.v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_IDENTITY_AUDIT_SCHEMA = (
    "teaching_skill_miner.teachobs_source_identity_audit.v1"
)
_BENCHMARK_INPUT_SCHEMA = "teaching_skill_miner.teachobs_benchmark_input.v1"
_EXPECTED_TRAIN_SOURCE_VALUE_COUNT = 7
_EXPECTED_TEST_SOURCE_VALUE_COUNT = 6
_EXPECTED_OVERLAPPING_SOURCE_VALUE_COUNT = 3
_EXPECTED_SOURCE_VALUE_UNION_COUNT = 10
_AUDIO_FEATURE_NAMES = (
    "coverage_fraction",
    "rms_amplitude",
    "mean_absolute_amplitude",
    "peak_absolute_amplitude",
    "dc_offset",
    "zero_crossing_rate",
    "silence_fraction",
)
_IMAGE_METRIC_COMPONENTS = (
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_stddev_r",
    "rgb_stddev_g",
    "rgb_stddev_b",
    "luminance_mean",
    "edge_difference_mean",
    "dark_pixel_fraction",
    "bright_pixel_fraction",
)
_OCR_NUMERIC_COMPONENTS = (
    "ocr_raw_word_count",
    "ocr_accepted_word_count",
    "ocr_accepted_mean_confidence",
)
_OCR_STATUSES = ("completed", "unavailable", "failed", "disabled")
_TRANSITION_COMPONENTS = (
    "previous_dhash_hamming_distance",
    "previous_dhash_distance_fraction",
    "previous_edge_difference_delta",
)
_TRANSITION_FLAGS = (
    "scene_change",
    "slide_change",
    "board_build_up",
)
_VISUAL_NUMERIC_FEATURE_NAMES = (
    *tuple(
        name
        for component in _IMAGE_METRIC_COMPONENTS
        for name in (component, f"{component}_missing")
    ),
    *tuple(
        name
        for component in _OCR_NUMERIC_COMPONENTS
        for name in (component, f"{component}_missing")
    ),
    *(f"ocr_status_{status}" for status in _OCR_STATUSES),
    "ocr_text_nonempty",
    *tuple(
        name
        for component in _TRANSITION_COMPONENTS
        for name in (component, f"{component}_missing")
    ),
    *(f"transition_{event_type}" for event_type in _TRANSITION_FLAGS),
)
_ARMS = (
    "transcript_only",
    "transcript_audio",
    "transcript_visual",
    "full",
)
_METRICS = (
    "micro_f1",
    "macro_f1",
    "subset_accuracy",
    "hamming_accuracy",
    "visual_macro_f1",
    "nonvisual_macro_f1",
)
_COMPARISONS = (
    ("transcript_audio_minus_transcript_only", "transcript_audio", "transcript_only"),
    (
        "transcript_visual_minus_transcript_only",
        "transcript_visual",
        "transcript_only",
    ),
    ("full_minus_transcript_only", "full", "transcript_only"),
    ("full_minus_transcript_audio", "full", "transcript_audio"),
    ("full_minus_transcript_visual", "full", "transcript_visual"),
)


class TeachObsMultimodalBenchmarkError(TeachObsBenchmarkError):
    """Raised when private features or the four-arm protocol are invalid."""


@dataclass(frozen=True, slots=True)
class TeachObsBenchmarkProfile:
    """Pinned lesson-level protocol selected before any fitted operation."""

    profile_id: str
    test_lesson_ids: tuple[str, ...]
    excluded_official_test_lesson_ids: tuple[str, ...]
    expected_test_scene_count: int
    source_alignment: str
    published_six_lesson_intersection: bool


@dataclass(frozen=True, slots=True)
class ResolvedTeachObsBenchmarkProfile:
    """One profile resolved against an audited full TeachObs release."""

    spec: TeachObsBenchmarkProfile
    dataset: TeachObsTextDataset
    selected_train_lesson_ids: tuple[str, ...]
    selected_test_lesson_ids: tuple[str, ...]
    selected_train_sample_ids: tuple[str, ...]
    selected_test_sample_ids: tuple[str, ...]
    dataset_profile_fingerprint: str


TEACHOBS_BENCHMARK_PROFILES: dict[str, TeachObsBenchmarkProfile] = {
    FULL_23_TRAIN_7_TEST_PROFILE: TeachObsBenchmarkProfile(
        profile_id=FULL_23_TRAIN_7_TEST_PROFILE,
        test_lesson_ids=_FULL_TEST_LESSON_IDS,
        excluded_official_test_lesson_ids=(),
        expected_test_scene_count=_FULL_EXPECTED_TEST_SCENE_COUNT,
        source_alignment="official_released_23_train_7_test_split",
        published_six_lesson_intersection=False,
    ),
    PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE: TeachObsBenchmarkProfile(
        profile_id=PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
        test_lesson_ids=PAPER_TRACK1_TEST_LESSON_IDS,
        excluded_official_test_lesson_ids=("S4",),
        expected_test_scene_count=PAPER_TRACK1_EXPECTED_TEST_SCENE_COUNT,
        source_alignment=(
            "arxiv_2605.30673v2_section_4.1_track1_text_frame_intersection"
        ),
        published_six_lesson_intersection=True,
    ),
}


@dataclass(frozen=True, slots=True)
class TeachObsMultimodalFeatures:
    """Validated matrices ordered exactly like the audited text dataset."""

    train_audio: Any
    test_audio: Any
    train_visual_numeric: Any
    test_visual_numeric: Any
    train_clip: Any
    test_clip: Any
    train_ocr_text: tuple[str, ...]
    test_ocr_text: tuple[str, ...]
    feature_manifest_sha256: str
    audio_feature_names: tuple[str, ...]
    visual_numeric_feature_names: tuple[str, ...]
    visual_dimension: int
    visual_numeric_dimension: int
    clip_dimension: int
    clip_source_revision: str
    clip_weight_manifest_sha256: str
    visual_feature_layout: str
    visual_evidence_set_sha256: str
    visual_evidence_configuration_set_sha256: str
    visual_evidence_configuration_sha256_values: tuple[str, ...]
    ocr_status_counts: dict[str, int]
    ocr_nonempty_scene_count: int
    ocr_completed_scene_count: int
    source_file_count: int
    benchmark_profile: str
    dataset_profile_fingerprint: str
    selected_train_lesson_ids: tuple[str, ...]
    selected_test_lesson_ids: tuple[str, ...]
    selected_train_sample_ids: tuple[str, ...]
    selected_test_sample_ids: tuple[str, ...]

    @property
    def train_visual(self) -> Any:
        """Compatibility view containing numeric evidence followed by CLIP."""

        dependencies = _require_dependencies()
        return dependencies["np"].concatenate(
            (self.train_visual_numeric, self.train_clip), axis=1
        )

    @property
    def test_visual(self) -> Any:
        """Compatibility view containing numeric evidence followed by CLIP."""

        dependencies = _require_dependencies()
        return dependencies["np"].concatenate(
            (self.test_visual_numeric, self.test_clip), axis=1
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ordered_lesson_ids(scenes: Sequence[Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(scene.lesson_id for scene in scenes))


def _sample_ids(scenes: Sequence[Any]) -> tuple[str, ...]:
    return tuple(f"{scene.lesson_id}:{scene.scene_no}" for scene in scenes)


def _dataset_profile_contract(
    *,
    spec: TeachObsBenchmarkProfile,
    repository_dataset_sha256: str,
    train_lesson_ids: Sequence[str],
    test_lesson_ids: Sequence[str],
    train_sample_ids: Sequence[str],
    test_sample_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": _PROFILE_SCHEMA,
        "profile_id": spec.profile_id,
        "source_alignment": spec.source_alignment,
        "published_six_lesson_intersection": (
            spec.published_six_lesson_intersection
        ),
        "repository_dataset_sha256": repository_dataset_sha256,
        "selected_train_lesson_ids": list(train_lesson_ids),
        "selected_test_lesson_ids": list(test_lesson_ids),
        "excluded_official_test_lesson_ids": list(
            spec.excluded_official_test_lesson_ids
        ),
        "selected_train_sample_ids": list(train_sample_ids),
        "selected_test_sample_ids": list(test_sample_ids),
        "selected_train_scene_count": len(train_sample_ids),
        "selected_test_scene_count": len(test_sample_ids),
    }


def resolve_teachobs_benchmark_profile(
    full_dataset: TeachObsTextDataset,
    profile: str = DEFAULT_BENCHMARK_PROFILE,
) -> ResolvedTeachObsBenchmarkProfile:
    """Select a pinned profile before any transform, model, or metric is run."""

    try:
        spec = TEACHOBS_BENCHMARK_PROFILES[str(profile)]
    except KeyError as exc:
        raise TeachObsMultimodalBenchmarkError(
            f"unknown TeachObs multimodal benchmark profile: {profile}"
        ) from exc
    full_train_ids = _ordered_lesson_ids(full_dataset.train_scenes)
    full_test_ids = _ordered_lesson_ids(full_dataset.test_scenes)
    if (
        full_train_ids != _FULL_TRAIN_LESSON_IDS
        or full_test_ids != _FULL_TEST_LESSON_IDS
        or set(full_train_ids) & set(full_test_ids)
    ):
        raise TeachObsMultimodalBenchmarkError(
            "TeachObs profile resolution requires the audited official 23/7 split"
        )
    selected_test = tuple(
        scene
        for scene in full_dataset.test_scenes
        if scene.lesson_id in spec.test_lesson_ids
    )
    selected_test_ids = _ordered_lesson_ids(selected_test)
    if (
        selected_test_ids != spec.test_lesson_ids
        or any(
            scene.lesson_id in spec.excluded_official_test_lesson_ids
            for scene in selected_test
        )
    ):
        raise TeachObsMultimodalBenchmarkError(
            "TeachObs test scenes differ from the pinned benchmark profile"
        )
    release_sized = (
        len(full_dataset.train_scenes) + len(full_dataset.test_scenes)
        == _PINNED_RELEASE_SCENE_COUNT
    )
    if release_sized and (
        len(full_dataset.train_scenes) != _FULL_EXPECTED_TRAIN_SCENE_COUNT
        or len(full_dataset.test_scenes) != _FULL_EXPECTED_TEST_SCENE_COUNT
        or len(selected_test) != spec.expected_test_scene_count
    ):
        raise TeachObsMultimodalBenchmarkError(
            "TeachObs scene counts differ from the pinned benchmark profile"
        )
    train_sample_ids = _sample_ids(full_dataset.train_scenes)
    test_sample_ids = _sample_ids(selected_test)
    contract = _dataset_profile_contract(
        spec=spec,
        repository_dataset_sha256=full_dataset.dataset_sha256,
        train_lesson_ids=full_train_ids,
        test_lesson_ids=selected_test_ids,
        train_sample_ids=train_sample_ids,
        test_sample_ids=test_sample_ids,
    )
    selected_dataset = TeachObsTextDataset(
        code_names=full_dataset.code_names,
        code_groups=full_dataset.code_groups,
        train_scenes=full_dataset.train_scenes,
        test_scenes=selected_test,
        dataset_sha256=full_dataset.dataset_sha256,
        source_file_count=full_dataset.source_file_count,
    )
    return ResolvedTeachObsBenchmarkProfile(
        spec=spec,
        dataset=selected_dataset,
        selected_train_lesson_ids=full_train_ids,
        selected_test_lesson_ids=selected_test_ids,
        selected_train_sample_ids=train_sample_ids,
        selected_test_sample_ids=test_sample_ids,
        dataset_profile_fingerprint=_canonical_sha256(contract),
    )


def frozen_profile_training_binding(
    resolved: ResolvedTeachObsBenchmarkProfile,
) -> dict[str, Any]:
    """Return the private, transitive profile fields embedded in frozen models."""

    contract = _dataset_profile_contract(
        spec=resolved.spec,
        repository_dataset_sha256=resolved.dataset.dataset_sha256,
        train_lesson_ids=resolved.selected_train_lesson_ids,
        test_lesson_ids=resolved.selected_test_lesson_ids,
        train_sample_ids=resolved.selected_train_sample_ids,
        test_sample_ids=resolved.selected_test_sample_ids,
    )
    return {
        "benchmark_profile": resolved.spec.profile_id,
        "profile_source_alignment": resolved.spec.source_alignment,
        "published_six_lesson_intersection": (
            resolved.spec.published_six_lesson_intersection
        ),
        "excluded_official_test_lesson_ids": list(
            resolved.spec.excluded_official_test_lesson_ids
        ),
        "selected_train_lesson_ids": list(resolved.selected_train_lesson_ids),
        "selected_test_lesson_ids": list(resolved.selected_test_lesson_ids),
        "selected_train_sample_ids": list(resolved.selected_train_sample_ids),
        "selected_test_sample_ids": list(resolved.selected_test_sample_ids),
        "selected_train_sample_order_sha256": _canonical_sha256(
            list(resolved.selected_train_sample_ids)
        ),
        "selected_test_sample_order_sha256": _canonical_sha256(
            list(resolved.selected_test_sample_ids)
        ),
        "dataset_profile_fingerprint": _canonical_sha256(contract),
    }


_FROZEN_TRANSCRIPT_MATERIALIZATION_FIELDS = {
    "schema",
    "profile_id",
    "manifest_file_sha256",
    "manifest_sha256",
    "materialization_fingerprint_sha256",
    "ordered_sample_id_sha256",
    "ordered_scene_text_sha256",
    "repository_binding_sha256",
    "input_hashes_sha256",
    "selected_train_transcript_order_sha256",
    "selected_test_transcript_order_sha256",
    "released_transcript_fallback_used",
    "labels_read_or_used",
}


def _benchmark_input_contract(
    *,
    dataset_profile_fingerprint: str,
    feature_manifest_sha256: str,
    transcript_materialization: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": _BENCHMARK_INPUT_SCHEMA,
        "dataset_profile_fingerprint": dataset_profile_fingerprint,
        "feature_manifest_sha256": feature_manifest_sha256,
        "transcript_materialization_manifest_file_sha256": (
            transcript_materialization["manifest_file_sha256"]
        ),
        "transcript_materialization_manifest_sha256": (
            transcript_materialization["manifest_sha256"]
        ),
        "transcript_materialization_fingerprint_sha256": (
            transcript_materialization["materialization_fingerprint_sha256"]
        ),
        "transcript_ordered_sample_id_sha256": transcript_materialization[
            "ordered_sample_id_sha256"
        ],
        "transcript_ordered_scene_text_sha256": transcript_materialization[
            "ordered_scene_text_sha256"
        ],
    }


def _validate_frozen_transcript_materialization_binding(
    training: Mapping[str, Any],
) -> None:
    raw = training.get("transcript_materialization")
    if not isinstance(raw, Mapping) or set(raw) != (
        _FROZEN_TRANSCRIPT_MATERIALIZATION_FIELDS
    ):
        raise TeachObsMultimodalBenchmarkError(
            "frozen transcript materialization binding is malformed"
        )
    digest_fields = {
        "manifest_file_sha256",
        "manifest_sha256",
        "materialization_fingerprint_sha256",
        "ordered_sample_id_sha256",
        "ordered_scene_text_sha256",
        "repository_binding_sha256",
        "input_hashes_sha256",
        "selected_train_transcript_order_sha256",
        "selected_test_transcript_order_sha256",
    }
    if (
        raw.get("schema") != TRANSCRIPT_MATERIALIZATION_SCHEMA
        or raw.get("profile_id") != training.get("benchmark_profile")
        or any(
            not isinstance(raw.get(field), str)
            or _SHA256.fullmatch(str(raw[field])) is None
            for field in digest_fields
        )
        or raw.get("released_transcript_fallback_used") is not False
        or raw.get("labels_read_or_used") is not False
    ):
        raise TeachObsMultimodalBenchmarkError(
            "frozen transcript materialization provenance is invalid"
        )
    expected_input_fingerprint = _canonical_sha256(
        _benchmark_input_contract(
            dataset_profile_fingerprint=str(
                training.get("dataset_profile_fingerprint", "")
            ),
            feature_manifest_sha256=str(
                training.get("feature_manifest_sha256", "")
            ),
            transcript_materialization=raw,
        )
    )
    if training.get("benchmark_input_fingerprint") != expected_input_fingerprint:
        raise TeachObsMultimodalBenchmarkError(
            "frozen benchmark input fingerprint differs"
        )


def _load_source_identity_audit(
    repository_root: str | Path,
    profile_resolution: ResolvedTeachObsBenchmarkProfile,
) -> dict[str, Any]:
    """Audit provenance values without treating the ``source`` field as a site ID."""

    repository = Path(repository_root).expanduser().resolve()
    metadata_path = repository / "data" / "lessons.csv"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise TeachObsMultimodalBenchmarkError(
            "TeachObs lesson metadata is missing or unsafe"
        )
    try:
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = tuple(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise TeachObsMultimodalBenchmarkError(
            "invalid TeachObs lesson metadata CSV"
        ) from exc
    if (
        len(fieldnames) != len(set(fieldnames))
        or not {"id", "split", "source"}.issubset(fieldnames)
        or any(
            field in fieldnames
            for field in ("site_id", "teacher_id", "classroom_id")
        )
    ):
        raise TeachObsMultimodalBenchmarkError(
            "TeachObs lesson metadata lacks the pinned source-identity contract"
        )
    expected_ids = {f"S{index}" for index in range(1, 31)}
    sources_by_lesson: dict[str, str] = {}
    declared_split: dict[str, str] = {}
    for row in rows:
        lesson_id = str(row.get("id", "")).strip()
        split = str(row.get("split", "")).strip()
        source = str(row.get("source", "")).strip()
        if (
            lesson_id not in expected_ids
            or lesson_id in sources_by_lesson
            or split not in {"train", "test"}
            or not source
            or "\x00" in source
            or len(source) > 512
        ):
            raise TeachObsMultimodalBenchmarkError(
                "TeachObs lesson source metadata is malformed"
            )
        sources_by_lesson[lesson_id] = source
        declared_split[lesson_id] = split
    if set(sources_by_lesson) != expected_ids:
        raise TeachObsMultimodalBenchmarkError(
            "TeachObs source metadata does not cover exactly S1-S30"
        )
    if any(
        declared_split[lesson_id]
        != ("test" if lesson_id in _FULL_TEST_LESSON_IDS else "train")
        for lesson_id in expected_ids
    ):
        raise TeachObsMultimodalBenchmarkError(
            "TeachObs source metadata differs from the official lesson split"
        )
    train_sources = {
        sources_by_lesson[lesson_id]
        for lesson_id in profile_resolution.selected_train_lesson_ids
    }
    test_sources = {
        sources_by_lesson[lesson_id]
        for lesson_id in profile_resolution.selected_test_lesson_ids
    }
    overlap = train_sources & test_sources
    union = train_sources | test_sources
    if (
        len(train_sources) != _EXPECTED_TRAIN_SOURCE_VALUE_COUNT
        or len(test_sources) != _EXPECTED_TEST_SOURCE_VALUE_COUNT
        or len(overlap) != _EXPECTED_OVERLAPPING_SOURCE_VALUE_COUNT
        or len(union) != _EXPECTED_SOURCE_VALUE_UNION_COUNT
    ):
        raise TeachObsMultimodalBenchmarkError(
            "TeachObs source-value train/test overlap differs from the pinned release"
        )
    return {
        "schema": _SOURCE_IDENTITY_AUDIT_SCHEMA,
        "metadata_file_sha256": _file_sha256(metadata_path),
        "metadata_lesson_count": len(rows),
        "source_metadata_field": "source",
        "source_field_present_for_every_lesson": True,
        "selected_train_lesson_count": len(
            profile_resolution.selected_train_lesson_ids
        ),
        "selected_test_lesson_count": len(
            profile_resolution.selected_test_lesson_ids
        ),
        "selected_train_source_value_count": len(train_sources),
        "selected_test_source_value_count": len(test_sources),
        "overlapping_source_value_count": len(overlap),
        "source_value_union_count": len(union),
        "train_source_value_set_sha256": _canonical_sha256(
            sorted(train_sources)
        ),
        "test_source_value_set_sha256": _canonical_sha256(sorted(test_sources)),
        "overlapping_source_value_set_sha256": _canonical_sha256(
            sorted(overlap)
        ),
        "train_test_source_value_overlap_detected": bool(overlap),
        "source_value_disjoint_split": not overlap,
        "canonical_site_id_field_present": False,
        "canonical_teacher_id_field_present": False,
        "canonical_classroom_id_field_present": False,
        "source_field_used_as_canonical_site_id": False,
        "source_field_semantics": (
            "upstream_provenance_or_collection_value_not_canonical_site_id"
        ),
        "site_disjointness_verified": False,
        "teacher_disjointness_verified": False,
        "classroom_disjointness_verified": False,
        "source_family_sensitivity_metric_computed": False,
        "site_held_out_evaluation_performed": False,
        "site_held_out_accuracy_established": False,
    }


def _validate_source_identity_audit(
    value: Any, *, selected_test_lesson_count: int
) -> dict[str, Any]:
    """Validate the path-free source audit before it enters a public receipt."""

    if not isinstance(value, dict):
        raise TeachObsMultimodalBenchmarkError(
            "TeachObs source identity audit is missing"
        )
    digest_fields = (
        "metadata_file_sha256",
        "train_source_value_set_sha256",
        "test_source_value_set_sha256",
        "overlapping_source_value_set_sha256",
    )
    fixed = {
        "schema": _SOURCE_IDENTITY_AUDIT_SCHEMA,
        "metadata_lesson_count": 30,
        "source_metadata_field": "source",
        "source_field_present_for_every_lesson": True,
        "selected_train_lesson_count": 23,
        "selected_test_lesson_count": selected_test_lesson_count,
        "selected_train_source_value_count": _EXPECTED_TRAIN_SOURCE_VALUE_COUNT,
        "selected_test_source_value_count": _EXPECTED_TEST_SOURCE_VALUE_COUNT,
        "overlapping_source_value_count": _EXPECTED_OVERLAPPING_SOURCE_VALUE_COUNT,
        "source_value_union_count": _EXPECTED_SOURCE_VALUE_UNION_COUNT,
        "train_test_source_value_overlap_detected": True,
        "source_value_disjoint_split": False,
        "canonical_site_id_field_present": False,
        "canonical_teacher_id_field_present": False,
        "canonical_classroom_id_field_present": False,
        "source_field_used_as_canonical_site_id": False,
        "source_field_semantics": (
            "upstream_provenance_or_collection_value_not_canonical_site_id"
        ),
        "site_disjointness_verified": False,
        "teacher_disjointness_verified": False,
        "classroom_disjointness_verified": False,
        "source_family_sensitivity_metric_computed": False,
        "site_held_out_evaluation_performed": False,
        "site_held_out_accuracy_established": False,
    }
    if any(value.get(key) != expected for key, expected in fixed.items()) or any(
        not isinstance(value.get(field), str)
        or _SHA256.fullmatch(str(value[field])) is None
        for field in digest_fields
    ):
        raise TeachObsMultimodalBenchmarkError(
            "TeachObs source identity audit weakens the pinned claim boundary"
        )
    return dict(value)


def validate_frozen_profile_training_binding(training: Mapping[str, Any]) -> None:
    """Fail closed when a frozen profile/list/fingerprint binding is inconsistent."""

    profile = training.get("benchmark_profile")
    try:
        spec = TEACHOBS_BENCHMARK_PROFILES[str(profile)]
    except KeyError as exc:
        raise TeachObsMultimodalBenchmarkError(
            "frozen model references an unknown TeachObs benchmark profile"
        ) from exc
    train_lessons = training.get("selected_train_lesson_ids")
    test_lessons = training.get("selected_test_lesson_ids")
    train_samples = training.get("selected_train_sample_ids")
    test_samples = training.get("selected_test_sample_ids")
    excluded = training.get("excluded_official_test_lesson_ids")
    if (
        not isinstance(train_lessons, list)
        or not isinstance(test_lessons, list)
        or not isinstance(train_samples, list)
        or not isinstance(test_samples, list)
        or not isinstance(excluded, list)
        or tuple(train_lessons) != _FULL_TRAIN_LESSON_IDS
        or tuple(test_lessons) != spec.test_lesson_ids
        or tuple(excluded) != spec.excluded_official_test_lesson_ids
        or training.get("profile_source_alignment") != spec.source_alignment
        or training.get("published_six_lesson_intersection")
        is not spec.published_six_lesson_intersection
    ):
        raise TeachObsMultimodalBenchmarkError(
            "frozen TeachObs profile lesson binding is malformed"
        )
    all_ids = {f"S{index}" for index in range(1, 31)}
    if (
        set(train_lessons) & set(_FULL_TEST_LESSON_IDS)
        or (set(train_lessons) | set(test_lessons) | set(excluded)) != all_ids
    ):
        raise TeachObsMultimodalBenchmarkError(
            "frozen TeachObs profile does not reconstruct S1-S30"
        )

    def validate_samples(samples: list[Any], lessons: list[Any]) -> None:
        observed: dict[str, list[int]] = {}
        for sample_id in samples:
            if not isinstance(sample_id, str):
                raise TeachObsMultimodalBenchmarkError(
                    "frozen TeachObs sample identity is malformed"
                )
            match = re.fullmatch(r"(S(?:[1-9]|[12][0-9]|30)):(\d+)", sample_id)
            if match is None:
                raise TeachObsMultimodalBenchmarkError(
                    "frozen TeachObs sample identity is malformed"
                )
            observed.setdefault(match.group(1), []).append(int(match.group(2)))
        if tuple(observed) != tuple(lessons) or any(
            values != list(range(1, len(values) + 1))
            for values in observed.values()
        ):
            raise TeachObsMultimodalBenchmarkError(
                "frozen TeachObs sample order differs from its lesson profile"
            )

    validate_samples(train_samples, train_lessons)
    validate_samples(test_samples, test_lessons)
    if (
        training.get("selected_train_sample_order_sha256")
        != _canonical_sha256(train_samples)
        or training.get("selected_test_sample_order_sha256")
        != _canonical_sha256(test_samples)
    ):
        raise TeachObsMultimodalBenchmarkError(
            "frozen TeachObs sample-order hash differs"
        )
    repository_digest = training.get("training_repository_dataset_sha256")
    if not isinstance(repository_digest, str) or _SHA256.fullmatch(
        repository_digest
    ) is None:
        raise TeachObsMultimodalBenchmarkError(
            "frozen TeachObs repository fingerprint is malformed"
        )
    contract = _dataset_profile_contract(
        spec=spec,
        repository_dataset_sha256=repository_digest,
        train_lesson_ids=train_lessons,
        test_lesson_ids=test_lessons,
        train_sample_ids=train_samples,
        test_sample_ids=test_samples,
    )
    if training.get("dataset_profile_fingerprint") != _canonical_sha256(contract):
        raise TeachObsMultimodalBenchmarkError(
            "frozen TeachObs dataset/profile fingerprint differs"
        )
    _validate_frozen_transcript_materialization_binding(training)


def _array_sha256(value: Any) -> str:
    dependencies = _require_dependencies()
    np = dependencies["np"]
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _read_json_object(path: Path, *, purpose: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TeachObsMultimodalBenchmarkError(
            f"invalid {purpose} JSON"
        ) from exc
    if not isinstance(value, dict):
        raise TeachObsMultimodalBenchmarkError(f"{purpose} must be a JSON object")
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    digest = str(value).strip().lower()
    if not _SHA256.fullmatch(digest):
        raise TeachObsMultimodalBenchmarkError(f"invalid {field} SHA-256")
    return digest


def _bound_feature_path(
    root: Path,
    relative_value: Any,
    expected_sha256: Any,
    *,
    purpose: str,
) -> Path:
    if not isinstance(relative_value, str) or not relative_value.strip():
        raise TeachObsMultimodalBenchmarkError(f"missing private {purpose} path")
    text = relative_value.strip()
    relative = PurePosixPath(text)
    if (
        relative.is_absolute()
        or "\\" in text
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise TeachObsMultimodalBenchmarkError(f"unsafe private {purpose} path")
    path = root.joinpath(*relative.parts)
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise TeachObsMultimodalBenchmarkError(
            f"private {purpose} file is missing"
        ) from exc
    if path.is_symlink() or not resolved.is_relative_to(root) or not resolved.is_file():
        raise TeachObsMultimodalBenchmarkError(f"unsafe private {purpose} file")
    expected = _require_sha256(expected_sha256, field=purpose)
    if _file_sha256(resolved) != expected:
        raise TeachObsMultimodalBenchmarkError(f"private {purpose} hash mismatch")
    return resolved


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TeachObsMultimodalBenchmarkError(f"non-numeric {field}")
    result = float(value)
    if not math.isfinite(result):
        raise TeachObsMultimodalBenchmarkError(f"non-finite {field}")
    return result


def _official_manifest_sha256(repository_root: Path, lesson_id: str) -> str:
    path = repository_root / "data" / "scenes" / lesson_id / "manifest.jsonl"
    if path.is_symlink() or not path.is_file():
        raise TeachObsMultimodalBenchmarkError(
            "official TeachObs scene manifest is missing"
        )
    return _file_sha256(path)


def _load_audio_rows(
    path: Path,
    *,
    lesson_id: str,
    scene_count: int,
    media_sha256: str,
    scene_manifest_sha256: str,
) -> list[list[float]]:
    value = _read_json_object(path, purpose="TeachObs audio features")
    if (
        value.get("schema") != AUDIO_SCHEMA
        or value.get("lesson_id") != lesson_id
        or value.get("media_sha256") != media_sha256
        or value.get("scene_manifest_sha256") != scene_manifest_sha256
        or value.get("scene_count") != scene_count
        or tuple(value.get("feature_names", ())) != _AUDIO_FEATURE_NAMES
    ):
        raise TeachObsMultimodalBenchmarkError(
            f"audio feature binding mismatch for {lesson_id}"
        )
    rows = value.get("scenes")
    if not isinstance(rows, list) or len(rows) != scene_count:
        raise TeachObsMultimodalBenchmarkError(
            f"audio scene count mismatch for {lesson_id}"
        )
    result: list[list[float]] = []
    for expected_scene_no, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or row.get("scene_no") != expected_scene_no:
            raise TeachObsMultimodalBenchmarkError(
                f"audio scene alignment mismatch for {lesson_id}"
            )
        result.append(
            [
                _finite_number(row.get(name), field=f"audio {name}")
                for name in _AUDIO_FEATURE_NAMES
            ]
        )
    return result


def _load_visual_rows(
    path: Path,
    *,
    lesson_id: str,
    scene_count: int,
    media_sha256: str,
    frame_task_sha256: str,
    task_frames: Sequence[Mapping[str, Any]],
) -> tuple[list[list[float]], str, str]:
    value = _read_json_object(path, purpose="TeachObs CLIP features")
    source_revision = str(value.get("teachobs_source_revision", "")).strip()
    provenance = value.get("model_provenance")
    weight_manifest = provenance.get("weight_manifest") if isinstance(provenance, dict) else None
    weight_digest = (
        weight_manifest.get("manifest_sha256")
        if isinstance(weight_manifest, dict)
        else None
    )
    if (
        value.get("schema") != SCHEMA_RESULT
        or value.get("video_id") != lesson_id
        or value.get("media_sha256") != media_sha256
        or value.get("teachobs_task_file_sha256") != frame_task_sha256
        or value.get("frame_count") != scene_count
        or not source_revision
    ):
        raise TeachObsMultimodalBenchmarkError(
            f"visual feature binding mismatch for {lesson_id}"
        )
    weight_sha256 = _require_sha256(weight_digest, field="CLIP weight manifest")
    frames = value.get("frames")
    if not isinstance(frames, list) or len(frames) != scene_count:
        raise TeachObsMultimodalBenchmarkError(
            f"visual scene count mismatch for {lesson_id}"
        )
    result: list[list[float]] = []
    dimension: int | None = None
    for row, task_frame in zip(frames, task_frames, strict=True):
        if (
            not isinstance(row, dict)
            or row.get("frame_id") != task_frame["frame_id"]
            or not math.isclose(
                _finite_number(row.get("timestamp"), field="visual timestamp"),
                float(task_frame["timestamp"]),
                abs_tol=1e-6,
            )
            or row.get("path") != task_frame["path"]
            or row.get("sha256") != task_frame["sha256"]
        ):
            raise TeachObsMultimodalBenchmarkError(
                f"visual scene alignment mismatch for {lesson_id}"
            )
        embedding = row.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise TeachObsMultimodalBenchmarkError(
                f"visual embedding is missing for {lesson_id}"
            )
        vector = [
            _finite_number(item, field="visual embedding") for item in embedding
        ]
        if dimension is None:
            dimension = len(vector)
            if dimension < 2 or dimension > 4096:
                raise TeachObsMultimodalBenchmarkError(
                    "unsupported CLIP embedding dimension"
                )
        elif len(vector) != dimension:
            raise TeachObsMultimodalBenchmarkError(
                f"visual embedding dimension mismatch for {lesson_id}"
            )
        if row.get("embedding_sha256") != _canonical_sha256(vector):
            raise TeachObsMultimodalBenchmarkError(
                f"visual embedding hash mismatch for {lesson_id}"
            )
        result.append(vector)
    return result, source_revision, weight_sha256


def _load_bound_frame_task(
    path: Path,
    *,
    feature_root: Path,
    lesson_id: str,
    scene_count: int,
    media_sha256: str,
    scene_manifest_sha256: str,
    source_files: set[Path],
) -> list[dict[str, Any]]:
    """Validate one per-lesson task, including every referenced private frame."""

    task = _read_json_object(path, purpose="TeachObs per-lesson frame task")
    frames = task.get("frames")
    if (
        task.get("schema") != FRAME_TASK_SCHEMA
        or task.get("video_id") != lesson_id
        or task.get("media_sha256") != media_sha256
        or task.get("scene_manifest_sha256") != scene_manifest_sha256
        or task.get("frame_count") != scene_count
        or not isinstance(frames, list)
        or len(frames) != scene_count
    ):
        raise TeachObsMultimodalBenchmarkError(
            f"per-lesson frame-task binding mismatch for {lesson_id}"
        )
    task_clamp_fields = {
        "media_duration_seconds",
        "tail_clamp_margin_seconds",
        "maximum_tail_clamp_delta_seconds",
    }
    has_any_task_clamp_field = bool(task_clamp_fields & set(task))
    if has_any_task_clamp_field and not task_clamp_fields <= set(task):
        raise TeachObsMultimodalBenchmarkError(
            f"incomplete frame-task clamp metadata for {lesson_id}"
        )
    media_duration: float | None = None
    clamp_margin: float | None = None
    maximum_clamp_delta: float | None = None
    if has_any_task_clamp_field:
        media_duration = _finite_number(
            task["media_duration_seconds"], field="frame-task media duration"
        )
        clamp_margin = _finite_number(
            task["tail_clamp_margin_seconds"], field="frame-task clamp margin"
        )
        maximum_clamp_delta = _finite_number(
            task["maximum_tail_clamp_delta_seconds"],
            field="frame-task maximum clamp delta",
        )
        if (
            media_duration <= 0
            or not math.isclose(clamp_margin, 0.25, abs_tol=1e-9)
            or not math.isclose(maximum_clamp_delta, 15.0, abs_tol=1e-9)
        ):
            raise TeachObsMultimodalBenchmarkError(
                f"invalid frame-task clamp policy for {lesson_id}"
            )

    frame_clamp_fields = {
        "requested_timestamp",
        "timestamp_clamped",
        "timestamp_clamp_delta_seconds",
    }
    result: list[dict[str, Any]] = []
    clamped_frame_count = 0
    task_directory = path.parent.resolve()
    for expected_scene_no, frame in enumerate(frames, start=1):
        expected_id = f"{lesson_id}_scene_{expected_scene_no:04d}"
        expected_requested_timestamp = (expected_scene_no - 0.5) * 15.0
        if not isinstance(frame, dict):
            raise TeachObsMultimodalBenchmarkError(
                f"invalid frame-task row for {lesson_id}"
            )
        has_any_frame_clamp_field = bool(frame_clamp_fields & set(frame))
        if has_any_frame_clamp_field != has_any_task_clamp_field or (
            has_any_frame_clamp_field and not frame_clamp_fields <= set(frame)
        ):
            raise TeachObsMultimodalBenchmarkError(
                f"incomplete frame clamp metadata for {lesson_id}"
            )
        actual_timestamp = _finite_number(
            frame.get("timestamp"), field="frame timestamp"
        )
        if has_any_frame_clamp_field:
            requested_timestamp = _finite_number(
                frame["requested_timestamp"], field="requested frame timestamp"
            )
            clamped = frame["timestamp_clamped"]
            clamp_delta = _finite_number(
                frame["timestamp_clamp_delta_seconds"],
                field="frame timestamp clamp delta",
            )
            if (
                not isinstance(clamped, bool)
                or not math.isclose(
                    requested_timestamp, expected_requested_timestamp, abs_tol=1e-6
                )
                or actual_timestamp > requested_timestamp + 1e-6
                or not math.isclose(
                    requested_timestamp - actual_timestamp,
                    clamp_delta,
                    abs_tol=1e-6,
                )
            ):
                raise TeachObsMultimodalBenchmarkError(
                    f"frame-task clamp binding mismatch for {lesson_id}"
                )
            assert media_duration is not None
            assert clamp_margin is not None
            assert maximum_clamp_delta is not None
            expected_actual = max(0.0, media_duration - clamp_margin)
            should_clamp = requested_timestamp > expected_actual + 1e-6
            if clamped:
                clamped_frame_count += 1
                scene_start = (expected_scene_no - 1) * 15.0
                scene_end = expected_scene_no * 15.0
                if (
                    expected_scene_no != scene_count
                    or not should_clamp
                    or clamp_delta <= 0
                    or clamp_delta > maximum_clamp_delta + 1e-6
                    or not math.isclose(
                        actual_timestamp, expected_actual, abs_tol=1e-6
                    )
                    or actual_timestamp < scene_start - 1e-6
                    or actual_timestamp >= scene_end
                ):
                    raise TeachObsMultimodalBenchmarkError(
                        f"invalid tail-clamped frame for {lesson_id}"
                    )
            elif (
                should_clamp
                or not math.isclose(
                    actual_timestamp, requested_timestamp, abs_tol=1e-6
                )
                or not math.isclose(clamp_delta, 0.0, abs_tol=1e-9)
            ):
                raise TeachObsMultimodalBenchmarkError(
                    f"invalid unclamped frame timestamp for {lesson_id}"
                )
        else:
            requested_timestamp = expected_requested_timestamp
            clamped = False
            clamp_delta = 0.0
            if not math.isclose(
                actual_timestamp, expected_requested_timestamp, abs_tol=1e-6
            ):
                raise TeachObsMultimodalBenchmarkError(
                    f"legacy frame-task scene alignment mismatch for {lesson_id}"
                )
        relative_text = str(frame.get("path", ""))
        relative = PurePosixPath(relative_text)
        if (
            relative.is_absolute()
            or "\\" in relative_text
            or len(relative.parts) != 1
            or relative.name in {"", ".", ".."}
        ):
            raise TeachObsMultimodalBenchmarkError(
                f"unsafe frame path for {lesson_id}"
            )
        frame_sha256 = _require_sha256(frame.get("sha256"), field="frame")
        if (
            frame.get("frame_id") != expected_id
            or frame.get("scene_no") != expected_scene_no
        ):
            raise TeachObsMultimodalBenchmarkError(
                f"frame-task scene alignment mismatch for {lesson_id}"
            )
        image_path = (task_directory / relative.name).resolve()
        if (
            not image_path.is_relative_to(feature_root)
            or image_path.is_symlink()
            or not image_path.is_file()
            or _file_sha256(image_path) != frame_sha256
        ):
            raise TeachObsMultimodalBenchmarkError(
                f"private frame hash/path mismatch for {lesson_id}"
            )
        source_files.add(image_path)
        result.append(
            {
                "frame_id": expected_id,
                "scene_no": expected_scene_no,
                "requested_timestamp": requested_timestamp,
                "timestamp": actual_timestamp,
                "timestamp_clamped": clamped,
                "timestamp_clamp_delta_seconds": clamp_delta,
                "path": relative.name,
                "combined_path": f"{lesson_id}/{relative.name}",
                "sha256": frame_sha256,
                "_clamp_metadata_present": has_any_frame_clamp_field,
            }
        )
    if "tail_frame_clamp_applied" in task:
        declared_clamp = task["tail_frame_clamp_applied"]
        if not isinstance(declared_clamp, bool) or declared_clamp != (
            clamped_frame_count == 1
        ):
            raise TeachObsMultimodalBenchmarkError(
                f"frame-task clamp summary mismatch for {lesson_id}"
            )
    return result


def _forbidden_visual_evidence_key(value: Any) -> str | None:
    forbidden = {
        "label",
        "labels",
        "gold",
        "codes",
        "prediction",
        "predictions",
        "target",
        "targets",
        "y_true",
        "y_pred",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in forbidden:
                return str(key)
            nested = _forbidden_visual_evidence_key(child)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for child in value:
            nested = _forbidden_visual_evidence_key(child)
            if nested is not None:
                return nested
    return None


def _numeric_with_missing(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> list[float]:
    if value is None:
        return [0.0, 1.0]
    numeric = _finite_number(value, field=field)
    if minimum is not None and numeric < minimum:
        raise TeachObsMultimodalBenchmarkError(f"out-of-range {field}")
    if maximum is not None and numeric > maximum:
        raise TeachObsMultimodalBenchmarkError(f"out-of-range {field}")
    return [numeric, 0.0]


def _image_metric_vector(value: Any, *, lesson_id: str) -> list[float]:
    if not isinstance(value, dict):
        raise TeachObsMultimodalBenchmarkError(
            f"visual image metrics are invalid for {lesson_id}"
        )
    backend = str(value.get("backend", "")).strip()
    if not backend:
        raise TeachObsMultimodalBenchmarkError(
            f"visual image metric backend is missing for {lesson_id}"
        )
    width = value.get("width")
    height = value.get("height")
    dimensions_valid = (
        not isinstance(width, bool)
        and isinstance(width, int)
        and width >= 1
        and not isinstance(height, bool)
        and isinstance(height, int)
        and height >= 1
    )
    if (width is not None or height is not None) and not dimensions_valid:
        raise TeachObsMultimodalBenchmarkError(
            f"visual image dimensions are invalid for {lesson_id}"
        )
    dhash64 = value.get("dhash64")
    if dhash64 is not None and not re.fullmatch(r"[0-9a-fA-F]{16}", str(dhash64)):
        raise TeachObsMultimodalBenchmarkError(
            f"visual dHash is invalid for {lesson_id}"
        )
    result: list[float] = []
    for field in ("rgb_mean", "rgb_stddev"):
        components = value.get(field)
        if components is None:
            components = [None, None, None]
        if not isinstance(components, list) or len(components) != 3:
            raise TeachObsMultimodalBenchmarkError(
                f"visual {field} is invalid for {lesson_id}"
            )
        for index, component in enumerate(components):
            result.extend(
                _numeric_with_missing(
                    component,
                    field=f"visual {field}[{index}]",
                    minimum=0.0,
                    maximum=255.0,
                )
            )
    for field, minimum, maximum in (
        ("luminance_mean", 0.0, 255.0),
        ("edge_difference_mean", 0.0, 255.0),
        ("dark_pixel_fraction", 0.0, 1.0),
        ("bright_pixel_fraction", 0.0, 1.0),
    ):
        result.extend(
            _numeric_with_missing(
                value.get(field),
                field=f"visual {field}",
                minimum=minimum,
                maximum=maximum,
            )
        )
    if backend == "Pillow" and (
        not dimensions_valid
        or dhash64 is None
        or any(result[index] == 1.0 for index in range(1, 20, 2))
    ):
        raise TeachObsMultimodalBenchmarkError(
            f"Pillow image metrics are incomplete for {lesson_id}"
        )
    return result


def _ocr_vector_and_text(
    audit: Any,
    text: Any,
    *,
    lesson_id: str,
) -> tuple[list[float], str, str]:
    if not isinstance(audit, dict) or not isinstance(text, str):
        raise TeachObsMultimodalBenchmarkError(
            f"OCR evidence is invalid for {lesson_id}"
        )
    if "\x00" in text:
        raise TeachObsMultimodalBenchmarkError(
            f"OCR text contains a NUL character for {lesson_id}"
        )
    status = str(audit.get("status", "")).strip()
    if status not in _OCR_STATUSES:
        raise TeachObsMultimodalBenchmarkError(
            f"OCR status is invalid for {lesson_id}"
        )
    raw = audit.get("raw_word_count")
    accepted = audit.get("accepted_word_count")
    if (
        isinstance(raw, bool)
        or not isinstance(raw, int)
        or raw < 0
        or isinstance(accepted, bool)
        or not isinstance(accepted, int)
        or accepted < 0
        or accepted > raw
    ):
        raise TeachObsMultimodalBenchmarkError(
            f"OCR word counts are invalid for {lesson_id}"
        )
    mean_confidence = audit.get("accepted_mean_confidence")
    if mean_confidence is not None:
        confidence = _finite_number(mean_confidence, field="OCR mean confidence")
        if not 0 <= confidence <= 100:
            raise TeachObsMultimodalBenchmarkError(
                f"OCR confidence is out of range for {lesson_id}"
            )
    if (accepted == 0) != (mean_confidence is None):
        raise TeachObsMultimodalBenchmarkError(
            f"OCR accepted count/confidence mismatch for {lesson_id}"
        )
    if audit.get("confidence_is_calibrated_probability") is not False:
        raise TeachObsMultimodalBenchmarkError(
            f"OCR confidence semantics are invalid for {lesson_id}"
        )
    if status != "completed" and (
        text.strip() or raw != 0 or accepted != 0 or mean_confidence is not None
    ):
        raise TeachObsMultimodalBenchmarkError(
            f"unavailable OCR contains fabricated evidence for {lesson_id}"
        )
    result: list[float] = []
    result.extend(_numeric_with_missing(raw, field="OCR raw word count", minimum=0.0))
    result.extend(
        _numeric_with_missing(
            accepted, field="OCR accepted word count", minimum=0.0
        )
    )
    result.extend(
        _numeric_with_missing(
            mean_confidence,
            field="OCR accepted mean confidence",
            minimum=0.0,
            maximum=100.0,
        )
    )
    result.extend(float(status == expected) for expected in _OCR_STATUSES)
    result.append(float(bool(text.strip())))
    return result, text, status


def _transition_vector(
    transition: Any | None,
    *,
    lesson_id: str,
) -> list[float]:
    if transition is None:
        return [0.0, 1.0] * len(_TRANSITION_COMPONENTS) + [0.0] * len(
            _TRANSITION_FLAGS
        )
    if not isinstance(transition, dict):
        raise TeachObsMultimodalBenchmarkError(
            f"visual transition is invalid for {lesson_id}"
        )
    result: list[float] = []
    result.extend(
        _numeric_with_missing(
            transition.get("dhash_hamming_distance"),
            field="transition dHash Hamming distance",
            minimum=0.0,
            maximum=64.0,
        )
    )
    result.extend(
        _numeric_with_missing(
            transition.get("dhash_distance_fraction"),
            field="transition dHash distance fraction",
            minimum=0.0,
            maximum=1.0,
        )
    )
    result.extend(
        _numeric_with_missing(
            transition.get("edge_difference_delta"),
            field="transition edge difference delta",
        )
    )
    event_types = transition.get("event_types")
    if (
        not isinstance(event_types, list)
        or any(not isinstance(item, str) for item in event_types)
        or len(event_types) != len(set(event_types))
    ):
        raise TeachObsMultimodalBenchmarkError(
            f"visual transition event types are invalid for {lesson_id}"
        )
    result.extend(float(event_type in event_types) for event_type in _TRANSITION_FLAGS)
    return result


def _load_visual_evidence_rows(
    path: Path,
    *,
    lesson_id: str,
    scene_count: int,
    media_sha256: str,
    scene_manifest_sha256: str,
    frame_task_sha256: str,
    task_frames: Sequence[Mapping[str, Any]],
) -> tuple[list[list[float]], list[str], dict[str, int], str]:
    value = _read_json_object(path, purpose="TeachObs visual/OCR evidence")
    forbidden_key = _forbidden_visual_evidence_key(value)
    if forbidden_key is not None:
        raise TeachObsMultimodalBenchmarkError(
            f"visual evidence contains forbidden label field: {forbidden_key}"
        )
    configuration = value.get("configuration")
    configuration_sha256 = _require_sha256(
        value.get("configuration_sha256"), field="visual evidence configuration"
    )
    if (
        not isinstance(configuration, dict)
        or configuration.get("event_type_counting_policy")
        != VISUAL_EVIDENCE_EVENT_COUNTING_POLICY
        or _canonical_sha256(configuration) != configuration_sha256
    ):
        raise TeachObsMultimodalBenchmarkError(
            f"visual evidence configuration hash mismatch for {lesson_id}"
        )
    scenes = value.get("scenes")
    transitions = value.get("transitions")
    events = value.get("events")
    event_count = value.get("event_count")
    if (
        value.get("schema") != VISUAL_EVIDENCE_SCHEMA
        or value.get("lesson_id") != lesson_id
        or value.get("media_sha256") != media_sha256
        or value.get("scene_manifest_sha256") != scene_manifest_sha256
        or value.get("task_file_sha256") != frame_task_sha256
        or value.get("scene_count") != scene_count
        or value.get("transition_count") != max(0, scene_count - 1)
        or value.get("private_artifact") is not True
        or value.get("public_release_authorized") is not False
        or not isinstance(scenes, list)
        or len(scenes) != scene_count
        or not isinstance(transitions, list)
        or len(transitions) != max(0, scene_count - 1)
        or isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or event_count < 0
        or not isinstance(events, list)
        or len(events) != event_count
    ):
        raise TeachObsMultimodalBenchmarkError(
            f"visual evidence binding mismatch for {lesson_id}"
        )
    claims = value.get("claim_boundary")
    if (
        not isinstance(claims, dict)
        or claims.get("events_are_deterministic_heuristics") is not True
        or claims.get("event_accuracy_established") is not False
        or claims.get("human_ground_truth_used") is not False
    ):
        raise TeachObsMultimodalBenchmarkError(
            f"visual evidence claim boundary mismatch for {lesson_id}"
        )

    transition_by_after: dict[str, Mapping[str, Any]] = {}
    for index, transition in enumerate(transitions, 1):
        before = task_frames[index - 1]
        after = task_frames[index]
        if (
            not isinstance(transition, dict)
            or transition.get("before_frame_id") != before["frame_id"]
            or transition.get("after_frame_id") != after["frame_id"]
            or not math.isclose(
                _finite_number(transition.get("start"), field="transition start"),
                float(before["timestamp"]),
                abs_tol=1e-6,
            )
            or not math.isclose(
                _finite_number(transition.get("end"), field="transition end"),
                float(after["timestamp"]),
                abs_tol=1e-6,
            )
        ):
            raise TeachObsMultimodalBenchmarkError(
                f"visual transition scene alignment mismatch for {lesson_id}"
            )
        transition_by_after[str(after["frame_id"])] = transition

    numeric_rows: list[list[float]] = []
    ocr_texts: list[str] = []
    status_counts = {status: 0 for status in _OCR_STATUSES}
    transition_flag_counts = {event_type: 0 for event_type in _TRANSITION_FLAGS}
    for expected_scene_no, (row, task_frame) in enumerate(
        zip(scenes, task_frames, strict=True), start=1
    ):
        core_fields = ("frame_id", "scene_no", "timestamp", "path", "sha256")
        clamp_fields = (
            "requested_timestamp",
            "timestamp_clamped",
            "timestamp_clamp_delta_seconds",
        )
        clamp_metadata_present = bool(task_frame["_clamp_metadata_present"])
        legacy_optional_mismatch = (
            isinstance(row, dict)
            and not clamp_metadata_present
            and any(
                field in row and row.get(field) != task_frame[field]
                for field in clamp_fields
            )
        )
        if (
            not isinstance(row, dict)
            or any(row.get(field) != task_frame[field] for field in core_fields)
            or (
                clamp_metadata_present
                and any(row.get(field) != task_frame[field] for field in clamp_fields)
            )
            or legacy_optional_mismatch
        ):
            raise TeachObsMultimodalBenchmarkError(
                f"visual evidence scene alignment mismatch for {lesson_id}"
            )
        vector = _image_metric_vector(row.get("image_metrics"), lesson_id=lesson_id)
        ocr_vector, ocr_text, status = _ocr_vector_and_text(
            row.get("ocr_audit"), row.get("ocr_text"), lesson_id=lesson_id
        )
        vector.extend(ocr_vector)
        transition = transition_by_after.get(str(task_frame["frame_id"]))
        transition_vector = _transition_vector(transition, lesson_id=lesson_id)
        vector.extend(transition_vector)
        if len(vector) != len(_VISUAL_NUMERIC_FEATURE_NAMES):
            raise AssertionError("visual numeric feature layout drift")
        numeric_rows.append(vector)
        ocr_texts.append(ocr_text)
        status_counts[status] += 1
        for event_type, flag in zip(
            _TRANSITION_FLAGS, transition_vector[-len(_TRANSITION_FLAGS) :], strict=True
        ):
            transition_flag_counts[event_type] += int(flag)
        if expected_scene_no == 1 and transition is not None:
            raise AssertionError("first scene cannot have a previous transition")

    declared_status_counts = value.get("ocr_status_counts")
    valid_declared_status_counts = isinstance(declared_status_counts, dict) and all(
        key in _OCR_STATUSES
        and not isinstance(count, bool)
        and isinstance(count, int)
        and count > 0
        for key, count in declared_status_counts.items()
    )
    if not valid_declared_status_counts or declared_status_counts != {
        key: count for key, count in status_counts.items() if count
    }:
        raise TeachObsMultimodalBenchmarkError(
            f"visual evidence OCR status counts mismatch for {lesson_id}"
        )
    declared_event_counts = value.get("event_type_counts")
    computed_event_counts = {
        event_type: 0 for event_type in VISUAL_EVIDENCE_EVENT_TYPES
    }
    for event in events:
        event_type = event.get("type") if isinstance(event, dict) else None
        if event_type not in computed_event_counts:
            raise TeachObsMultimodalBenchmarkError(
                f"visual evidence event type is invalid for {lesson_id}"
            )
        computed_event_counts[event_type] += 1
    valid_declared_event_counts = (
        isinstance(declared_event_counts, dict)
        and set(declared_event_counts) == set(VISUAL_EVIDENCE_EVENT_TYPES)
        and all(
            not isinstance(count, bool) and isinstance(count, int) and count >= 0
            for count in declared_event_counts.values()
        )
        and declared_event_counts == computed_event_counts
        and sum(declared_event_counts.values()) == event_count
        and all(
            computed_event_counts[event_type]
            == transition_flag_counts[event_type]
            for event_type in _TRANSITION_FLAGS
        )
    )
    if not valid_declared_event_counts:
        raise TeachObsMultimodalBenchmarkError(
            f"visual evidence event type counts mismatch for {lesson_id}"
        )
    return numeric_rows, ocr_texts, status_counts, configuration_sha256


def _load_combined_visual_rows(
    *,
    feature_root: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    scene_counts: dict[str, int],
    repository: Path,
    source_files: set[Path],
    profile_resolution: ResolvedTeachObsBenchmarkProfile,
) -> tuple[dict[tuple[str, int], list[float]], str, str]:
    """Validate the profile-selected combined CLIP task/result and vectors."""

    task_path = _bound_feature_path(
        feature_root,
        manifest.get("combined_clip_task_path"),
        manifest.get("combined_clip_task_sha256"),
        purpose="combined CLIP task",
    )
    result_path = _bound_feature_path(
        feature_root,
        manifest.get("combined_clip_result_path"),
        manifest.get("combined_clip_result_sha256"),
        purpose="combined CLIP result",
    )
    if task_path == result_path:
        raise TeachObsMultimodalBenchmarkError(
            "combined CLIP task and result cannot reuse one file"
        )
    source_files.update((task_path, result_path))
    expected_frames: list[dict[str, Any]] = []
    media_set: list[dict[str, str]] = []
    scene_manifest_set: list[dict[str, str]] = []
    for record in records:
        lesson_id = str(record.get("lesson_id", ""))
        if lesson_id not in scene_counts:
            raise TeachObsMultimodalBenchmarkError(
                "combined CLIP manifest contains an unknown lesson"
            )
        media_sha256 = _require_sha256(record.get("media_sha256"), field="media")
        official_manifest_digest = _official_manifest_sha256(repository, lesson_id)
        if record.get("scene_manifest_sha256") != official_manifest_digest:
            raise TeachObsMultimodalBenchmarkError(
                f"official scene-manifest hash mismatch for {lesson_id}"
            )
        expected_task_relative = PurePosixPath("frames") / lesson_id / "task.json"
        if PurePosixPath(str(record.get("frame_task_path", ""))) != expected_task_relative:
            raise TeachObsMultimodalBenchmarkError(
                f"combined CLIP frame-task path mismatch for {lesson_id}"
            )
        per_lesson_task = _bound_feature_path(
            feature_root,
            record.get("frame_task_path"),
            record.get("frame_task_sha256"),
            purpose="per-lesson frame task",
        )
        source_files.add(per_lesson_task)
        lesson_frames = _load_bound_frame_task(
            per_lesson_task,
            feature_root=feature_root,
            lesson_id=lesson_id,
            scene_count=scene_counts[lesson_id],
            media_sha256=media_sha256,
            scene_manifest_sha256=official_manifest_digest,
            source_files=source_files,
        )
        expected_frames.extend(
            {**frame, "path": frame["combined_path"]} for frame in lesson_frames
        )
        media_set.append({"lesson_id": lesson_id, "media_sha256": media_sha256})
        scene_manifest_set.append(
            {
                "lesson_id": lesson_id,
                "scene_manifest_sha256": official_manifest_digest,
            }
        )
    expected_frame_count = sum(scene_counts.values())
    task = _read_json_object(task_path, purpose="combined CLIP task")
    task_frames = task.get("frames")
    media_set_sha256 = _canonical_sha256(media_set)
    scene_manifest_set_sha256 = _canonical_sha256(scene_manifest_set)
    paper_partial = (
        profile_resolution.spec.profile_id
        == PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE
    )
    expected_included_ids = [
        f"S{index}" for index in range(1, 31) if index != 4
    ]
    partial_task_binding_valid = not paper_partial or (
        task.get("complete") is False
        and task.get("lesson_count") == 29
        and task.get("expected_lesson_count") == 30
        and task.get("included_lesson_ids") == expected_included_ids
        and task.get("expected_lesson_ids")
        == [f"S{index}" for index in range(1, 31)]
        and task.get("failed_lesson_ids") == ["S4"]
    )
    if (
        task.get("schema") != FRAME_TASK_SCHEMA
        or task.get("video_id") != "teachobs_combined_scene_midpoints"
        or task.get("media_set") != media_set
        or task.get("media_sha256") != media_set_sha256
        or task.get("scene_manifest_set_sha256") != scene_manifest_set_sha256
        or task.get("frame_count") != expected_frame_count
        or manifest.get("combined_clip_frame_count") != expected_frame_count
        or not isinstance(task_frames, list)
        or len(task_frames) != expected_frame_count
        or task.get("private_artifact") is not True
        or task.get("public_release_authorized") is not False
        or not partial_task_binding_valid
    ):
        raise TeachObsMultimodalBenchmarkError(
            "combined CLIP task binding or frame count mismatch"
        )
    optional_manifest_bindings = {
        "combined_clip_media_set_sha256": media_set_sha256,
        "combined_clip_scene_manifest_set_sha256": scene_manifest_set_sha256,
        "combined_clip_task_manifest_sha256": _canonical_sha256(task),
    }
    for field, expected in optional_manifest_bindings.items():
        if field in manifest and manifest[field] != expected:
            raise TeachObsMultimodalBenchmarkError(
                f"combined CLIP top-level {field} mismatch"
            )
    for expected, actual in zip(expected_frames, task_frames, strict=True):
        core_fields = ("frame_id", "scene_no", "timestamp", "path", "sha256")
        clamp_fields = (
            "requested_timestamp",
            "timestamp_clamped",
            "timestamp_clamp_delta_seconds",
        )
        clamp_metadata_present = expected["_clamp_metadata_present"]
        if (
            not isinstance(actual, dict)
            or any(actual.get(field) != expected[field] for field in core_fields)
            or (
                clamp_metadata_present
                and any(actual.get(field) != expected[field] for field in clamp_fields)
            )
            or (
                not clamp_metadata_present
                and any(field in actual for field in clamp_fields)
            )
        ):
            raise TeachObsMultimodalBenchmarkError(
                "combined CLIP task has a wrong lesson/scene/frame mapping"
            )

    result = _read_json_object(result_path, purpose="combined CLIP result")
    result_frames = result.get("frames")
    task_file_sha256 = _file_sha256(task_path)
    provenance = result.get("model_provenance")
    weight_manifest = provenance.get("weight_manifest") if isinstance(provenance, dict) else None
    weight_sha256 = _require_sha256(
        weight_manifest.get("manifest_sha256")
        if isinstance(weight_manifest, dict)
        else None,
        field="CLIP weight manifest",
    )
    source_revision = str(result.get("teachobs_source_revision", "")).strip()
    if (
        result.get("schema") != SCHEMA_RESULT
        or result.get("video_id") != "teachobs_combined_scene_midpoints"
        or result.get("media_sha256") != media_set_sha256
        or result.get("task_manifest_sha256") != _canonical_sha256(task)
        or result.get("teachobs_task_file_sha256") != task_file_sha256
        or result.get("frame_count") != expected_frame_count
        or not isinstance(result_frames, list)
        or len(result_frames) != expected_frame_count
        or not source_revision
        or result.get("private_artifact") is not True
        or result.get("public_release_authorized") is not False
    ):
        raise TeachObsMultimodalBenchmarkError(
            "combined CLIP result/task/media binding mismatch"
        )
    if "combined_clip_model_provenance_sha256" in manifest and manifest[
        "combined_clip_model_provenance_sha256"
    ] != _canonical_sha256(provenance):
        raise TeachObsMultimodalBenchmarkError(
            "combined CLIP top-level model provenance mismatch"
        )

    vectors: dict[tuple[str, int], list[float]] = {}
    dimension: int | None = None
    for expected, task_frame, row in zip(
        expected_frames, task_frames, result_frames, strict=True
    ):
        if not isinstance(row, dict) or any(
            row.get(field) != task_frame.get(field)
            for field in ("frame_id", "timestamp", "path", "sha256")
        ):
            raise TeachObsMultimodalBenchmarkError(
                "combined CLIP result has a wrong lesson/scene/frame mapping"
            )
        embedding = row.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise TeachObsMultimodalBenchmarkError(
                "combined CLIP result is missing an embedding"
            )
        vector = [_finite_number(item, field="visual embedding") for item in embedding]
        if dimension is None:
            dimension = len(vector)
            if dimension < 2 or dimension > 4096:
                raise TeachObsMultimodalBenchmarkError(
                    "unsupported CLIP embedding dimension"
                )
        elif len(vector) != dimension:
            raise TeachObsMultimodalBenchmarkError(
                "combined CLIP embedding dimension mismatch"
            )
        if row.get("embedding_sha256") != _canonical_sha256(vector):
            raise TeachObsMultimodalBenchmarkError(
                "combined CLIP embedding hash mismatch"
            )
        frame_match = re.fullmatch(r"(S(?:[1-9]|[12][0-9]|30))_scene_(\d{4})", expected["frame_id"])
        assert frame_match is not None
        key = (frame_match.group(1), int(frame_match.group(2)))
        if key in vectors:
            raise TeachObsMultimodalBenchmarkError(
                "combined CLIP result contains a duplicate scene"
            )
        vectors[key] = vector
    if len(vectors) != expected_frame_count:
        raise TeachObsMultimodalBenchmarkError(
            "combined CLIP result does not cover every scene exactly once"
        )
    return vectors, source_revision, weight_sha256


def load_teachobs_multimodal_features(
    repository_root: str | Path,
    feature_manifest_path: str | Path,
    *,
    dataset: TeachObsTextDataset | None = None,
    profile: str = DEFAULT_BENCHMARK_PROFILE,
) -> TeachObsMultimodalFeatures:
    """Load audio, OCR, image/change evidence, and CLIP for selected scenes."""

    dependencies = _require_dependencies()
    np = dependencies["np"]
    repository = Path(repository_root).expanduser().resolve()
    if not repository.is_dir():
        raise TeachObsMultimodalBenchmarkError("TeachObs repository root is missing")
    full_dataset = dataset or load_teachobs_text_dataset(repository)
    profile_resolution = resolve_teachobs_benchmark_profile(full_dataset, profile)
    checked_dataset = profile_resolution.dataset
    manifest_file = Path(feature_manifest_path).expanduser()
    if manifest_file.is_symlink() or not manifest_file.is_file():
        raise TeachObsMultimodalBenchmarkError(
            "TeachObs private feature manifest is missing or unsafe"
        )
    manifest_file = manifest_file.resolve()
    root = manifest_file.parent.resolve()
    manifest = _read_json_object(
        manifest_file, purpose="TeachObs private feature manifest"
    )
    all_scenes = checked_dataset.train_scenes + checked_dataset.test_scenes
    scene_counts: dict[str, int] = {}
    for scene in all_scenes:
        scene_counts[scene.lesson_id] = scene_counts.get(scene.lesson_id, 0) + 1
    rows = manifest.get("lessons")
    paper_partial = (
        profile_resolution.spec.profile_id
        == PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE
    )
    expected_covered_lesson_ids = [
        f"S{index}" for index in range(1, 31) if not paper_partial or index != 4
    ]
    base_manifest_valid = (
        manifest.get("schema") != FEATURE_SCHEMA
        or manifest.get("private_artifact") is not True
        or manifest.get("public_release_authorized") is not False
        or manifest.get("audio_statistics_included") is not True
        or manifest.get("visual_evidence_included") is not True
        or manifest.get("clip_embeddings_included") is not True
        or not isinstance(rows, list)
        or manifest.get("lesson_count") != len(scene_counts)
        or manifest.get("expected_lesson_count") != 30
        or manifest.get("scene_count") != len(all_scenes)
        or len(rows) != len(scene_counts)
        or [str(row.get("lesson_id", "")) for row in rows if isinstance(row, dict)]
        != expected_covered_lesson_ids
    )
    if paper_partial:
        claim_boundary = manifest.get("claim_boundary")
        profile_manifest_valid = (
            manifest.get("complete") is False
            and manifest.get("lesson_count") == 29
            and manifest.get("failed_lesson_count") == 1
            and manifest.get("failed_lesson_ids") == ["S4"]
            and isinstance(manifest.get("failure_records_path"), str)
            and _SHA256.fullmatch(
                str(manifest.get("failure_records_sha256", ""))
            )
            is not None
            and manifest.get("clip_embeddings_complete_for_all_lessons") is False
            and manifest.get("combined_clip_complete") is False
            and manifest.get("combined_clip_lesson_count") == 29
            and manifest.get("combined_clip_expected_lesson_count") == 30
            and manifest.get("combined_clip_lesson_ids")
            == expected_covered_lesson_ids
            and manifest.get("combined_clip_failed_lesson_ids") == ["S4"]
            and isinstance(claim_boundary, dict)
            and claim_boundary.get("full_scene_midpoint_frames_extracted") is False
            and claim_boundary.get("audio_statistics_computed") is False
            and claim_boundary.get("image_metrics_computed") is False
            and claim_boundary.get("adjacent_visual_events_inferred") is False
            and claim_boundary.get("clip_visual_embeddings_computed") is True
            and claim_boundary.get(
                "clip_visual_embeddings_complete_for_all_lessons"
            )
            is False
            and claim_boundary.get("clip_visual_embeddings_partial_success_only")
            is True
        )
    else:
        profile_manifest_valid = (
            manifest.get("complete") is True
            and manifest.get("lesson_count") == 30
            and manifest.get("failed_lesson_count", 0) == 0
            and manifest.get("failed_lesson_ids", []) == []
            and manifest.get("clip_embeddings_complete_for_all_lessons", True)
            is True
            and manifest.get("combined_clip_complete", True) is True
        )
    if base_manifest_valid or not profile_manifest_valid:
        raise TeachObsMultimodalBenchmarkError(
            "TeachObs feature manifest is incompatible with the selected profile"
        )
    combined_fields = {
        "combined_clip_task_path",
        "combined_clip_task_sha256",
        "combined_clip_result_path",
        "combined_clip_result_sha256",
        "combined_clip_frame_count",
    }
    present_combined_fields = combined_fields & set(manifest)
    if present_combined_fields and present_combined_fields != combined_fields:
        raise TeachObsMultimodalBenchmarkError(
            "TeachObs combined CLIP manifest binding is incomplete"
        )
    combined_layout = present_combined_fields == combined_fields
    if paper_partial and not combined_layout:
        raise TeachObsMultimodalBenchmarkError(
            "paper Track 1 profile requires its exact partial combined CLIP binding"
        )

    audio_by_scene: dict[tuple[str, int], list[float]] = {}
    visual_numeric_by_scene: dict[tuple[str, int], list[float]] = {}
    ocr_text_by_scene: dict[tuple[str, int], str] = {}
    clip_by_scene: dict[tuple[str, int], list[float]] = {}
    seen_lessons: set[str] = set()
    revisions: set[str] = set()
    weight_digests: set[str] = set()
    visual_dimensions: set[int] = set()
    evidence_receipts: list[dict[str, str]] = []
    evidence_configuration_receipts: list[dict[str, str]] = []
    aggregate_ocr_status_counts = {status: 0 for status in _OCR_STATUSES}
    source_files: set[Path] = {manifest_file}
    if paper_partial:
        failure_path = _bound_feature_path(
            root,
            manifest.get("failure_records_path"),
            manifest.get("failure_records_sha256"),
            purpose="feature failure records",
        )
        source_files.add(failure_path)
        failure_receipt = _read_json_object(
            failure_path, purpose="TeachObs feature failure records"
        )
        failures = failure_receipt.get("failures")
        if (
            failure_receipt.get("schema") != FEATURE_FAILURE_SCHEMA
            or failure_receipt.get("private_artifact") is not True
            or failure_receipt.get("public_release_authorized") is not False
            or failure_receipt.get("failure_count") != 1
            or not isinstance(failures, list)
            or len(failures) != 1
            or not isinstance(failures[0], dict)
            or failures[0].get("lesson_id") != "S4"
        ):
            raise TeachObsMultimodalBenchmarkError(
                "paper Track 1 feature failure receipt is not the unique S4 hole"
            )
    for record in rows:
        if not isinstance(record, dict):
            raise TeachObsMultimodalBenchmarkError(
                "TeachObs feature manifest contains a non-object lesson"
            )
        lesson_id = str(record.get("lesson_id", ""))
        if lesson_id not in scene_counts or lesson_id in seen_lessons:
            raise TeachObsMultimodalBenchmarkError(
                "TeachObs feature manifest has an unknown or duplicate lesson"
            )
        seen_lessons.add(lesson_id)
        scene_count = scene_counts[lesson_id]
        if record.get("scene_count") != scene_count:
            raise TeachObsMultimodalBenchmarkError(
                f"feature scene count mismatch for {lesson_id}"
            )
        media_sha256 = _require_sha256(
            record.get("media_sha256"), field="media"
        )
        official_manifest_digest = _official_manifest_sha256(repository, lesson_id)
        if record.get("scene_manifest_sha256") != official_manifest_digest:
            raise TeachObsMultimodalBenchmarkError(
                f"official scene-manifest hash mismatch for {lesson_id}"
            )
        frame_task_path = _bound_feature_path(
            root,
            record.get("frame_task_path"),
            record.get("frame_task_sha256"),
            purpose="per-lesson frame task",
        )
        if frame_task_path in source_files:
            raise TeachObsMultimodalBenchmarkError("private feature file is reused")
        source_files.add(frame_task_path)
        frame_task_sha256 = _file_sha256(frame_task_path)
        task_frames = _load_bound_frame_task(
            frame_task_path,
            feature_root=root,
            lesson_id=lesson_id,
            scene_count=scene_count,
            media_sha256=media_sha256,
            scene_manifest_sha256=official_manifest_digest,
            source_files=source_files,
        )
        evidence_path = _bound_feature_path(
            root,
            record.get("visual_evidence_path"),
            record.get("visual_evidence_sha256"),
            purpose="visual evidence",
        )
        if evidence_path in source_files:
            raise TeachObsMultimodalBenchmarkError("private feature file is reused")
        source_files.add(evidence_path)
        (
            numeric_rows,
            ocr_texts,
            lesson_ocr_status_counts,
            configuration_sha256,
        ) = _load_visual_evidence_rows(
            evidence_path,
            lesson_id=lesson_id,
            scene_count=scene_count,
            media_sha256=media_sha256,
            scene_manifest_sha256=official_manifest_digest,
            frame_task_sha256=frame_task_sha256,
            task_frames=task_frames,
        )
        if "ocr_status_counts" in record and record.get(
            "ocr_status_counts"
        ) != {
            key: count for key, count in lesson_ocr_status_counts.items() if count
        }:
            raise TeachObsMultimodalBenchmarkError(
                f"manifest OCR status counts mismatch for {lesson_id}"
            )
        evidence_digest = _file_sha256(evidence_path)
        evidence_receipts.append(
            {"lesson_id": lesson_id, "visual_evidence_sha256": evidence_digest}
        )
        evidence_configuration_receipts.append(
            {
                "lesson_id": lesson_id,
                "configuration_sha256": configuration_sha256,
            }
        )
        for status, count in lesson_ocr_status_counts.items():
            aggregate_ocr_status_counts[status] += count
        audio_path = _bound_feature_path(
            root,
            record.get("audio_feature_path"),
            record.get("audio_feature_sha256"),
            purpose="audio feature",
        )
        if audio_path in source_files:
            raise TeachObsMultimodalBenchmarkError("private feature file is reused")
        source_files.add(audio_path)
        audio_rows = _load_audio_rows(
            audio_path,
            lesson_id=lesson_id,
            scene_count=scene_count,
            media_sha256=media_sha256,
            scene_manifest_sha256=official_manifest_digest,
        )
        visual_rows: list[list[float]] | None = None
        if not combined_layout:
            visual_path = _bound_feature_path(
                root,
                record.get("visual_feature_path"),
                record.get("visual_feature_sha256"),
                purpose="visual feature",
            )
            if visual_path in source_files:
                raise TeachObsMultimodalBenchmarkError(
                    "private feature file is reused"
                )
            source_files.add(visual_path)
            visual_rows, revision, weight_digest = _load_visual_rows(
                visual_path,
                lesson_id=lesson_id,
                scene_count=scene_count,
                media_sha256=media_sha256,
                frame_task_sha256=frame_task_sha256,
                task_frames=task_frames,
            )
            revisions.add(revision)
            weight_digests.add(weight_digest)
            visual_dimensions.add(len(visual_rows[0]))
        for scene_no, audio in enumerate(audio_rows, start=1):
            key = (lesson_id, scene_no)
            audio_by_scene[key] = audio
            visual_numeric_by_scene[key] = numeric_rows[scene_no - 1]
            ocr_text_by_scene[key] = ocr_texts[scene_no - 1]
            if visual_rows is not None:
                clip_by_scene[key] = visual_rows[scene_no - 1]

    if seen_lessons != set(scene_counts):
        raise TeachObsMultimodalBenchmarkError(
            "TeachObs feature manifest does not cover every profile-selected lesson"
        )
    if combined_layout:
        clip_by_scene, revision, weight_digest = _load_combined_visual_rows(
            feature_root=root,
            manifest=manifest,
            records=rows,
            scene_counts=scene_counts,
            repository=repository,
            source_files=source_files,
            profile_resolution=profile_resolution,
        )
        revisions.add(revision)
        weight_digests.add(weight_digest)
        visual_dimensions.add(len(next(iter(clip_by_scene.values()))))
    if len(revisions) != 1 or len(weight_digests) != 1 or len(visual_dimensions) != 1:
        raise TeachObsMultimodalBenchmarkError(
            "CLIP model provenance or embedding dimension differs across lessons"
        )

    def matrix(scenes: tuple[Any, ...], source: dict[tuple[str, int], list[float]]) -> Any:
        try:
            value = np.asarray(
                [source[(scene.lesson_id, scene.scene_no)] for scene in scenes],
                dtype=np.float64,
            )
        except KeyError as exc:
            raise TeachObsMultimodalBenchmarkError(
                "private feature rows do not align with the official scenes"
            ) from exc
        if value.ndim != 2 or not np.isfinite(value).all():
            raise TeachObsMultimodalBenchmarkError("invalid private feature matrix")
        return value

    def texts(scenes: tuple[Any, ...]) -> tuple[str, ...]:
        try:
            return tuple(
                ocr_text_by_scene[(scene.lesson_id, scene.scene_no)] for scene in scenes
            )
        except KeyError as exc:
            raise TeachObsMultimodalBenchmarkError(
                "private OCR rows do not align with the official scenes"
            ) from exc

    clip_dimension = visual_dimensions.pop()
    numeric_dimension = len(_VISUAL_NUMERIC_FEATURE_NAMES)
    ocr_status_counts = {
        key: count for key, count in aggregate_ocr_status_counts.items() if count
    }
    train_ocr = texts(checked_dataset.train_scenes)
    test_ocr = texts(checked_dataset.test_scenes)

    return TeachObsMultimodalFeatures(
        train_audio=matrix(checked_dataset.train_scenes, audio_by_scene),
        test_audio=matrix(checked_dataset.test_scenes, audio_by_scene),
        train_visual_numeric=matrix(
            checked_dataset.train_scenes, visual_numeric_by_scene
        ),
        test_visual_numeric=matrix(
            checked_dataset.test_scenes, visual_numeric_by_scene
        ),
        train_clip=matrix(checked_dataset.train_scenes, clip_by_scene),
        test_clip=matrix(checked_dataset.test_scenes, clip_by_scene),
        train_ocr_text=train_ocr,
        test_ocr_text=test_ocr,
        feature_manifest_sha256=_file_sha256(manifest_file),
        audio_feature_names=_AUDIO_FEATURE_NAMES,
        visual_numeric_feature_names=_VISUAL_NUMERIC_FEATURE_NAMES,
        visual_dimension=numeric_dimension + clip_dimension,
        visual_numeric_dimension=numeric_dimension,
        clip_dimension=clip_dimension,
        clip_source_revision=revisions.pop(),
        clip_weight_manifest_sha256=weight_digests.pop(),
        visual_feature_layout=(
            "single_combined_clip_result" if combined_layout else "legacy_per_lesson_clip_results"
        ),
        visual_evidence_set_sha256=_canonical_sha256(evidence_receipts),
        visual_evidence_configuration_set_sha256=_canonical_sha256(
            evidence_configuration_receipts
        ),
        visual_evidence_configuration_sha256_values=tuple(
            sorted(
                {
                    item["configuration_sha256"]
                    for item in evidence_configuration_receipts
                }
            )
        ),
        ocr_status_counts=ocr_status_counts,
        ocr_nonempty_scene_count=sum(bool(text.strip()) for text in train_ocr + test_ocr),
        ocr_completed_scene_count=aggregate_ocr_status_counts["completed"],
        source_file_count=len(source_files),
        benchmark_profile=profile_resolution.spec.profile_id,
        dataset_profile_fingerprint=(
            profile_resolution.dataset_profile_fingerprint
        ),
        selected_train_lesson_ids=(
            profile_resolution.selected_train_lesson_ids
        ),
        selected_test_lesson_ids=profile_resolution.selected_test_lesson_ids,
        selected_train_sample_ids=profile_resolution.selected_train_sample_ids,
        selected_test_sample_ids=profile_resolution.selected_test_sample_ids,
    )


def _require_dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        from scipy.sparse import csr_matrix, hstack
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TeachObsBenchmarkDependencyError(
            "TeachObs multimodal benchmarking requires the 'recognition' optional "
            "dependencies (NumPy, SciPy, and scikit-learn)"
        ) from exc
    return {
        "np": np,
        "csr_matrix": csr_matrix,
        "hstack": hstack,
        "TfidfVectorizer": TfidfVectorizer,
        "LogisticRegression": LogisticRegression,
        "GroupKFold": GroupKFold,
        "StandardScaler": StandardScaler,
    }


def _fit_train_only_tfidf(
    train_documents: Sequence[str],
    test_documents: Sequence[str],
    *,
    analyzer: str,
    ngram_range: tuple[int, int],
    max_features: int,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Fit one text block on training scenes only, with an empty OCR fallback."""

    dependencies = _require_dependencies()
    if not all(isinstance(value, str) for value in (*train_documents, *test_documents)):
        raise TeachObsMultimodalBenchmarkError("TF-IDF documents must be strings")
    if not any(value.strip() for value in train_documents):
        empty_state = {
            "configuration": {
                "analyzer": analyzer,
                "ngram_range": list(ngram_range),
                "lowercase": True,
                "min_df": 1,
                "max_features": max_features,
                "sublinear_tf": True,
                "norm": "l2",
                "use_idf": True,
                "smooth_idf": True,
            },
            "vocabulary": {},
            "idf": dependencies["np"].zeros(0, dtype=dependencies["np"].float64),
        }
        return (
            dependencies["csr_matrix"]((len(train_documents), 0), dtype=float),
            dependencies["csr_matrix"]((len(test_documents), 0), dtype=float),
            {
                "input_sample_count": len(train_documents),
                "fit_sample_count": 0,
                "feature_count": 0,
                "vocabulary_sha256": _canonical_sha256([]),
                "training_text_available": False,
            },
            empty_state,
        )
    vectorizer = dependencies["TfidfVectorizer"](
        analyzer=analyzer,
        ngram_range=ngram_range,
        lowercase=True,
        min_df=1,
        max_features=max_features,
        sublinear_tf=True,
        dtype=dependencies["np"].float64,
    )
    train_matrix = vectorizer.fit_transform(train_documents)
    test_matrix = vectorizer.transform(test_documents)
    vocabulary = {
        str(key): int(value) for key, value in vectorizer.vocabulary_.items()
    }
    audit = {
        "input_sample_count": len(train_documents),
        "fit_sample_count": len(train_documents),
        "feature_count": int(train_matrix.shape[1]),
        "vocabulary_sha256": _canonical_sha256(sorted(vocabulary.items())),
        "training_text_available": True,
    }
    state = {
        "configuration": {
            "analyzer": analyzer,
            "ngram_range": list(ngram_range),
            "lowercase": True,
            "min_df": 1,
            "max_features": max_features,
            "sublinear_tf": True,
            "norm": "l2",
            "use_idf": True,
            "smooth_idf": True,
        },
        "vocabulary": vocabulary,
        "idf": dependencies["np"].asarray(vectorizer.idf_, dtype=dependencies["np"].float64),
    }
    return train_matrix, test_matrix, audit, state


def _rounded(value: Any) -> float:
    return round(float(value), 6)


def _metric_values(y_true: Any, y_pred: Any, groups: tuple[str, ...]) -> dict[str, float]:
    dependencies = _require_dependencies()
    np = dependencies["np"]
    truth = np.asarray(y_true, dtype=np.uint8)
    predictions = np.asarray(y_pred, dtype=np.uint8)
    tp = np.logical_and(truth == 1, predictions == 1).sum(axis=0, dtype=np.int64)
    fp = np.logical_and(truth == 0, predictions == 1).sum(axis=0, dtype=np.int64)
    fn = np.logical_and(truth == 1, predictions == 0).sum(axis=0, dtype=np.int64)
    denominator = 2 * tp + fp + fn
    per_label_f1 = np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(tp, dtype=np.float64),
        where=denominator != 0,
    )
    total_denominator = int(denominator.sum())
    micro = float(2 * tp.sum() / total_denominator) if total_denominator else 0.0
    visual = np.asarray([group == "visual" for group in groups], dtype=bool)
    return {
        "micro_f1": micro,
        "macro_f1": float(per_label_f1.mean()),
        "subset_accuracy": float(np.all(truth == predictions, axis=1).mean()),
        "hamming_accuracy": float((truth == predictions).mean()),
        "visual_macro_f1": float(per_label_f1[visual].mean()),
        "nonvisual_macro_f1": float(per_label_f1[~visual].mean()),
    }


def _metric_summary(y_true: Any, y_pred: Any, groups: tuple[str, ...]) -> dict[str, float]:
    return {
        key: _rounded(value)
        for key, value in _metric_values(y_true, y_pred, groups).items()
    }


def _fit_predict_arm_with_state(
    x_train: Any,
    y_train: Any,
    x_test: Any,
    *,
    regularization_c: float = 1.0,
    class_weight: str | None = "balanced",
    thresholds: Any | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    dependencies = _require_dependencies()
    np = dependencies["np"]
    if (
        isinstance(regularization_c, bool)
        or not isinstance(regularization_c, (int, float))
        or not math.isfinite(float(regularization_c))
        or float(regularization_c) <= 0.0
        or class_weight not in {None, "balanced"}
    ):
        raise TeachObsMultimodalBenchmarkError(
            "invalid train-only logistic-regression configuration"
        )
    predictions = np.zeros((x_test.shape[0], y_train.shape[1]), dtype=np.uint8)
    probabilities = np.zeros(
        (x_test.shape[0], y_train.shape[1]), dtype=np.float64
    )
    coefficient = np.zeros(
        (y_train.shape[1], x_train.shape[1]), dtype=np.float64
    )
    intercept = np.zeros(y_train.shape[1], dtype=np.float64)
    if thresholds is None:
        checked_thresholds = np.full(
            y_train.shape[1], _DECISION_THRESHOLD, dtype=np.float64
        )
    else:
        checked_thresholds = np.ascontiguousarray(thresholds, dtype=np.float64)
        if (
            checked_thresholds.shape != (y_train.shape[1],)
            or not np.isfinite(checked_thresholds).all()
            or np.any(checked_thresholds <= 0.0)
            or np.any(checked_thresholds >= 1.0)
        ):
            raise TeachObsMultimodalBenchmarkError(
                "invalid train-only per-label decision thresholds"
            )
    model_kinds: list[str] = []
    model_classes: list[list[int]] = []
    converged: list[bool] = []
    for label_index in range(y_train.shape[1]):
        column = y_train[:, label_index]
        unique = np.unique(column)
        if unique.size == 1:
            constant = int(unique[0])
            predictions[:, label_index] = constant
            probabilities[:, label_index] = float(constant)
            model_kinds.append(f"constant_{constant}")
            model_classes.append([constant])
            converged.append(True)
            continue
        classifier = dependencies["LogisticRegression"](
            C=float(regularization_c),
            class_weight=class_weight,
            max_iter=_LOGISTIC_MAX_ITER,
            random_state=_RANDOM_STATE,
            solver=_LOGISTIC_SOLVER,
        )
        classifier.fit(x_train, column)
        positive_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
        probability = classifier.predict_proba(x_test)[:, positive_index]
        probabilities[:, label_index] = probability
        predictions[:, label_index] = (probability >= checked_thresholds[label_index]).astype(
            np.uint8
        )
        coefficient[label_index] = np.asarray(
            classifier.coef_[0], dtype=np.float64
        )
        intercept[label_index] = float(classifier.intercept_[0])
        model_kinds.append("logistic_regression")
        model_classes.append([int(value) for value in classifier.classes_])
        converged.append(int(classifier.n_iter_[0]) < _LOGISTIC_MAX_ITER)
    audit = {
        "model_kind_counts": {
            kind: model_kinds.count(kind) for kind in sorted(set(model_kinds))
        },
        "all_iterative_models_converged": all(converged),
        "regularization_c": float(regularization_c),
        "class_weight": class_weight,
        "threshold_count": int(checked_thresholds.size),
        "threshold_sha256": _array_sha256(checked_thresholds),
    }
    state = {
        "coefficient": coefficient,
        "intercept": intercept,
        "thresholds": checked_thresholds,
        "model_kinds": model_kinds,
        "model_classes": model_classes,
        "test_probabilities": probabilities,
        "regularization_c": float(regularization_c),
        "class_weight": class_weight,
    }
    return predictions, audit, state


def _fit_predict_arm(
    x_train: Any, y_train: Any, x_test: Any
) -> tuple[Any, dict[str, Any]]:
    predictions, audit, _ = _fit_predict_arm_with_state(
        x_train, y_train, x_test
    )
    return predictions, audit


def _threshold_objective(
    y_true: Any,
    probabilities: Any,
    thresholds: Any,
    groups: tuple[str, ...],
) -> tuple[dict[str, float], Any]:
    dependencies = _require_dependencies()
    np = dependencies["np"]
    predictions = (
        np.asarray(probabilities, dtype=np.float64)
        >= np.asarray(thresholds, dtype=np.float64)[None, :]
    ).astype(np.uint8)
    return _metric_values(y_true, predictions, groups), predictions


def _selection_metric_key(metrics: Mapping[str, float]) -> tuple[float, ...]:
    """Lexicographic train-OOF objective; never reads held-out test outcomes."""

    return (
        round(float(metrics["micro_f1"]), 12),
        round(float(metrics["hamming_accuracy"]), 12),
        round(float(metrics["macro_f1"]), 12),
    )


def _select_train_oof_thresholds(
    y_train: Any,
    probabilities: Any,
    groups: tuple[str, ...],
) -> tuple[Any, dict[str, Any]]:
    """Select deterministic per-label thresholds on pooled lesson-grouped OOF."""

    dependencies = _require_dependencies()
    np = dependencies["np"]
    checked_probabilities = np.asarray(probabilities, dtype=np.float64)
    if (
        checked_probabilities.shape != np.asarray(y_train).shape
        or not np.isfinite(checked_probabilities).all()
        or np.any(checked_probabilities < 0.0)
        or np.any(checked_probabilities > 1.0)
    ):
        raise TeachObsMultimodalBenchmarkError(
            "invalid grouped-OOF probability matrix"
        )
    thresholds = np.full(
        checked_probabilities.shape[1], _DECISION_THRESHOLD, dtype=np.float64
    )
    baseline_metrics, _ = _threshold_objective(
        y_train, checked_probabilities, thresholds, groups
    )
    completed_passes = 0
    changed_by_pass: list[int] = []
    for _ in range(_ADVANCED_THRESHOLD_PASSES):
        changed = 0
        for label_index in range(thresholds.size):
            current_threshold = float(thresholds[label_index])
            best_threshold = current_threshold
            best_metrics, _ = _threshold_objective(
                y_train, checked_probabilities, thresholds, groups
            )
            best_key = (
                *_selection_metric_key(best_metrics),
                -abs(current_threshold - _DECISION_THRESHOLD),
                -current_threshold,
            )
            for candidate_threshold in _ADVANCED_THRESHOLDS:
                proposed = thresholds.copy()
                proposed[label_index] = float(candidate_threshold)
                metrics, _ = _threshold_objective(
                    y_train, checked_probabilities, proposed, groups
                )
                key = (
                    *_selection_metric_key(metrics),
                    -abs(float(candidate_threshold) - _DECISION_THRESHOLD),
                    -float(candidate_threshold),
                )
                if key > best_key:
                    best_threshold = float(candidate_threshold)
                    best_metrics = metrics
                    best_key = key
            if best_threshold != current_threshold:
                thresholds[label_index] = best_threshold
                changed += 1
        completed_passes += 1
        changed_by_pass.append(changed)
        if changed == 0:
            break
    selected_metrics, predictions = _threshold_objective(
        y_train, checked_probabilities, thresholds, groups
    )
    if _selection_metric_key(selected_metrics) < _selection_metric_key(
        baseline_metrics
    ):
        raise TeachObsMultimodalBenchmarkError(
            "grouped-OOF threshold selection regressed its fixed baseline"
        )
    return thresholds, {
        "threshold_candidate_grid": list(_ADVANCED_THRESHOLDS),
        "initial_threshold": _DECISION_THRESHOLD,
        "coordinate_order": "fixed_official_label_order",
        "maximum_coordinate_passes": _ADVANCED_THRESHOLD_PASSES,
        "completed_coordinate_passes": completed_passes,
        "threshold_changes_by_pass": changed_by_pass,
        "selection_objective": [
            "pooled_oof_micro_f1",
            "pooled_oof_hamming_accuracy_tiebreak",
            "pooled_oof_macro_f1_tiebreak",
            "closest_to_0.5_tiebreak",
            "lower_threshold_final_tiebreak",
        ],
        "fixed_0_5_metrics": {
            key: _rounded(value) for key, value in baseline_metrics.items()
        },
        "selected_metrics": {
            key: _rounded(value) for key, value in selected_metrics.items()
        },
        "selected_thresholds": [float(value) for value in thresholds],
        "selected_thresholds_sha256": _array_sha256(thresholds),
        "oof_prediction_matrix_sha256": _array_sha256(predictions),
    }


def _fold_preprocessed_arm_matrices(
    dataset: TeachObsTextDataset,
    features: TeachObsMultimodalFeatures,
    fit_indices: Any,
    validation_indices: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Refit every learned transform inside one training-lesson OOF fold."""

    dependencies = _require_dependencies()
    np = dependencies["np"]
    fit = np.asarray(fit_indices, dtype=np.int64)
    validation = np.asarray(validation_indices, dtype=np.int64)

    def selected(values: Sequence[str], indices: Any) -> list[str]:
        return [values[int(index)] for index in indices]

    train_transcripts = tuple(scene.transcript for scene in dataset.train_scenes)
    (
        fit_text,
        validation_text,
        transcript_audit,
        _,
    ) = _fit_train_only_tfidf(
        selected(train_transcripts, fit),
        selected(train_transcripts, validation),
        analyzer="char_wb",
        ngram_range=_TFIDF_NGRAM_RANGE,
        max_features=_TFIDF_MAX_FEATURES,
    )
    (
        fit_ocr_char,
        validation_ocr_char,
        ocr_char_audit,
        _,
    ) = _fit_train_only_tfidf(
        selected(features.train_ocr_text, fit),
        selected(features.train_ocr_text, validation),
        analyzer="char_wb",
        ngram_range=_OCR_CHAR_NGRAM_RANGE,
        max_features=_OCR_CHAR_MAX_FEATURES,
    )
    (
        fit_ocr_word,
        validation_ocr_word,
        ocr_word_audit,
        _,
    ) = _fit_train_only_tfidf(
        selected(features.train_ocr_text, fit),
        selected(features.train_ocr_text, validation),
        analyzer="word",
        ngram_range=_OCR_WORD_NGRAM_RANGE,
        max_features=_OCR_WORD_MAX_FEATURES,
    )

    numeric: dict[str, tuple[Any, Any, int]] = {}
    for block, values in (
        ("audio_numeric", features.train_audio),
        ("visual_numeric", features.train_visual_numeric),
        ("clip_embedding", features.train_clip),
    ):
        scaler = dependencies["StandardScaler"]()
        fit_values = scaler.fit_transform(np.asarray(values)[fit])
        validation_values = scaler.transform(np.asarray(values)[validation])
        if not np.isfinite(fit_values).all() or not np.isfinite(
            validation_values
        ).all():
            raise TeachObsMultimodalBenchmarkError(
                f"grouped-OOF {block} scaler produced non-finite values"
            )
        numeric[block] = (
            fit_values,
            validation_values,
            int(scaler.n_samples_seen_),
        )
    fit_audio, validation_audio, _ = numeric["audio_numeric"]
    fit_visual_numeric, validation_visual_numeric, _ = numeric[
        "visual_numeric"
    ]
    fit_clip, validation_clip, _ = numeric["clip_embedding"]
    hstack = dependencies["hstack"]
    fit_visual = hstack(
        (fit_ocr_char, fit_ocr_word, fit_visual_numeric, fit_clip),
        format="csr",
    )
    validation_visual = hstack(
        (
            validation_ocr_char,
            validation_ocr_word,
            validation_visual_numeric,
            validation_clip,
        ),
        format="csr",
    )
    fit_matrices = {
        "transcript_only": fit_text,
        "transcript_audio": hstack((fit_text, fit_audio), format="csr"),
        "transcript_visual": hstack((fit_text, fit_visual), format="csr"),
        "full": hstack((fit_text, fit_audio, fit_visual), format="csr"),
    }
    validation_matrices = {
        "transcript_only": validation_text,
        "transcript_audio": hstack(
            (validation_text, validation_audio), format="csr"
        ),
        "transcript_visual": hstack(
            (validation_text, validation_visual), format="csr"
        ),
        "full": hstack(
            (validation_text, validation_audio, validation_visual),
            format="csr",
        ),
    }
    audit = {
        "fit_scene_count": int(fit.size),
        "validation_scene_count": int(validation.size),
        "transcript_tfidf_fit_sample_count": int(
            transcript_audit["fit_sample_count"]
        ),
        "ocr_char_tfidf_fit_sample_count": int(
            ocr_char_audit["fit_sample_count"]
        ),
        "ocr_word_tfidf_fit_sample_count": int(
            ocr_word_audit["fit_sample_count"]
        ),
        "audio_scaler_fit_sample_count": numeric["audio_numeric"][2],
        "visual_numeric_scaler_fit_sample_count": numeric["visual_numeric"][2],
        "clip_scaler_fit_sample_count": numeric["clip_embedding"][2],
        "transcript_vocabulary_sha256": transcript_audit[
            "vocabulary_sha256"
        ],
        "ocr_char_vocabulary_sha256": ocr_char_audit["vocabulary_sha256"],
        "ocr_word_vocabulary_sha256": ocr_word_audit["vocabulary_sha256"],
    }
    return fit_matrices, validation_matrices, audit


def _train_lesson_grouped_oof_selection(
    dataset: TeachObsTextDataset,
    features: TeachObsMultimodalFeatures,
    y_train: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Select regularization and thresholds using only nested train-lesson OOF."""

    dependencies = _require_dependencies()
    np = dependencies["np"]
    groups = np.asarray(
        [scene.lesson_id for scene in dataset.train_scenes], dtype=str
    )
    unique_groups = tuple(dict.fromkeys(groups.tolist()))
    if len(unique_groups) < _ADVANCED_GROUP_FOLDS:
        raise TeachObsMultimodalBenchmarkError(
            "too few training lessons for fixed grouped-OOF selection"
        )
    splitter = dependencies["GroupKFold"](n_splits=_ADVANCED_GROUP_FOLDS)
    candidate_probabilities = {
        arm: {
            str(candidate["candidate_id"]): np.full(
                y_train.shape, np.nan, dtype=np.float64
            )
            for candidate in _ADVANCED_CLASSIFIER_CANDIDATES
        }
        for arm in _ARMS
    }
    fold_rows: list[dict[str, Any]] = []
    seen_validation = np.zeros(y_train.shape[0], dtype=np.uint8)
    sample_indices = np.arange(y_train.shape[0], dtype=np.int64)
    for fold_index, (fit_indices, validation_indices) in enumerate(
        splitter.split(sample_indices, y_train, groups=groups),
        start=1,
    ):
        fit_lessons = tuple(dict.fromkeys(groups[fit_indices].tolist()))
        validation_lessons = tuple(
            dict.fromkeys(groups[validation_indices].tolist())
        )
        if set(fit_lessons) & set(validation_lessons):
            raise TeachObsMultimodalBenchmarkError(
                "training lesson leaked across grouped-OOF fold"
            )
        seen_validation[validation_indices] += 1
        fit_matrices, validation_matrices, preprocessing_audit = (
            _fold_preprocessed_arm_matrices(
                dataset, features, fit_indices, validation_indices
            )
        )
        candidate_rows: dict[str, dict[str, Any]] = {}
        for candidate in _ADVANCED_CLASSIFIER_CANDIDATES:
            candidate_id = str(candidate["candidate_id"])
            arm_rows: dict[str, Any] = {}
            for arm in _ARMS:
                _, model_audit, fitted = _fit_predict_arm_with_state(
                    fit_matrices[arm],
                    y_train[fit_indices],
                    validation_matrices[arm],
                    regularization_c=float(candidate["regularization_c"]),
                    class_weight=str(candidate["class_weight"]),
                )
                candidate_probabilities[arm][candidate_id][
                    validation_indices
                ] = fitted["test_probabilities"]
                arm_rows[arm] = {
                    **model_audit,
                    "validation_probability_sha256": _array_sha256(
                        fitted["test_probabilities"]
                    ),
                }
            candidate_rows[candidate_id] = arm_rows
        fold_rows.append(
            {
                "fold_index": fold_index,
                "fit_lesson_ids": list(fit_lessons),
                "validation_lesson_ids": list(validation_lessons),
                "fit_sample_order_sha256": _canonical_sha256(
                    [
                        (
                            f"{dataset.train_scenes[int(index)].lesson_id}:"
                            f"{dataset.train_scenes[int(index)].scene_no}"
                        )
                        for index in fit_indices
                    ]
                ),
                "validation_sample_order_sha256": _canonical_sha256(
                    [
                        (
                            f"{dataset.train_scenes[int(index)].lesson_id}:"
                            f"{dataset.train_scenes[int(index)].scene_no}"
                        )
                        for index in validation_indices
                    ]
                ),
                "preprocessing": preprocessing_audit,
                "classifier_candidates": candidate_rows,
            }
        )
    if not np.all(seen_validation == 1):
        raise TeachObsMultimodalBenchmarkError(
            "training grouped-OOF predictions do not cover each scene exactly once"
        )

    selected: dict[str, dict[str, Any]] = {}
    arm_audits: dict[str, Any] = {}
    for arm in _ARMS:
        candidate_reports: list[dict[str, Any]] = []
        best_key: tuple[float, ...] | None = None
        best: dict[str, Any] | None = None
        for candidate_order, candidate in enumerate(
            _ADVANCED_CLASSIFIER_CANDIDATES
        ):
            candidate_id = str(candidate["candidate_id"])
            probabilities = candidate_probabilities[arm][candidate_id]
            if not np.isfinite(probabilities).all():
                raise TeachObsMultimodalBenchmarkError(
                    "grouped-OOF candidate has missing probabilities"
                )
            thresholds, threshold_audit = _select_train_oof_thresholds(
                y_train, probabilities, dataset.code_groups
            )
            selected_metrics = threshold_audit["selected_metrics"]
            key = (
                *_selection_metric_key(selected_metrics),
                -abs(float(candidate["regularization_c"]) - 1.0),
                -float(candidate_order),
            )
            report = {
                "candidate_id": candidate_id,
                "regularization_c": float(candidate["regularization_c"]),
                "class_weight": candidate["class_weight"],
                "oof_probability_matrix_sha256": _array_sha256(
                    probabilities
                ),
                "threshold_selection": threshold_audit,
                "selected": False,
            }
            candidate_reports.append(report)
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    "candidate": candidate,
                    "thresholds": thresholds,
                    "report": report,
                }
        if best is None:  # pragma: no cover - fixed candidate registry
            raise TeachObsMultimodalBenchmarkError(
                "grouped-OOF selection has no classifier candidate"
            )
        best["report"]["selected"] = True
        selected_candidate = best["candidate"]
        selected[arm] = {
            "regularization_c": float(selected_candidate["regularization_c"]),
            "class_weight": selected_candidate["class_weight"],
            "thresholds": best["thresholds"],
        }
        arm_audits[arm] = {
            "selected_candidate_id": selected_candidate["candidate_id"],
            "selected_regularization_c": float(
                selected_candidate["regularization_c"]
            ),
            "selected_class_weight": selected_candidate["class_weight"],
            "selected_thresholds": [
                float(value) for value in best["thresholds"]
            ],
            "selected_thresholds_sha256": _array_sha256(best["thresholds"]),
            "selected_oof_metrics": best["report"]["threshold_selection"][
                "selected_metrics"
            ],
            "candidate_reports": candidate_reports,
        }
    audit = {
        "schema": _ADVANCED_SELECTION_SCHEMA,
        "exploratory_revision_after_initial_public_test_evaluation": True,
        "confirmatory_selection": False,
        "fit_scope": "official_23_training_lessons_only",
        "outer_test_labels_used_for_selection": False,
        "outer_test_features_used_for_selection": False,
        "preprocessing_refit_inside_every_group_fold": True,
        "fixed_raw_audio_visual_clip_features_reused": True,
        "splitter": "GroupKFold",
        "group_field": "lesson_id",
        "fold_count": _ADVANCED_GROUP_FOLDS,
        "classifier_random_state": _RANDOM_STATE,
        "candidate_registry": [
            dict(candidate) for candidate in _ADVANCED_CLASSIFIER_CANDIDATES
        ],
        "threshold_candidate_grid": list(_ADVANCED_THRESHOLDS),
        "selection_criterion": (
            "pooled OOF Micro-F1; Hamming accuracy then Macro-F1; "
            "C closest to 1 then registry order"
        ),
        "each_training_scene_has_exactly_one_oof_prediction": True,
        "folds": fold_rows,
        "arms": arm_audits,
    }
    return selected, audit


def _arm_private_result(
    *,
    name: str,
    predictions: Any,
    y_train: Any,
    y_test: Any,
    dataset: TeachObsTextDataset,
    model_audit: dict[str, Any],
    feature_count: int,
    feature_blocks: Mapping[str, int],
) -> dict[str, Any]:
    dependencies = _require_dependencies()
    np = dependencies["np"]
    per_label: list[dict[str, Any]] = []
    for index, (code, group) in enumerate(
        zip(dataset.code_names, dataset.code_groups, strict=True)
    ):
        truth = y_test[:, index]
        predicted = predictions[:, index]
        tp = int(np.logical_and(truth == 1, predicted == 1).sum())
        fp = int(np.logical_and(truth == 0, predicted == 1).sum())
        fn = int(np.logical_and(truth == 1, predicted == 0).sum())
        per_label.append(
            {
                "code": code,
                "group": group,
                "train_positive_count": int(y_train[:, index].sum()),
                "test_positive_count": int(truth.sum()),
                "predicted_positive_count": int(predicted.sum()),
                "precision": _rounded(tp / (tp + fp) if tp + fp else 0.0),
                "recall": _rounded(tp / (tp + fn) if tp + fn else 0.0),
                "f1": _rounded(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0),
            }
        )
    lesson_indices: dict[str, list[int]] = {}
    for index, scene in enumerate(dataset.test_scenes):
        lesson_indices.setdefault(scene.lesson_id, []).append(index)
    per_lesson = [
        {
            "lesson_id": lesson_id,
            "scene_count": len(indices),
            "metrics": _metric_summary(
                y_test[indices], predictions[indices], dataset.code_groups
            ),
        }
        for lesson_id, indices in lesson_indices.items()
    ]
    checked_blocks = {str(key): int(value) for key, value in feature_blocks.items()}
    if any(value < 0 for value in checked_blocks.values()) or sum(
        checked_blocks.values()
    ) != int(feature_count):
        raise TeachObsMultimodalBenchmarkError(
            f"feature-block dimensions do not sum to arm {name}"
        )
    return {
        "arm": name,
        "metrics": _metric_summary(y_test, predictions, dataset.code_groups),
        "feature_audit": {
            "feature_count": int(feature_count),
            "feature_blocks": checked_blocks,
            **model_audit,
        },
        "prediction_matrix_sha256": _array_sha256(predictions),
        "prediction_shape": list(predictions.shape),
        "per_label": per_label,
        "per_lesson": per_lesson,
    }


def _paired_cluster_bootstrap(
    y_test: Any,
    predictions: dict[str, Any],
    dataset: TeachObsTextDataset,
    profile_resolution: ResolvedTeachObsBenchmarkProfile,
) -> dict[str, Any]:
    dependencies = _require_dependencies()
    np = dependencies["np"]
    lesson_indices: dict[str, Any] = {}
    for index, scene in enumerate(dataset.test_scenes):
        lesson_indices.setdefault(scene.lesson_id, []).append(index)
    if tuple(lesson_indices) != profile_resolution.selected_test_lesson_ids:
        raise TeachObsMultimodalBenchmarkError(
            "paired bootstrap clusters differ from the selected test profile"
        )
    clusters = [np.asarray(indices, dtype=np.int64) for indices in lesson_indices.values()]
    point_metrics = {
        arm: _metric_values(y_test, value, dataset.code_groups)
        for arm, value in predictions.items()
    }
    distributions = {
        name: {metric: [] for metric in _METRICS}
        for name, _, _ in _COMPARISONS
    }
    generator = np.random.default_rng(_BOOTSTRAP_SEED)
    for _ in range(_BOOTSTRAP_REPLICATES):
        sampled = generator.integers(0, len(clusters), size=len(clusters))
        indices = np.concatenate([clusters[int(index)] for index in sampled])
        truth = y_test[indices]
        replicate_metrics = {
            arm: _metric_values(truth, value[indices], dataset.code_groups)
            for arm, value in predictions.items()
        }
        for name, numerator, denominator in _COMPARISONS:
            for metric in _METRICS:
                distributions[name][metric].append(
                    replicate_metrics[numerator][metric]
                    - replicate_metrics[denominator][metric]
                )
    comparisons: dict[str, Any] = {}
    for name, numerator, denominator in _COMPARISONS:
        comparisons[name] = {}
        for metric in _METRICS:
            interval = np.quantile(distributions[name][metric], [0.025, 0.975])
            comparisons[name][metric] = {
                "point_delta": _rounded(
                    point_metrics[numerator][metric]
                    - point_metrics[denominator][metric]
                ),
                "percentile_95_ci": [_rounded(value) for value in interval],
            }
    return {
        "bootstrap_unit": "profile_selected_test_lesson_cluster",
        "cluster_count": len(clusters),
        "replicates": _BOOTSTRAP_REPLICATES,
        "seed": _BOOTSTRAP_SEED,
        "paired_resampling": True,
        "interval_method": "percentile_95",
        "comparisons": comparisons,
    }


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


def _load_materialized_transcript_dataset(
    profile_resolution: ResolvedTeachObsBenchmarkProfile,
    manifest_path: str | Path,
) -> tuple[TeachObsTextDataset, dict[str, Any]]:
    """Replace released scene text with one exact audited materialization."""

    try:
        validated = validate_teachobs_transcript_materialization(
            manifest_path,
            expected_profile=profile_resolution.spec.profile_id,
        )
    except TeachObsTranscriptMaterializationError as exc:
        raise TeachObsMultimodalBenchmarkError(
            f"TeachObs transcript materialization failed validation: {exc}"
        ) from exc
    manifest = validated["manifest"]
    rows = validated["rows"]
    text_by_sample_id = validated["text_by_sample_id"]
    expected_sample_ids = (
        profile_resolution.selected_train_sample_ids
        + profile_resolution.selected_test_sample_ids
    )
    observed_sample_ids = tuple(row["sample_id"] for row in rows)
    if (
        observed_sample_ids != expected_sample_ids
        or set(text_by_sample_id) != set(expected_sample_ids)
        or len(text_by_sample_id) != len(expected_sample_ids)
    ):
        raise TeachObsMultimodalBenchmarkError(
            "TeachObs transcript materialization sample order differs from "
            "the selected benchmark profile"
        )

    source_dataset = profile_resolution.dataset

    def replace_scenes(scenes: Sequence[TeachObsScene]) -> tuple[TeachObsScene, ...]:
        materialized: list[TeachObsScene] = []
        for scene in scenes:
            sample_id = f"{scene.lesson_id}:{scene.scene_no}"
            # Deliberately use direct indexing.  Empty materialized scenes must
            # remain empty and may never fall back to released scene text.
            text = text_by_sample_id[sample_id]
            if not isinstance(text, str):
                raise TeachObsMultimodalBenchmarkError(
                    "TeachObs materialized transcript text is not a string"
                )
            materialized.append(
                TeachObsScene(
                    lesson_id=scene.lesson_id,
                    scene_no=scene.scene_no,
                    transcript=text,
                    labels=scene.labels,
                )
            )
        return tuple(materialized)

    train_scenes = replace_scenes(source_dataset.train_scenes)
    test_scenes = replace_scenes(source_dataset.test_scenes)
    train_texts = [scene.transcript for scene in train_scenes]
    test_texts = [scene.transcript for scene in test_scenes]
    policy = manifest["policy"]
    claims = manifest["claims"]
    if (
        policy.get("released_transcript_fallback_used") is not False
        or policy.get("labels_read_or_used") is not False
        or claims.get("released_transcript_fallback_used") is not False
        or claims.get("labels_read_or_used") is not False
        or claims.get("empty_scenes_filled_from_released_transcript") is not False
    ):
        raise TeachObsMultimodalBenchmarkError(
            "TeachObs transcript materialization weakens the no-fallback boundary"
        )
    audit = {
        "schema": manifest["schema"],
        "profile_id": manifest["profile_id"],
        "manifest_file_sha256": _file_sha256(validated["manifest_path"]),
        "manifest_sha256": manifest["manifest_sha256"],
        "materialization_fingerprint_sha256": manifest[
            "materialization_fingerprint_sha256"
        ],
        "ordered_sample_id_sha256": manifest["ordered_sample_id_sha256"],
        "ordered_scene_text_sha256": manifest["ordered_scene_text_sha256"],
        "repository_binding_sha256": _canonical_sha256(
            manifest["repository_binding"]
        ),
        "input_hashes_sha256": _canonical_sha256(manifest["input_hashes"]),
        "selected_train_transcript_order_sha256": _canonical_sha256(
            train_texts
        ),
        "selected_test_transcript_order_sha256": _canonical_sha256(test_texts),
        "lesson_count": int(manifest["aggregate"]["lesson_count"]),
        "scene_count": int(manifest["aggregate"]["scene_count"]),
        "nonempty_scene_count": int(
            manifest["aggregate"]["nonempty_scene_count"]
        ),
        "empty_scene_count": int(manifest["aggregate"]["empty_scene_count"]),
        "source_tier_lesson_counts": dict(
            manifest["aggregate"]["source_tier_lesson_counts"]
        ),
        "source_tier_scene_counts": dict(
            manifest["aggregate"]["source_tier_scene_counts"]
        ),
        "all_selected_scene_transcripts_materialized": True,
        "sample_order_matches_benchmark_profile": True,
        "released_transcript_fallback_used": False,
        "labels_read_or_used": False,
        "empty_scenes_filled_from_released_transcript": False,
        "released_repository_transcripts_used_for_model_input": False,
    }
    dataset = TeachObsTextDataset(
        code_names=source_dataset.code_names,
        code_groups=source_dataset.code_groups,
        train_scenes=train_scenes,
        test_scenes=test_scenes,
        dataset_sha256=source_dataset.dataset_sha256,
        source_file_count=source_dataset.source_file_count,
    )
    return dataset, audit


def run_teachobs_multimodal_benchmark(
    repository_root: str | Path,
    feature_manifest_path: str | Path,
    *,
    transcript_materialization_manifest_path: str | Path,
    frozen_model_output: str | Path | None = None,
    profile: str = DEFAULT_BENCHMARK_PROFILE,
) -> dict[str, Any]:
    """Run transcript, +audio, +visual, and full arms without test tuning."""

    dependencies = _require_dependencies()
    np = dependencies["np"]
    full_dataset = load_teachobs_text_dataset(repository_root)
    profile_resolution = resolve_teachobs_benchmark_profile(full_dataset, profile)
    dataset, transcript_audit = _load_materialized_transcript_dataset(
        profile_resolution,
        transcript_materialization_manifest_path,
    )
    source_identity_audit = _load_source_identity_audit(
        repository_root, profile_resolution
    )
    features = load_teachobs_multimodal_features(
        repository_root,
        feature_manifest_path,
        dataset=full_dataset,
        profile=profile_resolution.spec.profile_id,
    )
    if (
        features.benchmark_profile != profile_resolution.spec.profile_id
        or features.dataset_profile_fingerprint
        != profile_resolution.dataset_profile_fingerprint
        or features.selected_train_sample_ids
        != profile_resolution.selected_train_sample_ids
        or features.selected_test_sample_ids
        != profile_resolution.selected_test_sample_ids
    ):
        raise TeachObsMultimodalBenchmarkError(
            "TeachObs feature/profile resolution changed before fitting"
        )
    benchmark_input_fingerprint = _canonical_sha256(
        _benchmark_input_contract(
            dataset_profile_fingerprint=(
                profile_resolution.dataset_profile_fingerprint
            ),
            feature_manifest_sha256=features.feature_manifest_sha256,
            transcript_materialization=transcript_audit,
        )
    )
    transcript_audit["benchmark_input_fingerprint"] = (
        benchmark_input_fingerprint
    )
    y_train = np.asarray([scene.labels for scene in dataset.train_scenes], dtype=np.uint8)
    y_test = np.asarray([scene.labels for scene in dataset.test_scenes], dtype=np.uint8)
    (
        train_text,
        test_text,
        transcript_tfidf_audit,
        transcript_tfidf_state,
    ) = _fit_train_only_tfidf(
        [scene.transcript for scene in dataset.train_scenes],
        [scene.transcript for scene in dataset.test_scenes],
        analyzer="char_wb",
        ngram_range=_TFIDF_NGRAM_RANGE,
        max_features=_TFIDF_MAX_FEATURES,
    )
    (
        train_ocr_char,
        test_ocr_char,
        ocr_char_tfidf_audit,
        ocr_char_tfidf_state,
    ) = _fit_train_only_tfidf(
        features.train_ocr_text,
        features.test_ocr_text,
        analyzer="char_wb",
        ngram_range=_OCR_CHAR_NGRAM_RANGE,
        max_features=_OCR_CHAR_MAX_FEATURES,
    )
    (
        train_ocr_word,
        test_ocr_word,
        ocr_word_tfidf_audit,
        ocr_word_tfidf_state,
    ) = _fit_train_only_tfidf(
        features.train_ocr_text,
        features.test_ocr_text,
        analyzer="word",
        ngram_range=_OCR_WORD_NGRAM_RANGE,
        max_features=_OCR_WORD_MAX_FEATURES,
    )

    audio_scaler = dependencies["StandardScaler"]()
    train_audio = audio_scaler.fit_transform(features.train_audio)
    test_audio = audio_scaler.transform(features.test_audio)
    visual_numeric_scaler = dependencies["StandardScaler"]()
    train_visual_numeric = visual_numeric_scaler.fit_transform(
        features.train_visual_numeric
    )
    test_visual_numeric = visual_numeric_scaler.transform(
        features.test_visual_numeric
    )
    clip_scaler = dependencies["StandardScaler"]()
    train_clip = clip_scaler.fit_transform(features.train_clip)
    test_clip = clip_scaler.transform(features.test_clip)
    if not all(
        np.isfinite(value).all()
        for value in (
            train_audio,
            test_audio,
            train_visual_numeric,
            test_visual_numeric,
            train_clip,
            test_clip,
        )
    ):
        raise TeachObsMultimodalBenchmarkError(
            "standardized TeachObs features contain non-finite values"
        )
    hstack = dependencies["hstack"]
    train_visual_blocks = (
        train_ocr_char,
        train_ocr_word,
        train_visual_numeric,
        train_clip,
    )
    test_visual_blocks = (
        test_ocr_char,
        test_ocr_word,
        test_visual_numeric,
        test_clip,
    )
    train_visual = hstack(train_visual_blocks, format="csr")
    test_visual = hstack(test_visual_blocks, format="csr")
    train_matrices = {
        "transcript_only": train_text,
        "transcript_audio": hstack((train_text, train_audio), format="csr"),
        "transcript_visual": hstack((train_text, train_visual), format="csr"),
        "full": hstack((train_text, train_audio, train_visual), format="csr"),
    }
    test_matrices = {
        "transcript_only": test_text,
        "transcript_audio": hstack((test_text, test_audio), format="csr"),
        "transcript_visual": hstack((test_text, test_visual), format="csr"),
        "full": hstack((test_text, test_audio, test_visual), format="csr"),
    }
    feature_block_dimensions = {
        "transcript_tfidf": int(train_text.shape[1]),
        "audio_numeric": int(train_audio.shape[1]),
        "ocr_char_tfidf": int(train_ocr_char.shape[1]),
        "ocr_word_tfidf": int(train_ocr_word.shape[1]),
        "visual_numeric": int(train_visual_numeric.shape[1]),
        "clip_embedding": int(train_clip.shape[1]),
    }
    arm_feature_blocks = {
        "transcript_only": {
            "transcript_tfidf": feature_block_dimensions["transcript_tfidf"]
        },
        "transcript_audio": {
            key: feature_block_dimensions[key]
            for key in ("transcript_tfidf", "audio_numeric")
        },
        "transcript_visual": {
            key: feature_block_dimensions[key]
            for key in (
                "transcript_tfidf",
                "ocr_char_tfidf",
                "ocr_word_tfidf",
                "visual_numeric",
                "clip_embedding",
            )
        },
        "full": dict(feature_block_dimensions),
    }
    selected_arm_configurations, train_oof_selection_audit = (
        _train_lesson_grouped_oof_selection(dataset, features, y_train)
    )
    predictions: dict[str, Any] = {}
    arms: dict[str, Any] = {}
    fitted_arm_states: dict[str, dict[str, Any]] = {}
    for arm in _ARMS:
        selected_configuration = selected_arm_configurations[arm]
        prediction, model_audit, fitted_state = _fit_predict_arm_with_state(
            train_matrices[arm],
            y_train,
            test_matrices[arm],
            regularization_c=selected_configuration["regularization_c"],
            class_weight=selected_configuration["class_weight"],
            thresholds=selected_configuration["thresholds"],
        )
        fitted_state["selection_provenance_sha256"] = _canonical_sha256(
            train_oof_selection_audit["arms"][arm]
        )
        predictions[arm] = prediction
        fitted_arm_states[arm] = fitted_state
        arms[arm] = _arm_private_result(
            name=arm,
            predictions=prediction,
            y_train=y_train,
            y_test=y_test,
            dataset=dataset,
            model_audit=model_audit,
            feature_count=train_matrices[arm].shape[1],
            feature_blocks=arm_feature_blocks[arm],
        )

    result = {
        "schema_version": "1.0",
        "benchmark_kind": BENCHMARK_KIND,
        "dataset_audit": {
            "dataset_name": "TeachObs",
            "benchmark_profile": profile_resolution.spec.profile_id,
            "profile_source_alignment": profile_resolution.spec.source_alignment,
            "published_six_lesson_intersection": (
                profile_resolution.spec.published_six_lesson_intersection
            ),
            "lesson_count": len(profile_resolution.selected_train_lesson_ids)
            + len(profile_resolution.selected_test_lesson_ids),
            "train_lesson_count": 23,
            "test_lesson_count": len(
                profile_resolution.selected_test_lesson_ids
            ),
            "scene_count": len(dataset.train_scenes) + len(dataset.test_scenes),
            "train_scene_count": len(dataset.train_scenes),
            "test_scene_count": len(dataset.test_scenes),
            "code_count": len(dataset.code_names),
            "visual_code_count": dataset.code_groups.count("visual"),
            "nonvisual_code_count": dataset.code_groups.count("nonvisual"),
            "dataset_sha256": dataset.dataset_sha256,
            "dataset_profile_fingerprint": (
                profile_resolution.dataset_profile_fingerprint
            ),
            "benchmark_input_fingerprint": benchmark_input_fingerprint,
            "selected_train_lesson_ids": list(
                profile_resolution.selected_train_lesson_ids
            ),
            "selected_test_lesson_ids": list(
                profile_resolution.selected_test_lesson_ids
            ),
            "excluded_official_test_lesson_ids": list(
                profile_resolution.spec.excluded_official_test_lesson_ids
            ),
            "selected_train_sample_order_sha256": _canonical_sha256(
                list(profile_resolution.selected_train_sample_ids)
            ),
            "selected_test_sample_order_sha256": _canonical_sha256(
                list(profile_resolution.selected_test_sample_ids)
            ),
        },
        "private_transcript_audit": transcript_audit,
        "private_feature_audit": {
            "feature_manifest_sha256": features.feature_manifest_sha256,
            "source_file_count": features.source_file_count,
            "audio_feature_count": len(features.audio_feature_names),
            "visual_feature_count": int(train_visual.shape[1]),
            "visual_raw_numeric_feature_count": features.visual_numeric_dimension,
            "ocr_char_tfidf_feature_count": int(train_ocr_char.shape[1]),
            "ocr_word_tfidf_feature_count": int(train_ocr_word.shape[1]),
            "clip_embedding_feature_count": features.clip_dimension,
            "feature_block_dimensions": feature_block_dimensions,
            "visual_feature_layout": features.visual_feature_layout,
            "visual_evidence_set_sha256": features.visual_evidence_set_sha256,
            "visual_evidence_configuration_set_sha256": (
                features.visual_evidence_configuration_set_sha256
            ),
            "visual_numeric_train_matrix_sha256": _array_sha256(
                features.train_visual_numeric
            ),
            "visual_numeric_test_matrix_sha256": _array_sha256(
                features.test_visual_numeric
            ),
            "clip_source_revision": features.clip_source_revision,
            "clip_weight_manifest_sha256": features.clip_weight_manifest_sha256,
            "ocr_status_counts": features.ocr_status_counts,
            "ocr_completed_scene_count": features.ocr_completed_scene_count,
            "ocr_nonempty_scene_count": features.ocr_nonempty_scene_count,
            "ocr_completed_scene_fraction": _rounded(
                features.ocr_completed_scene_count
                / (len(dataset.train_scenes) + len(dataset.test_scenes))
            ),
            "ocr_complete_for_every_scene": (
                features.ocr_completed_scene_count
                == len(dataset.train_scenes) + len(dataset.test_scenes)
                and set(features.ocr_status_counts) <= {"completed"}
            ),
            "all_scenes_have_audio_and_visual_features": True,
            "scene_identity_and_hash_binding_verified": True,
            "profile_selected_lessons_have_complete_audio_visual_clip": True,
        },
        "source_identity_audit": source_identity_audit,
        "training_only_model_selection_audit": train_oof_selection_audit,
        "training_only_preprocessing_audit": {
            "tfidf_fit_sample_count": transcript_tfidf_audit["fit_sample_count"],
            "tfidf_feature_count": transcript_tfidf_audit["feature_count"],
            "tfidf_vocabulary_sha256": transcript_tfidf_audit[
                "vocabulary_sha256"
            ],
            "transcript_tfidf": transcript_tfidf_audit,
            "ocr_char_tfidf": ocr_char_tfidf_audit,
            "ocr_word_tfidf": ocr_word_tfidf_audit,
            "audio_scaler_fit_sample_count": int(audio_scaler.n_samples_seen_),
            "audio_scaler_mean_sha256": _array_sha256(audio_scaler.mean_),
            "audio_scaler_scale_sha256": _array_sha256(audio_scaler.scale_),
            "visual_scaler_fit_sample_count": int(
                visual_numeric_scaler.n_samples_seen_
            ),
            "visual_numeric_scaler_fit_sample_count": int(
                visual_numeric_scaler.n_samples_seen_
            ),
            "visual_numeric_scaler_mean_sha256": _array_sha256(
                visual_numeric_scaler.mean_
            ),
            "visual_numeric_scaler_scale_sha256": _array_sha256(
                visual_numeric_scaler.scale_
            ),
            "clip_scaler_fit_sample_count": int(clip_scaler.n_samples_seen_),
            "clip_scaler_mean_sha256": _array_sha256(clip_scaler.mean_),
            "clip_scaler_scale_sha256": _array_sha256(clip_scaler.scale_),
        },
        "protocol": {
            "arms": list(_ARMS),
            "benchmark_profile": profile_resolution.spec.profile_id,
            "profile_source_alignment": profile_resolution.spec.source_alignment,
            "published_six_lesson_intersection": (
                profile_resolution.spec.published_six_lesson_intersection
            ),
            "official_lesson_disjoint_split": True,
            "tfidf_fit_scope": "official_23_training_lessons_only",
            "ocr_tfidf_fit_scope": "official_23_training_lessons_only",
            "numeric_scaler_fit_scope": "official_23_training_lessons_only",
            "classifier_fit_scope": "official_23_training_lessons_only",
            "classifier": "independent_binary_logistic_regression",
            "class_weight": "balanced_in_fixed_candidate_registry",
            "regularization_c_selection": (
                "per_arm_official_training_lesson_grouped_oof"
            ),
            "constant_train_label_handling": "predict_that_constant",
            "decision_threshold_policy": (
                "per_label_official_training_lesson_grouped_oof"
            ),
            "decision_threshold_candidate_grid": list(
                _ADVANCED_THRESHOLDS
            ),
            "thresholds_frozen_before_current_revised_test_scoring": True,
            "public_test_outcomes_previously_observed_before_revision": True,
            "current_revision_is_post_test_exploratory": True,
            "test_labels_used_for_feature_fitting_training_or_tuning": False,
            "test_threshold_or_hyperparameter_tuning_performed": False,
            "grouped_oof_preprocessing_refit_inside_every_fold": True,
            "grouped_oof_fold_count": _ADVANCED_GROUP_FOLDS,
            "exploratory_revision_after_initial_public_test_evaluation": True,
            "excluded_test_labels_used_for_fitting_tuning_or_metrics": False,
            "s4_labels_used_for_fitting_tuning_or_metrics": False,
            "same_test_scenes_used_for_all_arms": True,
            "same_materialized_transcript_scenes_used_for_all_arms": True,
            "released_repository_transcripts_used_for_model_input": False,
            "transcript_materialization_manifest_required": True,
            "benchmark_input_fingerprint": benchmark_input_fingerprint,
            "visual_arm_blocks": [
                "training_only_ocr_char_tfidf",
                "training_only_ocr_word_tfidf",
                "deterministic_image_ocr_transition_numeric_evidence",
                "hash_bound_clip_embedding",
            ],
            "random_state": _RANDOM_STATE,
        },
        "arms": arms,
        "paired_cluster_bootstrap": _paired_cluster_bootstrap(
            y_test, predictions, dataset, profile_resolution
        ),
        "evidence_scope": {
            "benchmark_profile": profile_resolution.spec.profile_id,
            "source_aligned_published_six_lesson_intersection": (
                profile_resolution.spec.published_six_lesson_intersection
            ),
            "official_full_seven_lesson_test_profile": (
                profile_resolution.spec.profile_id
                == FULL_23_TRAIN_7_TEST_PROFILE
            ),
            "released_consensus_gold_evaluated": True,
            "audited_transcript_materialization_used": True,
            "released_repository_transcripts_used_for_model_input": False,
            "transcript_content_accuracy_established": False,
            "transcript_word_error_rate_established": False,
            "public_test_labels_were_accessible_before_implementation": True,
            "provisional_result": True,
            "exploratory_multimodal_gain_estimate": True,
            "confirmatory_multimodal_gain_established": False,
            "external_lockbox_established": False,
            "deployment_accuracy_established": False,
            "learning_effectiveness_established": False,
            "source_value_train_test_overlap_detected": True,
            "source_field_used_as_canonical_site_id": False,
            "site_disjointness_verified": False,
            "teacher_disjointness_verified": False,
            "classroom_disjointness_verified": False,
            "source_family_sensitivity_metric_computed": False,
            "site_held_out_evaluation_performed": False,
            "site_held_out_accuracy_established": False,
            "valid_claim": (
                "provisional exploratory four-arm multimodal comparison on the "
                + (
                    "publicly accessible TeachObs source-aligned published "
                    "six-lesson Track 1 text/frame intersection"
                    if profile_resolution.spec.published_six_lesson_intersection
                    else "publicly accessible TeachObs official 23/7 split"
                )
            ),
            "invalid_claims": [
                "confirmatory multimodal gain",
                "external lockbox accuracy",
                "deployment accuracy",
                "learning effectiveness",
            ],
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": _version("numpy"),
            "scipy": _version("scipy"),
            "scikit_learn": _version("scikit-learn"),
        },
    }
    if frozen_model_output is not None:
        from .teachobs_frozen_model import (
            TeachObsFrozenModelError,
            export_teachobs_frozen_bundle,
            load_teachobs_frozen_bundle,
            predict_teachobs_frozen_bundle,
        )

        if len(features.visual_evidence_configuration_sha256_values) != 1:
            raise TeachObsMultimodalBenchmarkError(
                "frozen deployment export requires one visual extractor configuration"
            )

        benchmark_configuration = {
            "arms": list(_ARMS),
            "benchmark_profile": profile_resolution.spec.profile_id,
            "dataset_profile_fingerprint": (
                profile_resolution.dataset_profile_fingerprint
            ),
            "feature_blocks_by_arm": arm_feature_blocks,
            "decision_threshold_policy": (
                "per_label_official_training_lesson_grouped_oof"
            ),
            "decision_threshold_candidate_grid": list(
                _ADVANCED_THRESHOLDS
            ),
            "per_arm_model_selection": {
                arm: {
                    "regularization_c": float(
                        selected_arm_configurations[arm][
                            "regularization_c"
                        ]
                    ),
                    "class_weight": selected_arm_configurations[arm][
                        "class_weight"
                    ],
                    "thresholds_sha256": _array_sha256(
                        selected_arm_configurations[arm]["thresholds"]
                    ),
                    "selection_provenance_sha256": fitted_arm_states[arm][
                        "selection_provenance_sha256"
                    ],
                }
                for arm in _ARMS
            },
            "tfidf": {
                "transcript": transcript_tfidf_state["configuration"],
                "ocr_char": ocr_char_tfidf_state["configuration"],
                "ocr_word": ocr_word_tfidf_state["configuration"],
            },
            "classifier": {
                "solver": _LOGISTIC_SOLVER,
                "maximum_iterations": _LOGISTIC_MAX_ITER,
                "random_state": _RANDOM_STATE,
                "class_weight": "balanced_in_fixed_candidate_registry",
                "regularization_c_candidates": [
                    float(candidate["regularization_c"])
                    for candidate in _ADVANCED_CLASSIFIER_CANDIDATES
                ],
            },
        }
        training_provenance = {
            **frozen_profile_training_binding(profile_resolution),
            "training_repository_dataset_sha256": dataset.dataset_sha256,
            "feature_manifest_sha256": features.feature_manifest_sha256,
            "transcript_materialization": {
                key: transcript_audit[key]
                for key in (
                    "schema",
                    "profile_id",
                    "manifest_file_sha256",
                    "manifest_sha256",
                    "materialization_fingerprint_sha256",
                    "ordered_sample_id_sha256",
                    "ordered_scene_text_sha256",
                    "repository_binding_sha256",
                    "input_hashes_sha256",
                    "selected_train_transcript_order_sha256",
                    "selected_test_transcript_order_sha256",
                    "released_transcript_fallback_used",
                    "labels_read_or_used",
                )
            },
            "benchmark_input_fingerprint": benchmark_input_fingerprint,
            "visual_evidence_set_sha256": features.visual_evidence_set_sha256,
            "visual_evidence_configuration_set_sha256": (
                features.visual_evidence_configuration_set_sha256
            ),
            "label_order_sha256": _canonical_sha256(list(dataset.code_names)),
            "benchmark_configuration_sha256": _canonical_sha256(
                benchmark_configuration
            ),
            "benchmark_source_sha256": _file_sha256(Path(__file__)),
        }
        software_provenance = {
            "python": sys.version.split()[0],
            "numpy": _version("numpy"),
            "scipy": _version("scipy"),
            "scikit_learn": _version("scikit-learn"),
        }
        try:
            export_receipt = export_teachobs_frozen_bundle(
                frozen_model_output,
                label_names=dataset.code_names,
                label_groups=dataset.code_groups,
                feature_blocks_by_arm=arm_feature_blocks,
                text_states={
                    "transcript_tfidf": transcript_tfidf_state,
                    "ocr_char_tfidf": ocr_char_tfidf_state,
                    "ocr_word_tfidf": ocr_word_tfidf_state,
                },
                numeric_states={
                    "audio_numeric": {
                        "mean": audio_scaler.mean_,
                        "scale": audio_scaler.scale_,
                    },
                    "visual_numeric": {
                        "mean": visual_numeric_scaler.mean_,
                        "scale": visual_numeric_scaler.scale_,
                    },
                    "clip_embedding": {
                        "mean": clip_scaler.mean_,
                        "scale": clip_scaler.scale_,
                    },
                },
                numeric_feature_names={
                    "audio_numeric": features.audio_feature_names,
                    "visual_numeric": features.visual_numeric_feature_names,
                },
                numeric_provenance={
                    "audio_numeric": {
                        "feature_schema_sha256": _canonical_sha256(
                            list(features.audio_feature_names)
                        ),
                        "audio_feature_schema": AUDIO_SCHEMA,
                    },
                    "visual_numeric": {
                        "feature_schema_sha256": _canonical_sha256(
                            list(features.visual_numeric_feature_names)
                        ),
                        "visual_evidence_configuration_set_sha256": (
                            features.visual_evidence_configuration_set_sha256
                        ),
                        "visual_evidence_schema": VISUAL_EVIDENCE_SCHEMA,
                        "configuration_sha256_values": list(
                            features.visual_evidence_configuration_sha256_values
                        ),
                    },
                    "clip_embedding": {
                        "clip_source_revision": features.clip_source_revision,
                        "clip_weight_manifest_sha256": (
                            features.clip_weight_manifest_sha256
                        ),
                    },
                },
                fitted_arm_states=fitted_arm_states,
                training_provenance=training_provenance,
                software_provenance=software_provenance,
            )
            frozen_bundle = load_teachobs_frozen_bundle(
                frozen_model_output,
                expected_bundle_manifest_file_sha256=export_receipt[
                    "bundle_manifest_file_sha256"
                ],
                expected_benchmark_profile=profile_resolution.spec.profile_id,
                expected_dataset_profile_fingerprint=(
                    profile_resolution.dataset_profile_fingerprint
                ),
                expected_transcript_materialization_fingerprint=(
                    transcript_audit["materialization_fingerprint_sha256"]
                ),
                expected_benchmark_input_fingerprint=(
                    benchmark_input_fingerprint
                ),
            )
            frozen_outputs = predict_teachobs_frozen_bundle(
                frozen_bundle,
                transcripts=[scene.transcript for scene in dataset.test_scenes],
                transcript_input_materialization_fingerprint_sha256=(
                    transcript_audit["materialization_fingerprint_sha256"]
                ),
                ocr_texts=features.test_ocr_text,
                audio=features.test_audio,
                visual_numeric=features.test_visual_numeric,
                clip=features.test_clip,
                label_names=dataset.code_names,
                audio_feature_names=features.audio_feature_names,
                visual_numeric_feature_names=features.visual_numeric_feature_names,
                clip_source_revision=features.clip_source_revision,
                clip_weight_manifest_sha256=features.clip_weight_manifest_sha256,
                audio_feature_schema=AUDIO_SCHEMA,
                visual_evidence_schema=VISUAL_EVIDENCE_SCHEMA,
                visual_evidence_configuration_sha256=(
                    features.visual_evidence_configuration_sha256_values[0]
                ),
                sample_ids=list(profile_resolution.selected_test_sample_ids),
                benchmark_profile=profile_resolution.spec.profile_id,
                dataset_profile_fingerprint=(
                    profile_resolution.dataset_profile_fingerprint
                ),
            )
        except TeachObsFrozenModelError as exc:
            raise TeachObsMultimodalBenchmarkError(
                f"safe frozen-model export failed: {exc}"
            ) from exc
        for arm in _ARMS:
            if not np.array_equal(
                frozen_outputs[arm]["predictions"], predictions[arm]
            ) or not np.allclose(
                frozen_outputs[arm]["probabilities"],
                fitted_arm_states[arm]["test_probabilities"],
                rtol=1e-12,
                atol=1e-12,
            ):
                raise TeachObsMultimodalBenchmarkError(
                    f"frozen/in-memory prediction parity failed for {arm}"
                )
        result["frozen_model_export"] = {
            **export_receipt,
            "test_prediction_bitwise_parity_verified": True,
            "test_probability_maximum_absolute_delta": _rounded(
                max(
                    np.max(
                        np.abs(
                            frozen_outputs[arm]["probabilities"]
                            - fitted_arm_states[arm]["test_probabilities"]
                        )
                    )
                    for arm in _ARMS
                )
            ),
            "public_test_labels_were_accessible_before_freeze": True,
            "public_test_outcomes_previously_observed_before_revision": True,
            "confirmatory_lockbox_result_established": False,
            "deployment_accuracy_established": False,
        }
    return result


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def build_public_teachobs_multimodal_receipt(
    private_result: dict[str, Any],
) -> dict[str, Any]:
    """Build an aggregate receipt without ids, paths, text, vectors, or row labels."""

    if private_result.get("benchmark_kind") != BENCHMARK_KIND:
        raise TeachObsMultimodalBenchmarkError(
            "not a TeachObs four-arm multimodal benchmark result"
        )
    audit = private_result.get("dataset_audit")
    transcripts = private_result.get("private_transcript_audit")
    features = private_result.get("private_feature_audit")
    source_identity = private_result.get("source_identity_audit")
    model_selection = private_result.get(
        "training_only_model_selection_audit"
    )
    protocol = private_result.get("protocol")
    arms = private_result.get("arms")
    bootstrap = private_result.get("paired_cluster_bootstrap")
    evidence = private_result.get("evidence_scope")
    if not all(
        isinstance(value, dict)
        for value in (
            audit,
            transcripts,
            features,
            source_identity,
            model_selection,
            protocol,
            arms,
            bootstrap,
            evidence,
        )
    ) or tuple(arms) != _ARMS:
        raise TeachObsMultimodalBenchmarkError(
            "TeachObs multimodal private result is malformed"
        )
    checked_source_identity = _validate_source_identity_audit(
        source_identity,
        selected_test_lesson_count=int(audit.get("test_lesson_count", -1)),
    )
    source_claim_boundary = {
        "source_value_train_test_overlap_detected": True,
        "source_field_used_as_canonical_site_id": False,
        "site_disjointness_verified": False,
        "teacher_disjointness_verified": False,
        "classroom_disjointness_verified": False,
        "source_family_sensitivity_metric_computed": False,
        "site_held_out_evaluation_performed": False,
        "site_held_out_accuracy_established": False,
    }
    if any(
        evidence.get(key) is not expected
        for key, expected in source_claim_boundary.items()
    ):
        raise TeachObsMultimodalBenchmarkError(
            "TeachObs source/site claim boundary is weakened"
        )
    aggregate_arms: dict[str, Any] = {}
    for arm in _ARMS:
        run = arms[arm]
        metrics = run.get("metrics") if isinstance(run, dict) else None
        feature_audit = run.get("feature_audit") if isinstance(run, dict) else None
        if not isinstance(metrics, dict) or not isinstance(feature_audit, dict):
            raise TeachObsMultimodalBenchmarkError("TeachObs arm is malformed")
        aggregate_arms[arm] = {
            "metrics": {name: float(metrics[name]) for name in _METRICS},
            "feature_count": int(feature_audit["feature_count"]),
            "feature_blocks": {
                str(key): int(value)
                for key, value in feature_audit["feature_blocks"].items()
            },
            "model_kind_counts": {
                str(key): int(value)
                for key, value in feature_audit["model_kind_counts"].items()
            },
            "all_iterative_models_converged": bool(
                feature_audit["all_iterative_models_converged"]
            ),
        }
    receipt = {
        "schema_version": "1.0",
        "receipt_kind": "teachobs_aggregate_four_arm_multimodal_benchmark",
        "private_result_canonical_sha256": _canonical_sha256(private_result),
        "dataset_aggregate": {
            key: audit[key]
            for key in (
                "dataset_name",
                "benchmark_profile",
                "profile_source_alignment",
                "published_six_lesson_intersection",
                "lesson_count",
                "train_lesson_count",
                "test_lesson_count",
                "scene_count",
                "train_scene_count",
                "test_scene_count",
                "code_count",
                "visual_code_count",
                "nonvisual_code_count",
                "dataset_sha256",
                "dataset_profile_fingerprint",
                "benchmark_input_fingerprint",
            )
        },
        "transcript_aggregate": {
            key: transcripts[key]
            for key in (
                "schema",
                "profile_id",
                "manifest_file_sha256",
                "manifest_sha256",
                "materialization_fingerprint_sha256",
                "ordered_sample_id_sha256",
                "ordered_scene_text_sha256",
                "repository_binding_sha256",
                "input_hashes_sha256",
                "selected_train_transcript_order_sha256",
                "selected_test_transcript_order_sha256",
                "lesson_count",
                "scene_count",
                "nonempty_scene_count",
                "empty_scene_count",
                "source_tier_lesson_counts",
                "source_tier_scene_counts",
                "all_selected_scene_transcripts_materialized",
                "sample_order_matches_benchmark_profile",
                "released_transcript_fallback_used",
                "labels_read_or_used",
                "empty_scenes_filled_from_released_transcript",
                "released_repository_transcripts_used_for_model_input",
                "benchmark_input_fingerprint",
            )
        },
        "feature_aggregate": {
            "feature_manifest_sha256": str(features["feature_manifest_sha256"]),
            "audio_feature_count": int(features["audio_feature_count"]),
            "visual_feature_count": int(features["visual_feature_count"]),
            "visual_raw_numeric_feature_count": int(
                features["visual_raw_numeric_feature_count"]
            ),
            "ocr_char_tfidf_feature_count": int(
                features["ocr_char_tfidf_feature_count"]
            ),
            "ocr_word_tfidf_feature_count": int(
                features["ocr_word_tfidf_feature_count"]
            ),
            "clip_embedding_feature_count": int(
                features["clip_embedding_feature_count"]
            ),
            "feature_block_dimensions": {
                str(key): int(value)
                for key, value in features["feature_block_dimensions"].items()
            },
            "visual_feature_layout": str(features["visual_feature_layout"]),
            "visual_evidence_set_sha256": str(
                features["visual_evidence_set_sha256"]
            ),
            "visual_evidence_configuration_set_sha256": str(
                features["visual_evidence_configuration_set_sha256"]
            ),
            "clip_weight_manifest_sha256": str(
                features["clip_weight_manifest_sha256"]
            ),
            "ocr_status_counts": {
                str(key): int(value)
                for key, value in features["ocr_status_counts"].items()
            },
            "ocr_completed_scene_count": int(
                features["ocr_completed_scene_count"]
            ),
            "ocr_nonempty_scene_count": int(features["ocr_nonempty_scene_count"]),
            "ocr_completed_scene_fraction": float(
                features["ocr_completed_scene_fraction"]
            ),
            "ocr_complete_for_every_scene": bool(
                features["ocr_complete_for_every_scene"]
            ),
            "all_scenes_have_audio_and_visual_features": bool(
                features["all_scenes_have_audio_and_visual_features"]
            ),
            "scene_identity_and_hash_binding_verified": bool(
                features["scene_identity_and_hash_binding_verified"]
            ),
            "profile_selected_lessons_have_complete_audio_visual_clip": bool(
                features[
                    "profile_selected_lessons_have_complete_audio_visual_clip"
                ]
            ),
        },
        "source_identity_aggregate": {
            key: checked_source_identity[key]
            for key in (
                "schema",
                "metadata_file_sha256",
                "metadata_lesson_count",
                "source_metadata_field",
                "source_field_present_for_every_lesson",
                "selected_train_lesson_count",
                "selected_test_lesson_count",
                "selected_train_source_value_count",
                "selected_test_source_value_count",
                "overlapping_source_value_count",
                "source_value_union_count",
                "train_source_value_set_sha256",
                "test_source_value_set_sha256",
                "overlapping_source_value_set_sha256",
                "train_test_source_value_overlap_detected",
                "source_value_disjoint_split",
                "canonical_site_id_field_present",
                "canonical_teacher_id_field_present",
                "canonical_classroom_id_field_present",
                "source_field_used_as_canonical_site_id",
                "source_field_semantics",
                "site_disjointness_verified",
                "teacher_disjointness_verified",
                "classroom_disjointness_verified",
                "source_family_sensitivity_metric_computed",
                "site_held_out_evaluation_performed",
                "site_held_out_accuracy_established",
            )
        },
        "fixed_protocol": {
            "arms": list(_ARMS),
            "benchmark_profile": str(protocol["benchmark_profile"]),
            "profile_source_alignment": str(
                protocol["profile_source_alignment"]
            ),
            "published_six_lesson_intersection": bool(
                protocol["published_six_lesson_intersection"]
            ),
            "decision_threshold_policy": str(
                protocol["decision_threshold_policy"]
            ),
            "decision_threshold_candidate_grid": [
                float(value)
                for value in protocol[
                    "decision_threshold_candidate_grid"
                ]
            ],
            "official_lesson_disjoint_split": bool(
                protocol["official_lesson_disjoint_split"]
            ),
            "training_only_feature_fitting": True,
            "test_labels_used_for_feature_fitting_training_or_tuning": bool(
                protocol[
                    "test_labels_used_for_feature_fitting_training_or_tuning"
                ]
            ),
            "test_threshold_or_hyperparameter_tuning_performed": bool(
                protocol["test_threshold_or_hyperparameter_tuning_performed"]
            ),
            "thresholds_frozen_before_current_revised_test_scoring": bool(
                protocol[
                    "thresholds_frozen_before_current_revised_test_scoring"
                ]
            ),
            "public_test_outcomes_previously_observed_before_revision": bool(
                protocol[
                    "public_test_outcomes_previously_observed_before_revision"
                ]
            ),
            "current_revision_is_post_test_exploratory": bool(
                protocol["current_revision_is_post_test_exploratory"]
            ),
            "same_test_scenes_used_for_all_arms": bool(
                protocol["same_test_scenes_used_for_all_arms"]
            ),
            "s4_labels_used_for_fitting_tuning_or_metrics": bool(
                protocol["s4_labels_used_for_fitting_tuning_or_metrics"]
            ),
        },
        "training_only_model_selection": {
            "schema": str(model_selection["schema"]),
            "exploratory_revision_after_initial_public_test_evaluation": bool(
                model_selection[
                    "exploratory_revision_after_initial_public_test_evaluation"
                ]
            ),
            "confirmatory_selection": bool(
                model_selection["confirmatory_selection"]
            ),
            "fit_scope": str(model_selection["fit_scope"]),
            "outer_test_labels_used_for_selection": bool(
                model_selection["outer_test_labels_used_for_selection"]
            ),
            "outer_test_features_used_for_selection": bool(
                model_selection["outer_test_features_used_for_selection"]
            ),
            "preprocessing_refit_inside_every_group_fold": bool(
                model_selection[
                    "preprocessing_refit_inside_every_group_fold"
                ]
            ),
            "splitter": str(model_selection["splitter"]),
            "group_unit": "complete_classroom_lesson",
            "fold_count": int(model_selection["fold_count"]),
            "classifier_random_state": int(
                model_selection["classifier_random_state"]
            ),
            "candidate_registry": [
                {
                    "candidate_id": str(row["candidate_id"]),
                    "regularization_c": float(row["regularization_c"]),
                    "class_weight": row["class_weight"],
                }
                for row in model_selection["candidate_registry"]
            ],
            "threshold_candidate_grid": [
                float(value)
                for value in model_selection["threshold_candidate_grid"]
            ],
            "selection_criterion": str(
                model_selection["selection_criterion"]
            ),
            "each_training_scene_has_exactly_one_oof_prediction": bool(
                model_selection[
                    "each_training_scene_has_exactly_one_oof_prediction"
                ]
            ),
            "arms": {
                arm: {
                    "selected_candidate_id": str(
                        model_selection["arms"][arm][
                            "selected_candidate_id"
                        ]
                    ),
                    "selected_regularization_c": float(
                        model_selection["arms"][arm][
                            "selected_regularization_c"
                        ]
                    ),
                    "selected_class_weight": model_selection["arms"][arm][
                        "selected_class_weight"
                    ],
                    "selected_thresholds_sha256": str(
                        model_selection["arms"][arm][
                            "selected_thresholds_sha256"
                        ]
                    ),
                    "selected_oof_metrics": {
                        key: float(value)
                        for key, value in model_selection["arms"][arm][
                            "selected_oof_metrics"
                        ].items()
                    },
                }
                for arm in _ARMS
            },
        },
        "aggregate_arms": aggregate_arms,
        "paired_cluster_bootstrap": bootstrap,
        "claim_boundaries": {
            key: evidence[key]
            for key in (
                "released_consensus_gold_evaluated",
                "public_test_labels_were_accessible_before_implementation",
                "provisional_result",
                "exploratory_multimodal_gain_estimate",
                "confirmatory_multimodal_gain_established",
                "external_lockbox_established",
                "deployment_accuracy_established",
                "learning_effectiveness_established",
                "source_value_train_test_overlap_detected",
                "source_field_used_as_canonical_site_id",
                "site_disjointness_verified",
                "teacher_disjointness_verified",
                "classroom_disjointness_verified",
                "source_family_sensitivity_metric_computed",
                "site_held_out_evaluation_performed",
                "site_held_out_accuracy_established",
                "valid_claim",
                "invalid_claims",
            )
        },
        "row_level_data_included": False,
        "lesson_or_scene_ids_included": False,
        "source_video_urls_included": False,
        "transcript_text_included": False,
        "ocr_text_included": False,
        "audio_rows_included": False,
        "visual_embeddings_included": False,
        "per_label_or_lesson_results_included": False,
    }
    forbidden_keys = {
        "lesson_id",
        "scene_no",
        "path",
        "youtube_url",
        "source_url",
        "transcript",
        "ocr_text",
        "embedding",
        "per_label",
        "per_lesson",
        "predictions",
        "labels",
        "code",
    }
    leaked = forbidden_keys & {key.casefold() for key in _walk_keys(receipt)}
    if leaked:  # pragma: no cover - fixed allowlist defense in depth
        raise TeachObsMultimodalBenchmarkError(
            f"public TeachObs multimodal receipt contains forbidden fields: {sorted(leaked)}"
        )
    return receipt
