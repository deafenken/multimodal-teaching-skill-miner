"""Deterministic text-only benchmark for the public TeachObs release.

The benchmark consumes the fixed 23-lesson train / 7-lesson test split from
TeachObs.  It deliberately has no threshold-selection or test-set tuning API:
the TF-IDF vocabulary and every binary classifier are fitted on the official
training lessons, and the decision threshold is always 0.5.

TeachObs publishes consensus gold labels, not the individual files from its
seven coders.  Consequently this module can establish a reproducible held-out
benchmark against the released gold, but it cannot independently recompute
inter-rater reliability or turn the public test set into an external lockbox.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Iterable


TEACHOBS_EXPECTED_LESSON_COUNT = 30
TEACHOBS_EXPECTED_TRAIN_LESSON_COUNT = 23
TEACHOBS_EXPECTED_TEST_LESSON_COUNT = 7
TEACHOBS_EXPECTED_SCENE_COUNT = 5158
TEACHOBS_EXPECTED_CODE_COUNT = 39
TEACHOBS_EXPECTED_VISUAL_CODE_COUNT = 20
TEACHOBS_EXPECTED_NONVISUAL_CODE_COUNT = 19
TEACHOBS_EXPECTED_TEST_IDS = frozenset(
    {"S2", "S4", "S5", "S19", "S24", "S28", "S30"}
)

_DECISION_THRESHOLD = 0.5
_TFIDF_MAX_FEATURES = 30_000
_TFIDF_NGRAM_RANGE = (3, 5)
_LOGISTIC_MAX_ITER = 2_000
_LOGISTIC_SOLVER = "liblinear"
_RANDOM_STATE = 0

_SPEAKER_TURN_START = re.compile(r"(?=\[[^\]\r\n]{1,80}\])")
_SPEAKER_TURN = re.compile(
    r"^\[(?P<speaker>[^\]\r\n]{1,80})\]\s*(?P<body>.*)$", re.DOTALL
)
_NUMBERED_SPEAKER_SUFFIX = re.compile(r"(?:\s+|#)\d+$")
_WHITESPACE = re.compile(r"\s+")


class TeachObsBenchmarkError(ValueError):
    """Raised when TeachObs assets or benchmark inputs fail validation."""


class TeachObsBenchmarkDependencyError(RuntimeError):
    """Raised when the optional recognition dependencies are unavailable."""


@dataclass(frozen=True, slots=True)
class TeachObsScene:
    """One released 15-second TeachObs scene kept in private process memory."""

    lesson_id: str
    scene_no: int
    transcript: str
    labels: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TeachObsTextDataset:
    """Validated official TeachObs text and consensus-label split."""

    code_names: tuple[str, ...]
    code_groups: tuple[str, ...]
    train_scenes: tuple[TeachObsScene, ...]
    test_scenes: tuple[TeachObsScene, ...]
    dataset_sha256: str
    source_file_count: int


class _FileFingerprintCollector:
    def __init__(self, repository_root: Path) -> None:
        self.root = repository_root
        self.entries: dict[str, dict[str, Any]] = {}

    def read(self, path: Path) -> bytes:
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise TeachObsBenchmarkError(
                f"required TeachObs asset is missing: {path.name}"
            ) from exc
        if not resolved.is_relative_to(self.root):
            raise TeachObsBenchmarkError(
                f"TeachObs asset resolves outside the repository: {path.name}"
            )
        if not resolved.is_file():
            raise TeachObsBenchmarkError(
                f"required TeachObs asset is not a regular file: {path.name}"
            )
        relative = resolved.relative_to(self.root).as_posix()
        payload = resolved.read_bytes()
        current = {
            "path": relative,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        previous = self.entries.get(relative)
        if previous is not None and previous != current:
            raise TeachObsBenchmarkError(
                f"TeachObs asset changed while it was being read: {path.name}"
            )
        self.entries[relative] = current
        return payload

    def canonical_sha256(self) -> str:
        payload = json.dumps(
            [self.entries[key] for key in sorted(self.entries)],
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _decode_text(payload: bytes, *, name: str) -> str:
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise TeachObsBenchmarkError(
            f"TeachObs asset is not valid UTF-8: {name}"
        ) from exc


def _load_code_scheme(
    collector: _FileFingerprintCollector,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    path = collector.root / "data" / "track_a" / "coding_scheme.json"
    try:
        value = json.loads(_decode_text(collector.read(path), name=path.name))
    except json.JSONDecodeError as exc:
        raise TeachObsBenchmarkError("TeachObs coding scheme is invalid JSON") from exc
    rows = value.get("codes") if isinstance(value, dict) else None
    if not isinstance(rows, list) or value.get("n_codes") != len(rows):
        raise TeachObsBenchmarkError("TeachObs coding scheme has an invalid code list")
    names: list[str] = []
    groups: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise TeachObsBenchmarkError("TeachObs coding scheme contains a non-object")
        name = str(row.get("name", "")).strip()
        group = str(row.get("group", "")).strip()
        if not name or name in names or group not in {"visual", "nonvisual"}:
            raise TeachObsBenchmarkError("TeachObs coding scheme contains an invalid code")
        names.append(name)
        groups.append(group)
    if len(names) != TEACHOBS_EXPECTED_CODE_COUNT:
        raise TeachObsBenchmarkError(
            "TeachObs coding scheme does not contain the expected 39 codes"
        )
    if groups.count("visual") != TEACHOBS_EXPECTED_VISUAL_CODE_COUNT:
        raise TeachObsBenchmarkError("TeachObs visual-code count does not equal 20")
    if groups.count("nonvisual") != TEACHOBS_EXPECTED_NONVISUAL_CODE_COUNT:
        raise TeachObsBenchmarkError("TeachObs nonvisual-code count does not equal 19")
    return tuple(names), tuple(groups)


def _load_split_ids(
    collector: _FileFingerprintCollector, split: str
) -> tuple[str, ...]:
    path = collector.root / "data" / "track_a" / "splits" / f"{split}_ids.txt"
    values = tuple(
        line.strip()
        for line in _decode_text(collector.read(path), name=path.name).splitlines()
        if line.strip()
    )
    if not values or len(values) != len(set(values)):
        raise TeachObsBenchmarkError(f"TeachObs {split} split is empty or duplicated")
    return values


def _validate_official_split(
    collector: _FileFingerprintCollector,
    train_ids: tuple[str, ...],
    test_ids: tuple[str, ...],
) -> None:
    if len(train_ids) != TEACHOBS_EXPECTED_TRAIN_LESSON_COUNT:
        raise TeachObsBenchmarkError("TeachObs train split does not contain 23 lessons")
    if len(test_ids) != TEACHOBS_EXPECTED_TEST_LESSON_COUNT:
        raise TeachObsBenchmarkError("TeachObs test split does not contain 7 lessons")
    if set(train_ids) & set(test_ids):
        raise TeachObsBenchmarkError("TeachObs train and test lesson ids overlap")
    expected_ids = {
        f"S{index}" for index in range(1, TEACHOBS_EXPECTED_LESSON_COUNT + 1)
    }
    if set(train_ids) | set(test_ids) != expected_ids:
        raise TeachObsBenchmarkError("TeachObs split does not cover exactly S1-S30")
    if set(test_ids) != TEACHOBS_EXPECTED_TEST_IDS:
        raise TeachObsBenchmarkError("TeachObs test split differs from the pinned release")

    lessons_path = collector.root / "data" / "lessons.csv"
    lessons_text = _decode_text(
        collector.read(lessons_path), name=lessons_path.name
    )
    rows = list(csv.DictReader(io.StringIO(lessons_text)))
    if not rows or not {"id", "split"}.issubset(rows[0]):
        raise TeachObsBenchmarkError("TeachObs lessons.csv has an invalid schema")
    row_ids = [str(row["id"]).strip() for row in rows]
    if len(rows) != TEACHOBS_EXPECTED_LESSON_COUNT or set(row_ids) != expected_ids:
        raise TeachObsBenchmarkError("TeachObs lessons.csv does not cover exactly S1-S30")
    if len(row_ids) != len(set(row_ids)):
        raise TeachObsBenchmarkError("TeachObs lessons.csv contains duplicate ids")
    declared = {
        "train": {str(row["id"]).strip() for row in rows if row["split"] == "train"},
        "test": {str(row["id"]).strip() for row in rows if row["split"] == "test"},
    }
    if any(row["split"] not in {"train", "test"} for row in rows):
        raise TeachObsBenchmarkError("TeachObs lessons.csv contains an invalid split")
    if declared["train"] != set(train_ids) or declared["test"] != set(test_ids):
        raise TeachObsBenchmarkError(
            "TeachObs lessons.csv disagrees with the official split files"
        )


def _parse_jsonl(payload: bytes, *, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        _decode_text(payload, name=name).splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TeachObsBenchmarkError(
                f"invalid JSONL in {name} at line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise TeachObsBenchmarkError(
                f"non-object JSONL row in {name} at line {line_number}"
            )
        rows.append(value)
    if not rows:
        raise TeachObsBenchmarkError(f"TeachObs JSONL file is empty: {name}")
    return rows


def _safe_transcript_name(value: Any) -> str:
    if not isinstance(value, str):
        raise TeachObsBenchmarkError("unsafe transcript path in a scene manifest")
    name = value.strip()
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or len(path.parts) != 1
        or path.parts[0] in {".", ".."}
    ):
        raise TeachObsBenchmarkError("unsafe transcript path in a scene manifest")
    return name


def _load_lesson_scenes(
    collector: _FileFingerprintCollector,
    *,
    split: str,
    lesson_id: str,
    code_names: tuple[str, ...],
) -> list[TeachObsScene]:
    scene_dir = collector.root / "data" / "scenes" / lesson_id
    manifest_path = scene_dir / "manifest.jsonl"
    gold_path = (
        collector.root
        / "data"
        / "track_a"
        / "gold"
        / split
        / f"{lesson_id}.jsonl"
    )
    manifest = _parse_jsonl(
        collector.read(manifest_path), name=f"{lesson_id} manifest"
    )
    gold = _parse_jsonl(collector.read(gold_path), name=f"{lesson_id} gold")
    if len(manifest) != len(gold):
        raise TeachObsBenchmarkError(
            f"TeachObs scene/gold count mismatch for {lesson_id}"
        )

    scenes: list[TeachObsScene] = []
    transcript_names: set[str] = set()
    for expected_scene_no, (scene_row, gold_row) in enumerate(
        zip(manifest, gold, strict=True), start=1
    ):
        if scene_row.get("id") != lesson_id or gold_row.get("lesson_id") != lesson_id:
            raise TeachObsBenchmarkError(
                f"TeachObs lesson binding mismatch for {lesson_id}"
            )
        if (
            scene_row.get("scene_no") != expected_scene_no
            or gold_row.get("scene_no") != expected_scene_no
        ):
            raise TeachObsBenchmarkError(
                f"TeachObs scenes are not contiguous for {lesson_id}"
            )
        start = scene_row.get("start")
        end = scene_row.get("end")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or float(start) < 0
            or float(end) <= float(start)
        ):
            raise TeachObsBenchmarkError(
                f"TeachObs scene timing is invalid for {lesson_id}"
            )
        transcript_name = _safe_transcript_name(scene_row.get("transcript_file"))
        if transcript_name in transcript_names:
            raise TeachObsBenchmarkError(
                f"TeachObs transcript is reused within {lesson_id}"
            )
        transcript_names.add(transcript_name)
        transcript_path = scene_dir / transcript_name
        transcript = _decode_text(
            collector.read(transcript_path), name=transcript_name
        ).strip()

        codes = gold_row.get("codes")
        if not isinstance(codes, dict) or tuple(codes) != code_names:
            raise TeachObsBenchmarkError(
                f"TeachObs gold code vector does not match the scheme for {lesson_id}"
            )
        label_values = tuple(codes[name] for name in code_names)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1}
            for value in label_values
        ):
            raise TeachObsBenchmarkError(
                f"TeachObs labels are not integer 0/1 for {lesson_id}"
            )
        scenes.append(
            TeachObsScene(
                lesson_id=lesson_id,
                scene_no=expected_scene_no,
                transcript=transcript,
                labels=label_values,
            )
        )
    return scenes


def load_teachobs_text_dataset(
    repository_root: str | Path,
) -> TeachObsTextDataset:
    """Load and fail-closed audit the official TeachObs text benchmark assets."""

    root = Path(repository_root).expanduser().resolve()
    if not root.is_dir():
        raise TeachObsBenchmarkError("TeachObs repository root is not a directory")
    collector = _FileFingerprintCollector(root)
    code_names, code_groups = _load_code_scheme(collector)
    train_ids = _load_split_ids(collector, "train")
    test_ids = _load_split_ids(collector, "test")
    _validate_official_split(collector, train_ids, test_ids)

    train_scenes: list[TeachObsScene] = []
    test_scenes: list[TeachObsScene] = []
    for split, lesson_ids, destination in (
        ("train", train_ids, train_scenes),
        ("test", test_ids, test_scenes),
    ):
        for lesson_id in lesson_ids:
            destination.extend(
                _load_lesson_scenes(
                    collector,
                    split=split,
                    lesson_id=lesson_id,
                    code_names=code_names,
                )
            )

    if len(train_scenes) + len(test_scenes) != TEACHOBS_EXPECTED_SCENE_COUNT:
        raise TeachObsBenchmarkError(
            "TeachObs scene count does not equal the pinned 5,158-scene release"
        )
    if {scene.lesson_id for scene in train_scenes} & {
        scene.lesson_id for scene in test_scenes
    }:
        raise TeachObsBenchmarkError("TeachObs train/test scene leakage detected")
    return TeachObsTextDataset(
        code_names=code_names,
        code_groups=code_groups,
        train_scenes=tuple(train_scenes),
        test_scenes=tuple(test_scenes),
        dataset_sha256=collector.canonical_sha256(),
        source_file_count=len(collector.entries),
    )


def sanitize_repeated_transcript(text: str) -> tuple[str, dict[str, int | bool]]:
    """Collapse exact repeated numbered-speaker turns without using any labels.

    The released transcripts occasionally contain hundreds of copies of the
    same utterance under ``[Student 1]``, ``[Student 2]``, and so on.  This
    deterministic sensitivity transform keeps the first exact role/utterance
    pair and removes later copies.  It does not paraphrase, translate, truncate,
    or inspect gold labels.  Because multiple students can legitimately give
    the same response, the raw benchmark remains the primary result.
    """

    original = str(text)
    segments = _SPEAKER_TURN_START.split(original)
    seen: set[tuple[str, str]] = set()
    kept: list[str] = []
    turn_count = 0
    removed = 0
    for segment in segments:
        if not segment:
            continue
        match = _SPEAKER_TURN.match(segment)
        if match is None:
            kept.append(segment)
            continue
        turn_count += 1
        original_speaker = _WHITESPACE.sub(
            " ", match.group("speaker")
        ).strip().casefold()
        numbered_speaker = _NUMBERED_SPEAKER_SUFFIX.search(original_speaker) is not None
        speaker = _NUMBERED_SPEAKER_SUFFIX.sub("", original_speaker)
        body = _WHITESPACE.sub(" ", match.group("body")).strip()
        canonical_body = body.strip(" \t\r\n\"'“”‘’").casefold()
        key = (speaker, canonical_body)
        if numbered_speaker and canonical_body and key in seen:
            removed += 1
            continue
        if numbered_speaker and canonical_body:
            seen.add(key)
        kept.append(segment)
    sanitized = "".join(kept).strip()
    diagnostics: dict[str, int | bool] = {
        "changed": removed > 0,
        "original_character_count": len(original),
        "sanitized_character_count": len(sanitized),
        "speaker_turn_count": turn_count,
        "removed_duplicate_turn_count": removed,
    }
    return sanitized, diagnostics


def _prepare_text_variants(
    scenes: tuple[TeachObsScene, ...],
) -> tuple[list[str], list[str], dict[str, int | float]]:
    raw: list[str] = []
    sanitized: list[str] = []
    affected = 0
    removed_turns = 0
    original_characters = 0
    sanitized_characters = 0
    for scene in scenes:
        raw.append(scene.transcript)
        clean, diagnostics = sanitize_repeated_transcript(scene.transcript)
        sanitized.append(clean)
        affected += int(bool(diagnostics["changed"]))
        removed_turns += int(diagnostics["removed_duplicate_turn_count"])
        original_characters += int(diagnostics["original_character_count"])
        sanitized_characters += int(diagnostics["sanitized_character_count"])
    reduction = (
        1.0 - sanitized_characters / original_characters
        if original_characters
        else 0.0
    )
    return raw, sanitized, {
        "scene_count": len(scenes),
        "affected_scene_count": affected,
        "removed_duplicate_turn_count": removed_turns,
        "original_character_count": original_characters,
        "sanitized_character_count": sanitized_characters,
        "character_reduction_fraction": _rounded(reduction),
    }


def _require_recognition_dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            hamming_loss,
            precision_recall_fscore_support,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TeachObsBenchmarkDependencyError(
            "TeachObs benchmarking requires the 'recognition' optional dependencies "
            "(numpy and scikit-learn)"
        ) from exc
    return {
        "np": np,
        "TfidfVectorizer": TfidfVectorizer,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "f1_score": f1_score,
        "hamming_loss": hamming_loss,
        "precision_recall_fscore_support": precision_recall_fscore_support,
    }


def _rounded(value: Any) -> float:
    return round(float(value), 6)


def _metric_summary(y_true: Any, y_pred: Any, groups: tuple[str, ...]) -> dict[str, float]:
    dependencies = _require_recognition_dependencies()
    np = dependencies["np"]
    f1_score = dependencies["f1_score"]
    visual = np.asarray([group == "visual" for group in groups], dtype=bool)
    nonvisual = ~visual
    return {
        "micro_f1": _rounded(
            f1_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "macro_f1": _rounded(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "subset_accuracy": _rounded(
            dependencies["accuracy_score"](y_true, y_pred)
        ),
        "hamming_accuracy": _rounded(
            1.0 - dependencies["hamming_loss"](y_true, y_pred)
        ),
        "visual_macro_f1": _rounded(
            f1_score(
                y_true[:, visual],
                y_pred[:, visual],
                average="macro",
                zero_division=0,
            )
        ),
        "nonvisual_macro_f1": _rounded(
            f1_score(
                y_true[:, nonvisual],
                y_pred[:, nonvisual],
                average="macro",
                zero_division=0,
            )
        ),
    }


def _run_text_arm(
    dataset: TeachObsTextDataset,
    *,
    train_texts: list[str],
    test_texts: list[str],
    variant: str,
) -> dict[str, Any]:
    dependencies = _require_recognition_dependencies()
    np = dependencies["np"]
    if len(train_texts) != len(dataset.train_scenes):
        raise TeachObsBenchmarkError("training text/scene count mismatch")
    if len(test_texts) != len(dataset.test_scenes):
        raise TeachObsBenchmarkError("test text/scene count mismatch")
    if not any(text.strip() for text in train_texts):
        raise TeachObsBenchmarkError("all TeachObs training transcripts are empty")

    y_train = np.asarray(
        [scene.labels for scene in dataset.train_scenes], dtype=np.uint8
    )
    y_test = np.asarray(
        [scene.labels for scene in dataset.test_scenes], dtype=np.uint8
    )
    vectorizer = dependencies["TfidfVectorizer"](
        analyzer="char_wb",
        ngram_range=_TFIDF_NGRAM_RANGE,
        lowercase=True,
        min_df=1,
        max_features=_TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float64,
    )
    train_features = vectorizer.fit_transform(train_texts)
    test_features = vectorizer.transform(test_texts)
    predictions = np.zeros_like(y_test, dtype=np.uint8)
    model_kinds: list[str] = []
    converged: list[bool] = []
    for label_index in range(y_train.shape[1]):
        train_column = y_train[:, label_index]
        unique = np.unique(train_column)
        if unique.size == 1:
            constant = int(unique[0])
            predictions[:, label_index] = constant
            model_kinds.append(f"constant_{constant}")
            converged.append(True)
            continue
        classifier = dependencies["LogisticRegression"](
            class_weight="balanced",
            max_iter=_LOGISTIC_MAX_ITER,
            random_state=_RANDOM_STATE,
            solver=_LOGISTIC_SOLVER,
        )
        classifier.fit(train_features, train_column)
        positive_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
        probability = classifier.predict_proba(test_features)[:, positive_index]
        predictions[:, label_index] = (probability >= _DECISION_THRESHOLD).astype(
            np.uint8
        )
        model_kinds.append("logistic_regression")
        converged.append(int(classifier.n_iter_[0]) < _LOGISTIC_MAX_ITER)

    precision, recall, f1, support = dependencies[
        "precision_recall_fscore_support"
    ](y_test, predictions, average=None, zero_division=0)
    per_label: list[dict[str, Any]] = []
    for index, (name, group) in enumerate(
        zip(dataset.code_names, dataset.code_groups, strict=True)
    ):
        per_label.append(
            {
                "code": name,
                "group": group,
                "train_positive_count": int(y_train[:, index].sum()),
                "test_positive_count": int(support[index]),
                "predicted_positive_count": int(predictions[:, index].sum()),
                "precision": _rounded(precision[index]),
                "recall": _rounded(recall[index]),
                "f1": _rounded(f1[index]),
                "model_kind": model_kinds[index],
                "converged": converged[index],
            }
        )

    lesson_indices: dict[str, list[int]] = {}
    for index, scene in enumerate(dataset.test_scenes):
        lesson_indices.setdefault(scene.lesson_id, []).append(index)
    per_lesson: list[dict[str, Any]] = []
    for lesson_id, indices in lesson_indices.items():
        lesson_truth = y_test[indices]
        lesson_predictions = predictions[indices]
        per_lesson.append(
            {
                "lesson_id": lesson_id,
                "scene_count": len(indices),
                "positive_label_count": int(lesson_truth.sum()),
                "predicted_positive_count": int(lesson_predictions.sum()),
                "metrics": _metric_summary(
                    lesson_truth, lesson_predictions, dataset.code_groups
                ),
            }
        )

    kind_counts = {
        kind: model_kinds.count(kind) for kind in sorted(set(model_kinds))
    }
    return {
        "variant": variant,
        "metrics": _metric_summary(y_test, predictions, dataset.code_groups),
        "feature_audit": {
            "tfidf_feature_count": int(train_features.shape[1]),
            "train_nonzero_feature_count": int(train_features.nnz),
            "test_nonzero_feature_count": int(test_features.nnz),
            "model_kind_counts": kind_counts,
            "all_iterative_models_converged": all(converged),
        },
        "per_label": per_label,
        "per_lesson": per_lesson,
    }


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


def run_teachobs_text_benchmark(
    repository_root: str | Path,
    *,
    compare_sanitized: bool = True,
) -> dict[str, Any]:
    """Run the fixed TeachObs transcript-only held-out benchmark.

    ``compare_sanitized`` adds a sensitivity analysis with exact duplicate
    numbered-speaker turns removed.  Raw transcripts remain the primary run.
    Neither arm exposes a test-tuning knob, and both use the fixed threshold
    0.5.
    """

    dataset = load_teachobs_text_dataset(repository_root)
    train_raw, train_sanitized, train_anomalies = _prepare_text_variants(
        dataset.train_scenes
    )
    test_raw, test_sanitized, test_anomalies = _prepare_text_variants(
        dataset.test_scenes
    )
    runs = {
        "raw": _run_text_arm(
            dataset,
            train_texts=train_raw,
            test_texts=test_raw,
            variant="raw_released_transcripts_primary",
        )
    }
    comparison: dict[str, float] | None = None
    if compare_sanitized:
        runs["sanitized"] = _run_text_arm(
            dataset,
            train_texts=train_sanitized,
            test_texts=test_sanitized,
            variant="exact_duplicate_numbered_speaker_turns_removed_sensitivity",
        )
        comparison = {
            metric: _rounded(
                runs["sanitized"]["metrics"][metric]
                - runs["raw"]["metrics"][metric]
            )
            for metric in runs["raw"]["metrics"]
        }

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "benchmark_kind": "teachobs_official_held_out_text_baseline",
        "primary_run": "raw",
        "dataset_audit": {
            "dataset_name": "TeachObs",
            "lesson_count": TEACHOBS_EXPECTED_LESSON_COUNT,
            "train_lesson_count": TEACHOBS_EXPECTED_TRAIN_LESSON_COUNT,
            "test_lesson_count": TEACHOBS_EXPECTED_TEST_LESSON_COUNT,
            "scene_count": len(dataset.train_scenes) + len(dataset.test_scenes),
            "train_scene_count": len(dataset.train_scenes),
            "test_scene_count": len(dataset.test_scenes),
            "code_count": len(dataset.code_names),
            "visual_code_count": dataset.code_groups.count("visual"),
            "nonvisual_code_count": dataset.code_groups.count("nonvisual"),
            "source_file_count": dataset.source_file_count,
            "dataset_sha256": dataset.dataset_sha256,
            "transcript_sanitization_audit": {
                "train": train_anomalies,
                "test": test_anomalies,
            },
        },
        "protocol": {
            "input_modality": "released_scene_transcripts_only",
            "official_lesson_disjoint_split": True,
            "train_lessons": 23,
            "test_lessons": 7,
            "tfidf_analyzer": "char_wb",
            "tfidf_ngram_range": list(_TFIDF_NGRAM_RANGE),
            "tfidf_min_df": 1,
            "tfidf_max_features": _TFIDF_MAX_FEATURES,
            "classifier": "independent_binary_logistic_regression",
            "class_weight": "balanced",
            "constant_train_label_handling": "predict_that_constant",
            "solver": _LOGISTIC_SOLVER,
            "max_iterations": _LOGISTIC_MAX_ITER,
            "random_state": _RANDOM_STATE,
            "decision_threshold": _DECISION_THRESHOLD,
            "feature_extractor_fit_scope": "official_train_lessons_only",
            "classifier_fit_scope": "official_train_lessons_only",
            "test_labels_used_for_training_or_tuning": False,
            "test_threshold_tuning_performed": False,
            "raw_transcript_result_is_primary": True,
        },
        "evidence_scope": {
            "released_consensus_gold_evaluated": True,
            "individual_coder_files_available_in_release": False,
            "inter_rater_reliability_independently_recomputed": False,
            "public_test_labels_were_accessible_before_this_run": True,
            "external_lockbox_established": False,
            "deployment_accuracy_established": False,
            "multimodal_gain_established_by_this_benchmark": False,
            "learning_effectiveness_established": False,
            "valid_claim": (
                "reproducible provisional held-out text baseline on the released "
                "TeachObs 23/7 split"
            ),
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": _version("numpy"),
            "scikit_learn": _version("scikit-learn"),
        },
        "runs": runs,
    }
    if comparison is not None:
        result["sanitized_minus_raw"] = comparison
    return result


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _select_public_run(run: dict[str, Any]) -> dict[str, Any]:
    metrics = run.get("metrics")
    feature_audit = run.get("feature_audit")
    if not isinstance(metrics, dict) or not isinstance(feature_audit, dict):
        raise TeachObsBenchmarkError("TeachObs private benchmark run is malformed")
    metric_names = (
        "micro_f1",
        "macro_f1",
        "subset_accuracy",
        "hamming_accuracy",
        "visual_macro_f1",
        "nonvisual_macro_f1",
    )
    return {
        "metrics": {name: float(metrics[name]) for name in metric_names},
        "aggregate_model_audit": {
            "tfidf_feature_count": int(feature_audit["tfidf_feature_count"]),
            "model_kind_counts": {
                str(key): int(value)
                for key, value in feature_audit["model_kind_counts"].items()
            },
            "all_iterative_models_converged": bool(
                feature_audit["all_iterative_models_converged"]
            ),
        },
    }


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def build_public_teachobs_benchmark_receipt(
    private_result: dict[str, Any],
) -> dict[str, Any]:
    """Create an aggregate-only receipt with no ids, text, paths, or row labels."""

    if private_result.get("benchmark_kind") != (
        "teachobs_official_held_out_text_baseline"
    ):
        raise TeachObsBenchmarkError("not a TeachObs text benchmark result")
    audit = private_result.get("dataset_audit")
    protocol = private_result.get("protocol")
    evidence = private_result.get("evidence_scope")
    runs = private_result.get("runs")
    if not all(isinstance(value, dict) for value in (audit, protocol, evidence, runs)):
        raise TeachObsBenchmarkError("TeachObs private benchmark result is malformed")
    selected_runs = {
        name: _select_public_run(runs[name])
        for name in ("raw", "sanitized")
        if name in runs
    }
    receipt: dict[str, Any] = {
        "schema_version": "1.0",
        "receipt_kind": "teachobs_aggregate_text_benchmark",
        "private_result_canonical_sha256": _canonical_sha256(private_result),
        "private_result_hash_serialization": (
            "UTF-8 JSON; sorted keys; compact separators; ensure_ascii=true; "
            "allow_nan=false"
        ),
        "dataset_aggregate": {
            "dataset_name": "TeachObs",
            "lesson_count": int(audit["lesson_count"]),
            "train_lesson_count": int(audit["train_lesson_count"]),
            "test_lesson_count": int(audit["test_lesson_count"]),
            "scene_count": int(audit["scene_count"]),
            "train_scene_count": int(audit["train_scene_count"]),
            "test_scene_count": int(audit["test_scene_count"]),
            "code_count": int(audit["code_count"]),
            "visual_code_count": int(audit["visual_code_count"]),
            "nonvisual_code_count": int(audit["nonvisual_code_count"]),
            "dataset_sha256": str(audit["dataset_sha256"]),
            "transcript_sanitization_audit": audit[
                "transcript_sanitization_audit"
            ],
        },
        "fixed_protocol": {
            "input_modality": protocol["input_modality"],
            "official_lesson_disjoint_split": bool(
                protocol["official_lesson_disjoint_split"]
            ),
            "decision_threshold": float(protocol["decision_threshold"]),
            "feature_extractor_fit_scope": protocol["feature_extractor_fit_scope"],
            "classifier_fit_scope": protocol["classifier_fit_scope"],
            "test_labels_used_for_training_or_tuning": bool(
                protocol["test_labels_used_for_training_or_tuning"]
            ),
            "test_threshold_tuning_performed": bool(
                protocol["test_threshold_tuning_performed"]
            ),
            "raw_transcript_result_is_primary": bool(
                protocol["raw_transcript_result_is_primary"]
            ),
        },
        "claim_boundaries": {
            "released_consensus_gold_evaluated": bool(
                evidence["released_consensus_gold_evaluated"]
            ),
            "individual_coder_files_available_in_release": bool(
                evidence["individual_coder_files_available_in_release"]
            ),
            "inter_rater_reliability_independently_recomputed": bool(
                evidence["inter_rater_reliability_independently_recomputed"]
            ),
            "external_lockbox_established": bool(
                evidence["external_lockbox_established"]
            ),
            "deployment_accuracy_established": bool(
                evidence["deployment_accuracy_established"]
            ),
            "multimodal_gain_established_by_this_benchmark": bool(
                evidence["multimodal_gain_established_by_this_benchmark"]
            ),
            "learning_effectiveness_established": bool(
                evidence["learning_effectiveness_established"]
            ),
            "valid_claim": str(evidence["valid_claim"]),
        },
        "aggregate_runs": selected_runs,
        "row_level_data_included": False,
        "source_video_urls_included": False,
        "transcript_text_included": False,
        "lesson_or_scene_ids_included": False,
    }
    if "sanitized_minus_raw" in private_result:
        receipt["sanitized_minus_raw"] = {
            str(key): float(value)
            for key, value in private_result["sanitized_minus_raw"].items()
        }
    forbidden_keys = {
        "lesson_id",
        "scene_no",
        "transcript",
        "youtube_url",
        "source_url",
        "path",
        "per_lesson",
        "per_label",
        "predictions",
        "labels",
    }
    leaked = forbidden_keys & {key.casefold() for key in _walk_keys(receipt)}
    if leaked:  # pragma: no cover - fixed allowlist defense in depth
        raise TeachObsBenchmarkError(
            f"public TeachObs receipt contains forbidden fields: {sorted(leaked)}"
        )
    return receipt
