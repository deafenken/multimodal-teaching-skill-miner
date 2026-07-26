from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import sklearn  # noqa: F401

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional recognition dependency
    SKLEARN_AVAILABLE = False

import teaching_skill_miner.teachobs_benchmark as benchmark
from teaching_skill_miner.teachobs_benchmark import (
    TeachObsBenchmarkError,
    build_public_teachobs_benchmark_receipt,
    load_teachobs_text_dataset,
    run_teachobs_text_benchmark,
    sanitize_repeated_transcript,
)


TEST_IDS = {"S2", "S4", "S5", "S19", "S24", "S28", "S30"}
CODE_NAMES = [f"Code {index:02d}" for index in range(39)]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _labels(lesson_number: int, scene_no: int, *, flip_test: bool) -> dict[str, int]:
    values: list[int] = []
    for code_index in range(39):
        if code_index == 0:
            value = (lesson_number + scene_no) % 2
            if flip_test and f"S{lesson_number}" in TEST_IDS:
                value = 1
        elif code_index == 1:
            value = 0
        elif code_index == 2:
            value = 1
        else:
            value = int((lesson_number + scene_no + code_index) % 3 == 0)
        values.append(value)
    return dict(zip(CODE_NAMES, values, strict=True))


def _build_repository(root: Path, *, flip_test: bool = False) -> Path:
    repository = root / "repository"
    track_a = repository / "data" / "track_a"
    (track_a / "splits").mkdir(parents=True)
    codes = [
        {
            "name": name,
            "group": "visual" if index < 20 else "nonvisual",
            "definition": "",
        }
        for index, name in enumerate(CODE_NAMES)
    ]
    (track_a / "coding_scheme.json").write_text(
        json.dumps({"n_codes": 39, "codes": codes}), encoding="utf-8"
    )
    train_ids = [f"S{index}" for index in range(1, 31) if f"S{index}" not in TEST_IDS]
    test_ids = [f"S{index}" for index in range(1, 31) if f"S{index}" in TEST_IDS]
    (track_a / "splits" / "train_ids.txt").write_text(
        "\n".join(train_ids) + "\n", encoding="utf-8"
    )
    (track_a / "splits" / "test_ids.txt").write_text(
        "\n".join(test_ids) + "\n", encoding="utf-8"
    )
    lessons = ["id,split"]
    for index in range(1, 31):
        lesson_id = f"S{index}"
        split = "test" if lesson_id in TEST_IDS else "train"
        lessons.append(f"{lesson_id},{split}")
        scene_dir = repository / "data" / "scenes" / lesson_id
        scene_dir.mkdir(parents=True)
        manifest: list[dict] = []
        gold: list[dict] = []
        for scene_no in (1, 2):
            transcript_name = f"{lesson_id}_scene_{scene_no:04d}.txt"
            text = (
                f"[Instructor] common lesson parity {index % 2} scene {scene_no}."
            )
            if lesson_id == "S1" and scene_no == 1:
                text += ' [Student 1] "same" [Student 2] "same"'
            if lesson_id == "S2":
                text += " test-only-never-fit-token"
            (scene_dir / transcript_name).write_text(text, encoding="utf-8")
            manifest.append(
                {
                    "id": lesson_id,
                    "scene_no": scene_no,
                    "start": float((scene_no - 1) * 15),
                    "end": float(scene_no * 15),
                    "transcript_file": transcript_name,
                }
            )
            gold.append(
                {
                    "lesson_id": lesson_id,
                    "scene_no": scene_no,
                    "time": "00:00-00:15",
                    "codes": _labels(index, scene_no, flip_test=flip_test),
                }
            )
        _write_jsonl(scene_dir / "manifest.jsonl", manifest)
        _write_jsonl(track_a / "gold" / split / f"{lesson_id}.jsonl", gold)
    (repository / "data" / "lessons.csv").write_text(
        "\n".join(lessons) + "\n", encoding="utf-8"
    )
    return repository


class TeachObsSanitizationTests(unittest.TestCase):
    def test_exact_numbered_speaker_duplicates_are_collapsed(self) -> None:
        original = (
            '[Student 1] "same answer" [Student 2] "same answer" '
            '[Student 3] "different answer"'
        )
        sanitized, audit = sanitize_repeated_transcript(original)
        self.assertIn("[Student 1]", sanitized)
        self.assertNotIn("[Student 2]", sanitized)
        self.assertIn("[Student 3]", sanitized)
        self.assertTrue(audit["changed"])
        self.assertEqual(audit["removed_duplicate_turn_count"], 1)

    def test_non_duplicate_text_is_byte_content_preserving_after_strip(self) -> None:
        original = '[Instructor] "first" [Student 1] "second"'
        sanitized, audit = sanitize_repeated_transcript(original)
        self.assertEqual(sanitized, original)
        self.assertFalse(audit["changed"])
        self.assertEqual(audit["removed_duplicate_turn_count"], 0)

    def test_repeated_unnumbered_speaker_is_preserved(self) -> None:
        original = '[Instructor] "repeat" [Instructor] "repeat"'
        sanitized, audit = sanitize_repeated_transcript(original)
        self.assertEqual(sanitized, original)
        self.assertFalse(audit["changed"])


class TeachObsLoaderTests(unittest.TestCase):
    def test_official_shape_and_constant_labels_are_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = _build_repository(Path(temporary))
            with patch.object(benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60):
                dataset = load_teachobs_text_dataset(repository)
        self.assertEqual(len(dataset.train_scenes), 46)
        self.assertEqual(len(dataset.test_scenes), 14)
        self.assertEqual(len(dataset.code_names), 39)
        self.assertEqual(dataset.code_groups.count("visual"), 20)
        self.assertEqual(len(dataset.dataset_sha256), 64)
        self.assertTrue(all(scene.labels[1] == 0 for scene in dataset.train_scenes))
        self.assertTrue(all(scene.labels[2] == 1 for scene in dataset.train_scenes))

    def test_manifest_path_traversal_fails_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = _build_repository(Path(temporary))
            manifest = repository / "data" / "scenes" / "S1" / "manifest.jsonl"
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            rows[0]["transcript_file"] = "../outside.txt"
            _write_jsonl(manifest, rows)
            with patch.object(benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60):
                with self.assertRaisesRegex(
                    TeachObsBenchmarkError, "unsafe transcript path"
                ):
                    load_teachobs_text_dataset(repository)


@unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is optional")
class TeachObsModelTests(unittest.TestCase):
    def test_fixed_protocol_is_deterministic_and_constant_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = _build_repository(Path(temporary))
            with patch.object(benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60):
                first = run_teachobs_text_benchmark(
                    repository, compare_sanitized=True
                )
                second = run_teachobs_text_benchmark(
                    repository, compare_sanitized=True
                )
        self.assertEqual(first, second)
        self.assertEqual(first["protocol"]["decision_threshold"], 0.5)
        self.assertFalse(
            first["protocol"]["test_labels_used_for_training_or_tuning"]
        )
        self.assertFalse(first["protocol"]["test_threshold_tuning_performed"])
        model_counts = first["runs"]["raw"]["feature_audit"][
            "model_kind_counts"
        ]
        self.assertEqual(model_counts["constant_0"], 1)
        self.assertEqual(model_counts["constant_1"], 1)
        self.assertEqual(model_counts["logistic_regression"], 37)
        self.assertEqual(len(first["runs"]["raw"]["per_label"]), 39)
        self.assertEqual(len(first["runs"]["raw"]["per_lesson"]), 7)
        self.assertIn("visual_macro_f1", first["runs"]["raw"]["metrics"])
        self.assertIn("nonvisual_macro_f1", first["runs"]["raw"]["metrics"])

    def test_changing_only_test_gold_does_not_change_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as first_temp, tempfile.TemporaryDirectory() as second_temp:
            first_repository = _build_repository(Path(first_temp), flip_test=False)
            second_repository = _build_repository(Path(second_temp), flip_test=True)
            with patch.object(benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60):
                first = run_teachobs_text_benchmark(
                    first_repository, compare_sanitized=False
                )
                second = run_teachobs_text_benchmark(
                    second_repository, compare_sanitized=False
                )
        first_predicted = [
            row["predicted_positive_count"]
            for row in first["runs"]["raw"]["per_label"]
        ]
        second_predicted = [
            row["predicted_positive_count"]
            for row in second["runs"]["raw"]["per_label"]
        ]
        self.assertEqual(first_predicted, second_predicted)
        self.assertEqual(
            first["runs"]["raw"]["feature_audit"],
            second["runs"]["raw"]["feature_audit"],
        )
        self.assertNotEqual(
            first["runs"]["raw"]["per_label"][0]["test_positive_count"],
            second["runs"]["raw"]["per_label"][0]["test_positive_count"],
        )

    def test_public_receipt_is_aggregate_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = _build_repository(Path(temporary))
            with patch.object(benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60):
                private = run_teachobs_text_benchmark(
                    repository, compare_sanitized=True
                )
            receipt = build_public_teachobs_benchmark_receipt(private)
        serialized = json.dumps(receipt, ensure_ascii=False)
        self.assertNotIn("per_lesson", serialized)
        self.assertNotIn("per_label", serialized)
        self.assertNotIn("lesson_id", serialized)
        self.assertNotIn("scene_no", serialized)
        self.assertNotIn("test-only-never-fit-token", serialized)
        self.assertNotIn("youtube", serialized.casefold())
        self.assertFalse(receipt["row_level_data_included"])
        self.assertFalse(receipt["lesson_or_scene_ids_included"])
        self.assertEqual(
            receipt["aggregate_runs"]["raw"]["metrics"],
            private["runs"]["raw"]["metrics"],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
