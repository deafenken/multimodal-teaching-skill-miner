from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import teaching_skill_miner.teachobs_human_annotation as human
from teaching_skill_miner.cli import main
from teaching_skill_miner.io_utils import write_json
from teaching_skill_miner.release_audit import audit_release_path


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _repository_fixture(root: Path) -> tuple[Path, dict[str, str]]:
    repository = root / "repository"
    (repository / "data" / "track_a" / "gold" / "train").mkdir(parents=True)
    # Invalid content proves that the independent workflow does not parse gold.
    (repository / "data" / "track_a" / "gold" / "train" / "S1.jsonl").write_text(
        "THIS MUST NEVER BE READ\n", encoding="utf-8"
    )
    lessons = repository / "data" / "lessons.csv"
    lessons.parent.mkdir(parents=True, exist_ok=True)
    with lessons.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "private_note"))
        writer.writeheader()
        writer.writerows(
            [
                {"id": "S1", "private_note": "not exported"},
                {"id": "S2", "private_note": "not exported"},
            ]
        )

    coding_scheme = repository / "data" / "track_a" / "coding_scheme.json"
    scheme = {
        "schema_version": "test",
        "n_codes": 3,
        "codes": [
            {"name": "Code One", "group": "visual", "definition": ""},
            {"name": "Code Two", "group": "visual", "definition": ""},
            {"name": "Code Three", "group": "nonvisual", "definition": ""},
        ],
    }
    coding_scheme.write_text(json.dumps(scheme), encoding="utf-8")

    scene_receipts = []
    for lesson_id in ("S1", "S2"):
        path = repository / "data" / "scenes" / lesson_id / "manifest.jsonl"
        path.parent.mkdir(parents=True)
        rows = [
            {
                "id": lesson_id,
                "scene_no": scene_no,
                "start": float((scene_no - 1) * 15),
                "end": float(scene_no * 15),
                "transcript_file": "private-transcript-must-not-be-read.txt",
            }
            for scene_no in (1, 2)
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        payload = path.read_bytes()
        scene_receipts.append(
            {
                "lesson_id": lesson_id,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )

    digests = {
        "lessons": hashlib.sha256(lessons.read_bytes()).hexdigest(),
        "codes": hashlib.sha256(coding_scheme.read_bytes()).hexdigest(),
        "scenes": _canonical_sha256(scene_receipts),
    }
    return repository, digests


def _patch_fixture_constants(digests: dict[str, str]):
    return patch.multiple(
        human,
        EXPECTED_LESSON_COUNT=2,
        EXPECTED_SCENE_COUNT=4,
        EXPECTED_CODE_COUNT=3,
        EXPECTED_VISUAL_CODE_COUNT=2,
        EXPECTED_NONVISUAL_CODE_COUNT=1,
        EXPECTED_LESSONS_SHA256=digests["lessons"],
        EXPECTED_CODING_SCHEME_SHA256=digests["codes"],
        EXPECTED_SCENE_MANIFEST_SET_SHA256=digests["scenes"],
    )


def _write_operational_codebook(path: Path) -> Path:
    value = {
        "schema": human.OPERATIONAL_CODEBOOK_SCHEMA,
        "version": "independent-expert-v1",
        "codes": [
            {
                "name": name,
                "definition": f"Synthetic test definition for {name}.",
                "inclusion_criteria": [f"Synthetic inclusion for {name}."],
                "exclusion_criteria": [f"Synthetic exclusion for {name}."],
                "positive_examples": [f"Synthetic positive example for {name}."],
                "negative_examples": [f"Synthetic negative example for {name}."],
            }
            for name in ("Code One", "Code Two", "Code Three")
        ],
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _complete_assignment(
    source: Path,
    target: Path,
    *,
    rater_id: str,
    signed_name: str,
    values_by_token: dict[str, dict[str, int]],
) -> None:
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for index, row in enumerate(rows):
        if index == 0:
            row["rater_id"] = rater_id
            row["completed_by_real_human"] = "YES"
            row["independent_without_gold_or_predictions"] = "YES"
            row["signed_name"] = signed_name
            row["attested_at_utc"] = "2026-07-23T00:00:00Z"
        for code_name, value in values_by_token[row["item_token"]].items():
            row[f"label::{code_name}"] = str(value)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TeachObsHumanAnnotationTests(unittest.TestCase):
    def test_top_level_cli_stays_pending_without_operational_codebook(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, digests = _repository_fixture(root)
            media_root = root / "private-media"
            media_root.mkdir()
            output = root / "assignments"
            pending_receipt = root / "public" / "pending.json"
            with _patch_fixture_constants(digests), redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "prepare-teachobs-double-annotation",
                        "--repository",
                        str(repository),
                        "--output",
                        str(output),
                        "--public-receipt",
                        str(pending_receipt),
                        "--media-root",
                        str(media_root),
                        "--lesson-id",
                        "S1",
                    ]
                )
            self.assertEqual(exit_code, 0)
            manifest = json.loads(
                (output / "assignment_manifest.json").read_text(encoding="utf-8")
            )
            receipt = json.loads(pending_receipt.read_text(encoding="utf-8"))
            self.assertFalse(
                manifest["operational_codebook"][
                    "operational_definitions_complete"
                ]
            )
            self.assertFalse(
                manifest["operational_codebook"]["annotation_execution_ready"]
            )
            self.assertFalse(receipt["human_completion"])
            self.assertIsNone(receipt["agreement"]["pooled_binary_cohen_kappa"])

            completed_receipt = root / "public" / "completed.json"
            with (
                _patch_fixture_constants(digests),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()) as error,
            ):
                exit_code = main(
                    [
                        "analyze-teachobs-double-annotation",
                        "--manifest",
                        str(output / "assignment_manifest.json"),
                        "--assignment-a",
                        str(output / "assignment_A.csv"),
                        "--assignment-b",
                        str(output / "assignment_B.csv"),
                        "--output",
                        str(root / "agreement"),
                        "--public-receipt",
                        str(completed_receipt),
                    ]
                )
            self.assertEqual(exit_code, 1)
            self.assertIn("operational definitions", error.getvalue())
            self.assertFalse(completed_receipt.exists())

    def test_prepare_is_blind_stable_private_and_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, digests = _repository_fixture(root)
            media = root / "private-media" / "videos"
            media.mkdir(parents=True)
            (media / "S1.mp4").write_bytes(b"private media reference target")
            output = root / "private-assignments"
            with _patch_fixture_constants(digests):
                manifest = human.prepare_teachobs_double_annotation(
                    repository,
                    output,
                    lesson_ids=["S1"],
                    media_root=media.parent,
                    require_media=True,
                )
                receipt = human.build_pending_public_teachobs_annotation_receipt(
                    manifest
                )

            self.assertEqual(manifest["selection"]["scene_item_count"], 2)
            self.assertFalse(manifest["human_completion"])
            self.assertFalse(manifest["blindness"]["gold_labels_included"])
            self.assertFalse(
                manifest["blindness"]["official_gold_files_read_by_this_workflow"]
            )
            self.assertTrue(manifest["blindness"]["assignment_orders_differ"])
            self.assertFalse(receipt["human_completion"])
            self.assertIsNone(receipt["agreement"]["pooled_binary_cohen_kappa"])
            self.assertFalse(
                manifest["operational_codebook"][
                    "operational_definitions_complete"
                ]
            )
            self.assertFalse(
                manifest["operational_codebook"]["annotation_execution_ready"]
            )
            self.assertFalse(receipt["status"]["annotation_execution_ready"])
            self.assertTrue(
                (output / "operational_codebook_template.json").is_file()
            )
            with self.assertRaisesRegex(ValueError, "operational definitions"):
                human.analyze_teachobs_double_annotations(
                    output / "assignment_manifest.json",
                    output / "assignment_A.csv",
                    output / "assignment_B.csv",
                    root / "must-not-exist",
                )

            with (output / "assignment_A.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows_a = list(csv.DictReader(handle))
            with (output / "assignment_B.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows_b = list(csv.DictReader(handle))
            self.assertEqual(
                {row["item_token"] for row in rows_a},
                {row["item_token"] for row in rows_b},
            )
            self.assertNotEqual(
                [row["item_token"] for row in rows_a],
                [row["item_token"] for row in rows_b],
            )
            self.assertTrue(
                all(
                    value == ""
                    for row in rows_a
                    for key, value in row.items()
                    if key.startswith("label::")
                )
            )
            header = set(rows_a[0])
            label_headers = {key for key in header if key.startswith("label::")}
            self.assertFalse(any("gold" in key.casefold() for key in label_headers))
            self.assertFalse(
                any("prediction" in key.casefold() for key in label_headers)
            )
            self.assertFalse(any(path.suffix == ".mp4" for path in output.rglob("*")))

    def test_import_computes_kappa_and_blank_third_party_template(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, digests = _repository_fixture(root)
            output = root / "assignments"
            media = root / "private-media" / "videos"
            media.mkdir(parents=True)
            (media / "S1.mp4").write_bytes(b"private media reference target")
            operational = _write_operational_codebook(root / "operational.json")
            with _patch_fixture_constants(digests):
                manifest = human.prepare_teachobs_double_annotation(
                    repository,
                    output,
                    lesson_ids=["S1"],
                    media_root=media.parent,
                    operational_codebook_path=operational,
                )
                item_tokens = [row["item_token"] for row in manifest["items"]]
                values_a = {
                    item_tokens[0]: {"Code One": 0, "Code Two": 0, "Code Three": 0},
                    item_tokens[1]: {"Code One": 1, "Code Two": 1, "Code Three": 0},
                }
                values_b = {
                    item_tokens[0]: {"Code One": 0, "Code Two": 0, "Code Three": 0},
                    item_tokens[1]: {"Code One": 0, "Code Two": 1, "Code Three": 0},
                }
                returned_a = root / "returned_A.csv"
                returned_b = root / "returned_B.csv"
                _complete_assignment(
                    output / "assignment_A.csv",
                    returned_a,
                    rater_id="human_rater_a",
                    signed_name="Human A",
                    values_by_token=values_a,
                )
                _complete_assignment(
                    output / "assignment_B.csv",
                    returned_b,
                    rater_id="human_rater_b",
                    signed_name="Human B",
                    values_by_token=values_b,
                )
                result = human.analyze_teachobs_double_annotations(
                    output / "assignment_manifest.json",
                    returned_a,
                    returned_b,
                    root / "agreement",
                )
                public = human.build_completed_public_teachobs_annotation_receipt(
                    manifest, result["report"]
                )

            report = result["report"]
            self.assertTrue(report["annotation_validation"]["human_completion"])
            self.assertTrue(
                report["operational_codebook"][
                    "operational_definitions_complete"
                ]
            )
            self.assertEqual(report["overall"]["disagreement_count"], 1)
            self.assertEqual(report["per_label"]["Code Two"]["cohen_kappa"], 1.0)
            self.assertIsNone(report["per_label"]["Code Three"]["cohen_kappa"])
            self.assertTrue(public["human_completion"])
            self.assertFalse(public["status"]["third_party_adjudication_completed"])
            encoded = json.dumps(public)
            self.assertNotIn("human_rater_a", encoded)
            self.assertNotIn("Code One", encoded)
            self.assertNotIn("S1", encoded)
            self.assertNotIn("videos/S1.mp4", encoded)

            with Path(result["adjudication_template_path"]).open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["adjudicated_value"], "")
            self.assertEqual(rows[0]["adjudicator_id"], "")

            public_dir = root / "public"
            write_json(public_dir / "receipt.json", public)
            audit = audit_release_path(public_dir)
            self.assertTrue(audit["passed"], audit["findings"])

            tampered = json.loads(json.dumps(report))
            tampered["overall"]["disagreement_count"] = 0
            with _patch_fixture_constants(digests), self.assertRaisesRegex(
                ValueError, "report SHA-256 mismatch"
            ):
                human.build_completed_public_teachobs_annotation_receipt(manifest, tampered)

    def test_same_rater_and_invalid_or_missing_labels_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, digests = _repository_fixture(root)
            output = root / "assignments"
            media = root / "private-media" / "videos"
            media.mkdir(parents=True)
            (media / "S1.mp4").write_bytes(b"private media reference target")
            operational = _write_operational_codebook(root / "operational.json")
            with _patch_fixture_constants(digests):
                manifest = human.prepare_teachobs_double_annotation(
                    repository,
                    output,
                    lesson_ids=["S1"],
                    media_root=media.parent,
                    operational_codebook_path=operational,
                )
                tokens = [row["item_token"] for row in manifest["items"]]
                values = {
                    token: {"Code One": 0, "Code Two": 1, "Code Three": 0}
                    for token in tokens
                }
                returned_a = root / "returned_A.csv"
                returned_b = root / "returned_B.csv"
                _complete_assignment(
                    output / "assignment_A.csv",
                    returned_a,
                    rater_id="same_human",
                    signed_name="First Human",
                    values_by_token=values,
                )
                _complete_assignment(
                    output / "assignment_B.csv",
                    returned_b,
                    rater_id="same_human",
                    signed_name="Second Human",
                    values_by_token=values,
                )
                with self.assertRaisesRegex(ValueError, "different rater_id"):
                    human.analyze_teachobs_double_annotations(
                        output / "assignment_manifest.json",
                        returned_a,
                        returned_b,
                        root / "agreement",
                    )

                with returned_b.open(
                    "r", encoding="utf-8", newline=""
                ) as handle:
                    reader = csv.DictReader(handle)
                    fieldnames = list(reader.fieldnames or [])
                    rows = list(reader)
                rows[0]["rater_id"] = "different_human"
                rows[0]["label::Code One"] = ""
                with returned_b.open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaisesRegex(ValueError, "missing/invalid binary label"):
                    human.analyze_teachobs_double_annotations(
                        output / "assignment_manifest.json",
                        returned_a,
                        returned_b,
                        root / "agreement",
                    )


if __name__ == "__main__":
    unittest.main()
