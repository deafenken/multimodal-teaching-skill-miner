from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from teaching_skill_miner.recognition.dipser import DipserFormatError, read_zip_directory
from teaching_skill_miner.recognition.dipser_experiment import (
    BulkTailCache,
    ScienceDBArchive,
    align_archive_windows,
    default_archive_paths,
    median_document_timestamp,
    query_sciencedb_archives,
    run_dipser_credible_experiment,
)


class MemoryRangeSource:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    @property
    def size(self) -> int:
        return len(self.payload)

    def read(self, start: int, end: int) -> bytes:
        return self.payload[start:end]


class BulkTailCacheTests(unittest.TestCase):
    def test_oversized_suffix_falls_back_to_explicit_member_ranges(self) -> None:
        payload = b"x" * 80_000
        cache = BulkTailCache(MemoryRangeSource(payload), footer_bytes=65_557)

        self.assertFalse(cache.prime_from(0, max_bytes=1_000))
        self.assertEqual(cache.read(10, 20), b"x" * 10)
        self.assertEqual(cache.network_range_reads, 2)


class FakeResponse:
    def __init__(self, payload: bytes, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def read(self) -> bytes:
        return self.payload

    def close(self) -> None:
        return None


def _watch(timestamp: str, value: float = 1.0) -> dict:
    def rows(dimensions: int) -> list[dict]:
        return [
            {
                "datetime": row_timestamp,
                **{
                    f"value{index}": value + index + offset
                    for index in range(dimensions)
                },
            }
            for offset, row_timestamp in (
                (0, timestamp),
                (1, timestamp[:-3] + "200"),
            )
        ]

    return {
        "data": {
            "Samsung HR None Wakeup Sensor": rows(1),
            "Samsung Linear Acceleration Sensor": rows(3),
            "LSM6DSO Gyroscope": rows(3),
            "Samsung Rotation Vector": rows(4),
            "OPT3007 Light": rows(1),
        }
    }


def _metadata(timestamp: str, yaw: float = 2.0) -> dict:
    del timestamp  # V5 metadata time is published in the member filename.
    return {
        # These forbidden values must never enter the stable visual feature list.
        "age": 22,
        "gender": "x",
        "race": "x",
        "person": {
            "face": {
                "facemesh": [{"x": 999}],
                "headpose": {"pose": {"pitch": 1.0, "yaw": yaw, "roll": 3.0}},
            },
            "body": {
                "body_pose": [
                    {
                        "x": 0.1 + index / 100.0,
                        "y": 0.2 + index / 100.0,
                        "z": 0.3 + index / 100.0,
                        "visibility": 0.9,
                        "presence": 0.8,
                    }
                    for index in range(33)
                ]
            },
        },
    }


def _archive_zip(*, start_second: int = 0, attention: int = 1) -> bytes:
    buffer = io.BytesIO()
    base = f"10:00:{start_second:02d}"
    next_second = start_second + 15
    with zipfile.ZipFile(buffer, "w") as archive:
        for labeler in range(1, 5):
            archive.writestr(
                f"labels/labeler_{labeler:02d}.json",
                json.dumps([{"datetime": f"{base}:000", "attention": attention}]),
                compress_type=zipfile.ZIP_DEFLATED,
            )
        archive.writestr(
            "labels/self_labeling.json",
            json.dumps([{"datetime": f"{base}:000", "attention": 5}]),
        )
        for second, suffix in ((start_second, "000"), (next_second, "000")):
            timestamp = f"10:00:{second:02d}:{suffix}"
            archive.writestr(
                f"watch_sensors/10_00_{second:02d}_{suffix}.json",
                json.dumps(_watch(timestamp, float(second + 1))),
                compress_type=zipfile.ZIP_DEFLATED,
            )
            # 0.2 seconds away: within the strict 0.6-second tolerance.
            metadata_timestamp = f"10:00:{second:02d}:300"
            archive.writestr(
                f"metadata/10_00_{second:02d}_300.json",
                json.dumps(_metadata(metadata_timestamp, float(second + 2))),
                compress_type=zipfile.ZIP_DEFLATED,
            )
    return buffer.getvalue()


class PreregistrationTests(unittest.TestCase):
    def test_default_is_52_unique_participants_across_27_sessions(self) -> None:
        paths = default_archive_paths()
        sessions = {"/".join(path.split("/")[:2]) for path in paths}
        participants = {
            f"{path.split('/')[0]}/{Path(path).stem}"
            for path in paths
        }

        self.assertEqual(len(paths), 52)
        self.assertEqual(len(sessions), 27)
        self.assertEqual(len(participants), 52)


class ScienceDBCatalogTests(unittest.TestCase):
    def test_post_query_validates_and_returns_requested_official_file(self) -> None:
        seen = {}

        def opener(request, timeout):
            seen["timeout"] = timeout
            seen["method"] = request.get_method()
            seen["body"] = json.loads(request.data)
            payload = {
                "code": 200,
                "data": {
                    "list": [
                        {
                            "fileName": "subject_01.zip",
                            "path": "/V5/DIPSER/group_01/experiment_01/subject_01.zip",
                            "id": "a" * 32,
                            "fileMd5": "b" * 32,
                            "fileSize": 12345,
                        }
                    ]
                },
            }
            return FakeResponse(json.dumps(payload).encode())

        result = query_sciencedb_archives(
            "group_01", "experiment_01", ["subject_01"], opener=opener
        )

        self.assertEqual(seen["method"], "POST")
        self.assertEqual(
            seen["body"]["path"], "/V5/DIPSER/group_01/experiment_01"
        )
        self.assertEqual(result[0].file_id, "a" * 32)
        self.assertEqual(result[0].md5, "b" * 32)
        self.assertEqual(result[0].size, 12345)

    def test_catalog_rejects_bad_md5_instead_of_accepting_unverified_file(self) -> None:
        def opener(request, timeout):
            del request, timeout
            return FakeResponse(
                json.dumps(
                    {
                        "data": [
                            {
                                "fileName": "subject_01.zip",
                                "fileId": "a" * 32,
                                "md5": "not-md5",
                                "size": 12345,
                            }
                        ]
                    }
                ).encode()
            )

        with self.assertRaisesRegex(DipserFormatError, "MD5"):
            query_sciencedb_archives(
                "group_01", "experiment_01", ["subject_01"], opener=opener
            )


class AlignmentTests(unittest.TestCase):
    def test_uses_actual_watch_median_and_nearest_actual_metadata_timestamp(self) -> None:
        payload = _archive_zip()
        source = MemoryRangeSource(payload)

        aligned, exclusions = align_archive_windows(
            source,
            read_zip_directory(source),
            interval_seconds=15.0,
            tolerance_seconds=0.6,
        )

        self.assertEqual(len(aligned), 2)
        self.assertEqual(exclusions, [])
        # watch rows are at .000 and .200 -> median .100, metadata at .300.
        self.assertAlmostEqual(aligned[0].watch_median_seconds, 36000.1)
        self.assertAlmostEqual(aligned[0].alignment_error_seconds, 0.2)

    def test_alignment_over_tolerance_is_audited_and_excluded(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "watch_sensors/10_00_00_000.json", json.dumps(_watch("10:00:00:000"))
            )
            archive.writestr(
                "metadata/10_00_02_000.json", json.dumps(_metadata("10:00:02:000"))
            )
        source = MemoryRangeSource(buffer.getvalue())

        aligned, exclusions = align_archive_windows(
            source, read_zip_directory(source), tolerance_seconds=0.6
        )

        self.assertEqual(aligned, [])
        self.assertEqual(exclusions[0]["reason"], "synchronization_tolerance_exceeded")

    def test_watch_filename_and_internal_clock_mismatch_is_excluded(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "watch_sensors/10_00_00_000.json",
                json.dumps(_watch("10:00:05:000")),
            )
            archive.writestr(
                "metadata/10_00_05_100.json",
                json.dumps(_metadata("10:00:05:100")),
            )
        source = MemoryRangeSource(buffer.getvalue())

        aligned, exclusions = align_archive_windows(
            source, read_zip_directory(source), watch_filename_tolerance_seconds=1.0
        )

        self.assertEqual(aligned, [])
        self.assertEqual(
            exclusions[0]["reason"],
            "watch_filename_internal_timestamp_mismatch",
        )

    def test_timestamp_comes_from_json_not_member_name(self) -> None:
        document = {
            "data": {
                "sensor": [
                    {"datetime": "10:00:04:000"},
                    {"datetime": "10:00:06:000"},
                ]
            }
        }
        self.assertEqual(median_document_timestamp(document), 36005.0)

    def test_publisher_time_field_wins_over_epoch_timestamp_clock(self) -> None:
        document = {
            "data": {
                "sensor": [
                    {
                        "time": "10:00:00:000",
                        "timestamp": 1_700_000_000_000,
                    },
                    {
                        "time": "10:00:02:000",
                        "timestamp": 1_700_000_002_000,
                    },
                ]
            }
        }

        self.assertEqual(median_document_timestamp(document), 36001.0)


class EndToEndSemanticsTests(unittest.TestCase):
    def test_artifacts_exclude_sensitive_features_and_never_claim_deployment(self) -> None:
        payload_by_path = {
            "group_01/experiment_01/subject_01.zip": _archive_zip(
                start_second=0, attention=1
            ),
            "group_01/experiment_02/subject_02.zip": _archive_zip(
                start_second=1, attention=5
            ),
        }

        def opener(request, timeout):
            del timeout
            body = json.loads(request.data)
            parent = body["path"].split("/DIPSER/", 1)[1]
            expected = next(path for path in payload_by_path if path.startswith(parent + "/"))
            subject = Path(expected).name
            payload = payload_by_path[expected]
            descriptor = {
                "fileName": subject,
                "filePath": body["path"],
                "fileId": hashlib.md5(expected.encode()).hexdigest(),
                "md5": hashlib.md5(payload).hexdigest(),
                "size": len(payload),
            }
            return FakeResponse(json.dumps({"data": [descriptor]}).encode())

        def source_factory(archive: ScienceDBArchive) -> MemoryRangeSource:
            relative = archive.official_path.split("/DIPSER/", 1)[-1]
            return MemoryRangeSource(payload_by_path[relative])

        with tempfile.TemporaryDirectory() as directory:
            progress_events = []
            report = run_dipser_credible_experiment(
                directory,
                archive_paths=list(payload_by_path),
                opener=opener,
                source_factory=source_factory,
                max_workers=2,
                min_valid_sessions=2,
                min_participants_for_claim=10,
                evaluation_kwargs={
                    "outer_splits": 2,
                    "inner_splits": 2,
                    "bootstrap_replicates": 20,
                    "permutation_replicates": 19,
                    "c_grid": (1.0,),
                },
                progress_callback=progress_events.append,
            )
            manifest = json.loads(Path(directory, "dataset_manifest.json").read_text())
            features = json.loads(Path(directory, "features.json").read_text())
            blocked = json.loads(
                Path(directory, "blocked_descriptive_report.json").read_text()
            )
            cached_progress = []

            def source_must_not_run(archive: ScienceDBArchive) -> MemoryRangeSource:
                raise AssertionError(f"unexpected network source for {archive.official_path}")

            cached_report = run_dipser_credible_experiment(
                directory,
                archive_paths=list(payload_by_path),
                opener=opener,
                source_factory=source_must_not_run,
                max_workers=1,
                min_valid_sessions=2,
                min_participants_for_claim=10,
                evaluation_kwargs={
                    "outer_splits": 2,
                    "inner_splits": 2,
                    "bootstrap_replicates": 20,
                    "permutation_replicates": 19,
                    "c_grid": (1.0,),
                },
                progress_callback=cached_progress.append,
            )

        self.assertFalse(report["deployment_accuracy_established"])
        self.assertFalse(report["formal_preregistration_available"])
        self.assertFalse(report["session_disjoint_accuracy_established"])
        self.assertFalse(report["session_disjoint_multimodal_gain_established"])
        self.assertFalse(report["participant_disjoint_multimodal_gain_established"])
        self.assertFalse(blocked["available"])
        self.assertFalse(blocked["claim_status"]["multimodal_gain_established"])
        self.assertIn("exactly 3 cohorts", blocked["failure_reason"])

        def established_values(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key.endswith("_established"):
                        yield item
                    yield from established_values(item)
            elif isinstance(value, list):
                for item in value:
                    yield from established_values(item)

        self.assertTrue(all(item is False for item in established_values(report)))
        self.assertEqual(len(progress_events), 2)
        self.assertEqual({event["status"] for event in progress_events}, {"ok"})
        self.assertEqual(cached_report["dataset_fingerprint"], report["dataset_fingerprint"])
        self.assertEqual({event["status"] for event in cached_progress}, {"cached"})
        self.assertIn("one site", report["deployment_accuracy_reason"])
        self.assertFalse(report["participant_disjoint_accuracy_established"])
        self.assertNotIn("teacher_id", manifest["records"][0])
        self.assertFalse(
            manifest["audit"]["identity_metadata_verified"]["teacher_id"]
        )
        names = features["feature_names_by_modality"]["visual"]
        self.assertFalse(any(token in name for name in names for token in ("age", "gender", "race", "mesh")))
        self.assertFalse(any(name.endswith("_available") for name in names))
        self.assertFalse(any(name.endswith("_count") for name in names))
        sensor_names = features["feature_names_by_modality"]["sensor"]
        self.assertFalse(
            any(name.endswith(("_present", "_count")) for name in sensor_names)
        )
        for record in manifest["records"]:
            self.assertEqual(len(record["content_sha256"]), 64)
            self.assertTrue(record["source"]["self_labeling_excluded"])
            self.assertEqual(
                record["source"]["ground_truth_source"],
                "three_band_agreement_of_four_publisher_expert_labelers",
            )
            self.assertEqual(len(record["source"]["expert_labeler_ids"]), 4)
            coverage = record["source"]["numeric_coverage"]
            self.assertEqual(coverage["body_pose_complete_landmark_count"], 33)
            self.assertTrue(
                all(
                    item["row_count"] == item["complete_row_count"] > 0
                    and item["row_count"] == item["timestamped_row_count"]
                    for item in coverage["watch"].values()
                )
            )


if __name__ == "__main__":
    unittest.main()
