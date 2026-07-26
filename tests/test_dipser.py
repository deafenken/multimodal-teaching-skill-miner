from __future__ import annotations

import io
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

from teaching_skill_miner.recognition.dipser import (
    DIPSER_DATASET_ID,
    DipserFormatError,
    HttpRangeSource,
    build_expert_attention_band_ground_truth,
    build_expert_attention_ground_truth,
    build_source_provenance,
    extract_visual_pose_features,
    extract_watch_features,
    fetch_selected_members,
    fetch_zip_member,
    parse_official_archive_identity,
    read_zip_directory,
    reconstruct_expert_labels,
    safe_member_destination,
    write_selected_members,
)


class MemoryRangeSource:
    def __init__(self, payload: bytes) -> None:
        self.payload = bytearray(payload)
        self.reads: list[tuple[int, int]] = []

    @property
    def size(self) -> int:
        return len(self.payload)

    def read(self, start: int, end: int) -> bytes:
        self.reads.append((start, end))
        return bytes(self.payload[start:end])


def _fixture_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.comment = b"auditable DIPSER fixture"
        archive.writestr(
            "labels/labeler_01.json",
            '[{"datetime":"10:00:00:000","attention":"3"}]',
            compress_type=zipfile.ZIP_DEFLATED,
        )
        archive.writestr(
            "watch_sensors/10_00_01.json",
            b'{"data":{}}',
            compress_type=zipfile.ZIP_STORED,
        )
    return buffer.getvalue()


class RemoteZipTests(unittest.TestCase):
    def test_reads_central_directory_and_fetches_store_and_deflate_with_ranges(self) -> None:
        source = MemoryRangeSource(_fixture_zip())

        members = read_zip_directory(source)
        self.assertEqual(
            [member.name for member in members],
            ["labels/labeler_01.json", "watch_sensors/10_00_01.json"],
        )
        self.assertEqual(members[0].compression, zipfile.ZIP_DEFLATED)
        self.assertEqual(members[1].compression, zipfile.ZIP_STORED)

        label = fetch_zip_member(source, members[0])
        watch = fetch_zip_member(source, members[1])
        self.assertEqual(json.loads(label)[0]["attention"], "3")
        self.assertEqual(json.loads(watch), {"data": {}})
        # The small fixture fits in one tail request; large DIPSER archives use
        # the same bounded (<= 65,557 byte) footer request.
        self.assertLessEqual(source.reads[0][1] - source.reads[0][0], 65_557)

    def test_selected_member_allowlist_never_fetches_unselected_payload(self) -> None:
        source = MemoryRangeSource(_fixture_zip())

        directory = read_zip_directory(source)
        watch_offset = next(
            member.local_header_offset
            for member in directory
            if member.name.startswith("watch_sensors/")
        )
        source.reads.clear()
        selected = fetch_selected_members(source, names=["labels/labeler_01.json"])

        self.assertEqual(list(selected), ["labels/labeler_01.json"])
        # Reads after footer + central directory belong only to the selected
        # member's local header and compressed bytes.
        self.assertNotIn(watch_offset, [start for start, _ in source.reads[2:]])
        with self.assertRaises(KeyError):
            fetch_selected_members(source, names=["labels/missing.json"])

    def test_crc_corruption_is_detected(self) -> None:
        source = MemoryRangeSource(_fixture_zip())
        stored = next(
            member for member in read_zip_directory(source) if member.compression == zipfile.ZIP_STORED
        )
        fixed = bytes(source.payload[stored.local_header_offset : stored.local_header_offset + 30])
        values = struct.unpack("<4s5H3I2H", fixed)
        data_start = stored.local_header_offset + 30 + values[9] + values[10]
        source.payload[data_start] ^= 0x01

        with self.assertRaisesRegex(DipserFormatError, "CRC-32 mismatch"):
            fetch_zip_member(source, stored)

    def test_zip64_footer_sentinel_fails_closed(self) -> None:
        payload = bytearray(_fixture_zip())
        eocd = payload.rfind(b"PK\x05\x06")
        struct.pack_into("<H", payload, eocd + 10, 0xFFFF)

        with self.assertRaisesRegex(DipserFormatError, "ZIP64"):
            read_zip_directory(MemoryRangeSource(bytes(payload)))

    def test_safe_extraction_rejects_traversal_and_writes_verified_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            written = write_selected_members(
                MemoryRangeSource(_fixture_zip()),
                root,
                names=["labels/labeler_01.json"],
            )
            self.assertEqual(len(written), 1)
            self.assertEqual(json.loads(written[0].read_text())[0]["attention"], "3")
            with self.assertRaisesRegex(DipserFormatError, "unsafe"):
                safe_member_destination(root, "../outside.json")


class FakeResponse:
    def __init__(self, payload: bytes, *, status: int, headers: dict[str, str]) -> None:
        self._payload = payload
        self.status = status
        self.headers = headers

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, amount: int = -1) -> bytes:
        return self._payload if amount < 0 else self._payload[:amount]


class HttpRangeTests(unittest.TestCase):
    def test_http_source_sends_exact_range_and_validates_content_range(self) -> None:
        observed_headers: list[dict[str, str]] = []

        def opener(request, timeout):
            del timeout
            observed_headers.append(dict(request.header_items()))
            return FakeResponse(
                b"bcd",
                status=206,
                headers={"Content-Range": "bytes 1-3/100"},
            )

        source = HttpRangeSource("https://example.test/archive.zip", size=100, opener=opener)
        self.assertEqual(source.read(1, 4), b"bcd")
        self.assertEqual(observed_headers[0]["Range"], "bytes=1-3")
        self.assertEqual(observed_headers[0]["Accept-encoding"], "identity")

    def test_http_source_rejects_server_that_ignores_range(self) -> None:
        def opener(request, timeout):
            del request, timeout
            return FakeResponse(b"whole archive", status=200, headers={})

        source = HttpRangeSource("https://example.test/archive.zip", size=100, opener=opener)
        with self.assertRaisesRegex(DipserFormatError, "ignored HTTP Range"):
            source.read(1, 4)


class DipserGroundTruthTests(unittest.TestCase):
    def test_attention_and_emotion_are_carried_forward_independently(self) -> None:
        labels = [
            {"datetime": "10:00:00:000", "attention": "2", "emotion": "1"},
            {"datetime": "10:00:01:000", "emotion": "8"},
            {"datetime": "10:00:02:000", "attention": "5"},
        ]

        rows = reconstruct_expert_labels(
            labels,
            ["09:59:59:999", "10:00:01:500", "10:00:02:000"],
        )

        self.assertEqual(rows[0]["attention"], None)
        self.assertEqual(rows[0]["emotion"], None)
        self.assertEqual(rows[1]["attention"], 2)
        self.assertEqual(rows[1]["emotion"], 8)
        self.assertEqual(rows[2]["attention"], 5)
        self.assertEqual(rows[2]["emotion"], 8)

    def test_four_expert_median_excludes_self_labeling_and_rounds_half_up(self) -> None:
        documents = {
            "labels/labeler_01.json": [
                {"datetime": "10:00:00:000", "attention": "1", "emotion": "1"},
                {"datetime": "10:00:01:000", "emotion": "9"},
            ],
            "labels/labeler_02.json": [
                {"datetime": "10:00:00:000", "attention": "2"},
                {"datetime": "10:00:02:000", "attention": "5"},
            ],
            "labels/labeler_03.json": [
                {"datetime": "10:00:00:000", "attention": "3"},
            ],
            "labels/labeler_04.json": [
                {"datetime": "10:00:00:000", "attention": "4"},
            ],
            "labels/self_labeling.json": [
                {"datetime": "10:00:00:000", "attention": "5"},
            ],
        }

        rows = build_expert_attention_ground_truth(
            documents,
            ["09:59:59:000", "10:00:01:500", "10:00:02:500"],
        )

        self.assertEqual(len(rows), 2)  # no truth before all four experts initialize
        self.assertEqual(rows[0]["expert_attention"]["labeler_01"], 1)
        self.assertEqual(rows[0]["attention_median_raw"], 2.5)
        self.assertEqual(rows[0]["attention"], 3)
        self.assertEqual(rows[0]["label"], 2)
        self.assertEqual(rows[1]["attention_median_raw"], 3.5)
        self.assertEqual(rows[1]["attention"], 4)
        self.assertTrue(rows[1]["self_labeling_excluded"])

    def test_missing_expert_fails_closed(self) -> None:
        documents = {
            f"labels/labeler_{number:02d}.json": [
                {"datetime": "10:00:00:000", "attention": "3"}
            ]
            for number in range(1, 4)
        }
        with self.assertRaisesRegex(DipserFormatError, "label files are required"):
            build_expert_attention_ground_truth(documents, ["10:00:01:000"])

    def test_fourth_publisher_expert_may_use_labeler_05_id(self) -> None:
        documents = {
            f"labels/labeler_{number:02d}.json": [
                {"datetime": "10:00:00:000", "attention": "4"}
            ]
            for number in (1, 2, 3, 5)
        }

        rows = build_expert_attention_band_ground_truth(
            documents, ["10:00:01:000"]
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["expert_labeler_ids"], ["01", "02", "03", "05"])
        self.assertEqual(rows[0]["expert_agreement"], 4)
        self.assertEqual(
            rows[0]["ground_truth_source"],
            "three_band_agreement_of_four_publisher_expert_labelers",
        )

    def test_three_band_truth_requires_three_of_four_experts(self) -> None:
        documents = {
            "labels/labeler_01.json": [
                {"datetime": "10:00:00:000", "attention": "1"},
                {"datetime": "10:00:02:000", "attention": "1"},
            ],
            "labels/labeler_02.json": [
                {"datetime": "10:00:00:000", "attention": "2"},
                {"datetime": "10:00:02:000", "attention": "2"},
            ],
            "labels/labeler_03.json": [
                {"datetime": "10:00:00:000", "attention": "2"},
                {"datetime": "10:00:02:000", "attention": "4"},
            ],
            "labels/labeler_04.json": [
                {"datetime": "10:00:00:000", "attention": "5"},
                {"datetime": "10:00:02:000", "attention": "5"},
            ],
            "labels/self_labeling.json": [
                {"datetime": "10:00:00:000", "attention": "5"}
            ],
        }

        rows = build_expert_attention_band_ground_truth(
            documents,
            ["10:00:01:000", "10:00:03:000"],
        )

        self.assertEqual(len(rows), 1)  # second timestamp is a 2--2 low/high split
        self.assertEqual(rows[0]["label"], 0)
        self.assertEqual(rows[0]["label_name"], "low")
        self.assertEqual(rows[0]["expert_agreement"], 3)
        self.assertTrue(rows[0]["self_labeling_excluded"])


class DipserFeatureTests(unittest.TestCase):
    def test_visual_feature_whitelist_excludes_demographic_and_face_identifiers(self) -> None:
        metadata = {
            "person": {
                "age": 22,
                "gender": "female",
                "race": {"white": 0.99},
                "face": {
                    "bounding_box": [1, 2, 3, 4],
                    "headpose": {"pose": {"pitch": 1, "yaw": -2, "roll": 3}},
                },
                "facemesh": [{"x": 0.1, "y": 0.2}],
                "body": {
                    "bounding_box": [1, 2, 3, 4],
                    "body_pose": [
                        {
                            "landmark": "left_shoulder",
                            "x": 0.2,
                            "y": 0.4,
                            "z": -0.1,
                            "visibility": 0.9,
                            "presence": 0.8,
                        },
                        {
                            "landmark": "right_shoulder",
                            "x": 0.8,
                            "y": 0.6,
                            "z": 0.1,
                            "visibility": 0.3,
                            "presence": 0.4,
                        },
                    ],
                },
            }
        }

        features = extract_visual_pose_features(metadata)

        self.assertEqual(features["head_pose_yaw"], -2.0)
        self.assertEqual(features["body_pose_landmark_count"], 2.0)
        self.assertEqual(features["body_pose_expected_landmark_count"], 33.0)
        self.assertEqual(features["body_pose_available"], 0.0)
        self.assertAlmostEqual(features["body_pose_x_mean"], 0.5)
        self.assertAlmostEqual(features["body_pose_visible_fraction_ge_0_5"], 0.5)
        forbidden = ("age", "gender", "race", "ethnicity", "face", "mesh", "hand")
        self.assertFalse(any(term in key for key in features for term in forbidden))

    def test_watch_features_have_fixed_statistics_and_audit_timestamps(self) -> None:
        watch = {
            "num_samples": {"samsung_linear_acceleration_sensor": 2},
            "data": {
                "samsung_hr_none_wakeup_sensor": [
                    {"timestamp": "10:00:00:000", "value0": 60},
                    {"timestamp": "10:00:01:000", "value0": 80},
                ],
                "samsung_linear_acceleration_sensor": [
                    {"timestamp": "10:00:00:000", "value0": 3, "value1": 4, "value2": 0},
                    {"timestamp": "10:00:00:010", "value0": 0, "value1": 0, "value2": 0},
                ],
            },
        }

        features = extract_watch_features(watch)

        self.assertEqual(features["watch_samsung_hr_none_wakeup_sensor_value0_mean"], 70.0)
        self.assertEqual(features["watch_samsung_linear_acceleration_sensor_sample_count"], 2.0)
        self.assertEqual(
            features["watch_samsung_linear_acceleration_sensor_magnitude3_mean"], 2.5
        )
        self.assertEqual(
            features[
                "watch_samsung_linear_acceleration_sensor_timestamped_sample_count"
            ],
            2.0,
        )
        self.assertEqual(features["watch_lsm6dso_gyroscope_present"], 0.0)

    def test_incomplete_watch_rows_fail_the_presence_audit(self) -> None:
        features = extract_watch_features(
            {
                "data": {
                    "samsung_linear_acceleration_sensor": [
                        {
                            "time": "10:00:00:000",
                            "value0": 1.0,
                            "value1": 2.0,
                        }
                    ]
                }
            }
        )

        prefix = "watch_samsung_linear_acceleration_sensor"
        self.assertEqual(features[f"{prefix}_sample_count"], 1.0)
        self.assertEqual(features[f"{prefix}_complete_sample_count"], 0.0)
        self.assertEqual(features[f"{prefix}_present"], 0.0)

    def test_missing_or_nonmonotonic_watch_timestamps_fail_presence_audit(self) -> None:
        prefix = "watch_samsung_hr_none_wakeup_sensor"
        missing = extract_watch_features(
            {"data": {"samsung_hr_none_wakeup_sensor": [{"value0": 72.0}]}}
        )
        descending = extract_watch_features(
            {
                "data": {
                    "samsung_hr_none_wakeup_sensor": [
                        {"time": "10:00:01:000", "value0": 72.0},
                        {"time": "10:00:00:000", "value0": 73.0},
                    ]
                }
            }
        )

        self.assertEqual(missing[f"{prefix}_timestamped_sample_count"], 0.0)
        self.assertEqual(missing[f"{prefix}_present"], 0.0)
        self.assertEqual(descending[f"{prefix}_timestamped_sample_count"], 2.0)
        self.assertEqual(descending[f"{prefix}_present"], 0.0)

    def test_legacy_datos_watch_key_is_supported(self) -> None:
        features = extract_watch_features(
            {
                "Datos": {
                    "opt3007_light": [{"timestamp": "10:00:00:000", "value0": 42}]
                }
            }
        )
        self.assertEqual(features["watch_opt3007_light_value0_mean"], 42.0)


class DipserIdentityTests(unittest.TestCase):
    def test_official_hierarchy_defines_session_and_stable_participant(self) -> None:
        first = parse_official_archive_identity(
            "/V5/DIPSER/group_01/experiment_01/subject_07.zip"
        )
        second = parse_official_archive_identity(
            "group_01/experiment_09/subject_07.zip"
        )

        self.assertEqual(first.session_id, "group_01/experiment_01")
        self.assertNotEqual(first.session_id, second.session_id)
        self.assertEqual(first.participant_id, second.participant_id)
        self.assertEqual(first.participant_id, "group_01/subject_07")

    def test_provenance_contains_sciencedb_file_proof_and_identity_basis(self) -> None:
        provenance = build_source_provenance(
            "group_03/experiment_09/subject_16.zip",
            source_file_id="8bdd2fbf3fb5938d952e3cc42ad1ce83",
            source_md5="197b03f8f0441f488cabce4f2e359953",
        )

        self.assertEqual(provenance["science_db_dataset_id"], DIPSER_DATASET_ID)
        self.assertEqual(provenance["session_id"], "group_03/experiment_09")
        self.assertEqual(provenance["participant_id"], "group_03/subject_16")
        self.assertTrue(provenance["identity_metadata_verified"]["session_id"])
        self.assertTrue(provenance["identity_metadata_verified"]["participant_id"])
        self.assertTrue(provenance["identity_metadata_verified"]["cohort_id"])
        self.assertTrue(provenance["identity_metadata_verified"]["site_id"])
        self.assertIn(provenance["source_file_id"], provenance["source_url"])
        self.assertIn("self labeling excluded", provenance["ground_truth_source"])
        self.assertFalse(provenance["teacher_id_available"])
        self.assertFalse(provenance["supports_cross_site_deployment_accuracy"])
        self.assertIn("publisher-declared", provenance["source_archive_md5_status"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
