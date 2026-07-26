from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from teaching_skill_miner.visual_semantics import (
    SCHEMA_TASK,
    _projected_feature_tensor,
    validate_task_manifest,
)


class _TensorLike:
    def norm(self) -> None:
        return None


class _TransformersFiveOutput:
    def __init__(self, pooled: object) -> None:
        self.pooler_output = pooled


class VisualSemanticExtractorTests(unittest.TestCase):
    def test_transformers_four_tensor_is_returned_directly(self) -> None:
        tensor = _TensorLike()
        self.assertIs(_projected_feature_tensor(tensor), tensor)

    def test_transformers_five_uses_projected_pooler_output(self) -> None:
        pooled = object()
        self.assertIs(
            _projected_feature_tensor(_TransformersFiveOutput(pooled)), pooled
        )

    def test_tuple_fallback_uses_pooled_projection(self) -> None:
        pooled = object()
        self.assertIs(_projected_feature_tensor((object(), pooled)), pooled)

    def test_unrecognised_output_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "pooler_output"):
            _projected_feature_tensor(object())

    def test_task_manifest_requires_confined_hash_matched_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "frames/frame.jpg"
            frame.parent.mkdir()
            frame.write_bytes(b"synthetic-frame-only")
            digest = hashlib.sha256(frame.read_bytes()).hexdigest()
            task = {
                "schema": SCHEMA_TASK,
                "frame_count": 1,
                "frames": [
                    {
                        "frame_id": "frame-1",
                        "timestamp": 1.0,
                        "path": "frames/frame.jpg",
                        "sha256": digest,
                    }
                ],
            }
            checked = validate_task_manifest(task, root)
            self.assertEqual(checked[0]["absolute_path"], str(frame.resolve()))

            task["frames"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validate_task_manifest(task, root)

            task["frames"][0]["path"] = "../outside.jpg"
            with self.assertRaisesRegex(ValueError, "escapes"):
                validate_task_manifest(task, root)


if __name__ == "__main__":
    unittest.main()
