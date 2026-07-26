from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from teaching_skill_miner.cli import main
from teaching_skill_miner.teachobs import (
    TEACHOBS_SOURCE_COMMIT,
    build_public_teachobs_receipt,
    download_and_audit_teachobs,
)


REPOSITORY = (
    Path(__file__).resolve().parents[1]
    / "artifacts/private/external_datasets/teachobs/repository"
)


@unittest.skipUnless(REPOSITORY.is_dir(), "private pinned TeachObs fixture unavailable")
class TeachObsImportTests(unittest.TestCase):
    def _local_fetch(self, relative_path: str, *, timeout: int) -> tuple[bytes, str]:
        del timeout
        payload = (REPOSITORY / relative_path).read_bytes()
        return (
            payload,
            "https://raw.githubusercontent.com/codingchild2424/teacherOps/"
            f"{TEACHOBS_SOURCE_COMMIT}/{relative_path}",
        )

    def test_imports_pinned_consensus_release_without_overclaiming(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "teaching_skill_miner.teachobs._fetch_bytes", side_effect=self._local_fetch
        ):
            output = Path(directory) / "private"
            audit = download_and_audit_teachobs(
                output,
                acknowledge_source_terms=True,
            )

            self.assertTrue(audit["import_complete"])
            self.assertFalse(audit["confirmatory_validation_ready"])
            self.assertEqual(audit["source_commit"], TEACHOBS_SOURCE_COMMIT)
            self.assertEqual(audit["lesson_count"], 30)
            self.assertEqual(audit["train_lesson_count"], 23)
            self.assertEqual(audit["test_lesson_count"], 7)
            self.assertEqual(audit["scene_count"], 5158)
            self.assertEqual(audit["code_count"], 39)
            self.assertEqual(audit["gold_status"], "provisional_consensus_gold")
            self.assertTrue(audit["all_code_definitions_blank"])
            self.assertFalse(
                audit["annotation_protocol_from_source"][
                    "coder_level_reliability_recomputable"
                ]
            )

            receipt = build_public_teachobs_receipt(
                audit,
                private_audit_path=output / "dataset_audit.json",
            )
            serialized = json.dumps(receipt, ensure_ascii=False)
            self.assertNotIn("youtube", serialized.casefold())
            self.assertNotIn(str(output), serialized)
            self.assertNotIn('"S2"', serialized)
            self.assertFalse(
                receipt["evidence"]["inter_rater_reliability_independently_recomputed"]
            )
            self.assertEqual(receipt["evidence"]["code_definition_count"], 0)

    def test_source_terms_acknowledgement_is_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            ValueError, "explicit acknowledgement"
        ):
            download_and_audit_teachobs(
                directory,
                acknowledge_source_terms=False,
            )

    def test_cli_help_and_missing_acknowledgement_fail_closed(self) -> None:
        with redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as raised:
            main(["fetch-teachobs", "--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("usage:", output.getvalue())

        with tempfile.TemporaryDirectory() as directory, redirect_stderr(
            io.StringIO()
        ) as error:
            status = main(["fetch-teachobs", "--output", directory])
        self.assertEqual(status, 1)
        self.assertIn("explicit acknowledgement", error.getvalue())


if __name__ == "__main__":
    unittest.main()
