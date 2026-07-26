#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teaching_skill_miner.public_multimodal_receipts import (  # noqa: E402
    write_public_multimodal_receipts,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build aggregate-only public validation receipts from private full-video "
            "multimodal and four-arm ablation artifacts."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/private/full_multimodal/dataset_manifest.semantic.json"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("artifacts/private/full_multimodal/data_audit.semantic.json"),
    )
    parser.add_argument(
        "--semantic-batch",
        type=Path,
        default=Path(
            "artifacts/private/full_multimodal/semantic_results/semantic_batch_receipt.json"
        ),
    )
    parser.add_argument(
        "--ablation-report",
        type=Path,
        default=Path(
            "artifacts/private/full_multimodal/ablation/ablation_report.json"
        ),
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path("artifacts/public/full_multimodal_validation_receipt.json"),
    )
    parser.add_argument(
        "--ablation-output",
        type=Path,
        default=Path("artifacts/public/multimodal_ablation_receipt.json"),
    )
    args = parser.parse_args()

    validation_path, ablation_path = write_public_multimodal_receipts(
        manifest_path=args.manifest,
        audit_path=args.audit,
        semantic_batch_path=args.semantic_batch,
        ablation_report_path=args.ablation_report,
        validation_output_path=args.validation_output,
        ablation_output_path=args.ablation_output,
    )
    print(validation_path.resolve())
    print(ablation_path.resolve())
    print("Public receipts contain aggregates and hashes only; no media/text/frames/embeddings.")
    print("Accuracy, Precision, Recall, F1, deployment accuracy, and learner effects remain unestablished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
