#!/usr/bin/env python3
"""Audit label coverage for the frozen, participant-disjoint DIPSER subset."""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teaching_skill_miner.recognition.dipser import (  # noqa: E402
    HttpRangeSource,
    build_expert_attention_band_ground_truth,
    build_expert_attention_ground_truth,
    fetch_zip_member,
    parse_dipser_timestamp,
    read_zip_directory,
)


TREE_URL = "https://www.scidb.cn/api/gin-sdb-filetree/public/file/childrenFileListByPath"
DATASET_ID = "7856c716c0cc4589a23ee4a23d8a0893"
VERSION = "V5"

# Fixed before labels are read. Each participant occurs in one session only.
FROZEN_SUBSET = {
    "group_01": {
        "experiment_01": ("subject_01", "subject_02"),
        "experiment_02": ("subject_03", "subject_04"),
        "experiment_03": ("subject_05", "subject_06"),
        "experiment_04": ("subject_07", "subject_08"),
        "experiment_05": ("subject_09", "subject_10"),
        "experiment_06": ("subject_12", "subject_13"),
        "experiment_07": ("subject_14", "subject_15"),
        "experiment_08": ("subject_16", "subject_17"),
        "experiment_09": ("subject_18", "subject_19"),
    },
    "group_02": {
        "experiment_01": ("subject_01", "subject_03"),
        "experiment_02": ("subject_07", "subject_11"),
        "experiment_03": ("subject_02", "subject_04"),
        "experiment_04": ("subject_05", "subject_06"),
        "experiment_05": ("subject_08", "subject_09"),
        "experiment_06": ("subject_10", "subject_12"),
        "experiment_07": ("subject_13", "subject_14"),
        "experiment_08": ("subject_15", "subject_16"),
        "experiment_09": ("subject_17", "subject_18"),
    },
    "group_03": {
        "experiment_01": ("subject_01", "subject_02"),
        "experiment_02": ("subject_03", "subject_04"),
        "experiment_03": ("subject_05", "subject_06"),
        "experiment_04": ("subject_07", "subject_09"),
        "experiment_05": ("subject_10", "subject_11"),
        "experiment_06": ("subject_08", "subject_12"),
        "experiment_07": ("subject_13",),
        "experiment_08": ("subject_14",),
        "experiment_09": ("subject_15", "subject_16"),
    },
}


def _post_json(payload: dict) -> dict:
    request = urllib.request.Request(
        TREE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _catalog() -> list[dict]:
    archives: list[dict] = []
    for group, experiments in FROZEN_SUBSET.items():
        for experiment, subjects in experiments.items():
            path = f"/{VERSION}/DIPSER/{group}/{experiment}"
            payload = _post_json(
                {
                    "dataSetId": DATASET_ID,
                    "version": VERSION,
                    "path": path,
                    "lastIndex": 0,
                    "pageSize": 200,
                }
            )
            if payload.get("code") != 20000:
                raise RuntimeError(f"ScienceDB tree query failed for {path}: {payload}")
            by_name = {item["fileName"]: item for item in payload["data"]}
            for subject in subjects:
                name = f"{subject}.zip"
                if name not in by_name:
                    raise RuntimeError(f"frozen archive is absent: {path}/{name}")
                item = by_name[name]
                archives.append(
                    {
                        "group_id": group,
                        "experiment_id": experiment,
                        "subject_id": subject,
                        "participant_id": f"{group}/{subject}",
                        "session_id": f"{group}/{experiment}",
                        "official_path": item["path"],
                        "file_id": item["id"],
                        "archive_size": int(item["size"]),
                        "archive_md5": item["md5"],
                    }
                )
    return archives


def _member_time(name: str) -> float:
    stem = Path(name).stem.replace("_", ":")
    return parse_dipser_timestamp(stem)


def _valid_member_times(directory: list, prefix: str) -> list[float]:
    values: list[float] = []
    for item in directory:
        if not item.name.startswith(prefix):
            continue
        try:
            values.append(_member_time(item.name))
        except Exception:
            # V5 contains a few auxiliary files such as ``out_...``. They are
            # not timestamped samples and are excluded before labels are read.
            continue
    return values


class _CachedTailSource:
    def __init__(self, size: int, start: int, payload: bytes) -> None:
        self.size = size
        self.start = start
        self.payload = payload

    def read(self, start: int, end: int) -> bytes:
        if start < self.start or end > self.size:
            raise RuntimeError("requested member is outside the cached label tail")
        return self.payload[start - self.start : end - self.start]


def _audit_archive(archive: dict, interval_seconds: int) -> dict:
    source = HttpRangeSource(
        f"https://china.scidb.cn/download?fileId={archive['file_id']}",
        size=archive["archive_size"],
        timeout_seconds=45,
    )
    directory = read_zip_directory(source)
    metadata_times = _valid_member_times(directory, "metadata/")
    watch_times = _valid_member_times(directory, "watch_sensors/")
    label_members = [item for item in directory if item.name.startswith("labels/")]
    label_names = [item.name for item in label_members]
    if not label_members:
        raise RuntimeError("missing label members")
    first_label_offset = min(item.local_header_offset for item in label_members)
    cached = _CachedTailSource(
        source.size,
        first_label_offset,
        source.read(first_label_offset, source.size),
    )
    labels = {item.name: fetch_zip_member(cached, item) for item in label_members}
    if not metadata_times or not watch_times:
        raise RuntimeError("missing metadata or watch members")
    start = max(min(metadata_times), min(watch_times) - 1.0)
    end = min(max(metadata_times), max(watch_times))
    first = math.ceil(start / interval_seconds) * interval_seconds
    timestamps = list(range(first, math.floor(end / interval_seconds) * interval_seconds + 1, interval_seconds))
    band_rows = build_expert_attention_band_ground_truth(labels, timestamps)
    median_rows = build_expert_attention_ground_truth(labels, timestamps)
    result = {
        **archive,
        "zip_member_count": len(directory),
        "metadata_member_count": len(metadata_times),
        "watch_member_count": len(watch_times),
        "label_members": sorted(label_names),
        "grid_count": len(timestamps),
        "four_expert_complete_count": len(median_rows),
        "three_of_four_band_consensus_count": len(band_rows),
        "band_counts": dict(sorted(Counter(row["label_name"] for row in band_rows).items())),
        "excluded_no_band_consensus_count": len(timestamps) - len(band_rows),
    }
    print(
        f"{archive['session_id']} {archive['subject_id']}: "
        f"grid={len(timestamps)} consensus={len(band_rows)} labels={result['band_counts']}",
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=15)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.interval_seconds < 1 or args.workers < 1:
        parser.error("interval and workers must be positive")
    archives = _catalog()
    if len({item["participant_id"] for item in archives}) != len(archives):
        raise RuntimeError("frozen subset unexpectedly repeats a participant")
    results: list[dict] = []
    errors: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_audit_archive, archive, args.interval_seconds): archive
            for archive in archives
        }
        for future in as_completed(futures):
            archive = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # continue to produce a complete audit trail
                errors.append({**archive, "error": f"{type(exc).__name__}: {exc}"})
                print(f"ERROR {archive['official_path']}: {exc}", flush=True)
    results.sort(key=lambda item: item["official_path"])
    errors.sort(key=lambda item: item["official_path"])
    total_counts = Counter()
    for item in results:
        total_counts.update(item["band_counts"])
    report = {
        "protocol": "dipser_frozen_label_coverage_v1",
        "dataset_id": DATASET_ID,
        "dataset_version": VERSION,
        "interval_seconds": args.interval_seconds,
        "archive_count_expected": len(archives),
        "archive_count_audited": len(results),
        "archive_error_count": len(errors),
        "participant_count": len({item["participant_id"] for item in archives}),
        "session_count": len({item["session_id"] for item in archives}),
        "participant_repeated_across_sessions": False,
        "grid_count": sum(item["grid_count"] for item in results),
        "three_of_four_band_consensus_count": sum(
            item["three_of_four_band_consensus_count"] for item in results
        ),
        "band_counts": dict(sorted(total_counts.items())),
        "archives": results,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("archives", "errors")}, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
