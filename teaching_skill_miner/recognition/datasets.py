from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


OUC_CGE_LABELS = {"low": 0, "mid": 1, "medium": 1, "high": 2}
OUC_CGE_CLASS_NAMES = ["low", "medium", "high"]
KNOWN_OUC_CGE_PUBLIC_SAMPLE_SHA256 = {
    "high/view1.mp4": "ab6d972f254c1313bc8eabc87fc61192749392b96d624b0d110d10646dc797ab",
    "high/view10.mp4": "27c0f9fe0479eca4bafaedbb16ba33bed298c0bb0839c4f928711f0f48affdf4",
    "high/view11.mp4": "c4bea8e5ba9aa8509bd48a9196781d9ba9f4e48cfe5d296aa28d0d03d6674c11",
    "high/view12.mp4": "06eccf817095c1cbfa0cb3508e2e826d25a741fd26800a167cd60d74e2a36590",
    "high/view2.mp4": "afb64c80a439c682c63d054b1ec838adc4477614aeb27caa59e902e93019ca9c",
    "high/view3.mp4": "f8c6454f738062cabd1b811807044f5d82996be925cb27866b5c521756d017eb",
    "high/view4.mp4": "adb67d2295762506cb7a5971d0986178ba9d13fc5c231cece0081f32c22e68f7",
    "high/view5.mp4": "1477f25ef997c2e207bdb9d00d9544636a5ef569afe9df84a7a67fe0816de879",
    "high/view6.mp4": "be7150c4c9d1efeb3ee27cb51124104e4e72068364a31ce3b27b1926e152e5d3",
    "high/view7.mp4": "8d8339340cabc924a0303c834eedcd9f8004f2d983313afeb2d1914497b7f256",
    "high/view8.mp4": "b0b877c09a598ef7da9f0899b916e4ef670d69a48f42a9d4780900c7c9d18b24",
    "high/view9.mp4": "66337d7775fac954a2b0c760108b72afc2937f21a39a93dac202e6d614fd5b42",
    "low/view1.mp4": "b69c177796a2e62e3c3dbd99a0c07d6cd79a6257e7e2e331be7bb2afeed8edf7",
    "low/view10.mp4": "95aab229514aef3d71397c4533aba1845bea99031122510cf48e5a3bb6a257fd",
    "low/view11.mp4": "349ad4d7de19373c936ce38f7702d716f3daabfd4d386f82debc386f2412198a",
    "low/view12.mp4": "686d9e273e887dfb7d1cbb1e7ce2f37887d354f751641894a05cc92b699ac18c",
    "low/view2.mp4": "b69c177796a2e62e3c3dbd99a0c07d6cd79a6257e7e2e331be7bb2afeed8edf7",
    "low/view3.mp4": "14e4e3fd383df03ac22e5173382dc72ca0983a16eeadc6b77d8b7b01a5bd9b6f",
    "low/view4.mp4": "825204a84c927ff6eab3478a133fcecd05525c97fb0bd065765582798b8eb481",
    "low/view5.mp4": "6c7acc50b41cfefdb340c2522ecbeb874775aaac7de50e583609f06f7b097ae4",
    "low/view6.mp4": "20a40a869b9dcfd37efcc8120d20973c4c7e4e3c7e58038a48d110f48eeeeeb6",
    "low/view7.mp4": "d01a90480e3ae1a917c6acdc3f1c048e1b819af53ff6cdaeba08953deb2ae4fc",
    "low/view8.mp4": "baea5ab9040cc9039059562ee47abb10b98a5f8116a0dba81356ea84e66fa311",
    "low/view9.mp4": "662c442312392a6b6e5493f4f7af2a400331ffc743e960cf00cf982a9f014793",
    "mid/view1.mp4": "16cb7ed1b72c6225450c32aeea7092ff4dc3102e323ca9ee8fd58d1dfbb0f4b2",
    "mid/view10.mp4": "c0f5bb9eb4dc2a23d1df68ec12cd4b1078320ecfec79f425f3e48d5a6e28a041",
    "mid/view11.mp4": "96e7af4f46f1d481e3a608ea337da5ef8cdda0864e5f1d42d2f1eaece1c5d7a7",
    "mid/view12.mp4": "7c29b6b4c98d1cd660f3547857afa2af69f4ea070e1671bec5fe1c71e161e0a6",
    "mid/view2.mp4": "ab3564410995684335749f6e7d45af3785d34ec7b9fa355a3702908bd04bb5a0",
    "mid/view3.mp4": "897354274052f3af2293c56012e80afae502b7ec10d9a21f8fea4a17e5d07adf",
    "mid/view4.mp4": "18ae480b627855176451f96f2c97789058135eb753d66045c9f22e93679db1e9",
    "mid/view5.mp4": "9fa2745b919cdaf514627721d7a556ea5216be8b87f4a43146f2c88d7679fea4",
    "mid/view6.mp4": "ce045e149f991cc69981cd274eef41e914631ccc8faced6c817fe48c7954ac4a",
    "mid/view7.mp4": "74c844c08a7644e3f68fda47bb303600c78249053cdc5f63b5a4489a3bde95ef",
    "mid/view8.mp4": "ce2dbb4ce665aa0078230172fa76f15ba7c012fd5064df5e209341777817f5eb",
    "mid/view9.mp4": "6f85676a5f3713d88464ea348ae0fbc767c56908a0803364fd7884963b39a725",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("real classroom recognition requires ffprobe on PATH")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=start_time,duration,size:stream=index,codec_type,codec_name,width,height,sample_rate,channels,start_time,duration,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {"valid": False, "error": result.stderr[-500:]}
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    media_format = payload.get("format", {})
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    primary_video = video_streams[0] if video_streams else {}
    format_duration = float(media_format.get("duration", 0) or 0)
    video_start = float(primary_video.get("start_time", media_format.get("start_time", 0)) or 0)
    video_duration = float(primary_video.get("duration", 0) or 0)
    if video_duration <= 0:
        video_duration = max(0.0, format_duration - video_start)
    video_end = video_start + video_duration
    overlapping_audio_streams = []
    for stream in audio_streams:
        audio_start = float(stream.get("start_time", 0) or 0)
        audio_duration = float(stream.get("duration", 0) or 0)
        audio_end = audio_start + audio_duration
        overlap = max(0.0, min(video_end, audio_end) - max(video_start, audio_start))
        if audio_duration >= 0.2 and overlap >= 0.2:
            overlapping_audio_streams.append(stream)
    return {
        "valid": bool(video_streams) and video_duration > 0,
        "duration_seconds": round(video_duration, 3),
        "format_duration_seconds": round(format_duration, 3),
        "video_start_time_seconds": round(video_start, 6),
        "size_bytes": int(media_format.get("size", 0) or 0),
        "has_audio": bool(overlapping_audio_streams),
        "has_audio_stream": bool(audio_streams),
        "audio_stream_count": len(audio_streams),
        "overlapping_audio_stream_count": len(overlapping_audio_streams),
        "selected_audio_stream_index": (
            int(overlapping_audio_streams[0]["index"])
            if overlapping_audio_streams and "index" in overlapping_audio_streams[0]
            else None
        ),
        "streams": streams,
    }


def _filename_group(stem: str) -> str:
    match = re.search(r"(\d+)$", stem)
    return f"view_{int(match.group(1)):06d}" if match else f"stem_{stem.lower()}"


def discover_ouc_cge(dataset_root: str | Path) -> dict[str, Any]:
    """Create an audited manifest without using filenames as model features."""

    root = Path(dataset_root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    records: list[dict[str, Any]] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        label_name = directory.name.lower()
        if label_name not in OUC_CGE_LABELS:
            continue
        label = OUC_CGE_LABELS[label_name]
        canonical_label = OUC_CGE_CLASS_NAMES[label]
        for path in sorted(directory.glob("*.mp4")):
            probe = probe_video(path)
            digest = file_sha256(path)
            records.append(
                {
                    "sample_id": hashlib.sha256(
                        f"{canonical_label}/{path.name}/{digest}".encode("utf-8")
                    ).hexdigest()[:20],
                    "relative_path": str(path.relative_to(root)),
                    "label": label,
                    "label_name": canonical_label,
                    "group_id": _filename_group(path.stem),
                    "video_sha256": digest,
                    **probe,
                }
            )
    if not records:
        raise ValueError(f"no OUC-CGE videos found under {root}")

    observed_sample_hashes = {
        record["relative_path"]: record["video_sha256"] for record in records
    }
    provenance_verified = observed_sample_hashes == KNOWN_OUC_CGE_PUBLIC_SAMPLE_SHA256

    sha_seen: dict[str, dict[str, Any]] = {}
    unique_records: list[dict[str, Any]] = []
    exact_duplicates: list[dict[str, str]] = []
    duplicate_label_conflicts: list[dict[str, str]] = []
    for record in records:
        previous = sha_seen.get(record["video_sha256"])
        if previous:
            record["duplicate_of"] = previous["sample_id"]
            duplicate = {
                "sample_id": record["sample_id"],
                "duplicate_of": previous["sample_id"],
            }
            exact_duplicates.append(duplicate)
            if record["label"] != previous["label"]:
                duplicate_label_conflicts.append(
                    {
                        **duplicate,
                        "label": record["label_name"],
                        "duplicate_of_label": previous["label_name"],
                    }
                )
        else:
            sha_seen[record["video_sha256"]] = record
            unique_records.append(record)

    label_counts = Counter(item["label_name"] for item in unique_records)
    invalid = [item["sample_id"] for item in unique_records if not item.get("valid")]
    duration_outliers = [
        item["sample_id"]
        for item in unique_records
        if not 8.0 <= float(item.get("duration_seconds", 0)) <= 16.0
    ]
    fingerprint_payload = [
        (item["sample_id"], item["video_sha256"], item["label"], item["group_id"])
        for item in unique_records
    ]
    dataset_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    audio_usable_by_label = {
        label_name: sum(
            bool(item.get("has_audio"))
            for item in unique_records
            if item["label_name"] == label_name
        )
        for label_name in OUC_CGE_CLASS_NAMES
    }
    audit = {
        "dataset_id": "OUC-CGE" if provenance_verified else "OUC-CGE-compatible-unverified",
        "dataset_variant": "verified_public_sample" if provenance_verified else "full_or_custom_unverified",
        "dataset_root": str(root),
        "source_url": "https://osf.io/brd2c/",
        "paper_url": "https://doi.org/10.1038/s41597-025-04987-w",
        "task": "three_class_group_engagement",
        "provenance_verified": provenance_verified,
        "provenance_verification": (
            "all 36 relative paths and SHA-256 values match the pinned official OSF sample archives"
            if provenance_verified
            else "directory layout is compatible, but official origin and human labels were not verified"
        ),
        "real_classroom_video": provenance_verified,
        "independent_human_ground_truth": provenance_verified,
        "contains_identifiable_people": True,
        "usage_scope": "research/non-commercial; follow the dataset page and participant-consent terms",
        "raw_file_count": len(records),
        "usable_unique_count": len(unique_records),
        "label_counts": dict(sorted(label_counts.items())),
        "surrogate_group_count": len({item["group_id"] for item in unique_records}),
        "surrogate_group_kind": "label_independent_filename_index; not a verified source/session group",
        "exact_duplicate_count": len(exact_duplicates),
        "exact_duplicates": exact_duplicates,
        "duplicate_label_conflict_count": len(duplicate_label_conflicts),
        "duplicate_label_conflicts": duplicate_label_conflicts,
        "invalid_video_count": len(invalid),
        "invalid_sample_ids": invalid,
        "duration_outlier_count": len(duration_outliers),
        "duration_outlier_sample_ids": duration_outliers,
        "usable_overlapping_audio_count": sum(bool(item.get("has_audio")) for item in unique_records),
        "usable_overlapping_audio_by_label": audio_usable_by_label,
        "dataset_fingerprint": dataset_fingerprint,
        "sample_release_is_final_benchmark_split": False,
        "warnings": [
            "The public sample archives contain variable-duration sample videos rather than the paper's final 10-second split.",
            "Media windows use the primary video stream timeline, not the potentially misleading container duration.",
            "Some files contain nominal audio streams that do not overlap the classroom video and are treated as missing audio.",
            "OUC-CGE does not publish session IDs in the sample filenames, so session-disjoint generalization cannot be established.",
            "Exact duplicates are removed before modeling.",
        ],
    }
    return {"schema_version": "1.0", "audit": audit, "records": unique_records}


def resolve_record_path(dataset_root: str | Path, record: dict[str, Any]) -> Path:
    root = Path(dataset_root).resolve()
    candidate = (root / str(record["relative_path"])).resolve()
    if root not in candidate.parents:
        raise ValueError("record path escapes dataset root")
    return candidate
