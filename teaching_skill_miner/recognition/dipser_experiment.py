"""End-to-end, fail-closed DIPSER multimodal credibility experiment.

The importer intentionally keeps network discovery, byte-range ZIP access, data
alignment, feature extraction, and statistical evaluation as separate steps.
That makes the public ScienceDB source independently auditable and lets tests
replace every network operation with in-memory fixtures.

No request is made at import time.  The default roster is analysis-frozen from
publisher IDs and the published attendance table.  It is not a formal
preregistration because part of the label inventory had already been inspected
before this implementation was frozen.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import re
import struct
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..io_utils import ensure_private_directory
from .blocked_evaluation import (
    BlockedEvaluationError,
    blocked_descriptive_evaluation,
)
from .dipser import (
    DIPSER_DATASET_ID,
    DIPSER_DATASET_URL,
    DIPSER_LICENSE_SNAPSHOT_DATE,
    DIPSER_SITE_ID,
    DIPSER_VERSION,
    DipserFormatError,
    HttpRangeSource,
    RangeSource,
    ZipMember,
    build_expert_attention_band_ground_truth,
    build_source_provenance,
    extract_visual_pose_features,
    extract_watch_features,
    fetch_zip_member,
    parse_dipser_timestamp,
    parse_official_archive_identity,
    read_zip_directory,
)
from .strict_evaluation import (
    CredibilityError,
    nested_grouped_multimodal_evaluation,
    strict_dataset_fingerprint,
)


SCIENCEDB_TREE_URL = (
    "https://www.scidb.cn/api/gin-sdb-filetree/public/file/childrenFileListByPath"
)

# Each participant occurs in exactly one experiment.  Thus a session-disjoint
# fold also has zero participant overlap.  G3 experiments 07 and 08 have one
# eligible participant in the publisher attendance table, yielding 52 archives.
# The legacy public symbol name is retained for compatibility.  Semantically
# this is an analysis-frozen exploratory roster, not a formal preregistration.
DEFAULT_PREREGISTERED_ASSIGNMENT: dict[str, dict[str, tuple[str, ...]]] = {
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

CLASS_NAMES = ("low", "medium", "high")
REQUIRED_IDENTITIES = ("session_id", "participant_id", "cohort_id", "site_id")
ARCHIVE_CACHE_PROTOCOL = "dipser_processed_archive_cache_v3_complete_numeric_modalities"


def _source_implementation_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name("dipser.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    digest.update(platform.python_implementation().encode("utf-8"))
    digest.update(b"\0")
    digest.update(platform.python_version().encode("utf-8"))
    return digest.hexdigest()


PROCESSING_IMPLEMENTATION_FINGERPRINT = _source_implementation_fingerprint()


def _runtime_provenance() -> dict[str, Any]:
    source_files = {}
    for name in (
        "blocked_evaluation.py",
        "dipser.py",
        "dipser_experiment.py",
        "strict_evaluation.py",
        "metrics.py",
    ):
        path = Path(__file__).with_name(name)
        source_files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    packages = {}
    for distribution in ("numpy", "scipy", "scikit-learn"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    environment = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "zlib_version": getattr(__import__("zlib"), "ZLIB_VERSION", None),
        "packages": packages,
        "source_file_sha256": source_files,
    }
    canonical = json.dumps(environment, sort_keys=True, separators=(",", ":"))
    return {
        **environment,
        "evaluation_environment_fingerprint": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
    }


@dataclass(frozen=True)
class ScienceDBArchive:
    """One verified ScienceDB file-tree result."""

    official_path: str
    file_id: str
    md5: str
    size: int

    @property
    def download_url(self) -> str:
        return f"https://china.scidb.cn/download?fileId={self.file_id}"


@dataclass(frozen=True)
class AlignedWindow:
    """A watch document paired to publisher RGB-derived pose metadata."""

    watch_member: ZipMember
    watch_bytes: bytes
    watch_filename_seconds: float
    watch_median_seconds: float
    watch_filename_error_seconds: float
    watch_max_internal_deviation_seconds: float
    metadata_member: ZipMember
    metadata_bytes: bytes
    metadata_median_seconds: float
    alignment_error_seconds: float


class BulkTailCache:
    """Coalesce many small ZIP-member reads into at most two suffix reads.

    A small footer suffix is loaded at construction so EOCD and the central
    directory are normally served locally.  After directory inspection,
    :meth:`prime_from` loads the non-image asset suffix once when it is bounded.
    Above the hard byte cap it returns ``False`` so callers can fetch only the
    explicitly selected members instead of downloading an unexpected gigabyte.
    """

    def __init__(self, source: RangeSource, *, footer_bytes: int = 2 * 1024 * 1024) -> None:
        if footer_bytes < 65_557:
            raise ValueError("footer_bytes must cover the maximum ordinary ZIP footer search")
        self.source = source
        self._start = max(0, source.size - int(footer_bytes))
        self._payload = source.read(self._start, source.size)
        if len(self._payload) != source.size - self._start:
            raise DipserFormatError("short bulk footer read")
        self.network_range_reads = 1

    @property
    def size(self) -> int:
        return self.source.size

    @property
    def cached_start(self) -> int:
        return self._start

    def prime_from(self, start: int, *, max_bytes: int = 128 * 1024 * 1024) -> bool:
        if start < 0 or start > self.size:
            raise ValueError("invalid bulk-tail start")
        if start >= self._start:
            return True
        requested = self.size - start
        if requested > max_bytes:
            return False
        payload = self.source.read(start, self.size)
        if len(payload) != requested:
            raise DipserFormatError("short bulk asset-tail read")
        self._start = start
        self._payload = payload
        self.network_range_reads += 1
        return True

    def read(self, start: int, end: int) -> bytes:
        if start < 0 or end < start or end > self.size:
            raise ValueError(f"invalid byte range [{start}, {end})")
        if start >= self._start:
            relative_start = start - self._start
            relative_end = end - self._start
            return self._payload[relative_start:relative_end]
        # Used only if a central directory exceeds the initial footer cache;
        # real processing primes the full asset suffix before member reads.
        payload = self.source.read(start, end)
        self.network_range_reads += 1
        return payload


def default_archive_paths() -> tuple[str, ...]:
    """Return the immutable 52-archive / 27-recording frozen roster."""

    paths = []
    for group_id in sorted(DEFAULT_PREREGISTERED_ASSIGNMENT):
        for experiment_id in sorted(DEFAULT_PREREGISTERED_ASSIGNMENT[group_id]):
            for subject_id in DEFAULT_PREREGISTERED_ASSIGNMENT[group_id][experiment_id]:
                paths.append(f"{group_id}/{experiment_id}/{subject_id}.zip")
    if len(paths) != 52 or len(set(paths)) != 52:
        raise RuntimeError("default DIPSER frozen roster must contain 52 unique archives")
    identities = [parse_official_archive_identity(path) for path in paths]
    if len({item.session_id for item in identities}) != 27:
        raise RuntimeError("default DIPSER frozen roster must contain 27 recordings")
    if len({item.participant_id for item in identities}) != 52:
        raise RuntimeError("default DIPSER frozen roster must contain 52 participants")
    return tuple(paths)


def _response_bytes(response: Any) -> bytes:
    try:
        status = getattr(response, "status", None)
        if status is None and hasattr(response, "getcode"):
            status = response.getcode()
        if status not in (None, 0, 200):
            raise ConnectionError(f"ScienceDB file-tree API returned HTTP {status}")
        return bytes(response.read())
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    opener: Callable[..., Any],
    timeout_seconds: float,
) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "teaching-skill-miner-dipser/1.0",
        },
    )
    raw = _response_bytes(
        _open_with_retry(opener, request, timeout_seconds=timeout_seconds)
    )
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DipserFormatError("ScienceDB file-tree response is not valid UTF-8 JSON") from exc


def _open_with_retry(
    opener: Callable[..., Any],
    request: Any,
    *,
    timeout_seconds: float,
    attempts: int = 4,
) -> Any:
    """Open a request with bounded 429/5xx retry and Retry-After support."""

    for attempt in range(attempts):
        try:
            try:
                return opener(request, timeout=timeout_seconds)
            except TypeError:
                # Tiny injected openers often deliberately omit the timeout.
                return opener(request)
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code <= 599
            if not retryable or attempt + 1 >= attempts:
                raise
            retry_after = None
            if getattr(exc, "headers", None) is not None:
                retry_after = exc.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after is not None else 2.0**attempt
            except ValueError:
                delay = 2.0**attempt
            time.sleep(min(10.0, max(0.0, delay)))
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt + 1 >= attempts:
                raise
            time.sleep(min(10.0, 2.0**attempt))
    raise RuntimeError("unreachable retry state")  # pragma: no cover


def _retrying_opener(
    opener: Callable[..., Any], *, timeout_seconds: float = 30.0
) -> Callable[..., Any]:
    def wrapped(request: Any, timeout: float | None = None) -> Any:
        return _open_with_retry(
            opener,
            request,
            timeout_seconds=float(timeout if timeout is not None else timeout_seconds),
        )

    return wrapped


def _walk_objects(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_objects(child)


def _first_value(item: Mapping[str, Any], names: Sequence[str]) -> Any:
    lowered = {str(key).lower(): value for key, value in item.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _normalized_tree_file(
    item: Mapping[str, Any],
    *,
    parent_path: str,
) -> ScienceDBArchive | None:
    file_id = _first_value(item, ("fileId", "file_id", "id"))
    md5 = _first_value(item, ("md5", "fileMd5", "file_md5", "md5Value"))
    size = _first_value(item, ("size", "fileSize", "file_size"))
    path = _first_value(item, ("filePath", "fullPath", "path"))
    name = _first_value(item, ("fileName", "name"))
    if file_id is None or md5 is None or size is None or (path is None and name is None):
        return None
    path_text = str(path or "").replace("\\", "/").rstrip("/")
    name_text = str(name or "").replace("\\", "/").strip("/")
    if path_text.lower().endswith(".zip"):
        full_path = path_text
    elif name_text:
        full_path = f"{path_text or parent_path}/{name_text}"
    else:  # pragma: no cover - guarded above
        return None
    normalized = str(PurePosixPath(full_path)).replace("\\", "/")
    try:
        numeric_size = int(size)
    except (TypeError, ValueError) as exc:
        raise DipserFormatError(f"ScienceDB file size is not an integer for {normalized}") from exc
    file_id_text = str(file_id).strip()
    md5_text = str(md5).strip().lower()
    if not re.fullmatch(r"[0-9a-fA-F]{16,64}", file_id_text):
        raise DipserFormatError(f"ScienceDB fileId is malformed for {normalized}")
    if not re.fullmatch(r"[0-9a-f]{32}", md5_text):
        raise DipserFormatError(f"ScienceDB MD5 is malformed for {normalized}")
    if numeric_size < 22:
        raise DipserFormatError(f"ScienceDB ZIP size is invalid for {normalized}")
    return ScienceDBArchive(normalized, file_id_text, md5_text, numeric_size)


def query_sciencedb_archives(
    group_id: str,
    experiment_id: str,
    subject_ids: Sequence[str],
    *,
    version: str = DIPSER_VERSION,
    opener: Callable[..., Any] | None = None,
    timeout_seconds: float = 30.0,
) -> list[ScienceDBArchive]:
    """POST one official folder query and strictly resolve requested subjects."""

    group = str(group_id).lower()
    experiment = str(experiment_id).lower()
    requested = tuple(str(value).lower() for value in subject_ids)
    if not re.fullmatch(r"group_\d{2}", group):
        raise ValueError("group_id must look like group_01")
    if not re.fullmatch(r"experiment_\d{2}", experiment):
        raise ValueError("experiment_id must look like experiment_01")
    if not requested or any(not re.fullmatch(r"subject_\d{2}", value) for value in requested):
        raise ValueError("subject_ids must contain values such as subject_01")
    if len(set(requested)) != len(requested):
        raise ValueError("subject_ids must not contain duplicates")
    normalized_version = str(version).upper()
    if not re.fullmatch(r"V\d+", normalized_version):
        raise ValueError("version must look like V5")

    parent_path = f"/{normalized_version}/DIPSER/{group}/{experiment}"
    document = _post_json(
        SCIENCEDB_TREE_URL,
        {
            "dataSetId": DIPSER_DATASET_ID,
            "version": normalized_version,
            "path": parent_path,
            "lastIndex": 0,
            "pageSize": 200,
        },
        opener=opener or urllib.request.urlopen,
        timeout_seconds=timeout_seconds,
    )
    candidates: list[ScienceDBArchive] = []
    for item in _walk_objects(document):
        candidate = _normalized_tree_file(item, parent_path=parent_path)
        if candidate is not None:
            candidates.append(candidate)

    resolved: list[ScienceDBArchive] = []
    for subject in requested:
        expected_relative = f"{group}/{experiment}/{subject}.zip"
        matches = []
        for candidate in candidates:
            try:
                identity = parse_official_archive_identity(candidate.official_path)
            except DipserFormatError:
                continue
            canonical_path = (
                f"{normalized_version}/DIPSER/{expected_relative}".lower()
            )
            observed_path = candidate.official_path.replace("\\", "/").strip("/").lower()
            if (
                identity.official_relative_path == expected_relative
                and observed_path == canonical_path
            ):
                matches.append(candidate)
        # Recursive response walking can see the exact same mapping through no
        # more than one path, but deduplicate defensively by all verified fields.
        matches = list(
            {
                (value.official_path, value.file_id, value.md5, value.size): value
                for value in matches
            }.values()
        )
        if len(matches) != 1:
            raise DipserFormatError(
                f"ScienceDB tree must contain exactly one verified {expected_relative}; "
                f"found {len(matches)}"
            )
        resolved.append(matches[0])
    return resolved


def discover_preregistered_archives(
    paths: Sequence[str] | None = None,
    *,
    opener: Callable[..., Any] | None = None,
    timeout_seconds: float = 30.0,
) -> list[ScienceDBArchive]:
    """Discover custom or analysis-frozen archives, querying each folder once.

    The function name is retained for API compatibility; the accompanying
    report explicitly records that no formal preregistration is available.
    """

    requested_paths = tuple(paths) if paths is not None else default_archive_paths()
    if not requested_paths:
        raise ValueError("at least one archive path is required")
    by_session: dict[tuple[str, str], list[str]] = {}
    for path in requested_paths:
        identity = parse_official_archive_identity(path)
        by_session.setdefault((identity.group_id, identity.experiment_id), []).append(
            identity.subject_id
        )
    archives: list[ScienceDBArchive] = []
    for (group, experiment), subjects in sorted(by_session.items()):
        archives.extend(
            query_sciencedb_archives(
                group,
                experiment,
                sorted(subjects),
                opener=opener,
                timeout_seconds=timeout_seconds,
            )
        )
    return sorted(
        archives,
        key=lambda item: parse_official_archive_identity(item.official_path).official_relative_path,
    )


def build_dipser_catalog(
    paths: Sequence[str] | None = None,
    *,
    opener: Callable[..., Any] | None = None,
    timeout_seconds: float = 30.0,
) -> list[dict[str, Any]]:
    """Return a JSON-ready, publisher-verified catalog without reading ZIPs."""

    return [
        {
            "official_path": archive.official_path,
            "file_id": archive.file_id,
            "publisher_md5": archive.md5,
            "size": archive.size,
            "download_url": archive.download_url,
        }
        for archive in discover_preregistered_archives(
            paths, opener=opener, timeout_seconds=timeout_seconds
        )
    ]


_TIME_KEYS = {
    "datetime",
    "date_time",
    "timestamp",
    "time_stamp",
    "time",
    "recorded_at",
    "capture_time",
}


def _numeric_time_of_day(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    absolute = abs(value)
    # Epoch values occur in seconds, milliseconds, microseconds, or nanoseconds.
    if absolute >= 1e17:
        value /= 1e9
    elif absolute >= 1e14:
        value /= 1e6
    elif absolute >= 1e11:
        value /= 1e3
    if abs(value) >= 86_400:
        value %= 86_400
    return float(value) if 0 <= value < 86_400 else None


def _timestamp_seconds(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _numeric_time_of_day(float(value))
    text = str(value).strip()
    if not text:
        return None
    try:
        return parse_dipser_timestamp(text)
    except DipserFormatError:
        pass
    # ISO dates and filenames may prefix the time-of-day.
    match = re.search(r"(?:T|\s)(\d{1,2}:\d{2}:\d{2}(?:[.:]\d{1,6})?)", text)
    if match:
        try:
            return parse_dipser_timestamp(match.group(1))
        except DipserFormatError:
            return None
    try:
        return _numeric_time_of_day(float(text))
    except ValueError:
        return None


def _document_timestamp_values(
    document: bytes | str | Mapping[str, Any],
) -> list[float]:

    if isinstance(document, bytes):
        try:
            root = json.loads(document.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DipserFormatError("DIPSER member is not valid UTF-8 JSON") from exc
    elif isinstance(document, str):
        try:
            root = json.loads(document)
        except json.JSONDecodeError as exc:
            raise DipserFormatError("DIPSER member is not valid JSON") from exc
    else:
        root = document
    # This priority follows the publisher's reference ``_process_sensors``
    # implementation: V5 sensor rows use ``time`` parsed as %H:%M:%S:%f.
    # Lower-priority aliases cover older exports.  Never mix clock domains.
    values_by_priority: dict[int, list[float]] = {0: [], 1: [], 2: [], 3: []}
    priorities = {
        "time": 0,
        "datetime": 1,
        "date_time": 2,
        "recorded_at": 2,
        "capture_time": 2,
        "timestamp": 3,
        "time_stamp": 3,
    }

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized_key = str(key).lower()
                if normalized_key in _TIME_KEYS:
                    timestamp = _timestamp_seconds(child)
                    if timestamp is not None:
                        values_by_priority[priorities[normalized_key]].append(timestamp)
                if isinstance(child, (Mapping, list)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(root)
    values = next(
        (values_by_priority[level] for level in sorted(values_by_priority) if values_by_priority[level]),
        [],
    )
    if not values:
        raise DipserFormatError("JSON member contains no usable internal timestamps")
    return [float(value) for value in values]


def median_document_timestamp(document: bytes | str | Mapping[str, Any]) -> float:
    """Return the median of the highest-priority clock domain in a JSON document."""

    return float(median(_document_timestamp_values(document)))


def _member_filename_timestamp(name: str) -> float | None:
    stem = PurePosixPath(name).stem
    matches = list(
        re.finditer(r"(?<!\d)(\d{1,2})[-_:](\d{2})[-_:](\d{2})(?:[-_:](\d{1,6}))?(?!\d)", stem)
    )
    if not matches:
        return None
    match = matches[-1]
    fraction = match.group(4)
    timestamp = f"{match.group(1)}:{match.group(2)}:{match.group(3)}"
    if fraction:
        timestamp += f":{fraction}"
    try:
        return parse_dipser_timestamp(timestamp)
    except DipserFormatError:
        return None


def _circular_time_distance(left: float, right: float) -> float:
    direct = abs(left - right)
    return min(direct, 86_400.0 - direct)


def _is_expert_label(member: ZipMember) -> bool:
    return bool(re.search(r"(?:^|/)labeler_0[1-5]\.json$", member.name.lower()))


def _is_watch(member: ZipMember) -> bool:
    lower = member.name.lower()
    return lower.endswith(".json") and any(
        token in lower for token in ("watch_sensors/", "smartwatch/", "watch/")
    )


def _is_metadata(member: ZipMember) -> bool:
    lower = member.name.lower()
    return (
        lower.endswith(".json")
        and "metadata" in lower
        and not _is_watch(member)
        and not _is_expert_label(member)
        and "self_label" not in lower
    )


def _select_fixed_interval(
    timed: Sequence[tuple[float, ZipMember, bytes]], interval_seconds: float
) -> list[tuple[float, ZipMember, bytes]]:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    ordered = sorted(timed, key=lambda row: (row[0], row[1].name))
    if not ordered:
        return []
    selected = [ordered[0]]
    next_target = ordered[0][0] + interval_seconds
    for row in ordered[1:]:
        if row[0] + 1e-9 >= next_target:
            selected.append(row)
            # Preserve a fixed grid even if one document arrives late.
            while next_target <= row[0] + 1e-9:
                next_target += interval_seconds
    return selected


def align_archive_windows(
    source: RangeSource,
    directory: Sequence[ZipMember],
    *,
    interval_seconds: float = 15.0,
    tolerance_seconds: float = 0.6,
    watch_filename_tolerance_seconds: float = 1.0,
    watch_internal_tolerance_seconds: float = 1.0,
) -> tuple[list[AlignedWindow], list[dict[str, Any]]]:
    """Align sampled watch documents to nearest internally timestamped metadata."""

    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be non-negative")
    if watch_filename_tolerance_seconds < 0 or watch_internal_tolerance_seconds < 0:
        raise ValueError("watch timestamp tolerances must be non-negative")
    exclusions: list[dict[str, Any]] = []
    # Select sparsely from central-directory names before fetching payloads.
    # V5 watch archives contain roughly one JSON per second; fetching all of
    # them both wastes requests and readily triggers ScienceDB rate limiting.
    named_watch: list[tuple[float, ZipMember, bytes]] = []
    for member in sorted((item for item in directory if _is_watch(item)), key=lambda x: x.name):
        filename_time = _member_filename_timestamp(member.name)
        if filename_time is None:
            exclusions.append(
                {"member": member.name, "reason": "invalid_watch_member_filename_timestamp"}
            )
            continue
        named_watch.append((filename_time, member, b""))
    selected_named_watch = _select_fixed_interval(named_watch, interval_seconds)
    timed_watch: list[tuple[float, float, float, float, ZipMember, bytes]] = []
    for filename_time, member, _ in selected_named_watch:
        payload = fetch_zip_member(source, member)
        try:
            timestamp_values = _document_timestamp_values(payload)
        except DipserFormatError as exc:
            exclusions.append({"member": member.name, "reason": "invalid_watch_timestamp", "detail": str(exc)})
            continue
        timestamp = float(median(timestamp_values))
        filename_error = _circular_time_distance(filename_time, timestamp)
        if filename_error > watch_filename_tolerance_seconds + 1e-12:
            exclusions.append(
                {
                    "member": member.name,
                    "reason": "watch_filename_internal_timestamp_mismatch",
                    "filename_internal_error_seconds": round(filename_error, 6),
                    "tolerance_seconds": watch_filename_tolerance_seconds,
                }
            )
            continue
        max_internal_deviation = max(
            _circular_time_distance(value, timestamp) for value in timestamp_values
        )
        if max_internal_deviation > watch_internal_tolerance_seconds + 1e-12:
            exclusions.append(
                {
                    "member": member.name,
                    "reason": "watch_internal_timestamp_span_exceeded",
                    "max_deviation_from_median_seconds": round(
                        max_internal_deviation, 6
                    ),
                    "tolerance_seconds": watch_internal_tolerance_seconds,
                }
            )
            continue
        timed_watch.append(
            (
                filename_time,
                timestamp,
                filename_error,
                max_internal_deviation,
                member,
                payload,
            )
        )

    metadata_members = sorted((item for item in directory if _is_metadata(item)), key=lambda x: x.name)
    filename_times = {
        member.name: _member_filename_timestamp(member.name) for member in metadata_members
    }
    metadata_cache: dict[str, bytes] = {}
    aligned: list[AlignedWindow] = []
    for (
        watch_filename_time,
        watch_time,
        watch_filename_error,
        watch_internal_deviation,
        watch_member,
        watch_bytes,
    ) in timed_watch:
        # DIPSER V5 metadata JSON has no timestamp field.  Its publisher member
        # filename is the official capture timestamp; watch time still comes
        # from the actual sensor rows inside the selected JSON document.
        ranked = sorted(
            (member for member in metadata_members if filename_times[member.name] is not None),
            key=lambda member: (
                _circular_time_distance(float(filename_times[member.name]), watch_time),
                member.name,
            ),
        )
        if not ranked:
            exclusions.append(
                {"member": watch_member.name, "reason": "missing_timestamped_visual_metadata_member"}
            )
            continue
        metadata_member = ranked[0]
        metadata_time = float(filename_times[metadata_member.name])
        error = _circular_time_distance(watch_time, metadata_time)
        if error > tolerance_seconds + 1e-12:
            exclusions.append(
                {
                    "member": watch_member.name,
                    "reason": "synchronization_tolerance_exceeded",
                    "alignment_error_seconds": round(float(error), 6),
                    "tolerance_seconds": tolerance_seconds,
                }
            )
            continue
        if metadata_member.name not in metadata_cache:
            metadata_cache[metadata_member.name] = fetch_zip_member(source, metadata_member)
        metadata_bytes = metadata_cache[metadata_member.name]
        aligned.append(
            AlignedWindow(
                watch_member=watch_member,
                watch_bytes=watch_bytes,
                watch_filename_seconds=watch_filename_time,
                watch_median_seconds=watch_time,
                watch_filename_error_seconds=watch_filename_error,
                watch_max_internal_deviation_seconds=watch_internal_deviation,
                metadata_member=metadata_member,
                metadata_bytes=metadata_bytes,
                metadata_median_seconds=metadata_time,
                alignment_error_seconds=float(error),
            )
        )
    return aligned, exclusions


def _member_evidence(member: ZipMember) -> dict[str, Any]:
    return {
        "path": member.name,
        "crc32": f"{member.crc32:08x}",
        "compressed_size": member.compressed_size,
        "uncompressed_size": member.uncompressed_size,
    }


def _content_hash(metadata_bytes: bytes, watch_bytes: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(b"DIPSER_SYNCHRONIZED_METADATA_WATCH_V1\0")
    for payload in (metadata_bytes, watch_bytes):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _process_archive(
    archive: ScienceDBArchive,
    *,
    source_factory: Callable[[ScienceDBArchive], RangeSource],
    interval_seconds: float,
    tolerance_seconds: float,
    watch_filename_tolerance_seconds: float,
    watch_internal_tolerance_seconds: float,
    footer_cache_bytes: int,
    max_bulk_tail_bytes: int,
) -> dict[str, Any]:
    identity = parse_official_archive_identity(archive.official_path)
    raw_source = source_factory(archive)
    if raw_source.size != archive.size:
        raise DipserFormatError(
            f"HTTP object size {raw_source.size} disagrees with ScienceDB size {archive.size}"
        )
    source = BulkTailCache(raw_source, footer_bytes=footer_cache_bytes)
    directory = read_zip_directory(source)
    relevant_members = [
        member
        for member in directory
        if _is_expert_label(member) or _is_watch(member) or _is_metadata(member)
    ]
    if not relevant_members:
        raise DipserFormatError("archive has no labels, watch data, or visual metadata")
    earliest_relevant_offset = min(
        member.local_header_offset for member in relevant_members
    )
    requested_bulk_tail_bytes = source.size - earliest_relevant_offset
    bulk_tail_primed = source.prime_from(
        earliest_relevant_offset,
        max_bytes=max_bulk_tail_bytes,
    )
    label_members = sorted((item for item in directory if _is_expert_label(item)), key=lambda x: x.name)
    if len(label_members) != 4:
        raise DipserFormatError(
            f"archive requires exactly four expert label members; found {len(label_members)}"
        )
    label_documents: dict[str, bytes] = {}
    for member in label_members:
        payload = fetch_zip_member(source, member)
        try:
            json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DipserFormatError(
                f"invalid expert-label JSON in {member.name}: {exc}"
            ) from exc
        label_documents[member.name] = payload
    named_watch_for_coverage = [
        (timestamp, member, b"")
        for member in directory
        if _is_watch(member)
        for timestamp in [_member_filename_timestamp(member.name)]
        if timestamp is not None
    ]
    selected_watch_count = len(
        _select_fixed_interval(named_watch_for_coverage, interval_seconds)
    )
    windows, exclusions = align_archive_windows(
        source,
        directory,
        interval_seconds=interval_seconds,
        tolerance_seconds=tolerance_seconds,
        watch_filename_tolerance_seconds=watch_filename_tolerance_seconds,
        watch_internal_tolerance_seconds=watch_internal_tolerance_seconds,
    )
    timestamps = [window.watch_median_seconds for window in windows]
    truth = build_expert_attention_band_ground_truth(
        label_documents,
        timestamps,
        minimum_agreement=3,
    )
    truth_by_time = {float(row["seconds_of_day"]): row for row in truth}
    provenance = build_source_provenance(
        archive.official_path,
        source_file_id=archive.file_id,
        source_md5=archive.md5,
    )
    records: list[dict[str, Any]] = []
    visual_rows: list[list[float]] = []
    sensor_rows: list[list[float]] = []
    visual_names = tuple(
        name
        for name in extract_visual_pose_features({}).keys()
        if not name.endswith(("_available", "_count"))
    )
    sensor_names = tuple(
        name
        for name in extract_watch_features({"data": {}}).keys()
        if not name.endswith(("_present", "_count"))
    )
    label_evidence = [_member_evidence(member) for member in label_members]
    for window in windows:
        row_truth = truth_by_time.get(window.watch_median_seconds)
        if row_truth is None:
            exclusions.append(
                {
                    "member": window.watch_member.name,
                    "reason": "missing_three_of_four_expert_consensus",
                }
            )
            continue
        try:
            visual = extract_visual_pose_features(window.metadata_bytes)
            sensor = extract_watch_features(window.watch_bytes)
        except (DipserFormatError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            exclusions.append(
                {
                    "member": window.watch_member.name,
                    "metadata_member": window.metadata_member.name,
                    "reason": "invalid_required_modality_json",
                    "detail": str(exc),
                }
            )
            continue
        # Require the complete registered modality set, then remove all
        # availability/sample-count indicators from the learned matrices. This
        # prevents detector success and device dropout from becoming shortcuts.
        visual_present = bool(
            visual.get("head_pose_available") == 1.0
            and visual.get("body_pose_available") == 1.0
        )
        sensor_presence = [
            value for name, value in sensor.items() if name.endswith("_present")
        ]
        sensor_present = bool(sensor_presence and all(value == 1.0 for value in sensor_presence))
        if not visual_present or not sensor_present:
            exclusions.append(
                {
                    "member": window.watch_member.name,
                    "reason": "missing_required_modality",
                    "visual_present": visual_present,
                    "sensor_present": sensor_present,
                }
            )
            continue
        content_hash = _content_hash(window.metadata_bytes, window.watch_bytes)
        millis = int(round(window.watch_median_seconds * 1000.0))
        sample_id = f"{identity.participant_id}/{identity.experiment_id}/t{millis:08d}"
        records.append(
            {
                "sample_id": sample_id,
                "content_sha256": content_hash,
                "label": int(row_truth["label"]),
                "label_name": row_truth["label_name"],
                "session_id": identity.session_id,
                "participant_id": identity.participant_id,
                "cohort_id": identity.group_id,
                "activity_id": identity.experiment_id,
                "site_id": DIPSER_SITE_ID,
                "modalities": {"visual": True, "sensor": True},
                "source": {
                    **provenance,
                    "ground_truth_source": row_truth["ground_truth_source"],
                    "watch_member": _member_evidence(window.watch_member),
                    "metadata_member": _member_evidence(window.metadata_member),
                    "expert_label_members": label_evidence,
                    "watch_median_seconds_of_day": round(window.watch_median_seconds, 6),
                    "watch_filename_seconds_of_day": round(
                        window.watch_filename_seconds, 6
                    ),
                    "watch_filename_internal_error_seconds": round(
                        window.watch_filename_error_seconds, 6
                    ),
                    "watch_max_internal_deviation_seconds": round(
                        window.watch_max_internal_deviation_seconds, 6
                    ),
                    "metadata_median_seconds_of_day": round(window.metadata_median_seconds, 6),
                    "alignment_error_seconds": round(window.alignment_error_seconds, 6),
                    "alignment_tolerance_seconds": tolerance_seconds,
                    "expert_agreement": row_truth["expert_agreement"],
                    "expert_labeler_ids": row_truth["expert_labeler_ids"],
                    "numeric_coverage": {
                        "head_pose_complete": visual["head_pose_available"] == 1.0,
                        "body_pose_complete_landmark_count": int(
                            visual["body_pose_landmark_count"]
                        ),
                        "body_pose_expected_landmark_count": int(
                            visual["body_pose_expected_landmark_count"]
                        ),
                        "watch": {
                            name.removeprefix("watch_").removesuffix("_present"): {
                                "row_count": int(
                                    sensor[name.removesuffix("_present") + "_sample_count"]
                                ),
                                "complete_row_count": int(
                                    sensor[
                                        name.removesuffix("_present")
                                        + "_complete_sample_count"
                                    ]
                                ),
                                "timestamped_row_count": int(
                                    sensor[
                                        name.removesuffix("_present")
                                        + "_timestamped_sample_count"
                                    ]
                                ),
                            }
                            for name in sensor
                            if name.endswith("_present")
                        },
                    },
                    "self_labeling_excluded": True,
                },
            }
        )
        visual_rows.append([float(visual[name]) for name in visual_names])
        sensor_rows.append([float(sensor[name]) for name in sensor_names])
    exclusion_counts = dict(sorted(Counter(row["reason"] for row in exclusions).items()))
    return {
        "archive": archive,
        "records": records,
        "visual": visual_rows,
        "sensor": sensor_rows,
        "visual_names": visual_names,
        "sensor_names": sensor_names,
        "exclusions": exclusions,
        "coverage_flow": {
            "timestamped_watch_member_count": len(named_watch_for_coverage),
            "selected_watch_grid_count": selected_watch_count,
            "timestamp_and_metadata_aligned_count": len(windows),
            "three_of_four_expert_consensus_count": len(truth),
            "complete_numeric_multimodal_count": len(records),
            "exclusion_reason_counts": exclusion_counts,
        },
        "provenance": provenance,
        "range_cache": {
            "strategy": (
                "footer_then_bounded_non_image_suffix"
                if bulk_tail_primed
                else "footer_then_explicit_selected_member_ranges"
            ),
            "range_reads": source.network_range_reads,
            "cached_start": source.cached_start,
            "cached_bytes": source.size - source.cached_start,
            "requested_bulk_tail_bytes": requested_bulk_tail_bytes,
            "bulk_tail_primed": bulk_tail_primed,
            "max_bulk_tail_bytes": max_bulk_tail_bytes,
        },
    }


def _archive_cache_key(
    archive: ScienceDBArchive,
    *,
    interval_seconds: float,
    tolerance_seconds: float,
    watch_filename_tolerance_seconds: float,
    watch_internal_tolerance_seconds: float,
    footer_cache_bytes: int,
    max_bulk_tail_bytes: int,
) -> str:
    payload = {
        "protocol": ARCHIVE_CACHE_PROTOCOL,
        "processing_implementation_fingerprint": PROCESSING_IMPLEMENTATION_FINGERPRINT,
        "official_path": archive.official_path,
        "file_id": archive.file_id,
        "publisher_md5": archive.md5,
        "size": archive.size,
        "interval_seconds": interval_seconds,
        "tolerance_seconds": tolerance_seconds,
        "watch_filename_tolerance_seconds": watch_filename_tolerance_seconds,
        "watch_internal_tolerance_seconds": watch_internal_tolerance_seconds,
        "footer_cache_bytes": footer_cache_bytes,
        "max_bulk_tail_bytes": max_bulk_tail_bytes,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cache_result_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key != "archive"
    }


def _write_archive_cache(path: Path, cache_key: str, result: Mapping[str, Any]) -> None:
    result_payload = _cache_result_payload(result)
    canonical = json.dumps(
        result_payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    document = {
        "protocol": ARCHIVE_CACHE_PROTOCOL,
        "processing_implementation_fingerprint": PROCESSING_IMPLEMENTATION_FINGERPRINT,
        "cache_key": cache_key,
        "result_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "result": result_payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_archive_cache(path: Path, cache_key: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        result = document["result"]
        canonical = json.dumps(
            result,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if (
            document.get("protocol") != ARCHIVE_CACHE_PROTOCOL
            or document.get("processing_implementation_fingerprint")
            != PROCESSING_IMPLEMENTATION_FINGERPRINT
            or document.get("cache_key") != cache_key
            or document.get("result_sha256") != fingerprint
            or not isinstance(result, dict)
        ):
            return None
        return result
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None


def feature_bundle_fingerprint(
    records: Sequence[Mapping[str, Any]],
    feature_names: Mapping[str, Sequence[str]],
    features: Mapping[str, Sequence[Sequence[float]]],
) -> str:
    """Bind records, schemas, and exact float64 matrices after strict validation.

    The fingerprint is intentionally independent from JSON formatting.  It
    binds the ordered sample IDs, source-content hashes, labels, feature-name
    order, matrix shapes, and every feature value.  Malformed bundles fail
    closed instead of producing a hash for a partially interpreted payload.
    """

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("records must be a sequence")
    if not records:
        raise ValueError("records must not be empty")
    if not isinstance(feature_names, Mapping) or not isinstance(features, Mapping):
        raise ValueError("feature names and matrices must be mappings")

    expected_modalities = {"visual", "sensor"}
    name_modalities = {str(value) for value in feature_names}
    matrix_modalities = {str(value) for value in features}
    if name_modalities != expected_modalities:
        raise ValueError("feature names must contain exactly visual and sensor")
    if matrix_modalities != expected_modalities:
        raise ValueError("feature matrices must contain exactly visual and sensor")

    digest = hashlib.sha256()

    def add_text(value: Any) -> None:
        payload = str(value).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    add_text("dipser_feature_bundle_v1")
    for row_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {row_index} must be an object")
        for field in ("sample_id", "content_sha256", "label"):
            if field not in record:
                raise ValueError(f"record {row_index} lacks {field}")
        add_text(record["sample_id"])
        add_text(record["content_sha256"])
        add_text(record["label"])
    for modality in sorted(features):
        add_text(modality)
        names = feature_names[modality]
        matrix = features[modality]
        if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
            raise ValueError(f"{modality} feature names must be a sequence")
        normalized_names = [str(name) for name in names]
        if not normalized_names or any(not name for name in normalized_names):
            raise ValueError(f"{modality} feature names must be non-empty strings")
        if len(set(normalized_names)) != len(normalized_names):
            raise ValueError(f"{modality} feature names must be unique")
        if not isinstance(matrix, Sequence) or isinstance(matrix, (str, bytes)):
            raise ValueError(f"{modality} feature matrix must be a sequence")
        if len(matrix) != len(records):
            raise ValueError(
                f"{modality} feature row count does not match records"
            )

        for name in normalized_names:
            add_text(name)
        add_text(len(matrix))
        add_text(len(normalized_names))
        for row_index, row in enumerate(matrix):
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                raise ValueError(
                    f"{modality} feature row {row_index} must be a sequence"
                )
            if len(row) != len(normalized_names):
                raise ValueError(
                    f"{modality} feature row {row_index} has the wrong width"
                )
            for column_index, value in enumerate(row):
                if isinstance(value, bool):
                    raise ValueError(
                        f"{modality} feature [{row_index}, {column_index}] is boolean"
                    )
                try:
                    numeric = float(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"{modality} feature [{row_index}, {column_index}] is not numeric"
                    ) from exc
                if not math.isfinite(numeric):
                    raise ValueError(
                        f"{modality} feature [{row_index}, {column_index}] is not finite"
                    )
                digest.update(struct.pack(">d", numeric))
    return digest.hexdigest()


# Backward-compatible private alias for any local code written before the
# verifier became part of the public artifact-consumption API.
_feature_bundle_fingerprint = feature_bundle_fingerprint


def _feature_provenance(names: Mapping[str, Sequence[str]]) -> dict[str, dict[str, Any]]:
    descriptions = {
        "visual": "DIPSER V5 publisher RGB-derived head/body pose metadata whitelist; identity, pixels, face mesh, demographics and availability/count markers excluded",
        "sensor": "DIPSER V5 five-channel raw watch descriptive statistics; presence and all count markers excluded",
    }
    output = {}
    for modality in ("visual", "sensor"):
        schema = json.dumps(
            {
                "extractor": descriptions[modality],
                "feature_names": list(names[modality]),
                "processing_implementation_fingerprint": (
                    PROCESSING_IMPLEMENTATION_FINGERPRINT
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        output[modality] = {
            "extractor_id": f"dipser-{modality}-fixed-whitelist-v1",
            "extractor_fingerprint": hashlib.sha256(schema.encode("utf-8")).hexdigest(),
            "processing_implementation_fingerprint": (
                PROCESSING_IMPLEMENTATION_FINGERPRINT
            ),
            "frozen_before_evaluation": True,
            "uses_ground_truth_labels": False,
            "fitted_on_evaluation_records": False,
        }
    return output


def _false_evaluation(group_field: str, reason: str, group_count: int) -> dict[str, Any]:
    return {
        "protocol": "nested_grouped_multimodal_evaluation_v1",
        "outer_group_field": group_field,
        "group_count": group_count,
        "metrics": None,
        "cross_group_accuracy_established": False,
        "session_disjoint_accuracy_established": False,
        "participant_disjoint_accuracy_established": False,
        "multimodal_gain_established": False,
        "not_established_reason": reason,
    }


def _force_all_established_flags_false(value: Any) -> None:
    """Make the single-site DIPSER report structurally non-confirmatory.

    The generic evaluator exposes several ``*_established`` fields whose gates
    are appropriate only when the chosen groups can support the corresponding
    population claim.  DIPSER's 27 recordings are cross-classified inside one
    site and only three cohorts, so this wrapper keeps numerical availability in
    separately named fields and clears every confirmatory claim recursively.
    """

    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).endswith("_established"):
                value[key] = False
            else:
                _force_all_established_flags_false(item)
    elif isinstance(value, list):
        for item in value:
            _force_all_established_flags_false(item)


def _evaluate(
    manifest: Mapping[str, Any],
    features: Mapping[str, Sequence[Sequence[float]]],
    provenance: Mapping[str, Mapping[str, Any]],
    *,
    group_field: str,
    min_claim_groups: int,
    evaluation_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    groups = {str(record[group_field]) for record in manifest["records"]}
    if len(groups) < min_claim_groups:
        return _false_evaluation(
            group_field,
            f"only {len(groups)} valid {group_field} groups; at least {min_claim_groups} required",
            len(groups),
        )
    kwargs = dict(evaluation_kwargs)
    kwargs.setdefault("outer_splits", min(5, len(groups)))
    kwargs.setdefault("inner_splits", 3)
    kwargs.setdefault("min_claim_groups", min_claim_groups)
    try:
        return nested_grouped_multimodal_evaluation(
            manifest,
            features,
            feature_provenance=provenance,
            class_names=CLASS_NAMES,
            feature_sample_ids=[record["sample_id"] for record in manifest["records"]],
            group_field=group_field,
            statistical_group_field=group_field,
            required_identity_fields=REQUIRED_IDENTITIES,
            **kwargs,
        )
    except (CredibilityError, RuntimeError, ValueError) as exc:
        return _false_evaluation(group_field, str(exc), len(groups))


def run_dipser_credible_experiment(
    output_dir: str | Path,
    *,
    archive_paths: Sequence[str] | None = None,
    opener: Callable[..., Any] | None = None,
    source_factory: Callable[[ScienceDBArchive], RangeSource] | None = None,
    interval_seconds: float = 15.0,
    alignment_tolerance_seconds: float = 0.6,
    watch_filename_tolerance_seconds: float = 1.0,
    watch_internal_tolerance_seconds: float = 1.0,
    max_workers: int = 2,
    min_valid_sessions: int = 10,
    min_participants_for_claim: int = 10,
    evaluation_kwargs: Mapping[str, Any] | None = None,
    footer_cache_bytes: int = 2 * 1024 * 1024,
    max_bulk_tail_bytes: int = 128 * 1024 * 1024,
    range_timeout_seconds: float = 120.0,
    resume_archive_cache: bool = True,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build manifests/features and run fail-closed descriptive evaluations.

    Failures are recorded per archive and do not abort other workers.  Archive
    internals remain deterministic; concurrency exists only across archives.
    """

    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if range_timeout_seconds <= 0:
        raise ValueError("range_timeout_seconds must be positive")
    if (
        alignment_tolerance_seconds < 0
        or watch_filename_tolerance_seconds < 0
        or watch_internal_tolerance_seconds < 0
    ):
        raise ValueError("alignment and watch timestamp tolerances must be non-negative")
    if min_valid_sessions < 2 or min_participants_for_claim < 2:
        raise ValueError("claim thresholds must be at least two")
    output = ensure_private_directory(output_dir)
    cache_root = output / "archive_cache"
    base_opener = opener or urllib.request.urlopen
    retrying_opener = _retrying_opener(base_opener)
    archives = discover_preregistered_archives(
        archive_paths,
        opener=base_opener,
    )
    catalog_payload = [
        {
            "official_path": archive.official_path,
            "file_id": archive.file_id,
            "publisher_md5": archive.md5,
            "size": archive.size,
        }
        for archive in archives
    ]
    archive_catalog_fingerprint = hashlib.sha256(
        json.dumps(
            catalog_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if source_factory is None:
        def default_source_factory(archive: ScienceDBArchive) -> RangeSource:
            return HttpRangeSource(
                archive.download_url,
                size=archive.size,
                timeout_seconds=range_timeout_seconds,
                opener=retrying_opener,
            )

        source_factory = default_source_factory

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    pending: list[tuple[ScienceDBArchive, str, Path]] = []
    cache_hit_count = 0
    progress_index = 0
    for archive in archives:
        cache_key = _archive_cache_key(
            archive,
            interval_seconds=interval_seconds,
            tolerance_seconds=alignment_tolerance_seconds,
            watch_filename_tolerance_seconds=watch_filename_tolerance_seconds,
            watch_internal_tolerance_seconds=watch_internal_tolerance_seconds,
            footer_cache_bytes=footer_cache_bytes,
            max_bulk_tail_bytes=max_bulk_tail_bytes,
        )
        cache_path = cache_root / f"{cache_key}.json"
        cached = (
            _read_archive_cache(cache_path, cache_key)
            if resume_archive_cache
            else None
        )
        if cached is None:
            pending.append((archive, cache_key, cache_path))
            continue
        completed.append(cached)
        cache_hit_count += 1
        progress_index += 1
        if progress_callback is not None:
            progress_callback(
                {
                    "completed": progress_index,
                    "total": len(archives),
                    "status": "cached",
                    "official_path": archive.official_path,
                }
            )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_archive = {
            executor.submit(
                _process_archive,
                archive,
                source_factory=source_factory,
                interval_seconds=interval_seconds,
                tolerance_seconds=alignment_tolerance_seconds,
                watch_filename_tolerance_seconds=watch_filename_tolerance_seconds,
                watch_internal_tolerance_seconds=watch_internal_tolerance_seconds,
                footer_cache_bytes=footer_cache_bytes,
                max_bulk_tail_bytes=max_bulk_tail_bytes,
            ): (archive, cache_key, cache_path)
            for archive, cache_key, cache_path in pending
        }
        for future in as_completed(future_to_archive):
            archive, cache_key, cache_path = future_to_archive[future]
            progress_index += 1
            try:
                result = future.result()
                if resume_archive_cache:
                    _write_archive_cache(cache_path, cache_key, result)
                completed.append(result)
                progress = {
                    "completed": progress_index,
                    "total": len(archives),
                    "status": "ok",
                    "official_path": archive.official_path,
                }
            except Exception as exc:  # per-archive fail-closed audit boundary
                failures.append(
                    {
                        "official_path": archive.official_path,
                        "file_id": archive.file_id,
                        "reason": type(exc).__name__,
                        "detail": str(exc),
                    }
                )
                progress = {
                    "completed": progress_index,
                    "total": len(archives),
                    "status": "failed",
                    "official_path": archive.official_path,
                    "reason": type(exc).__name__,
                }
            if progress_callback is not None:
                progress_callback(progress)
    completed.sort(key=lambda item: item["provenance"]["official_relative_path"])
    failures.sort(key=lambda item: item["official_path"])

    records: list[dict[str, Any]] = []
    visual_rows: list[list[float]] = []
    sensor_rows: list[list[float]] = []
    exclusions: list[dict[str, Any]] = []
    feature_names: dict[str, list[str]] | None = None
    for result in completed:
        current_names = {
            "visual": list(result["visual_names"]),
            "sensor": list(result["sensor_names"]),
        }
        if feature_names is None:
            feature_names = current_names
        elif current_names != feature_names:
            failures.append(
                {
                    "official_path": result["provenance"]["official_relative_path"],
                    "reason": "FeatureSchemaMismatch",
                    "detail": "fixed feature schema changed across archives",
                }
            )
            continue
        records.extend(result["records"])
        visual_rows.extend(result["visual"])
        sensor_rows.extend(result["sensor"])
        exclusions.extend(
            {
                "official_path": result["provenance"]["official_relative_path"],
                **row,
            }
            for row in result["exclusions"]
        )
    records_and_features = sorted(
        zip(records, visual_rows, sensor_rows), key=lambda row: row[0]["sample_id"]
    )
    records = [row[0] for row in records_and_features]
    visual_rows = [row[1] for row in records_and_features]
    sensor_rows = [row[2] for row in records_and_features]
    feature_names = feature_names or {
        "visual": [
            name
            for name in extract_visual_pose_features({}).keys()
            if not name.endswith(("_available", "_count"))
        ],
        "sensor": [
            name
            for name in extract_watch_features({"data": {}}).keys()
            if not name.endswith(("_present", "_count"))
        ],
    }
    feature_provenance = _feature_provenance(feature_names)

    coverage_fields = (
        "timestamped_watch_member_count",
        "selected_watch_grid_count",
        "timestamp_and_metadata_aligned_count",
        "three_of_four_expert_consensus_count",
        "complete_numeric_multimodal_count",
    )
    coverage_per_archive = [
        {
            "official_path": result["provenance"]["official_relative_path"],
            "session_id": result["provenance"]["session_id"],
            "cohort_id": result["provenance"]["cohort_id"],
            "activity_id": result["provenance"]["experiment_id"],
            **result["coverage_flow"],
        }
        for result in completed
    ]
    coverage_totals = {
        field: sum(int(item[field]) for item in coverage_per_archive)
        for field in coverage_fields
    }
    coverage_totals["complete_over_selected_fraction"] = round(
        coverage_totals["complete_numeric_multimodal_count"]
        / max(1, coverage_totals["selected_watch_grid_count"]),
        6,
    )

    def aggregate_coverage(group_field: str) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in coverage_per_archive:
            grouped.setdefault(str(item[group_field]), []).append(item)
        output_rows = []
        for group_value, items in sorted(grouped.items()):
            row = {
                group_field: group_value,
                "archive_count": len(items),
                **{
                    field: sum(int(item[field]) for item in items)
                    for field in coverage_fields
                },
            }
            row["complete_over_selected_fraction"] = round(
                row["complete_numeric_multimodal_count"]
                / max(1, row["selected_watch_grid_count"]),
                6,
            )
            output_rows.append(row)
        return output_rows

    identity_verified = {
        "session_id": True,
        "participant_id": True,
        "cohort_id": True,
        "activity_id": True,
        "site_id": True,
        "teacher_id": False,
    }
    audit = {
        "dataset_id": "DIPSER",
        "science_db_dataset_id": DIPSER_DATASET_ID,
        "dataset_version": DIPSER_VERSION,
        "dataset_url": DIPSER_DATASET_URL,
        "license_snapshot_checked_on": DIPSER_LICENSE_SNAPSHOT_DATE,
        "platform_license_label": "CC BY 4.0",
        "usage_boundary": (
            "non-commercial academic/research use pending clarification of the "
            "paper Usage Notes; commercial use requires explicit custodian approval"
        ),
        "provenance_verified": True,
        "real_classroom_recording": True,
        "independent_human_ground_truth": True,
        "synchronized_modalities_verified": True,
        "identity_metadata_verified": identity_verified,
        "identity_metadata_source": "publisher ScienceDB group/experiment/subject hierarchy and DIPSER site description",
        "ground_truth_source": (
            "3-of-4 band consensus from exactly four publisher expert labeler "
            "files (IDs may be 01..05); self labels excluded"
        ),
        "synchronization_source": (
            "actual watch JSON sensor-row median timestamp paired to the nearest "
            "publisher metadata member filename timestamp"
        ),
        "synchronization_tolerance_seconds": alignment_tolerance_seconds,
        "watch_filename_tolerance_seconds": watch_filename_tolerance_seconds,
        "watch_internal_tolerance_seconds": watch_internal_tolerance_seconds,
        "fixed_interval_seconds": interval_seconds,
        "visual_feature_scope": "publisher RGB-derived de-identified head/body pose metadata only",
        "forbidden_feature_fields": [
            "age",
            "gender",
            "sex",
            "race",
            "ethnicity",
            "face_mesh",
            "participant_id",
            "session_id",
            "cohort_id",
            "activity_id",
            "file_path",
            "modality_presence_flags",
            "sensor_sample_count",
            "visual_detection_availability",
        ],
        "teacher_id_available": False,
        "planned_archive_count": len(archives),
        "successfully_read_archive_count": len(completed),
        "failed_archive_count": len(failures),
        "excluded_window_count": len(exclusions),
        "coverage_flow": {
            "totals": coverage_totals,
            "per_archive": coverage_per_archive,
            "per_session": aggregate_coverage("session_id"),
            "per_cohort": aggregate_coverage("cohort_id"),
            "per_activity": aggregate_coverage("activity_id"),
        },
        "source_archive_fingerprints": [
            {
                "official_path": archive.official_path,
                "file_id": archive.file_id,
                "size": archive.size,
                "publisher_md5": archive.md5,
            }
            for archive in archives
        ],
        "roster_mode": "default_frozen_52" if archive_paths is None else "custom_exploratory",
        "archive_catalog_fingerprint": archive_catalog_fingerprint,
        "range_strategy": {
            "name": "footer_then_bounded_suffix_or_explicit_selected_member_ranges",
            "footer_cache_bytes": footer_cache_bytes,
            "max_bulk_tail_bytes": max_bulk_tail_bytes,
            "per_successful_archive": [
                {
                    "official_path": result["provenance"]["official_relative_path"],
                    **result["range_cache"],
                }
                for result in completed
            ],
        },
        "processed_archive_cache": {
            "enabled": resume_archive_cache,
            "protocol": ARCHIVE_CACHE_PROTOCOL,
            "processing_implementation_fingerprint": (
                PROCESSING_IMPLEMENTATION_FINGERPRINT
            ),
            "cache_hit_count": cache_hit_count,
            "cache_miss_count": len(pending),
            "stores_raw_zip_or_image_bytes": False,
        },
    }
    audit["dataset_fingerprint"] = strict_dataset_fingerprint(records)
    manifest = {"schema_version": "2.0", "audit": audit, "records": records}
    features = {"visual": visual_rows, "sensor": sensor_rows}
    feature_bundle_fingerprint_value = feature_bundle_fingerprint(
        records, feature_names, features
    )
    runtime_provenance = _runtime_provenance()
    audit["feature_bundle_fingerprint"] = feature_bundle_fingerprint_value
    audit["runtime_provenance"] = runtime_provenance

    session_count = len({record["session_id"] for record in records})
    participant_count = len({record["participant_id"] for record in records})
    participant_sessions: dict[str, set[str]] = {}
    for record in records:
        participant_sessions.setdefault(record["participant_id"], set()).add(record["session_id"])
    repeated_participants = sorted(
        participant for participant, sessions in participant_sessions.items() if len(sessions) > 1
    )
    eval_kwargs = dict(evaluation_kwargs or {})
    strict_evaluation_protocol = {
        "outer_splits": int(eval_kwargs.get("outer_splits", 5)),
        "inner_splits": int(eval_kwargs.get("inner_splits", 3)),
        "c_grid": [
            float(value)
            for value in eval_kwargs.get("c_grid", (0.01, 0.1, 1.0, 10.0))
        ],
        "seed": int(eval_kwargs.get("seed", 2026)),
        "bootstrap_replicates": int(
            eval_kwargs.get("bootstrap_replicates", 2000)
        ),
        "permutation_replicates": int(
            eval_kwargs.get("permutation_replicates", 5000)
        ),
        "alpha": float(eval_kwargs.get("alpha", 0.05)),
        "session_minimum_group_count_for_raw_generic_gate": min_valid_sessions,
        "participant_minimum_group_count_for_raw_generic_gate": (
            min_participants_for_claim
        ),
        "inferential_interpretation_allowed_for_dipser": False,
    }
    if not records:
        session_report = _false_evaluation("session_id", "no valid synchronized records", 0)
    else:
        session_report = _evaluate(
            manifest,
            features,
            feature_provenance,
            group_field="session_id",
            min_claim_groups=min_valid_sessions,
            evaluation_kwargs=eval_kwargs,
        )
    fold_participant_overlap = any(
        fold.get("participant_overlap") for fold in session_report.get("folds", [])
    )
    if repeated_participants or fold_participant_overlap:
        session_report["session_disjoint_accuracy_established"] = False
        session_report["multimodal_gain_established"] = False
        session_report["participant_overlap_audit_passed"] = False
        session_report["not_established_reason"] = (
            "session folds contain participant overlap; this protocol requires participant-disjoint sessions"
        )
    else:
        session_report["participant_overlap_audit_passed"] = True

    if participant_count >= min_participants_for_claim and records:
        participant_report = _evaluate(
            manifest,
            features,
            feature_provenance,
            group_field="participant_id",
            min_claim_groups=min_participants_for_claim,
            evaluation_kwargs=eval_kwargs,
        )
    else:
        participant_report = _false_evaluation(
            "participant_id",
            f"only {participant_count} valid participants; at least {min_participants_for_claim} required",
            participant_count,
        )

    participant_session_overlap_folds = [
        int(fold["fold"])
        for fold in participant_report.get("folds", [])
        if fold.get("identity_overlap", {}).get("session_id")
    ]
    participant_report["session_overlap_audit_passed"] = not bool(
        participant_session_overlap_folds
    )
    participant_report["session_overlap_folds"] = participant_session_overlap_folds
    participant_report["participant_only_raw_multimodal_gain_gate_passed"] = bool(
        participant_report.get("multimodal_gain_established", False)
    )
    participant_report["participant_blocked_oof_estimate_available"] = bool(
        participant_report.get("metrics")
        and participant_report.get("cross_group_accuracy_established", False)
        and participant_report.get("feature_sample_order_verified", False)
    )
    if participant_session_overlap_folds:
        participant_report["participant_disjoint_accuracy_established"] = False
        participant_report["multimodal_gain_established"] = False
        participant_report["not_established_reason"] = (
            "participant-only folds share classroom sessions across train/test; "
            "this analysis is descriptive only"
        )

    cohort_count = len({record["cohort_id"] for record in records})
    activity_count = len({record["activity_id"] for record in records})
    raw_session_gain_gate = bool(session_report.get("multimodal_gain_established", False))
    session_blocked_estimate_available = bool(
        session_report.get("metrics")
        and session_report.get("cross_group_accuracy_established", False)
        and session_report.get("participant_overlap_audit_passed", False)
        and session_report.get("feature_sample_order_verified", False)
    )
    session_report["session_cluster_only_multimodal_gain_gate_passed"] = (
        raw_session_gain_gate
    )
    session_report["hierarchical_independence_audit_passed"] = False
    session_report["multimodal_gain_established"] = False
    session_report["session_disjoint_accuracy_established"] = False
    session_report["session_blocked_oof_estimate_available"] = (
        session_blocked_estimate_available
    )
    session_report["accuracy_acceptability_established"] = False
    session_report["claim_scope"] = (
        "single-site, same-design, complete-case descriptive OOF for held-out "
        "cohort-by-activity recordings; train folds may contain the same cohort "
        "or the same activity type"
    )
    session_report["not_established_reason"] = (
        "the 27 recordings are cross-classified within only 3 cohorts and 9 "
        "repeated activities, so session-level bootstrap/permutation does not "
        "support an inferential multimodal-gain or acceptable-accuracy claim"
    )

    if records:
        try:
            blocked_report = blocked_descriptive_evaluation(
                records,
                visual_rows,
                sensor_rows,
                sample_ids=[record["sample_id"] for record in records],
                logistic_c=1.0,
                max_iter=3000,
                random_state=2026,
            )
            availability_by_design = {}
            for design, evaluation in blocked_report["evaluations"].items():
                fusion_report = evaluation["modalities"]["fusion"]
                coverage = float(fusion_report["oof_coverage"])
                availability_by_design[design] = {
                    "estimate_available": bool(
                        evaluation["evaluated_fold_count"] > 0
                        and fusion_report["pooled_oof_sample_weighted"]["accuracy"]
                        is not None
                    ),
                    "complete_oof_coverage": math.isclose(
                        coverage, 1.0, rel_tol=0.0, abs_tol=1e-12
                    ),
                    "oof_coverage": coverage,
                }
            blocked_report["report_generated"] = True
            blocked_report["availability_by_design"] = availability_by_design
            blocked_report["estimate_available"] = all(
                item["estimate_available"]
                for item in availability_by_design.values()
            )
            blocked_report["complete_oof_coverage"] = all(
                item["complete_oof_coverage"]
                for item in availability_by_design.values()
            )
            blocked_report["available"] = blocked_report["estimate_available"]
        except (BlockedEvaluationError, RuntimeError, ValueError) as exc:
            blocked_report = {
                "evaluation_kind": "single_site_descriptive_blocked_oof",
                "available": False,
                "report_generated": False,
                "estimate_available": False,
                "complete_oof_coverage": False,
                "failure_type": type(exc).__name__,
                "failure_reason": str(exc),
                "contains_inferential_statistics": False,
                "claim_status": {
                    "multimodal_gain_established": False,
                    "cross_site_accuracy_established": False,
                    "deployment_accuracy_established": False,
                },
            }
    else:
        blocked_report = {
            "evaluation_kind": "single_site_descriptive_blocked_oof",
            "available": False,
            "report_generated": False,
            "estimate_available": False,
            "complete_oof_coverage": False,
            "failure_type": "NoValidRecords",
            "failure_reason": "no valid synchronized records",
            "contains_inferential_statistics": False,
            "claim_status": {
                "multimodal_gain_established": False,
                "cross_site_accuracy_established": False,
                "deployment_accuracy_established": False,
            },
        }

    analysis_configuration = {
        "archive_catalog_fingerprint": archive_catalog_fingerprint,
        "processing_implementation_fingerprint": (
            PROCESSING_IMPLEMENTATION_FINGERPRINT
        ),
        "feature_bundle_fingerprint": feature_bundle_fingerprint_value,
        "interval_seconds": interval_seconds,
        "alignment_tolerance_seconds": alignment_tolerance_seconds,
        "watch_filename_tolerance_seconds": watch_filename_tolerance_seconds,
        "watch_internal_tolerance_seconds": watch_internal_tolerance_seconds,
        "ground_truth": "attention 1-2 low, 3 medium, 4-5 high; >=3/4 experts; self excluded",
        "evaluation_kwargs": eval_kwargs,
        "strict_session_participant_protocol": strict_evaluation_protocol,
        "blocked_descriptive_protocol": {
            "designs": [
                "leave_one_cohort_out",
                "leave_one_activity_out",
                "double_blocked_cohort_activity",
            ],
            "fixed_logistic_c": 1.0,
            "max_iter": 3000,
            "random_state": 2026,
            "inferential_statistics": False,
        },
        "evaluation_environment_fingerprint": runtime_provenance[
            "evaluation_environment_fingerprint"
        ],
        "minimum_valid_sessions_for_descriptive_oof": min_valid_sessions,
        "minimum_participants_for_secondary_analysis": min_participants_for_claim,
    }
    analysis_configuration_fingerprint = hashlib.sha256(
        json.dumps(
            analysis_configuration,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    audit["analysis_configuration"] = analysis_configuration
    audit["analysis_configuration_fingerprint"] = (
        analysis_configuration_fingerprint
    )
    blocked_report["artifact_binding"] = {
        "dataset_fingerprint": audit["dataset_fingerprint"],
        "feature_bundle_fingerprint": feature_bundle_fingerprint_value,
        "analysis_configuration_fingerprint": analysis_configuration_fingerprint,
        "evaluation_environment_fingerprint": runtime_provenance[
            "evaluation_environment_fingerprint"
        ],
        "blocked_evaluation_source_sha256": runtime_provenance[
            "source_file_sha256"
        ]["blocked_evaluation.py"],
    }
    blocked_canonical = json.dumps(
        blocked_report, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    blocked_report["blocked_report_fingerprint"] = hashlib.sha256(
        blocked_canonical.encode("utf-8")
    ).hexdigest()

    report = {
        "protocol": "dipser_frozen_exploratory_multimodal_experiment_v2",
        "dataset_fingerprint": audit["dataset_fingerprint"],
        "feature_bundle_fingerprint": feature_bundle_fingerprint_value,
        "runtime_provenance": runtime_provenance,
        "analysis_configuration": analysis_configuration,
        "analysis_configuration_fingerprint": (
            analysis_configuration_fingerprint
        ),
        "analysis_status": "analysis-frozen exploratory evaluation after partial label audit",
        "formal_preregistration_available": False,
        "planned_archive_count": len(archives),
        "valid_record_count": len(records),
        "valid_session_count": session_count,
        "valid_participant_count": participant_count,
        "cohort_count": cohort_count,
        "activity_count": activity_count,
        "hierarchical_design": {
            "cohort_count": cohort_count,
            "activity_count": activity_count,
            "session_cells": session_count,
            "sessions_are_independent_exchangeable_units": False,
        },
        "archive_failures": failures,
        "coverage_flow": audit["coverage_flow"],
        "window_exclusions": sorted(
            exclusions, key=lambda row: (row["official_path"], row.get("member", ""), row["reason"])
        ),
        "session_disjoint": session_report,
        "participant_disjoint": participant_report,
        "blocked_descriptive": blocked_report,
        "session_blocked_oof_estimate_available": session_blocked_estimate_available,
        "session_disjoint_accuracy_established": False,
        "session_disjoint_multimodal_gain_established": bool(
            session_report.get("multimodal_gain_established", False)
        ),
        "participant_disjoint_accuracy_established": False,
        "participant_disjoint_multimodal_gain_established": False,
        "deployment_accuracy_established": False,
        "deployment_accuracy_reason": (
            "DIPSER is retrospective data from one site; no frozen model was prospectively "
            "evaluated at an independent target deployment site."
        ),
    }
    _force_all_established_flags_false(report)
    features_artifact = {
        "schema_version": "1.0",
        "dataset_fingerprint": audit["dataset_fingerprint"],
        "feature_bundle_fingerprint": feature_bundle_fingerprint_value,
        "fingerprint_algorithm": "dipser_feature_bundle_v1",
        "sample_ids": [record["sample_id"] for record in records],
        "feature_names_by_modality": feature_names,
        "matrices": features,
        "feature_provenance": feature_provenance,
    }

    for filename, payload in (
        ("dataset_manifest.json", manifest),
        ("features.json", features_artifact),
        ("blocked_descriptive_report.json", blocked_report),
        ("credible_report.json", report),
    ):
        (output / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def build_dipser_dataset(
    output_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build audited manifest/features without running statistical CV.

    This is useful for a staged real download: catalog first, then materialize
    and inspect labels/alignment, and only then run the compute-heavy nested
    evaluation.  The generated credibility report explicitly leaves every
    accuracy/gain claim false.
    """

    options = dict(kwargs)
    options["min_valid_sessions"] = 1_000_000_000
    options["min_participants_for_claim"] = 1_000_000_000
    report = run_dipser_credible_experiment(output_dir, **options)
    output = Path(output_dir)
    return {
        "manifest": json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8")),
        "features": json.loads((output / "features.json").read_text(encoding="utf-8")),
        "build_report": report,
    }


__all__ = [
    "AlignedWindow",
    "BulkTailCache",
    "CLASS_NAMES",
    "DEFAULT_PREREGISTERED_ASSIGNMENT",
    "SCIENCEDB_TREE_URL",
    "ScienceDBArchive",
    "align_archive_windows",
    "build_dipser_catalog",
    "build_dipser_dataset",
    "default_archive_paths",
    "discover_preregistered_archives",
    "feature_bundle_fingerprint",
    "median_document_timestamp",
    "query_sciencedb_archives",
    "run_dipser_credible_experiment",
]
