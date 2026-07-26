"""Auditable, privacy-conscious helpers for the public DIPSER dataset.

The ScienceDB release stores each subject/session as a large ZIP archive.  This
module reads the ordinary (non-ZIP64) central directory with HTTP byte ranges,
so callers can retrieve labels, metadata, and watch records without downloading
the image-heavy archive in full.  It also implements fixed, label-blind feature
extractors and a four-expert attention consensus policy.

No network request is made at import time.  The ZIP parser accepts any object
implementing :class:`RangeSource`, which makes every byte-level operation
testable with an in-memory archive.
"""

from __future__ import annotations

import binascii
import json
import math
import re
import struct
import urllib.error
import urllib.request
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from statistics import fmean, median, pstdev
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


DIPSER_DATASET_ID = "7856c716c0cc4589a23ee4a23d8a0893"
DIPSER_VERSION = "V5"
DIPSER_DOI = "https://doi.org/10.57760/SCIENCEDB.11541"
DIPSER_DATASET_URL = (
    "https://www.scidb.cn/en/detail?dataSetId=7856c716c0cc4589a23ee4a23d8a0893"
)
DIPSER_LICENSE_SNAPSHOT_DATE = "2026-07-20"
DIPSER_PAPER_URL = "https://arxiv.org/abs/2502.20209"
DIPSER_CODE_URL = (
    "https://github.com/luis-marquez/"
    "DIPSEER-A-Dataset-for-In-Person-Student-Emotion-and-Engagement-Recognition-in-the-Wild"
)
DIPSER_SITE_ID = "university_of_alicante_faculty_of_education"
DIPSER_BODY_LANDMARK_COUNT = 33

_EOCD_SIGNATURE = b"PK\x05\x06"
_CENTRAL_SIGNATURE = b"PK\x01\x02"
_LOCAL_SIGNATURE = b"PK\x03\x04"
_MAX_EOCD_SEARCH = 22 + 65_535
_UINT16_MAX = (1 << 16) - 1
_UINT32_MAX = (1 << 32) - 1


class DipserFormatError(ValueError):
    """Raised when DIPSER metadata or a remote ZIP is malformed/unsupported."""


class RangeSource(Protocol):
    """Random-access byte source used by the ZIP parser.

    ``read`` uses a half-open interval: ``start`` is inclusive and ``end`` is
    exclusive.  Implementations must return exactly ``end - start`` bytes.
    """

    @property
    def size(self) -> int: ...

    def read(self, start: int, end: int) -> bytes: ...


class HttpRangeSource:
    """Strict HTTP Range source that refuses silent whole-archive responses."""

    def __init__(
        self,
        url: str,
        *,
        size: int | None = None,
        timeout_seconds: float = 30.0,
        opener: Callable[..., Any] | None = None,
        user_agent: str = "teaching-skill-miner-dipser/1.0",
    ) -> None:
        if not str(url).startswith(("https://", "http://")):
            raise ValueError("HTTP Range URL must start with http:// or https://")
        self.url = str(url)
        self.timeout_seconds = float(timeout_seconds)
        self._opener = opener or urllib.request.urlopen
        self._user_agent = user_agent
        self._size = int(size) if size is not None else self._discover_size()
        if self._size < 22:
            raise DipserFormatError("remote object is too small to be a ZIP archive")

    @property
    def size(self) -> int:
        return self._size

    def _request(self, request: urllib.request.Request) -> Any:
        try:
            return self._opener(request, timeout=self.timeout_seconds)
        except urllib.error.URLError as exc:  # pragma: no cover - network dependent
            raise ConnectionError(f"failed to read {self.url}: {exc}") from exc

    @staticmethod
    def _header(response: Any, name: str) -> str | None:
        headers = getattr(response, "headers", {})
        if hasattr(headers, "get"):
            value = headers.get(name)
            return None if value is None else str(value)
        return None

    @staticmethod
    def _status(response: Any) -> int:
        status = getattr(response, "status", None)
        if status is None and hasattr(response, "getcode"):
            status = response.getcode()
        return int(status or 0)

    def _discover_size(self) -> int:
        head_request = urllib.request.Request(
            self.url,
            method="HEAD",
            headers={"User-Agent": self._user_agent, "Accept-Encoding": "identity"},
        )
        try:
            with self._request(head_request) as response:
                content_length = self._header(response, "Content-Length")
                if content_length is not None:
                    return int(content_length)
        except (ConnectionError, ValueError):
            pass

        probe = urllib.request.Request(
            self.url,
            headers={
                "Range": "bytes=0-0",
                "User-Agent": self._user_agent,
                "Accept-Encoding": "identity",
            },
        )
        with self._request(probe) as response:
            if self._status(response) != 206:
                raise DipserFormatError("server did not honor a one-byte HTTP Range probe")
            content_range = self._header(response, "Content-Range") or ""
            match = re.fullmatch(r"bytes\s+0-0/(\d+)", content_range.strip(), re.IGNORECASE)
            if not match:
                raise DipserFormatError("HTTP Range probe lacks a valid Content-Range total")
            response.read(1)
            return int(match.group(1))

    def read(self, start: int, end: int) -> bytes:
        if start < 0 or end < start or end > self.size:
            raise ValueError(f"invalid byte range [{start}, {end}) for object of size {self.size}")
        if start == end:
            return b""
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{end - 1}",
                "User-Agent": self._user_agent,
                "Accept-Encoding": "identity",
            },
        )
        with self._request(request) as response:
            if self._status(response) != 206:
                raise DipserFormatError(
                    "server ignored HTTP Range; refusing to download the whole DIPSER archive"
                )
            content_range = self._header(response, "Content-Range") or ""
            expected = f"bytes {start}-{end - 1}/{self.size}"
            if content_range.strip().lower() != expected.lower():
                raise DipserFormatError(
                    f"unexpected Content-Range {content_range!r}; expected {expected!r}"
                )
            payload = response.read()
        if len(payload) != end - start:
            raise DipserFormatError(
                f"short HTTP Range read: received {len(payload)}, expected {end - start}"
            )
        return payload


@dataclass(frozen=True)
class ZipMember:
    """An ordinary ZIP central-directory record."""

    name: str
    compression: int
    flags: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


def _read_exact(source: RangeSource, start: int, end: int) -> bytes:
    payload = source.read(start, end)
    if len(payload) != end - start:
        raise DipserFormatError(
            f"range source returned {len(payload)} bytes for [{start}, {end}); expected {end-start}"
        )
    return payload


def read_zip_directory(source: RangeSource) -> list[ZipMember]:
    """Read a single-disk, ordinary ZIP central directory from byte ranges.

    ZIP64 and multi-disk archives fail explicitly instead of suffering integer
    truncation.  DIPSER's per-subject archives use the supported ordinary ZIP
    layout.
    """

    if source.size < 22:
        raise DipserFormatError("object is too small to contain an end-of-central-directory record")
    tail_start = max(0, source.size - _MAX_EOCD_SEARCH)
    tail = _read_exact(source, tail_start, source.size)
    relative_eocd = tail.rfind(_EOCD_SIGNATURE)
    if relative_eocd < 0 or len(tail) - relative_eocd < 22:
        raise DipserFormatError("ZIP end-of-central-directory record not found")
    (
        signature,
        disk_number,
        central_disk,
        entries_on_disk,
        entry_count,
        central_size,
        central_offset,
        comment_length,
    ) = struct.unpack_from("<4s4H2IH", tail, relative_eocd)
    if signature != _EOCD_SIGNATURE:
        raise DipserFormatError("invalid ZIP end-of-central-directory signature")
    eocd_end = relative_eocd + 22 + comment_length
    if eocd_end != len(tail):
        raise DipserFormatError("trailing bytes or malformed ZIP comment after central directory")
    if (
        entries_on_disk == _UINT16_MAX
        or entry_count == _UINT16_MAX
        or central_size == _UINT32_MAX
        or central_offset == _UINT32_MAX
    ):
        raise DipserFormatError("ZIP64 archives are unsupported")
    if disk_number or central_disk or entries_on_disk != entry_count:
        raise DipserFormatError("multi-disk ZIP archives are unsupported")
    if central_offset + central_size > tail_start + relative_eocd:
        raise DipserFormatError("central-directory bounds overlap or exceed the ZIP footer")

    central = _read_exact(source, central_offset, central_offset + central_size)
    members: list[ZipMember] = []
    cursor = 0
    seen_names: set[str] = set()
    while cursor < len(central):
        if len(central) - cursor < 46:
            raise DipserFormatError("truncated ZIP central-directory entry")
        values = struct.unpack_from("<4s6H3I5H2I", central, cursor)
        if values[0] != _CENTRAL_SIGNATURE:
            raise DipserFormatError("invalid ZIP central-directory entry signature")
        flags = int(values[3])
        compression = int(values[4])
        crc32 = int(values[7])
        compressed_size = int(values[8])
        uncompressed_size = int(values[9])
        name_length = int(values[10])
        extra_length = int(values[11])
        entry_comment_length = int(values[12])
        start_disk = int(values[13])
        local_header_offset = int(values[16])
        total_length = 46 + name_length + extra_length + entry_comment_length
        if cursor + total_length > len(central):
            raise DipserFormatError("ZIP central-directory variable fields are truncated")
        if start_disk:
            raise DipserFormatError("multi-disk ZIP member is unsupported")
        if (
            compressed_size == _UINT32_MAX
            or uncompressed_size == _UINT32_MAX
            or local_header_offset == _UINT32_MAX
        ):
            raise DipserFormatError("ZIP64 member is unsupported")
        if flags & 0x1:
            raise DipserFormatError("encrypted ZIP members are unsupported")
        raw_name = central[cursor + 46 : cursor + 46 + name_length]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        try:
            name = raw_name.decode(encoding)
        except UnicodeDecodeError as exc:
            raise DipserFormatError("ZIP member name is not valid text") from exc
        if not name or "\x00" in name:
            raise DipserFormatError("ZIP member has an empty or NUL-containing name")
        if name in seen_names:
            raise DipserFormatError(f"duplicate ZIP member name: {name}")
        seen_names.add(name)
        members.append(
            ZipMember(
                name=name,
                compression=compression,
                flags=flags,
                crc32=crc32,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                local_header_offset=local_header_offset,
            )
        )
        cursor += total_length
    if cursor != len(central) or len(members) != entry_count:
        raise DipserFormatError(
            f"central-directory entry count mismatch: parsed {len(members)}, expected {entry_count}"
        )
    return members


def fetch_zip_member(source: RangeSource, member: ZipMember) -> bytes:
    """Range-fetch, decompress, and CRC-check one central-directory member."""

    header_start = member.local_header_offset
    fixed_header = _read_exact(source, header_start, header_start + 30)
    values = struct.unpack("<4s5H3I2H", fixed_header)
    if values[0] != _LOCAL_SIGNATURE:
        raise DipserFormatError(f"invalid local header for ZIP member {member.name}")
    local_flags = int(values[2])
    local_compression = int(values[3])
    name_length = int(values[9])
    extra_length = int(values[10])
    if local_flags & 0x1:
        raise DipserFormatError(f"encrypted ZIP member is unsupported: {member.name}")
    if local_compression != member.compression:
        raise DipserFormatError(f"compression mismatch for ZIP member {member.name}")
    variable = _read_exact(source, header_start + 30, header_start + 30 + name_length + extra_length)
    encoding = "utf-8" if local_flags & 0x800 else "cp437"
    try:
        local_name = variable[:name_length].decode(encoding)
    except UnicodeDecodeError as exc:
        raise DipserFormatError(f"invalid local filename for ZIP member {member.name}") from exc
    if local_name != member.name:
        raise DipserFormatError(
            f"local/central filename mismatch: {local_name!r} != {member.name!r}"
        )
    data_start = header_start + 30 + name_length + extra_length
    compressed = _read_exact(source, data_start, data_start + member.compressed_size)
    try:
        if member.compression == 0:
            payload = compressed
        elif member.compression == 8:
            payload = zlib.decompress(compressed, -zlib.MAX_WBITS)
        else:
            raise DipserFormatError(
                f"unsupported ZIP compression method {member.compression} for {member.name}"
            )
    except zlib.error as exc:
        raise DipserFormatError(f"deflate decompression failed for {member.name}") from exc
    if len(payload) != member.uncompressed_size:
        raise DipserFormatError(
            f"uncompressed-size mismatch for {member.name}: "
            f"{len(payload)} != {member.uncompressed_size}"
        )
    actual_crc = binascii.crc32(payload) & _UINT32_MAX
    if actual_crc != member.crc32:
        raise DipserFormatError(
            f"CRC-32 mismatch for {member.name}: {actual_crc:08x} != {member.crc32:08x}"
        )
    return payload


def safe_member_destination(root: str | Path, member_name: str) -> Path:
    """Resolve a ZIP member below ``root`` and reject traversal/absolute paths."""

    pure = PurePosixPath(member_name)
    if pure.is_absolute() or not pure.parts or any(part in ("", ".", "..") for part in pure.parts):
        raise DipserFormatError(f"unsafe ZIP member path: {member_name!r}")
    root_path = Path(root).resolve()
    destination = root_path.joinpath(*pure.parts).resolve()
    if destination != root_path and root_path not in destination.parents:
        raise DipserFormatError(f"ZIP member escapes output root: {member_name!r}")
    return destination


def fetch_selected_members(
    source: RangeSource,
    *,
    names: Iterable[str] | None = None,
    predicate: Callable[[ZipMember], bool] | None = None,
) -> dict[str, bytes]:
    """Fetch an explicit/predicate-selected subset, preserving central order."""

    requested = set(names or ())
    directory = read_zip_directory(source)
    selected = [
        member
        for member in directory
        if member.name in requested or (predicate is not None and predicate(member))
    ]
    found = {member.name for member in selected}
    missing = requested - found
    if missing:
        raise KeyError("ZIP members not found: " + ", ".join(sorted(missing)))
    return {member.name: fetch_zip_member(source, member) for member in selected}


def _json_document(document: bytes | str | Mapping[str, Any] | Sequence[Any]) -> Any:
    if isinstance(document, bytes):
        return json.loads(document.decode("utf-8"))
    if isinstance(document, str):
        return json.loads(document)
    return document


def parse_dipser_timestamp(value: str | int | float) -> float:
    """Convert DIPSER ``HH:MM:SS:fraction`` timestamps to seconds of day."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        if not math.isfinite(result):
            raise DipserFormatError("timestamp must be finite")
        return result
    text = str(value).strip()
    match = re.fullmatch(r"(\d{1,2}):(\d{2}):(\d{2})(?:[.:](\d{1,6}))?", text)
    if not match:
        raise DipserFormatError(f"invalid DIPSER timestamp: {value!r}")
    hours, minutes, seconds = (int(match.group(index)) for index in range(1, 4))
    if hours > 23 or minutes > 59 or seconds > 59:
        raise DipserFormatError(f"out-of-range DIPSER timestamp: {value!r}")
    fraction_text = match.group(4) or ""
    fraction = int(fraction_text) / (10 ** len(fraction_text)) if fraction_text else 0.0
    return hours * 3600.0 + minutes * 60.0 + seconds + fraction


def _validated_scale(value: Any, *, field_name: str, minimum: int, maximum: int) -> int:
    try:
        numeric = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise DipserFormatError(f"{field_name} must be an integer") from exc
    if not minimum <= numeric <= maximum:
        raise DipserFormatError(f"{field_name} must be in [{minimum}, {maximum}]")
    return numeric


def reconstruct_expert_labels(
    document: bytes | str | Sequence[Mapping[str, Any]],
    timestamps: Sequence[str | int | float],
) -> list[dict[str, Any]]:
    """Carry attention and emotion change points forward independently.

    A DIPSER change-point can update only one field.  Treating omitted fields as
    null would erase the other signal, so the two timelines are maintained
    separately.  Rows before an expert's first change point remain ``None``.
    """

    raw = _json_document(document)
    if not isinstance(raw, list):
        raise DipserFormatError("DIPSER label document must be a JSON list")
    changes: list[tuple[float, int, int | None, int | None]] = []
    for sequence_number, item in enumerate(raw):
        if not isinstance(item, Mapping) or "datetime" not in item:
            raise DipserFormatError("each DIPSER label change must contain datetime")
        attention = (
            _validated_scale(item["attention"], field_name="attention", minimum=1, maximum=5)
            if "attention" in item and str(item["attention"]).strip() != ""
            else None
        )
        emotion = (
            _validated_scale(item["emotion"], field_name="emotion", minimum=1, maximum=9)
            if "emotion" in item and str(item["emotion"]).strip() != ""
            else None
        )
        if attention is None and emotion is None:
            raise DipserFormatError("label change updates neither attention nor emotion")
        changes.append((parse_dipser_timestamp(item["datetime"]), sequence_number, attention, emotion))
    changes.sort(key=lambda row: (row[0], row[1]))

    targets = [
        (parse_dipser_timestamp(value), index, value)
        for index, value in enumerate(timestamps)
    ]
    state_attention: int | None = None
    state_emotion: int | None = None
    cursor = 0
    output: list[dict[str, Any] | None] = [None] * len(targets)
    for seconds, target_index, original in sorted(targets, key=lambda pair: (pair[0], pair[1])):
        while cursor < len(changes) and changes[cursor][0] <= seconds:
            _, _, attention, emotion = changes[cursor]
            if attention is not None:
                state_attention = attention
            if emotion is not None:
                state_emotion = emotion
            cursor += 1
        output[target_index] = {
            "timestamp": original,
            "seconds_of_day": seconds,
            "attention": state_attention,
            "emotion": state_emotion,
        }
    if any(row is None for row in output):  # pragma: no cover - defensive invariant
        raise RuntimeError("failed to reconstruct one or more requested label timestamps")
    return [row for row in output if row is not None]


def _expert_number(name: str) -> str | None:
    normalized = str(PurePosixPath(name)).lower()
    match = re.search(r"(?:^|/)labeler_(0[1-5])\.json$", normalized)
    return match.group(1) if match else None


def _four_publisher_experts(
    label_documents: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve exactly four publisher expert files, excluding self-labeling.

    DIPSER uses evaluator IDs 01--05 but publishes four expert files per
    subject archive.  Real V5 archives contain both 01/02/03/04 and
    01/02/03/05 combinations; the student self-evaluation is a separately named
    file and is never eligible here.
    """

    experts: dict[str, Any] = {}
    for name, document in label_documents.items():
        expert = _expert_number(name)
        if expert is not None:
            if expert in experts:
                raise DipserFormatError(f"duplicate DIPSER expert labeler_{expert}")
            experts[expert] = document
    if len(experts) != 4:
        raise DipserFormatError(
            "exactly four publisher expert label files are required; "
            f"found IDs {sorted(experts)}"
        )
    return experts


def build_expert_attention_ground_truth(
    label_documents: Mapping[str, bytes | str | Sequence[Mapping[str, Any]]],
    timestamps: Sequence[str | int | float],
) -> list[dict[str, Any]]:
    """Build integer 1--5 truth from the median of exactly four experts.

    Exactly four publisher ``labeler_0X.json`` files are eligible.  Evaluator
    IDs may be 01--05 because real V5 archives use both 01/02/03/04 and
    01/02/03/05 combinations. ``self_labeling.json`` is ignored by construction.
    When four values have a half-step median, the consensus uses deterministic
    round-half-up (e.g. 2.5 -> 3), keeping the published 1--5 label space.
    Targets lacking a carried-forward value from any expert are omitted.
    """

    experts = _four_publisher_experts(label_documents)

    reconstructed = {
        expert: reconstruct_expert_labels(experts[expert], timestamps)
        for expert in sorted(experts)
    }
    consensus: list[dict[str, Any]] = []
    for index, timestamp in enumerate(timestamps):
        values = [reconstructed[expert][index]["attention"] for expert in sorted(experts)]
        if any(value is None for value in values):
            continue
        numeric_values = [int(value) for value in values if value is not None]
        raw_median = float(median(numeric_values))
        truth = min(5, max(1, int(math.floor(raw_median + 0.5))))
        consensus.append(
            {
                "timestamp": timestamp,
                "seconds_of_day": parse_dipser_timestamp(timestamp),
                "expert_attention": {
                    f"labeler_{expert}": int(reconstructed[expert][index]["attention"])
                    for expert in sorted(experts)
                },
                "attention_median_raw": raw_median,
                "attention": truth,
                "label": truth - 1,
                "expert_labeler_ids": sorted(experts),
                "ground_truth_source": "median_of_four_publisher_expert_labelers",
                "self_labeling_excluded": True,
            }
        )
    return consensus


def build_expert_attention_band_ground_truth(
    label_documents: Mapping[str, bytes | str | Sequence[Mapping[str, Any]]],
    timestamps: Sequence[str | int | float],
    *,
    minimum_agreement: int = 3,
) -> list[dict[str, Any]]:
    """Build low/medium/high truth only where at least 3/4 experts agree.

    Published attention values 1--2 map to ``low``, 3 maps to ``medium``, and
    4--5 map to ``high``.  Ambiguous 2--2 splits are omitted rather than forced
    into a class.  This is suitable as a conservative primary classification
    target; :func:`build_expert_attention_ground_truth` retains the full 1--5
    median as an auditable secondary target.
    """

    if minimum_agreement not in (3, 4):
        raise ValueError("minimum_agreement must be 3 or 4")
    experts = _four_publisher_experts(label_documents)
    reconstructed = {
        expert: reconstruct_expert_labels(experts[expert], timestamps)
        for expert in sorted(experts)
    }
    class_names = ("low", "medium", "high")

    rows: list[dict[str, Any]] = []
    for index, timestamp in enumerate(timestamps):
        attention = [reconstructed[expert][index]["attention"] for expert in sorted(experts)]
        if any(value is None for value in attention):
            continue
        bands = [0 if int(value) <= 2 else 1 if int(value) == 3 else 2 for value in attention]
        counts = Counter(bands)
        winning_band, agreement = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        if agreement < minimum_agreement:
            continue
        rows.append(
            {
                "timestamp": timestamp,
                "seconds_of_day": parse_dipser_timestamp(timestamp),
                "expert_attention": {
                    f"labeler_{expert}": int(reconstructed[expert][index]["attention"])
                    for expert in sorted(experts)
                },
                "expert_bands": {
                    f"labeler_{expert}": bands[position]
                    for position, expert in enumerate(sorted(experts))
                },
                "label": int(winning_band),
                "label_name": class_names[winning_band],
                "expert_agreement": int(agreement),
                "minimum_agreement": minimum_agreement,
                "expert_labeler_ids": sorted(experts),
                "ground_truth_source": "three_band_agreement_of_four_publisher_expert_labelers",
                "self_labeling_excluded": True,
            }
        )
    return rows


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _summary(prefix: str, values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_range": 0.0,
        }
    return {
        f"{prefix}_mean": float(fmean(values)),
        f"{prefix}_std": float(pstdev(values)) if len(values) > 1 else 0.0,
        f"{prefix}_min": float(min(values)),
        f"{prefix}_max": float(max(values)),
        f"{prefix}_range": float(max(values) - min(values)),
    }


def extract_visual_pose_features(
    document: bytes | str | Mapping[str, Any],
) -> dict[str, float]:
    """Extract a fixed, de-identified head/body-pose feature whitelist.

    Deliberately ignored fields include face mesh, face bounding boxes, hands,
    image pixels, age, gender, race, and ethnicity.  The returned mapping can
    therefore be audited simply by inspecting its stable key set.
    """

    raw = _json_document(document)
    if not isinstance(raw, Mapping):
        raise DipserFormatError("DIPSER metadata document must be a JSON object")
    person = raw.get("person", {})
    if not isinstance(person, Mapping):
        person = {}
    face = person.get("face", {})
    if not isinstance(face, Mapping):
        face = {}
    # V5 nests head pose under person.face.  The direct person.headpose fallback
    # keeps early processed exports readable while preserving the same whitelist.
    headpose = face.get("headpose", person.get("headpose", {}))
    if not isinstance(headpose, Mapping):
        headpose = {}
    pose = headpose.get("pose", {})
    if not isinstance(pose, Mapping):
        pose = {}

    features: dict[str, float] = {}
    head_values: list[float] = []
    for axis in ("pitch", "yaw", "roll"):
        value = _finite_number(pose.get(axis))
        features[f"head_pose_{axis}"] = value if value is not None else 0.0
        features[f"head_pose_{axis}_available"] = float(value is not None)
        if value is not None:
            head_values.append(value)
    features["head_pose_available"] = float(len(head_values) == 3)

    body = person.get("body", {})
    if not isinstance(body, Mapping):
        body = {}
    body_pose = body.get("body_pose", person.get("body_pose", []))
    if isinstance(body_pose, Mapping):
        body_pose = list(body_pose.values())
    if not isinstance(body_pose, list):
        body_pose = []
    coordinates: dict[str, list[float]] = {
        "x": [],
        "y": [],
        "z": [],
        "visibility": [],
        "presence": [],
    }
    complete_landmark_count = 0
    for landmark in body_pose:
        if not isinstance(landmark, Mapping):
            continue
        complete = True
        for field_name in coordinates:
            value = _finite_number(landmark.get(field_name))
            if value is not None:
                coordinates[field_name].append(value)
            else:
                complete = False
        complete_landmark_count += int(complete)
    features["body_pose_landmark_count"] = float(complete_landmark_count)
    features["body_pose_expected_landmark_count"] = float(DIPSER_BODY_LANDMARK_COUNT)
    features["body_pose_available"] = float(
        len(body_pose) == DIPSER_BODY_LANDMARK_COUNT
        and complete_landmark_count == DIPSER_BODY_LANDMARK_COUNT
    )
    for field_name, values in coordinates.items():
        features[f"body_pose_{field_name}_count"] = float(len(values))
        features.update(_summary(f"body_pose_{field_name}", values))
    visibility = coordinates["visibility"]
    presence = coordinates["presence"]
    features["body_pose_visible_fraction_ge_0_5"] = (
        sum(value >= 0.5 for value in visibility) / len(visibility) if visibility else 0.0
    )
    features["body_pose_present_fraction_ge_0_5"] = (
        sum(value >= 0.5 for value in presence) / len(presence) if presence else 0.0
    )
    return features


_WATCH_CHANNELS: dict[str, tuple[int, ...]] = {
    "samsung_hr_none_wakeup_sensor": (0,),
    "samsung_linear_acceleration_sensor": (0, 1, 2),
    "lsm6dso_gyroscope": (0, 1, 2),
    "samsung_rotation_vector": (0, 1, 2, 3),
    "opt3007_light": (0,),
}


def _normalized_sensor_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _sensor_row_timestamp(row: Mapping[str, Any]) -> float | None:
    """Return one row timestamp using the publisher clock-field priority."""

    normalized = {str(key).lower(): value for key, value in row.items()}
    for key in (
        "time",
        "datetime",
        "date_time",
        "recorded_at",
        "capture_time",
        "timestamp",
        "time_stamp",
    ):
        if key not in normalized:
            continue
        try:
            return parse_dipser_timestamp(normalized[key])
        except DipserFormatError:
            continue
    return None


def extract_watch_features(document: bytes | str | Mapping[str, Any]) -> dict[str, float]:
    """Extract fixed per-file statistics from the five published watch sensors."""

    raw = _json_document(document)
    if not isinstance(raw, Mapping):
        raise DipserFormatError("DIPSER watch document must be a JSON object")
    # V5 uses ``data``.  The authors' earlier reference loader used the
    # Spanish key ``Datos``; accepting it keeps provenance-preserving older
    # releases readable without weakening the fixed sensor whitelist.
    raw_data = raw.get("data", raw.get("Datos", {}))
    if not isinstance(raw_data, Mapping):
        raise DipserFormatError("DIPSER watch document data must be an object")
    normalized_data = {_normalized_sensor_name(name): rows for name, rows in raw_data.items()}

    features: dict[str, float] = {}
    for sensor_name, channel_indices in _WATCH_CHANNELS.items():
        rows = normalized_data.get(sensor_name, [])
        if not isinstance(rows, list):
            rows = []
        mapping_rows = [row for row in rows if isinstance(row, Mapping)]
        channel_values: dict[int, list[float]] = {index: [] for index in channel_indices}
        magnitudes: list[float] = []
        complete_sample_count = 0
        row_timestamps: list[float] = []
        for row in mapping_rows:
            row_timestamp = _sensor_row_timestamp(row)
            if row_timestamp is not None:
                row_timestamps.append(row_timestamp)
            vector: list[float] = []
            for index in channel_indices:
                value = _finite_number(row.get(f"value{index}"))
                if value is None:
                    vector = []
                    break
                vector.append(value)
            if len(vector) != len(channel_indices):
                continue
            complete_sample_count += 1
            for index, value in zip(channel_indices, vector):
                channel_values[index].append(value)
            if len(channel_indices) >= 3:
                magnitudes.append(math.sqrt(sum(value * value for value in vector[:3])))
        timestamps_monotonic = all(
            current + 1e-12 >= previous
            for previous, current in zip(row_timestamps, row_timestamps[1:])
        )
        features[f"watch_{sensor_name}_present"] = float(
            bool(mapping_rows)
            and complete_sample_count == len(mapping_rows) == len(rows)
            and len(row_timestamps) == len(mapping_rows)
            and timestamps_monotonic
        )
        features[f"watch_{sensor_name}_sample_count"] = float(len(mapping_rows))
        features[f"watch_{sensor_name}_complete_sample_count"] = float(
            complete_sample_count
        )
        features[f"watch_{sensor_name}_timestamped_sample_count"] = float(
            len(row_timestamps)
        )
        for index in channel_indices:
            features.update(
                _summary(f"watch_{sensor_name}_value{index}", channel_values[index])
            )
        if len(channel_indices) >= 3:
            features.update(_summary(f"watch_{sensor_name}_magnitude3", magnitudes))
    return features


@dataclass(frozen=True)
class DipserArchiveIdentity:
    """Verified identities encoded by the publisher's archive hierarchy."""

    group_id: str
    experiment_id: str
    subject_id: str
    session_id: str
    participant_id: str
    official_relative_path: str


def parse_official_archive_identity(path: str | PurePosixPath) -> DipserArchiveIdentity:
    """Parse ``group_XX/experiment_XX/subject_XX.zip`` from an official path."""

    normalized = str(path).replace("\\", "/").strip("/").lower()
    match = re.search(
        r"(?:^|/)(group_\d{2})/(experiment_\d{2})/(subject_\d{2})\.zip$",
        normalized,
    )
    if not match:
        raise DipserFormatError(
            "archive path must end in group_XX/experiment_XX/subject_XX.zip"
        )
    group_id, experiment_id, subject_basename = match.groups()
    official_relative_path = f"{group_id}/{experiment_id}/{subject_basename}.zip"
    return DipserArchiveIdentity(
        group_id=group_id,
        experiment_id=experiment_id,
        subject_id=subject_basename,
        session_id=f"{group_id}/{experiment_id}",
        participant_id=f"{group_id}/{subject_basename}",
        official_relative_path=official_relative_path,
    )


def build_source_provenance(
    archive_path: str | PurePosixPath,
    *,
    source_file_id: str,
    source_md5: str,
    version: str = DIPSER_VERSION,
) -> dict[str, Any]:
    """Document source identities and byte-level proof for one subject archive."""

    identity = parse_official_archive_identity(archive_path)
    file_id = str(source_file_id).strip()
    md5 = str(source_md5).strip().lower()
    if not file_id or not re.fullmatch(r"[0-9a-fA-F]{16,64}", file_id):
        raise DipserFormatError("ScienceDB source_file_id is missing or malformed")
    if not re.fullmatch(r"[0-9a-f]{32}", md5):
        raise DipserFormatError("ScienceDB source_md5 must be 32 lowercase hex characters")
    normalized_version = str(version).strip()
    if not re.fullmatch(r"V\d+", normalized_version):
        raise DipserFormatError("ScienceDB dataset version must look like V5")
    return {
        "dataset_id": "DIPSER",
        "science_db_dataset_id": DIPSER_DATASET_ID,
        "dataset_version": normalized_version,
        "dataset_doi": DIPSER_DOI,
        "dataset_url": DIPSER_DATASET_URL,
        "license_snapshot_checked_on": DIPSER_LICENSE_SNAPSHOT_DATE,
        "platform_license_label": "CC BY 4.0",
        "usage_boundary": (
            "non-commercial academic/research use pending clarification of the "
            "paper Usage Notes; commercial use requires explicit custodian approval"
        ),
        "paper_url": DIPSER_PAPER_URL,
        "reference_loader_url": DIPSER_CODE_URL,
        "site_id": DIPSER_SITE_ID,
        "group_id": identity.group_id,
        "cohort_id": identity.group_id,
        "experiment_id": identity.experiment_id,
        "subject_id": identity.subject_id,
        "participant_id": identity.participant_id,
        "session_id": identity.session_id,
        "official_relative_path": identity.official_relative_path,
        "source_file_id": file_id,
        "source_archive_md5": md5,
        "source_archive_md5_status": (
            "publisher-declared; range extraction does not claim a full-archive MD5 recomputation"
        ),
        "source_url": f"https://china.scidb.cn/download?fileId={file_id}",
        "range_member_integrity": "compressed bytes are decompressed and checked against ZIP CRC-32",
        "identity_metadata_source": (
            "publisher ScienceDB group/experiment/subject hierarchy; "
            "publisher paper capture-site description"
        ),
        "identity_metadata_verified": {
            "group_id": True,
            "cohort_id": True,
            "experiment_id": True,
            "subject_id": True,
            "participant_id": True,
            "session_id": True,
            "site_id": True,
        },
        "session_definition": "one cohort's recording of one published experiment",
        "participant_definition": "subject number scoped by cohort; stable across experiments",
        "ground_truth_source": (
            "exactly four independent publisher expert labeler files (IDs 01..05); "
            "exact IDs and task aggregation are recorded per sample; self labeling excluded"
        ),
        "teacher_id_available": False,
        "deployment_site_count": 1,
        "supports_cross_site_deployment_accuracy": False,
    }


def write_selected_members(
    source: RangeSource,
    output_root: str | Path,
    *,
    names: Iterable[str] | None = None,
    predicate: Callable[[ZipMember], bool] | None = None,
) -> list[Path]:
    """Materialize a verified subset beneath ``output_root``.

    This intentionally does not choose files on the caller's behalf.  Passing an
    explicit allowlist/predicate keeps downloads auditable and prevents a typo
    from triggering extraction of every RGB frame.
    """

    payloads = fetch_selected_members(source, names=names, predicate=predicate)
    written: list[Path] = []
    for name, payload in payloads.items():
        destination = safe_member_destination(output_root, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        written.append(destination)
    return written
