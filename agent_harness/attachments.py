"""Private, immutable local attachment ingestion.

The store deliberately treats a user-selected pathname as an untrusted import
source.  It opens every path component without following links, reads one
stable regular-file snapshot under a strict bound, identifies the bytes rather
than trusting the suffix, and publishes a private blob atomically.  Public
descriptors contain a display basename but never the source pathname.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Iterable, Mapping
import unicodedata
import zlib


ATTACHMENT_SCHEMA = "agent_harness.attachment.v1"

MAX_ATTACHMENTS_PER_TURN = 8
MAX_ATTACHMENTS_TOTAL_BYTES = 24 * 1024 * 1024
MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_PDF_BYTES = 16 * 1024 * 1024

_MAX_SOURCE_BYTES = max(MAX_TEXT_BYTES, MAX_IMAGE_BYTES, MAX_PDF_BYTES)
_READ_CHUNK_BYTES = 1024 * 1024
_ATTACHMENT_ID = re.compile(r"^att_[0-9a-f]{32}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_KINDS = frozenset({"text", "image", "pdf"})
_TEXT_MEDIA_TYPES = frozenset(
    {
        "text/plain; charset=utf-8",
        "text/markdown; charset=utf-8",
    }
)
_IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg"})
_TRUSTED_MACOS_ROOT_ALIASES = {
    "etc": "/private/etc",
    "tmp": "/private/tmp",
    "var": "/private/var",
}


class AttachmentError(RuntimeError):
    """A path-free attachment validation or storage failure."""

    def __init__(self, message: str, *, code: str = "attachment_error") -> None:
        self.code = code
        super().__init__(message)


def _plain_int(value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AttachmentError("attachment descriptor is invalid")
    return value


def _valid_display_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise AttachmentError("attachment descriptor is invalid")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError:
        raise AttachmentError("attachment descriptor is invalid") from None
    if len(encoded) > 255:
        raise AttachmentError("attachment descriptor is invalid")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise AttachmentError("attachment descriptor is invalid")
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise AttachmentError("attachment descriptor is invalid")
    return value


@dataclass(frozen=True, slots=True)
class AttachmentDescriptor:
    """Content metadata safe to persist in a session or expose in a UI."""

    schema: str
    attachment_id: str
    kind: str
    media_type: str
    display_name: str
    size_bytes: int
    sha256: str
    estimated_tokens: int
    width: int | None = None
    height: int | None = None

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> "AttachmentDescriptor":
        if not isinstance(value, Mapping):
            raise AttachmentError("attachment descriptor is invalid")
        kind = value.get("kind")
        if not isinstance(kind, str) or kind not in _KINDS:
            raise AttachmentError("attachment descriptor is invalid")
        required = {
            "schema",
            "attachment_id",
            "kind",
            "media_type",
            "display_name",
            "size_bytes",
            "sha256",
            "estimated_tokens",
        }
        expected = required | ({"width", "height"} if kind == "image" else set())
        if set(value) != expected:
            raise AttachmentError("attachment descriptor is invalid")

        schema = value.get("schema")
        attachment_id = value.get("attachment_id")
        media_type = value.get("media_type")
        digest = value.get("sha256")
        if schema != ATTACHMENT_SCHEMA:
            raise AttachmentError("attachment descriptor is invalid")
        if not isinstance(attachment_id, str) or not _ATTACHMENT_ID.fullmatch(attachment_id):
            raise AttachmentError("attachment descriptor is invalid")
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
            raise AttachmentError("attachment descriptor is invalid")
        if not isinstance(media_type, str):
            raise AttachmentError("attachment descriptor is invalid")

        size_bytes = _plain_int(value.get("size_bytes"), minimum=1)
        estimated_tokens = _plain_int(value.get("estimated_tokens"), minimum=1)
        display_name = _valid_display_name(value.get("display_name"))
        width: int | None = None
        height: int | None = None
        if kind == "text":
            if (
                media_type not in _TEXT_MEDIA_TYPES
                or size_bytes > MAX_TEXT_BYTES
                or estimated_tokens != size_bytes
            ):
                raise AttachmentError("attachment descriptor is invalid")
        elif kind == "pdf":
            if (
                media_type != "application/pdf"
                or size_bytes > MAX_PDF_BYTES
                or estimated_tokens != size_bytes
            ):
                raise AttachmentError("attachment descriptor is invalid")
        else:
            if (
                media_type not in _IMAGE_MEDIA_TYPES
                or size_bytes > MAX_IMAGE_BYTES
                or estimated_tokens != 512
            ):
                raise AttachmentError("attachment descriptor is invalid")
            width = _plain_int(value.get("width"), minimum=1)
            height = _plain_int(value.get("height"), minimum=1)
            if width > 8192 or height > 8192:
                raise AttachmentError("attachment descriptor is invalid")

        return cls(
            schema=schema,
            attachment_id=attachment_id,
            kind=kind,
            media_type=media_type,
            display_name=display_name,
            size_bytes=size_bytes,
            sha256=digest,
            estimated_tokens=estimated_tokens,
            width=width,
            height=height,
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "attachment_id": self.attachment_id,
            "kind": self.kind,
            "media_type": self.media_type,
            "display_name": self.display_name,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "estimated_tokens": self.estimated_tokens,
        }
        if self.kind == "image":
            value["width"] = self.width
            value["height"] = self.height
        # Round-trip validation prevents a manually constructed dataclass from
        # bypassing the same trust boundary as a deserialized descriptor.
        validated = type(self).from_value(value)
        if validated != self:
            raise AttachmentError("attachment descriptor is invalid")
        return value


@dataclass(frozen=True, slots=True)
class _PreparedAttachment:
    descriptor: AttachmentDescriptor
    content: bytes


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise AttachmentError("secure attachment access is unavailable")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _read_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise AttachmentError("secure attachment access is unavailable")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _write_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise AttachmentError("secure attachment access is unavailable")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _is_trusted_macos_root_alias(
    parent: Path,
    candidate: Path,
    metadata: os.stat_result,
) -> bool:
    if sys.platform != "darwin" or parent != Path("/"):
        return False
    expected = _TRUSTED_MACOS_ROOT_ALIASES.get(candidate.name)
    if expected is None or metadata.st_uid != 0:
        return False
    try:
        parent_metadata = parent.stat()
        target = os.readlink(candidate)
    except OSError:
        return False
    if parent_metadata.st_uid != 0 or stat.S_IMODE(parent_metadata.st_mode) & 0o022:
        return False
    resolved = os.path.normpath(os.path.join(os.fspath(parent), target))
    return resolved == expected


def _canonical_path(
    value: os.PathLike[str] | str,
    *,
    source: bool,
    allow_missing: bool,
) -> Path:
    try:
        raw_value = os.fspath(value)
    except TypeError:
        raise AttachmentError("attachment path is invalid") from None
    if not isinstance(raw_value, str) or not raw_value:
        raise AttachmentError("attachment path is invalid")
    if any(character in raw_value for character in ("\x00", "\n", "\r")):
        raise AttachmentError("attachment path is invalid")
    raw = Path(raw_value)
    expandable = raw_value.startswith("~")
    if not raw.is_absolute() and not expandable:
        raise AttachmentError("attachment path must be absolute or home-expanded")
    try:
        expanded = raw.expanduser()
    except (KeyError, RuntimeError):
        raise AttachmentError("attachment path is invalid") from None
    if not expanded.is_absolute():
        raise AttachmentError("attachment path must be absolute or home-expanded")

    path = Path(os.path.abspath(os.fspath(expanded)))
    current = Path(path.anchor)
    parts = path.parts[1:]
    for index, component in enumerate(parts):
        candidate = current / component
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            if allow_missing:
                return current.joinpath(*parts[index:])
            raise AttachmentError("attachment source is unavailable") from None
        except OSError:
            message = "attachment source is unavailable" if source else "attachment store is unavailable"
            raise AttachmentError(message) from None
        if stat.S_ISLNK(metadata.st_mode):
            if not _is_trusted_macos_root_alias(current, candidate, metadata):
                raise AttachmentError("attachment path contains an untrusted link")
            try:
                current = candidate.resolve(strict=True)
            except (OSError, RuntimeError):
                raise AttachmentError("attachment path contains an untrusted link") from None
        else:
            current = candidate
    return current


def _open_directory_chain(path: Path, *, create: bool) -> int:
    """Return an owned descriptor for ``path`` after a no-follow walk."""

    try:
        descriptor = os.open(path.anchor, _directory_flags())
    except OSError:
        raise AttachmentError("attachment store is unavailable") from None
    try:
        for component in path.parts[1:]:
            created = False
            try:
                child = os.open(component, _directory_flags(), dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise AttachmentError("attachment source is unavailable") from None
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                    child = os.open(component, _directory_flags(), dir_fd=descriptor)
                    created = True
                except OSError:
                    raise AttachmentError("attachment store is unavailable") from None
            except OSError:
                message = "attachment store is unavailable" if create else "attachment source is unavailable"
                raise AttachmentError(message) from None
            if created:
                try:
                    os.fchmod(child, 0o700)
                    os.fsync(child)
                except OSError:
                    os.close(child)
                    raise AttachmentError("attachment store is unavailable") from None
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_source_parent(path: Path) -> tuple[int, str]:
    if path == Path(path.anchor):
        raise AttachmentError("attachment source is not a regular file")
    parent = _open_directory_chain(path.parent, create=False)
    return parent, path.name


def _snapshot_fields(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_bounded(descriptor: int, *, expected_size: int, maximum: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        try:
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, maximum + 1 - total))
        except OSError:
            raise AttachmentError("attachment source could not be read") from None
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > maximum:
            raise AttachmentError("attachment exceeds the supported size limit")
    if total != expected_size:
        raise AttachmentError("attachment source changed during import")
    return b"".join(chunks)


def _safe_display_name(name: str) -> str:
    normalized = unicodedata.normalize("NFC", name)
    cleaned = "".join(
        "_" if character in {"/", "\\"} or unicodedata.category(character) == "Cc" else character
        for character in normalized
    ).strip()
    if cleaned in {"", ".", ".."}:
        cleaned = "attachment"
    try:
        while len(cleaned.encode("utf-8")) > 255:
            cleaned = cleaned[:-1]
    except UnicodeError:
        raise AttachmentError("attachment display name is invalid") from None
    return cleaned or "attachment"


def _png_dimensions(content: bytes) -> tuple[int, int]:
    if not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AttachmentError("attachment image data is invalid")
    offset = 8
    saw_header = False
    saw_data = False
    while offset < len(content):
        if len(content) - offset < 12:
            raise AttachmentError("attachment image data is invalid")
        length = int.from_bytes(content[offset : offset + 4], "big")
        chunk_type = content[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(content):
            raise AttachmentError("attachment image data is invalid")
        chunk_data = content[offset + 8 : offset + 8 + length]
        expected_crc = int.from_bytes(content[offset + 8 + length : end], "big")
        if zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF != expected_crc:
            raise AttachmentError("attachment image data is invalid")
        if not saw_header:
            if chunk_type != b"IHDR" or length != 13:
                raise AttachmentError("attachment image data is invalid")
            width = int.from_bytes(chunk_data[0:4], "big")
            height = int.from_bytes(chunk_data[4:8], "big")
            bit_depth = chunk_data[8]
            color_type = chunk_data[9]
            valid_depths = {
                0: {1, 2, 4, 8, 16},
                2: {8, 16},
                3: {1, 2, 4, 8},
                4: {8, 16},
                6: {8, 16},
            }
            if (
                width not in range(1, 8193)
                or height not in range(1, 8193)
                or bit_depth not in valid_depths.get(color_type, set())
                or chunk_data[10] != 0
                or chunk_data[11] != 0
                or chunk_data[12] not in {0, 1}
            ):
                raise AttachmentError("attachment image data is invalid")
            saw_header = True
        elif chunk_type == b"IHDR":
            raise AttachmentError("attachment image data is invalid")
        if chunk_type == b"IDAT":
            saw_data = True
        if chunk_type == b"IEND":
            if length != 0 or not saw_data or end != len(content):
                raise AttachmentError("attachment image data is invalid")
            return width, height
        offset = end
    raise AttachmentError("attachment image data is invalid")


def _jpeg_dimensions(content: bytes) -> tuple[int, int]:
    if len(content) < 4 or content[:2] != b"\xff\xd8" or content[-2:] != b"\xff\xd9":
        raise AttachmentError("attachment image data is invalid")
    offset = 2
    dimensions: tuple[int, int] | None = None
    frame_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while offset < len(content) - 2:
        if content[offset] != 0xFF:
            raise AttachmentError("attachment image data is invalid")
        while offset < len(content) and content[offset] == 0xFF:
            offset += 1
        if offset >= len(content):
            raise AttachmentError("attachment image data is invalid")
        marker = content[offset]
        offset += 1
        if marker == 0x00 or marker == 0xD8:
            raise AttachmentError("attachment image data is invalid")
        if marker == 0xD9:
            break
        if marker in range(0xD0, 0xD8) or marker == 0x01:
            continue
        if offset + 2 > len(content):
            raise AttachmentError("attachment image data is invalid")
        segment_length = int.from_bytes(content[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(content):
            raise AttachmentError("attachment image data is invalid")
        if marker in frame_markers:
            if segment_length < 8:
                raise AttachmentError("attachment image data is invalid")
            height = int.from_bytes(content[offset + 3 : offset + 5], "big")
            width = int.from_bytes(content[offset + 5 : offset + 7], "big")
            components = content[offset + 7]
            if segment_length != 8 + 3 * components:
                raise AttachmentError("attachment image data is invalid")
            if width not in range(1, 8193) or height not in range(1, 8193):
                raise AttachmentError("attachment image data is invalid")
            dimensions = (width, height)
        if marker == 0xDA:
            if dimensions is None:
                raise AttachmentError("attachment image data is invalid")
            return dimensions
        offset += segment_length
    if dimensions is None:
        raise AttachmentError("attachment image data is invalid")
    return dimensions


def _estimated_tokens(
    kind: str,
    content: bytes,
    *,
    width: int | None,
    height: int | None,
) -> int:
    del width, height
    if kind == "image":
        return 512
    return len(content)


def _identify(
    content: bytes,
    *,
    display_name: str,
) -> tuple[str, str, int | None, int | None, int]:
    kind: str
    media_type: str
    width: int | None = None
    height: int | None = None
    if content.startswith(b"%PDF"):
        if not content.startswith(b"%PDF-"):
            raise AttachmentError("attachment PDF data is invalid")
        if not content.rstrip(b" \t\r\n\f").endswith(b"%%EOF"):
            raise AttachmentError("attachment PDF data is invalid")
        kind = "pdf"
        media_type = "application/pdf"
        maximum = MAX_PDF_BYTES
    elif content.startswith(b"\x89PNG\r\n\x1a\n"):
        width, height = _png_dimensions(content)
        kind = "image"
        media_type = "image/png"
        maximum = MAX_IMAGE_BYTES
    elif content.startswith(b"\xff\xd8"):
        width, height = _jpeg_dimensions(content)
        kind = "image"
        media_type = "image/jpeg"
        maximum = MAX_IMAGE_BYTES
    else:
        try:
            content.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise AttachmentError("attachment type is unsupported") from None
        if b"\x00" in content:
            raise AttachmentError("attachment text contains a NUL byte")
        kind = "text"
        suffix = Path(display_name).suffix.casefold()
        media_type = (
            "text/markdown; charset=utf-8"
            if suffix in {".md", ".markdown"}
            else "text/plain; charset=utf-8"
        )
        maximum = MAX_TEXT_BYTES
    if len(content) > maximum:
        raise AttachmentError("attachment exceeds the supported size limit")
    tokens = _estimated_tokens(kind, content, width=width, height=height)
    return kind, media_type, width, height, tokens


class AttachmentStore:
    """Owner-private store for immutable attachment blobs."""

    def __init__(self, root: os.PathLike[str] | str) -> None:
        self.root = _canonical_path(root, source=False, allow_missing=True)
        descriptor = _open_directory_chain(self.root, create=True)
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                raise AttachmentError("attachment store permissions are invalid")
        finally:
            os.close(descriptor)

    def _open_store(self) -> int:
        descriptor = _open_directory_chain(self.root, create=False)
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                raise AttachmentError("attachment store permissions are invalid")
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def _prepare(self, path_value: os.PathLike[str] | str) -> _PreparedAttachment:
        path = _canonical_path(path_value, source=True, allow_missing=False)
        parent, name = _open_source_parent(path)
        source_descriptor: int | None = None
        try:
            try:
                before = os.stat(name, dir_fd=parent, follow_symlinks=False)
            except OSError:
                raise AttachmentError("attachment source is unavailable") from None
            if not stat.S_ISREG(before.st_mode):
                raise AttachmentError("attachment source is not a regular file")
            if before.st_size == 0:
                raise AttachmentError("empty attachments are unsupported")
            if before.st_size < 0 or before.st_size > _MAX_SOURCE_BYTES:
                raise AttachmentError("attachment exceeds the supported size limit")
            try:
                source_descriptor = os.open(name, _read_flags(), dir_fd=parent)
            except OSError:
                raise AttachmentError("attachment source could not be opened") from None
            opened = os.fstat(source_descriptor)
            if not stat.S_ISREG(opened.st_mode) or _snapshot_fields(opened) != _snapshot_fields(before):
                raise AttachmentError("attachment source changed during import")
            content = _read_bounded(
                source_descriptor,
                expected_size=opened.st_size,
                maximum=_MAX_SOURCE_BYTES,
            )
            after = os.fstat(source_descriptor)
            try:
                current = os.stat(name, dir_fd=parent, follow_symlinks=False)
            except OSError:
                raise AttachmentError("attachment source changed during import") from None
            snapshot = _snapshot_fields(opened)
            if _snapshot_fields(after) != snapshot or _snapshot_fields(current) != snapshot:
                raise AttachmentError("attachment source changed during import")
        finally:
            if source_descriptor is not None:
                os.close(source_descriptor)
            os.close(parent)

        display_name = _safe_display_name(path.name)
        kind, media_type, width, height, estimated_tokens = _identify(
            content,
            display_name=display_name,
        )
        descriptor = AttachmentDescriptor(
            schema=ATTACHMENT_SCHEMA,
            attachment_id=f"att_{os.urandom(16).hex()}",
            kind=kind,
            media_type=media_type,
            display_name=display_name,
            size_bytes=len(content),
            sha256=sha256(content).hexdigest(),
            estimated_tokens=estimated_tokens,
            width=width,
            height=height,
        )
        descriptor.to_dict()
        return _PreparedAttachment(descriptor=descriptor, content=content)

    def _publish(self, prepared: _PreparedAttachment) -> AttachmentDescriptor:
        directory = self._open_store()
        temporary_name = f".tmp_{os.urandom(16).hex()}"
        final_name = f"{prepared.descriptor.attachment_id}.blob"
        temporary_descriptor: int | None = None
        temporary_exists = False
        final_exists = False
        succeeded = False
        try:
            try:
                temporary_descriptor = os.open(
                    temporary_name,
                    _write_flags(),
                    0o600,
                    dir_fd=directory,
                )
                temporary_exists = True
                os.fchmod(temporary_descriptor, 0o600)
            except OSError:
                raise AttachmentError("attachment blob could not be created") from None
            view = memoryview(prepared.content)
            written = 0
            while written < len(view):
                try:
                    count = os.write(temporary_descriptor, view[written:])
                except OSError:
                    raise AttachmentError("attachment blob could not be written") from None
                if count <= 0:
                    raise AttachmentError("attachment blob could not be written")
                written += count
            try:
                os.fsync(temporary_descriptor)
                metadata = os.fstat(temporary_descriptor)
            except OSError:
                raise AttachmentError("attachment blob could not be committed") from None
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_nlink != 1
                or metadata.st_size != len(prepared.content)
            ):
                raise AttachmentError("attachment blob integrity check failed")
            os.close(temporary_descriptor)
            temporary_descriptor = None
            try:
                os.link(
                    temporary_name,
                    final_name,
                    src_dir_fd=directory,
                    dst_dir_fd=directory,
                    follow_symlinks=False,
                )
                final_exists = True
                os.unlink(temporary_name, dir_fd=directory)
                temporary_exists = False
                os.fsync(directory)
            except OSError:
                raise AttachmentError("attachment blob could not be committed") from None
            final_metadata = os.stat(final_name, dir_fd=directory, follow_symlinks=False)
            if (
                not stat.S_ISREG(final_metadata.st_mode)
                or final_metadata.st_uid != os.getuid()
                or stat.S_IMODE(final_metadata.st_mode) != 0o600
                or final_metadata.st_nlink != 1
                or final_metadata.st_size != len(prepared.content)
            ):
                raise AttachmentError("attachment blob integrity check failed")
            succeeded = True
            return prepared.descriptor
        finally:
            if temporary_descriptor is not None:
                os.close(temporary_descriptor)
            if temporary_exists:
                try:
                    os.unlink(temporary_name, dir_fd=directory)
                    os.fsync(directory)
                except OSError:
                    pass
            if final_exists and not succeeded:
                try:
                    os.unlink(final_name, dir_fd=directory)
                    os.fsync(directory)
                except OSError:
                    pass
            os.close(directory)

    def ingest(self, path: os.PathLike[str] | str) -> AttachmentDescriptor:
        """Import one explicit local pathname into a private immutable blob."""

        return self._publish(self._prepare(path))

    def ingest_many(
        self,
        paths: Iterable[os.PathLike[str] | str],
    ) -> tuple[AttachmentDescriptor, ...]:
        """Import one turn's bounded attachment batch without partial success."""

        if isinstance(paths, (str, bytes, os.PathLike)):
            raise AttachmentError("attachment batch is invalid")
        prepared: list[_PreparedAttachment] = []
        total = 0
        try:
            iterator = iter(paths)
        except TypeError:
            raise AttachmentError("attachment batch is invalid") from None
        for path in iterator:
            if len(prepared) >= MAX_ATTACHMENTS_PER_TURN:
                raise AttachmentError("attachment batch exceeds the item limit")
            item = self._prepare(path)
            total += item.descriptor.size_bytes
            if total > MAX_ATTACHMENTS_TOTAL_BYTES:
                raise AttachmentError("attachment batch exceeds the total size limit")
            prepared.append(item)

        published: list[AttachmentDescriptor] = []
        try:
            for item in prepared:
                published.append(self._publish(item))
        except BaseException:
            for descriptor in reversed(published):
                try:
                    self.discard(descriptor)
                except AttachmentError:
                    pass
            raise
        return tuple(published)

    def _verified_blob(
        self,
        descriptor: AttachmentDescriptor,
        *,
        discard: bool,
    ) -> bytes:
        if not isinstance(descriptor, AttachmentDescriptor):
            raise AttachmentError("attachment descriptor is invalid")
        descriptor = AttachmentDescriptor.from_value(descriptor.to_dict())
        directory = self._open_store()
        name = f"{descriptor.attachment_id}.blob"
        blob_descriptor: int | None = None
        try:
            try:
                before = os.stat(name, dir_fd=directory, follow_symlinks=False)
                blob_descriptor = os.open(name, _read_flags(), dir_fd=directory)
            except OSError:
                raise AttachmentError("attachment blob is unavailable") from None
            opened = os.fstat(blob_descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_uid != os.getuid()
                or stat.S_IMODE(opened.st_mode) != 0o600
                or opened.st_nlink != 1
                or _snapshot_fields(opened) != _snapshot_fields(before)
                or opened.st_size != descriptor.size_bytes
            ):
                raise AttachmentError("attachment blob integrity check failed")
            content = _read_bounded(
                blob_descriptor,
                expected_size=descriptor.size_bytes,
                maximum=_MAX_SOURCE_BYTES,
            )
            after = os.fstat(blob_descriptor)
            try:
                current = os.stat(name, dir_fd=directory, follow_symlinks=False)
            except OSError:
                raise AttachmentError("attachment blob integrity check failed") from None
            snapshot = _snapshot_fields(opened)
            if _snapshot_fields(after) != snapshot or _snapshot_fields(current) != snapshot:
                raise AttachmentError("attachment blob integrity check failed")
            if sha256(content).hexdigest() != descriptor.sha256:
                raise AttachmentError("attachment blob integrity check failed")
            kind, media_type, width, height, estimated_tokens = _identify(
                content,
                display_name=descriptor.display_name,
            )
            if (
                kind != descriptor.kind
                or media_type != descriptor.media_type
                or width != descriptor.width
                or height != descriptor.height
                or estimated_tokens != descriptor.estimated_tokens
            ):
                raise AttachmentError("attachment blob integrity check failed")
            if discard:
                try:
                    os.unlink(name, dir_fd=directory)
                    os.fsync(directory)
                except OSError:
                    raise AttachmentError("attachment blob could not be discarded") from None
            return content
        finally:
            if blob_descriptor is not None:
                os.close(blob_descriptor)
            os.close(directory)

    def read(self, descriptor: AttachmentDescriptor) -> bytes:
        """Read and revalidate one immutable blob."""

        return self._verified_blob(descriptor, discard=False)

    def discard(self, descriptor: AttachmentDescriptor) -> None:
        """Remove exactly one verified blob capability."""

        self._verified_blob(descriptor, discard=True)


__all__ = [
    "ATTACHMENT_SCHEMA",
    "AttachmentDescriptor",
    "AttachmentError",
    "AttachmentStore",
    "MAX_ATTACHMENTS_PER_TURN",
    "MAX_ATTACHMENTS_TOTAL_BYTES",
    "MAX_IMAGE_BYTES",
    "MAX_PDF_BYTES",
    "MAX_TEXT_BYTES",
]
