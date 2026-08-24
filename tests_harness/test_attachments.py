from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import stat
import zlib

import pytest

import agent_harness.attachments as attachments_module
from agent_harness.attachments import (
    ATTACHMENT_SCHEMA,
    MAX_ATTACHMENTS_PER_TURN,
    MAX_ATTACHMENTS_TOTAL_BYTES,
    MAX_IMAGE_BYTES,
    MAX_PDF_BYTES,
    MAX_TEXT_BYTES,
    AttachmentDescriptor,
    AttachmentError,
    AttachmentStore,
)


def _chunk(kind: bytes, content: bytes) -> bytes:
    checksum = zlib.crc32(kind + content) & 0xFFFFFFFF
    return len(content).to_bytes(4, "big") + kind + content + checksum.to_bytes(4, "big")


def _png(width: int = 2, height: int = 3, *, data: bytes | None = None) -> bytes:
    header = (
        width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + bytes([8, 2, 0, 0, 0])
    )
    pixels = data if data is not None else zlib.compress((b"\x00" + b"\x00" * width * 3) * height)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header) + _chunk(b"IDAT", pixels) + _chunk(b"IEND", b"")


def _jpeg(width: int = 17, height: int = 13) -> bytes:
    frame = (
        b"\x00\x11"
        + bytes([8])
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + bytes([3, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0])
    )
    scan = b"\x00\x0c" + bytes([3, 1, 0, 2, 0, 3, 0, 0, 63, 0])
    return b"\xff\xd8\xff\xc0" + frame + b"\xff\xda" + scan + b"\x00\xff\xd9"


def _pdf(size: int | None = None) -> bytes:
    prefix = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n"
    suffix = b"%%EOF\n"
    if size is None:
        return prefix + suffix
    assert size >= len(prefix) + len(suffix)
    return prefix + b"x" * (size - len(prefix) - len(suffix)) + suffix


def _write(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path


def _blobs(store: AttachmentStore) -> list[Path]:
    return sorted(store.root.glob("att_*.blob"))


def test_markdown_ingest_round_trip_private_blob_and_discard(tmp_path: Path) -> None:
    source = _write(tmp_path / "lesson.md", "# 标题\n内容\n".encode())
    store = AttachmentStore(tmp_path / "store")

    descriptor = store.ingest(source)

    assert descriptor.schema == ATTACHMENT_SCHEMA
    assert descriptor.attachment_id.startswith("att_")
    assert descriptor.kind == "text"
    assert descriptor.media_type == "text/markdown; charset=utf-8"
    assert descriptor.display_name == "lesson.md"
    assert descriptor.size_bytes == len(source.read_bytes())
    assert descriptor.estimated_tokens == descriptor.size_bytes
    assert descriptor.width is None
    assert descriptor.height is None
    assert "width" not in descriptor.to_dict()
    assert "height" not in descriptor.to_dict()
    assert AttachmentDescriptor.from_value(descriptor.to_dict()) == descriptor
    assert store.read(descriptor) == source.read_bytes()

    blob = _blobs(store)[0]
    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(blob.stat().st_mode) == 0o600
    assert blob.stat().st_nlink == 1
    assert os.fspath(source) not in repr(descriptor)

    store.discard(descriptor)
    assert not blob.exists()
    with pytest.raises(AttachmentError, match="unavailable"):
        store.read(descriptor)


def test_actual_bytes_identify_pdf_png_jpeg_and_ignore_misleading_suffix(tmp_path: Path) -> None:
    store = AttachmentStore(tmp_path / "store")
    pdf = store.ingest(_write(tmp_path / "looks-like-text.txt", _pdf()))
    png = store.ingest(_write(tmp_path / "picture.bin", _png(7, 9)))
    jpeg = store.ingest(_write(tmp_path / "picture.data", _jpeg(19, 11)))
    text = store.ingest(_write(tmp_path / "not-really.png", b"ordinary UTF-8 text\n"))

    assert (pdf.kind, pdf.media_type) == ("pdf", "application/pdf")
    assert (png.kind, png.media_type, png.width, png.height) == (
        "image",
        "image/png",
        7,
        9,
    )
    assert (jpeg.kind, jpeg.media_type, jpeg.width, jpeg.height) == (
        "image",
        "image/jpeg",
        19,
        11,
    )
    assert png.estimated_tokens == jpeg.estimated_tokens == 512
    assert (text.kind, text.media_type) == ("text", "text/plain; charset=utf-8")
    assert set(png.to_dict()) == {
        "schema",
        "attachment_id",
        "kind",
        "media_type",
        "display_name",
        "size_bytes",
        "sha256",
        "estimated_tokens",
        "width",
        "height",
    }


@pytest.mark.parametrize(
    "name,content,error",
    [
        ("empty.md", b"", "empty"),
        ("nul.md", b"hello\x00world", "NUL"),
        ("binary.dat", b"\x80\x81\x82", "unsupported"),
        ("bad.pdf", b"%PDF-1.7\nmissing eof", "PDF"),
        ("bad-magic.pdf", b"%PDFX this is not a PDF\n%%EOF", "PDF"),
        ("bad.png", _png()[:-1] + b"x", "image"),
        ("wide.png", _png(8193, 1), "image"),
        ("bad.jpg", _jpeg()[:-2], "image"),
    ],
)
def test_rejects_empty_unsupported_or_malformed_content(
    tmp_path: Path,
    name: str,
    content: bytes,
    error: str,
) -> None:
    source = _write(tmp_path / name, content)
    store = AttachmentStore(tmp_path / "store")

    with pytest.raises(AttachmentError, match=error) as caught:
        store.ingest(source)

    assert os.fspath(source) not in str(caught.value)
    assert not _blobs(store)


def test_rejects_relative_symlink_directory_fifo_and_linked_parent(tmp_path: Path) -> None:
    store = AttachmentStore(tmp_path / "store")
    target = _write(tmp_path / "target.md", b"safe\n")
    link = tmp_path / "link.md"
    link.symlink_to(target)
    linked_directory = tmp_path / "linked-directory"
    linked_directory.symlink_to(tmp_path, target_is_directory=True)
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)

    with pytest.raises(AttachmentError, match="absolute"):
        store.ingest("relative.md")
    with pytest.raises(AttachmentError, match="untrusted link"):
        store.ingest(link)
    with pytest.raises(AttachmentError, match="untrusted link"):
        store.ingest(linked_directory / "target.md")
    with pytest.raises(AttachmentError, match="regular file"):
        store.ingest(tmp_path)
    with pytest.raises(AttachmentError, match="regular file"):
        store.ingest(fifo)
    assert not _blobs(store)


def test_home_expansion_is_explicitly_supported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _write(tmp_path / "home.md", b"home\n")
    monkeypatch.setenv("HOME", os.fspath(tmp_path))
    store = AttachmentStore(tmp_path / "store")

    descriptor = store.ingest("~/home.md")

    assert descriptor.display_name == source.name
    assert store.read(descriptor) == b"home\n"


def test_store_root_must_be_private_and_must_not_be_a_symlink(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    link = tmp_path / "store-link"
    link.symlink_to(public, target_is_directory=True)

    with pytest.raises(AttachmentError, match="permissions"):
        AttachmentStore(public)
    with pytest.raises(AttachmentError, match="untrusted link"):
        AttachmentStore(link)


def test_text_limit_and_declared_batch_limits_are_enforced(tmp_path: Path) -> None:
    assert MAX_ATTACHMENTS_PER_TURN == 8
    assert MAX_ATTACHMENTS_TOTAL_BYTES == 24 * 1024 * 1024
    assert MAX_TEXT_BYTES == 2 * 1024 * 1024
    assert MAX_IMAGE_BYTES == 8 * 1024 * 1024
    assert MAX_PDF_BYTES == 16 * 1024 * 1024
    store = AttachmentStore(tmp_path / "store")
    maximum = _write(tmp_path / "maximum.txt", b"a" * MAX_TEXT_BYTES)
    oversized = _write(tmp_path / "oversized.txt", b"a" * (MAX_TEXT_BYTES + 1))

    descriptor = store.ingest(maximum)
    assert descriptor.size_bytes == MAX_TEXT_BYTES
    with pytest.raises(AttachmentError, match="size limit"):
        store.ingest(oversized)

    sources = [_write(tmp_path / f"item-{index}.md", b"x") for index in range(9)]
    before = set(_blobs(store))
    with pytest.raises(AttachmentError, match="item limit"):
        store.ingest_many(sources)
    assert set(_blobs(store)) == before


def test_batch_total_limit_is_preflighted_without_partial_publication(tmp_path: Path) -> None:
    store = AttachmentStore(tmp_path / "store")
    pdf = _write(tmp_path / "large.pdf", _pdf(MAX_PDF_BYTES))
    image_size = MAX_IMAGE_BYTES
    image_payload_size = image_size - 57
    image = _write(tmp_path / "large.png", _png(1, 1, data=b"x" * image_payload_size))
    assert image.stat().st_size == image_size
    extra = _write(tmp_path / "extra.txt", b"x")

    descriptors = store.ingest_many([pdf, image])
    assert sum(item.size_bytes for item in descriptors) == MAX_ATTACHMENTS_TOTAL_BYTES
    for descriptor in descriptors:
        store.discard(descriptor)
    with pytest.raises(AttachmentError, match="total size limit"):
        store.ingest_many([pdf, image, extra])
    assert not _blobs(store)


def test_batch_publish_failure_rolls_back_already_published_blobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AttachmentStore(tmp_path / "store")
    sources = [
        _write(tmp_path / "one.md", b"one"),
        _write(tmp_path / "two.md", b"two"),
    ]
    original_publish = AttachmentStore._publish
    calls = 0

    def fail_second(self: AttachmentStore, prepared: object) -> AttachmentDescriptor:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise AttachmentError("injected publication failure")
        return original_publish(self, prepared)  # type: ignore[arg-type]

    monkeypatch.setattr(AttachmentStore, "_publish", fail_second)

    with pytest.raises(AttachmentError, match="injected"):
        store.ingest_many(sources)
    assert not _blobs(store)


def test_source_change_during_read_is_rejected_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write(tmp_path / "changing.md", b"initial")
    store = AttachmentStore(tmp_path / "store")
    original_read = os.read
    changed = False

    def mutate_then_read(file_descriptor: int, size: int) -> bytes:
        nonlocal changed
        if not changed:
            changed = True
            with source.open("ab") as stream:
                stream.write(b"changed")
        return original_read(file_descriptor, size)

    monkeypatch.setattr(attachments_module.os, "read", mutate_then_read)

    with pytest.raises(AttachmentError, match="changed"):
        store.ingest(source)
    assert not _blobs(store)


def test_blob_digest_permissions_and_link_count_are_revalidated(tmp_path: Path) -> None:
    source = _write(tmp_path / "source.md", b"original")
    store = AttachmentStore(tmp_path / "store")
    descriptor = store.ingest(source)
    blob = _blobs(store)[0]

    blob.chmod(0o644)
    with pytest.raises(AttachmentError, match="integrity"):
        store.read(descriptor)
    blob.chmod(0o600)
    alias = tmp_path / "hard-link"
    os.link(blob, alias)
    with pytest.raises(AttachmentError, match="integrity"):
        store.read(descriptor)
    alias.unlink()
    blob.write_bytes(b"tampered")
    blob.chmod(0o600)
    with pytest.raises(AttachmentError, match="integrity"):
        store.read(descriptor)


def test_forged_descriptor_cannot_read_or_discard_another_blob(tmp_path: Path) -> None:
    source = _write(tmp_path / "source.md", b"content")
    store = AttachmentStore(tmp_path / "store")
    descriptor = store.ingest(source)
    forged = replace(descriptor, sha256="0" * 64)

    with pytest.raises(AttachmentError, match="integrity"):
        store.read(forged)
    with pytest.raises(AttachmentError, match="integrity"):
        store.discard(forged)
    assert store.read(descriptor) == b"content"


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "wrong"},
        {"attachment_id": "../blob"},
        {"kind": []},
        {"size_bytes": True},
        {"size_bytes": 0},
        {"estimated_tokens": 0},
        {"estimated_tokens": 999_999},
        {"display_name": "../source.md"},
        {"sha256": "A" * 64},
    ],
)
def test_descriptor_parser_is_exact_and_fail_closed(
    tmp_path: Path,
    change: dict[str, object],
) -> None:
    source = _write(tmp_path / "source.md", b"content")
    descriptor = AttachmentStore(tmp_path / "store").ingest(source)
    value = descriptor.to_dict()
    value.update(change)

    with pytest.raises(AttachmentError, match="descriptor"):
        AttachmentDescriptor.from_value(value)

    value = descriptor.to_dict()
    value["unexpected"] = "content"
    with pytest.raises(AttachmentError, match="descriptor"):
        AttachmentDescriptor.from_value(value)


def test_read_only_accepts_validated_descriptor_objects(tmp_path: Path) -> None:
    source = _write(tmp_path / "source.md", b"content")
    store = AttachmentStore(tmp_path / "store")
    descriptor = store.ingest(source)

    with pytest.raises(AttachmentError, match="descriptor"):
        store.read(descriptor.to_dict())  # type: ignore[arg-type]
