"""Local, bounded extraction for teacher-provided learning resources.

Raw files are processed ephemerally on the dashboard host.  Only bounded text,
content hashes, and non-sensitive extraction metadata leave this module.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import posixpath
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
import wave
import xml.etree.ElementTree as ET
import zipfile

from .teacher_agent_multimodal import (
    TemporalTranscriptionProvider,
    VisualSemanticError,
    analyze_temporal_media,
    multimodal_provider_spec,
)
from .teacher_agent_vision import (
    LocalVisualEvidenceError,
    SUPPORTED_IMAGE_MIME_TYPES,
    extract_local_visual_evidence,
    local_visual_extractor_available,
)
from .teacher_agent_resource_retrieval import (
    MAX_INDEXED_RESOURCE_TEXT_CHARS,
    ResourceRetrievalError,
    TeachingResourceIndexStore,
    validate_resource_chunk_index,
)
from .teacher_agent_worker_isolation import (
    WorkerIsolationError,
    parser_network_isolation_available,
    sandboxed_parser_command,
)


TEACHING_RESOURCE_SCHEMA = "teaching_skill_miner.teaching_resource.v1"
MAX_RESOURCE_BYTES = 12 * 1024 * 1024
MAX_RESOURCE_TEXT_CHARS = 12_000
MAX_TEACHING_RESOURCES = 6

_TEXT_EXTENSIONS = frozenset({".txt", ".md", ".markdown"})
_WORD_EXTENSIONS = frozenset({".doc", ".docx", ".rtf"})
_PRESENTATION_EXTENSIONS = frozenset({".ppt", ".pptx"})
_SPREADSHEET_EXTENSIONS = frozenset({".csv", ".tsv", ".xlsx"})
_AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".webm"})
_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov"})
_SUPPORTED_EXTENSIONS = (
    _TEXT_EXTENSIONS
    | _WORD_EXTENSIONS
    | _PRESENTATION_EXTENSIONS
    | _SPREADSHEET_EXTENSIONS
    | _AUDIO_EXTENSIONS
    | _VIDEO_EXTENSIONS
    | frozenset({".pdf", ".png", ".jpg", ".jpeg", ".webp"})
)
_IMAGE_MIME_BY_EXTENSION = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
_MIME_BY_EXTENSION = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".rtf": "application/rtf",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".webm": "audio/webm",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    **_IMAGE_MIME_BY_EXTENSION,
}


class TeachingResourceError(ValueError):
    """Raised when a teaching resource cannot be safely converted to text."""


def runtime_supported_resource_extensions(
    *,
    temporal_transcription_provider: TemporalTranscriptionProvider | None = None,
) -> tuple[str, ...]:
    """Return the exact resource formats usable by this running host.

    Pure-Python/OOXML formats are always present. Formats backed by optional
    host tools or providers are only advertised after a side-effect-free
    availability check. The result is suitable for a bootstrap capability
    contract and intentionally excludes a leading dot.
    """

    supported = set(_TEXT_EXTENSIONS | _SPREADSHEET_EXTENSIONS)
    supported.update({".docx", ".pptx"})
    parser_sandbox = parser_network_isolation_available()
    if parser_sandbox and shutil.which("pdftotext") is not None:
        supported.add(".pdf")
    if parser_sandbox and shutil.which("textutil") is not None:
        supported.update({".doc", ".rtf", ".ppt"})
    if parser_sandbox and local_visual_extractor_available():
        supported.update(_IMAGE_MIME_BY_EXTENSION)
    if parser_sandbox and temporal_transcription_provider is not None:
        try:
            temporal_spec = multimodal_provider_spec(temporal_transcription_provider)
        except VisualSemanticError:
            temporal_spec = None
        if (
            temporal_spec is not None
            and temporal_spec["execution_scope"] == "local"
            and "temporal_transcription" in temporal_spec["capabilities"]
            and "transcription" in temporal_spec["capabilities"]
        ):
            supported.update(_AUDIO_EXTENSIONS | _VIDEO_EXTENSIONS)
    return tuple(sorted(extension.removeprefix(".") for extension in supported))


def _validate_office_archive(archive: zipfile.ZipFile, *, label: str) -> None:
    """Bound OOXML expansion and reject encrypted/traversal-shaped members."""

    members = archive.infolist()
    if len(members) > 20_000:
        raise TeachingResourceError(f"{label} 部件数量超出安全上限")
    expanded = 0
    for member in members:
        normalized = posixpath.normpath(member.filename.replace("\\", "/"))
        if (
            normalized.startswith("../")
            or normalized.startswith("/")
            or member.flag_bits & 0x1
            or member.file_size < 0
            or member.file_size > 32 * 1024 * 1024
        ):
            raise TeachingResourceError(f"{label} 压缩部件不安全")
        expanded += member.file_size
        if expanded > 96 * 1024 * 1024:
            raise TeachingResourceError(f"{label} 解压大小超出安全上限")


def _clean_text(value: str) -> str:
    value = value.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n{4,}", "\n\n\n", value)
    return value.strip()


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "gb18030"):
        try:
            return _clean_text(data.decode(encoding))
        except UnicodeDecodeError:
            continue
    decoded = _clean_text(data.decode("utf-8", errors="replace"))
    if decoded.count("\ufffd") > max(4, len(decoded) // 50):
        raise TeachingResourceError("文本文稿编码无法可靠识别")
    return decoded


def _extract_delimited_table(
    data: bytes, *, delimiter: str, label: str
) -> tuple[str, int | None, str]:
    """Extract literal row/cell order without inferring spreadsheet meaning."""

    decoded = _decode_text(data)
    try:
        rows = csv.reader(io.StringIO(decoded), delimiter=delimiter)
        rendered: list[str] = [
            "[结构化表格转写：保留行列次序；未核验单位、格式、合并单元格或语义]"
        ]
        formula_seen = False
        for row_index, row in enumerate(rows, 1):
            if row_index > 2_000:
                rendered.append("[表格截断：仅处理前 2000 行]")
                break
            bounded = [_clean_text(str(cell))[:1_000] for cell in row[:100]]
            if any(cell.startswith("=") for cell in bounded):
                formula_seen = True
            rendered.append(f"行 {row_index}: " + " | ".join(bounded))
    except csv.Error as exc:
        raise TeachingResourceError("分隔表格无法可靠解析") from exc
    if len(rendered) == 1:
        raise TeachingResourceError("表格中没有可提取的行")
    if formula_seen:
        rendered.append(
            "[公式转写候选：检测到以 = 开头的单元格；公式计算结果与引用关系未核验]"
        )
    return _clean_text("\n".join(rendered)), None, f"stdlib_{label}_rows"


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    name = "xl/sharedStrings.xml"
    if name not in archive.namelist():
        return []
    try:
        root = ET.fromstring(archive.read(name))
    except ET.ParseError as exc:
        raise TeachingResourceError("XLSX 共享字符串 XML 已损坏") from exc
    values: list[str] = []
    for item in (node for node in root.iter() if _local_name(node) == "si"):
        values.append(
            _clean_text(
                "".join(
                    str(child.text or "")
                    for child in item.iter()
                    if _local_name(child) == "t"
                )
            )[:1_000]
        )
        if len(values) > 100_000:
            raise TeachingResourceError("XLSX 共享字符串数量超出安全上限")
    return values


def _xlsx_cell_text(cell: ET.Element, shared: Sequence[str]) -> tuple[str, bool]:
    cell_type = str(cell.attrib.get("t", ""))
    formula = next(
        (
            str(node.text or "").strip()
            for node in cell
            if _local_name(node) == "f" and str(node.text or "").strip()
        ),
        "",
    )
    inline = "".join(
        str(node.text or "")
        for node in cell.iter()
        if _local_name(node) == "t" and str(node.text or "")
    ).strip()
    cached = next(
        (str(node.text or "").strip() for node in cell if _local_name(node) == "v"),
        "",
    )
    value = inline or cached
    if cell_type == "s" and cached:
        try:
            value = shared[int(cached)]
        except (ValueError, IndexError) as exc:
            raise TeachingResourceError("XLSX 共享字符串引用已损坏") from exc
    if formula:
        rendered = f"公式={formula}"
        if value:
            rendered += f"；缓存值={value}"
        return rendered[:1_000], True
    return value[:1_000], False


def _extract_xlsx(data: bytes) -> tuple[str, int | None, str]:
    """Extract cell/formula transcription layers from XLSX without calculating."""

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            _validate_office_archive(archive, label="XLSX")
            names = archive.namelist()
            sheets = sorted(
                (
                    name
                    for name in names
                    if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)
                ),
                key=lambda value: int(re.search(r"(\d+)", value).group(1)),
            )
            if not sheets:
                raise TeachingResourceError("文件不是有效的 XLSX 工作簿")
            shared = _xlsx_shared_strings(archive)
            rendered = [
                "[电子表格层：单元格与公式均为文字转写；未执行公式、未核验图表或格式语义]"
            ]
            formula_seen = False
            for sheet_index, name in enumerate(sheets[:256], 1):
                try:
                    root = ET.fromstring(archive.read(name))
                except ET.ParseError as exc:
                    raise TeachingResourceError("XLSX 工作表 XML 已损坏") from exc
                rendered.append(f"[工作表 {sheet_index}]")
                row_count = 0
                for row in (node for node in root.iter() if _local_name(node) == "row"):
                    row_count += 1
                    if row_count > 2_000:
                        rendered.append("[工作表截断：仅处理前 2000 行]")
                        break
                    cells: list[str] = []
                    for cell in (node for node in row if _local_name(node) == "c"):
                        coordinate = str(cell.attrib.get("r", "?"))[:20]
                        value, has_formula = _xlsx_cell_text(cell, shared)
                        formula_seen = formula_seen or has_formula
                        cells.append(f"{coordinate}={value}")
                        if len(cells) >= 100:
                            break
                    if cells:
                        rendered.append(
                            f"行 {row.attrib.get('r', row_count)}: " + " | ".join(cells)
                        )
            if formula_seen:
                rendered.append(
                    "[公式转写候选：XLSX 缓存值不是重新计算结果；公式正确性与依赖关系未核验]"
                )
            text = _clean_text("\n".join(rendered))
    except zipfile.BadZipFile as exc:
        raise TeachingResourceError("XLSX 压缩结构已损坏") from exc
    return text, len(sheets), "stdlib_xlsx_cells_and_formulas"


def _xml_text(xml_bytes: bytes, *, paragraph_breaks: bool) -> str:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise TeachingResourceError("Office 文档 XML 已损坏") from exc
    pieces: list[str] = []
    for node in root.iter():
        local_name = node.tag.rsplit("}", 1)[-1]
        if local_name == "t" and node.text:
            pieces.append(node.text)
        elif local_name == "tab":
            pieces.append("\t")
        elif local_name in {"br", "cr"}:
            pieces.append("\n")
        elif paragraph_breaks and local_name == "p":
            pieces.append("\n")
    return _clean_text("".join(pieces))


def _local_name(node: ET.Element) -> str:
    return node.tag.rsplit("}", 1)[-1]


def _descendant_text(node: ET.Element) -> str:
    return _clean_text(
        " ".join(
            str(candidate.text).strip()
            for candidate in node.iter()
            if _local_name(candidate) in {"t", "v"}
            and str(candidate.text or "").strip()
        )
    )


def _ooxml_structured_tables(root: ET.Element) -> list[str]:
    """Recover bounded row/cell order without inferring visual formatting."""

    rendered: list[str] = []
    for table_index, table in enumerate(
        (node for node in root.iter() if _local_name(node) == "tbl"), 1
    ):
        if table_index > 8:
            break
        rows: list[str] = []
        for row_index, row in enumerate(
            (node for node in table.iter() if _local_name(node) == "tr"), 1
        ):
            if row_index > 40:
                break
            cells = [
                _descendant_text(cell) for cell in row if _local_name(cell) == "tc"
            ][:20]
            if any(cells):
                rows.append(f"行 {row_index}: " + " | ".join(cells))
        if rows:
            rendered.append(
                "[结构化表格：来自 OOXML 单元格顺序，不含颜色或版式含义]\n"
                + "\n".join(rows)
            )
    return rendered


def _ooxml_formula_candidates(root: ET.Element) -> list[str]:
    """Return literal Office Math text while preserving an abstention marker."""

    formulas: list[str] = []
    for node in (item for item in root.iter() if _local_name(item) == "oMath"):
        text = _descendant_text(node)
        if text and text not in formulas:
            formulas.append(text[:500])
        if len(formulas) >= 16:
            break
    if not formulas:
        return []
    return [
        "[公式 OOXML 文字转写候选]\n" + "\n".join(formulas),
        "[视觉复核：公式仅完成线性文字转写；上下标、分式、矩阵与二维符号关系未核验]",
    ]


def _extract_docx(data: bytes) -> tuple[str, int | None, str]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            _validate_office_archive(archive, label="DOCX")
            if "word/document.xml" not in archive.namelist():
                raise TeachingResourceError("文件不是有效的 DOCX 文稿")
            document_xml = archive.read("word/document.xml")
            text = _xml_text(document_xml, paragraph_breaks=True)
            try:
                root = ET.fromstring(document_xml)
            except ET.ParseError as exc:
                raise TeachingResourceError("Office 文档 XML 已损坏") from exc
            structured = [
                *_ooxml_structured_tables(root),
                *_ooxml_formula_candidates(root),
            ]
            if structured:
                text = _clean_text(text + "\n\n" + "\n\n".join(structured))
    except zipfile.BadZipFile as exc:
        raise TeachingResourceError("DOCX 压缩结构已损坏") from exc
    return text, None, "stdlib_docx_xml"


def _slide_number(name: str) -> int:
    match = re.search(r"slide(\d+)\.xml$", name)
    return int(match.group(1)) if match else 0


def _pptx_related_parts(
    archive: zipfile.ZipFile, slide_name: str
) -> tuple[str | None, dict[str, list[str]]]:
    """Resolve notes and visual relationships for one slide without rendering."""

    basename = posixpath.basename(slide_name)
    relationship_name = posixpath.join(
        posixpath.dirname(slide_name), "_rels", basename + ".rels"
    )
    notes_name: str | None = None
    visual_parts: dict[str, list[str]] = {
        "image": [],
        "chart": [],
        "diagram": [],
    }
    if relationship_name not in archive.namelist():
        return notes_name, visual_parts
    try:
        root = ET.fromstring(archive.read(relationship_name))
    except ET.ParseError as exc:
        raise TeachingResourceError("PPTX 幻灯片关系 XML 已损坏") from exc
    for relationship in root.iter():
        if relationship.tag.rsplit("}", 1)[-1] != "Relationship":
            continue
        relationship_type = str(relationship.attrib.get("Type", ""))
        target = str(relationship.attrib.get("Target", ""))
        if not target or relationship.attrib.get("TargetMode") == "External":
            continue
        normalized = posixpath.normpath(
            posixpath.join(posixpath.dirname(slide_name), target)
        ).lstrip("/")
        if normalized.startswith("../") or not normalized.startswith("ppt/"):
            continue
        if relationship_type.endswith("/notesSlide"):
            notes_name = normalized
        elif relationship_type.endswith("/image"):
            visual_parts["image"].append(normalized)
        elif relationship_type.endswith("/chart"):
            visual_parts["chart"].append(normalized)
        elif relationship_type.endswith("/diagramData"):
            visual_parts["diagram"].append(normalized)
    return notes_name, visual_parts


def _pptx_chart_cache(xml_bytes: bytes) -> str:
    """Render cached chart series without inferring axes, shapes, or trends."""

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise TeachingResourceError("PPTX 图表 XML 已损坏") from exc
    series_rows: list[str] = []
    for series_index, series in enumerate(
        (node for node in root.iter() if _local_name(node) == "ser"), 1
    ):
        if series_index > 12:
            break
        name = ""
        categories: list[str] = []
        values: list[str] = []
        for child in series:
            kind = _local_name(child)
            descendants = [
                str(item.text).strip()
                for item in child.iter()
                if _local_name(item) == "v" and str(item.text or "").strip()
            ]
            if kind == "tx" and descendants:
                name = descendants[0]
            elif kind in {"cat", "xVal"}:
                categories = descendants[:40]
            elif kind in {"val", "yVal"}:
                values = descendants[:40]
        fields = [f"系列 {series_index}" + (f"（{name}）" if name else "")]
        if categories:
            fields.append("类别=" + " | ".join(categories))
        if values:
            fields.append("数值=" + " | ".join(values))
        if categories or values or name:
            series_rows.append("；".join(fields))
    if not series_rows:
        return ""
    return (
        "[结构化图表数据：来自 OOXML 缓存；不代表已理解坐标轴、颜色、形状或趋势]\n"
        + "\n".join(series_rows)
    )


def _pptx_diagram_text(xml_bytes: bytes) -> str:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise TeachingResourceError("PPTX 图示 XML 已损坏") from exc
    values: list[str] = []
    for node in root.iter():
        if _local_name(node) == "t" and str(node.text or "").strip():
            value = str(node.text).strip()
            if value not in values:
                values.append(value[:300])
        if len(values) >= 80:
            break
    if not values:
        return ""
    return (
        "[结构化图示文字：来自 OOXML 节点；连接、方向与空间关系未核验]\n"
        + " | ".join(values)
    )


def _pptx_inline_visual_kinds(xml_bytes: bytes) -> set[str]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise TeachingResourceError("PPTX 幻灯片 XML 已损坏") from exc
    kinds: set[str] = set()
    for node in root.iter():
        local_name = node.tag.rsplit("}", 1)[-1]
        if local_name in {"pic", "blip"}:
            kinds.add("image")
        elif local_name == "chart":
            kinds.add("chart")
        elif local_name in {"graphicFrame", "graphicData"}:
            kinds.add("diagram")
    return kinds


def _slide_notes_conflict(slide_text: str, notes_text: str) -> str | None:
    """Detect only explicit lexical conflict signals; never decide which is true."""

    if not slide_text or not notes_text:
        return None
    slide_numbers = set(re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", slide_text))
    notes_numbers = set(re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", notes_text))
    if slide_numbers and notes_numbers and slide_numbers != notes_numbers:
        return "正文与讲者备注包含不同数值；需教师确认版本，当前不得自动合并"
    negative = re.compile(r"(?:\bnot\b|\bnever\b|不应|不是|不能|无需|禁止)", re.I)
    if bool(negative.search(slide_text)) != bool(negative.search(notes_text)):
        slide_terms = set(re.findall(r"[A-Za-z]{3,}|[\u4e00-\u9fff]{2,}", slide_text))
        note_terms = set(re.findall(r"[A-Za-z]{3,}|[\u4e00-\u9fff]{2,}", notes_text))
        if slide_terms & note_terms:
            return "正文与讲者备注出现相反的否定表述；需教师确认，当前必须弃权"
    return None


def _extract_pptx(
    data: bytes,
) -> tuple[str, int | None, str]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            _validate_office_archive(archive, label="PPTX")
            slides = sorted(
                (
                    name
                    for name in archive.namelist()
                    if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
                ),
                key=_slide_number,
            )
            if not slides:
                raise TeachingResourceError("文件不是有效的 PPTX 演示文稿")
            rendered: list[str] = []
            for index, name in enumerate(slides, 1):
                slide_xml = archive.read(name)
                slide_text = _xml_text(slide_xml, paragraph_breaks=False)
                notes_name, relationship_visuals = _pptx_related_parts(archive, name)
                visual_kinds = {
                    kind for kind, paths in relationship_visuals.items() if paths
                } | _pptx_inline_visual_kinds(slide_xml)
                notes_text = ""
                if notes_name and notes_name in archive.namelist():
                    notes_text = _xml_text(
                        archive.read(notes_name), paragraph_breaks=True
                    )
                parts = [f"[第 {index} 页]"]
                if slide_text:
                    parts.append(slide_text)
                try:
                    slide_root = ET.fromstring(slide_xml)
                except ET.ParseError as exc:
                    raise TeachingResourceError("PPTX 幻灯片 XML 已损坏") from exc
                parts.extend(_ooxml_structured_tables(slide_root))
                parts.extend(_ooxml_formula_candidates(slide_root))
                for chart_name in relationship_visuals["chart"][:8]:
                    if chart_name in archive.namelist():
                        chart_data = _pptx_chart_cache(archive.read(chart_name))
                        if chart_data:
                            parts.append(chart_data)
                for diagram_name in relationship_visuals["diagram"][:8]:
                    if diagram_name in archive.namelist():
                        diagram_text = _pptx_diagram_text(archive.read(diagram_name))
                        if diagram_text:
                            parts.append(diagram_text)
                if visual_kinds:
                    parts.append(
                        "[视觉复核：本页含图片、图表或图示；结构化缓存不包含完整视觉编码与空间语义]"
                    )
                if notes_text:
                    conflict = _slide_notes_conflict(slide_text, notes_text)
                    if conflict:
                        parts.append(
                            "[视觉复核：本页正文与讲者备注存在未解决冲突；"
                            "确认前两层均不得用于教学合成]"
                        )
                    parts.extend(("[讲者备注]", notes_text))
                    if conflict:
                        parts.append(f"[内容冲突待确认：{conflict}]")
                        parts.append(
                            "[视觉复核：本页正文与讲者备注存在未解决冲突；"
                            "确认前两层均不得用于教学合成]"
                        )
                rendered.append("\n".join(parts))
            text = "\n\n".join(rendered)
    except zipfile.BadZipFile as exc:
        raise TeachingResourceError("PPTX 压缩结构已损坏") from exc
    return text, len(slides), "stdlib_pptx_xml_with_notes"


def _run_file_converter(
    data: bytes,
    *,
    suffix: str,
    command: list[str],
    engine: str,
) -> tuple[str, int | None, str]:
    executable = shutil.which(command[0])
    if executable is None:
        raise TeachingResourceError(f"本机缺少 {command[0]}，无法读取该文件")
    with tempfile.NamedTemporaryFile(suffix=suffix) as source:
        source.write(data)
        source.flush()
        try:
            parser_command = sandboxed_parser_command(
                [executable, *command[1:], source.name]
            )
        except WorkerIsolationError as exc:
            raise TeachingResourceError(
                "本机解析器隔离不可用，无法安全读取该文件"
            ) from exc
        completed = subprocess.run(
            parser_command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    if completed.returncode != 0:
        raise TeachingResourceError(f"{engine} 无法读取该文件")
    text = _clean_text(completed.stdout.decode("utf-8", errors="replace"))
    return text, None, engine


def _extract_pdf(data: bytes) -> tuple[str, int | None, str]:
    if not data.startswith(b"%PDF-"):
        raise TeachingResourceError("文件扩展名为 PDF，但内容不是有效 PDF")
    executable = shutil.which("pdftotext")
    if executable is None:
        raise TeachingResourceError("本机缺少 pdftotext，无法读取 PDF")
    with tempfile.NamedTemporaryFile(suffix=".pdf") as source:
        source.write(data)
        source.flush()
        try:
            parser_command = sandboxed_parser_command(
                [executable, "-layout", source.name, "-"]
            )
        except WorkerIsolationError as exc:
            raise TeachingResourceError(
                "本机解析器隔离不可用，无法安全读取 PDF"
            ) from exc
        completed = subprocess.run(
            parser_command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    if completed.returncode != 0:
        raise TeachingResourceError("pdftotext 无法读取该 PDF")
    decoded = completed.stdout.decode("utf-8", errors="replace")
    raw_pages = decoded.split("\f")
    if raw_pages and not raw_pages[-1].strip():
        raw_pages.pop()
    if not raw_pages:
        raw_pages = [""]
    page_count = len(raw_pages) or None
    rendered_pages: list[str] = []
    for index, page in enumerate(raw_pages, 1):
        page_text = _clean_text(page)
        if page_text:
            layer = page_text
            review = (
                "[视觉复核：PDF 仅完成嵌入文字提取；本页图表、几何、手写、"
                "公式二维关系与布局未核验]"
            )
        else:
            layer = "[扫描页检测：未发现可用嵌入文字；未配置页面 OCR/视觉提供商]"
            review = (
                "[视觉复核：扫描 PDF 页当前必须弃权；不得从空转写推断语义、"
                "评分或掌握度]"
            )
        rendered_pages.append(f"[第 {index} 页]\n{layer}\n{review}")
    rendered = "\n\n".join(rendered_pages)
    return _clean_text(rendered), page_count, "pdftotext_local_cli_pages"


def _extract_legacy_document(
    data: bytes, extension: str
) -> tuple[str, int | None, str]:
    return _run_file_converter(
        data,
        suffix=extension,
        command=["textutil", "-convert", "txt", "-stdout"],
        engine="macos_textutil_local_cli",
    )


def _temporal_signature(data: bytes, mime_type: str) -> bool:
    if mime_type in {"audio/wav", "audio/x-wav"}:
        return data.startswith(b"RIFF") and data[8:12] == b"WAVE"
    if mime_type == "audio/mpeg":
        return data.startswith(b"ID3") or (
            len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0
        )
    if mime_type in {"audio/mp4", "audio/x-m4a", "video/mp4", "video/quicktime"}:
        return len(data) >= 12 and data[4:8] == b"ftyp"
    if mime_type in {"audio/webm", "video/webm"}:
        return data.startswith(b"\x1aE\xdf\xa3")
    return False


def _safe_nonnegative_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed >= 0 else 0


def inspect_temporal_media_metadata(
    data: bytes, mime_type: str, *, display_name: str
) -> dict[str, Any]:
    """Return local-only audio/video container metadata without transcription."""

    if not isinstance(data, bytes) or not 1 <= len(data) <= MAX_RESOURCE_BYTES:
        raise TeachingResourceError("音视频大小超出教学资源安全上限")
    normalized_mime = str(mime_type or "").split(";", 1)[0].strip().lower()
    if not (
        normalized_mime.startswith("audio/") or normalized_mime.startswith("video/")
    ) or not _temporal_signature(data, normalized_mime):
        raise TeachingResourceError("音视频容器签名与 MIME 类型不匹配")
    source_modality = "audio" if normalized_mime.startswith("audio/") else "video"
    safe_name = Path(str(display_name or "media")).name[:160]
    metadata: dict[str, Any] = {
        "schema": "teaching_skill_miner.temporal_media_metadata.v1",
        "source_modality": source_modality,
        "display_name": safe_name,
        "mime_type": normalized_mime,
        "byte_size": len(data),
        "content_sha256": hashlib.sha256(data).hexdigest(),
        "metadata_status": "container_signature_only",
        "metadata_engine": "stdlib_signature_probe",
        "duration_ms": None,
        "audio_streams": [],
        "video_streams": [],
        "raw_media_retained": False,
        "remote_media_sent": False,
        "transcription_performed": False,
    }
    if normalized_mime in {"audio/wav", "audio/x-wav"}:
        try:
            with wave.open(io.BytesIO(data), "rb") as source:
                frame_count = source.getnframes()
                sample_rate = source.getframerate()
                channels = source.getnchannels()
                sample_width = source.getsampwidth()
        except (EOFError, wave.Error) as exc:
            raise TeachingResourceError("WAV 音频头无法可靠解析") from exc
        duration_ms = round(frame_count * 1000 / sample_rate) if sample_rate else None
        metadata.update(
            {
                "metadata_status": "parsed",
                "metadata_engine": "stdlib_wave_header",
                "duration_ms": duration_ms,
                "audio_streams": [
                    {
                        "stream_index": 0,
                        "codec": "pcm",
                        "channels": channels,
                        "sample_rate_hz": sample_rate,
                        "sample_width_bytes": sample_width,
                    }
                ],
            }
        )
        return metadata

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return metadata
    suffix = Path(safe_name).suffix or (
        ".mp4" if source_modality == "video" else ".bin"
    )
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix) as source:
            source.write(data)
            source.flush()
            completed = subprocess.run(
                sandboxed_parser_command(
                    [
                        ffprobe,
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration:stream=index,codec_type,codec_name,width,height,sample_rate,channels",
                        "-of",
                        "json",
                        source.name,
                    ]
                ),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
            )
    except (OSError, subprocess.SubprocessError, WorkerIsolationError):
        return metadata
    if completed.returncode != 0 or len(completed.stdout) > 256_000:
        return metadata
    try:
        probed = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return metadata
    if not isinstance(probed, Mapping):
        return metadata
    duration_raw = (
        (probed.get("format") or {}).get("duration")
        if isinstance(probed.get("format"), Mapping)
        else None
    )
    try:
        duration = float(duration_raw)
    except (TypeError, ValueError):
        duration = math.nan
    if math.isfinite(duration) and 0 <= duration <= 86_400:
        metadata["duration_ms"] = round(duration * 1000)
    streams = probed.get("streams", [])
    if isinstance(streams, list):
        for item in streams[:16]:
            if not isinstance(item, Mapping):
                continue
            stream_type = item.get("codec_type")
            base = {
                "stream_index": _safe_nonnegative_int(item.get("index", 0)),
                "codec": str(item.get("codec_name", "unknown"))[:40],
            }
            if stream_type == "audio":
                metadata["audio_streams"].append(
                    {
                        **base,
                        "channels": _safe_nonnegative_int(item.get("channels", 0)),
                        "sample_rate_hz": _safe_nonnegative_int(
                            item.get("sample_rate", 0)
                        ),
                    }
                )
            elif stream_type == "video":
                metadata["video_streams"].append(
                    {
                        **base,
                        "width": _safe_nonnegative_int(item.get("width", 0)),
                        "height": _safe_nonnegative_int(item.get("height", 0)),
                    }
                )
    metadata["metadata_status"] = "parsed"
    metadata["metadata_engine"] = "ffprobe_local_cli"
    return metadata


def _timestamp(milliseconds: int) -> str:
    seconds, millis = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _extract_temporal_media(
    data: bytes,
    mime_type: str,
    display_name: str,
    provider: TemporalTranscriptionProvider | None,
) -> tuple[str, int | None, str, dict[str, Any], dict[str, Any]]:
    metadata = inspect_temporal_media_metadata(
        data, mime_type, display_name=display_name
    )
    if provider is None:
        raise TeachingResourceError(
            "本地音视频转写适配器未配置；已能读取容器元数据，但元数据不能代替转写"
        )
    try:
        evidence = analyze_temporal_media(
            data,
            mime_type,
            task_context=(
                "逐字转写教学音视频，保留每段开始和结束时间；不推断学习者正确性或掌握度。"
            ),
            provider=provider,
        )
    except VisualSemanticError as exc:
        raise TeachingResourceError("本地音视频转写未通过证据边界") from exc
    transcription = evidence["transcription"]
    segments = transcription["segments"]
    lines = [
        "[音视频转写层：时间戳来自本地适配器；内容未经语义或事实核验，不可用于评分]"
    ]
    for segment in segments:
        lines.append(
            f"[{_timestamp(segment['start_ms'])}-{_timestamp(segment['end_ms'])}; "
            f"{segment['evidence_locator']}] {segment['text']}"
        )
    metadata["transcription_performed"] = True
    metadata["transcription_provider_id"] = evidence["provider_id"]
    metadata["transcription_sha256"] = hashlib.sha256(
        transcription["text"].encode("utf-8")
    ).hexdigest()
    return (
        _clean_text("\n".join(lines)),
        None,
        "local_timestamped_multimodal_adapter",
        metadata,
        evidence,
    )


def _extract_image(
    data: bytes, mime_type: str, display_name: str
) -> tuple[str, int | None, str, bool]:
    try:
        evidence = extract_local_visual_evidence(
            data, mime_type, display_name=display_name
        )
    except (LocalVisualEvidenceError, OSError, TypeError, ValueError) as exc:
        raise TeachingResourceError("本机无法读取该图片") from exc
    text = _clean_text(str(evidence.get("recognized_text", "")))
    if not text:
        raise TeachingResourceError(
            "图片中未识别到可用文字；复杂图表或纯视觉内容需要接入视觉模型"
        )
    return (
        text,
        1,
        str(evidence.get("engine") or "local_image_ocr"),
        bool(evidence.get("needs_student_confirmation", False)),
    )


def _resource_evidence_contract(
    text: str,
    *,
    resource_type: str,
    page_count: int | None,
    temporal_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Project extraction layers without promoting any layer to assessment."""

    layers: list[dict[str, Any]] = []

    def add_layer(kind: str, locator: str, status: str) -> None:
        if len(layers) >= 256:
            return
        layers.append(
            {
                "layer_id": f"layer_{len(layers) + 1:03d}",
                "kind": kind,
                "evidence_locator": locator,
                "status": status,
                "semantic_understanding_established": False,
                "visual_verification_status": "not_verified",
                "grading_evidence_allowed": False,
                "mastery_evidence_allowed": False,
            }
        )

    add_layer("text_transcription", "resource/extracted-text", "candidate")
    marker_layers = (
        ("[结构化表格", "table_structure", "resource/table"),
        ("[电子表格层", "spreadsheet_cells", "resource/spreadsheet"),
        ("[结构化图表", "chart_cache", "resource/chart-cache"),
        ("[结构化图示", "diagram_text", "resource/diagram-text"),
        ("[公式", "formula_transcription", "resource/formula"),
        ("[扫描页检测", "scanned_page", "resource/scanned-page"),
        ("[讲者备注]", "speaker_notes", "resource/speaker-notes"),
        ("[音视频转写层", "temporal_transcription", "resource/timeline"),
    )
    for marker, kind, locator in marker_layers:
        if marker in text:
            status = "unavailable" if kind == "scanned_page" else "candidate"
            add_layer(kind, locator, status)
    visual_pending = "[视觉复核：" in text or any(
        layer["kind"]
        in {
            "chart_cache",
            "diagram_text",
            "formula_transcription",
            "scanned_page",
        }
        for layer in layers
    )
    if visual_pending:
        add_layer("visual_semantics", "resource/visual-layer", "not_verified")

    conflict_matches = re.findall(r"\[内容冲突待确认：([^\]]+)\]", text)
    conflicts = [
        {
            "conflict_id": f"resource_conflict_{index:03d}",
            "kind": "slide_notes_disagreement",
            "description": description[:500],
            "evidence_locators": ["slide/body", "slide/speaker-notes"],
            "resolution_status": "unresolved",
        }
        for index, description in enumerate(conflict_matches[:32], 1)
    ]
    if conflicts:
        decision = "requires_confirmation"
    elif visual_pending or resource_type in {"audio", "video"}:
        decision = "abstain_from_unverified_semantics"
    else:
        decision = "usable_as_untrusted_teaching_context"
    contract = {
        "schema": "teaching_skill_miner.resource_evidence_layers.v1",
        "layers": layers,
        "conflicts": conflicts,
        "decision": decision,
        "transcription_is_semantic_understanding": False,
        "semantic_analysis_is_answer_correctness": False,
        "grading_evidence_allowed": False,
        "mastery_evidence_allowed": False,
        "visual_verification_status": "not_verified"
        if visual_pending
        else "not_applicable",
        "page_count_bound": page_count,
        "temporal_provenance": [],
    }
    if isinstance(temporal_evidence, Mapping):
        transcription = temporal_evidence.get("transcription", {})
        if isinstance(transcription, Mapping):
            contract["temporal_provenance"] = [
                {
                    "segment_id": segment.get("segment_id"),
                    "start_ms": segment.get("start_ms"),
                    "end_ms": segment.get("end_ms"),
                    "evidence_locator": segment.get("evidence_locator"),
                    "confidence": segment.get("confidence"),
                }
                for segment in transcription.get("segments", [])
                if isinstance(segment, Mapping)
            ]
    return contract


def extract_teaching_resource(
    data: bytes,
    mime_type: str,
    *,
    display_name: str,
    index_store: TeachingResourceIndexStore | None = None,
    temporal_transcription_provider: TemporalTranscriptionProvider | None = None,
) -> dict[str, Any]:
    """Extract one resource without retaining or remotely sending raw media.

    When a private ``index_store`` is supplied, a larger but still bounded
    local text projection is atomically indexed before this function returns.
    The returned live-session descriptor remains capped at 12,000 characters.
    """

    if not isinstance(data, bytes) or not 1 <= len(data) <= MAX_RESOURCE_BYTES:
        raise TeachingResourceError(
            f"教学资源大小必须在 1 到 {MAX_RESOURCE_BYTES} 字节之间"
        )
    safe_name = Path(str(display_name or "")).name[:160].strip()
    if not safe_name:
        raise TeachingResourceError("教学资源文件名不能为空")
    extension = Path(safe_name).suffix.lower()
    if extension not in _SUPPORTED_EXTENSIONS:
        raise TeachingResourceError("暂不支持该教学资源格式")
    declared_mime = str(mime_type or "").split(";", 1)[0].strip().lower()
    normalized_mime = _MIME_BY_EXTENSION[extension]
    if extension == ".webm":
        if declared_mime not in {"audio/webm", "video/webm"}:
            raise TeachingResourceError(
                "WebM 必须声明 audio/webm 或 video/webm，不能从扩展名猜测模态"
            )
        normalized_mime = declared_mime
    elif extension == ".mp4" and declared_mime == "audio/mp4":
        normalized_mime = declared_mime
    elif extension == ".m4a" and declared_mime == "audio/x-m4a":
        normalized_mime = declared_mime
    temporal_metadata: dict[str, Any] | None = None
    temporal_evidence: dict[str, Any] | None = None
    if extension in _IMAGE_MIME_BY_EXTENSION:
        image_mime = declared_mime or normalized_mime
        if image_mime not in SUPPORTED_IMAGE_MIME_TYPES:
            raise TeachingResourceError("图片 MIME 类型不受支持")
        text, page_count, engine, needs_review = _extract_image(
            data, image_mime, safe_name
        )
        text = text + "\n[视觉复核：图片仅完成 OCR；纯视觉内容与空间关系未被理解]"
        resource_type = "image_ocr"
        normalized_mime = image_mime
    elif extension in _TEXT_EXTENSIONS:
        text, page_count, engine, needs_review = (
            _decode_text(data),
            None,
            "bounded_local_text_decode",
            False,
        )
        resource_type = "text"
    elif extension == ".csv":
        text, page_count, engine = _extract_delimited_table(
            data, delimiter=",", label="csv"
        )
        needs_review = True
        resource_type = "spreadsheet"
    elif extension == ".tsv":
        text, page_count, engine = _extract_delimited_table(
            data, delimiter="\t", label="tsv"
        )
        needs_review = True
        resource_type = "spreadsheet"
    elif extension == ".xlsx":
        text, page_count, engine = _extract_xlsx(data)
        needs_review = True
        resource_type = "spreadsheet"
    elif extension == ".docx":
        text, page_count, engine = _extract_docx(data)
        needs_review = False
        resource_type = "document"
    elif extension == ".pptx":
        text, page_count, engine = _extract_pptx(data)
        needs_review = False
        resource_type = "presentation"
    elif extension == ".pdf":
        text, page_count, engine = _extract_pdf(data)
        needs_review = False
        resource_type = "pdf"
    elif extension in _WORD_EXTENSIONS | _PRESENTATION_EXTENSIONS:
        text, page_count, engine = _extract_legacy_document(data, extension)
        needs_review = False
        resource_type = "presentation" if extension == ".ppt" else "document"
    elif extension in _AUDIO_EXTENSIONS | _VIDEO_EXTENSIONS:
        text, page_count, engine, temporal_metadata, temporal_evidence = (
            _extract_temporal_media(
                data,
                normalized_mime,
                safe_name,
                temporal_transcription_provider,
            )
        )
        needs_review = True
        resource_type = "audio" if normalized_mime.startswith("audio/") else "video"
    else:  # pragma: no cover - the supported-extension guard is exhaustive.
        raise TeachingResourceError("暂不支持该教学资源格式")
    if not text:
        raise TeachingResourceError("教学资源中没有提取到可用文字")
    # A successful text/OCR extraction is not equivalent to verified visual
    # semantics.  Preserve that distinction at the resource level as well as
    # in the per-page marker used by retrieval.
    needs_review = bool(
        needs_review
        or "[视觉复核：" in text
        or "[内容冲突待确认：" in text
        or "[公式转写候选：" in text
    )
    original_char_count = len(text)
    extracted_text = text[:MAX_RESOURCE_TEXT_CHARS].rstrip()
    truncated = original_char_count > len(extracted_text)
    content_sha256 = hashlib.sha256(data).hexdigest()
    resource_id = "res_" + content_sha256[:20]
    evidence_contract = _resource_evidence_contract(
        text,
        resource_type=resource_type,
        page_count=page_count,
        temporal_evidence=temporal_evidence,
    )
    resource = {
        "schema": TEACHING_RESOURCE_SCHEMA,
        "resource_id": resource_id,
        "display_name": safe_name,
        "resource_type": resource_type,
        "mime_type": normalized_mime,
        "byte_size": len(data),
        "content_sha256": content_sha256,
        "extracted_text": extracted_text,
        "extracted_char_count": len(extracted_text),
        "original_extracted_char_count": original_char_count,
        "truncated": truncated,
        "page_count": page_count,
        "extraction_engine": engine,
        "needs_review": needs_review,
        "raw_media_retained": False,
        "remote_media_sent": False,
        "remote_representation": "bounded_redacted_text_only",
        "evidence_contract": evidence_contract,
        "grading_evidence_allowed": False,
        "mastery_evidence_allowed": False,
        "requires_confirmation": evidence_contract["decision"]
        == "requires_confirmation",
        "temporal_metadata": temporal_metadata,
        "temporal_transcription_receipt": (
            {
                "provider_id": temporal_evidence.get("provider_id"),
                "provider_spec_sha256": temporal_evidence.get("provider_spec_sha256"),
                "provider_result_sha256": temporal_evidence.get(
                    "provider_result_sha256"
                ),
                "media_sha256": temporal_evidence.get("media_sha256"),
                "remote_media_sent": temporal_evidence.get("remote_media_sent"),
            }
            if isinstance(temporal_evidence, Mapping)
            else None
        ),
    }
    if index_store is not None:
        if not isinstance(index_store, TeachingResourceIndexStore):
            raise TeachingResourceError("教学资源索引存储器无效")
        indexed_text = text[:MAX_INDEXED_RESOURCE_TEXT_CHARS].rstrip()
        try:
            index_store.put(resource, indexed_text=indexed_text)
        except ResourceRetrievalError as exc:
            raise TeachingResourceError("无法持久化教学资源检索索引") from exc
    return resource


def teaching_resource_for_session(resource: Mapping[str, Any]) -> dict[str, Any]:
    """Remove staging-only metadata and validate the persisted text descriptor."""

    candidate = {
        str(key): value
        for key, value in dict(resource).items()
        if key != "staged_resource_id"
    }
    validate_teaching_resources([candidate])
    contract = candidate.get("evidence_contract", {})
    decision = contract.get("decision") if isinstance(contract, Mapping) else None
    if candidate.get("requires_confirmation") is True or decision == (
        "requires_confirmation"
    ):
        raise TeachingResourceError(
            "teaching resource has an unresolved cross-layer conflict; "
            "teacher confirmation is required before session use"
        )
    if decision == "abstain_from_unverified_semantics":
        raise TeachingResourceError(
            "teaching resource abstains from unverified semantics; a reviewed "
            "resource projection is required before session use"
        )
    return candidate


def _validate_resource_evidence_contract(
    contract: Any, *, resource: Mapping[str, Any]
) -> None:
    if not isinstance(contract, Mapping) or contract.get("schema") != (
        "teaching_skill_miner.resource_evidence_layers.v1"
    ):
        raise TeachingResourceError("teaching resource evidence contract is invalid")
    layers = contract.get("layers")
    if not isinstance(layers, list) or not 1 <= len(layers) <= 256:
        raise TeachingResourceError("teaching resource evidence layers are invalid")
    layer_kinds = {
        "text_transcription",
        "table_structure",
        "spreadsheet_cells",
        "chart_cache",
        "diagram_text",
        "formula_transcription",
        "scanned_page",
        "speaker_notes",
        "temporal_transcription",
        "visual_semantics",
    }
    for index, layer in enumerate(layers, 1):
        if (
            not isinstance(layer, Mapping)
            or layer.get("layer_id") != f"layer_{index:03d}"
            or layer.get("kind") not in layer_kinds
            or not isinstance(layer.get("evidence_locator"), str)
            or not str(layer["evidence_locator"]).strip()
            or len(str(layer["evidence_locator"])) > 240
            or layer.get("status") not in {"candidate", "unavailable", "not_verified"}
            or layer.get("semantic_understanding_established") is not False
            or layer.get("grading_evidence_allowed") is not False
            or layer.get("mastery_evidence_allowed") is not False
            or layer.get("visual_verification_status") != "not_verified"
        ):
            raise TeachingResourceError("teaching resource evidence layer is invalid")
    conflicts = contract.get("conflicts")
    if not isinstance(conflicts, list) or len(conflicts) > 32:
        raise TeachingResourceError("teaching resource conflicts are invalid")
    for index, conflict in enumerate(conflicts, 1):
        if (
            not isinstance(conflict, Mapping)
            or conflict.get("conflict_id") != f"resource_conflict_{index:03d}"
            or conflict.get("kind") != "slide_notes_disagreement"
            or not isinstance(conflict.get("description"), str)
            or not str(conflict["description"]).strip()
            or len(str(conflict["description"])) > 500
            or conflict.get("evidence_locators")
            != ["slide/body", "slide/speaker-notes"]
            or conflict.get("resolution_status") != "unresolved"
        ):
            raise TeachingResourceError("teaching resource conflict is invalid")
    decision = contract.get("decision")
    if decision not in {
        "usable_as_untrusted_teaching_context",
        "requires_confirmation",
        "abstain_from_unverified_semantics",
    } or (bool(conflicts) != (decision == "requires_confirmation")):
        raise TeachingResourceError("teaching resource review decision is invalid")
    if (
        contract.get("transcription_is_semantic_understanding") is not False
        or contract.get("semantic_analysis_is_answer_correctness") is not False
        or contract.get("grading_evidence_allowed") is not False
        or contract.get("mastery_evidence_allowed") is not False
        or contract.get("visual_verification_status")
        not in {"not_verified", "not_applicable"}
        or contract.get("page_count_bound") != resource.get("page_count")
    ):
        raise TeachingResourceError("teaching resource assessment boundary is invalid")
    temporal = contract.get("temporal_provenance")
    if not isinstance(temporal, list) or len(temporal) > 512:
        raise TeachingResourceError("teaching resource temporal provenance is invalid")
    previous_start = -1
    for index, segment in enumerate(temporal, 1):
        if (
            not isinstance(segment, Mapping)
            or segment.get("segment_id") != f"segment_{index:04d}"
            or isinstance(segment.get("start_ms"), bool)
            or not isinstance(segment.get("start_ms"), int)
            or isinstance(segment.get("end_ms"), bool)
            or not isinstance(segment.get("end_ms"), int)
            or segment["start_ms"] < previous_start
            or segment["end_ms"] <= segment["start_ms"]
            or not isinstance(segment.get("evidence_locator"), str)
            or not str(segment["evidence_locator"]).strip()
            or isinstance(segment.get("confidence"), bool)
            or not isinstance(segment.get("confidence"), (int, float))
            or not math.isfinite(float(segment["confidence"]))
            or not 0 <= float(segment["confidence"]) <= 1
        ):
            raise TeachingResourceError(
                "teaching resource temporal provenance segment is invalid"
            )
        previous_start = segment["start_ms"]
    is_temporal = resource.get("resource_type") in {"audio", "video"}
    if is_temporal != bool(temporal):
        raise TeachingResourceError("teaching resource temporal provenance is missing")


def validate_teaching_resources(resources: Any) -> None:
    if not isinstance(resources, Sequence) or isinstance(
        resources, (str, bytes, bytearray)
    ):
        raise TeachingResourceError("teaching_resources must be an array")
    if len(resources) > MAX_TEACHING_RESOURCES:
        raise TeachingResourceError("too many teaching resources")
    identifiers: set[str] = set()
    for resource in resources:
        if not isinstance(resource, Mapping):
            raise TeachingResourceError("teaching resource must be an object")
        if resource.get("schema") != TEACHING_RESOURCE_SCHEMA:
            raise TeachingResourceError("teaching resource schema is invalid")
        resource_id = resource.get("resource_id")
        content_sha256 = resource.get("content_sha256")
        extracted_text = resource.get("extracted_text")
        display_name = resource.get("display_name")
        resource_type = resource.get("resource_type")
        resource_mime = resource.get("mime_type")
        extension = Path(str(display_name or "")).suffix.lower()
        allowed_mimes = {_MIME_BY_EXTENSION.get(extension)}
        if extension == ".webm":
            allowed_mimes = {"audio/webm", "video/webm"}
        elif extension == ".mp4":
            allowed_mimes = {"video/mp4", "audio/mp4"}
        elif extension == ".m4a":
            allowed_mimes = {"audio/mp4", "audio/x-m4a"}
        if (
            not isinstance(resource_id, str)
            or not resource_id.startswith("res_")
            or resource_id in identifiers
            or not isinstance(content_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", content_sha256)
            or not isinstance(display_name, str)
            or not display_name.strip()
            or len(display_name) > 160
            or not isinstance(extracted_text, str)
            or not extracted_text.strip()
            or len(extracted_text) > MAX_RESOURCE_TEXT_CHARS
            or resource_type
            not in {
                "text",
                "document",
                "presentation",
                "pdf",
                "image_ocr",
                "spreadsheet",
                "audio",
                "video",
            }
            or not isinstance(resource_mime, str)
            or resource_mime not in allowed_mimes
            or isinstance(resource.get("byte_size"), bool)
            or not isinstance(resource.get("byte_size"), int)
            or not 1 <= resource["byte_size"] <= MAX_RESOURCE_BYTES
            or isinstance(resource.get("extracted_char_count"), bool)
            or resource.get("extracted_char_count") != len(extracted_text)
            or isinstance(resource.get("original_extracted_char_count"), bool)
            or not isinstance(resource.get("original_extracted_char_count"), int)
            or resource["original_extracted_char_count"] < len(extracted_text)
            or resource.get("truncated")
            is not (resource["original_extracted_char_count"] > len(extracted_text))
            or resource.get("page_count") is not None
            and (
                isinstance(resource.get("page_count"), bool)
                or not isinstance(resource.get("page_count"), int)
                or resource["page_count"] < 1
            )
            or not isinstance(resource.get("extraction_engine"), str)
            or not str(resource["extraction_engine"]).strip()
            or not isinstance(resource.get("needs_review"), bool)
        ):
            raise TeachingResourceError("teaching resource identity or text is invalid")
        if resource_id != "res_" + content_sha256[:20]:
            raise TeachingResourceError(
                "teaching resource ID does not match its content hash"
            )
        if resource.get("raw_media_retained") is not False:
            raise TeachingResourceError("teaching resource must not retain raw media")
        if resource.get("remote_media_sent") is not False:
            raise TeachingResourceError("raw teaching media must not be sent remotely")
        if resource.get("remote_representation") != "bounded_redacted_text_only":
            raise TeachingResourceError(
                "teaching resource remote representation is invalid"
            )
        evidence_contract = resource.get("evidence_contract")
        if evidence_contract is not None:
            _validate_resource_evidence_contract(evidence_contract, resource=resource)
            if (
                resource.get("grading_evidence_allowed") is not False
                or resource.get("mastery_evidence_allowed") is not False
                or resource.get("requires_confirmation")
                is not (evidence_contract.get("decision") == "requires_confirmation")
            ):
                raise TeachingResourceError(
                    "teaching resource assessment boundary is invalid"
                )
        temporal_metadata = resource.get("temporal_metadata")
        temporal_receipt = resource.get("temporal_transcription_receipt")
        if resource_type in {"audio", "video"}:
            duration = (
                temporal_metadata.get("duration_ms")
                if isinstance(temporal_metadata, Mapping)
                else None
            )
            temporal_segments = (
                evidence_contract.get("temporal_provenance", [])
                if isinstance(evidence_contract, Mapping)
                else []
            )
            if (
                not isinstance(temporal_metadata, Mapping)
                or not isinstance(evidence_contract, Mapping)
                or temporal_metadata.get("schema")
                != "teaching_skill_miner.temporal_media_metadata.v1"
                or temporal_metadata.get("content_sha256") != content_sha256
                or temporal_metadata.get("byte_size") != resource.get("byte_size")
                or temporal_metadata.get("mime_type") != resource.get("mime_type")
                or temporal_metadata.get("source_modality")
                != resource.get("resource_type")
                or temporal_metadata.get("metadata_status")
                not in {"parsed", "container_signature_only"}
                or not isinstance(temporal_metadata.get("metadata_engine"), str)
                or not str(temporal_metadata["metadata_engine"]).strip()
                or (
                    duration is not None
                    and (
                        isinstance(duration, bool)
                        or not isinstance(duration, int)
                        or not 0 <= duration <= 86_400_000
                    )
                )
                or not isinstance(temporal_metadata.get("audio_streams"), list)
                or not isinstance(temporal_metadata.get("video_streams"), list)
                or temporal_metadata.get("raw_media_retained") is not False
                or temporal_metadata.get("remote_media_sent") is not False
                or temporal_metadata.get("transcription_performed") is not True
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(temporal_metadata.get("transcription_sha256", "")),
                )
                or not isinstance(
                    temporal_metadata.get("transcription_provider_id"), str
                )
                or not isinstance(temporal_receipt, Mapping)
                or temporal_receipt.get("media_sha256") != content_sha256
                or temporal_receipt.get("remote_media_sent") is not False
                or not isinstance(temporal_receipt.get("provider_id"), str)
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(temporal_receipt.get("provider_spec_sha256", "")),
                )
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(temporal_receipt.get("provider_result_sha256", "")),
                )
            ):
                raise TeachingResourceError(
                    "teaching resource temporal receipt is invalid"
                )
            if duration is not None and any(
                int(segment.get("end_ms", duration + 1)) > duration + 1_000
                for segment in temporal_segments
                if isinstance(segment, Mapping)
            ):
                raise TeachingResourceError(
                    "teaching resource transcript exceeds media duration"
                )
        elif temporal_metadata is not None or temporal_receipt is not None:
            raise TeachingResourceError(
                "non-temporal teaching resource has temporal metadata"
            )
        retrieval_index = resource.get("retrieval_index")
        if retrieval_index is not None:
            try:
                validate_resource_chunk_index(
                    retrieval_index,
                    resource_id=resource_id,
                    resource_content_sha256=content_sha256,
                    extracted_text=extracted_text,
                )
            except ResourceRetrievalError as exc:
                raise TeachingResourceError(
                    "teaching resource retrieval index is invalid"
                ) from exc
        identifiers.add(resource_id)
