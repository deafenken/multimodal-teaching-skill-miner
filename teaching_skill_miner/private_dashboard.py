"""Loopback-only dashboard for replaying private TeachObs evidence.

The public dashboard intentionally contains no row-level material.  This module
keeps that contract intact: it reads explicitly local private artifacts,
recomputes the frozen four-arm predictions in memory, and exposes only a small
capability-token protected HTTP surface on the loopback interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
import csv
import json
import math
import mimetypes
from pathlib import Path
import re
import secrets
import threading
from typing import Any, BinaryIO, Mapping
from urllib.parse import unquote, urlsplit
import webbrowser


PRIVATE_DASHBOARD_PACKAGE = "teaching_skill_miner.web"
PRIVATE_DASHBOARD_RESOURCE = "private_demo.html"
PRIVATE_DASHBOARD_SKILL_STYLE_RESOURCE = "private_skill_demo.css"
PRIVATE_DASHBOARD_SKILL_SCRIPT_RESOURCE = "private_skill_demo.js"
DEFAULT_PRIVATE_TEACHOBS_ROOT = Path(
    "artifacts/private/external_datasets/teachobs"
)
DEFAULT_PRIVATE_SKILL_ROOT = Path("artifacts/private/full_multimodal")
DEFAULT_PRIVATE_LESSON = "S24"
DEFAULT_PRIVATE_SKILL = "linear_algebra_l03"
PRIVATE_DASHBOARD_ARMS = (
    "transcript_only",
    "transcript_audio",
    "transcript_visual",
    "full",
)

PRIVATE_SKILL_EVALUATION_DIMENSIONS = {
    "structural_completeness": 0.12,
    "evidence_grounding": 0.18,
    "executability": 0.18,
    "method_fidelity": 0.22,
    "pedagogical_quality": 0.12,
    "generalizability": 0.09,
    "traceability": 0.09,
}
PRIVATE_SKILL_EVALUATION_GATES = (
    "schema_valid",
    "grounded",
    "executable",
    "testable",
    "multimodal_consistent",
    "method_distilled_from_video",
)

BEHAVIOR_PROJECTION_SCHEMA = "teachobs_frozen_behavior_projection.v1"

# TeachObs Track A is a teaching-practice ontology whose labelled subject is the
# teacher.  These two allowlists are deliberately explicit: the first contains
# codes that name directly visible teacher actions; the second contains teacher
# interaction codes that can provide context about learner participation.  The
# latter must never be presented as direct student-action recognition.
TEACHER_VISIBLE_ACTION_CODES: tuple[tuple[str, str], ...] = (
    ("Demonstration", "示范"),
    ("Board work", "板书"),
    ("Pointing", "指向"),
    ("Underlining", "下划线强调"),
    ("Enclosing", "圈画"),
    ("Marking", "标记"),
    ("Linking", "连接内容"),
    ("Moving", "移动"),
    ("Stationary", "原地讲授"),
    ("Gesture", "手势"),
)

LEARNER_RELATED_TEACHING_SIGNAL_CODES: tuple[tuple[str, str], ...] = (
    ("Responding", "回应"),
    ("Reinforcing", "强化反馈"),
    ("Checking", "检查理解"),
    ("Cueing", "提示引导"),
    ("Monitoring", "巡视观察"),
    ("Incorporating", "吸收学生观点"),
    ("Assessment", "评估"),
    ("Reflection", "引导反思"),
    ("Prior Knowledge Elicitation", "唤起先备知识"),
)

_LESSON_ID = re.compile(r"S(?:[1-9]|[12][0-9]|30)")
_SCENE_ROUTE = re.compile(r"api/lessons/(S(?:[1-9]|[12][0-9]|30))/scenes/(\d+)")
_SKILL_ROUTE = re.compile(r"api/skills/([a-z0-9][a-z0-9_-]{0,79})")
_MEDIA_ROUTE = re.compile(r"media/(S(?:[1-9]|[12][0-9]|30))")
_FRAME_ROUTE = re.compile(r"frames/(S(?:[1-9]|[12][0-9]|30))/(\d+)")
_CAPTION_ROUTE = re.compile(r"captions/(S(?:[1-9]|[12][0-9]|30))\.vtt")
_SINGLE_RANGE = re.compile(r"bytes=(\d*)-(\d*)")
_SAFE_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_CSP = (
    "default-src 'none'; style-src 'self' 'unsafe-inline'; "
    "script-src 'self' 'unsafe-inline'; "
    "img-src 'self'; media-src 'self'; connect-src 'self'; base-uri 'none'; "
    "object-src 'none'; frame-ancestors 'none'; form-action 'none'; "
    "worker-src 'none'; manifest-src 'none'"
)


class PrivateDashboardError(RuntimeError):
    """Raised when local private evidence cannot be safely replayed."""


class RangeNotSatisfiable(ValueError):
    """Raised for malformed, multipart, or unsatisfiable byte ranges."""


def project_teachobs_behavior_view(
    labels: list[dict[str, Any]],
) -> dict[str, Any]:
    """Project existing frozen label rows into an honest presentation view.

    No label, score or actor is inferred from OCR text.  Unknown codes remain
    outside the projection, and learner-related rows are explicitly described
    as indirect interaction cues rather than observed student actions.
    """

    by_name: dict[str, dict[str, Any]] = {}
    for row in labels:
        label = str(row.get("label", ""))
        if label and label not in by_name:
            by_name[label] = row

    def select(
        codes: tuple[tuple[str, str], ...],
        *,
        actor_scope: str,
        direct_student_action: bool,
    ) -> list[dict[str, Any]]:
        projected: list[dict[str, Any]] = []
        for code, display_label in codes:
            source = by_name.get(code)
            if source is None:
                continue
            projected.append(
                {
                    "label": code,
                    "display_label": display_label,
                    "group": str(source.get("group", "")),
                    "predicted": bool(source.get("predicted", False)),
                    "score": float(source.get("score", 0.0)),
                    "reference": bool(source.get("reference", False)),
                    "actor_scope": actor_scope,
                    "direct_student_action": direct_student_action,
                    "mapping_source": "explicit_teachobs_code_allowlist",
                }
            )
        return projected

    return {
        "schema": BEHAVIOR_PROJECTION_SCHEMA,
        "kind": "frozen_teachobs_label_projection",
        "teacher_visible_actions": select(
            TEACHER_VISIBLE_ACTION_CODES,
            actor_scope="teacher",
            direct_student_action=False,
        ),
        "student_actions": [],
        "student_action_status": "not_modeled_no_scene_level_ground_truth",
        "learner_related_teaching_signals": select(
            LEARNER_RELATED_TEACHING_SIGNAL_CODES,
            actor_scope="teacher_student_interaction",
            direct_student_action=False,
        ),
        "pose_tracking_performed": False,
        "identity_recognition_performed": False,
        "direct_student_action_recognition": False,
        "action_subgroup_accuracy_established": False,
    }


@dataclass(frozen=True, slots=True)
class PrivateDashboardConfig:
    """Locations and presentation defaults for one private dashboard run."""

    root: Path
    initial_lesson: str = DEFAULT_PRIVATE_LESSON
    skill_root: Path | None = None
    initial_skill: str = DEFAULT_PRIVATE_SKILL


@dataclass(slots=True)
class PrivateLessonSnapshot:
    """Browser-safe rows plus allowlisted local files for one lesson."""

    lesson_id: str
    catalog_row: dict[str, Any]
    media_path: Path
    expected_media_sha256: str
    scene_manifest_sha256: str
    scenes: tuple[dict[str, Any], ...]
    frame_paths: tuple[Path, ...]
    expected_frame_sha256: tuple[str, ...]
    captions_vtt: bytes
    _verified_media: bool = False
    _verified_frames: set[int] = field(default_factory=set)
    _verification_lock: threading.Lock = field(default_factory=threading.Lock)

    def verify_media(self) -> None:
        """Hash the media once before the first byte is exposed."""

        with self._verification_lock:
            if self._verified_media:
                return
            if _file_sha256(self.media_path) != self.expected_media_sha256:
                raise PrivateDashboardError(
                    f"private media integrity mismatch for {self.lesson_id}"
                )
            self._verified_media = True

    def verify_frame(self, scene_number: int) -> None:
        """Hash one allowlisted frame before serving it."""

        index = scene_number - 1
        with self._verification_lock:
            if index in self._verified_frames:
                return
            if _file_sha256(self.frame_paths[index]) != self.expected_frame_sha256[index]:
                raise PrivateDashboardError(
                    f"private frame integrity mismatch for {self.lesson_id}"
                )
            self._verified_frames.add(index)


@dataclass(frozen=True, slots=True)
class PrivateSkillSnapshot:
    """One browser-safe distilled Skill with no local path-bearing fields."""

    video_id: str
    catalog_row: dict[str, Any]
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PrivateDashboardSnapshot:
    """Immutable, path-safe snapshot used by the loopback request handler."""

    lessons: Mapping[str, PrivateLessonSnapshot]
    default_lesson: str
    bundle_fingerprint: str
    transcript_fingerprint: str
    prediction_input_fingerprint: str
    prediction_matrix_sha256: Mapping[str, str]
    template: bytes
    skills: Mapping[str, PrivateSkillSnapshot] = field(default_factory=dict)
    default_skill: str | None = None
    skill_aggregate: Mapping[str, Any] = field(default_factory=dict)

    def catalog(self) -> dict[str, Any]:
        skill_available = bool(self.skills)
        return {
            "schema_version": "1.0",
            "mode": "local_private_real_data",
            "dataset": "TeachObs",
            "profile": "paper_track1_23_train_6_test",
            "default_lesson": self.default_lesson,
            "lessons": [self.lessons[key].catalog_row for key in self.lessons],
            "products": [
                {
                    "id": "behavior_recognition",
                    "title": "TeachObs 行为识别",
                    "input": "真实课堂 15 秒场景",
                    "output": "39 类教师教学行为预测",
                    "available": True,
                },
                {
                    "id": "skill_distillation",
                    "title": "MIT Skill 蒸馏",
                    "input": "完整讲次与对齐多模态证据",
                    "output": "有证据的九环节可执行 Skill",
                    "available": skill_available,
                },
            ],
            "skill_distillation": {
                "available": skill_available,
                "kind": "complete_video_skill_distillation",
                "dataset": "MIT_OCW_10_lecture_longform_track",
                "default_skill": self.default_skill,
                "skills": [
                    self.skills[key].catalog_row for key in self.skills
                ],
                "aggregate": dict(self.skill_aggregate),
                "relationship_to_recognition": (
                    "independent_validation_track_not_direct_prediction_input"
                ),
                "claim_boundary": {
                    "procedure_is_text_led": True,
                    "multimodal_events_assist_strategy_scoring": True,
                    "independent_event_ground_truth_used": False,
                    "human_skill_quality_labels_used": False,
                    "teaching_effectiveness_established": False,
                },
            },
            "inference": {
                "kind": "frozen_four_arm_feature_replay",
                "arms": list(PRIVATE_DASHBOARD_ARMS),
                "default_arm": "full",
                "behavior_view": "frozen_teachobs_label_projection",
                "teacher_visible_action_codes": len(TEACHER_VISIBLE_ACTION_CODES),
                "direct_student_action_recognition": False,
                "student_action_status": "not_modeled_no_scene_level_ground_truth",
                "live_feature_extraction": False,
                "calibrated_probability": False,
                "post_test_exploratory": True,
                "deployment_accuracy_established": False,
            },
            "privacy": {
                "loopback_only": True,
                "persistent_browser_storage": False,
                "private_files_copied": False,
                "identity_recognition_performed": False,
            },
            "provenance": {
                "bundle_fingerprint": _short_hash(self.bundle_fingerprint),
                "transcript_fingerprint": _short_hash(self.transcript_fingerprint),
                "prediction_input_fingerprint": _short_hash(
                    self.prediction_input_fingerprint
                ),
            },
        }

    def scene(self, lesson_id: str, scene_number: int) -> dict[str, Any]:
        lesson = self.lessons.get(lesson_id)
        if lesson is None or scene_number < 1 or scene_number > len(lesson.scenes):
            raise KeyError((lesson_id, scene_number))
        return lesson.scenes[scene_number - 1]

    def skill(self, video_id: str) -> dict[str, Any]:
        skill = self.skills.get(video_id)
        if skill is None:
            raise KeyError(video_id)
        return skill.payload

    def verify_all_private_files(self) -> dict[str, int | bool]:
        """Hash every selected video and frame for an explicit preflight check."""

        media_bytes = 0
        frame_bytes = 0
        frame_count = 0
        for lesson in self.lessons.values():
            lesson.verify_media()
            media_bytes += lesson.media_path.stat().st_size
            for scene_number, frame_path in enumerate(lesson.frame_paths, start=1):
                lesson.verify_frame(scene_number)
                frame_bytes += frame_path.stat().st_size
                frame_count += 1
        return {
            "all_selected_private_files_sha256_verified": True,
            "verified_video_count": len(self.lessons),
            "verified_video_bytes": media_bytes,
            "verified_frame_count": frame_count,
            "verified_frame_bytes": frame_bytes,
            "validated_skill_artifact_count": len(self.skills),
        }


def private_dashboard_html_bytes() -> bytes:
    """Return the generic private replay UI; it contains no private data."""

    return (
        resources.files(PRIVATE_DASHBOARD_PACKAGE)
        .joinpath(PRIVATE_DASHBOARD_RESOURCE)
        .read_bytes()
    )


def private_dashboard_asset_bytes(resource_name: str) -> bytes:
    """Return one allowlisted generic dashboard asset."""

    if resource_name not in {
        PRIVATE_DASHBOARD_SKILL_STYLE_RESOURCE,
        PRIVATE_DASHBOARD_SKILL_SCRIPT_RESOURCE,
    }:
        raise PrivateDashboardError("unknown private dashboard asset")
    return (
        resources.files(PRIVATE_DASHBOARD_PACKAGE)
        .joinpath(resource_name)
        .read_bytes()
    )


def private_dashboard_template_self_check() -> dict[str, Any]:
    """Validate the generic template without reading any private artifact."""

    payload = private_dashboard_html_bytes()
    style = private_dashboard_asset_bytes(PRIVATE_DASHBOARD_SKILL_STYLE_RESOURCE)
    script = private_dashboard_asset_bytes(PRIVATE_DASHBOARD_SKILL_SCRIPT_RESOURCE)
    text = "\n".join(
        value.decode("utf-8") for value in (payload, style, script)
    )
    required = (
        "local_private_real_data",
        "data-screen-label=\"01 Behavior recognition\"",
        "data-screen-label=\"02 Skill distillation\"",
        "data-screen-label=\"03 Teaching process generation\"",
        "data-screen-label=\"04 Automatic evaluation\"",
        "<video",
        "fetch(",
        ".textContent",
        "private_skill_demo.css",
        "private_skill_demo.js",
        "真实 TeachObs 完整视频",
        "MIT Skill 蒸馏",
        "九环节可执行 Skill",
        "视频中观察到",
        "建议补全",
        "字幕证据主导",
        "直接 mme_* 引用数",
        "用当前 Skill 生成一段可执行教学过程",
        "七维量表 + 六项门槛",
        "结构质量与内部证据一致性量表，不是 Accuracy",
        "教师侧可观察教学动作",
        "学生动作",
        "当前未建模",
        "冻结标签投影",
        "不是实时部署推理",
        "post-test exploratory",
    )
    forbidden = (
        "artifacts/private",
        "/Volumes/",
        "/Users/",
        "/data/winbeau_zhao",
        "localStorage",
        "sessionStorage",
        ".innerHTML",
        "http://",
        "https://",
    )
    missing = [marker for marker in required if marker not in text]
    forbidden_matches = [marker for marker in forbidden if marker in text]
    same_origin_assets_allowed = all(
        directive in text and directive in _CSP
        for directive in (
            "style-src 'self' 'unsafe-inline'",
            "script-src 'self' 'unsafe-inline'",
        )
    )
    return {
        "schema_version": "1.0",
        "dashboard_kind": "local_private_real_evidence_template",
        "resource": f"{PRIVATE_DASHBOARD_PACKAGE}:{PRIVATE_DASHBOARD_RESOURCE}",
        "size_bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
        "asset_sha256": {
            PRIVATE_DASHBOARD_SKILL_STYLE_RESOURCE: sha256(style).hexdigest(),
            PRIVATE_DASHBOARD_SKILL_SCRIPT_RESOURCE: sha256(script).hexdigest(),
        },
        "required_markers_present": not missing,
        "missing_markers": missing,
        "forbidden_markers_absent": not forbidden_matches,
        "forbidden_matches": forbidden_matches,
        "same_origin_assets_allowed": same_origin_assets_allowed,
        "contains_private_data": False,
        "passed": not missing and not forbidden_matches and same_origin_assets_allowed,
    }


def _short_hash(value: str) -> str:
    return f"{value[:12]}…{value[-8:]}"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_safe_path(
    root: Path,
    candidate: Path,
    *,
    kind: str,
) -> Path:
    """Resolve one required path while rejecting escape and symlink components."""

    lexical = candidate if candidate.is_absolute() else root / candidate
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise PrivateDashboardError(f"private {kind} path escapes its root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise PrivateDashboardError(f"private {kind} path contains a symlink")
    try:
        resolved = lexical.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PrivateDashboardError(f"required private {kind} is missing") from exc
    if not resolved.is_relative_to(root):
        raise PrivateDashboardError(f"private {kind} resolves outside its root")
    return resolved


def _safe_file(root: Path, candidate: Path, *, kind: str) -> Path:
    resolved = _ensure_safe_path(root, candidate, kind=kind)
    if not resolved.is_file():
        raise PrivateDashboardError(f"private {kind} is not a regular file")
    return resolved


def _safe_directory(root: Path, candidate: Path, *, kind: str) -> Path:
    resolved = _ensure_safe_path(root, candidate, kind=kind)
    if not resolved.is_dir():
        raise PrivateDashboardError(f"private {kind} is not a directory")
    return resolved


def _read_json(path: Path, *, purpose: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrivateDashboardError(f"invalid private {purpose}") from exc
    if not isinstance(value, dict):
        raise PrivateDashboardError(f"private {purpose} must be an object")
    return value


def _read_jsonl(path: Path, *, purpose: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError
            rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise PrivateDashboardError(f"invalid private {purpose}") from exc
    return rows


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _safe_number(value: Any, *, purpose: str, minimum: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PrivateDashboardError(f"invalid private {purpose}") from exc
    if not math.isfinite(result) or result < minimum:
        raise PrivateDashboardError(f"invalid private {purpose}")
    return result


def _safe_text(value: Any, *, purpose: str, limit: int = 1200) -> str:
    if not isinstance(value, str):
        raise PrivateDashboardError(f"invalid private {purpose}")
    result = " ".join(value.split())
    if not result:
        raise PrivateDashboardError(f"invalid private {purpose}")
    return result if len(result) <= limit else f"{result[: limit - 1]}…"


def _safe_string_list(
    value: Any,
    *,
    purpose: str,
    limit: int = 80,
) -> list[str]:
    if not isinstance(value, list):
        raise PrivateDashboardError(f"invalid private {purpose}")
    return [
        _safe_text(item, purpose=purpose, limit=limit)
        for item in value
    ]


def _sanitize_multimodal_event(value: dict[str, Any]) -> dict[str, Any]:
    event_id = _safe_text(value.get("event_id"), purpose="event id", limit=40)
    event_type = _safe_text(value.get("type"), purpose="event type", limit=80)
    if not re.fullmatch(r"mme_[0-9]{4,}", event_id) or not re.fullmatch(
        r"[a-z0-9_]+", event_type
    ):
        raise PrivateDashboardError("invalid private multimodal event identity")
    start = _safe_number(value.get("start"), purpose="event start")
    end = _safe_number(value.get("end"), purpose="event end")
    if end <= start:
        raise PrivateDashboardError("invalid private multimodal event timing")
    modalities = _safe_string_list(
        value.get("modalities"), purpose="event modalities", limit=24
    )
    if not modalities or set(modalities) - {"transcript", "audio", "visual", "ocr"}:
        raise PrivateDashboardError("invalid private multimodal event modalities")
    supports = _safe_string_list(
        value.get("supports"), purpose="event strategy support", limit=80
    )
    evidence = value.get("evidence")
    if not isinstance(evidence, dict):
        raise PrivateDashboardError("invalid private multimodal event evidence")
    speech_excerpt = evidence.get("speech_quote")
    if speech_excerpt is not None:
        speech_excerpt = _safe_text(
            speech_excerpt,
            purpose="event speech excerpt",
            limit=240,
        )
    wait_seconds = evidence.get("wait_seconds")
    if wait_seconds is not None:
        wait_seconds = round(
            _safe_number(wait_seconds, purpose="event wait seconds"),
            3,
        )
    visual_labels: list[dict[str, Any]] = []
    seen_visual_labels: set[str] = set()
    raw_semantics = evidence.get("visual_semantic_labels", [])
    if not isinstance(raw_semantics, list):
        raise PrivateDashboardError("invalid private visual semantic evidence")
    for row in raw_semantics:
        if not isinstance(row, dict):
            raise PrivateDashboardError("invalid private visual semantic row")
        label = _safe_text(
            row.get("top_label"),
            purpose="visual semantic label",
            limit=80,
        )
        if label in seen_visual_labels:
            continue
        seen_visual_labels.add(label)
        visual_labels.append(
            {
                "label": label,
                "relative_score": round(
                    _safe_number(
                        row.get("top_relative_score"),
                        purpose="visual semantic relative score",
                    ),
                    6,
                ),
            }
        )
        if len(visual_labels) == 3:
            break
    confidence = round(
        _safe_number(value.get("confidence"), purpose="event confidence"),
        6,
    )
    if confidence > 1:
        raise PrivateDashboardError("invalid private event confidence")
    return {
        "event_id": event_id,
        "type": event_type,
        "start": round(start, 3),
        "end": round(end, 3),
        "modalities": modalities,
        "supports": supports,
        "speech_excerpt": speech_excerpt,
        "wait_seconds": wait_seconds,
        "visual_semantic_labels": visual_labels,
        "confidence": confidence,
        "confidence_is_calibrated_probability": False,
    }


def _build_private_skill_snapshots(
    root_input: Path,
    *,
    initial_skill: str,
) -> tuple[dict[str, PrivateSkillSnapshot], str, dict[str, Any]]:
    """Validate ten Full Skills and project only presentation-safe fields."""

    if root_input.is_symlink():
        raise PrivateDashboardError("private Skill root may not be a symlink")
    try:
        root = root_input.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise PrivateDashboardError("private Skill root is missing") from exc
    if not root.is_dir():
        raise PrivateDashboardError("private Skill root is not a directory")

    manifest_path = _safe_file(
        root,
        Path("dataset_manifest.semantic.json"),
        kind="semantic dataset manifest",
    )
    report_path = _safe_file(
        root,
        Path("ablation/ablation_report.json"),
        kind="multimodal ablation report",
    )
    manifest = _read_json(manifest_path, purpose="semantic dataset manifest")
    report = _read_json(report_path, purpose="multimodal ablation report")
    manifest_rows = manifest.get("videos")
    report_rows = report.get("per_lecture")
    if (
        manifest.get("video_count") != 10
        or report.get("paired_lecture_count") != 10
        or not isinstance(manifest_rows, list)
        or not isinstance(report_rows, list)
        or len(manifest_rows) != 10
        or len(report_rows) != 10
    ):
        raise PrivateDashboardError("private Skill track is not the frozen ten-lecture set")
    if manifest.get("aggregate", {}).get("semantic_feature_complete_count") != 10:
        raise PrivateDashboardError("private Skill visual semantics are incomplete")

    manifest_by_id: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        if not isinstance(row, dict):
            raise PrivateDashboardError("private Skill manifest row is malformed")
        video_id = row.get("video_id")
        if (
            not isinstance(video_id, str)
            or not re.fullmatch(r"(?:linear_algebra|python)_l0[1-5]", video_id)
            or video_id in manifest_by_id
        ):
            raise PrivateDashboardError("private Skill manifest identity is malformed")
        manifest_by_id[video_id] = row

    from .models import validate_skill

    skills: dict[str, PrivateSkillSnapshot] = {}
    selected_event_type_frequency: dict[str, int] = {}
    candidate_event_type_frequency: dict[str, int] = {}
    totals = {
        "procedure_step_count": 0,
        "observed_method_step_count": 0,
        "recommended_enrichment_step_count": 0,
        "text_evidence_count": 0,
        "selected_multimodal_event_count": 0,
        "verification_step_count": 0,
        "procedure_direct_mme_reference_count": 0,
    }
    total_duration = 0.0

    for report_row in report_rows:
        if not isinstance(report_row, dict):
            raise PrivateDashboardError("private Skill ablation row is malformed")
        video_id = report_row.get("video_id")
        manifest_row = manifest_by_id.get(str(video_id))
        if manifest_row is None or video_id in skills:
            raise PrivateDashboardError("private Skill lecture identities disagree")
        if any(
            report_row.get(field) != manifest_row.get(field)
            for field in ("video_id", "course_id", "title")
        ):
            raise PrivateDashboardError("private Skill lecture metadata disagrees")
        arms = report_row.get("arms")
        full_arm = arms.get("full") if isinstance(arms, dict) else None
        if not isinstance(full_arm, dict):
            raise PrivateDashboardError("private Full Skill arm is missing")
        expected_skill_relative = f"skills/{video_id}.full.skill.json"
        expected_evaluation_relative = f"evaluations/{video_id}.full.evaluation.json"
        if (
            full_arm.get("skill_artifact") != expected_skill_relative
            or full_arm.get("evaluation_artifact") != expected_evaluation_relative
        ):
            raise PrivateDashboardError("private Full Skill artifact binding disagrees")
        skill_path = _safe_file(
            root,
            Path("ablation") / expected_skill_relative,
            kind="Full Skill artifact",
        )
        evaluation_path = _safe_file(
            root,
            Path("ablation") / expected_evaluation_relative,
            kind="Full Skill evaluation",
        )
        skill = _read_json(skill_path, purpose="Full Skill artifact")
        evaluation = _read_json(evaluation_path, purpose="Full Skill evaluation")
        validation = validate_skill(skill)
        if not validation.valid:
            raise PrivateDashboardError("private Full Skill schema validation failed")
        skill_metrics = full_arm.get("skill_metrics")
        if not isinstance(skill_metrics, dict) or skill_metrics.get(
            "skill_fingerprint_sha256"
        ) != _canonical_json_sha256(skill):
            raise PrivateDashboardError("private Full Skill fingerprint disagrees")
        source = skill.get("source")
        if not isinstance(source, dict) or any(
            source.get(field) != report_row.get(field)
            for field in ("video_id", "course_id", "title")
        ):
            raise PrivateDashboardError("private Full Skill source metadata disagrees")
        if set(source.get("modalities_available", [])) != {
            "transcript",
            "audio",
            "visual",
            "ocr",
        }:
            raise PrivateDashboardError("private Full Skill modalities disagree")
        if evaluation.get("skill_id") != skill.get("skill_id") or not evaluation.get(
            "passed"
        ):
            raise PrivateDashboardError("private Full Skill evaluation disagrees")

        manifest_summary = manifest_row.get("summary")
        analysis_metrics = full_arm.get("analysis_metrics")
        if not isinstance(manifest_summary, dict) or not isinstance(
            analysis_metrics, dict
        ):
            raise PrivateDashboardError("private Full Skill event summary is malformed")
        candidate_event_count = int(manifest_summary.get("fused_event_count", -1))
        if candidate_event_count != int(analysis_metrics.get("event_count", -2)):
            raise PrivateDashboardError("private Full Skill event counts disagree")
        for event_type, count in analysis_metrics.get(
            "event_type_frequency", {}
        ).items():
            if not isinstance(event_type, str) or not isinstance(count, int):
                raise PrivateDashboardError("private Full Skill event frequency is invalid")
            candidate_event_type_frequency[event_type] = (
                candidate_event_type_frequency.get(event_type, 0) + count
            )

        text_evidence_raw = source.get("evidence")
        multimodal_raw = source.get("multimodal_evidence")
        procedure_raw = skill.get("procedure")
        strategies_raw = skill.get("strategies")
        verification_raw = skill.get("verification")
        failure_modes_raw = skill.get("failure_modes")
        if not all(
            isinstance(value, list)
            for value in (
                text_evidence_raw,
                multimodal_raw,
                procedure_raw,
                strategies_raw,
                verification_raw,
                failure_modes_raw,
            )
        ):
            raise PrivateDashboardError("private Full Skill content is malformed")

        evidence_rows: list[dict[str, Any]] = []
        evidence_ids: set[str] = set()
        for item in text_evidence_raw:
            if not isinstance(item, dict):
                raise PrivateDashboardError("private Skill text evidence is malformed")
            evidence_id = _safe_text(
                item.get("evidence_id"), purpose="text evidence id", limit=40
            )
            if not re.fullmatch(r"evi_[0-9a-f]{16}", evidence_id) or evidence_id in evidence_ids:
                raise PrivateDashboardError("private Skill text evidence id is invalid")
            evidence_ids.add(evidence_id)
            start = _safe_number(item.get("start"), purpose="text evidence start")
            end = _safe_number(item.get("end"), purpose="text evidence end")
            if end <= start:
                raise PrivateDashboardError("private Skill text evidence timing is invalid")
            evidence_rows.append(
                {
                    "evidence_id": evidence_id,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "quote": _safe_text(
                        item.get("quote"), purpose="text evidence excerpt", limit=260
                    ),
                    "supports": _safe_string_list(
                        item.get("supports"),
                        purpose="text evidence support",
                        limit=80,
                    ),
                }
            )

        multimodal_events = [
            _sanitize_multimodal_event(item)
            for item in multimodal_raw
            if isinstance(item, dict)
        ]
        if len(multimodal_events) != len(multimodal_raw):
            raise PrivateDashboardError("private Skill multimodal evidence is malformed")
        multimodal_events.sort(key=lambda row: (row["start"], row["event_id"]))
        for event in multimodal_events:
            event_type = str(event["type"])
            selected_event_type_frequency[event_type] = (
                selected_event_type_frequency.get(event_type, 0) + 1
            )

        procedure_rows: list[dict[str, Any]] = []
        observed_count = 0
        recommended_count = 0
        direct_mme_count = 0
        for item in procedure_raw:
            if not isinstance(item, dict):
                raise PrivateDashboardError("private Skill procedure is malformed")
            origin = item.get("origin")
            if origin not in {"observed_method", "recommended_enrichment"}:
                raise PrivateDashboardError("private Skill procedure origin is invalid")
            observed_count += origin == "observed_method"
            recommended_count += origin == "recommended_enrichment"
            references = _safe_string_list(
                item.get("evidence_ids"),
                purpose="procedure evidence references",
                limit=40,
            )
            direct_mme_count += sum(value.startswith("mme_") for value in references)
            span = item.get("observed_span")
            safe_span = None
            if span is not None:
                if not isinstance(span, dict):
                    raise PrivateDashboardError("private Skill observed span is invalid")
                span_start = _safe_number(
                    span.get("start"), purpose="procedure observed span start"
                )
                span_end = _safe_number(
                    span.get("end"), purpose="procedure observed span end"
                )
                if span_end <= span_start:
                    raise PrivateDashboardError("private Skill observed span is invalid")
                safe_span = {
                    "start": round(span_start, 3),
                    "end": round(span_end, 3),
                }
            provenance = item.get("provenance")
            if not isinstance(provenance, dict):
                raise PrivateDashboardError("private Skill step provenance is invalid")
            procedure_rows.append(
                {
                    "step": int(item.get("step")),
                    "teaching_phase": _safe_text(
                        item.get("teaching_phase"), purpose="teaching phase", limit=100
                    ),
                    "teaching_phase_name": _safe_text(
                        item.get("teaching_phase_name"),
                        purpose="teaching phase name",
                        limit=100,
                    ),
                    "canonical_phase_rank": int(item.get("canonical_phase_rank")),
                    "teacher_action": _safe_text(
                        item.get("teacher_action"), purpose="teacher action", limit=80
                    ),
                    "instruction": _safe_text(
                        item.get("instruction"), purpose="Skill instruction"
                    ),
                    "expected_signal": _safe_text(
                        item.get("expected_signal"), purpose="expected learner signal"
                    ),
                    "fallback": _safe_text(
                        item.get("fallback"), purpose="Skill fallback"
                    ),
                    "origin": origin,
                    "evidence_ids": references,
                    "observed_span": safe_span,
                    "matched_cues": _safe_string_list(
                        item.get("matched_cues"),
                        purpose="teaching phase cues",
                        limit=80,
                    ),
                    "strategy_id": provenance.get("strategy_id"),
                }
            )
        if len(procedure_rows) != 9 or observed_count not in {5, 6, 7}:
            raise PrivateDashboardError("private Skill nine-phase procedure disagrees")
        if direct_mme_count:
            raise PrivateDashboardError(
                "private Skill procedure now directly references multimodal events; "
                "the dashboard claim boundary must be revised"
            )

        strategy_rows: list[dict[str, Any]] = []
        for item in strategies_raw:
            if not isinstance(item, dict):
                raise PrivateDashboardError("private Skill strategy is malformed")
            strategy_rows.append(
                {
                    "id": _safe_text(item.get("id"), purpose="strategy id", limit=80),
                    "name": _safe_text(
                        item.get("name"), purpose="strategy name", limit=120
                    ),
                    "evidence_count": int(item.get("evidence_count", 0)),
                    "text_evidence_count": int(item.get("text_evidence_count", 0)),
                    "multimodal_evidence_count": int(
                        item.get("multimodal_evidence_count", 0)
                    ),
                    "confidence": round(
                        _safe_number(
                            item.get("confidence"), purpose="strategy confidence"
                        ),
                        6,
                    ),
                    "origin": item.get("origin"),
                }
            )

        coverage = source.get("provenance", {}).get("transcript_coverage", {})
        if not isinstance(coverage, dict) or not coverage.get("completeness_verified"):
            raise PrivateDashboardError("private Skill caption coverage is not verified")
        duration = _safe_number(
            manifest_summary.get("duration_seconds"), purpose="Skill video duration"
        )
        source_duration = _safe_number(
            coverage.get("source_duration_seconds"), purpose="Skill source duration"
        )
        if abs(duration - source_duration) > 0.01:
            raise PrivateDashboardError("private Skill duration binding disagrees")
        parameters = skill.get("parameters")
        if not isinstance(parameters, dict):
            raise PrivateDashboardError("private Skill execution parameters are malformed")
        execution_parameters: dict[str, dict[str, Any]] = {}
        for parameter_name in ("concept", "learner_level"):
            parameter = parameters.get(parameter_name)
            if (
                not isinstance(parameter, dict)
                or parameter.get("type") != "string"
                or not isinstance(parameter.get("required"), bool)
            ):
                raise PrivateDashboardError(
                    "private Skill execution parameters are malformed"
                )
            execution_parameters[parameter_name] = {
                "type": "string",
                "required": parameter["required"],
                "default": _safe_text(
                    parameter.get("default"),
                    purpose=f"Skill {parameter_name} default",
                    limit=160,
                ),
            }

        evaluation_dimensions = evaluation.get("dimensions")
        evaluation_weights = evaluation.get("weights")
        evaluation_gates = evaluation.get("gates")
        if not all(
            isinstance(value, dict)
            for value in (
                evaluation_dimensions,
                evaluation_weights,
                evaluation_gates,
            )
        ):
            raise PrivateDashboardError("private Skill evaluation fields are malformed")
        if set(evaluation_dimensions) != set(PRIVATE_SKILL_EVALUATION_DIMENSIONS) or set(
            evaluation_weights
        ) != set(PRIVATE_SKILL_EVALUATION_DIMENSIONS):
            raise PrivateDashboardError("private Skill evaluation dimensions disagree")
        safe_dimensions: dict[str, float] = {}
        safe_weights: dict[str, float] = {}
        for dimension, expected_weight in PRIVATE_SKILL_EVALUATION_DIMENSIONS.items():
            score = _safe_number(
                evaluation_dimensions.get(dimension),
                purpose=f"Skill {dimension} score",
            )
            weight = _safe_number(
                evaluation_weights.get(dimension),
                purpose=f"Skill {dimension} weight",
            )
            if score > 100 or abs(weight - expected_weight) > 1e-9:
                raise PrivateDashboardError("private Skill evaluation dimensions disagree")
            safe_dimensions[dimension] = round(score, 1)
            safe_weights[dimension] = weight
        if abs(sum(safe_weights.values()) - 1.0) > 1e-9:
            raise PrivateDashboardError("private Skill evaluation weights disagree")
        if set(evaluation_gates) != set(PRIVATE_SKILL_EVALUATION_GATES) or not all(
            isinstance(evaluation_gates.get(name), bool)
            for name in PRIVATE_SKILL_EVALUATION_GATES
        ):
            raise PrivateDashboardError("private Skill evaluation gates disagree")
        safe_gates = {
            name: evaluation_gates[name] for name in PRIVATE_SKILL_EVALUATION_GATES
        }
        overall_score = _safe_number(
            evaluation.get("overall_score"), purpose="Skill overall score"
        )
        threshold = _safe_number(
            evaluation.get("threshold"), purpose="Skill evaluation threshold"
        )
        grade = evaluation.get("grade")
        passed = evaluation.get("passed")
        if (
            overall_score > 100
            or threshold > 100
            or grade not in {"A", "B", "C", "D"}
            or not isinstance(passed, bool)
        ):
            raise PrivateDashboardError("private Skill evaluation summary is malformed")
        multimodal_evaluation = evaluation.get("multimodal_evaluation")
        if not isinstance(multimodal_evaluation, dict):
            raise PrivateDashboardError("private Skill multimodal evaluation is malformed")
        internal_evidence_score = _safe_number(
            multimodal_evaluation.get("score"),
            purpose="Skill internal evidence consistency score",
        )
        if internal_evidence_score > 100:
            raise PrivateDashboardError("private Skill multimodal evaluation is malformed")
        expected_overall = round(
            sum(
                safe_dimensions[name]
                * PRIVATE_SKILL_EVALUATION_DIMENSIONS[name]
                for name in PRIVATE_SKILL_EVALUATION_DIMENSIONS
            ),
            1,
        )
        expected_grade = (
            "A"
            if overall_score >= 90
            else "B"
            if overall_score >= 80
            else "C"
            if overall_score >= 70
            else "D"
        )
        if (
            evaluation.get("score_scope")
            != "structural_quality_and_internal_evidence_consistency"
            or evaluation.get("teaching_effectiveness_established") is not False
            or evaluation.get("real_world_recognition_accuracy_established") is not False
            or threshold != 75
            or overall_score != expected_overall
            or grade != expected_grade
            or passed != (overall_score >= threshold and all(safe_gates.values()))
            or multimodal_evaluation.get("independent_ground_truth_used") is not False
        ):
            raise PrivateDashboardError("private Skill evaluation semantics disagree")
        report_dimensions = skill_metrics.get("internal_dimensions")
        report_gates = skill_metrics.get("internal_gates")
        if (
            not isinstance(report_dimensions, dict)
            or not isinstance(report_gates, dict)
            or set(report_dimensions) != set(PRIVATE_SKILL_EVALUATION_DIMENSIONS)
            or set(report_gates) != set(PRIVATE_SKILL_EVALUATION_GATES)
            or safe_dimensions
            != {
                name: round(
                    _safe_number(
                        report_dimensions.get(name),
                        purpose=f"reported Skill {name} score",
                    ),
                    1,
                )
                for name in PRIVATE_SKILL_EVALUATION_DIMENSIONS
            }
            or safe_gates
            != {name: report_gates.get(name) for name in PRIVATE_SKILL_EVALUATION_GATES}
            or overall_score
            != _safe_number(
                skill_metrics.get("internal_overall_score"),
                purpose="reported Skill overall score",
            )
            or internal_evidence_score
            != _safe_number(
                skill_metrics.get("internal_evidence_consistency_score"),
                purpose="reported Skill evidence consistency score",
            )
            or passed is not skill_metrics.get("pipeline_internal_evaluation_passed")
        ):
            raise PrivateDashboardError("private Skill evaluation report binding disagrees")
        method_fidelity = _safe_number(
            safe_dimensions.get("method_fidelity"),
            purpose="Skill method fidelity",
        )
        method_fidelity_detail = evaluation.get("method_fidelity")
        expected_method_fidelity = skill_metrics.get("internal_dimensions", {}).get(
            "method_fidelity"
        )
        if (
            not isinstance(method_fidelity_detail, dict)
            or method_fidelity
            != _safe_number(
                method_fidelity_detail.get("score"),
                purpose="Skill method fidelity detail",
            )
            or method_fidelity != float(expected_method_fidelity)
        ):
            raise PrivateDashboardError("private Skill method fidelity disagrees")

        verification_rows = [
            {
                "type": _safe_text(item.get("type"), purpose="verification type", limit=80),
                "prompt": _safe_text(item.get("prompt"), purpose="verification prompt"),
                "pass_condition": _safe_text(
                    item.get("pass_condition"), purpose="verification pass condition"
                ),
            }
            for item in verification_raw
            if isinstance(item, dict)
        ]
        failure_mode_rows = [
            {
                "mode": _safe_text(item.get("mode"), purpose="failure mode", limit=240),
                "mitigation": _safe_text(
                    item.get("mitigation"), purpose="failure mitigation", limit=600
                ),
            }
            for item in failure_modes_raw
            if isinstance(item, dict)
        ]
        if len(verification_rows) != len(verification_raw) or len(
            failure_mode_rows
        ) != len(failure_modes_raw):
            raise PrivateDashboardError("private Skill runtime content is malformed")

        mining = skill.get("mining_metadata")
        phase_analysis = mining.get("teaching_phase_analysis") if isinstance(
            mining, dict
        ) else None
        if not isinstance(phase_analysis, dict):
            raise PrivateDashboardError("private Skill phase analysis is malformed")
        observed_sequence = phase_analysis.get("observed_phase_sequence")
        if not isinstance(observed_sequence, list):
            raise PrivateDashboardError("private Skill observed phase sequence is malformed")

        skill_fingerprint = str(skill_metrics["skill_fingerprint_sha256"])
        payload = {
            "schema_version": "1.0",
            "kind": "local_private_distilled_skill",
            "video_id": video_id,
            "skill": {
                "skill_id": _safe_text(
                    skill.get("skill_id"), purpose="Skill id", limit=120
                ),
                "name": _safe_text(skill.get("name"), purpose="Skill name", limit=240),
                "goal": _safe_text(skill.get("goal"), purpose="Skill goal"),
                "learning_objective": {
                    "statement": _safe_text(
                        skill.get("learning_objective", {}).get("statement"),
                        purpose="learning objective",
                    ),
                    "bloom_level": _safe_text(
                        skill.get("learning_objective", {}).get("bloom_level"),
                        purpose="Bloom level",
                        limit=40,
                    ),
                    "assessment": _safe_text(
                        skill.get("learning_objective", {}).get("assessment"),
                        purpose="learning assessment",
                    ),
                },
                "trigger": _safe_string_list(
                    skill.get("trigger"), purpose="Skill trigger", limit=500
                ),
                "preconditions": _safe_string_list(
                    skill.get("preconditions"),
                    purpose="Skill preconditions",
                    limit=500,
                ),
                "strategies": strategy_rows,
                "procedure": procedure_rows,
                "verification": verification_rows,
                "failure_modes": failure_mode_rows,
                "parameters": execution_parameters,
            },
            "source": {
                "course_id": report_row["course_id"],
                "title": report_row["title"],
                "language": source.get("language"),
                "transcript_kind": source.get("transcript_kind"),
                "duration_seconds": round(duration, 3),
                "caption_coverage_fraction": round(
                    _safe_number(
                        coverage.get("coverage_fraction"),
                        purpose="caption coverage fraction",
                    ),
                    6,
                ),
                "caption_source_verified": bool(
                    source.get("provenance", {}).get("caption_source_verified")
                ),
                "modalities_available": list(source["modalities_available"]),
            },
            "distillation": {
                "method": mining.get("method"),
                "canonical_phase_count": 9,
                "observed_phase_count": observed_count,
                "recommended_phase_count": recommended_count,
                "observed_phase_sequence": observed_sequence,
                "segment_count": int(mining.get("segment_count", 0)),
                "candidate_multimodal_event_count": candidate_event_count,
                "selected_multimodal_event_count": len(multimodal_events),
                "procedure_direct_mme_reference_count": direct_mme_count,
            },
            "text_evidence": evidence_rows,
            "multimodal_events": multimodal_events,
            "evaluation": {
                "score_scope": _safe_text(
                    evaluation.get("score_scope"),
                    purpose="Skill score scope",
                    limit=120,
                ),
                "internal_overall_score": round(overall_score, 1),
                "internal_grade": grade,
                "internal_passed": passed,
                "internal_threshold": round(threshold, 1),
                "dimensions": safe_dimensions,
                "weights": safe_weights,
                "gates": safe_gates,
                "method_fidelity": method_fidelity,
                "schema_valid": safe_gates["schema_valid"],
                "grounded": safe_gates["grounded"],
                "executable": safe_gates["executable"],
                "multimodal_consistent": safe_gates["multimodal_consistent"],
                "internal_evidence_consistency_score": round(
                    internal_evidence_score, 1
                ),
                "internal_score_is_accuracy": False,
                "teaching_effectiveness_established": False,
                "real_world_recognition_accuracy_established": False,
            },
            "runtime_contract": {
                "states": ["READY", "PROCEDURE", "VERIFY", "COMPLETE"],
                "success_transition": "advance",
                "failure_transition": "retry_current_step_with_fallback",
                "completion_semantics": "state_machine_path_completion_only",
            },
            "claim_boundary": {
                "full_video_bytes_downloaded_and_hashed": True,
                "official_caption_timeline_alignment_verified": True,
                "procedure_is_text_led": True,
                "multimodal_events_assist_strategy_scoring": True,
                "multimodal_events_directly_generate_procedure_steps": False,
                "independent_event_ground_truth_used": False,
                "event_recognition_accuracy_established": False,
                "human_skill_quality_labels_used": False,
                "teaching_effectiveness_established": False,
                "internal_score_is_accuracy": False,
            },
            "provenance": {
                "dataset_manifest_sha256": _short_hash(
                    _file_sha256(manifest_path)
                ),
                "ablation_report_sha256": _short_hash(_file_sha256(report_path)),
                "skill_fingerprint": _short_hash(skill_fingerprint),
                "skill_file_sha256": _short_hash(_file_sha256(skill_path)),
                "evaluation_file_sha256": _short_hash(
                    _file_sha256(evaluation_path)
                ),
                "source_transcript_fingerprint": _short_hash(
                    str(report_row["source_transcript_fingerprint_sha256"])
                ),
            },
        }
        catalog_row = {
            "id": video_id,
            "course_id": report_row["course_id"],
            "title": report_row["title"],
            "duration_seconds": round(duration, 3),
            "candidate_event_count": candidate_event_count,
            "selected_multimodal_event_count": len(multimodal_events),
            "observed_phase_count": observed_count,
            "recommended_phase_count": recommended_count,
            "procedure_step_count": len(procedure_rows),
            "method_fidelity": method_fidelity,
            "detail": f"api/skills/{video_id}",
        }
        skills[video_id] = PrivateSkillSnapshot(
            video_id=video_id,
            catalog_row=catalog_row,
            payload=payload,
        )
        totals["procedure_step_count"] += len(procedure_rows)
        totals["observed_method_step_count"] += observed_count
        totals["recommended_enrichment_step_count"] += recommended_count
        totals["text_evidence_count"] += len(evidence_rows)
        totals["selected_multimodal_event_count"] += len(multimodal_events)
        totals["verification_step_count"] += len(verification_rows)
        totals["procedure_direct_mme_reference_count"] += direct_mme_count
        total_duration += duration

    if len(skills) != 10 or initial_skill not in skills:
        raise PrivateDashboardError("private Skill default is outside the frozen track")
    aggregate_raw = manifest.get("aggregate")
    if not isinstance(aggregate_raw, dict):
        raise PrivateDashboardError("private Skill aggregate is malformed")
    aggregate_internal = report.get("aggregate_internal_metrics", {})
    full_internal = aggregate_internal.get("full", {})
    text_internal = aggregate_internal.get("transcript_only", {})
    if not isinstance(full_internal, dict) or not isinstance(text_internal, dict):
        raise PrivateDashboardError("private Skill internal ablation summary is malformed")
    aggregate = {
        "complete_lecture_count": len(skills),
        "course_count": len(
            {snapshot.catalog_row["course_id"] for snapshot in skills.values()}
        ),
        "duration_hours": round(total_duration / 3600, 6),
        "keyframe_count": int(aggregate_raw.get("keyframe_count", 0)),
        "candidate_multimodal_event_count": int(
            aggregate_raw.get("fused_event_count", 0)
        ),
        "candidate_event_type_frequency": dict(
            sorted(candidate_event_type_frequency.items())
        ),
        "selected_event_type_frequency": dict(
            sorted(selected_event_type_frequency.items())
        ),
        **totals,
        "full_internal_scale_mean": full_internal.get(
            "mean_internal_overall_score"
        ),
        "full_minus_transcript_internal_scale_delta": round(
            float(full_internal.get("mean_internal_overall_score"))
            - float(text_internal.get("mean_internal_overall_score")),
            6,
        ),
        "internal_scale_is_accuracy": False,
        "candidate_events_are_human_verified_truth": False,
        "expert_skill_gold_established": False,
        "teaching_effectiveness_established": False,
    }
    if aggregate["candidate_multimodal_event_count"] != sum(
        row.catalog_row["candidate_event_count"] for row in skills.values()
    ):
        raise PrivateDashboardError("private Skill aggregate event count disagrees")
    return skills, initial_skill, aggregate


def _parse_duration(value: str) -> float:
    pieces = value.split(":")
    if len(pieces) not in {2, 3}:
        raise PrivateDashboardError("invalid TeachObs lesson duration")
    try:
        numbers = [int(piece) for piece in pieces]
    except ValueError as exc:
        raise PrivateDashboardError("invalid TeachObs lesson duration") from exc
    if any(number < 0 for number in numbers) or numbers[-1] >= 60:
        raise PrivateDashboardError("invalid TeachObs lesson duration")
    if len(numbers) == 2:
        minutes, seconds = numbers
        return float(minutes * 60 + seconds)
    hours, minutes, seconds = numbers
    if minutes >= 60:
        raise PrivateDashboardError("invalid TeachObs lesson duration")
    return float(hours * 3600 + minutes * 60 + seconds)


def _vtt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def _vtt_text(value: str) -> str:
    return " ".join(value.replace("-->", "→").split())


def _build_vtt(rows: list[dict[str, Any]]) -> bytes:
    lines = ["WEBVTT", ""]
    for row in rows:
        start = float(row["start_seconds"])
        end = float(row["end_seconds"])
        text = _vtt_text(str(row["text"]))
        if not text:
            continue
        lines.extend(
            [
                str(row["scene_no"]),
                f"{_vtt_timestamp(start)} --> {_vtt_timestamp(end)}",
                text,
                "",
            ]
        )
    return "\n".join(lines).encode("utf-8")


def _lesson_metadata(repository_root: Path) -> dict[str, dict[str, str]]:
    path = _safe_file(
        repository_root,
        Path("data/lessons.csv"),
        kind="lesson metadata",
    )
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise PrivateDashboardError("invalid private lesson metadata") from exc
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        lesson_id = str(row.get("id", ""))
        if _LESSON_ID.fullmatch(lesson_id):
            result[lesson_id] = {key: str(value or "") for key, value in row.items()}
    return result


def _feature_manifest_rows(
    root: Path,
    path: Path,
) -> dict[str, dict[str, Any]]:
    manifest = _read_json(path, purpose="feature manifest")
    rows = manifest.get("lessons")
    if not isinstance(rows, list):
        raise PrivateDashboardError("private feature manifest has no lesson rows")
    result: dict[str, dict[str, Any]] = {}
    for value in rows:
        if not isinstance(value, dict):
            raise PrivateDashboardError("private feature lesson row is malformed")
        lesson_id = value.get("lesson_id")
        if not isinstance(lesson_id, str) or not _LESSON_ID.fullmatch(lesson_id):
            raise PrivateDashboardError("private feature lesson id is malformed")
        result[lesson_id] = value
    if not result or path.parent != root / "media":
        raise PrivateDashboardError("private feature manifest root differs")
    return result


def build_private_snapshot(
    config: PrivateDashboardConfig,
) -> PrivateDashboardSnapshot:
    """Validate local evidence and recompute every held-out frozen prediction."""

    root_input = Path(config.root).expanduser()
    if root_input.is_symlink():
        raise PrivateDashboardError("private TeachObs root may not be a symlink")
    try:
        root = root_input.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PrivateDashboardError("private TeachObs root is missing") from exc
    if not root.is_dir():
        raise PrivateDashboardError("private TeachObs root is not a directory")

    repository_root = _safe_directory(root, Path("repository"), kind="repository")
    _safe_directory(root, Path("media"), kind="media root")
    frozen_root = _safe_directory(root, Path("frozen_models"), kind="frozen model")
    transcript_manifest = _safe_file(
        root,
        Path("materialized_transcripts/manifest.json"),
        kind="transcript manifest",
    )
    feature_manifest = _safe_file(
        root,
        Path("media/feature_manifest.json"),
        kind="feature manifest",
    )
    template_check = private_dashboard_template_self_check()
    if not template_check["passed"]:
        raise PrivateDashboardError("private dashboard template failed its self-check")
    skills: dict[str, PrivateSkillSnapshot] = {}
    default_skill: str | None = None
    skill_aggregate: dict[str, Any] = {}
    if config.skill_root is not None:
        skills, default_skill, skill_aggregate = _build_private_skill_snapshots(
            Path(config.skill_root),
            initial_skill=config.initial_skill,
        )

    # Heavy optional imports stay inside the private path so the public dashboard
    # remains zero-dependency.
    from .teachobs_benchmark import load_teachobs_text_dataset
    from .teachobs_frozen_model import (
        load_teachobs_frozen_bundle,
        predict_teachobs_frozen_bundle,
    )
    from .teachobs_media import AUDIO_SCHEMA, VISUAL_EVIDENCE_SCHEMA
    from .teachobs_multimodal_benchmark import (
        PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
        PAPER_TRACK1_TEST_LESSON_IDS,
        _load_materialized_transcript_dataset,
        load_teachobs_multimodal_features,
        resolve_teachobs_benchmark_profile,
    )

    full_dataset = load_teachobs_text_dataset(repository_root)
    profile = resolve_teachobs_benchmark_profile(
        full_dataset,
        PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
    )
    dataset, transcript_audit = _load_materialized_transcript_dataset(
        profile,
        transcript_manifest,
    )
    features = load_teachobs_multimodal_features(
        repository_root,
        feature_manifest,
        dataset=full_dataset,
        profile=PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
    )
    bundle = load_teachobs_frozen_bundle(
        frozen_root,
        expected_benchmark_profile=PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
        expected_dataset_profile_fingerprint=profile.dataset_profile_fingerprint,
        expected_transcript_materialization_fingerprint=transcript_audit[
            "materialization_fingerprint_sha256"
        ],
    )
    if len(features.visual_evidence_configuration_sha256_values) != 1:
        raise PrivateDashboardError(
            "private visual evidence has multiple extractor configurations"
        )
    outputs = predict_teachobs_frozen_bundle(
        bundle,
        transcripts=[scene.transcript for scene in dataset.test_scenes],
        transcript_input_materialization_fingerprint_sha256=transcript_audit[
            "materialization_fingerprint_sha256"
        ],
        ocr_texts=features.test_ocr_text,
        audio=features.test_audio,
        visual_numeric=features.test_visual_numeric,
        clip=features.test_clip,
        label_names=dataset.code_names,
        audio_feature_names=features.audio_feature_names,
        visual_numeric_feature_names=features.visual_numeric_feature_names,
        clip_source_revision=features.clip_source_revision,
        clip_weight_manifest_sha256=features.clip_weight_manifest_sha256,
        audio_feature_schema=AUDIO_SCHEMA,
        visual_evidence_schema=VISUAL_EVIDENCE_SCHEMA,
        visual_evidence_configuration_sha256=(
            features.visual_evidence_configuration_sha256_values[0]
        ),
        sample_ids=list(profile.selected_test_sample_ids),
        benchmark_profile=PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
        dataset_profile_fingerprint=profile.dataset_profile_fingerprint,
    )
    input_fingerprints = {
        str(outputs[arm]["prediction_input_fingerprint"])
        for arm in PRIVATE_DASHBOARD_ARMS
    }
    if len(input_fingerprints) != 1:
        raise PrivateDashboardError("frozen arms disagree on prediction input")
    prediction_input_fingerprint = input_fingerprints.pop()

    metadata = _lesson_metadata(repository_root)
    manifest_rows = _feature_manifest_rows(root, feature_manifest)
    scenes_by_lesson: dict[str, list[tuple[int, Any]]] = {
        lesson_id: [] for lesson_id in PAPER_TRACK1_TEST_LESSON_IDS
    }
    for global_index, scene in enumerate(dataset.test_scenes):
        if scene.lesson_id not in scenes_by_lesson:
            raise PrivateDashboardError("unexpected lesson in selected test scenes")
        scenes_by_lesson[scene.lesson_id].append((global_index, scene))

    lessons: dict[str, PrivateLessonSnapshot] = {}
    for lesson_id in PAPER_TRACK1_TEST_LESSON_IDS:
        feature_row = manifest_rows.get(lesson_id)
        lesson_meta = metadata.get(lesson_id)
        if feature_row is None or lesson_meta is None:
            raise PrivateDashboardError(f"private lesson metadata is missing for {lesson_id}")
        media_path = _safe_file(
            root,
            Path(f"media/videos/{lesson_id}.mp4"),
            kind="lesson video",
        )
        visual_path = _safe_file(
            root,
            Path(f"media/features/visual_evidence/{lesson_id}.json"),
            kind="visual evidence",
        )
        transcript_path = _safe_file(
            root,
            Path(f"materialized_transcripts/lessons/{lesson_id}.jsonl"),
            kind="materialized transcript",
        )
        visual = _read_json(visual_path, purpose="visual evidence")
        visual_rows = visual.get("scenes")
        if not isinstance(visual_rows, list):
            raise PrivateDashboardError("private visual evidence has no scenes")
        transcript_rows = _read_jsonl(
            transcript_path,
            purpose="materialized transcript",
        )
        selected = scenes_by_lesson[lesson_id]
        if not (
            len(selected) == len(visual_rows) == len(transcript_rows)
            == int(feature_row.get("scene_count", -1))
        ):
            raise PrivateDashboardError(
                f"private scene counts disagree for {lesson_id}"
            )

        dto_rows: list[dict[str, Any]] = []
        frame_paths: list[Path] = []
        frame_hashes: list[str] = []
        for local_index, ((global_index, scene), visual_row, transcript_row) in enumerate(
            zip(selected, visual_rows, transcript_rows, strict=True),
            start=1,
        ):
            if not isinstance(visual_row, dict):
                raise PrivateDashboardError("private visual scene is malformed")
            if (
                scene.scene_no != local_index
                or visual_row.get("scene_no") != local_index
                or transcript_row.get("scene_no") != local_index
                or transcript_row.get("lesson_id") != lesson_id
                or transcript_row.get("text") != scene.transcript
            ):
                raise PrivateDashboardError(
                    f"private scene ordering differs for {lesson_id}"
                )
            start_seconds = float(transcript_row.get("start_seconds"))
            end_seconds = float(transcript_row.get("end_seconds"))
            if (
                not math.isfinite(start_seconds)
                or not math.isfinite(end_seconds)
                or start_seconds < 0
                or end_seconds <= start_seconds
            ):
                raise PrivateDashboardError("private scene timing is invalid")
            frame_name = visual_row.get("path")
            frame_sha = visual_row.get("sha256")
            if (
                not isinstance(frame_name, str)
                or Path(frame_name).name != frame_name
                or Path(frame_name).suffix.casefold() not in {".jpg", ".jpeg"}
                or not isinstance(frame_sha, str)
                or not re.fullmatch(r"[0-9a-f]{64}", frame_sha)
            ):
                raise PrivateDashboardError("private frame binding is malformed")
            frame_path = _safe_file(
                root,
                Path(f"media/frames/{lesson_id}") / frame_name,
                kind="scene frame",
            )
            frame_paths.append(frame_path)
            frame_hashes.append(frame_sha)

            truth = {
                index for index, value in enumerate(scene.labels) if int(value) == 1
            }
            arm_rows: dict[str, Any] = {}
            for arm in PRIVATE_DASHBOARD_ARMS:
                probabilities = outputs[arm]["probabilities"][global_index]
                predictions = outputs[arm]["predictions"][global_index]
                predicted = {
                    index for index, value in enumerate(predictions) if int(value) == 1
                }
                labels = [
                    {
                        "label": dataset.code_names[index],
                        "group": dataset.code_groups[index],
                        "predicted": index in predicted,
                        "score": round(float(probabilities[index]), 6),
                        "reference": index in truth,
                    }
                    for index in range(len(dataset.code_names))
                ]
                labels.sort(
                    key=lambda row: (
                        not bool(row["predicted"]),
                        -float(row["score"]),
                        str(row["label"]),
                    )
                )
                arm_rows[arm] = {
                    "labels": labels,
                    "behavior_view": project_teachobs_behavior_view(labels),
                    "summary": {
                        "predicted_positive": len(predicted),
                        "reference_positive": len(truth),
                        "true_positive": len(predicted & truth),
                        "false_positive": len(predicted - truth),
                        "false_negative": len(truth - predicted),
                    },
                }
            dto_rows.append(
                {
                    "schema_version": "1.1",
                    "lesson": lesson_id,
                    "scene_number": local_index,
                    "scene_count": len(selected),
                    "start_seconds": round(start_seconds, 3),
                    "end_seconds": round(end_seconds, 3),
                    "transcript": {
                        "text": scene.transcript,
                        "source_tier": str(transcript_row.get("source_tier", "unknown")),
                        "empty": not bool(scene.transcript.strip()),
                    },
                    "visual": {
                        "frame_url": f"frames/{lesson_id}/{local_index}",
                        "frame_kind": "time_aligned_scene_frame",
                        "ocr_used_as_frozen_model_input": True,
                        "ocr_displayed": False,
                    },
                    "reference": {
                        "kind": "released_consensus_labels",
                        "positive_labels": [
                            dataset.code_names[index] for index in sorted(truth)
                        ],
                    },
                    "predictions": arm_rows,
                    "provenance": {
                        "media_sha256": _short_hash(str(feature_row["media_sha256"])),
                        "scene_manifest_sha256": _short_hash(
                            str(feature_row["scene_manifest_sha256"])
                        ),
                        "frame_sha256": _short_hash(frame_sha),
                        "bundle_fingerprint": _short_hash(bundle.bundle_fingerprint),
                        "prediction_input_fingerprint": _short_hash(
                            prediction_input_fingerprint
                        ),
                    },
                    "claim_boundary": {
                        "real_video": True,
                        "real_materialized_transcript": True,
                        "real_ocr": True,
                        "ocr_displayed": False,
                        "frozen_model_prediction": True,
                        "teacher_behavior_view": True,
                        "behavior_view_is_frozen_label_projection": True,
                        "direct_student_action_recognition": False,
                        "pose_tracking_performed": False,
                        "live_feature_extraction": False,
                        "ocr_accuracy_established": False,
                        "action_subgroup_accuracy_established": False,
                        "score_is_calibrated_probability": False,
                        "deployment_accuracy_established": False,
                    },
                }
            )

        duration = _parse_duration(lesson_meta["duration"])
        showcase_candidates = [
            row
            for row in dto_rows
            if row["transcript"]["text"]
            and row["predictions"]["full"]["summary"]["true_positive"] > 0
            and any(
                action["predicted"] or action["reference"]
                for group in (
                    "teacher_visible_actions",
                    "learner_related_teaching_signals",
                )
                for action in row["predictions"]["full"]["behavior_view"][group]
            )
        ]
        showcase = max(
            showcase_candidates or dto_rows,
            key=lambda row: (
                row["predictions"]["full"]["summary"]["true_positive"],
                -row["predictions"]["full"]["summary"]["false_negative"],
                sum(
                    bool(action["predicted"] or action["reference"])
                    for group in (
                        "teacher_visible_actions",
                        "learner_related_teaching_signals",
                    )
                    for action in row["predictions"]["full"]["behavior_view"][group]
                ),
                len(row["transcript"]["text"]),
            ),
        )
        catalog_row = {
            "id": lesson_id,
            "subject": lesson_meta["subject"],
            "school_level": lesson_meta["school_level"],
            "country": lesson_meta["country"],
            "source": lesson_meta["source"],
            "duration_seconds": duration,
            "scene_count": len(dto_rows),
            "suggested_scene": showcase["scene_number"],
            "video": f"media/{lesson_id}",
            "captions": f"captions/{lesson_id}.vtt",
        }
        expected_media_sha = str(feature_row.get("media_sha256", ""))
        scene_manifest_sha = str(feature_row.get("scene_manifest_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", expected_media_sha) or not re.fullmatch(
            r"[0-9a-f]{64}", scene_manifest_sha
        ):
            raise PrivateDashboardError("private media binding is malformed")
        lessons[lesson_id] = PrivateLessonSnapshot(
            lesson_id=lesson_id,
            catalog_row=catalog_row,
            media_path=media_path,
            expected_media_sha256=expected_media_sha,
            scene_manifest_sha256=scene_manifest_sha,
            scenes=tuple(dto_rows),
            frame_paths=tuple(frame_paths),
            expected_frame_sha256=tuple(frame_hashes),
            captions_vtt=_build_vtt(transcript_rows),
        )

    if tuple(lessons) != tuple(PAPER_TRACK1_TEST_LESSON_IDS):
        raise PrivateDashboardError("private lesson order differs from frozen profile")
    if config.initial_lesson not in lessons:
        raise PrivateDashboardError("initial private lesson is outside the frozen profile")
    prediction_hashes = {
        arm: str(outputs[arm]["prediction_matrix_sha256"])
        for arm in PRIVATE_DASHBOARD_ARMS
    }
    return PrivateDashboardSnapshot(
        lessons=lessons,
        default_lesson=config.initial_lesson,
        bundle_fingerprint=bundle.bundle_fingerprint,
        transcript_fingerprint=transcript_audit[
            "materialization_fingerprint_sha256"
        ],
        prediction_input_fingerprint=prediction_input_fingerprint,
        prediction_matrix_sha256=prediction_hashes,
        template=private_dashboard_html_bytes(),
        skills=skills,
        default_skill=default_skill,
        skill_aggregate=skill_aggregate,
    )


def parse_single_byte_range(
    header: str | None,
    size: int,
) -> tuple[int, int] | None:
    """Parse one RFC 7233 byte range, rejecting multipart and invalid forms."""

    if size < 0:
        raise ValueError("size must be nonnegative")
    if header is None:
        return None
    match = _SINGLE_RANGE.fullmatch(header.strip())
    if match is None or "," in header:
        raise RangeNotSatisfiable("only one byte range is supported")
    first, last = match.groups()
    if not first and not last:
        raise RangeNotSatisfiable("empty byte range")
    if size == 0:
        raise RangeNotSatisfiable("empty resource")
    if not first:
        suffix = int(last)
        if suffix <= 0:
            raise RangeNotSatisfiable("invalid suffix range")
        return max(0, size - suffix), size - 1
    start = int(first)
    if start >= size:
        raise RangeNotSatisfiable("range starts after the resource")
    if not last:
        return start, size - 1
    end = int(last)
    if end < start:
        raise RangeNotSatisfiable("range end precedes start")
    return start, min(end, size - 1)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _copy_range(
    source: BinaryIO,
    target: BinaryIO,
    length: int,
) -> None:
    remaining = length
    while remaining:
        chunk = source.read(min(1024 * 1024, remaining))
        if not chunk:
            break
        target.write(chunk)
        remaining -= len(chunk)


def create_private_dashboard_server(
    snapshot: PrivateDashboardSnapshot,
    *,
    port: int = 0,
    capability_token: str | None = None,
) -> tuple[ThreadingHTTPServer, str]:
    """Create, but do not start, a loopback-only capability server."""

    if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
        raise PrivateDashboardError("private dashboard port is invalid")
    token = capability_token or secrets.token_urlsafe(24)
    if not re.fullmatch(r"[A-Za-z0-9_-]{20,128}", token):
        raise PrivateDashboardError("private dashboard capability token is invalid")
    prefix = f"/{token}/"

    class PrivateDashboardHandler(BaseHTTPRequestHandler):
        server_version = "TeachingSkillMinerPrivate/1.0"
        sys_version = ""

        def log_message(self, _format: str, *args: Any) -> None:
            del args

        def _secure_headers(self) -> None:
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header(
                "Permissions-Policy",
                "camera=(), microphone=(), geolocation=(), payment=()",
            )
            self.send_header("Content-Security-Policy", _CSP)

        def _request_is_local(self) -> bool:
            host_header = self.headers.get("Host", "")
            try:
                parsed_authority = urlsplit(f"//{host_header}")
                parsed_host = (parsed_authority.hostname or "").casefold()
                parsed_port = parsed_authority.port
            except ValueError:
                return False
            if parsed_host not in _SAFE_HOSTS or parsed_port not in {
                None,
                self.server.server_port,
            }:
                return False
            if self.headers.get("Sec-Fetch-Site", "").casefold() in {
                "cross-site",
                "cross-origin",
            }:
                return False
            origin = self.headers.get("Origin")
            if origin:
                parsed = urlsplit(origin)
                if parsed.scheme != "http" or (parsed.hostname or "").casefold() not in _SAFE_HOSTS:
                    return False
                if parsed.port not in {None, self.server.server_port}:
                    return False
            return self.client_address[0] in {"127.0.0.1", "::1"}

        def _error(self, status: HTTPStatus, message: str) -> None:
            body = _json_bytes({"error": message, "status": int(status)})
            self.send_response(status)
            self._secure_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _payload(
            self,
            payload: bytes,
            *,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            self.send_response(status)
            self._secure_headers()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        def _route(self) -> None:
            if not self._request_is_local():
                self._error(HTTPStatus.FORBIDDEN, "loopback origin required")
                return
            target = urlsplit(self.path)
            if target.query or target.fragment:
                self._error(HTTPStatus.BAD_REQUEST, "query strings are not supported")
                return
            decoded = unquote(target.path)
            if ".." in decoded.split("/") or not decoded.startswith(prefix):
                self._error(HTTPStatus.FORBIDDEN, "valid capability path required")
                return
            route = decoded[len(prefix) :]
            if route in {"", "index.html"}:
                self._payload(
                    snapshot.template,
                    content_type="text/html; charset=utf-8",
                )
                return
            if route == "assets/private_skill_demo.css":
                self._payload(
                    private_dashboard_asset_bytes(
                        PRIVATE_DASHBOARD_SKILL_STYLE_RESOURCE
                    ),
                    content_type="text/css; charset=utf-8",
                )
                return
            if route == "assets/private_skill_demo.js":
                self._payload(
                    private_dashboard_asset_bytes(
                        PRIVATE_DASHBOARD_SKILL_SCRIPT_RESOURCE
                    ),
                    content_type="text/javascript; charset=utf-8",
                )
                return
            if route == "api/catalog":
                self._payload(
                    _json_bytes(snapshot.catalog()),
                    content_type="application/json; charset=utf-8",
                )
                return
            skill_match = _SKILL_ROUTE.fullmatch(route)
            if skill_match:
                try:
                    skill = snapshot.skill(skill_match.group(1))
                except KeyError:
                    self._error(HTTPStatus.NOT_FOUND, "Skill not found")
                    return
                self._payload(
                    _json_bytes(skill),
                    content_type="application/json; charset=utf-8",
                )
                return
            scene_match = _SCENE_ROUTE.fullmatch(route)
            if scene_match:
                lesson_id, scene_number_text = scene_match.groups()
                try:
                    scene = snapshot.scene(lesson_id, int(scene_number_text))
                except KeyError:
                    self._error(HTTPStatus.NOT_FOUND, "scene not found")
                    return
                self._payload(
                    _json_bytes(scene),
                    content_type="application/json; charset=utf-8",
                )
                return
            media_match = _MEDIA_ROUTE.fullmatch(route)
            if media_match:
                self._serve_media(media_match.group(1))
                return
            frame_match = _FRAME_ROUTE.fullmatch(route)
            if frame_match:
                self._serve_frame(frame_match.group(1), int(frame_match.group(2)))
                return
            caption_match = _CAPTION_ROUTE.fullmatch(route)
            if caption_match:
                lesson = snapshot.lessons.get(caption_match.group(1))
                if lesson is None:
                    self._error(HTTPStatus.NOT_FOUND, "captions not found")
                    return
                self._payload(
                    lesson.captions_vtt,
                    content_type="text/vtt; charset=utf-8",
                )
                return
            self._error(HTTPStatus.NOT_FOUND, "resource not found")

        def _serve_frame(self, lesson_id: str, scene_number: int) -> None:
            lesson = snapshot.lessons.get(lesson_id)
            if lesson is None or not 1 <= scene_number <= len(lesson.frame_paths):
                self._error(HTTPStatus.NOT_FOUND, "frame not found")
                return
            try:
                lesson.verify_frame(scene_number)
                payload = lesson.frame_paths[scene_number - 1].read_bytes()
            except (OSError, PrivateDashboardError):
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "frame verification failed")
                return
            content_type = mimetypes.guess_type(lesson.frame_paths[scene_number - 1].name)[0]
            self._payload(payload, content_type=content_type or "image/jpeg")

        def _serve_media(self, lesson_id: str) -> None:
            lesson = snapshot.lessons.get(lesson_id)
            if lesson is None:
                self._error(HTTPStatus.NOT_FOUND, "video not found")
                return
            try:
                lesson.verify_media()
                size = lesson.media_path.stat().st_size
                selected = parse_single_byte_range(self.headers.get("Range"), size)
            except RangeNotSatisfiable:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self._secure_headers()
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            except (OSError, PrivateDashboardError):
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "video verification failed")
                return
            start, end = selected if selected is not None else (0, size - 1)
            length = end - start + 1
            status = HTTPStatus.PARTIAL_CONTENT if selected is not None else HTTPStatus.OK
            self.send_response(status)
            self._secure_headers()
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            if selected is not None:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            if self.command == "HEAD":
                return
            try:
                with lesson.media_path.open("rb") as handle:
                    handle.seek(start)
                    _copy_range(handle, self.wfile, length)
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            self._route()

        def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler API
            self._route()

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            self._error(HTTPStatus.METHOD_NOT_ALLOWED, "read-only dashboard")

    server = ThreadingHTTPServer(("127.0.0.1", port), PrivateDashboardHandler)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{server.server_port}/{token}/"
    return server, url


def private_snapshot_summary(snapshot: PrivateDashboardSnapshot) -> dict[str, Any]:
    """Return a path-free status object suitable for CLI output."""

    return {
        "schema_version": "1.0",
        "dashboard_kind": "local_private_real_evidence",
        "mode": "local_private_real_data",
        "lesson_count": len(snapshot.lessons),
        "scene_count": sum(len(lesson.scenes) for lesson in snapshot.lessons.values()),
        "default_lesson": snapshot.default_lesson,
        "prediction_arms": list(PRIVATE_DASHBOARD_ARMS),
        "bundle_fingerprint": snapshot.bundle_fingerprint,
        "prediction_input_fingerprint": snapshot.prediction_input_fingerprint,
        "product_count": 2,
        "real_video": True,
        "real_ocr": True,
        "ocr_displayed": False,
        "real_materialized_transcript": True,
        "frozen_per_scene_predictions": True,
        "teacher_behavior_projection": True,
        "direct_student_action_recognition": False,
        "skill_lesson_count": len(snapshot.skills),
        "skill_procedure_step_count": int(
            snapshot.skill_aggregate.get("procedure_step_count", 0)
        ),
        "real_skill_distillation": bool(snapshot.skills),
        "skill_procedure_is_text_led": True,
        "skill_direct_mme_reference_count": int(
            snapshot.skill_aggregate.get(
                "procedure_direct_mme_reference_count", 0
            )
        ),
        "expert_skill_gold_established": False,
        "skill_accuracy_established": False,
        "teaching_effectiveness_established": False,
        "live_feature_extraction": False,
        "deployment_accuracy_established": False,
        "private_paths_disclosed": False,
        "passed": True,
    }


def serve_private_dashboard(
    root: str | Path,
    *,
    skill_root: str | Path | None = None,
    initial_lesson: str = DEFAULT_PRIVATE_LESSON,
    initial_skill: str = DEFAULT_PRIVATE_SKILL,
    port: int = 0,
    open_browser: bool = True,
) -> int:
    """Build and serve the real-evidence dashboard until interrupted."""

    snapshot = build_private_snapshot(
        PrivateDashboardConfig(
            root=Path(root),
            initial_lesson=initial_lesson,
            skill_root=Path(skill_root) if skill_root is not None else None,
            initial_skill=initial_skill,
        )
    )
    server, url = create_private_dashboard_server(snapshot, port=port)
    status = {
        **private_snapshot_summary(snapshot),
        "dashboard_url": url,
        "loopback_only": True,
        "cache_control": "no-store",
        "browser_open_requested": open_browser,
    }
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0
