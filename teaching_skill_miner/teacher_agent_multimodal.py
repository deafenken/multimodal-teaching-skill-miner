"""Auditable multimodal analysis above local extraction boundaries.

The contract deliberately separates three questions that are often collapsed:

* transcription: what symbols or words a provider says are present;
* semantic analysis: what a provider says a chart/diagram/media segment means;
* assessment: whether a learner answer is correct or demonstrates mastery.

Only the first two are represented here.  Neither is assessment evidence.  Raw
remote image processing requires a purpose-, provider-, region-, retention-,
and data-category-bound server consent receipt.  The current consent authority
does not define raw audio/video categories, so temporal media is local-only and
fails closed before a remote provider can be called.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Callable, Mapping, Protocol

from .teacher_agent_consent import RemoteConsentStore
from .teacher_agent_vision import extract_local_visual_evidence, validate_local_image


# The v1 visual names remain public compatibility aliases for the dashboard and
# existing persisted attachment records.  New consumers should bind to the
# general provider/result/evidence contracts below.
VISUAL_SEMANTIC_EVIDENCE_SCHEMA = "teaching_skill_miner.visual_semantic_evidence.v1"
VISUAL_PROVIDER_RESULT_SCHEMA = "teaching_skill_miner.visual_provider_result.v1"
MULTIMODAL_EVIDENCE_SCHEMA = "teaching_skill_miner.multimodal_evidence.v2"
MULTIMODAL_PROVIDER_SPEC_SCHEMA = "teaching_skill_miner.multimodal_provider_spec.v1"
MULTIMODAL_PROVIDER_RESULT_SCHEMA = "teaching_skill_miner.multimodal_provider_result.v1"

MAX_VISUAL_CONTEXT_CHARS = 1_200
MAX_VISUAL_DESCRIPTION_CHARS = 2_000
MAX_VISUAL_CLAIMS = 12
MAX_MULTIMODAL_MEDIA_BYTES = 32 * 1024 * 1024
MAX_TRANSCRIPT_SEGMENTS = 512
MAX_TRANSCRIPT_CHARS = 24_000

_SEMANTIC_MODALITIES = frozenset(
    {
        "chart",
        "table",
        "diagram",
        "geometry",
        "formula",
        "handwriting",
        "mixed",
        "unknown",
    }
)
_SOURCE_MODALITIES = frozenset(
    {"image", "scanned_pdf_page", "slide", "audio", "video"}
)
_CAPABILITIES = frozenset(
    {"transcription", "semantic_analysis", "temporal_transcription"}
)
_DECISIONS = frozenset({"observation", "requires_confirmation", "abstain"})
_PROVIDER_ID = re.compile(r"^[A-Za-z0-9_.-]{2,80}$")
_VERSION = re.compile(r"^[A-Za-z0-9_.+-]{1,80}$")
_REGION = re.compile(r"^[A-Za-z0-9_.-]{2,40}$")
_MIME = re.compile(r"^[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*$")

SUPPORTED_MULTIMODAL_MIME_TYPES = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/webp",
        "application/pdf",
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp4",
        "audio/x-m4a",
        "audio/webm",
        "video/mp4",
        "video/quicktime",
        "video/webm",
    }
)


class VisualSemanticError(ValueError):
    """Raised when multimodal semantics cannot cross the evidence boundary."""


class VisualSemanticProvider(Protocol):
    """Compatibility protocol for image-only v1 providers."""

    provider_id: str
    processing_region: str
    sends_raw_media_remotely: bool

    def analyze(
        self, image_bytes: bytes, mime_type: str, *, task_context: str
    ) -> Mapping[str, Any]: ...


class MultimodalProvider(Protocol):
    """Versioned provider contract for local or explicitly consented analysis."""

    provider_spec: Mapping[str, Any]

    def analyze(
        self, media_bytes: bytes, mime_type: str, *, task_context: str
    ) -> Mapping[str, Any]: ...


class TemporalTranscriptionProvider(MultimodalProvider, Protocol):
    """A local audio/video transcription adapter with timestamped segments."""


class LocalOCRVisualSemanticProvider:
    """On-device, modest OCR adapter that never claims visual understanding.

    OCR can establish a bounded transcription candidate.  It cannot establish
    a chart trend, geometry relation, handwriting intent, formula correctness,
    learner correctness, or mastery.
    """

    provider_id = "local-ocr-visual-v1"
    processing_region = "on_device"
    sends_raw_media_remotely = False
    provider_spec = {
        "schema": MULTIMODAL_PROVIDER_SPEC_SCHEMA,
        "provider_id": provider_id,
        "adapter_version": "1.0.0",
        "execution_scope": "local",
        "processing_region": processing_region,
        "raw_media_transport": "none",
        "provider_retention_days": 0,
        "supported_source_modalities": ["image"],
        "supported_mime_types": ["image/jpeg", "image/png", "image/webp"],
        "capabilities": ["transcription"],
        "deterministic": False,
    }

    def __init__(self, extractor: Callable[..., dict[str, Any]] | None = None) -> None:
        self._extractor = extractor or extract_local_visual_evidence

    def analyze(
        self, image_bytes: bytes, mime_type: str, *, task_context: str
    ) -> Mapping[str, Any]:
        del task_context
        evidence = self._extractor(
            image_bytes,
            mime_type,
            display_name="local-visual-semantic-analysis",
        )
        text = str(evidence.get("recognized_text", "")).strip()
        reliable = evidence.get("status") == "recognized" and not bool(
            evidence.get("needs_student_confirmation")
        )
        uncertainties = [] if reliable else ["本地转录需要学习者或教师复核"]
        return {
            "schema": MULTIMODAL_PROVIDER_RESULT_SCHEMA,
            "source_modality": "image",
            "modality": (
                "formula"
                if evidence.get("formula_like_text_detected") is True
                else "unknown"
            ),
            "transcription": {
                "status": "candidate" if text else "unavailable",
                "text": text,
                "confidence": float(evidence.get("transcription_confidence", 0.0)),
                "language": "und",
                "segments": [],
            },
            "semantic_analysis": {
                "status": "not_performed",
                "description": (
                    "本地适配器只完成 OCR 转写，没有执行视觉语义分析。"
                ),
                "claims": [],
                "uncertainties": uncertainties
                or ["图像的图形、空间关系与含义未核验"],
                "conflicts": [],
            },
            "decision": "abstain",
        }


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise VisualSemanticError(
            "multimodal provider result is not canonical JSON"
        ) from exc


def _sha_text(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _number(value: Any, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
    ):
        raise VisualSemanticError(f"{field} is invalid")
    return round(float(value), 6)


def _require_exact_keys(
    value: Mapping[str, Any], *, allowed: set[str], field: str
) -> None:
    extras = set(value) - allowed
    if extras:
        raise VisualSemanticError(
            f"{field} contains unsupported fields: {','.join(sorted(extras))}"
        )


def _normalized_provider_spec(provider: object) -> dict[str, Any]:
    raw_spec = getattr(provider, "provider_spec", None)
    if callable(raw_spec):
        raw_spec = raw_spec()
    compatibility_adapter = raw_spec is None
    if raw_spec is None:
        provider_id = str(getattr(provider, "provider_id", ""))
        region = str(getattr(provider, "processing_region", ""))
        remote = getattr(provider, "sends_raw_media_remotely", None)
        raw_spec = {
            "schema": MULTIMODAL_PROVIDER_SPEC_SCHEMA,
            "provider_id": provider_id,
            "adapter_version": "legacy-visual-v1",
            "execution_scope": "remote" if remote is True else "local",
            "processing_region": region,
            "raw_media_transport": "remote" if remote is True else "none",
            "provider_retention_days": None if remote is True else 0,
            "supported_source_modalities": ["image"],
            "supported_mime_types": ["image/jpeg", "image/png", "image/webp"],
            "capabilities": ["semantic_analysis"],
            "deterministic": False,
        }
    if not isinstance(raw_spec, Mapping):
        raise VisualSemanticError("multimodal provider spec is invalid")
    spec = deepcopy(dict(raw_spec))
    _require_exact_keys(
        spec,
        allowed={
            "schema",
            "provider_id",
            "adapter_version",
            "execution_scope",
            "processing_region",
            "raw_media_transport",
            "provider_retention_days",
            "supported_source_modalities",
            "supported_mime_types",
            "capabilities",
            "deterministic",
            "compatibility_adapter",
        },
        field="multimodal provider spec",
    )
    if spec.get("schema") != MULTIMODAL_PROVIDER_SPEC_SCHEMA:
        raise VisualSemanticError("multimodal provider spec schema is invalid")
    provider_id = spec.get("provider_id")
    adapter_version = spec.get("adapter_version")
    execution_scope = spec.get("execution_scope")
    region = spec.get("processing_region")
    transport = spec.get("raw_media_transport")
    retention = spec.get("provider_retention_days")
    source_modalities = spec.get("supported_source_modalities")
    mime_types = spec.get("supported_mime_types")
    capabilities = spec.get("capabilities")
    deterministic = spec.get("deterministic")
    if "compatibility_adapter" in spec and spec["compatibility_adapter"] is not False:
        raise VisualSemanticError(
            "explicit multimodal provider spec cannot claim compatibility mode"
        )
    if not isinstance(provider_id, str) or not _PROVIDER_ID.fullmatch(provider_id):
        raise VisualSemanticError("multimodal provider identity is invalid")
    if not isinstance(adapter_version, str) or not _VERSION.fullmatch(adapter_version):
        raise VisualSemanticError("multimodal provider adapter version is invalid")
    if execution_scope not in {"local", "remote"}:
        raise VisualSemanticError("multimodal provider execution scope is invalid")
    if not isinstance(region, str) or not _REGION.fullmatch(region):
        raise VisualSemanticError("multimodal provider processing region is invalid")
    if transport not in {"none", "remote"} or (transport == "remote") != (
        execution_scope == "remote"
    ):
        raise VisualSemanticError("multimodal provider raw-media policy is invalid")
    if (
        (retention is not None and (
            isinstance(retention, bool)
            or not isinstance(retention, int)
            or not 0 <= retention <= 365
        ))
        or (execution_scope == "local" and retention != 0)
        or (
            execution_scope == "remote"
            and retention is None
            and not compatibility_adapter
        )
    ):
        raise VisualSemanticError("multimodal provider retention policy is invalid")
    if (
        not isinstance(source_modalities, list)
        or not source_modalities
        or source_modalities != sorted(set(source_modalities))
        or any(item not in _SOURCE_MODALITIES for item in source_modalities)
    ):
        raise VisualSemanticError("multimodal provider source modalities are invalid")
    if (
        not isinstance(mime_types, list)
        or not mime_types
        or mime_types != sorted(set(mime_types))
        or any(
            not isinstance(item, str)
            or _MIME.fullmatch(item) is None
            or item not in SUPPORTED_MULTIMODAL_MIME_TYPES
            for item in mime_types
        )
    ):
        raise VisualSemanticError("multimodal provider MIME types are invalid")
    if (
        not isinstance(capabilities, list)
        or not capabilities
        or capabilities != sorted(set(capabilities))
        or any(item not in _CAPABILITIES for item in capabilities)
    ):
        raise VisualSemanticError("multimodal provider capabilities are invalid")
    if not isinstance(deterministic, bool):
        raise VisualSemanticError("multimodal provider determinism is invalid")
    if str(getattr(provider, "provider_id", provider_id)) != provider_id:
        raise VisualSemanticError("multimodal provider identity contradicts its spec")
    legacy_region = getattr(provider, "processing_region", region)
    if str(legacy_region) != region:
        raise VisualSemanticError("multimodal provider region contradicts its spec")
    legacy_remote = getattr(
        provider, "sends_raw_media_remotely", execution_scope == "remote"
    )
    if not isinstance(legacy_remote, bool) or legacy_remote != (
        execution_scope == "remote"
    ):
        raise VisualSemanticError("multimodal provider transport contradicts its spec")
    normalized = {
        "schema": MULTIMODAL_PROVIDER_SPEC_SCHEMA,
        "provider_id": provider_id,
        "adapter_version": adapter_version,
        "execution_scope": execution_scope,
        "processing_region": region,
        "raw_media_transport": transport,
        "provider_retention_days": retention,
        "supported_source_modalities": list(source_modalities),
        "supported_mime_types": list(mime_types),
        "capabilities": list(capabilities),
        "deterministic": deterministic,
        "compatibility_adapter": compatibility_adapter,
    }
    _canonical(normalized)
    return normalized


def multimodal_provider_spec(provider: object) -> dict[str, Any]:
    """Return the canonical, versioned public spec for one provider."""

    return deepcopy(_normalized_provider_spec(provider))


def _validated_claims(raw_claims: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_claims, list) or len(raw_claims) > MAX_VISUAL_CLAIMS:
        raise VisualSemanticError("multimodal provider claims are invalid")
    claims: list[dict[str, Any]] = []
    for index, claim in enumerate(raw_claims):
        if not isinstance(claim, Mapping):
            raise VisualSemanticError(f"multimodal provider claim {index} is invalid")
        _require_exact_keys(
            claim,
            allowed={"statement", "evidence_locator", "confidence"},
            field=f"multimodal provider claim {index}",
        )
        statement = claim.get("statement")
        locator = claim.get("evidence_locator")
        if (
            not isinstance(statement, str)
            or not statement.strip()
            or len(statement) > 500
            or not isinstance(locator, str)
            or not locator.strip()
            or len(locator) > 240
        ):
            raise VisualSemanticError(f"multimodal provider claim {index} is invalid")
        claims.append(
            {
                "claim_id": f"multimodal_claim_{index + 1:02d}",
                "statement": statement.strip(),
                "evidence_locator": locator.strip(),
                "confidence": _number(
                    claim.get("confidence"),
                    field=f"multimodal provider claim {index} confidence",
                ),
                "confidence_semantics": (
                    "provider_observation_likelihood_not_correctness_or_mastery"
                ),
                "verification_status": "unverified_provider_observation",
                "untrusted_instruction_data": True,
                "grading_evidence_allowed": False,
                "mastery_evidence_allowed": False,
            }
        )
    return claims


def _validated_strings(raw: Any, *, field: str) -> list[str]:
    if (
        not isinstance(raw, list)
        or len(raw) > MAX_VISUAL_CLAIMS
        or any(
            not isinstance(item, str) or not item.strip() or len(item) > 500
            for item in raw
        )
    ):
        raise VisualSemanticError(f"multimodal provider {field} are invalid")
    return [str(item).strip() for item in raw]


def _validated_conflicts(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or len(raw) > MAX_VISUAL_CLAIMS:
        raise VisualSemanticError("multimodal provider conflicts are invalid")
    conflicts: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise VisualSemanticError(
                f"multimodal provider conflict {index} is invalid"
            )
        _require_exact_keys(
            item,
            allowed={"kind", "description", "evidence_locators"},
            field=f"multimodal provider conflict {index}",
        )
        kind = item.get("kind")
        description = item.get("description")
        locators = item.get("evidence_locators")
        if (
            kind not in {
                "transcription_disagreement",
                "slide_notes_disagreement",
                "cross_layer_disagreement",
                "ambiguous_visual_relation",
            }
            or not isinstance(description, str)
            or not description.strip()
            or len(description) > 500
            or not isinstance(locators, list)
            or not 2 <= len(locators) <= 8
            or any(
                not isinstance(locator, str)
                or not locator.strip()
                or len(locator) > 240
                for locator in locators
            )
        ):
            raise VisualSemanticError(
                f"multimodal provider conflict {index} is invalid"
            )
        conflicts.append(
            {
                "conflict_id": f"multimodal_conflict_{index + 1:02d}",
                "kind": kind,
                "description": description.strip(),
                "evidence_locators": [str(value).strip() for value in locators],
                "resolution_status": "unresolved",
            }
        )
    return conflicts


def _validated_transcription(raw: Any, *, source_modality: str) -> dict[str, Any]:
    if raw is None:
        raw = {
            "status": "not_performed",
            "text": "",
            "confidence": 0.0,
            "language": "und",
            "segments": [],
        }
    if not isinstance(raw, Mapping):
        raise VisualSemanticError("multimodal transcription is invalid")
    _require_exact_keys(
        raw,
        allowed={"status", "text", "confidence", "language", "segments"},
        field="multimodal transcription",
    )
    status = raw.get("status")
    text = raw.get("text")
    confidence = raw.get("confidence")
    language = raw.get("language")
    segments = raw.get("segments")
    if status not in {"not_performed", "unavailable", "candidate", "conflict"}:
        raise VisualSemanticError("multimodal transcription status is invalid")
    if (
        not isinstance(text, str)
        or len(text) > MAX_TRANSCRIPT_CHARS
        or (status in {"candidate", "conflict"} and not text.strip())
        or (status in {"not_performed", "unavailable"} and text)
    ):
        raise VisualSemanticError("multimodal transcription text is invalid")
    bounded_confidence = _number(confidence, field="multimodal transcription confidence")
    if status in {"not_performed", "unavailable"} and bounded_confidence != 0:
        raise VisualSemanticError(
            "unavailable multimodal transcription cannot carry confidence"
        )
    if not isinstance(language, str) or not re.fullmatch(r"[A-Za-z0-9-]{2,35}", language):
        raise VisualSemanticError("multimodal transcription language is invalid")
    if (
        not isinstance(segments, list)
        or len(segments) > MAX_TRANSCRIPT_SEGMENTS
    ):
        raise VisualSemanticError("multimodal transcript segments are invalid")
    temporal = source_modality in {"audio", "video"}
    normalized_segments: list[dict[str, Any]] = []
    previous_start = -1
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise VisualSemanticError(
                f"multimodal transcript segment {index} is invalid"
            )
        _require_exact_keys(
            segment,
            allowed={
                "start_ms",
                "end_ms",
                "text",
                "evidence_locator",
                "confidence",
            },
            field=f"multimodal transcript segment {index}",
        )
        segment_text = segment.get("text")
        locator = segment.get("evidence_locator")
        start_ms = segment.get("start_ms")
        end_ms = segment.get("end_ms")
        if (
            not isinstance(segment_text, str)
            or not segment_text.strip()
            or len(segment_text) > 1_000
            or not isinstance(locator, str)
            or not locator.strip()
            or len(locator) > 240
            or isinstance(start_ms, bool)
            or not isinstance(start_ms, int)
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, int)
            or start_ms < 0
            or end_ms <= start_ms
            or start_ms < previous_start
        ):
            raise VisualSemanticError(
                f"multimodal transcript segment {index} is invalid"
            )
        if not temporal:
            raise VisualSemanticError(
                "timestamps are only valid for audio/video transcription"
            )
        previous_start = start_ms
        normalized_segments.append(
            {
                "segment_id": f"segment_{index + 1:04d}",
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": segment_text.strip(),
                "evidence_locator": locator.strip(),
                "confidence": _number(
                    segment.get("confidence"),
                    field=f"multimodal transcript segment {index} confidence",
                ),
                "grading_evidence_allowed": False,
                "mastery_evidence_allowed": False,
            }
        )
    if temporal and status == "candidate" and not normalized_segments:
        raise VisualSemanticError(
            "audio/video transcription requires timestamped provenance"
        )
    if temporal and normalized_segments:
        normalized_full = re.sub(r"\s+", "", text).casefold()
        normalized_parts = re.sub(
            r"\s+",
            "",
            "".join(segment["text"] for segment in normalized_segments),
        ).casefold()
        if normalized_full != normalized_parts:
            raise VisualSemanticError(
                "audio/video transcript text diverges from timestamped segments"
            )
    if not temporal and normalized_segments:
        raise VisualSemanticError("visual transcription cannot contain timestamps")
    return {
        "status": status,
        "text": text.strip(),
        "confidence": bounded_confidence,
        "confidence_semantics": (
            "provider_transcription_likelihood_not_semantic_truth_or_correctness"
        ),
        "language": language,
        "segments": normalized_segments,
        "student_or_teacher_confirmed": False,
        "grading_evidence_allowed": False,
        "mastery_evidence_allowed": False,
    }


def _convert_v1_result(raw: Mapping[str, Any], *, source_modality: str) -> dict[str, Any]:
    _require_exact_keys(
        raw,
        allowed={"schema", "modality", "description", "claims", "uncertainties"},
        field="visual provider result",
    )
    modality = raw.get("modality")
    description = raw.get("description")
    claims = raw.get("claims")
    uncertainties = raw.get("uncertainties")
    if modality not in _SEMANTIC_MODALITIES:
        raise VisualSemanticError("visual provider modality is invalid")
    if (
        not isinstance(description, str)
        or not description.strip()
        or len(description) > MAX_VISUAL_DESCRIPTION_CHARS
    ):
        raise VisualSemanticError("visual provider description is invalid")
    # Full validation occurs after conversion.  An uncertainty in a legacy
    # result is an abstention, never an implicit invitation to keep using it.
    return {
        "schema": MULTIMODAL_PROVIDER_RESULT_SCHEMA,
        "source_modality": source_modality,
        "modality": modality,
        "transcription": None,
        "semantic_analysis": {
            "status": "provider_observation",
            "description": description,
            "claims": claims,
            "uncertainties": uncertainties,
            "conflicts": [],
        },
        "decision": "abstain" if uncertainties else "observation",
    }


def _validated_result(raw: Any, *, source_modality: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise VisualSemanticError("multimodal provider result is invalid")
    if raw.get("schema") == VISUAL_PROVIDER_RESULT_SCHEMA:
        raw = _convert_v1_result(raw, source_modality=source_modality)
    if raw.get("schema") != MULTIMODAL_PROVIDER_RESULT_SCHEMA:
        raise VisualSemanticError("multimodal provider result schema is invalid")
    _require_exact_keys(
        raw,
        allowed={
            "schema",
            "source_modality",
            "modality",
            "transcription",
            "semantic_analysis",
            "decision",
        },
        field="multimodal provider result",
    )
    if raw.get("source_modality") != source_modality:
        raise VisualSemanticError("multimodal provider source modality is invalid")
    modality = raw.get("modality")
    if modality not in _SEMANTIC_MODALITIES:
        raise VisualSemanticError("multimodal provider modality is invalid")
    transcription = _validated_transcription(
        raw.get("transcription"), source_modality=source_modality
    )
    semantic = raw.get("semantic_analysis")
    if not isinstance(semantic, Mapping):
        raise VisualSemanticError("multimodal semantic analysis is invalid")
    _require_exact_keys(
        semantic,
        allowed={
            "status",
            "description",
            "claims",
            "uncertainties",
            "conflicts",
        },
        field="multimodal semantic analysis",
    )
    status = semantic.get("status")
    description = semantic.get("description")
    if status not in {
        "not_performed",
        "provider_observation",
        "conflict",
        "abstained",
    }:
        raise VisualSemanticError("multimodal semantic analysis status is invalid")
    if (
        not isinstance(description, str)
        or not description.strip()
        or len(description) > MAX_VISUAL_DESCRIPTION_CHARS
    ):
        raise VisualSemanticError("multimodal semantic description is invalid")
    claims = _validated_claims(semantic.get("claims"))
    uncertainties = _validated_strings(
        semantic.get("uncertainties"), field="uncertainties"
    )
    conflicts = _validated_conflicts(semantic.get("conflicts"))
    decision = raw.get("decision")
    if decision not in _DECISIONS:
        raise VisualSemanticError("multimodal provider decision is invalid")
    if conflicts and decision not in {"requires_confirmation", "abstain"}:
        raise VisualSemanticError("multimodal conflict requires confirmation or abstention")
    if uncertainties and decision not in {"requires_confirmation", "abstain"}:
        raise VisualSemanticError(
            "multimodal uncertainty requires confirmation or abstention"
        )
    if status in {"conflict", "abstained"} and decision == "observation":
        raise VisualSemanticError("multimodal provider decision contradicts status")
    if status == "not_performed" and claims:
        raise VisualSemanticError(
            "missing semantic analysis cannot carry semantic claims"
        )
    if status == "conflict" and not conflicts:
        raise VisualSemanticError(
            "conflicting semantic analysis requires conflict provenance"
        )
    if status == "abstained" and decision != "abstain":
        raise VisualSemanticError(
            "abstained semantic analysis requires an abstain decision"
        )
    if transcription["status"] == "conflict" and decision == "observation":
        raise VisualSemanticError(
            "transcription conflict requires confirmation or abstention"
        )
    if status == "not_performed" and decision != "abstain":
        raise VisualSemanticError(
            "missing semantic analysis requires explicit abstention"
        )
    if decision == "requires_confirmation" and not (
        conflicts or uncertainties or transcription["status"] == "conflict"
    ):
        raise VisualSemanticError(
            "multimodal confirmation decision requires a recorded uncertainty or conflict"
        )
    return {
        "schema": MULTIMODAL_PROVIDER_RESULT_SCHEMA,
        "source_modality": source_modality,
        "modality": modality,
        "transcription": transcription,
        "semantic_analysis": {
            "status": status,
            "description": description.strip(),
            "claims": claims,
            "uncertainties": uncertainties,
            "conflicts": conflicts,
            "visual_verification_status": "unverified_provider_observation",
            "grading_evidence_allowed": False,
            "mastery_evidence_allowed": False,
        },
        "decision": decision,
    }


def _source_modality_for_mime(mime_type: str) -> str:
    if mime_type.startswith("image/"):
        return "image"
    if mime_type == "application/pdf":
        return "scanned_pdf_page"
    if mime_type.startswith("audio/"):
        return "audio"
    if mime_type.startswith("video/"):
        return "video"
    raise VisualSemanticError("multimodal MIME type is unsupported")


def _validate_media(media_bytes: bytes, mime_type: str) -> tuple[str, str]:
    if (
        not isinstance(media_bytes, bytes)
        or not 1 <= len(media_bytes) <= MAX_MULTIMODAL_MEDIA_BYTES
    ):
        raise VisualSemanticError("multimodal media size is invalid")
    normalized_mime = str(mime_type or "").split(";", 1)[0].strip().lower()
    if normalized_mime not in SUPPORTED_MULTIMODAL_MIME_TYPES:
        raise VisualSemanticError("multimodal MIME type is unsupported")
    source_modality = _source_modality_for_mime(normalized_mime)
    if source_modality == "image":
        normalized_mime = validate_local_image(media_bytes, normalized_mime)
    elif source_modality == "scanned_pdf_page" and not media_bytes.startswith(b"%PDF-"):
        raise VisualSemanticError("multimodal PDF signature is invalid")
    elif normalized_mime in {"audio/wav", "audio/x-wav"} and not (
        media_bytes.startswith(b"RIFF") and media_bytes[8:12] == b"WAVE"
    ):
        raise VisualSemanticError("multimodal WAV signature is invalid")
    elif normalized_mime in {"audio/webm", "video/webm"} and not media_bytes.startswith(
        b"\x1aE\xdf\xa3"
    ):
        raise VisualSemanticError("multimodal WebM signature is invalid")
    elif normalized_mime in {"audio/mp4", "audio/x-m4a", "video/mp4"} and (
        len(media_bytes) < 12 or media_bytes[4:8] != b"ftyp"
    ):
        raise VisualSemanticError("multimodal MP4 signature is invalid")
    elif normalized_mime == "video/quicktime" and (
        len(media_bytes) < 12 or media_bytes[4:8] != b"ftyp"
    ):
        raise VisualSemanticError("multimodal QuickTime signature is invalid")
    elif normalized_mime == "audio/mpeg" and not (
        media_bytes.startswith(b"ID3")
        or (len(media_bytes) >= 2 and media_bytes[0] == 0xFF and media_bytes[1] & 0xE0 == 0xE0)
    ):
        raise VisualSemanticError("multimodal MP3 signature is invalid")
    return normalized_mime, source_modality


def analyze_multimodal_semantics(
    media_bytes: bytes,
    mime_type: str,
    *,
    task_context: str,
    provider: MultimodalProvider | VisualSemanticProvider,
    subject_id: str | None = None,
    consent_id: str | None = None,
    consent_store: RemoteConsentStore | None = None,
    source_modality: str | None = None,
) -> dict[str, Any]:
    """Analyze bounded media while preserving consent and assessment boundaries."""

    detected_mime, inferred_source_modality = _validate_media(media_bytes, mime_type)
    if source_modality is None:
        source_modality = inferred_source_modality
    elif source_modality not in _SOURCE_MODALITIES:
        raise VisualSemanticError("multimodal source modality is invalid")
    elif source_modality == "slide" and not detected_mime.startswith("image/"):
        raise VisualSemanticError("slide analysis requires a rendered slide image")
    elif source_modality != "slide" and source_modality != inferred_source_modality:
        raise VisualSemanticError("multimodal source modality contradicts MIME type")
    context = re.sub(r"\s+", " ", str(task_context)).strip()
    if not context or len(context) > MAX_VISUAL_CONTEXT_CHARS:
        raise VisualSemanticError("multimodal task context is empty or too long")
    spec = _normalized_provider_spec(provider)
    if source_modality not in spec["supported_source_modalities"]:
        raise VisualSemanticError("multimodal provider does not support this modality")
    if detected_mime not in spec["supported_mime_types"]:
        raise VisualSemanticError("multimodal provider does not support this MIME type")

    remote = spec["execution_scope"] == "remote"
    consent_receipt: Mapping[str, Any] | None = None
    if remote:
        # The existing consent registry intentionally has no learner_audio or
        # learner_video category.  Do not silently stretch learner_image.
        if source_modality != "image":
            raise VisualSemanticError(
                "remote audio/video/PDF processing has no authorized consent category"
            )
        if (
            consent_store is None
            or not isinstance(subject_id, str)
            or not subject_id
            or not isinstance(consent_id, str)
            or not consent_id
        ):
            raise VisualSemanticError(
                "remote visual analysis requires a server consent receipt"
            )
        try:
            consent_receipt = consent_store.verify(
                consent_id,
                subject_id=subject_id,
                purpose="remote_visual_analysis",
                provider_id=spec["provider_id"],
                required_data_categories=["learner_image"],
            )
        except ValueError as exc:
            raise VisualSemanticError(
                "remote visual analysis consent is invalid"
            ) from exc
        if consent_receipt.get("processing_region") != spec["processing_region"]:
            raise VisualSemanticError("remote visual consent region is invalid")
        declared_retention = spec["provider_retention_days"]
        if declared_retention is not None and consent_receipt.get(
            "provider_retention_days"
        ) != declared_retention:
            raise VisualSemanticError("remote visual consent retention is invalid")
    elif consent_id is not None:
        raise VisualSemanticError("local multimodal analysis does not accept consent")

    try:
        raw_result = provider.analyze(
            media_bytes, detected_mime, task_context=context
        )
    except Exception as exc:  # noqa: BLE001 - third-party adapters fail closed.
        raise VisualSemanticError("multimodal provider failed") from exc
    result = _validated_result(raw_result, source_modality=source_modality)
    transcription_status = result["transcription"]["status"]
    semantic_status = result["semantic_analysis"]["status"]
    if transcription_status in {"candidate", "conflict"} and not (
        {"transcription", "temporal_transcription"} & set(spec["capabilities"])
    ):
        raise VisualSemanticError(
            "multimodal result exceeds provider transcription capability"
        )
    if semantic_status in {"provider_observation", "conflict"} and (
        "semantic_analysis" not in spec["capabilities"]
    ):
        raise VisualSemanticError(
            "multimodal result exceeds provider semantic capability"
        )
    result_hash = _sha_text(result)
    media_hash = hashlib.sha256(media_bytes).hexdigest()
    uncertainties = result["semantic_analysis"]["uncertainties"]
    conflicts = result["semantic_analysis"]["conflicts"]
    evidence = {
        "schema": MULTIMODAL_EVIDENCE_SCHEMA,
        "source_modality": source_modality,
        "media_sha256": media_hash,
        # Compatibility for persisted visual records.
        "image_sha256": media_hash if source_modality == "image" else None,
        "mime_type": detected_mime,
        "provider": spec,
        "provider_spec_sha256": _sha_text(spec),
        "provider_id": spec["provider_id"],
        "processing_region": spec["processing_region"],
        "remote_media_sent": remote,
        "raw_media_retained": False,
        "task_context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
        "provider_result_sha256": result_hash,
        "modality": result["modality"],
        "transcription": result["transcription"],
        "semantic_analysis": result["semantic_analysis"],
        # Compatibility projection for v1 visual consumers.
        "description": result["semantic_analysis"]["description"],
        "claims": result["semantic_analysis"]["claims"],
        "uncertainties": uncertainties,
        "conflicts": conflicts,
        "decision": result["decision"],
        "semantic_understanding_status": (
            "provider_observation_requires_review"
            if result["semantic_analysis"]["status"] == "provider_observation"
            else result["semantic_analysis"]["status"]
        ),
        "visual_verification_status": "not_verified",
        "provider_confidence_is_answer_correctness": False,
        "transcription_is_semantic_understanding": False,
        "semantic_analysis_is_answer_correctness": False,
        "grading_evidence_allowed": False,
        "mastery_evidence_allowed": False,
        "requires_teacher_review": True,
        "requires_learner_confirmation": result["decision"]
        == "requires_confirmation",
        "abstained": result["decision"] == "abstain",
        "untrusted_instruction_data": True,
        "privacy_receipt": {
            "execution_scope": spec["execution_scope"],
            "raw_media_sent": remote,
            "processing_region": spec["processing_region"],
            "provider_retention_days": (
                consent_receipt.get("provider_retention_days")
                if isinstance(consent_receipt, Mapping)
                else 0
            ),
            "consent_id": (
                consent_receipt.get("consent_id")
                if isinstance(consent_receipt, Mapping)
                else None
            ),
            "consent_receipt_sha256": (
                consent_receipt.get("receipt_sha256")
                if isinstance(consent_receipt, Mapping)
                else None
            ),
            "consent_policy_version": (
                consent_receipt.get("policy_version")
                if isinstance(consent_receipt, Mapping)
                else None
            ),
            "data_categories": (
                list(consent_receipt.get("data_categories", []))
                if isinstance(consent_receipt, Mapping)
                else []
            ),
        },
        "consent_receipt_sha256": (
            consent_receipt.get("receipt_sha256")
            if isinstance(consent_receipt, Mapping)
            else None
        ),
    }
    _canonical(evidence)
    return deepcopy(evidence)


def analyze_visual_semantics(
    image_bytes: bytes,
    mime_type: str,
    *,
    task_context: str,
    provider: VisualSemanticProvider,
    subject_id: str | None = None,
    consent_id: str | None = None,
    consent_store: RemoteConsentStore | None = None,
) -> dict[str, Any]:
    """Compatibility image API backed by the general multimodal contract."""

    evidence = analyze_multimodal_semantics(
        image_bytes,
        mime_type,
        task_context=task_context,
        provider=provider,
        subject_id=subject_id,
        consent_id=consent_id,
        consent_store=consent_store,
    )
    evidence["schema"] = VISUAL_SEMANTIC_EVIDENCE_SCHEMA
    evidence["multimodal_contract_schema"] = MULTIMODAL_EVIDENCE_SCHEMA
    return evidence


def analyze_temporal_media(
    media_bytes: bytes,
    mime_type: str,
    *,
    task_context: str,
    provider: TemporalTranscriptionProvider | None,
) -> dict[str, Any]:
    """Run a local timestamped audio/video adapter or refuse honestly."""

    if provider is None:
        raise VisualSemanticError(
            "local audio/video transcription provider is not configured"
        )
    spec = _normalized_provider_spec(provider)
    if spec["execution_scope"] != "local":
        raise VisualSemanticError(
            "audio/video transcription must remain local under the current consent policy"
        )
    if "temporal_transcription" not in spec["capabilities"]:
        raise VisualSemanticError(
            "audio/video provider lacks temporal transcription capability"
        )
    evidence = analyze_multimodal_semantics(
        media_bytes,
        mime_type,
        task_context=task_context,
        provider=provider,
    )
    if evidence["source_modality"] not in {"audio", "video"}:
        raise VisualSemanticError("temporal media must be audio or video")
    transcription = evidence["transcription"]
    if transcription["status"] != "candidate" or not transcription["segments"]:
        raise VisualSemanticError(
            "audio/video provider did not return timestamped transcription"
        )
    return evidence


__all__ = [
    "MAX_MULTIMODAL_MEDIA_BYTES",
    "MAX_TRANSCRIPT_CHARS",
    "MAX_TRANSCRIPT_SEGMENTS",
    "MAX_VISUAL_CLAIMS",
    "MAX_VISUAL_CONTEXT_CHARS",
    "LocalOCRVisualSemanticProvider",
    "MULTIMODAL_EVIDENCE_SCHEMA",
    "MULTIMODAL_PROVIDER_RESULT_SCHEMA",
    "MULTIMODAL_PROVIDER_SPEC_SCHEMA",
    "MultimodalProvider",
    "SUPPORTED_MULTIMODAL_MIME_TYPES",
    "TemporalTranscriptionProvider",
    "VISUAL_PROVIDER_RESULT_SCHEMA",
    "VISUAL_SEMANTIC_EVIDENCE_SCHEMA",
    "VisualSemanticError",
    "VisualSemanticProvider",
    "analyze_multimodal_semantics",
    "analyze_temporal_media",
    "analyze_visual_semantics",
    "multimodal_provider_spec",
]
