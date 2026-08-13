"""Curriculum dependency graph and measurement-blueprint core.

The teaching-syllabus generator intentionally produces planning text, not a
grading authority.  This module turns that outline into an auditable graph and
keeps the two trust domains explicit:

* model-generated or legacy outlines are always ``generated_unvalidated``;
* a teacher-owned specification becomes authoritative only through a separate,
  content-bound authority receipt.

The graph is independent from the dashboard and live lesson loop.  Consumers
may therefore validate/import curricula without starting a learner session.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import base64
from hashlib import sha256
import json
from pathlib import Path
import re
import threading
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

from .io_utils import ensure_private_directory, read_json, write_json
from .student_model import stable_knowledge_component_id


CURRICULUM_BLUEPRINT_SCHEMA = "teaching_skill_miner.curriculum_blueprint.v1"
TEACHER_AUTHORITY_RECEIPT_SCHEMA = (
    "teaching_skill_miner.teacher_curriculum_authority_receipt.v2"
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CURRICULUM_ID = re.compile(r"^cur_[0-9a-f]{24}$")
_OBJECTIVE_ID = re.compile(r"^obj_[0-9a-f]{20}$")
_LESSON_ID = re.compile(r"^lsn_[0-9a-f]{20}$")
_KC_ID = re.compile(r"^kc_[a-z0-9][a-z0-9_-]{2,80}$")
_ENTITY_ID = re.compile(r"^(?:edge|span|claim|rubric|item|remedy|review)_[0-9a-f]{20}$")
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
_SAFE_RESOURCE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,159}$")
_SIGNING_KEY_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,119}$")
_MAX_BLUEPRINT_BYTES = 8 * 1024 * 1024

_MEASURABLE_CJK = re.compile(
    r"定义|解释|说明|描述|列出|识别|辨认|区分|比较|计算|求解|推导|证明|"
    r"构建|编写|写出|给出|表达|实现|操作|应用|使用|判断|分析|设计|翻译|复述|发音|造句|分类|"
    r"预测|标注|演示|完成|迁移|纠正|改写|概括|总结|选择|排序|检验"
)
_MEASURABLE_LATIN = re.compile(
    r"\b(?:define|explain|describe|list|identify|distinguish|compare|calculate|"
    r"compute|solve|derive|prove|construct|build|implement|apply|use|"
    r"determine|analy[sz]e|design|translate|classify|predict|label|"
    r"demonstrate|produce|write|speak|pronounce|revise|summari[sz]e|"
    r"select|sequence|test)\b",
    re.IGNORECASE,
)
_UNMEASURABLE_ONLY = re.compile(
    r"^(?:能够?|学生(?:将|能|能够)?|学习者(?:将|能|能够)?|to\s+)?\s*"
    r"(?:理解|了解|掌握|熟悉|认识|知道|体会|欣赏|"
    r"understand|know|learn|appreciate|be\s+aware\s+of|be\s+familiar\s+with)\b",
    re.IGNORECASE,
)


class CurriculumBlueprintError(ValueError):
    """Raised when a curriculum graph or authority receipt fails closed."""


class TeachingCurriculumBlueprintStore:
    """Private atomic storage for independently sealed curriculum graphs."""

    def __init__(
        self,
        root: str | Path,
        *,
        trusted_teacher_public_keys: Mapping[str, bytes | str] | None = None,
    ) -> None:
        self.root = ensure_private_directory(Path(root).expanduser().resolve())
        self._lock = threading.Lock()
        self._trusted_teacher_public_keys = dict(trusted_teacher_public_keys or {})

    def _path(self, curriculum_id: str) -> Path:
        if not isinstance(curriculum_id, str) or not _CURRICULUM_ID.fullmatch(
            curriculum_id
        ):
            raise CurriculumBlueprintError("curriculum_id is invalid")
        return self.root / f"{curriculum_id}.json"

    def save(self, blueprint: Mapping[str, Any]) -> bool:
        validate_curriculum_blueprint(
            blueprint,
            trusted_teacher_public_keys=self._trusted_teacher_public_keys,
        )
        path = self._path(str(blueprint["curriculum_id"]))
        with self._lock:
            created = not path.exists()
            write_json(path, deepcopy(dict(blueprint)))
        return created

    def read(self, curriculum_id: str) -> dict[str, Any]:
        path = self._path(curriculum_id)
        with self._lock:
            try:
                value = read_json(path)
            except FileNotFoundError as exc:
                raise CurriculumBlueprintError("curriculum_id was not found") from exc
        validate_curriculum_blueprint(
            value,
            trusted_teacher_public_keys=self._trusted_teacher_public_keys,
        )
        return deepcopy(dict(value))

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            paths = sorted(self.root.glob("cur_*.json"))
            try:
                values = [read_json(path) for path in paths]
            except (OSError, json.JSONDecodeError) as exc:
                raise CurriculumBlueprintError(
                    "stored curriculum blueprint cannot be read"
                ) from exc
        for value in values:
            validate_curriculum_blueprint(
                value,
                trusted_teacher_public_keys=self._trusted_teacher_public_keys,
            )
        return sorted(
            (deepcopy(dict(value)) for value in values),
            key=lambda row: row["curriculum_id"],
        )


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CurriculumBlueprintError("curriculum must be canonical JSON") from exc


def _digest(prefix: str, value: Any, *, length: int = 20) -> str:
    return f"{prefix}_{sha256(_canonical_json(value)).hexdigest()[:length]}"


def _canonical_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _text(value: Any, field: str, *, maximum: int, minimum: int = 1) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not minimum <= len(value) <= maximum
    ):
        raise CurriculumBlueprintError(
            f"{field} must be a trimmed string with length in [{minimum}, {maximum}]"
        )
    return value


def _integer(value: Any, field: str, *, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise CurriculumBlueprintError(
            f"{field} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise CurriculumBlueprintError(f"{field} must be a boolean")
    return value


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CurriculumBlueprintError(f"{field} must be an object")
    return value


def _array(
    value: Any, field: str, *, minimum: int = 0, maximum: int = 1_000
) -> list[Any]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise CurriculumBlueprintError(
            f"{field} must be an array with {minimum} to {maximum} items"
        )
    return value


def _strict_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise CurriculumBlueprintError(
            f"{field} has invalid fields ({'; '.join(details)})"
        )


def _string_ids(
    value: Any,
    field: str,
    *,
    pattern: re.Pattern[str],
    minimum: int = 0,
    maximum: int = 1_000,
) -> list[str]:
    values = _array(value, field, minimum=minimum, maximum=maximum)
    result: list[str] = []
    for index, item in enumerate(values):
        text = _text(item, f"{field}[{index}]", maximum=120)
        if not pattern.fullmatch(text):
            raise CurriculumBlueprintError(f"{field}[{index}] has an invalid ID")
        result.append(text)
    if len(result) != len(set(result)):
        raise CurriculumBlueprintError(f"{field} must not contain duplicates")
    return result


def _timestamp(value: Any, field: str) -> str:
    result = _text(value, field, maximum=27)
    if not _UTC_TIMESTAMP.fullmatch(result):
        raise CurriculumBlueprintError(f"{field} must be an RFC3339 UTC timestamp")
    try:
        datetime.fromisoformat(result[:-1] + "+00:00")
    except ValueError as exc:
        raise CurriculumBlueprintError(f"{field} is not a real timestamp") from exc
    return result


def _hash(value: Any, field: str) -> str:
    result = _text(value, field, maximum=64)
    if not _HEX64.fullmatch(result):
        raise CurriculumBlueprintError(f"{field} must be a lowercase SHA-256 hash")
    return result


def _index_by_id(
    rows: Sequence[Mapping[str, Any]], key: str, field: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        identifier = str(row[key])
        if identifier in result:
            raise CurriculumBlueprintError(f"{field}[{index}].{key} is duplicated")
        result[identifier] = row
    return result


def stable_curriculum_objective_id(statement: str) -> str:
    """Return a position-independent objective ID."""

    safe = _text(statement, "objective statement", maximum=600)
    return _digest("obj", {"statement": _canonical_text(safe)})


def stable_curriculum_lesson_id(
    title: str,
    *,
    objective_ids: Sequence[str],
    knowledge_component_ids: Sequence[str],
) -> str:
    """Return a lesson ID that is stable across reordering/insertion."""

    safe_title = _text(title, "lesson title", maximum=160)
    return _digest(
        "lsn",
        {
            "title": _canonical_text(safe_title),
            "objective_ids": sorted(set(objective_ids)),
            "knowledge_component_ids": sorted(set(knowledge_component_ids)),
        },
    )


def stable_curriculum_source_span_id(
    *,
    resource_id: str,
    content_sha256: str,
    excerpt_sha256: str,
    locator_kind: str,
    start: int,
    end: int,
) -> str:
    """Return the content-derived ID for one source location."""

    row = {
        "resource_id": resource_id,
        "content_sha256": content_sha256,
        "excerpt_sha256": excerpt_sha256,
        "locator": {"kind": locator_kind, "start": start, "end": end},
    }
    # Reuse the strict row validator by supplying the two derived fields after
    # the ID has been calculated.
    identifier = _source_span_id(row)
    checked = {**row, "source_span_id": identifier, "authority": True}
    _validate_source_span(checked, 0)
    return identifier


def _source_span_id(row: Mapping[str, Any]) -> str:
    material = {
        "resource_id": row["resource_id"],
        "content_sha256": row["content_sha256"],
        "excerpt_sha256": row["excerpt_sha256"],
        "locator": row["locator"],
    }
    return _digest("span", material)


def _edge_id(prerequisite_kc_id: str, dependent_kc_id: str) -> str:
    return _digest(
        "edge",
        {
            "prerequisite_kc_id": prerequisite_kc_id,
            "dependent_kc_id": dependent_kc_id,
        },
    )


def _claim_id(
    statement: str, kc_ids: Sequence[str], source_span_ids: Sequence[str]
) -> str:
    return _digest(
        "claim",
        {
            "statement": _canonical_text(statement),
            "kc_ids": sorted(kc_ids),
            "source_span_ids": sorted(source_span_ids),
        },
    )


def _rubric_id(
    objective_id: str,
    kc_id: str,
    description: str,
    source_span_ids: Sequence[str],
) -> str:
    return _digest(
        "rubric",
        {
            "objective_id": objective_id,
            "kc_id": kc_id,
            "description": _canonical_text(description),
            "source_span_ids": sorted(source_span_ids),
        },
    )


def _item_id(
    objective_id: str,
    kc_id: str,
    rubric_id: str,
    kind: str,
    prompt_intent: str,
    source_span_ids: Sequence[str],
) -> str:
    return _digest(
        "item",
        {
            "objective_id": objective_id,
            "kc_id": kc_id,
            "rubric_id": rubric_id,
            "kind": kind,
            "prompt_intent": _canonical_text(prompt_intent),
            "source_span_ids": sorted(source_span_ids),
        },
    )


def _remediation_id(
    objective_id: str, kc_id: str, trigger: str, lesson_id: str, action: str
) -> str:
    return _digest(
        "remedy",
        {
            "objective_id": objective_id,
            "kc_id": kc_id,
            "trigger": trigger,
            "lesson_id": lesson_id,
            "action": _canonical_text(action),
        },
    )


def _review_id(
    objective_id: str, kc_id: str, after_days: int, item_blueprint_id: str
) -> str:
    return _digest(
        "review",
        {
            "objective_id": objective_id,
            "kc_id": kc_id,
            "after_days": after_days,
            "item_blueprint_id": item_blueprint_id,
        },
    )


def _is_measurable(statement: str) -> bool:
    normalized = unicodedata.normalize("NFKC", statement).strip()
    if _MEASURABLE_CJK.search(normalized) or _MEASURABLE_LATIN.search(normalized):
        return True
    # Explicitly surface the common attack instead of treating length as
    # measurability.  The final ``False`` also rejects vague noun phrases.
    if _UNMEASURABLE_ONLY.search(normalized):
        return False
    return False


def _objective_lesson_alignment_score(
    statement: str,
    *,
    lesson_title: str,
    lesson_objective: str,
    kc_labels: Sequence[str],
) -> tuple[int, int]:
    """Return a deterministic lexical score for generated-outline projection.

    It selects coverage, not truth or grading authority.  The result is only a
    conservative structural mapping and remains ``generated_unvalidated``.
    """

    objective_text = _canonical_text(statement)
    lesson_text = _canonical_text(
        " ".join((lesson_title, lesson_objective, *kc_labels))
    )
    latin_objective = set(re.findall(r"[a-z][a-z0-9+#.-]*", objective_text))
    latin_lesson = set(re.findall(r"[a-z][a-z0-9+#.-]*", lesson_text))
    cjk_objective = set(re.findall(r"[\u3400-\u9fff]", objective_text))
    cjk_lesson = set(re.findall(r"[\u3400-\u9fff]", lesson_text))
    overlap = 4 * len(latin_objective & latin_lesson) + len(cjk_objective & cjk_lesson)
    exact_bonus = 1000 if objective_text and objective_text in lesson_text else 0
    return exact_bonus + overlap, -abs(len(objective_text) - len(lesson_text))


def _canonical_topological_order(
    kc_ids: Iterable[str],
    edges: Iterable[tuple[str, str]],
    lesson_rank: Mapping[str, int],
) -> list[str]:
    nodes = set(kc_ids)
    outgoing = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for prerequisite, dependent in edges:
        if prerequisite not in nodes or dependent not in nodes:
            raise CurriculumBlueprintError("prerequisite edge contains a dangling KC")
        if prerequisite == dependent:
            raise CurriculumBlueprintError("a KC cannot be its own prerequisite")
        if dependent not in outgoing[prerequisite]:
            outgoing[prerequisite].add(dependent)
            indegree[dependent] += 1
    ready = sorted(
        (node for node, count in indegree.items() if count == 0),
        key=lambda node: (lesson_rank[node], node),
    )
    result: list[str] = []
    while ready:
        node = ready.pop(0)
        result.append(node)
        for dependent in sorted(outgoing[node]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
                ready.sort(key=lambda item: (lesson_rank[item], item))
    if len(result) != len(nodes):
        raise CurriculumBlueprintError("prerequisite graph contains a cycle")
    return result


def _validate_authority_receipt(
    receipt: Any, *, expected_spec_sha256: str | None = None
) -> Mapping[str, Any]:
    row = _mapping(receipt, "authority.receipt")
    _strict_keys(
        row,
        {
            "schema",
            "receipt_id",
            "teacher_id_hash",
            "reviewed_at",
            "spec_sha256",
            "scope",
            "authority_assertion",
            "identity_assurance",
            "authoritative_for_runtime_grading",
            "signature_algorithm",
            "signing_key_id",
            "public_key_base64",
            "public_key_sha256",
            "signature_base64",
            "receipt_sha256",
        },
        "authority.receipt",
    )
    if row["schema"] != TEACHER_AUTHORITY_RECEIPT_SCHEMA:
        raise CurriculumBlueprintError("authority receipt schema is invalid")
    _hash(row["teacher_id_hash"], "authority.receipt.teacher_id_hash")
    _timestamp(row["reviewed_at"], "authority.receipt.reviewed_at")
    spec_hash = _hash(row["spec_sha256"], "authority.receipt.spec_sha256")
    if expected_spec_sha256 is not None and spec_hash != expected_spec_sha256:
        raise CurriculumBlueprintError("authority receipt does not bind this spec")
    if row["scope"] != "curriculum_graph_measurement_and_sources":
        raise CurriculumBlueprintError("authority receipt scope is invalid")
    if row["authority_assertion"] != "teacher_owned_spec_explicit_review":
        raise CurriculumBlueprintError("authority receipt assertion is invalid")
    if row["identity_assurance"] != "authenticated_teacher":
        raise CurriculumBlueprintError(
            "authority receipt identity assurance is invalid"
        )
    if row["authoritative_for_runtime_grading"] is not True:
        raise CurriculumBlueprintError(
            "authority receipt must explicitly authorize grading"
        )
    if row["signature_algorithm"] != "ed25519":
        raise CurriculumBlueprintError(
            "authority receipt signature algorithm is invalid"
        )
    signing_key_id = _text(
        row["signing_key_id"], "authority.receipt.signing_key_id", maximum=120
    )
    if not _SIGNING_KEY_ID.fullmatch(signing_key_id):
        raise CurriculumBlueprintError("authority receipt signing_key_id is invalid")
    public_key = _strict_base64(
        row["public_key_base64"],
        "authority.receipt.public_key_base64",
        expected_bytes=32,
    )
    public_key_hash = _hash(
        row["public_key_sha256"], "authority.receipt.public_key_sha256"
    )
    if public_key_hash != sha256(public_key).hexdigest():
        raise CurriculumBlueprintError("authority receipt public key hash mismatch")
    signature = _strict_base64(
        row["signature_base64"],
        "authority.receipt.signature_base64",
        expected_bytes=64,
    )
    unsigned = deepcopy(dict(row))
    for key in ("receipt_id", "receipt_sha256", "signature_base64"):
        unsigned.pop(key)
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature, _canonical_json(unsigned)
        )
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise CurriculumBlueprintError(
            "authority receipt signature is invalid"
        ) from exc
    material = deepcopy(dict(row))
    declared_hash = _hash(
        material.pop("receipt_sha256"), "authority.receipt.receipt_sha256"
    )
    declared_id = material.pop("receipt_id")
    calculated = sha256(_canonical_json(material)).hexdigest()
    if declared_hash != calculated or declared_id != f"receipt_{calculated[:20]}":
        raise CurriculumBlueprintError("authority receipt integrity mismatch")
    return row


def _strict_base64(value: Any, field: str, *, expected_bytes: int) -> bytes:
    text = _text(value, field, maximum=256)
    try:
        decoded = base64.b64decode(text.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise CurriculumBlueprintError(f"{field} is not canonical base64") from exc
    if (
        len(decoded) != expected_bytes
        or base64.b64encode(decoded).decode("ascii") != text
    ):
        raise CurriculumBlueprintError(f"{field} has an invalid length or encoding")
    return decoded


def _public_key_bytes(value: bytes | str) -> bytes:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    raw = value.encode("utf-8") if isinstance(value, str) else value
    try:
        if raw.startswith(b"-----BEGIN"):
            public_key = serialization.load_pem_public_key(raw)
            if not isinstance(public_key, Ed25519PublicKey):
                raise CurriculumBlueprintError("trusted teacher key must be Ed25519")
            return public_key.public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw,
            )
        if len(raw) == 32:
            Ed25519PublicKey.from_public_bytes(raw)
            return bytes(raw)
    except (TypeError, ValueError) as exc:
        raise CurriculumBlueprintError("trusted teacher public key is invalid") from exc
    raise CurriculumBlueprintError("trusted teacher public key is invalid")


def create_teacher_curriculum_authority_receipt(
    spec: Mapping[str, Any],
    *,
    teacher_id_hash: str,
    reviewed_at: str,
    teacher_confirmed_authority: bool,
    signing_key_id: str,
    private_key_pem: bytes,
) -> dict[str, Any]:
    """Create an explicit, content-bound teacher review receipt.

    The affirmative boolean is intentionally required.  Merely importing a
    file, or a model mentioning a teacher, cannot mint this receipt.
    """

    if teacher_confirmed_authority is not True:
        raise CurriculumBlueprintError(
            "explicit teacher authority confirmation is required"
        )
    teacher_hash = _hash(teacher_id_hash, "teacher_id_hash")
    timestamp = _timestamp(reviewed_at, "reviewed_at")
    key_id = _text(signing_key_id, "signing_key_id", maximum=120)
    if not _SIGNING_KEY_ID.fullmatch(key_id):
        raise CurriculumBlueprintError("signing_key_id is invalid")
    if not isinstance(private_key_pem, bytes):
        raise CurriculumBlueprintError("private_key_pem must contain an Ed25519 key")
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        private_key = serialization.load_pem_private_key(private_key_pem, password=None)
        if not isinstance(private_key, Ed25519PrivateKey):
            raise CurriculumBlueprintError("teacher signing key must be Ed25519")
    except (TypeError, ValueError) as exc:
        raise CurriculumBlueprintError("teacher signing key is invalid") from exc
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    material = {
        "schema": TEACHER_AUTHORITY_RECEIPT_SCHEMA,
        "teacher_id_hash": teacher_hash,
        "reviewed_at": timestamp,
        "spec_sha256": sha256(_canonical_json(spec)).hexdigest(),
        "scope": "curriculum_graph_measurement_and_sources",
        "authority_assertion": "teacher_owned_spec_explicit_review",
        "identity_assurance": "authenticated_teacher",
        "authoritative_for_runtime_grading": True,
        "signature_algorithm": "ed25519",
        "signing_key_id": key_id,
        "public_key_base64": base64.b64encode(public_key).decode("ascii"),
        "public_key_sha256": sha256(public_key).hexdigest(),
    }
    signature = private_key.sign(_canonical_json(material))
    signed = {
        **material,
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }
    receipt_hash = sha256(_canonical_json(signed)).hexdigest()
    return {
        **signed,
        "receipt_id": f"receipt_{receipt_hash[:20]}",
        "receipt_sha256": receipt_hash,
    }


def verify_teacher_curriculum_authority_receipt(
    receipt: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    trusted_teacher_public_keys: Mapping[str, bytes | str],
) -> dict[str, Any]:
    """Verify cryptographic identity against a deployment-owned trust root."""

    checked = _validate_authority_receipt(
        receipt, expected_spec_sha256=sha256(_canonical_json(spec)).hexdigest()
    )
    key_id = str(checked["signing_key_id"])
    trusted = trusted_teacher_public_keys.get(key_id)
    if trusted is None:
        raise CurriculumBlueprintError("teacher signing key is not trusted")
    if _public_key_bytes(trusted) != _strict_base64(
        checked["public_key_base64"],
        "authority.receipt.public_key_base64",
        expected_bytes=32,
    ):
        raise CurriculumBlueprintError(
            "teacher signing key does not match trust registry"
        )
    return deepcopy(dict(checked))


def verify_teacher_curriculum_runtime_authority(
    blueprint: Mapping[str, Any],
    *,
    trusted_teacher_public_keys: Mapping[str, bytes | str],
) -> dict[str, Any]:
    """Re-establish authority at the runtime boundary, never from JSON alone."""

    _validate_curriculum_blueprint_structure(blueprint)
    origin = _mapping(blueprint.get("origin"), "origin")
    authority = _mapping(blueprint.get("authority"), "authority")
    if (
        origin.get("kind") != "teacher_owned_spec"
        or authority.get("authority") is not True
    ):
        raise CurriculumBlueprintError(
            "curriculum is not teacher-owned authoritative material"
        )
    spec = {
        key: deepcopy(blueprint[key])
        for key in (
            "title",
            "lessons",
            "objectives",
            "knowledge_components",
            "prerequisite_edges",
            "topological_order",
            "source_spans",
            "factual_claims",
            "rubrics",
            "item_blueprints",
            "remediation_branches",
            "delayed_reviews",
        )
    }
    receipt = verify_teacher_curriculum_authority_receipt(
        _mapping(authority.get("receipt"), "authority.receipt"),
        spec,
        trusted_teacher_public_keys=trusted_teacher_public_keys,
    )
    return {
        "authoritative_for_runtime_grading": True,
        "curriculum_id": str(blueprint["curriculum_id"]),
        "receipt_id": str(receipt["receipt_id"]),
        "signing_key_id": str(receipt["signing_key_id"]),
        "receipt_sha256": str(receipt["receipt_sha256"]),
    }


def _validate_source_span(row: Mapping[str, Any], index: int) -> None:
    field = f"source_spans[{index}]"
    _strict_keys(
        row,
        {
            "source_span_id",
            "resource_id",
            "content_sha256",
            "excerpt_sha256",
            "locator",
            "authority",
        },
        field,
    )
    identifier = _text(row["source_span_id"], f"{field}.source_span_id", maximum=25)
    if not _ENTITY_ID.fullmatch(identifier) or not identifier.startswith("span_"):
        raise CurriculumBlueprintError(f"{field}.source_span_id is invalid")
    resource_id = _text(row["resource_id"], f"{field}.resource_id", maximum=160)
    if not _SAFE_RESOURCE_ID.fullmatch(resource_id):
        raise CurriculumBlueprintError(f"{field}.resource_id is unsafe")
    _hash(row["content_sha256"], f"{field}.content_sha256")
    _hash(row["excerpt_sha256"], f"{field}.excerpt_sha256")
    locator = _mapping(row["locator"], f"{field}.locator")
    _strict_keys(locator, {"kind", "start", "end"}, f"{field}.locator")
    if locator["kind"] not in {"page", "slide", "line", "section"}:
        raise CurriculumBlueprintError(f"{field}.locator.kind is invalid")
    start = _integer(
        locator["start"], f"{field}.locator.start", minimum=1, maximum=10_000_000
    )
    end = _integer(
        locator["end"], f"{field}.locator.end", minimum=1, maximum=10_000_000
    )
    if end < start:
        raise CurriculumBlueprintError(f"{field}.locator end precedes start")
    _boolean(row["authority"], f"{field}.authority")
    if row["source_span_id"] != _source_span_id(row):
        raise CurriculumBlueprintError(f"{field}.source_span_id is not content-derived")


def _validate_integrity(value: Mapping[str, Any]) -> None:
    integrity = _mapping(value["integrity"], "integrity")
    _strict_keys(integrity, {"algorithm", "content_sha256"}, "integrity")
    if integrity["algorithm"] != "sha256_canonical_json_without_integrity":
        raise CurriculumBlueprintError("integrity.algorithm is invalid")
    declared = _hash(integrity["content_sha256"], "integrity.content_sha256")
    material = deepcopy(dict(value))
    material.pop("integrity", None)
    if declared != sha256(_canonical_json(material)).hexdigest():
        raise CurriculumBlueprintError("integrity.content_sha256 mismatch")
    identity_material = deepcopy(material)
    identity_material.pop("curriculum_id", None)
    expected_id = "cur_" + sha256(_canonical_json(identity_material)).hexdigest()[:24]
    if value["curriculum_id"] != expected_id:
        raise CurriculumBlueprintError("curriculum_id does not bind the blueprint")


def _validate_curriculum_blueprint_structure(value: Any) -> None:
    """Validate graph topology, reference closure, coverage, and authority."""

    root = _mapping(value, "curriculum blueprint")
    _strict_keys(
        root,
        {
            "schema",
            "curriculum_id",
            "title",
            "origin",
            "lessons",
            "objectives",
            "knowledge_components",
            "prerequisite_edges",
            "topological_order",
            "source_spans",
            "factual_claims",
            "rubrics",
            "item_blueprints",
            "remediation_branches",
            "delayed_reviews",
            "authority",
            "integrity",
        },
        "curriculum blueprint",
    )
    if root["schema"] != CURRICULUM_BLUEPRINT_SCHEMA:
        raise CurriculumBlueprintError("curriculum blueprint schema is invalid")
    curriculum_id = _text(root["curriculum_id"], "curriculum_id", maximum=28)
    if not _CURRICULUM_ID.fullmatch(curriculum_id):
        raise CurriculumBlueprintError("curriculum_id is invalid")
    _text(root["title"], "title", maximum=240)

    origin = _mapping(root["origin"], "origin")
    _strict_keys(
        origin, {"kind", "source_schema", "source_id", "source_sha256"}, "origin"
    )
    if origin["kind"] not in {
        "generated_unvalidated",
        "legacy_generated_unvalidated",
        "teacher_owned_spec",
    }:
        raise CurriculumBlueprintError("origin.kind is invalid")
    _text(origin["source_schema"], "origin.source_schema", maximum=120)
    _text(origin["source_id"], "origin.source_id", maximum=160)
    _hash(origin["source_sha256"], "origin.source_sha256")

    lesson_rows = [
        _mapping(row, f"lessons[{index}]")
        for index, row in enumerate(
            _array(root["lessons"], "lessons", minimum=1, maximum=100)
        )
    ]
    for index, row in enumerate(lesson_rows):
        field = f"lessons[{index}]"
        _strict_keys(
            row,
            {
                "lesson_id",
                "legacy_lesson_id",
                "title",
                "order",
                "objective_ids",
                "kc_ids",
            },
            field,
        )
        lesson_id = _text(row["lesson_id"], f"{field}.lesson_id", maximum=24)
        if not _LESSON_ID.fullmatch(lesson_id):
            raise CurriculumBlueprintError(f"{field}.lesson_id is invalid")
        _text(
            row["legacy_lesson_id"], f"{field}.legacy_lesson_id", maximum=120, minimum=0
        )
        title = _text(row["title"], f"{field}.title", maximum=160)
        _integer(row["order"], f"{field}.order", minimum=1, maximum=100)
        objective_ids = _string_ids(
            row["objective_ids"],
            f"{field}.objective_ids",
            pattern=_OBJECTIVE_ID,
            minimum=1,
            maximum=32,
        )
        kc_ids = _string_ids(
            row["kc_ids"], f"{field}.kc_ids", pattern=_KC_ID, minimum=1, maximum=32
        )
        if lesson_id != stable_curriculum_lesson_id(
            title, objective_ids=objective_ids, knowledge_component_ids=kc_ids
        ):
            raise CurriculumBlueprintError(f"{field}.lesson_id is not content-derived")
    lesson_by_id = _index_by_id(lesson_rows, "lesson_id", "lessons")
    lesson_orders = [int(row["order"]) for row in lesson_rows]
    if sorted(lesson_orders) != list(range(1, len(lesson_rows) + 1)):
        raise CurriculumBlueprintError("lesson order must be globally contiguous")

    objective_rows = [
        _mapping(row, f"objectives[{index}]")
        for index, row in enumerate(
            _array(root["objectives"], "objectives", minimum=1, maximum=200)
        )
    ]
    for index, row in enumerate(objective_rows):
        field = f"objectives[{index}]"
        _strict_keys(
            row,
            {
                "objective_id",
                "statement",
                "lesson_ids",
                "kc_ids",
                "rubric_ids",
                "item_blueprint_ids",
                "remediation_branch_ids",
                "delayed_review_ids",
            },
            field,
        )
        statement = _text(row["statement"], f"{field}.statement", maximum=600)
        if not _is_measurable(statement):
            raise CurriculumBlueprintError(f"{field}.statement is not measurable")
        if row["objective_id"] != stable_curriculum_objective_id(statement):
            raise CurriculumBlueprintError(f"{field}.objective_id is not stable")
        _string_ids(
            row["lesson_ids"],
            f"{field}.lesson_ids",
            pattern=_LESSON_ID,
            minimum=1,
            maximum=100,
        )
        _string_ids(
            row["kc_ids"], f"{field}.kc_ids", pattern=_KC_ID, minimum=1, maximum=64
        )
        for key, prefix in (
            ("rubric_ids", "rubric_"),
            ("item_blueprint_ids", "item_"),
            ("remediation_branch_ids", "remedy_"),
            ("delayed_review_ids", "review_"),
        ):
            values = _string_ids(
                row[key], f"{field}.{key}", pattern=_ENTITY_ID, minimum=1, maximum=256
            )
            if any(not item.startswith(prefix) for item in values):
                raise CurriculumBlueprintError(
                    f"{field}.{key} contains a wrong ID kind"
                )
    objective_by_id = _index_by_id(objective_rows, "objective_id", "objectives")

    kc_rows = [
        _mapping(row, f"knowledge_components[{index}]")
        for index, row in enumerate(
            _array(
                root["knowledge_components"],
                "knowledge_components",
                minimum=1,
                maximum=720,
            )
        )
    ]
    for index, row in enumerate(kc_rows):
        field = f"knowledge_components[{index}]"
        _strict_keys(
            row,
            {
                "kc_id",
                "label",
                "introduced_lesson_id",
                "objective_ids",
                "prerequisite_kc_ids",
                "source_span_ids",
                "rubric_ids",
                "item_blueprint_ids",
                "remediation_branch_ids",
                "delayed_review_ids",
            },
            field,
        )
        label = _text(row["label"], f"{field}.label", maximum=160)
        if row["kc_id"] != stable_knowledge_component_id(label):
            raise CurriculumBlueprintError(f"{field}.kc_id is not stable")
        introduced = _text(
            row["introduced_lesson_id"], f"{field}.introduced_lesson_id", maximum=24
        )
        if not _LESSON_ID.fullmatch(introduced):
            raise CurriculumBlueprintError(f"{field}.introduced_lesson_id is invalid")
        _string_ids(
            row["objective_ids"],
            f"{field}.objective_ids",
            pattern=_OBJECTIVE_ID,
            minimum=1,
            maximum=200,
        )
        _string_ids(
            row["prerequisite_kc_ids"],
            f"{field}.prerequisite_kc_ids",
            pattern=_KC_ID,
            maximum=64,
        )
        for key, prefix, minimum in (
            ("source_span_ids", "span_", 0),
            ("rubric_ids", "rubric_", 1),
            ("item_blueprint_ids", "item_", 2),
            ("remediation_branch_ids", "remedy_", 1),
            ("delayed_review_ids", "review_", 1),
        ):
            values = _string_ids(
                row[key],
                f"{field}.{key}",
                pattern=_ENTITY_ID,
                minimum=minimum,
                maximum=512,
            )
            if any(not item.startswith(prefix) for item in values):
                raise CurriculumBlueprintError(
                    f"{field}.{key} contains a wrong ID kind"
                )
    kc_by_id = _index_by_id(kc_rows, "kc_id", "knowledge_components")

    edge_rows = [
        _mapping(row, f"prerequisite_edges[{index}]")
        for index, row in enumerate(
            _array(root["prerequisite_edges"], "prerequisite_edges", maximum=2_000)
        )
    ]
    edge_pairs: list[tuple[str, str]] = []
    for index, row in enumerate(edge_rows):
        field = f"prerequisite_edges[{index}]"
        _strict_keys(
            row, {"edge_id", "prerequisite_kc_id", "dependent_kc_id", "origin"}, field
        )
        prerequisite = _text(
            row["prerequisite_kc_id"], f"{field}.prerequisite_kc_id", maximum=84
        )
        dependent = _text(
            row["dependent_kc_id"], f"{field}.dependent_kc_id", maximum=84
        )
        if not _KC_ID.fullmatch(prerequisite) or not _KC_ID.fullmatch(dependent):
            raise CurriculumBlueprintError(f"{field} contains an invalid KC ID")
        if row["edge_id"] != _edge_id(prerequisite, dependent):
            raise CurriculumBlueprintError(f"{field}.edge_id is not content-derived")
        if row["origin"] not in {
            "outline_order_inferred_unvalidated",
            "teacher_owned_spec",
        }:
            raise CurriculumBlueprintError(f"{field}.origin is invalid")
        edge_pairs.append((prerequisite, dependent))
    if len(edge_pairs) != len(set(edge_pairs)):
        raise CurriculumBlueprintError("prerequisite_edges must not contain duplicates")

    source_rows = [
        _mapping(row, f"source_spans[{index}]")
        for index, row in enumerate(
            _array(root["source_spans"], "source_spans", maximum=2_000)
        )
    ]
    for index, row in enumerate(source_rows):
        _validate_source_span(row, index)
    source_by_id = _index_by_id(source_rows, "source_span_id", "source_spans")

    claim_rows = [
        _mapping(row, f"factual_claims[{index}]")
        for index, row in enumerate(
            _array(root["factual_claims"], "factual_claims", maximum=2_000)
        )
    ]
    for index, row in enumerate(claim_rows):
        field = f"factual_claims[{index}]"
        _strict_keys(row, {"claim_id", "statement", "kc_ids", "source_span_ids"}, field)
        statement = _text(row["statement"], f"{field}.statement", maximum=2_000)
        kc_ids = _string_ids(
            row["kc_ids"], f"{field}.kc_ids", pattern=_KC_ID, minimum=1, maximum=16
        )
        span_ids = _string_ids(
            row["source_span_ids"],
            f"{field}.source_span_ids",
            pattern=_ENTITY_ID,
            minimum=1,
            maximum=16,
        )
        if any(not item.startswith("span_") for item in span_ids):
            raise CurriculumBlueprintError(
                f"{field}.source_span_ids has a wrong ID kind"
            )
        if row["claim_id"] != _claim_id(statement, kc_ids, span_ids):
            raise CurriculumBlueprintError(f"{field}.claim_id is not content-derived")

    rubric_rows = [
        _mapping(row, f"rubrics[{index}]")
        for index, row in enumerate(
            _array(root["rubrics"], "rubrics", minimum=1, maximum=4_000)
        )
    ]
    for index, row in enumerate(rubric_rows):
        field = f"rubrics[{index}]"
        _strict_keys(
            row,
            {
                "rubric_id",
                "objective_id",
                "kc_id",
                "description",
                "source_span_ids",
                "authority",
                "assessment_eligible",
            },
            field,
        )
        objective_id = _text(row["objective_id"], f"{field}.objective_id", maximum=24)
        kc_id = _text(row["kc_id"], f"{field}.kc_id", maximum=84)
        description = _text(row["description"], f"{field}.description", maximum=1_000)
        span_ids = _string_ids(
            row["source_span_ids"],
            f"{field}.source_span_ids",
            pattern=_ENTITY_ID,
            maximum=32,
        )
        if any(not item.startswith("span_") for item in span_ids):
            raise CurriculumBlueprintError(
                f"{field}.source_span_ids has a wrong ID kind"
            )
        authority = _boolean(row["authority"], f"{field}.authority")
        eligible = _boolean(row["assessment_eligible"], f"{field}.assessment_eligible")
        if eligible and not authority:
            raise CurriculumBlueprintError(
                f"{field} cannot be eligible without authority"
            )
        if row["rubric_id"] != _rubric_id(objective_id, kc_id, description, span_ids):
            raise CurriculumBlueprintError(f"{field}.rubric_id is not content-derived")
    rubric_by_id = _index_by_id(rubric_rows, "rubric_id", "rubrics")

    item_rows = [
        _mapping(row, f"item_blueprints[{index}]")
        for index, row in enumerate(
            _array(root["item_blueprints"], "item_blueprints", minimum=2, maximum=8_000)
        )
    ]
    for index, row in enumerate(item_rows):
        field = f"item_blueprints[{index}]"
        _strict_keys(
            row,
            {
                "item_blueprint_id",
                "objective_id",
                "kc_id",
                "rubric_id",
                "kind",
                "prompt_intent",
                "source_span_ids",
                "authority",
                "assessment_eligible",
            },
            field,
        )
        objective_id = _text(row["objective_id"], f"{field}.objective_id", maximum=24)
        kc_id = _text(row["kc_id"], f"{field}.kc_id", maximum=84)
        rubric_id = _text(row["rubric_id"], f"{field}.rubric_id", maximum=27)
        kind = _text(row["kind"], f"{field}.kind", maximum=40)
        if kind not in {
            "worked_application",
            "near_transfer",
            "far_transfer",
            "delayed_retrieval",
            "language_production",
        }:
            raise CurriculumBlueprintError(f"{field}.kind is invalid")
        prompt_intent = _text(
            row["prompt_intent"], f"{field}.prompt_intent", maximum=1_000
        )
        span_ids = _string_ids(
            row["source_span_ids"],
            f"{field}.source_span_ids",
            pattern=_ENTITY_ID,
            maximum=32,
        )
        if any(not item.startswith("span_") for item in span_ids):
            raise CurriculumBlueprintError(
                f"{field}.source_span_ids has a wrong ID kind"
            )
        authority = _boolean(row["authority"], f"{field}.authority")
        eligible = _boolean(row["assessment_eligible"], f"{field}.assessment_eligible")
        if eligible and not authority:
            raise CurriculumBlueprintError(
                f"{field} cannot be eligible without authority"
            )
        expected = _item_id(
            objective_id, kc_id, rubric_id, kind, prompt_intent, span_ids
        )
        if row["item_blueprint_id"] != expected:
            raise CurriculumBlueprintError(
                f"{field}.item_blueprint_id is not content-derived"
            )
    item_by_id = _index_by_id(item_rows, "item_blueprint_id", "item_blueprints")

    remediation_rows = [
        _mapping(row, f"remediation_branches[{index}]")
        for index, row in enumerate(
            _array(
                root["remediation_branches"],
                "remediation_branches",
                minimum=1,
                maximum=4_000,
            )
        )
    ]
    for index, row in enumerate(remediation_rows):
        field = f"remediation_branches[{index}]"
        _strict_keys(
            row,
            {
                "remediation_branch_id",
                "objective_id",
                "kc_id",
                "trigger",
                "lesson_id",
                "action",
            },
            field,
        )
        objective_id = _text(row["objective_id"], f"{field}.objective_id", maximum=24)
        kc_id = _text(row["kc_id"], f"{field}.kc_id", maximum=84)
        trigger = _text(row["trigger"], f"{field}.trigger", maximum=80)
        if trigger not in {"incorrect", "partial", "misconception", "uncertain"}:
            raise CurriculumBlueprintError(f"{field}.trigger is invalid")
        lesson_id = _text(row["lesson_id"], f"{field}.lesson_id", maximum=24)
        action = _text(row["action"], f"{field}.action", maximum=1_000)
        if row["remediation_branch_id"] != _remediation_id(
            objective_id, kc_id, trigger, lesson_id, action
        ):
            raise CurriculumBlueprintError(
                f"{field}.remediation_branch_id is not content-derived"
            )
    remediation_by_id = _index_by_id(
        remediation_rows, "remediation_branch_id", "remediation_branches"
    )

    review_rows = [
        _mapping(row, f"delayed_reviews[{index}]")
        for index, row in enumerate(
            _array(root["delayed_reviews"], "delayed_reviews", minimum=1, maximum=4_000)
        )
    ]
    for index, row in enumerate(review_rows):
        field = f"delayed_reviews[{index}]"
        _strict_keys(
            row,
            {
                "delayed_review_id",
                "objective_id",
                "kc_id",
                "after_days",
                "item_blueprint_id",
            },
            field,
        )
        objective_id = _text(row["objective_id"], f"{field}.objective_id", maximum=24)
        kc_id = _text(row["kc_id"], f"{field}.kc_id", maximum=84)
        after_days = _integer(
            row["after_days"], f"{field}.after_days", minimum=1, maximum=365
        )
        item_id = _text(
            row["item_blueprint_id"], f"{field}.item_blueprint_id", maximum=25
        )
        if row["delayed_review_id"] != _review_id(
            objective_id, kc_id, after_days, item_id
        ):
            raise CurriculumBlueprintError(
                f"{field}.delayed_review_id is not content-derived"
            )
    review_by_id = _index_by_id(review_rows, "delayed_review_id", "delayed_reviews")

    # Reference closure and bidirectional mapping checks.
    def require_refs(refs: Iterable[str], index: Mapping[str, Any], field: str) -> None:
        missing = sorted(set(refs) - set(index))
        if missing:
            raise CurriculumBlueprintError(
                f"{field} contains dangling refs: {','.join(missing)}"
            )

    for lesson in lesson_rows:
        require_refs(lesson["objective_ids"], objective_by_id, "lesson.objective_ids")
        require_refs(lesson["kc_ids"], kc_by_id, "lesson.kc_ids")
    for objective in objective_rows:
        require_refs(objective["lesson_ids"], lesson_by_id, "objective.lesson_ids")
        require_refs(objective["kc_ids"], kc_by_id, "objective.kc_ids")
        require_refs(objective["rubric_ids"], rubric_by_id, "objective.rubric_ids")
        require_refs(
            objective["item_blueprint_ids"], item_by_id, "objective.item_blueprint_ids"
        )
        require_refs(
            objective["remediation_branch_ids"],
            remediation_by_id,
            "objective.remediation_branch_ids",
        )
        require_refs(
            objective["delayed_review_ids"],
            review_by_id,
            "objective.delayed_review_ids",
        )
    for kc in kc_rows:
        require_refs(
            [kc["introduced_lesson_id"]], lesson_by_id, "KC introduced_lesson_id"
        )
        require_refs(kc["objective_ids"], objective_by_id, "KC objective_ids")
        require_refs(kc["prerequisite_kc_ids"], kc_by_id, "KC prerequisite_kc_ids")
        require_refs(kc["source_span_ids"], source_by_id, "KC source_span_ids")
        require_refs(kc["rubric_ids"], rubric_by_id, "KC rubric_ids")
        require_refs(kc["item_blueprint_ids"], item_by_id, "KC item_blueprint_ids")
        require_refs(
            kc["remediation_branch_ids"], remediation_by_id, "KC remediation_branch_ids"
        )
        require_refs(kc["delayed_review_ids"], review_by_id, "KC delayed_review_ids")
    for claim in claim_rows:
        require_refs(claim["kc_ids"], kc_by_id, "claim.kc_ids")
        require_refs(claim["source_span_ids"], source_by_id, "claim.source_span_ids")
    for rubric in rubric_rows:
        require_refs([rubric["objective_id"]], objective_by_id, "rubric.objective_id")
        require_refs([rubric["kc_id"]], kc_by_id, "rubric.kc_id")
        require_refs(rubric["source_span_ids"], source_by_id, "rubric.source_span_ids")
    for item in item_rows:
        require_refs([item["objective_id"]], objective_by_id, "item.objective_id")
        require_refs([item["kc_id"]], kc_by_id, "item.kc_id")
        require_refs([item["rubric_id"]], rubric_by_id, "item.rubric_id")
        require_refs(item["source_span_ids"], source_by_id, "item.source_span_ids")
        rubric = rubric_by_id[item["rubric_id"]]
        if (item["objective_id"], item["kc_id"]) != (
            rubric["objective_id"],
            rubric["kc_id"],
        ):
            raise CurriculumBlueprintError(
                "item blueprint does not match its rubric target"
            )
    for remediation in remediation_rows:
        require_refs(
            [remediation["objective_id"]], objective_by_id, "remediation.objective_id"
        )
        require_refs([remediation["kc_id"]], kc_by_id, "remediation.kc_id")
        require_refs([remediation["lesson_id"]], lesson_by_id, "remediation.lesson_id")
    for review in review_rows:
        require_refs([review["objective_id"]], objective_by_id, "review.objective_id")
        require_refs([review["kc_id"]], kc_by_id, "review.kc_id")
        require_refs(
            [review["item_blueprint_id"]], item_by_id, "review.item_blueprint_id"
        )

    for lesson in lesson_rows:
        for objective_id in lesson["objective_ids"]:
            if lesson["lesson_id"] not in objective_by_id[objective_id]["lesson_ids"]:
                raise CurriculumBlueprintError(
                    "lesson/objective mapping is not bidirectional"
                )
        for kc_id in lesson["kc_ids"]:
            if lesson["lesson_id"] != kc_by_id[kc_id][
                "introduced_lesson_id"
            ] and lesson["lesson_id"] not in {
                lesson_id
                for objective_id in kc_by_id[kc_id]["objective_ids"]
                for lesson_id in objective_by_id[objective_id]["lesson_ids"]
            }:
                raise CurriculumBlueprintError(
                    "lesson/KC mapping is not covered by an objective"
                )
    for objective in objective_rows:
        for lesson_id in objective["lesson_ids"]:
            if (
                objective["objective_id"]
                not in lesson_by_id[lesson_id]["objective_ids"]
            ):
                raise CurriculumBlueprintError(
                    "objective/lesson mapping is not bidirectional"
                )
        for kc_id in objective["kc_ids"]:
            kc = kc_by_id[kc_id]
            if objective["objective_id"] not in kc["objective_ids"]:
                raise CurriculumBlueprintError(
                    "objective/KC mapping is not bidirectional"
                )
            pair_rubrics = [
                row
                for row in rubric_rows
                if row["objective_id"] == objective["objective_id"]
                and row["kc_id"] == kc_id
            ]
            pair_items = [
                row
                for row in item_rows
                if row["objective_id"] == objective["objective_id"]
                and row["kc_id"] == kc_id
            ]
            pair_remediation = [
                row
                for row in remediation_rows
                if row["objective_id"] == objective["objective_id"]
                and row["kc_id"] == kc_id
            ]
            pair_reviews = [
                row
                for row in review_rows
                if row["objective_id"] == objective["objective_id"]
                and row["kc_id"] == kc_id
            ]
            if not pair_rubrics:
                raise CurriculumBlueprintError("objective/KC pair has no rubric")
            if len(pair_items) < 2:
                raise CurriculumBlueprintError(
                    "objective/KC pair needs at least two item blueprints"
                )
            if not pair_remediation:
                raise CurriculumBlueprintError(
                    "objective/KC pair has no remediation branch"
                )
            if not pair_reviews:
                raise CurriculumBlueprintError(
                    "objective/KC pair has no delayed review"
                )

    expected_prerequisites = {
        (prerequisite, str(kc["kc_id"]))
        for kc in kc_rows
        for prerequisite in kc["prerequisite_kc_ids"]
    }
    if set(edge_pairs) != expected_prerequisites:
        raise CurriculumBlueprintError("KC prerequisites and edge list do not match")
    lesson_rank = {
        str(kc["kc_id"]): int(lesson_by_id[str(kc["introduced_lesson_id"])]["order"])
        for kc in kc_rows
    }
    for prerequisite, dependent in edge_pairs:
        if lesson_rank[prerequisite] >= lesson_rank[dependent]:
            raise CurriculumBlueprintError(
                "prerequisite must be introduced in an earlier lesson than its dependent"
            )
    expected_topological = _canonical_topological_order(
        kc_by_id, edge_pairs, lesson_rank
    )
    topological = _string_ids(
        root["topological_order"],
        "topological_order",
        pattern=_KC_ID,
        minimum=len(kc_rows),
        maximum=len(kc_rows),
    )
    if topological != expected_topological:
        raise CurriculumBlueprintError("topological_order is invalid or non-canonical")

    authority = _mapping(root["authority"], "authority")
    _strict_keys(
        authority,
        {"status", "authority", "authoritative_for_runtime_grading", "receipt"},
        "authority",
    )
    generated = origin["kind"] in {
        "generated_unvalidated",
        "legacy_generated_unvalidated",
    }
    if generated:
        if dict(authority) != {
            "status": "generated_unvalidated",
            "authority": False,
            "authoritative_for_runtime_grading": False,
            "receipt": None,
        }:
            raise CurriculumBlueprintError(
                "generated curriculum authority boundary is invalid"
            )
        if claim_rows:
            raise CurriculumBlueprintError(
                "generated curriculum cannot assert factual claims"
            )
        if source_rows or any(
            row["authority"] or row["assessment_eligible"]
            for row in rubric_rows + item_rows
        ):
            raise CurriculumBlueprintError(
                "generated curriculum cannot self-authorize sources or assessment"
            )
        if any(
            row["origin"] != "outline_order_inferred_unvalidated" for row in edge_rows
        ):
            raise CurriculumBlueprintError(
                "generated prerequisite edges must remain unvalidated"
            )
    else:
        if authority["status"] != "teacher_owned_authoritative":
            raise CurriculumBlueprintError(
                "teacher-owned curriculum authority status is invalid"
            )
        if (
            authority["authority"] is not True
            or authority["authoritative_for_runtime_grading"] is not True
        ):
            raise CurriculumBlueprintError(
                "teacher-owned curriculum needs explicit authority"
            )
        _validate_authority_receipt(
            authority["receipt"], expected_spec_sha256=origin["source_sha256"]
        )
        if not source_rows:
            raise CurriculumBlueprintError(
                "teacher-owned curriculum needs source spans"
            )
        if any(not row["authority"] for row in source_rows):
            raise CurriculumBlueprintError(
                "teacher-owned source spans must be authoritative"
            )
        if any(not kc["source_span_ids"] for kc in kc_rows):
            raise CurriculumBlueprintError(
                "every teacher-owned KC needs an authoritative source span"
            )
        if any(not row["source_span_ids"] for row in rubric_rows + item_rows):
            raise CurriculumBlueprintError(
                "teacher-owned rubric/item needs a source span"
            )
        if any(
            not row["authority"] or not row["assessment_eligible"]
            for row in rubric_rows + item_rows
        ):
            raise CurriculumBlueprintError(
                "teacher-owned measurement rows must be authoritative and eligible"
            )
        for claim in claim_rows:
            if any(
                not source_by_id[source_id]["authority"]
                for source_id in claim["source_span_ids"]
            ):
                raise CurriculumBlueprintError(
                    "factual claim lacks authoritative source support"
                )
        if any(row["origin"] != "teacher_owned_spec" for row in edge_rows):
            raise CurriculumBlueprintError(
                "teacher prerequisite edges must come from teacher spec"
            )

    _validate_integrity(root)
    if len(_canonical_json(root)) > _MAX_BLUEPRINT_BYTES:
        raise CurriculumBlueprintError("curriculum blueprint exceeds the safety limit")


def validate_curriculum_blueprint(
    value: Any,
    *,
    trusted_teacher_public_keys: Mapping[str, bytes | str] | None = None,
) -> None:
    """Validate structure and re-establish any claimed runtime authority.

    Generated projections need no trust configuration.  A teacher-owned graph
    fails closed unless the caller supplies a deployment-owned key registry;
    JSON/schema validity or a self-signed receipt alone is never authority.
    """

    _validate_curriculum_blueprint_structure(value)
    if isinstance(value, Mapping):
        authority = value.get("authority")
        if isinstance(authority, Mapping) and authority.get("authority") is True:
            if not trusted_teacher_public_keys:
                raise CurriculumBlueprintError(
                    "authoritative curriculum validation requires configured trusted teacher keys"
                )
            verify_teacher_curriculum_runtime_authority(
                value,
                trusted_teacher_public_keys=trusted_teacher_public_keys,
            )


def _seal_blueprint(material: Mapping[str, Any]) -> dict[str, Any]:
    output = deepcopy(dict(material))
    output["curriculum_id"] = "cur_" + "0" * 24
    # The identifier is calculated with its placeholder omitted, preventing a
    # recursive ID/hash dependency while still binding every other field.
    without_integrity = deepcopy(output)
    without_integrity.pop("integrity", None)
    without_integrity.pop("curriculum_id", None)
    digest_material = deepcopy(without_integrity)
    curriculum_hash = sha256(_canonical_json(digest_material)).hexdigest()
    output["curriculum_id"] = "cur_" + curriculum_hash[:24]
    content_material = deepcopy(output)
    content_material.pop("integrity", None)
    output["integrity"] = {
        "algorithm": "sha256_canonical_json_without_integrity",
        "content_sha256": sha256(_canonical_json(content_material)).hexdigest(),
    }
    # ``validate`` derives the curriculum ID from the full material with ID;
    # use the same non-recursive convention by checking in a small wrapper.
    _validate_curriculum_blueprint_structure(output)
    return output


def _measurement_rows(
    *,
    objective_rows: Sequence[Mapping[str, Any]],
    lesson_by_objective: Mapping[str, str],
    source_span_ids_by_kc: Mapping[str, Sequence[str]],
    authoritative: bool,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    rubrics: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    remediations: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    for objective in objective_rows:
        objective_id = str(objective["objective_id"])
        statement = str(objective["statement"])
        for kc_id in objective["kc_ids"]:
            spans = list(source_span_ids_by_kc.get(str(kc_id), ()))
            description = f"学习者以可核验产出来证明该知识组件达到目标：{statement}"[
                :1_000
            ]
            rubric_id = _rubric_id(objective_id, str(kc_id), description, spans)
            rubrics.append(
                {
                    "rubric_id": rubric_id,
                    "objective_id": objective_id,
                    "kc_id": kc_id,
                    "description": description,
                    "source_span_ids": spans,
                    "authority": authoritative,
                    "assessment_eligible": authoritative,
                }
            )
            pair_items: list[dict[str, Any]] = []
            for kind, prompt_intent in (
                ("worked_application", f"在熟悉情境中产出能核验“{statement}”的回答。"),
                (
                    "near_transfer",
                    f"改变一个表面条件，再产出能核验“{statement}”的回答。",
                ),
            ):
                item_id = _item_id(
                    objective_id,
                    str(kc_id),
                    rubric_id,
                    kind,
                    prompt_intent,
                    spans,
                )
                pair_items.append(
                    {
                        "item_blueprint_id": item_id,
                        "objective_id": objective_id,
                        "kc_id": kc_id,
                        "rubric_id": rubric_id,
                        "kind": kind,
                        "prompt_intent": prompt_intent,
                        "source_span_ids": spans,
                        "authority": authoritative,
                        "assessment_eligible": authoritative,
                    }
                )
            items.extend(pair_items)
            lesson_id = lesson_by_objective[objective_id]
            action = "回到对应课节，用更小示例重教该知识组件，再进行一次新题核验。"
            remediation_id = _remediation_id(
                objective_id, str(kc_id), "partial", lesson_id, action
            )
            remediations.append(
                {
                    "remediation_branch_id": remediation_id,
                    "objective_id": objective_id,
                    "kc_id": kc_id,
                    "trigger": "partial",
                    "lesson_id": lesson_id,
                    "action": action,
                }
            )
            review_id = _review_id(
                objective_id,
                str(kc_id),
                7,
                str(pair_items[1]["item_blueprint_id"]),
            )
            reviews.append(
                {
                    "delayed_review_id": review_id,
                    "objective_id": objective_id,
                    "kc_id": kc_id,
                    "after_days": 7,
                    "item_blueprint_id": pair_items[1]["item_blueprint_id"],
                }
            )
    return rubrics, items, remediations, reviews


def build_teacher_owned_curriculum_spec(
    *,
    title: str,
    lessons: Sequence[Mapping[str, Any]],
    source_spans: Sequence[Mapping[str, Any]],
    factual_claims: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build the normalized spec a teacher reviews before signing a receipt.

    Each lesson declares one measurable ``objective`` and one or more knowledge
    components.  A component declaration has ``label``, ``prerequisites`` and
    either ``source_resource_ids`` or ``source_span_ids``.  Source rows contain
    hashes plus an exact page/slide/line/section locator; raw resource text is
    deliberately not copied into the blueprint.

    Rubric, two item-blueprint, remediation, and delayed-review rows are
    materialized here, but remain unauthoritative until the returned complete
    spec is explicitly reviewed and bound by
    :func:`create_teacher_curriculum_authority_receipt`.
    """

    safe_title = _text(title, "title", maximum=240)
    raw_sources = list(source_spans)
    if not raw_sources:
        raise CurriculumBlueprintError("teacher-owned curriculum needs source spans")
    normalized_sources: list[dict[str, Any]] = []
    spans_by_resource: dict[str, list[str]] = {}
    for index, raw in enumerate(raw_sources):
        row = _mapping(raw, f"source_spans[{index}]")
        _strict_keys(
            row,
            {"resource_id", "content_sha256", "excerpt_sha256", "locator"},
            f"source_spans[{index}]",
        )
        locator = _mapping(row["locator"], f"source_spans[{index}].locator")
        normalized = {
            "resource_id": deepcopy(row["resource_id"]),
            "content_sha256": deepcopy(row["content_sha256"]),
            "excerpt_sha256": deepcopy(row["excerpt_sha256"]),
            "locator": deepcopy(dict(locator)),
            "authority": True,
        }
        normalized["source_span_id"] = _source_span_id(normalized)
        _validate_source_span(normalized, index)
        if any(
            existing["source_span_id"] == normalized["source_span_id"]
            for existing in normalized_sources
        ):
            raise CurriculumBlueprintError("source_spans must not contain duplicates")
        normalized_sources.append(normalized)
        spans_by_resource.setdefault(str(normalized["resource_id"]), []).append(
            str(normalized["source_span_id"])
        )

    raw_lessons = list(lessons)
    if not 1 <= len(raw_lessons) <= 100:
        raise CurriculumBlueprintError("lessons must contain 1 to 100 rows")
    parsed_lessons: list[dict[str, Any]] = []
    component_definitions: dict[str, dict[str, Any]] = {}
    for lesson_index, raw in enumerate(raw_lessons, start=1):
        row = _mapping(raw, f"lessons[{lesson_index - 1}]")
        if not set(row) <= {
            "legacy_lesson_id",
            "title",
            "objective",
            "knowledge_components",
        } or not {"title", "objective", "knowledge_components"} <= set(row):
            raise CurriculumBlueprintError(
                f"lessons[{lesson_index - 1}] has invalid fields"
            )
        legacy_lesson_id = _text(
            row.get("legacy_lesson_id", ""),
            "lesson.legacy_lesson_id",
            maximum=120,
            minimum=0,
        )
        lesson_title = _text(row["title"], "lesson.title", maximum=160)
        statement = _text(row["objective"], "lesson.objective", maximum=600)
        if not _is_measurable(statement):
            raise CurriculumBlueprintError("lesson objective is not measurable")
        objective_id = stable_curriculum_objective_id(statement)
        raw_components = _array(
            row["knowledge_components"],
            "lesson.knowledge_components",
            minimum=1,
            maximum=32,
        )
        kc_ids: list[str] = []
        for component_index, raw_component in enumerate(raw_components):
            component = _mapping(
                raw_component,
                f"lessons[{lesson_index - 1}].knowledge_components[{component_index}]",
            )
            allowed_keys = {
                "label",
                "prerequisites",
                "source_resource_ids",
                "source_span_ids",
            }
            if not set(component) <= allowed_keys or not {
                "label",
                "prerequisites",
            } <= set(component):
                raise CurriculumBlueprintError(
                    "knowledge component needs label, prerequisites, and source refs"
                )
            label = _text(component["label"], "knowledge component label", maximum=160)
            kc_id = stable_knowledge_component_id(label)
            prerequisite_labels = [
                _text(item, "prerequisite label", maximum=160)
                for item in _array(
                    component["prerequisites"], "prerequisites", maximum=64
                )
            ]
            span_ids = [
                _text(item, "source_span_id", maximum=25)
                for item in _array(
                    component.get("source_span_ids", []),
                    "source_span_ids",
                    maximum=128,
                )
            ]
            resource_ids = [
                _text(item, "source_resource_id", maximum=160)
                for item in _array(
                    component.get("source_resource_ids", []),
                    "source_resource_ids",
                    maximum=128,
                )
            ]
            for resource_id in resource_ids:
                if resource_id not in spans_by_resource:
                    raise CurriculumBlueprintError(
                        f"knowledge component references unknown resource {resource_id}"
                    )
                span_ids.extend(spans_by_resource[resource_id])
            span_ids = list(dict.fromkeys(span_ids))
            if not span_ids:
                raise CurriculumBlueprintError(
                    "every teacher-owned KC needs at least one source span"
                )
            declared_span_ids = {
                str(source["source_span_id"]) for source in normalized_sources
            }
            if not set(span_ids) <= declared_span_ids:
                raise CurriculumBlueprintError(
                    "knowledge component has a dangling source span"
                )
            current = {
                "kc_id": kc_id,
                "label": label,
                "introduced_lesson_index": lesson_index - 1,
                "prerequisite_labels": prerequisite_labels,
                "source_span_ids": span_ids,
            }
            previous = component_definitions.get(kc_id)
            if previous is not None:
                if (
                    previous["label"] != label
                    or previous["prerequisite_labels"] != prerequisite_labels
                    or previous["source_span_ids"] != span_ids
                ):
                    raise CurriculumBlueprintError(
                        "repeated KC declarations must have identical prerequisites and sources"
                    )
            else:
                component_definitions[kc_id] = current
            kc_ids.append(kc_id)
        if len(kc_ids) != len(set(kc_ids)):
            raise CurriculumBlueprintError("lesson KCs must not contain duplicates")
        parsed_lessons.append(
            {
                "legacy_lesson_id": legacy_lesson_id,
                "title": lesson_title,
                "objective_id": objective_id,
                "statement": statement,
                "kc_ids": kc_ids,
            }
        )

    label_to_kc = {
        _canonical_text(str(row["label"])): kc_id
        for kc_id, row in component_definitions.items()
    }
    for row in component_definitions.values():
        for label in row["prerequisite_labels"]:
            if _canonical_text(label) not in label_to_kc:
                raise CurriculumBlueprintError(
                    f"prerequisite label is dangling: {label}"
                )

    lesson_rows: list[dict[str, Any]] = []
    for order, parsed in enumerate(parsed_lessons, start=1):
        objective_ids = [str(parsed["objective_id"])]
        lesson_id = stable_curriculum_lesson_id(
            str(parsed["title"]),
            objective_ids=objective_ids,
            knowledge_component_ids=parsed["kc_ids"],
        )
        lesson_rows.append(
            {
                "lesson_id": lesson_id,
                "legacy_lesson_id": parsed["legacy_lesson_id"],
                "title": parsed["title"],
                "order": order,
                "objective_ids": objective_ids,
                "kc_ids": parsed["kc_ids"],
            }
        )

    objective_builders: dict[str, dict[str, Any]] = {}
    for parsed, lesson in zip(parsed_lessons, lesson_rows, strict=True):
        objective_id = str(parsed["objective_id"])
        builder = objective_builders.setdefault(
            objective_id,
            {
                "objective_id": objective_id,
                "statement": parsed["statement"],
                "lesson_ids": [],
                "kc_ids": [],
            },
        )
        builder["lesson_ids"].append(lesson["lesson_id"])
        builder["kc_ids"] = list(
            dict.fromkeys(builder["kc_ids"] + list(parsed["kc_ids"]))
        )
    objective_rows = [
        {
            **row,
            "rubric_ids": [],
            "item_blueprint_ids": [],
            "remediation_branch_ids": [],
            "delayed_review_ids": [],
        }
        for row in objective_builders.values()
    ]
    lesson_by_objective = {
        str(row["objective_id"]): str(row["lesson_ids"][0]) for row in objective_rows
    }
    spans_by_kc = {
        kc_id: list(row["source_span_ids"])
        for kc_id, row in component_definitions.items()
    }
    rubrics, items, remediations, reviews = _measurement_rows(
        objective_rows=objective_rows,
        lesson_by_objective=lesson_by_objective,
        source_span_ids_by_kc=spans_by_kc,
        authoritative=True,
    )
    for objective in objective_rows:
        objective_id = objective["objective_id"]
        objective["rubric_ids"] = [
            row["rubric_id"] for row in rubrics if row["objective_id"] == objective_id
        ]
        objective["item_blueprint_ids"] = [
            row["item_blueprint_id"]
            for row in items
            if row["objective_id"] == objective_id
        ]
        objective["remediation_branch_ids"] = [
            row["remediation_branch_id"]
            for row in remediations
            if row["objective_id"] == objective_id
        ]
        objective["delayed_review_ids"] = [
            row["delayed_review_id"]
            for row in reviews
            if row["objective_id"] == objective_id
        ]

    edge_pairs: list[tuple[str, str]] = []
    kc_rows: list[dict[str, Any]] = []
    for kc_id, definition in component_definitions.items():
        prerequisite_ids = [
            label_to_kc[_canonical_text(label)]
            for label in definition["prerequisite_labels"]
        ]
        edge_pairs.extend((prerequisite, kc_id) for prerequisite in prerequisite_ids)
        objective_refs = [
            row["objective_id"] for row in objective_rows if kc_id in row["kc_ids"]
        ]
        introduced_lesson_id = lesson_rows[int(definition["introduced_lesson_index"])][
            "lesson_id"
        ]
        kc_rows.append(
            {
                "kc_id": kc_id,
                "label": definition["label"],
                "introduced_lesson_id": introduced_lesson_id,
                "objective_ids": objective_refs,
                "prerequisite_kc_ids": prerequisite_ids,
                "source_span_ids": definition["source_span_ids"],
                "rubric_ids": [
                    row["rubric_id"] for row in rubrics if row["kc_id"] == kc_id
                ],
                "item_blueprint_ids": [
                    row["item_blueprint_id"] for row in items if row["kc_id"] == kc_id
                ],
                "remediation_branch_ids": [
                    row["remediation_branch_id"]
                    for row in remediations
                    if row["kc_id"] == kc_id
                ],
                "delayed_review_ids": [
                    row["delayed_review_id"] for row in reviews if row["kc_id"] == kc_id
                ],
            }
        )
    lesson_rank = {
        row["kc_id"]: int(
            next(
                lesson["order"]
                for lesson in lesson_rows
                if lesson["lesson_id"] == row["introduced_lesson_id"]
            )
        )
        for row in kc_rows
    }
    topological = _canonical_topological_order(
        component_definitions, edge_pairs, lesson_rank
    )
    edge_rows = [
        {
            "edge_id": _edge_id(prerequisite, dependent),
            "prerequisite_kc_id": prerequisite,
            "dependent_kc_id": dependent,
            "origin": "teacher_owned_spec",
        }
        for prerequisite, dependent in edge_pairs
    ]

    claim_rows: list[dict[str, Any]] = []
    for index, raw_claim in enumerate(factual_claims):
        claim = _mapping(raw_claim, f"factual_claims[{index}]")
        allowed = {
            "statement",
            "knowledge_components",
            "source_resource_ids",
            "source_span_ids",
        }
        if not set(claim) <= allowed or not {
            "statement",
            "knowledge_components",
        } <= set(claim):
            raise CurriculumBlueprintError(
                "factual claim needs statement, KCs, and source refs"
            )
        statement = _text(claim["statement"], "factual claim statement", maximum=2_000)
        kc_ids = []
        for label in _array(
            claim["knowledge_components"],
            "factual claim knowledge_components",
            minimum=1,
            maximum=16,
        ):
            canonical = _canonical_text(
                _text(label, "factual claim knowledge component", maximum=160)
            )
            if canonical not in label_to_kc:
                raise CurriculumBlueprintError("factual claim references an unknown KC")
            kc_ids.append(label_to_kc[canonical])
        claim_span_ids = [
            _text(item, "factual claim source_span_id", maximum=25)
            for item in _array(
                claim.get("source_span_ids", []),
                "factual claim source_span_ids",
                maximum=16,
            )
        ]
        for resource_id in _array(
            claim.get("source_resource_ids", []),
            "factual claim source_resource_ids",
            maximum=16,
        ):
            safe_resource_id = _text(
                resource_id, "factual claim source_resource_id", maximum=160
            )
            if safe_resource_id not in spans_by_resource:
                raise CurriculumBlueprintError(
                    "factual claim references an unknown source resource"
                )
            claim_span_ids.extend(spans_by_resource[safe_resource_id])
        claim_span_ids = list(dict.fromkeys(claim_span_ids))
        if not claim_span_ids:
            raise CurriculumBlueprintError(
                "factual claim requires an authoritative source span"
            )
        if not set(claim_span_ids) <= {
            str(row["source_span_id"]) for row in normalized_sources
        }:
            raise CurriculumBlueprintError("factual claim has a dangling source span")
        claim_rows.append(
            {
                "claim_id": _claim_id(statement, kc_ids, claim_span_ids),
                "statement": statement,
                "kc_ids": kc_ids,
                "source_span_ids": claim_span_ids,
            }
        )

    return {
        "title": safe_title,
        "lessons": lesson_rows,
        "objectives": objective_rows,
        "knowledge_components": kc_rows,
        "prerequisite_edges": edge_rows,
        "topological_order": topological,
        "source_spans": normalized_sources,
        "factual_claims": claim_rows,
        "rubrics": rubrics,
        "item_blueprints": items,
        "remediation_branches": remediations,
        "delayed_reviews": reviews,
    }


def derive_generated_curriculum_blueprint(
    syllabus: Mapping[str, Any],
) -> dict[str, Any]:
    """Safely migrate/project a generated v1 syllabus into a curriculum graph.

    No matter what text appears in the old document, the projection remains
    non-authoritative and contains no factual claims or assessment-eligible
    items.  This is also the compatibility path for stored legacy syllabi.
    """

    # Lazy import avoids a module cycle when teacher_agent_syllabus re-exports
    # the curriculum API.
    from .teacher_agent_syllabus import validate_teaching_syllabus

    validate_teaching_syllabus(syllabus)
    source = _mapping(syllabus.get("source"), "syllabus.source")
    if source.get("kind") not in {"generated", "teacher_edited"}:
        raise CurriculumBlueprintError(
            "only generated or structurally teacher-edited syllabi use this projection"
        )

    raw_lessons: list[Mapping[str, Any]] = [
        _mapping(lesson, "syllabus lesson")
        for module in _array(
            syllabus.get("modules"), "syllabus.modules", minimum=1, maximum=12
        )
        for lesson in _array(
            module.get("lessons"), "syllabus module lessons", minimum=1, maximum=16
        )
    ]
    all_labels: list[str] = []
    introduced_index: dict[str, int] = {}
    lesson_component_ids: list[list[str]] = []
    for lesson_index, lesson in enumerate(raw_lessons):
        ids: list[str] = []
        for label in _array(
            lesson.get("knowledge_components"),
            "lesson.knowledge_components",
            minimum=1,
            maximum=12,
        ):
            safe_label = _text(label, "knowledge component", maximum=160)
            kc_id = stable_knowledge_component_id(safe_label)
            ids.append(kc_id)
            if kc_id not in introduced_index:
                introduced_index[kc_id] = lesson_index
                all_labels.append(safe_label)
        lesson_component_ids.append(ids)

    program_statements = [
        _text(item, "learning objective", maximum=600)
        for item in _array(
            syllabus.get("learning_objectives"),
            "learning_objectives",
            minimum=1,
            maximum=16,
        )
    ]
    lesson_statements = [
        _text(lesson.get("objective"), "lesson objective", maximum=600)
        for lesson in raw_lessons
    ]
    statements: list[str] = []
    for statement in program_statements + lesson_statements:
        if statement not in statements:
            statements.append(statement)
    for statement in statements:
        if not _is_measurable(statement):
            raise CurriculumBlueprintError(
                "generated objective is not measurable; teacher review/rewrite is required"
            )

    objective_ids = {
        statement: stable_curriculum_objective_id(statement) for statement in statements
    }
    program_mappings: dict[str, int] = {}
    for statement in program_statements:
        ranked = sorted(
            range(len(raw_lessons)),
            key=lambda index: (
                _objective_lesson_alignment_score(
                    statement,
                    lesson_title=str(raw_lessons[index]["title"]),
                    lesson_objective=lesson_statements[index],
                    kc_labels=[
                        str(item) for item in raw_lessons[index]["knowledge_components"]
                    ],
                ),
                _canonical_text(str(raw_lessons[index]["title"])),
                _canonical_text(lesson_statements[index]),
            ),
            reverse=True,
        )
        program_mappings[statement] = ranked[0]
    lesson_rows: list[dict[str, Any]] = []
    lesson_ids: list[str] = []
    for index, (lesson, kc_ids, statement) in enumerate(
        zip(raw_lessons, lesson_component_ids, lesson_statements, strict=True)
    ):
        local_objectives = list(
            dict.fromkeys(
                [
                    objective_ids[program]
                    for program in program_statements
                    if program_mappings[program] == index
                ]
                + [objective_ids[statement]]
            )
        )
        lesson_id = stable_curriculum_lesson_id(
            str(lesson["title"]),
            objective_ids=local_objectives,
            knowledge_component_ids=kc_ids,
        )
        lesson_ids.append(lesson_id)
        lesson_rows.append(
            {
                "lesson_id": lesson_id,
                "legacy_lesson_id": str(lesson.get("lesson_id") or ""),
                "title": str(lesson["title"]),
                "order": index + 1,
                "objective_ids": local_objectives,
                "kc_ids": list(kc_ids),
            }
        )

    objective_rows: list[dict[str, Any]] = []
    for statement in statements:
        objective_id = objective_ids[statement]
        mapped_indices = [
            index
            for index, candidate in enumerate(lesson_statements)
            if candidate == statement
        ]
        if statement in program_statements:
            mapped_indices.append(program_mappings[statement])
        mapped_indices = list(dict.fromkeys(mapped_indices))
        mapped_lessons = [lesson_ids[index] for index in mapped_indices]
        mapped_kcs = list(
            dict.fromkeys(
                kc_id
                for index in mapped_indices
                for kc_id in lesson_component_ids[index]
            )
        )
        objective_rows.append(
            {
                "objective_id": objective_id,
                "statement": statement,
                "lesson_ids": mapped_lessons,
                "kc_ids": mapped_kcs,
                "rubric_ids": [],
                "item_blueprint_ids": [],
                "remediation_branch_ids": [],
                "delayed_review_ids": [],
            }
        )

    # Only infer a simple first-introduction chain.  It is useful for ordering
    # but carries an explicit unvalidated origin and never grading authority.
    ordered_unique_kcs = list(
        dict.fromkeys(kc_id for row in lesson_component_ids for kc_id in row)
    )
    introduction_groups: list[list[str]] = []
    for lesson_index in range(len(raw_lessons)):
        introduced_here = [
            kc_id
            for kc_id in ordered_unique_kcs
            if introduced_index[kc_id] == lesson_index
        ]
        if introduced_here:
            introduction_groups.append(introduced_here)
    edge_pairs = [
        (prior_group[-1], next_group[0])
        for prior_group, next_group in zip(introduction_groups, introduction_groups[1:])
    ]
    prerequisites: dict[str, list[str]] = {kc_id: [] for kc_id in ordered_unique_kcs}
    for prerequisite, dependent in edge_pairs:
        prerequisites[dependent].append(prerequisite)

    lesson_by_objective = {
        str(row["objective_id"]): str(row["lesson_ids"][0]) for row in objective_rows
    }
    rubrics, items, remediations, reviews = _measurement_rows(
        objective_rows=objective_rows,
        lesson_by_objective=lesson_by_objective,
        source_span_ids_by_kc={},
        authoritative=False,
    )
    for objective in objective_rows:
        objective_id = objective["objective_id"]
        objective["rubric_ids"] = [
            row["rubric_id"] for row in rubrics if row["objective_id"] == objective_id
        ]
        objective["item_blueprint_ids"] = [
            row["item_blueprint_id"]
            for row in items
            if row["objective_id"] == objective_id
        ]
        objective["remediation_branch_ids"] = [
            row["remediation_branch_id"]
            for row in remediations
            if row["objective_id"] == objective_id
        ]
        objective["delayed_review_ids"] = [
            row["delayed_review_id"]
            for row in reviews
            if row["objective_id"] == objective_id
        ]

    label_by_kc = {stable_knowledge_component_id(label): label for label in all_labels}
    kc_rows: list[dict[str, Any]] = []
    for kc_id in ordered_unique_kcs:
        objective_refs = [
            row["objective_id"] for row in objective_rows if kc_id in row["kc_ids"]
        ]
        kc_rows.append(
            {
                "kc_id": kc_id,
                "label": label_by_kc[kc_id],
                "introduced_lesson_id": lesson_ids[introduced_index[kc_id]],
                "objective_ids": objective_refs,
                "prerequisite_kc_ids": prerequisites[kc_id],
                "source_span_ids": [],
                "rubric_ids": [
                    row["rubric_id"] for row in rubrics if row["kc_id"] == kc_id
                ],
                "item_blueprint_ids": [
                    row["item_blueprint_id"] for row in items if row["kc_id"] == kc_id
                ],
                "remediation_branch_ids": [
                    row["remediation_branch_id"]
                    for row in remediations
                    if row["kc_id"] == kc_id
                ],
                "delayed_review_ids": [
                    row["delayed_review_id"] for row in reviews if row["kc_id"] == kc_id
                ],
            }
        )

    edge_rows = [
        {
            "edge_id": _edge_id(prerequisite, dependent),
            "prerequisite_kc_id": prerequisite,
            "dependent_kc_id": dependent,
            "origin": "outline_order_inferred_unvalidated",
        }
        for prerequisite, dependent in edge_pairs
    ]
    lesson_rank = {
        row["kc_id"]: int(lesson_rows[introduced_index[row["kc_id"]]]["order"])
        for row in kc_rows
    }
    material = {
        "schema": CURRICULUM_BLUEPRINT_SCHEMA,
        "title": str(syllabus["title"]),
        "origin": {
            "kind": "legacy_generated_unvalidated",
            "source_schema": str(syllabus["schema"]),
            "source_id": str(syllabus["syllabus_id"]),
            "source_sha256": str(syllabus["integrity"]["content_sha256"]),
        },
        "lessons": lesson_rows,
        "objectives": objective_rows,
        "knowledge_components": kc_rows,
        "prerequisite_edges": edge_rows,
        "topological_order": _canonical_topological_order(
            ordered_unique_kcs, edge_pairs, lesson_rank
        ),
        "source_spans": [],
        "factual_claims": [],
        "rubrics": rubrics,
        "item_blueprints": items,
        "remediation_branches": remediations,
        "delayed_reviews": reviews,
        "authority": {
            "status": "generated_unvalidated",
            "authority": False,
            "authoritative_for_runtime_grading": False,
            "receipt": None,
        },
    }
    return _seal_blueprint(material)


def seal_teacher_owned_curriculum_blueprint(
    spec: Mapping[str, Any],
    authority_receipt: Mapping[str, Any],
    *,
    trusted_teacher_public_keys: Mapping[str, bytes | str],
) -> dict[str, Any]:
    """Seal a teacher-authored, already-normalized measurement specification.

    ``spec`` contains every graph/measurement row except ``schema``, ``origin``,
    ``authority``, ``curriculum_id``, and ``integrity``.  It is never repaired or
    completed by a model: missing mappings fail validation.
    """

    draft = _mapping(spec, "teacher curriculum spec")
    expected = {
        "title",
        "lessons",
        "objectives",
        "knowledge_components",
        "prerequisite_edges",
        "topological_order",
        "source_spans",
        "factual_claims",
        "rubrics",
        "item_blueprints",
        "remediation_branches",
        "delayed_reviews",
    }
    _strict_keys(draft, expected, "teacher curriculum spec")
    spec_hash = sha256(_canonical_json(draft)).hexdigest()
    receipt = verify_teacher_curriculum_authority_receipt(
        authority_receipt,
        draft,
        trusted_teacher_public_keys=trusted_teacher_public_keys,
    )
    material = {
        "schema": CURRICULUM_BLUEPRINT_SCHEMA,
        "title": deepcopy(draft["title"]),
        "origin": {
            "kind": "teacher_owned_spec",
            "source_schema": "teaching_skill_miner.teacher_curriculum_spec.v1",
            "source_id": str(receipt["receipt_id"]),
            "source_sha256": spec_hash,
        },
        "lessons": deepcopy(draft["lessons"]),
        "objectives": deepcopy(draft["objectives"]),
        "knowledge_components": deepcopy(draft["knowledge_components"]),
        "prerequisite_edges": deepcopy(draft["prerequisite_edges"]),
        "topological_order": deepcopy(draft["topological_order"]),
        "source_spans": deepcopy(draft["source_spans"]),
        "factual_claims": deepcopy(draft["factual_claims"]),
        "rubrics": deepcopy(draft["rubrics"]),
        "item_blueprints": deepcopy(draft["item_blueprints"]),
        "remediation_branches": deepcopy(draft["remediation_branches"]),
        "delayed_reviews": deepcopy(draft["delayed_reviews"]),
        "authority": {
            "status": "teacher_owned_authoritative",
            "authority": True,
            "authoritative_for_runtime_grading": True,
            "receipt": deepcopy(dict(receipt)),
        },
    }
    return _seal_blueprint(material)


# Clear compatibility alias: migration never modifies the stored v1 document.
migrate_legacy_syllabus_to_curriculum_blueprint = derive_generated_curriculum_blueprint


__all__ = [
    "CURRICULUM_BLUEPRINT_SCHEMA",
    "TEACHER_AUTHORITY_RECEIPT_SCHEMA",
    "CurriculumBlueprintError",
    "TeachingCurriculumBlueprintStore",
    "build_teacher_owned_curriculum_spec",
    "create_teacher_curriculum_authority_receipt",
    "derive_generated_curriculum_blueprint",
    "migrate_legacy_syllabus_to_curriculum_blueprint",
    "seal_teacher_owned_curriculum_blueprint",
    "stable_curriculum_lesson_id",
    "stable_curriculum_objective_id",
    "stable_curriculum_source_span_id",
    "validate_curriculum_blueprint",
    "verify_teacher_curriculum_authority_receipt",
    "verify_teacher_curriculum_runtime_authority",
]
