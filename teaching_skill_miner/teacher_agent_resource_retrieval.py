"""Deterministic, bounded retrieval over teacher-imported resources.

The index deliberately contains offsets and hashes, not a second copy of the
resource text.  Raw media never reaches this module.  Retrieval slices only the
locally extracted, bounded ``extracted_text`` field and returns small excerpts
with provenance.  The lexical scorer is useful without an embedding service;
The retrieval pipeline is deliberately provider-neutral: a caller may attach
an embedding model, query expander, reranker, and claim assessor, but every
provider must declare its execution/privacy properties.  Remote processing is
fail-closed unless the caller explicitly authorizes it.  When no embedding is
configured the receipt says so and the complete lexical path remains usable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from threading import RLock
from typing import Any, Mapping, Protocol, Sequence


RESOURCE_CHUNK_INDEX_SCHEMA = "teaching_skill_miner.teaching_resource_chunk_index.v1"
RESOURCE_RETRIEVAL_SCHEMA = "teaching_skill_miner.resource_retrieval.v1"
RESOURCE_STORE_DOCUMENT_SCHEMA = "teaching_skill_miner.resource_store_document.v1"
MAX_CHUNK_CHARS = 1_200
MAX_CHUNKS_PER_RESOURCE = 256
MAX_RETRIEVAL_RESULTS = 6
MAX_RETRIEVAL_TOTAL_CHARS = 4_800
MAX_RETRIEVAL_QUERY_CHARS = 300
MAX_QUERY_EXPANSIONS = 4
MAX_QUERY_EXPANSION_CHARS = 120
MAX_INDEXED_RESOURCE_TEXT_CHARS = 240_000
MAX_RETRIEVAL_RESOURCES = 6
MAX_RETRIEVAL_CANDIDATES = MAX_RETRIEVAL_RESOURCES * MAX_CHUNKS_PER_RESOURCE
MAX_PROVIDER_CANDIDATES = 192
MAX_PROVIDER_TEXT_CHARS = 120_000
MAX_PROVIDER_CALLS = 4
MAX_RERANK_CANDIDATES = 48
MAX_QUERY_CONTEXT_TERMS = 24
MAX_QUERY_CONTEXT_CHARS = 2_400
_RRF_K = 60

_PAGE_MARKER = re.compile(r"(?m)^\[第\s*(\d+)\s*页\]\s*$")
_NOTES_MARKER = re.compile(r"(?m)^\[讲者备注\]\s*$")
_VISUAL_REVIEW_MARKER = re.compile(r"(?m)^\[视觉复核：[^\]]+\]\s*$")
_LATIN_TERM = re.compile(r"[a-zA-Z0-9_]+(?:[.+#-][a-zA-Z0-9_]+)*")
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


class ResourceRetrievalError(ValueError):
    """Raised when a resource index or retrieval request fails closed."""


class RetrievalProvider(Protocol):
    """Privacy/determinism declaration shared by text-processing providers."""

    provider_id: str
    execution_scope: str
    sends_source_text_off_device: bool
    deterministic: bool


class VectorScoreProvider(RetrievalProvider, Protocol):
    """Optional similarity scorer over bounded extracted-text excerpts."""

    def score(self, query: str, excerpts: Sequence[str]) -> Sequence[float]: ...


class EmbeddingProvider(RetrievalProvider, Protocol):
    """Pluggable embedding provider returning one vector per bounded string."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class RerankScoreProvider(RetrievalProvider, Protocol):
    """Pluggable reranker over a bounded hybrid candidate set."""

    def score(self, query: str, excerpts: Sequence[str]) -> Sequence[float]: ...


class QueryExpansionProvider(RetrievalProvider, Protocol):
    """Pluggable expansion provider; it receives only query/context strings."""

    def expand(self, query: str, context_terms: Sequence[str]) -> Sequence[str]: ...


class ClaimAssessmentProvider(RetrievalProvider, Protocol):
    """Optional local claim checker for contradiction-aware synthesis.

    The provider receives extracted text only.  Its result is advisory for
    teaching synthesis and is never accepted as learner-answer or grading
    evidence.
    """

    def assess(self, query: str, excerpts: Sequence[str]) -> Sequence[str]: ...


@dataclass(frozen=True)
class _ProviderManifest:
    capability: str
    provider_id: str
    execution_scope: str
    sends_source_text_off_device: bool
    deterministic: bool

    def receipt(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "provider_id": self.provider_id,
            "execution_scope": self.execution_scope,
            "sends_source_text_off_device": self.sends_source_text_off_device,
            "deterministic": self.deterministic,
        }


class LocalHashingVectorScoreProvider:
    """Dependency-free, deterministic local vector similarity.

    This is deliberately described as a hashing vector rather than a semantic
    embedding model.  It improves mixed Chinese/Latin and paraphrased-token
    retrieval without sending teacher material to a remote embedding service.
    A stronger local embedding implementation can still be supplied through
    ``VectorScoreProvider``.
    """

    provider_id = "local_hashing_vector_sha256_v1"
    execution_scope = "local"
    sends_source_text_off_device = False
    deterministic = True

    def __init__(self, *, dimensions: int = 1_024) -> None:
        if not isinstance(dimensions, int) or not 128 <= dimensions <= 8_192:
            raise ResourceRetrievalError("hashing vector dimensions are invalid")
        self.dimensions = dimensions

    def _vector(self, value: str) -> dict[int, float]:
        features = _terms(value)
        counts = Counter(features)
        vector: dict[int, float] = {}
        for feature, count in counts.items():
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] = vector.get(index, 0.0) + sign * math.log1p(count)
        norm = math.sqrt(sum(item * item for item in vector.values()))
        if norm:
            return {index: item / norm for index, item in vector.items()}
        return {}

    def score(self, query: str, excerpts: Sequence[str]) -> Sequence[float]:
        query_vector = self._vector(str(query))
        scores: list[float] = []
        for excerpt in excerpts:
            vector = self._vector(str(excerpt))
            scores.append(
                sum(
                    query_value * vector.get(index, 0.0)
                    for index, query_value in query_vector.items()
                )
            )
        return scores


class LocalHashingEmbeddingProvider(LocalHashingVectorScoreProvider):
    """Embedding-shaped adapter for the deterministic local hashing vectors."""

    provider_id = "local_hashing_embedding_sha256_v1"

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            sparse = self._vector(str(text))
            dense = [0.0] * self.dimensions
            for index, value in sparse.items():
                dense[index] = value
            vectors.append(dense)
        return vectors


class DeterministicTermRerankProvider:
    """Local term-coverage/proximity reranker used when no model is supplied."""

    provider_id = "local_term_coverage_proximity_v1"
    execution_scope = "local"
    sends_source_text_off_device = False
    deterministic = True

    def score(self, query: str, excerpts: Sequence[str]) -> Sequence[float]:
        query_terms = list(dict.fromkeys(_terms(query)))
        query_set = set(query_terms)
        scores: list[float] = []
        for excerpt in excerpts:
            document_terms = _terms(str(excerpt))
            document_set = set(document_terms)
            coverage = len(query_set & document_set) / max(1, len(query_set))
            positions = [
                index
                for index, term in enumerate(document_terms)
                if term in query_set
            ]
            span = (
                max(positions) - min(positions) + 1 if len(positions) > 1 else 1
            )
            proximity = min(1.0, len(positions) / max(1, span))
            exact = 1.0 if query.casefold() in str(excerpt).casefold() else 0.0
            scores.append(4.0 * coverage + proximity + 2.0 * exact)
        return scores


class StaticQueryExpansionProvider:
    """Deterministic local synonym/teacher-vocabulary expansion provider."""

    provider_id = "local_static_query_expansion_v1"
    execution_scope = "local"
    sends_source_text_off_device = False
    deterministic = True

    def __init__(self, vocabulary: Mapping[str, Sequence[str]]) -> None:
        normalized: dict[str, tuple[str, ...]] = {}
        if not isinstance(vocabulary, Mapping) or len(vocabulary) > 256:
            raise ResourceRetrievalError("query expansion vocabulary is invalid")
        for raw_key, raw_values in vocabulary.items():
            key = re.sub(r"\s+", " ", str(raw_key)).strip()
            values = _validated_query_expansions(raw_values)
            if not key or len(key) > MAX_QUERY_EXPANSION_CHARS or not values:
                raise ResourceRetrievalError("query expansion vocabulary is invalid")
            normalized[key.casefold()] = tuple(values)
        self._vocabulary = normalized

    def expand(self, query: str, context_terms: Sequence[str]) -> Sequence[str]:
        haystack = " ".join([query, *[str(item) for item in context_terms]]).casefold()
        result: list[str] = []
        for key in sorted(self._vocabulary):
            if key not in haystack:
                continue
            for expansion in self._vocabulary[key]:
                if expansion not in result:
                    result.append(expansion)
                if len(result) >= MAX_QUERY_EXPANSIONS:
                    return result
        return result


class TeachingResourceIndexStore:
    """Private atomic JSON store, deduplicated by raw-file content hash.

    Each immutable resource lives in its own hash-named document, so concurrent
    writers cannot lose unrelated entries.  The stored descriptor contains
    bounded extracted text and its offset index, never raw media.  This class
    intentionally has no global default path: callers must choose a private
    project/session-owned directory.
    """

    _MAX_DOCUMENT_BYTES = 2_000_000
    _HASH = re.compile(r"[0-9a-f]{64}")
    _STAGE_ID = re.compile(r"stage_([0-9a-f]{24})")

    def __init__(self, root: str | Path) -> None:
        candidate = Path(root)
        if candidate.exists() and candidate.is_symlink():
            raise ResourceRetrievalError("resource store root cannot be a symlink")
        candidate.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            candidate.chmod(0o700)
        except OSError as exc:
            raise ResourceRetrievalError(
                "resource store permissions are invalid"
            ) from exc
        self.root = candidate.resolve()
        self._lock = RLock()

    def _path(self, content_sha256: str) -> Path:
        if self._HASH.fullmatch(str(content_sha256)) is None:
            raise ResourceRetrievalError("resource store content hash is invalid")
        return self.root / f"{content_sha256}.json"

    @staticmethod
    def _validated_resource(resource: Mapping[str, Any]) -> dict[str, Any]:
        # Lazy import avoids a module cycle: extraction builds the chunk index.
        from .teacher_agent_resources import validate_teaching_resources

        candidate = json.loads(
            json.dumps(dict(resource), ensure_ascii=False, sort_keys=True)
        )
        validate_teaching_resources([candidate])
        if candidate.get("raw_media_retained") is not False:
            raise ResourceRetrievalError("resource store refuses raw media")
        return candidate

    def _decode_document(self, path: Path) -> dict[str, Any]:
        if path.is_symlink() or not path.is_file():
            raise ResourceRetrievalError("resource store document is invalid")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ResourceRetrievalError(
                "resource store document cannot be read"
            ) from exc
        if not raw or len(raw) > self._MAX_DOCUMENT_BYTES:
            raise ResourceRetrievalError("resource store document size is invalid")
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResourceRetrievalError(
                "resource store document JSON is invalid"
            ) from exc
        if (
            not isinstance(document, Mapping)
            or document.get("schema") != RESOURCE_STORE_DOCUMENT_SCHEMA
            or document.get("content_sha256") != path.stem
            or not isinstance(document.get("resource"), Mapping)
            or not isinstance(document.get("indexed_text"), str)
            or not isinstance(document.get("retrieval_index"), Mapping)
            or not isinstance(document.get("indexed_text_truncated"), bool)
            or not isinstance(document.get("indexed_char_count"), int)
        ):
            raise ResourceRetrievalError("resource store document contract is invalid")
        resource = self._validated_resource(document["resource"])
        if resource.get("content_sha256") != path.stem:
            raise ResourceRetrievalError(
                "resource store document hash binding is invalid"
            )
        indexed_text = str(document["indexed_text"])
        if (
            not indexed_text.strip()
            or len(indexed_text) > MAX_INDEXED_RESOURCE_TEXT_CHARS
            or not indexed_text.startswith(str(resource.get("extracted_text", "")))
            or document.get("indexed_char_count") != len(indexed_text)
            or document.get("indexed_text_truncated")
            != (
                int(resource.get("original_extracted_char_count", len(indexed_text)))
                > len(indexed_text)
            )
        ):
            raise ResourceRetrievalError("resource store indexed text is invalid")
        validate_resource_chunk_index(
            document["retrieval_index"],
            resource_id=str(resource["resource_id"]),
            resource_content_sha256=str(resource["content_sha256"]),
            extracted_text=indexed_text,
        )
        return {
            "schema": RESOURCE_STORE_DOCUMENT_SCHEMA,
            "content_sha256": path.stem,
            "resource": resource,
            "indexed_text": indexed_text,
            "indexed_char_count": len(indexed_text),
            "indexed_text_truncated": bool(document["indexed_text_truncated"]),
            "retrieval_index": dict(document["retrieval_index"]),
        }

    def put(
        self, resource: Mapping[str, Any], *, indexed_text: str | None = None
    ) -> dict[str, Any]:
        """Persist once and return the canonical deduplicated descriptor."""

        candidate = self._validated_resource(resource)
        if indexed_text is None and candidate.get("truncated") is True:
            raise ResourceRetrievalError(
                "truncated resource requires the longer local indexed text"
            )
        content_hash = str(candidate["content_sha256"])
        full_text = str(
            indexed_text
            if indexed_text is not None
            else candidate.get("extracted_text", "")
        )
        if (
            not full_text.strip()
            or len(full_text) > MAX_INDEXED_RESOURCE_TEXT_CHARS
            or not full_text.startswith(str(candidate.get("extracted_text", "")))
        ):
            raise ResourceRetrievalError("resource indexed text is invalid")
        retrieval_index = build_resource_chunk_index(
            resource_id=str(candidate["resource_id"]),
            resource_content_sha256=content_hash,
            resource_type=str(candidate.get("resource_type", "document")),
            extracted_text=full_text,
        )
        path = self._path(content_hash)
        with self._lock:
            if path.exists():
                existing = self._decode_document(path)
                return {
                    "created": False,
                    "resource": existing["resource"],
                    "retrieval_index": existing["retrieval_index"],
                }
            document = {
                "schema": RESOURCE_STORE_DOCUMENT_SCHEMA,
                "content_sha256": content_hash,
                "resource": candidate,
                "indexed_text": full_text,
                "indexed_char_count": len(full_text),
                "indexed_text_truncated": int(
                    candidate.get("original_extracted_char_count", len(full_text))
                )
                > len(full_text),
                "retrieval_index": retrieval_index,
            }
            encoded = json.dumps(
                document,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            if len(encoded) > self._MAX_DOCUMENT_BYTES:
                raise ResourceRetrievalError(
                    "resource store document exceeds its budget"
                )
            temporary_name = ""
            try:
                descriptor, temporary_name = tempfile.mkstemp(
                    dir=self.root, prefix=".resource-", suffix=".tmp"
                )
                with os.fdopen(descriptor, "wb") as handle:
                    os.fchmod(handle.fileno(), 0o600)
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    # Hard-link publication is atomic and never overwrites a
                    # winner from another process.  Same-filesystem temp files
                    # make this a strict content-hash compare-and-create.
                    os.link(temporary_name, path)
                except FileExistsError:
                    os.unlink(temporary_name)
                    temporary_name = ""
                    return {
                        "created": False,
                        "resource": self._decode_document(path)["resource"],
                        "retrieval_index": self._decode_document(path)[
                            "retrieval_index"
                        ],
                    }
                os.unlink(temporary_name)
                temporary_name = ""
                directory_descriptor = os.open(self.root, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            except OSError as exc:
                raise ResourceRetrievalError(
                    "resource store atomic write failed"
                ) from exc
            finally:
                if temporary_name:
                    try:
                        os.unlink(temporary_name)
                    except FileNotFoundError:
                        pass
            return {
                "created": True,
                "resource": candidate,
                "retrieval_index": retrieval_index,
            }

    def get_by_content_hash(self, content_sha256: str) -> dict[str, Any] | None:
        path = self._path(content_sha256)
        with self._lock:
            if not path.exists():
                return None
            return self._decode_document(path)["resource"]

    def get_index_document(self, content_sha256: str) -> dict[str, Any] | None:
        """Load a directly usable, hash-bound retrieval document."""

        path = self._path(content_sha256)
        with self._lock:
            if not path.exists():
                return None
            return self._decode_document(path)

    def get_retrieval_resource(self, content_sha256: str) -> dict[str, Any] | None:
        """Project one stored document into ``retrieve_teaching_resources`` input.

        This projection is not a live-session descriptor: its longer text stays
        local and must only be passed to the bounded retrieval function/tool.
        """

        document = self.get_index_document(content_sha256)
        if document is None:
            return None
        return {
            **dict(document["resource"]),
            "extracted_text": document["indexed_text"],
            "retrieval_index": document["retrieval_index"],
            "retrieval_indexed_char_count": document["indexed_char_count"],
            "retrieval_index_truncated": document["indexed_text_truncated"],
        }

    def get_staged_resource(self, staged_resource_id: str) -> dict[str, Any] | None:
        """Recover the bounded live-session descriptor for a deterministic stage ID.

        A stage ID contains a 96-bit prefix of the immutable raw-file hash.  The
        lookup therefore remains local and content-bound across process restarts.
        Prefix collisions fail closed instead of selecting an arbitrary resource.
        The returned descriptor is the store's validated 12k projection, never
        the longer private ``indexed_text`` used by the retrieval tool.
        """

        match = self._STAGE_ID.fullmatch(str(staged_resource_id))
        if match is None:
            raise ResourceRetrievalError("resource store staged resource ID is invalid")
        prefix = match.group(1)
        with self._lock:
            matches = sorted(
                path
                for path in self.root.glob(f"{prefix}*.json")
                if self._HASH.fullmatch(path.stem) is not None
            )
            if len(matches) > 1:
                raise ResourceRetrievalError(
                    "resource store staged resource ID prefix is ambiguous"
                )
            if not matches:
                return None
            document = self._decode_document(matches[0])
            content_hash = str(document["content_sha256"])
            if f"stage_{content_hash[:24]}" != staged_resource_id:
                raise ResourceRetrievalError(
                    "resource store staged resource ID binding is invalid"
                )
            return dict(document["resource"])

    def get(self, resource_id: str) -> dict[str, Any] | None:
        if re.fullmatch(r"res_[0-9a-f]{20}", str(resource_id)) is None:
            raise ResourceRetrievalError("resource store resource ID is invalid")
        prefix = str(resource_id).removeprefix("res_")
        with self._lock:
            matches = sorted(self.root.glob(f"{prefix}[0-9a-f]*.json"))
            if len(matches) > 1:
                raise ResourceRetrievalError("resource ID prefix is ambiguous")
            if not matches:
                return None
            resource = self._decode_document(matches[0])["resource"]
            if resource.get("resource_id") != resource_id:
                raise ResourceRetrievalError(
                    "resource store resource ID binding is invalid"
                )
            return resource

    def list_metadata(self) -> list[dict[str, Any]]:
        """List descriptors without extracted text; index offsets carry no text."""

        with self._lock:
            resources = [
                self._decode_document(path)["resource"]
                for path in sorted(self.root.glob("[0-9a-f]" * 64 + ".json"))
            ]
        return [
            {key: value for key, value in resource.items() if key != "extracted_text"}
            for resource in resources
        ]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _provider_manifest(
    provider: RetrievalProvider,
    *,
    capability: str,
    receives_source_text: bool,
    allow_remote_processing: bool,
) -> _ProviderManifest:
    provider_id = str(getattr(provider, "provider_id", ""))
    execution_scope = str(getattr(provider, "execution_scope", ""))
    sends_source_text = getattr(provider, "sends_source_text_off_device", None)
    deterministic = getattr(provider, "deterministic", None)
    if (
        re.fullmatch(r"[a-z0-9][a-z0-9._:-]{2,119}", provider_id) is None
        or execution_scope not in {"local", "remote"}
        or not isinstance(sends_source_text, bool)
        or not isinstance(deterministic, bool)
    ):
        raise ResourceRetrievalError(
            f"{capability} provider privacy declaration is invalid"
        )
    if receives_source_text and execution_scope == "remote" and not sends_source_text:
        raise ResourceRetrievalError(
            f"{capability} provider remote text declaration is inconsistent"
        )
    if (execution_scope == "remote" or sends_source_text) and not allow_remote_processing:
        raise ResourceRetrievalError(
            f"{capability} provider requires explicit remote-processing authorization"
        )
    if not deterministic:
        raise ResourceRetrievalError(
            f"{capability} provider must declare deterministic execution"
        )
    return _ProviderManifest(
        capability=capability,
        provider_id=provider_id,
        execution_scope=execution_scope,
        sends_source_text_off_device=sends_source_text,
        deterministic=deterministic,
    )


def _validated_provider_scores(
    values: Sequence[float], *, expected: int, capability: str
) -> list[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ResourceRetrievalError(f"{capability} provider returned invalid scores")
    if len(values) != expected:
        raise ResourceRetrievalError(f"{capability} provider returned invalid scores")
    scores: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise ResourceRetrievalError(
                f"{capability} provider returned invalid scores"
            )
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ResourceRetrievalError(
                f"{capability} provider returned invalid scores"
            ) from exc
        if not math.isfinite(score):
            raise ResourceRetrievalError(
                f"{capability} provider returned invalid scores"
            )
        scores.append(score)
    return scores


def _embedding_scores(
    provider: EmbeddingProvider, query: str, excerpts: Sequence[str]
) -> list[float]:
    try:
        raw_vectors = provider.embed([query, *excerpts])
    except Exception as exc:
        raise ResourceRetrievalError("embedding provider execution failed") from exc
    if isinstance(raw_vectors, (str, bytes)) or not isinstance(
        raw_vectors, Sequence
    ):
        raise ResourceRetrievalError("embedding provider returned invalid vectors")
    if len(raw_vectors) != len(excerpts) + 1:
        raise ResourceRetrievalError("embedding provider returned invalid vectors")
    vectors: list[list[float]] = []
    dimension: int | None = None
    for raw_vector in raw_vectors:
        if isinstance(raw_vector, (str, bytes)) or not isinstance(
            raw_vector, Sequence
        ):
            raise ResourceRetrievalError("embedding provider returned invalid vectors")
        if not raw_vector or len(raw_vector) > 8_192:
            raise ResourceRetrievalError("embedding provider returned invalid vectors")
        vector = _validated_provider_scores(
            raw_vector, expected=len(raw_vector), capability="embedding"
        )
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise ResourceRetrievalError(
                "embedding provider vector dimensions are inconsistent"
            )
        vectors.append(vector)
    query_vector = vectors[0]
    query_norm = math.sqrt(sum(value * value for value in query_vector))
    scores: list[float] = []
    for vector in vectors[1:]:
        norm = math.sqrt(sum(value * value for value in vector))
        if not query_norm or not norm:
            scores.append(0.0)
        else:
            scores.append(
                sum(
                    left * right
                    for left, right in zip(query_vector, vector, strict=True)
                )
                / (query_norm * norm)
            )
    return scores


def _bounded_provider_candidate_positions(
    candidates: Sequence[dict[str, Any]],
    lexical_scores: Sequence[float],
    *,
    reserved_chars: int,
) -> tuple[list[int], bool, int]:
    """Choose a deterministic, relevance-plus-coverage provider batch."""

    lexical_order = sorted(
        range(len(candidates)),
        key=lambda index: (
            -float(lexical_scores[index]),
            str(candidates[index]["chunk"]["chunk_id"]),
        ),
    )
    # Reserve one quarter for stable corpus coverage.  This lets a semantic
    # provider recover candidates with zero lexical overlap without allowing
    # unbounded teacher text to cross the provider boundary.
    coverage_order = sorted(
        range(len(candidates)),
        key=lambda index: _sha256(str(candidates[index]["chunk"]["chunk_id"])),
    )
    priority = [
        *lexical_order[: MAX_PROVIDER_CANDIDATES * 3 // 4],
        *coverage_order,
        *lexical_order,
    ]
    selected: list[int] = []
    selected_set: set[int] = set()
    character_count = 0
    for position in priority:
        if position in selected_set:
            continue
        excerpt_length = len(str(candidates[position]["excerpt"]))
        if (
            len(selected) >= MAX_PROVIDER_CANDIDATES
            or reserved_chars + character_count + excerpt_length
            > MAX_PROVIDER_TEXT_CHARS
        ):
            continue
        selected.append(position)
        selected_set.add(position)
        character_count += excerpt_length
    selected.sort()
    return selected, len(selected) < len(candidates), character_count


def _rank_map(
    values: Mapping[int, float], *, positive_only: bool, minimum_score: float = 0.0
) -> dict[int, int]:
    eligible = [
        (position, score)
        for position, score in values.items()
        if not positive_only or score > minimum_score
    ]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    return {position: rank for rank, (position, _) in enumerate(eligible, 1)}


_NEGATION = re.compile(
    r"(?:不会|不能|不是|并非|未曾|没有|无需|不|非|未|无|\bnot\b|\bno\b|\bnever\b)",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?")


def _conflict_signature(value: str) -> tuple[set[str], bool, tuple[str, ...]]:
    without_negation = _NEGATION.sub(" ", value.casefold())
    without_numbers = _NUMBER.sub(" ", without_negation)
    terms = {
        term
        for term in _terms(without_numbers)
        if len(term) > 1 or term.isascii()
    }
    numbers = tuple(sorted(set(_NUMBER.findall(value))))
    return terms, _NEGATION.search(value) is not None, numbers


def _detect_source_conflicts(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for left_index, left in enumerate(results):
        left_provenance = left.get("provenance", {})
        left_terms, left_negated, left_numbers = _conflict_signature(
            str(left.get("excerpt", ""))
        )
        for right in results[left_index + 1 :]:
            right_provenance = right.get("provenance", {})
            if left_provenance.get("resource_id") == right_provenance.get(
                "resource_id"
            ):
                continue
            right_terms, right_negated, right_numbers = _conflict_signature(
                str(right.get("excerpt", ""))
            )
            union = left_terms | right_terms
            overlap = len(left_terms & right_terms) / len(union) if union else 0.0
            reason = ""
            if overlap >= 0.42 and left_negated != right_negated:
                reason = "matched_statement_opposite_polarity"
            elif (
                overlap >= 0.72
                and left_numbers
                and right_numbers
                and left_numbers != right_numbers
            ):
                reason = "matched_statement_incompatible_numeric_values"
            if not reason:
                continue
            conflicts.append(
                {
                    "left_citation_id": left.get("citation_id"),
                    "right_citation_id": right.get("citation_id"),
                    "left_chunk_id": left_provenance.get("chunk_id"),
                    "right_chunk_id": right_provenance.get("chunk_id"),
                    "reason": reason,
                    "term_overlap": round(overlap, 6),
                    "detector": "deterministic_high_precision_pairwise_v1",
                }
            )
    return conflicts


def _terms(value: str) -> list[str]:
    lowered = value.casefold()
    terms = [match.group(0) for match in _LATIN_TERM.finditer(lowered)]
    for match in _CJK_RUN.finditer(lowered):
        run = match.group(0)
        terms.extend(run)
        if len(run) > 1:
            terms.extend(run[index : index + 2] for index in range(len(run) - 1))
    return terms


def _bounded_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < end and len(spans) < MAX_CHUNKS_PER_RESOURCE:
        while cursor < end and text[cursor].isspace():
            cursor += 1
        if cursor >= end:
            break
        hard_end = min(end, cursor + MAX_CHUNK_CHARS)
        span_end = hard_end
        if hard_end < end:
            minimum_break = cursor + MAX_CHUNK_CHARS // 2
            candidates = [
                text.rfind("\n\n", minimum_break, hard_end),
                text.rfind("\n", minimum_break, hard_end),
                text.rfind("。", minimum_break, hard_end),
                text.rfind("；", minimum_break, hard_end),
                text.rfind(" ", minimum_break, hard_end),
            ]
            best = max(candidates)
            if best >= minimum_break:
                span_end = best + (1 if text[best] in "。；" else 0)
        while span_end > cursor and text[span_end - 1].isspace():
            span_end -= 1
        if span_end <= cursor:
            span_end = hard_end
        spans.append((cursor, span_end))
        cursor = max(span_end, cursor + 1)
    return spans


def _source_sections(
    text: str, resource_type: str
) -> list[tuple[int, int, dict[str, Any], str]]:
    """Return source-aligned ranges: page/slide, then speaker notes."""

    markers = list(_PAGE_MARKER.finditer(text))
    if not markers:
        paragraphs: list[tuple[int, int, dict[str, Any], str]] = []
        for index, match in enumerate(re.finditer(r"[^\n]+", text), 1):
            if match.group(0).strip():
                paragraphs.append(
                    (
                        match.start(),
                        match.end(),
                        {"kind": "paragraph", "index": index},
                        "body",
                    )
                )
        return paragraphs
    location_kind = "slide" if resource_type == "presentation" else "page"
    sections: list[tuple[int, int, dict[str, Any], str]] = []
    for position, marker in enumerate(markers):
        section_start = marker.end()
        section_end = (
            markers[position + 1].start() if position + 1 < len(markers) else len(text)
        )
        index = int(marker.group(1))
        location = {"kind": location_kind, "index": index}
        note_markers = list(_NOTES_MARKER.finditer(text, section_start, section_end))
        if not note_markers:
            sections.append((section_start, section_end, location, "body"))
            continue
        first_note = note_markers[0]
        sections.append((section_start, first_note.start(), location, "body"))
        sections.append((first_note.end(), section_end, location, "speaker_notes"))
    return sections


def build_resource_chunk_index(
    *,
    resource_id: str,
    resource_content_sha256: str,
    resource_type: str,
    extracted_text: str,
    visual_review_locations: Sequence[int] = (),
) -> dict[str, Any]:
    """Build a stable offset index over one already-bounded text descriptor."""

    if not resource_id.startswith("res_") or not re.fullmatch(
        r"[0-9a-f]{64}", resource_content_sha256
    ):
        raise ResourceRetrievalError("resource identity is invalid")
    if not extracted_text.strip():
        raise ResourceRetrievalError("resource text is empty")
    visual_locations = {int(item) for item in visual_review_locations if int(item) > 0}
    chunks: list[dict[str, Any]] = []
    for section_start, section_end, location, content_kind in _source_sections(
        extracted_text, resource_type
    ):
        for start, end in _bounded_spans(extracted_text, section_start, section_end):
            if len(chunks) >= MAX_CHUNKS_PER_RESOURCE:
                break
            excerpt = extracted_text[start:end]
            ordinal = len(chunks) + 1
            chunk_id = f"{resource_id}_c{ordinal:03d}"
            source_ref = (
                f"{resource_id}#{location['kind']}={location['index']}&chunk={ordinal}"
            )
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "start_char": start,
                    "end_char": end,
                    "char_count": len(excerpt),
                    "content_sha256": _sha256(excerpt),
                    "location": dict(location),
                    "content_kind": content_kind,
                    "source_ref": source_ref,
                    "needs_visual_review": (
                        location.get("index") in visual_locations
                        or _VISUAL_REVIEW_MARKER.search(
                            extracted_text[section_start:section_end]
                        )
                        is not None
                    ),
                }
            )
        if len(chunks) >= MAX_CHUNKS_PER_RESOURCE:
            break
    if not chunks:
        raise ResourceRetrievalError("resource text has no indexable content")
    return {
        "schema": RESOURCE_CHUNK_INDEX_SCHEMA,
        "algorithm": "deterministic_source_chunks_v1",
        "text_field": "extracted_text",
        "resource_content_sha256": resource_content_sha256,
        "indexed_char_count": len(extracted_text),
        "chunk_count": len(chunks),
        "vector_index_present": False,
        "chunks": chunks,
    }


def validate_resource_chunk_index(
    index: Any, *, resource_id: str, resource_content_sha256: str, extracted_text: str
) -> None:
    if (
        not isinstance(index, Mapping)
        or index.get("schema") != RESOURCE_CHUNK_INDEX_SCHEMA
    ):
        raise ResourceRetrievalError("resource chunk index schema is invalid")
    if index.get("text_field") != "extracted_text":
        raise ResourceRetrievalError("resource chunk index text field is invalid")
    if (
        index.get("algorithm") != "deterministic_source_chunks_v1"
        or index.get("vector_index_present") is not False
    ):
        raise ResourceRetrievalError("resource chunk index algorithm is invalid")
    if index.get("resource_content_sha256") != resource_content_sha256:
        raise ResourceRetrievalError("resource chunk index hash binding is invalid")
    if index.get("indexed_char_count") != len(extracted_text):
        raise ResourceRetrievalError("resource chunk index length binding is invalid")
    chunks = index.get("chunks")
    if (
        not isinstance(chunks, list)
        or not chunks
        or len(chunks) > MAX_CHUNKS_PER_RESOURCE
    ):
        raise ResourceRetrievalError("resource chunk index chunks are invalid")
    if index.get("chunk_count") != len(chunks):
        raise ResourceRetrievalError("resource chunk index count is invalid")
    previous_end = -1
    for ordinal, chunk in enumerate(chunks, 1):
        if not isinstance(chunk, Mapping):
            raise ResourceRetrievalError("resource chunk is invalid")
        start = chunk.get("start_char")
        end = chunk.get("end_char")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or end > len(extracted_text)
            or start < previous_end
            or end - start > MAX_CHUNK_CHARS
        ):
            raise ResourceRetrievalError("resource chunk offsets are invalid")
        excerpt = extracted_text[start:end]
        if chunk.get("content_sha256") != _sha256(excerpt):
            raise ResourceRetrievalError("resource chunk content hash is invalid")
        if chunk.get("char_count") != len(excerpt):
            raise ResourceRetrievalError("resource chunk character count is invalid")
        if chunk.get("chunk_id") != f"{resource_id}_c{ordinal:03d}":
            raise ResourceRetrievalError("resource chunk ID is invalid")
        location = chunk.get("location")
        if (
            not isinstance(location, Mapping)
            or location.get("kind") not in {"page", "slide", "paragraph"}
            or not isinstance(location.get("index"), int)
            or int(location["index"]) < 1
        ):
            raise ResourceRetrievalError("resource chunk location is invalid")
        if chunk.get("content_kind") not in {"body", "speaker_notes"}:
            raise ResourceRetrievalError("resource chunk content kind is invalid")
        expected_source_ref = (
            f"{resource_id}#{location['kind']}={location['index']}&chunk={ordinal}"
        )
        if (
            chunk.get("source_ref") != expected_source_ref
            or not isinstance(chunk.get("needs_visual_review"), bool)
        ):
            raise ResourceRetrievalError("resource chunk provenance is invalid")
        previous_end = end


def _candidate_chunks(
    resources: Sequence[Mapping[str, Any]],
    resource_ids: set[str] | None,
    *,
    exclude_visual_review_pending: bool,
) -> tuple[list[dict[str, Any]], int, list[tuple[str, str]]]:
    if isinstance(resources, (str, bytes)) or not isinstance(resources, Sequence):
        raise ResourceRetrievalError("resources must be an array")
    if len(resources) > MAX_RETRIEVAL_RESOURCES:
        raise ResourceRetrievalError("too many resources for one retrieval request")
    candidates: list[dict[str, Any]] = []
    excluded_visual_review_count = 0
    corpus_binding: list[tuple[str, str]] = []
    seen_resources: set[tuple[str, str]] = set()
    for resource in resources:
        if not isinstance(resource, Mapping):
            raise ResourceRetrievalError("retrieval resource must be an object")
        resource_id = str(resource.get("resource_id", ""))
        if re.fullmatch(r"res_[0-9a-f]{20}", resource_id) is None:
            raise ResourceRetrievalError("retrieval resource ID is invalid")
        if resource_ids is not None and resource_id not in resource_ids:
            continue
        text = resource.get("extracted_text")
        content_hash = str(resource.get("content_sha256", ""))
        index = resource.get("retrieval_index")
        if (
            not isinstance(text, str)
            or not text.strip()
            or len(text) > MAX_INDEXED_RESOURCE_TEXT_CHARS
            or re.fullmatch(r"[0-9a-f]{64}", content_hash) is None
            or resource.get("raw_media_retained") is not False
        ):
            raise ResourceRetrievalError("retrieval resource contract is invalid")
        identity = (resource_id, content_hash)
        if identity in seen_resources:
            raise ResourceRetrievalError("retrieval resources contain duplicates")
        seen_resources.add(identity)
        corpus_binding.append(identity)
        if index is None:
            index = build_resource_chunk_index(
                resource_id=resource_id,
                resource_content_sha256=content_hash,
                resource_type=str(resource.get("resource_type", "document")),
                extracted_text=text,
            )
        validate_resource_chunk_index(
            index,
            resource_id=resource_id,
            resource_content_sha256=content_hash,
            extracted_text=text,
        )
        for chunk in index["chunks"]:
            if (
                exclude_visual_review_pending
                and chunk.get("needs_visual_review") is True
            ):
                excluded_visual_review_count += 1
                continue
            start, end = chunk["start_char"], chunk["end_char"]
            candidates.append(
                {
                    "resource": resource,
                    "chunk": chunk,
                    "excerpt": text[start:end],
                }
            )
            if len(candidates) > MAX_RETRIEVAL_CANDIDATES:
                raise ResourceRetrievalError("retrieval candidate budget exceeded")
    return candidates, excluded_visual_review_count, sorted(corpus_binding)


def contextual_query_expansions(query: str, context_terms: Sequence[str]) -> list[str]:
    """Select bounded teacher-context terms related to a retrieval query."""

    query_terms = set(_terms(str(query)))
    if not query_terms:
        return []
    expansions: list[str] = []
    for raw in context_terms:
        candidate = re.sub(r"\s+", " ", str(raw)).strip()
        if (
            not candidate
            or len(candidate) > MAX_QUERY_EXPANSION_CHARS
            or candidate == query
        ):
            continue
        candidate_terms = set(_terms(candidate))
        if not query_terms.intersection(candidate_terms):
            continue
        if candidate not in expansions:
            expansions.append(candidate)
        if len(expansions) >= MAX_QUERY_EXPANSIONS:
            break
    return expansions


def _validated_query_expansions(value: Sequence[str]) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ResourceRetrievalError("resource query expansions must be an array")
    if len(value) > MAX_QUERY_EXPANSIONS:
        raise ResourceRetrievalError("too many resource query expansions")
    result: list[str] = []
    for raw in value:
        candidate = re.sub(r"\s+", " ", str(raw)).strip()
        if not candidate or len(candidate) > MAX_QUERY_EXPANSION_CHARS:
            raise ResourceRetrievalError("resource query expansion is invalid")
        if candidate not in result:
            result.append(candidate)
    return result


def _jaccard_terms(left: str, right: str) -> float:
    left_terms = set(_terms(left))
    right_terms = set(_terms(right))
    union = left_terms | right_terms
    if not union:
        return 0.0
    return len(left_terms & right_terms) / len(union)


def _duplicate_similarity(left: str, right: str) -> float:
    """Do not MMR-suppress a likely contradiction as a near duplicate."""

    left_signature, left_negated, left_numbers = _conflict_signature(left)
    right_signature, right_negated, right_numbers = _conflict_signature(right)
    union = left_signature | right_signature
    overlap = (
        len(left_signature & right_signature) / len(union) if union else 0.0
    )
    if overlap >= 0.42 and left_negated != right_negated:
        return 0.0
    if (
        overlap >= 0.72
        and left_numbers
        and right_numbers
        and left_numbers != right_numbers
    ):
        return 0.0
    return _jaccard_terms(left, right)


def _diversified_candidates(
    scored: Sequence[tuple[float, str, dict[str, Any]]], *, maximum: int
) -> list[tuple[float, str, dict[str, Any]]]:
    """Greedy MMR-style selection that keeps source diversity deterministic."""

    remaining = list(scored)
    selected: list[tuple[float, str, dict[str, Any]]] = []
    while remaining and len(selected) < maximum:
        ranked: list[tuple[float, float, str, tuple[float, str, dict[str, Any]]]] = []
        for item in remaining:
            relevance, chunk_id, candidate = item
            duplicate_penalty = max(
                (
                    _duplicate_similarity(
                        str(candidate["excerpt"]), str(chosen[2]["excerpt"])
                    )
                    for chosen in selected
                ),
                default=0.0,
            )
            same_resource_penalty = (
                0.12
                if any(
                    chosen[2]["resource"].get("resource_id")
                    == candidate["resource"].get("resource_id")
                    for chosen in selected
                )
                else 0.0
            )
            diversified_score = (
                relevance - 0.8 * duplicate_penalty - same_resource_penalty
            )
            ranked.append((diversified_score, relevance, chunk_id, item))
        ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))
        winner = ranked[0][3]
        selected.append(winner)
        remaining.remove(winner)
    return selected


def retrieve_teaching_resources(
    resources: Sequence[Mapping[str, Any]],
    query: str,
    *,
    max_results: int = 4,
    max_total_chars: int = 3_600,
    resource_ids: Sequence[str] | None = None,
    vector_scorer: VectorScoreProvider | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    rerank_scorer: RerankScoreProvider | None = None,
    query_expansions: Sequence[str] = (),
    query_expander: QueryExpansionProvider | None = None,
    query_context_terms: Sequence[str] = (),
    claim_assessor: ClaimAssessmentProvider | None = None,
    exclude_visual_review_pending: bool = True,
    allow_remote_processing: bool = False,
) -> dict[str, Any]:
    """Return bounded excerpts and a hash-bound, privacy-aware pipeline receipt.

    Raw media is never passed to any provider.  Providers see at most bounded
    extracted-text excerpts and must declare deterministic execution.  The
    legacy ``vector_scorer`` extension remains supported; new integrations
    should prefer ``embedding_provider`` so vector shape can be validated here.
    """

    query = str(query).strip()
    if not query or len(query) > MAX_RETRIEVAL_QUERY_CHARS:
        raise ResourceRetrievalError("resource query length is invalid")
    if not 1 <= int(max_results) <= MAX_RETRIEVAL_RESULTS:
        raise ResourceRetrievalError("resource result limit is invalid")
    if not 200 <= int(max_total_chars) <= MAX_RETRIEVAL_TOTAL_CHARS:
        raise ResourceRetrievalError("resource character budget is invalid")
    if not isinstance(exclude_visual_review_pending, bool):
        raise ResourceRetrievalError("visual-review exclusion policy is invalid")
    if not isinstance(allow_remote_processing, bool):
        raise ResourceRetrievalError("remote-processing policy is invalid")
    if vector_scorer is not None and embedding_provider is not None:
        raise ResourceRetrievalError(
            "configure either vector_scorer or embedding_provider, not both"
        )
    selected_ids = None
    if resource_ids is not None:
        if (
            isinstance(resource_ids, (str, bytes))
            or not isinstance(resource_ids, Sequence)
            or len(resource_ids) > MAX_RETRIEVAL_RESOURCES
            or any(
                re.fullmatch(r"res_[0-9a-f]{20}", str(item)) is None
                for item in resource_ids
            )
            or len({str(item) for item in resource_ids}) != len(resource_ids)
        ):
            raise ResourceRetrievalError("resource ID filter is invalid")
        selected_ids = {str(item) for item in resource_ids}
    expansions = _validated_query_expansions(query_expansions)
    if isinstance(query_context_terms, (str, bytes)) or not isinstance(
        query_context_terms, Sequence
    ):
        raise ResourceRetrievalError("resource query context must be an array")
    if len(query_context_terms) > MAX_QUERY_CONTEXT_TERMS:
        raise ResourceRetrievalError("resource query context exceeds its bound")
    normalized_context: list[str] = []
    for raw in query_context_terms:
        context = re.sub(r"\s+", " ", str(raw)).strip()
        if not context or len(context) > MAX_QUERY_EXPANSION_CHARS:
            raise ResourceRetrievalError("resource query context is invalid")
        normalized_context.append(context)
    if sum(len(item) for item in normalized_context) > MAX_QUERY_CONTEXT_CHARS:
        raise ResourceRetrievalError("resource query context exceeds its budget")

    provider_calls: list[dict[str, Any]] = []

    def record_provider_call(
        manifest: _ProviderManifest,
        *,
        input_item_count: int,
        input_char_count: int,
        source_text_input: bool,
    ) -> None:
        if (
            input_item_count < 1
            or input_char_count < 1
            or input_char_count > MAX_PROVIDER_TEXT_CHARS
            or len(provider_calls) >= MAX_PROVIDER_CALLS
        ):
            raise ResourceRetrievalError("retrieval provider call budget exceeded")
        provider_calls.append(
            {
                **manifest.receipt(),
                "input_item_count": input_item_count,
                "input_char_count": input_char_count,
                "source_text_input": source_text_input,
            }
        )

    configured_provider_manifests: list[_ProviderManifest] = []
    if query_expander is not None:
        expansion_manifest = _provider_manifest(
            query_expander,
            capability="query_expansion",
            receives_source_text=False,
            allow_remote_processing=allow_remote_processing,
        )
        configured_provider_manifests.append(expansion_manifest)
        try:
            raw_generated = query_expander.expand(query, tuple(normalized_context))
        except Exception as exc:
            raise ResourceRetrievalError(
                "query expansion provider execution failed"
            ) from exc
        generated = _validated_query_expansions(raw_generated)
        expansions = list(dict.fromkeys([*expansions, *generated]))[
            :MAX_QUERY_EXPANSIONS
        ]
        record_provider_call(
            expansion_manifest,
            input_item_count=1 + len(normalized_context),
            input_char_count=len(query) + sum(map(len, normalized_context)),
            source_text_input=False,
        )
    vector_provider = embedding_provider or vector_scorer
    vector_capability = "embedding" if embedding_provider is not None else "vector_score"
    vector_manifest: _ProviderManifest | None = None
    if vector_provider is not None:
        vector_manifest = _provider_manifest(
            vector_provider,
            capability=vector_capability,
            receives_source_text=True,
            allow_remote_processing=allow_remote_processing,
        )
        configured_provider_manifests.append(vector_manifest)
    reranker: RerankScoreProvider = rerank_scorer or DeterministicTermRerankProvider()
    rerank_manifest = _provider_manifest(
        reranker,
        capability="rerank",
        receives_source_text=True,
        allow_remote_processing=allow_remote_processing,
    )
    configured_provider_manifests.append(rerank_manifest)
    claim_manifest: _ProviderManifest | None = None
    if claim_assessor is not None:
        claim_manifest = _provider_manifest(
            claim_assessor,
            capability="claim_assessment",
            receives_source_text=True,
            allow_remote_processing=allow_remote_processing,
        )
        configured_provider_manifests.append(claim_manifest)
    effective_query = " ".join([query, *expansions])
    candidates, excluded_visual_review_count, corpus_binding = _candidate_chunks(
        resources,
        selected_ids,
        exclude_visual_review_pending=exclude_visual_review_pending,
    )
    query_terms = _terms(effective_query)
    documents = [_terms(str(item["excerpt"])) for item in candidates]
    document_frequency: Counter[str] = Counter()
    for terms in documents:
        document_frequency.update(set(terms))
    average_length = max(
        1.0,
        sum(len(item) for item in documents) / len(documents) if documents else 1.0,
    )
    query_counts = Counter(query_terms)

    lexical_scores: list[float] = []
    for candidate, terms in zip(candidates, documents, strict=True):
        counts = Counter(terms)
        length = max(1, len(terms))
        score = 0.0
        for term, query_weight in query_counts.items():
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            inverse_document_frequency = math.log(
                1.0
                + (len(documents) - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            denominator = frequency + 1.2 * (0.25 + 0.75 * length / average_length)
            score += (
                query_weight
                * inverse_document_frequency
                * (frequency * 2.2 / denominator)
            )
        excerpt = str(candidate["excerpt"])
        if query.casefold() in excerpt.casefold():
            score += 3.0
        # A one-character CJK overlap between a filename and query is too weak
        # to admit an otherwise unrelated chunk into the candidate set.
        title_terms = {
            term
            for term in _terms(str(candidate["resource"].get("display_name", "")))
            if len(term) > 1
        }
        meaningful_query_terms = {term for term in query_terms if len(term) > 1}
        score += 0.25 * len(meaningful_query_terms.intersection(title_terms))
        lexical_scores.append(score)

    vector_score_map: dict[int, float] = {}
    provider_candidate_count = 0
    provider_candidate_chars = 0
    provider_candidates_truncated = False
    if vector_provider is not None and candidates:
        assert vector_manifest is not None
        positions, provider_candidates_truncated, provider_candidate_chars = (
            _bounded_provider_candidate_positions(
                candidates,
                lexical_scores,
                reserved_chars=len(effective_query),
            )
        )
        excerpts = [str(candidates[position]["excerpt"]) for position in positions]
        provider_candidate_count = len(positions)
        if embedding_provider is not None:
            provided_scores = _embedding_scores(
                embedding_provider, effective_query, excerpts
            )
        else:
            assert vector_scorer is not None
            try:
                raw_vector_scores = vector_scorer.score(
                    effective_query, tuple(excerpts)
                )
            except Exception as exc:
                raise ResourceRetrievalError(
                    "vector score provider execution failed"
                ) from exc
            provided_scores = _validated_provider_scores(
                raw_vector_scores,
                expected=len(excerpts),
                capability="vector score",
            )
        vector_score_map = dict(zip(positions, provided_scores, strict=True))
        record_provider_call(
            vector_manifest,
            input_item_count=1 + len(excerpts),
            input_char_count=len(effective_query) + provider_candidate_chars,
            source_text_input=True,
        )

    lexical_ranks = _rank_map(
        {index: score for index, score in enumerate(lexical_scores)},
        positive_only=True,
    )
    vector_ranks = _rank_map(
        vector_score_map,
        positive_only=True,
        # Cosine/hash collisions around zero must not turn lexically unrelated
        # chunks into results merely because their floating-point score is
        # microscopically positive.
        minimum_score=0.08,
    )
    fusion_scores: dict[int, float] = {}
    for position in sorted(set(lexical_ranks) | set(vector_ranks)):
        score = 0.0
        if position in lexical_ranks:
            lexical_weight = 0.65 if vector_ranks else 1.0
            score += lexical_weight / (_RRF_K + lexical_ranks[position])
        if position in vector_ranks:
            score += 0.35 / (_RRF_K + vector_ranks[position])
        fusion_scores[position] = score

    fusion_order = sorted(
        fusion_scores,
        key=lambda position: (
            -fusion_scores[position],
            str(candidates[position]["chunk"]["chunk_id"]),
        ),
    )[:MAX_RERANK_CANDIDATES]
    rerank_score_map: dict[int, float] = {}
    if fusion_order:
        rerank_excerpts = [
            str(candidates[position]["excerpt"]) for position in fusion_order
        ]
        rerank_char_count = sum(map(len, rerank_excerpts))
        if len(effective_query) + rerank_char_count > MAX_PROVIDER_TEXT_CHARS:
            raise ResourceRetrievalError("rerank provider text budget exceeded")
        try:
            raw_rerank_scores = reranker.score(
                effective_query, tuple(rerank_excerpts)
            )
        except Exception as exc:
            raise ResourceRetrievalError("rerank provider execution failed") from exc
        rerank_scores = _validated_provider_scores(
            raw_rerank_scores,
            expected=len(rerank_excerpts),
            capability="rerank",
        )
        rerank_score_map = dict(zip(fusion_order, rerank_scores, strict=True))
        record_provider_call(
            rerank_manifest,
            input_item_count=1 + len(rerank_excerpts),
            input_char_count=len(effective_query) + rerank_char_count,
            source_text_input=True,
        )
    rerank_ranks = _rank_map(rerank_score_map, positive_only=False)
    fusion_ranks = {
        position: rank for rank, position in enumerate(fusion_order, 1)
    }
    final_scores: dict[int, float] = {}
    for position in fusion_order:
        score = 0.8 / (_RRF_K + fusion_ranks[position])
        if position in rerank_ranks:
            score += 0.2 / (_RRF_K + rerank_ranks[position])
        final_scores[position] = score * 100.0

    scored = [
        (
            final_scores[position],
            str(candidates[position]["chunk"]["chunk_id"]),
            {**candidates[position], "_candidate_position": position},
        )
        for position in fusion_order
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    diversified = _diversified_candidates(scored, maximum=int(max_results))

    results: list[dict[str, Any]] = []
    remaining = int(max_total_chars)
    for score, _, candidate in diversified:
        if score <= 0 or len(results) >= int(max_results) or remaining <= 0:
            break
        full_excerpt = str(candidate["excerpt"])
        excerpt = full_excerpt
        if len(excerpt) > remaining:
            excerpt = excerpt[:remaining].rstrip()
        if not excerpt:
            continue
        resource = candidate["resource"]
        chunk = candidate["chunk"]
        position = int(candidate["_candidate_position"])
        excerpt_hash = _sha256(excerpt)
        citation_id = "cite_" + _sha256(
            "\n".join(
                [
                    str(resource.get("content_sha256")),
                    str(chunk.get("chunk_id")),
                    excerpt_hash,
                ]
            )
        )[:24]
        results.append(
            {
                "citation_id": citation_id,
                "excerpt": excerpt,
                "score": round(score, 6),
                "score_components": {
                    "bm25": round(lexical_scores[position], 6),
                    "vector_similarity": (
                        round(vector_score_map[position], 6)
                        if position in vector_score_map
                        else None
                    ),
                    "rank_fusion": round(fusion_scores[position], 9),
                    "rerank": round(rerank_score_map[position], 6),
                },
                "provenance": {
                    "resource_id": resource.get("resource_id"),
                    "display_name": resource.get("display_name"),
                    "resource_content_sha256": resource.get("content_sha256"),
                    "chunk_id": chunk.get("chunk_id"),
                    "chunk_content_sha256": chunk.get("content_sha256"),
                    "excerpt_content_sha256": excerpt_hash,
                    "excerpt_is_partial": len(excerpt) != len(full_excerpt),
                    "chunk_start_char": chunk.get("start_char"),
                    "chunk_end_char": chunk.get("end_char"),
                    "excerpt_start_char": chunk.get("start_char"),
                    "excerpt_end_char": int(chunk.get("start_char", 0))
                    + len(excerpt),
                    "location": dict(chunk.get("location", {})),
                    "content_kind": chunk.get("content_kind"),
                    "source_ref": chunk.get("source_ref"),
                    "extraction_engine": resource.get("extraction_engine"),
                    "needs_visual_review": chunk.get("needs_visual_review") is True,
                    "representation": "bounded_local_extracted_text",
                    "retrieval_index_truncated": resource.get(
                        "retrieval_index_truncated"
                    )
                    is True,
                    "content_is_untrusted_instruction_data": True,
                },
                "eligible_for_grading": False,
                "eligible_for_synthesis": chunk.get("needs_visual_review") is False,
            }
        )
        remaining -= len(excerpt)
    claim_relations = ["not_verified"] * len(results)
    if claim_assessor is not None and results:
        assert claim_manifest is not None
        try:
            assessed = claim_assessor.assess(
                query, tuple(str(item["excerpt"]) for item in results)
            )
        except Exception as exc:
            raise ResourceRetrievalError(
                "claim assessment provider execution failed"
            ) from exc
        if len(assessed) != len(results) or any(
            item not in {"entails", "contradicts", "irrelevant", "uncertain"}
            for item in assessed
        ):
            raise ResourceRetrievalError("claim assessor returned invalid relations")
        claim_relations = [str(item) for item in assessed]
        record_provider_call(
            claim_manifest,
            input_item_count=1 + len(results),
            input_char_count=len(query)
            + sum(len(str(item["excerpt"])) for item in results),
            source_text_input=True,
        )
    for result, relation in zip(results, claim_relations, strict=True):
        result["claim_relation"] = relation
    source_conflicts = _detect_source_conflicts(results)
    has_entailment = "entails" in claim_relations
    has_contradiction = "contradicts" in claim_relations
    if has_entailment and has_contradiction:
        entailing = [item for item in results if item["claim_relation"] == "entails"]
        contradicting = [
            item for item in results if item["claim_relation"] == "contradicts"
        ]
        known_pairs = {
            (item["left_citation_id"], item["right_citation_id"])
            for item in source_conflicts
        }
        for left in entailing:
            for right in contradicting:
                pair = (left["citation_id"], right["citation_id"])
                if pair in known_pairs:
                    continue
                source_conflicts.append(
                    {
                        "left_citation_id": left["citation_id"],
                        "right_citation_id": right["citation_id"],
                        "left_chunk_id": left["provenance"]["chunk_id"],
                        "right_chunk_id": right["provenance"]["chunk_id"],
                        "reason": "claim_assessor_opposite_relations",
                        "term_overlap": None,
                        "detector": str(
                            getattr(claim_assessor, "provider_id", "claim_assessor")
                        ),
                    }
                )
        consistency = "conflicting_sources"
    elif source_conflicts:
        consistency = "conflicting_sources"
    elif has_contradiction:
        consistency = "query_contradicted"
    elif has_entailment:
        consistency = "query_supported"
    elif claim_assessor is not None:
        consistency = "uncertain"
    else:
        consistency = "not_verified"
    visual_pending_returned = sum(
        item["provenance"]["needs_visual_review"] is True for item in results
    )
    safe_to_quote = bool(results) and visual_pending_returned == 0
    safe_to_synthesize = (
        consistency == "query_supported"
        and bool(results)
        and not source_conflicts
        and visual_pending_returned == 0
    )
    abstention_reasons: list[str] = []
    if not results:
        abstention_reasons.append("no_retrieval_result")
    if consistency == "not_verified":
        abstention_reasons.append("claim_not_verified")
    elif consistency == "uncertain":
        abstention_reasons.append("claim_assessment_uncertain")
    elif consistency == "query_contradicted":
        abstention_reasons.append("query_contradicted_by_sources")
    elif consistency == "conflicting_sources":
        abstention_reasons.append("conflicting_sources")
    if visual_pending_returned:
        abstention_reasons.append("visual_review_pending")

    query_hash = _sha256(query)
    claim_id = "claim_" + query_hash[:24]
    citations = [
        {
            "citation_id": item["citation_id"],
            "resource_id": item["provenance"]["resource_id"],
            "resource_content_sha256": item["provenance"][
                "resource_content_sha256"
            ],
            "chunk_id": item["provenance"]["chunk_id"],
            "chunk_content_sha256": item["provenance"]["chunk_content_sha256"],
            "excerpt_content_sha256": item["provenance"][
                "excerpt_content_sha256"
            ],
            "source_ref": item["provenance"]["source_ref"],
            "relation": item["claim_relation"],
            "needs_visual_review": item["provenance"]["needs_visual_review"],
        }
        for item in results
    ]
    claim_trace = {
        "claim_id": claim_id,
        "claim_sha256": query_hash,
        "assessment_scope": "returned_bounded_excerpts",
        "assessment_complete": claim_assessor is not None,
        "consistency_status": consistency,
        "decision": "supported" if safe_to_synthesize else "abstain",
        "abstention_reasons": abstention_reasons,
        "citations": citations,
    }
    external_processing_used = any(
        call["execution_scope"] == "remote"
        or call["sends_source_text_off_device"] is True
        for call in provider_calls
    )
    receipt = {
        "schema": RESOURCE_RETRIEVAL_SCHEMA,
        "query_sha256": query_hash,
        "retrieval_method": (
            "bm25_lexical_plus_vector_v1"
            if vector_score_map
            else "bm25_lexical_v1"
        ),
        "pipeline_method": "bounded_hybrid_rrf_rerank_v2",
        "lexical_fallback_used": vector_provider is None,
        "vector_scores_used": bool(vector_score_map),
        "embedding_scores_used": embedding_provider is not None
        and bool(vector_score_map),
        "fusion_method": (
            "weighted_reciprocal_rank_fusion_v1"
            if vector_score_map
            else "lexical_rank_v1"
        ),
        "rerank_method": str(getattr(reranker, "provider_id", "")),
        "query_expansions": expansions,
        "query_expansions_sha256": _sha256("\n".join(expansions)),
        "diversification_method": "mmr_term_jaccard_v1",
        "excluded_visual_review_chunk_count": excluded_visual_review_count,
        "returned_visual_review_chunk_count": visual_pending_returned,
        "claim_consistency_status": consistency,
        "source_conflict_count": len(source_conflicts),
        "source_conflicts": source_conflicts,
        "safe_to_quote": safe_to_quote,
        "safe_to_synthesize": safe_to_synthesize,
        "grading_evidence_allowed": False,
        "result_count": len(results),
        "returned_char_count": sum(len(item["excerpt"]) for item in results),
        "results": results,
        "claim_trace": claim_trace,
        "claim_trace_sha256": _canonical_sha256(claim_trace),
        "abstention": {
            "required": not safe_to_synthesize,
            "reason_codes": abstention_reasons,
        },
        "evidence_policy": {
            "teacher_resources_are_grading_evidence": False,
            "visual_review_pending_allowed_for_grading": False,
            "visual_review_pending_allowed_for_synthesis": False,
            "unverified_claims_allowed_for_synthesis": False,
        },
        "privacy_receipt": {
            "mode": "authorized_remote" if external_processing_used else "local_only",
            "remote_processing_authorized": allow_remote_processing,
            "external_processing_used": external_processing_used,
            "raw_media_shared_with_provider": False,
            "provider_inputs": "bounded_extracted_text_or_query_context_only",
            "configured_providers": [
                item.receipt() for item in configured_provider_manifests
            ],
            "providers": provider_calls,
        },
        "budget_receipt": {
            "query_chars": len(query),
            "query_char_limit": MAX_RETRIEVAL_QUERY_CHARS,
            "resource_count": len(corpus_binding),
            "resource_limit": MAX_RETRIEVAL_RESOURCES,
            "candidate_count": len(candidates),
            "candidate_limit": MAX_RETRIEVAL_CANDIDATES,
            "candidate_chars": sum(
                len(str(candidate["excerpt"])) for candidate in candidates
            ),
            "provider_candidate_count": provider_candidate_count,
            "provider_candidate_limit": MAX_PROVIDER_CANDIDATES,
            "provider_candidate_chars": provider_candidate_chars,
            "provider_text_char_limit_per_call": MAX_PROVIDER_TEXT_CHARS,
            "provider_call_count": len(provider_calls),
            "provider_call_limit": MAX_PROVIDER_CALLS,
            "provider_input_chars": sum(
                int(call["input_char_count"]) for call in provider_calls
            ),
            "provider_input_char_limit": MAX_PROVIDER_CALLS
            * MAX_PROVIDER_TEXT_CHARS,
            "provider_candidates_truncated": provider_candidates_truncated,
            "result_limit": int(max_results),
            "returned_chars": sum(len(item["excerpt"]) for item in results),
            "returned_char_limit": int(max_total_chars),
        },
        "corpus_fingerprint_sha256": _canonical_sha256(corpus_binding),
        "student_evidence_used": False,
    }
    receipt["determinism_receipt"] = {
        "deterministic": True,
        "stable_tie_breaker": "chunk_id_ascending",
        "request_fingerprint_sha256": _canonical_sha256(
            {
                "query_sha256": query_hash,
                "corpus_fingerprint_sha256": receipt[
                    "corpus_fingerprint_sha256"
                ],
                "query_expansions_sha256": receipt["query_expansions_sha256"],
                "max_results": int(max_results),
                "max_total_chars": int(max_total_chars),
                "resource_ids": sorted(selected_ids) if selected_ids else None,
                "exclude_visual_review_pending": exclude_visual_review_pending,
                "providers": [
                    {
                        "capability": item.capability,
                        "provider_id": item.provider_id,
                    }
                    for item in configured_provider_manifests
                ],
            }
        ),
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


__all__ = [
    "ClaimAssessmentProvider",
    "DeterministicTermRerankProvider",
    "EmbeddingProvider",
    "LocalHashingEmbeddingProvider",
    "LocalHashingVectorScoreProvider",
    "MAX_CHUNK_CHARS",
    "MAX_CHUNKS_PER_RESOURCE",
    "MAX_INDEXED_RESOURCE_TEXT_CHARS",
    "MAX_QUERY_EXPANSION_CHARS",
    "MAX_QUERY_EXPANSIONS",
    "MAX_QUERY_CONTEXT_CHARS",
    "MAX_QUERY_CONTEXT_TERMS",
    "MAX_PROVIDER_CANDIDATES",
    "MAX_PROVIDER_CALLS",
    "MAX_PROVIDER_TEXT_CHARS",
    "MAX_RETRIEVAL_CANDIDATES",
    "MAX_RETRIEVAL_QUERY_CHARS",
    "MAX_RETRIEVAL_RESOURCES",
    "MAX_RETRIEVAL_RESULTS",
    "MAX_RETRIEVAL_TOTAL_CHARS",
    "RESOURCE_CHUNK_INDEX_SCHEMA",
    "RESOURCE_RETRIEVAL_SCHEMA",
    "RESOURCE_STORE_DOCUMENT_SCHEMA",
    "ResourceRetrievalError",
    "RerankScoreProvider",
    "QueryExpansionProvider",
    "StaticQueryExpansionProvider",
    "TeachingResourceIndexStore",
    "VectorScoreProvider",
    "build_resource_chunk_index",
    "contextual_query_expansions",
    "retrieve_teaching_resources",
    "validate_resource_chunk_index",
]
