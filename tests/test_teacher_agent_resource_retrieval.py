from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from teaching_skill_miner.teacher_agent_resource_retrieval import (
    LocalHashingEmbeddingProvider,
    LocalHashingVectorScoreProvider,
    ResourceRetrievalError,
    StaticQueryExpansionProvider,
    TeachingResourceIndexStore,
    build_resource_chunk_index,
    retrieve_teaching_resources,
    validate_resource_chunk_index,
)
from teaching_skill_miner.teacher_agent_resources import extract_teaching_resource


def _resource() -> dict[str, object]:
    return extract_teaching_resource(
        (
            "状态定义说明问题中需要记住的信息。\n\n"
            "状态转移说明当前状态怎样由更小的子问题得到。\n\n"
            "边界条件为递推提供最小规模的已知答案。"
        ).encode("utf-8"),
        "text/plain",
        display_name="动态规划讲义.md",
    )


def _indexed_resource() -> dict[str, object]:
    resource = _resource()
    resource["retrieval_index"] = build_resource_chunk_index(
        resource_id=str(resource["resource_id"]),
        resource_content_sha256=str(resource["content_sha256"]),
        resource_type=str(resource["resource_type"]),
        extracted_text=str(resource["extracted_text"]),
    )
    return resource


def test_lexical_retrieval_is_deterministic_bounded_and_provenanced() -> None:
    resource = _resource()

    first = retrieve_teaching_resources(
        [resource], "状态转移 更小子问题", max_results=2, max_total_chars=500
    )
    second = retrieve_teaching_resources(
        [resource], "状态转移 更小子问题", max_results=2, max_total_chars=500
    )

    assert first == second
    assert first["result_count"] >= 1
    assert first["returned_char_count"] <= 500
    assert first["student_evidence_used"] is False
    hit = first["results"][0]
    assert "状态转移" in hit["excerpt"]
    assert hit["provenance"]["resource_id"] == resource["resource_id"]
    assert hit["provenance"]["location"]["kind"] == "paragraph"
    assert len(hit["provenance"]["chunk_content_sha256"]) == 64
    assert "learner" not in str(first).casefold()


def test_retrieval_fails_closed_when_chunk_text_or_hash_is_tampered() -> None:
    resource = _indexed_resource()
    tampered = deepcopy(resource)
    tampered["retrieval_index"]["chunks"][0]["content_sha256"] = "0" * 64

    with pytest.raises(ResourceRetrievalError, match="content hash"):
        retrieve_teaching_resources([tampered], "状态")


def test_old_descriptor_without_index_gets_same_deterministic_on_demand_index() -> None:
    resource = _indexed_resource()
    legacy = deepcopy(resource)
    legacy.pop("retrieval_index")

    result = retrieve_teaching_resources([legacy], "边界条件")

    assert result["result_count"] == 1
    assert "边界条件" in result["results"][0]["excerpt"]


def test_index_validation_binds_offsets_to_resource_hash_and_text() -> None:
    resource = _indexed_resource()

    validate_resource_chunk_index(
        resource["retrieval_index"],
        resource_id=str(resource["resource_id"]),
        resource_content_sha256=str(resource["content_sha256"]),
        extracted_text=str(resource["extracted_text"]),
    )
    with pytest.raises(ResourceRetrievalError, match="length binding"):
        validate_resource_chunk_index(
            resource["retrieval_index"],
            resource_id=str(resource["resource_id"]),
            resource_content_sha256=str(resource["content_sha256"]),
            extracted_text=str(resource["extracted_text"]) + "篡改",
        )


def test_private_atomic_store_deduplicates_and_reloads_resource(tmp_path) -> None:
    resource = _resource()
    store = TeachingResourceIndexStore(tmp_path / "private-resources")

    created = store.put(resource)
    duplicate = store.put(resource)
    reloaded = TeachingResourceIndexStore(tmp_path / "private-resources")

    assert created["created"] is True
    assert duplicate["created"] is False
    assert reloaded.get(str(resource["resource_id"])) == resource
    assert reloaded.get_by_content_hash(str(resource["content_sha256"])) == resource
    assert "extracted_text" not in reloaded.list_metadata()[0]
    assert len(list((tmp_path / "private-resources").glob("*.json"))) == 1
    assert ((tmp_path / "private-resources").stat().st_mode & 0o777) == 0o700
    document = next((tmp_path / "private-resources").glob("*.json"))
    assert (document.stat().st_mode & 0o777) == 0o600


def test_extractor_can_persist_text_beyond_live_session_prefix(tmp_path) -> None:
    store = TeachingResourceIndexStore(tmp_path / "resources")
    payload = ("开头段落\n" + "中间内容" * 5_000 + "\n末尾检索锚点").encode("utf-8")
    resource = extract_teaching_resource(
        payload,
        "text/plain",
        display_name="长讲义.txt",
        index_store=store,
    )
    retrieval_resource = store.get_retrieval_resource(resource["content_sha256"])
    assert retrieval_resource is not None

    assert len(resource["extracted_text"]) == 12_000
    assert len(retrieval_resource["extracted_text"]) > 12_000
    assert retrieval_resource["retrieval_index_truncated"] is False
    result = retrieve_teaching_resources([retrieval_resource], "末尾检索锚点")
    assert result["result_count"] == 1
    assert "末尾检索锚点" in result["results"][0]["excerpt"]


def test_store_refuses_to_mistake_a_truncated_session_prefix_for_full_index(
    tmp_path,
) -> None:
    resource = extract_teaching_resource(
        ("内容" * 7_000).encode("utf-8"),
        "text/plain",
        display_name="被截断讲义.txt",
    )
    assert resource["truncated"] is True

    with pytest.raises(ResourceRetrievalError, match="longer local indexed text"):
        TeachingResourceIndexStore(tmp_path / "resources").put(resource)


def test_resource_store_rejects_corrupt_hash_binding(tmp_path) -> None:
    resource = _resource()
    store = TeachingResourceIndexStore(tmp_path / "resources")
    store.put(resource)
    document = next((tmp_path / "resources").glob("*.json"))
    document.write_text("{}", encoding="utf-8")

    with pytest.raises(ResourceRetrievalError, match="contract"):
        store.get_by_content_hash(str(resource["content_sha256"]))


def test_local_hashing_vector_and_query_expansion_are_private_and_deterministic() -> (
    None
):
    resource = _resource()
    scorer = LocalHashingVectorScoreProvider(dimensions=512)

    first = retrieve_teaching_resources(
        [resource],
        "较小子问题怎样连接",
        query_expansions=["状态转移"],
        vector_scorer=scorer,
    )
    second = retrieve_teaching_resources(
        [resource],
        "较小子问题怎样连接",
        query_expansions=["状态转移"],
        vector_scorer=scorer,
    )

    assert first == second
    assert first["vector_scores_used"] is True
    assert first["query_expansions"] == ["状态转移"]
    assert first["result_count"] >= 1
    assert "状态转移" in first["results"][0]["excerpt"]
    assert first["grading_evidence_allowed"] is False
    assert first["safe_to_synthesize"] is False
    assert first["safe_to_quote"] is True


def test_visual_review_pending_chunks_are_excluded_by_default() -> None:
    resource = _indexed_resource()
    resource["retrieval_index"]["chunks"][0]["needs_visual_review"] = True

    result = retrieve_teaching_resources([resource], "状态定义")

    assert result["excluded_visual_review_chunk_count"] == 1
    assert all("状态定义" not in item["excerpt"] for item in result["results"])


def test_claim_conflict_forces_abstention_and_never_becomes_grading_evidence() -> None:
    first = extract_teaching_resource(
        "水在标准大气压下达到100摄氏度会沸腾。".encode(),
        "text/plain",
        display_name="教师材料甲.txt",
    )
    second = extract_teaching_resource(
        "水在标准大气压下达到100摄氏度不会沸腾。".encode(),
        "text/plain",
        display_name="教师材料乙.txt",
    )

    class ConflictAssessor:
        provider_id = "test_local_conflict_assessor"
        execution_scope = "local"
        sends_source_text_off_device = False
        deterministic = True

        def assess(self, _query, excerpts):
            return [
                "contradicts" if "不会" in excerpt else "entails"
                for excerpt in excerpts
            ]

    result = retrieve_teaching_resources(
        [first, second],
        "水在标准大气压100摄氏度是否沸腾",
        max_results=2,
        claim_assessor=ConflictAssessor(),
    )

    assert result["result_count"] == 2
    assert result["claim_consistency_status"] == "conflicting_sources"
    assert result["safe_to_synthesize"] is False
    assert result["grading_evidence_allowed"] is False
    assert {item["claim_relation"] for item in result["results"]} == {
        "entails",
        "contradicts",
    }


def test_mmr_diversifies_near_duplicate_retrieval_results() -> None:
    resources = [
        extract_teaching_resource(text.encode(), "text/plain", display_name=name)
        for name, text in (
            ("重复一.txt", "边界条件给出递推的起点。\n边界条件给出递推的起点。"),
            ("补充二.txt", "边界条件必须与状态定义的语义保持一致。"),
        )
    ]

    result = retrieve_teaching_resources(
        resources, "边界条件 递推 起点 状态定义", max_results=2
    )

    assert result["result_count"] == 2
    assert result["diversification_method"] == "mmr_term_jaccard_v1"
    assert len({item["provenance"]["resource_id"] for item in result["results"]}) == 2


def test_pluggable_embedding_expansion_rerank_and_claim_trace_are_deterministic() -> (
    None
):
    target = extract_teaching_resource(
        "递推关系把当前状态与较小规模子问题连接起来。".encode(),
        "text/plain",
        display_name="目标材料.txt",
    )
    distractor = extract_teaching_resource(
        "蒸发是液体表面的汽化过程。".encode(),
        "text/plain",
        display_name="干扰材料.txt",
    )

    class Reranker:
        provider_id = "test_local_reranker"
        execution_scope = "local"
        sends_source_text_off_device = False
        deterministic = True

        def score(self, _query, excerpts):
            return [10.0 if "递推关系" in excerpt else 0.0 for excerpt in excerpts]

    expander = StaticQueryExpansionProvider({"recurrence": ["递推关系"]})
    kwargs = {
        "embedding_provider": LocalHashingEmbeddingProvider(dimensions=256),
        "query_expander": expander,
        "query_context_terms": ["离散数学 recurrence"],
        "rerank_scorer": Reranker(),
    }

    first = retrieve_teaching_resources(
        [target, distractor], "recurrence", **kwargs
    )
    second = retrieve_teaching_resources(
        [target, distractor], "recurrence", **kwargs
    )

    assert first == second
    assert first["embedding_scores_used"] is True
    assert first["lexical_fallback_used"] is False
    assert first["fusion_method"] == "weighted_reciprocal_rank_fusion_v1"
    assert first["query_expansions"] == ["递推关系"]
    assert first["results"][0]["provenance"]["resource_id"] == target["resource_id"]
    assert first["claim_trace"]["claim_sha256"] == first["query_sha256"]
    assert first["claim_trace"]["citations"][0]["citation_id"] == first[
        "results"
    ][0]["citation_id"]
    assert len(first["receipt_sha256"]) == 64
    assert {
        item["capability"] for item in first["privacy_receipt"]["providers"]
    } == {"embedding", "query_expansion", "rerank"}
    assert first["privacy_receipt"]["external_processing_used"] is False
    assert first["budget_receipt"]["provider_candidate_count"] <= first[
        "budget_receipt"
    ]["provider_candidate_limit"]


def test_remote_text_provider_requires_explicit_authorization_and_is_receipted() -> (
    None
):
    resource = _resource()

    class RemoteScorer:
        provider_id = "test_remote_vector_service"
        execution_scope = "remote"
        sends_source_text_off_device = True
        deterministic = True

        def score(self, _query, excerpts):
            return [0.9 for _ in excerpts]

    with pytest.raises(ResourceRetrievalError, match="explicit remote-processing"):
        retrieve_teaching_resources(
            [resource], "状态定义", vector_scorer=RemoteScorer()
        )

    receipt = retrieve_teaching_resources(
        [resource],
        "状态定义",
        vector_scorer=RemoteScorer(),
        allow_remote_processing=True,
    )

    assert receipt["privacy_receipt"]["mode"] == "authorized_remote"
    assert receipt["privacy_receipt"]["external_processing_used"] is True
    assert receipt["privacy_receipt"]["raw_media_shared_with_provider"] is False


def test_unattested_or_nondeterministic_provider_fails_closed() -> None:
    resource = _resource()

    class Unattested:
        def score(self, _query, excerpts):
            return [1.0 for _ in excerpts]

    class Nondeterministic:
        provider_id = "test_nondeterministic_vector"
        execution_scope = "local"
        sends_source_text_off_device = False
        deterministic = False

        def score(self, _query, excerpts):
            return [1.0 for _ in excerpts]

    with pytest.raises(ResourceRetrievalError, match="privacy declaration"):
        retrieve_teaching_resources(
            [resource], "状态定义", vector_scorer=Unattested()
        )
    with pytest.raises(ResourceRetrievalError, match="deterministic"):
        retrieve_teaching_resources(
            [resource], "状态定义", vector_scorer=Nondeterministic()
        )


def test_provider_failure_is_sanitized_into_retrieval_boundary_error() -> None:
    class BrokenProvider:
        provider_id = "test_broken_local_vector"
        execution_scope = "local"
        sends_source_text_off_device = False
        deterministic = True

        def score(self, _query, _excerpts):
            raise RuntimeError("provider secret detail")

    with pytest.raises(
        ResourceRetrievalError, match="vector score provider execution failed"
    ) as caught:
        retrieve_teaching_resources(
            [_resource()], "状态定义", vector_scorer=BrokenProvider()
        )

    assert "secret" not in str(caught.value)


def test_visual_pending_result_cannot_be_synthesized_even_when_assessor_entails() -> (
    None
):
    resource = _indexed_resource()
    resource["retrieval_index"]["chunks"][0]["needs_visual_review"] = True

    class Entails:
        provider_id = "test_local_entailment_assessor"
        execution_scope = "local"
        sends_source_text_off_device = False
        deterministic = True

        def assess(self, _query, excerpts):
            return ["entails" for _ in excerpts]

    receipt = retrieve_teaching_resources(
        [resource],
        "状态定义",
        exclude_visual_review_pending=False,
        claim_assessor=Entails(),
    )

    assert receipt["returned_visual_review_chunk_count"] == 1
    assert receipt["results"][0]["eligible_for_grading"] is False
    assert receipt["results"][0]["eligible_for_synthesis"] is False
    assert receipt["safe_to_quote"] is False
    assert receipt["safe_to_synthesize"] is False
    assert receipt["abstention"]["reason_codes"] == ["visual_review_pending"]
    assert receipt["evidence_policy"][
        "visual_review_pending_allowed_for_grading"
    ] is False


def test_deterministic_conflict_detector_abstains_without_claim_model() -> None:
    yes = extract_teaching_resource(
        "水在标准大气压下达到100摄氏度会沸腾。".encode(),
        "text/plain",
        display_name="来源甲.txt",
    )
    no = extract_teaching_resource(
        "水在标准大气压下达到100摄氏度不会沸腾。".encode(),
        "text/plain",
        display_name="来源乙.txt",
    )

    receipt = retrieve_teaching_resources(
        [yes, no], "水在标准大气压100摄氏度是否沸腾", max_results=2
    )

    assert receipt["claim_consistency_status"] == "conflicting_sources"
    assert receipt["source_conflict_count"] == 1
    assert receipt["source_conflicts"][0]["reason"] == (
        "matched_statement_opposite_polarity"
    )
    assert receipt["safe_to_synthesize"] is False
    assert receipt["claim_trace"]["decision"] == "abstain"
    assert receipt["claim_trace"]["assessment_complete"] is False


def test_partial_excerpt_has_distinct_chunk_and_excerpt_hashes_bound_to_citation() -> (
    None
):
    resource = extract_teaching_resource(
        ("检索锚点 " + "状态转移描述。" * 120).encode(),
        "text/plain",
        display_name="长段落.txt",
    )

    receipt = retrieve_teaching_resources(
        [resource], "检索锚点", max_results=1, max_total_chars=200
    )
    hit = receipt["results"][0]
    provenance = hit["provenance"]

    assert len(hit["excerpt"]) <= 200
    assert provenance["excerpt_is_partial"] is True
    assert provenance["chunk_content_sha256"] != provenance[
        "excerpt_content_sha256"
    ]
    assert provenance["excerpt_content_sha256"] == hashlib.sha256(
        hit["excerpt"].encode()
    ).hexdigest()
    assert receipt["claim_trace"]["citations"][0][
        "excerpt_content_sha256"
    ] == provenance["excerpt_content_sha256"]


def test_no_embedding_is_honest_bounded_lexical_fallback() -> None:
    receipt = retrieve_teaching_resources([_resource()], "边界条件")

    assert receipt["retrieval_method"] == "bm25_lexical_v1"
    assert receipt["lexical_fallback_used"] is True
    assert receipt["vector_scores_used"] is False
    assert receipt["embedding_scores_used"] is False
    assert receipt["budget_receipt"]["candidate_count"] >= 1
    assert receipt["privacy_receipt"]["mode"] == "local_only"
