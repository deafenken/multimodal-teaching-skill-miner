from __future__ import annotations

from copy import deepcopy

import pytest

from teaching_skill_miner.teacher_agent_resource_retrieval import (
    LocalHashingVectorScoreProvider,
    build_resource_chunk_index,
    retrieve_teaching_resources,
)
from teaching_skill_miner.teacher_agent_resources import extract_teaching_resource
from teaching_skill_miner.teacher_agent_retrieval_eval import (
    ResourceRetrievalEvaluationError,
    run_resource_retrieval_evaluation,
    score_resource_retrieval_predictions,
)


def _resource(name: str, text: str) -> dict:
    return extract_teaching_resource(
        text.encode("utf-8"), "text/plain", display_name=name
    )


def test_hidden_gold_retrieval_eval_scores_relevance_citations_and_conflicts() -> None:
    state = _resource("状态讲义.txt", "状态定义说明 dp[i] 的下标和保存值分别代表什么。")
    transition = _resource(
        "转移讲义.txt", "状态转移说明当前状态怎样由更小规模的子问题得到。"
    )
    boiling_yes = _resource("沸腾材料甲.txt", "水在标准大气压下达到100摄氏度会沸腾。")
    boiling_no = _resource("沸腾材料乙.txt", "水在标准大气压下达到100摄氏度不会沸腾。")
    resources = [state, transition, boiling_yes, boiling_no]

    class Assessor:
        provider_id = "test_local_eval_assessor"
        execution_scope = "local"
        sends_source_text_off_device = False
        deterministic = True

        def assess(self, query, excerpts):
            if "沸腾" not in query:
                return ["entails" for _ in excerpts]
            return [
                "contradicts" if "不会" in excerpt else "entails"
                for excerpt in excerpts
            ]

    cases = [
        {
            "case_id": "state-transition",
            "query": "状态转移 更小规模子问题",
            "relevant_chunk_ids": [f"{transition['resource_id']}_c001"],
            "forbidden_chunk_ids": [],
            "expected_consistency": "query_supported",
            "k": 1,
        },
        {
            "case_id": "boiling-conflict",
            "query": "水在标准大气压100摄氏度是否沸腾",
            "relevant_chunk_ids": [
                f"{boiling_yes['resource_id']}_c001",
                f"{boiling_no['resource_id']}_c001",
            ],
            "forbidden_chunk_ids": [],
            "expected_consistency": "conflicting_sources",
            "k": 2,
        },
    ]
    queries_seen: list[str] = []

    def retrieve(query: str):
        queries_seen.append(query)
        return retrieve_teaching_resources(
            resources,
            query,
            max_results=2,
            vector_scorer=LocalHashingVectorScoreProvider(),
            claim_assessor=Assessor(),
        )

    report = run_resource_retrieval_evaluation(cases, retrieve)

    assert queries_seen == [case["query"] for case in cases]
    assert report["benchmark_gold_exposed_to_retriever"] is False
    assert report["student_evidence_used"] is False
    assert report["mean_recall_at_k"] == 1.0
    assert report["mean_ndcg_at_k"] == 1.0
    assert report["citation_integrity_rate"] == 1.0
    assert report["consistency_accuracy"] == 1.0
    assert report["contradiction_abstention_accuracy"] == 1.0
    assert report["forbidden_chunk_hit_rate"] == 0.0
    assert len(report["report_sha256"]) == 64


def test_eval_rejects_tampered_excerpt_hash_and_gold_id_mismatch() -> None:
    resource = _resource("材料.txt", "边界条件必须与状态定义保持一致。")
    case = {
        "case_id": "boundary",
        "query": "边界条件",
        "relevant_chunk_ids": [f"{resource['resource_id']}_c001"],
        "forbidden_chunk_ids": [],
        "expected_consistency": "not_verified",
        "k": 1,
    }
    receipt = retrieve_teaching_resources([resource], case["query"])
    tampered = deepcopy(receipt)
    tampered["results"][0]["excerpt"] += "篡改"

    with pytest.raises(ResourceRetrievalEvaluationError, match="integrity"):
        score_resource_retrieval_predictions([case], {"boundary": tampered})
    with pytest.raises(ResourceRetrievalEvaluationError, match="exactly match"):
        score_resource_retrieval_predictions([case], {"wrong-id": receipt})


def test_multimodal_corpus_reports_recall_ndcg_citation_entailment_and_abstention() -> (
    None
):
    text = _resource(
        "文本讲义.txt", "拓扑排序只适用于有向无环图，入度为零的节点可以先处理。"
    )
    table = _resource(
        "复杂度表.txt",
        "[结构化表格：来自单元格顺序]\n行 1: 归并排序 | O(n log n) | 稳定",
    )
    chart = _resource(
        "图表转写.txt",
        "[图表缓存已由教师核验] 系列A在2024年的值为42\n"
        "[视觉复核：图表坐标轴含义尚未核验] 系列B的趋势候选",
    )
    chart["retrieval_index"] = build_resource_chunk_index(
        resource_id=str(chart["resource_id"]),
        resource_content_sha256=str(chart["content_sha256"]),
        resource_type=str(chart["resource_type"]),
        extracted_text=str(chart["extracted_text"]),
        visual_review_locations=[2],
    )
    equation = _resource(
        "公式讲义.txt", "欧拉恒等式写作 e^(iπ)+1=0，它连接指数函数与三角函数。"
    )
    boiling_yes = _resource(
        "物理来源甲.txt", "水在标准大气压下达到100摄氏度会沸腾。"
    )
    boiling_no = _resource(
        "物理来源乙.txt", "水在标准大气压下达到100摄氏度不会沸腾。"
    )
    resources = [text, table, chart, equation, boiling_yes, boiling_no]

    class Assessor:
        provider_id = "test_local_multimodal_assessor"
        execution_scope = "local"
        sends_source_text_off_device = False
        deterministic = True

        def assess(self, _query, excerpts):
            return [
                "contradicts" if "不会沸腾" in excerpt else "entails"
                for excerpt in excerpts
            ]

    identities = {
        "text": f"{text['resource_id']}_c001",
        "table": f"{table['resource_id']}_c002",
        "chart": f"{chart['resource_id']}_c001",
        "chart_unverified": f"{chart['resource_id']}_c002",
        "equation": f"{equation['resource_id']}_c001",
        "yes": f"{boiling_yes['resource_id']}_c001",
        "no": f"{boiling_no['resource_id']}_c001",
    }
    cases = [
        {
            "case_id": "modality-text",
            "modality": "text",
            "query": "拓扑排序 有向无环图 入度为零",
            "relevant_chunk_ids": [identities["text"]],
            "relevance_grades": {identities["text"]: 3},
            "entailing_chunk_ids": [identities["text"]],
            "contradicting_chunk_ids": [],
            "forbidden_chunk_ids": [],
            "expected_consistency": "query_supported",
            "expected_abstain": False,
            "k": 1,
        },
        {
            "case_id": "modality-table",
            "modality": "table",
            "query": "归并排序 O(n log n) 稳定",
            "relevant_chunk_ids": [identities["table"]],
            "relevance_grades": {identities["table"]: 3},
            "entailing_chunk_ids": [identities["table"]],
            "contradicting_chunk_ids": [],
            "forbidden_chunk_ids": [],
            "expected_consistency": "query_supported",
            "expected_abstain": False,
            "k": 1,
        },
        {
            "case_id": "modality-chart",
            "modality": "chart",
            "query": "系列A 2024 值42",
            "relevant_chunk_ids": [identities["chart"]],
            "relevance_grades": {identities["chart"]: 3},
            "entailing_chunk_ids": [identities["chart"]],
            "contradicting_chunk_ids": [],
            "forbidden_chunk_ids": [identities["chart_unverified"]],
            "expected_consistency": "query_supported",
            "expected_abstain": False,
            "k": 1,
        },
        {
            "case_id": "modality-equation",
            "modality": "equation",
            "query": "欧拉恒等式 e^(iπ)+1=0",
            "relevant_chunk_ids": [identities["equation"]],
            "relevance_grades": {identities["equation"]: 3},
            "entailing_chunk_ids": [identities["equation"]],
            "contradicting_chunk_ids": [],
            "forbidden_chunk_ids": [],
            "expected_consistency": "query_supported",
            "expected_abstain": False,
            "k": 1,
        },
        {
            "case_id": "modality-contradiction",
            "modality": "contradiction",
            "query": "水在标准大气压100摄氏度是否沸腾",
            "relevant_chunk_ids": [identities["yes"], identities["no"]],
            "relevance_grades": {identities["yes"]: 3, identities["no"]: 3},
            "entailing_chunk_ids": [identities["yes"]],
            "contradicting_chunk_ids": [identities["no"]],
            "forbidden_chunk_ids": [],
            "expected_consistency": "conflicting_sources",
            "expected_abstain": True,
            "k": 2,
        },
    ]

    report = run_resource_retrieval_evaluation(
        cases,
        lambda query: retrieve_teaching_resources(
            resources,
            query,
            max_results=2,
            vector_scorer=LocalHashingVectorScoreProvider(),
            claim_assessor=Assessor(),
        ),
    )

    assert report["mean_recall_at_k"] == 1.0
    assert report["mean_ndcg_at_k"] == 1.0
    assert report["citation_precision_at_k"] == 1.0
    assert report["citation_integrity_rate"] == 1.0
    assert report["claim_trace_integrity_rate"] == 1.0
    assert report["entailment_relation_accuracy"] == 1.0
    assert report["contradiction_abstention_accuracy"] == 1.0
    assert report["forbidden_chunk_hit_rate"] == 0.0
    assert set(report["modality_metrics"]) == {
        "text",
        "table",
        "chart",
        "equation",
        "contradiction",
    }


def test_eval_rejects_tampered_claim_to_chunk_trace() -> None:
    resource = _resource("引用材料.txt", "最短路径满足三角不等式。")
    case = {
        "case_id": "trace-binding",
        "query": "最短路径 三角不等式",
        "relevant_chunk_ids": [f"{resource['resource_id']}_c001"],
        "forbidden_chunk_ids": [],
        "expected_consistency": "not_verified",
        "k": 1,
    }
    receipt = retrieve_teaching_resources([resource], case["query"])
    tampered = deepcopy(receipt)
    tampered["claim_trace"]["citations"][0]["chunk_id"] = "res_fake_c001"

    with pytest.raises(ResourceRetrievalEvaluationError, match="claim-to-chunk"):
        score_resource_retrieval_predictions([case], {"trace-binding": tampered})
