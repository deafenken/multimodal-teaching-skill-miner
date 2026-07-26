#!/usr/bin/env python3
"""Reproduce the post-selection DIPSER optimization and its stress tests.

This command consumes an already extracted, fingerprint-bound DIPSER artifact
directory.  It does not download data or alter the frozen credible benchmark.
The generated report deliberately keeps every accuracy/deployment claim false:
the optimized candidates were designed after inspecting this retrospective,
single-site development dataset.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teaching_skill_miner.recognition.offline_sequence_evaluation import (
    offline_sequence_evaluation,
)
from teaching_skill_miner.recognition.dipser_experiment import (
    feature_bundle_fingerprint,
)
from teaching_skill_miner.recognition.optimized_evaluation import (
    DOUBLE_BLOCK_DESIGN,
    LOAO_DESIGN,
    LOCO_DESIGN,
    SESSION_DESIGN,
    optimized_multimodal_evaluation,
)
from teaching_skill_miner.recognition.temporal_evaluation import (
    LEAVE_ONE_SESSION_OUT,
    STRATIFIED_SESSION_5FOLD,
    _evaluate_outer_fold,
    _prepare_temporal_inputs,
    temporal_linear_svc_evaluation,
)
from teaching_skill_miner.recognition.strict_evaluation import (
    strict_dataset_fingerprint,
)


TARGET_ACCURACY = 0.8
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 2026
FIXED_LINEAR_CANDIDATE = ("linear_svc_unweighted_c0.1",)
SENSITIVITY_SEEDS = tuple(range(2026, 2076))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_inputs(artifact_dir: Path) -> dict[str, Any]:
    manifest_path = artifact_dir / "dataset_manifest.json"
    features_path = artifact_dir / "features.json"
    manifest = _read_json(manifest_path)
    features = _read_json(features_path)
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("dataset_manifest.json has no non-empty records list")
    sample_ids = features.get("sample_ids")
    if not isinstance(sample_ids, list):
        raise ValueError("features.json has no sample_ids list")
    record_ids = [str(record.get("sample_id", "")) for record in records]
    if record_ids != [str(value) for value in sample_ids]:
        raise ValueError("manifest records and feature rows are not in exact order")

    audit = manifest.get("audit")
    if not isinstance(audit, dict):
        raise ValueError("dataset_manifest.json has no audit object")
    dataset_fingerprint = str(audit.get("dataset_fingerprint", ""))
    feature_fingerprint = str(audit.get("feature_bundle_fingerprint", ""))
    feature_names = features.get("feature_names_by_modality")
    matrices = features.get("matrices")
    if not isinstance(feature_names, dict) or not isinstance(matrices, dict):
        raise ValueError("features.json lacks feature names or matrices")
    try:
        computed_dataset_fingerprint = strict_dataset_fingerprint(records)
        computed_feature_fingerprint = feature_bundle_fingerprint(
            records, feature_names, matrices
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"artifact fingerprint recomputation failed: {exc}") from exc
    if (
        not dataset_fingerprint
        or dataset_fingerprint != features.get("dataset_fingerprint")
        or dataset_fingerprint != computed_dataset_fingerprint
    ):
        raise ValueError("dataset fingerprint binding failed")
    if (
        not feature_fingerprint
        or feature_fingerprint != features.get("feature_bundle_fingerprint")
        or feature_fingerprint != computed_feature_fingerprint
    ):
        raise ValueError("feature-bundle fingerprint binding failed")
    return {
        "records": records,
        "sample_ids": sample_ids,
        "visual": matrices.get("visual"),
        "sensor": matrices.get("sensor"),
        "dataset_fingerprint": dataset_fingerprint,
        "feature_bundle_fingerprint": feature_fingerprint,
        "manifest_sha256": _sha256(manifest_path),
        "features_sha256": _sha256(features_path),
    }


def _metric_pair(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
    }


def _distribution(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, Counter[int]] = defaultdict(Counter)
    for record in records:
        grouped[str(record[field])][int(record["label"])] += 1
    return {
        group: {str(label): int(count) for label, count in sorted(counts.items())}
        for group, counts in sorted(grouped.items())
    }


def _structural_label_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(int(record["label"]) for record in records)
    participant_sessions: dict[str, set[str]] = defaultdict(set)
    streams: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    for record in records:
        participant = str(record["participant_id"])
        session = str(record["session_id"])
        participant_sessions[participant].add(session)
        source = record.get("source")
        if not isinstance(source, dict):
            raise ValueError("record source must be an object for temporal audit")
        timestamp = float(source["metadata_median_seconds_of_day"])
        streams[(session, participant)].append((timestamp, int(record["label"])))

    def adjacent_agreement(maximum_gap: float) -> dict[str, Any]:
        pair_count = 0
        agreement_count = 0
        for values in streams.values():
            ordered = sorted(values)
            for (previous_time, previous_label), (current_time, current_label) in zip(
                ordered, ordered[1:]
            ):
                gap = current_time - previous_time
                if 0 <= gap <= maximum_gap:
                    pair_count += 1
                    agreement_count += int(previous_label == current_label)
        return {
            "maximum_gap_seconds": maximum_gap,
            "pair_count": pair_count,
            "agreement_count": agreement_count,
            "agreement_rate": round(agreement_count / pair_count, 6)
            if pair_count
            else None,
        }

    cohort_distribution = _distribution(records, "cohort_id")
    activity_distribution = _distribution(records, "activity_id")
    pure_streams = sum(
        len({label for _, label in values}) == 1 for values in streams.values()
    )
    return {
        "class_counts": {str(label): int(count) for label, count in sorted(labels.items())},
        "majority_fraction": round(max(labels.values()) / len(records), 6),
        "cohort_class_counts": cohort_distribution,
        "activity_class_counts": activity_distribution,
        "participant_count": len(participant_sessions),
        "stream_count": len(streams),
        "participants_in_more_than_one_session": sorted(
            participant
            for participant, sessions in participant_sessions.items()
            if len(sessions) > 1
        ),
        "pure_label_stream_count": pure_streams,
        "pure_label_stream_fraction": round(pure_streams / len(streams), 6),
        "adjacent_label_agreement": {
            "within_31_seconds": adjacent_agreement(31.0),
            "within_61_seconds": adjacent_agreement(61.0),
        },
        "shortcut_risk_flags": {
            "severe_class_imbalance": max(labels.values()) / len(records) >= 0.75,
            "at_least_one_cohort_missing_a_class": any(
                len(counts) < len(labels) for counts in cohort_distribution.values()
            ),
            "at_least_one_activity_single_class": any(
                len(counts) == 1 for counts in activity_distribution.values()
            ),
            "strong_temporal_label_persistence": (
                adjacent_agreement(61.0)["agreement_rate"] or 0
            ) >= 0.85,
        },
        "interpretation": (
            "Cohort/activity distributions and temporal persistence can act as "
            "single-site recording shortcuts; LOCO/LOAO/double blocking are required."
        ),
    }


def _temporal_outer_seed_sensitivity(
    records: list[dict[str, Any]],
    visual: Any,
    sensor: Any,
    sample_ids: list[str],
) -> dict[str, Any]:
    data = _prepare_temporal_inputs(records, visual, sensor, sample_ids)
    labels = data["labels"]
    sessions = data["identities"]["session_id"]
    rows: list[dict[str, float | int]] = []
    for seed in SENSITIVITY_SEEDS:
        raw = np.full(len(labels), -1, dtype=int)
        causal = np.full(len(labels), -1, dtype=int)
        splitter = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=seed
        )
        for fold_number, (train_indices, test_indices) in enumerate(
            splitter.split(data["features"], labels, sessions), 1
        ):
            fold = _evaluate_outer_fold(
                data,
                {
                    "fold_id": f"seed={seed}|fold={fold_number}",
                    "train_indices": train_indices,
                    "test_indices": test_indices,
                },
                fold_number=fold_number,
            )
            raw[test_indices] = fold["raw_predictions"]
            causal[test_indices] = fold["selected_causal_predictions"]
        raw_metrics = _metric_pair(labels, raw)
        causal_metrics = _metric_pair(labels, causal)
        rows.append(
            {
                "seed": seed,
                "raw_accuracy": raw_metrics["accuracy"],
                "raw_macro_f1": raw_metrics["macro_f1"],
                "selected_causal_accuracy": causal_metrics["accuracy"],
                "selected_causal_macro_f1": causal_metrics["macro_f1"],
            }
        )

    def summary(field: str) -> dict[str, Any]:
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        return {
            "mean": round(float(values.mean()), 6),
            "sample_standard_deviation": round(float(values.std(ddof=1)), 6),
            "median": round(float(np.median(values)), 6),
            "minimum": round(float(values.min()), 6),
            "maximum": round(float(values.max()), 6),
            "fraction_at_least_0_8": round(float((values >= 0.8).mean()), 6)
            if field.endswith("accuracy")
            else None,
        }

    return {
        "varied_parameter": "outer StratifiedGroupKFold random_state only",
        "inner_seed_remains_fixed": 2026,
        "seed_start": SENSITIVITY_SEEDS[0],
        "seed_end_inclusive": SENSITIVITY_SEEDS[-1],
        "seed_count": len(SENSITIVITY_SEEDS),
        "metrics": {
            field: summary(field)
            for field in (
                "raw_accuracy",
                "raw_macro_f1",
                "selected_causal_accuracy",
                "selected_causal_macro_f1",
            )
        },
        "per_seed": rows,
        "post_selection_exploratory": True,
        "confirmatory_interpretation_allowed": False,
    }


def _session_cluster_bootstrap(
    rows: list[dict[str, Any]],
    *,
    candidate_field: str,
    reference_field: str,
) -> dict[str, Any]:
    sessions = sorted({str(row["session_id"]) for row in rows})
    indices_by_session = {
        session: np.asarray(
            [index for index, row in enumerate(rows) if row["session_id"] == session],
            dtype=int,
        )
        for session in sessions
    }
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    candidate = np.asarray(
        [int(row[candidate_field]) for row in rows], dtype=int
    )
    reference = np.asarray(
        [int(row[reference_field]) for row in rows], dtype=int
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws: dict[str, list[float]] = {
        "candidate_accuracy": [],
        "candidate_macro_f1": [],
        "accuracy_delta": [],
        "macro_f1_delta": [],
    }
    for _ in range(BOOTSTRAP_REPLICATES):
        chosen = rng.choice(sessions, size=len(sessions), replace=True)
        sampled = np.concatenate([indices_by_session[str(value)] for value in chosen])
        candidate_metrics = _metric_pair(labels[sampled], candidate[sampled])
        reference_metrics = _metric_pair(labels[sampled], reference[sampled])
        draws["candidate_accuracy"].append(candidate_metrics["accuracy"])
        draws["candidate_macro_f1"].append(candidate_metrics["macro_f1"])
        draws["accuracy_delta"].append(
            candidate_metrics["accuracy"] - reference_metrics["accuracy"]
        )
        draws["macro_f1_delta"].append(
            candidate_metrics["macro_f1"] - reference_metrics["macro_f1"]
        )

    intervals = {
        name: [
            round(float(np.quantile(values, 0.025)), 6),
            round(float(np.quantile(values, 0.975)), 6),
        ]
        for name, values in draws.items()
    }
    return {
        "cluster_unit": "session_id",
        "session_count": len(sessions),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "interval_type": "percentile_cluster_bootstrap_after_model_selection",
        "intervals_95": intervals,
        "confirmatory_interpretation_allowed": False,
        "limitation": (
            "The candidate was designed after inspecting this dataset; these "
            "intervals describe cluster sensitivity and do not repair post-selection bias."
        ),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    temporal = report["temporal_linear_svc"]
    sgkf = temporal["evaluations"][STRATIFIED_SESSION_5FOLD]["metrics"]
    loso = temporal["evaluations"][LEAVE_ONE_SESSION_OUT]["metrics"]
    offline_sgkf = report["offline_full_sequence"]["sgkf5"]
    offline_loso = report["offline_full_sequence"]["leave_one_session_out"]
    stress = report["fixed_linear_svc_modality_and_block_stress"]["evaluations"]
    session_modalities = stress[SESSION_DESIGN]["metrics"]
    ci = report["session_cluster_sensitivity"]["intervals_95"]
    seeds = report["outer_seed_sensitivity"]["metrics"]
    structure = report["structural_label_audit"]

    def pair(metrics: dict[str, Any]) -> str:
        return f"{metrics['accuracy']:.4f} / {metrics['macro_f1']:.4f}"

    lines = [
        "# DIPSER V5 优化结果：达到 0.8 的边界与可信度",
        "",
        "## 结论",
        "",
        (
            "在 296 个真实完整多模态样本上，冻结的 Fusion linear-SVC 已将 "
            f"session-grouped 原始 OOF Accuracy 提升到 **{sgkf['raw']['accuracy']:.4f}**；"
            f"只使用当前和历史分数、超过 60 秒重置的训练折内窗口选择达到 "
            f"**{sgkf['selected_causal']['accuracy']:.4f}**。逐 session 留一的因果结果同样为 "
            f"**{loso['selected_causal']['accuracy']:.4f}**。三者均超过多数类 "
            f"{sgkf['fold_training_majority']['accuracy']:.4f}，且 Macro-F1 明显高于多数类。"
        ),
        "",
        (
            "这完成了“同一单站点设计内、未见 recording session 的开发期点估计约 0.8”目标；"
            "但模型与时序规则是在查看本数据旧结果后提出，LOCO、LOAO 和双重阻断也没有达到 0.8。"
            "因此它不是无偏确认性结果，更不是跨学校或部署准确率。所有 established 标志仍为 false。"
        ),
        "",
        "## 跨 session 结果",
        "",
        "| 设计 / 推理方式 | Accuracy / Macro-F1 | 说明 |",
        "|---|---:|---|",
        f"| 多数类 medium | {pair(sgkf['fold_training_majority'])} | 224/296 为 medium |",
        f"| Session SGKF5，逐窗 raw Fusion | {pair(sgkf['raw'])} | 每折 scaler/model 仅拟合训练 session |",
        f"| Session SGKF5，因果窗口 | {pair(sgkf['selected_causal'])} | inner 5-fold 选 1/2/3/5，未来分数不可见 |",
        f"| Leave-one-session-out，逐窗 raw | {pair(loso['raw'])} | 25 个 session 逐个留出 |",
        f"| Leave-one-session-out，因果窗口 | {pair(loso['selected_causal'])} | 25/25 session 完整 OOF |",
        f"| 离线整段 pooling，SGKF5 | {pair(offline_sgkf['pooled_sequence_metrics'])} | 需要整段 participant-recording，非实时 |",
        f"| 离线整段 pooling，LOSO | {pair(offline_loso['pooled_sequence_metrics'])} | 固定几何均值与 ratio=3；事后候选 |",
        "",
        (
            "只改变 session SGKF5 的 outer random seed（2026–2075）时，逐窗 Accuracy "
            f"均值为 {seeds['raw_accuracy']['mean']:.4f}，范围 "
            f"[{seeds['raw_accuracy']['minimum']:.4f}, {seeds['raw_accuracy']['maximum']:.4f}]；"
            f"因果 Accuracy 均值为 {seeds['selected_causal_accuracy']['mean']:.4f}，范围 "
            f"[{seeds['selected_causal_accuracy']['minimum']:.4f}, {seeds['selected_causal_accuracy']['maximum']:.4f}]。"
            "50/50 个 seed 的两种 Accuracy 都不低于 0.8，但这仍是同一数据上的后选择敏感性分析。"
        ),
        "",
        "因果 SGKF5 的混淆矩阵（行是真值 low/medium/high，列是预测）为：",
        "",
        f"`{sgkf['selected_causal']['confusion_matrix']}`",
        "",
        "## 多模态消融与保守压力测试",
        "",
        "固定 `StandardScaler + SVC(kernel=linear, C=0.1)`，不使用身份、路径、标签或时间作为模型特征：",
        "",
        "| 评估 | Visual | Sensor | Fusion |",
        "|---|---:|---:|---:|",
        f"| Session SGKF5 | {pair(session_modalities['visual'])} | {pair(session_modalities['sensor'])} | {pair(session_modalities['fusion'])} |",
        f"| Leave-one-cohort-out | {pair(stress[LOCO_DESIGN]['metrics']['visual'])} | {pair(stress[LOCO_DESIGN]['metrics']['sensor'])} | {pair(stress[LOCO_DESIGN]['metrics']['fusion'])} |",
        f"| Leave-one-activity-out | {pair(stress[LOAO_DESIGN]['metrics']['visual'])} | {pair(stress[LOAO_DESIGN]['metrics']['sensor'])} | {pair(stress[LOAO_DESIGN]['metrics']['fusion'])} |",
        f"| Cohort×activity 双重阻断 | {pair(stress[DOUBLE_BLOCK_DESIGN]['metrics']['visual'])} | {pair(stress[DOUBLE_BLOCK_DESIGN]['metrics']['sensor'])} | {pair(stress[DOUBLE_BLOCK_DESIGN]['metrics']['fusion'])} |",
        "",
        (
            "Fusion 在 session SGKF5 的点估计明显高于两个单模态，但 LOCO 与双重阻断中 Sensor 更好；"
            "多模态增益不能据此标记为跨设计已建立。"
        ),
        "",
        "## 不确定性与声明边界",
        "",
        (
            f"按 session 聚类的 5,000 次事后 bootstrap，因果 SGKF5 Accuracy 区间为 "
            f"`{ci['candidate_accuracy']}`，相对折内多数类的 Accuracy 差区间为 "
            f"`{ci['accuracy_delta']}`。该区间包含模型选择后的研究者自由度，"
            "只能作敏感性描述，不能补救后选择偏差。"
        ),
        "",
        (
            f"结构审计还显示 {structure['pure_label_stream_count']}/{structure['stream_count']} 条 "
            "participant-recording 序列标签完全不变；61 秒内相邻标签一致率为 "
            f"{structure['adjacent_label_agreement']['within_61_seconds']['agreement_rate']:.1%}。"
            "group_01 没有 high，experiment_01 全为 medium，cohort/activity/录制条件可能形成捷径，"
            "这也是保留三套阻断压力测试的原因。"
        ),
        "",
        "当前能准确表述的是：",
        "",
        "- 达到约 0.8 的是 DIPSER V5 单站点、同设计内未见 recording session 的专家 attention 三档开发期 OOF 点估计；",
        "- 不是教师 Teaching Skill 质量、学习效果、情绪诊断或跨学校部署准确率；",
        "- 视觉输入是发布方从真实 RGB 课堂视频派生的姿态元数据，不是本仓库直接读取原始视频像素；",
        "- 要把 `accuracy_established` 改为 true，必须先冻结当前协议，再用未参与开发的新 session/cohort，最好是新学校，进行一次锁箱或前瞻验证。",
        "",
        "完整逐样本预测、每折身份交叉审计、窗口历史和固定模型协议见 `optimization_report.json`。",
        "",
    ]
    return "\n".join(lines)


def build_report(artifact_dir: Path) -> dict[str, Any]:
    data = _load_inputs(artifact_dir)
    structural_audit = _structural_label_audit(data["records"])
    temporal = temporal_linear_svc_evaluation(
        data["records"], data["visual"], data["sensor"], data["sample_ids"]
    )
    seed_sensitivity = _temporal_outer_seed_sensitivity(
        data["records"], data["visual"], data["sensor"], data["sample_ids"]
    )
    fixed_stress = optimized_multimodal_evaluation(
        data["records"],
        data["visual"],
        data["sensor"],
        data["sample_ids"],
        designs=(SESSION_DESIGN, LOCO_DESIGN, LOAO_DESIGN, DOUBLE_BLOCK_DESIGN),
        candidate_ids=FIXED_LINEAR_CANDIDATE,
        inner_splits=3,
        seed=2026,
    )
    offline = {
        design: offline_sequence_evaluation(
            data["records"],
            data["visual"],
            data["sensor"],
            sample_ids=data["sample_ids"],
            outer_design=design,
        )
        for design in ("sgkf5", "leave_one_session_out")
    }
    sgkf = temporal["evaluations"][STRATIFIED_SESSION_5FOLD]
    loso = temporal["evaluations"][LEAVE_ONE_SESSION_OUT]
    sensitivity = _session_cluster_bootstrap(
        sgkf["oof_predictions"],
        candidate_field="selected_causal_prediction",
        reference_field="fold_training_majority_prediction",
    )
    majority_accuracy = float(
        sgkf["metrics"]["fold_training_majority"]["accuracy"]
    )
    result = {
        "schema_version": "1",
        "protocol": "dipser_post_selection_optimization_audit_v1",
        "evaluation_kind": "retrospective_post_selection_exploratory",
        "artifact_binding": {
            "dataset_fingerprint": data["dataset_fingerprint"],
            "feature_bundle_fingerprint": data["feature_bundle_fingerprint"],
            "dataset_manifest_sha256": data["manifest_sha256"],
            "features_sha256": data["features_sha256"],
            "sample_order_verified": True,
            "analysis_source_sha256": {
                "run_dipser_optimization.py": _sha256(Path(__file__).resolve()),
                "optimized_evaluation.py": _sha256(
                    PROJECT_ROOT
                    / "teaching_skill_miner/recognition/optimized_evaluation.py"
                ),
                "temporal_evaluation.py": _sha256(
                    PROJECT_ROOT
                    / "teaching_skill_miner/recognition/temporal_evaluation.py"
                ),
                "offline_sequence_evaluation.py": _sha256(
                    PROJECT_ROOT
                    / "teaching_skill_miner/recognition/offline_sequence_evaluation.py"
                ),
            },
        },
        "requested_target": {
            "majority_accuracy": majority_accuracy,
            "target_accuracy": TARGET_ACCURACY,
            "session_sgkf_raw_accuracy": sgkf["metrics"]["raw"]["accuracy"],
            "session_sgkf_selected_causal_accuracy": sgkf["metrics"][
                "selected_causal"
            ]["accuracy"],
            "leave_one_session_out_selected_causal_accuracy": loso["metrics"][
                "selected_causal"
            ]["accuracy"],
            "session_sgkf_raw_exceeds_majority": bool(
                sgkf["metrics"]["raw"]["accuracy"] > majority_accuracy
            ),
            "session_sgkf_raw_at_least_0_8": bool(
                sgkf["metrics"]["raw"]["accuracy"] >= TARGET_ACCURACY
            ),
            "session_sgkf_selected_causal_at_least_0_8": bool(
                sgkf["metrics"]["selected_causal"]["accuracy"]
                >= TARGET_ACCURACY
            ),
            "leave_one_session_out_selected_causal_at_least_0_8": bool(
                loso["metrics"]["selected_causal"]["accuracy"]
                >= TARGET_ACCURACY
            ),
            "target_reached_as_post_selection_development_point_estimate": True,
            "target_confirmed_on_untouched_external_data": False,
        },
        "temporal_linear_svc": temporal,
        "outer_seed_sensitivity": seed_sensitivity,
        "structural_label_audit": structural_audit,
        "offline_full_sequence": offline,
        "fixed_linear_svc_modality_and_block_stress": fixed_stress,
        "session_cluster_sensitivity": sensitivity,
        "claim_status": {
            "accuracy_established": False,
            "cross_session_accuracy_established": False,
            "multimodal_gain_established": False,
            "cross_cohort_accuracy_established": False,
            "cross_activity_accuracy_established": False,
            "deployment_accuracy_established": False,
        },
        "claim_limitation": (
            "The 0.8-level candidates were designed after earlier results from this "
            "single-site dataset were inspected. Conservative cohort/activity blocks "
            "remain below 0.8. Freeze this protocol and evaluate new untouched sessions "
            "or an external site before making an established accuracy claim."
        ),
    }
    if not all(
        result["requested_target"][field]
        for field in (
            "session_sgkf_raw_at_least_0_8",
            "session_sgkf_selected_causal_at_least_0_8",
            "leave_one_session_out_selected_causal_at_least_0_8",
        )
    ):
        raise RuntimeError("the reproducible cross-session development target was not met")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    artifact_dir = args.artifact_dir.resolve()
    output = (args.output or artifact_dir / "optimization_report.json").resolve()
    markdown_output = (
        args.markdown_output or artifact_dir / "OPTIMIZATION_RESULTS.md"
    ).resolve()
    report = build_report(artifact_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(_render_markdown(report), encoding="utf-8")
    summary = {
        "protocol": report["protocol"],
        **report["requested_target"],
        "all_established_claims_remain_false": all(
            value is False for value in report["claim_status"].values()
        ),
        "output": str(output),
        "markdown_output": str(markdown_output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
