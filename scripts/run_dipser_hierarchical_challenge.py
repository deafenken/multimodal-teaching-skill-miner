#!/usr/bin/env python3
"""Reproduce the DIPSER offline hierarchical 0.9 challenge.

The command consumes an already extracted, fingerprint-bound artifact bundle.
It evaluates the post-selected full-session transductive candidate with honest
inner refits, 50 outer SGKF seeds, deterministic leave-one-session-out, and
fixed-gate cohort/activity blocking.  It never upgrades the development score
to a real-time, inductive, or deployment claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version as package_version
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teaching_skill_miner.recognition.hierarchical_evaluation import (
    DOUBLE_BLOCKED_COHORT_ACTIVITY,
    LEAVE_ONE_ACTIVITY_OUT,
    LEAVE_ONE_COHORT_OUT,
    LEAVE_ONE_SESSION_OUT,
    SGKF5,
    fixed_hierarchical_stress_evaluation,
    hierarchical_offline_evaluation,
)
from teaching_skill_miner.recognition.hierarchical_counterfactual import (
    CAUSAL_PREFIX,
    FULL,
    LEAVE_CURRENT_OUT,
    hierarchical_counterfactual_evaluation,
)
from teaching_skill_miner.recognition.dipser_experiment import (
    feature_bundle_fingerprint,
)
from teaching_skill_miner.recognition.strict_evaluation import (
    strict_dataset_fingerprint,
)


TARGET_ACCURACY = 0.9
OUTER_SEEDS = tuple(range(2026, 2076))
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 2026
FIXED_STRESS_GATE = "high_override_plus_low_confirmation"


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
    sample_ids = features.get("sample_ids")
    feature_names = features.get("feature_names_by_modality")
    matrices = features.get("matrices")
    if not isinstance(records, list) or not records:
        raise ValueError("dataset_manifest.json has no non-empty records list")
    if (
        not isinstance(sample_ids, list)
        or not isinstance(feature_names, dict)
        or not isinstance(matrices, dict)
    ):
        raise ValueError("features.json lacks sample_ids, feature names, or matrices")
    record_ids = [str(record.get("sample_id", "")) for record in records]
    if record_ids != [str(value) for value in sample_ids]:
        raise ValueError("manifest records and feature rows are not in exact order")
    audit = manifest.get("audit")
    if not isinstance(audit, dict):
        raise ValueError("dataset_manifest.json has no audit object")
    dataset_fingerprint = str(audit.get("dataset_fingerprint", ""))
    feature_fingerprint = str(audit.get("feature_bundle_fingerprint", ""))
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
        "visual_features": matrices.get("visual"),
        "sensor_features": matrices.get("sensor"),
        "dataset_fingerprint": dataset_fingerprint,
        "feature_bundle_fingerprint": feature_fingerprint,
        "manifest_sha256": _sha256(manifest_path),
        "features_sha256": _sha256(features_path),
    }


def _seed_sensitivity(inputs: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    aggregate_gate_counts: Counter[str] = Counter()
    for seed in OUTER_SEEDS:
        evaluation = hierarchical_offline_evaluation(
            inputs["records"],
            inputs["visual_features"],
            inputs["sensor_features"],
            sample_ids=inputs["sample_ids"],
            outer_design=SGKF5,
            outer_seed=seed,
        )
        metrics = evaluation["nested_hierarchical_metrics"]
        gate_counts = Counter(
            fold["inner_gate_selection"]["selected_gate_id"]
            for fold in evaluation["folds"]
        )
        aggregate_gate_counts.update(gate_counts)
        rows.append(
            {
                "seed": seed,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "confusion_matrix": metrics["confusion_matrix"],
                "selected_gate_counts": dict(sorted(gate_counts.items())),
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
            "fraction_at_least_0_9": round(float((values >= TARGET_ACCURACY).mean()), 6)
            if field == "accuracy"
            else None,
        }

    return {
        "varied_parameter": "outer StratifiedGroupKFold random_state only",
        "inner_seed_fixed": 2026,
        "seed_start": OUTER_SEEDS[0],
        "seed_end_inclusive": OUTER_SEEDS[-1],
        "seed_count": len(OUTER_SEEDS),
        "metrics": {
            "accuracy": summary("accuracy"),
            "macro_f1": summary("macro_f1"),
        },
        "aggregate_selected_gate_counts": dict(
            sorted(aggregate_gate_counts.items())
        ),
        "per_seed": rows,
        "post_selection_exploratory": True,
        "confirmatory_interpretation_allowed": False,
    }


def _session_cluster_sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sessions = sorted({str(row["session_id"]) for row in rows})
    indices_by_session = {
        session: np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if str(row["session_id"]) == session
            ],
            dtype=int,
        )
        for session in sessions
    }
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    predictions = np.asarray(
        [int(row["nested_hierarchical_prediction"]) for row in rows], dtype=int
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    accuracy_draws = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    macro_draws = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        chosen = rng.choice(sessions, size=len(sessions), replace=True)
        sampled = np.concatenate([indices_by_session[str(value)] for value in chosen])
        accuracy_draws[replicate] = accuracy_score(
            labels[sampled], predictions[sampled]
        )
        macro_draws[replicate] = f1_score(
            labels[sampled],
            predictions[sampled],
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    return {
        "cluster_unit": "session_id",
        "session_count": len(sessions),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "descriptive_percentile_interval_95": {
            "accuracy": [
                round(float(value), 6)
                for value in np.quantile(accuracy_draws, [0.025, 0.975])
            ],
            "macro_f1": [
                round(float(value), 6)
                for value in np.quantile(macro_draws, [0.025, 0.975])
            ],
        },
        "confirmatory_confidence_interval": False,
        "limitation": (
            "The method was designed after inspecting this dataset. Cluster "
            "resampling describes sensitivity but does not repair post-selection bias."
        ),
    }


def _paired_session_cluster_gain(
    fusion_rows: list[dict[str, Any]], comparator_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if [row["sample_id"] for row in fusion_rows] != [
        row["sample_id"] for row in comparator_rows
    ]:
        raise ValueError("paired modality rows are not in exact sample order")
    sessions = sorted({str(row["session_id"]) for row in fusion_rows})
    indices_by_session = {
        session: np.asarray(
            [
                index
                for index, row in enumerate(fusion_rows)
                if str(row["session_id"]) == session
            ],
            dtype=int,
        )
        for session in sessions
    }
    labels = np.asarray([int(row["label"]) for row in fusion_rows], dtype=int)
    fusion = np.asarray(
        [int(row["nested_hierarchical_prediction"]) for row in fusion_rows],
        dtype=int,
    )
    comparator = np.asarray(
        [int(row["nested_hierarchical_prediction"]) for row in comparator_rows],
        dtype=int,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    accuracy_delta = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    macro_delta = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        chosen = rng.choice(sessions, size=len(sessions), replace=True)
        sampled = np.concatenate([indices_by_session[str(value)] for value in chosen])
        accuracy_delta[replicate] = accuracy_score(
            labels[sampled], fusion[sampled]
        ) - accuracy_score(labels[sampled], comparator[sampled])
        macro_delta[replicate] = f1_score(
            labels[sampled],
            fusion[sampled],
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        ) - f1_score(
            labels[sampled],
            comparator[sampled],
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    return {
        "cluster_unit": "session_id",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "descriptive_percentile_interval_95": {
            "accuracy_delta": [
                round(float(value), 6)
                for value in np.quantile(accuracy_delta, [0.025, 0.975])
            ],
            "macro_f1_delta": [
                round(float(value), 6)
                for value in np.quantile(macro_delta, [0.025, 0.975])
            ],
        },
        "confirmatory_multimodal_inference_allowed": False,
        "limitation": (
            "This paired cluster sensitivity is conditional on a modality/fusion "
            "pipeline designed after inspecting the same development dataset."
        ),
    }


def _prior_causal_reference(artifact_dir: Path) -> dict[str, Any] | None:
    path = artifact_dir / "optimization_report.json"
    if not path.exists():
        return None
    report = _read_json(path)
    temporal = report.get("temporal_linear_svc")
    if not isinstance(temporal, dict):
        return None
    evaluations = temporal.get("evaluations")
    if not isinstance(evaluations, dict):
        return None
    sgkf = evaluations.get("stratified_session_5fold")
    loso = evaluations.get("leave_one_session_out")
    if not isinstance(sgkf, dict) or not isinstance(loso, dict):
        return None
    return {
        "source_path": path.name,
        "source_sha256": _sha256(path),
        "evaluation_kind": "post_selected_strict_causal_reference",
        "session_sgkf5": sgkf["metrics"]["selected_causal"],
        "leave_one_session_out": loso["metrics"]["selected_causal"],
        "real_time_compatible": True,
        "post_selection_exploratory": True,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    sgkf = report["nested_offline_full_session"][SGKF5][
        "nested_hierarchical_metrics"
    ]
    loso = report["nested_offline_full_session"][LEAVE_ONE_SESSION_OUT][
        "nested_hierarchical_metrics"
    ]
    seed = report["outer_seed_sensitivity"]["metrics"]
    stress = report["fixed_gate_block_stress"]
    oracle = report["nested_offline_full_session"][SGKF5][
        "constant_sequence_oracle"
    ]["metrics"]
    session_target_counts = report["nested_offline_full_session"][SGKF5][
        "session_majority_target_audit"
    ]["class_counts"]
    bootstrap = report["session_cluster_sensitivity"]
    ablation = report["nested_modality_ablation"]
    counterfactual = report["context_counterfactual"]
    causal = report.get("strict_causal_reference")

    def pair(metrics: dict[str, Any]) -> str:
        return f"{metrics['accuracy']:.4f} / {metrics['macro_f1']:.4f}"

    lines = [
        "# DIPSER 离线层级算法：0.9 挑战与可信度审计",
        "",
        "## 结论",
        "",
        (
            "在 296 个真实、完整的视觉+手表多模态窗口上，严格 inner-fold 重训的离线层级模型，"
            f"在固定 SGKF5 seed=2026 上达到 **Accuracy {sgkf['accuracy']:.4f}** "
            f"和 **Macro-F1 {sgkf['macro_f1']:.4f}**。这是真实计算得到的 OOF 点估计。"
            "SGKF5 使用全体标签构造分层 session 外折；除此之外，outer-test 标签不参与"
            "拟合、上下文聚合或 gate 选择，身份、路径和样本数也不作为特征。"
        ),
        "",
        (
            "但它没有建立可信的 0.9：确定性 LOSO 只有 "
            f"**{loso['accuracy']:.4f}**；50 个 outer seed 的均值为 "
            f"**{seed['accuracy']['mean']:.4f}**，范围 "
            f"[{seed['accuracy']['minimum']:.4f}, {seed['accuracy']['maximum']:.4f}]，"
            f"只有 {seed['accuracy']['fraction_at_least_0_9']:.0%} 达到 0.9。"
        ),
        "",
        (
            "该模型必须看到 held-out participant 的完整序列，并用完整课堂 session 中其他学生和未来窗口的"
            "无标签特征做 session median/IQR。因此正确名称是 `offline full-session transductive`；"
            "它不是实时、单学生、归纳式或部署 Accuracy。"
        ),
        "",
        "## 核心结果",
        "",
        "| 设计 | Accuracy / Macro-F1 | 解释 |",
        "|---|---:|---|",
        f"| SGKF5 seed=2026 | **{pair(sgkf)}** | 267/296；事后开发点估计 |",
        f"| Leave-one-session-out | **{pair(loso)}** | 266/296；确定性 session 留一 |",
        (
            f"| 50-seed SGKF 中心 | **{seed['accuracy']['mean']:.4f} / "
            f"{seed['macro_f1']['mean']:.4f}** | 21/50 达到 0.9020 |"
        ),
    ]
    if causal is not None:
        lines.append(
            f"| 严格因果 SGKF5 参考 | **{pair(causal['session_sgkf5'])}** | "
            "只用当前/历史，同 participant，gap>60 秒重置 |"
        )
    lines.extend(
        [
        f"| 每 sequence 恒定标签 oracle | **{pair(oracle)}** | 标签先知诊断，绝非模型分数 |",
            "",
            "SGKF5 混淆矩阵（行是真实 low/medium/high，列是预测）为：",
            "",
            f"`{sgkf['confusion_matrix']}`",
            "",
            "逐类 Recall："
            f"low={sgkf['per_class']['low']['recall']:.4f}，"
            f"medium={sgkf['per_class']['medium']['recall']:.4f}，"
            f"high={sgkf['per_class']['high']['recall']:.4f}。"
            "低类只有 26 个样本，Recall 仍仅 0.2692，故 Accuracy 0.9 不代表三类均衡识别。",
            "",
            "## 同协议多模态消融",
            "",
            "每个模态都使用同一 nested outer/inner 协议，并在各自 outer-training 内选择 gate：",
            "",
            "| 模态 | SGKF5 Accuracy / Macro-F1 | LOSO Accuracy / Macro-F1 |",
            "|---|---:|---:|",
            f"| Visual | {pair(ablation['sgkf5']['visual']['metrics'])} | {pair(ablation['leave_one_session_out']['visual']['metrics'])} |",
            f"| Sensor | {pair(ablation['sgkf5']['sensor']['metrics'])} | {pair(ablation['leave_one_session_out']['sensor']['metrics'])} |",
            f"| Fusion | **{pair(ablation['sgkf5']['fusion']['metrics'])}** | **{pair(ablation['leave_one_session_out']['fusion']['metrics'])}** |",
            "",
            (
                "以 outer 描述指标较高的单模态 Sensor 作保守、非预声明比较，Fusion 的 "
                "SGKF5 Accuracy / Macro-F1 观察增量为 "
                f"+{ablation['sgkf5']['fusion_minus_best_unimodal']['accuracy_delta']:.4f} / "
                f"+{ablation['sgkf5']['fusion_minus_best_unimodal']['macro_f1_delta']:.4f}；"
                "LOSO 增量为 "
                f"+{ablation['leave_one_session_out']['fusion_minus_best_unimodal']['accuracy_delta']:.4f} / "
                f"+{ablation['leave_one_session_out']['fusion_minus_best_unimodal']['macro_f1_delta']:.4f}。"
            ),
            "",
            (
                "SGKF5 的 paired session-cluster 描述性 Accuracy-delta 区间为 "
                f"{ablation['sgkf5']['paired_session_cluster_gain_vs_best_unimodal']['descriptive_percentile_interval_95']['accuracy_delta']}。"
                "它支持当前数据上的多模态互补现象，但由于算法和特征组合是在同一数据上事后形成，"
                "尚不能建立确认性的多模态增益。"
            ),
            "",
            "## 完整测试 batch / 未来上下文反事实",
            "",
            "每种上下文都在各 outer-training partition 内重新做 inner gate 选择：",
            "",
            "| 上下文 | SGKF5 Accuracy / Macro-F1 | LOSO Accuracy / Macro-F1 |",
            "|---|---:|---:|",
            f"| 完整 sequence + session | **{pair(counterfactual['sgkf5']['mode_results']['full']['metrics'])}** | **{pair(counterfactual['leave_one_session_out']['mode_results']['full']['metrics'])}** |",
            f"| 排除当前预测行 | {pair(counterfactual['sgkf5']['mode_results']['leave_current_out']['metrics'])} | {pair(counterfactual['leave_one_session_out']['mode_results']['leave_current_out']['metrics'])} |",
            f"| 严格时间前缀，gap>60秒重置 | {pair(counterfactual['sgkf5']['mode_results']['causal_prefix']['metrics'])} | {pair(counterfactual['leave_one_session_out']['mode_results']['causal_prefix']['metrics'])} |",
            "",
            (
                "SGKF5 完整 batch 中，"
                f"{counterfactual['sgkf5']['mode_results']['full']['rows_using_future_sequence_context']} 行使用了未来 sequence 特征，"
                f"{counterfactual['sgkf5']['mode_results']['full']['rows_using_future_session_context']} 行使用了未来 session 特征，"
                f"{counterfactual['sgkf5']['mode_results']['full']['rows_using_other_participant_session_context']} 行使用了其他学生上下文。"
                "严格前缀审计的未来来源计数为 0。完整 batch 改成排除当前行即下降到 0.8547，"
                "说明 0.902 对 transductive 测试上下文有实质依赖。"
            ),
            "",
            "## 跨 cohort / activity 压力测试",
            "",
            "固定一个在 SGKF2026+LOSO 主开发评估中频繁入选、且不依赖概率阈值的事后 gate "
            "`high_override_plus_low_confirmation`；不在压力测试外折中重新调 gate：",
            "",
            "| 阻断设计 | Accuracy / Macro-F1 |",
            "|---|---:|",
            f"| LOCO | {pair(stress[LEAVE_ONE_COHORT_OUT]['metrics'])} |",
            f"| LOAO | {pair(stress[LEAVE_ONE_ACTIVITY_OUT]['metrics'])} |",
            f"| Cohort + activity 双重阻断 | {pair(stress[DOUBLE_BLOCKED_COHORT_ACTIVITY]['metrics'])} |",
            "",
            (
                "LOCO 和双重阻断大幅下降，说明模型很可能利用了单站点 cohort/activity/设备/环境的结构信号。"
                "因此跨 cohort、多课堂和部署泛化均未建立。"
            ),
            "",
            "## 为什么固定 split 能到 0.902",
            "",
            (
                f"模型最终对每个 participant-session 输出常数标签，而这种受限预测的标签先知上界仅为 "
                f"{oracle['accuracy']:.4f}。模型的 0.9020 已几乎贴住该结构上界，说明它主要识别整段 recording 的主状态，"
                "不是把每个窗口的动态状态都识别到 90%。"
            ),
            "",
            (
                "session-level RBF-SVC 的有效监督单位只有 25 个 session；完整数据的 majority-target "
                f"计数为 low={session_target_counts.get('0', 0)}、"
                f"medium={session_target_counts.get('1', 0)}、high={session_target_counts.get('2', 0)}，"
                "即没有 low-majority session；"
                "low 由 participant-sequence 分支处理。这种小样本层级结构使单个划分和概率校准更敏感。"
            ),
            "",
            "session 整簇重采样的描述性 95% 区间为 Accuracy "
            f"{bootstrap['descriptive_percentile_interval_95']['accuracy']}；"
            "因算法是在本数据上事后设计，这不是确认性置信区间。",
            "",
            "## 可接受的陈述",
            "",
            "可以陈述：**在当前单站点 DIPSER 开发数据上，一个完整 session 离线批处理层级模型，在固定 SGKF5 划分触及 0.9020。**",
            "",
            "不能陈述：可信跨 session Accuracy 已稳定达到 0.9、实时视频识别达到 0.9、跨 cohort/学校达到 0.9，或部署 Accuracy 达到 0.9。",
            "",
            "下一步确认必须冻结全部代码和阈值，在从未参与设计的新课堂/新 cohort 上一次性评估；预注册主指标应同时包含 Accuracy、Macro-F1 和每类 Recall。",
            "",
            "## 复现",
            "",
            "```bash",
            "python3 scripts/run_dipser_hierarchical_challenge.py \\",
            "  --artifact-dir artifacts/dipser_credible/full_v5_52_complete_v5",
            "```",
            "",
            "输出：`hierarchical_0_9_report.json` 和本文件。",
            "",
        ]
    )
    return "\n".join(lines)


def run(artifact_dir: Path) -> dict[str, Any]:
    inputs = _load_inputs(artifact_dir)
    evaluation_inputs = {
        key: inputs[key]
        for key in (
            "records",
            "visual_features",
            "sensor_features",
            "sample_ids",
        )
    }
    sgkf = hierarchical_offline_evaluation(
        **evaluation_inputs, outer_design=SGKF5, outer_seed=2026
    )
    loso = hierarchical_offline_evaluation(
        **evaluation_inputs, outer_design=LEAVE_ONE_SESSION_OUT
    )
    modality_reports: dict[str, dict[str, Any]] = {
        SGKF5: {"fusion": sgkf},
        LEAVE_ONE_SESSION_OUT: {"fusion": loso},
    }
    for design in (SGKF5, LEAVE_ONE_SESSION_OUT):
        for feature_mode in ("visual", "sensor"):
            modality_reports[design][feature_mode] = hierarchical_offline_evaluation(
                **evaluation_inputs,
                outer_design=design,
                outer_seed=2026,
                feature_mode=feature_mode,
            )

    modality_ablation: dict[str, Any] = {}
    for design, evaluations in modality_reports.items():
        compact: dict[str, Any] = {}
        for feature_mode, evaluation in evaluations.items():
            compact[feature_mode] = {
                "metrics": evaluation["nested_hierarchical_metrics"],
                "selected_gate_ids": [
                    fold["inner_gate_selection"]["selected_gate_id"]
                    for fold in evaluation["folds"]
                ],
                "feature_dimensions": evaluation["feature_dimensions"],
                "outer_test_labels_used_for_training_or_gate_selection": False,
                "outer_labels_used_for_stratified_split_construction": (
                    design == SGKF5
                ),
            }
        best_unimodal = max(
            ("visual", "sensor"),
            key=lambda mode: (
                compact[mode]["metrics"]["accuracy"],
                compact[mode]["metrics"]["macro_f1"],
            ),
        )
        compact["fusion_minus_best_unimodal"] = {
            "best_unimodal": best_unimodal,
            "best_unimodal_selected_from_outer_descriptive_metrics": True,
            "selection_is_confirmatory": False,
            "accuracy_delta": round(
                compact["fusion"]["metrics"]["accuracy"]
                - compact[best_unimodal]["metrics"]["accuracy"],
                6,
            ),
            "macro_f1_delta": round(
                compact["fusion"]["metrics"]["macro_f1"]
                - compact[best_unimodal]["metrics"]["macro_f1"],
                6,
            ),
            "confirmatory_gain_established": False,
        }
        compact["paired_session_cluster_gain_vs_best_unimodal"] = (
            _paired_session_cluster_gain(
                evaluations["fusion"]["oof_predictions"],
                evaluations[best_unimodal]["oof_predictions"],
            )
        )
        modality_ablation[design] = compact
    context_evaluations = {
        design: hierarchical_counterfactual_evaluation(
            **evaluation_inputs,
            outer_design=design,
            outer_seed=2026,
        )
        for design in (SGKF5, LEAVE_ONE_SESSION_OUT)
    }
    context_counterfactual: dict[str, Any] = {}
    for design, evaluation in context_evaluations.items():
        primary = modality_reports[design]["fusion"]
        primary_predictions = [
            int(row["nested_hierarchical_prediction"])
            for row in primary["oof_predictions"]
        ]
        counterfactual_full_predictions = [
            int(row["modes"][FULL]["prediction"])
            for row in evaluation["oof_predictions"]
        ]
        mismatch_count = sum(
            left != right
            for left, right in zip(
                primary_predictions, counterfactual_full_predictions
            )
        )
        if mismatch_count:
            raise RuntimeError(
                f"counterfactual full mode differs from primary {design}: "
                f"{mismatch_count} rows"
            )
        context_counterfactual[design] = {
            "mode_results": evaluation["mode_results"],
            "full_mode_matches_primary_prediction_rows": True,
            "full_mode_prediction_mismatch_count": mismatch_count,
            "outer_test_labels_used_for_training_or_gate_selection": False,
            "outer_labels_used_for_stratified_split_construction": (
                design == SGKF5
            ),
            "fold_selected_gate_ids_by_mode": [
                {
                    mode: fold["gate_selection_by_mode"][mode][
                        "selected_gate_id"
                    ]
                    for mode in (FULL, LEAVE_CURRENT_OUT, CAUSAL_PREFIX)
                }
                for fold in evaluation["folds"]
            ],
            "claim_status": evaluation["claim_status"],
        }
    stress = {
        design: fixed_hierarchical_stress_evaluation(
            **evaluation_inputs,
            design=design,
            gate_id=FIXED_STRESS_GATE,
        )
        for design in (
            LEAVE_ONE_COHORT_OUT,
            LEAVE_ONE_ACTIVITY_OUT,
            DOUBLE_BLOCKED_COHORT_ACTIVITY,
        )
    }
    seed_sensitivity = _seed_sensitivity(inputs)
    source_path = PROJECT_ROOT / "teaching_skill_miner/recognition/hierarchical_evaluation.py"
    counterfactual_source_path = (
        PROJECT_ROOT
        / "teaching_skill_miner/recognition/hierarchical_counterfactual.py"
    )
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "evaluation_kind": "post_selected_offline_hierarchical_0_9_challenge",
        "artifact_binding": {
            "dataset_fingerprint": inputs["dataset_fingerprint"],
            "feature_bundle_fingerprint": inputs["feature_bundle_fingerprint"],
            "dataset_manifest_sha256": inputs["manifest_sha256"],
            "features_sha256": inputs["features_sha256"],
            "evaluation_source_sha256": _sha256(source_path),
            "counterfactual_source_sha256": _sha256(
                counterfactual_source_path
            ),
            "runner_source_sha256": _sha256(Path(__file__)),
            "sample_order_verified": True,
        },
        "runtime_environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": package_version("scikit-learn"),
            "scipy": package_version("scipy"),
        },
        "requested_target": {
            "target_accuracy": TARGET_ACCURACY,
            "fixed_sgkf_seed_2026_reached": (
                sgkf["nested_hierarchical_metrics"]["accuracy"] >= TARGET_ACCURACY
            ),
            "leave_one_session_out_reached": (
                loso["nested_hierarchical_metrics"]["accuracy"] >= TARGET_ACCURACY
            ),
            "fifty_seed_mean_reached": (
                seed_sensitivity["metrics"]["accuracy"]["mean"] >= TARGET_ACCURACY
            ),
            "cross_cohort_reached": (
                stress[LEAVE_ONE_COHORT_OUT]["metrics"]["accuracy"]
                >= TARGET_ACCURACY
            ),
            "credible_deployment_target_established": False,
        },
        "nested_offline_full_session": {SGKF5: sgkf, LEAVE_ONE_SESSION_OUT: loso},
        "nested_modality_ablation": modality_ablation,
        "context_counterfactual": context_counterfactual,
        "outer_seed_sensitivity": seed_sensitivity,
        "fixed_gate_block_stress": stress,
        "session_cluster_sensitivity": _session_cluster_sensitivity(
            sgkf["oof_predictions"]
        ),
        "strict_causal_reference": _prior_causal_reference(artifact_dir),
        "claim_status": {
            "accuracy_0_9_established": False,
            "stable_cross_session_accuracy_0_9_established": False,
            "multimodal_gain_established": False,
            "real_time_accuracy_established": False,
            "cross_cohort_accuracy_established": False,
            "cross_activity_accuracy_established": False,
            "deployment_accuracy_established": False,
        },
        "claim_limitation": (
            "The 0.902027 score is a reproducible point estimate for one "
            "post-selected, full-session transductive development split. LOSO, "
            "50-seed central tendency, cohort blocking, causal inference, and "
            "class-balanced performance do not establish a stable or deployment 0.9."
        ),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/dipser_credible/full_v5_52_complete_v5"),
    )
    args = parser.parse_args()
    artifact_dir = args.artifact_dir.resolve()
    report = run(artifact_dir)
    report_path = artifact_dir / "hierarchical_0_9_report.json"
    markdown_path = artifact_dir / "HIERARCHICAL_0_9_RESULTS.md"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    summary = {
        "report": str(report_path),
        "markdown": str(markdown_path),
        "sgkf_accuracy": report["nested_offline_full_session"][SGKF5][
            "nested_hierarchical_metrics"
        ]["accuracy"],
        "loso_accuracy": report["nested_offline_full_session"][
            LEAVE_ONE_SESSION_OUT
        ]["nested_hierarchical_metrics"]["accuracy"],
        "fifty_seed_mean_accuracy": report["outer_seed_sensitivity"]["metrics"][
            "accuracy"
        ]["mean"],
        "credible_deployment_target_established": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
