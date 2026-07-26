"""Real classroom recognition with independent human labels.

This package is intentionally separate from the Teaching Skill internal-consistency
evaluator. Recognition metrics here are computed from held-out human ground truth.
"""

from .datasets import discover_ouc_cge
from .blocked_evaluation import blocked_descriptive_evaluation
from .dipser import (
    build_expert_attention_band_ground_truth,
    build_expert_attention_ground_truth,
    build_source_provenance,
    extract_visual_pose_features,
    extract_watch_features,
)
from .experiment import infer_classroom_engagement, run_real_classroom_benchmark
from .offline_sequence_evaluation import offline_sequence_evaluation
from .hierarchical_evaluation import (
    fixed_hierarchical_stress_evaluation,
    hierarchical_offline_evaluation,
)
from .hierarchical_counterfactual import hierarchical_counterfactual_evaluation
from .optimized_evaluation import optimized_multimodal_evaluation
from .temporal_evaluation import temporal_linear_svc_evaluation
from .strict_evaluation import (
    ClaimContract,
    FrozenDeploymentModel,
    evaluate_frozen_external_deployment,
    fit_frozen_deployment_model,
    frozen_evaluation_report_fingerprint,
    frozen_model_to_artifact,
    load_frozen_deployment_model,
    strict_dataset_fingerprint,
    strict_coverage_evidence_fingerprint,
    strict_feature_bundle_fingerprint,
    validate_evaluation_coverage,
)
from .raw_feature_bridge import (
    extract_strict_feature_bundle,
    raw_source_binding_fingerprint,
    synchronized_content_sha256,
    validate_bundle_against_frozen_model,
    validate_extractor_suite_binding,
    validate_raw_source_binding,
)
from .dipser_experiment import (
    build_dipser_catalog,
    build_dipser_dataset,
    feature_bundle_fingerprint,
    run_dipser_credible_experiment,
)

__all__ = [
    "build_expert_attention_band_ground_truth",
    "build_expert_attention_ground_truth",
    "build_source_provenance",
    "ClaimContract",
    "FrozenDeploymentModel",
    "build_dipser_catalog",
    "build_dipser_dataset",
    "blocked_descriptive_evaluation",
    "discover_ouc_cge",
    "extract_visual_pose_features",
    "extract_watch_features",
    "evaluate_frozen_external_deployment",
    "extract_strict_feature_bundle",
    "fit_frozen_deployment_model",
    "frozen_evaluation_report_fingerprint",
    "frozen_model_to_artifact",
    "feature_bundle_fingerprint",
    "infer_classroom_engagement",
    "fixed_hierarchical_stress_evaluation",
    "hierarchical_offline_evaluation",
    "hierarchical_counterfactual_evaluation",
    "load_frozen_deployment_model",
    "raw_source_binding_fingerprint",
    "strict_dataset_fingerprint",
    "strict_coverage_evidence_fingerprint",
    "strict_feature_bundle_fingerprint",
    "synchronized_content_sha256",
    "validate_bundle_against_frozen_model",
    "validate_extractor_suite_binding",
    "validate_evaluation_coverage",
    "validate_raw_source_binding",
    "offline_sequence_evaluation",
    "optimized_multimodal_evaluation",
    "run_real_classroom_benchmark",
    "run_dipser_credible_experiment",
    "temporal_linear_svc_evaluation",
]
