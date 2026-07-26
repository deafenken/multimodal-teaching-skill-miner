from __future__ import annotations

import copy
from contextlib import redirect_stdout
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional recognition dependency
    CRYPTOGRAPHY_AVAILABLE = False

try:
    import jsonschema

    JSONSCHEMA_AVAILABLE = True
except ImportError:  # pragma: no cover - dev dependency
    JSONSCHEMA_AVAILABLE = False

from scripts.prepare_teachobs_lockbox_preregistration import main as prepare_main
from teaching_skill_miner.cli import main as cli_main
from teaching_skill_miner.external_evidence import finalize_external_research_evidence
from teaching_skill_miner.io_utils import project_root, read_json
from scripts.check_teachobs_external_study import StageCheckError, check_lockbox
from teaching_skill_miner.teachobs_lockbox import (
    ARM_ORDER,
    TeachObsLockboxError,
    build_teachobs_lockbox_preregistration,
    sign_teachobs_lockbox_preregistration,
    teachobs_analysis_plan_fingerprint,
    validate_teachobs_confirmatory_evidence_handoff,
    validate_teachobs_lockbox_preregistration,
    verify_teachobs_lockbox_artifact_files,
    verify_teachobs_lockbox_preregistration_attestation,
)


class TeachObsLockboxPreregistrationTests(unittest.TestCase):
    def _digest(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _artifact_files(
        self,
        root: Path,
        *,
        fixture: str = "fixture",
    ) -> tuple[Path, Path, dict[str, Path]]:
        import numpy as np

        from teaching_skill_miner import teachobs_multimodal_benchmark as benchmark
        from teaching_skill_miner.teachobs_frozen_model import (
            export_teachobs_frozen_bundle,
        )

        root.mkdir(parents=True, exist_ok=True)
        model_root = root / "frozen_models"
        analysis = root / "analysis.py"
        analysis.write_text(
            f"# test-only fixed paired bootstrap: {fixture}\n",
            encoding="utf-8",
        )
        digest = self._digest(fixture)
        audio_feature_schema_sha256 = hashlib.sha256(
            json.dumps(
                list(benchmark._AUDIO_FEATURE_NAMES),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        visual_feature_schema_sha256 = hashlib.sha256(
            json.dumps(
                list(benchmark._VISUAL_NUMERIC_FEATURE_NAMES),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        blocks = {
            "transcript_only": {"transcript_tfidf": 1},
            "transcript_audio": {
                "transcript_tfidf": 1,
                "audio_numeric": len(benchmark._AUDIO_FEATURE_NAMES),
            },
            "transcript_visual": {
                "transcript_tfidf": 1,
                "ocr_char_tfidf": 1,
                "ocr_word_tfidf": 1,
                "visual_numeric": len(benchmark._VISUAL_NUMERIC_FEATURE_NAMES),
                "clip_embedding": 2,
            },
            "full": {
                "transcript_tfidf": 1,
                "audio_numeric": len(benchmark._AUDIO_FEATURE_NAMES),
                "ocr_char_tfidf": 1,
                "ocr_word_tfidf": 1,
                "visual_numeric": len(benchmark._VISUAL_NUMERIC_FEATURE_NAMES),
                "clip_embedding": 2,
            },
        }
        fitted = {}
        for arm_index, (arm, dimensions) in enumerate(blocks.items()):
            total = sum(dimensions.values())
            fitted[arm] = {
                "coefficient": np.zeros((39, total), dtype=np.float64),
                "intercept": np.zeros(39, dtype=np.float64),
                "thresholds": np.full(
                    39, (0.3, 0.4, 0.5, 0.6)[arm_index], dtype=np.float64
                ),
                "model_kinds": ["constant_0"] * 39,
                "model_classes": [[0] for _ in range(39)],
                "regularization_c": 1.0,
                "class_weight": "balanced",
                "selection_provenance_sha256": self._digest(
                    f"{fixture}:{arm}:train-oof-selection"
                ),
            }
        profile_spec = benchmark.TEACHOBS_BENCHMARK_PROFILES[
            benchmark.FULL_23_TRAIN_7_TEST_PROFILE
        ]
        train_lesson_ids = [
            f"S{index}"
            for index in range(1, 31)
            if f"S{index}" not in profile_spec.test_lesson_ids
        ]
        test_lesson_ids = list(profile_spec.test_lesson_ids)
        train_sample_ids = [f"{lesson_id}:1" for lesson_id in train_lesson_ids]
        test_sample_ids = [f"{lesson_id}:1" for lesson_id in test_lesson_ids]
        profile_contract = benchmark._dataset_profile_contract(
            spec=profile_spec,
            repository_dataset_sha256=digest,
            train_lesson_ids=train_lesson_ids,
            test_lesson_ids=test_lesson_ids,
            train_sample_ids=train_sample_ids,
            test_sample_ids=test_sample_ids,
        )
        profile_binding = {
            "benchmark_profile": profile_spec.profile_id,
            "profile_source_alignment": profile_spec.source_alignment,
            "published_six_lesson_intersection": False,
            "excluded_official_test_lesson_ids": [],
            "selected_train_lesson_ids": train_lesson_ids,
            "selected_test_lesson_ids": test_lesson_ids,
            "selected_train_sample_ids": train_sample_ids,
            "selected_test_sample_ids": test_sample_ids,
            "selected_train_sample_order_sha256": hashlib.sha256(
                json.dumps(
                    train_sample_ids,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "selected_test_sample_order_sha256": hashlib.sha256(
                json.dumps(
                    test_sample_ids,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "dataset_profile_fingerprint": hashlib.sha256(
                json.dumps(
                    profile_contract,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
        transcript_materialization = {
            "schema": benchmark.TRANSCRIPT_MATERIALIZATION_SCHEMA,
            "profile_id": profile_spec.profile_id,
            "manifest_file_sha256": self._digest(
                f"{fixture}:transcript-manifest-file"
            ),
            "manifest_sha256": self._digest(f"{fixture}:transcript-manifest"),
            "materialization_fingerprint_sha256": self._digest(
                f"{fixture}:transcript-materialization"
            ),
            "ordered_sample_id_sha256": self._digest(
                f"{fixture}:transcript-sample-order"
            ),
            "ordered_scene_text_sha256": self._digest(
                f"{fixture}:transcript-text-order"
            ),
            "repository_binding_sha256": self._digest(
                f"{fixture}:transcript-repository-binding"
            ),
            "input_hashes_sha256": self._digest(
                f"{fixture}:transcript-input-hashes"
            ),
            "selected_train_transcript_order_sha256": self._digest(
                f"{fixture}:train-transcript-order"
            ),
            "selected_test_transcript_order_sha256": self._digest(
                f"{fixture}:test-transcript-order"
            ),
            "released_transcript_fallback_used": False,
            "labels_read_or_used": False,
        }
        benchmark_input_contract = benchmark._benchmark_input_contract(
            dataset_profile_fingerprint=profile_binding[
                "dataset_profile_fingerprint"
            ],
            feature_manifest_sha256=digest,
            transcript_materialization=transcript_materialization,
        )
        benchmark_input_fingerprint = hashlib.sha256(
            json.dumps(
                benchmark_input_contract,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        export_teachobs_frozen_bundle(
            model_root,
            label_names=[f"code_{index:02d}" for index in range(39)],
            label_groups=["visual"] * 20 + ["nonvisual"] * 19,
            feature_blocks_by_arm=blocks,
            text_states={
                "transcript_tfidf": {
                    "configuration": {
                        "analyzer": "char_wb",
                        "ngram_range": list(benchmark._TFIDF_NGRAM_RANGE),
                        "lowercase": True,
                        "min_df": 1,
                        "max_features": benchmark._TFIDF_MAX_FEATURES,
                        "sublinear_tf": True,
                        "norm": "l2",
                        "use_idf": True,
                        "smooth_idf": True,
                    },
                    "vocabulary": {"a": 0},
                    "idf": np.ones(1),
                },
                "ocr_char_tfidf": {
                    "configuration": {
                        "analyzer": "char_wb",
                        "ngram_range": list(benchmark._OCR_CHAR_NGRAM_RANGE),
                        "lowercase": True,
                        "min_df": 1,
                        "max_features": benchmark._OCR_CHAR_MAX_FEATURES,
                        "sublinear_tf": True,
                        "norm": "l2",
                        "use_idf": True,
                        "smooth_idf": True,
                    },
                    "vocabulary": {"a": 0},
                    "idf": np.ones(1),
                },
                "ocr_word_tfidf": {
                    "configuration": {
                        "analyzer": "word",
                        "ngram_range": list(benchmark._OCR_WORD_NGRAM_RANGE),
                        "lowercase": True,
                        "min_df": 1,
                        "max_features": benchmark._OCR_WORD_MAX_FEATURES,
                        "sublinear_tf": True,
                        "norm": "l2",
                        "use_idf": True,
                        "smooth_idf": True,
                    },
                    "vocabulary": {"a": 0},
                    "idf": np.ones(1),
                },
            },
            numeric_states={
                "audio_numeric": {
                    "mean": np.zeros(len(benchmark._AUDIO_FEATURE_NAMES)),
                    "scale": np.ones(len(benchmark._AUDIO_FEATURE_NAMES)),
                },
                "visual_numeric": {
                    "mean": np.zeros(len(benchmark._VISUAL_NUMERIC_FEATURE_NAMES)),
                    "scale": np.ones(len(benchmark._VISUAL_NUMERIC_FEATURE_NAMES)),
                },
                "clip_embedding": {"mean": np.zeros(2), "scale": np.ones(2)},
            },
            numeric_feature_names={
                "audio_numeric": benchmark._AUDIO_FEATURE_NAMES,
                "visual_numeric": benchmark._VISUAL_NUMERIC_FEATURE_NAMES,
            },
            numeric_provenance={
                "audio_numeric": {
                    "feature_schema_sha256": audio_feature_schema_sha256,
                    "audio_feature_schema": benchmark.AUDIO_SCHEMA,
                },
                "visual_numeric": {
                    "feature_schema_sha256": visual_feature_schema_sha256,
                    "visual_evidence_configuration_set_sha256": digest,
                    "visual_evidence_schema": benchmark.VISUAL_EVIDENCE_SCHEMA,
                    "configuration_sha256_values": [digest],
                },
                "clip_embedding": {
                    "clip_source_revision": "test-only-revision",
                    "clip_weight_manifest_sha256": digest,
                },
            },
            fitted_arm_states=fitted,
            training_provenance={
                **profile_binding,
                "training_repository_dataset_sha256": digest,
                "feature_manifest_sha256": digest,
                "transcript_materialization": transcript_materialization,
                "benchmark_input_fingerprint": benchmark_input_fingerprint,
                "visual_evidence_set_sha256": digest,
                "visual_evidence_configuration_set_sha256": digest,
                "label_order_sha256": hashlib.sha256(
                    json.dumps(
                        [f"code_{index:02d}" for index in range(39)],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "benchmark_configuration_sha256": digest,
                "benchmark_source_sha256": hashlib.sha256(
                    Path(benchmark.__file__).read_bytes()
                ).hexdigest(),
            },
            software_provenance={
                "python": sys.version.split()[0],
                "numpy": importlib.metadata.version("numpy"),
                "scipy": importlib.metadata.version("scipy"),
                "scikit_learn": importlib.metadata.version("scikit-learn"),
            },
        )
        models = {
            arm: model_root / arm / "manifest.json" for arm in ARM_ORDER
        }
        return model_root / "bundle_manifest.json", analysis, models

    def _complete(
        self,
        root: Path,
        *,
        fixture: str = "fixture",
    ) -> tuple[dict, Path, Path, dict[str, Path]]:
        system, analysis, models = self._artifact_files(root, fixture=fixture)
        preregistration = build_teachobs_lockbox_preregistration(
            study_id="new-site-lockbox-2026",
            system_artifact_path=system,
            analysis_code_path=analysis,
            arm_model_artifact_paths=models,
            created_at_utc="2026-07-01T00:00:00Z",
        )
        return preregistration, system, analysis, models

    def test_external_checker_rehashes_exact_draft_bundle_and_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preregistration, system_a, analysis_a, _ = self._complete(
                root / "a",
                fixture="bundle-a",
            )
            system_b, analysis_b, _ = self._artifact_files(
                root / "b",
                fixture="bundle-b",
            )
            draft = root / "draft.json"
            draft.write_text(
                json.dumps(preregistration, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            report = check_lockbox(
                str(draft),
                "new-site-lockbox-2026",
                evaluation_profile="full_23_train_7_test",
                frozen_model_output=str(system_a.parent),
                analysis_code_path=str(analysis_a),
            )
            self.assertTrue(report["artifact_file_bindings_verified"])
            self.assertTrue(report["transitive_companion_arrays_verified"])

            with self.assertRaisesRegex(
                StageCheckError,
                "lockbox frozen artifact binding failed",
            ):
                check_lockbox(
                    str(draft),
                    "new-site-lockbox-2026",
                    evaluation_profile="full_23_train_7_test",
                    frozen_model_output=str(system_b.parent),
                    analysis_code_path=str(analysis_b),
                )

    def test_pending_draft_is_valid_but_every_claim_remains_false(self) -> None:
        draft = build_teachobs_lockbox_preregistration(
            study_id="new-site-lockbox-draft",
            created_at_utc="2026-07-01T00:00:00Z",
        )
        report = validate_teachobs_lockbox_preregistration(draft)
        self.assertTrue(report["valid"])
        self.assertFalse(report["frozen_artifact_set_complete"])
        self.assertFalse(report["preregistration_execution_ready"])
        self.assertFalse(report["confirmatory_multimodal_gain_established"])
        self.assertFalse(report["deployment_accuracy_established"])
        self.assertFalse(report["learner_effectiveness_established"])
        threshold_policy = draft["analysis_plan"]["decision_threshold_policy"]
        self.assertFalse(threshold_policy["binding_complete"])
        self.assertEqual(threshold_policy["label_count"], 39)
        self.assertTrue(
            all(
                value is None
                for value in threshold_policy[
                    "arm_model_thresholds_sha256"
                ].values()
            )
        )
        self.assertFalse(
            draft["scope"]["development_split_eligible_as_confirmatory_lockbox"]
        )
        self.assertTrue(
            draft["target_lockbox_protocol"][
                "public_teachobs_23_7_explicitly_excluded"
            ]
        )

    @unittest.skipUnless(JSONSCHEMA_AVAILABLE, "jsonschema is a dev dependency")
    def test_generated_draft_matches_bundled_json_schema(self) -> None:
        draft = build_teachobs_lockbox_preregistration(
            study_id="new-site-lockbox-schema",
            created_at_utc="2026-07-01T00:00:00Z",
        )
        schema = read_json(
            project_root() / "schema" / "teachobs_lockbox_preregistration.schema.json"
        )
        jsonschema.Draft202012Validator(schema).validate(draft)

    def test_complete_preregistration_rehashes_every_frozen_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preregistration, system, analysis, models = self._complete(root)
            report = validate_teachobs_lockbox_preregistration(preregistration)
            self.assertTrue(report["frozen_artifact_set_complete"])
            bundle = read_json(system)
            training_provenance = bundle["training_provenance"]
            expected_training_provenance_sha256 = hashlib.sha256(
                json.dumps(
                    training_provenance,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            transitive = preregistration["frozen_artifacts"][
                "transitive_model_binding"
            ]
            self.assertEqual(
                transitive["training_provenance_sha256"],
                expected_training_provenance_sha256,
            )
            self.assertEqual(
                training_provenance["transcript_materialization"]["profile_id"],
                training_provenance["benchmark_profile"],
            )
            self.assertFalse(
                training_provenance["transcript_materialization"][
                    "released_transcript_fallback_used"
                ]
            )
            self.assertFalse(
                training_provenance["transcript_materialization"][
                    "labels_read_or_used"
                ]
            )
            self.assertFalse(
                preregistration["claim_status"][
                    "confirmatory_multimodal_gain_established"
                ]
            )
            self.assertFalse(
                preregistration["claim_status"]["deployment_accuracy_established"]
            )
            self.assertFalse(
                preregistration["claim_status"][
                    "learner_effectiveness_established"
                ]
            )
            threshold_policy = preregistration["analysis_plan"][
                "decision_threshold_policy"
            ]
            self.assertTrue(threshold_policy["binding_complete"])
            self.assertEqual(threshold_policy["label_count"], 39)
            self.assertFalse(threshold_policy["threshold_values_included"])
            self.assertEqual(
                threshold_policy["frozen_artifact_set_fingerprint"],
                transitive["artifact_set_fingerprint"],
            )
            self.assertEqual(
                threshold_policy["arm_model_thresholds_sha256"],
                {
                    arm: transitive["arms"][arm][
                        "model_thresholds_sha256"
                    ]
                    for arm in ARM_ORDER
                },
            )
            self.assertEqual(
                len(set(threshold_policy["arm_model_thresholds_sha256"].values())),
                len(ARM_ORDER),
            )
            verification = verify_teachobs_lockbox_artifact_files(
                preregistration,
                system_artifact_path=system,
                analysis_code_path=analysis,
                arm_model_artifact_paths=models,
            )
            self.assertTrue(verification["verified"])
            self.assertTrue(
                verification["transitive_model_and_companion_arrays_verified"]
            )
            self.assertEqual(set(verification["arm_model_sha256"]), set(ARM_ORDER))
            self.assertEqual(
                set(verification["arm_numeric_state_sha256"]), set(ARM_ORDER)
            )
            self.assertEqual(
                verification["arm_model_thresholds_sha256"],
                threshold_policy["arm_model_thresholds_sha256"],
            )
            self.assertEqual(verification["decision_threshold_count_per_arm"], 39)
            self.assertFalse(verification["paths_included"])

            full_arrays = models["full"].with_name("arrays.npz")
            full_arrays_bytes = full_arrays.read_bytes()
            full_arrays.unlink()
            with self.assertRaisesRegex(
                TeachObsLockboxError, "companion verification failed"
            ):
                verify_teachobs_lockbox_artifact_files(
                    preregistration,
                    system_artifact_path=system,
                    analysis_code_path=analysis,
                    arm_model_artifact_paths=models,
                )
            full_arrays.write_bytes(full_arrays_bytes)

            models["full"].write_bytes(b"changed after freeze\n")
            with self.assertRaisesRegex(
                TeachObsLockboxError, "full model artifact hash/size mismatch"
            ):
                verify_teachobs_lockbox_artifact_files(
                    preregistration,
                    system_artifact_path=system,
                    analysis_code_path=analysis,
                    arm_model_artifact_paths=models,
                )

    def test_plain_files_cannot_be_declared_a_complete_frozen_model_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            system = root / "bundle_manifest.json"
            analysis = root / "analysis.py"
            system.write_text("not a frozen bundle", encoding="utf-8")
            analysis.write_text("# analysis", encoding="utf-8")
            models = {}
            for arm in ARM_ORDER:
                arm_root = root / arm
                arm_root.mkdir()
                models[arm] = arm_root / "manifest.json"
                models[arm].write_text("not a model", encoding="utf-8")
            with self.assertRaisesRegex(
                TeachObsLockboxError, "invalid transitive frozen-model artifact set"
            ):
                build_teachobs_lockbox_preregistration(
                    study_id="new-site-lockbox-plain-files",
                    system_artifact_path=system,
                    analysis_code_path=analysis,
                    arm_model_artifact_paths=models,
                    created_at_utc="2026-07-01T00:00:00Z",
                )

    def test_cross_arm_manifest_cannot_masquerade_as_four_frozen_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            system, analysis, models = self._artifact_files(root)
            repeated = {arm: models["full"] for arm in ARM_ORDER}
            with self.assertRaisesRegex(
                TeachObsLockboxError, "not the arm declared by this bundle"
            ):
                build_teachobs_lockbox_preregistration(
                    study_id="new-site-lockbox-duplicate-models",
                    system_artifact_path=system,
                    analysis_code_path=analysis,
                    arm_model_artifact_paths=repeated,
                    created_at_utc="2026-07-01T00:00:00Z",
                )

    def test_legacy_or_tampered_transcript_provenance_cannot_enter_lockbox(
        self,
    ) -> None:
        def rewrite_bundle_fingerprint(bundle: dict) -> None:
            payload = {
                key: value
                for key, value in bundle.items()
                if key != "bundle_fingerprint"
            }
            bundle["bundle_fingerprint"] = hashlib.sha256(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()

        def remove_materialization(training: dict) -> None:
            training.pop("transcript_materialization")

        def tamper_benchmark_input(training: dict) -> None:
            training["benchmark_input_fingerprint"] = "f" * 64

        def tamper_materialized_text_order(training: dict) -> None:
            training["transcript_materialization"][
                "ordered_scene_text_sha256"
            ] = "f" * 64

        mutations = {
            "legacy_without_materialization": remove_materialization,
            "benchmark_input_fingerprint": tamper_benchmark_input,
            "materialized_text_order": tamper_materialized_text_order,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    system, analysis, models = self._artifact_files(
                        root / name,
                        fixture=name,
                    )
                    bundle = read_json(system)
                    mutate(bundle["training_provenance"])
                    rewrite_bundle_fingerprint(bundle)
                    system.write_text(
                        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        TeachObsLockboxError,
                        "invalid transitive frozen-model artifact set",
                    ):
                        build_teachobs_lockbox_preregistration(
                            study_id=f"new-site-lockbox-{name}",
                            system_artifact_path=system,
                            analysis_code_path=analysis,
                            arm_model_artifact_paths=models,
                            created_at_utc="2026-07-01T00:00:00Z",
                        )

    def test_protocol_tampering_and_outcome_injection_fail_closed(self) -> None:
        draft = build_teachobs_lockbox_preregistration(
            study_id="new-site-lockbox-tamper",
            created_at_utc="2026-07-01T00:00:00Z",
        )
        mutations = []
        public_relabel = copy.deepcopy(draft)
        public_relabel["scope"][
            "development_split_eligible_as_confirmatory_lockbox"
        ] = True
        mutations.append((public_relabel, "exclude public TeachObs"))
        unpaired = copy.deepcopy(draft)
        unpaired["analysis_plan"][
            "same_eligible_samples_scored_by_every_arm"
        ] = False
        mutations.append((unpaired, "same eligible samples"))
        weak_bootstrap = copy.deepcopy(draft)
        weak_bootstrap["analysis_plan"]["paired_cluster_bootstrap"][
            "replicates"
        ] = 1999
        mutations.append((weak_bootstrap, "2,000-replicate"))
        outcome = copy.deepcopy(draft)
        outcome["execution_evidence"]["results_computed"] = True
        mutations.append((outcome, "must not contain target outcomes"))
        claimed = copy.deepcopy(draft)
        claimed["claim_status"]["confirmatory_multimodal_gain_established"] = True
        mutations.append((claimed, "requires confirmatory_multimodal_gain"))
        for value, message in mutations:
            with self.subTest(message=message), self.assertRaises(
                TeachObsLockboxError
            ):
                validate_teachobs_lockbox_preregistration(value)

    def test_generic_external_evidence_must_bind_exact_four_arm_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preregistration, _, _, _ = self._complete(root)
            artifacts = preregistration["frozen_artifacts"]
            draft = {
                "schema_version": "1.0",
                "protocol": "externally_governed_research_evidence_v1",
                "evidence_id": "new-site-four-arm-evidence",
                "evidence_kind": "confirmatory_multimodal_gain",
                "study_id": "new-site-lockbox-2026",
                "registered_at_utc": "2026-07-02T00:00:00Z",
                "completed_at_utc": "2026-07-20T00:00:00Z",
                "preregistration": {
                    "registry_name": "test-only external registry",
                    "registration_id": "new-site-lockbox-registration",
                    "protocol_sha256": preregistration[
                        "preregistration_fingerprint"
                    ],
                    "statistical_analysis_plan_sha256": (
                        teachobs_analysis_plan_fingerprint(preregistration)
                    ),
                    "frozen_before_outcome_access": True,
                },
                "governance": {
                    "independent_external_governance": True,
                    "data_or_analysis_independent_of_system_developers": True,
                    "data_not_used_to_modify_system_or_claim_thresholds": True,
                    "conflicts_of_interest_disclosed": True,
                    "complete_registered_outcome_reporting": True,
                    "trusted_key_provisioned_out_of_band": True,
                },
                "population": {
                    "real_world_data": True,
                    "data_context": "real_classroom",
                    "site_count": 1,
                    "independent_site_count": 1,
                    "participant_count": 12,
                    "eligible_unit_count": 100,
                    "evaluated_unit_count": 90,
                    "excluded_unit_count": 10,
                    "exclusion_reason_counts": {"registered_quality_rule": 10},
                    "coverage_fraction": 0.9,
                    "coverage_evidence_sha256": self._digest("coverage"),
                    "identity_disjoint_from_development": True,
                },
                "artifact_bindings": {
                    "dataset_or_study_data_fingerprint": self._digest("dataset"),
                    "evaluated_system_fingerprint": artifacts["system_artifact"][
                        "sha256"
                    ],
                    "comparator_fingerprint": artifacts["arms"][
                        "transcript_only"
                    ]["model_artifact"]["sha256"],
                    "analysis_code_sha256": artifacts["analysis_code_artifact"][
                        "sha256"
                    ],
                    "statistical_output_sha256": self._digest("statistics"),
                    "aggregate_result_table_sha256": self._digest("table"),
                },
                "claim": {
                    "claim_type": "confirmatory_multimodal_gain",
                    "evaluation_design": "paired_same_samples_external_lockbox",
                    "baseline_modalities": ["transcript"],
                    "added_modalities": ["audio", "visual"],
                    "primary_metric": "macro_f1",
                    "baseline_estimate": 0.6,
                    "multimodal_estimate": 0.66,
                    "absolute_gain": 0.06,
                    "confidence_level": 0.95,
                    "confidence_interval_lower": 0.03,
                    "confidence_interval_upper": 0.09,
                    "p_value": 0.01,
                    "alpha": 0.05,
                    "minimum_confirmatory_gain": 0.02,
                    "paired_cluster_aware_inference": True,
                    "claim_cluster_field": "classroom_id",
                    "claim_cluster_count": 12,
                    "all_registered_primary_analyses_reported": True,
                    "multimodal_gain_established": True,
                },
            }
            evidence = finalize_external_research_evidence(draft)
            report = validate_teachobs_confirmatory_evidence_handoff(
                preregistration, evidence
            )
            self.assertTrue(report["handoff_verified"])
            self.assertTrue(report["unsigned_external_evidence_claim_gates_passed"])
            self.assertFalse(report["external_evidence_signature_verified"])
            self.assertFalse(report["confirmatory_multimodal_gain_established"])

            mismatched = copy.deepcopy(draft)
            mismatched["preregistration"]["protocol_sha256"] = self._digest(
                "another preregistration"
            )
            mismatched_evidence = finalize_external_research_evidence(mismatched)
            with self.assertRaisesRegex(
                TeachObsLockboxError, "does not bind this preregistration"
            ):
                validate_teachobs_confirmatory_evidence_handoff(
                    preregistration, mismatched_evidence
                )

    def test_incomplete_draft_cannot_be_signed(self) -> None:
        draft = build_teachobs_lockbox_preregistration(
            study_id="new-site-lockbox-unsigned",
            created_at_utc="2026-07-01T00:00:00Z",
        )
        with self.assertRaisesRegex(
            TeachObsLockboxError, "requires system, analysis code, and all four models"
        ):
            sign_teachobs_lockbox_preregistration(
                draft,
                private_key_pem=b"not-needed-before-artifact-gate",
                issuer="test-only",
                key_id="test-key-2026",
            )

    def test_script_writes_an_explicitly_pending_draft(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preregistration.json"
            with redirect_stdout(io.StringIO()) as stdout:
                code = prepare_main(
                    [
                        "--study-id",
                        "new-site-lockbox-script",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(code, 0)
            summary = json.loads(stdout.getvalue())
            self.assertFalse(summary["frozen_artifact_set_complete"])
            self.assertFalse(summary["confirmatory_multimodal_gain_established"])
            self.assertFalse(summary["deployment_accuracy_established"])
            draft = read_json(output)
            self.assertFalse(
                draft["claim_status"]["preregistration_execution_ready"]
            )

    def test_installed_cli_writes_the_same_fail_closed_draft(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cli-preregistration.json"
            with redirect_stdout(io.StringIO()) as stdout:
                code = cli_main(
                    [
                        "prepare-teachobs-lockbox-preregistration",
                        "--study-id",
                        "new-site-lockbox-installed-cli",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(code, 0)
            summary = json.loads(stdout.getvalue())
            self.assertFalse(summary["frozen_artifact_set_complete"])
            self.assertFalse(summary["preregistration_execution_ready"])
            self.assertFalse(summary["confirmatory_multimodal_gain_established"])
            draft = read_json(output)
            self.assertFalse(draft["claim_status"]["external_lockbox_established"])


@unittest.skipUnless(CRYPTOGRAPHY_AVAILABLE, "cryptography is optional")
class TeachObsLockboxSignatureTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, dict[str, Path]]:
        helper = TeachObsLockboxPreregistrationTests()
        return helper._artifact_files(root)

    def test_external_attestation_verifies_but_does_not_establish_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            system, analysis, models = self._fixture(root)
            preregistration = build_teachobs_lockbox_preregistration(
                study_id="new-site-lockbox-signed",
                system_artifact_path=system,
                analysis_code_path=analysis,
                arm_model_artifact_paths=models,
                created_at_utc="2026-07-01T00:00:00Z",
            )
            private = Ed25519PrivateKey.generate()
            private_pem = private.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
            public_pem = private.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            attestation = sign_teachobs_lockbox_preregistration(
                preregistration,
                private_key_pem=private_pem,
                issuer="test-only-external-governance",
                key_id="test-key-2026",
            )
            verified = verify_teachobs_lockbox_preregistration_attestation(
                attestation,
                trusted_public_key_pem=public_pem,
                expected_preregistration=preregistration,
            )
            self.assertTrue(verified["signature_verified"])
            self.assertTrue(
                verified["external_governance_registration_gate_verified"]
            )
            self.assertFalse(verified["signature_alone_proves_signer_independence"])
            self.assertFalse(verified["target_execution_evidence_complete"])
            self.assertFalse(
                verified["confirmatory_multimodal_gain_established"]
            )
            self.assertFalse(verified["deployment_accuracy_established"])
            self.assertFalse(verified["learner_effectiveness_established"])

            tampered = copy.deepcopy(attestation)
            tampered["statement"]["preregistration"]["analysis_plan"][
                "decision_threshold_policy"
            ]["arm_model_thresholds_sha256"]["full"] = "f" * 64
            with self.assertRaises(TeachObsLockboxError):
                verify_teachobs_lockbox_preregistration_attestation(
                    tampered,
                    trusted_public_key_pem=public_pem,
                )


if __name__ == "__main__":
    unittest.main()
