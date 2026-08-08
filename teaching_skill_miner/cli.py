from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import io
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__, teachobs_asr_handoff
from .attestation import (
    build_freeze_registration_request,
    sign_evaluation_report_receipt,
    sign_freeze_registration,
)
from .audit import audit_dataset
from .benchmark import benchmark_transfer
from .delivery import delivery_markdown, verify_delivery
from .dashboard import dashboard_self_check, open_dashboard
from .deepseek_client import ALLOWED_MODELS, DeepSeekClient, DeepSeekConfig
from .private_dashboard import (
    DEFAULT_PRIVATE_LESSON,
    DEFAULT_PRIVATE_SKILL,
    DEFAULT_PRIVATE_SKILL_ROOT,
    DEFAULT_PRIVATE_TEACHOBS_ROOT,
    PrivateDashboardConfig,
    build_private_snapshot,
    private_dashboard_template_self_check,
    private_snapshot_summary,
    serve_private_dashboard,
)
from .evaluator import evaluate_collection, evaluate_skill
from .executor import execute_skill
from .general_skill import (
    build_general_skill_receipt,
    distill_general_skill,
    evaluate_general_skill,
    execute_general_skill,
    render_general_skill_summary,
    validate_general_skill,
)
from .external_evidence import (
    EVIDENCE_KINDS,
    finalize_external_research_evidence,
    sign_external_research_evidence,
    validate_external_research_evidence,
    verify_external_research_evidence_files,
)
from .formal_captions import import_mit_ocw_formal_captions
from .full_video_dataset import (
    build_public_full_video_receipt,
    download_full_video_dataset,
)
from .io_utils import (
    ensure_private_directory,
    read_json,
    resolve_manifest_root,
    resolve_resource_path,
    write_json,
    write_text,
)
from .human_eval import skill_review_fingerprint, summarize_human_review
from .llm_backend import refine_skill_with_api
from .miner import mine_skill
from .models import validate_skill, validate_transcript
from .multimodal import VIDEO_EXTENSIONS, analyze_video, enrich_transcript_with_multimodal
from .longform_multimodal import process_longform_dataset
from .learner_effect_study import (
    analyze_learner_effect_study,
    generate_learner_effect_study_package,
    write_learner_effect_analysis,
)
from .multimodal_ablation import ARM_ORDER, evaluate_multimodal_ablation
from .multimodal_benchmark import evaluate_multimodal_fixture
from .preprocess import preprocess_file
from .project_health import doctor_report
from .release_audit import audit_release_path, export_public_dipser_report
from .teachobs import (
    build_public_teachobs_receipt,
    download_and_audit_teachobs,
)
from .teachobs_benchmark import (
    build_public_teachobs_benchmark_receipt,
    run_teachobs_text_benchmark,
)
from .teachobs_captions import (
    build_public_teachobs_caption_receipt,
    retrieve_and_audit_teachobs_captions,
)
from .teachobs_asr_handoff import (
    build_pending_teachobs_asr_receipt,
    build_teachobs_asr_job_manifest,
    import_teachobs_asr_results,
    model_directory_sha256,
    run_teachobs_asr_gpu_jobs,
    write_teachobs_asr_import,
)
from .teachobs_human_annotation import (
    analyze_teachobs_double_annotations,
    build_completed_public_teachobs_annotation_receipt,
    build_pending_public_teachobs_annotation_receipt,
    prepare_teachobs_double_annotation,
)
from .teachobs_lockbox import (
    ARM_ORDER as TEACHOBS_LOCKBOX_ARM_ORDER,
    build_teachobs_lockbox_preregistration,
    validate_teachobs_lockbox_preregistration,
)
from .teachobs_media import prepare_teachobs_media_dataset
from .teachobs_multimodal_benchmark import (
    FULL_23_TRAIN_7_TEST_PROFILE,
    PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
    build_public_teachobs_multimodal_receipt,
    run_teachobs_multimodal_benchmark,
)
from .teachobs_transcript_materialization import (
    materialize_teachobs_transcripts,
)
from .visual_semantics import execute_visual_semantic_extraction
from .visual_semantics_apply import execute_visual_semantic_apply
from .visual_semantics_dataset import execute_visual_semantic_dataset
from .recognition import (
    evaluate_frozen_external_deployment,
    feature_bundle_fingerprint as dipser_feature_bundle_fingerprint,
    fit_frozen_deployment_model,
    frozen_model_to_artifact,
    frozen_evaluation_report_fingerprint,
    discover_ouc_cge,
    infer_classroom_engagement,
    load_frozen_deployment_model,
    run_dipser_credible_experiment,
    run_real_classroom_benchmark,
    strict_dataset_fingerprint,
    strict_feature_bundle_fingerprint,
    validate_evaluation_coverage,
)
from .recognition.raw_feature_bridge import (
    extract_strict_feature_bundle,
    validate_bundle_against_frozen_model,
    validate_extractor_suite_binding,
    validate_raw_source_binding,
)
from .recognition.strict_evaluation import (
    validate_class_schema,
    validate_feature_provenance,
    validate_strict_manifest,
)
from .runtime import SkillRuntime, run_scripted_session
from .teacher_agent import (
    advance_teacher_agent_session,
    evaluate_teacher_agent,
    session_turn_summary,
    start_teacher_agent_session,
)
from .teacher_agent_dashboard import (
    serve_teacher_agent_dashboard,
    teacher_agent_dashboard_self_check,
)
from .teacher_agent_live import LiveAgentOptions
from .teacher_agent_benchmark import (
    benchmark_exit_code,
    run_teacher_agent_benchmark,
)
from .teacher_agent_benchmark_v2 import (
    predictions_from_live_cases,
    score_benchmark_v2,
    validate_benchmark_gold,
    validate_benchmark_inputs,
)
from .teacher_agent_outcomes import evaluate_learning_observation


def _mine(transcript: dict[str, Any], backend: str) -> dict[str, Any]:
    baseline = mine_skill(transcript)
    skill = refine_skill_with_api(transcript, baseline) if backend == "api" else baseline
    skill.setdefault("mining_metadata", {})["backend"] = backend
    return skill


def command_preprocess(args: argparse.Namespace) -> int:
    transcript = preprocess_file(
        args.input,
        video_id=args.video_id,
        course_id=args.course_id,
        title=args.title,
        source_url=args.source_url,
        language=args.language,
    )
    target = write_json(args.output, transcript)
    print(f"transcript: {target}")
    print(f"segments: {len(transcript['segments'])}")
    return 0


def command_mine(args: argparse.Namespace) -> int:
    transcript = read_json(args.transcript)
    skill = _mine(transcript, args.backend)
    target = write_json(args.output, skill)
    print(f"skill: {target}")
    print(f"strategies: {', '.join(item['id'] for item in skill['strategies'])}")
    return 0


def command_distill_general_skill(args: argparse.Namespace) -> int:
    skill_root = Path(args.skill_root).expanduser().resolve()
    if not skill_root.is_dir():
        raise FileNotFoundError(f"Skill directory does not exist: {skill_root}")
    pattern = str(args.pattern)
    pattern_path = Path(pattern)
    if pattern_path.is_absolute() or ".." in pattern_path.parts:
        raise ValueError("--pattern must stay inside --skill-root")
    skill_paths = sorted(
        path for path in skill_root.glob(pattern) if path.is_file()
    )
    if not skill_paths:
        raise FileNotFoundError(
            f"no Skill files matched {pattern!r} beneath {skill_root}"
        )

    source_skills = [read_json(path) for path in skill_paths]
    general_skill = distill_general_skill(
        source_skills,
        overall_support_threshold=args.overall_support_threshold,
        per_course_support_threshold=args.per_course_support_threshold,
        minimum_course_count=args.minimum_course_count,
        minimum_skills_per_course=args.minimum_skills_per_course,
    )
    validation = validate_general_skill(general_skill)
    if not validation.valid:
        raise ValueError(
            "invalid distilled general Skill: " + "; ".join(validation.errors)
        )
    evaluation = evaluate_general_skill(general_skill)
    teaching_process = execute_general_skill(
        general_skill,
        concept=args.example_concept,
        learner_level=args.learner_level,
    )

    output_dir = ensure_private_directory(args.output_dir)
    skill_path = write_json(output_dir / "general_skill.json", general_skill)
    evaluation_path = write_json(
        output_dir / "general_skill_evaluation.json", evaluation
    )
    process_path = write_text(
        output_dir / "example_teaching_process.md", teaching_process
    )
    summary_path = write_text(
        output_dir / "general_skill_summary.md",
        render_general_skill_summary(general_skill),
    )
    distillation = general_skill.get("distillation", {})
    receipt = build_general_skill_receipt(general_skill, evaluation)
    receipt_path = write_json(
        output_dir / "general_skill_receipt.json", receipt
    )
    summary = {
        "general_skill": str(skill_path.resolve()),
        "evaluation": str(evaluation_path.resolve()),
        "receipt": str(receipt_path.resolve()),
        "example_teaching_process": str(process_path.resolve()),
        "summary": str(summary_path.resolve()),
        "input_skill_count": distillation.get("input_skill_count"),
        "input_course_count": distillation.get("input_course_count"),
        "consensus_strategy_count": len(
            general_skill.get("skill", {}).get("strategies", [])
        ),
        "observed_consensus_phase_count": sum(
            step.get("origin") == "cross_lecture_observed_consensus"
            for step in general_skill.get("skill", {}).get("procedure", [])
        ),
        "recommended_phase_count": sum(
            step.get("origin") == "recommended_enrichment"
            for step in general_skill.get("skill", {}).get("procedure", [])
        ),
        "internal_evaluation_passed": bool(evaluation.get("passed")),
        "internal_score_is_accuracy": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if evaluation.get("passed") else 2


def command_apply_general_skill(args: argparse.Namespace) -> int:
    general_skill = read_json(args.skill)
    validation = validate_general_skill(general_skill)
    if not validation.valid:
        raise ValueError("invalid general Skill: " + "; ".join(validation.errors))
    teaching_process = execute_general_skill(
        general_skill,
        concept=args.concept,
        learner_level=args.learner_level,
    )
    if args.output:
        target = write_text(args.output, teaching_process)
        print(f"teaching process: {target}")
    else:
        print(teaching_process, end="")
    return 0


def command_evaluate_general_skill(args: argparse.Namespace) -> int:
    general_skill = read_json(args.skill)
    report = evaluate_general_skill(general_skill)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("passed") else 2


def command_teach(args: argparse.Namespace) -> int:
    skill = read_json(args.skill)
    validation = validate_skill(skill)
    if not validation.valid:
        raise ValueError("invalid skill: " + "; ".join(validation.errors))
    lesson = execute_skill(skill, concept=args.concept, learner_level=args.learner_level)
    if args.output:
        target = write_text(args.output, lesson)
        print(f"teaching process: {target}")
    else:
        print(lesson, end="")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    skill = read_json(args.skill)
    transcript = read_json(args.transcript) if args.transcript else None
    report = evaluate_skill(skill, transcript)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


def command_validate(args: argparse.Namespace) -> int:
    value = read_json(args.path)
    if args.kind == "transcript":
        result = validate_transcript(value)
    elif args.kind == "skill":
        result = validate_skill(value)
    else:
        result = validate_general_skill(value)
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return 0 if result.valid else 2


def _script_for_complete_session(skill: dict[str, Any]) -> list[dict[str, str]]:
    total = len(skill.get("procedure", [])) + len(skill.get("verification", []))
    responses = [{"response": "我还不确定，可能需要一个更简单的例子。", "signal": "not_achieved"}]
    responses.extend(
        {"response": f"第 {index + 1} 次作答：我能说明关键条件、理由和一个例子。", "signal": "achieved"}
        for index in range(total)
    )
    return responses


def command_interact(args: argparse.Namespace) -> int:
    skill = read_json(args.skill)
    validation = validate_skill(skill)
    if not validation.valid:
        raise ValueError("invalid skill: " + "; ".join(validation.errors))
    if args.script:
        script = read_json(args.script)
        responses = script.get("responses", script) if isinstance(script, dict) else script
        result = run_scripted_session(
            skill,
            concept=args.concept,
            learner_level=args.learner_level,
            responses=responses,
        )
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["completed"] else 2

    runtime = SkillRuntime(skill, concept=args.concept, learner_level=args.learner_level)
    while not runtime.completed:
        turn = runtime.current_turn()
        print(f"\n教师[{turn['teacher_action']}]：{turn['teacher_message']}")
        print(f"观察标准：{turn['expected_signal']}")
        response = input("学生：").strip()
        judgement = input("是否达到观察标准？[y/n] ").strip().lower()
        outcome = runtime.observe(response, "achieved" if judgement in {"y", "yes", "是"} else "not_achieved")
        if outcome["event"].get("fallback_message"):
            print("教师回退：" + outcome["event"]["fallback_message"])
    result = runtime.snapshot()
    if args.output:
        write_json(args.output, result)
    print("\n教学与验证完成。")
    return 0


def command_teacher_agent_start(args: argparse.Namespace) -> int:
    payload = read_json(resolve_resource_path(args.input))
    library = read_json(resolve_resource_path(args.library))
    if not isinstance(payload, dict):
        raise ValueError("teacher Agent input must be one JSON object")
    session = start_teacher_agent_session(
        payload.get("goal", {}),
        payload.get("student_profile", {}),
        library,
        policy=args.policy,
        fixed_skill_id=args.fixed_skill_id,
    )
    ensure_private_directory(Path(args.session).parent)
    target = write_json(args.session, session)
    print(json.dumps(session_turn_summary(session), ensure_ascii=False, indent=2))
    print(f"private session: {target.resolve()}")
    return 0


def command_teacher_agent_step(args: argparse.Namespace) -> int:
    session = read_json(args.session)
    if args.response_file:
        response = Path(args.response_file).read_text(encoding="utf-8").strip()
    else:
        response = args.response or ""
    updated = advance_teacher_agent_session(
        session,
        learner_response=response,
        signal=args.signal,
        misconception_tag=args.misconception_tag,
        signal_confidence=args.signal_confidence,
    )
    target = write_json(args.session, updated)
    print(json.dumps(session_turn_summary(updated), ensure_ascii=False, indent=2))
    print(f"private session: {target.resolve()}")
    return 0


def command_teacher_agent_evaluate(args: argparse.Namespace) -> int:
    library = read_json(resolve_resource_path(args.library))
    cases = read_json(resolve_resource_path(args.cases))
    report = evaluate_teacher_agent(library, cases)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


def command_teacher_agent_benchmark(args: argparse.Namespace) -> int:
    if args.online and not args.allow_remote_benchmark_data:
        raise ValueError("--online requires --allow-remote-benchmark-data")
    dataset = read_json(resolve_resource_path(args.benchmark))
    library = read_json(resolve_resource_path(args.skill_library))
    client = None
    if args.online:
        client = DeepSeekClient(
            DeepSeekConfig.from_environment(
                api_key_file=args.api_key_file,
                allow_remote_student_data=True,
                model=args.model,
            )
        )
    report = run_teacher_agent_benchmark(
        dataset,
        library,
        client=client,
        fixed_skill_id=args.fixed_skill_id,
        repeats=args.repeats,
        minimum_review_confidence=args.minimum_review_confidence,
    )
    if args.output:
        target = write_json(args.output, report)
        print(
            json.dumps(
                {
                    "output": str(target),
                    "run_status": report["run_status"],
                    "run_fingerprint": report["run_fingerprint"],
                    "content_sha256": report["content_sha256"],
                    "input_content_printed": False,
                    "api_key_printed": False,
                },
                ensure_ascii=False,
            )
        )
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return benchmark_exit_code(report)


def command_teacher_agent_benchmark_v2(args: argparse.Namespace) -> int:
    """Validate or score the product benchmark without mixing input and gold."""

    dataset = read_json(resolve_resource_path(args.benchmark))
    gold = read_json(resolve_resource_path(args.gold))
    library = read_json(resolve_resource_path(args.skill_library))
    validate_benchmark_inputs(dataset, library)
    validate_benchmark_gold(gold, dataset, library)
    if args.validate_only:
        summary = {
            "schema": "teaching_skill_miner.teacher_agent_benchmark_v2_validation.v1",
            "benchmark_id": dataset["benchmark_id"],
            "split": dataset["split"],
            "case_count": len(dataset["cases"]),
            "gold_is_separate": True,
            "input_contains_gold": False,
            "deployment_accuracy_established": False,
            "real_learning_effect_established": False,
            "passed": True,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    predictions = None
    if args.predictions:
        predictions = read_json(args.predictions)
    elif args.online:
        if not args.allow_remote_benchmark_data:
            raise ValueError("--online requires --allow-remote-benchmark-data")
        if not args.predictions_output:
            raise ValueError("--online requires --predictions-output in a private path")
        client = DeepSeekClient(
            DeepSeekConfig.from_environment(
                api_key_file=args.api_key_file,
                allow_remote_student_data=True,
                model=args.model,
            )
        )
        predictions = predictions_from_live_cases(dataset, library, client)
        ensure_private_directory(Path(args.predictions_output).parent)
        write_json(args.predictions_output, predictions)
    else:
        raise ValueError(
            "provide --predictions for offline scoring, or use --online with explicit consent"
        )
    report = score_benchmark_v2(
        dataset,
        gold,
        predictions,
        library,
        acknowledge_held_out=args.acknowledge_held_out,
    )
    if args.output:
        target = write_json(args.output, report)
        print(
            json.dumps(
                {
                    "output": str(target),
                    "benchmark_id": report["benchmark_id"],
                    "split": report["split"],
                    "run_fingerprint": report["run_fingerprint"],
                    "content_sha256": report["content_sha256"],
                    "raw_teacher_messages_printed": False,
                    "deployment_accuracy_established": False,
                    "real_learning_effect_established": False,
                },
                ensure_ascii=False,
            )
        )
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def command_teacher_agent_outcome_evaluate(args: argparse.Namespace) -> int:
    observation = read_json(resolve_resource_path(args.input))
    report = evaluate_learning_observation(observation)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def command_teacher_agent_demo(args: argparse.Namespace) -> int:
    payload = read_json(resolve_resource_path(args.input))
    library = read_json(resolve_resource_path(args.library))
    cases = read_json(resolve_resource_path(args.cases))
    if not isinstance(payload, dict):
        raise ValueError("teacher Agent demo input must be one JSON object")
    sequence = payload.get("demo_feedback_sequence")
    if not isinstance(sequence, list) or not sequence:
        raise ValueError("teacher Agent demo input requires demo_feedback_sequence")
    session = start_teacher_agent_session(
        payload.get("goal", {}), payload.get("student_profile", {}), library
    )
    timeline = [session_turn_summary(session)]
    for item in sequence:
        if session["status"] != "active":
            break
        if not isinstance(item, dict):
            raise ValueError("each demo feedback item must be an object")
        session = advance_teacher_agent_session(
            session,
            learner_response=str(item.get("response", "")),
            signal=str(item.get("signal", "")),
            misconception_tag=item.get("misconception_tag"),
            signal_confidence=float(item.get("confidence", 1.0)),
        )
        timeline.append(session_turn_summary(session))
    evaluation = evaluate_teacher_agent(library, cases)
    output = ensure_private_directory(args.output_dir)
    final_path = write_json(output / "teacher_agent_session.json", session)
    timeline_path = write_json(output / "teacher_agent_timeline.json", timeline)
    evaluation_path = write_json(output / "teacher_agent_evaluation.json", evaluation)
    summary = {
        "status": session["status"],
        "rounds_completed": session["round"],
        "skill_switch_count": session["control"]["skill_switch_count"],
        "selected_skill_ids": [
            item["action"]["primary_skill"]["skill_id"]
            for item in session["history"]
        ],
        "explicit_student_state": session["student_state"],
        "evaluation_passed": evaluation["passed"],
        "simulated_mean_gain_delta": evaluation["aggregate"][
            "simulated_mean_gain_delta"
        ],
        "real_learning_effectiveness_established": False,
        "artifacts": {
            "session": str(final_path.resolve()),
            "timeline": str(timeline_path.resolve()),
            "evaluation": str(evaluation_path.resolve()),
        },
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if session["status"] == "succeeded" and evaluation["passed"] else 2


def command_teacher_agent_dashboard(args: argparse.Namespace) -> int:
    library = resolve_resource_path(args.library)
    demo_input = resolve_resource_path(args.input)
    cases = resolve_resource_path(args.cases)
    if args.check:
        report = teacher_agent_dashboard_self_check(library, demo_input, cases)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 2
    client = None
    if args.agent_backend == "deepseek":
        config = DeepSeekConfig.from_environment(
            api_key_file=args.api_key_file,
            allow_remote_student_data=args.allow_remote_student_data,
            model=args.model,
        )
        client = DeepSeekClient(config)
    return serve_teacher_agent_dashboard(
        library,
        demo_input,
        cases,
        port=args.port,
        open_browser=not args.no_browser,
        client=client,
        live_options=LiveAgentOptions(
            fallback_to_rules=not args.no_rule_fallback,
            action_only_repair_enabled=True,
            state_first_route_adjudication_enabled=True,
            agent_loop_enabled=True,
            agent_loop_post_assessment_enabled=True,
            maximum_agent_steps=4,
        ),
        neural_v1_manifest_path=resolve_resource_path(args.neural_v1_manifest),
        learning_outcome_path=resolve_resource_path(args.learning_outcome),
        free_text_benchmark_receipt_path=resolve_resource_path(
            args.free_text_benchmark_receipt
        ),
        store_path=args.session_store,
    )


def command_audit(args: argparse.Namespace) -> int:
    manifest_path = resolve_resource_path(args.manifest)
    manifest = read_json(manifest_path)
    root = resolve_manifest_root(manifest_path, manifest)
    report = audit_dataset(manifest, root)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    passed = (
        report["formal_empirical_ready"]
        if args.require_formal
        else report["dataset_structure_passed"]
    )
    return 0 if passed else 2


def command_fetch_formal_captions(args: argparse.Namespace) -> int:
    if not args.acknowledge_source_terms:
        raise ValueError(
            "formal caption download requires --acknowledge-source-terms; "
            "official captions retain the upstream license"
        )
    result = import_mit_ocw_formal_captions(
        resolve_resource_path(args.source_manifest),
        args.output,
        ffprobe_command=args.ffprobe,
        timeout_seconds=args.timeout,
        progress=lambda message: print(f"verified: {message}", flush=True),
    )
    if args.public_receipt:
        write_json(args.public_receipt, result["receipt"])
    summary = {
        "manifest": result["manifest_path"],
        "audit": result["audit_path"],
        "private_receipt": result["receipt_path"],
        "public_receipt": (
            str(Path(args.public_receipt).resolve()) if args.public_receipt else None
        ),
        "video_count": result["audit"]["video_count"],
        "research_grade_transcript_count": result["audit"][
            "research_grade_transcript_count"
        ],
        "formal_empirical_ready": result["audit"]["formal_empirical_ready"],
        "caption_text_publicly_exported": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result["audit"]["formal_empirical_ready"] else 2


def command_fetch_full_videos(args: argparse.Namespace) -> int:
    if not args.acknowledge_source_terms:
        raise ValueError(
            "full video download requires --acknowledge-source-terms; "
            "MIT OCW media retains the upstream license and remains private"
        )
    result = download_full_video_dataset(
        resolve_resource_path(args.source_manifest),
        resolve_resource_path(args.formal_caption_manifest),
        args.output,
        acknowledge_source_terms=True,
        curl_command=args.curl,
        ffprobe_command=args.ffprobe,
        connect_timeout_seconds=args.connect_timeout,
        download_timeout_seconds=args.download_timeout,
        ffprobe_timeout_seconds=args.ffprobe_timeout,
        progress=lambda message: print(f"verified: {message}", flush=True),
    )
    if args.public_receipt:
        write_json(
            args.public_receipt,
            build_public_full_video_receipt(result["manifest"]),
        )
    summary = {
        "media_manifest": result["manifest_path"],
        "private_receipt": result["receipt_path"],
        "complete": result["manifest"]["complete"],
        "video_count": result["manifest"]["video_count"],
        "total_media_bytes": result["receipt"]["total_media_bytes"],
        "raw_media_publicly_exported": False,
        "publisher_media_hashes_pinned": False,
        "public_receipt": (
            str(Path(args.public_receipt).resolve())
            if args.public_receipt
            else None
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["complete"] else 2


def _load_skills(directory: str | Path) -> list[dict[str, Any]]:
    paths = sorted(Path(directory).glob("*.skill.json"))
    if not paths:
        paths = sorted(Path(directory).glob("*.json"))
    if not paths:
        raise ValueError(f"no skill JSON files found in {directory}")
    return [read_json(path) for path in paths]


def command_benchmark(args: argparse.Namespace) -> int:
    skills = _load_skills(args.skills)
    cases_payload = read_json(resolve_resource_path(args.cases))
    cases = cases_payload.get("cases", cases_payload) if isinstance(cases_payload, dict) else cases_payload
    report = benchmark_transfer(skills, cases)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


def command_pipeline(args: argparse.Namespace) -> int:
    output = ensure_private_directory(args.output)
    input_path = Path(args.input)
    is_video_input = input_path.suffix.lower() in VIDEO_EXTENSIONS
    if input_path.suffix.lower() == ".json":
        if args.transcript:
            raise ValueError("--transcript is only valid when the primary input is a video")
        transcript = read_json(input_path)
        validation = validate_transcript(transcript)
        if not validation.valid:
            raise ValueError("invalid transcript: " + "; ".join(validation.errors))
        transcript_source_mode = "input_transcript_json"
    else:
        provided_transcript_path = Path(args.transcript) if args.transcript else None
        if provided_transcript_path is not None and not is_video_input:
            raise ValueError("--transcript requires a video as the primary input")
        if provided_transcript_path is not None and provided_transcript_path.suffix.lower() == ".json":
            transcript = read_json(provided_transcript_path)
            validation = validate_transcript(transcript)
            if not validation.valid:
                raise ValueError(
                    "invalid provided transcript: " + "; ".join(validation.errors)
                )
            for field in ("video_id", "course_id", "title", "source_url"):
                supplied = getattr(args, field)
                if supplied and str(transcript.get(field)) != str(supplied):
                    raise ValueError(
                        f"provided transcript {field} does not match --{field.replace('_', '-')}"
                    )
            transcript_source_mode = "provided_transcript_json"
        else:
            missing = [
                name
                for name in ("video_id", "course_id", "title", "source_url")
                if not getattr(args, name)
            ]
            if missing:
                raise ValueError(
                    "non-JSON transcript input requires: "
                    + ", ".join("--" + name.replace("_", "-") for name in missing)
                )
            transcript_input = provided_transcript_path or input_path
            transcript = preprocess_file(
                transcript_input,
                video_id=args.video_id,
                course_id=args.course_id,
                title=args.title,
                source_url=args.source_url,
                language=args.language,
            )
            if provided_transcript_path is not None:
                transcript_source_mode = "provided_caption_or_text"
            elif is_video_input:
                transcript_source_mode = "media_asr"
            else:
                transcript_source_mode = "caption_or_text_input"
        if is_video_input and not args.no_multimodal:
            observations = None
            if args.observations:
                observation_payload = read_json(args.observations)
                observations = observation_payload.get("observations", observation_payload) if isinstance(observation_payload, dict) else observation_payload
            analysis = analyze_video(
                input_path,
                transcript,
                output / "multimodal",
                observations=observations,
                use_ocr=not args.no_ocr,
                frame_interval_seconds=args.frame_interval,
                maximum_frames=args.max_frames,
            )
            transcript = enrich_transcript_with_multimodal(transcript, analysis)
    skill = _mine(transcript, args.backend)
    evaluation = evaluate_skill(skill, transcript)
    lesson = execute_skill(skill, concept=args.concept, learner_level=args.learner_level)
    session = run_scripted_session(
        skill,
        concept=args.concept,
        learner_level=args.learner_level,
        responses=_script_for_complete_session(skill),
    )
    write_json(output / "transcript.json", transcript)
    write_json(output / "skill.json", skill)
    write_json(output / "evaluation.json", evaluation)
    write_text(output / "teaching_process.md", lesson)
    write_json(output / "interactive_session.json", session)
    multimodal = transcript.get("multimodal", {})
    language_evidence = multimodal.get("language", {}) if isinstance(multimodal, dict) else {}
    summary = {
        "video_id": transcript["video_id"],
        "skill_id": skill["skill_id"],
        "transcript_source_mode": transcript_source_mode,
        "multimodal_analysis_performed": bool(multimodal),
        "language_evidence_status": language_evidence.get(
            "status", "transcript_only"
        ),
        "audio_content_verified": bool(
            language_evidence.get("audio_content_verified", False)
        ),
        "evaluation_score": evaluation["overall_score"],
        "evaluation_passed": evaluation["passed"],
        "interactive_session_completed": session["completed"],
        "interactive_session_mode": session["session_mode"],
        "interactive_completion_semantics": session["completion_semantics"],
        "learning_effectiveness_established": session[
            "learning_effectiveness_established"
        ],
        "fallback_count": session["fallback_count"],
        "artifacts": ["transcript.json", "skill.json", "evaluation.json", "teaching_process.md", "interactive_session.json"],
    }
    write_json(output / "pipeline_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if evaluation["passed"] and session["completed"] else 2


def command_multimodal(args: argparse.Namespace) -> int:
    transcript = read_json(args.transcript)
    validation = validate_transcript(transcript)
    if not validation.valid:
        raise ValueError("invalid transcript: " + "; ".join(validation.errors))
    observations = None
    if args.observations:
        payload = read_json(args.observations)
        observations = payload.get("observations", payload) if isinstance(payload, dict) else payload
    analysis = analyze_video(
        args.video,
        transcript,
        args.artifacts_dir,
        observations=observations,
        use_ocr=not args.no_ocr,
        frame_interval_seconds=args.frame_interval,
        maximum_frames=args.max_frames,
    )
    enriched = enrich_transcript_with_multimodal(transcript, analysis)
    write_json(args.output, enriched)
    summary = {
        "output": str(Path(args.output).resolve()),
        "modalities_available": analysis["modalities_available"],
        "silence_count": len(analysis["audio"]["silences"]),
        "keyframe_count": len(analysis["visual"]["keyframes"]),
        "visual_event_count": len(analysis["visual"]["events"]),
        "fused_event_count": len(analysis["events"]),
        "event_types": sorted({event["type"] for event in analysis["events"]}),
    }
    write_json(Path(args.artifacts_dir) / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_fetch_teachobs(args: argparse.Namespace) -> int:
    audit = download_and_audit_teachobs(
        args.output,
        acknowledge_source_terms=args.acknowledge_source_terms,
        timeout=args.timeout,
    )
    audit_path = Path(args.output) / "dataset_audit.json"
    if args.public_receipt:
        receipt = build_public_teachobs_receipt(
            audit,
            private_audit_path=audit_path,
        )
        write_json(args.public_receipt, receipt)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "dataset_id": audit["dataset_id"],
                "lesson_count": audit["lesson_count"],
                "scene_count": audit["scene_count"],
                "code_count": audit["code_count"],
                "independent_coder_count_from_source_protocol": audit[
                    "annotation_protocol_from_source"
                ]["independent_coder_count"],
                "system_evaluated_on_labels": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_teachobs_text_benchmark(args: argparse.Namespace) -> int:
    result = run_teachobs_text_benchmark(
        args.repository,
        compare_sanitized=not args.no_sanitized_sensitivity,
    )
    output = write_json(args.output, result)
    if args.public_receipt:
        receipt = build_public_teachobs_benchmark_receipt(result)
        write_json(args.public_receipt, receipt)
    summary = {
        "output": str(output.resolve()),
        "train_lessons": result["dataset_audit"]["train_lesson_count"],
        "test_lessons": result["dataset_audit"]["test_lesson_count"],
        "test_scenes": result["dataset_audit"]["test_scene_count"],
        "label_count": result["dataset_audit"]["code_count"],
        "primary_metrics": result["runs"]["raw"]["metrics"],
        "valid_claim": result["evidence_scope"]["valid_claim"],
        "external_lockbox_established": False,
        "deployment_accuracy_established": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_audit_teachobs_captions(args: argparse.Namespace) -> int:
    yt_dlp_command: list[str] | str | None
    if args.yt_dlp_python:
        yt_dlp_command = [args.yt_dlp_python, "-m", "yt_dlp"]
    else:
        yt_dlp_command = args.yt_dlp_executable
    result = retrieve_and_audit_teachobs_captions(
        args.repository,
        args.media_plan,
        args.output,
        acknowledge_source_terms=args.acknowledge_source_terms,
        acquisition_receipt_path=args.acquisition_receipt,
        lesson_ids=args.lesson_id,
        yt_dlp_command=yt_dlp_command,
        js_runtime=args.js_runtime,
        yt_dlp_direct=args.yt_dlp_direct,
        yt_dlp_impersonate=args.yt_dlp_impersonate,
        yt_dlp_youtube_client=args.yt_dlp_youtube_client,
        cookies_from_browser=args.cookies_from_browser,
        timeout_seconds=args.timeout,
        max_workers=args.jobs,
    )
    audit = result["audit"]
    if args.public_receipt:
        receipt = build_public_teachobs_caption_receipt(
            audit,
            private_audit_path=result["private_audit_path"],
        )
        write_json(args.public_receipt, receipt)
    aggregate = audit["aggregate"]
    summary = {
        "private_output": str(Path(args.output).resolve()),
        "lesson_count": aggregate["lesson_count"],
        "caption_timeline_audited_lesson_count": aggregate[
            "caption_timeline_audited_lesson_count"
        ],
        "caption_unavailable_or_failed_lesson_count": aggregate[
            "caption_unavailable_or_failed_lesson_count"
        ],
        "manual_creator_provided_track_count": aggregate[
            "manual_creator_provided_track_count"
        ],
        "youtube_automatic_caption_track_count": aggregate[
            "youtube_automatic_caption_track_count"
        ],
        "formal_caption_timeline_audit_completed": aggregate[
            "formal_caption_timeline_audit_completed"
        ],
        "caption_content_accuracy_established": False,
        "word_error_rate_established": False,
        "independent_human_transcript_audit_completed": False,
        "whisper_executed": False,
        "source_urls_or_lesson_ids_printed": False,
        "public_receipt_written": bool(args.public_receipt),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if aggregate["formal_caption_timeline_audit_completed"] else 2


def command_prepare_teachobs_asr_handoff(args: argparse.Namespace) -> int:
    if args.pending_only:
        receipt = build_pending_teachobs_asr_receipt(
            args.media_manifest,
            args.caption_audit,
        )
        write_json(args.public_receipt, receipt)
        print(
            json.dumps(
                {
                    "handoff_status": "pending",
                    "hash_bound_private_media_lesson_count": receipt["aggregate"][
                        "hash_bound_private_media_lesson_count"
                    ],
                    "covered_by_platform_caption_lesson_count": (
                        receipt["aggregate"]["covered_lesson_count"]
                    ),
                    "asr_job_count": 0,
                    "whisper_executed": False,
                    "model_or_media_downloaded": False,
                    "asr_is_official_caption": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    required = {
        "--model-id": args.model_id,
        "--model-revision": args.model_revision,
        "--model-files-sha256": args.model_files_sha256,
        "--faster-whisper-version": args.faster_whisper_version,
        "--ctranslate2-version": args.ctranslate2_version,
        "--container-image-digest": args.container_image_digest,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "non-pending ASR handoff requires exact model/runtime fields: "
            + ", ".join(missing)
        )
    manifest = build_teachobs_asr_job_manifest(
        args.media_manifest,
        args.media_root,
        args.caption_audit,
        model_id=args.model_id,
        model_revision=args.model_revision,
        model_files_sha256=args.model_files_sha256,
        faster_whisper_version=args.faster_whisper_version,
        ctranslate2_version=args.ctranslate2_version,
        container_image_digest=args.container_image_digest,
        fallback_only=not args.all_lessons,
        lesson_ids=args.lesson_id,
        default_language=args.language,
        language_map_path=args.language_map,
        min_timeline_span_fraction=args.min_timeline_span_fraction,
        max_endpoint_gap_fraction=args.max_endpoint_gap_fraction,
        require_all_selected_media=args.require_all_selected_media,
    )
    target = write_json(args.output, manifest)
    receipt = build_pending_teachobs_asr_receipt(
        args.media_manifest,
        args.caption_audit,
        job_manifest=manifest,
        job_manifest_path=target,
    )
    write_json(args.public_receipt, receipt)
    print(
        json.dumps(
            {
                "private_job_manifest": str(target.resolve()),
                "job_count": len(manifest["jobs"]),
                "media_pending_count": len(manifest["media_pending"]),
                "handoff_status": "pending_gpu_execution",
                "whisper_executed": False,
                "model_or_media_downloaded": False,
                "asr_is_official_caption": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_import_teachobs_asr_results(args: argparse.Namespace) -> int:
    result = import_teachobs_asr_results(
        args.job_manifest,
        args.media_manifest,
        args.media_root,
        args.caption_audit,
        args.results,
    )
    outputs = write_teachobs_asr_import(
        result,
        audit_path=args.output,
        coverage_matrix_path=args.coverage_matrix,
        public_receipt_path=args.public_receipt,
    )
    audit = result["audit"]["aggregate"]
    coverage = result["coverage_matrix"]["aggregate"]
    print(
        json.dumps(
            {
                "private_audit": str(outputs["audit"].resolve()),
                "private_coverage_matrix": str(
                    outputs["coverage_matrix"].resolve()
                ),
                "asr_job_count": audit["job_count"],
                "valid_asr_result_count": audit["valid_result_count"],
                "pending_asr_result_count": audit["pending_result_count"],
                "covered_lesson_count": coverage["covered_lesson_count"],
                "transcript_source_coverage_complete": coverage[
                    "transcript_source_coverage_complete"
                ],
                "asr_is_official_caption": False,
                "content_accuracy_established": False,
                "word_error_rate_established": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if coverage["transcript_source_coverage_complete"] else 2


def command_materialize_teachobs_transcripts(args: argparse.Namespace) -> int:
    result = materialize_teachobs_transcripts(
        args.media_plan,
        args.media_manifest,
        args.caption_audit,
        args.asr_import_audit,
        args.coverage_matrix,
        args.asr_job_manifest,
        args.asr_results,
        args.output,
        public_receipt_path=args.public_receipt,
    )
    manifest = result["manifest"]
    print(
        json.dumps(
            {
                "private_manifest": result["manifest_path"],
                "public_receipt": result["public_receipt_path"],
                "profile_id": manifest["profile_id"],
                "lesson_count": manifest["aggregate"]["lesson_count"],
                "scene_count": manifest["aggregate"]["scene_count"],
                "materialization_fingerprint_sha256": manifest[
                    "materialization_fingerprint_sha256"
                ],
                "released_transcript_fallback_used": False,
                "labels_read_or_used": False,
                "content_accuracy_established": False,
                "word_error_rate_established": False,
                "recognition_accuracy_established": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_hash_teachobs_asr_model(args: argparse.Namespace) -> int:
    print(model_directory_sha256(args.model_directory))
    return 0


def command_run_teachobs_asr_gpu(args: argparse.Namespace) -> int:
    runner_source = Path(teachobs_asr_handoff.__file__).resolve()
    summary = run_teachobs_asr_gpu_jobs(
        args.job_manifest,
        args.media_root,
        args.model_directory,
        args.output,
        container_image_digest=args.container_image_digest,
        runner_source_sha256=sha256(runner_source.read_bytes()).hexdigest(),
        lesson_ids=args.lesson_id,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_prepare_teachobs_double_annotation(args: argparse.Namespace) -> int:
    manifest = prepare_teachobs_double_annotation(
        args.repository,
        args.output,
        lesson_ids=args.lesson_id,
        media_root=args.media_root,
        media_reference_prefix=args.media_reference_prefix,
        operational_codebook_path=args.operational_codebook,
        seed_a=args.seed_a,
        seed_b=args.seed_b,
        require_media=args.require_media,
    )
    if args.public_receipt:
        write_json(
            args.public_receipt,
            build_pending_public_teachobs_annotation_receipt(manifest),
        )
    summary = {
        "private_output": str(Path(args.output).resolve()),
        "selected_lesson_count": manifest["selection"]["lesson_count"],
        "assigned_scene_count": manifest["selection"]["scene_item_count"],
        "code_count": len(manifest["codes"]),
        "assignment_orders_differ": manifest["blindness"][
            "assignment_orders_differ"
        ],
        "operational_definitions_complete": manifest["operational_codebook"][
            "operational_definitions_complete"
        ],
        "annotation_execution_ready": manifest["operational_codebook"][
            "annotation_execution_ready"
        ],
        "gold_predictions_or_transcript_text_included": False,
        "media_bytes_copied": False,
        "human_completion": False,
        "inter_rater_reliability_computed": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_analyze_teachobs_double_annotation(args: argparse.Namespace) -> int:
    result = analyze_teachobs_double_annotations(
        args.manifest,
        args.assignment_a,
        args.assignment_b,
        args.output,
    )
    manifest = read_json(args.manifest)
    if args.public_receipt:
        write_json(
            args.public_receipt,
            build_completed_public_teachobs_annotation_receipt(
                manifest, result["report"]
            ),
        )
    report = result["report"]
    overall = report["overall"]
    summary = {
        "private_output": str(Path(args.output).resolve()),
        "validated_annotator_count": report["annotation_validation"][
            "annotator_count"
        ],
        "human_completion": report["annotation_validation"]["human_completion"],
        "completion_basis": report["annotation_validation"][
            "human_completion_basis"
        ],
        "human_identity_independently_verified": report["annotation_validation"][
            "human_identity_independently_verified"
        ],
        "macro_label_cohen_kappa": overall["macro_label_cohen_kappa"],
        "pooled_binary_cohen_kappa": overall["pooled_binary"]["cohen_kappa"],
        "disagreement_count": overall["disagreement_count"],
        "adjudication_completed": False,
        "recognition_accuracy_established": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_prepare_teachobs_media(args: argparse.Namespace) -> int:
    yt_dlp_command: list[str] | str | None
    if args.yt_dlp_python:
        yt_dlp_command = [args.yt_dlp_python, "-m", "yt_dlp"]
    else:
        yt_dlp_command = args.yt_dlp_executable
    result = prepare_teachobs_media_dataset(
        args.repository,
        args.output,
        lesson_ids=args.lesson_id,
        source_override_manifest_path=args.source_override_manifest,
        acknowledge_source_terms=args.acknowledge_source_terms,
        acknowledge_override_source_terms=(
            args.acknowledge_override_source_terms
        ),
        dry_run=args.dry_run,
        include_audio_statistics=not args.no_audio_statistics,
        include_visual_evidence=not args.no_visual_evidence,
        use_ocr=not args.no_ocr,
        ocr_language=args.ocr_language,
        clip_model=args.clip_model,
        clip_source_revision=args.clip_source_revision,
        clip_device=args.clip_device,
        clip_batch_size=args.clip_batch_size,
        yt_dlp_command=yt_dlp_command,
        js_runtime=args.js_runtime,
        cookies_from_browser=args.cookies_from_browser,
        yt_dlp_direct=args.yt_dlp_direct,
        yt_dlp_impersonate=args.yt_dlp_impersonate,
        ffmpeg_command=args.ffmpeg,
        ffprobe_command=args.ffprobe,
        download_jobs=args.jobs,
        feature_jobs=args.feature_jobs,
        visual_jobs=args.visual_jobs,
        ocr_jobs=args.ocr_jobs,
    )
    plan = result["plan"]
    summary: dict[str, Any] = {
        "mode": result["mode"],
        "private_output": str(Path(args.output).resolve()),
        "lesson_count": plan["lesson_count"],
        "scene_count": plan["scene_count"],
        "duration_hours": round(plan["reference_duration_seconds"] / 3600, 6),
        "source_urls_or_lesson_ids_printed": False,
        "public_release_authorized": False,
        "recognition_accuracy_established": False,
        "multimodal_gain_established": False,
    }
    if result["mode"] != "dry_run":
        summary.update(
            {
                "media_complete": result["media"]["manifest"][
                    "selected_complete"
                ],
                "feature_complete": result["features"]["manifest"]["complete"],
            }
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_teachobs_multimodal_benchmark(args: argparse.Namespace) -> int:
    result = run_teachobs_multimodal_benchmark(
        args.repository,
        args.feature_manifest,
        transcript_materialization_manifest_path=(
            args.transcript_materialization_manifest
        ),
        frozen_model_output=args.frozen_model_output,
        profile=args.evaluation_profile,
    )
    output = write_json(args.output, result)
    if args.public_receipt:
        receipt = build_public_teachobs_multimodal_receipt(result)
        write_json(args.public_receipt, receipt)
    summary = {
        "output": str(output.resolve()),
        "evaluation_profile": result["dataset_audit"]["benchmark_profile"],
        "train_lessons": result["dataset_audit"]["train_lesson_count"],
        "test_lessons": result["dataset_audit"]["test_lesson_count"],
        "test_scenes": result["dataset_audit"]["test_scene_count"],
        "label_count": result["dataset_audit"]["code_count"],
        "visual_feature_layout": result["private_feature_audit"][
            "visual_feature_layout"
        ],
        "arm_metrics": {
            name: arm["metrics"] for name, arm in result["arms"].items()
        },
        "paired_cluster_bootstrap": result["paired_cluster_bootstrap"],
        "valid_claim": result["evidence_scope"]["valid_claim"],
        "frozen_model_export": result.get("frozen_model_export"),
        "confirmatory_multimodal_gain_established": False,
        "external_lockbox_established": False,
        "deployment_accuracy_established": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _teachobs_lockbox_arm_models(values: list[str]) -> dict[str, str]:
    models: dict[str, str] = {}
    for value in values:
        arm, separator, path = value.partition("=")
        if (
            not separator
            or arm not in TEACHOBS_LOCKBOX_ARM_ORDER
            or not path.strip()
        ):
            raise ValueError(
                "--arm-model must use one of "
                f"{','.join(TEACHOBS_LOCKBOX_ARM_ORDER)}=PATH"
            )
        if arm in models:
            raise ValueError(f"duplicate --arm-model for {arm}")
        models[arm] = path
    return models


def command_prepare_teachobs_lockbox_preregistration(
    args: argparse.Namespace,
) -> int:
    if args.expected_development_profile:
        if not args.system_artifact:
            raise ValueError(
                "--expected-development-profile requires --system-artifact"
            )
        from .teachobs_frozen_model import load_teachobs_frozen_bundle

        system_path = Path(args.system_artifact).expanduser().resolve()
        if system_path.name != "bundle_manifest.json":
            raise ValueError(
                "profile-pinned lockbox system artifact must be bundle_manifest.json"
            )
        load_teachobs_frozen_bundle(
            system_path.parent,
            expected_benchmark_profile=args.expected_development_profile,
        )
    preregistration = build_teachobs_lockbox_preregistration(
        study_id=args.study_id,
        system_artifact_path=args.system_artifact,
        analysis_code_path=args.analysis_code,
        arm_model_artifact_paths=_teachobs_lockbox_arm_models(args.arm_model),
        target_cluster_field=args.target_cluster_field,
    )
    validation = validate_teachobs_lockbox_preregistration(preregistration)
    output = write_json(args.output, preregistration)
    summary = {
        "output": str(output.resolve()),
        "expected_development_profile": args.expected_development_profile,
        "preregistration_fingerprint": validation[
            "preregistration_fingerprint"
        ],
        "frozen_artifact_set_complete": validation[
            "frozen_artifact_set_complete"
        ],
        "per_arm_label_thresholds_bound": validation[
            "gates"
        ]["per_arm_label_thresholds_bound"],
        "preregistration_execution_ready": False,
        "external_registration_signature_verified": False,
        "target_execution_evidence_complete": False,
        "confirmatory_multimodal_gain_established": False,
        "deployment_accuracy_established": False,
        "learner_effectiveness_established": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_longform_multimodal_dataset(args: argparse.Namespace) -> int:
    report = process_longform_dataset(
        args.media_manifest,
        args.transcript_manifest,
        args.output,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        frame_interval_seconds=args.frame_interval,
        scene_threshold=args.scene_threshold,
        max_scene_frames_per_chunk=args.max_scenes_per_chunk,
        use_ocr=not args.no_ocr,
        ocr_workers=args.ocr_workers,
        resume=not args.no_resume,
    )
    summary = {
        "manifest": str((Path(args.output) / "dataset_manifest.json").resolve()),
        "video_count": report["video_count"],
        **report["aggregate"],
        "recognition_accuracy_established": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    passed = bool(
        report["video_count"]
        and report["aggregate"]["full_timeline_sampling_passed_count"]
        == report["video_count"]
        and report["aggregate"][
            "caption_timeline_media_binding_verified_count"
        ]
        == report["video_count"]
    )
    return 0 if passed else 2


def command_multimodal_ablation(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    manifest = read_json(manifest_path)
    transcripts = []
    for item in manifest.get("videos", []):
        value = Path(str(item["transcript_path"]))
        candidates = (value, manifest_path.parent / value, Path.cwd() / value)
        transcript_path = next(
            (candidate for candidate in candidates if candidate.is_file()), None
        )
        if transcript_path is None:
            raise FileNotFoundError(value)
        transcripts.append(read_json(transcript_path))
    report = evaluate_multimodal_ablation(transcripts)
    payloads = report.pop("_payloads")
    output = ensure_private_directory(args.output)
    skill_dir = ensure_private_directory(output / "skills")
    evaluation_dir = ensure_private_directory(output / "evaluations")
    rows = {str(row["video_id"]): row for row in report["per_lecture"]}
    for video_id, arms in payloads.items():
        for arm in ARM_ORDER:
            skill, evaluation = arms[arm]
            skill_path = write_json(
                skill_dir / f"{video_id}.{arm}.skill.json", skill
            )
            evaluation_path = write_json(
                evaluation_dir / f"{video_id}.{arm}.evaluation.json", evaluation
            )
            rows[video_id]["arms"][arm]["skill_artifact"] = str(
                skill_path.relative_to(output)
            )
            rows[video_id]["arms"][arm]["evaluation_artifact"] = str(
                evaluation_path.relative_to(output)
            )
    target = write_json(output / "ablation_report.json", report)
    print(
        json.dumps(
            {
                "report": str(target.resolve()),
                "paired_lecture_count": report["paired_lecture_count"],
                "aggregate_internal_metrics": report[
                    "aggregate_internal_metrics"
                ],
                "recognition_accuracy_established": False,
                "causal_multimodal_gain_established": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_visual_semantic_extract(args: argparse.Namespace) -> int:
    return execute_visual_semantic_extraction(args)


def command_visual_semantic_dataset(args: argparse.Namespace) -> int:
    return execute_visual_semantic_dataset(args)


def command_visual_semantic_apply(args: argparse.Namespace) -> int:
    return execute_visual_semantic_apply(args)


def command_multimodal_benchmark(args: argparse.Namespace) -> int:
    transcript = read_json(args.transcript)
    analysis = transcript.get("multimodal")
    if not isinstance(analysis, dict):
        raise ValueError("transcript does not contain multimodal analysis")
    report = evaluate_multimodal_fixture(analysis, read_json(args.ground_truth))
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


def command_real_data_audit(args: argparse.Namespace) -> int:
    manifest = discover_ouc_cge(args.dataset_root)
    audit = manifest["audit"]
    if args.output:
        write_json(args.output, manifest)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["invalid_video_count"] == 0 and audit["provenance_verified"] else 2


def command_real_recognition_benchmark(args: argparse.Namespace) -> int:
    report = run_real_classroom_benchmark(
        args.dataset_root,
        args.output_dir,
        folds=args.folds,
        seed=args.seed,
        frame_count=args.frames,
    )
    summary = {
        "benchmark_kind": report["benchmark_kind"],
        "dataset_variant": report["dataset_variant"],
        "usable_unique_sample_count": report["usable_unique_sample_count"],
        "surrogate_group_count": report["surrogate_group_count"],
        "majority_macro_f1": report["majority_baseline"]["macro_f1"],
        "encoding_metadata_macro_f1": report["encoding_metadata_nuisance_baseline"]["macro_f1"],
        "visual_macro_f1": report["modality_results"]["visual"]["macro_f1"],
        "audio_macro_f1": report["modality_results"]["audio"]["macro_f1"],
        "fusion_macro_f1": report["modality_results"]["fusion"]["macro_f1"],
        "fusion_accuracy": report["modality_results"]["fusion"]["accuracy"],
        "multimodal_gain_established_on_pilot": report["multimodal_gain_established_on_pilot"],
        "multimodal_audio_coverage_valid": report["audio_availability_audit"][
            "eligible_for_multimodal_accuracy_claim"
        ],
        "released_checkpoint_modality": report["released_checkpoint_modality"],
        "sample_shortcut_risk_detected": report["sample_shortcut_risk_detected"],
        "shuffled_label_mean_macro_f1": report["post_feature_label_permutation_sanity"][
            "mean_macro_f1"
        ],
        "real_world_recognition_accuracy_established": report[
            "real_world_recognition_accuracy_established"
        ],
        "artifacts": str(Path(args.output_dir).resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(report["quality_gates"].values()) else 2


def command_real_recognition_infer(args: argparse.Namespace) -> int:
    prediction = infer_classroom_engagement(args.video, args.checkpoint)
    if args.output:
        write_json(args.output, prediction)
    print(json.dumps(prediction, ensure_ascii=False, indent=2))
    return 0


def command_dipser_credible_benchmark(args: argparse.Namespace) -> int:
    def progress(event: dict[str, Any]) -> None:
        reason = f" ({event['reason']})" if event.get("reason") else ""
        print(
            f"DIPSER {event['completed']}/{event['total']} "
            f"{event['status']}: {event['official_path']}{reason}",
            file=sys.stderr,
            flush=True,
        )

    report = run_dipser_credible_experiment(
        args.output_dir,
        archive_paths=args.archive,
        interval_seconds=args.interval_seconds,
        alignment_tolerance_seconds=args.alignment_tolerance_seconds,
        watch_filename_tolerance_seconds=args.watch_filename_tolerance_seconds,
        watch_internal_tolerance_seconds=args.watch_internal_tolerance_seconds,
        max_workers=args.workers,
        range_timeout_seconds=args.range_timeout_seconds,
        resume_archive_cache=not args.no_resume,
        evaluation_kwargs={
            "outer_splits": args.outer_splits,
            "inner_splits": args.inner_splits,
            "bootstrap_replicates": args.bootstrap_replicates,
            "permutation_replicates": args.permutation_replicates,
            "seed": args.seed,
        },
        progress_callback=progress,
    )
    session = report["session_disjoint"]
    session_metrics = session.get("metrics") or {}
    fusion_metrics = session_metrics.get("fusion") or {}
    fusion_intervals = (
        session.get("cluster_bootstrap_95_intervals", {}).get("fusion", {})
    )
    gain = session.get("fusion_vs_best_unimodal_nested", {}).get("macro_f1", {})
    blocked_summary = {}
    for name, evaluation in report.get("blocked_descriptive", {}).get(
        "evaluations", {}
    ).items():
        fusion = evaluation.get("modalities", {}).get("fusion", {})
        blocked_summary[name] = {
            "evaluated_fold_count": evaluation.get("evaluated_fold_count"),
            "unevaluable_fold_count": evaluation.get("unevaluable_fold_count"),
            "fusion_sample_weighted": fusion.get("pooled_oof_sample_weighted"),
            "fusion_cell_equal_weighted": fusion.get(
                "test_cell_session_equal_weighted"
            ),
            "descriptive_fusion_deltas": evaluation.get(
                "descriptive_fusion_deltas"
            ),
        }
    summary = {
        "protocol": report["protocol"],
        "dataset_fingerprint": report["dataset_fingerprint"],
        "planned_archive_count": report["planned_archive_count"],
        "failed_archive_count": len(report["archive_failures"]),
        "valid_record_count": report["valid_record_count"],
        "valid_session_count": report["valid_session_count"],
        "valid_participant_count": report["valid_participant_count"],
        "session_disjoint_fusion_accuracy": fusion_metrics.get("accuracy"),
        "session_disjoint_fusion_accuracy_95_ci": fusion_intervals.get("accuracy_95_ci"),
        "session_disjoint_fusion_macro_f1": fusion_metrics.get("macro_f1"),
        "session_disjoint_fusion_macro_f1_95_ci": fusion_intervals.get("macro_f1_95_ci"),
        "fusion_vs_nested_best_unimodal_macro_f1_delta": gain.get("observed_delta"),
        "gain_delta_95_ci": gain.get("paired_group_bootstrap_95_ci"),
        "gain_one_sided_group_permutation_p": gain.get(
            "one_sided_group_swap_permutation_p"
        ),
        "blocked_descriptive": blocked_summary,
        "session_disjoint_accuracy_established": report[
            "session_disjoint_accuracy_established"
        ],
        "session_blocked_oof_estimate_available": report[
            "session_blocked_oof_estimate_available"
        ],
        "session_disjoint_multimodal_gain_established": report[
            "session_disjoint_multimodal_gain_established"
        ],
        "participant_disjoint_accuracy_established": report[
            "participant_disjoint_accuracy_established"
        ],
        "participant_disjoint_multimodal_gain_established": report[
            "participant_disjoint_multimodal_gain_established"
        ],
        "deployment_accuracy_established": report["deployment_accuracy_established"],
        "artifacts": str(Path(args.output_dir).resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report["session_blocked_oof_estimate_available"] else 2


def command_human_evaluate(args: argparse.Namespace) -> int:
    expected_skill_ids = None
    expected_skill_fingerprints = None
    if args.skills:
        skills = _load_skills(args.skills)
        expected_skill_ids = sorted(str(skill["skill_id"]) for skill in skills)
        expected_skill_fingerprints = {
            str(skill["skill_id"]): skill_review_fingerprint(skill)
            for skill in skills
        }
    report = summarize_human_review(
        args.input,
        expected_skill_ids=expected_skill_ids,
        expected_skill_fingerprints=expected_skill_fingerprints,
    )
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


def _write_human_review(path: Path, skills: list[dict[str, Any]]) -> bool:
    """Create a blank review template once, never overwrite human work."""

    if path.exists():
        return False
    fields = [
        "skill_id",
        "skill_fingerprint",
        "reviewer_id",
        "goal_clarity_1_5",
        "evidence_fidelity_1_5",
        "procedure_executability_1_5",
        "adaptivity_1_5",
        "transferability_1_5",
        "harm_or_bias_flag_0_1",
        "comments",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for skill in skills:
        row = {
            "skill_id": skill["skill_id"],
            "skill_fingerprint": skill_review_fingerprint(skill),
        }
        writer.writerow(row)
        writer.writerow(row)
    write_text(path, "\ufeff" + buffer.getvalue())
    return True


def _summary_markdown(summary: dict[str, Any], reports: list[dict[str, Any]]) -> str:
    lines = [
        "# Teaching Skill 自动评估汇总",
        "",
        f"- Skill 数：{summary['skill_count']}",
        f"- 课程数：{summary['course_count']}",
        f"- 视频数：{summary['video_count']}",
        f"- 平均分：{summary['average_score']}",
        f"- 通过率：{summary['pass_rate']:.0%}",
        f"- 教学策略类型数：{summary['strategy_type_count']}",
        f"- 自动化核心覆盖：{'通过' if summary.get('core_collection_passed') else '未通过'}",
        f"- 工程演示闭环：{'通过' if summary['passed'] else '未通过'}",
        f"- 独立人工复核：{'完成' if summary.get('human_validation_completed') else '未完成（模板不等于结果）'}",
        f"- 完整研究验证：{'完成' if summary.get('research_validation_complete') else '未完成'}",
        f"- 跨领域能力基准：{'通过' if summary.get('transfer_benchmark_passed') else '未通过'}",
        f"- Skill/静态基线：{summary.get('transfer_skill_mean', 0):.1f} / {summary.get('transfer_baseline_mean', 0):.1f}（能力覆盖分，不是学习增益）",
        f"- 数据结构审计：{'通过' if summary.get('dataset_structure_passed') else '未通过'}",
        f"- 正式实证数据就绪：{'是' if summary.get('formal_empirical_ready') else '否（当前为离线释义节选）'}",
        "",
        "| Skill | 总分 | 等级 | 结果 |",
        "|---|---:|:---:|:---:|",
    ]
    for report in reports:
        lines.append(
            f"| `{report['skill_id']}` | {report['overall_score']:.1f} | {report['grade']} | {'通过' if report['passed'] else '未通过'} |"
        )
    lines.extend(["", "策略覆盖：" + "、".join(summary["strategy_coverage"]), ""])
    return "\n".join(lines)


def command_demo(args: argparse.Namespace) -> int:
    manifest_path = resolve_resource_path(
        args.manifest or "data/dataset_manifest.json"
    )
    manifest = read_json(manifest_path)
    root = resolve_manifest_root(manifest_path, manifest)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    skills: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for item in manifest.get("videos", []):
        transcript_path = root / item["transcript_path"]
        transcript = read_json(transcript_path)
        skill = _mine(transcript, args.backend)
        report = evaluate_skill(skill, transcript)
        write_json(output / "skills" / f"{item['video_id']}.skill.json", skill)
        write_json(output / "evaluations" / f"{item['video_id']}.evaluation.json", report)
        skills.append(skill)
        reports.append(report)
    summary = evaluate_collection(skills, reports, manifest)
    audit = audit_dataset(manifest, root)
    cases_path = root / "data" / "evaluation_cases.json"
    if not cases_path.is_file():
        cases_path = resolve_resource_path("data/evaluation_cases.json")
    cases_payload = read_json(cases_path)
    benchmark = benchmark_transfer(skills, cases_payload["cases"])
    core_passed = summary["passed"]
    summary["core_collection_passed"] = core_passed
    summary["transfer_benchmark_passed"] = benchmark["passed"]
    summary["transfer_skill_mean"] = benchmark["skill_mean"]
    summary["transfer_baseline_mean"] = benchmark["static_baseline_mean"]
    summary["transfer_mean_delta"] = benchmark["mean_delta"]
    summary["dataset_structure_passed"] = audit["dataset_structure_passed"]
    summary["formal_empirical_ready"] = audit["formal_empirical_ready"]
    summary["passed"] = core_passed and benchmark["passed"] and audit["dataset_structure_passed"]
    review_path = output / "human_review.csv"
    review_template_created = _write_human_review(review_path, skills)
    human_report = summarize_human_review(
        review_path,
        expected_skill_ids=sorted(str(skill["skill_id"]) for skill in skills),
        expected_skill_fingerprints={
            str(skill["skill_id"]): skill_review_fingerprint(skill)
            for skill in skills
        },
    )
    summary["human_validation_completed"] = human_report["passed"]
    summary["human_validation_status"] = human_report["validation_status"]
    summary["human_review_template_created"] = review_template_created
    summary["formal_transcript_and_human_review_complete"] = bool(
        audit["formal_empirical_ready"] and human_report["passed"]
    )
    summary["research_validation_complete"] = False
    research_validation_pending = []
    if not audit["formal_empirical_ready"]:
        research_validation_pending.append(
            "formal_full_transcripts_or_audited_asr"
        )
    if not human_report["passed"]:
        research_validation_pending.append("independent_human_review")
    research_validation_pending.extend(
        [
            "confirmatory_external_multimodal_gain",
            "cryptographically_registered_prospective_deployment_evaluation",
            "real_learner_effectiveness_study",
        ]
    )
    summary["research_validation_pending"] = research_validation_pending
    summary["passed_semantics"] = (
        "engineering demonstration gates only; external human and learner evidence are separate"
    )
    write_json(output / "summary.json", summary)
    write_json(output / "data_audit.json", audit)
    write_json(output / "transfer_benchmark.json", benchmark)
    write_json(output / "human_review_status.json", human_report)
    write_text(output / "summary.md", _summary_markdown(summary, reports))
    demo_skill = next(
        (skill for skill in skills if skill.get("source", {}).get("video_id") == args.demo_video_id),
        skills[0],
    )
    write_text(
        output / "teaching_demo.md",
        execute_skill(demo_skill, concept=args.concept, learner_level=args.learner_level),
    )
    write_json(
        output / "interactive_demo.json",
        run_scripted_session(
            demo_skill,
            concept=args.concept,
            learner_level=args.learner_level,
            responses=_script_for_complete_session(demo_skill),
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"artifacts: {output.resolve()}")
    return 0 if summary["passed"] else 2


def command_doctor(args: argparse.Namespace) -> int:
    report = doctor_report()
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["capabilities"]["offline_demo_ready"] else 2


def command_dashboard(args: argparse.Namespace) -> int:
    if args.check:
        report = dashboard_self_check()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 2
    return open_dashboard(
        output=args.output,
        open_browser=not args.no_browser,
    )


def command_private_dashboard(args: argparse.Namespace) -> int:
    if args.check_template:
        report = private_dashboard_template_self_check()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 2
    if args.check_data:
        snapshot = build_private_snapshot(
            PrivateDashboardConfig(
                root=args.teachobs_root,
                initial_lesson=args.initial_lesson,
                skill_root=args.skill_root,
                initial_skill=args.initial_skill,
            )
        )
        print(
            json.dumps(
                {
                    **private_snapshot_summary(snapshot),
                    "file_integrity": snapshot.verify_all_private_files(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    return serve_private_dashboard(
        args.teachobs_root,
        skill_root=args.skill_root,
        initial_lesson=args.initial_lesson,
        initial_skill=args.initial_skill,
        port=args.port,
        open_browser=not args.no_browser,
    )


def command_verify_delivery(args: argparse.Namespace) -> int:
    manifest_path = resolve_resource_path(args.manifest)
    cases_path = resolve_resource_path(args.cases)
    human_review = args.human_review
    if human_review is None:
        default_review = Path("artifacts/human_review.csv")
        human_review = str(default_review) if default_review.is_file() else None
    dipser_report = args.dipser_report
    if dipser_report is None:
        default_dipser = Path(
            "artifacts/dipser_credible/full_v5_52_complete_v5/"
            "hierarchical_0_9_report.json"
        )
        dipser_report = str(default_dipser) if default_dipser.is_file() else None
    report = verify_delivery(
        manifest_path,
        cases_path,
        formal_manifest_path=(
            resolve_resource_path(args.formal_manifest)
            if args.formal_manifest
            else None
        ),
        human_review_path=human_review,
        dipser_report_path=dipser_report,
        external_deployment_report_path=args.external_deployment_report,
        external_evaluation_receipt_path=args.external_evaluation_receipt,
        trusted_attestation_public_key_path=args.trusted_attestation_public_key,
        external_multimodal_evidence_path=args.external_multimodal_evidence,
        external_multimodal_attestation_path=args.external_multimodal_attestation,
        trusted_multimodal_public_key_path=args.trusted_multimodal_public_key,
        external_learner_evidence_path=args.external_learner_evidence,
        external_learner_attestation_path=args.external_learner_attestation,
        trusted_learner_public_key_path=args.trusted_learner_public_key,
        external_evaluated_system_artifact_path=(
            args.external_evaluated_system_artifact
        ),
    )
    if args.output:
        write_json(args.output, report)
    if args.markdown:
        write_text(args.markdown, delivery_markdown(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_external_validation:
        return 0 if report["research_validation_complete"] else 2
    return 0 if report["engineering_delivery_ready"] else 2


def command_release_audit(args: argparse.Namespace) -> int:
    report = audit_release_path(args.path)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


def command_export_public_dipser(args: argparse.Namespace) -> int:
    report = export_public_dipser_report(args.input, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _load_strict_feature_bundle(
    path: str | Path, manifest: dict[str, Any]
) -> tuple[
    dict[str, Any],
    dict[str, list[str]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("feature bundle must be one JSON object")
    if payload.get("schema_version") != "1.0":
        raise ValueError("feature bundle schema_version must be 1.0")
    records = manifest.get("records")
    audit = manifest.get("audit")
    if not isinstance(records, list) or not records or not isinstance(audit, dict):
        raise ValueError("manifest must contain non-empty records and an audit object")
    expected_dataset_fingerprint = strict_dataset_fingerprint(records)
    if audit.get("dataset_fingerprint") != expected_dataset_fingerprint:
        raise ValueError("manifest dataset_fingerprint does not match its records")
    if payload.get("dataset_fingerprint") != expected_dataset_fingerprint:
        raise ValueError("feature bundle dataset_fingerprint does not match the manifest")
    sample_ids = payload.get("sample_ids")
    matrices = payload.get("matrices")
    names = payload.get("feature_names_by_modality")
    provenance = payload.get("feature_provenance")
    if not isinstance(sample_ids, list):
        raise ValueError("feature bundle requires sample_ids")
    if not isinstance(matrices, dict) or not isinstance(names, dict) or not isinstance(provenance, dict):
        raise ValueError(
            "feature bundle requires matrices, feature_names_by_modality, and feature_provenance objects"
        )
    if (
        not sample_ids
        or any(not isinstance(value, str) or not value for value in sample_ids)
        or len(set(sample_ids)) != len(sample_ids)
    ):
        raise ValueError("feature bundle sample_ids must be unique non-empty strings")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("manifest records must be JSON objects")
    record_ids = [record.get("sample_id") for record in records]
    if record_ids != sample_ids:
        raise ValueError("manifest records and feature bundle rows are not in exact order")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("feature modality names must be non-empty strings")
    normalized_names: dict[str, list[str]] = {}
    for name, values in names.items():
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"feature names for {name} must be unique non-empty strings")
        normalized_names[name] = list(values)
    matrix_modalities = set(matrices)
    provenance_modalities = set(provenance)
    if set(normalized_names) != matrix_modalities or matrix_modalities != provenance_modalities:
        raise ValueError("feature bundle modality schemas are inconsistent")
    algorithm = payload.get("fingerprint_algorithm")
    if algorithm is None:
        archive_fingerprint = audit.get("archive_catalog_fingerprint")
        legacy_dipser = bool(
            audit.get("dataset_id") == "DIPSER"
            and matrix_modalities == {"visual", "sensor"}
            and isinstance(archive_fingerprint, str)
            and len(archive_fingerprint) == 64
            and all(char in "0123456789abcdef" for char in archive_fingerprint.lower())
            and audit.get("feature_bundle_fingerprint")
            == payload.get("feature_bundle_fingerprint")
        )
        if not legacy_dipser:
            raise ValueError("feature bundle requires an explicit fingerprint_algorithm")
        algorithm = "dipser_feature_bundle_v1"
    strict_fingerprint = strict_feature_bundle_fingerprint(
        records, normalized_names, matrices, provenance
    )
    if algorithm == "strict_feature_bundle_v1":
        expected_bundle_fingerprint = strict_fingerprint
    elif algorithm == "dipser_feature_bundle_v1":
        archive_fingerprint = audit.get("archive_catalog_fingerprint")
        if (
            audit.get("dataset_id") != "DIPSER"
            or matrix_modalities != {"visual", "sensor"}
            or not isinstance(archive_fingerprint, str)
            or len(archive_fingerprint) != 64
        ):
            raise ValueError("legacy DIPSER fingerprint is only valid for DIPSER visual/sensor bundles")
        expected_bundle_fingerprint = dipser_feature_bundle_fingerprint(
            records, normalized_names, matrices
        )
    else:
        raise ValueError(f"unsupported feature bundle fingerprint_algorithm: {algorithm}")
    if payload.get("feature_bundle_fingerprint") != expected_bundle_fingerprint:
        raise ValueError("feature_bundle_fingerprint does not match exact bundle contents")
    manifest_bundle_fingerprint = audit.get("feature_bundle_fingerprint")
    if manifest_bundle_fingerprint is not None and manifest_bundle_fingerprint != expected_bundle_fingerprint:
        raise ValueError("manifest and feature bundle fingerprints disagree")
    raw_binding = payload.get("raw_source_binding")
    raw_binding_summary: dict[str, Any] | None = None
    if raw_binding is not None:
        raw_binding_summary = validate_raw_source_binding(raw_binding, records)
    extractor_suite = payload.get("extractor_suite")
    extractor_suite_summary: dict[str, Any] | None = None
    if extractor_suite is not None:
        extractor_suite_summary = validate_extractor_suite_binding(
            extractor_suite, normalized_names, provenance
        )
    if (raw_binding is None) != (extractor_suite is None):
        raise ValueError(
            "raw-generated bundles require both raw_source_binding and extractor_suite"
        )
    return matrices, normalized_names, provenance, {
        "input_fingerprint_algorithm": algorithm,
        "input_feature_bundle_fingerprint": expected_bundle_fingerprint,
        "strict_feature_bundle_fingerprint": strict_fingerprint,
        "legacy_input_upgraded_to_strict_binding": algorithm != "strict_feature_bundle_v1",
        "raw_source_binding_verified": raw_binding_summary is not None,
        "raw_source_binding_fingerprint": (
            raw_binding_summary["binding_fingerprint"]
            if raw_binding_summary is not None
            else None
        ),
        "extractor_suite_binding_verified": extractor_suite_summary is not None,
        "extractor_configuration_fingerprint": (
            extractor_suite_summary["configuration_fingerprint"]
            if extractor_suite_summary is not None
            else None
        ),
    }


def command_extract_strict_features(args: argparse.Namespace) -> int:
    manifest = read_json(args.manifest)
    config = read_json(args.extractor_config)
    model = load_frozen_deployment_model(read_json(args.model)) if args.model else None
    identity_fields = (
        list(model.required_identity_fields)
        if model is not None
        else args.identity_field or ["session_id", "teacher_id", "site_id"]
    )
    bundle = extract_strict_feature_bundle(
        manifest,
        args.raw_root,
        config,
        required_identity_fields=identity_fields,
    )
    compatibility = (
        validate_bundle_against_frozen_model(bundle, model)
        if model is not None
        else {
            "verified": False,
            "reason": "no frozen model supplied; feature extraction only",
            "prediction_performed": False,
        }
    )
    output_directory = ensure_private_directory(args.output_dir)
    target = write_json(output_directory / "strict_feature_bundle.json", bundle)
    raw_binding = bundle["raw_source_binding"]
    summary = {
        "output": str(target.resolve()),
        "sample_count": len(bundle["sample_ids"]),
        "modalities": list(bundle["matrices"]),
        "dataset_fingerprint": bundle["dataset_fingerprint"],
        "feature_bundle_fingerprint": bundle["feature_bundle_fingerprint"],
        "raw_source_binding_fingerprint": raw_binding["binding_fingerprint"],
        "extractor_configuration_fingerprint": bundle["extractor_suite"][
            "configuration_fingerprint"
        ],
        "frozen_model_compatibility": compatibility,
        "automatic_recognition_performed": False,
        "recognition_accuracy_established": False,
        "deployment_accuracy_established": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_freeze_recognition_model(args: argparse.Namespace) -> int:
    manifest = read_json(args.manifest)
    features, names, provenance, feature_binding = _load_strict_feature_bundle(
        args.features, manifest
    )
    contract = read_json(args.claim_contract)
    identity_fields = args.identity_field or ["session_id", "teacher_id", "site_id"]
    model = fit_frozen_deployment_model(
        manifest,
        features,
        names,
        feature_provenance=provenance,
        class_names=args.class_name,
        modality=args.modality,
        group_field=args.group_field,
        claim_cluster_field=args.claim_cluster_field,
        required_identity_fields=identity_fields,
        claim_contract=contract,
        inner_splits=args.inner_splits,
        c_grid=args.c_grid,
        seed=args.seed,
    )
    artifact = frozen_model_to_artifact(model)
    if (
        artifact["training_feature_bundle_fingerprint"]
        != feature_binding["strict_feature_bundle_fingerprint"]
    ):
        raise RuntimeError(
            "frozen checkpoint feature binding differs from verified input bundle"
        )
    target = write_json(args.output, artifact)
    summary = {
        "output": str(target.resolve()),
        "model_fingerprint": artifact["model_fingerprint"],
        "protocol": artifact["protocol"],
        "modality": artifact["modality"],
        "required_identity_fields": artifact["required_identity_fields"],
        "claim_cluster_field": artifact["claim_cluster_field"],
        "selected_C": artifact["selected_C"],
        "training_feature_bundle_fingerprint": artifact[
            "training_feature_bundle_fingerprint"
        ],
        "input_feature_bundle_binding": feature_binding,
        "claim_contract": artifact["claim_contract"],
        "external_evaluation_performed": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_create_freeze_registration(args: argparse.Namespace) -> int:
    artifact = read_json(args.model)
    model = load_frozen_deployment_model(artifact)
    manifest = read_json(args.manifest)
    features, names, provenance, feature_binding = _load_strict_feature_bundle(
        args.features, manifest
    )
    records = manifest.get("records", [])
    manifest_summary = validate_strict_manifest(
        manifest,
        required_modalities=model.component_modalities,
        required_identity_fields=model.required_identity_fields,
    )
    if manifest_summary["class_count"] != len(model.class_names):
        raise ValueError("external labels do not match the frozen model class count")
    validate_class_schema(
        model.class_names,
        records,
        expected_class_count=manifest_summary["class_count"],
    )
    expected_modalities = set(model.component_modalities)
    if set(features) != expected_modalities or set(names) != expected_modalities:
        raise ValueError("external feature modalities do not match the frozen model")
    normalized_provenance = validate_feature_provenance(
        provenance, model.component_modalities
    )
    if normalized_provenance != model.feature_provenance_by_modality:
        raise ValueError("external feature provenance does not match the frozen model")
    for modality in model.component_modalities:
        if tuple(names[modality]) != model.feature_names_by_modality[modality]:
            raise ValueError(
                f"external feature schema does not match frozen modality {modality}"
            )
    external_sets = {
        "sample": {str(record["sample_id"]) for record in records},
        "content": {str(record["content_sha256"]) for record in records},
        **{
            field: {str(record[field]) for record in records}
            for field in model.required_identity_fields
        },
    }
    training_sets = {
        "sample": model.training_sample_ids,
        "content": model.training_content_hashes,
        **model.training_identity_values_by_field,
    }
    overlaps = {
        name: sorted(training_sets[name] & values)
        for name, values in external_sets.items()
        if training_sets[name] & values
    }
    if overlaps:
        raise ValueError(
            f"external lockbox overlaps frozen training identities/content: {overlaps}"
        )
    audit = manifest.get("audit", {})
    required_flags = (
        "held_out_from_model_development",
        "deployment_target_documented",
        "prospective_deployment_collection",
        "one_time_lockbox_evaluation",
    )
    missing = [name for name in required_flags if audit.get(name) is not True]
    if missing:
        raise ValueError(
            "external manifest cannot be registered until these flags are true: "
            + ", ".join(missing)
        )
    request = build_freeze_registration_request(
        registration_id=args.registration_id,
        model_fingerprint=model.model_fingerprint,
        claim_contract_fingerprint=artifact["claim_contract_fingerprint"],
        training_dataset_fingerprint=model.training_dataset_fingerprint,
        training_feature_bundle_fingerprint=(
            model.training_feature_bundle_fingerprint
        ),
        external_dataset_fingerprint=manifest_summary["dataset_fingerprint"],
        external_feature_bundle_fingerprint=str(
            feature_binding["strict_feature_bundle_fingerprint"]
        ),
        external_coverage_evidence_fingerprint=validate_evaluation_coverage(
            audit,
            evaluated_sample_count=len(manifest.get("records", [])),
        )["coverage_evidence_fingerprint"],
    )
    target = write_json(args.output, request)
    print(
        json.dumps(
            {
                "output": str(target.resolve()),
                "registration_id": request["registration_id"],
                "request_fingerprint": request["request_fingerprint"],
                "requires_independent_ed25519_signature": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_sign_freeze_registration(args: argparse.Namespace) -> int:
    request = read_json(args.request)
    private_key = Path(args.private_key).read_bytes()
    attestation = sign_freeze_registration(
        request,
        private_key_pem=private_key,
        issuer=args.issuer,
        key_id=args.key_id,
    )
    target = write_json(args.output, attestation)
    print(
        json.dumps(
            {
                "output": str(target.resolve()),
                "registration_id": attestation["request"]["registration_id"],
                "issuer": attestation["issuer"],
                "public_key_sha256": attestation["public_key_sha256"],
                "private_key_written_to_output": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_sign_evaluation_report(args: argparse.Namespace) -> int:
    report = read_json(args.report)
    if not isinstance(report, dict):
        raise ValueError("evaluation report must be one JSON object")
    observed = report.get("evaluation_report_fingerprint")
    expected = frozen_evaluation_report_fingerprint(report)
    if observed != expected:
        raise ValueError("evaluation report fingerprint does not match report contents")
    registration = report.get("freeze_registration")
    if not isinstance(registration, dict) or registration.get("verified") is not True:
        raise ValueError("evaluation report lacks a verified freeze registration")
    receipt = sign_evaluation_report_receipt(
        evaluation_report_fingerprint=expected,
        registration_id=str(registration.get("registration_id", "")),
        registration_attestation_fingerprint=str(
            registration.get("attestation_fingerprint", "")
        ),
        deployment_accuracy_established=(
            report.get("deployment_accuracy_established") is True
        ),
        private_key_pem=Path(args.private_key).read_bytes(),
        issuer=args.issuer,
        key_id=args.key_id,
    )
    target = write_json(args.output, receipt)
    print(
        json.dumps(
            {
                "output": str(target.resolve()),
                "evaluation_report_fingerprint": expected,
                "deployment_accuracy_established": receipt["payload"][
                    "deployment_accuracy_established"
                ],
                "private_key_written_to_output": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_prepare_external_research_evidence(args: argparse.Namespace) -> int:
    draft = read_json(args.input)
    if not isinstance(draft, dict):
        raise ValueError("external evidence draft must be one JSON object")
    evidence = finalize_external_research_evidence(draft)
    validation = validate_external_research_evidence(evidence)
    target = write_json(args.output, evidence)
    print(
        json.dumps(
            {
                "output": str(target.resolve()),
                "evidence_id": validation["evidence_id"],
                "evidence_kind": validation["evidence_kind"],
                "evidence_fingerprint": validation["evidence_fingerprint"],
                "claim_established_by_recomputed_gates": validation[
                    "claim_established"
                ],
                "source_artifact_authenticity_proven_by_preparation": False,
                "requires_external_ed25519_attestation": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_prepare_learner_effect_study(args: argparse.Namespace) -> int:
    result = generate_learner_effect_study_package(
        args.output_dir,
        study_id=args.study_id,
        cluster_unit=args.cluster_unit,
        cluster_count=args.cluster_count,
        participants_per_cluster=args.participants_per_cluster,
        randomization_seed=args.randomization_seed,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
        minimum_cluster_count=args.minimum_cluster_count,
        minimum_primary_coverage=args.minimum_primary_coverage,
        maximum_attrition_fraction=args.maximum_attrition_fraction,
        minimum_meaningful_effect=args.minimum_meaningful_effect,
        score_minimum=args.score_minimum,
        score_maximum=args.score_maximum,
        intervention_description=args.intervention_description,
        comparator_description=args.comparator_description,
        data_origin=args.data_origin,
        ethics_approval_id=args.ethics_approval_id,
        informed_consent_or_approved_waiver=(
            args.informed_consent_or_approved_waiver
        ),
        preregistration_frozen_before_allocation=(
            args.preregistration_frozen_before_allocation
        ),
        allocation_concealment_procedure_declared=(
            args.allocation_concealment_procedure_declared
        ),
    )
    print(
        json.dumps(
            {
                "mode": result["mode"],
                "output_directory": result["output_directory"],
                "planned_cluster_count": result["planned_cluster_count"],
                "planned_participant_count": result["planned_participant_count"],
                "identity_fields_included": False,
                "real_participant_data_generated": False,
                "learner_effectiveness_established": False,
                "next_step": "authorized site custodian fills the outcome template",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_analyze_learner_effect_study(args: argparse.Namespace) -> int:
    result = analyze_learner_effect_study(
        args.preregistration,
        args.allocation,
        args.outcomes,
    )
    target = write_learner_effect_analysis(args.output, result)
    print(
        json.dumps(
            {
                "output": str(target.resolve()),
                "cluster_count": result["design"]["cluster_count"],
                "randomized_participant_count": result["design"][
                    "randomized_participant_count"
                ],
                "primary_outcome_coverage_fraction": result[
                    "coverage_and_attrition"
                ]["primary_outcome_coverage_fraction"],
                "adjusted_standardized_effect": result["primary_analysis"][
                    "adjusted_standardized_effect"
                ],
                "percentile_95_ci": result["primary_analysis"][
                    "cluster_bootstrap"
                ]["percentile_95_ci"],
                "eligible_for_external_governance_review": result[
                    "eligible_for_external_governance_review"
                ],
                "learner_effectiveness_established": False,
                "external_signature_still_required": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_sign_external_research_evidence(args: argparse.Namespace) -> int:
    evidence = read_json(args.evidence)
    if not isinstance(evidence, dict):
        raise ValueError("external evidence manifest must be one JSON object")
    attestation = sign_external_research_evidence(
        evidence,
        private_key_pem=Path(args.private_key).read_bytes(),
        issuer=args.issuer,
        key_id=args.key_id,
    )
    target = write_json(args.output, attestation)
    validation = validate_external_research_evidence(evidence)
    print(
        json.dumps(
            {
                "output": str(target.resolve()),
                "evidence_id": validation["evidence_id"],
                "evidence_kind": validation["evidence_kind"],
                "evidence_fingerprint": validation["evidence_fingerprint"],
                "claim_established_by_recomputed_gates": validation[
                    "claim_established"
                ],
                "public_key_sha256": attestation["public_key_sha256"],
                "private_key_written_to_output": False,
                "signature_alone_proves_signer_independence": False,
                "delivery_requires_out_of_band_trusted_public_key": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_verify_external_research_evidence(args: argparse.Namespace) -> int:
    report = verify_external_research_evidence_files(
        args.evidence,
        args.attestation,
        args.trusted_public_key,
        expected_kind=args.expected_kind,
        expected_system_artifact_path=args.expected_system_artifact,
    )
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_claim_established:
        return 0 if report["claim_established"] else 2
    return 0


def command_evaluate_frozen_recognition(args: argparse.Namespace) -> int:
    model = load_frozen_deployment_model(read_json(args.model))
    manifest = read_json(args.manifest)
    features, names, provenance, feature_binding = _load_strict_feature_bundle(
        args.features, manifest
    )
    report = evaluate_frozen_external_deployment(
        model,
        manifest,
        features,
        names,
        feature_provenance=provenance,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        registration_attestation=(
            read_json(args.registration_attestation)
            if args.registration_attestation
            else None
        ),
        trusted_public_key_pem=(
            Path(args.trusted_public_key).read_bytes()
            if args.trusted_public_key
            else None
        ),
        one_time_ledger_dir=args.one_time_ledger,
    )
    target = write_json(args.output, report)
    summary = {
        "output": str(target.resolve()),
        "protocol": report["protocol"],
        "model_fingerprint": report["model_fingerprint"],
        "external_dataset_fingerprint": report["external_dataset_fingerprint"],
        "external_feature_bundle_fingerprint": report[
            "external_feature_bundle_fingerprint"
        ],
        "input_feature_bundle_binding": feature_binding,
        "metrics": report["metrics"],
        "claim_cluster_bootstrap": report["claim_cluster_bootstrap"],
        "claim_gate_results": report["claim_gate_results"],
        "all_frozen_claim_gates_passed": report["all_frozen_claim_gates_passed"],
        "freeze_registration": report["freeze_registration"],
        "one_time_consumption_receipt": report[
            "one_time_consumption_receipt"
        ],
        "deployment_accuracy_established": report["deployment_accuracy_established"],
        "predictions_written_to_private_artifact": True,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report["all_frozen_claim_gates_passed"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsm", description="Teaching Skill Mining and Evaluation System")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("preprocess", help="convert media/subtitle/text to transcript JSON")
    preprocess_parser.add_argument("input")
    preprocess_parser.add_argument("--video-id", required=True)
    preprocess_parser.add_argument("--course-id", required=True)
    preprocess_parser.add_argument("--title", required=True)
    preprocess_parser.add_argument("--source-url", required=True)
    preprocess_parser.add_argument("--language", default="en")
    preprocess_parser.add_argument("--output", required=True)
    preprocess_parser.set_defaults(func=command_preprocess)

    mine_parser = subparsers.add_parser("mine", help="extract one Teaching Skill")
    mine_parser.add_argument("--transcript", required=True)
    mine_parser.add_argument("--backend", choices=("heuristic", "api"), default="heuristic")
    mine_parser.add_argument("--output", required=True)
    mine_parser.set_defaults(func=command_mine)

    distill_general_parser = subparsers.add_parser(
        "distill-general-skill",
        help="aggregate validated per-lecture Skills into one course-balanced general Skill",
    )
    distill_general_parser.add_argument("--skill-root", required=True)
    distill_general_parser.add_argument(
        "--pattern",
        default="*.full.skill.json",
        help="relative glob beneath --skill-root",
    )
    distill_general_parser.add_argument("--output-dir", required=True)
    distill_general_parser.add_argument("--example-concept", default="动态规划")
    distill_general_parser.add_argument("--learner-level", default="beginner")
    distill_general_parser.add_argument(
        "--overall-support-threshold", type=float, default=0.8
    )
    distill_general_parser.add_argument(
        "--per-course-support-threshold", type=float, default=0.6
    )
    distill_general_parser.add_argument(
        "--minimum-course-count", type=int, default=2
    )
    distill_general_parser.add_argument(
        "--minimum-skills-per-course", type=int, default=5
    )
    distill_general_parser.set_defaults(func=command_distill_general_skill)

    apply_general_parser = subparsers.add_parser(
        "apply-general-skill",
        help="generate an executable teaching process for a new concept",
    )
    apply_general_parser.add_argument("--skill", required=True)
    apply_general_parser.add_argument("--concept", required=True)
    apply_general_parser.add_argument("--learner-level", default="beginner")
    apply_general_parser.add_argument("--output")
    apply_general_parser.set_defaults(func=command_apply_general_skill)

    evaluate_general_parser = subparsers.add_parser(
        "evaluate-general-skill",
        help="audit a general Skill's support, provenance, and executability",
    )
    evaluate_general_parser.add_argument("--skill", required=True)
    evaluate_general_parser.add_argument("--output")
    evaluate_general_parser.set_defaults(func=command_evaluate_general_skill)

    teach_parser = subparsers.add_parser("teach", help="execute a Teaching Skill for a new concept")
    teach_parser.add_argument("--skill", required=True)
    teach_parser.add_argument("--concept", required=True)
    teach_parser.add_argument("--learner-level", default="beginner")
    teach_parser.add_argument("--output")
    teach_parser.set_defaults(func=command_teach)

    evaluate_parser = subparsers.add_parser("evaluate", help="score a Teaching Skill")
    evaluate_parser.add_argument("--skill", required=True)
    evaluate_parser.add_argument("--transcript")
    evaluate_parser.add_argument("--output")
    evaluate_parser.set_defaults(func=command_evaluate)

    validate_parser = subparsers.add_parser("validate", help="validate a transcript or skill")
    validate_parser.add_argument(
        "kind", choices=("transcript", "skill", "general-skill")
    )
    validate_parser.add_argument("path")
    validate_parser.set_defaults(func=command_validate)

    interact_parser = subparsers.add_parser("interact", help="run an executable Skill as a state machine")
    interact_parser.add_argument("--skill", required=True)
    interact_parser.add_argument("--concept", required=True)
    interact_parser.add_argument("--learner-level", default="beginner")
    interact_parser.add_argument("--script", help="JSON response script; omit for terminal interaction")
    interact_parser.add_argument("--output")
    interact_parser.set_defaults(func=command_interact)

    teacher_start_parser = subparsers.add_parser(
        "teacher-agent-start",
        help="start one stateful task-two Agent session and emit only its first action",
    )
    teacher_start_parser.add_argument(
        "--input", default="data/teacher_agent_demo_input.json"
    )
    teacher_start_parser.add_argument(
        "--library", default="data/teacher_agent_skill_library_v2.json"
    )
    teacher_start_parser.add_argument("--session", required=True)
    teacher_start_parser.add_argument(
        "--policy",
        choices=("adaptive_skill_library", "fixed_single_skill_baseline"),
        default="adaptive_skill_library",
    )
    teacher_start_parser.add_argument("--fixed-skill-id")
    teacher_start_parser.set_defaults(func=command_teacher_agent_start)

    teacher_step_parser = subparsers.add_parser(
        "teacher-agent-step",
        help="consume one learner response and emit exactly one next action",
    )
    teacher_step_parser.add_argument("--session", required=True)
    response_group = teacher_step_parser.add_mutually_exclusive_group(required=True)
    response_group.add_argument("--response")
    response_group.add_argument("--response-file")
    teacher_step_parser.add_argument(
        "--signal",
        choices=("correct", "partial", "misconception", "confused", "no_response"),
        required=True,
    )
    teacher_step_parser.add_argument("--misconception-tag")
    teacher_step_parser.add_argument("--signal-confidence", type=float, default=1.0)
    teacher_step_parser.set_defaults(func=command_teacher_agent_step)

    teacher_evaluate_parser = subparsers.add_parser(
        "teacher-agent-evaluate",
        help="compare the adaptive Agent with a fixed single-Skill baseline",
    )
    teacher_evaluate_parser.add_argument(
        "--library", default="data/teacher_agent_skill_library_v2.json"
    )
    teacher_evaluate_parser.add_argument(
        "--cases", default="data/teacher_agent_evaluation_cases.json"
    )
    teacher_evaluate_parser.add_argument("--output")
    teacher_evaluate_parser.set_defaults(func=command_teacher_agent_evaluate)

    teacher_benchmark_parser = subparsers.add_parser(
        "teacher-agent-benchmark",
        help="run the privacy-safe free-text benchmark, offline by default",
    )
    teacher_benchmark_parser.add_argument(
        "--benchmark", default="data/teacher_agent_free_text_benchmark.json"
    )
    teacher_benchmark_parser.add_argument(
        "--skill-library", default="data/teacher_agent_skill_library_v2.json"
    )
    teacher_benchmark_parser.add_argument("--output")
    teacher_benchmark_parser.add_argument(
        "--online",
        action="store_true",
        help="send benchmark learner text to DeepSeek; requires explicit consent",
    )
    teacher_benchmark_parser.add_argument(
        "--allow-remote-benchmark-data",
        action="store_true",
        help="explicitly authorize sending the author-constructed benchmark text",
    )
    teacher_benchmark_parser.add_argument("--api-key-file")
    teacher_benchmark_parser.add_argument(
        "--model", choices=sorted(ALLOWED_MODELS), default="deepseek-v4-flash"
    )
    teacher_benchmark_parser.add_argument("--repeats", type=int, default=1)
    teacher_benchmark_parser.add_argument(
        "--minimum-review-confidence", type=float, default=0.50
    )
    teacher_benchmark_parser.add_argument(
        "--fixed-skill-id", default="skill_diagnostic_questioning"
    )
    teacher_benchmark_parser.set_defaults(func=command_teacher_agent_benchmark)

    teacher_benchmark_v2_parser = subparsers.add_parser(
        "teacher-agent-benchmark-v2",
        help="validate or score the product benchmark with separately governed gold",
    )
    teacher_benchmark_v2_parser.add_argument(
        "--benchmark", default="data/teacher_agent_benchmark_v2_development.json"
    )
    teacher_benchmark_v2_parser.add_argument(
        "--gold", default="data/teacher_agent_benchmark_v2_development_gold.json"
    )
    teacher_benchmark_v2_parser.add_argument(
        "--skill-library", default="data/teacher_agent_skill_library_v2.json"
    )
    teacher_benchmark_v2_parser.add_argument("--predictions")
    teacher_benchmark_v2_parser.add_argument("--predictions-output")
    teacher_benchmark_v2_parser.add_argument("--output")
    teacher_benchmark_v2_parser.add_argument(
        "--validate-only", action="store_true", help="only validate input/gold separation"
    )
    teacher_benchmark_v2_parser.add_argument(
        "--online", action="store_true", help="run the real DeepSeek executor"
    )
    teacher_benchmark_v2_parser.add_argument(
        "--allow-remote-benchmark-data",
        action="store_true",
        help="explicitly authorize sending benchmark learner text to DeepSeek",
    )
    teacher_benchmark_v2_parser.add_argument("--api-key-file")
    teacher_benchmark_v2_parser.add_argument(
        "--model", choices=sorted(ALLOWED_MODELS), default="deepseek-v4-flash"
    )
    teacher_benchmark_v2_parser.add_argument(
        "--acknowledge-held-out", action="store_true"
    )
    teacher_benchmark_v2_parser.set_defaults(func=command_teacher_agent_benchmark_v2)

    teacher_outcome_parser = subparsers.add_parser(
        "teacher-agent-outcome-evaluate",
        help="score paired pre/post, transfer, and optional delayed assessments",
    )
    teacher_outcome_parser.add_argument(
        "--input", default="data/teacher_agent_learning_outcome_demo.json"
    )
    teacher_outcome_parser.add_argument("--output")
    teacher_outcome_parser.set_defaults(func=command_teacher_agent_outcome_evaluate)

    teacher_demo_parser = subparsers.add_parser(
        "teacher-agent-demo",
        help="run the deterministic task-two teaching and baseline-evaluation demo",
    )
    teacher_demo_parser.add_argument(
        "--input", default="data/teacher_agent_demo_input.json"
    )
    teacher_demo_parser.add_argument(
        "--library", default="data/teacher_agent_skill_library_v2.json"
    )
    teacher_demo_parser.add_argument(
        "--cases", default="data/teacher_agent_evaluation_cases.json"
    )
    teacher_demo_parser.add_argument(
        "--output-dir", default="artifacts/private/teacher_agent_demo"
    )
    teacher_demo_parser.set_defaults(func=command_teacher_agent_demo)

    teacher_dashboard_parser = subparsers.add_parser(
        "teacher-agent-dashboard",
        help="serve the loopback-only live task-two Agent demonstration",
    )
    teacher_dashboard_parser.add_argument(
        "--input", default="data/teacher_agent_demo_input.json"
    )
    teacher_dashboard_parser.add_argument(
        "--library", default="data/teacher_agent_skill_library_v2.json"
    )
    teacher_dashboard_parser.add_argument(
        "--cases", default="data/teacher_agent_evaluation_cases.json"
    )
    teacher_dashboard_parser.add_argument("--port", type=int, default=0)
    teacher_dashboard_parser.add_argument(
        "--neural-v1-manifest", default="data/neural_v1_runtime_manifest.json"
    )
    teacher_dashboard_parser.add_argument(
        "--learning-outcome",
        default="data/teacher_agent_learning_outcome_demo.json",
    )
    teacher_dashboard_parser.add_argument(
        "--free-text-benchmark-receipt",
        default="data/teacher_agent_free_text_benchmark_receipt.json",
    )
    teacher_dashboard_parser.add_argument(
        "--agent-backend",
        choices=("deepseek", "deterministic"),
        default="deepseek",
    )
    teacher_dashboard_parser.add_argument(
        "--model", choices=sorted(ALLOWED_MODELS), default="deepseek-v4-flash"
    )
    teacher_dashboard_parser.add_argument("--api-key-file")
    teacher_dashboard_parser.add_argument(
        "--allow-remote-student-data",
        action="store_true",
        help="explicitly allow redacted learner text and necessary context to be sent to DeepSeek",
    )
    teacher_dashboard_parser.add_argument(
        "--no-rule-fallback",
        action="store_true",
        help="fail the turn instead of using the deterministic safety fallback",
    )
    teacher_dashboard_parser.add_argument("--no-browser", action="store_true")
    teacher_dashboard_parser.add_argument(
        "--session-store",
        help=(
            "opt in to cold resume by writing local learner/session state to "
            "this append-only JSONL file"
        ),
    )
    teacher_dashboard_parser.add_argument("--check", action="store_true")
    teacher_dashboard_parser.set_defaults(func=command_teacher_agent_dashboard)

    audit_parser = subparsers.add_parser("audit", help="audit dataset completeness and research readiness")
    audit_parser.add_argument("--manifest", default="data/dataset_manifest.json")
    audit_parser.add_argument("--output")
    audit_parser.add_argument(
        "--require-formal",
        action="store_true",
        help="return a failing exit status unless every transcript is formal-ready",
    )
    audit_parser.set_defaults(func=command_audit)

    formal_caption_parser = subparsers.add_parser(
        "fetch-formal-captions",
        help=(
            "download hash-pinned MIT OCW WebVTT tracks and build a private "
            "formal transcript dataset"
        ),
    )
    formal_caption_parser.add_argument(
        "--source-manifest", default="data/formal_caption_sources.json"
    )
    formal_caption_parser.add_argument(
        "--output", default="artifacts/private/formal_captions"
    )
    formal_caption_parser.add_argument(
        "--public-receipt",
        help="optional aggregate/hash-only receipt; never includes caption text",
    )
    formal_caption_parser.add_argument("--ffprobe", default="ffprobe")
    formal_caption_parser.add_argument("--timeout", type=int, default=120)
    formal_caption_parser.add_argument(
        "--acknowledge-source-terms", action="store_true"
    )
    formal_caption_parser.set_defaults(func=command_fetch_formal_captions)

    full_video_parser = subparsers.add_parser(
        "fetch-full-videos",
        help=(
            "download, hash, and ffprobe the 10 complete MIT OCW videos "
            "bound to the formal captions"
        ),
    )
    full_video_parser.add_argument(
        "--source-manifest", default="data/formal_caption_sources.json"
    )
    full_video_parser.add_argument(
        "--formal-caption-manifest",
        default="artifacts/private/formal_captions/dataset_manifest.json",
    )
    full_video_parser.add_argument(
        "--output", default="artifacts/private/full_videos"
    )
    full_video_parser.add_argument("--public-receipt")
    full_video_parser.add_argument("--curl", default="curl")
    full_video_parser.add_argument("--ffprobe", default="ffprobe")
    full_video_parser.add_argument("--connect-timeout", type=int, default=30)
    full_video_parser.add_argument("--download-timeout", type=int, default=14_400)
    full_video_parser.add_argument("--ffprobe-timeout", type=int, default=180)
    full_video_parser.add_argument(
        "--acknowledge-source-terms", action="store_true"
    )
    full_video_parser.set_defaults(func=command_fetch_full_videos)

    teachobs_parser = subparsers.add_parser(
        "fetch-teachobs",
        help=(
            "download and audit the commit-pinned TeachObs annotation release; "
            "source classroom videos remain separate"
        ),
    )
    teachobs_parser.add_argument(
        "--output",
        default="artifacts/private/external_datasets/teachobs/imported_annotations",
    )
    teachobs_parser.add_argument(
        "--public-receipt",
        help="optional aggregate/hash-only receipt without URLs, ids, text, or labels",
    )
    teachobs_parser.add_argument("--timeout", type=int, default=120)
    teachobs_parser.add_argument(
        "--acknowledge-source-terms",
        action="store_true",
        help=(
            "acknowledge CC BY 4.0 for released data and separate original "
            "terms for source videos"
        ),
    )
    teachobs_parser.set_defaults(func=command_fetch_teachobs)

    teachobs_benchmark_parser = subparsers.add_parser(
        "benchmark-teachobs-text",
        help=(
            "run the fixed transcript-only baseline on the official TeachObs "
            "23/7 lesson split"
        ),
    )
    teachobs_benchmark_parser.add_argument(
        "--repository",
        default="artifacts/private/external_datasets/teachobs/repository",
    )
    teachobs_benchmark_parser.add_argument(
        "--output",
        default=(
            "artifacts/private/external_datasets/teachobs/"
            "text_benchmark_result.json"
        ),
    )
    teachobs_benchmark_parser.add_argument(
        "--public-receipt",
        default="artifacts/public/teachobs_text_benchmark_receipt.json",
    )
    teachobs_benchmark_parser.add_argument(
        "--no-sanitized-sensitivity",
        action="store_true",
        help="skip the duplicate-speaker-turn sensitivity arm; raw stays primary",
    )
    teachobs_benchmark_parser.set_defaults(func=command_teachobs_text_benchmark)

    teachobs_caption_parser = subparsers.add_parser(
        "audit-teachobs-captions",
        aliases=["fetch-teachobs-captions"],
        help=(
            "privately fetch each TeachObs lesson's YouTube caption tracks and "
            "audit their timelines against the released 15-second text"
        ),
    )
    teachobs_caption_parser.add_argument(
        "--repository",
        default="artifacts/private/external_datasets/teachobs/repository",
    )
    teachobs_caption_parser.add_argument(
        "--media-plan",
        default="artifacts/private/external_datasets/teachobs/media/media_plan.json",
    )
    teachobs_caption_parser.add_argument(
        "--acquisition-receipt",
        default=(
            "artifacts/private/external_datasets/teachobs/"
            "acquisition_receipt.json"
        ),
    )
    teachobs_caption_parser.add_argument(
        "--output",
        default="artifacts/private/external_datasets/teachobs/captions",
    )
    teachobs_caption_parser.add_argument(
        "--public-receipt",
        help=(
            "optional aggregate-only receipt without caption text, source URLs, "
            "lesson/video ids, paths, or per-scene records"
        ),
    )
    teachobs_caption_parser.add_argument(
        "--lesson-id",
        action="append",
        help="optional repeatable S1-S30 selection; default is all planned lessons",
    )
    teachobs_caption_parser.add_argument("--jobs", type=int, default=4)
    teachobs_caption_parser.add_argument("--yt-dlp-python")
    teachobs_caption_parser.add_argument("--yt-dlp-executable")
    teachobs_caption_parser.add_argument(
        "--cookies-from-browser",
        help=(
            "explicit private browser or browser:profile-name opt-in; the "
            "selector and cookie material are never written to receipts"
        ),
    )
    teachobs_caption_parser.add_argument(
        "--yt-dlp-direct",
        action="store_true",
        help="explicitly bypass inherited HTTP(S) proxy variables",
    )
    teachobs_caption_parser.add_argument(
        "--yt-dlp-impersonate",
        choices=("chrome",),
        help="request local yt-dlp/curl_cffi Chrome HTTP impersonation",
    )
    teachobs_caption_parser.add_argument(
        "--yt-dlp-youtube-client",
        choices=("android_vr",),
        help=(
            "explicit fixed YouTube player client for both caption metadata "
            "and VTT retrieval; disabled by default"
        ),
    )
    teachobs_caption_parser.add_argument(
        "--js-runtime",
        help=(
            "optional local yt-dlp JavaScript runtime, for example "
            "node:/opt/homebrew/bin/node; remote EJS components stay disabled"
        ),
    )
    teachobs_caption_parser.add_argument("--timeout", type=int, default=600)
    teachobs_caption_parser.add_argument(
        "--acknowledge-source-terms", action="store_true"
    )
    teachobs_caption_parser.set_defaults(func=command_audit_teachobs_captions)

    teachobs_asr_prepare_parser = subparsers.add_parser(
        "prepare-teachobs-asr-handoff",
        help=(
            "create a hash-bound, content-free private GPU ASR job manifest, "
            "or publish an honest pending aggregate receipt"
        ),
    )
    teachobs_asr_prepare_parser.add_argument(
        "--media-manifest",
        default=(
            "artifacts/private/external_datasets/teachobs/media/"
            "media_manifest.json"
        ),
    )
    teachobs_asr_prepare_parser.add_argument(
        "--media-root",
        default="artifacts/private/external_datasets/teachobs/media",
    )
    teachobs_asr_prepare_parser.add_argument(
        "--caption-audit",
        default=(
            "artifacts/private/external_datasets/teachobs/captions/"
            "caption_audit.json"
        ),
    )
    teachobs_asr_prepare_parser.add_argument(
        "--output",
        default=(
            "artifacts/private/external_datasets/teachobs/asr/"
            "job_manifest.json"
        ),
    )
    teachobs_asr_prepare_parser.add_argument(
        "--public-receipt",
        default="artifacts/public/teachobs_asr_receipt.json",
    )
    teachobs_asr_prepare_parser.add_argument(
        "--pending-only",
        action="store_true",
        help="write no job manifest and truthfully record that ASR has not run",
    )
    teachobs_asr_prepare_parser.add_argument("--model-id")
    teachobs_asr_prepare_parser.add_argument("--model-revision")
    teachobs_asr_prepare_parser.add_argument("--model-files-sha256")
    teachobs_asr_prepare_parser.add_argument("--faster-whisper-version")
    teachobs_asr_prepare_parser.add_argument("--ctranslate2-version")
    teachobs_asr_prepare_parser.add_argument(
        "--container-image-digest",
        help=(
            "exact sha256:<64-hex> GPU container digest; required and frozen "
            "for non-pending handoffs"
        ),
    )
    teachobs_asr_prepare_parser.add_argument(
        "--all-lessons",
        action="store_true",
        help="schedule every selected lesson instead of caption-fallback lessons",
    )
    teachobs_asr_prepare_parser.add_argument("--lesson-id", action="append")
    teachobs_asr_prepare_parser.add_argument("--language", default="auto")
    teachobs_asr_prepare_parser.add_argument(
        "--language-map", help="private JSON mapping lesson ids to language tags"
    )
    teachobs_asr_prepare_parser.add_argument(
        "--min-timeline-span-fraction", type=float, default=0.90
    )
    teachobs_asr_prepare_parser.add_argument(
        "--max-endpoint-gap-fraction",
        type=float,
        default=0.10,
        help=(
            "symmetric maximum first/last VAD speech-anchor gap as a fraction "
            "of hash-bound full-media duration; must be no greater than "
            "1 - --min-timeline-span-fraction"
        ),
    )
    teachobs_asr_prepare_parser.add_argument(
        "--require-all-selected-media", action="store_true"
    )
    teachobs_asr_prepare_parser.set_defaults(
        func=command_prepare_teachobs_asr_handoff
    )

    teachobs_asr_import_parser = subparsers.add_parser(
        "import-teachobs-asr-results",
        help=(
            "strictly revalidate private GPU ASR JSON and build a caption-first "
            "30-lesson coverage matrix"
        ),
    )
    teachobs_asr_import_parser.add_argument("--job-manifest", required=True)
    teachobs_asr_import_parser.add_argument(
        "--media-manifest",
        default=(
            "artifacts/private/external_datasets/teachobs/media/"
            "media_manifest.json"
        ),
    )
    teachobs_asr_import_parser.add_argument(
        "--media-root",
        default="artifacts/private/external_datasets/teachobs/media",
    )
    teachobs_asr_import_parser.add_argument(
        "--caption-audit",
        default=(
            "artifacts/private/external_datasets/teachobs/captions/"
            "caption_audit.json"
        ),
    )
    teachobs_asr_import_parser.add_argument("--results", required=True)
    teachobs_asr_import_parser.add_argument(
        "--output",
        default=(
            "artifacts/private/external_datasets/teachobs/asr/"
            "import_audit.json"
        ),
    )
    teachobs_asr_import_parser.add_argument(
        "--coverage-matrix",
        default=(
            "artifacts/private/external_datasets/teachobs/asr/"
            "transcript_coverage_matrix.json"
        ),
    )
    teachobs_asr_import_parser.add_argument(
        "--public-receipt",
        default="artifacts/public/teachobs_asr_receipt.json",
    )
    teachobs_asr_import_parser.set_defaults(
        func=command_import_teachobs_asr_results
    )

    teachobs_materialize_parser = subparsers.add_parser(
        "materialize-teachobs-transcripts",
        help=(
            "materialize the fixed 29-lesson/4,945-scene TeachObs paper profile "
            "from official captions and media-bound audited ASR without "
            "released-transcript fallback"
        ),
    )
    teachobs_materialize_parser.add_argument("--media-plan", required=True)
    teachobs_materialize_parser.add_argument("--media-manifest", required=True)
    teachobs_materialize_parser.add_argument("--caption-audit", required=True)
    teachobs_materialize_parser.add_argument(
        "--asr-import-audit", required=True
    )
    teachobs_materialize_parser.add_argument(
        "--coverage-matrix", required=True
    )
    teachobs_materialize_parser.add_argument(
        "--asr-job-manifest", required=True
    )
    teachobs_materialize_parser.add_argument("--asr-results", required=True)
    teachobs_materialize_parser.add_argument("--output", required=True)
    teachobs_materialize_parser.add_argument("--public-receipt", required=True)
    teachobs_materialize_parser.set_defaults(
        func=command_materialize_teachobs_transcripts
    )

    teachobs_asr_hash_parser = subparsers.add_parser(
        "hash-teachobs-asr-model",
        help=(
            "hash a prepositioned local Whisper snapshot for an offline, "
            "content-addressed GPU handoff"
        ),
    )
    teachobs_asr_hash_parser.add_argument("--model-directory", required=True)
    teachobs_asr_hash_parser.set_defaults(
        func=command_hash_teachobs_asr_model
    )

    teachobs_asr_run_parser = subparsers.add_parser(
        "run-teachobs-asr-gpu",
        help=(
            "execute hash-bound private TeachObs ASR jobs using an already "
            "installed CUDA runtime and local model snapshot"
        ),
    )
    teachobs_asr_run_parser.add_argument("--job-manifest", required=True)
    teachobs_asr_run_parser.add_argument("--media-root", required=True)
    teachobs_asr_run_parser.add_argument("--model-directory", required=True)
    teachobs_asr_run_parser.add_argument("--output", required=True)
    teachobs_asr_run_parser.add_argument(
        "--container-image-digest",
        required=True,
        help=(
            "sha256:<64-hex> digest that must exactly match the frozen job "
            "manifest runtime contract"
        ),
    )
    teachobs_asr_run_parser.add_argument(
        "--lesson-id",
        action="append",
        help="optional repeatable subset of private jobs",
    )
    teachobs_asr_run_parser.set_defaults(func=command_run_teachobs_asr_gpu)

    teachobs_annotation_prepare_parser = subparsers.add_parser(
        "prepare-teachobs-double-annotation",
        help=(
            "generate private blind A/B TeachObs assignments; without a complete "
            "external operational codebook this remains a non-executable template"
        ),
    )
    teachobs_annotation_prepare_parser.add_argument(
        "--repository",
        default="artifacts/private/external_datasets/teachobs/repository",
    )
    teachobs_annotation_prepare_parser.add_argument(
        "--output",
        default="artifacts/private/external_datasets/teachobs/human_annotation",
    )
    teachobs_annotation_prepare_parser.add_argument(
        "--public-receipt",
        default="artifacts/public/teachobs_human_annotation_receipt.json",
    )
    teachobs_annotation_prepare_parser.add_argument(
        "--lesson-id",
        action="append",
        help="optional repeatable S1-S30 subset; default uses all 5158 scenes",
    )
    teachobs_annotation_prepare_parser.add_argument(
        "--media-root",
        default="artifacts/private/external_datasets/teachobs/media",
        help="private base directory; assignments contain only relative references",
    )
    teachobs_annotation_prepare_parser.add_argument(
        "--media-reference-prefix", default="videos"
    )
    teachobs_annotation_prepare_parser.add_argument(
        "--operational-codebook",
        help=(
            "external 39/39 operational JSON codebook; when omitted the output "
            "is a non-executable template and cannot be analyzed"
        ),
    )
    teachobs_annotation_prepare_parser.add_argument("--seed-a", type=int, default=1729)
    teachobs_annotation_prepare_parser.add_argument("--seed-b", type=int, default=2718)
    teachobs_annotation_prepare_parser.add_argument(
        "--require-media",
        action="store_true",
        help="fail unless every selected private videos/S*.mp4 exists",
    )
    teachobs_annotation_prepare_parser.set_defaults(
        func=command_prepare_teachobs_double_annotation
    )

    teachobs_annotation_analyze_parser = subparsers.add_parser(
        "analyze-teachobs-double-annotation",
        help=(
            "validate two complete independently signed TeachObs assignments and "
            "compute private agreement artifacts"
        ),
    )
    teachobs_annotation_analyze_parser.add_argument("--manifest", required=True)
    teachobs_annotation_analyze_parser.add_argument(
        "--assignment-a", required=True
    )
    teachobs_annotation_analyze_parser.add_argument(
        "--assignment-b", required=True
    )
    teachobs_annotation_analyze_parser.add_argument("--output", required=True)
    teachobs_annotation_analyze_parser.add_argument("--public-receipt")
    teachobs_annotation_analyze_parser.set_defaults(
        func=command_analyze_teachobs_double_annotation
    )

    teachobs_media_parser = subparsers.add_parser(
        "prepare-teachobs-media",
        help=(
            "privately download, hash, ffprobe, sample, OCR, and optionally "
            "CLIP-encode complete TeachObs lessons"
        ),
    )
    teachobs_media_parser.add_argument(
        "--repository",
        default="artifacts/private/external_datasets/teachobs/repository",
    )
    teachobs_media_parser.add_argument(
        "--output",
        default="artifacts/private/external_datasets/teachobs/media",
    )
    teachobs_media_parser.add_argument(
        "--lesson-id",
        action="append",
        help="optional repeatable S1-S30 selection; default is all 30 lessons",
    )
    teachobs_media_parser.add_argument("--jobs", type=int, default=4)
    teachobs_media_parser.add_argument(
        "--feature-jobs",
        type=int,
        default=2,
        help="parallel lesson feature workers (1-8; nested worker product capped at 8)",
    )
    teachobs_media_parser.add_argument(
        "--visual-jobs", type=int, default=1, help="image-metric workers per lesson (1-8)"
    )
    teachobs_media_parser.add_argument(
        "--ocr-jobs", type=int, default=1, help="OCR workers per lesson (1-8)"
    )
    teachobs_media_parser.add_argument("--yt-dlp-python")
    teachobs_media_parser.add_argument("--yt-dlp-executable")
    teachobs_media_parser.add_argument(
        "--cookies-from-browser",
        help=(
            "explicit private browser or browser:profile-name opt-in; the "
            "selector and cookie material are never written to receipts"
        ),
    )
    teachobs_media_parser.add_argument(
        "--yt-dlp-direct",
        action="store_true",
        help="explicitly bypass inherited HTTP(S) proxy variables",
    )
    teachobs_media_parser.add_argument(
        "--yt-dlp-impersonate",
        choices=("chrome",),
        help="request local yt-dlp/curl_cffi Chrome HTTP impersonation",
    )
    teachobs_media_parser.add_argument("--js-runtime")
    teachobs_media_parser.add_argument(
        "--source-override-manifest",
        help=(
            "private explicit mirror manifest; canonical repository URLs remain "
            "unchanged and every override is hash-bound"
        ),
    )
    teachobs_media_parser.add_argument("--ffmpeg", default="ffmpeg")
    teachobs_media_parser.add_argument("--ffprobe", default="ffprobe")
    teachobs_media_parser.add_argument("--no-audio-statistics", action="store_true")
    teachobs_media_parser.add_argument("--no-visual-evidence", action="store_true")
    teachobs_media_parser.add_argument("--no-ocr", action="store_true")
    teachobs_media_parser.add_argument("--ocr-language", default="eng")
    teachobs_media_parser.add_argument("--clip-model")
    teachobs_media_parser.add_argument("--clip-source-revision")
    teachobs_media_parser.add_argument("--clip-device", default="cpu")
    teachobs_media_parser.add_argument("--clip-batch-size", type=int, default=32)
    teachobs_media_parser.add_argument("--dry-run", action="store_true")
    teachobs_media_parser.add_argument(
        "--acknowledge-source-terms", action="store_true"
    )
    teachobs_media_parser.add_argument(
        "--acknowledge-override-source-terms",
        action="store_true",
        help=(
            "separately acknowledge the explicit mirror's platform/source terms; "
            "required in addition to --acknowledge-source-terms"
        ),
    )
    teachobs_media_parser.set_defaults(func=command_prepare_teachobs_media)

    teachobs_multimodal_parser = subparsers.add_parser(
        "benchmark-teachobs-multimodal",
        help=(
            "run transcript-only, +audio, +visual, and full models on the "
            "selected pinned TeachObs lesson-level evaluation profile"
        ),
    )
    teachobs_multimodal_parser.add_argument(
        "--evaluation-profile",
        choices=(
            FULL_23_TRAIN_7_TEST_PROFILE,
            PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
        ),
        default=FULL_23_TRAIN_7_TEST_PROFILE,
        help=(
            "default keeps the full official 23/7 split; the paper Track 1 "
            "profile uses the published six-test-lesson text/frame intersection"
        ),
    )
    teachobs_multimodal_parser.add_argument(
        "--repository",
        default="artifacts/private/external_datasets/teachobs/repository",
    )
    teachobs_multimodal_parser.add_argument(
        "--feature-manifest",
        default=(
            "artifacts/private/external_datasets/teachobs/media/"
            "feature_manifest.json"
        ),
    )
    teachobs_multimodal_parser.add_argument(
        "--transcript-materialization-manifest",
        required=True,
        help=(
            "required private, fully validated official-caption/audited-ASR "
            "scene materialization; released repository transcripts are never "
            "used as a fallback"
        ),
    )
    teachobs_multimodal_parser.add_argument(
        "--output",
        default=(
            "artifacts/private/external_datasets/teachobs/"
            "multimodal_benchmark_result.json"
        ),
    )
    teachobs_multimodal_parser.add_argument(
        "--public-receipt",
        default="artifacts/public/teachobs_multimodal_benchmark_receipt.json",
    )
    teachobs_multimodal_parser.add_argument(
        "--frozen-model-output",
        help=(
            "optional private directory for deterministic, non-pickle four-arm "
            "model artifacts; export does not establish deployment evidence"
        ),
    )
    teachobs_multimodal_parser.set_defaults(
        func=command_teachobs_multimodal_benchmark
    )

    teachobs_lockbox_parser = subparsers.add_parser(
        "prepare-teachobs-lockbox-preregistration",
        help=(
            "freeze a no-outcome four-arm protocol for a prospective new-site "
            "TeachObs confirmation"
        ),
    )
    teachobs_lockbox_parser.add_argument("--study-id", required=True)
    teachobs_lockbox_parser.add_argument("--output", required=True)
    teachobs_lockbox_parser.add_argument(
        "--system-artifact",
        help=(
            "serialized frozen four-arm system bundle; omit only for a pending "
            "draft"
        ),
    )
    teachobs_lockbox_parser.add_argument(
        "--analysis-code",
        help=(
            "frozen evaluator/bootstrap source or archive; omit only for a "
            "pending draft"
        ),
    )
    teachobs_lockbox_parser.add_argument(
        "--arm-model",
        action="append",
        default=[],
        metavar="ARM=PATH",
        help=(
            "repeat for transcript_only, transcript_audio, transcript_visual, "
            "and full; a partial set remains non-executable"
        ),
    )
    teachobs_lockbox_parser.add_argument(
        "--target-cluster-field",
        choices=("classroom_id", "session_id"),
        default="classroom_id",
    )
    teachobs_lockbox_parser.add_argument(
        "--expected-development-profile",
        choices=(
            FULL_23_TRAIN_7_TEST_PROFILE,
            PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
        ),
        help=(
            "fail unless the supplied frozen bundle is bound to this exact "
            "TeachObs development evaluation profile"
        ),
    )
    teachobs_lockbox_parser.set_defaults(
        func=command_prepare_teachobs_lockbox_preregistration
    )

    benchmark_parser = subparsers.add_parser("benchmark", help="run held-out cross-domain capability tests")
    benchmark_parser.add_argument("--skills", default="artifacts/skills")
    benchmark_parser.add_argument("--cases", default="data/evaluation_cases.json")
    benchmark_parser.add_argument("--output")
    benchmark_parser.set_defaults(func=command_benchmark)

    human_parser = subparsers.add_parser("human-evaluate", help="aggregate two-reviewer human validation CSV")
    human_parser.add_argument("--input", required=True)
    human_parser.add_argument(
        "--skills",
        help="directory containing the complete expected Skill set; enables 100% coverage validation",
    )
    human_parser.add_argument("--output")
    human_parser.set_defaults(func=command_human_evaluate)

    doctor_parser = subparsers.add_parser(
        "doctor", help="check packaged resources, dependencies, tools, API safety, and readiness"
    )
    doctor_parser.add_argument("--output")
    doctor_parser.set_defaults(func=command_doctor)

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="open the public aggregate evidence dashboard",
    )
    dashboard_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="materialize and print the dashboard path without opening a browser",
    )
    dashboard_parser.add_argument(
        "--output",
        help="optional HTML output path; defaults to a private random temporary file",
    )
    dashboard_parser.add_argument(
        "--check",
        action="store_true",
        help="validate the packaged dashboard and exit",
    )
    dashboard_parser.set_defaults(func=command_dashboard)

    private_dashboard_parser = subparsers.add_parser(
        "dashboard-real",
        help=(
            "show real private TeachObs behavior recognition and MIT Skill "
            "distillation outcomes locally"
        ),
    )
    private_dashboard_parser.add_argument(
        "--teachobs-root",
        type=Path,
        default=DEFAULT_PRIVATE_TEACHOBS_ROOT,
        help=(
            "private TeachObs artifact root; defaults to "
            "artifacts/private/external_datasets/teachobs"
        ),
    )
    private_dashboard_parser.add_argument(
        "--initial-lesson",
        default=DEFAULT_PRIVATE_LESSON,
        help="initial held-out lesson shown in the browser (default: S24)",
    )
    private_dashboard_parser.add_argument(
        "--skill-root",
        type=Path,
        default=DEFAULT_PRIVATE_SKILL_ROOT,
        help=(
            "private long-form Skill artifact root; defaults to "
            "artifacts/private/full_multimodal"
        ),
    )
    private_dashboard_parser.add_argument(
        "--initial-skill",
        default=DEFAULT_PRIVATE_SKILL,
        help=(
            "initial distilled Skill shown in the browser "
            "(default: linear_algebra_l03)"
        ),
    )
    private_dashboard_parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="loopback port; 0 selects an unused random port",
    )
    private_dashboard_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="serve locally and print the capability URL without opening a browser",
    )
    private_dashboard_checks = private_dashboard_parser.add_mutually_exclusive_group()
    private_dashboard_checks.add_argument(
        "--check-template",
        action="store_true",
        help="validate the generic private dashboard template without reading private data",
    )
    private_dashboard_checks.add_argument(
        "--check-data",
        action="store_true",
        help="validate private artifacts and frozen predictions, then exit without serving",
    )
    private_dashboard_parser.set_defaults(func=command_private_dashboard)

    delivery_parser = subparsers.add_parser(
        "verify-delivery",
        help="rebuild and verify the local project evidence while listing external-validation gaps",
    )
    delivery_parser.add_argument("--manifest", default="data/dataset_manifest.json")
    delivery_parser.add_argument(
        "--formal-manifest",
        help=(
            "separate complete-caption/ASR manifest used only for the formal "
            "transcript evidence gate"
        ),
    )
    delivery_parser.add_argument("--cases", default="data/evaluation_cases.json")
    delivery_parser.add_argument("--human-review")
    delivery_parser.add_argument("--dipser-report")
    delivery_parser.add_argument("--external-deployment-report")
    delivery_parser.add_argument("--external-evaluation-receipt")
    delivery_parser.add_argument("--trusted-attestation-public-key")
    delivery_parser.add_argument("--external-multimodal-evidence")
    delivery_parser.add_argument("--external-multimodal-attestation")
    delivery_parser.add_argument("--trusted-multimodal-public-key")
    delivery_parser.add_argument("--external-learner-evidence")
    delivery_parser.add_argument("--external-learner-attestation")
    delivery_parser.add_argument("--trusted-learner-public-key")
    delivery_parser.add_argument(
        "--external-evaluated-system-artifact",
        help=(
            "exact evaluated system artifact (normally the release wheel); its "
            "SHA-256 must match both external research manifests"
        ),
    )
    delivery_parser.add_argument("--output")
    delivery_parser.add_argument("--markdown")
    delivery_parser.add_argument(
        "--require-external-validation",
        action="store_true",
        help=(
            "fail unless formal transcripts, human review, confirmatory multimodal "
            "gain, lockbox deployment, and learner evidence are complete"
        ),
    )
    delivery_parser.set_defaults(func=command_verify_delivery)

    release_parser = subparsers.add_parser(
        "release-audit", help="fail closed when a public wheel/zip/directory contains media, secrets, or row-level identities"
    )
    release_parser.add_argument("path")
    release_parser.add_argument("--output")
    release_parser.set_defaults(func=command_release_audit)

    public_report_parser = subparsers.add_parser(
        "export-public-dipser",
        help="export an aggregate-only DIPSER report without samples, identities, paths, or features",
    )
    public_report_parser.add_argument("--input", required=True)
    public_report_parser.add_argument("--output", required=True)
    public_report_parser.set_defaults(func=command_export_public_dipser)

    raw_features_parser = subparsers.add_parser(
        "extract-strict-features",
        help=(
            "generate a content-addressed strict feature bundle from private raw "
            "media/sensor inputs without making predictions"
        ),
    )
    raw_features_parser.add_argument("--manifest", required=True)
    raw_features_parser.add_argument("--raw-root", required=True)
    raw_features_parser.add_argument("--extractor-config", required=True)
    raw_features_parser.add_argument(
        "--identity-field",
        action="append",
        help="verified identity field required by the intended protocol; repeat as needed",
    )
    raw_features_parser.add_argument(
        "--model",
        help="optional frozen v3 checkpoint; verifies extractor/schema compatibility but does not predict",
    )
    raw_features_parser.add_argument(
        "--output-dir",
        required=True,
        help="private artifact directory (created or restricted to mode 0700 on POSIX)",
    )
    raw_features_parser.set_defaults(func=command_extract_strict_features)

    freeze_parser = subparsers.add_parser(
        "freeze-recognition-model",
        help="fit once on source-only strict data and freeze identity, feature, and claim contracts",
    )
    freeze_parser.add_argument("--manifest", required=True)
    freeze_parser.add_argument("--features", required=True)
    freeze_parser.add_argument("--claim-contract", required=True)
    freeze_parser.add_argument("--class-name", action="append", required=True)
    freeze_parser.add_argument("--identity-field", action="append")
    freeze_parser.add_argument("--modality", default="fusion")
    freeze_parser.add_argument("--group-field", default="session_id")
    freeze_parser.add_argument("--claim-cluster-field", default="session_id")
    freeze_parser.add_argument("--inner-splits", type=int, default=5)
    freeze_parser.add_argument(
        "--c-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0]
    )
    freeze_parser.add_argument("--seed", type=int, default=2026)
    freeze_parser.add_argument("--output", required=True)
    freeze_parser.set_defaults(func=command_freeze_recognition_model)

    registration_parser = subparsers.add_parser(
        "create-freeze-registration",
        help="bind a frozen model and exact external lockbox into a request for independent signature",
    )
    registration_parser.add_argument("--model", required=True)
    registration_parser.add_argument("--manifest", required=True)
    registration_parser.add_argument("--features", required=True)
    registration_parser.add_argument("--registration-id", required=True)
    registration_parser.add_argument("--output", required=True)
    registration_parser.set_defaults(func=command_create_freeze_registration)

    sign_registration_parser = subparsers.add_parser(
        "sign-freeze-registration",
        help="sign a lockbox registration request with an independent Ed25519 custodian key",
    )
    sign_registration_parser.add_argument("--request", required=True)
    sign_registration_parser.add_argument("--private-key", required=True)
    sign_registration_parser.add_argument("--issuer", required=True)
    sign_registration_parser.add_argument("--key-id", required=True)
    sign_registration_parser.add_argument("--output", required=True)
    sign_registration_parser.set_defaults(func=command_sign_freeze_registration)

    sign_evaluation_parser = subparsers.add_parser(
        "sign-evaluation-report",
        help="sign a completed registered lockbox report for independent delivery verification",
    )
    sign_evaluation_parser.add_argument("--report", required=True)
    sign_evaluation_parser.add_argument("--private-key", required=True)
    sign_evaluation_parser.add_argument("--issuer", required=True)
    sign_evaluation_parser.add_argument("--key-id", required=True)
    sign_evaluation_parser.add_argument("--output", required=True)
    sign_evaluation_parser.set_defaults(func=command_sign_evaluation_report)

    learner_study_parser = subparsers.add_parser(
        "prepare-learner-effect-study",
        help=(
            "create private cluster-randomized preregistration, token-only "
            "allocation, and blinded outcome templates"
        ),
    )
    learner_study_parser.add_argument("--output-dir", required=True)
    learner_study_parser.add_argument("--study-id", required=True)
    learner_study_parser.add_argument(
        "--cluster-unit", choices=("teacher", "classroom"), default="classroom"
    )
    learner_study_parser.add_argument("--cluster-count", type=int, default=6)
    learner_study_parser.add_argument(
        "--participants-per-cluster", type=int, default=5
    )
    learner_study_parser.add_argument(
        "--randomization-seed", type=int, default=20260723
    )
    learner_study_parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    learner_study_parser.add_argument(
        "--bootstrap-replicates", type=int, default=2000
    )
    learner_study_parser.add_argument(
        "--minimum-cluster-count", type=int, default=6
    )
    learner_study_parser.add_argument(
        "--minimum-primary-coverage", type=float, default=0.8
    )
    learner_study_parser.add_argument(
        "--maximum-attrition-fraction", type=float, default=0.2
    )
    learner_study_parser.add_argument(
        "--minimum-meaningful-effect", type=float, default=0.1
    )
    learner_study_parser.add_argument("--score-minimum", type=float, default=0.0)
    learner_study_parser.add_argument("--score-maximum", type=float, default=100.0)
    learner_study_parser.add_argument(
        "--intervention-description",
        default="teaching-skill-assisted instruction",
    )
    learner_study_parser.add_argument(
        "--comparator-description",
        default="preregistered active teaching comparator",
    )
    learner_study_parser.add_argument(
        "--data-origin", choices=("template", "synthetic", "real"), default="template"
    )
    learner_study_parser.add_argument("--ethics-approval-id", default="")
    learner_study_parser.add_argument(
        "--informed-consent-or-approved-waiver", action="store_true"
    )
    learner_study_parser.add_argument(
        "--preregistration-frozen-before-allocation", action="store_true"
    )
    learner_study_parser.add_argument(
        "--allocation-concealment-procedure-declared", action="store_true"
    )
    learner_study_parser.set_defaults(func=command_prepare_learner_effect_study)

    learner_analysis_parser = subparsers.add_parser(
        "analyze-learner-effect-study",
        help=(
            "fail-closed ITT ANCOVA and cluster-bootstrap analysis of completed "
            "token-only learner-study tables"
        ),
    )
    learner_analysis_parser.add_argument("--preregistration", required=True)
    learner_analysis_parser.add_argument("--allocation", required=True)
    learner_analysis_parser.add_argument("--outcomes", required=True)
    learner_analysis_parser.add_argument("--output", required=True)
    learner_analysis_parser.set_defaults(func=command_analyze_learner_effect_study)

    prepare_external_evidence_parser = subparsers.add_parser(
        "prepare-external-research-evidence",
        help="validate an aggregate external study manifest and add its canonical fingerprint",
    )
    prepare_external_evidence_parser.add_argument("--input", required=True)
    prepare_external_evidence_parser.add_argument("--output", required=True)
    prepare_external_evidence_parser.set_defaults(
        func=command_prepare_external_research_evidence
    )

    sign_external_evidence_parser = subparsers.add_parser(
        "sign-external-research-evidence",
        help="sign a validated external multimodal-gain or learner-effectiveness manifest",
    )
    sign_external_evidence_parser.add_argument("--evidence", required=True)
    sign_external_evidence_parser.add_argument("--private-key", required=True)
    sign_external_evidence_parser.add_argument("--issuer", required=True)
    sign_external_evidence_parser.add_argument("--key-id", required=True)
    sign_external_evidence_parser.add_argument("--output", required=True)
    sign_external_evidence_parser.set_defaults(
        func=command_sign_external_research_evidence
    )

    verify_external_evidence_parser = subparsers.add_parser(
        "verify-external-research-evidence",
        help="verify an exact signed evidence manifest against an out-of-band trusted key",
    )
    verify_external_evidence_parser.add_argument("--evidence", required=True)
    verify_external_evidence_parser.add_argument("--attestation", required=True)
    verify_external_evidence_parser.add_argument(
        "--trusted-public-key", required=True
    )
    verify_external_evidence_parser.add_argument(
        "--expected-system-artifact",
        required=True,
        help="exact local system artifact whose SHA-256 the signed evidence must bind",
    )
    verify_external_evidence_parser.add_argument(
        "--expected-kind", choices=EVIDENCE_KINDS
    )
    verify_external_evidence_parser.add_argument(
        "--require-claim-established", action="store_true"
    )
    verify_external_evidence_parser.add_argument("--output")
    verify_external_evidence_parser.set_defaults(
        func=command_verify_external_research_evidence
    )

    frozen_eval_parser = subparsers.add_parser(
        "evaluate-frozen-recognition",
        help="evaluate a fingerprinted model once on an identity-disjoint external lockbox",
    )
    frozen_eval_parser.add_argument("--model", required=True)
    frozen_eval_parser.add_argument("--manifest", required=True)
    frozen_eval_parser.add_argument("--features", required=True)
    frozen_eval_parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    frozen_eval_parser.add_argument("--seed", type=int, default=2026)
    frozen_eval_parser.add_argument("--registration-attestation")
    frozen_eval_parser.add_argument("--trusted-public-key")
    frozen_eval_parser.add_argument("--one-time-ledger")
    frozen_eval_parser.add_argument("--output", required=True)
    frozen_eval_parser.set_defaults(func=command_evaluate_frozen_recognition)

    multimodal_parser = subparsers.add_parser("multimodal", help="align transcript with audio pauses, keyframes, OCR, and classroom observations")
    multimodal_parser.add_argument("--video", required=True)
    multimodal_parser.add_argument("--transcript", required=True)
    multimodal_parser.add_argument("--artifacts-dir", required=True)
    multimodal_parser.add_argument("--observations", help="optional anonymized classroom observation JSON")
    multimodal_parser.add_argument("--no-ocr", action="store_true")
    multimodal_parser.add_argument("--frame-interval", type=float, default=30.0)
    multimodal_parser.add_argument("--max-frames", type=int, default=48)
    multimodal_parser.add_argument("--output", required=True)
    multimodal_parser.set_defaults(func=command_multimodal)

    longform_parser = subparsers.add_parser(
        "multimodal-longform-dataset",
        help=(
            "process complete videos with resumable full-timeline chunks, OCR, "
            "audio alignment, and visual events"
        ),
    )
    longform_parser.add_argument("--media-manifest", required=True)
    longform_parser.add_argument("--transcript-manifest", required=True)
    longform_parser.add_argument("--output", required=True)
    longform_parser.add_argument("--chunk-seconds", type=float, default=300.0)
    longform_parser.add_argument("--overlap-seconds", type=float, default=2.0)
    longform_parser.add_argument("--frame-interval", type=float, default=15.0)
    longform_parser.add_argument("--scene-threshold", type=float, default=0.32)
    longform_parser.add_argument(
        "--max-scenes-per-chunk", type=int, default=12
    )
    longform_parser.add_argument("--ocr-workers", type=int, default=4)
    longform_parser.add_argument("--no-ocr", action="store_true")
    longform_parser.add_argument("--no-resume", action="store_true")
    longform_parser.set_defaults(func=command_longform_multimodal_dataset)

    ablation_parser = subparsers.add_parser(
        "multimodal-ablation",
        help=(
            "run paired transcript-only, +audio, +visual, and full internal "
            "pipeline ablations without claiming recognition accuracy"
        ),
    )
    ablation_parser.add_argument("--manifest", required=True)
    ablation_parser.add_argument("--output", required=True)
    ablation_parser.set_defaults(func=command_multimodal_ablation)

    visual_extract_parser = subparsers.add_parser(
        "visual-semantic-extract",
        help=(
            "compute hash-bound CLIP embeddings and closed-ontology relative "
            "prompt scores for one lecture"
        ),
    )
    visual_extract_parser.add_argument("--tasks", type=Path, required=True)
    visual_extract_parser.add_argument("--frame-root", type=Path, required=True)
    visual_extract_parser.add_argument("--output", type=Path, required=True)
    visual_extract_parser.add_argument(
        "--model", default="openai/clip-vit-base-patch32"
    )
    visual_extract_parser.add_argument("--revision")
    visual_extract_parser.add_argument("--source-model-id")
    visual_extract_parser.add_argument("--source-revision")
    visual_extract_parser.add_argument("--device", default="cuda:0")
    visual_extract_parser.add_argument("--batch-size", type=int, default=64)
    visual_extract_parser.set_defaults(func=command_visual_semantic_extract)

    visual_dataset_parser = subparsers.add_parser(
        "visual-semantic-dataset",
        help="run hash-bound CLIP extraction for every lecture in a dataset",
    )
    visual_dataset_parser.add_argument("--manifest", type=Path, required=True)
    visual_dataset_parser.add_argument("--model", type=Path, required=True)
    visual_dataset_parser.add_argument("--output-dir", type=Path, required=True)
    visual_dataset_parser.add_argument(
        "--source-model-id", default="openai/clip-vit-base-patch32"
    )
    visual_dataset_parser.add_argument("--source-revision", required=True)
    visual_dataset_parser.add_argument("--device", default="cpu")
    visual_dataset_parser.add_argument("--batch-size", type=int, default=16)
    visual_dataset_parser.set_defaults(func=command_visual_semantic_dataset)

    visual_apply_parser = subparsers.add_parser(
        "visual-semantic-apply",
        help="attach hash-matched visual-semantic results to a longform dataset",
    )
    visual_apply_parser.add_argument("--manifest", type=Path, required=True)
    visual_apply_parser.add_argument(
        "--semantic-results-dir", type=Path, required=True
    )
    visual_apply_parser.add_argument(
        "--output-manifest", type=Path, required=True
    )
    visual_apply_parser.set_defaults(func=command_visual_semantic_apply)

    multimodal_benchmark_parser = subparsers.add_parser(
        "multimodal-benchmark",
        help="score a labeled synthetic fixture without claiming real-world accuracy",
    )
    multimodal_benchmark_parser.add_argument("--transcript", required=True)
    multimodal_benchmark_parser.add_argument("--ground-truth", required=True)
    multimodal_benchmark_parser.add_argument("--output")
    multimodal_benchmark_parser.set_defaults(func=command_multimodal_benchmark)

    real_audit_parser = subparsers.add_parser(
        "real-data-audit",
        help="audit local OUC-CGE real-classroom videos and remove exact duplicates from the manifest",
    )
    real_audit_parser.add_argument("--dataset-root", required=True)
    real_audit_parser.add_argument("--output")
    real_audit_parser.set_defaults(func=command_real_data_audit)

    real_benchmark_parser = subparsers.add_parser(
        "real-recognition-benchmark",
        help="run filename-surrogate out-of-fold visual/audio/fusion evaluation with provenance checks",
    )
    real_benchmark_parser.add_argument("--dataset-root", required=True)
    real_benchmark_parser.add_argument("--output-dir", required=True)
    real_benchmark_parser.add_argument("--folds", type=int, default=4)
    real_benchmark_parser.add_argument("--frames", type=int, default=12)
    real_benchmark_parser.add_argument("--seed", type=int, default=2026)
    real_benchmark_parser.set_defaults(func=command_real_recognition_benchmark)

    real_infer_parser = subparsers.add_parser(
        "real-recognition-infer",
        help="predict OUC-CGE group engagement for a classroom video with a trained checkpoint",
    )
    real_infer_parser.add_argument("--video", required=True)
    real_infer_parser.add_argument("--checkpoint", required=True)
    real_infer_parser.add_argument("--output")
    real_infer_parser.set_defaults(func=command_real_recognition_infer)

    dipser_parser = subparsers.add_parser(
        "dipser-credible-benchmark",
        help="run the analysis-frozen real-classroom pose+watch descriptive evaluation",
    )
    dipser_parser.add_argument("--output-dir", required=True)
    dipser_parser.add_argument(
        "--archive",
        action="append",
        help="official group_XX/experiment_YY/subject_ZZ.zip; repeat to override the 52-archive roster",
    )
    dipser_parser.add_argument("--interval-seconds", type=float, default=15.0)
    dipser_parser.add_argument("--alignment-tolerance-seconds", type=float, default=0.6)
    dipser_parser.add_argument("--watch-filename-tolerance-seconds", type=float, default=1.0)
    dipser_parser.add_argument("--watch-internal-tolerance-seconds", type=float, default=1.0)
    dipser_parser.add_argument("--workers", type=int, default=1)
    dipser_parser.add_argument("--range-timeout-seconds", type=float, default=120.0)
    dipser_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore and do not write integrity-checked per-archive feature checkpoints",
    )
    dipser_parser.add_argument("--outer-splits", type=int, default=5)
    dipser_parser.add_argument("--inner-splits", type=int, default=3)
    dipser_parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    dipser_parser.add_argument("--permutation-replicates", type=int, default=5000)
    dipser_parser.add_argument("--seed", type=int, default=2026)
    dipser_parser.set_defaults(func=command_dipser_credible_benchmark)

    pipeline_parser = subparsers.add_parser("pipeline", help="one command: input -> skill -> teaching -> evaluation")
    pipeline_parser.add_argument("input")
    pipeline_parser.add_argument("--video-id")
    pipeline_parser.add_argument("--course-id")
    pipeline_parser.add_argument("--title")
    pipeline_parser.add_argument("--source-url")
    pipeline_parser.add_argument("--language", default="en")
    pipeline_parser.add_argument(
        "--transcript",
        help="optional JSON/SRT/VTT/TXT transcript for a video input; bypasses ASR while retaining real FFmpeg/OCR analysis",
    )
    pipeline_parser.add_argument("--backend", choices=("heuristic", "api"), default="heuristic")
    pipeline_parser.add_argument("--concept", required=True)
    pipeline_parser.add_argument("--learner-level", default="beginner")
    pipeline_parser.add_argument("--observations", help="optional anonymized classroom observation JSON")
    pipeline_parser.add_argument("--no-multimodal", action="store_true", help="disable audio/visual analysis for video inputs")
    pipeline_parser.add_argument("--no-ocr", action="store_true")
    pipeline_parser.add_argument("--frame-interval", type=float, default=30.0)
    pipeline_parser.add_argument("--max-frames", type=int, default=48)
    pipeline_parser.add_argument("--output", required=True)
    pipeline_parser.set_defaults(func=command_pipeline)

    demo_parser = subparsers.add_parser("demo", help="run the 10-video offline demonstration")
    demo_parser.add_argument("--manifest")
    demo_parser.add_argument("--output", default="artifacts")
    demo_parser.add_argument("--backend", choices=("heuristic", "api"), default="heuristic")
    demo_parser.add_argument("--demo-video-id", default="python_l03")
    demo_parser.add_argument("--concept", default="二分查找")
    demo_parser.add_argument("--learner-level", default="beginner")
    demo_parser.set_defaults(func=command_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
