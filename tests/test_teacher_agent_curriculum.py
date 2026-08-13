from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from tempfile import TemporaryDirectory

import jsonschema
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from teaching_skill_miner.io_utils import project_root, read_json
from teaching_skill_miner.teacher_agent_curriculum import (
    CurriculumBlueprintError,
    TeachingCurriculumBlueprintStore,
    _seal_blueprint,
    build_teacher_owned_curriculum_spec,
    create_teacher_curriculum_authority_receipt,
    derive_generated_curriculum_blueprint,
    seal_teacher_owned_curriculum_blueprint,
    validate_curriculum_blueprint,
    verify_teacher_curriculum_runtime_authority,
)
from teaching_skill_miner.teacher_agent_syllabus import (
    TeachingSyllabusStore,
    _seal_generated_syllabus,
    syllabus_lesson_start_payload,
)


_DOMAIN_CASES = read_json(
    project_root() / "tests/fixtures/teacher_curriculum_domains_v1.json"
)["cases"]

_TEACHER_KEY_ID = "teacher-key-2026"
_TEACHER_SIGNING_KEY = Ed25519PrivateKey.generate()
_TEACHER_PRIVATE_PEM = _TEACHER_SIGNING_KEY.private_bytes(
    serialization.Encoding.PEM,
    serialization.PrivateFormat.PKCS8,
    serialization.NoEncryption(),
)
_TRUSTED_TEACHER_KEYS = {
    _TEACHER_KEY_ID: _TEACHER_SIGNING_KEY.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
}


def _generated_draft() -> dict:
    return {
        "title": "动态规划基础",
        "description": "从状态定义走向最小递推与迁移。",
        "audience": "算法初学者",
        "estimated_duration_minutes": 60,
        "learning_objectives": ["区分状态、转移和边界条件"],
        "prerequisites": [],
        "modules": [
            {
                "title": "最小递推",
                "description": "建立有向的知识依赖。",
                "lessons": [
                    {
                        "title": "状态定义",
                        "objective": "解释状态如何表示一个子问题。",
                        "summary": "先确定子问题和已知边界。",
                        "duration_minutes": 30,
                        "knowledge_components": ["状态定义"],
                        "materials": {
                            "example": "观察一个最小硬币情境。",
                            "practice": "标出子问题。",
                            "transfer_task": "把表示方法迁移到台阶问题。",
                        },
                    },
                    {
                        "title": "递推关系",
                        "objective": "构建从更小状态到当前状态的最小递推。",
                        "summary": "比较所有可行前驱并取最小值。",
                        "duration_minutes": 30,
                        "knowledge_components": ["递推关系"],
                        "materials": {
                            "example": "观察前驱状态如何产生候选值。",
                            "practice": "列出一个状态的候选前驱。",
                            "transfer_task": "迁移到不同面额集合。",
                        },
                    },
                ],
            }
        ],
    }


def _generated_blueprint(draft: dict | None = None) -> dict:
    syllabus = _seal_generated_syllabus(
        draft or _generated_draft(),
        model="deepseek-v4-flash",
        source_resource_ids=[],
        created_at="2026-08-12T01:00:00Z",
    )
    return derive_generated_curriculum_blueprint(syllabus)


def _source(resource_id: str = "teacher_source") -> dict:
    return {
        "resource_id": resource_id,
        "content_sha256": sha256(f"{resource_id}:document".encode()).hexdigest(),
        "excerpt_sha256": sha256(f"{resource_id}:excerpt".encode()).hexdigest(),
        "locator": {"kind": "page", "start": 2, "end": 2},
    }


def _teacher_spec(
    *,
    title: str = "光合作用",
    first_objective: str = "识别光合作用的反应输入。",
    second_objective: str = "解释光合作用输入与产物之间的关系。",
    first_label: str = "反应输入",
    second_label: str = "反应产物",
) -> dict:
    return build_teacher_owned_curriculum_spec(
        title=title,
        lessons=[
            {
                "title": f"{first_label}课",
                "objective": first_objective,
                "knowledge_components": [
                    {
                        "label": first_label,
                        "prerequisites": [],
                        "source_resource_ids": ["teacher_source"],
                    }
                ],
            },
            {
                "title": f"{second_label}课",
                "objective": second_objective,
                "knowledge_components": [
                    {
                        "label": second_label,
                        "prerequisites": [first_label],
                        "source_resource_ids": ["teacher_source"],
                    }
                ],
            },
        ],
        source_spans=[_source()],
        factual_claims=[
            {
                "statement": "植物利用光能把二氧化碳和水转化为有机物。",
                "knowledge_components": [first_label, second_label],
                "source_resource_ids": ["teacher_source"],
            }
        ],
    )


def _receipt(spec: dict) -> dict:
    return create_teacher_curriculum_authority_receipt(
        spec,
        teacher_id_hash=sha256(b"teacher-7").hexdigest(),
        reviewed_at="2026-08-12T02:00:00Z",
        teacher_confirmed_authority=True,
        signing_key_id=_TEACHER_KEY_ID,
        private_key_pem=_TEACHER_PRIVATE_PEM,
    )


def _seal_teacher(spec: dict, receipt: dict | None = None) -> dict:
    return seal_teacher_owned_curriculum_blueprint(
        spec,
        receipt or _receipt(spec),
        trusted_teacher_public_keys=_TRUSTED_TEACHER_KEYS,
    )


def _material(blueprint: dict) -> dict:
    value = deepcopy(blueprint)
    value.pop("curriculum_id")
    value.pop("integrity")
    return value


def test_generated_projection_has_stable_graph_and_complete_non_authoritative_blueprint() -> (
    None
):
    blueprint = _generated_blueprint()
    validate_curriculum_blueprint(blueprint)

    assert blueprint["authority"] == {
        "status": "generated_unvalidated",
        "authority": False,
        "authoritative_for_runtime_grading": False,
        "receipt": None,
    }
    assert blueprint["factual_claims"] == []
    assert blueprint["source_spans"] == []
    assert all(not row["authority"] for row in blueprint["rubrics"])
    assert all(not row["assessment_eligible"] for row in blueprint["item_blueprints"])
    assert len(blueprint["topological_order"]) == len(blueprint["knowledge_components"])
    for objective in blueprint["objectives"]:
        assert len(objective["item_blueprint_ids"]) >= 2 * len(objective["kc_ids"])
        assert objective["rubric_ids"]
        assert objective["remediation_branch_ids"]
        assert objective["delayed_review_ids"]


def test_lesson_objective_and_kc_ids_survive_reordering() -> None:
    first = _generated_blueprint()
    reordered = _generated_draft()
    reordered["modules"][0]["lessons"].reverse()
    second = _generated_blueprint(reordered)

    assert {row["lesson_id"] for row in first["lessons"]} == {
        row["lesson_id"] for row in second["lessons"]
    }
    assert {row["objective_id"] for row in first["objectives"]} == {
        row["objective_id"] for row in second["objectives"]
    }
    assert {row["kc_id"] for row in first["knowledge_components"]} == {
        row["kc_id"] for row in second["knowledge_components"]
    }


@pytest.mark.parametrize(
    "case",
    _DOMAIN_CASES,
    ids=[case["case_id"] for case in _DOMAIN_CASES],
)
def test_teacher_owned_cross_domain_specs_require_content_bound_receipt(
    case: dict,
) -> None:
    spec = _teacher_spec(
        title=case["title"],
        first_objective=case["first_objective"],
        second_objective=case["second_objective"],
        first_label=case["first_label"],
        second_label=case["second_label"],
    )
    blueprint = _seal_teacher(spec)

    validate_curriculum_blueprint(
        blueprint, trusted_teacher_public_keys=_TRUSTED_TEACHER_KEYS
    )
    assert blueprint["authority"]["authoritative_for_runtime_grading"] is True
    assert blueprint["authority"]["receipt"]["identity_assurance"] == (
        "authenticated_teacher"
    )
    assert (
        verify_teacher_curriculum_runtime_authority(
            blueprint, trusted_teacher_public_keys=_TRUSTED_TEACHER_KEYS
        )["authoritative_for_runtime_grading"]
        is True
    )
    assert all(row["authority"] for row in blueprint["source_spans"])
    assert all(row["assessment_eligible"] for row in blueprint["item_blueprints"])


def test_blueprint_validates_against_dedicated_json_schema() -> None:
    schema = read_json(
        project_root() / "schema/teaching_curriculum_blueprint.schema.json"
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(_generated_blueprint())
    teacher = _teacher_spec()
    jsonschema.Draft202012Validator(schema).validate(_seal_teacher(teacher))


def test_explicit_confirmation_and_exact_spec_hash_are_required() -> None:
    spec = _teacher_spec()
    with pytest.raises(CurriculumBlueprintError, match="explicit teacher"):
        create_teacher_curriculum_authority_receipt(
            spec,
            teacher_id_hash=sha256(b"teacher-7").hexdigest(),
            reviewed_at="2026-08-12T02:00:00Z",
            teacher_confirmed_authority=False,
            signing_key_id=_TEACHER_KEY_ID,
            private_key_pem=_TEACHER_PRIVATE_PEM,
        )

    receipt = _receipt(spec)
    changed = deepcopy(spec)
    changed["title"] = "changed after review"
    with pytest.raises(CurriculumBlueprintError, match="does not bind"):
        _seal_teacher(changed, receipt)


def test_teacher_authority_requires_a_deployment_trust_root() -> None:
    spec = _teacher_spec()
    receipt = _receipt(spec)
    blueprint = _seal_teacher(spec, receipt)
    with pytest.raises(CurriculumBlueprintError, match="trusted teacher keys"):
        validate_curriculum_blueprint(blueprint)
    with pytest.raises(CurriculumBlueprintError, match="not trusted"):
        seal_teacher_owned_curriculum_blueprint(
            spec,
            receipt,
            trusted_teacher_public_keys={},
        )

    attacker = Ed25519PrivateKey.generate()
    attacker_pem = attacker.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    forged = create_teacher_curriculum_authority_receipt(
        spec,
        teacher_id_hash=sha256(b"attacker").hexdigest(),
        reviewed_at="2026-08-12T02:00:00Z",
        teacher_confirmed_authority=True,
        signing_key_id=_TEACHER_KEY_ID,
        private_key_pem=attacker_pem,
    )
    with pytest.raises(CurriculumBlueprintError, match="does not match"):
        seal_teacher_owned_curriculum_blueprint(
            spec,
            forged,
            trusted_teacher_public_keys=_TRUSTED_TEACHER_KEYS,
        )


def test_authoritative_store_revalidates_trust_on_every_read(tmp_path) -> None:
    spec = _teacher_spec()
    blueprint = _seal_teacher(spec)
    untrusted = TeachingCurriculumBlueprintStore(tmp_path / "untrusted")
    with pytest.raises(CurriculumBlueprintError, match="trusted teacher keys"):
        untrusted.save(blueprint)

    trusted_root = tmp_path / "trusted"
    trusted = TeachingCurriculumBlueprintStore(
        trusted_root,
        trusted_teacher_public_keys=_TRUSTED_TEACHER_KEYS,
    )
    assert trusted.save(blueprint) is True
    assert trusted.read(blueprint["curriculum_id"]) == blueprint
    reopened_without_trust = TeachingCurriculumBlueprintStore(trusted_root)
    with pytest.raises(CurriculumBlueprintError, match="trusted teacher keys"):
        reopened_without_trust.read(blueprint["curriculum_id"])


def test_cycle_and_prerequisite_inversion_fail_closed() -> None:
    with pytest.raises(CurriculumBlueprintError, match="cycle"):
        build_teacher_owned_curriculum_spec(
            title="循环攻击",
            lessons=[
                {
                    "title": "A",
                    "objective": "识别知识 A。",
                    "knowledge_components": [
                        {
                            "label": "A",
                            "prerequisites": ["B"],
                            "source_resource_ids": ["teacher_source"],
                        }
                    ],
                },
                {
                    "title": "B",
                    "objective": "解释知识 B。",
                    "knowledge_components": [
                        {
                            "label": "B",
                            "prerequisites": ["A"],
                            "source_resource_ids": ["teacher_source"],
                        }
                    ],
                },
            ],
            source_spans=[_source()],
        )

    inverted = build_teacher_owned_curriculum_spec(
        title="逆序攻击",
        lessons=[
            {
                "title": "依赖项先教",
                "objective": "解释知识 B。",
                "knowledge_components": [
                    {
                        "label": "B",
                        "prerequisites": ["A"],
                        "source_resource_ids": ["teacher_source"],
                    }
                ],
            },
            {
                "title": "先修项后教",
                "objective": "识别知识 A。",
                "knowledge_components": [
                    {
                        "label": "A",
                        "prerequisites": [],
                        "source_resource_ids": ["teacher_source"],
                    }
                ],
            },
        ],
        source_spans=[_source()],
    )
    with pytest.raises(CurriculumBlueprintError, match="earlier lesson"):
        _seal_teacher(inverted)


def test_dangling_unmeasurable_and_unsupported_fact_attacks_fail_closed() -> None:
    with pytest.raises(CurriculumBlueprintError, match="not measurable"):
        build_teacher_owned_curriculum_spec(
            title="不可测目标",
            lessons=[
                {
                    "title": "模糊目标",
                    "objective": "理解并掌握这个主题。",
                    "knowledge_components": [
                        {
                            "label": "主题",
                            "prerequisites": [],
                            "source_resource_ids": ["teacher_source"],
                        }
                    ],
                }
            ],
            source_spans=[_source()],
        )

    with pytest.raises(CurriculumBlueprintError, match="requires an authoritative"):
        build_teacher_owned_curriculum_spec(
            title="事实无来源",
            lessons=[
                {
                    "title": "事实",
                    "objective": "解释一个可核验事实。",
                    "knowledge_components": [
                        {
                            "label": "事实",
                            "prerequisites": [],
                            "source_resource_ids": ["teacher_source"],
                        }
                    ],
                }
            ],
            source_spans=[_source()],
            factual_claims=[
                {
                    "statement": "这是一个没有来源的事实。",
                    "knowledge_components": ["事实"],
                }
            ],
        )

    spec = _teacher_spec()
    spec["objectives"][0]["kc_ids"] = ["kc_dangling"]
    receipt = _receipt(spec)
    with pytest.raises(CurriculumBlueprintError, match="dangling"):
        _seal_teacher(spec, receipt)


def test_missing_rubric_or_second_item_blueprint_fails_closed() -> None:
    missing_rubric = _teacher_spec()
    removed = missing_rubric["rubrics"].pop(0)
    missing_rubric["objectives"][0]["rubric_ids"].remove(removed["rubric_id"])
    missing_rubric["knowledge_components"][0]["rubric_ids"].remove(removed["rubric_id"])
    with pytest.raises(CurriculumBlueprintError, match="rubric"):
        _seal_teacher(missing_rubric)

    missing_item = _teacher_spec()
    target_objective = missing_item["objectives"][0]["objective_id"]
    target_kc = missing_item["objectives"][0]["kc_ids"][0]
    candidates = [
        row
        for row in missing_item["item_blueprints"]
        if row["objective_id"] == target_objective and row["kc_id"] == target_kc
    ]
    removed_item = candidates[0]
    missing_item["item_blueprints"].remove(removed_item)
    missing_item["objectives"][0]["item_blueprint_ids"].remove(
        removed_item["item_blueprint_id"]
    )
    missing_item["knowledge_components"][0]["item_blueprint_ids"].remove(
        removed_item["item_blueprint_id"]
    )
    with pytest.raises(CurriculumBlueprintError, match="item_blueprint|at least two"):
        _seal_teacher(missing_item)


def test_generated_text_cannot_be_promoted_by_flipping_authority_flags() -> None:
    generated = _material(_generated_blueprint())
    generated["authority"] = {
        "status": "teacher_owned_authoritative",
        "authority": True,
        "authoritative_for_runtime_grading": True,
        "receipt": _receipt(_teacher_spec()),
    }
    for row in generated["rubrics"] + generated["item_blueprints"]:
        row["authority"] = True
        row["assessment_eligible"] = True
    with pytest.raises(
        CurriculumBlueprintError, match="generated curriculum authority"
    ):
        _seal_blueprint(generated)


def test_legacy_projection_is_pure_and_does_not_fabricate_authority() -> None:
    syllabus = _seal_generated_syllabus(
        _generated_draft(),
        model="deepseek-v4-flash",
        source_resource_ids=[],
        created_at="2026-08-12T01:00:00Z",
    )
    before = deepcopy(syllabus)

    projection = derive_generated_curriculum_blueprint(syllabus)

    assert syllabus == before
    assert projection["origin"]["kind"] == "legacy_generated_unvalidated"
    assert projection["origin"]["source_id"] == syllabus["syllabus_id"]
    assert projection["authority"]["authority"] is False


def test_syllabus_store_exposes_projection_and_blueprint_store_persists_it() -> None:
    syllabus = _seal_generated_syllabus(
        _generated_draft(),
        model="deepseek-v4-flash",
        source_resource_ids=[],
        created_at="2026-08-12T01:00:00Z",
    )
    with (
        TemporaryDirectory() as syllabus_directory,
        TemporaryDirectory() as graph_directory,
    ):
        syllabus_store = TeachingSyllabusStore(syllabus_directory)
        syllabus_store.save(syllabus)
        projection = syllabus_store.read_curriculum_blueprint(syllabus["syllabus_id"])

        graph_store = TeachingCurriculumBlueprintStore(graph_directory)
        assert graph_store.save(projection) is True
        assert graph_store.save(projection) is False
        assert graph_store.read(projection["curriculum_id"]) == projection
        assert graph_store.list() == [projection]


def test_lesson_start_exposes_compact_blueprint_ref_without_bloating_live_goal() -> (
    None
):
    syllabus = _seal_generated_syllabus(
        _generated_draft(),
        model="deepseek-v4-flash",
        source_resource_ids=[],
        created_at="2026-08-12T01:00:00Z",
    )
    payload = syllabus_lesson_start_payload(syllabus, "lesson_01_01")

    assert payload["curriculum_blueprint_ref"]["status"] == "available"
    assert payload["curriculum_blueprint_ref"]["authority"] is False
    assert (
        payload["curriculum_blueprint_ref"]["authoritative_for_runtime_grading"]
        is False
    )
    assert payload["curriculum_lesson_mapping"]["legacy_lesson_id"] == ("lesson_01_01")
    assert "curriculum_blueprint" not in payload["goal"]
    assert payload["claim_boundary"]["curriculum_blueprint_is_gold"] is False


def test_curriculum_schema_is_explicitly_packaged_and_public_allowlisted() -> None:
    root = project_root()
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    allowlist = (root / "release/public_json_resources.txt").read_text(encoding="utf-8")
    schema_path = "schema/teaching_curriculum_blueprint.schema.json"
    assert f'"{schema_path}"' in pyproject
    assert schema_path in allowlist.splitlines()
