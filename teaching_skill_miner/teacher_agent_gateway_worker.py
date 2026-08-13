"""Private loopback worker for the authenticated TeachLab gateway.

The public API process owns authentication and scope selection.  This process
receives one bounded bootstrap document over stdin, builds exactly one
tenant/owner-isolated dashboard snapshot, and serves it on an ephemeral
loopback port.  Capability material and private paths are intentionally never
accepted on argv or environment variables and are never written to stdout or
stderr.

This module is not a public multi-tenant server.  Sharing one instance between
access scopes would violate its security contract.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import sysconfig
import threading
from typing import Any, Mapping, NoReturn

from .deepseek_client import DeepSeekClient, DeepSeekConfig
from .teacher_agent_dashboard import (
    build_teacher_agent_dashboard_snapshot,
    create_teacher_agent_dashboard_server,
)
from .teacher_agent_live import LiveAgentOptions
from .teacher_agent_authority import TeacherAuthorityVerifier
from .teacher_agent_consent import (
    ConsentError,
    validated_provider_policy,
    validated_subject_policy,
)
from .teacher_agent_curriculum_authority_store import (
    TeachingCurriculumAuthorityStore,
)
from .teacher_agent_curriculum_signing import CurriculumSigningKeyring
from .teacher_agent_safeguarding import (
    EscalationDeliveryConfig,
    TeacherAgentSafeguardingStore,
    default_emergency_resource_policy,
)
from .teacher_agent_safeguarding_authority import (
    CompositeSafeguardingAuthorizationVerifier,
    InternalSafeguardingStaffAuthority,
    InternalSafeguardingSystemAuthority,
)
from .teacher_agent_safeguarding_dispatch import (
    HttpsSafeguardingDispatcher,
    SafeguardingDispatchError,
    SafeguardingDispatcher,
)
from .teacher_agent_safeguarding_supervisor import (
    write_safeguarding_route_record,
)
from .teacher_agent_worker_isolation import (
    WorkerIsolationError,
    install_worker_filesystem_isolation,
    install_worker_process_resource_limits,
)


WORKER_BOOTSTRAP_SCHEMA = "teaching_skill_miner.gateway_worker_bootstrap.v1"
WORKER_STATUS_SCHEMA = "teaching_skill_miner.gateway_worker_status.v1"

_CONFIG_FIELDS = frozenset(
    {
        "schema",
        "scope_id",
        "scope_key_version",
        "worker_id",
        "capability_token",
        "private_root",
        "scope_key_material",
        "learner_scope_id",
        "authority_scope_bindings",
        "agent_backend",
        "api_key_file",
        "filesystem_isolation_required",
        "process_resource_limits",
        "remote_provider_policy",
        "remote_subject_policy",
        "safeguarding_locale",
        "safeguarding_dispatcher",
    }
)
_SCOPE_PATTERN = re.compile(r"scope_[0-9a-f]{48}")
_KEY_VERSION_PATTERN = re.compile(r"k[1-9][0-9]{0,8}")
_WORKER_PATTERN = re.compile(r"worker_[0-9a-f]{32}")
_CAPABILITY_PATTERN = re.compile(r"[A-Za-z0-9_-]{43,128}")
_LOCALE_PATTERN = re.compile(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8}){0,3}")
_PROVIDER_PATTERN = re.compile(r"[A-Za-z0-9_.-]{2,80}")
_MAX_BOOTSTRAP_BYTES = 16 * 1024
_REQUIRED_RUNTIME_DATA = (
    "teacher_agent_skill_library_v2.json",
    "teacher_agent_demo_input.json",
    "teacher_agent_evaluation_cases.json",
    "neural_v1_runtime_manifest.json",
    "teacher_agent_learning_outcome_demo.json",
    "teacher_agent_free_text_benchmark_receipt.json",
)


class GatewayWorkerConfigurationError(RuntimeError):
    """Raised when the private parent-to-child bootstrap cannot be trusted."""


@dataclass(frozen=True)
class _PreparedRuntime:
    """Validated, confined worker inputs shared by normal and canary startup."""

    scope_id: str
    key_version: str
    worker_id: str
    capability: str
    root: Path
    key: bytes
    learner_scope_id: str
    authority_scope_bindings: tuple[tuple[str, str], ...]
    client: DeepSeekClient | None
    remote_provider_policy: Mapping[str, Any] | None
    remote_subject_policy: Mapping[str, Any]
    remote_processing_region: str
    remote_provider_retention_days: int
    safeguarding_locale: str
    safeguarding_dispatcher: SafeguardingDispatcher | None
    data_root: Path
    filesystem_isolation: str
    process_resource_limits: Mapping[str, Any]


def _fail(code: str, *, exit_code: int = 2) -> NoReturn:
    """Emit only a stable, content-free failure code."""

    payload = {
        "schema": WORKER_STATUS_SCHEMA,
        "status": "failed",
        "code": code,
    }
    try:
        sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    finally:
        raise SystemExit(exit_code)


def _read_bootstrap() -> dict[str, Any]:
    raw = sys.stdin.buffer.readline(_MAX_BOOTSTRAP_BYTES + 1)
    if not raw or len(raw) > _MAX_BOOTSTRAP_BYTES or not raw.endswith(b"\n"):
        raise GatewayWorkerConfigurationError("bootstrap envelope is invalid")
    if sys.stdin.buffer.read(1):
        raise GatewayWorkerConfigurationError("bootstrap envelope has trailing data")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GatewayWorkerConfigurationError("bootstrap envelope is invalid") from exc
    if (
        not isinstance(value, dict)
        or set(value) != _CONFIG_FIELDS
        or value.get("schema") != WORKER_BOOTSTRAP_SCHEMA
    ):
        raise GatewayWorkerConfigurationError("bootstrap envelope is invalid")
    return value


def _required_match(value: Any, pattern: re.Pattern[str], field: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise GatewayWorkerConfigurationError(f"{field} is invalid")
    return value


def _private_root(value: Any, scope_id: str, key_version: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise GatewayWorkerConfigurationError("private root is invalid")
    root = Path(value)
    if not root.is_absolute():
        raise GatewayWorkerConfigurationError("private root is invalid")
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise GatewayWorkerConfigurationError("private root is unavailable") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_mode & 0o077
        or root.name != scope_id
        or root.parent.name != key_version
    ):
        raise GatewayWorkerConfigurationError("private root is unsafe")
    return root.resolve(strict=True)


def _key_material(value: Any) -> bytes:
    if not isinstance(value, str) or len(value) > 256:
        raise GatewayWorkerConfigurationError("scope key material is invalid")
    try:
        decoded = base64.b64decode(value, altchars=b"-_", validate=True)
    except (ValueError, TypeError) as exc:
        raise GatewayWorkerConfigurationError("scope key material is invalid") from exc
    if len(decoded) != 32:
        raise GatewayWorkerConfigurationError("scope key material is invalid")
    return decoded


def _authority_scope_bindings(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= 16:
        raise GatewayWorkerConfigurationError("authority scope bindings are invalid")
    bindings: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"scope_id", "key_version"}:
            raise GatewayWorkerConfigurationError("authority scope binding is invalid")
        scope_id = _required_match(item.get("scope_id"), _SCOPE_PATTERN, "scope_id")
        key_version = _required_match(
            item.get("key_version"), _KEY_VERSION_PATTERN, "key_version"
        )
        binding = (scope_id, key_version)
        if binding in bindings:
            raise GatewayWorkerConfigurationError(
                "authority scope bindings contain duplicates"
            )
        bindings.append(binding)
    return tuple(bindings)


def _safeguarding_dispatcher(value: Any) -> SafeguardingDispatcher | None:
    if value is None:
        return None
    required = {
        "endpoint",
        "bearer_secret",
        "route_locator",
        "policy_version",
        "timeout_ms",
        "maximum_response_bytes",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise GatewayWorkerConfigurationError(
            "safeguarding dispatcher configuration is invalid"
        )
    timeout_ms = value.get("timeout_ms")
    maximum_response_bytes = value.get("maximum_response_bytes")
    if (
        isinstance(timeout_ms, bool)
        or not isinstance(timeout_ms, int)
        or isinstance(maximum_response_bytes, bool)
        or not isinstance(maximum_response_bytes, int)
    ):
        raise GatewayWorkerConfigurationError(
            "safeguarding dispatcher configuration is invalid"
        )
    try:
        return HttpsSafeguardingDispatcher(
            endpoint=value.get("endpoint"),
            bearer_secret=value.get("bearer_secret"),
            route_locator=value.get("route_locator"),
            policy_version=value.get("policy_version"),
            timeout_seconds=timeout_ms / 1000,
            maximum_response_bytes=maximum_response_bytes,
        )
    except (SafeguardingDispatchError, TypeError, ValueError) as exc:
        raise GatewayWorkerConfigurationError(
            "safeguarding dispatcher configuration is invalid"
        ) from exc


def _derived_secret(key: bytes, label: bytes) -> bytes:
    return hmac.new(key, b"teachlab-gateway-worker-v1\x00" + label, sha256).digest()


def _runtime_data_root() -> Path:
    """Resolve immutable runtime fixtures in source and installed-wheel layouts."""

    candidates = (
        Path(__file__).resolve().parent.parent / "data",
        Path(sysconfig.get_path("data")) / "share" / "teaching-skill-miner" / "data",
    )
    for candidate in candidates:
        try:
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                continue
            for name in _REQUIRED_RUNTIME_DATA:
                item = candidate / name
                item_metadata = item.lstat()
                if stat.S_ISLNK(item_metadata.st_mode) or not stat.S_ISREG(
                    item_metadata.st_mode
                ):
                    raise GatewayWorkerConfigurationError(
                        "runtime fixture data is unsafe"
                    )
            return candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise GatewayWorkerConfigurationError(
                "runtime fixture data is unavailable"
            ) from exc
    raise GatewayWorkerConfigurationError("runtime fixture data is unavailable")


def _expected_capability(
    key: bytes, *, scope_id: str, key_version: str, worker_id: str
) -> str:
    binding = (
        "capability-v1\x00" + key_version + "\x00" + scope_id + "\x00" + worker_id
    ).encode("ascii")
    return (
        base64.urlsafe_b64encode(hmac.new(key, binding, sha256).digest())
        .decode("ascii")
        .rstrip("=")
    )


def _client(config: Mapping[str, Any]) -> DeepSeekClient | None:
    backend = config.get("agent_backend")
    key_file = config.get("api_key_file")
    if backend == "deterministic":
        if key_file is not None:
            raise GatewayWorkerConfigurationError(
                "deterministic backend cannot receive an API key file"
            )
        return None
    if backend != "deepseek" or not isinstance(key_file, str) or not key_file:
        raise GatewayWorkerConfigurationError("agent backend is invalid")
    path = Path(key_file)
    if not path.is_absolute():
        raise GatewayWorkerConfigurationError("provider credential is invalid")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise GatewayWorkerConfigurationError(
            "provider credential is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o077
        or not 1 <= metadata.st_size <= 4096
    ):
        raise GatewayWorkerConfigurationError("provider credential is unsafe")
    try:
        api_key = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise GatewayWorkerConfigurationError(
            "provider credential is unavailable"
        ) from exc
    if not api_key or any(character.isspace() for character in api_key):
        raise GatewayWorkerConfigurationError("provider credential is malformed")
    return DeepSeekClient(
        DeepSeekConfig(
            allow_remote_student_data=True,
            api_key_file=path,
        ),
        api_key=api_key,
    )


def _prepare_runtime(config: Mapping[str, Any]) -> _PreparedRuntime:
    """Exercise the production bootstrap/policy/path/isolation boundary.

    This phase deliberately does not construct any durable store or contact a
    remote provider.  The readiness canary stops here; a normal worker uses the
    returned values to construct the tenant-scoped dashboard.
    """

    scope_id = _required_match(config.get("scope_id"), _SCOPE_PATTERN, "scope_id")
    key_version = _required_match(
        config.get("scope_key_version"), _KEY_VERSION_PATTERN, "scope_key_version"
    )
    worker_id = _required_match(config.get("worker_id"), _WORKER_PATTERN, "worker_id")
    capability = _required_match(
        config.get("capability_token"), _CAPABILITY_PATTERN, "capability_token"
    )
    root = _private_root(config.get("private_root"), scope_id, key_version)
    key = _key_material(config.get("scope_key_material"))
    learner_scope_id = _required_match(
        config.get("learner_scope_id"), _SCOPE_PATTERN, "learner_scope_id"
    )
    authority_scope_bindings = _authority_scope_bindings(
        config.get("authority_scope_bindings")
    )
    if not hmac.compare_digest(
        capability,
        _expected_capability(
            key,
            scope_id=scope_id,
            key_version=key_version,
            worker_id=worker_id,
        ),
    ):
        raise GatewayWorkerConfigurationError(
            "capability is not bound to this worker identity"
        )
    client = _client(config)
    remote_provider_policy = config.get("remote_provider_policy")
    remote_subject_policy = config.get("remote_subject_policy")
    safeguarding_locale = _required_match(
        config.get("safeguarding_locale"),
        _LOCALE_PATTERN,
        "safeguarding_locale",
    )
    safeguarding_dispatcher = _safeguarding_dispatcher(
        config.get("safeguarding_dispatcher")
    )
    if client is not None and (
        not isinstance(remote_provider_policy, Mapping)
        or not isinstance(remote_subject_policy, Mapping)
    ):
        raise GatewayWorkerConfigurationError(
            "remote processing deployment policies are unavailable"
        )
    if remote_provider_policy is not None and not isinstance(
        remote_provider_policy, Mapping
    ):
        raise GatewayWorkerConfigurationError("remote provider policy is invalid")
    if remote_subject_policy is not None and not isinstance(
        remote_subject_policy, Mapping
    ):
        raise GatewayWorkerConfigurationError("remote subject policy is invalid")
    remote_processing_region = (
        remote_provider_policy.get("processing_region")
        if isinstance(remote_provider_policy, Mapping)
        else "provider_managed"
    )
    remote_provider_retention_days = (
        remote_provider_policy.get("provider_retention_days")
        if isinstance(remote_provider_policy, Mapping)
        else 30
    )
    raw_subject = (
        dict(remote_subject_policy)
        if isinstance(remote_subject_policy, Mapping)
        else None
    )
    likely_minor = raw_subject.get("likely_minor", False) if raw_subject else False
    guardian_policy = (
        raw_subject.get("guardian_or_school_policy", "not_required")
        if raw_subject
        else "not_required"
    )
    if not isinstance(likely_minor, bool) or not isinstance(guardian_policy, str):
        raise GatewayWorkerConfigurationError("remote subject policy is invalid")
    try:
        normalized_subject_policy = validated_subject_policy(
            raw_subject,
            likely_minor=likely_minor,
            guardian_or_school_policy=guardian_policy,
        )
    except ConsentError as exc:
        raise GatewayWorkerConfigurationError(
            "remote subject policy is invalid"
        ) from exc
    normalized_provider_policy: Mapping[str, Any] | None = remote_provider_policy
    if client is not None:
        provider_status = client.public_status()
        provider_id = str(
            provider_status.get("provider") or provider_status.get("model") or ""
        ).strip()
        if _PROVIDER_PATTERN.fullmatch(provider_id) is None:
            raise GatewayWorkerConfigurationError(
                "remote provider identity is unavailable"
            )
        try:
            normalized_provider_policy = validated_provider_policy(
                remote_provider_policy,
                provider_id=provider_id,
                processing_region=remote_processing_region,
                provider_retention_days=remote_provider_retention_days,
            )
        except ConsentError as exc:
            raise GatewayWorkerConfigurationError(
                "remote provider policy is invalid"
            ) from exc
        if (
            normalized_provider_policy["policy_source"]
            != "deployment_operator_asserted_external_terms_not_repository_verified"
            or normalized_subject_policy["policy_source"]
            != "organization_oidc_or_roster_policy"
        ):
            raise GatewayWorkerConfigurationError(
                "remote processing policies lack server authority"
            )
    isolation_required = config.get("filesystem_isolation_required")
    if not isinstance(isolation_required, bool):
        raise GatewayWorkerConfigurationError("filesystem isolation policy is invalid")
    data_root = _runtime_data_root()
    try:
        filesystem_isolation = install_worker_filesystem_isolation(
            required=isolation_required,
            private_root=root,
            runtime_data_root=data_root,
            provider_key_file=(
                Path(str(config["api_key_file"]))
                if isinstance(config.get("api_key_file"), str)
                else None
            ),
            application_root=Path.cwd(),
        )
    except WorkerIsolationError as exc:
        raise GatewayWorkerConfigurationError(
            "required worker filesystem isolation is unavailable"
        ) from exc
    try:
        process_resource_limits = install_worker_process_resource_limits(
            required=isolation_required,
            policy=config.get("process_resource_limits"),
        )
    except WorkerIsolationError as exc:
        raise GatewayWorkerConfigurationError(
            "required worker process resource limits are unavailable"
        ) from exc
    if isolation_required:
        # Set this only after the filesystem sandbox succeeds. Besides keeping
        # failed startup side-effect free in in-process contract tests, this
        # ensures optional parser capabilities are never advertised unless the
        # parent worker confinement is already active.
        os.environ["TSM_REQUIRE_PARSER_NETWORK_ISOLATION"] = "1"
    return _PreparedRuntime(
        scope_id=scope_id,
        key_version=key_version,
        worker_id=worker_id,
        capability=capability,
        root=root,
        key=key,
        learner_scope_id=learner_scope_id,
        authority_scope_bindings=authority_scope_bindings,
        client=client,
        remote_provider_policy=normalized_provider_policy,
        remote_subject_policy=normalized_subject_policy,
        remote_processing_region=str(remote_processing_region),
        remote_provider_retention_days=int(remote_provider_retention_days),
        safeguarding_locale=safeguarding_locale,
        safeguarding_dispatcher=safeguarding_dispatcher,
        data_root=data_root,
        filesystem_isolation=filesystem_isolation,
        process_resource_limits=process_resource_limits,
    )


def _serve(config: Mapping[str, Any]) -> int:
    prepared = _prepare_runtime(config)
    scope_id = prepared.scope_id
    key_version = prepared.key_version
    worker_id = prepared.worker_id
    capability = prepared.capability
    root = prepared.root
    key = prepared.key
    authority_verifier = TeacherAuthorityVerifier(
        key=_derived_secret(key, b"teacher-authority-v1"),
        scope_id=scope_id,
        scope_key_version=key_version,
        legacy_scope_bindings=prepared.authority_scope_bindings,
        replay_store_path=root / "teacher_authority_replay.jsonl",
    )
    curriculum_signing_keyring = CurriculumSigningKeyring(
        root / "syllabi" / ".curriculum_signing_keyring.json",
        integrity_key=_derived_secret(key, b"curriculum-signing-keyring-v1"),
    )
    curriculum_authority_store = TeachingCurriculumAuthorityStore(
        root / "syllabi" / ".curriculum_authority.json",
        scope_id=prepared.learner_scope_id,
        gateway_scope_ids=[
            scope_id,
            *(binding[0] for binding in prepared.authority_scope_bindings),
        ],
        trusted_teacher_public_keys=(curriculum_signing_keyring.trusted_public_keys),
        known_teacher_public_keys=curriculum_signing_keyring.all_public_keys,
        gateway_receipt_validator=(authority_verifier.verify_verification_receipt),
    )
    safeguarding_scope_sha256 = _derived_secret(key, b"safeguarding-scope-v1").hex()
    safeguarding_authority = InternalSafeguardingSystemAuthority(
        key=_derived_secret(key, b"safeguarding-system-authority-v1"),
        scope_sha256=safeguarding_scope_sha256,
    )
    safeguarding_staff_authority = InternalSafeguardingStaffAuthority(
        key=_derived_secret(key, b"safeguarding-staff-authority-v1"),
        scope_sha256=safeguarding_scope_sha256,
        gateway_receipt_verifier=authority_verifier.verify_verification_receipt,
    )
    safeguarding_authorization = CompositeSafeguardingAuthorizationVerifier(
        system=safeguarding_authority,
        staff=safeguarding_staff_authority,
    )
    dispatcher = prepared.safeguarding_dispatcher
    safeguarding_store = TeacherAgentSafeguardingStore(
        root / "safeguarding",
        authorization_verifier=safeguarding_authorization.verify,
        emergency_resource_policy=default_emergency_resource_policy(
            {safeguarding_scope_sha256: prepared.safeguarding_locale}
        ),
        escalation_delivery=(
            EscalationDeliveryConfig(
                policy_version="scope_durable_safeguarding_outbox_v1",
                queue_sha256=str(dispatcher.queue_sha256),
                sla_seconds_by_severity={
                    "elevated": 24 * 60 * 60,
                    "high": 4 * 60 * 60,
                    "urgent": 15 * 60,
                },
            )
            if dispatcher is not None and dispatcher.queue_sha256 is not None
            else None
        ),
    )
    if isinstance(dispatcher, HttpsSafeguardingDispatcher):
        write_safeguarding_route_record(
            safeguarding_store.root,
            dispatcher.supervisor_route_locator(),
        )
    snapshot = build_teacher_agent_dashboard_snapshot(
        prepared.data_root / "teacher_agent_skill_library_v2.json",
        prepared.data_root / "teacher_agent_demo_input.json",
        prepared.data_root / "teacher_agent_evaluation_cases.json",
        client=prepared.client,
        live_options=LiveAgentOptions(
            fallback_to_rules=True,
            action_only_repair_enabled=True,
            state_first_route_adjudication_enabled=True,
            agent_loop_enabled=True,
            agent_loop_post_assessment_enabled=True,
            maximum_agent_steps=4,
        ),
        neural_v1_manifest_path=(
            prepared.data_root / "neural_v1_runtime_manifest.json"
        ),
        learning_outcome_path=(
            prepared.data_root / "teacher_agent_learning_outcome_demo.json"
        ),
        free_text_benchmark_receipt_path=(
            prepared.data_root / "teacher_agent_free_text_benchmark_receipt.json"
        ),
        store_path=root / "sessions.jsonl",
        syllabus_store_path=root / "syllabi",
        project_store_path=root / "projects",
        resource_index_store_path=root / "resource_index",
        resource_review_store_path=root / "resource_reviews",
        learning_record_store_path=root / "learning_records",
        metacognition_store_path=root / "metacognition",
        adjudication_store_path=root / "adjudication",
        teacher_authority_verifier=authority_verifier,
        curriculum_authority_store=curriculum_authority_store,
        curriculum_signing_keyring=curriculum_signing_keyring,
        consent_store_path=root / "consent",
        consent_signing_secret=_derived_secret(key, b"consent"),
        remote_processing_region=prepared.remote_processing_region,
        remote_provider_retention_days=prepared.remote_provider_retention_days,
        remote_provider_policy=prepared.remote_provider_policy,
        remote_subject_policy=prepared.remote_subject_policy,
        learner_key_secret=_derived_secret(key, b"learner-key"),
        learner_tenant_id=prepared.learner_scope_id,
        trusted_learner_profile_ref=(
            "profile_" + _derived_secret(key, b"learner-profile-ref").hex()
        ),
        safeguarding_store=safeguarding_store,
        safeguarding_scope_sha256=safeguarding_scope_sha256,
        safeguarding_system_authority_issuer=safeguarding_authority.issue,
        safeguarding_system_authority_verifier=safeguarding_authority.verify,
        safeguarding_staff_authority_issuer=safeguarding_staff_authority.issue,
        safeguarding_dispatcher=dispatcher,
    )
    server, _private_url = create_teacher_agent_dashboard_server(
        snapshot,
        port=0,
        capability_token=capability,
    )

    stopping = threading.Event()

    def request_stop(_signal: int, _frame: Any) -> None:
        if stopping.is_set():
            return
        stopping.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    status = {
        "schema": WORKER_STATUS_SCHEMA,
        "status": "ready",
        "worker_id": worker_id,
        "scope_key_version": key_version,
        "port": server.server_port,
        "backend": config["agent_backend"],
        "filesystem_isolation": prepared.filesystem_isolation,
        "process_resource_limits": prepared.process_resource_limits,
    }
    sys.stdout.write(json.dumps(status, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        stopping.set()
        server.server_close()
    return 0


def _runtime_canary(config: Mapping[str, Any]) -> int:
    """Validate worker startup without remote I/O or durable tenant stores."""

    prepared = _prepare_runtime(config)
    stopping = threading.Event()

    def request_stop(_signal: int, _frame: Any) -> None:
        stopping.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    status = {
        "schema": WORKER_STATUS_SCHEMA,
        "status": "canary_ready",
        "worker_id": prepared.worker_id,
        "scope_key_version": prepared.key_version,
        "backend": config["agent_backend"],
        "filesystem_isolation": prepared.filesystem_isolation,
        "process_resource_limits": prepared.process_resource_limits,
        "remote_provider_network": "not_contacted",
        "provider_credential": (
            "loaded_not_provider_validated"
            if prepared.client is not None
            else "not_applicable"
        ),
        "persistent_tenant_data_created": False,
    }
    sys.stdout.write(json.dumps(status, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    stopping.wait()
    return 0


def _provider_readiness(config: Mapping[str, Any]) -> int:
    """Run one authenticated, content-free probe against the real provider path."""

    prepared = _prepare_runtime(config)
    if prepared.client is None:
        status = {
            "schema": WORKER_STATUS_SCHEMA,
            "status": "provider_not_required",
            "backend": config["agent_backend"],
            "credential_validated": False,
            "provider_network_validated": False,
            "configured_model_available": False,
            "learner_content_sent": False,
            "generation_created": False,
            "persistent_tenant_data_created": False,
        }
    else:
        probe = prepared.client.probe_model_availability(timeout_seconds=5.0)
        status = {
            "schema": WORKER_STATUS_SCHEMA,
            "status": "provider_ready",
            "backend": config["agent_backend"],
            **{key: value for key, value in probe.items() if key != "schema"},
            "persistent_tenant_data_created": False,
        }
    sys.stdout.write(json.dumps(status, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    return 0


def main() -> int:
    # A restrictive umask is a second fence behind each store's own private
    # directory/file validation.
    os.umask(0o077)
    if sys.argv[1:] in (["--help"], ["-h"]):
        sys.stdout.write(
            "usage: python -m teaching_skill_miner.teacher_agent_gateway_worker "
            "[--self-check|--runtime-canary|--provider-readiness]\n"
        )
        return 0
    if sys.argv[1:] == ["--self-check"]:
        # Import-time contract check only. It deliberately reads no stdin,
        # secret, path, network endpoint, or tenant state and starts no server.
        if not WORKER_STATUS_SCHEMA or not callable(
            create_teacher_agent_dashboard_server
        ):
            return 2
        sys.stdout.write("teacher_agent_gateway_worker self-check passed\n")
        return 0
    runtime_canary = sys.argv[1:] == ["--runtime-canary"]
    provider_readiness = sys.argv[1:] == ["--provider-readiness"]
    if sys.argv[1:] and not runtime_canary and not provider_readiness:
        _fail("worker_configuration_invalid")
    try:
        bootstrap = _read_bootstrap()
        if runtime_canary:
            return _runtime_canary(bootstrap)
        if provider_readiness:
            return _provider_readiness(bootstrap)
        return _serve(bootstrap)
    except GatewayWorkerConfigurationError:
        _fail("worker_configuration_invalid")
    except BaseException:
        # Never serialize exception messages: they may include a capability,
        # provider credential location, or tenant-private store path.
        _fail("worker_startup_failed")


if __name__ == "__main__":
    raise SystemExit(main())
