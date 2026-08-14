from __future__ import annotations

import copy
import json
from pathlib import Path
import re

import pytest

from scripts.verify_teacher_agent_deployment import (
    verify_caddyfile,
    verify_rendered_compose,
)
from scripts.generate_python_environment_sbom import build_sbom, write_sbom


ROOT = Path(__file__).resolve().parents[1]


def _rendered_contract() -> str:
    source = (ROOT / "deploy/production/compose.yaml").read_text(encoding="utf-8")
    replacements = {
        "${TEACHLAB_CADDY_IMAGE:?set a digest-pinned Caddy image}": "registry.invalid/caddy@sha256:"
        + "1" * 64,
        "${TEACHLAB_CONSOLE_IMAGE:?set a digest-pinned Console image}": "registry.invalid/console@sha256:"
        + "2" * 64,
        "${TEACHLAB_API_IMAGE:?set a digest-pinned API image}": "registry.invalid/api@sha256:"
        + "3" * 64,
    }
    for key, value in replacements.items():
        source = source.replace(key, value)
    # A rendered fixture does not need to preserve deployment-specific values;
    # the verifier's unresolved-variable gate is exercised separately.
    source = source.replace("${TEACHLAB_HTTP_PORT:-80}", "80")
    source = source.replace("${TEACHLAB_HTTPS_PORT:-443}", "443")
    source = re_sub_variables(source)
    return source


def re_sub_variables(source: str) -> str:
    import re

    return re.sub(r"\$\{[^}]+\}", "/private/fixture", source)


def _rendered_json_contract() -> dict[str, object]:
    def service(
        image: str,
        user: str,
        limits: dict[str, object],
        reservations: dict[str, object],
    ) -> dict[str, object]:
        return {
            "image": image,
            "restart": "unless-stopped",
            "read_only": True,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
            "user": user,
            "init": True,
            "healthcheck": {"test": ["CMD", "probe"], "retries": 3},
            "deploy": {
                "replicas": 1,
                "resources": {"limits": limits, "reservations": reservations},
            },
        }

    edge = service(
        "registry.invalid/caddy@sha256:" + "1" * 64,
        "1000:1000",
        {"cpus": 1, "memory": "268435456"},
        {"cpus": 0.1, "memory": "67108864"},
    )
    edge.update({"cap_add": ["NET_BIND_SERVICE"], "ports": [{"target": 80}]})
    console = service(
        "registry.invalid/console@sha256:" + "2" * 64,
        "10001:10001",
        {"cpus": 2, "memory": "805306368"},
        {"cpus": 0.25, "memory": "268435456"},
    )
    console["environment"] = {
        "NEXT_PUBLIC_TEACHLAB_AUTH_MODE": "oidc",
        "TEACHLAB_HARNESS_MODE": "authenticated_apps_api",
        "TEACHLAB_APPS_API_INTERNAL_URL": "http://api:4000",
        "TEACHLAB_APPS_API_URL": "https://console.example.invalid",
    }
    api = service(
        "registry.invalid/api@sha256:" + "3" * 64,
        "10001:10001",
        {"cpus": 4, "memory": "4294967296", "pids": 256},
        {"cpus": 0.5, "memory": "1073741824"},
    )
    api["pids_limit"] = 256
    api["healthcheck"] = {
        "test": [
            "CMD",
            "node",
            "-e",
            "fetch('http://127.0.0.1:4000/ready').then(r=>r.ok)",
        ],
        "retries": 5,
    }
    api["volumes"] = [
        {"type": "volume", "source": "scope_data", "target": "/var/lib/teachlab"},
        {
            "type": "bind",
            "source": "/private/teachlab/secrets",
            "target": "/run/teachlab",
            "read_only": True,
            "bind": {},
        },
    ]
    api_environment = {
        "HARNESS_GATEWAY_ENABLED": "true",
        "HARNESS_WORKER_ROOT": "/var/lib/teachlab/scopes",
        "HARNESS_WORKER_BACKEND": "deepseek",
        "HARNESS_WORKER_FILESYSTEM_ISOLATION_REQUIRED": "true",
        "HARNESS_WORKER_RLIMIT_CORE_BYTES": "0",
        "HARNESS_WORKER_RLIMIT_NOFILE": "256",
        "HARNESS_WORKER_RLIMIT_FSIZE_BYTES": "536870912",
        "HARNESS_WORKER_RLIMIT_AS_BYTES": "1610612736",
        "AUTH_MODE": "oidc",
        "DATA_BACKEND": "postgres",
        "PG_SSL_MODE": "verify-full",
        "SAFEGUARDING_AUTHORITY_ROLES": "safeguarding",
        "SESSION_SECRET_FILE": "/run/teachlab/session-secret",
        "OIDC_CLIENT_SECRET_FILE": "/run/teachlab/oidc-client-secret",
        "OIDC_TRANSACTION_SECRET_FILE": "/run/teachlab/oidc-transaction-secret",
        "ACCOUNT_SCOPE_SECRET_FILE": "/run/teachlab/account-scope-secret",
        "ACCOUNT_IDENTITY_NAMESPACE_SECRET_FILE": "/run/teachlab/account-identity-namespace-secret",
        "ACCOUNT_DELETION_STATUS_SECRET_FILE": "/run/teachlab/account-deletion-status-secret",
        "ACCOUNT_CACHE_SCOPE_SECRET_FILE": "/run/teachlab/account-cache-scope-secret",
        "HARNESS_SCOPE_SECRET_FILE": "/run/teachlab/harness-scope-secret",
        "HARNESS_PROVIDER_API_KEY_FILE": "/run/teachlab/provider-api-key",
        "HARNESS_SAFEGUARDING_DISPATCH_BEARER_SECRET_FILE": "/run/teachlab/safeguarding-dispatch-bearer",
        "HARNESS_SAFEGUARDING_RETENTION_AUTHORITY_SECRET_FILE": "/run/teachlab/safeguarding-retention-authority",
        "TEACHER_ENTITLEMENT_DIRECTORY_BEARER_SECRET_FILE": "/run/teachlab/teacher-entitlement-directory-bearer",
        "TEACHER_ENTITLEMENT_BINDING_KEY_FILE": "/run/teachlab/teacher-entitlement-binding-key",
        "TEACHER_ENTITLEMENT_RECEIPT_KEY_FILE": "/run/teachlab/teacher-entitlement-receipt-key",
        "METRICS_TOKEN_FILE": "/run/teachlab/metrics-token",
    }
    for key in (
        "OIDC_ACCOUNT_AAL2_ACR_VALUES",
        "TEACHER_ENTITLEMENT_DIRECTORY_URL",
        "TEACHER_ENTITLEMENT_POLICY_ID",
        "TEACHER_ENTITLEMENT_POLICY_VERSION",
        "TEACHER_ENTITLEMENT_FRESHNESS_TTL_MS",
        "TEACHER_ENTITLEMENT_CACHE_TTL_MS",
        "TEACHER_ENTITLEMENT_PROVIDER_TIMEOUT_MS",
        "TEACHER_ENTITLEMENT_MAX_RESPONSE_BYTES",
        "TEACHER_ENTITLEMENT_MAX_CLOCK_SKEW_MS",
        "TEACHER_ENTITLEMENT_MAX_CACHE_ENTRIES",
        "TEACHER_ENTITLEMENT_MIN_ASSURANCE_LEVEL",
        "TEACHER_AUTHORITY_ROLES",
        "HARNESS_PROVIDER_POLICY_ID",
        "HARNESS_PROVIDER_POLICY_VERSION",
        "HARNESS_PROVIDER_PROCESSING_REGION",
        "HARNESS_PROVIDER_RETENTION_DAYS",
        "HARNESS_PROVIDER_DELETION_STATUS",
        "HARNESS_PROVIDER_DOCUMENTATION_URL",
        "HARNESS_SAFEGUARDING_LOCALE",
        "HARNESS_SAFEGUARDING_DISPATCH_URL",
        "HARNESS_SAFEGUARDING_DISPATCH_POLICY_VERSION",
        "HARNESS_SAFEGUARDING_DISPATCH_TIMEOUT_MS",
        "HARNESS_SAFEGUARDING_DISPATCH_MAX_RESPONSE_BYTES",
        "HARNESS_SAFEGUARDING_RETENTION_POLICY_VERSION",
        "HARNESS_SAFEGUARDING_RETENTION_MINIMUM_CLOSED_AGE_SECONDS",
        "HARNESS_SAFEGUARDING_RETENTION_MAXIMUM_CASES_PER_RUN",
        "HARNESS_SAFEGUARDING_RETENTION_DEPLOYMENT_CONTEXT_SHA256",
        "REMOTE_SUBJECT_POLICY_ID",
        "REMOTE_SUBJECT_POLICY_VERSION",
    ):
        api_environment[key] = "configured"
    api["environment"] = api_environment
    return {"name": "teachlab-production", "services": {"edge": edge, "console": console, "api": api}}


def test_rendered_single_replica_contract_passes(tmp_path: Path) -> None:
    rendered = tmp_path / "compose.yaml"
    rendered.write_text(_rendered_contract(), encoding="utf-8")
    receipt = verify_rendered_compose(rendered)
    assert receipt == {
        "status": "deployment_contract_verified",
        "service_count": 3,
        "images_digest_pinned": True,
        "api_replicas": 1,
        "api_container_pid_limit": 256,
        "distributed_worker_scheduler": False,
        "external_image_signing_verified": False,
    }


def test_canonical_compose_json_contract_passes(tmp_path: Path) -> None:
    rendered = tmp_path / "compose.json"
    rendered.write_text(json.dumps(_rendered_json_contract()), encoding="utf-8")
    receipt = verify_rendered_compose(rendered)
    assert receipt["status"] == "deployment_contract_verified"
    assert receipt["api_container_pid_limit"] == 256


@pytest.mark.parametrize(
    "mutation",
    [
        lambda model: model["services"].update({"surprise": {}}),
        lambda model: model["services"]["api"].update({"pids_limit": 1024}),
        lambda model: model["services"]["api"]["deploy"]["resources"]["limits"].update({"pids": 1024}),
        lambda model: model["services"]["console"].update({"ports": [{"target": 3000}]}),
        lambda model: model["services"]["api"].update({"image": "registry.invalid/api:latest"}),
        lambda model: model["services"]["api"]["deploy"].update({"replicas": 2}),
        lambda model: model["services"]["api"]["environment"].update({"TEACHLAB_SESSION_SECRET": "inline"}),
        lambda model: model["services"]["api"]["volumes"][1].update({"read_only": False}),
    ],
)
def test_unsafe_canonical_compose_json_fails_closed(tmp_path: Path, mutation) -> None:
    model = copy.deepcopy(_rendered_json_contract())
    mutation(model)
    rendered = tmp_path / "compose.json"
    rendered.write_text(json.dumps(model), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_rendered_compose(rendered)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda text: text.replace("replicas: 1", "replicas: 2", 1),
        lambda text: text.replace("    read_only: true", "    read_only: false", 1),
        lambda text: text.replace("    read_only: true", "    read_only: false", 2),
        lambda text: text.replace("    read_only: true", "    read_only: false", 3),
        lambda text: text.replace("    cap_drop: [ALL]", "    cap_drop: [NET_RAW]", 1),
        lambda text: text.replace(
            "    security_opt: [no-new-privileges:true]",
            "    security_opt: [no-new-privileges:false]",
            1,
        ),
        lambda text: text.replace('    user: "10001:10001"', '    user: "0:0"', 1),
        lambda text: text.replace(
            "registry.invalid/api@sha256:" + "3" * 64, "registry.invalid/api:latest"
        ),
        lambda text: text.replace("SESSION_SECRET_FILE:", "SESSION_SECRET="),
        lambda text: text.replace(
            "HARNESS_WORKER_BACKEND: deepseek",
            "HARNESS_WORKER_BACKEND: deterministic",
        ),
        lambda text: text.replace(
            'HARNESS_WORKER_RLIMIT_CORE_BYTES: "0"',
            'HARNESS_WORKER_RLIMIT_CORE_BYTES: "1"',
        ),
        lambda text: text.replace(
            'HARNESS_WORKER_RLIMIT_NOFILE: "256"',
            'HARNESS_WORKER_RLIMIT_NOFILE: "128"',
        ),
        lambda text: text.replace(
            'HARNESS_WORKER_RLIMIT_FSIZE_BYTES: "536870912"',
            'HARNESS_WORKER_RLIMIT_FSIZE_BYTES: "335544320"',
        ),
        lambda text: text.replace(
            'HARNESS_WORKER_RLIMIT_AS_BYTES: "1610612736"',
            'HARNESS_WORKER_RLIMIT_AS_BYTES: "1073741824"',
        ),
        lambda text: text.replace("    pids_limit: 256", "    pids_limit: 1024"),
        lambda text: text.replace(
            'limits: {cpus: "4.00", memory: 4G, pids: 256}',
            'limits: {cpus: "4.00", memory: 4G, pids: 1024}',
        ),
        lambda text: text.replace(
            "HARNESS_SAFEGUARDING_RETENTION_POLICY_VERSION:",
            "HARNESS_SAFEGUARDING_RETENTION_POLICY_MISSING:",
        ),
        lambda text: text.replace(
            "HARNESS_SAFEGUARDING_RETENTION_MINIMUM_CLOSED_AGE_SECONDS:",
            "HARNESS_SAFEGUARDING_RETENTION_MINIMUM_CLOSED_AGE_MISSING:",
        ),
        lambda text: text.replace(
            "HARNESS_SAFEGUARDING_RETENTION_MAXIMUM_CASES_PER_RUN:",
            "HARNESS_SAFEGUARDING_RETENTION_MAXIMUM_CASES_MISSING:",
        ),
        lambda text: text.replace(
            "HARNESS_SAFEGUARDING_RETENTION_AUTHORITY_SECRET_FILE:",
            "HARNESS_SAFEGUARDING_RETENTION_AUTHORITY_SECRET=",
        ),
        lambda text: text.replace(
            "HARNESS_SAFEGUARDING_RETENTION_DEPLOYMENT_CONTEXT_SHA256:",
            "HARNESS_SAFEGUARDING_RETENTION_DEPLOYMENT_CONTEXT_MISSING:",
        ),
        lambda text: text.replace(
            "HARNESS_PROVIDER_POLICY_ID:", "HARNESS_PROVIDER_POLICY_MISSING:"
        ),
        lambda text: text.replace(
            "TEACHER_ENTITLEMENT_DIRECTORY_URL:",
            "TEACHER_ENTITLEMENT_DIRECTORY_MISSING:",
        ),
        lambda text: text.replace(
            "TEACHER_ENTITLEMENT_RECEIPT_KEY_FILE:",
            "TEACHER_ENTITLEMENT_RECEIPT_KEY_MISSING:",
        ),
        lambda text: text.replace(
            "HARNESS_SAFEGUARDING_LOCALE:",
            "HARNESS_SAFEGUARDING_LOCALE_MISSING:",
        ),
        lambda text: text.replace(
            "TEACHER_ENTITLEMENT_DIRECTORY_URL:",
            "TEACHER_ENTITLEMENT_DIRECTORY_MISSING:",
        ),
        lambda text: text.replace(
            "TEACHER_ENTITLEMENT_RECEIPT_KEY_FILE:",
            "TEACHER_ENTITLEMENT_RECEIPT_KEY=",
        ),
        lambda text: text.replace(
            ":/run/teachlab:ro", ":/run/teachlab/session-secret:ro"
        ),
        lambda text: text.replace(
            "ACCOUNT_CACHE_SCOPE_SECRET_FILE:", "ACCOUNT_CACHE_SCOPE_SECRET="
        ),
        lambda text: text + "\n${UNRESOLVED}\n",
        lambda text: text + "\n  ports:\n    - 4000:4000\n",
        lambda text: text.replace("  console:\n", "  surprise:\n", 1),
    ],
)
def test_unsafe_or_unrendered_deployments_fail_closed(tmp_path: Path, mutation) -> None:
    rendered = tmp_path / "compose.yaml"
    rendered.write_text(mutation(_rendered_contract()), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_rendered_compose(rendered)


def test_dockerfiles_reject_mutable_base_images_and_run_as_non_root() -> None:
    for name in ("teachlab_api.Dockerfile", "teachlab_console.Dockerfile"):
        source = (ROOT / "docker" / name).read_text(encoding="utf-8")
        assert "@sha256:" in source
        assert "must be digest pinned" in source
        assert "USER 10001:10001" in source
        assert "latest" not in source

    api = (ROOT / "docker/teachlab_api.Dockerfile").read_text(encoding="utf-8")
    assert api.index("ARG PYTHON_RUNTIME_IMAGE") < api.index("FROM ")
    assert "PROJECT_WHEEL_SHA256" in api
    assert "release wheel SHA-256 mismatch" in api
    assert "teacher_agent_gateway_worker --self-check" in api
    assert "teacher_agent_safeguarding_supervisor --self-check" in api
    assert "--require-hashes --only-binary=:all:" in api
    wheel_path = "/tmp/teaching_skill_miner-1.2.0-py3-none-any.whl"
    assert f"COPY ${{PROJECT_WHEEL}} {wheel_path}" in api
    assert f"Path('{wheel_path}')" in api
    assert f"--no-deps {wheel_path}" in api
    assert f"rm {wheel_path}" in api
    assert "teaching-skill-miner.whl" not in api
    assert "python -m pip check" in api
    assert "TEACHLAB_RELEASE_VERSION=${TEACHLAB_RELEASE_VERSION}" in api
    assert "TEACHLAB_RELEASE_ID=${TEACHLAB_RELEASE_ID}" in api
    assert "COPY apps/api/tsconfig.json apps/api/tsconfig.build.json ./" in api
    console = (ROOT / "docker/teachlab_console.Dockerfile").read_text(encoding="utf-8")
    assert console.index("ARG NODE_RUNTIME_IMAGE") < console.index("FROM ")
    assert "NEXT_PUBLIC_TEACHLAB_AUTH_MODE=oidc" in console
    assert "TEACHLAB_RELEASE_VERSION=${TEACHLAB_RELEASE_VERSION}" in console
    assert "TEACHLAB_RELEASE_ID=${TEACHLAB_RELEASE_ID}" in console
    assert "COPY apps/console/ ./" in console
    compose = (ROOT / "deploy/production/compose.yaml").read_text(encoding="utf-8")
    assert "NEXT_PUBLIC_TEACHLAB_AUTH_MODE: oidc" in compose
    assert "TEACHLAB_APPS_API_INTERNAL_URL: http://api:4000" in compose
    assert "OIDC_ACCOUNT_AAL2_ACR_VALUES:" in compose
    assert "TEACHER_ENTITLEMENT_DIRECTORY_URL:" in compose
    assert "TEACHER_ENTITLEMENT_DIRECTORY_BEARER_SECRET_FILE:" in compose
    assert "TEACHER_ENTITLEMENT_BINDING_KEY_FILE:" in compose
    assert "TEACHER_ENTITLEMENT_RECEIPT_KEY_FILE:" in compose
    assert "TEACHER_ENTITLEMENT_CACHE_TTL_MS:" in compose
    assert "ACCOUNT_SCOPE_SECRET_FILE:" in compose
    assert "ACCOUNT_IDENTITY_NAMESPACE_SECRET_FILE:" in compose
    assert "ACCOUNT_DELETION_STATUS_SECRET_FILE:" in compose
    assert "ACCOUNT_CACHE_SCOPE_SECRET_FILE:" in compose
    assert "TEACHER_ENTITLEMENT_DIRECTORY_BEARER_SECRET_FILE:" in compose
    assert "TEACHER_ENTITLEMENT_BINDING_KEY_FILE:" in compose
    assert "TEACHER_ENTITLEMENT_RECEIPT_KEY_FILE:" in compose
    assert "HARNESS_SAFEGUARDING_LOCALE:" in compose
    assert "SAFEGUARDING_AUTHORITY_ROLES: safeguarding" in compose
    assert "HARNESS_SAFEGUARDING_DISPATCH_URL:" in compose
    assert "HARNESS_SAFEGUARDING_DISPATCH_BEARER_SECRET_FILE:" in compose
    assert "HARNESS_SAFEGUARDING_DISPATCH_POLICY_VERSION:" in compose
    assert "HARNESS_WORKER_BACKEND: deepseek" in compose
    assert 'HARNESS_WORKER_RLIMIT_CORE_BYTES: "0"' in compose
    assert 'HARNESS_WORKER_RLIMIT_NOFILE: "256"' in compose
    assert 'HARNESS_WORKER_RLIMIT_FSIZE_BYTES: "536870912"' in compose
    assert 'HARNESS_WORKER_RLIMIT_AS_BYTES: "1610612736"' in compose
    assert "pids_limit: 256" in compose
    assert 'limits: {cpus: "4.00", memory: 4G, pids: 256}' in compose
    assert "HARNESS_SAFEGUARDING_RETENTION_POLICY_VERSION:" in compose
    assert "HARNESS_SAFEGUARDING_RETENTION_MINIMUM_CLOSED_AGE_SECONDS:" in compose
    assert "HARNESS_SAFEGUARDING_RETENTION_MAXIMUM_CASES_PER_RUN:" in compose
    assert "HARNESS_SAFEGUARDING_RETENTION_AUTHORITY_SECRET_FILE:" in compose
    assert "HARNESS_SAFEGUARDING_RETENTION_DEPLOYMENT_CONTEXT_SHA256:" in compose
    assert "TEACHLAB_SECRETS_DIR:" in compose
    assert "TEACHLAB_HARNESS_BACKEND" not in compose
    caddy = (ROOT / "deploy/production/Caddyfile").read_text(encoding="utf-8")
    assert "handle /internal/metrics" in caddy
    assert "header_up X-Forwarded-For {remote_host}" in caddy
    assert "rewrite * /metrics" in caddy
    assert "reverse_proxy api:4000" in caddy
    assert caddy.count("health_uri /ready") == 3
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "**/node_modules" in dockerignore
    assert "**/.next" in dockerignore
    assert ".private" in dockerignore
    runtime_lock = (ROOT / "deploy/production/python-runtime.lock").read_text(
        encoding="utf-8"
    )
    assert runtime_lock.count("==") == 3
    assert runtime_lock.count("--hash=sha256:") == 9


def test_caddy_edge_contract_is_exact_and_fail_closed(tmp_path: Path) -> None:
    source = (ROOT / "deploy/production/Caddyfile").read_text(encoding="utf-8")
    path = tmp_path / "Caddyfile"
    path.write_text(source, encoding="utf-8")
    assert verify_caddyfile(path) == {
        "status": "edge_contract_verified",
        "public_api_prefix": "/api/v1/*",
        "public_metrics_path": "/internal/metrics",
        "direct_metrics_exposed": False,
        "ready_via_console_internal_probe": True,
        "runtime_unhealthy_upstreams_ejected": True,
        "client_forwarding_source": "edge_remote_host",
    }


@pytest.mark.parametrize(
    "before,after",
    [
        ("admin off", "admin localhost:2019"),
        ("handle /internal/metrics", "handle /metrics"),
        ("rewrite * /metrics", "rewrite * /api/v1/metrics"),
        ("@api path /api/v1/*", "@api path /api/*"),
        ("reverse_proxy @api api:4000", "reverse_proxy @api console:3000"),
        ("reverse_proxy console:3000", "reverse_proxy api:4000"),
        ("health_interval 10s", "health_interval 10m"),
        ("health_timeout 2s", "health_timeout 20s"),
        ("health_fails 2", "health_fails 200"),
        ("health_passes 2", "health_passes 200"),
        ("encode zstd gzip", "handle /ready { respond 200 }"),
        (
            "header_up X-Forwarded-For {remote_host}",
            "header_up X-Forwarded-For {http.request.header.X-Forwarded-For}",
        ),
        ('Referrer-Policy "no-referrer"', 'Referrer-Policy "unsafe-url"'),
    ],
)
def test_caddy_route_or_security_mutations_fail_closed(
    tmp_path: Path, before: str, after: str
) -> None:
    source = (ROOT / "deploy/production/Caddyfile").read_text(encoding="utf-8")
    assert before in source
    path = tmp_path / "Caddyfile"
    path.write_text(source.replace(before, after, 1), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_caddyfile(path)


def test_cross_browser_python_requirements_are_linux_hash_locked() -> None:
    lock = (ROOT / "deploy/production/playwright-requirements.lock").read_text(
        encoding="utf-8"
    )
    assert "Ubuntu x86_64 / CPython 3.11" in lock
    assert lock.count("==") == 4
    assert lock.count("--hash=sha256:") == 4
    assert (
        "playwright==1.60.0" in lock
        and "1c2bfae7884fb3fb05b853290eab8f343d524e5016f2f1def702acbbdf14c93e"
        in lock
    )
    assert (
        "greenlet==3.3.2" in lock
        and "8e2cd90d413acbf5e77ae41e5d3c9b3ac1d011a756d7284d7f3f2b806bbd6358"
        in lock
    )
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "pip install --require-hashes -r deploy/production/playwright-requirements.lock" in workflow


def test_ci_supply_chain_and_production_image_gate_are_commit_or_digest_pinned() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    uses = re.findall(r"(?m)^\s*- uses:\s*([^\s#]+)", workflow)
    assert uses
    assert all(re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", value) for value in uses)
    service_images = re.findall(r"(?m)^\s+image:\s*([^\s]+)", workflow)
    assert service_images
    assert all(re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", value) for value in service_images)
    for marker in (
        "production-images:",
        "docker build --pull",
        "teacher_agent_gateway_worker --self-check",
        "teacher_agent_safeguarding_supervisor --self-check",
        "docker compose -f deploy/production/compose.yaml config",
        "config --format json > /tmp/teachlab-rendered-compose.json",
        "--caddyfile deploy/production/Caddyfile",
        "TEACHLAB_TEACHER_ENTITLEMENT_DIRECTORY_URL:",
        "TEACHLAB_SAFEGUARDING_LOCALE:",
        "TEACHLAB_SAFEGUARDING_DISPATCH_URL:",
        "TEACHLAB_SAFEGUARDING_DISPATCH_POLICY_VERSION:",
        "TEACHLAB_SAFEGUARDING_RETENTION_POLICY_VERSION:",
        "TEACHLAB_SAFEGUARDING_RETENTION_MINIMUM_CLOSED_AGE_SECONDS:",
        "TEACHLAB_SAFEGUARDING_RETENTION_MAXIMUM_CASES_PER_RUN:",
        "TEACHLAB_SAFEGUARDING_RETENTION_DEPLOYMENT_CONTEXT_SHA256:",
    ):
        assert marker in workflow
    assert "cyclonedx-bom" not in workflow
    assert "generate_python_environment_sbom.py" in workflow
    assert "pip install --upgrade pip" not in workflow


def test_local_python_environment_sbom_is_deterministic_and_honest(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_sbom(first)
    write_sbom(second)
    assert first.read_bytes() == second.read_bytes()
    receipt = build_sbom()
    assert receipt["bomFormat"] == "CycloneDX"
    assert receipt["specVersion"] == "1.5"
    assert receipt["components"]
    assert "inventory_only_not_vulnerability_or_provenance_attestation" in first.read_text(
        encoding="utf-8"
    )
