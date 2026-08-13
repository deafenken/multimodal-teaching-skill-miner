#!/usr/bin/env python3
"""Fail-closed static verifier for the rendered single-replica deployment."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


_DIGEST = re.compile(r"^[A-Za-z0-9./:_-]+@sha256:[0-9a-f]{64}$")
_SERVICE_HEADER = re.compile(r"(?m)^  ([a-z][a-z0-9_-]{0,63}):\s*$")


def _service_blocks(text: str) -> dict[str, str]:
    marker = "services:\n"
    if text.count(marker) != 1:
        raise ValueError("deployment must contain one services mapping")
    start = text.index(marker) + len(marker)
    end_match = re.search(r"(?m)^[A-Za-z][A-Za-z0-9_-]*:\s*$", text[start:])
    section = text[start : start + end_match.start() if end_match else len(text)]
    headers = list(_SERVICE_HEADER.finditer(section))
    blocks: dict[str, str] = {}
    for index, header in enumerate(headers):
        name = header.group(1)
        if name in blocks:
            raise ValueError("deployment service names must be unique")
        stop = headers[index + 1].start() if index + 1 < len(headers) else len(section)
        blocks[name] = section[header.start() : stop]
    if set(blocks) != {"edge", "console", "api"}:
        raise ValueError("deployment must contain exactly edge, console, and api")
    return blocks


def _line(block: str, pattern: str) -> bool:
    return re.search(rf"(?m)^{pattern}$", block) is not None


def _verify_service(name: str, block: str) -> str:
    image_match = re.search(r"(?m)^    image:\s*[\"']?([^\s\"']+)", block)
    if not image_match or not _DIGEST.fullmatch(image_match.group(1)):
        raise ValueError(f"{name} image must use an exact sha256 digest")
    required_lines = (
        r"    restart: unless-stopped",
        r"    read_only: true",
        r"    cap_drop: \[ALL\]",
        r"    security_opt: \[no-new-privileges:true\]",
        r"    user: [\"']?[1-9][0-9]*:[1-9][0-9]*[\"']?",
        r"    init: true",
        r"    healthcheck:",
        r"      retries: [1-9][0-9]*",
        r"    deploy:",
        r"      replicas: 1",
        r"      resources:",
        r"        limits: \{cpus: [\"'][0-9.]+[\"'], memory: [0-9]+[MG]\}",
        r"        reservations: \{cpus: [\"'][0-9.]+[\"'], memory: [0-9]+[MG]\}",
    )
    if any(not _line(block, pattern) for pattern in required_lines):
        raise ValueError(f"{name} is missing a hardened service-scoped contract")
    has_ports = _line(block, r"    ports:")
    if (name == "edge") != has_ports:
        raise ValueError("only edge may publish host ports")
    return image_match.group(1)


def verify_rendered_compose(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    if len(text.encode("utf-8")) > 2_000_000:
        raise ValueError("rendered deployment is unexpectedly large")
    for unresolved in (
        "${",
        "TEACHLAB_SESSION_SECRET=",
        "OIDC_CLIENT_SECRET=",
        "HARNESS_SCOPE_SECRET=",
        "ACCOUNT_SCOPE_SECRET=",
        "ACCOUNT_IDENTITY_NAMESPACE_SECRET=",
        "ACCOUNT_DELETION_STATUS_SECRET=",
        "ACCOUNT_CACHE_SCOPE_SECRET=",
        "TEACHER_ENTITLEMENT_DIRECTORY_BEARER_SECRET=",
        "TEACHER_ENTITLEMENT_BINDING_KEY=",
        "TEACHER_ENTITLEMENT_RECEIPT_KEY=",
        "HARNESS_SAFEGUARDING_DISPATCH_BEARER_SECRET=",
        "HARNESS_SAFEGUARDING_RETENTION_AUTHORITY_SECRET=",
    ):
        if unresolved in text:
            raise ValueError(
                "rendered deployment contains unresolved or inline secret material"
            )

    services = _service_blocks(text)
    for name in ("edge", "console", "api"):
        _verify_service(name, services[name])
    if len(re.findall(r"(?m)^\s+ports:\s*$", text)) != 1:
        raise ValueError("exactly the edge service must publish host ports")
    api_required = (
        'HARNESS_GATEWAY_ENABLED: "true"',
        "HARNESS_WORKER_ROOT: /var/lib/teachlab/scopes",
        "HARNESS_WORKER_BACKEND: deepseek",
        'HARNESS_WORKER_FILESYSTEM_ISOLATION_REQUIRED: "true"',
        'HARNESS_WORKER_RLIMIT_CORE_BYTES: "0"',
        'HARNESS_WORKER_RLIMIT_NOFILE: "256"',
        'HARNESS_WORKER_RLIMIT_FSIZE_BYTES: "536870912"',
        'HARNESS_WORKER_RLIMIT_AS_BYTES: "1610612736"',
        "pids_limit: 256",
        "AUTH_MODE: oidc",
        "DATA_BACKEND: postgres",
        "PG_SSL_MODE: verify-full",
        "SESSION_SECRET_FILE:",
        "OIDC_CLIENT_SECRET_FILE:",
        "OIDC_TRANSACTION_SECRET_FILE:",
        "OIDC_ACCOUNT_AAL2_ACR_VALUES:",
        "TEACHER_ENTITLEMENT_DIRECTORY_URL:",
        "TEACHER_ENTITLEMENT_DIRECTORY_BEARER_SECRET_FILE:",
        "TEACHER_ENTITLEMENT_BINDING_KEY_FILE:",
        "TEACHER_ENTITLEMENT_RECEIPT_KEY_FILE:",
        "TEACHER_ENTITLEMENT_POLICY_ID:",
        "TEACHER_ENTITLEMENT_POLICY_VERSION:",
        "TEACHER_ENTITLEMENT_FRESHNESS_TTL_MS:",
        "TEACHER_ENTITLEMENT_CACHE_TTL_MS:",
        "TEACHER_ENTITLEMENT_PROVIDER_TIMEOUT_MS:",
        "TEACHER_ENTITLEMENT_MAX_RESPONSE_BYTES:",
        "TEACHER_ENTITLEMENT_MAX_CLOCK_SKEW_MS:",
        "TEACHER_ENTITLEMENT_MAX_CACHE_ENTRIES:",
        "TEACHER_ENTITLEMENT_MIN_ASSURANCE_LEVEL:",
        "TEACHER_AUTHORITY_ROLES:",
        "SAFEGUARDING_AUTHORITY_ROLES: safeguarding",
        "ACCOUNT_SCOPE_SECRET_FILE:",
        "ACCOUNT_IDENTITY_NAMESPACE_SECRET_FILE:",
        "ACCOUNT_DELETION_STATUS_SECRET_FILE:",
        "ACCOUNT_CACHE_SCOPE_SECRET_FILE:",
        "HARNESS_SCOPE_SECRET_FILE:",
        "HARNESS_PROVIDER_API_KEY_FILE:",
        "HARNESS_PROVIDER_POLICY_ID:",
        "HARNESS_PROVIDER_POLICY_VERSION:",
        "HARNESS_PROVIDER_PROCESSING_REGION:",
        "HARNESS_PROVIDER_RETENTION_DAYS:",
        "HARNESS_PROVIDER_DELETION_STATUS:",
        "HARNESS_PROVIDER_DOCUMENTATION_URL:",
        "HARNESS_SAFEGUARDING_LOCALE:",
        "HARNESS_SAFEGUARDING_DISPATCH_URL:",
        "HARNESS_SAFEGUARDING_DISPATCH_BEARER_SECRET_FILE:",
        "HARNESS_SAFEGUARDING_DISPATCH_POLICY_VERSION:",
        "HARNESS_SAFEGUARDING_DISPATCH_TIMEOUT_MS:",
        "HARNESS_SAFEGUARDING_DISPATCH_MAX_RESPONSE_BYTES:",
        "HARNESS_SAFEGUARDING_RETENTION_POLICY_VERSION:",
        "HARNESS_SAFEGUARDING_RETENTION_MINIMUM_CLOSED_AGE_SECONDS:",
        "HARNESS_SAFEGUARDING_RETENTION_MAXIMUM_CASES_PER_RUN:",
        "HARNESS_SAFEGUARDING_RETENTION_AUTHORITY_SECRET_FILE: /run/teachlab/safeguarding-retention-authority",
        "HARNESS_SAFEGUARDING_RETENTION_DEPLOYMENT_CONTEXT_SHA256:",
        "REMOTE_SUBJECT_POLICY_ID:",
        "REMOTE_SUBJECT_POLICY_VERSION:",
        "METRICS_TOKEN_FILE:",
        "fetch('http://127.0.0.1:4000/ready')",
    )
    missing = [value for value in api_required if value not in services["api"]]
    if missing:
        raise ValueError("api is missing hardened runtime contracts")
    if ":/run/teachlab:ro" not in services["api"]:
        raise ValueError("api must mount one private secrets directory")
    if len(re.findall(r":/run/teachlab(?:/[^:\s]+)?:ro", services["api"])) != 1:
        raise ValueError("api must not use individual secret-file bind mounts")
    console_required = (
        "NEXT_PUBLIC_TEACHLAB_AUTH_MODE: oidc",
        "TEACHLAB_HARNESS_MODE: authenticated_apps_api",
        "TEACHLAB_APPS_API_URL:",
    )
    if any(value not in services["console"] for value in console_required):
        raise ValueError("console is missing authenticated production contracts")
    if "cap_add: [NET_BIND_SERVICE]" not in services["edge"]:
        raise ValueError("edge lacks its one explicit network-bind capability")
    return {
        "status": "deployment_contract_verified",
        "service_count": len(services),
        "images_digest_pinned": True,
        "api_replicas": 1,
        "api_container_pid_limit": 256,
        "distributed_worker_scheduler": False,
        "external_image_signing_verified": False,
    }


def _caddy_block(text: str, header: str) -> str:
    """Return one exact Caddy block, rejecting missing/duplicate/unclosed blocks."""

    pattern = re.compile(rf"(?m)^[ \t]*{re.escape(header)}[ \t]*\{{[ \t]*$")
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"Caddyfile must contain exactly one {header!r} block")
    # A header may itself contain a Caddy placeholder (for example
    # {$TEACHLAB_PUBLIC_HOST}); the block delimiter is the final opening brace.
    opening = text.rfind("{", matches[0].start(), matches[0].end())
    depth = 0
    quoted = False
    escaped = False
    for index in range(opening, len(text)):
        character = text[index]
        if escaped:
            escaped = False
            continue
        if character == "\\" and quoted:
            escaped = True
            continue
        if character == '"':
            quoted = not quoted
            continue
        if quoted:
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[matches[0].start() : index + 1]
            if depth < 0:
                break
    raise ValueError(f"Caddyfile contains an unclosed {header!r} block")


def _exact_caddy_line(block: str, line: str, *, count: int = 1) -> None:
    actual = len(re.findall(rf"(?m)^[ \t]*{re.escape(line)}[ \t]*$", block))
    if actual != count:
        raise ValueError(f"Caddyfile requires {count} exact {line!r} line(s)")


def verify_caddyfile(path: Path) -> dict[str, object]:
    """Verify the public edge routes rather than trusting whole-file substrings."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("Caddyfile must be a regular non-symlink file")
    text = path.read_text(encoding="utf-8")
    if not text or len(text.encode("utf-8")) > 256_000:
        raise ValueError("Caddyfile has an invalid size")
    _exact_caddy_line(text, "admin off")
    site = _caddy_block(text, "{$TEACHLAB_PUBLIC_HOST}")
    metrics = _caddy_block(site, "handle /internal/metrics")
    api = _caddy_block(site, "reverse_proxy @api api:4000")
    console = _caddy_block(site, "reverse_proxy console:3000")
    security_headers = _caddy_block(site, "header")

    _exact_caddy_line(site, "@api path /api/v1/*")
    _exact_caddy_line(metrics, "rewrite * /metrics")
    _exact_caddy_line(metrics, "header_up X-Forwarded-For {remote_host}")
    _exact_caddy_line(metrics, "header_up X-Forwarded-Proto {scheme}")
    _exact_caddy_line(metrics, "header_up X-Forwarded-Host {host}")
    _exact_caddy_line(api, "header_up X-Forwarded-For {remote_host}")
    _exact_caddy_line(api, "header_up X-Forwarded-Proto https")
    _exact_caddy_line(api, "header_up X-Forwarded-Host {host}")
    _exact_caddy_line(console, "header_up X-Forwarded-Proto https")
    _exact_caddy_line(console, "header_up X-Forwarded-Host {host}")
    for proxy in (metrics, api, console):
        for line in (
            "health_uri /ready",
            "health_interval 10s",
            "health_timeout 2s",
            "health_fails 2",
            "health_passes 2",
        ):
            _exact_caddy_line(proxy, line)
    for line in (
        "-Server",
        'Strict-Transport-Security "max-age=31536000; includeSubDomains"',
        'X-Content-Type-Options "nosniff"',
        'Referrer-Policy "no-referrer"',
    ):
        _exact_caddy_line(security_headers, line)

    # /ready intentionally reaches Console, which performs exactly one
    # server-side probe through its separately configured internal API base.
    # Metrics retain the API bearer check and direct /metrics is not routed.
    forbidden = (
        "handle /metrics",
        "handle /ready",
        "handle_path /ready",
        "path /ready",
        "path /api/*",
        "path /*",
        "trusted_proxies",
        "{http.request.header.X-Forwarded-For}",
    )
    if any(value in site for value in forbidden):
        raise ValueError("Caddyfile contains a broadened or spoofable public route")
    if len(re.findall(r"(?m)^[ \t]*reverse_proxy[ \t]", site)) != 3:
        raise ValueError("Caddyfile must contain exactly metrics, API, and Console proxies")
    return {
        "status": "edge_contract_verified",
        "public_api_prefix": "/api/v1/*",
        "public_metrics_path": "/internal/metrics",
        "direct_metrics_exposed": False,
        "ready_via_console_internal_probe": True,
        "runtime_unhealthy_upstreams_ejected": True,
        "client_forwarding_source": "edge_remote_host",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose", type=Path, required=True)
    parser.add_argument("--caddyfile", type=Path, required=True)
    args = parser.parse_args()
    receipt = {
        "compose": verify_rendered_compose(args.compose),
        "edge": verify_caddyfile(args.caddyfile),
    }
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
