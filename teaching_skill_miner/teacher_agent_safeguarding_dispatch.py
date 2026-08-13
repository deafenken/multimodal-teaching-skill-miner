"""HTTPS dispatcher for the content-free safeguarding outbox.

The endpoint, bearer credential, and opaque routing locator are trusted worker
bootstrap configuration.  No request body field can select or override them.
Dispatch is deliberately idempotent by ``delivery_id`` and does not mark the
durable outbox acknowledged: the receiving safeguarding workflow must call the
separate, freshly-authorized acknowledgement API after it owns the case.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any, Protocol
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)


SAFEGUARDING_DISPATCH_SCHEMA = "teaching_skill_miner.safeguarding_dispatch.v1"
SAFEGUARDING_DISPATCH_ACK_SCHEMA = "teaching_skill_miner.safeguarding_dispatch_ack.v1"
SAFEGUARDING_DISPATCH_READINESS_SCHEMA = (
    "teaching_skill_miner.safeguarding_dispatch_readiness.v1"
)

_DELIVERY_ID = re.compile(r"sge_[0-9a-f]{24}")
_CASE_ID = re.compile(r"sgc_[0-9a-f]{24}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,2047}")
_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z")
_MAX_REQUEST_BYTES = 16 * 1024


class SafeguardingDispatchError(RuntimeError):
    """Stable, content-free dispatcher failure."""


class SafeguardingDispatcher(Protocol):
    configured: bool
    route_locator_sha256: str | None
    queue_sha256: str | None

    def dispatch(self, row: Mapping[str, Any]) -> Mapping[str, Any]: ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SafeguardingDispatchError(
            "safeguarding dispatch payload is invalid"
        ) from exc


def _canonical_https_endpoint(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SafeguardingDispatchError(
            "safeguarding dispatcher configuration is invalid"
        )
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.port not in {None, 443}
        or value.endswith("/")
    ):
        raise SafeguardingDispatchError(
            "safeguarding dispatcher configuration is invalid"
        )
    return value


def _validated_row(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SafeguardingDispatchError("safeguarding dispatch row is invalid")
    required = {
        "case_id",
        "case_version",
        "scope_sha256",
        "category",
        "severity",
        "observed_at_utc",
        "content_sha256",
        "delivery_id",
        "delivery_status",
        "queue_sha256",
        "sla_due_at_utc",
    }
    if set(value) != required:
        raise SafeguardingDispatchError("safeguarding dispatch row is invalid")
    row = deepcopy(dict(value))
    if (
        not isinstance(row["case_id"], str)
        or _CASE_ID.fullmatch(row["case_id"]) is None
        or isinstance(row["case_version"], bool)
        or not isinstance(row["case_version"], int)
        or row["case_version"] < 1
        or row["category"]
        not in {
            "self_harm",
            "abuse_disclosure",
            "bullying_disclosure",
            "minor_sexual_content",
            "urgent_medical",
            "harm_to_others",
            "other_safeguarding",
        }
        or row["severity"] not in {"elevated", "high", "urgent"}
        or row["delivery_status"] not in {"pending", "overdue"}
        or not isinstance(row["delivery_id"], str)
        or _DELIVERY_ID.fullmatch(row["delivery_id"]) is None
    ):
        raise SafeguardingDispatchError("safeguarding dispatch row is invalid")
    for field in ("scope_sha256", "content_sha256", "queue_sha256"):
        if not isinstance(row[field], str) or _DIGEST.fullmatch(row[field]) is None:
            raise SafeguardingDispatchError("safeguarding dispatch row is invalid")
    for field in ("observed_at_utc", "sla_due_at_utc"):
        if (
            not isinstance(row[field], str)
            or _UTC_TIMESTAMP.fullmatch(row[field]) is None
        ):
            raise SafeguardingDispatchError("safeguarding dispatch row is invalid")
    return row


@dataclass(frozen=True, slots=True)
class UnconfiguredSafeguardingDispatcher:
    """Explicit fail-closed dispatcher used when no staff route is installed."""

    configured: bool = False
    route_locator_sha256: str | None = None
    queue_sha256: str | None = None

    def dispatch(self, _row: Mapping[str, Any]) -> Mapping[str, Any]:
        raise SafeguardingDispatchError("safeguarding dispatcher is unavailable")


class HttpsSafeguardingDispatcher:
    """Send one exact, minimized outbox row to a configured HTTPS receiver."""

    configured = True

    def __init__(
        self,
        *,
        endpoint: str,
        bearer_secret: str,
        route_locator: str,
        policy_version: str,
        timeout_seconds: float = 5.0,
        maximum_response_bytes: int = 32 * 1024,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.endpoint = _canonical_https_endpoint(endpoint)
        if (
            not isinstance(bearer_secret, str)
            or bearer_secret != bearer_secret.strip()
            or len(bearer_secret.encode("utf-8")) < 32
            or any(character in bearer_secret for character in "\r\n\x00")
            or not isinstance(route_locator, str)
            or _SAFE_TOKEN.fullmatch(route_locator) is None
            or not isinstance(policy_version, str)
            or _SAFE_TOKEN.fullmatch(policy_version) is None
            or not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not 0.1 <= float(timeout_seconds) <= 30.0
            or not isinstance(maximum_response_bytes, int)
            or isinstance(maximum_response_bytes, bool)
            or not 256 <= maximum_response_bytes <= 1024 * 1024
        ):
            raise SafeguardingDispatchError(
                "safeguarding dispatcher configuration is invalid"
            )
        self._bearer_secret = bearer_secret
        self._route_locator = route_locator
        self.policy_version = policy_version
        self.timeout_seconds = float(timeout_seconds)
        self.maximum_response_bytes = maximum_response_bytes
        self.route_locator_sha256 = sha256(route_locator.encode("utf-8")).hexdigest()
        self.queue_sha256 = sha256(
            _canonical_json_bytes(
                {
                    "schema": "teaching_skill_miner.safeguarding_queue.v1",
                    "endpoint": endpoint,
                    "policy_version": policy_version,
                    "route_locator_sha256": self.route_locator_sha256,
                }
            )
        ).hexdigest()
        self._open = opener or build_opener(ProxyHandler({}), _NoRedirect()).open

    def supervisor_route_locator(self) -> str:
        """Return the opaque route only to the private delivery supervisor.

        The locator is never included in public status or browser responses;
        persisting it beside the content-free outbox lets delivery survive the
        learner worker lifecycle without exposing a tenant identity.
        """

        return self._route_locator

    def probe_readiness(self) -> Mapping[str, Any]:
        """Authenticate to the receiver without creating a case or sending data."""

        readiness_request = Request(
            self.endpoint,
            method="GET",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._bearer_secret}",
                "Cache-Control": "no-store",
                "User-Agent": "TeachLab-Safeguarding-Readiness/1",
            },
        )
        try:
            with self._open(
                readiness_request, timeout=self.timeout_seconds
            ) as response:
                status = int(response.getcode())
                content_type = str(response.headers.get("Content-Type", ""))
                declared = response.headers.get("Content-Length")
                if (
                    status != 200
                    or content_type.split(";", 1)[0].strip().lower()
                    != "application/json"
                    or (
                        declared is not None
                        and (
                            not str(declared).isdigit()
                            or int(str(declared)) > self.maximum_response_bytes
                        )
                    )
                ):
                    raise SafeguardingDispatchError(
                        "safeguarding dispatcher readiness failed"
                    )
                raw = response.read(self.maximum_response_bytes + 1)
        except SafeguardingDispatchError:
            raise
        except Exception as exc:
            raise SafeguardingDispatchError(
                "safeguarding dispatcher readiness failed"
            ) from exc
        if len(raw) > self.maximum_response_bytes:
            raise SafeguardingDispatchError("safeguarding dispatcher readiness failed")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SafeguardingDispatchError(
                "safeguarding dispatcher readiness failed"
            ) from exc
        if (
            not isinstance(value, Mapping)
            or set(value) != {"schema", "status", "policy_version"}
            or value.get("schema") != SAFEGUARDING_DISPATCH_READINESS_SCHEMA
            or value.get("status") != "ready"
            or value.get("policy_version") != self.policy_version
        ):
            raise SafeguardingDispatchError("safeguarding dispatcher readiness failed")
        return {
            "schema": SAFEGUARDING_DISPATCH_READINESS_SCHEMA,
            "status": "ready",
            "policy_version_validated": True,
            "receiver_network_validated": True,
            "credential_validated": True,
            "learner_content_sent": False,
            "case_created": False,
        }

    def dispatch(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        row = _validated_row(value)
        if row["queue_sha256"] != self.queue_sha256:
            raise SafeguardingDispatchError(
                "safeguarding dispatch queue binding failed"
            )
        payload = {
            "schema": SAFEGUARDING_DISPATCH_SCHEMA,
            "policy_version": self.policy_version,
            "route_locator": self._route_locator,
            **row,
        }
        encoded = _canonical_json_bytes(payload)
        if len(encoded) > _MAX_REQUEST_BYTES:
            raise SafeguardingDispatchError("safeguarding dispatch payload is invalid")
        request = Request(
            self.endpoint,
            data=encoded,
            method="POST",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._bearer_secret}",
                "Content-Type": "application/json",
                "Idempotency-Key": str(row["delivery_id"]),
                "User-Agent": "TeachLab-Safeguarding-Dispatcher/1",
            },
        )
        try:
            with self._open(request, timeout=self.timeout_seconds) as response:
                status = int(response.getcode())
                content_type = str(response.headers.get("Content-Type", ""))
                declared = response.headers.get("Content-Length")
                if (
                    status not in {200, 202}
                    or content_type.split(";", 1)[0].strip().lower()
                    != "application/json"
                    or (
                        declared is not None
                        and (
                            not str(declared).isdigit()
                            or int(str(declared)) > self.maximum_response_bytes
                        )
                    )
                ):
                    raise SafeguardingDispatchError("safeguarding dispatch failed")
                raw = response.read(self.maximum_response_bytes + 1)
        except SafeguardingDispatchError:
            raise
        except Exception as exc:
            # Adapters used by deployments (or tests) are not allowed to leak
            # network/library exception text into the staff API surface.
            raise SafeguardingDispatchError("safeguarding dispatch failed") from exc
        if len(raw) > self.maximum_response_bytes:
            raise SafeguardingDispatchError("safeguarding dispatch failed")
        try:
            acknowledgement = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SafeguardingDispatchError("safeguarding dispatch failed") from exc
        if (
            not isinstance(acknowledgement, Mapping)
            or set(acknowledgement)
            != {"schema", "status", "delivery_id", "route_locator_sha256"}
            or acknowledgement.get("schema") != SAFEGUARDING_DISPATCH_ACK_SCHEMA
            or acknowledgement.get("status") != "accepted"
            or acknowledgement.get("delivery_id") != row["delivery_id"]
            or acknowledgement.get("route_locator_sha256") != self.route_locator_sha256
        ):
            raise SafeguardingDispatchError("safeguarding dispatch failed")
        return {
            "schema": SAFEGUARDING_DISPATCH_ACK_SCHEMA,
            "status": "accepted",
            "delivery_id": row["delivery_id"],
            "route_locator_sha256": self.route_locator_sha256,
            "raw_learner_text_sent": False,
            "durable_delivery_acknowledged": False,
        }


__all__ = [
    "HttpsSafeguardingDispatcher",
    "SAFEGUARDING_DISPATCH_ACK_SCHEMA",
    "SAFEGUARDING_DISPATCH_READINESS_SCHEMA",
    "SAFEGUARDING_DISPATCH_SCHEMA",
    "SafeguardingDispatchError",
    "SafeguardingDispatcher",
    "UnconfiguredSafeguardingDispatcher",
]
