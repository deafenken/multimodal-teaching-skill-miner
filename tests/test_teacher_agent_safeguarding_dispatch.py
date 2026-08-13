from __future__ import annotations

from hashlib import sha256
import json

import pytest

from teaching_skill_miner.teacher_agent_safeguarding_dispatch import (
    HttpsSafeguardingDispatcher,
    SAFEGUARDING_DISPATCH_ACK_SCHEMA,
    SAFEGUARDING_DISPATCH_READINESS_SCHEMA,
    SAFEGUARDING_DISPATCH_SCHEMA,
    SafeguardingDispatchError,
    UnconfiguredSafeguardingDispatcher,
)


class Response:
    def __init__(self, payload: dict, *, status: int = 202) -> None:
        self.payload = json.dumps(payload, separators=(",", ":")).encode()
        self.status = status
        self.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(self.payload)),
        }

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def getcode(self) -> int:
        return self.status

    def read(self, maximum: int) -> bytes:
        return self.payload[:maximum]


def _dispatcher(calls: list, *, bad_ack: bool = False):
    route = "sgr1_k1_" + "A" * 100
    route_sha256 = sha256(route.encode()).hexdigest()

    def open_request(request, *, timeout):
        calls.append((request, timeout))
        return Response(
            {
                "schema": SAFEGUARDING_DISPATCH_ACK_SCHEMA,
                "status": "accepted",
                "delivery_id": "sge_" + ("2" if bad_ack else "1") * 24,
                "route_locator_sha256": route_sha256,
            }
        )

    return HttpsSafeguardingDispatcher(
        endpoint="https://safeguarding.school.example/v1/cases",
        bearer_secret="dispatcher-private-bearer-secret-at-least-32-bytes",
        route_locator=route,
        policy_version="school-policy-v1",
        timeout_seconds=1.5,
        maximum_response_bytes=8192,
        opener=open_request,
    )


def _row(dispatcher: HttpsSafeguardingDispatcher) -> dict:
    return {
        "case_id": "sgc_" + "1" * 24,
        "case_version": 3,
        "scope_sha256": "2" * 64,
        "category": "self_harm",
        "severity": "urgent",
        "observed_at_utc": "2026-08-12T04:00:00Z",
        "content_sha256": "3" * 64,
        "delivery_id": "sge_" + "1" * 24,
        "delivery_status": "overdue",
        "queue_sha256": dispatcher.queue_sha256,
        "sla_due_at_utc": "2026-08-12T04:15:00Z",
    }


def test_https_dispatch_is_exact_hash_only_and_idempotent_by_delivery_id() -> None:
    calls: list = []
    dispatcher = _dispatcher(calls)
    result = dispatcher.dispatch(_row(dispatcher))
    assert result["status"] == "accepted"
    assert result["durable_delivery_acknowledged"] is False
    request, timeout = calls[0]
    assert timeout == 1.5
    assert request.get_header("Authorization").startswith("Bearer ")
    assert request.get_header("Idempotency-key") == "sge_" + "1" * 24
    payload = json.loads(request.data)
    assert payload["schema"] == SAFEGUARDING_DISPATCH_SCHEMA
    assert set(payload) == {
        "schema",
        "policy_version",
        "route_locator",
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
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "learner_text" not in serialized
    assert "message" not in serialized
    assert "dispatcher-private-bearer" not in serialized


def test_dispatch_ack_mismatch_outage_and_unconfigured_state_fail_closed() -> None:
    dispatcher = _dispatcher([], bad_ack=True)
    with pytest.raises(SafeguardingDispatchError, match="dispatch failed"):
        dispatcher.dispatch(_row(dispatcher))
    with pytest.raises(SafeguardingDispatchError, match="unavailable"):
        UnconfiguredSafeguardingDispatcher().dispatch({})

    def private_adapter_failure(*_args, **_kwargs):
        raise RuntimeError("private host and library detail")

    failed = HttpsSafeguardingDispatcher(
        endpoint="https://safeguarding.school.example/v1/cases",
        bearer_secret="dispatcher-private-bearer-secret-at-least-32-bytes",
        route_locator="sgr1_k1_" + "A" * 100,
        policy_version="school-policy-v1",
        opener=private_adapter_failure,
    )
    with pytest.raises(
        SafeguardingDispatchError, match="^safeguarding dispatch failed$"
    ):
        failed.dispatch(_row(failed))


def test_receiver_readiness_is_authenticated_content_free_and_exact() -> None:
    calls: list = []

    def open_request(request, *, timeout):
        calls.append((request, timeout))
        return Response(
            {
                "schema": SAFEGUARDING_DISPATCH_READINESS_SCHEMA,
                "status": "ready",
                "policy_version": "school-policy-v1",
            },
            status=200,
        )

    dispatcher = HttpsSafeguardingDispatcher(
        endpoint="https://safeguarding.school.example/v1/cases",
        bearer_secret="dispatcher-private-bearer-secret-at-least-32-bytes",
        route_locator="sgr1_k1_" + "A" * 100,
        policy_version="school-policy-v1",
        opener=open_request,
    )
    result = dispatcher.probe_readiness()
    request, _timeout = calls[0]
    assert request.get_method() == "GET"
    assert request.data is None
    assert request.get_header("Authorization").startswith("Bearer ")
    assert result["receiver_network_validated"] is True
    assert result["credential_validated"] is True
    assert result["learner_content_sent"] is False
    assert result["case_created"] is False
    assert "dispatcher-private-bearer" not in json.dumps(result)

    def wrong_policy(*_args, **_kwargs):
        return Response(
            {
                "schema": SAFEGUARDING_DISPATCH_READINESS_SCHEMA,
                "status": "ready",
                "policy_version": "wrong-policy",
            },
            status=200,
        )

    dispatcher._open = wrong_policy
    with pytest.raises(
        SafeguardingDispatchError,
        match="^safeguarding dispatcher readiness failed$",
    ):
        dispatcher.probe_readiness()


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://safeguarding.school.example/v1/cases",
        "https://user:pass@safeguarding.school.example/v1/cases",
        "https://safeguarding.school.example/v1/cases?tenant=x",
        "https://safeguarding.school.example:444/v1/cases",
        "https://safeguarding.school.example/v1/cases/",
    ],
)
def test_dispatch_endpoint_is_canonical_https(endpoint: str) -> None:
    with pytest.raises(SafeguardingDispatchError, match="configuration"):
        HttpsSafeguardingDispatcher(
            endpoint=endpoint,
            bearer_secret="dispatcher-private-bearer-secret-at-least-32-bytes",
            route_locator="sgr1_k1_" + "A" * 100,
            policy_version="school-policy-v1",
        )
