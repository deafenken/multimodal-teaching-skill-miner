from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent_harness.core import HarnessContractError, HarnessModelResponse
from agent_harness.core.provider_registry import (
    ProviderErrorKind,
    ProviderFailure,
    ProviderModelSpec,
    ProviderRegistry,
    approximate_tokens,
    provider_stream_contract,
)
from agent_harness.core.providers import (
    ProviderCapabilities,
    ProviderStreamEvent,
)


@dataclass
class _Adapter:
    model_spec: ProviderModelSpec

    def plan(self, *_args, **_kwargs):
        return HarnessModelResponse(kind="final", output={"ok": True})

    def classify_error(self, _error: BaseException) -> ProviderFailure:
        return ProviderFailure(
            kind=ProviderErrorKind.UNKNOWN,
            retryable=False,
            safe_code="unknown_provider_failure",
        )


def _spec(model: str, *, context: int, vision: bool = False) -> ProviderModelSpec:
    return ProviderModelSpec(
        provider="test-provider",
        model=model,
        capabilities=ProviderCapabilities(
            provider="test-provider",
            model=model,
            structured_output=True,
            native_stream=True,
            vision=vision,
            cancellation=True,
        ),
        context_window_tokens=context,
        maximum_output_tokens=1_024,
    )


def test_registry_selects_smallest_sufficient_configured_model() -> None:
    registry = ProviderRegistry()
    small = _spec("small", context=8_192)
    large = _spec("large", context=64_000, vision=True)
    registry.register(small, lambda: _Adapter(small))
    registry.register(large, lambda: _Adapter(large))

    assert registry.resolve(minimum_context_tokens=5_000).model_spec.model == "small"
    assert (
        registry.resolve(required_capabilities={"vision"}).model_spec.model == "large"
    )
    with pytest.raises(HarnessContractError, match="no configured provider"):
        registry.resolve(required_capabilities={"web_search"})


def test_factory_model_drift_and_duplicate_registration_fail_closed() -> None:
    registry = ProviderRegistry()
    expected = _spec("expected", context=8_192)
    other = _spec("other", context=8_192)
    registry.register(expected, lambda: _Adapter(other))
    with pytest.raises(HarnessContractError, match="different model spec"):
        registry.resolve(model="expected")
    with pytest.raises(HarnessContractError, match="already registered"):
        registry.register(expected, lambda: _Adapter(expected))


def test_failure_taxonomy_and_token_estimate_are_bounded() -> None:
    assert approximate_tokens("abcd") == 2
    assert approximate_tokens("动态规划") == 4
    ProviderFailure(
        kind=ProviderErrorKind.RATE_LIMIT,
        retryable=True,
        safe_code="provider_rate_limit",
        retry_after_seconds=2.5,
    ).validated()
    with pytest.raises(HarnessContractError, match="retryability"):
        ProviderFailure(
            kind=ProviderErrorKind.AUTHENTICATION,
            retryable=True,
            safe_code="auth_failed",
        ).validated()


def test_stream_conformance_requires_exact_start_and_end() -> None:
    provider_stream_contract(
        [
            ProviderStreamEvent(type="message.start"),
            ProviderStreamEvent(type="message.delta", delta="hello"),
            ProviderStreamEvent(type="message.end"),
        ]
    )
    with pytest.raises(HarnessContractError, match="start"):
        provider_stream_contract([ProviderStreamEvent(type="message.end")])
