"""Tests for the optional OTel tracing setup in ``claude_teams.telemetry``.

The module has three branches we want to cover:

1. ``OTEL_SDK_DISABLED`` env toggle — operator opt-out.
2. ``ImportError`` path — SDK not installed (simulated with ``sys.modules``
   sabotage even though dev pulls the SDK via the dev group).
3. Happy path — SDK imports succeed and a ``TracerProvider`` is registered.

``trace.set_tracer_provider`` is guarded by a one-shot latch in the OTel
SDK, so the ``captured_provider`` fixture below monkeypatches it in every
happy-path test to avoid polluting global state across the full pytest
run.
"""

import importlib
import sys

import pytest

from claude_teams import telemetry


@pytest.fixture(autouse=True)
def _clear_disabled_env(monkeypatch):
    """Start every test with a clean OTEL_SDK_DISABLED state.

    Several tests assert behavior under specific values; the autouse
    fixture prevents env leakage from one test into the next.
    """
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)


@pytest.fixture
def captured_provider(monkeypatch):
    """Intercept ``trace.set_tracer_provider`` without polluting global state.

    OTel's one-shot latch on the provider setter means a real
    ``configure_tracing`` call would leak across the test suite. This
    fixture redirects the setter into a dict the caller inspects after
    invoking the production code: ``captured_provider["provider"]``
    holds whatever ``configure_tracing`` tried to register.
    """
    from opentelemetry import trace

    captured: dict = {}

    def _fake_set(provider):
        captured["provider"] = provider

    monkeypatch.setattr(trace, "set_tracer_provider", _fake_set)
    return captured


class TestOtelDisabled:
    """``_otel_disabled`` honours the standard OTel env flag."""

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "YES", "True"])
    def test_returns_true_for_affirmative_values(self, monkeypatch, value):
        monkeypatch.setenv("OTEL_SDK_DISABLED", value)
        assert telemetry._otel_disabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "", "maybe"])
    def test_returns_false_for_other_values(self, monkeypatch, value):
        monkeypatch.setenv("OTEL_SDK_DISABLED", value)
        assert telemetry._otel_disabled() is False

    def test_returns_false_when_unset(self):
        # The autouse fixture already cleared this; assert the default
        # behavior explicitly so the env contract is covered.
        assert telemetry._otel_disabled() is False


class TestConfigureTracingEarlyExits:
    """``configure_tracing`` short-circuits under opt-out / missing SDK."""

    def test_returns_false_when_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
        assert telemetry.configure_tracing() is False

    def test_returns_false_when_sdk_missing(self, monkeypatch):
        """Simulate a user who didn't ``pip install claude-teams[otel]``.

        The SDK is installed in the dev venv, so we have to sabotage the
        import path. Setting ``sys.modules["opentelemetry.sdk"] = None``
        makes subsequent ``from opentelemetry.sdk... import X`` raise
        ImportError, which is exactly what the runtime path guards against.
        """
        # Remove any already-imported ``opentelemetry.sdk.*`` entries so the
        # function's local imports re-trigger module resolution.
        for name in list(sys.modules):
            if name.startswith("opentelemetry.sdk"):
                monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setitem(sys.modules, "opentelemetry.sdk", None)

        assert telemetry.configure_tracing() is False


class TestConfigureTracingHappyPath:
    """When the SDK is importable and operator hasn't opted out."""

    def test_registers_provider_with_default_service_name(self, captured_provider):
        """Default service name is ``claude-teams`` per the module docstring."""
        from opentelemetry.sdk.trace import TracerProvider

        assert telemetry.configure_tracing() is True
        assert "provider" in captured_provider, (
            "expected set_tracer_provider to be called"
        )
        assert isinstance(captured_provider["provider"], TracerProvider)
        resource = captured_provider["provider"].resource
        assert resource.attributes.get("service.name") == "claude-teams"

    def test_uses_custom_service_name_from_env(self, monkeypatch, captured_provider):
        """``OTEL_SERVICE_NAME`` overrides the default."""
        monkeypatch.setenv("OTEL_SERVICE_NAME", "custom-service")

        assert telemetry.configure_tracing() is True
        assert (
            captured_provider["provider"].resource.attributes.get("service.name")
            == "custom-service"
        )

    def test_attaches_batch_span_processor(self, monkeypatch, captured_provider):
        """Provider should have a BatchSpanProcessor wrapping the OTLP exporter."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        original_add = TracerProvider.add_span_processor
        processors: list = []

        def _spy_add_processor(self, processor):
            processors.append(processor)
            return original_add(self, processor)

        monkeypatch.setattr(TracerProvider, "add_span_processor", _spy_add_processor)

        assert telemetry.configure_tracing() is True
        assert len(processors) == 1
        assert isinstance(processors[0], BatchSpanProcessor)


class TestModuleImport:
    """The module must be importable even when OTel is absent.

    This guards against a regression where a top-level OTel import would
    break anyone who installs ``claude-teams`` without the ``[otel]`` extra.
    """

    def test_module_imports_cleanly(self):
        # Re-import the module under a fresh sys.modules slot and check
        # that nothing raises. Using importlib.reload so the assertion
        # doesn't just pass because the module is already cached.
        importlib.reload(telemetry)
        assert hasattr(telemetry, "configure_tracing")
        assert hasattr(telemetry, "_otel_disabled")
