"""Optional OpenTelemetry tracing setup for the claude-teams MCP server.

FastMCP emits server spans for every tool/resource/prompt invocation when
an OTel SDK tracer provider is registered globally (see
``fastmcp/server/telemetry.py``). This module wires that provider up so
operators who installed the ``otel`` extra — ``pip install claude-teams[otel]``
— can export traces over OTLP/HTTP without touching server code.

Configuration is read from the standard OTel environment variables:

- ``OTEL_SERVICE_NAME`` (defaults to ``claude-teams``)
- ``OTEL_EXPORTER_OTLP_ENDPOINT`` (e.g. ``http://localhost:4318``)
- ``OTEL_EXPORTER_OTLP_HEADERS`` (for authentication)
- ``OTEL_SDK_DISABLED`` — set to ``true``/``1`` to disable without uninstalling

The OTel imports live inside ``configure_tracing`` so this module stays
importable even when the ``otel`` extra is not installed; a missing SDK
just makes the function a no-op returning ``False``.
"""

import logging
import os

logger = logging.getLogger(__name__)


def _otel_disabled() -> bool:
    return os.environ.get("OTEL_SDK_DISABLED", "").lower() in {"1", "true", "yes"}


def configure_tracing() -> bool:
    """Register a global OTel tracer provider if the SDK is available.

    Returns:
        bool: ``True`` when tracing was configured; ``False`` when the
            SDK is missing or the operator opted out via
            ``OTEL_SDK_DISABLED``. A ``False`` result is not an error —
            the server runs normally and FastMCP's span emission
            becomes a silent no-op.

    """
    if _otel_disabled():
        return False

    # OpenTelemetry ships as the ``otel`` extra; the imports below must stay
    # inside this function so a missing extra degrades to the ImportError
    # branch below instead of breaking every module that imports this file.
    try:
        from opentelemetry import trace  # noqa: PLC0415
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415
    except ImportError:
        logger.debug(
            "OpenTelemetry SDK not installed; tracing disabled. "
            "Install with: pip install claude-teams[otel]"
        )
        return False

    service_name = os.environ.get("OTEL_SERVICE_NAME", "claude-teams")
    resource = Resource.create({"service.name": service_name})

    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)

    logger.info("OpenTelemetry tracing enabled for service %r", service_name)
    return True
