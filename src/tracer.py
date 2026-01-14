"""Phoenix tracer setup for observability."""

import os
from phoenix.otel import register
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from config import PHOENIX_COLLECTOR_ENDPOINT

# Register Phoenix tracer
tracer_provider = register(
    project_name="fashion-forward-chatbot",
    endpoint=f"{PHOENIX_COLLECTOR_ENDPOINT}/v1/traces"
)

# Get tracer for manual instrumentation
tracer = tracer_provider.get_tracer(__name__)

# Decorator for easy tracing
def trace_function(span_kind="tool"):
    """Decorator to trace functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(
                func.__name__,
                attributes={"openinference.span.kind": span_kind}
            ) as span:
                result = func(*args, **kwargs)
                return result
        return wrapper
    return decorator