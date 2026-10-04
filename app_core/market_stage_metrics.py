"""Observe market-stage call costs without changing prediction behavior."""
from app_core.performance_spans import PerformanceSpan


def measured_call(name, operation, *args, ids=None, **kwargs):
    with PerformanceSpan("market_enrichment_component", ids=ids, table_or_kind=name) as span:
        value = operation(*args, **kwargs)
        frame = value[0] if isinstance(value, tuple) and value else value
        span.set(records_returned=len(frame) if hasattr(frame, "__len__") else None)
        return value
