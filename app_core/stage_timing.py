"""Lightweight stage timing with optional UI progress; never logs input data."""
import logging
from time import perf_counter
from app_core.performance_spans import PerformanceSpan, operation_ids

class StageTimer:
    def __init__(self, progress=None, *, ids=None):
        self.progress = progress
        self.timings = {}
        self.name = None
        self.records = None
        self.started = perf_counter()
        self.ids = ids or operation_ids(refresh_run_id=None)
        self.span = None

    def start(self, name, *, records=None):
        self.finish()
        self.name = name
        self.records = records
        self.started = perf_counter()
        self.span = PerformanceSpan('refresh_stage', ids=self.ids, table_or_kind=name)
        self.span.__enter__()
        if self.progress:
            elapsed = sum(self.timings.values())
            self.progress(f"{name} — {elapsed:.0f}s elapsed")

    def finish(self):
        if self.name is not None:
            elapsed = round(perf_counter()-self.started, 3)
            self.timings[self.name] = self.timings.get(self.name, 0)+elapsed
            logging.getLogger(__name__).warning('PERFORMANCE stage=%s seconds=%.3f records=%s',
                                                self.name, elapsed, self.records)
            self.span.set(records_returned=self.records)
            self.span.__exit__(None, None, None)
            self.span = None
            self.name = None
            self.records = None
