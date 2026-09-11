"""Lightweight stage timing with optional UI progress; never logs input data."""
import logging
from time import perf_counter

class StageTimer:
    def __init__(self, progress=None):
        self.progress = progress
        self.timings = {}
        self.name = None
        self.started = perf_counter()

    def start(self, name):
        self.finish()
        self.name = name
        self.started = perf_counter()
        if self.progress:
            elapsed = sum(self.timings.values())
            self.progress(f"{name} — {elapsed:.0f}s elapsed")

    def finish(self):
        if self.name is not None:
            elapsed = round(perf_counter()-self.started, 3)
            self.timings[self.name] = self.timings.get(self.name, 0)+elapsed
            logging.getLogger(__name__).warning('PERFORMANCE stage=%s seconds=%.3f', self.name, elapsed)
            self.name = None
