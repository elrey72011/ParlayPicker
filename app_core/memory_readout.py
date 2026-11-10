"""How much memory the app process is using, and at which Refresh stage it grew (owner 2026-11-10).

Streamlit Community Cloud restricted the app for using too much memory, and nothing recorded
where the memory went. ``record(stage)`` is called at the end of every Refresh stage
(``stage_timing.StageTimer.finish``): it notes the process's resident memory, how much it
rose since the previous note, and the peak, in a short in-process history and in one
``MEMORY`` log line. ``readout()`` is what the app shows.

It reads two numbers from the operating system and never looks at any data. It changes
no decision, and a failure here never reaches a caller.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import logging
import threading

LIMIT_MB = 2700            # Community Cloud's limit is about 2.7 GB; shown as a guide, not enforced here
NEAR_LIMIT = 0.8
HISTORY = 60

_history = deque(maxlen=HISTORY)
_lock = threading.Lock()


def rss_mb():
    """Resident memory of this process in MB, or None where the platform cannot say."""
    try:
        with open("/proc/self/statm", encoding="ascii") as fh:
            pages = int(fh.read().split()[1])
        import os
        return pages * os.sysconf("SC_PAGE_SIZE") / 1_048_576
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def peak_mb():
    """The most resident memory this process has held, in MB, or None."""
    try:
        import resource
        import sys
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 1_048_576 if sys.platform == "darwin" else peak / 1024
    except (ImportError, OSError, ValueError):
        return None


def record(stage, *, read=rss_mb, peak=peak_mb, clock=None):
    """Note memory at the end of ``stage``. Returns the note, or None when memory cannot be read."""
    try:
        now_mb = read()
        if now_mb is None:
            return None
        with _lock:
            previous = _history[-1]["rss_mb"] if _history else None
            note = {"at": (clock() if clock else datetime.now(timezone.utc)).isoformat(timespec="seconds"),
                    "stage": str(stage)[:120], "rss_mb": round(now_mb),
                    "rose_mb": None if previous is None else round(now_mb - previous),
                    "peak_mb": None if peak() is None else round(peak())}
            _history.append(note)
        logging.getLogger(__name__).warning("MEMORY stage=%s rss_mb=%s rose_mb=%s peak_mb=%s",
                                            note["stage"], note["rss_mb"], note["rose_mb"], note["peak_mb"])
        return note
    except Exception:  # a readout must never break a Refresh
        return None


def readout(*, read=rss_mb, peak=peak_mb):
    """Current and peak memory, the share of the limit, and the notes so far (oldest first)."""
    now_mb, top = read(), peak()
    with _lock:
        notes = list(_history)
    largest = max((n for n in notes if n["rose_mb"]), key=lambda n: n["rose_mb"], default=None)
    return {"rss_mb": None if now_mb is None else round(now_mb), "peak_mb": None if top is None else round(top),
            "limit_mb": LIMIT_MB, "share_of_limit": None if now_mb is None else now_mb / LIMIT_MB,
            "near_limit": bool(now_mb is not None and now_mb >= NEAR_LIMIT * LIMIT_MB),
            "largest_rise": largest, "notes": notes}


def clear():
    with _lock:
        _history.clear()
