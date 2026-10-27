"""Test-owned watchdog scheduling for a deliberately simulated driver clock.

The real acquisition driver is unchanged. A logical-time test completes its
inline slice before draining the watchdog callback against its signalled event.
Real hard termination is exercised in separate owned processes instead.
"""
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch


@contextmanager
def logical_watchdogs(driver):
    owned = []

    class LogicalEvent:
        def __init__(self):
            self.signalled = False

        def set(self):
            self.signalled = True

        def wait(self, timeout):
            # A broken test cleanup is an assertion, never a real process exit
            # in this logical-time parent. The real Event is tested separately.
            assert self.signalled and timeout >= 0
            return True

    class LogicalThread:
        def __init__(self, *, target, daemon):
            assert daemon is True
            self.target = target
            self.started = self.joined = False
            owned.append(self)

        def start(self):
            assert not self.started
            self.started = True

        def join(self):
            assert self.started and not self.joined
            # capture_block's finally has set the actual event. No real-time
            # watchdog is allowed to terminate a logical-time parent test.
            self.target()
            self.joined = True

    threading = SimpleNamespace(**dict(vars(driver.threading), Thread=LogicalThread, Event=LogicalEvent))
    with patch.object(driver, 'threading', threading):
        yield owned
    assert owned and all(thread.started and thread.joined for thread in owned)
