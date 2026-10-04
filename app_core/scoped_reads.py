"""Process-local coalescing of equivalent reads; completed results are not authority."""
from concurrent.futures import Future
from threading import Lock

_mutex = Lock()
_flights = {}


def singleflight(key, operation):
    with _mutex:
        future = _flights.get(key)
        owner = future is None
        if owner:
            future = _flights[key] = Future()
    if not owner:
        return future.result(), True
    try:
        value = operation()
        future.set_result(value)
        return value, False
    except BaseException as exc:
        future.set_exception(exc)
        raise
    finally:
        with _mutex:
            if _flights.get(key) is future:
                del _flights[key]
