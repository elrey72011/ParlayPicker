"""Exact row deduplication for frames containing structured metadata."""
import numpy as np


def _key(value):
    if isinstance(value, dict):
        return (dict, frozenset((_key(k), _key(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return (type(value), tuple(_key(v) for v in value))
    if isinstance(value, (set, frozenset)):
        return (type(value), frozenset(_key(v) for v in value))
    if isinstance(value, np.ndarray):
        return (np.ndarray, str(value.dtype), value.shape, _key(value.tolist()))
    return value


def drop_duplicate_records(frame):
    """Compare complete rows; preserve original values and conflicting records."""
    if frame.empty:
        return frame.copy()
    keys = frame.map(_key)
    return frame.loc[~keys.duplicated()].copy()
