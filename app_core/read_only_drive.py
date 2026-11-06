"""A Drive client wrapper that reads evidence and refuses every write."""
from __future__ import annotations


class ReadOnlyViolation(RuntimeError):
    pass


class ReadOnlyDrive:
    """Delegates reads to the real client; refuses every write."""

    def __init__(self, client, cache_dir=None):
        self._client = client
        self._cache_dir = cache_dir
        self.writes_refused = 0

    def read_objects(self, *, Prefix):
        if self._cache_dir is not None and callable(getattr(self._client, 'read_cached_objects', None)):
            return self._client.read_cached_objects(Prefix=Prefix, cache_dir=self._cache_dir)
        return self._client.read_objects(Prefix=Prefix)

    def put_object(self, **kwargs):
        self.writes_refused += 1
        raise ReadOnlyViolation('READ_ONLY_VIOLATION:put_object:' + str(kwargs.get('Key', ''))[:80])

    def run_parallel(self, operation, items, progress=None):
        # Run through this wrapper so no worker client can bypass put_object.
        out = []
        for i, item in enumerate(items, 1):
            out.append(operation(self, item))
            if progress:
                progress(i, len(items))
        return out

    def __getattr__(self, name):
        return getattr(self._client, name)
