from io import BytesIO
import pytest
from app_core import mlb_prospective_store as store
from app_core.evidence_drive import AlreadyExists


class Cloud:
    def __init__(self):
        self.objects = {}
        self.lists = self.reads = self.writes = 0
        self.corrupt_readback = False
    def get_paginator(self, name): return self
    def paginate(self, **kw):
        self.lists += 1
        yield {"Contents": [{"Key": k} for k in self.objects]}
    def get_object(self, Key, **kw):
        self.reads += 1
        return {"Body": BytesIO(b"bad" if self.corrupt_readback else self.objects[Key])}
    def put_object(self, Key, Body, **kw):
        self.writes += 1
        if Key in self.objects: raise AlreadyExists()
        self.objects[Key] = Body


def test_backups_only_send_new_records_and_new_run_rechecks_all(tmp_path):
    cloud = Cloud(); session = {}; path = tmp_path / "a.db"
    store.save("model", {"v": 1}, path)
    store.sync(path, client=cloud, folder="f", session=session)
    assert (cloud.lists, cloud.reads, cloud.writes) == (1, 1, 1)
    store.sync(path, client=cloud, folder="f", session=session)
    assert (cloud.lists, cloud.reads, cloud.writes) == (1, 1, 1)
    store.save("scores", {"v": 2}, path)
    store.sync(path, client=cloud, folder="f", session=session)
    assert (cloud.lists, cloud.reads, cloud.writes) == (1, 2, 2)
    store.sync(path, client=cloud, folder="f", session={})
    assert (cloud.lists, cloud.reads, cloud.writes) == (2, 4, 2)


def test_restore_detects_corruption_and_cannot_mark_complete(tmp_path):
    cloud = Cloud(); path = tmp_path / "a.db"
    store.save("model", {}, path)
    store.sync(path, client=cloud, folder="f")
    cloud.objects[next(iter(cloud.objects))] = b"tampered"
    session = {}
    with pytest.raises(ValueError, match="integrity"):
        store.sync(tmp_path / "b.db", client=cloud, folder="f", session=session)
    assert not session.get("restored")


def test_failed_readback_is_not_cached(tmp_path):
    cloud = Cloud(); session = {}; path = tmp_path / "a.db"
    store.save("model", {}, path)
    cloud.corrupt_readback = True
    with pytest.raises(ValueError, match="read-back"):
        store.sync(path, client=cloud, folder="f", session=session)
    assert not session["verified"]
    cloud.corrupt_readback = False
    store.sync(path, client=cloud, folder="f", session=session)
    assert len(session["verified"]) == 1


def test_bulk_restore_used_and_scoped_to_local_database(tmp_path):
    cloud = Cloud(); first = tmp_path / "a.db"
    store.save("model", {}, first)
    store.sync(first, client=cloud, folder="f")
    calls = []
    def bulk(**kw):
        calls.append(kw)
        return list(cloud.objects.items())
    cloud.read_objects = bulk
    session = {}
    for name in ("b.db", "c.db"):
        store.sync(tmp_path / name, client=cloud, folder="f", session=session)
        assert len(store.records(tmp_path / name)) == 1
    assert len(calls) == 2
