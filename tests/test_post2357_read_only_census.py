import hashlib
import json
from types import SimpleNamespace

import pytest

from app_core.evidence_drive import DriveInventory
from app_core.read_only_census import (
    CANONICAL_PREFIX,
    SCOPES,
    SOURCE_PREFIXES,
    run_census,
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def source_object(sport, event_id):
    value = {
        "schema": 1,
        "kind": "capture",
        "created_at": "2026-09-29T12:00:00+00:00",
        "data": {"sport": sport, "events": [{"event_id": event_id}]},
    }
    raw = canonical(value)
    return SOURCE_PREFIXES[sport] + hashlib.sha256(raw).hexdigest() + ".json", raw


def canonical_object(table, columns, row, primary):
    value = {"schema": 1, "table": table, "columns": columns, "row": row}
    raw = canonical(value)
    identity = [row[columns.index(field)] for field in primary]
    digest = hashlib.sha256(canonical([table, identity])).hexdigest()
    return f"{CANONICAL_PREFIX}{table}/{digest}.json", raw


class FakeReadOnlyDrive:
    def __init__(self, objects, scope="scope-a", fail=False):
        self.objects = dict(objects)
        self.scope = scope
        self.fail = fail
        self.read_names = []
        self.last_read_report = None
        self.mutation_calls = []

    def storage_scope_hash(self):
        return self.scope

    def discover_complete_inventory(self, namespace=None):
        files = tuple({"id": "id-" + hashlib.sha256(name.encode()).hexdigest(), "name": name,
                       "sha256Checksum": hashlib.sha256(raw).hexdigest()}
                      for name, raw in sorted(self.objects.items()))
        return DriveInventory(self.scope, "inventory-op", namespace, files, 1, len(files))

    def read_verified_prefixes(self, *, Prefixes, inventory, cache_dir, full_verify):
        assert cache_dir is None
        if self.fail:
            raise RuntimeError("private-token=never-serialize")
        result = {}
        downloaded = 0
        byte_count = 0
        for name in Prefixes:
            self.read_names.append(name)
            raw = self.objects[name]
            result[name] = [(name, raw)]
            downloaded += 1
            byte_count += len(raw)
        self.last_read_report = SimpleNamespace(
            objects_downloaded=downloaded, bytes_downloaded=byte_count)
        return result

    def _mutation(self, *args, **kwargs):
        self.mutation_calls.append((args, kwargs))
        raise AssertionError("mutation route called")

    put_object = sync = restore = backup = reconcile = activate = bill = wager = _mutation


def all_source_objects():
    return dict(source_object(sport, f"{sport}-event") for sport in SOURCE_PREFIXES)


def many_source_objects(count=20):
    objects = all_source_objects()
    objects.update(source_object("NFL", f"NFL-extra-{index}")
                   for index in range(count))
    return objects


def test_complete_six_sport_census_is_explicitly_read_only(tmp_path):
    objects = all_source_objects()
    name, raw = canonical_object(
        "prospective_model",
        ["model_id", "sport", "market_family", "artifact_hash",
         "independent_event_count", "training_cutoff"],
        ["nfl-model", "NFL", "SPREAD", "model-hash", 12,
         "2026-09-01T00:00:00+00:00"],
        ["model_id"],
    )
    objects[name] = raw
    client = FakeReadOnlyDrive(objects)

    report = run_census(
        client, checkpoint_path=tmp_path / "checkpoint.json",
        output_path=tmp_path / "report.json", source_revision="head-sha",
        max_objects=100, max_bytes=1_000_000, deadline_seconds=60)

    assert report["status"] == "COMPLETE"
    assert [scope["scope"] for scope in report["scopes"]] == list(SCOPES)
    assert all(scope["census_state"] == "COMPLETE" for scope in report["scopes"])
    assert report["read_only_contract"] == {
        "remote_operations": ["complete_inventory", "verified_object_read"],
        "restore_calls": 0,
        "reconciliation_calls": 0,
        "capture_calls": 0,
        "receipt_recovery_calls": 0,
        "backup_calls": 0,
        "activation_calls": 0,
        "billing_calls": 0,
        "wager_calls": 0,
    }
    assert client.mutation_calls == []
    nfl_spread = next(scope for scope in report["scopes"]
                      if scope["scope"] == "NFL/SPREAD")
    assert nfl_spread["model_records"][0]["model_id"] == "nfl-model"
    assert nfl_spread["counts"]["raw_objects"] == 2
    assert nfl_spread["counts"]["unique_events"] == 1


def test_slice_resumes_without_duplicate_reads(tmp_path):
    objects = all_source_objects()
    client = FakeReadOnlyDrive(objects)
    checkpoint = tmp_path / "checkpoint.json"

    first = run_census(client, checkpoint_path=checkpoint,
                       output_path=tmp_path / "first.json", max_objects=2,
                       max_bytes=1_000_000, deadline_seconds=60)
    first_names = tuple(client.read_names)
    assert first["status"] == "PARTIAL"
    assert first["terminal_reason"] == "OBJECT_LIMIT_REACHED"
    assert len(first_names) == 2

    second = run_census(client, checkpoint_path=checkpoint,
                        output_path=tmp_path / "second.json", max_objects=100,
                        max_bytes=1_000_000, deadline_seconds=60)
    assert second["status"] == "COMPLETE"
    assert second["metrics"]["verified_reused_objects"] == 2
    assert set(first_names).isdisjoint(client.read_names[2:])
    assert len(client.read_names) == len(objects)


def test_full_verification_redownloads_checkpointed_objects(tmp_path):
    objects = all_source_objects()
    checkpoint = tmp_path / "checkpoint.json"
    run_census(FakeReadOnlyDrive(objects), checkpoint_path=checkpoint,
               output_path=tmp_path / "first.json", max_objects=100,
               max_bytes=1_000_000, deadline_seconds=60)
    resumed = FakeReadOnlyDrive(objects)

    report = run_census(resumed, checkpoint_path=checkpoint,
                        output_path=tmp_path / "second.json", max_objects=100,
                        max_bytes=1_000_000, deadline_seconds=60,
                        full_verify=True)

    assert report["metrics"]["verified_reused_objects"] == 0
    assert report["metrics"]["invalidated_checkpoint_objects"][
        "full_verification_requested"] == len(objects)
    assert len(resumed.read_names) == len(objects)


@pytest.mark.parametrize("change,reason", [
    ("tamper", "CHECKPOINT_DIGEST_MISMATCH"),
    ("scope", "CHECKPOINT_STORAGE_SCOPE_MISMATCH"),
    ("revision", "CHECKPOINT_SOURCE_REVISION_MISMATCH"),
])
def test_corrupt_or_wrong_scope_checkpoint_is_not_reused(tmp_path, change, reason):
    objects = all_source_objects()
    checkpoint = tmp_path / "checkpoint.json"
    run_census(FakeReadOnlyDrive(objects), checkpoint_path=checkpoint,
               output_path=tmp_path / "first.json", max_objects=100,
               max_bytes=1_000_000, deadline_seconds=60,
               source_revision="revision-a")
    if change == "tamper":
        value = json.loads(checkpoint.read_text())
        value["processed"].pop(next(iter(value["processed"])))
        checkpoint.write_text(json.dumps(value))
        resumed = FakeReadOnlyDrive(objects)
    elif change == "scope":
        resumed = FakeReadOnlyDrive(objects, scope="scope-b")
    else:
        resumed = FakeReadOnlyDrive(objects)

    report = run_census(resumed, checkpoint_path=checkpoint,
                        output_path=tmp_path / "second.json", max_objects=100,
                        max_bytes=1_000_000, deadline_seconds=60,
                        source_revision=("revision-b" if change == "revision"
                                         else "revision-a"))
    assert report["checkpoint"]["rejection_reason"] == reason
    assert report["metrics"]["verified_reused_objects"] == 0
    assert len(resumed.read_names) == len(objects)


def test_changed_metadata_and_deletion_invalidate_checkpoint_entries(tmp_path):
    objects = all_source_objects()
    checkpoint = tmp_path / "checkpoint.json"
    run_census(FakeReadOnlyDrive(objects), checkpoint_path=checkpoint,
               output_path=tmp_path / "first.json", max_objects=100,
               max_bytes=1_000_000, deadline_seconds=60)
    names = list(objects)
    deleted = names[0]
    changed = names[1]
    revised = dict(objects)
    revised.pop(deleted)
    # A newly valid immutable object under the same namespace changes both its
    # name and checksum; the old name is treated as a deletion and the new one
    # must be read.
    prefix = next(prefix for prefix in SOURCE_PREFIXES.values()
                  if changed.startswith(prefix))
    sport = next(sport for sport, candidate in SOURCE_PREFIXES.items()
                 if candidate == prefix)
    revised.pop(changed)
    new_name, new_raw = source_object(sport, f"{sport}-changed")
    revised[new_name] = new_raw
    resumed = FakeReadOnlyDrive(revised)

    report = run_census(resumed, checkpoint_path=checkpoint,
                        output_path=tmp_path / "second.json", max_objects=100,
                        max_bytes=1_000_000, deadline_seconds=60)
    assert report["metrics"]["invalidated_checkpoint_objects"]["deleted"] == 2
    assert resumed.read_names == [new_name]


def test_conflicting_duplicate_metadata_blocks_namespace_without_read(tmp_path):
    name, raw = source_object("NFL", "nfl-event")

    class ConflictDrive(FakeReadOnlyDrive):
        def discover_complete_inventory(self, namespace=None):
            files = (
                {"id": "one", "name": name, "sha256Checksum": hashlib.sha256(raw).hexdigest()},
                {"id": "two", "name": name, "sha256Checksum": "f" * 64},
            )
            return DriveInventory(self.scope, "inventory-op", namespace, files, 1, 2)

    client = ConflictDrive({name: raw})
    report = run_census(client, checkpoint_path=tmp_path / "checkpoint.json",
                        output_path=tmp_path / "report.json", max_objects=100,
                        max_bytes=1_000_000, deadline_seconds=60)

    assert report["namespaces"][SOURCE_PREFIXES["NFL"]]["status"] == "BLOCKED"
    assert client.read_names == []


def test_verified_read_failure_retains_sanitized_terminal_report(tmp_path):
    client = FakeReadOnlyDrive(all_source_objects(), fail=True)
    output = tmp_path / "report.json"
    report = run_census(client, checkpoint_path=tmp_path / "checkpoint.json",
                        output_path=output, max_objects=100,
                        max_bytes=1_000_000, deadline_seconds=60)

    assert report["status"] == "BLOCKED"
    assert report["terminal_reason"] == "VERIFIED_READ_FAILED"
    saved = output.read_text()
    assert "private-token" not in saved
    assert "VERIFIED_READ_FAILED" in saved


def test_byte_budget_stops_after_one_bounded_batch_and_retains_state(tmp_path):
    client = FakeReadOnlyDrive(many_source_objects())
    checkpoint = tmp_path / "checkpoint.json"
    output = tmp_path / "report.json"

    report = run_census(client, checkpoint_path=checkpoint, output_path=output,
                        max_objects=100, max_bytes=1, deadline_seconds=60)

    assert report["status"] == "PARTIAL"
    assert report["terminal_reason"] == "BYTE_LIMIT_REACHED"
    assert len(client.read_names) == report["budget_contract"]["read_batch_objects"] == 8
    assert report["budget_contract"]["byte_limit_enforcement"] == "POST_BATCH_SOFT_LIMIT"
    assert report["budget_contract"]["maximum_byte_limit_overrun"] == 480_000_000
    assert checkpoint.is_file() and output.is_file()


def test_deadline_self_interruption_retains_checkpoint_and_resumes_without_duplicates(tmp_path):
    class ControlledClock:
        def __init__(self):
            self.values = iter([0] * 8 + [61] * 100)

        def __call__(self):
            return next(self.values)

    objects = many_source_objects()
    client = FakeReadOnlyDrive(objects)
    checkpoint = tmp_path / "checkpoint.json"
    output = tmp_path / "report.json"

    first = run_census(client, checkpoint_path=checkpoint, output_path=output,
                       max_objects=100, max_bytes=1_000_000,
                       deadline_seconds=60, clock=ControlledClock())
    first_names = tuple(client.read_names)

    assert first["terminal_reason"] == "DEADLINE_REACHED"
    assert first["status"] == "PARTIAL"
    assert len(first_names) == 8
    assert checkpoint.is_file() and json.loads(checkpoint.read_text())["processed"]
    assert json.loads(output.read_text())["terminal_reason"] == "DEADLINE_REACHED"

    second = run_census(client, checkpoint_path=checkpoint, output_path=output,
                        max_objects=100, max_bytes=1_000_000, deadline_seconds=60)
    assert second["status"] == "COMPLETE"
    assert second["metrics"]["verified_reused_objects"] == 8
    assert set(first_names).isdisjoint(client.read_names[len(first_names):])
    assert len(client.read_names) == len(objects)
