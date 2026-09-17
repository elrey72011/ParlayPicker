"""Run-scoped incremental backup for append-only research records."""
import hashlib
import json


def sync_records(store, path, client, folder, session=None):
    from app_core.evidence_drive import AlreadyExists
    # No cache survives a scheduler invocation. Manual sync always restores.
    session = {} if session is None else session
    scope = (store.PREFIX, str(path), folder, id(client))
    if session.get("scope") != scope:
        session.clear()
        session["scope"] = scope
    verified = session.setdefault("verified", set())
    restored = 0
    if not session.get("restored"):
        def remote_objects():
            if callable(getattr(client, "read_objects", None)):
                yield from client.read_objects(Prefix=store.PREFIX)
            else:
                for page in client.get_paginator("list_objects_v2").paginate(Bucket=folder, Prefix=store.PREFIX):
                    for item in page.get("Contents", []):
                        with client.get_object(Bucket=folder, Key=item["Key"])["Body"] as body:
                            yield item["Key"], body.read(40_000_001)
        for key, raw in remote_objects():
            if len(raw) > 40_000_000 or store.PREFIX + hashlib.sha256(raw).hexdigest() + ".json" != key:
                raise ValueError("Prospective backup integrity failure")
            record_id = store.insert(json.loads(raw), path)
            # Verify canonical local bytes before treating a restored object as backed up.
            if key != store.PREFIX + record_id + ".json":
                raise ValueError("Prospective backup canonical integrity failure")
            verified.add(key)
            restored += 1
        session["restored"] = True
    saved = 0
    for record in store.records(path):
        key = store.PREFIX + record["id"] + ".json"
        if key in verified:
            continue
        raw = store.encode({k: v for k, v in record.items() if k != "id"})
        try:
            client.put_object(Bucket=folder, Key=key, Body=raw, ContentType="application/json", IfNoneMatch="*")
        except AlreadyExists:
            pass
        with client.get_object(Bucket=folder, Key=key)["Body"] as body:
            if body.read(40_000_001) != raw:
                raise ValueError("Prospective backup read-back failed")
        verified.add(key)
        saved += 1
    return {"remote_records_read": restored, "records_verified": len(verified), "new_records_verified": saved}
