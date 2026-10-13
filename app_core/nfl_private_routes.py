"""Explicit version dispatch; private routes cannot borrow one another's review."""
from contextlib import contextmanager
import json
from app_core import nfl_owner_research as v1, nfl_observed_research as v2, nfl_scoring_research as v3, nfl_interpreted_research as v4

REASONS = v1.REASONS | v2.REASONS | v3.REASONS | v4.REASONS
require = v1.require
MAX_PACKETS = v1.MAX_PACKETS
private = v1.private
finish = v1.finish
retain_ui_refresh = v1.retain_ui_refresh


def route(packet):
    version = packet.get('payload', {}).get('version') if isinstance(packet, dict) else None
    v1.require(version in {v1.VERSION, v2.VERSION, v3.VERSION, v4.VERSION}, 'NFL_PRIVATE_PACKET_SCHEMA')
    return {v1.VERSION:v1,v2.VERSION:v2,v3.VERSION:v3,v4.VERSION:v4}[version]


def load(raw, *, owner_upload=False):
    v1.require(isinstance(raw, bytes) and 0 < len(raw) <= v3.MAX_PACKET, 'NFL_PRIVATE_PACKET_SCHEMA')
    p = json.loads(raw, object_pairs_hook=v1.custody.unique)
    return route(p).load(raw, owner_upload=owner_upload)


@contextmanager
def selected(packets=()):
    v1.require(isinstance(packets, (list, tuple)) and len(packets) <= v1.MAX_PACKETS, 'NFL_PRIVATE_PACKET_SCHEMA')
    v1.require(sum(len(v1.encode(p).encode()) for p in packets) <= v1.MAX_SELECTED and len({p['sha256'] for p in packets}) == len(packets), 'NFL_PRIVATE_PACKET_AMBIGUOUS')
    with v1.selected([p for p in packets if route(p) is v1]), v2.selected([p for p in packets if route(p) is v2]), v3.selected([p for p in packets if route(p) is v3]), v4.selected([p for p in packets if route(p) is v4]):
        yield


def selection_requested():
    # Both recursion guards must block dispatch during unchanged mathematics.
    return not (v1.IN_NATIVE.get() or v2.IN_NATIVE.get() or v3.IN_NATIVE.get() or v4.IN_NATIVE.get()) and bool(v1.SELECTED.get() or v2.SELECTED.get() or v3.SELECTED.get() or v4.SELECTED.get())


def predict(source):
    offered = v1.producer._matches(source)
    packets = v1.SELECTED.get() + v2.SELECTED.get() + v3.SELECTED.get() + v4.SELECTED.get()
    matching = [p for p in packets if p['payload']['quote']['market_type'] == source.get('market_type') and
        any(m.get('provider_event_id') == p['payload']['event']['provider_event_id'] and
            m.get('provider_namespace') == p['payload']['event']['provider_namespace'] for m in offered)]
    if len(matching) == 1:
        return route(matching[0]).predict(source)
    # Reject cross-version ambiguity before either reader or numerical inference.
    token = v1.SELECTED.set(tuple(matching))
    try: return v1.predict(source)
    finally: v1.SELECTED.reset(token)


def saved_route(source):
    try:
        p = json.loads(source['ml_estimate_metadata'])['nfl_private_inputs']['payload']
        return {v1.RESULT:v1,v2.RESULT:v2,v3.RESULT:v3,v4.RESULT:v4}.get(p.get('version'),v1)
    except (KeyError, ValueError, TypeError): return v1


def diagnose(source): return saved_route(source).diagnose(source)
def private_display(source): return saved_route(source).private_display(source)
