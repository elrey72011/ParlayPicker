"""Exact old input bytes only; never cache a mutable validation decision."""
import hashlib,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
BINDING='docs/paid-launch/ncaaf-staged-capacity-binding-v1.json'
PROFILE='scripts/ci_test_file_costs_v1.json'

def exact_predecessor_bytes(path,source):
    from scripts import check_launch_change_scope as guard
    # Canonical Git source only; this helper never handles provider originals.
    source=source.replace(b'\r\n',b'\n')
    raw=(ROOT/BINDING).read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(raw).hexdigest()==guard.NCAAF_STAGED_CAPACITY_BINDING_SHA256
    info=json.loads(raw)['source_reconstructions'][path]
    observed=hashlib.sha256(source).hexdigest()
    assert observed in (info['after_sha256'],info['before_sha256'])
    if observed==info['before_sha256']:return source
    previous=guard._ncaaf_staged_capacity_previous_source(path,source)
    assert hashlib.sha256(previous).hexdigest()==info['before_sha256']
    return previous

@pytest.fixture
def historical_ncaaf_staged_capacity_profile(tmp_path):
    root=tmp_path/'SYNTHETIC-exact-seven-shard-predecessor'
    destination=root/PROFILE;destination.parent.mkdir(parents=True)
    destination.write_bytes(exact_predecessor_bytes(PROFILE,(ROOT/PROFILE).read_bytes().replace(b'\r\n',b'\n')))
    old='docs/paid-launch/nfl-pdf-timing-binding-v1.json'
    destination=root/old;destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_bytes((ROOT/old).read_bytes())
    return root
