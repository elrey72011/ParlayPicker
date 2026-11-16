"""Exact predecessor inputs for historical assertions, never live decisions."""
import hashlib
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
PROFILE='scripts/ci_test_file_costs_v1.json'
BINDING='docs/paid-launch/nfl-pdf-timing-binding-v1.json'


def predecessor_profile(root=ROOT):
    from scripts import check_launch_change_scope as guard
    binding_raw=(root/BINDING).read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(binding_raw).hexdigest()==guard.NFL_PDF_TIMING_BINDING_SHA256
    binding=json.loads(binding_raw)
    source=(root/PROFILE).read_bytes().replace(b'\r\n',b'\n')
    from tests.ncaaf_staged_capacity_fixtures import exact_predecessor_bytes
    source=exact_predecessor_bytes(PROFILE,source)
    info=binding['source_reconstructions'][PROFILE]
    assert hashlib.sha256(source).hexdigest()==info['after_sha256']
    prior=guard._nfl_pdf_timing_previous_source(PROFILE,source)
    assert hashlib.sha256(prior).hexdigest()==info['before_sha256']
    return prior


@pytest.fixture
def historical_nfl_pdf_capacity_profile(tmp_path):
    root=tmp_path/'SYNTHETIC-exact-six-shard-predecessor'
    dest=root/PROFILE;dest.parent.mkdir(parents=True)
    dest.write_bytes(predecessor_profile())
    old='docs/paid-launch/ncaaf-socket-ci-binding-v1.json'
    dest=root/old;dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes((ROOT/old).read_bytes())
    return root
