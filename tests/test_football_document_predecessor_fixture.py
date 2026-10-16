"""SYNTHETIC immutable predecessor input; no source acquisition/inference."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

from scripts import check_launch_change_scope as guard
from tests import test_nfl_listing_capacity_scope as historical

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'tests/test_ci_test_shards.py'


def test_exact_five_shard_input_and_idempotence():
    current = (ROOT/TARGET).read_bytes().replace(b'\r\n', b'\n')
    prior = historical._historical_five_shard_bytes(TARGET, current)
    record = json.loads(guard.FOOTBALL_DOCSIX_RECONSTRUCTION_JSON)[TARGET]
    assert hashlib.sha256(prior).hexdigest() == record['before_sha256']
    assert prior != current
    assert historical._historical_five_shard_bytes(TARGET, prior) == prior
    assert b"'1', '2', '3', '4', '5', '6'" in current


@pytest.mark.parametrize('attack', ['append', 'replace', 'wrong_path'])
def test_unknown_changed_sources_are_not_reconstructed(attack):
    source = (ROOT/TARGET).read_bytes().replace(b'\r\n', b'\n')
    path = TARGET
    if attack == 'append':
        source += b'\nUNREVIEWED'
    elif attack == 'replace':
        source = source.replace(b"'6'", b"'7'", 1)
    else:
        path = 'tests/UNREVIEWED.py'
    assert historical._historical_five_shard_bytes(path, source) == source


def test_reconstruction_hash_tampering_rejects(monkeypatch):
    source = (ROOT/TARGET).read_bytes().replace(b'\r\n', b'\n')
    records = json.loads(guard.FOOTBALL_DOCSIX_RECONSTRUCTION_JSON)
    records[TARGET]['before_sha256'] = '0'*64
    monkeypatch.setattr(guard, 'FOOTBALL_DOCSIX_RECONSTRUCTION_JSON', json.dumps(records).encode())
    with pytest.raises(AssertionError):
        historical._historical_five_shard_bytes(TARGET, source)


def test_fixture_copies_are_isolated_and_original_assertions_remain(tmp_path):
    first = historical.historical_five_shard_source.__wrapped__(tmp_path/'one')
    second = historical.historical_five_shard_source.__wrapped__(tmp_path/'two')
    original = (ROOT/TARGET).read_bytes()
    historical.test_exact_capacity_source_projection_and_unknown_bytes_preserved(first)
    (first/TARGET).write_bytes((first/TARGET).read_bytes()+b'\nUNREVIEWED')
    with pytest.raises(AssertionError):
        historical.test_exact_capacity_source_projection_and_unknown_bytes_preserved(first)
    historical.test_exact_capacity_source_projection_and_unknown_bytes_preserved(second)
    assert (ROOT/TARGET).read_bytes() == original


def test_all_old_assertions_and_unknown_test_source_preserved():
    path = 'tests/test_nfl_listing_capacity_scope.py'
    current = (ROOT/path).read_bytes().replace(b'\r\n', b'\n')
    prior = guard._docpred_previous_source(path, current)
    # Guard preimages are canonical Git LF bytes; the retained Windows checkout
    # receipt separately hashes its unchanged CRLF representation.
    assert hashlib.sha256(prior).hexdigest() == '3b58be5f526e369a34bbf4b4a8a39dceb2c85ceedc883bbd67ca0dfd970e4f84'
    def assertions(raw):
        return {n.name: [ast.dump(a) for a in ast.walk(n) if isinstance(a, ast.Assert)]
                for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)}
    old = assertions(prior); new = assertions(current)
    assert all(new[name] == values for name, values in old.items())
    assert guard._docpred_previous_source(path, current+b'\nUNREVIEWED') == current+b'\nUNREVIEWED'


# These tests assert a frozen six-to-five projection. Supply its exact six-way
# input; current seven-way gate tests continue to read the live checkout.
def _exact_six_shard_input(path, source):
    if path == guard.GUARD_PATH:
        result = guard._nfl_pdf_timing_previous_source(path, source)
        assert hashlib.sha256(result).hexdigest() == guard.NFL_PDF_TIMING_PREVIOUS_GUARD_SHA256
        return result
    record = json.loads(guard.NFL_PDF_TIMING_RECONSTRUCTION_JSON).get(path)
    if record is None:
        return source
    identity = hashlib.sha256(source).hexdigest()
    if identity == record['before_sha256']:
        return source
    assert identity == record['after_sha256']
    result = guard._nfl_pdf_timing_previous_source(path, source)
    assert hashlib.sha256(result).hexdigest() == record['before_sha256']
    return result


@pytest.fixture(autouse=True)
def historical_six_shard_input(tmp_path, monkeypatch):
    root = tmp_path/'SYNTHETIC-exact-six-way-predecessor'
    paths = set(json.loads(guard.LISTING_CAPACITY_RECONSTRUCTION_JSON)) | {TARGET, 'tests/test_nfl_listing_capacity_scope.py'}
    assert len(paths) <= 32
    for path in sorted(paths):
        source = (ROOT/path).read_bytes().replace(b'\r\n', b'\n')
        result = _exact_six_shard_input(path, source)
        destination = root/path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(result)
    monkeypatch.setattr(__import__(__name__, fromlist=['ROOT']), 'ROOT', root)
    monkeypatch.setattr(historical, '__file__', str(root/'tests/test_nfl_listing_capacity_scope.py'))
    return root
