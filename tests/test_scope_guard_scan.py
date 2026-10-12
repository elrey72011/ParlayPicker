"""Labelled synthetic/exact-source tests; no numerical or external execution."""
from pathlib import Path
import hashlib
import pytest
from scripts import check_launch_change_scope as guard

def source():return Path(guard.__file__).read_bytes().replace(b'\r\n',b'\n')

def test_exact_previous_guard_and_warm_marker_scan():
    guard._guard_scan_cached_base_source.cache_clear()
    guard._guard_scan_v1_presence.cache_clear()
    current=source();previous=guard._guard_scan_base_source(current)
    assert hashlib.sha256(previous).hexdigest()==guard.GUARD_SCAN_PREVIOUS_GUARD_SHA256
    frozen=guard._guard_performance_base_source(current)
    for _ in range(100):assert guard._guard_performance_base_source(current)==frozen
    assert guard._guard_scan_cached_base_source.cache_info().hits>=100
    assert guard._guard_scan_v1_presence.cache_info().hits>=100
    assert guard._guard_scan_cached_base_source.cache_info().maxsize==2
    assert guard._guard_scan_v1_presence.cache_info().maxsize==2

def test_changed_source_and_oversize_reject_before_cache_admission():
    current=source()
    for changed in (current+b'\nSYNTHETIC_UNREVIEWED=True\n',current+b'x'*(8*1024*1024)):
        with pytest.raises(ValueError,match='SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED'):
            guard._guard_performance_base_source(changed)
    assert guard._guard_scan_cached_base_source.cache_info().currsize<=2
    assert guard._guard_scan_v1_presence.cache_info().currsize<=2

def test_opaque_unchanged_bytes_have_no_repository_or_decision_identity(tmp_path,monkeypatch):
    raw=b'LABELLED SYNTHETIC predecessor bytes without successor marker'
    for root in (tmp_path/'one',tmp_path/'two'):
        root.mkdir();monkeypatch.setattr(guard,'ROOT',root)
        assert guard._guard_scan_base_source(raw)==raw
        assert guard._guard_performance_base_source(raw)==raw
    changed=raw+b' DIFFERENT SUBJECT'
    assert guard._guard_performance_base_source(changed)==changed
    assert guard._guard_performance_base_source(raw)==raw
