"""Second reviewed NCAAF runtime transition (roadmap D15): exact hashes only, research-only."""
import pytest

from app_core import research_model_recovery as recovery


def _model(runtime):
    artifact = {"protocol": {"production_eligible": False}}
    return {"kind": "model", "id": "frozen", "data": {"artifact": artifact,
            "artifact_hash": recovery.ARTIFACT, "runtime_hash": runtime}}


@pytest.mark.parametrize("source", [recovery.TARGET_RUNTIME, recovery.SOURCE_RUNTIME])
def test_reviewed_alias_transition_appends_a_research_only_cohort(monkeypatch, tmp_path, source):
    monkeypatch.setattr(recovery.ncaaf, "runtime_hash", lambda: recovery.ALIAS_RUNTIME)
    monkeypatch.setattr(recovery.ncaaf, "digest", lambda _: recovery.ARTIFACT)
    path = tmp_path / "cohorts.sqlite3"
    new = recovery.recover_ncaaf(_model(source), path)
    assert new["data"]["runtime_hash"] == recovery.ALIAS_RUNTIME
    assert new["data"]["recovery"]["source_runtime"] == source
    assert new["data"]["recovery"]["production_eligible"] is False
    assert "93a7f236" in new["data"]["recovery"]["reason"]
    assert recovery.recover_ncaaf(_model(source), path)["id"] == new["id"]  # idempotent
    assert recovery.recover_ncaaf(new, path) == new


def test_the_first_transition_is_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(recovery.ncaaf, "runtime_hash", lambda: recovery.TARGET_RUNTIME)
    monkeypatch.setattr(recovery.ncaaf, "digest", lambda _: recovery.ARTIFACT)
    new = recovery.recover_ncaaf(_model(recovery.SOURCE_RUNTIME), tmp_path / "c.sqlite3")
    assert new["data"]["recovery"]["reason"] == "reviewed_NFL_alias_only_change_0d36bd74"


@pytest.mark.parametrize("pair", [
    ("ALIAS", "TARGET"),        # never backwards
    ("TARGET", "unknown"),      # any further code change needs its own review
    ("unknown", "ALIAS"),
])
def test_unreviewed_transitions_still_refuse(monkeypatch, tmp_path, pair):
    named = {"ALIAS": recovery.ALIAS_RUNTIME, "TARGET": recovery.TARGET_RUNTIME}
    source, current = (named.get(x, x) for x in pair)
    monkeypatch.setattr(recovery.ncaaf, "runtime_hash", lambda: current)
    monkeypatch.setattr(recovery.ncaaf, "digest", lambda _: recovery.ARTIFACT)
    with pytest.raises(ValueError, match="stale_frozen_model"):
        recovery.recover_ncaaf(_model(source), tmp_path / "absent.sqlite3")
    assert not (tmp_path / "absent.sqlite3").exists()
