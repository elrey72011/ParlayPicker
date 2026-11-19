"""A Refresh still shows its board when the saved evidence cannot be read (2026-11-19).

SYNTHETIC Drive, evidence and game throughout; no network. On 19 November two Drive files under one evidence name
stopped every restore. The Refresh warned at its start, as designed, then asked for the saved history again while
preparing review candidates and stopped there: "Pipeline crashed ... Drive contains conflicting duplicate evidence
names", after sixteen minutes and with no board. Three things are fixed here and tested below: a restore that has
just failed is not repeated a moment later, the review step goes on without the history, and two Refreshes at once
can no longer register one bundle twice, which is how the two files came to exist.
"""
from concurrent.futures import ThreadPoolExecutor
import threading

import pandas as pd
import pytest

from activation_fixture import NOW
from app_core import evidence_remote as remote
from app_core.evidence_config import EvidenceStorageError
from app_core.prediction_evidence import connect
from test_candidate_authority_projection import build, pregame_clock, source  # noqa: F401  (pregame_clock is a fixture)
from test_evidence_conflicts import _again
from test_evidence_index import DAY, MANIFEST, PREFIX, START, _add_run, _sha, world  # noqa: F401  (world is a fixture)

REFUSAL = "Drive contains conflicting duplicate evidence names"


@pytest.fixture
def memory(monkeypatch):
    """This process's memory of restores and bundles, empty, with a clock the test moves."""
    for name, value in (("_status", {}), ("_restored", set()), ("_verified", {}), ("_failed", {}), ("_bundles", {})):
        monkeypatch.setattr(remote, name, value)
    clock = [50_000.0]
    monkeypatch.setattr(remote, "_now", lambda: clock[0])
    return clock


@pytest.fixture
def store(monkeypatch, tmp_path, memory):
    """A configured evidence store whose restore refuses or succeeds as the test says, counting each real try."""
    monkeypatch.setenv("PARLAYPICKER_DRIVE_FOLDER_ID", "synthetic-folder")
    monkeypatch.setenv("PARLAYPICKER_EVIDENCE_DIR", str(tmp_path / "evidence"))
    state = {"refuse": True, "tries": []}

    def restore(path=None, **options):
        state["tries"].append(options)
        if state["refuse"]:
            raise EvidenceStorageError(REFUSAL)
        return 0
    monkeypatch.setattr(remote, "restore", restore)
    state["clock"] = memory
    return state


def test_a_restore_that_just_failed_is_not_tried_again_until_the_wait_is_over(store, tmp_path):
    path = tmp_path / "evidence.sqlite3"
    with pytest.raises(remote.EvidenceRestoreError, match=REFUSAL) as first:
        remote.restore_once(path)
    assert isinstance(first.value, RuntimeError)                    # what every caller already catches
    assert remote.remote_status()["error"] == REFUSAL and len(store["tries"]) == 1
    with pytest.raises(remote.EvidenceRestoreError, match=REFUSAL):
        remote.restore_once(path)                                   # the same Refresh asks again a moment later
    connect(path).close()                                           # and by then the local database exists
    store["clock"][0] += remote.RETRY_AFTER_SECONDS - 1
    with pytest.raises(remote.EvidenceRestoreError, match=REFUSAL) as later:
        remote.restore_once(path)
    assert str(later.value) == (REFUSAL + " (not tried again: that was 29 min ago and the next try comes 30 min after it; "
                                "Workspace > Prediction Evidence Status > Restore and sync evidence storage tries now)")
    assert len(store["tries"]) == 1                                 # the folder was listed once, not three times
    assert remote.remote_status()["error"] == REFUSAL               # the status keeps the refusal itself
    store["clock"][0] += 1
    with pytest.raises(remote.EvidenceRestoreError, match=REFUSAL):
        remote.restore_once(path)                                   # the wait is over: it looks again
    assert len(store["tries"]) == 2
    store["refuse"] = False                                         # the duplicate has been removed meanwhile
    with pytest.raises(remote.EvidenceRestoreError, match=REFUSAL):
        remote.restore_once(path)                                   # not known yet: the last try was a moment ago
    remote.restore_once(path, full_verification=True)               # a full verification always tries
    assert len(store["tries"]) == 3 and store["tries"][-1]["full_verification"] is True
    assert remote._failed == {}
    remote.restore_once(path)                                       # restored: nothing more to do
    assert len(store["tries"]) == 3


def test_a_request_that_failed_is_tried_again_sooner_than_a_store_that_refused_its_contents(store, tmp_path, monkeypatch):
    def unreachable(path=None, **options):
        store["tries"].append(options)
        raise ConnectionError("PRIVATE_MARKER connection reset")
    monkeypatch.setattr(remote, "restore", unreachable)
    path = tmp_path / "evidence.sqlite3"
    with pytest.raises(remote.EvidenceRestoreError) as first:
        remote.restore_once(path)
    assert "PRIVATE_MARKER" not in str(first.value)                 # only the safe message is kept and shown
    with pytest.raises(remote.EvidenceRestoreError, match="not tried again: that was 0 min ago and the next try comes 5 min after it"):
        remote.restore_once(path)                                   # the rest of that Refresh does not list again
    store["clock"][0] += remote.RETRY_AFTER_FAILED_REQUEST_SECONDS
    assert remote.RETRY_AFTER_FAILED_REQUEST_SECONDS < remote.RETRY_AFTER_SECONDS
    with pytest.raises(remote.EvidenceRestoreError):
        remote.restore_once(path)                                   # the next Refresh does
    assert len(store["tries"]) == 2


def test_the_wait_is_kept_for_each_store_and_database_apart(store, tmp_path, monkeypatch):
    one, other = tmp_path / "one" / "evidence.sqlite3", tmp_path / "other" / "evidence.sqlite3"
    with pytest.raises(remote.EvidenceRestoreError):
        remote.restore_once(one)
    store["refuse"] = False
    remote.restore_once(other)                                      # another database is not held back by the first
    assert len(store["tries"]) == 2
    monkeypatch.setenv("PARLAYPICKER_DRIVE_PREFIX", "another/prefix")
    remote.restore_once(one)                                        # nor is the same database against another store
    assert len(store["tries"]) == 3
    monkeypatch.delenv("PARLAYPICKER_DRIVE_FOLDER_ID")
    assert remote.restore_once(one) is None and len(store["tries"]) == 3      # no remote store configured: nothing to try


def test_once_the_duplicate_is_gone_an_explicit_restore_is_tried_at_once_and_clears_the_wait(world, monkeypatch):
    """The real restore against the synthetic Drive: what the owner does after moving the later copy to the trash."""
    monkeypatch.setattr(remote, "_failed", {})
    _add_run(world, START - DAY)
    first, second = _again(world, "bundles", lambda row: row.__setitem__(1, "2031-03-10T14:59:00Z"))
    path = world.fresh()
    with pytest.raises(remote.EvidenceRestoreError, match=REFUSAL):
        remote.restore_once(path)
    asked = len(world.drive.asked)
    with pytest.raises(remote.EvidenceRestoreError, match=REFUSAL):
        remote.restore_once(path)
    assert len(world.drive.asked) == asked                           # Drive was asked nothing the second time
    world.drive.files.remove(second)                                 # the later copy is in the trash: listings leave it out
    assert remote.restore(path) == 1                                 # "Restore and sync evidence storage" tries at once
    assert remote._failed == {} and remote.remote_status()["error"] is None
    asked = len(world.drive.asked)
    remote.restore_once(path)                                        # and the next Refresh starts from the restored store
    assert len(world.drive.asked) == asked
    with connect(path) as db:
        assert db.execute("SELECT version, frozen_at FROM bundles").fetchall() == [(_sha(MANIFEST), "2031-01-01T00:00:00Z")]
    assert first["name"].startswith(f"{PREFIX}/bundles/")


def _pool(tmp_path, monkeypatch):
    row, _, _ = source(tmp_path)
    _, diagnostics = build([row], monkeypatch)
    return diagnostics.get("candidate_authority_df", diagnostics["candidate_audit_df"])


def test_review_preparation_without_the_history_gives_what_a_store_with_no_plan_gives(tmp_path, monkeypatch):
    from app_core.controlled_trial_pipeline import prepare_review_candidates
    from core import prospective_uncertainty as uncertainty
    pool = _pool(tmp_path, monkeypatch)
    real = uncertainty.prepare_live
    # A store that can be read and holds no validation plan: no forecast is made, as in production today.
    monkeypatch.setattr(uncertainty, "prepare_live",
                        lambda frame: real(frame, database=tmp_path / "prior.db", plan_dir=tmp_path / "plans", now=NOW))
    readable = {"candidate_authority_df": pool.copy()}
    expected_pool, expected_trials = prepare_review_candidates(readable, now=NOW)
    assert "evidence_history_unavailable" not in readable and len(expected_pool) == len(pool) == 1

    def unreadable(frame):
        raise remote.EvidenceRestoreError(REFUSAL)
    monkeypatch.setattr(uncertainty, "prepare_live", unreadable)
    diagnostics = {"candidate_authority_df": pool.copy()}
    got_pool, got_trials = prepare_review_candidates(diagnostics, now=NOW)
    pd.testing.assert_frame_equal(got_pool, expected_pool)
    pd.testing.assert_frame_equal(got_trials, expected_trials)
    assert diagnostics.pop("evidence_history_unavailable") == REFUSAL
    assert diagnostics.keys() == readable.keys()
    assert all(diagnostics[key] == readable[key] for key in diagnostics if key != "candidate_authority_df")


@pytest.mark.parametrize("error", [ValueError("Saved snapshot changed"), RuntimeError("anything else"),
                                   EvidenceStorageError("Restore conflicts with immutable local evidence")])
def test_review_preparation_still_stops_on_any_other_failure(tmp_path, monkeypatch, error):
    """Only an evidence store that could not be restored is passed over. Evidence that reads wrong still stops the run."""
    from app_core.controlled_trial_pipeline import prepare_review_candidates
    from core import prospective_uncertainty as uncertainty

    def broken(frame):
        raise error
    monkeypatch.setattr(uncertainty, "prepare_live", broken)
    diagnostics = {"candidate_authority_df": _pool(tmp_path, monkeypatch)}
    with pytest.raises(type(error), match=str(error)):
        prepare_review_candidates(diagnostics, now=NOW)
    assert "evidence_history_unavailable" not in diagnostics


def test_a_refresh_whose_evidence_cannot_be_restored_still_returns_its_board(store, tmp_path, monkeypatch):
    """The Refresh of 19 November, end to end: begin_run and the review step both reach the real restore_once."""
    import streamlit_app as app
    from core import live_wager_contract as authority
    row, policy, config = source(tmp_path)
    row.pop("exact_quote_verified")
    row.pop("quote_time")
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats", lambda: {})
    monkeypatch.setattr("core.probability_calibration.load_calibration", lambda: None)
    monkeypatch.setattr(app, "run_analysis_pipeline", lambda **kwargs: (pd.DataFrame([row]), pd.DataFrame(), {}))
    monkeypatch.setattr(app, "optimize_portfolio_allocation", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(app, "generate_parlays", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(app, "run_bankroll_simulation", lambda *a, **k: {})
    monkeypatch.setattr(app, "_enrich_with_kalshi_safe", lambda df: (df, None))
    monkeypatch.setattr(app, "_recompute_consensus_from_kalshi", lambda df, require_ml=False: df)
    real_finalize = authority.finalize_live_wagers
    monkeypatch.setattr(authority, "finalize_live_wagers", lambda frame, best, bankroll:
                        real_finalize(frame, best, bankroll, now=NOW, policies={"NFL": policy}, config=config, reviews=frame))
    state, warnings, errors = app._run_pipeline({"sports": ["NFL"], "use_ml": False, "theover_spreads": None,
                                                 "theover_totals": None, "bankroll": 1000., "use_gemini": False})
    assert len(store["tries"]) == 1                                  # the folder was listed once in the whole Refresh
    assert f"Prediction evidence could not initialize: {REFUSAL}" in warnings
    board, diagnostics = state["best_picks_df"], state["diagnostics"]
    assert len(board) == 1 and board.iloc[0]["best_pick"] == "Indianapolis Colts -2.5"
    # The review step was told the same refusal, and that the folder was not being listed again for it.
    assert diagnostics["evidence_history_unavailable"] == (
        REFUSAL + " (not tried again: that was 0 min ago and the next try comes 30 min after it; "
                  "Workspace > Prediction Evidence Status > Restore and sync evidence storage tries now)")
    assert not diagnostics.get("prediction_snapshot_saved", False)   # nothing was saved, and the warning said so
    assert not (tmp_path / "evidence" / "evidence.sqlite3").exists() or not connect().execute("SELECT 1 FROM snapshots").fetchall()


def test_two_refreshes_at_once_register_one_bundle(world, monkeypatch):
    """Both find no bundle for the new code and both go to register it, each with its own freeze time."""
    monkeypatch.setattr(remote, "_bundles", {})
    version = _sha(MANIFEST)
    both_inside = threading.Barrier(2, timeout=5)
    plain = world.store.put_object

    def slow_search_then_create(**request):
        # Drive's search has not caught up with a file made a moment ago: each caller that gets this far finds no
        # file of that name. Without the lock both get here together, and both create.
        try:
            both_inside.wait(timeout=0.5)
        except threading.BrokenBarrierError:
            pass
        return plain(**request)
    monkeypatch.setattr(world.store, "put_object", slow_search_then_create)
    with ThreadPoolExecutor(2) as pool:
        frozen = list(pool.map(lambda moment: remote.register_bundle(version, moment, MANIFEST),
                               ["2031-03-10T15:00:00Z", "2031-03-10T15:00:01Z"]))
    files = [item for item in world.drive.files if "/bundles/" in item["name"]]
    assert len(files) == 1                                           # one Drive file, not two under one name
    assert len(set(frozen)) == 1 and frozen[0] in {"2031-03-10T15:00:00Z", "2031-03-10T15:00:01Z"}
    assert remote._decode(files[0]["content"], "bundles") == (version, frozen[0], MANIFEST)
    uploads = [detail for kind, detail in world.drive.asked if kind == "upload"]
    assert uploads == [files[0]["name"]]
    # A later Refresh in the same process takes the same freeze time and asks Drive nothing.
    asked = len(world.drive.asked)
    assert remote.register_bundle(version, "2031-03-10T16:00:00Z", MANIFEST) == frozen[0]
    assert len(world.drive.asked) == asked
    assert remote.restore(world.fresh()) == 0                        # and the store restores: no run yet, no conflict


def test_a_bundle_another_process_registered_first_is_adopted_as_before(world, monkeypatch):
    monkeypatch.setattr(remote, "_bundles", {})
    _add_run(world, START - DAY)                                     # its bundle was frozen on 2031-01-01 by someone else
    assert remote.register_bundle(_sha(MANIFEST), "2031-03-10T15:00:00Z", MANIFEST) == "2031-01-01T00:00:00Z"
    assert len([item for item in world.drive.files if "/bundles/" in item["name"]]) == 1
    monkeypatch.delenv("PARLAYPICKER_DRIVE_FOLDER_ID")
    assert remote.register_bundle("unsaved", "2031-03-10T15:00:00Z", MANIFEST) == "2031-03-10T15:00:00Z"   # no remote store
