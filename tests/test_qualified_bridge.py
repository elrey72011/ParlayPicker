"""C6 bridge: qualified bound predictions become rows the market gate accepts. SYNTHETIC store."""
from datetime import datetime, timedelta

from app_core import prospective_evidence as pe
from app_core import qualified_bridge as qb
from core.live_wager_contract import adapt_candidate
from core.sport_market_gate import market_gate
from tests.test_prospective_evidence import (  # noqa: F401  (clock is a fixture)
    _trained_market, at, clock, prediction, provider_event, provider_market)


def _upcoming(path, clock, *, book="draftkings", event_id="upcoming"):  # noqa: F811
    clock(hours=8)
    pe.insert_event(path, dict(event_id=event_id, sport="NBA", game_id=event_id, provider_namespace="THE_ODDS_API",
                               provider_event_id=event_id, home_team="Home", away_team="Away",
                               home_team_id="provider:1", away_team_id="provider:2", scheduled_start=at(12),
                               observed_at=at(8), source_id=f"event-{event_id}",
                               raw_source=provider_event(event_id, 12, 8, book=book)))
    pe.insert_quote(path, dict(quote_id=f"{event_id}-quote", event_id=event_id, sport="NBA", market_family="SPREAD",
                               selection="Home", line=-2.5, american_odds=-110, decimal_odds=1.909090909,
                               sportsbook=book, quote_timestamp=at(8), quote_source="THE_ODDS_API",
                               quote_verified=True, source_id=f"{event_id}:{book}:spreads:Home:{at(8)}",
                               raw_source=provider_market(8)))
    prediction(path, event_id=event_id, quote_id=f"{event_id}-quote", observation_id=f"{event_id}-pred", stamp=8)


def _deployment(row):
    return dict(sport="NBA", market_family="SPREAD", deployment_state="PROVISIONAL_VALIDATED",
                validation_state="PROVISIONAL_VALIDATED", validation_id="validation", artifact_id="artifact",
                **{k: row[k] for k in ("model_id", "model_version", "calibration_id", "calibration_version")})


def test_bridge_row_passes_every_row_level_gate_check(tmp_path, clock):  # noqa: F811
    path = tmp_path / "SYNTHETIC.sqlite3"
    _trained_market(path, clock)
    _upcoming(path, clock)
    now = datetime.fromisoformat(at(8, 10))
    rows = qb.rows(path, {("NBA", "SPREAD")}, now=now)
    assert len(rows) == 1
    row = rows[0]
    assert row["market_type"] == "spread_home" and row["best_pick"] == "Home -2.5" and row["book"] == "draftkings"
    assert row["model_id"] == "model" and row["calibration_id"] == "calibration" and row["team_ids"] == ["provider:1", "provider:2"]
    policy, blockers = market_gate(adapt_candidate(row), _deployment(row), None, None, now, evidence_path=path)
    # Only the owner's activation is missing; every row-level authority check passes.
    assert policy is None and blockers == ["owner_market_activation_or_exposure_missing"]


def test_stale_quote_is_blocked_by_the_gate(tmp_path, clock):  # noqa: F811
    path = tmp_path / "SYNTHETIC.sqlite3"
    _trained_market(path, clock)
    _upcoming(path, clock)
    later = datetime.fromisoformat(at(9))                         # quote is 60 minutes old
    row = qb.rows(path, {("NBA", "SPREAD")}, now=later)[0]
    _, blockers = market_gate(adapt_candidate(row), _deployment(row), None, None, later, evidence_path=path)
    assert "exact_pregame_quote_missing_or_stale" in blockers


def test_only_qualified_upcoming_markets_and_never_pinnacle(tmp_path, clock):  # noqa: F811
    path = tmp_path / "SYNTHETIC.sqlite3"
    _trained_market(path, clock)
    _upcoming(path, clock)
    now = datetime.fromisoformat(at(8, 10))
    assert qb.rows(path, set(), now=now) == []                                   # nothing qualified
    assert qb.rows(path, {("NBA", "TOTAL")}, now=now) == []                       # other market
    assert qb.rows(path, {("NBA", "SPREAD")}, now=datetime.fromisoformat(at(13))) == []   # game started
    path2 = tmp_path / "SYNTHETIC-2.sqlite3"
    clock(hours=0)                                                                # fresh store, fresh clock
    _trained_market(path2, clock)
    _upcoming(path2, clock, book="pinnacle")
    assert qb.rows(path2, {("NBA", "SPREAD")}, now=now) == []                     # reference book


def test_inject_is_off_by_default_and_replaces_only_qualified_market_rows():
    import pandas as pd
    analysis = pd.DataFrame([{"league": "NCAAB", "market_type": "spread_home", "best_pick": "app pick"},
                             {"league": "NCAAB", "market_type": "total_over", "best_pick": "app total"},
                             {"league": "NBA", "market_type": "spread_home", "best_pick": "nba pick"}])
    best = analysis.copy()
    off = qb.inject(analysis, best, setting=lambda *_: "")
    assert off[2] == {"status": "OFF"} and off[0] is analysis
    bridged = [{"league": "NCAAB", "market_type": "spread_away", "best_pick": "Away +3.5",
                "candidate_source": "qualified_bridge"}]
    seen = {}

    def make_rows(path, qualified, now=None, policy_versions=None):
        seen.update(path=path, qualified=qualified, versions=policy_versions)
        return bridged
    config = {"prospective_evidence_path": "evidence.sqlite3",
              "market_activations": {"NCAAB:SPREAD": {"policy": {"version": "ncaab-policy-v1"}}}}
    a, b, diag = qb.inject(analysis, best, setting=lambda *_: "true", configuration=config,
                           assess=lambda path: {("NCAAB", "SPREAD")}, make_rows=make_rows)
    assert diag == {"status": "INJECTED", "qualified_markets": ["NCAAB SPREAD"], "rows": 1}
    assert seen["path"] == "evidence.sqlite3" and seen["versions"] == {"NCAAB:SPREAD": "ncaab-policy-v1"}
    for frame in (a, b):
        assert sorted(frame["best_pick"]) == ["Away +3.5", "app total", "nba pick"]   # only NCAAB spread replaced


def test_inject_with_nothing_qualified_changes_nothing():
    import pandas as pd
    best = pd.DataFrame([{"league": "NCAAB", "market_type": "spread_home", "best_pick": "app pick"}])
    a, b, diag = qb.inject(best, best, setting=lambda *_: "true", configuration={}, assess=lambda path: set())
    assert diag == {"status": "NO_QUALIFIED_MARKET"} and b is best
