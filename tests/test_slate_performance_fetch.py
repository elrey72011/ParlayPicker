# Re-issued 2026-11-18 without the tests of panels the owner removed from the app; the original is in tests_archive/, unchanged.
import pandas as pd
from streamlit.testing.v1 import AppTest
from app_core import feature_processing as fp
from app_core.stage_timing import StageTimer


def test_fetches_only_requested_leagues(monkeypatch):
    calls=[]
    for league,name in [('NBA','fetch_nba_stats'),('NFL','fetch_nfl_stats'),('NCAAF','fetch_ncaaf_stats'),('NHL','fetch_nhl_stats'),('NCAAB','fetch_ncaab_stats'),('MLB','fetch_from_espn_mlb'),('WNBA','fetch_from_espn_wnba')]:
        monkeypatch.setattr(fp,name,lambda year, lg=league: calls.append((lg,year)) or [])
    fp.fetch_team_stats({},2026,leagues={'MLB','NFL'})
    assert calls==[('NFL',2026),('MLB',2026)]
    calls.clear()
    fp.fetch_team_stats({},2025,leagues=set())
    assert calls==[]


def test_enrichment_scopes_current_and_fallback(monkeypatch):
    calls=[]
    def fetch(clients,season_year=None,*,leagues=None):
        calls.append((season_year,leagues))
        return pd.DataFrame()
    monkeypatch.setattr(fp,'fetch_team_stats',fetch)
    fp.enrich_with_model_features(pd.DataFrame([{'league':'MLB','home_team':'Boston','away_team':'Seattle'}]),{},season_year=2026)
    assert calls==[(2026,{'MLB'}),(2025,{'MLB'})]


def test_timer_records_stages_and_reports_progress(monkeypatch):
    from app_core import stage_timing
    ticks=iter([0,0,2,2,5])
    monkeypatch.setattr(stage_timing,'perf_counter',lambda: next(ticks))
    messages=[]
    timer=StageTimer(messages.append)
    timer.start('Inputs');timer.start('Models');timer.finish()
    assert timer.timings=={'Inputs':2,'Models':3}
    assert messages==['Inputs — 0s elapsed','Models — 2s elapsed']
