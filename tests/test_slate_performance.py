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


def test_strategy_render_never_fetches_until_refresh(monkeypatch):
    from app.ui import strategy_lab_dashboard as dash
    calls=[]
    monkeypatch.setattr(dash,'run_performance_pipeline',lambda: calls.append(1) or pd.DataFrame())
    at=AppTest.from_string('import pandas as pd\nfrom app.ui.strategy_lab_dashboard import _render_realized_strategy_lab\n_render_realized_strategy_lab(pd.DataFrame())').run()
    assert not at.exception and not calls
    at.run()
    assert not calls
    at.button(key='strategy_refresh_scores').click().run()
    assert not at.exception and calls==[1]
    at.run()
    assert calls==[1]


def test_timer_records_stages_and_reports_progress(monkeypatch):
    from app_core import stage_timing
    ticks=iter([0,0,2,2,5])
    monkeypatch.setattr(stage_timing,'perf_counter',lambda: next(ticks))
    messages=[]
    timer=StageTimer(messages.append)
    timer.start('Inputs');timer.start('Models');timer.finish()
    assert timer.timings=={'Inputs':2,'Models':3}
    assert messages==['Inputs','Models']
