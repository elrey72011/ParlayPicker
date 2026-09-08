import json
from types import SimpleNamespace
import pandas as pd
import pytest
from app_core import gemini_review_budget as budget
from app_core.gemini_review_comparison import review_comparison
from integrations.gemini_client import verified_context, _attach_gemini_results

@pytest.fixture(autouse=True)
def local_store(tmp_path, monkeypatch):
    monkeypatch.setenv('PARLAYPICKER_EVIDENCE_DIR', str(tmp_path))
    monkeypatch.setenv('PARLAYPICKER_GEMINI_DAILY_REQUESTS', '2')


def test_budget_cap_and_cache_expiry(monkeypatch):
    assert budget.reserve() and budget.reserve()
    assert not budget.reserve()
    key = budget.request_key('model', 'payload')
    budget.save(key, {'a': 1})
    assert budget.lookup(key) == {'a': 1}
    assert budget.lookup(budget.request_key('other', 'payload')) is None
    now = budget.time.time()
    monkeypatch.setattr(budget.time, 'time', lambda: now+601)
    assert budget.lookup(key) is None


def test_invalid_cap_blocks_requests(monkeypatch):
    monkeypatch.setenv('PARLAYPICKER_GEMINI_DAILY_REQUESTS', 'invalid')
    assert not budget.reserve()


def test_context_requires_source_and_recent_timestamp():
    now = pd.Timestamp.now(tz='UTC')
    assert verified_context({'weather':'rain'}) == {}
    row = {'weather':'rain', 'weather_source':'provider', 'weather_recorded_at':now.isoformat(),
           'injuries':'none', 'injuries_source':'provider', 'injuries_recorded_at':(now-pd.Timedelta(hours=2)).isoformat()}
    assert list(verified_context(row)) == ['weather']


def test_invented_evidence_cannot_authorize_review():
    df = pd.DataFrame([{'best_pick':'Over 8', 'gemini_verified_context':'{}'}])
    response = {'x': {'recommended_bet':'Over 8','explanation':'x','risk_notes':'x','confidence':'HIGH','flags':[], 'supporting_evidence':['invented']}}
    reviewed = _attach_gemini_results(df, ['x'], response)
    assert not reviewed.iloc[0].gemini_reviewed
    assert reviewed.iloc[0].gemini_agreement == 'unavailable'


def test_comparison_excludes_postgame_and_duplicate_runs():
    base = {'matchup_id':'g1','best_pick':'Over 8','game_start_utc':'2026-09-09T20:00:00Z',
            'gemini_review_model':'model','gemini_review_input_hash':'a'*64,
            'gemini_reviewed_at':'2026-09-09T19:00:00Z','gemini_agreement':'agree','candidate_outcome':'WIN'}
    rows = [base, dict(base, gemini_reviewed_at='2026-09-09T19:30:00Z'),
            dict(base, matchup_id='g2', gemini_reviewed_at='2026-09-09T21:00:00Z'),
            dict(base, matchup_id='g3', gemini_agreement='disagree', candidate_outcome='LOSS')]
    report = review_comparison(pd.DataFrame(rows))
    assert report.Selections.tolist() == [2, 1]
    assert report['Win rate'].tolist() == [.5, 1.0]


def test_batch_cache_and_exhaustion_do_not_call_provider(monkeypatch):
    from app_core import llm_assistant as llm
    class Models:
        calls = 0
        def generate_content(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(text=json.dumps([{'game_id':'g', 'recommended_bet':'Over 8', 'confidence':'MEDIUM', 'explanation':'x','risk_notes':'x','flags':[]}]))
    models = Models()
    monkeypatch.setattr(llm, '_GEMINI_AVAILABLE', True)
    monkeypatch.setattr(llm, 'initialize_gemini', lambda: (SimpleNamespace(models=models), None))
    monkeypatch.setattr(llm, 'genai', SimpleNamespace(types=SimpleNamespace(GenerateContentConfig=lambda **k:k)))
    monkeypatch.setattr(llm.time, 'sleep', lambda _:None)
    data = [{'game_id':'g','best_pick':'Over 8'}]
    first = llm.generate_batch_confidence_explanation(data)
    assert llm.generate_batch_confidence_explanation(data) == first
    assert models.calls == 1
    monkeypatch.setenv('PARLAYPICKER_GEMINI_DAILY_REQUESTS','0')
    assert llm.generate_batch_confidence_explanation([{'game_id':'different'}]) == {}
    assert models.calls == 1

def test_exact_candidate_context_reaches_review(monkeypatch):
    from integrations.gemini_client import run_gemini_analysis
    from app_core import llm_assistant
    captured = []
    monkeypatch.setattr(llm_assistant, 'generate_batch_confidence_explanation', lambda rows, state: captured.extend(rows) or {})
    frame = pd.DataFrame([{'matchup_id':'g', 'game_id':'g', 'market_type':'total_over','best_pick':'Over 8'}])
    audit = frame.assign(weather='rain', weather_source='provider', weather_recorded_at=pd.Timestamp.now(tz='UTC').isoformat())
    run_gemini_analysis(frame, {}, audit)
    assert captured[0]['verified_context']['weather']['value'] == 'rain'
    assert 'lineups' in captured[0]['missing_context']
