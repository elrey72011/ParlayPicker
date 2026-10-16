"""SYNTHETIC, offline PDF margin prose; no authentic numerical execution."""
import base64
from copy import deepcopy
import hashlib
import json
import pytest
from scripts.benchmark_drive_history_loading import blocked_network
from test_nfl_pdf_evidence import fixture, make_pdf, resign
from test_source_contract_pipeline import INFERENCE

PROSE = "calculated as <team> 's points minus <opponent> 's points recorded within <time period>"
OLD = "reference team's final score minus opposing team's final score"


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():
        yield


def prose_fixture(monkeypatch, away=False):
    from app_core import nfl_pdf_evidence as pdf, nfl_pdf_interpretation as interpretation
    packet, baseline = fixture(monkeypatch, away=away)
    p = packet['payload']; obj = p['objects']['rules']
    text = obj['extraction']['pages'][0]['text']
    assert OLD in text
    raw = make_pdf(text.replace(OLD, PROSE))
    p['objects']['rules'] = pdf.retained_pdf(raw, receipt=deepcopy(obj['receipt']),
                                          prepared_at=obj['extraction']['prepared_at'])
    obj = p['objects']['rules']; page = obj['extraction']['pages'][0]
    for field, fact in p['projections']['rules']['payload']['facts'].items():
        for support in fact['supports']:
            exact = PROSE if field == 'reference_definition' else support['pdf']['exact']
            start = page['text'].index(exact)
            support['pdf'].update(page=0, start=start, end=start+len(exact), exact=exact,
                                  page_sha256=page['text_sha256'], extraction_sha256=pdf.digest(obj['extraction']))
    resign(p)
    return interpretation.seal(p), baseline


@pytest.mark.parametrize('text', [PROSE, PROSE.replace(" 's", "'s"), PROSE.replace(' ', '\n \n')])
def test_exact_template_margin_preserves_direction_and_value(text):
    from app_core.nfl_pdf_interpretation import normalize
    assert normalize('reference_definition', text) == normalize('reference_definition', OLD)


@pytest.mark.parametrize('text', [
    "calculated as <opponent> 's points minus <team> 's points recorded within <time period>",
    PROSE.replace('<team>', '<home>'), PROSE.replace('<opponent>', '<other team>'),
    PROSE.replace('minus', 'plus'), PROSE.replace('<time period>', 'first half'),
    PROSE.replace('<time period>', 'regulation'), PROSE+' unless the opponent is ahead',
    PROSE.replace('<team>', 'Seattle Seahawks'), PROSE.replace('minus', 'less than'),
])
def test_other_direction_teams_periods_or_qualifiers_remain_unsupported(text):
    from app_core.nfl_pdf_interpretation import normalize
    with pytest.raises(ValueError, match='NFL_PRIVATE_INTERPRETATION_UNSUPPORTED'):
        normalize('reference_definition', text)


@pytest.mark.parametrize('away', [False, True])
def test_prose_actual_caller_capture_replay_display_matches_native(monkeypatch, tmp_path, away):
    from app_core import nfl_pdf_research as reader, nfl_private_routes as routes, research_replay
    from test_nfl_owner_research import actual_pipeline, private_app
    from test_source_contract_pipeline import exported
    from streamlit.testing.v1 import AppTest
    packet, baseline = prose_fixture(monkeypatch, away)
    original = deepcopy(packet)
    features, _ = reader.checked(packet, INFERENCE)
    assert features == packet['payload']['features']
    with routes.selected([packet]):
        analysis = actual_pipeline(monkeypatch)
    i = int(away); row = analysis.iloc[i]
    assert row.ml_probability == pytest.approx(baseline.iloc[i].ml_probability, abs=1e-12)
    result = exported(monkeypatch, tmp_path, analysis)
    receipt = research_replay.retain_export(result['frames'], result['package'], result['card'], result['captured'], path=result['db'])
    saved, sources = research_replay.read_export(receipt['export_id'], path=result['db'])
    retained = research_replay.frame_from_payload(next(iter(sources.values()))['original']['producer'])
    display = routes.private_display(retained.iloc[i])
    assert display['raw_probability'] == row.ml_probability and display['calibrated'] is False
    assert json.loads(retained.iloc[i].ml_estimate_metadata)['nfl_private_inputs']['payload']['original_packet'] == original
    assert 'bytes_base64' not in json.dumps(saved['package'])
    assert all(c['production_bet_amount'] == 0 for c in result['card'].wager_contract)
    assert all(r['status'] == 'PASS' and r['win_estimate'] is None and r['ev'] is None
               for family in ('overall','sides','totals') for r in saved['package']['games'][family])
    app = AppTest.from_function(private_app); app.session_state['private_candidates'] = retained; app.run()
    assert not app.exception and any(m.value == f'{row.ml_probability:.2%}' for m in app.metric)
    assert any('Original PDF bytes' in c.value for c in app.caption)
    assert original == packet


@pytest.mark.parametrize('attack', ['body', 'extraction', 'locator', 'output', 'review', 'missing_review',
                                  'period', 'applicability', 'edition', 'orientation', 'old_review'])
def test_new_wording_cannot_supply_other_facts_or_borrow_review(monkeypatch, attack):
    from app_core import nfl_pdf_research as reader, nfl_pdf_interpretation as interpretation
    packet, _ = prose_fixture(monkeypatch); p = packet['payload']
    if attack == 'body':
        p['objects']['rules']['bytes_base64'] = base64.b64encode(b'SYNTHETIC substituted original').decode()
    elif attack == 'extraction':
        p['objects']['rules']['extraction']['pages'][0]['text'] += ' SYNTHETIC change'
    elif attack == 'locator':
        p['projections']['rules']['payload']['facts']['reference_definition']['supports'][0]['pdf']['start'] += 1
    elif attack == 'output':
        p['projections']['rules']['payload']['values']['reference_definition'] = 'opponent_minus_reference_final_points'
    elif attack == 'review':
        p['projections']['rules']['payload']['review']['subject_sha256'] = '0'*64
    elif attack == 'missing_review':
        p['owner_review'] = {}
    elif attack == 'period':
        p['projections']['rules']['payload']['values']['period'] = 'first_half'
    elif attack == 'applicability':
        p['projections']['listing']['payload']['values']['applicability']['provider_event_id'] = 'OTHER'
    elif attack == 'edition':
        p['projections']['rules']['payload']['values']['rule_edition'] = 'NFL-001 edition 2026-10-02'
    elif attack == 'orientation':
        p['event']['home'], p['event']['away'] = p['event']['away'], p['event']['home']
    else:
        p['owner_review']['version'] = 'nfl-owner-interpreted-verification-v4'
    # Rehashing an outer envelope grants no new custody, interpretation or review.
    with pytest.raises(ValueError):
        reader.checked(interpretation.seal(p), INFERENCE)


@pytest.mark.parametrize('attack', ['body', 'extraction', 'locator', 'output', 'period', 'applicability', 'edition'])
def test_rehashed_projections_still_reject_forged_facts(monkeypatch, attack):
    from app_core import nfl_pdf_research as reader, nfl_pdf_interpretation as interpretation
    packet, _ = prose_fixture(monkeypatch); p = packet['payload']
    if attack == 'body':
        raw = make_pdf('SYNTHETIC substituted PDF with different original facts')
        p['objects']['rules']['bytes_base64'] = base64.b64encode(raw).decode()
        p['objects']['rules']['body_sha256'] = hashlib.sha256(raw).hexdigest()
    elif attack == 'extraction':
        p['objects']['rules']['extraction']['pages'][0]['text'] += ' SYNTHETIC forged text'
    elif attack == 'locator':
        p['projections']['rules']['payload']['facts']['reference_definition']['supports'][0]['pdf']['exact'] = PROSE.replace('minus','plus')
    elif attack == 'output':
        p['projections']['rules']['payload']['values']['reference_definition'] = 'opponent_minus_reference_final_points'
    elif attack == 'period':
        p['projections']['rules']['payload']['values']['period'] = 'first_half'
    elif attack == 'applicability':
        p['projections']['listing']['payload']['values']['applicability']['listing_id'] = 'OTHER'
    else:
        p['projections']['rules']['payload']['values']['rule_edition'] = 'NFL-001 edition 2026-10-02'
    resign(p)  # SYNTHETIC review fixture only; not an owner action.
    with pytest.raises(ValueError):
        reader.checked(interpretation.seal(p), INFERENCE)
