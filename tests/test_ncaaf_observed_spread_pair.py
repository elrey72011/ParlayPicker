"""Explicit synthetic spread masses and exact two-side private computation."""
from copy import deepcopy
import json
import math
import pytest
from app_core import ncaaf_owner_spread as owner,ncaaf_spread_pair as pair,ncaaf_spread_mass as mass
from app_core import ncaaf_pipeline_evidence as adapter,ncaaf_research as native
from test_ncaaf_model_compatibility import synthetic
from ncaaf_future_pair_fixtures import fixture,pair_fixture,inference_authorization,NOW


@pytest.mark.parametrize('line',[-7,0,7,-3.5,3.5])
def test_actual_single_caller_unchanged_native_masses(synthetic,monkeypatch,tmp_path,line):
    packet,row=fixture(synthetic,monkeypatch,tmp_path,line)
    owner.load(packet)
    a=inference_authorization(packet,monkeypatch)
    result=owner.admitted_analysis(packet,row,authorization=a,custody_root=tmp_path)
    r=result.loc[result.market_type.eq('spread_home')].iloc[0]
    assert r.ml_inference_status=='success',r.ml_unavailable_reason
    p=json.loads(r.ml_estimate_metadata)['ncaaf_inputs']['payload'];saved=p['computation']['payload']
    receipt=saved['custody_computation']['payload']['native_computation']['payload']
    expected=mass.from_native(native.probabilities(receipt['projection'],receipt['sigma'],-line,total=False),'spread_home')
    assert saved['outcome_probabilities']==expected and r.ml_probability==expected['cover']
    assert r.push_probability==expected['push'] and r.loss_probability==expected['loss']
    owner.inspect_result(packet,p['computation'],p['inference_time'])
    assert adapter._offer_matches(packet,r), (r.get('market_type'),r.get('spread_line'),r.get('market_line_used'))
    from app_core.research_estimate_trace import origin_rejection,encode
    original=json.loads(r.ml_estimate_metadata);original.pop('ncaaf_inputs')
    projected=owner.projection.presentation(packet,r)
    assert origin_rejection(dict(projected,ml_estimate_metadata=encode(original))) is None,(
        origin_rejection(dict(projected,ml_estimate_metadata=encode(original))),
        {k:(v.get('value'),projected.get(k)) for k,v in original['identity'].items() if v.get('state')=='VALUE' and str(v.get('value'))!=str(projected.get(k))},
        {k:(original[k],r.get(k)) for k in ('line','market_type','predictor_id','target')})
    assert adapter.diagnose(r)==dict(status='COMPLETE',reason='AVAILABLE')
    assert saved['wager_action']=='PASS' and saved['live_stake']==0


@pytest.mark.parametrize('line',[-7,0,7,-3.5,3.5])
def test_actual_pair_same_market_both_results_complement(synthetic,monkeypatch,tmp_path,line):
    packet,rows=pair_fixture(synthetic,monkeypatch,tmp_path,line);before=deepcopy(packet)
    a=inference_authorization(packet,monkeypatch,paired=True)
    out=pair.admitted_analysis(packet,rows,authorization=a,custody_root=tmp_path)
    assert len(out)==2 and out.ml_inference_status.eq('success').all()
    saved=out.attrs['ncaaf_pair_result'];p=pair.inspect_result(packet,saved);c=p['comparison']
    assert math.isclose(c['home_cover']+c['away_cover']+c['shared_push'],1.,abs_tol=1e-12)
    assert all(adapter.diagnose(row)==dict(status='COMPLETE',reason='AVAILABLE') for row in out.to_dict('records'))
    assert packet==before and len(p['children'])==2
    assert c['higher_cover'] in {'home','away','tie'} and p['live_stake']==0 and p['wager_action']=='PASS'
    with pytest.raises(ValueError,match='ALREADY_SPENT'):
        pair.admitted_analysis(packet,rows,authorization=a,custody_root=tmp_path)


def test_zero_denominator_and_probability_contract_are_explicit():
    assert mass.conditional_no_push(dict(cover=0.,push=1.,loss=0.))==dict(semantics='cover_conditional_on_no_push',state='UNKNOWN',value=None)
    assert mass.conditional_no_push(dict(cover=.4,push=.2,loss=.4))['value']==pytest.approx(.5)
    assert mass.compare(dict(cover=.4,push=.2,loss=.4),dict(cover=.4,push=.2,loss=.4))['higher_cover']=='tie'
    with pytest.raises(ValueError):mass.normalize(dict(cover=.4,push=.3,loss=.4))
