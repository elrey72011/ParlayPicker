"""Deployment state is sport-local and never inferred from an APPROVED label."""
from dataclasses import replace
import pytest
from test_wager_integrity_audit import candidate,policy,NOW
from core.wager_decisions import candidate_decision
from core.sport_policy import research_policies

@pytest.mark.parametrize('sport',['NFL','NCAAF','MLB','NBA','NHL','NCAAB'])
@pytest.mark.parametrize('state,allowed',[
 ('UNVALIDATED',set()),('PROVISIONAL_VALIDATED',{'PROVISIONAL'}),
 ('STANDARD_VALIDATED',{'PROVISIONAL','STANDARD'}),
 ('PREMIUM_VALIDATED',{'PROVISIONAL','STANDARD','PREMIUM'})])
def test_deployment_limits_maturity(sport,state,allowed):
 p=replace(policy(sport),deployment_state=state,provisional_allowed=True)
 for maturity in ['RESEARCH','QUALIFIED','PROVISIONAL','STANDARD','PREMIUM']:
  r=candidate_decision(candidate(sport=sport,book='Novig',maturity=maturity),p,NOW)
  assert (r['recommended_fraction']>0)==(maturity in allowed)
  assert r['maturity']==maturity

@pytest.mark.parametrize('sport',['NFL','NCAAF'])
def test_football_separate_validated_provisional_evidence(sport):
 p=replace(policy(sport),deployment_state='PROVISIONAL_VALIDATED',minimum_evidence=200,provisional_minimum_evidence=20)
 r=candidate(sport=sport,maturity='PROVISIONAL',evidence_effective_sample_size=30)
 assert candidate_decision(r,p,NOW)['production_eligible']
 assert not candidate_decision(dict(r,evidence_effective_sample_size=10),p,NOW)['production_eligible']
 assert not candidate_decision(dict(r,calibration_validated=False),p,NOW)['production_eligible']

def test_default_states_do_not_authorize_even_with_legacy_approval():
 for sport,p in research_policies().items():
  r=candidate_decision(candidate(sport=sport,book='Novig',APPROVED=True),p,NOW)
  assert p.deployment_state=='UNVALIDATED' and r['recommended_fraction']==0
