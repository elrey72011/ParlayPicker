from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from app_core.public_parlays import build_parlays, build_research_parlays
from app_core.public_board import validate_package
from test_public_parlays import row, NOW


def leg(i, **changes):
    return dict(row(i), **{'status': 'PASS', 'quote_source': 'Novig', 'quote_time': NOW.isoformat(), **changes})


def package(rows):
    qualified = build_parlays(rows, NOW, qualified_only=True)
    return dict(schema_version=5, selection_policy='qualified-v1', research_parlay_policy='positive-edge-v1',
                built_at=NOW.isoformat(), stale_after_minutes=30, games={key: deepcopy(rows) for key in ('overall','sides','totals')},
                props=[], dfs=[], results=[], parlays=qualified,
                research_parlays=build_research_parlays(rows, NOW, qualified_parlays=qualified))


def test_pass_combinations_are_separate_disjoint_and_never_promoted():
    rows = [leg(i) for i in range(8)]
    before = deepcopy(rows)
    result = package(rows)
    validate_package(result)
    assert not result['parlays'] and len(result['research_parlays']) == 3
    assert rows == before
    assert all(ticket['status'] == 'RESEARCH ONLY' and not ticket['approved_legs'] for ticket in result['research_parlays'])
    names = [name for ticket in result['research_parlays'] for leg in ticket['legs'] for name in leg['game'].split(' at ')]
    assert len(names) == len(set(names))
    assert build_research_parlays(list(reversed(rows)), NOW) == result['research_parlays']


@pytest.mark.parametrize('changes', [
    {'ev': 0}, {'ev': -0.1}, {'win_estimate': .4}, {'odds': None},
    {'quote_source': 'Unavailable'}, {'quote_time': None},
    {'quote_time': '2026-09-09T15:29:59Z'}, {'quote_time': '2026-09-09T16:01:00Z'},
    {'as_of': '2026-09-09T15:29:59Z'}, {'start': NOW.isoformat()}, {'start': None},
    {'start': '2026-09-10T20:00:00Z'}, {'pick': 'A1 line unresolved'},
    {'market': 'unknown'}, {'sport': 'MLB', 'quote_source': 'DraftKings'},
])
def test_research_rejects_ineligible_leg(changes):
    assert build_research_parlays([leg(0), leg(1, **changes)], NOW) == []


def test_college_sources_and_observation_limits_are_preserved():
    rows = [leg(i, sport='NCAAF', quote_source='DraftKings', quote_time_basis='espn_observed',
                quote_time='2026-09-09T15:30:00Z', as_of=NOW.isoformat()) for i in range(2)]
    tickets = build_research_parlays(rows, NOW)
    assert len(tickets) == 1
    assert all(r['quote_time_basis'] == 'espn_observed' and r['status'] == 'PASS' for r in tickets[0]['legs'])
    assert build_research_parlays(rows, datetime.fromisoformat('2026-09-09T16:00:01Z')) == []
    rows[1].update(quote_source='FanDuel')
    rows[1].pop('quote_time_basis')
    assert build_research_parlays(rows, NOW) == []  # Never multiply prices from different books.
    rows[0].pop('quote_time_basis')
    rows[0]['quote_source'] = 'FanDuel'
    assert len(build_research_parlays(rows, NOW)) == 1


def test_canonical_alias_duplicates_and_qualified_teams_are_excluded():
    rows = [leg(0), leg(1)]
    for row in rows:
        row.update(sport='NCAAF', game='Grambling State at Tcu', pick='Tcu +1.5', market='spread_home')
    rows[1]['game'] = 'Grambling at Tcu'
    rows += [leg(2), leg(3)]
    tickets = build_research_parlays(rows, NOW)
    assert len(tickets) == 1 and all('Grambling' not in r['game'] for r in tickets[0]['legs'])
    qualified_rows = [leg(i, status='APPROVED') for i in range(6)]
    qualified = build_parlays(qualified_rows, NOW, qualified_only=True)
    research = build_research_parlays(qualified_rows + [leg(6), leg(7)], NOW, qualified_parlays=qualified)
    assert len(research) == 1
    assert {r['game'] for ticket in research for r in ticket['legs']} == {'A6 at B6','A7 at B7'}


@pytest.mark.parametrize('change', ['promote','payout','leg','policy','missing_policy','move_to_qualified'])
def test_research_package_rejects_tampering(change):
    result = package([leg(0), leg(1)])
    if change == 'promote': result['research_parlays'][0]['status'] = 'APPROVED'
    if change == 'payout': result['research_parlays'][0]['decimal_odds_estimate'] = 99
    if change == 'leg': result['research_parlays'][0]['legs'][0]['pick'] = 'Changed +1.5'
    if change == 'policy': result['research_parlay_policy'] = 'anything'
    if change == 'missing_policy': result.pop('research_parlay_policy')
    if change == 'move_to_qualified': result['parlays'] = result['research_parlays']
    with pytest.raises(ValueError): validate_package(result)


def test_saved_research_tickets_grade_original_legs_once():
    from app_core.public_history import digest, report
    original = package([leg(0), leg(1)])
    revised = deepcopy(original)
    for r in revised['games']['overall']: r['pick'] = r['pick'].replace('+1.5','-1.5')
    revised['research_parlays'] = build_research_parlays(revised['games']['overall'], NOW)
    publications = [{'package': p, 'package_hash': digest(p), 'confirmed_at': at} for p, at in
                    [(original,'2026-09-09T16:01:00Z'),(revised,'2026-09-09T16:02:00Z')]]
    scores = [{'sport':'MLB','away':f'A{i}','home':f'B{i}','event_id':str(i),'start':'2026-09-09T20:00:00Z',
               'away_score':5,'home_score':4} for i in range(2)]
    rows = report(publications,[{'recorded_at':'2026-09-10T00:00:00Z','scores':scores}])
    tickets = [r for r in rows if r['category']=='parlays']
    assert len(tickets)==1 and tickets[0]['group']=='Research' and tickets[0]['outcome']=='WIN'
    assert tickets[0]['picks'].count('+1.5') == 2


def test_browser_separates_research_and_reports_expiration(tmp_path):
    node = os.environ.get('NODE_BINARY') or shutil.which('node')
    if not node: pytest.skip('Node required for browser regression')
    template = Path('publishing/board.html').read_text(encoding='utf-8')
    functions = template[template.index('function renderParlays()'):template.index('function flatStakeMetrics(')]
    data = package([leg(0),leg(1)])
    script = r"""
const assert=require('node:assert/strict');
class Element {constructor(tag,text){this.tag=tag;this.textContent=text||'';this.children=[];}append(...nodes){this.children.push(...nodes);}replaceChildren(...nodes){this.children=nodes;}}
const nodes={parlayRows:new Element('div'),researchParlayRows:new Element('div')};
const document={getElementById:id=>nodes[id]};
const el=(tag,text)=>new Element(tag,text),fmt=x=>String(x),table=rows=>el('table',JSON.stringify(rows));
let expired=false;const state=r=>expired?'STALE':r.status;
function text(n){return n.textContent+' '+n.children.map(text).join(' ');}
""" + 'const data='+json.dumps(data)+';\n' + functions + r"""
renderParlays();
assert.match(text(nodes.parlayRows),/0 individually approved/);
assert.match(text(nodes.researchParlayRows),/Research parlay 1/);
assert.match(text(nodes.researchParlayRows),/RESEARCH ONLY/);
assert.match(text(nodes.researchParlayRows),/Sportsbook: Novig/);
expired=true;renderParlays();assert.match(text(nodes.researchParlayRows),/EXPIRED/);
data.research_parlays=[];renderParlays();assert.match(text(nodes.researchParlayRows),/same sportsbook/);
assert.doesNotMatch(text(nodes.researchParlayRows),/Build a new preview with at least two eligible games/);
delete data.research_parlay_policy;renderParlays();assert.match(text(nodes.researchParlayRows),/predates Research Parlays/);
"""
    path = tmp_path/'research.cjs'; path.write_text(script,encoding='utf-8')
    subprocess.run([node,str(path)],check=True,capture_output=True,text=True)
