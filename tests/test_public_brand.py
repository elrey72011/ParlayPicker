import re
import shutil
import subprocess
from pathlib import Path

import pytest

from app_core.public_site_shell import STYLES, header
from scripts.publish_board import render


def empty_package():
    return {'schema_version': 1, 'built_at': '2026-09-17T16:00:00Z', 'stale_after_minutes': 30,
            'games': {'overall': [], 'sides': [], 'totals': []}, 'props': [], 'dfs': []}


def test_public_brand_does_not_use_teal_or_mint_primary_tokens():
    html = render(empty_package())
    css = '\n'.join(re.findall(r'<style>(.*?)</style>', html, re.S)).lower()
    for old in ('#5eead4', '#6fe1ba', '#07111f', '#38bdf8', '#0b111b', '#b93c16'):
        assert old not in css
    assert '--pp-bg: #f4efe7' in css
    assert '--pp-brand: #a84a16' in css
    assert 'color-scheme: light' in css
    assert 'border-bottom-color: var(--pp-brand)' in css
    assert STYLES in html and header(board=True) in html


def test_public_heading_and_navigation_semantics():
    html = render(empty_package())
    assert len(re.findall(r'<header\b', html)) == 1
    assert '<section class="page-heading" aria-labelledby="page-title">' in html
    assert '<h1 id="page-title">' in html
    assert 'role="tabpanel"' not in html and 'role="tablist"' not in html
    assert ':focus-visible { outline: 3px solid var(--pp-brand)' in html
    assert 'Research selections' not in html
    assert 'Not approved' in html and 'No approved plays right now' in html
    assert 'Analysis time unavailable' in html
    assert "badge.className='badge approved'" not in html  # A lock is not wager approval.


def test_methodology_uses_shared_light_shell(tmp_path):
    from scripts.prepare_methodology_page import prepare
    source = tmp_path/'method.html'
    source.write_text('<html><head><title>pick-inputs-and-weights.html</title></head><body><iframe sandbox="allow-scripts"></iframe></body></html>')
    prepare(source, tmp_path/'site')
    html = (tmp_path/'site/how-picks-work/index.html').read_text(encoding='utf-8')
    assert STYLES in html and header('method') in html
    assert len(re.findall(r'<header\b', html)) == 1
    assert 'color-scheme:dark' not in html
    assert 'aria-labelledby="page-title"' in html


def test_editorial_text_and_status_contrast():
    # WCAG relative luminance: normal-sized text needs at least 4.5:1.
    def luminance(hex_value):
        rgb = [int(hex_value[i:i+2], 16)/255 for i in (1, 3, 5)]
        linear = [v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4 for v in rgb]
        return sum(v*w for v, w in zip(linear, (.2126, .7152, .0722)))
    for text, surface in [('#1C2430','#F4EFE7'), ('#5F6875','#F4EFE7'),
                          ('#A84A16','#FFFFFF'), ('#8F3D12','#F4EFE7'), ('#8F3D12','#FFFFFF'), ('#B42318','#FFFFFF'), ('#2F6B4F','#FFFFFF'), ('#8A5A00','#FFFFFF'), ('#5F6875','#FFFFFF'),
                          ('#2F6B4F','#E5F1EB'), ('#8A5A00','#F6EBCD'),
                          ('#B42318','#FBE9E7'), ('#7A5132','#EFE4D8'), ('#5F6875','#EEF0F2')]:
        assert (luminance(surface)+.05)/(luminance(text)+.05) >= 4.5


def test_analysis_clock_and_approved_hero(tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for public presentation regression')
    html = Path('publishing/board.html').read_text(encoding='utf-8')
    helpers = html[html.index('function analysisTimestamp('):html.index('function probabilityOrder(')]
    hero = next(line for line in html.splitlines() if line.startswith('function renderBoardSummary('))
    script = r"""
const assert=require('node:assert/strict');
Date.now=()=>Date.parse('2026-09-17T16:00:00Z');
""" + helpers + r"""
const board=(at,built='2026-09-17T16:00:00Z')=>({built_at:built,games:{overall:[{as_of:at,quote_time:'2026-09-17T16:00:00Z'}]},props:[]});
const old=board('2026-09-12T16:00:00Z');
assert.equal(analysisFreshness(old,{published_at:'2026-09-17T16:00:00Z'}).text,'Analysis 5d old · site published just now');
assert.equal(analysisFreshness(old).state,'old');
const fresh=board('2026-09-17T15:56:00Z');
assert.equal(analysisFreshness(fresh,{published_at:'2026-09-17T15:58:00Z'}).text,'Analysis updated 4 min ago');
assert.equal(analysisFreshness(fresh).state,'fresh');
assert.equal(analysisFreshness(board('2026-09-17T15:13:00Z')).primary,'Analysis 47 min old');
assert.equal(analysisFreshness(board('2026-09-17T15:13:00Z')).state,'aging');
old.props=[{as_of:'2026-09-17T15:59:00Z'}];assert.equal(analysisFreshness(old).primary,'Analysis 5d old');
for(const bad of [undefined,'invalid','2027-01-01T00:00:00Z'])assert.equal(analysisFreshness(board(bad)).primary,'Analysis time unavailable');
assert.equal(analysisTimestamp({games:{overall:[]},props:[],built_at:'2026-09-17T15:56:00Z'}),'2026-09-17T15:56:00.000Z');
class E{constructor(tag,text){this.tag=tag;this.text=text||'';this.children=[];this.dataset={};}append(...nodes){this.children.push(...nodes)}replaceChildren(...nodes){this.children=nodes}}
const root=new E('section'),el=(tag,text)=>new E(tag,text),document={getElementById:()=>root};
const qualifiedPick=r=>r.eligible;
const data={games:{overall:[{status:'APPROVED',eligible:true},{status:'APPROVED',eligible:false},{status:'PASS',eligible:false}]},props:[]};
""" + hero + r"""
const content=n=>n.text+' '+n.children.map(content).join(' ');
renderBoardSummary();assert.equal(root.dataset.approved,'1');assert.match(content(root),/1 approved play/);assert.match(content(root),/Not approved\s+2/);
data.games.overall[0].eligible=false;renderBoardSummary();assert.equal(root.dataset.approved,'0');assert.match(content(root),/No approved plays right now/);assert.doesNotMatch(content(root),/0 approved/);assert.match(content(root),/Not approved\s+3/);
"""
    target = tmp_path/'brand-freshness.cjs'
    target.write_text(script, encoding='utf-8')
    subprocess.run([node, str(target)], check=True, capture_output=True, text=True)


def test_methodology_sandbox_keeps_content_and_uses_public_tokens():
    from html import escape, unescape
    from scripts.prepare_methodology_page import theme_sandboxed_content
    inner = '<html><head></head><body><p>Original methodology</p><script>const saved=42;</script></body></html>'
    original = '<iframe sandbox="allow-scripts" data-srcdoc="'+escape(inner, quote=True)+'"></iframe>'
    result = theme_sandboxed_content(original)
    assert 'sandbox="allow-scripts"' in result
    decoded = unescape(re.search(r'data-srcdoc="([^"]*)"', result).group(1))
    assert '<p>Original methodology</p><script>const saved=42;</script>' in decoded
    assert 'color-scheme:light!important' in decoded
    assert '--viz-series-1:var(--pp-brand)' in decoded
    assert '--pp-bg: #F4EFE7' in decoded


def test_v21_display_typography_and_color_separation():
    css = Path('publishing/site.css').read_text(encoding='utf-8')
    colors = dict(re.findall(r'(--pp-[\w-]+):\s*(#[0-9A-Fa-f]{6})', css))
    assert colors['--pp-brand'] == '#A84A16'
    assert colors['--pp-danger'] == '#B42318'
    brand = [int(colors['--pp-brand'][i:i+2], 16) for i in (1, 3, 5)]
    danger = [int(colors['--pp-danger'][i:i+2], 16) for i in (1, 3, 5)]
    assert sum((a-b)**2 for a, b in zip(brand, danger))**.5 > 30
    assert '--pp-font-display: ui-serif,Georgia,Cambria,' in css
    display_rules = re.findall(r'([^{}]+)\{[^{}]*font-family:\s*var\(--pp-font-display\)[^{}]*\}', css)
    assert [selector.strip() for selector in display_rules] == ['.page-heading h1,.pp-board-hero h1']
    assert '.pp-matchup { font-family: var(--pp-font-sans)' in css
    assert '0 3px 10px rgba(28,36,48,.035)' in css
    assert '@import' not in css


def test_v21_research_switcher_markup_and_state_contract():
    html = render(empty_package())
    assert 'role="group" aria-label="Research view"' in html
    assert html.count('data-board-view="overall"') == 1
    assert html.count('data-board-view="sides"') == 1
    assert html.count('data-board-view="totals"') == 1
    assert "let activeBoardView='overall'" in html
    assert 'boardView:activeBoardView' in html
    assert 'role="tablist"' not in html
    assert '<summary>Top 10 across all leagues</summary>' in html


def test_v211_section_headings_and_visible_research_label():
    html = render(empty_package())
    for section, duplicate in [('props', 'Player Props'), ('results', 'Published Pick Results'), ('dfs', 'DraftKings DFS')]:
        content = re.search(r'<section id="'+section+r'"[^>]*>(.*?)</section>', html, re.S).group(1)
        assert '<h2>'+duplicate+'</h2>' not in content
    parlays = re.search(r'<section id="parlays"[^>]*>(.*?)</section>', html, re.S).group(1)
    assert '<h2>Parlays</h2>' not in parlays
    assert '<h2>Parlay products</h2>' in parlays
    assert '<h2>Legacy qualified combinations</h2>' in parlays
    assert '<h2>Research Parlays</h2>' in parlays
    assert '<div class="pp-research-view-block"><div class="pp-control-label">Research view</div><div class="pp-view-switcher" role="group" aria-label="Research view">' in html
    assert '.pp-research-view-block { margin-top: 20px; }' in html
