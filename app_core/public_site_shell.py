"""Shared public navigation for the saved board and its methodology page."""
from html import escape
from pathlib import Path

STYLES = (Path(__file__).resolve().parents[1] / 'publishing/site.css').read_text(encoding='utf-8')

ITEMS=(('games','Picks'),('props','Player Props'),('parlays','Parlays'),('results','Results'),('dfs','DraftKings DFS'),('method','How Picks Work'))

def header(active='games', *, board=False):
    links=[]
    for key,label in ITEMS:
        url='/how-picks-work/' if key=='method' else ('#' if board else '/#')+key
        current=' aria-current="page"' if key==active else ''
        marker=f' data-site-tab="{key}"' if board and key!='method' else ''
        extra=' data-extra' if key in ('dfs','method') else ''
        links.append(f'<a{extra} href="{escape(url)}"{marker}{current}>{label}</a>')
    freshness = '<div class="pp-freshness" id="siteFreshness"><span class="pp-live-dot" id="siteFreshnessIcon" aria-hidden="true">◷</span> <span id="siteFreshnessText">Analysis time unavailable</span></div>' if board else '<span class="site-label">Sports research &amp; results</span>'
    more = '<details class="pp-nav-more"><summary>More</summary><div>'+''.join(links[-2:])+'</div></details>'
    return '<header class="pp-header"><div class="site-brandbar"><a class="site-brand" href="/">Parlay<span>Picker</span></a>'+freshness+'</div><nav class="site-nav" aria-label="Main navigation">'+''.join(links)+more+'</nav></header>'
