"""Shared public navigation for the saved board and its methodology page."""
from html import escape

STYLES = '''
.site-brandbar{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:6px 0 22px;border-bottom:1px solid #263347}
.site-brand{color:#e7edf5;text-decoration:none;font-size:24px;font-weight:750;letter-spacing:-.7px}.site-brand span{color:#6fe1ba}.site-label{color:#9baac0;font-size:12px;letter-spacing:1px;text-transform:uppercase}
.site-nav{display:flex;flex-wrap:wrap;gap:6px;margin:18px 0 28px}.site-nav a{display:inline-flex;align-items:center;padding:11px 16px;border-radius:8px;color:#b9c6d9;text-decoration:none;font-size:14px;font-weight:600;border:1px solid transparent}
.site-nav a:hover{background:#172334;color:#fff}.site-nav a[aria-current=page]{background:#6fe1ba;color:#081511}.site-nav a:focus-visible{outline:2px solid #6fe1ba;outline-offset:3px}
.page-heading{margin:28px 0 20px}.page-heading h1{font-size:34px;letter-spacing:-1px}.page-heading p{max-width:760px}.board-meta{font-size:13px;color:#9baac0;line-height:1.6}
@media(max-width:600px){.site-brandbar{align-items:flex-start}.site-label{max-width:110px;text-align:right}.site-nav{gap:4px}.site-nav a{padding:10px 12px}.page-heading h1{font-size:28px}}
'''
ITEMS=(('games','Picks'),('props','Player Props'),('parlays','Parlays'),('results','Results'),('dfs','DraftKings DFS'),('method','How Picks Work'))

def header(active='games', *, board=False):
    links=[]
    for key,label in ITEMS:
        url='/how-picks-work/' if key=='method' else ('#' if board else '/#')+key
        current=' aria-current="page"' if key==active else ''
        marker=f' data-site-tab="{key}"' if board and key!='method' else ''
        links.append(f'<a href="{escape(url)}"{marker}{current}>{label}</a>')
    return '<div class="site-brandbar"><a class="site-brand" href="/">Parlay<span>Picker</span></a><span class="site-label">Sports research &amp; results</span></div><nav class="site-nav" aria-label="Main navigation">'+''.join(links)+'</nav>'
