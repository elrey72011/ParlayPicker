"""Wrap an exported sandboxed visualization with public site navigation."""
import argparse
import re
from html import escape, unescape
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app_core.public_site_shell import STYLES, header

def theme_sandboxed_content(original):
    """Map the exported visualization's theme hooks to the same public palette."""
    tokens = STYLES[STYLES.index(':root'):STYLES.index('\n* {')]
    css = tokens + """
:root {color-scheme:light!important;--background:var(--pp-bg);--foreground:var(--pp-text);
--color-background-primary:var(--pp-bg);--card:var(--pp-surface);--card-foreground:var(--pp-text);
--popover:var(--pp-surface);--popover-foreground:var(--pp-text);--primary:var(--pp-brand);
--primary-foreground:var(--pp-surface);--secondary:var(--pp-surface-raised);--secondary-foreground:var(--pp-text);
--muted:var(--pp-surface-soft);--muted-foreground:var(--pp-text-muted);--accent:var(--pp-brand-soft);
--accent-foreground:var(--pp-brand-dark);--border:var(--pp-border);--input:var(--pp-border-strong);
--ring:var(--pp-brand);--destructive:var(--pp-danger);--viz-accent:var(--pp-brand);--viz-text:var(--pp-text);
--viz-series-1:var(--pp-brand);--viz-series-2:var(--pp-research);--viz-series-3:var(--pp-text-muted);
--viz-series-4:var(--pp-warning);--viz-series-5:var(--pp-text);--viz-series-6:var(--pp-border-strong)}
a{color:var(--pp-brand-dark)}:focus-visible{outline:3px solid var(--pp-brand);outline-offset:3px}
"""
    def themed(match):
        inner = unescape(match.group(2))
        if '</body>' not in inner:
            return match.group(0)
        inner = inner.replace('</body>', '<style data-public-theme="v2">'+css+'</style></body>')
        return match.group(1)+'="'+escape(inner, quote=True)+'"'
    return re.sub(r'(data-srcdoc|srcdoc)="([^"]*)"', themed, original)


def prepare(source, destination):
    original=Path(source).read_text(encoding='utf-8-sig')
    if '<body>' not in original or 'sandbox="allow-scripts"' not in original:
        raise ValueError('Expected a standalone sandboxed visualization export')
    original = theme_sandboxed_content(original)
    css = STYLES + '\niframe{max-width:100%;min-height:600px;border:1px solid var(--pp-border);background:var(--pp-surface)}\n'
    result=original.replace('</head>','<style>'+css+'</style></head>',1)
    result=result.replace('<title>pick-inputs-and-weights.html</title>','<title>How Picks Work · ParlayPicker</title>',1)
    result=result.replace('<body>','<body>'+header('method')+'<section class="page-heading" aria-labelledby="page-title"><div class="site-label">OUR PROCESS</div><h1 id="page-title">How picks work</h1><p>Follow the inputs, explore the weighting rules, and see where review and locking fit.</p></section>',1)
    destination=Path(destination);folder=destination/'how-picks-work';folder.mkdir(parents=True,exist_ok=True)
    (folder/'index.html').write_text(result,encoding='utf-8')
    archive=destination/'how-picks-work-navigation.zip'
    info=zipfile.ZipInfo('how-picks-work/index.html');info.create_system=3;info.external_attr=0o100644<<16
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:z.writestr(info,result.encode('utf-8'))
    return archive

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');args=p.parse_args()
    print(prepare(args.source,args.destination))
