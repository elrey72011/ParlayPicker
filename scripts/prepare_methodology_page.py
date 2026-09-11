"""Wrap an exported sandboxed visualization with public site navigation."""
import argparse
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app_core.public_site_shell import STYLES, header

def prepare(source, destination):
    original=Path(source).read_text(encoding='utf-8-sig')
    if '<body>' not in original or 'sandbox="allow-scripts"' not in original:
        raise ValueError('Expected a standalone sandboxed visualization export')
    css='''\n:root{color-scheme:dark;background:#0b111b;color:#e7edf5;font-family:system-ui,sans-serif}body{max-width:1250px;margin:auto;padding:32px 20px;box-sizing:border-box}iframe{max-width:100%;min-height:600px}h1{margin:4px 0}p{color:#9baac0;line-height:1.6}@media(max-width:600px){body{padding:20px 12px}}\n'''+STYLES
    result=original.replace('</head>','<style>'+css+'</style></head>',1)
    result=result.replace('<title>pick-inputs-and-weights.html</title>','<title>How Picks Work · ParlayPicker</title>',1)
    result=result.replace('<body>','<body>'+header('method')+'<header class="page-heading"><div class="site-label">OUR PROCESS</div><h1>How picks work</h1><p>Follow the inputs, explore the weighting rules, and see where review and locking fit.</p></header>',1)
    destination=Path(destination);folder=destination/'how-picks-work';folder.mkdir(parents=True,exist_ok=True)
    (folder/'index.html').write_text(result,encoding='utf-8')
    archive=destination/'how-picks-work-navigation.zip'
    info=zipfile.ZipInfo('how-picks-work/index.html');info.create_system=3;info.external_attr=0o100644<<16
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:z.writestr(info,result.encode('utf-8'))
    return archive

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');args=p.parse_args()
    print(prepare(args.source,args.destination))
