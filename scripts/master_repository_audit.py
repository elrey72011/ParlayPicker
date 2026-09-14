"""Read-only master-audit inventory. Does not fit models or modify live decisions."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path

SPORTS=('NFL','NCAAF','NBA','NCAAB','MLB','NHL')

def number(value):
    import math
    try:
        value=float(value)
        return value if math.isfinite(value) else None
    except (ValueError,TypeError): return None

def audit_input(path):
    raw=Path(path).read_bytes()
    rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    groups=defaultdict(list)
    for row in rows:
        groups[row.get('league','Unknown')].append(row)
    result={'file':Path(path).name,'sha256':hashlib.sha256(raw).hexdigest(),'candidate_rows':len(rows),'sports':{}}
    for sport,pool in sorted(groups.items()):
        games=defaultdict(list)
        for row in pool: games[(row.get('export_run_id'),row.get('matchup_id'))].append(row)
        counts=Counter();books=Counter()
        for game, candidates in games.items():
            available=set(); provider=set()
            for row in candidates:
                market=row.get('market_type','').split('_')[0]
                odds=number(row.get('odds_american'))
                line=number(row.get('spread_line') if market=='spread' else row.get('total_line'))
                if odds is not None and abs(odds)>=100 and (line is not None or market=='moneyline'):
                    available.add(market)
                try: quotes=json.loads(row.get('provider_quotes') or '[]')
                except ValueError: quotes=[]
                for q in quotes:
                    price=number(q.get('price'))
                    if price is not None and abs(price)>=100:
                        provider.add((q.get('book'),str(q.get('market_type','')).split('_')[0]))
            for family in ('spread','total','moneyline'): counts['candidate_'+family+'_games']+=family in available
            counts['both_spread_total_games']+= {'spread','total'}<=available
            counts['spread_only_games']+= 'spread' in available and 'total' not in available
            counts['total_only_games']+= 'total' in available and 'spread' not in available
            counts['neither_spread_total_games']+=not {'spread','total'}&available
            for book,family in provider: books[str(book)+':'+family]+=1
        selected=[r for r in pool if str(r.get('best_available_selected')).lower() in {'true','1'}]
        result['sports'][sport]={'games':len(games),'candidate_rows':len(pool),'selected_rows':len(selected),
            **dict(counts),'selected_markets':dict(Counter(r.get('market_type') for r in selected)),
            'selection_sources':dict(Counter(r.get('selection_probability_source') for r in selected)),
            'recorded_qualification_reasons':dict(Counter(r.get('qualification_reason') for r in selected)),
            'provider_quoted_games_by_book_family':dict(sorted(books.items())),
            'approved_flag_rows':sum(str(r.get('wager_approved')).lower() in {'true','1'} for r in selected)}
    return result

def import_inventory(root):
    modules={}
    for path in root.rglob('*.py'):
        relative=path.relative_to(root)
        if any(p.startswith('.') or p in {'outputs','test-results','node_modules','__pycache__'} for p in relative.parts): continue
        module='.'.join(relative.with_suffix('').parts)
        if module.endswith('.__init__'): module=module[:-9]
        modules[module]=path
    graph={};errors=[]
    for module,path in modules.items():
        try: tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        except (ValueError,SyntaxError,UnicodeError) as exc:
            errors.append({'module':module,'error':type(exc).__name__});continue
        edges=set()
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): edges.update(a.name for a in node.names)
            if isinstance(node,ast.ImportFrom):
                package=module if path.name=='__init__.py' else module.rpartition('.')[0]
                parts=package.split('.') if package else []
                base='.'.join(parts[:len(parts)-node.level+1]) if node.level else ''
                base='.'.join(x for x in (base,node.module or '') if x)
                if base: edges.add(base)
                edges.update(base+'.'+a.name if base else a.name for a in node.names)
        graph[module]=sorted(edges & modules.keys())
    reachable=set();pending=['streamlit_app']
    while pending:
        module=pending.pop()
        if module not in reachable:
            reachable.add(module);pending+=graph.get(module,[])
    return {'module_count':len(modules),'parse_errors':errors,'graph':graph,'possibly_reachable_from_streamlit':sorted(reachable),
            'limitation':'Static imports include optional and function-local paths; not a runtime trace. Unreachable does not prove dead code.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',action='append',default=[])
    parser.add_argument('--accuracy-report',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report={'version':1,'baseline_commit':'d9d4b5353ebdc25de6845fe3be04cd09194f7b02',
            'inputs':[audit_input(p) for p in args.audit],
            'scope':'Archived input snapshots; quote presence is not verified availability at lock time or wager approval.'}
    if args.accuracy_report:
        raw=args.accuracy_report.read_bytes();accuracy=json.loads(raw)
        report['evaluation']={'source':args.accuracy_report.name,'sha256':hashlib.sha256(raw).hexdigest(),
            'status':accuracy['status'],'inventory':accuracy['validation']['inventory'],'exclusions':accuracy['validation']['exclusions']}
        if accuracy['validation']['inventory']['eligible_events']==0:
            report['threshold_study']=[{'sport':s,'threshold':t,'verified_games':0,'win_rate':None,'roi':None,'brier':None,'log_loss':None,'ece':None,
                'status':'not_estimable_without_verified_pregame_cohort'} for s in SPORTS for t in (.55,.60,.65,.70,.75)]
    root=Path(__file__).resolve().parents[1]
    imports=import_inventory(root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    args.output.with_name(args.output.stem+'-imports.json').write_text(json.dumps(imports,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report['inputs'],indent=2))
if __name__=='__main__': main()
