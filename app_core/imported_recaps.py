"""Historical per-game CSV imports are research evidence, never public receipts."""
from datetime import datetime
from zoneinfo import ZoneInfo
from app_core.public_board import build_package
from app_core.public_history import digest, event_key


def import_exports(overall, sides, totals):
    package=build_package(overall,sides,totals)
    # Do not synthesize historical parlay tickets from today's generator.
    value={'games':package['games']}
    return {**value,'id':digest(value)}


def imported_selections(imports):
    chosen={}
    candidates=[]
    for batch in imports:
        for category,rows in batch['games'].items():
            for leg in rows:
                try:
                    start=datetime.fromisoformat(leg['start']);exported=datetime.fromisoformat(leg['as_of'])
                    if exported>=start or event_key(leg) is None or leg['odds'] is None:
                        continue
                    if leg['market'] not in {'spread_home','spread_away','total_over','total_under','moneyline_home','moneyline_away','h2h_home','h2h_away'}:
                        continue
                    date=start.astimezone(ZoneInfo('America/New_York')).date().isoformat()
                    identity=('imported',category,*event_key(leg)[:3],date)
                    candidates.append((exported.isoformat(),batch['id'],identity,category,date,leg))
                except (ValueError,TypeError):
                    continue
    for _,_,identity,category,date,leg in sorted(candidates,key=lambda x:x[:2]):
        chosen.setdefault(identity,{'id':digest(identity),'category':category,'date':date,'group':'Imported research',
                                  'published_at':'','legs':[leg]})
    return list(chosen.values())
