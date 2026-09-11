"""Bounded per-event recovery when the college bulk response omits Novig."""
from copy import deepcopy
from datetime import datetime, timezone
import logging
import re
import requests

log=logging.getLogger(__name__)


def recover_college_novig(games, api_key, *, get=None, max_requests=5):
    result=deepcopy(games)
    get=get or requests.get
    calls=0
    for game in result:
        books=game.get('bookmakers',[])
        present={m.get('key') for b in books if str(b.get('key','')).lower() in {'novig','novig_us'} for m in b.get('markets',[]) if m.get('outcomes')}
        missing={'spreads','totals'}-present
        if not missing:
            continue
        event=game.get('id','')
        if not api_key or not re.fullmatch(r'[a-zA-Z0-9_-]+',event):
            continue
        try:
            start=datetime.fromisoformat(game['commence_time'].replace('Z','+00:00'))
            if start <= datetime.now(timezone.utc):
                continue
        except (KeyError,TypeError,ValueError):
            continue
        if calls>=max_requests:
            log.warning('NCAAF Novig recovery request limit reached')
            break
        calls+=1
        try:
            response=get('https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/events/'+event+'/odds',
                         params={'apiKey':api_key,'bookmakers':'novig','markets':','.join(sorted(missing)),
                                 'oddsFormat':'american','dateFormat':'iso'},timeout=5)
            if response.status_code!=200:
                log.warning('NCAAF Novig event recovery HTTP %s for %s',response.status_code,event)
                continue
            recovered=response.json()
            if any(recovered.get(k)!=game.get(k) for k in ('id','home_team','away_team','commence_time')):
                log.warning('NCAAF Novig event recovery identity mismatch for %s',event)
                continue
            count=0
            for book in recovered.get('bookmakers',[]):
                if str(book.get('key','')).lower() not in {'novig','novig_us'}:
                    continue
                markets=[m for m in book.get('markets',[]) if m.get('key') in missing and m.get('outcomes')]
                if markets:
                    books.append({**book,'markets':markets})
                    count+=len(markets)
            game['bookmakers']=books
            log.info('NCAAF Novig event recovery %s: %s markets recovered',event,count)
        except (requests.RequestException,ValueError,TypeError,AttributeError):
            log.warning('NCAAF Novig event recovery failed for %s; original quotes retained',event)
    return result
