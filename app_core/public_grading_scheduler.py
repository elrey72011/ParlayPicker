"""Bounded public-history grading, invoked by the existing research scheduler."""
from datetime import datetime, timedelta, timezone
from app_core.public_history import History, report, selections, fetch_scores, digest
from app_core.imported_recaps import imported_selections
from app_core.research_schedule import is_open


def run(site, folder, client, sports, *, clock=None, fetch=None):
    clock=clock or (lambda: datetime.now(timezone.utc))
    fetch=fetch or fetch_scores
    at=clock()
    result={'started_at':at.isoformat(),'status':'ok','checked_batches':0,'newly_settled':0,'pending':0,'errors':[]}
    if not is_open():
        return {**result,'status':'outside_operating_window'}
    store=History(site,folder,client)
    # Restore must succeed before any scores are fetched or records changed.
    pubs=store.publications();imports=store.all('imports');revisions=store.all('scores')
    locks=store.all('locks')
    statuses=store.all('grading_runs')
    checked={}
    for status in sorted(statuses,key=lambda r:r['started_at']):
        checked.update(status.get('checked',{}))
    before=report(pubs,revisions,imports,locks)
    pending_ids={r['id'] for r in before if r['outcome']=='PENDING'}
    from app_core.locked_picks import locked_selections
    entries=selections(pubs)+imported_selections(imports)+locked_selections(locks)
    eligible=set()
    for row in entries:
        if row['id'] not in pending_ids:continue
        for leg in row['legs']:
            start=datetime.fromisoformat(leg['start'])
            # Do not poll new games immediately or old unresolved records forever.
            if leg['sport'] in sports and timedelta(hours=3)<=at-start<=timedelta(days=30):
                eligible.add((row['date'],leg['sport']))
    def last_checked(batch):
        return checked.get('|'.join(batch),'1970-01-01T00:00:00+00:00')
    due=[b for b in eligible if at-datetime.fromisoformat(last_checked(b))>=timedelta(hours=1)]
    result['checked']={}
    for day,sport in sorted(due,key=lambda b:(last_checked(b),b))[:2]:
        if not is_open():
            result['status']='operating_window_closed'
            break
        batch=day+'|'+sport
        result['checked'][batch]=clock().isoformat()
        try:
            revision=fetch(datetime.fromisoformat(day).date(),{sport})
            # No new empty/repeated score blobs for unfinished games.
            known={digest(r['scores']) for r in revisions}
            if revision['scores'] and digest(revision['scores']) not in known:
                store.put('scores/'+digest(revision)+'.json',revision)
                revisions.append(revision)
            result['checked_batches']+=1
        except Exception as exc:
            result['errors'].append(sport+':'+type(exc).__name__)
    after=report(pubs,revisions,imports,locks)
    result['pending']=sum(r['outcome']=='PENDING' for r in after)
    result['newly_settled']=sum(r['id'] in pending_ids and r['outcome']!='PENDING' for r in after)
    result['finished_at']=clock().isoformat()
    if result['errors']:result['status']='error'
    successes=[r['finished_at'] for r in statuses if r.get('status')=='ok']
    result['last_success_at']=result['finished_at'] if result['status']=='ok' else max(successes,default=None)
    store.put('grading_runs/'+digest(result)+'.json',result)
    return result
