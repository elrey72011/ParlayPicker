"""Published MLB prop evidence; explicit bounded boxscore grading only."""
import math
import re
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo
from app_core.public_history import digest, now

REVIEW_REASONS={
    'Ambiguous or missing game match',
    'Publication timing conflicts with provider game start',
    'Player missing or ambiguous in final box score',
    'No recorded appearance in final box score; settlement unverified',
    'Required statistic missing from final box score',
}

MARKETS={'batter_hits','batter_total_bases','pitcher_strikeouts','pitcher_walks','pitcher_outs'}

def norm(value):
    value=unicodedata.normalize('NFKD',str(value))
    return re.sub(r'[^a-z0-9]+',' ',''.join(c for c in value if not unicodedata.combining(c)).lower()).strip()

def terms(leg):
    market=re.sub(r'_(over|under)$','',leg['market'])
    match=re.fullmatch(r'(.+?)\s+(Over|Under)\s+(\d+(?:\.\d+)?)\s+.+',leg['pick'],re.I)
    if leg['sport'].upper()!='MLB' or market not in MARKETS or not match or norm(match[1])!=norm(leg['player']):return None
    side=match[2].lower()
    if leg['market'].endswith(('_over','_under')) and not leg['market'].endswith('_'+side):return None
    line=float(match[3])
    return (market,side,line) if math.isfinite(line) else None

def matchup(value):
    parts=re.split(r'\s+(?:at|@|vs?\.?)\s+',value,flags=re.I)
    return tuple(norm(p) for p in parts) if len(parts)==2 else None

def selections(publications,imports=()):
    from app_core.public_prop_timing import with_game_starts
    chosen={}
    sources=[(p['confirmed_at'],p['package_hash'],with_game_starts(p['package'].get('props',[]),p['package']['games']['overall']),False) for p in publications]
    sources += [(b['as_of'],b['id'],b['props'],True) for b in imports]
    for confirmed,key,legs,imported in sorted(sources):
        for leg in legs:
            parsed=terms(leg)
            if not parsed or not matchup(leg['game']):continue
            try:
                start=datetime.fromisoformat(leg['start']);at=datetime.fromisoformat(leg['as_of']);published=datetime.fromisoformat(confirmed)
                if not start.tzinfo or not at.tzinfo or not published.tzinfo:continue
                if at>=start or (not imported and not(at<=published<start and (published-at).total_seconds()<=900)):continue
                if leg['odds'] is None or not math.isfinite(leg['odds']) or abs(leg['odds'])<100:continue
            except (ValueError,TypeError):continue
            day=start.astimezone(ZoneInfo('America/New_York')).date().isoformat()
            # First player/stat per matchup/day; later line or direction changes cannot inflate the record.
            identity=('imported_props' if imported else 'props',day,matchup(leg['game']),norm(leg['player']),parsed[0])
            if identity in chosen:continue
            chosen[identity]={'id':digest(identity),'date':day,'category':'props','sport':'MLB','market':parsed[0],
                'group':'Imported research' if imported else 'Approved' if leg['status']=='APPROVED' else 'Research',
                'published_at':'' if imported else confirmed,'leg':leg}
    return list(chosen.values())

def import_export(frame):
    from app_core.public_board import pick_record
    required={'market_type','player','best_pick','matchup','odds_american','export_run_id'}
    if frame.empty or not required.issubset(frame.columns):raise ValueError('Use the original combined prop export with player, matchup and export_run_id.')
    runs=frame['export_run_id'].dropna().astype(str).unique()
    if len(runs)!=1 or frame['export_run_id'].isna().any():raise ValueError('Import one original analysis run at a time.')
    props=[pick_record(row,prop=True) for _,row in frame.iterrows() if str(row.get('league','')).upper()=='MLB']
    batch={'as_of':min((p['as_of'] for p in props),default=''),'props':props}
    batch['id']=digest(batch)
    if not selections([], [batch]):raise ValueError('No supported MLB props with original pregame timestamps and game start times. No timestamps were inferred.')
    return batch

def report(publications,revisions=(),imports=()):
    latest={};reasons={}
    for revision in sorted(revisions,key=lambda r:r['recorded_at']):
        for actual in revision['actuals']:latest[actual['id']]=actual
        for item in revision.get('unresolved',[]):reasons[item['id']]=item['reason']
    rows=[]
    from scripts.grade_props import grade_side
    for entry in selections(publications,imports):
        leg=entry['leg'];market,side,line=terms(leg);actual=latest.get(entry['id']);value=actual.get('value') if actual else None
        valid=isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0
        rows.append({**{k:v for k,v in entry.items() if k!='leg'},'outcome':grade_side(side,line,value) if valid else 'NEEDS_REVIEW' if reasons.get(entry['id']) in REVIEW_REASONS else 'PENDING',
            'picks':leg['game']+': '+leg['pick'],'odds':str(leg['odds']),
            **({'expected_stat':leg['expected_stat']} if 'expected_stat' in leg else {}),
            'final_score':str(value)+' '+market.removeprefix('batter_').removeprefix('pitcher_') if valid else reasons.get(entry['id'],'Pending player statistics')})
    return rows

def fetch_actuals(day,entries,*,http_get=None,max_games=10):
    """One schedule plus at most ten final boxscores. Missing/DNP/ambiguous stays pending."""
    import requests
    from scripts.grade_props import _stat_for_market, _ip_to_outs
    get=http_get or requests.get
    base='https://statsapi.mlb.com/api/v1'
    def request(path,**kwargs):
        response=get(base+path,timeout=10,**kwargs);response.raise_for_status();return response.json()
    schedule=request('/schedule',params={'sportId':1,'date':day.isoformat()})
    games=[g for d in schedule.get('dates',[]) for g in d.get('games',[])]
    actuals=[];boxes={};checked=[];unresolved=[]
    def pending(entry,reason):unresolved.append({'id':entry['id'],'reason':reason})
    for entry in entries:
        leg=entry['leg'];wanted=matchup(leg['game'])
        checked.append(entry['id'])
        same_day=[g for g in games if matchup(g['teams']['away']['team']['name']+' @ '+g['teams']['home']['team']['name'])==wanted
            and datetime.fromisoformat(g['gameDate'].replace('Z','+00:00')).astimezone(ZoneInfo('America/New_York')).date().isoformat()==entry['date']]
        matches=[g for g in same_day if abs((datetime.fromisoformat(g['gameDate'].replace('Z','+00:00'))-datetime.fromisoformat(leg['start'])).total_seconds())<=1800]
        fallback=False
        if not matches and len(same_day)==1:
            matches=same_day;fallback=True
        if len(matches)!=1:
            pending(entry,'Ambiguous or missing game match');continue
        game=matches[0];game_id=str(game['gamePk'])
        provider_start=datetime.fromisoformat(game['gameDate'].replace('Z','+00:00'))
        # A corrected start may establish an earlier kickoff. Never grade a post-start publication as pregame.
        if datetime.fromisoformat(leg['as_of'])>=provider_start or (entry['published_at'] and datetime.fromisoformat(entry['published_at'])>=provider_start):
            pending(entry,'Publication timing conflicts with provider game start');continue
        if game.get('status',{}).get('detailedState') not in {'Final','Game Over'}:
            pending(entry,'Game not final');continue
        if game_id not in boxes:
            if len(boxes)>=max_games:
                pending(entry,'Batch limit reached; run the next grading batch');continue
            boxes[game_id]=request('/game/'+game_id+'/boxscore')
        players=[p for team in boxes[game_id].get('teams',{}).values() for p in team.get('players',{}).values() if norm(p.get('person',{}).get('fullName',''))==norm(leg['player'])]
        if len(players)!=1:
            pending(entry,'Player missing or ambiguous in final box score');continue
        player=players[0];market,_,_=terms(leg);batter=market.startswith('batter_');stats=player.get('stats',{}).get('batting' if batter else 'pitching',{})
        appearance=stats.get('plateAppearances' if batter else 'battersFaced')
        if not isinstance(appearance,(int,float)) or appearance<=0:
            pending(entry,'No recorded appearance in final box score; settlement unverified');continue
        stat=_stat_for_market(market,leg['pick']);field={'hits':'hits','total_bases':'totalBases','ks':'strikeOuts','walks':'baseOnBalls','outs':'inningsPitched'}[stat]
        value=stats.get(field)
        if stat=='outs':value=_ip_to_outs(value) if value is not None and re.fullmatch(r'\d+(?:\.[012])?',str(value)) else None
        if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or value<0:
            pending(entry,'Required statistic missing from final box score');continue
        actuals.append({'id':entry['id'],'value':value,'stat':stat,'game_id':game_id,'player_id':str(player['person']['id']),'source':'mlb_final_boxscore','provider_start':provider_start.isoformat(),'match_method':'unique_matchup_date' if fallback else 'start_time'})
    return {'recorded_at':now(),'actuals':actuals,'checked':checked,'unresolved':unresolved}
