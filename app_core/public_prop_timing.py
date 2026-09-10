"""Recover MLB prop start times only from the same archived analysis package."""
import re
from app_core.public_board import timestamp


def game_key(row):
    from app_core.theover_ingest import TEAM_ALIAS_MAP_BY_LEAGUE
    sport=row.get('sport','').upper()
    if sport!='MLB':return None
    parts=re.split(r'\s+(?:at|@|vs?\.?)\s+',row.get('game',''),flags=re.I)
    if len(parts)!=2:return None
    clean=lambda text:re.sub(r'[^a-z0-9]+',' ',text.casefold()).strip()
    aliases={clean(k):clean(v) for k,v in TEAM_ALIAS_MAP_BY_LEAGUE['MLB'].items()}
    def team(value):
        name=clean(value)
        for _ in range(3):name=aliases.get(name,name)
        return name
    return sport,*(team(p) for p in parts)


def with_game_starts(props,games):
    """Return copies. Ambiguous/missing matches and explicit prop starts are untouched."""
    index={}
    for game in games:
        key=game_key(game)
        if key is not None:index.setdefault(key,[]).append(game)
    resolved=[]
    for prop in props:
        leg=dict(prop)
        if not leg.get('start'):
            candidates=index.get(game_key(leg),[])
            # One overall row per event. Do not collapse doubleheaders or incomplete duplicates.
            if len(candidates)==1:
                game=candidates[0]
                try:
                    start=timestamp(game.get('start'))
                    # Both timestamps must identify the same original analysis run.
                    same_run=timestamp(game.get('as_of'))==timestamp(leg.get('as_of'))
                    if start and same_run:leg['start']=start
                except (TypeError,ValueError):pass
        resolved.append(leg)
    return resolved
