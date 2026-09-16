"""Allowlisted public data built from finalized exports; no provider requests."""
from app_core.public_quote_policy import supported_quote
from app_core.quote_freshness import QUOTE_MAX_AGE_MINUTES, package_age_minutes
import math
import re
from datetime import datetime, timezone
import pandas as pd


def text(row, *keys):
    for key in keys:
        value = row.get(key)
        if value is not None and pd.notna(value) and str(value).strip():
            return str(value).strip()
    return ''


def number(row, *keys):
    for key in keys:
        try:
            value = float(row.get(key))
            if math.isfinite(value):
                return value
        except (TypeError, ValueError):
            pass
    return None


def timestamp(value):
    if not value:
        return None
    if re.fullmatch(r'\d{8}T\d{6}(?:\.\d+)?Z', str(value)):
        value = datetime.strptime(value, '%Y%m%dT%H%M%S.%fZ' if '.' in value else '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
    elif str(value).endswith(' ET'):
        value = pd.Timestamp(str(value)[:-3]).tz_localize('America/New_York', ambiguous='raise', nonexistent='raise')
    parsed = pd.Timestamp(value)
    if pd.isna(parsed) or parsed.tzinfo is None:
        raise ValueError('Timestamps must include a timezone')
    return parsed.tz_convert('UTC').isoformat()


def pick_record(row, *, prop=False, as_of=None):
    at = timestamp(text(row, 'prediction_generated_at', 'export_run_id') or as_of)
    if at is None:
        raise ValueError('Every selection needs its original analysis timestamp')
    start = timestamp(text(row, 'game_start_utc', 'start', 'game_time_est'))
    probability = number(row, *(['ConservativeWinProbability', 'CalibratedProbability'] if prop else ['win_probability']))
    if probability is not None and not 0 <= probability <= 1:
        raise ValueError('Win estimate must be between zero and one')
    odds = number(row, *(['odds_american'] if prop else ['odds']))
    if odds is not None and abs(odds) < 100:
        raise ValueError('Invalid American odds')
    # Missing or conflicting authorization never grants approval.
    approved = text(row, 'Bettable').lower() in {'true','1','yes'} and (number(row, 'Play_Stake','Kelly_Bet_Size','Suggested_Stake') or 0) > 0
    if not prop:
        approved = approved and text(row, 'status') == 'APPROVED'
    for field in ('production_eligible', 'wager_approved'):
        if field in row and text(row, field).lower() not in {'true','1','yes'}:
            approved = False
    if odds is None or probability is None:
        approved = False
    from core.market_policy import production_market, moneyline_context_only
    if not prop and moneyline_context_only(text(row, 'market_type')):
        raise ValueError('Public game picks must be spread or total; moneyline is context only')
    if not prop and not production_market(text(row, 'market_type')):
        # Missing quotes remain visible as PASS rows, never approved wagers.
        approved = False
    pick = text(row, 'best_pick' if prop else 'pick')
    record = {'sport':text(row, 'league','League'), 'game':text(row, 'matchup'),
            'pick':pick, 'player':text(row, 'player') if prop else '',
            'market':text(row, 'market_type'), 'odds':odds, 'win_estimate':probability,
            'ev':number(row, 'expected_value' if prop else 'ev'),
            'status':'APPROVED' if approved else 'PASS', 'start':start, 'as_of':at}
    if not prop and 'qualification_reason' in row:
        record['qualification_reason'] = text(row, 'qualification_reason')
    if not prop and 'quote_source' in row:
        record['quote_source'] = text(row, 'quote_source')
        record['quote_time'] = timestamp(text(row, 'quote_time'))
        if text(row, 'quote_time_basis'):
            record['quote_time_basis'] = text(row, 'quote_time_basis')
            record['status'] = 'PASS'
        if 'quote_reason' in row:
            record['quote_reason'] = text(row, 'quote_reason')
    if not prop:
        for field in ('maturity', 'gemini_review_status'):
            if text(row,field): record[field] = text(row,field)
        for field in ('espn_event_id','mlb_game_pk','game_number'):
            if text(row,field): record[field] = text(row,field)
        if number(row,'conservative_ev') is not None: record['conservative_ev'] = number(row,'conservative_ev')
    if not prop and isinstance(row.get('wager_contract'),dict):
        record['wager_contract']=dict(row['wager_contract'])
        if record['wager_contract'].get('selection')==record['pick']:
            record['win_estimate']=record['wager_contract'].get('conservative_probability')
            record['ev']=record['wager_contract'].get('conservative_ev')
        record['status']='APPROVED' if record['wager_contract'].get('production_eligible') is True and (record['wager_contract'].get('production_bet_amount') or 0)>0 else 'PASS'
    if prop:
        projection = number(row, 'expected_count')
        if record['sport'].upper() == 'NFL' and (number(row, 'FormSampleSize') or 0) <= 0:
            projection = None
        if projection is not None and projection >= 0:
            record['expected_stat'] = projection
    if not prop:
        from app_core.total_signal_quality import public_fields
        from app_core.price_value_display import display
        record.update(public_fields(row))
        record.update(display(record["win_estimate"], record["odds"], record["ev"]))
    return record


def build_package(overall, sides, totals, *, props=None, props_as_of=None, dfs=None, dfs_sport=None, dfs_slate=None, dfs_start=None):
    frames = {'overall':overall, 'sides':sides, 'totals':totals}
    identities = None
    run = None
    games = {}
    for family, frame in frames.items():
        required = {'matchup_id','export_run_id','pick','status','Bettable','Play_Stake'}
        if frame.empty or not required.issubset(frame.columns):
            raise ValueError('Supply all three nonempty per-game exports from the same run')
        ids = frame.matchup_id.fillna('').astype(str)
        runs = set(frame.export_run_id.dropna().astype(str))
        if ids.eq('').any() or ids.duplicated().any() or len(runs) != 1 or frame.export_run_id.isna().any():
            raise ValueError('Missing or duplicate game identity / run timestamp')
        if identities is not None and (set(ids) != identities or runs != run):
            raise ValueError('The three boards must contain the same games and run')
        identities, run = set(ids), runs
        games[family] = [pick_record(row) for _, row in frame.iterrows()]
    lineups = []
    if dfs is not None and not dfs.empty:
        from app_core.draftkings_classic import DK_MLB_CLASSIC_ROSTER_SLOTS, DK_NFL_CLASSIC_ROSTER_SLOTS
        if dfs_sport not in {'MLB','NFL'} or not dfs_slate or not dfs_start:
            raise ValueError('DFS needs sport MLB/NFL, a slate label, and a timezone-aware lock time')
        slots = DK_MLB_CLASSIC_ROSTER_SLOTS if dfs_sport == 'MLB' else DK_NFL_CLASSIC_ROSTER_SLOTS
        for _, row in dfs.iterrows():
            players = [text(row, slot) for slot in slots]
            ids = [re.search(r'\((\d+)\)$', p).group(1) if re.search(r'\((\d+)\)$', p) else p.casefold() for p in players]
            salary = number(row, 'Salary')
            if not all(players) or len(set(ids)) != len(slots) or salary is None or not 0 < salary <= 50000:
                raise ValueError('DFS lineup has missing/duplicate players or invalid salary')
            lineups.append({'sport':dfs_sport, 'slate':dfs_slate, 'format':'Classic', 'start':timestamp(dfs_start),
                            'players':[{'slot':slot, 'name':name} for slot,name in zip(slots,players)],
                            'salary':salary, 'salary_remaining':50000-salary,
                            'projected_points':number(row, 'Projected Points'),
                            'projection_basis':text(row, 'Projection Sources') or 'Unavailable'})
    from app_core.public_parlays import build_parlays, build_research_parlays
    built_at = datetime.now(timezone.utc)
    from app_core.public_prop_timing import with_game_starts
    public_props=[] if props is None else [pick_record(row, prop=True, as_of=props_as_of) for _,row in props.iterrows()]
    public_props=with_game_starts(public_props,games['overall'])
    from app_core.production_parlays import build_production_parlays, canonical_funnel
    summary=canonical_funnel(games['overall'],built_at)
    summary.pop('combinations')
    qualified_parlays = build_production_parlays(games['overall'], built_at)
    research_parlays = build_research_parlays(games['overall'], built_at, qualified_parlays=qualified_parlays, diversified=True)
    return {'schema_version':2, 'parlay_funnel':summary, 'parlay_policy':'canonical-v3', 'research_parlay_policy':'diversified-v2', 'research_parlays':research_parlays, 'selection_policy':'qualified-v1', 'top_ten_policy':'first-publication-v1', 'parlays':qualified_parlays, 'built_at':built_at.isoformat(), 'stale_after_minutes':QUOTE_MAX_AGE_MINUTES,
            'games':games, 'props':public_props,
            'dfs':lineups}


def validate_package(package):
    """Reject injected private fields before rendering or publishing saved drafts."""
    def exact(obj, keys):
        if not isinstance(obj, dict) or set(obj) != set(keys.split()):
            raise ValueError('Unexpected or missing public fields')
    from app_core.total_signal_quality import FIELDS as TQ_FIELDS, validate as validate_quality
    from app_core.price_value_display import FIELDS as VALUE_FIELDS, display as display_value
    def projection_metric(row):
        value = row.get('expected_stat')
        if 'expected_stat' in row and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise ValueError('Invalid expected statistic')
    exact(package, 'schema_version built_at stale_after_minutes games props dfs' + (' research_parlay_policy research_parlays' if 'research_parlay_policy' in package else '') + (' selection_policy' if 'selection_policy' in package else '') + (' top_ten_policy' if 'top_ten_policy' in package else '') + (' parlays' if package.get('schema_version') in {2,3,4,5} else '') + (' results' if package.get('schema_version') in {3,4,5} else '') + (' parlay_policy' if 'parlay_policy' in package else '') + (' parlay_funnel' if 'parlay_funnel' in package else ''))
    if package.get("parlay_policy", "supported-v2") not in {"supported-v2", "canonical-v3"}:
        raise ValueError("Unsupported parlay policy")
    package_age_minutes(package)
    if package['schema_version'] not in {1,2,3,4,5}:
        raise ValueError('Unsupported public package version/policy')
    if 'selection_policy' in package and package['selection_policy'] != 'qualified-v1':
        raise ValueError('Unsupported selection policy')
    if 'research_parlay_policy' in package and (package['research_parlay_policy'] not in {'positive-edge-v1', 'diversified-v2'} or package.get('selection_policy') != 'qualified-v1' or package['schema_version'] not in {2,3,4,5}):
        raise ValueError('Unsupported research parlay policy')
    if 'top_ten_policy' in package and (package['top_ten_policy'] != 'first-publication-v1' or package.get('selection_policy') != 'qualified-v1'):
        raise ValueError('Unsupported Top 10 policy')
    timestamp(package['built_at'])
    exact(package['games'], 'overall sides totals')
    for rows in [*package['games'].values(), package['props']]:
        if not isinstance(rows, list):
            raise ValueError('Selections must be lists')
        for row in rows:
            exact(row, 'sport game pick player market odds win_estimate ev status start as_of' + (' qualification_reason' if 'qualification_reason' in row else '') + ((' quote_source quote_time' + (' quote_reason' if 'quote_reason' in row else '') + (' quote_time_basis' if 'quote_time_basis' in row else '')) if rows is not package['props'] and 'quote_source' in row else '') + (' expected_stat' if rows is package['props'] and 'expected_stat' in row else '') + (' wager_contract' if 'wager_contract' in row else '') + ''.join(' '+k for k in ('maturity','gemini_review_status','conservative_ev','espn_event_id','mlb_game_pk','game_number', *TQ_FIELDS, *VALUE_FIELDS) if k in row))
            if any(k in row for k in TQ_FIELDS):
                validate_quality({k:row[k] for k in TQ_FIELDS if k in row})
                if row['sport'] != 'MLB' or row['market'] not in {'total_over','total_under'}:
                    raise ValueError('Totals diagnostics require MLB totals')
            if any(k in row for k in VALUE_FIELDS):
                if {k:row[k] for k in VALUE_FIELDS if k in row} != display_value(row['win_estimate'],row['odds'],row['ev']):
                    raise ValueError('Price display must match saved estimates')
            if 'wager_contract' in row:
                from core.live_wager_contract import PUBLIC_FIELDS, validate_snapshot
                exact(row['wager_contract'],' '.join(PUBLIC_FIELDS))
                validate_snapshot(row['wager_contract'])
            projection_metric(row)
            if any(k in row and not isinstance(row[k],str) for k in ('maturity','gemini_review_status','espn_event_id','mlb_game_pk','game_number')):
                raise ValueError('Invalid public review labels')
            if 'conservative_ev' in row and (isinstance(row['conservative_ev'],bool) or not isinstance(row['conservative_ev'],(int,float)) or not math.isfinite(row['conservative_ev'])):
                raise ValueError('Invalid conservative EV')
            if 'qualification_reason' in row and not isinstance(row['qualification_reason'],str):
                raise ValueError('Invalid qualification reason')
            if 'quote_reason' in row and not isinstance(row['quote_reason'],str):
                raise ValueError('Invalid quote reason')
            if 'quote_source' in row:
                if row['quote_source'] != 'Unavailable' and not supported_quote(row):
                    raise ValueError('Invalid quote source')
                if supported_quote(row) and not timestamp(row['quote_time']):
                    raise ValueError('Missing sportsbook quote time')
                if 'quote_time_basis' in row and (not supported_quote(row) or row['status'] != 'PASS' or not timestamp(row['as_of']) or timestamp(row['quote_time']) > timestamp(row['as_of'])):
                    raise ValueError('Invalid research observation provenance')
            for key in ('sport','game','pick','player','market','status'):
                if not isinstance(row[key], str):
                    raise ValueError('Public labels must be text')
            if row['status'] not in {'APPROVED','PASS'}:
                raise ValueError('Invalid status')
            if not timestamp(row['as_of']):
                raise ValueError('Missing analysis time')
            timestamp(row['start'])
            for key in ('odds','win_estimate','ev'):
                if row[key] is not None and (not isinstance(row[key], (int,float)) or not math.isfinite(row[key])):
                    raise ValueError('Invalid metric')
    if not isinstance(package['dfs'], list):
        raise ValueError('DFS must be a list')
    for row in package['dfs']:
        exact(row, 'sport slate format start players salary salary_remaining projected_points projection_basis')
        if not timestamp(row['start']) or row['sport'] not in {'MLB','NFL'} or row['format'] != 'Classic':
            raise ValueError('Invalid DFS slate metadata')
        for key in ('sport','slate','format','projection_basis'):
            if not isinstance(row[key], str):
                raise ValueError('DFS metadata must be text')
        if not isinstance(row['salary'], (int,float)) or not math.isfinite(row['salary']) or not 0 < row['salary'] <= 50000 or row['salary_remaining'] != 50000-row['salary']:
            raise ValueError('Invalid DFS salary')
        if row['projected_points'] is not None and (not isinstance(row['projected_points'], (int,float)) or not math.isfinite(row['projected_points'])):
            raise ValueError('Invalid projection')
        if not isinstance(row['players'], list) or len(row['players']) != (10 if row['sport']=='MLB' else 9):
            raise ValueError('Incomplete DFS roster')
        for player in row['players']:
            exact(player, 'slot name')
            if not all(isinstance(v, str) and v.strip() for v in player.values()):
                raise ValueError('Invalid player fields')
    if package['schema_version'] in {2,3,4,5}:
        from app_core.public_parlays import build_parlays
        expected = build_parlays(package['games']['overall'], datetime.fromisoformat(package['built_at']), qualified_only=package.get('selection_policy')=='qualified-v1', max_age_minutes=package_age_minutes(package), legacy='parlay_policy' not in package)
        if package.get('parlay_policy')=='canonical-v3':
            from app_core.production_parlays import build_production_parlays
            expected=build_production_parlays(package['games']['overall'],datetime.fromisoformat(package['built_at']))
        if 'parlay_funnel' in package:
            from app_core.production_parlays import canonical_funnel
            expected_funnel=canonical_funnel(package['games']['overall'],datetime.fromisoformat(package['built_at']))
            expected_funnel.pop('combinations')
            if package['parlay_funnel'] != expected_funnel: raise ValueError('Invalid parlay funnel')
        if package['parlays'] != expected:
            raise ValueError('Parlays must match the original disjoint selections and estimates')
        if 'research_parlay_policy' in package:
            from app_core.public_parlays import build_research_parlays
            research = build_research_parlays(package['games']['overall'], datetime.fromisoformat(package['built_at']), qualified_parlays=expected, max_age_minutes=package_age_minutes(package), diversified=package['research_parlay_policy']=='diversified-v2')
            if package['research_parlays'] != research:
                raise ValueError('Research parlays must match the original eligible selections and estimates')
    if package['schema_version'] in {3,4,5}:
        if not isinstance(package['results'], list):
            raise ValueError('Invalid public results')
        seen=set()
        for row in package['results']:
            exact(row, 'id category date group published_at outcome picks odds final_score' + (' quote_source' if 'quote_source' in row else '') + (' quote_time quote_time_basis' if 'quote_time_basis' in row else '') + (' sport market' if row.get('category')=='props' and package['schema_version'] in {4,5} else '') + (' sport' if row.get('category')!='props' and 'sport' in row else '') + (' market' if row.get('category')!='props' and 'market' in row else '') + (' grading_diagnostics' if 'grading_diagnostics' in row else '') + (' original_win_estimate' if 'original_win_estimate' in row else '') + (' expected_stat' if row.get('category')=='props' and 'expected_stat' in row else ''))
            projection_metric(row)
            if not all(isinstance(v,str) for k,v in row.items() if k not in {'expected_stat', 'original_win_estimate', 'grading_diagnostics'}) or row['id'] in seen:
                raise ValueError('Invalid or duplicate result fields')
            if 'grading_diagnostics' in row:
                diagnostics = row['grading_diagnostics']
                allowed = {'INVALID_SAVED_EVENT', 'INVALID_RESULT_EVENT', 'NO_MATCH', 'DATE_MISMATCH', 'PROVIDER_ID_NOT_FOUND', 'PROVIDER_ID_CONFLICT', 'DOUBLEHEADER_AMBIGUOUS', 'MULTIPLE_MATCHING_EVENTS', 'RESCHEDULED_NEEDS_REVIEW', 'NO_FINAL_PROVIDER_RESULT', 'FINAL_SCORE_INVALID', 'PROVIDER_SCORE_CONFLICT', 'RESCHEDULED_SETTLEMENT_REVIEW'}
                if not isinstance(diagnostics, list) or not diagnostics:
                    raise ValueError('Invalid grading diagnostics')
                for diagnostic in diagnostics:
                    exact(diagnostic, 'event_match_status grading_reason event_match_method settlement_review_required')
                    if diagnostic['event_match_status'] not in allowed or diagnostic['grading_reason'] not in allowed or diagnostic['event_match_method'] not in {None, 'provider_id', 'teams_scheduled_date', 'game_number'} or not isinstance(diagnostic['settlement_review_required'], bool):
                        raise ValueError('Invalid grading diagnostics')
            if 'original_win_estimate' in row:
                value = row['original_win_estimate']
                if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                        or not 0 <= value <= 1 or row['category'] == 'parlays' or row['group'] == 'Imported research'
                        or not row.get('sport') or not row.get('market')):
                    raise ValueError('Invalid original win estimate')
            seen.add(row['id'])
            if row['category'] not in (({'overall','sides','totals','parlays','props'} | ({'top10'} if package['schema_version']==5 else set())) if package['schema_version'] in {4,5} else {'overall','sides','totals','parlays'}) or row['group'] not in ({'Approved','Research','Imported research','Locked'} if package['schema_version']==5 else {'Approved','Research','Imported research'}) or row['outcome'] not in ({'WIN','LOSS','PUSH','PENDING','NEEDS_REVIEW'} if package['schema_version']==5 and row['category']=='props' else {'WIN','LOSS','PUSH','PENDING'}):
                raise ValueError('Invalid result category or outcome')
            if row['category']=='top10' and row['group']!='Approved':
                raise ValueError('Top 10 results must be approved published selections')
            if row['group']=='Locked' and row['category']!='overall':
                raise ValueError('Only overall picks can be locked')
            if row['category']=='props':
                from app_core.public_prop_history import MARKETS
                if row['sport']!='MLB' or row['market'] not in MARKETS:raise ValueError('Unsupported prop results')
            datetime.strptime(row['date'], '%Y-%m-%d')
            if 'quote_time_basis' in row and (row['group'] != 'Locked' or not supported_quote(row) or not timestamp(row['quote_time'])):
                raise ValueError('Invalid locked observation provenance')
            if 'quote_source' in row and (row['group'] != 'Locked' or row['quote_source'] not in {'Novig', 'DraftKings', 'FanDuel', 'BetMGM'}):
                raise ValueError('Invalid locked sportsbook label')
            if row['group'] != 'Imported research' and not timestamp(row['published_at']):
                raise ValueError('Missing publication time')
    return package
