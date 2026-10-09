"""Read-only independent-slate reconciliation. Never a wagering authority reader.

No schedules are fetched, historical numbers rebuilt, or stores opened here.
Coverage PASS/UNVERIFIED do not replace the existing zero-stake PASS action.
"""
from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timezone
from hashlib import sha256
import json
import math
import re
from zoneinfo import ZoneInfo

VERSION = 'slate-coverage-v1'
ET = ZoneInfo('America/New_York')
MARKETS = ('spread_home', 'spread_away', 'total_over', 'total_under')
STATES = {'APPROVED', 'PASS', 'UNVERIFIED'}
GATES = ('schedule_identity', 'pregame', 'provider_request', 'odds_match',
         'quote_clock', 'candidate_generation', 'model_evidence', 'finalization')
EXPLANATIONS = {
    'NO_MATCHING_ODDS_EVENT': 'No matching odds event',
    'PROVIDER_SUCCESS_EMPTY': 'Provider returned no events',
    'PROVIDER_FAILURE': 'Provider request failed',
    'PROVIDER_STATUS_UNKNOWN': 'Original provider outcome is unavailable',
    'ODDS_MARKET_MISSING': 'Required market has no retained offer',
    'IDENTITY_AMBIGUOUS': 'Schedule identity is ambiguous',
    'IDENTITY_CONFLICT': 'Schedule and provider identities conflict',
    'GAME_STARTED': 'Game has started; new wagering is unavailable',
    'START_UNAVAILABLE': 'Original game start is unavailable',
    'QUOTE_TIMESTAMP_MISSING': 'Quote timestamp is missing',
    'QUOTE_TIMESTAMP_INVALID': 'Quote timestamp is malformed or lacks timezone',
    'QUOTE_TIMESTAMP_FUTURE': 'Quote timestamp is after the coverage clock',
    'QUOTE_TIMESTAMP_STALE': 'Quote timestamp is stale',
    'CANDIDATE_GENERATION_LOSS': 'Matched offer has no candidate decision',
    'FINALIZATION_NOT_EVALUATED': 'Final wagering checks were not recorded',
    'MODEL_EVIDENCE_MISSING': 'Original compatible model evidence is missing',
    'MODEL_INCOMPATIBLE': 'Model incompatible',
    'MODEL_INFERENCE_UNAVAILABLE': 'Model inference is unavailable',
    'FINAL_APPROVAL_NOT_CURRENT': 'Saved approval is not current',
    'WAGER_REJECTION_RECORDED': 'Candidate rejected by the existing wagering gates',
    'POLICY_EXCLUSION': 'Excluded by the declared evaluation policy',
    'NCAAF_COMPAT_DEPENDENCY_BYTES_MISSING': 'Original feature dependency bytes are missing',
    'NCAAF_COMPAT_DEPENDENCY_BYTES_CORRUPT': 'Original feature dependency bytes fail integrity checks',
    'NCAAF_COMPAT_DEPENDENCY_IDENTITY_CONFLICT': 'Feature dependencies conflict with the selected event or teams',
    'NCAAF_COMPAT_DEPENDENCY_REFERENCE_CONFLICT': 'Feature dependency references do not match the consumed inputs',
    'NCAAF_COMPAT_FEATURE_DERIVATION_CONFLICT': 'Recorded features differ from the verified feature derivation',
    'NCAAF_COMPAT_MINIMUM_HISTORY_MISSING': 'Required prior scoring or yardage history is missing',
    'NCAAF_COMPAT_DEPENDENCY_SOURCE_REVIEW_MISSING_OR_CONFLICT': 'Exact feature-provider review is missing or conflicts with the dependencies',
    'NCAAF_COMPAT_DEPENDENCY_RIGHTS_UNAVAILABLE': 'Feature-provider use or public derived-output permission is unavailable',
    'NCAAF_COMPAT_EVENT_REVIEW_NOT_ACCEPTED': 'Exact event mapping has no accepted independent review',
    'NCAAF_SOURCE_REVIEW_NOT_ACCEPTED': 'Applicable source review has not been accepted',
    'NCAAF_PUBLIC_DERIVED_RIGHTS_UNAVAILABLE': 'Public derived-output permission is unavailable',
    'NCAAF_COMPAT_NEW_INFERENCE_CLOCK_CONFLICT': 'Quote or start clock does not support a new inference',
    'NCAAF_INTEGER_PUSH_MODEL_UNVALIDATED': 'Integer-line push model is unvalidated',
    'NCAAF_COMPAT_RUNTIME_COMPONENT_CHANGED': 'Model incompatible with an unreviewed runtime component',
}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def clock(value):
    try:
        if isinstance(value, str) and len(value) >= 16 and value[8] == 'T' and value.endswith('Z') and '-' not in value:
            value = datetime.strptime(value, '%Y%m%dT%H%M%S.%fZ' if '.' in value else '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except (ValueError, TypeError):
        return None


def text(row, *keys):
    for key in keys:
        val = row.get(key)
        if isinstance(val, str) and val.strip() and val.lower() not in {'nan', 'none'}:
            return val.strip()
    return ''


def rows(value):
    return value.to_dict('records') if hasattr(value, 'to_dict') else list(value or [])


def gate(name, status, code=None):
    return dict(gate=name, status=status, code=code)


def native_ncaaf(inventory, selected_date):
    """Project already-retained canonical inventory, preserving its source hash."""
    first, last = inventory['start_date'], inventory['end_date']
    usable = first <= selected_date <= last
    events = []
    for event in inventory.get('events', []) if usable else []:
        start = clock(event.get('kickoff'))
        if start and start.astimezone(ET).date().isoformat() != selected_date:
            continue
        events.append(dict(canonical_event_id=event['schedule_event_id'],
            home_team=event['home_team'], away_team=event['away_team'],
            home_team_id=event['home_team_id'], away_team_id=event['away_team_id'],
            original_start=event['kickoff'], schedule_status=event['schedule_status'],
            identity_conflict=event['identity_conflict'], divisions=event['divisions'],
            home_aliases=event['home_aliases'], away_aliases=event['away_aliases'],
            kickoff_revisions=event['kickoff_revisions'], provider_ids={'espn': event['schedule_provider_id']}))
    return dict(version='slate-inventory-v1', league='NCAAF', source='espn_schedule',
        selected_date=selected_date, timezone='America/New_York', observed_at=inventory['observed_at'],
        status=inventory['status'] if usable else 'UNAVAILABLE',
        completeness_basis='retained FBS/FCS scoreboard and event-index reconciliation',
        reasons=list(inventory['reasons']) if usable else ['DATE_OUTSIDE_RETAINED_WINDOW'],
        source_inventory_hash=digest(inventory), source_window=[first, last], events=events)


def native_football(observation, selected_date):
    """Already fetched identity reader observations are PARTIAL, never full slate."""
    sport = observation['sport']
    events = {}; issues = list(observation.get('reasons', []))
    for original in observation.get('events', []):
        eid = str(original.get('id') or '')
        if not eid:
            issues.append('SCHEDULE_CANONICAL_ID_UNAVAILABLE')
            continue
        competitions = original.get('competitions', [])
        if len(competitions) != 1:
            issues.append('SCHEDULE_FACTS_INCOMPLETE')
        competition = competitions[0] if len(competitions) == 1 else {}
        competitors = competition.get('competitors', [])
        home = [t.get('team', {}) for t in competitors if t.get('homeAway') == 'home']
        away = [t.get('team', {}) for t in competitors if t.get('homeAway') == 'away']
        start_value = competition.get('date') or original.get('date')
        start = clock(start_value)
        if start and start.astimezone(ET).date().isoformat() != selected_date:
            continue
        incomplete = len(home) != 1 or len(away) != 1
        if incomplete:
            issues.append('SCHEDULE_FACTS_INCOMPLETE')
        home = home if len(home) == 1 else [{}]
        away = away if len(away) == 1 else [{}]
        status = (competition.get('status') or original.get('status') or {}).get('type', {})
        event = dict(canonical_event_id=f'espn:{sport.lower()}:{eid}',
            home_team=home[0].get('displayName') or home[0].get('name') or '',
            away_team=away[0].get('displayName') or away[0].get('name') or '',
            home_team_id=f"espn:{sport.lower()}:{home[0].get('id')}",
            away_team_id=f"espn:{sport.lower()}:{away[0].get('id')}",
            original_start=start_value, provider_ids={'espn': eid}, identity_conflict=incomplete,
            schedule_status='STARTED' if status.get('state') in {'in', 'post'} else 'SCHEDULED' if status.get('state') == 'pre' else 'UNKNOWN',
            home_aliases=[home[0].get(k) for k in ('displayName', 'shortDisplayName', 'name') if home[0].get(k)],
            away_aliases=[away[0].get(k) for k in ('displayName', 'shortDisplayName', 'name') if away[0].get(k)])
        if eid in events and events[eid] != event:
            events[eid]['identity_conflict'] = True
            issues.append('SCHEDULE_REVISION_CONFLICT')
        else:
            events[eid] = event
    return dict(version='slate-inventory-v1', league=sport, source='retained_espn_identity_observation',
        selected_date=selected_date, timezone='America/New_York', observed_at=observation['observed_at'],
        status='UNAVAILABLE' if observation['status'] == 'UNAVAILABLE' else 'PARTIAL',
        completeness_basis='bounded quote-derived UTC date queries; no independent complete index',
        reasons=sorted(set(issues)), source_inventory_hash=digest(observation),
        source_window=observation.get('requested_utc_dates', []), events=list(events.values()))


def _normalize(inventory, selected_date, at):
    value = deepcopy(inventory)
    if value.get('version') != 'slate-inventory-v1' or value.get('timezone') != 'America/New_York':
        raise ValueError('COVERAGE_INVENTORY_SCHEMA')
    if value.get('selected_date') != selected_date or value.get('status') not in {'COMPLETE', 'PARTIAL', 'UNAVAILABLE'}:
        raise ValueError('COVERAGE_INVENTORY_DATE_OR_STATUS')
    if not all(text(value, k) for k in ('league', 'source')) or not isinstance(value.get('events'), list):
        raise ValueError('COVERAGE_INVENTORY_SCHEMA')
    observed = clock(value.get('observed_at'))
    reasons = list(value.get('reasons', []))
    if observed is None or observed > at:
        reasons.append('INVENTORY_CLOCK_UNVERIFIED')
    if value['status'] == 'COMPLETE' and (reasons or not text(value, 'completeness_basis')):
        value['status'] = 'PARTIAL'
        reasons.append('COMPLETENESS_REQUIREMENTS_UNVERIFIED')
    value['reasons'] = sorted(set(reasons))
    ids = set()
    for event in value['events']:
        eid = text(event, 'canonical_event_id')
        if not eid or eid in ids:
            raise ValueError('COVERAGE_DUPLICATE_OR_MISSING_CANONICAL_EVENT')
        ids.add(eid)
        if not all(text(event, k) for k in ('home_team', 'away_team')):
            value['status'] = 'PARTIAL'
            value['reasons'] = sorted(set(value['reasons'] + ['SCHEDULE_FACTS_INCOMPLETE']))
        start = clock(event.get('original_start'))
        if not start:
            value['status'] = 'PARTIAL'
            value['reasons'] = sorted(set(value['reasons'] + ['SCHEDULE_FACTS_INCOMPLETE']))
        if start and start.astimezone(ET).date().isoformat() != selected_date:
            raise ValueError('COVERAGE_EVENT_OUTSIDE_EASTERN_DATE')
    if value['status'] == 'UNAVAILABLE' and value['events']:
        raise ValueError('COVERAGE_UNAVAILABLE_WITH_EVENTS')
    value['inventory_id'] = digest(inventory)
    return value


def _match(row, events):
    """IDs do not waive named orientation/start checks. No fuzzy/date-only join."""
    league = text(row, 'league', 'sport', 'League').upper()
    eid = text(row, 'canonical_event_id', 'schedule_event_id')
    rid = text(row, 'matchup_id', 'game_id', 'id')
    home, away = text(row, 'home_team', 'Home'), text(row, 'away_team', 'Away')
    start = clock(text(row, 'game_start_utc', 'commence_time', 'start'))
    possible = []
    canonical_rid = rid if rid in {e['canonical_event_id'] for e in events} else None
    for event in events:
        if event['league'] != league:
            continue
        if eid and eid != event['canonical_event_id']:
            continue
        if canonical_rid and canonical_rid != event['canonical_event_id']:
            continue
        aliases = []
        for side, name in (('home', home), ('away', away)):
            source = event.get(side + '_aliases', []) + [event[side + '_team']]
            if league == 'NCAAF':
                from app_core.ncaaf_identity import normalize_ncaaf_team
                aliases.append(normalize_ncaaf_team(name) in {normalize_ncaaf_team(s) for s in source} if name else False)
            elif league == 'NFL':
                from app_core.nfl_identity import nfl_result_name
                aliases.append(nfl_result_name(name) in {nfl_result_name(s) for s in source} if name else False)
            else:
                aliases.append(name.casefold() in {s.casefold() for s in source} if name else False)
        same_id = eid == event['canonical_event_id'] or rid == event['canonical_event_id']
        ids = row.get('provider_ids')
        supplied = ids if isinstance(ids, dict) else {}
        known = event.get('provider_ids', {})
        if any(k in known and str(v) != str(known[k]) for k, v in supplied.items()):
            continue
        start_agrees = start is not None and start == clock(event.get('original_start'))
        if all(aliases) and start_agrees:
            possible.append(event)
        elif same_id:
            return None, 'IDENTITY_CONFLICT', [event['canonical_event_id']]
    if len(possible) != 1:
        return None, 'IDENTITY_AMBIGUOUS' if len(possible) > 1 else 'NO_SCHEDULE_MATCH', [e['canonical_event_id'] for e in possible]
    event = possible[0]
    if event.get('identity_conflict') or text(row, 'football_identity_status') == 'CONFLICT' or text(row, 'schedule_match_status') in {'AMBIGUOUS_PROVIDER_ID', 'AMBIGUOUS', 'KICKOFF_OR_IDENTITY_CONFLICT'}:
        return None, 'IDENTITY_CONFLICT', [event['canonical_event_id']]
    return event['canonical_event_id'], 'MATCHED', []


def _quotes(candidate, market):
    raw = candidate.get('provider_quotes')
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            raw = []
    return [q for q in raw or [] if isinstance(q, dict) and q.get('market_type') == market]


def _quote_gate(candidate, market, at):
    # Exact candidate binding stays with the evaluated offer, never another row.
    qt = text(candidate, 'odds_recorded_at', 'quote_time')
    if not qt:
        matches = _quotes(candidate, market)
        qt = matches[0].get('recorded_at') if len(matches) == 1 else None
    if not qt:
        return gate('quote_clock', 'FAIL', 'QUOTE_TIMESTAMP_MISSING')
    parsed = clock(qt)
    if parsed is None:
        return gate('quote_clock', 'FAIL', 'QUOTE_TIMESTAMP_INVALID')
    age = (at - parsed).total_seconds()
    if age < 0:
        return gate('quote_clock', 'FAIL', 'QUOTE_TIMESTAMP_FUTURE')
    from app_core.quote_freshness import QUOTE_MAX_AGE_SECONDS
    return gate('quote_clock', 'FAIL', 'QUOTE_TIMESTAMP_STALE') if age > QUOTE_MAX_AGE_SECONDS else gate('quote_clock', 'PASS')


def _current_approval(row, candidate, at, run):
    """Verify an existing finalized ticket only. Cannot create an approval."""
    contract = row.get('wager_contract')
    if not isinstance(contract, dict) or contract.get('production_eligible') is not True:
        return False
    try:
        from core.live_wager_contract import validate_snapshot, enforce_frame
        import pandas as pd
        validate_snapshot(contract)
        checked = enforce_frame(pd.DataFrame([row])).iloc[0]
        if not bool(checked.get('Bettable')) or not bool(row.get('Bettable')) or float(row.get('Play_Stake') or 0) <= 0:
            return False
        if text(row, 'export_run_id') != run or text(row, 'candidate_id') != text(candidate, 'candidate_id') or not text(candidate, 'candidate_id'):
            return False
        for field in ('market_type', 'odds_american', 'best_pick'):
            if row.get(field) != candidate.get(field):
                return False
        for field, key in (('selection', 'best_pick'), ('odds', 'odds_american'), ('market_type', 'market_type')):
            if contract.get(field) != candidate.get(key):
                return False
        if contract.get('sport') != text(candidate, 'league', 'sport').upper():
            return False
        if str(contract.get('game_id')) not in {text(candidate, 'game_id'), text(candidate, 'matchup_id')}:
            return False
        line = candidate.get('total_line' if str(candidate.get('market_type')).startswith('total') else 'spread_line')
        if contract.get('line') != line:
            return False
        from app_core.public_quote_policy import canonical_book_label
        if canonical_book_label(contract.get('sportsbook')) != canonical_book_label(text(candidate, 'quote_bookmaker', 'opposing_odds_source', 'book', 'odds_source')):
            return False
        original_quote = clock(text(candidate, 'odds_recorded_at', 'quote_time'))
        if original_quote is None or original_quote != clock(contract.get('quote_timestamp')):
            return False
        start, qt, generated = clock(contract.get('start')), clock(contract.get('quote_timestamp')), clock(text(row, 'prediction_generated_at') or run)
        from app_core.quote_freshness import QUOTE_MAX_AGE_SECONDS
        return bool(start and qt and generated and start > at and 0 <= (at-qt).total_seconds() <= QUOTE_MAX_AGE_SECONDS
                    and 0 <= (at-generated).total_seconds() <= QUOTE_MAX_AGE_SECONDS and contract.get('quote_fresh') is True)
    except (ValueError, TypeError, KeyError):
        return False


def _market(event, market, providers, candidates, finals, audit, at, run, health, conflicts):
    observed = [gate(name, 'NOT_EVALUATED') for name in GATES]
    def set_gate(name, status, code=None):
        observed[GATES.index(name)] = gate(name, status, code)
    set_gate('schedule_identity', 'FAIL' if conflicts or event.get('identity_conflict') else 'PASS',
             'IDENTITY_AMBIGUOUS' if conflicts else 'IDENTITY_CONFLICT' if event.get('identity_conflict') else None)
    start = clock(event.get('original_start'))
    set_gate('pregame', 'FAIL' if not start or start <= at else 'PASS', 'START_UNAVAILABLE' if not start else 'GAME_STARTED' if start <= at else None)
    outcome = health.get('outcome', 'NOT_RECORDED')
    failed = outcome not in {'SUCCESS', 'SUCCESS_EMPTY', 'NOT_RECORDED'} or health.get('processing') == 'FAILED'
    set_gate('provider_request', 'FAIL' if failed else 'UNKNOWN' if outcome == 'NOT_RECORDED' else 'PASS',
             'PROVIDER_FAILURE' if failed else 'PROVIDER_STATUS_UNKNOWN' if outcome == 'NOT_RECORDED' else None)
    pool = [c for c in candidates if text(c, 'market_type') == market]
    offers = [p for p in providers if _quotes(p, market)]
    matched = bool(providers or candidates)
    set_gate('odds_match', 'PASS' if matched else 'FAIL',
             None if matched else 'PROVIDER_SUCCESS_EMPTY' if outcome == 'SUCCESS_EMPTY' else 'NO_MATCHING_ODDS_EVENT')
    items = []
    if pool or offers:
        clocks = [_quote_gate(c, market, at) for c in pool or offers]
        good = any(g['status'] == 'PASS' for g in clocks)
        set_gate('quote_clock', 'PASS' if good else 'FAIL', None if good else clocks[0]['code'])
        set_gate('candidate_generation', 'PASS' if pool else 'FAIL', None if pool else 'CANDIDATE_GENERATION_LOSS')
    elif matched:
        set_gate('quote_clock', 'FAIL', 'ODDS_MARKET_MISSING')
    approved = False
    for candidate in pool:
        cid = text(candidate, 'candidate_id')
        receipts = [a for a in audit if text(a, 'candidate_id') == cid and cid]
        trace = deepcopy(receipts[0].get('coverage_gate_trace', [])) if len(receipts) == 1 else []
        receipt_matches = False
        if len(receipts) == 1:
            receipt = receipts[0]
            from app_core.public_quote_policy import canonical_book_label
            evaluated = clock(receipt.get('coverage_evaluated_at'))
            receipt_matches = bool(evaluated and evaluated <= at and
                receipt.get('coverage_run_id') == (text(candidate, 'finalization_run_id') or run) and
                receipt.get('sport') == event['league'] and receipt.get('market_type') == market and
                receipt.get('selection') == candidate.get('best_pick') and
                receipt.get('odds') == candidate.get('odds_american') and
                clock(receipt.get('start')) == start and
                str(receipt.get('game_id')) in {text(candidate, 'game_id'), text(candidate, 'matchup_id')} and
                receipt.get('line') == candidate.get('total_line' if market.startswith('total') else 'spread_line') and
                canonical_book_label(receipt.get('sportsbook')) == canonical_book_label(text(candidate, 'quote_bookmaker', 'opposing_odds_source', 'book', 'odds_source')) and
                clock(receipt.get('quote_timestamp')) == clock(text(candidate, 'odds_recorded_at', 'quote_time')))
            if not receipt_matches:
                trace = []
        if any(g.get('status') not in {'PASS', 'FAIL', 'NOT_EVALUATED', 'UNKNOWN'} for g in trace):
            raise ValueError('COVERAGE_GATE_SCHEMA')
        qt = _quote_gate(candidate, market, at)
        model_code = text(candidate, 'ml_unavailable_reason')
        status = text(candidate, 'ml_inference_status').lower()
        incompatible = any(k in model_code for k in ('RUNTIME_MISMATCH', 'MODEL_SCHEMA', 'ARTIFACT_READER', 'TARGET_CONFLICT', 'INCOMPATIBLE'))
        model = gate('model_evidence', 'FAIL' if incompatible or status in {'unavailable', 'failed', 'error'} else 'UNKNOWN',
            'MODEL_INCOMPATIBLE' if incompatible else 'MODEL_INFERENCE_UNAVAILABLE' if status in {'unavailable', 'failed', 'error'} else 'MODEL_EVIDENCE_MISSING')
        if text(candidate, 'league', 'League').upper() == 'NCAAF' and status in {'unavailable', 'failed', 'error'}:
            from app_core.ncaaf_pipeline_evidence import RESULT_VERSIONS
            from app_core.ncaaf_pipeline_evidence import PUBLIC_REASONS
            try:
                retained = json.loads(candidate.get('ml_estimate_metadata', ''))
                if retained['ncaaf_inputs']['payload']['version'] in RESULT_VERSIONS and model_code in PUBLIC_REASONS:
                    model = gate('model_evidence', 'FAIL', model_code)
            except (ValueError, TypeError, KeyError):
                pass
        # A completed actual finalizer receipt resolves its own prerequisites;
        # no absent receipt is promoted using a legacy probability alias.
        if trace and not model_code and not any('model' in str(g.get('code', '')).lower() or 'probability' in str(g.get('code', '')).lower() for g in trace if g['status'] == 'FAIL'):
            model = gate('model_evidence', 'PASS')
        resolved = bool(trace) and qt['status'] == 'PASS' and model['status'] == 'PASS'
        actual = [g for g in trace if g['status'] == 'FAIL']
        missing = any(any(word in str(g.get('code', '')).lower() for word in ('missing', 'unverified', 'unvalidated', 'unavailable', 'stale', 'invalid', 'future', 'mismatch', 'unsupported')) for g in actual)
        is_approved = bool(receipt_matches and trace and not actual and
                          any(_current_approval(f, candidate, at, run) for f in finals))
        approved |= is_approved
        items.append(dict(candidate_id=cid, market=market, decision_state='APPROVED' if is_approved else 'PASS' if resolved and not missing and actual else 'UNVERIFIED',
            gate_results=[qt, model, *trace, gate('finalization', 'PASS' if is_approved else 'FAIL' if actual else 'UNKNOWN' if trace else 'NOT_EVALUATED',
                None if is_approved else 'WAGER_REJECTION_RECORDED' if actual else 'FINAL_APPROVAL_NOT_CURRENT' if trace else 'FINALIZATION_NOT_EVALUATED')],
            actual_gate_results=trace, first_actual_failure=actual[0] if actual else None,
            evaluated_offer=dict(selection=candidate['best_pick'], market=market,
                signed_line=receipt['line'], price=receipt['odds'], operator=receipt['sportsbook'],
                quote_clock=receipt['quote_timestamp']) if receipt_matches and trace else None,
            original_model_blocker=model_code or None))
    if pool:
        model_ok = any(i['gate_results'][1]['status'] == 'PASS' for i in items)
        set_gate('model_evidence', 'PASS' if model_ok else 'FAIL', None if model_ok else items[0]['gate_results'][1]['code'])
        set_gate('finalization', 'PASS' if approved else 'FAIL' if any(i['decision_state'] == 'PASS' for i in items) else 'NOT_EVALUATED',
            None if approved else 'WAGER_REJECTION_RECORDED' if any(i['decision_state'] == 'PASS' for i in items) else 'FINALIZATION_NOT_EVALUATED')
    resolved = bool(items) and all(i['decision_state'] in {'PASS', 'APPROVED'} for i in items)
    state = 'APPROVED' if approved else 'PASS' if resolved and all(g['status'] not in {'FAIL', 'UNKNOWN'} for g in observed[:-1]) else 'UNVERIFIED'
    failures = [g for g in observed if g['status'] == 'FAIL'] + [g for item in items for g in item['gate_results'] if g['status'] == 'FAIL']
    return dict(market=market, state=state, gate_results=observed, candidates=items,
        first_observed_failure=failures[0] if failures else None, observed_failures=failures)


def build_coverage(inventories, *, selected_date, as_of, run_id, candidates=(), provider_events=(), final=(),
                   gate_audit=(), provider_health=None, required_markets=MARKETS, policy_exclusions=(), leagues=None):
    """One decision per canonical retained event, independent of candidate survival."""
    if date.fromisoformat(selected_date).isoformat() != selected_date or clock(as_of) is None or not run_id:
        raise ValueError('COVERAGE_DATE_AS_OF_RUN_REQUIRED')
    at = clock(as_of)
    invs = [_normalize(i, selected_date, at) for i in inventories]
    declared_leagues = list(leagues if leagues is not None else [i['league'] for i in invs] or ['NCAAF', 'NFL'])
    for league in declared_leagues:
        if league not in {i['league'] for i in invs}:
            invs.append(_normalize(dict(version='slate-inventory-v1', league=league, source='unavailable',
                selected_date=selected_date, timezone='America/New_York', observed_at=as_of,
                status='UNAVAILABLE', completeness_basis='none', reasons=['INDEPENDENT_SCHEDULE_NOT_RETAINED'], events=[]), selected_date, at))
    if len({i['league'] for i in invs}) != len(invs):
        raise ValueError('COVERAGE_CONFLICTING_LEAGUE_INVENTORIES')
    events = [dict(e, league=i['league'], inventory_id=i['inventory_id']) for i in invs for e in i['events']]
    if len({e['canonical_event_id'] for e in events}) != len(events):
        raise ValueError('COVERAGE_DUPLICATE_OR_MISSING_CANONICAL_EVENT')
    stages = {}; orphans = []; conflicts = set()
    for name, values in (('providers', provider_events), ('candidates', candidates), ('final', final)):
        grouped = {}
        for value in rows(values):
            recorded_run = text(value, 'export_run_id', 'run_id')
            if recorded_run and recorded_run != run_id:
                raise ValueError('COVERAGE_CONFLICTING_RUN_IDENTITIES')
            eid, status, possible = _match(value, events)
            if eid:
                grouped.setdefault(eid, []).append(value)
            else:
                conflicts.update(possible)
                orphans.append(dict(stage=name, league=text(value, 'league', 'sport', 'League'),
                    provider_event_id=text(value, 'provider_event_id', 'id', 'matchup_id'),
                    status=status, possible_canonical_events=possible))
        stages[name] = grouped
    from app_core.provider_health import sanitized_health
    health = sanitized_health(provider_health).get('sports', {})
    keys = {'NFL': 'americanfootball_nfl', 'NCAAF': 'americanfootball_ncaaf'}
    decisions = []
    exclusions = {p['canonical_event_id']: p for p in policy_exclusions}
    for event in events:
        eid = event['canonical_event_id']
        markets = [_market(event, m, stages['providers'].get(eid, []), stages['candidates'].get(eid, []),
            stages['final'].get(eid, []), rows(gate_audit), at, run_id, health.get(keys.get(event['league']), {}), eid in conflicts)
            for m in required_markets]
        excluded = exclusions.get(eid)
        if excluded and (not all(text(excluded, k) for k in ('policy_id', 'reason_code')) or not excluded.get('scope')):
            raise ValueError('COVERAGE_POLICY_EXCLUSION_REFERENCE_REQUIRED')
        excluded_markets = set()
        if excluded:
            scope = excluded['scope']
            if isinstance(scope, list):
                if not all(isinstance(m, str) and m in MARKETS for m in scope) or len(set(scope)) != len(scope):
                    raise ValueError('COVERAGE_POLICY_EXCLUSION_SCOPE_INVALID')
                excluded_markets = set(scope)
            elif isinstance(scope, str):
                # A cohort policy such as Stage 1 is retained separately. It
                # cannot resolve research markets without an exact market scope.
                excluded_markets = {scope} if scope in MARKETS else set()
            else:
                raise ValueError('COVERAGE_POLICY_EXCLUSION_SCOPE_INVALID')
            # A declared policy exclusion resolves only its exact markets.
            # Existing candidate observations remain attached and unaltered.
            for market in markets:
                if market['market'] not in excluded_markets:
                    continue
                market['gate_results'] = [gate(name, 'NOT_EVALUATED') for name in GATES]
                policy_gate = gate('evaluation_policy', 'FAIL', excluded['reason_code'])
                market['gate_results'].append(policy_gate)
                market['observed_failures'] = [policy_gate] + [g for c in market['candidates'] for g in c['gate_results'] if g['status'] == 'FAIL']
                market['first_observed_failure'] = market['observed_failures'][0]
                if market['state'] != 'APPROVED':
                    market['state'] = 'PASS'
        state = 'APPROVED' if any(m['state'] == 'APPROVED' for m in markets) else 'PASS' if all(m['state'] == 'PASS' for m in markets) and markets else 'UNVERIFIED'
        failures = [g for m in markets for g in m['observed_failures']]
        codes = list(dict.fromkeys(g['code'] for g in failures if g.get('code')))
        if not codes and state == 'UNVERIFIED':
            codes = ['FINALIZATION_NOT_EVALUATED']
        decisions.append(dict(canonical_event_id=eid, league=event['league'], home_team=event['home_team'],
            away_team=event['away_team'], home_team_id=event.get('home_team_id'), away_team_id=event.get('away_team_id'),
            original_start=event.get('original_start'), selected_date=selected_date, timezone='America/New_York',
            as_of=at.isoformat(), run_id=run_id, inventory_id=event['inventory_id'], coverage_decision_state=state,
            blocker_codes=codes, explanation=EXPLANATIONS.get(codes[0], codes[0]) if codes else 'Current finalized approval' if state == 'APPROVED' else 'Required evaluation resolved without an approved selection',
            required_markets=list(required_markets), market_results=markets,
            first_observed_failure=failures[0] if failures else None, observed_failures=failures,
            policy_exclusion=deepcopy(excluded), divisions=event.get('divisions', []),
            stage1_cohort='SEPARATE_FROZEN_FBS_ONLY_POLICY_NOT_EVALUATED'))
    status = 'UNAVAILABLE' if not invs or all(i['status'] == 'UNAVAILABLE' for i in invs) else 'COMPLETE' if all(i['status'] == 'COMPLETE' for i in invs) else 'PARTIAL'
    payload = dict(version=VERSION, selected_date=selected_date, timezone='America/New_York', as_of=at.isoformat(), run_id=run_id,
        inventory_status=status, inventory_scope=[{k: i.get(k) for k in ('league', 'source', 'inventory_id', 'status', 'observed_at', 'source_window', 'completeness_basis', 'reasons')} for i in invs],
        scheduled_event_ids=[e['canonical_event_id'] for e in events], decisions=decisions, orphan_events=orphans,
        counts=dict(scheduled_events=len(events), decision_rows=len(decisions), states={s: sum(d['coverage_decision_state'] == s for d in decisions) for s in sorted(STATES)}),
        blockers_by_league_market=dict(Counter(f"{d['league']}:{m['market']}:{g['code']}" for d in decisions for m in d['market_results'] for g in m['observed_failures'] if g.get('code'))),
        scientific_acceptance=False, wagering_authority=False)
    payload.update(reconciliation=reconcile(payload), fully_reconciled=status == 'COMPLETE')
    return payload


def reconcile(report):
    expected = set(report['scheduled_event_ids'])
    actual = [r['canonical_event_id'] for r in report['decisions']]
    duplicates = sorted(k for k, n in Counter(actual).items() if n > 1)
    result = dict(missing=sorted(expected - set(actual)), duplicates=duplicates, extras=sorted(set(actual) - expected))
    if any(result.values()) or len(expected) != len(report['scheduled_event_ids']):
        raise ValueError('COVERAGE_INTERNAL_RECONCILIATION_MISMATCH:' + json.dumps(result, sort_keys=True))
    return result


def refresh(diagnostics, candidates, final, *, selected_date, as_of, run_id):
    """Use retained diagnostics only; never call a source or initialize storage."""
    inventories = []
    ncaaf = diagnostics.get('ncaaf_schedule')
    if isinstance(ncaaf, dict):
        inventories.append(native_ncaaf(ncaaf, selected_date))
    inventories.extend(diagnostics.get('retained_slate_inventories', []))
    inventories.extend(native_football(o, selected_date) for o in diagnostics.get('retained_football_schedules', []) if o.get('sport') == 'NFL')
    report = build_coverage(inventories, selected_date=selected_date, as_of=as_of, run_id=run_id,
        candidates=candidates, final=final, provider_events=diagnostics.get('coverage_provider_events', diagnostics.get('ncaaf_provider_games', [])),
        gate_audit=diagnostics.get('wager_contract_audit', []), provider_health=diagnostics.get('provider_health'),
        leagues=diagnostics.get('coverage_leagues'))
    diagnostics['slate_coverage'] = report
    return report


def publication_rows(best, candidate_audit, report):
    """Independent placeholders contain identity/reasons, never rejected metrics."""
    import pandas as pd
    from app_core.game_coverage import publication_games
    reconcile(report)
    if report['inventory_status'] == 'UNAVAILABLE':
        legacy, missing = publication_games(best, candidate_audit)
        legacy.attrs['slate_coverage'] = deepcopy(report)
        return legacy, missing
    scoped_leagues = {i['league'] for i in report['inventory_scope']}
    records = rows(best)
    events = report['decisions']
    indexed = {}
    binding_failures = []
    for row in records:
        run = text(row, 'export_run_id')
        if run and run != report['run_id']:
            raise ValueError('COVERAGE_CONFLICTING_RUN_IDENTITIES')
        from app_core.coverage_presentation import resolve, CoverageBindingConflict
        try:
            decision = resolve(row, events)
        except CoverageBindingConflict as exc:
            # Preserve decisions, but reject preview before package construction.
            # No rejected selection or metrics migrate to a placeholder.
            binding_failures.append(exc.diagnostic)
            continue
        eid = decision['canonical_event_id'] if decision else None
        if eid:
            if eid in indexed:
                raise ValueError('COVERAGE_DUPLICATE_FINAL_GAME')
            indexed[eid] = row
    output = []; placeholders = []
    for decision in events:
        eid = decision['canonical_event_id']
        row = dict(indexed[eid]) if eid in indexed else dict(
            export_run_id=report['run_id'], matchup_id=eid, league=decision['league'],
            Home=decision['home_team'], Away=decision['away_team'],
            home_team=decision['home_team'], away_team=decision['away_team'],
            game_start_utc=decision['original_start'], **{'Local Date': report['selected_date'],
            'Commence (Local)': decision['original_start'] or ''},
            Bettable=False, Play_Stake=0.0, production_eligible=False, wager_approved=False,
            coverage_reason=decision['explanation'], coverage_only=True)
        row.update(coverage_decision_state=decision['coverage_decision_state'],
            coverage_blocker_codes='|'.join(decision['blocker_codes']),
            coverage_explanation=decision['explanation'], coverage_inventory_id=decision['inventory_id'],
            coverage_as_of=report['as_of'], coverage_selected_date=report['selected_date'],
            coverage_decision=json.dumps(decision, sort_keys=True, allow_nan=False))
        output.append(row)
        if eid not in indexed:
            placeholders.append(row)
    # Existing other-league display remains unchanged and outside this inventory.
    # Unmatched scoped events are reported as orphans, never added to the slate.
    outside = best.loc[[text(r, 'league', 'sport', 'League').upper() not in scoped_leagues for r in records]] if not best.empty else best
    audit = candidate_audit if isinstance(candidate_audit, pd.DataFrame) else pd.DataFrame()
    audit = audit.loc[[text(r, 'league', 'sport', 'League').upper() not in scoped_leagues for r in rows(audit)]] if not audit.empty else audit
    if not outside.empty or not audit.empty:
        legacy, _ = publication_games(outside, audit)
        output.extend(legacy.to_dict('records'))
    result = pd.DataFrame(output)
    result.attrs['slate_coverage'] = deepcopy(report)
    if binding_failures:
        result.attrs['coverage_binding_failures'] = binding_failures
    return result, pd.DataFrame(placeholders)


def validate_report(report):
    """Strict informational schema, shared by owner exports and public package."""
    expected = {'version', 'selected_date', 'timezone', 'as_of', 'run_id', 'inventory_status', 'inventory_scope',
        'scheduled_event_ids', 'decisions', 'orphan_events', 'counts', 'blockers_by_league_market',
        'scientific_acceptance', 'wagering_authority', 'reconciliation', 'fully_reconciled'}
    if set(report) != expected or report['version'] != VERSION or report['timezone'] != 'America/New_York' or clock(report['as_of']) is None:
        raise ValueError('COVERAGE_REPORT_SCHEMA')
    if report['scientific_acceptance'] is not False or report['wagering_authority'] is not False:
        raise ValueError('COVERAGE_AUTHORITY_FORBIDDEN')
    if report['inventory_status'] not in {'COMPLETE', 'PARTIAL', 'UNAVAILABLE'}:
        raise ValueError('COVERAGE_REPORT_SCHEMA')
    if report['fully_reconciled'] != (report['inventory_status'] == 'COMPLETE') or report['reconciliation'] != reconcile(report):
        raise ValueError('COVERAGE_COMPLETENESS_CONFLICT')
    decision_keys = {'canonical_event_id', 'league', 'home_team', 'away_team', 'home_team_id', 'away_team_id',
        'original_start', 'selected_date', 'timezone', 'as_of', 'run_id', 'inventory_id', 'coverage_decision_state',
        'blocker_codes', 'explanation', 'required_markets', 'market_results', 'first_observed_failure',
        'observed_failures', 'policy_exclusion', 'divisions', 'stage1_cohort'}
    def check_gate(g):
        if not isinstance(g, dict) or set(g) != {'gate', 'status', 'code'} or g['status'] not in {'PASS', 'FAIL', 'NOT_EVALUATED', 'UNKNOWN'}:
            raise ValueError('COVERAGE_GATE_SCHEMA')
        if not isinstance(g['gate'], str) or g['code'] is not None and not isinstance(g['code'], str):
            raise ValueError('COVERAGE_GATE_SCHEMA')
        if g['code'] is not None and not re.fullmatch(r'[A-Za-z0-9_]{1,120}', g['code']):
            raise ValueError('COVERAGE_DIAGNOSTIC_CODE_REQUIRED')
    for item in report['inventory_scope']:
        if set(item) != {'league', 'source', 'inventory_id', 'status', 'observed_at', 'source_window', 'completeness_basis', 'reasons'}:
            raise ValueError('COVERAGE_INVENTORY_SCHEMA')
    for orphan in report['orphan_events']:
        if set(orphan) != {'stage', 'league', 'provider_event_id', 'status', 'possible_canonical_events'}:
            raise ValueError('COVERAGE_ORPHAN_SCHEMA')
    for decision in report['decisions']:
        if set(decision) != decision_keys or decision['coverage_decision_state'] not in STATES:
            raise ValueError('COVERAGE_DECISION_SCHEMA')
        for key in ('selected_date', 'timezone', 'as_of', 'run_id'):
            if decision[key] != report[key]:
                raise ValueError('COVERAGE_DECISION_BINDING_CONFLICT')
        for g in decision['observed_failures'] + ([decision['first_observed_failure']] if decision['first_observed_failure'] else []):
            check_gate(g)
        for market in decision['market_results']:
            if set(market) != {'market', 'state', 'gate_results', 'candidates', 'first_observed_failure', 'observed_failures'} or market['state'] not in STATES:
                raise ValueError('COVERAGE_MARKET_SCHEMA')
            for g in market['gate_results'] + market['observed_failures']:
                check_gate(g)
            if market['first_observed_failure']:
                check_gate(market['first_observed_failure'])
            for candidate in market['candidates']:
                if set(candidate) != {'candidate_id', 'market', 'decision_state', 'gate_results', 'original_model_blocker', 'actual_gate_results', 'first_actual_failure', 'evaluated_offer'} or candidate['decision_state'] not in STATES:
                    raise ValueError('COVERAGE_CANDIDATE_SCHEMA')
                for g in candidate['gate_results'] + candidate['actual_gate_results'] + ([candidate['first_actual_failure']] if candidate['first_actual_failure'] else []):
                    check_gate(g)
                if candidate['evaluated_offer'] is not None and set(candidate['evaluated_offer']) != {'selection','market','signed_line','price','operator','quote_clock'}:
                    raise ValueError('COVERAGE_OFFER_SCHEMA')
        if decision['policy_exclusion'] is not None and set(decision['policy_exclusion']) != {'canonical_event_id', 'policy_id', 'reason_code', 'scope'}:
            raise ValueError('COVERAGE_POLICY_EXCLUSION_REFERENCE_REQUIRED')
    expected_states = {s: sum(d['coverage_decision_state'] == s for d in report['decisions']) for s in sorted(STATES)}
    if report['counts'] != dict(scheduled_events=len(report['scheduled_event_ids']), decision_rows=len(report['decisions']), states=expected_states):
        raise ValueError('COVERAGE_COUNTS_CONFLICT')
    return report


def validate_public_coverage(report, games):
    validate_report(report)
    expected = {d['canonical_event_id']: d for d in report['decisions']}
    for family, group in games.items():
        for row in group:
            d = row.get('coverage_decision')
            if d:
                from app_core.coverage_presentation import CoverageBindingConflict
                for field, expected_value in (('sport', d['league']), ('game', d['away_team'] + ' at ' + d['home_team']), ('start', d['original_start'])):
                    agrees = clock(row[field]) == clock(expected_value) if field == 'start' else row[field] == expected_value
                    if not agrees:
                        raise CoverageBindingConflict(d, family, field, expected_value, row[field])
        observed = [r['coverage_decision'] for r in group if 'coverage_decision' in r]
        if len(observed) != len(expected) or {r['canonical_event_id'] for r in observed} != set(expected):
            raise ValueError('COVERAGE_PUBLIC_RECONCILIATION_MISMATCH')
        if any(r != expected[r['canonical_event_id']] for r in observed):
            raise ValueError('COVERAGE_PUBLIC_DECISION_CONFLICT')
    for eid, decision in expected.items():
        if decision['coverage_decision_state'] != 'APPROVED':
            continue
        approved = [r for group in games.values() for r in group if r.get('coverage_decision', {}).get('canonical_event_id') == eid and r.get('status') == 'APPROVED']
        if not approved:
            raise ValueError('COVERAGE_APPROVAL_NOT_ON_PUBLIC_BOARD')
