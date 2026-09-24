"""Fail-closed evaluation of exact Standard, Same Game, and Cross Game tickets.

This module does not fetch prices, activate a product, or place a wager. Callers
must supply immutable pregame evidence, an actual sportsbook ticket quote, an
independently validated product policy, owner authorization, and a fresh ledger
snapshot. Missing inputs remain explicit blockers and have no stake authority.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any

from app_core.public_quote_policy import canonical_book_label, supported_quote
from core.exposure_ledger import verify_snapshot
from core.market_policy import sport_market_family
from core.wager_decisions import aware, decimal_price, finite


PRODUCTS = frozenset({'STANDARD_PARLAY', 'SAME_GAME_PARLAY', 'CROSS_GAME_PARLAY'})
VALIDATION_STATES = frozenset({'UNVALIDATED', 'PROVISIONAL_VALIDATED', 'STANDARD_VALIDATED'})
DEPENDENCE = frozenset({'INDEPENDENT_VERIFIED', 'LOW_DEPENDENCE', 'MATERIAL_DEPENDENCE', 'UNKNOWN'})
LEG_IDENTITY = ('candidate_id', 'game_id', 'sport', 'provider_namespace',
                'provider_event_id', 'provider_market_id', 'provider_selection_id', 'market_type',
                'selection', 'line', 'sportsbook', 'market_family', 'model_id',
                'model_version', 'calibration_id', 'calibration_version',
                'validation_id', 'validation_artifact_id', 'deployment_state')


def digest(value: Any) -> str:
    """Stable identity for exact immutable inputs; never an evidence ID."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False, default=str).encode()).hexdigest()


def _number(value: Any, *, low: float | None = None, high: float | None = None) -> float | None:
    result = finite(value)
    return result if result is not None and (low is None or result >= low) and (high is None or result <= high) else None


def _at(value: Any, now: datetime) -> bool:
    parsed = aware(value)
    return parsed is not None and parsed <= now


def _future(value: Any, now: datetime) -> bool:
    parsed = aware(value)
    return parsed is not None and parsed > now


def _fresh(value: Any, now: datetime, seconds: int = 1800) -> bool:
    parsed = aware(value)
    return parsed is not None and 0 <= (now - parsed).total_seconds() <= seconds


def _required_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def normalize_leg(row: dict) -> dict:
    """Adapt a saved public game row without inventing missing provenance."""
    c = row.get('wager_contract') if isinstance(row.get('wager_contract'), dict) else {}
    return {
        'candidate_id': row.get('candidate_id'),
        'game_id': c.get('game_id') or row.get('game_id'),
        'sport': c.get('sport') or row.get('sport'),
        'provider_namespace': c.get('provider_namespace') or row.get('provider_namespace'),
        'provider_event_id': c.get('provider_event_id') or row.get('provider_event_id'),
        'provider_market_id': c.get('provider_market_id') or row.get('provider_market_id'),
        'provider_selection_id': c.get('provider_selection_id') or row.get('provider_selection_id'),
        'team_ids': row.get('team_ids'),
        'market_type': c.get('market_type') or row.get('market'),
        'selection': c.get('selection') or row.get('pick'),
        'line': c.get('line'),
        'sportsbook': c.get('sportsbook') or row.get('quote_source'),
        'american_odds': c.get('odds') if c.get('odds') is not None else row.get('odds'),
        'quote_timestamp': c.get('quote_timestamp') or row.get('quote_time'),
        'analysis_timestamp': row.get('as_of'),
        'start': c.get('start') or row.get('start'),
        'identity_verified': c.get('identity_verified'),
        'quote_verified': c.get('quote_verified'),
        'production_eligible': c.get('production_eligible'),
        'model_version': c.get('model_version'),
        'model_id': c.get('model_id'),
        'model_trained_through': row.get('model_trained_through'),
        'model_available_at': row.get('model_available_at'),
        'calibration_version': c.get('calibration_version'),
        'calibration_id': c.get('calibration_id'),
        'calibration_available_at': row.get('calibration_available_at'),
        'market_family': c.get('market_family'),
        'validation_id': c.get('validation_id'),
        'validation_artifact_id': c.get('validation_artifact_id'),
        'deployment_state': c.get('deployment_state'),
        'policy_version': c.get('sport_policy_version'),
        'evidence_snapshot_id': c.get('evidence_version'),
        'evidence_frozen_at': row.get('evidence_frozen_at'),
        'critical_input_state': row.get('critical_input_state'),
        'material_news_status': row.get('material_news_status'),
        'probability_semantics': row.get('probability_semantics'),
        'probability_mean': c.get('calibrated_probability'),
        'probability_conservative': c.get('conservative_probability'),
        'probability_push': row.get('probability_push'),
        'probability_loss': row.get('probability_loss'),
    }


def leg_identity(leg: dict) -> dict:
    return {key: leg.get(key) for key in LEG_IDENTITY}


def leg_hash(leg: dict) -> str:
    return digest(leg_identity(leg))


def exposure_leg_key(leg: dict) -> str:
    """Market identity shared by straight and parlay ledger entries."""
    return digest({key: leg.get(key) for key in ('sport', 'game_id', 'market_type', 'selection', 'line')})


def leg_blockers(leg: dict, now: datetime) -> list[str]:
    """Exact, independent straight-wager admission. Unknown is never approval."""
    if not isinstance(leg, dict):
        return ['LEG_IDENTITY_MISSING']
    reasons = []
    if any(not _required_text(leg.get(k)) for k in ('candidate_id', 'game_id', 'sport', 'market_type', 'selection')) or _number(leg.get('line')) is None:
        reasons.append('LEG_IDENTITY_MISSING')
    if any(not _required_text(leg.get(k)) for k in ('provider_namespace', 'provider_event_id')):
        reasons.append('LEG_PROVIDER_EVENT_MISSING')
    teams = leg.get('team_ids')
    if (not isinstance(teams, list) or len(teams) != 2 or
            any(not _required_text(team) for team in teams) or len(set(teams)) != 2):
        reasons.append('LEG_TEAM_IDENTITY_MISSING')
    book = canonical_book_label(leg.get('sportsbook'))
    if not supported_quote({'quote_source': book, 'sport': leg.get('sport') or ''}):
        reasons.append('LEG_BOOK_UNSUPPORTED')
    american = _number(leg.get('american_odds'))
    if american is None or abs(american) > 10000 or decimal_price(american) is None:
        reasons.append('LEG_PRICE_INVALID')
    market = str(leg.get('market_type') or '')
    selection = str(leg.get('selection') or '')
    selected_line = re.search(r'([+-]?\d+(?:\.\d+)?)\s*$', selection)
    line_value = _number(leg.get('line'))
    if (market.startswith('total_') and not selection.casefold().startswith(('over ' if market.endswith('over') else 'under '))
            or market.startswith(('spread_', 'total_')) and
            (selected_line is None or line_value is None or abs(float(selected_line.group(1)) - line_value) > 1e-8)):
        reasons.append('LEG_SELECTION_LINE_MISMATCH')
    if leg.get('identity_verified') is not True or leg.get('quote_verified') is not True:
        reasons.append('LEG_IDENTITY_UNVERIFIED')
    if not _fresh(leg.get('quote_timestamp'), now):
        reasons.append('LEG_QUOTE_STALE')
    if not _fresh(leg.get('analysis_timestamp'), now):
        reasons.append('LEG_ANALYSIS_STALE')
    if not _future(leg.get('start'), now):
        reasons.append('LEG_STARTED_OR_TIME_UNKNOWN')
    if leg.get('production_eligible') is not True:
        reasons.append('LEG_NOT_PRODUCTION_ELIGIBLE')
    if (leg.get('market_family') != sport_market_family(leg.get('sport'), leg.get('market_type'))
            or leg.get('deployment_state') not in {'PROVISIONAL_VALIDATED','STANDARD_VALIDATED','PREMIUM_VALIDATED'}
            or not all(_required_text(leg.get(k)) for k in
                       ('model_id','model_version','calibration_id','calibration_version',
                        'validation_id','validation_artifact_id'))):
        reasons.append('LEG_EXACT_MARKET_VALIDATION_MISSING')
    if not all(_required_text(leg.get(k)) for k in ('model_version', 'calibration_version', 'policy_version', 'evidence_snapshot_id')):
        reasons.append('LEG_EVIDENCE_MISSING')
    trained = aware(leg.get('model_trained_through'))
    analysis = aware(leg.get('analysis_timestamp'))
    if (trained is None or analysis is None or trained >= analysis or
            any(not _at(leg.get(k), analysis) for k in ('model_available_at', 'calibration_available_at', 'evidence_frozen_at'))):
        reasons.append('LEG_CHRONOLOGY_INVALID')
    if leg.get('critical_input_state') != 'CLEAR':
        reasons.append('LEG_CRITICAL_INPUT')
    if leg.get('material_news_status') != 'CLEAR':
        reasons.append('LEG_MATERIAL_NEWS')
    mean, conservative, push, loss = (_number(leg.get(k), low=0, high=1) for k in
                                      ('probability_mean', 'probability_conservative', 'probability_push', 'probability_loss'))
    if (leg.get('probability_semantics') != 'UNCONDITIONAL' or None in (mean, conservative, push, loss)
            or (mean is not None and conservative is not None and conservative > mean)
            or (None not in (mean, push, loss) and abs(mean + push + loss - 1) > 1e-6)):
        reasons.append('LEG_PROBABILITY_INVALID')
    line = _number(leg.get('line'))
    if (line is not None and push is not None and abs(line * 2 - round(line * 2)) < 1e-9
            and abs(line - round(line)) > 1e-9 and push > 1e-9):
        reasons.append('HALF_POINT_PUSH_IMPOSSIBLE')
    return reasons


def mutually_exclusive(left: dict, right: dict) -> bool:
    """Known opposite selections that cannot both win under exact lines."""
    if left.get('game_id') != right.get('game_id'):
        return False
    lm, rm = str(left.get('market_type') or ''), str(right.get('market_type') or '')
    ll, rl = _number(left.get('line')), _number(right.get('line'))
    if None in (ll, rl):
        return False
    if {lm, rm} == {'total_over', 'total_under'}:
        over = ll if lm == 'total_over' else rl
        under = ll if lm == 'total_under' else rl
        return over >= under
    if {lm, rm} == {'spread_home', 'spread_away'}:
        return ll + rl <= 0
    return lm == rm and ll == rl and left.get('selection') != right.get('selection')


def _component_legs(component: dict) -> list[dict]:
    return component.get('legs', []) if component.get('product_type') == 'SAME_GAME_PARLAY' else [component]


def _flatten_components(components: list[dict]) -> list[dict]:
    legs = []
    for component in components:
        if component.get('product_type') == 'SAME_GAME_PARLAY':
            legs.extend({**leg, 'sgp_component_id': component.get('parlay_id')} for leg in _component_legs(component))
        else:
            legs.append(component)
    return legs


def _component_hash(component: dict) -> str:
    if component.get('product_type') == 'SAME_GAME_PARLAY':
        return digest({'sgp_ticket_hash': component.get('ticket_hash'),
                       'leg_hashes': sorted(leg_hash(r) for r in component.get('legs', []))})
    return leg_hash(component)


def component_hashes(components: list[dict]) -> list[str]:
    """Hashes a joint model must bind before any line or selection is changed."""
    return sorted(_component_hash(component) for component in components)


def ticket_hash(product_type: str, components: list[dict], sportsbook: str | None) -> str:
    return digest({'product_type': product_type, 'sportsbook': canonical_book_label(sportsbook),
                   'components': component_hashes(components)})


def _joint_blockers(joint: dict | None, product_type: str, now: datetime, policy: dict) -> list[str]:
    if not isinstance(joint, dict):
        return ['JOINT_MODEL_UNAVAILABLE']
    reasons = []
    if not all(_required_text(joint.get(k)) for k in ('model_id', 'model_version', 'calibration_version', 'validation_id', 'evidence_snapshot_id', 'method')):
        reasons.append('JOINT_MODEL_UNAVAILABLE')
    if joint.get('product_type') != product_type or joint.get('validation_id') != policy.get('validation_id'):
        reasons.append('JOINT_VALIDATION_MISMATCH')
    if product_type == 'SAME_GAME_PARLAY' and 'INDEPEND' in str(joint.get('method')).upper():
        reasons.append('SGP_INDEPENDENCE_FORBIDDEN')
    if product_type == 'SAME_GAME_PARLAY' and (
            not _required_text(joint.get('correlation_method')) or
            'INDEPEND' in str(joint.get('correlation_method')).upper()):
        reasons.append('SGP_CORRELATION_METHOD_UNAVAILABLE')
    if product_type == 'CROSS_GAME_PARLAY' and any(
            not _required_text(joint.get(key)) for key in
            ('component_dependence_method', 'shared_factor_method',
             'final_calibration_id', 'final_calibration_version')):
        reasons.append('CROSS_GAME_DEPENDENCE_UNAVAILABLE')
    generated = aware(joint.get('generated_at'))
    trained = aware(joint.get('model_trained_through'))
    if (generated is None or generated > now or trained is None or trained >= generated or
            any(not _at(joint.get(k), generated) for k in ('model_available_at', 'calibration_available_at', 'evidence_frozen_at'))):
        reasons.append('JOINT_CHRONOLOGY_INVALID')
    mean, conservative, push, loss = (_number(joint.get(k), low=0, high=1) for k in
                                      ('probability_mean', 'probability_conservative', 'probability_push', 'probability_loss'))
    partial = joint.get('partial_outcomes', [])
    partial_mean = partial_conservative = 0.0
    if not isinstance(partial, list):
        reasons.append('JOINT_PROBABILITY_INVALID')
    else:
        seen = set()
        for state in partial:
            if not isinstance(state, dict) or not _required_text(state.get('state_id')) or state['state_id'] in seen:
                reasons.append('JOINT_PROBABILITY_INVALID')
                continue
            seen.add(state['state_id'])
            pm = _number(state.get('probability_mean'), low=0, high=1)
            pc = _number(state.get('probability_conservative'), low=0, high=1)
            if pm is None or pc is None or pc > pm:
                reasons.append('JOINT_PROBABILITY_INVALID')
                continue
            partial_mean += pm
            partial_conservative += pc
    if (joint.get('probability_semantics') != 'UNCONDITIONAL' or None in (mean, conservative, push, loss)
            or (mean is not None and conservative is not None and conservative > mean)
            or (None not in (mean, push, loss) and abs(mean + push + loss + partial_mean - 1) > 1e-6)
            or (None not in (conservative, push) and conservative + push + partial_conservative > 1 + 1e-6)):
        reasons.append('JOINT_PROBABILITY_INVALID')
    return reasons


def minimum_acceptable_price(p_win: Any, p_push: Any, minimum_ev: Any, minimum_edge: Any,
                             *, other_return: Any = None) -> float | None:
    """Worst decimal ticket price satisfying both conservative EV and edge."""
    win, push = _number(p_win, low=0, high=1), _number(p_push, low=0, high=1)
    ev, edge = _number(minimum_ev, low=0), _number(minimum_edge, low=0)
    baseline = push if other_return is None else _number(other_return, low=0)
    if None in (win, push, ev, edge, baseline) or win <= edge or win <= 0 or win + push > 1:
        return None
    return max(1.0, (1 - baseline + ev) / win, (1 - baseline) / (win - edge))


def american_price(decimal: Any) -> int | None:
    value = _number(decimal)
    if value is None or value <= 1:
        return None
    return math.ceil((value - 1) * 100) if value >= 2 else -math.floor(100 / (value - 1))


def _partial_return(joint: dict, quote: dict, legs: list[dict]) -> tuple[float | None, list[str]]:
    """Value partial pushes using the actual sportsbook's reduced payouts."""
    states = joint.get('partial_outcomes', [])
    leg_push = any((_number(leg.get('probability_push'), low=0) or 0) > 0 for leg in legs)
    if not leg_push and not states:
        return _number(joint.get('probability_push'), low=0, high=1), []
    if leg_push and not states:
        return None, ['SETTLEMENT_MODEL_UNAVAILABLE']
    if (not _required_text(joint.get('settlement_rules_id')) or
            joint.get('settlement_rules_id') != quote.get('settlement_rules_id') or
            not isinstance(states, list) or not isinstance(quote.get('partial_payouts'), list)):
        return None, ['SETTLEMENT_MODEL_UNAVAILABLE']
    payouts = {}
    for item in quote['partial_payouts']:
        if not isinstance(item, dict) or not _required_text(item.get('state_id')) or item['state_id'] in payouts:
            return None, ['SETTLEMENT_MODEL_UNAVAILABLE']
        price = _number(item.get('decimal_return'))
        full_price = _number(quote.get('decimal_odds'))
        if price is None or price <= 1 or full_price is None or price > full_price:
            return None, ['SETTLEMENT_MODEL_UNAVAILABLE']
        payouts[item['state_id']] = price
    if set(payouts) != {state.get('state_id') for state in states if isinstance(state, dict)}:
        return None, ['SETTLEMENT_MODEL_UNAVAILABLE']
    result = _number(joint.get('probability_push'), low=0, high=1)
    if result is None:
        return None, ['SETTLEMENT_MODEL_UNAVAILABLE']
    for state in states:
        probability = _number(state.get('probability_conservative'), low=0, high=1)
        if probability is None:
            return None, ['SETTLEMENT_MODEL_UNAVAILABLE']
        result += probability * payouts[state['state_id']]
    return result, []


def _allocation_keys(legs: list[dict], ticket_id: str, product_type: str, policy: dict,
                     exposure: dict) -> dict[str, float] | None:
    caps = policy['exposure_caps']
    keys = {'total': exposure['total_cap'], 'daily': exposure['daily_cap'], 'weekly': exposure['weekly_cap'],
            f'parlay:{ticket_id}': caps['parlay']}
    if product_type == 'SAME_GAME_PARLAY' or any(leg.get('sgp_component_id') for leg in legs):
        keys['sgp:total'] = caps['sgp']
    for leg in legs:
        sport, game = leg.get('sport'), leg.get('game_id')
        if not sport or not game or not isinstance(leg.get('team_ids'), list) or len(leg['team_ids']) != 2:
            return None
        keys[f'sport:{sport}'] = caps['sport']
        keys[f'game:{sport}:{game}'] = exposure['game_cap']
        market_key = exposure_leg_key(leg)
        keys[f'leg:{market_key}'] = caps['leg']
        keys[f'overlap:{market_key}'] = caps['overlap']
        for team in leg['team_ids']:
            keys[f'team:{sport}:{team}'] = exposure['team_cap']
    return keys


def _allocation(legs: list[dict], ticket_id: str, product_type: str, decimal: float,
                p_win: float, p_push: float, policy: dict, exposure: dict | None,
                review: dict | None, now: datetime,
                reservations: dict[str, float] | None = None) -> tuple[float, float, list[str], str | None]:
    reasons = []
    try:
        verify_snapshot(exposure, now=now)
    except (ValueError, TypeError, KeyError, AttributeError):
        return 0.0, 0.0, ['EXPOSURE_UNAVAILABLE'], None
    caps = policy.get('exposure_caps')
    if not isinstance(caps, dict) or not all(_number(caps.get(k), low=0, high=1) is not None for k in ('sport', 'leg', 'parlay', 'sgp', 'overlap')):
        return 0.0, 0.0, ['EXPOSURE_POLICY_MISSING'], exposure.get('snapshot_hash')
    bankroll = _number(exposure.get('bankroll'), low=0)
    if not bankroll:
        return 0.0, 0.0, ['BANKROLL_UNAVAILABLE'], exposure.get('snapshot_hash')
    used = exposure['committed']
    keys = _allocation_keys(legs, ticket_id, product_type, policy, exposure)
    if keys is None:
        return 0.0, 0.0, ['EXPOSURE_IDENTITY_MISSING'], exposure.get('snapshot_hash')
    reservations = reservations or {}
    remaining = min(max(0.0, limit - (finite(used.get(key, 0)) or 0)
                        - (finite(reservations.get(key, 0)) or 0)) for key, limit in keys.items())
    stake_cap = _number(policy.get('stake_cap'), low=0, high=1)
    provisional_cap = _number(policy.get('provisional_cap'), low=0, high=1)
    kelly_fraction = _number(policy.get('kelly_fraction'), low=0, high=1)
    if None in (stake_cap, provisional_cap, kelly_fraction):
        return 0.0, 0.0, ['STAKE_POLICY_MISSING'], exposure.get('snapshot_hash')
    if policy['validation_state'] == 'PROVISIONAL_VALIDATED':
        stake_cap = min(stake_cap, provisional_cap)
    # Full-ticket push returns the stake. Treat partial wins as losses here;
    # that understates Kelly and can only reduce the proposed fraction.
    win_profit = decimal - 1
    kelly = max(0.0, (p_win * win_profit - (1 - p_win - p_push)) / ((1 - p_push) * win_profit)) if win_profit > 0 and p_push < 1 else 0.0
    multiplier = 1.0 if review.get('status') == 'CONFIRM' else _number(review.get('stake_multiplier'), low=0, high=1)
    if multiplier is None:
        reasons.append('REVIEW_MULTIPLIER_INVALID')
        multiplier = 0.0
    fraction = max(0.0, min(stake_cap, remaining, kelly * kelly_fraction * multiplier))
    if fraction <= 0:
        reasons.append('EXPOSURE_OR_KELLY_EXHAUSTED')
    return bankroll * fraction, fraction, reasons, exposure.get('snapshot_hash')


def evaluate_ticket(product_type: str, components: list[dict], *, sportsbook: str | None,
                    quote: dict | None = None, joint: dict | None = None,
                    dependence: dict | None = None, policy: dict | None = None,
                    authorization: dict | None = None, exposure: dict | None = None,
                    review: dict | None = None, now: datetime | None = None,
                    reservations: dict[str, float] | None = None) -> dict:
    """Evaluate an exact ticket; all external authority is explicit and fail closed."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError('Evaluation time must be timezone-aware')
    now = now.astimezone(timezone.utc)
    malformed_components = not isinstance(components, (list, tuple)) or any(
        not isinstance(c, dict) or (c.get('product_type') == 'SAME_GAME_PARLAY' and
                                    (not isinstance(c.get('legs'), list) or
                                     any(not isinstance(leg, dict) for leg in c['legs'])))
        for c in (components if isinstance(components, (list, tuple)) else []))
    components = list(components) if not malformed_components else []
    policy = policy if isinstance(policy, dict) else {}
    authorization = authorization if isinstance(authorization, dict) else {}
    review = review if isinstance(review, dict) else {}
    dependence = dependence if isinstance(dependence, dict) else {}
    joint = joint if isinstance(joint, dict) else None
    legs = _flatten_components(components)
    book = canonical_book_label(sportsbook)
    identity_hash = ticket_hash(product_type, components, book)
    blockers = ['COMPONENTS_INVALID'] if malformed_components else []
    if product_type not in PRODUCTS:
        blockers.append('PRODUCT_UNSUPPORTED')
    if not legs or any(not isinstance(leg, dict) for leg in legs):
        blockers.append('LEGS_MISSING')
        legs = [leg for leg in legs if isinstance(leg, dict)]
    count = len(legs)
    if product_type == 'STANDARD_PARLAY' and not 2 <= count <= 4:
        blockers.append('LEG_COUNT_INVALID')
    if product_type == 'SAME_GAME_PARLAY' and not 2 <= count <= 4:
        blockers.append('LEG_COUNT_INVALID')
    if product_type == 'CROSS_GAME_PARLAY' and not 2 <= count <= 6:
        blockers.append('LEG_COUNT_INVALID')
    ids = [leg_hash(leg) for leg in legs]
    if len(set(ids)) != count or len({leg.get('candidate_id') for leg in legs}) != count:
        blockers.append('DUPLICATE_LEG')
    games = [leg.get('game_id') for leg in legs]
    if product_type == 'STANDARD_PARLAY' and len(set(games)) != count:
        blockers.append('DUPLICATE_GAME')
    if product_type == 'SAME_GAME_PARLAY' and len(set(games)) != 1:
        blockers.append('NOT_SAME_GAME')
    if product_type == 'CROSS_GAME_PARLAY' and (len(components) < 2 or len({tuple(sorted({leg.get('game_id') for leg in _component_legs(c)})) for c in components}) != len(components)):
        blockers.append('COMPONENT_GAME_CONFLICT')
    if any(mutually_exclusive(a, b) for i, a in enumerate(legs) for b in legs[i + 1:]):
        blockers.append('MUTUALLY_EXCLUSIVE_LEGS')
    if any(c.get('product_type') == 'SAME_GAME_PARLAY' for c in components):
        if product_type != 'CROSS_GAME_PARLAY':
            blockers.append('SGP_COMPONENT_NOT_ALLOWED')
        for c in components:
            if c.get('product_type') == 'SAME_GAME_PARLAY':
                expected_sgp_hash = ticket_hash('SAME_GAME_PARLAY', c.get('legs', []), c.get('sportsbook'))
                if c.get('ticket_hash') != expected_sgp_hash:
                    blockers.append('SGP_COMPONENT_HASH_MISMATCH')
                if (c.get('production_eligible') is not True or c.get('status') != 'ACTIONABLE' or
                        not _required_text(c.get('validation_id')) or
                        c.get('validation_state') not in {'PROVISIONAL_VALIDATED', 'STANDARD_VALIDATED'} or
                        c.get('quote_verification_state') != 'VERIFIED' or
                        not _fresh(c.get('quoted_at'), now) or not _future(c.get('expires_at'), now) or
                        c.get('blockers')):
                    blockers.append('SGP_COMPONENT_UNVALIDATED')
    for leg in legs:
        blockers.extend(leg_blockers(leg, now))
        if canonical_book_label(leg.get('sportsbook')) != book:
            blockers.append('BOOK_MISMATCH')
    dependency = dependence.get('classification') if isinstance(dependence, dict) else None
    if product_type != 'SAME_GAME_PARLAY':
        if dependency not in DEPENDENCE or not _at((dependence or {}).get('assessed_at'), now) or not _future((dependence or {}).get('expires_at'), now) or not _required_text((dependence or {}).get('evidence_id')):
            blockers.append('DEPENDENCE_UNKNOWN')
        if dependency in {'UNKNOWN', 'MATERIAL_DEPENDENCE'} and not joint:
            blockers.append('DEPENDENCE_JOINT_MODEL_REQUIRED')
        if 'INDEPEND' in str((joint or {}).get('method')).upper() and dependency != 'INDEPENDENT_VERIFIED':
            blockers.append('INDEPENDENCE_NOT_VERIFIED')
    if product_type == 'SAME_GAME_PARLAY' and len(components) != count:
        blockers.append('SGP_COMPONENT_SHAPE_INVALID')
    if policy.get('product_type') != product_type or policy.get('validation_state') not in VALIDATION_STATES or policy.get('validation_state') == 'UNVALIDATED' or not _required_text(policy.get('validation_id')) or not _required_text(policy.get('policy_version')) or not _future(policy.get('expires_at'), now) or not _at(policy.get('frozen_at'), now):
        blockers.append('PRODUCT_UNVALIDATED')
    blockers.extend(_joint_blockers(joint, product_type, now, policy))
    if isinstance(joint, dict) and joint.get('component_hashes') != component_hashes(components):
        blockers.append('JOINT_INPUT_MISMATCH')
    authorized_products = authorization.get('product_types')
    if (authorization.get('status') != 'ACTIVE' or
            not isinstance(authorized_products, (list, tuple, set)) or product_type not in authorized_products or
            not _required_text(authorization.get('authorization_id')) or not _at(authorization.get('granted_at'), now) or
            not _future(authorization.get('expires_at'), now) or authorization.get('revoked_at')):
        blockers.append('OWNER_AUTHORIZATION_MISSING')
    if review.get('status') not in {'CONFIRM', 'REDUCE'}:
        blockers.append('REVIEW_HOLD')
    if review.get('status') == 'REDUCE' and _number(review.get('stake_multiplier'), low=0, high=1) is None:
        blockers.append('REVIEW_MULTIPLIER_INVALID')
    if any(_number(policy.get(key), low=0, high=1) is None for key in ('stake_cap', 'provisional_cap', 'kelly_fraction')):
        blockers.append('STAKE_POLICY_MISSING')
    caps = policy.get('exposure_caps')
    if not isinstance(caps, dict) or any(_number(caps.get(key), low=0, high=1) is None for key in ('sport', 'leg', 'parlay', 'sgp', 'overlap')):
        blockers.append('EXPOSURE_POLICY_MISSING')
    try:
        verify_snapshot(exposure, now=now)
    except (ValueError, TypeError, KeyError, AttributeError):
        blockers.append('EXPOSURE_UNAVAILABLE')
    quote = quote if isinstance(quote, dict) else {}
    quoted_decimal = _number(quote.get('decimal_odds'))
    quoted_american = _number(quote.get('american_odds'))
    quote_reasons = []
    if not _required_text(quote.get('quote_id')) or not _required_text(quote.get('provider_ticket_id')):
        quote_reasons.append('TICKET_QUOTE_ID_MISSING')
    if quote.get('source') not in {'SPORTSBOOK', 'OWNER_CONFIRMED'} or quote.get('verification_state') != 'VERIFIED':
        quote_reasons.append('TICKET_QUOTE_UNVERIFIED')
    if (not _required_text(quote.get('provider')) or
            not isinstance(quote.get('raw_evidence_hash'), str) or
            re.fullmatch(r'[0-9a-f]{64}', quote['raw_evidence_hash']) is None or
            not _required_text(quote.get('settlement_rules_id')) or
            (quote.get('source') == 'SPORTSBOOK' and
             not _required_text(quote.get('provider_response_id'))) or
            (quote.get('source') == 'OWNER_CONFIRMED' and
             any(not _required_text(quote.get(k)) for k in
                 ('owner_id', 'owner_authorization_id', 'owner_confirmed_at', 'artifact_reference')))):
        quote_reasons.append('TICKET_QUOTE_EVIDENCE_MISSING')
    if quote.get('source') == 'OWNER_CONFIRMED':
        confirmed_at = aware(quote.get('owner_confirmed_at'))
        quote_at = aware(quote.get('quoted_at'))
        if confirmed_at is None or quote_at is None or not quote_at <= confirmed_at <= now:
            quote_reasons.append('OWNER_QUOTE_CHRONOLOGY_INVALID')
    try:
        from app_core.parlay_ticket_quotes import bind_ticket_request
        exact_binding = bind_ticket_request({'product_type': product_type,
                                             'components': components, 'sportsbook': book})
    except (ValueError, TypeError, OverflowError):
        exact_binding = None
    if (exact_binding is None or
            quote.get('selection_bindings') != exact_binding['selection_bindings'] or
            quote.get('component_hashes') != exact_binding['component_hashes'] or
            quote.get('sgp_components') != exact_binding['sgp_components']):
        quote_reasons.append('TICKET_BINDING_MISMATCH')
    if quote.get('ticket_hash') != identity_hash or quote.get('leg_hashes') != sorted(ids):
        quote_reasons.append('TICKET_HASH_MISMATCH')
    if canonical_book_label(quote.get('sportsbook')) != book:
        quote_reasons.append('TICKET_BOOK_MISMATCH')
    if not _fresh(quote.get('quoted_at'), now):
        quote_reasons.append('TICKET_QUOTE_STALE')
    if not _future(quote.get('expires_at'), now):
        quote_reasons.append('TICKET_QUOTE_EXPIRED')
    if (quoted_decimal is None or quoted_decimal <= 1 or quoted_american is None or
            decimal_price(quoted_american) is None or abs(decimal_price(quoted_american) - quoted_decimal) > 0.002):
        quote_reasons.append('TICKET_PRICE_CONFLICT')
    if quote_reasons:
        blockers.append('PRICE_UNAVAILABLE')
        blockers.extend(quote_reasons)
    mean = _number((joint or {}).get('probability_mean'), low=0, high=1)
    conservative = _number((joint or {}).get('probability_conservative'), low=0, high=1)
    push = _number((joint or {}).get('probability_push'), low=0, high=1)
    partial_states = joint.get('partial_outcomes', []) if isinstance(joint, dict) else None
    partial_values = ([_number(state.get('probability_conservative'), low=0, high=1)
                       if isinstance(state, dict) else None for state in partial_states]
                      if isinstance(partial_states, list) else None)
    partial_probability = (sum(partial_values) if partial_values is not None and
                           all(value is not None for value in partial_values) else None)
    other_return, settlement_blockers = _partial_return(joint or {}, quote, legs)
    blockers.extend(settlement_blockers)
    leg_means = [_number(leg.get('probability_mean'), low=0, high=1) for leg in legs]
    leg_pushes = [_number(leg.get('probability_push'), low=0, high=1) for leg in legs]
    if (leg_means and mean is not None and all(value is not None for value in leg_means) and
            mean > min(leg_means) + 1e-9) or (push is not None and all(value is not None for value in leg_pushes)
                                           and push > sum(leg_pushes) + 1e-9):
        blockers.append('JOINT_MASS_INCONSISTENT_WITH_LEGS')
    if ('INDEPEND' in str((joint or {}).get('method')).upper() and mean is not None
            and all(value is not None for value in leg_means)
            and abs(mean - math.prod(leg_means)) > 1e-6):
        blockers.append('JOINT_INDEPENDENCE_MISMATCH')
    min_ev = _number(policy.get('minimum_ev'), low=0)
    min_edge = _number(policy.get('minimum_edge'), low=0)
    minimum = minimum_acceptable_price(conservative, push, min_ev, min_edge, other_return=other_return)
    conservative_ev = conservative * quoted_decimal + other_return - 1 if None not in (conservative, quoted_decimal, other_return) else None
    break_even = (1 - other_return) / quoted_decimal if None not in (other_return, quoted_decimal) and quoted_decimal > 1 else None
    edge = conservative - break_even if None not in (conservative, break_even) else None
    if min_ev is None or min_edge is None:
        blockers.append('PRICE_POLICY_MISSING')
    if conservative_ev is not None and min_ev is not None and conservative_ev <= max(0.0, min_ev):
        blockers.append('NONPOSITIVE_CONSERVATIVE_EV')
    if edge is not None and min_edge is not None and edge < min_edge:
        blockers.append('EDGE_BELOW_POLICY')
    if minimum is not None and quoted_decimal is not None and quoted_decimal + 1e-9 < minimum:
        blockers.append('PRICE_MOVED')
    blockers = sorted(set(blockers))
    stake = fraction = 0.0
    exposure_id = None
    if not blockers and None not in (quoted_decimal, conservative, push):
        stake, fraction, alloc_blockers, exposure_id = _allocation(legs, identity_hash, product_type, quoted_decimal,
                                                                    conservative, push, policy, exposure, review, now,
                                                                    reservations)
        blockers = sorted(set(alloc_blockers))
    elif exposure is not None:
        exposure_id = exposure.get('snapshot_hash') if isinstance(exposure, dict) else None
    status = ('ACTIONABLE' if not blockers and stake > 0 else
              'STALE' if any('STALE' in b or 'STARTED' in b or 'EXPIRED' in b for b in blockers) else
              'PRICE_UNAVAILABLE' if 'PRICE_UNAVAILABLE' in blockers else
              'PRICE_MOVED' if 'PRICE_MOVED' in blockers else
              'JOINT_MODEL_UNAVAILABLE' if any(b.startswith('JOINT_') for b in blockers) else
              'UNVALIDATED' if 'PRODUCT_UNVALIDATED' in blockers else 'WAIT' if 'REVIEW_HOLD' in blockers else 'RESEARCH')
    return {
        'schema_version': 1, 'parlay_id': identity_hash, 'product_type': product_type,
        'status': status, 'decision_at': now.isoformat(), 'sportsbook': book or None,
        'legs': [{**leg_identity(leg), 'leg_hash': leg_hash(leg),
                  **{key: leg.get(key) for key in (
                      'team_ids', 'american_odds', 'start', 'quote_timestamp',
                      'analysis_timestamp', 'identity_verified', 'quote_verified',
                      'production_eligible', 'model_version', 'model_trained_through',
                      'model_available_at', 'calibration_version',
                      'calibration_available_at', 'policy_version',
                      'evidence_snapshot_id', 'evidence_frozen_at',
                      'critical_input_state', 'material_news_status',
                      'probability_semantics', 'probability_mean',
                      'probability_conservative', 'probability_push',
                      'probability_loss', 'sgp_component_id')}} for leg in legs],
        'ticket_hash': identity_hash, 'quote_id': quote.get('quote_id'),
        'ticket_binding': exact_binding,
        'provider_ticket_id': quote.get('provider_ticket_id'), 'quote_source': quote.get('source'),
        'quote_verification_state': quote.get('verification_state'),
        'quoted_american_odds': quoted_american, 'quoted_decimal_odds': quoted_decimal,
        'quoted_at': quote.get('quoted_at'), 'expires_at': quote.get('expires_at'),
        'probability_mean': mean, 'probability_conservative': conservative, 'probability_push': push,
        'probability_partial': partial_probability,
        'joint_partial_outcomes': (joint or {}).get('partial_outcomes'),
        'partial_payouts': quote.get('partial_payouts'),
        'settlement_rules_id': quote.get('settlement_rules_id'),
        'break_even_probability': break_even, 'probability_method': (joint or {}).get('method'),
        'joint_model_id': (joint or {}).get('model_id'), 'joint_model_version': (joint or {}).get('model_version'),
        'joint_correlation_method': (joint or {}).get('correlation_method'),
        'joint_dependence_methodology_id': (joint or {}).get('dependence_methodology_id'),
        'joint_component_dependence_method': (joint or {}).get('component_dependence_method'),
        'joint_shared_factor_method': (joint or {}).get('shared_factor_method'),
        'joint_final_calibration_id': (joint or {}).get('final_calibration_id'),
        'joint_final_calibration_version': (joint or {}).get('final_calibration_version'),
        'model_id': (joint or {}).get('model_id'), 'model_version': (joint or {}).get('model_version'),
        'joint_evidence_snapshot_id': (joint or {}).get('evidence_snapshot_id'),
        'joint_component_hashes': (joint or {}).get('component_hashes'),
        'sgp_component_count': sum(component.get('product_type') == 'SAME_GAME_PARLAY'
                                   for component in components),
        'sgp_component_validation_ids': [component.get('validation_id') for component in components
                                         if component.get('product_type') == 'SAME_GAME_PARLAY'],
        'joint_generated_at': (joint or {}).get('generated_at'),
        'joint_model_trained_through': (joint or {}).get('model_trained_through'),
        'joint_model_available_at': (joint or {}).get('model_available_at'),
        'joint_calibration_available_at': (joint or {}).get('calibration_available_at'),
        'joint_evidence_frozen_at': (joint or {}).get('evidence_frozen_at'),
        'probability_loss': _number((joint or {}).get('probability_loss'), low=0, high=1),
        'calibration_version': (joint or {}).get('calibration_version'),
        'validation_id': policy.get('validation_id'), 'validation_state': policy.get('validation_state', 'UNVALIDATED'),
        'policy_id': policy.get('policy_id'), 'policy_version': policy.get('policy_version'),
        'conservative_edge': edge,
        'conservative_ev': conservative_ev, 'minimum_acceptable_decimal': minimum,
        'minimum_acceptable_american': american_price(minimum),
        'dependence_status': dependency or ('MATERIAL_DEPENDENCE' if product_type == 'SAME_GAME_PARLAY' else 'UNKNOWN'),
        'overlap_leg_hashes': sorted({exposure_leg_key(leg) for leg in legs}),
        'exposure_snapshot_id': exposure_id, 'recommended_stake': stake, 'recommended_fraction': fraction,
        'review_status': review.get('status'), 'authorization_status': authorization.get('status'),
        'production_eligible': status == 'ACTIONABLE', 'automated_bet_placement': False,
        'blockers': blockers,
    }


def optimize_portfolio(candidates: list[dict], exposure: dict | None, *,
                       now: datetime | None = None, max_tickets: int = 5) -> dict:
    """Deterministically allocate ticket-level EV against shared portfolio limits.

    Each candidate is a keyword-argument mapping for ``evaluate_ticket``.
    Reservations are local to this proposal batch; actual commitments still
    require an owner-maintained exposure-ledger event.
    """
    now = now or datetime.now(timezone.utc)
    if max_tickets < 0:
        raise ValueError('Invalid candidate limit')
    previews = [(evaluate_ticket(**{**candidate, 'exposure': exposure, 'now': now}), candidate)
                for candidate in candidates]
    previews.sort(key=lambda pair: (-(pair[0]['conservative_ev'] if pair[0]['conservative_ev'] is not None else -math.inf),
                                    pair[0]['parlay_id'], str(pair[0].get('quote_id'))))
    reservations: dict[str, float] = {}
    output = []
    for _, candidate in previews[:max_tickets]:
        result = evaluate_ticket(**{**candidate, 'exposure': exposure, 'now': now,
                                    'reservations': reservations})
        output.append(result)
        if result['production_eligible']:
            legs = _flatten_components(candidate['components'])
            keys = _allocation_keys(legs, result['parlay_id'], result['product_type'],
                                    candidate['policy'], exposure)
            for key in keys or {}:
                reservations[key] = reservations.get(key, 0.0) + result['recommended_fraction']
    return {'records': output, 'diagnostics': {'candidate_combinations': len(candidates),
             'evaluated': len(output), 'truncated': max(0, len(candidates) - len(output)),
             'actionable': sum(r['production_eligible'] for r in output),
             'blockers': dict(sorted(__import__('collections').Counter(
                 b for r in output for b in r['blockers']).items()))}}


def settle_ticket(leg_outcomes: list[str], decimal_odds: Any, stake: Any, *,
                  push_rule: str = 'VOID_LEG', void_rule: str = 'VOID_LEG',
                  reduced_decimal_odds: Any = None) -> dict:
    """Grade using captured sportsbook rules; unknown rules require review."""
    allowed = {'WIN', 'LOSS', 'PUSH', 'VOID'}
    if not leg_outcomes or any(o not in allowed for o in leg_outcomes):
        return {'status': 'NEEDS_REVIEW', 'net_return': None}
    if 'LOSS' in leg_outcomes:
        return {'status': 'LOSS', 'net_return': -float(stake)} if _number(stake, low=0) is not None else {'status': 'NEEDS_REVIEW', 'net_return': None}
    if ('PUSH' in leg_outcomes and push_rule != 'VOID_LEG') or ('VOID' in leg_outcomes and void_rule != 'VOID_LEG'):
        return {'status': 'NEEDS_REVIEW', 'net_return': None}
    if all(o in {'PUSH', 'VOID'} for o in leg_outcomes):
        return {'status': 'VOID', 'net_return': 0.0}
    price = _number(reduced_decimal_odds if any(o in {'PUSH', 'VOID'} for o in leg_outcomes) else decimal_odds)
    amount = _number(stake, low=0)
    if price is None or price <= 1 or amount is None:
        return {'status': 'NEEDS_REVIEW', 'net_return': None}
    return {'status': 'WIN', 'net_return': amount * (price - 1)}
