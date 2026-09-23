"""Deterministic public research funnel for the three independent parlay products.

The saved board has no ticket-quote or product-validation feed. This module
therefore publishes transparent, zero-stake research records. The core engine
accepts actual evidence in the private owner flow when it becomes available.
"""

from collections import Counter, defaultdict
from itertools import combinations, islice
from math import comb

from core.true_parlay_engine import evaluate_ticket, normalize_leg, leg_hash


POLICY_VERSION = 'true-parlay-research-v1'
PRODUCT_LIMIT = 20


def _public_leg(row):
    leg = normalize_leg(row)
    # A display label groups research combinations; it does not verify identity.
    leg['game_id'] = leg['game_id'] or row.get('game')
    return leg


def _identity(row):
    return (str(row.get('sport') or ''), str(row.get('game') or ''),
            str(row.get('market') or ''), str(row.get('pick') or ''),
            str(row.get('wager_contract', {}).get('line') if isinstance(row.get('wager_contract'), dict) else ''),
            str(row.get('quote_source') or ''))


def build_product_board(games, props, now, *, limit=PRODUCT_LIMIT):
    """Return saved candidate records and a reconciled stage/blocker funnel."""
    if not isinstance(games, dict) or not 0 < limit <= PRODUCT_LIMIT:
        raise ValueError('Invalid parlay candidate input/limit')
    primary = list(games.get('overall') or [])
    all_rows = list(primary) + list(games.get('sides') or []) + list(games.get('totals') or [])
    # Player props can be displayed as research only. They currently lack the
    # exact game/line production contract needed by the common leg gate.
    all_rows += list(props or [])
    unique = {}
    for row in all_rows:
        unique.setdefault(_identity(row), row)
    by_game = defaultdict(list)
    for row in unique.values():
        by_game[(row.get('sport'), row.get('game'))].append(row)
    primary = list({_identity(row): row for row in primary}.values())
    primary.sort(key=_identity)
    records = []
    potential = Counter()
    truncated = Counter()

    def add(product, rows_or_components):
        components = [_public_leg(r) if isinstance(r, dict) and 'product_type' not in r else r
                      for r in rows_or_components]
        legs = [leg for c in components for leg in (c['legs'] if c.get('product_type') == 'SAME_GAME_PARLAY' else [c])]
        books = {leg.get('sportsbook') for leg in legs}
        book = next(iter(books)) if len(books) == 1 else None
        record = evaluate_ticket(product, components, sportsbook=book, now=now)
        # Keep only the public, exact ticket facts and machine-readable gates.
        records.append(record)
        return record

    # Standard: enumerate distinct-game combinations; preserve the raw count
    # separately from the bounded number actually evaluated and published.
    n = len(primary)
    potential['STANDARD_PARLAY'] = sum(comb(n, size) for size in range(2, min(4, n) + 1))
    standard = 0
    for size in range(2, min(4, n) + 1):
        for rows in islice(combinations(primary, size), max(0, limit - standard)):
            if len({(r.get('sport'), r.get('game')) for r in rows}) != size:
                continue
            add('STANDARD_PARLAY', rows)
            standard += 1
    if potential['STANDARD_PARLAY'] > standard:
        truncated['STANDARD_PARLAY'] = potential['STANDARD_PARLAY'] - standard

    # Same Game: group exact side/total/prop selections. No independence
    # shortcut or multiplied single-leg price is ever used for authorization.
    sgp_records = []
    for key in sorted(by_game, key=str):
        group = sorted(by_game[key], key=_identity)
        for size in range(2, min(4, len(group)) + 1):
            potential['SAME_GAME_PARLAY'] += comb(len(group), size)
            for rows in islice(combinations(group, size), max(0, limit - len(sgp_records))):
                sgp_records.append(add('SAME_GAME_PARLAY', rows))
    if potential['SAME_GAME_PARLAY'] > len(sgp_records):
        truncated['SAME_GAME_PARLAY'] = potential['SAME_GAME_PARLAY'] - len(sgp_records)

    # Cross Game admits 2-6 independent-game legs, with or without a validated
    # SGP block. Reserve space for SGP examples so a large raw pool cannot hide
    # them. It never unpacks an SGP for joint-probability authorization.
    cross = 0
    raw_quota = max(1, limit // 2) if sgp_records else limit
    potential['CROSS_GAME_PARLAY'] += sum(comb(n, size) for size in range(2, min(6, n) + 1))
    for size in range(2, min(6, n) + 1):
        for rows in combinations(primary, size):
            if cross >= raw_quota:
                break
            if len({(r.get('sport'), r.get('game')) for r in rows}) == size:
                add('CROSS_GAME_PARLAY', rows)
                cross += 1
        if cross >= raw_quota:
            break
    for sgp in sgp_records:
        sgp_game = sgp['legs'][0].get('game_id') if sgp['legs'] else None
        others = [row for row in primary if row.get('game') != sgp_game]
        remaining = min(6 - len(sgp['legs']), len(others))
        potential['CROSS_GAME_PARLAY'] += sum(comb(len(others), size) for size in range(1, remaining + 1))
        for size in range(1, remaining + 1):
            for rows in combinations(others, size):
                if cross >= limit:
                    break
                add('CROSS_GAME_PARLAY', [sgp, *rows])
                cross += 1
            if cross >= limit:
                break
    if potential['CROSS_GAME_PARLAY'] > cross:
        truncated['CROSS_GAME_PARLAY'] = potential['CROSS_GAME_PARLAY'] - cross

    # Stable product/identity order is part of the immutable package contract.
    records.sort(key=lambda r: (r['product_type'], r['parlay_id']))
    blockers = Counter(reason for record in records for reason in record['blockers'])
    counts = Counter({'candidate_combinations': len(records), 'price_verified': 0,
                      'joint_probability_verified': 0, 'positive_conservative_ev': 0,
                      'production_validated': 0, 'actionable_now': 0})
    for record in records:
        if 'PRICE_UNAVAILABLE' in record['blockers']:
            continue
        counts['price_verified'] += 1
        if any(reason.startswith('JOINT_') for reason in record['blockers']):
            continue
        counts['joint_probability_verified'] += 1
        if record['conservative_ev'] is None or record['conservative_ev'] <= 0:
            continue
        counts['positive_conservative_ev'] += 1
        if 'PRODUCT_UNVALIDATED' in record['blockers']:
            continue
        counts['production_validated'] += 1
        if record['status'] == 'ACTIONABLE':
            counts['actionable_now'] += 1
    funnel = {'schema_version': 1, 'counts': dict(counts), 'blockers': dict(sorted(blockers.items())),
              'potential_combinations': dict(sorted(potential.items())),
              'truncated': dict(sorted(truncated.items())),
              'evaluated_by_product': dict(sorted(Counter(r['product_type'] for r in records).items()))}
    return records, funnel
