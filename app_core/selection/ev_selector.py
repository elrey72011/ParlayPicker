"""Market-anchored EV selector with an explicit No Bet (roadmap P1.1 and P1.2).

For each game it returns exactly one decision, BET or NO_BET, always with a reason:

1. **Fair market probability.** A two-way multiplicative de-vig of the offered price
   and the opposing price for the same line. Without the opposing price there is
   no fair market, and the candidate is unusable (``MARKET_PRICE_UNPAIRED``). A
   vig-inclusive price is never treated as fair.
2. **Anchored probability.** The market, nudged by the *independent* model
   (``ml_probability``) on the log-odds scale with weight ``model_weight``, then
   capped at ``± max_adjustment`` of the market. ``calibrated_probability`` is not
   used, because it already contains the market. With no independent model the
   anchored probability equals the market, so the edge is 0.
3. **Edge and EV.** Edge is anchored minus market; EV is anchored × decimal − 1
   at the offered price.
4. **Decision.** Take the highest-EV usable candidate per game. Bet only if its
   edge ≥ ``min_edge``, its EV > ``min_ev``, *and* its league × market is in
   ``qualified``. ``qualified`` is empty until the owner sets thresholds (D1) and
   a market passes them. Until then every game is NO_BET with the reason it
   would or would not otherwise qualify.

This module is pure: it reads a candidate table and returns a decision table. It
does not change picks, stakes or publication.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import pandas as pd

VERSION = 'ev-selector-v1'


@dataclass(frozen=True)
class Policy:
    model_weight: float = 0.25      # share of the model's log-odds difference from the market
    max_adjustment: float = 0.05    # the anchored probability stays within ±5 points of the market
    min_edge: float = 0.02
    min_ev: float = 0.0
    version: str = VERSION


FAMILY = {'spread_home': 'SPREAD', 'spread_away': 'SPREAD', 'total_over': 'TOTAL', 'total_under': 'TOTAL',
          'h2h_home': 'MONEYLINE', 'h2h_away': 'MONEYLINE', 'moneyline_home': 'MONEYLINE', 'moneyline_away': 'MONEYLINE'}


def american_to_decimal(odds):
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return None
    if isinstance(odds, bool) or not math.isfinite(o) or abs(o) < 100:
        return None
    return 1 + o / 100 if o > 0 else 1 + 100 / -o


def fair_probability(odds, opposing):
    """Two-way multiplicative de-vig for the selected side, or None without both prices."""
    d, q = american_to_decimal(odds), american_to_decimal(opposing)
    if d is None or q is None:
        return None
    p, r = 1 / d, 1 / q
    return p / (p + r)


def _logit(p):
    return math.log(p / (1 - p))


def anchored_probability(market, model, policy: Policy):
    """Market nudged toward an independent model, bounded by ``max_adjustment``."""
    if model is None or not 0 < model < 1:
        return market
    nudged = 1 / (1 + math.exp(-(_logit(market) + policy.model_weight * (_logit(model) - _logit(market)))))
    return min(max(nudged, market - policy.max_adjustment), market + policy.max_adjustment)


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def evaluate(row, policy: Policy):
    """Market, anchored probability, edge and EV for one candidate, or a reason it is unusable."""
    decimal = american_to_decimal(row.get('odds_american'))
    if decimal is None:
        return dict(usable=False, reason='PRICE_MISSING')
    market = fair_probability(row.get('odds_american'), row.get('opposing_odds_american'))
    if market is None:
        return dict(usable=False, reason='MARKET_PRICE_UNPAIRED', decimal_odds=decimal)
    model = _number(row.get('ml_probability'))
    model = model if model is not None and 0 < model < 1 else None
    anchored = anchored_probability(market, model, policy)
    return dict(usable=True, reason='', decimal_odds=decimal, market_probability=market,
                model_probability=model, anchored_probability=anchored, edge=anchored - market,
                ev=anchored * decimal - 1, has_model=model is not None)


def select(candidates: pd.DataFrame, *, qualified=frozenset(), policy: Policy = Policy(), current=None):
    """One decision per game (``matchup_id``), with the reason and the chosen candidate.

    ``current`` may be today's selection (a frame with ``matchup_id`` and
    ``market_type``). Each decision then records whether the shadow selector
    agrees with it.
    """
    if candidates is None or candidates.empty or 'matchup_id' not in candidates.columns:
        return pd.DataFrame()
    chosen_now = {}
    if current is not None and not current.empty and {'matchup_id', 'market_type'} <= set(current.columns):
        chosen_now = {str(r['matchup_id']): str(r['market_type']) for r in current.to_dict('records')}
    decisions = []
    for game, group in candidates.groupby(candidates['matchup_id'].astype(str), sort=True):
        rows = group.to_dict('records')
        evaluated = [(r, evaluate(r, policy)) for r in rows]
        usable = [(r, e) for r, e in evaluated if e['usable']]
        base = dict(matchup_id=game, league=str(rows[0].get('league', rows[0].get('League', ''))).upper(),
                    home_team=rows[0].get('home_team'), away_team=rows[0].get('away_team'),
                    candidates=len(rows), usable_candidates=len(usable), policy=policy.version,
                    current_market_type=chosen_now.get(game))
        if not usable:
            reasons = sorted({e['reason'] for _, e in evaluated})
            decisions.append(dict(base, decision='NO_BET', reason='NO_USABLE_CANDIDATE:' + '|'.join(reasons)))
            continue
        row, best = max(usable, key=lambda item: item[1]['ev'])
        market_type = str(row.get('market_type', '')).lower()
        family = FAMILY.get(market_type, 'UNKNOWN')
        line = row.get('total_line') if family == 'TOTAL' else row.get('spread_line')
        if not best['has_model']:
            reason = 'NO_INDEPENDENT_MODEL'
        elif best['edge'] < policy.min_edge:
            reason = 'EDGE_BELOW_THRESHOLD'
        elif best['ev'] <= policy.min_ev:
            reason = 'EV_NOT_POSITIVE'
        elif (base['league'], family) not in qualified:
            reason = 'MARKET_NOT_QUALIFIED'
        else:
            reason = 'QUALIFIED_EDGE'
        decisions.append(dict(
            base, decision='BET' if reason == 'QUALIFIED_EDGE' else 'NO_BET', reason=reason,
            market_type=market_type, family=family, line=_number(line), odds_american=_number(row.get('odds_american')),
            market_probability=round(best['market_probability'], 4),
            model_probability=None if best['model_probability'] is None else round(best['model_probability'], 4),
            anchored_probability=round(best['anchored_probability'], 4), edge=round(best['edge'], 4),
            ev=round(best['ev'], 4),
            agrees_with_current=None if base['current_market_type'] is None else base['current_market_type'] == market_type))
    return pd.DataFrame(decisions)


def summary(decisions: pd.DataFrame, policy: Policy = Policy()):
    """Counts for diagnostics: decisions, reasons and agreement with the current selector."""
    if decisions is None or decisions.empty:
        return dict(games=0, policy=asdict(policy))
    agree = decisions['agrees_with_current'].dropna() if 'agrees_with_current' in decisions else pd.Series(dtype=bool)
    return dict(games=int(len(decisions)), bet=int((decisions['decision'] == 'BET').sum()),
                reasons={str(k): int(v) for k, v in decisions['reason'].value_counts().items()},
                compared_with_current=int(len(agree)), agrees_with_current=int(agree.astype(bool).sum()),
                policy=asdict(policy))
