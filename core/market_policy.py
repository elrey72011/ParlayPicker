"""Non-configurable production market boundary. Moneyline remains context only."""
from __future__ import annotations

MARKET_POLICY_VERSION = "spread-total-only-v1"
PRODUCTION_MARKETS = frozenset({"spread_home", "spread_away", "total_over", "total_under"})
MONEYLINE_MARKETS = frozenset({"moneyline", "moneyline_home", "moneyline_away", "h2h", "h2h_home", "h2h_away", "ml"})


def production_market(value: object) -> bool:
    # Alternate lines must use a normal side/direction plus independently verified
    # exact quote metadata; a market name alone never certifies an alternate.
    return str(value).strip().casefold() in PRODUCTION_MARKETS


def moneyline_context_only(value: object) -> bool:
    return str(value).strip().casefold() in MONEYLINE_MARKETS
