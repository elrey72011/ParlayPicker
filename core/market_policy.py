"""Non-configurable production market boundary. Moneyline remains context only."""
from __future__ import annotations

MARKET_POLICY_VERSION = "spread-total-only-v1"
PRODUCTION_MARKETS = frozenset({"spread_home", "spread_away", "total_over", "total_under"})
MONEYLINE_MARKETS = frozenset({"moneyline", "moneyline_home", "moneyline_away", "h2h", "h2h_home", "h2h_away", "ml"})
SPORT_MARKET_FAMILIES = {
    "NFL": ("SPREAD", "TOTAL"),
    "NCAAF": ("SPREAD", "TOTAL"),
    "NBA": ("SPREAD", "TOTAL"),
    "NCAAB": ("SPREAD", "TOTAL"),
    "MLB": ("RUN_LINE", "TOTAL"),
    "NHL": ("PUCK_LINE", "TOTAL"),
}


def production_market(value: object) -> bool:
    # Alternate lines must use a normal side/direction plus independently verified
    # exact quote metadata; a market name alone never certifies an alternate.
    return str(value).strip().casefold() in PRODUCTION_MARKETS


def moneyline_context_only(value: object) -> bool:
    return str(value).strip().casefold() in MONEYLINE_MARKETS


def sport_market_family(sport: object, market_type: object) -> str | None:
    """Map a priced side/total to its exact sport-specific validation family."""
    code = str(sport).strip().upper()
    market = str(market_type).strip().casefold()
    if code not in SPORT_MARKET_FAMILIES or market not in PRODUCTION_MARKETS:
        return None
    if market.startswith("total_"):
        return "TOTAL"
    return {"MLB": "RUN_LINE", "NHL": "PUCK_LINE"}.get(code, "SPREAD")
