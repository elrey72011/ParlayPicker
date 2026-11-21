"""Does anything on this board carry a live price? (owner 2026-11-20, "I ran out of credits")

On 2026-11-20 the Odds API answered every sport with HTTP 401: the month's credits were used up.
The Refresh carried on from the uploaded rows, priced every market at a -110 placeholder and said
nothing on the board. ``outage`` names that case so the board can say it plainly: no candidate
has a live price, so nothing on the page is a pick.

Read only by the homepage board. Never used for a rating, a selection or a stake.
"""
from __future__ import annotations

PLACEHOLDER_SOURCES = ("fallback_novig", "espn_draftkings_fallback")   # as ``streamlit_pipeline._is_nonproduction_odds_source``
OUT_OF_CREDITS = "OUT_OF_USAGE_CREDITS"
_NO_PICKS = ("Every price below is a -110 placeholder, not a quote: nothing on this page is a pick, "
             "and none can be staked.")


def _source(value) -> str:
    if value is None or value != value:      # None or NaN
        return ""
    return str(value).strip().lower()


def is_placeholder(value) -> bool:
    source = _source(value)
    return any(marker in source for marker in PLACEHOLDER_SOURCES)


def outage(candidates, refusal=None) -> dict | None:
    """None while any candidate has a live price; else what to tell the owner.

    A board with no recorded price source at all says nothing: only placeholder prices with no
    live one beside them are an outage. ``refusal`` is ``odds_api.last_refusal`` (the provider's
    latest 401 or 403: status, its own error code, its credits-remaining header).
    """
    if candidates is None or getattr(candidates, "empty", True) or "odds_source" not in candidates.columns:
        return None
    sources = [_source(value) for value in candidates["odds_source"]]
    if not any(is_placeholder(s) for s in sources) or any(s and not is_placeholder(s) for s in sources):
        return None
    refusal = dict(refusal or {})
    status, code, remaining = refusal.get("http_status"), refusal.get("code"), refusal.get("remaining")
    said = f"HTTP {status}" + (f", {code}" if code else "") if status else None
    if code == OUT_OF_CREDITS or (status in (401, 403) and str(remaining).strip() == "0"):
        why = (f"The Odds API refused this Refresh ({said}): this month's credits are used up. "
               "Prices come back with more credits (a plan upgrade counts at once, same key) or when the "
               "allowance resets on the first of the month.")
    elif code:
        why = f"The Odds API refused this Refresh ({said}). Check the Odds API key in the app's secrets."
    elif status in (401, 403):
        why = (f"The Odds API refused this Refresh ({said}). The usual cause is the month's credits running "
               "out; a key that is wrong gives the same answer.")
    else:
        why = "The live odds feed returned no prices for this Refresh."
    return {"games": int(candidates["matchup_id"].nunique()) if "matchup_id" in candidates.columns else None,
            "http_status": status, "code": code, "remaining": remaining,
            "message": "No live prices. " + why + " " + _NO_PICKS}
