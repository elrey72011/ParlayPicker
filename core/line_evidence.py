"""Shared line rejection checks for captured and historical candidates."""
import math


def line_rejected(row):
    def text(key):
        return str(row.get(key, "")).strip().lower()

    if text("final_line_rejected") in {"true", "1"}:
        return True
    if any(text(key).startswith("rejected") for key in
           ("odds_source", "line_source", "market_line_source")):
        return True
    if any(text(key) in {"false", "0"} for key in
           ("line_consistency_flag", "line_event_identity_match_flag")):
        return True
    pick = text("best_pick")
    if "line unresolved" in pick or "no line" in pick:
        return True
    market = text("market_type")
    if market.startswith(("spread", "total")):
        try:
            line = float(row.get("total_line" if market.startswith("total") else "spread_line"))
            return not math.isfinite(line)
        except (TypeError, ValueError):
            return True
    return False
