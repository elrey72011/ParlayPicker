"""Exact live candidate binding for research MLB model inference."""
import json
from app_core.mlb_history import timestamp
from app_core.mlb_spread_total_model import finite, receipt_features
from app_core.public_quote_policy import canonical_book_label
from core.wager_decisions import decimal_price


def verify(row, receipt, *, now):
    p, _ = receipt_features(receipt)
    q = p["quote"]
    ids = row.get("provider_ids")
    if isinstance(ids, str):
        ids = json.loads(ids)
    if not isinstance(ids, dict) or str(ids.get("mlb")) != str(p["provider_event_id"]) or str(ids.get("odds_api")) != str(q["provider_event_id"]):
        raise ValueError("challenger provider mapping mismatch")
    if q.get("provider_namespace") != "odds_api":
        raise ValueError("challenger quote namespace mismatch")
    # Explicit candidate provider identity may use either verified namespace.
    ns, event = row.get("provider_namespace"), row.get("provider_event_id")
    supplied_ns = isinstance(ns, str) and ns not in ("", "nan")
    supplied_event = event is not None and str(event) not in ("", "nan", "<NA>")
    if supplied_ns != supplied_event:
        raise ValueError("challenger partial provider identity")
    if supplied_ns:
        if ns not in ids or str(event) != str(ids[ns]):
            raise ValueError("challenger candidate event mismatch")
    for key in ("home_team_id", "away_team_id"):
        if str(row.get(key)) != str(p[key]):
            raise ValueError("challenger team mismatch")
    def first(*keys):
        for key in keys:
            v = row.get(key)
            if v is not None and str(v) not in ("", "nan", "NaT", "<NA>"):
                return v
        raise ValueError("missing candidate quote fact")
    source_start = timestamp(q.get("source_game_start_utc", p["game_start_utc"]))
    if abs((source_start - timestamp(p["game_start_utc"])).total_seconds()) > 600:
        raise ValueError("challenger source start mismatch")
    if timestamp(first("game_start_utc", "start", "commence_time_raw", "commence_time")) != source_start:
        raise ValueError("challenger start mismatch")
    if row.get("market_type") != q["market_type"]:
        raise ValueError("challenger market mismatch")
    line = first("line", "spread_line" if q["market_type"].startswith("spread") else "total_line")
    if finite(line) != finite(q["line"]):
        raise ValueError("challenger line mismatch")
    book = canonical_book_label(first("quote_bookmaker", "sportsbook", "odds_source"))
    if book not in {"Novig", "DraftKings", "FanDuel", "BetMGM"}:
        # The expansion layer labels the feed, not always the selected book.
        # Reuse its exact price/line/provider matcher; ambiguity stays blocked.
        from app_core.prediction_evidence import ensure_authoritative_quote_binding
        bound = ensure_authoritative_quote_binding(row)
        if bound.get("quote_binding_verified") is not True:
            raise ValueError("challenger quote unverified")
        book = canonical_book_label(bound.get("quote_bookmaker"))
    if book != canonical_book_label(q["sportsbook"]):
        raise ValueError("challenger book mismatch")
    price = decimal_price(first("odds_american", "american_odds"))
    if price is None or abs(price - finite(q["decimal_odds"])) > 1e-9:
        raise ValueError("challenger price mismatch")
    if not 0 <= (now - timestamp(q["observed_at"])).total_seconds() <= 1800:
        raise ValueError("challenger stale quote")
    if not timestamp(p["captured_at"]) <= now < min(source_start, timestamp(p["game_start_utc"])):
        raise ValueError("challenger not pregame")
