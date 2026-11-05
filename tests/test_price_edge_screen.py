"""Cross-book price screen (roadmap C5). SYNTHETIC rows; no network."""
from app_core import price_edge_screen as screen


def _decimal(american):
    return 1 + (american / 100 if american > 0 else 100 / abs(american))


def _quote(event_id, book, selection, line, american, market="SPREAD"):
    return {"event_id": event_id, "market_family": market, "selection": selection, "line": line,
            "american_odds": american, "decimal_odds": _decimal(american), "sportsbook": book,
            "quote_verified": True}


EVENT = {"event_id": "e1", "sport": "NBA", "home_team": "Home", "away_team": "Away"}


def _market():
    # Three books at -110/-110 (fair 50/50); one soft book offers Home +105 at the same line.
    quotes = []
    for book in ("a", "b", "c"):
        quotes += [_quote("e1", book, "Home", -3.5, -110), _quote("e1", book, "Away", 3.5, -110)]
    quotes += [_quote("e1", "soft", "Home", -3.5, 105), _quote("e1", "soft", "Away", 3.5, -130)]
    return quotes


def test_consensus_flags_the_soft_price_and_grades_clv_and_outcome():
    closes = [{"event_id": "e1", "market_family": "SPREAD", "selection": "Home", "sportsbook": "soft",
               "line": -3.5, "decimal_odds": _decimal(-115), "close_verified": True}]
    scores = [{"event_id": "e1", "home_score": 110, "away_score": 100}]
    rows = screen.screen([EVENT], _market(), closes, scores)
    home = next(r for r in rows if r["selection"] == "Home")
    assert home["sportsbook"] == "soft" and home["books"] == 4
    assert 0.0 < home["ev"] < 0.05                  # fair 0.5 (median of 4) at +105: EV +2.5%
    assert home["price_clv"] > 0                     # +105 beat a -115 close
    assert home["outcome"] == "WIN"                  # 110 - 100 - 3.5 > 0
    summary = screen.summarize(rows)["0.00"]
    assert summary["picks"] >= 1 and summary["beat_close_share"] == 1.0 and summary["wins"] >= 1


def test_too_few_books_or_unpaired_sides_are_skipped():
    two_books = [q for q in _market() if q["sportsbook"] in ("a", "b")]
    assert screen.screen([EVENT], two_books, [], []) == []
    one_sided = [q for q in _market() if q["selection"] == "Home"]
    assert screen.screen([EVENT], one_sided, [], []) == []


def test_totals_share_one_key_and_a_different_line_is_not_comparable():
    quotes = []
    for book in ("a", "b", "c"):
        quotes += [_quote("e1", book, "Over", 220.5, -110, "TOTAL"), _quote("e1", book, "Under", 220.5, -110, "TOTAL")]
    closes = [{"event_id": "e1", "market_family": "TOTAL", "selection": "Over", "sportsbook": "a",
               "line": 221.5, "decimal_odds": 1.9, "close_verified": True}]
    rows = screen.screen([EVENT], quotes, closes, [])
    assert {r["selection"] for r in rows} == {"Over", "Under"}
    assert all("price_clv" not in r for r in rows)   # close moved to another line
