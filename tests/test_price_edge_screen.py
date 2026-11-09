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


def test_all_bucket_grades_every_best_price_even_without_a_flag():
    quotes = []
    for book in ("a", "b", "c"):
        quotes += [_quote("e1", book, "Home", -3.5, -110), _quote("e1", book, "Away", 3.5, -110)]
    closes = [{"event_id": "e1", "market_family": "SPREAD", "selection": "Home", "sportsbook": "c",
               "line": -3.5, "decimal_odds": _decimal(-120), "close_verified": True}]
    # Equal prices: the tie-break takes the last book by name ("c").
    summary = screen.summarize(screen.screen([EVENT], quotes, closes, []))
    assert summary["0.00"]["picks"] == 0                 # -110 at a fair 50% is negative EV
    assert summary["all"]["picks"] == 2 and summary["all"]["mean_ev"] < 0
    assert summary["all"]["with_comparable_close"] == 1 and summary["all"]["beat_close_share"] == 1.0


def test_breakdown_by_book_and_by_quote_timing_with_standard_errors():
    event = dict(EVENT, scheduled_start="2026-11-10T00:00:00+00:00")
    quotes = [dict(q, quote_timestamp="2026-11-09T22:00:00+00:00") for q in _market()]     # 2 hours before
    closes = [{"event_id": "e1", "market_family": "SPREAD", "selection": "Home", "sportsbook": "soft",
               "line": -3.5, "decimal_odds": _decimal(-115), "close_verified": True}]
    rows = screen.screen([event], quotes, closes, [{"event_id": "e1", "home_score": 110, "away_score": 100}])
    assert {round(r["hours_before_start"], 3) for r in rows} == {2.0}
    out = screen.breakdown(rows)
    assert set(out["by_timing"]) == {"1-3 h"} and out["by_timing"]["1-3 h"]["picks"] == len(rows) == 2
    soft = out["by_book"]["soft"]
    assert soft["picks"] == 1 and soft["with_comparable_close"] == 1 and soft["beat_close_share"] == 1.0
    assert soft["wins"] == 1 and soft["price_clv_se"] is None and soft["roi_se"] is None   # one pick: no error bar
    assert sum(g["picks"] for g in out["by_book"].values()) == 2
    assert screen.timing_label(0.5) == "under 1 h" and screen.timing_label(30) == "24 h or more"
    assert screen.timing_label(None) is None and screen.timing_label(-1) is None           # quoted after the start
    mean, se = screen._mean_se([0.01, 0.03])
    assert mean == 0.02 and round(se, 4) == 0.01
    assert screen.breakdown(screen.screen([EVENT], _market(), [], []))["by_timing"] == {}  # no timestamps stored
