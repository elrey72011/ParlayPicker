"""Offline regressions for the two late #2377 diagnostic findings."""
from copy import deepcopy
from io import StringIO
import json

import pandas as pd
import pytest

from app_core import ncaaf_schedule as ns
from app_core.prediction_evidence import provider_quotes
from test_ncaaf_schedule_coverage import (event, game, inv, no_network, calendar, response, index)

NOW = "2026-10-03T17:00Z"


def incomplete(kind):
    value = event()
    if kind == "competitions":
        value["competitions"] = []
        value.pop("date")
        value.pop("status")
    elif kind == "kickoff":
        value.pop("date")
        value["competitions"][0].pop("date")
    elif kind == "status":
        value.pop("status")
    elif kind == "team_names":
        for t in value["competitions"][0]["competitors"]:
            t["team"].pop("displayName")
    elif kind == "team_ids":
        for t in value["competitions"][0]["competitors"]:
            t["team"].pop("id")
    return value


@pytest.mark.parametrize("kind", ["competitions", "kickoff", "status", "team_names", "team_ids"])
def test_valid_revision_fills_missing_facts_without_clearing_conflict_or_granting_authority(kind):
    batches = [("FBS", [incomplete(kind)]), ("FCS", [event()])]
    before = deepcopy(batches)
    result = ns.inventory_from_events(batches, "2026-10-03", "2026-10-03", complete=True)
    canonical = result["events"][0]
    assert canonical["home_team"] == "LSU Tigers"
    assert canonical["away_team"] == "McNeese Cowboys"
    assert canonical["home_team_id"] == "espn:ncaaf:99"
    assert canonical["away_team_id"] == "espn:ncaaf:88"
    assert canonical["home_aliases"] and canonical["away_aliases"]
    assert canonical["kickoff"] == "2026-10-03T23:45:00+00:00"
    assert canonical["schedule_status"] == "SCHEDULED"
    assert canonical["identity_conflict"] and result["status"] == "PARTIAL"
    assert canonical["divisions"] == ["FBS", "FCS"]
    assert len(result["events"]) == 1 and len(result["identity_events"]) == 2
    matched, state = ns.match_event(game(), result)
    assert matched is canonical and state == "KICKOFF_OR_IDENTITY_CONFLICT"
    report = ns.coverage(result, [game()], [game()], [game()], now=NOW)
    assert report["counts"] == dict(scheduled=1, matched=0, quoted=0, timestamped=0, ranked=0, qualified=0)
    assert report["rows"][0]["home_team"] == "LSU Tigers"
    assert "IDENTITY_UNRESOLVED" in report["rows"][0]["exclusion_reasons"]
    assert batches == before


def test_actual_seed_and_week_revisions_recover_facts_within_unchanged_budget(monkeypatch):
    seed = incomplete("competitions")
    calls = []
    def get(url, params, timeout):
        calls.append((url, deepcopy(params)))
        assert timeout == (3, 5)
        if url == ns.SCOREBOARD:
            return response({"events": [seed] if params.get("dates") else [event()], "leagues": calendar()})
        return response(index(["1"]))
    monkeypatch.setattr(ns.requests, "get", get)
    result = ns.fetch_schedule("2026-10-03", "2026-10-03")
    assert len(calls) == 5 <= ns.MAX_REQUESTS == 6
    assert result["events"][0]["away_team_id"] == "espn:ncaaf:88"
    assert result["events"][0]["kickoff"] and result["status"] == "PARTIAL"
    assert result["events"][0]["identity_conflict"]
    assert seed["competitions"] == []


@pytest.mark.parametrize("conflict", ["teams", "kickoff", "status"])
def test_known_revision_disagreements_remain_visible_and_do_not_overwrite_first_facts(conflict):
    first, later = event(), event()
    if conflict == "teams":
        later["competitions"][0]["competitors"][0]["team"] = {"id": "55", "displayName": "Other School"}
    elif conflict == "kickoff":
        later["date"] = later["competitions"][0]["date"] = "2026-10-04T01:00Z"
    else:
        later["status"]["type"]["name"] = "STATUS_CANCELED"
    result = ns.inventory_from_events([("FBS", [first, later])], "2026-10-03", "2026-10-03", complete=True)
    canonical = result["events"][0]
    assert canonical["home_team_id"] == "espn:ncaaf:99"
    assert canonical["home_team"] == "LSU Tigers"
    assert "other school" not in canonical["home_aliases"]
    assert canonical["kickoff"] == "2026-10-03T23:45:00+00:00"
    assert canonical["schedule_status"] == "SCHEDULED"
    assert canonical["identity_conflict"] and not result["complete"]
    assert ns.match_event(game(), result)[1] == "KICKOFF_OR_IDENTITY_CONFLICT"
    if conflict == "kickoff":
        assert len(canonical["kickoff_revisions"]) == 2


def test_incomplete_later_revision_cannot_erase_healthy_facts():
    result = ns.inventory_from_events([("FBS", [event(), incomplete("competitions")])],
                                      "2026-10-03", "2026-10-03", complete=True)
    canonical = result["events"][0]
    assert canonical["home_team_id"] == "espn:ncaaf:99" and canonical["away_aliases"]
    assert canonical["kickoff"] and canonical["schedule_status"] == "SCHEDULED"
    assert canonical["identity_conflict"] and not result["complete"]


@pytest.mark.parametrize("price", [None, "", "bad", float("nan"), float("inf"), -float("inf"),
    "NaN", "Infinity", "1e999", True, False, 0, -99, 99, 1.91, 10**400])
def test_timestamped_unpriced_or_invalid_price_receipts_do_not_count_as_quotes(price):
    q = {"provider_namespace": "odds_api", "provider_event_id": "primary", "book": "novig",
         "price": price, "recorded_at": "2026-10-03T16:40Z"}
    row = {**game(), "provider_quotes": [q]}
    before = deepcopy(row)
    report = ns.coverage(inv(), [row], [row], [row], now=NOW)
    output = report["rows"][0]
    assert report["counts"] == dict(scheduled=1, matched=1, quoted=0, timestamped=0, ranked=1, qualified=0)
    assert output["quote_count"] == output["timestamped_quote_count"] == 0
    assert output["quote_receipt_count"] == output["invalid_price_quote_count"] == 1
    assert output["quote_coverage"] == "UNAVAILABLE" and "QUOTE_UNAVAILABLE" in output["exclusion_reasons"]
    assert json.loads(output["provider_identities"])  # Invalid receipts still retain identity evidence.
    assert row["provider_quotes"][0] is q and row["bookmakers"] == before["bookmakers"]


@pytest.mark.parametrize("price", [-110, 100, -100, "+125", -125.5])
@pytest.mark.parametrize("recorded_at,expected", [("2026-10-03T16:40Z", 1), (None, 0),
    ("2026-10-03T18:00Z", 0), ("2026-10-03T23:45Z", 0)])
def test_real_american_prices_and_provider_chronology_remain_separate(price, recorded_at, expected):
    q = {"price": price, "recorded_at": recorded_at, "observed_at": "2026-10-03T16:45Z"}
    row = {**game(), "provider_quotes": [q]}
    report = ns.coverage(inv(), [row], [row], now=NOW)
    output = report["rows"][0]
    assert report["counts"]["quoted"] == output["quote_count"] == 1
    assert report["counts"]["timestamped"] == output["timestamped_quote_count"] == expected
    assert output["quote_receipt_count"] == 1 and output["invalid_price_quote_count"] == 0
    assert report["counts"]["qualified"] == 0
    assert q["recorded_at"] == recorded_at


def test_real_quote_producer_preserves_unpriced_receipts_but_coverage_excludes_them():
    row = game()
    for outcome in row["bookmakers"][0]["markets"][0]["outcomes"]:
        outcome.pop("price")
    receipts = provider_quotes(row)
    assert len(json.loads(receipts)) == 2 and all(q["recorded_at"] for q in json.loads(receipts))
    report = ns.coverage(inv(), [{**row, "provider_quotes": receipts}], now=NOW)
    assert report["counts"]["matched"] == 1 and report["counts"]["quoted"] == report["counts"]["timestamped"] == 0
    assert report["rows"][0]["quote_receipt_count"] == report["rows"][0]["invalid_price_quote_count"] == 2


def test_mixed_inventory_counts_reconcile_with_ui_and_csv_without_manufactured_authority(monkeypatch):
    repaired = incomplete("competitions")
    repaired["id"] = "3"
    inventory = ns.inventory_from_events([("FBS", [event(), event("2", away="Other Team"), repaired]),
        ("FCS", [event("3", away="Third Team")])], "2026-10-03", "2026-10-03", complete=True)
    quotes = [{"book": "priced", "price": -110, "recorded_at": "2026-10-03T16:40Z"},
              {"book": "undated", "price": 120, "recorded_at": None},
              {"book": "missing", "recorded_at": "2026-10-03T16:40Z"},
              {"book": "infinite", "price": float("inf"), "recorded_at": "2026-10-03T16:40Z"}]
    first = {**game("espn-1"), "provider_quotes": quotes}
    second = {**game("espn-2", away="Other Team"), "provider_quotes": [{"price": None, "recorded_at": NOW}]}
    third = {**game("espn-3", away="Third Team"), "provider_quotes": [{"price": -110, "recorded_at": NOW}]}
    report = ns.coverage(inventory, [first, second, third], [first, second], [first, second], now=NOW)
    assert report["counts"] == dict(scheduled=3, matched=2, quoted=1, timestamped=1, ranked=2, qualified=0)
    assert [(r["quote_receipt_count"], r["quote_count"], r["invalid_price_quote_count"], r["timestamped_quote_count"])
            for r in report["rows"]] == [(4, 2, 2, 1), (1, 0, 1, 0), (0, 0, 0, 0)]
    for row in report["rows"]:
        assert row["quote_count"] + row["invalid_price_quote_count"] == row["quote_receipt_count"]
        assert row["timestamped_quote_count"] <= row["quote_count"]
        assert row["selection_status"] == "PASS"
    from app.ui import ncaaf_inventory as panel
    captured = {}
    class UI:
        def subheader(self, *args): pass
        def caption(self, value): captured.setdefault("captions", []).append(value)
        def warning(self, value): captured["warning"] = value
        def dataframe(self, value, **kwargs): captured["table"] = value
        def download_button(self, label, data, **kwargs): captured["csv"] = data
    monkeypatch.setattr(panel, "st", UI())
    panel.render_inventory({"ncaaf_coverage": report})
    assert "Scheduled: 3 · Matched: 2 · Quoted: 1 · Timestamped: 1 · Ranked: 2 · Qualified: 0" in captured["captions"]
    csv = pd.read_csv(StringIO(captured["csv"]))
    assert csv["quote_count"].tolist() == [2, 0, 0]
    assert csv["timestamped_quote_count"].tolist() == [1, 0, 0]
    assert csv["quote_receipt_count"].tolist() == [4, 1, 0]
    assert csv["home_team"].tolist() == ["LSU Tigers"] * 3
