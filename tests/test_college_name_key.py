"""College basketball name key: standard spellings fold together; an uploaded game takes the feed's names only when certain. SYNTHETIC."""
import pandas as pd
import pytest

from app_core import college_name_key as nk

DAY = "2026-11-17"
NOTHING = {"reconciled": [], "ambiguous": []}


def _id(home, away, day):
    return "|".join(sorted((home.upper().replace(" ", ""), away.upper().replace(" ", "")))) + "|" + day


def _upload(*games):
    """Uploaded rows, an Over and an Under for each (league, day, home, away)."""
    return pd.DataFrame([{"league": league, "game_date": day, "home_team": home, "away_team": away, "market_type": market,
                          "matchup_id": _id(home, away, day), "total_line": 164.5,
                          "game_key": f"{league}|{home.upper()}|{away.upper()}"}
                         for league, day, home, away in games for market in ("total_over", "total_under")])


def _live(*games):
    return pd.DataFrame([{"league": league, "game_date": day, "home_team": home, "away_team": away,
                          "matchup_id": _id(home, away, day)} for league, day, home, away in games])


def test_the_key_folds_the_standard_variants_and_nothing_else():
    assert nk.LEAGUES == frozenset({"NCAAB"})
    same = [("Tennessee State", "Tennessee St"), ("Long Beach State", "Long Beach St."), ("Mount St. Mary's", "Mt St Marys"),
            ("Mount Saint Mary's", "Mt. St. Mary's"), ("Seattle U", "Seattle"), ("Seattle University", "Seattle"),
            ("St. John's", "Saint Johns"), ("William and Mary", "William Mary"), ("Cal St Fullerton", "Cal State Fullerton"),
            ("UT Arlington", "Utarlington"), ("SOUTH FLORIDA", "South  Florida"), ("San José St", "San Jose State")]
    for left, right in same:
        assert nk.team_key(left) == nk.team_key(right) != "", (left, right)
    different = [("Tennessee", "Tennessee St"), ("East Tennessee State", "Tennessee State"), ("Tennesee State", "Tennessee St"),
                 ("Boston College", "Boston U"), ("Miami Oh", "Miami"), ("St Francis", "St Francis Brooklyn"),
                 ("Michigan", "Michigan State"), ("Mount St Marys", "St Marys"), ("Illinois", "Illinois State")]
    for left, right in different:
        assert nk.team_key(left) != nk.team_key(right), (left, right)
    assert nk.team_key("Mt St Marys") == "mountsaintmarys" and nk.team_key("Kent St") == "kentstate"
    assert nk.team_key("St Louis U") == "saintlouis" and nk.team_key("U") == "u"        # a closing U goes; a lone word stays
    assert nk.team_key(None) == "" and nk.team_key(float("nan")) == "" and nk.team_key(pd.NA) == ""
    assert nk.pair_key("Tennessee St", "South Florida") == nk.pair_key("South Florida", "Tennessee State")


def test_an_uploaded_game_takes_the_feeds_names_when_exactly_one_game_fits():
    upload = _upload(("NCAAB", DAY, "South Florida", "Tennessee State"), ("NCAAB", DAY, "Mount St Marys", "Lehigh"),
                     ("NCAAB", DAY, "Duke", "Illinois"))
    live = _live(("NCAAB", DAY, "South Florida", "Tennessee St"),
                 ("NCAAB", DAY, "Lehigh", "Mt St Marys"),                    # the feed lists this one the other way round
                 ("NCAAB", DAY, "Duke", "Illinois"), ("NCAAB", DAY, "Tennessee", "Michigan State"))
    before = upload.copy()
    out, report = nk.reconcile(upload, live)
    pd.testing.assert_frame_equal(upload, before)                           # the caller's frame is not touched
    first = out[out["matchup_id"].eq(_id("South Florida", "Tennessee St", DAY))]
    assert len(first) == 2 and set(first["home_team"]) == {"South Florida"} and set(first["away_team"]) == {"Tennessee St"}
    assert set(first["game_key"]) == {"NCAAB|SOUTH FLORIDA|TENNESSEE ST"} and set(first["total_line"]) == {164.5}
    assert set(first["market_type"]) == {"total_over", "total_under"}
    # The other way round in the feed: the upload keeps its own home and away, in the feed's spelling.
    second = out[out["matchup_id"].eq(_id("Lehigh", "Mt St Marys", DAY))]
    assert len(second) == 2 and set(second["home_team"]) == {"Mt St Marys"} and set(second["away_team"]) == {"Lehigh"}
    # A game already the same on both sides is left out of it, and so is the feed game nobody uploaded.
    assert out[out["home_team"].eq("Duke")].equals(upload[upload["home_team"].eq("Duke")])
    assert [(e["uploaded"], e["feed"], e["changed"], e["rows"]) for e in report["reconciled"]] == [
        ("Tennessee State at South Florida", "Tennessee St at South Florida", "Tennessee State -> Tennessee St", 2),
        ("Lehigh at Mount St Marys", "Mt St Marys at Lehigh", "Mount St Marys -> Mt St Marys", 2)]
    assert report["ambiguous"] == []
    assert nk.notes(report) == {_id("South Florida", "Tennessee St", DAY): "Tennessee State -> Tennessee St",
                                _id("Lehigh", "Mt St Marys", DAY): "Mount St Marys -> Mt St Marys"}
    assert nk.notes(None) == {} and nk.notes(NOTHING) == {}


def test_anything_less_certain_is_left_as_it_was():
    live = _live(("NCAAB", DAY, "South Florida", "Tennessee St"), ("NCAAB", DAY, "California Baptist", "Seattle"),
                 ("NBA", DAY, "St Louis", "Chicago"), ("NCAAB", "2026-11-18", "Ucla", "Long Beach St"))
    cases = {"a typo is not a variant": _upload(("NCAAB", DAY, "South Florida", "Tennesee State")),
             "another league": _upload(("NBA", DAY, "Saint Louis", "Chicago")),
             "another day": _upload(("NCAAB", DAY, "Ucla", "Long Beach State")),
             "only one of the two teams": _upload(("NCAAB", DAY, "North Florida", "Tennessee State"))}
    for label, upload in cases.items():
        out, report = nk.reconcile(upload, live)
        assert out is upload and report == NOTHING, label
    # One game uploaded under two spellings: neither is renamed, and the report says so.
    twice = _upload(("NCAAB", DAY, "California Baptist", "Seattle U"), ("NCAAB", DAY, "California Baptist", "Seattle University"))
    out, report = nk.reconcile(twice, live)
    assert out is twice and report["reconciled"] == []
    assert report["ambiguous"] == [{"league": "NCAAB", "game_date": DAY, "feed": ["Seattle at California Baptist"],
                                    "uploaded": ["Seattle U at California Baptist", "Seattle University at California Baptist"]}]
    # Two feed games with one key: the same.
    doubled = pd.concat([live, _live(("NCAAB", DAY, "California Baptist", "Seattle U"))], ignore_index=True)
    out, report = nk.reconcile(_upload(("NCAAB", DAY, "California Baptist", "Seattle University")), doubled)
    assert report["reconciled"] == [] and len(report["ambiguous"]) == 1 and len(report["ambiguous"][0]["feed"]) == 2
    # A feed game an uploaded game already matches exactly is not offered to a second spelling of it.
    both = _upload(("NCAAB", DAY, "South Florida", "Tennessee St"), ("NCAAB", DAY, "South Florida", "Tennessee State"))
    out, report = nk.reconcile(both, live)
    assert out is both and report == NOTHING
    # The same names with another id (a day kept differently, say) is not this step's business, and a row with no
    # name has no key at all.
    same = _upload(("NCAAB", DAY, "South Florida", "Tennessee St"))
    same["matchup_id"] = "another-id"
    out, report = nk.reconcile(same, live)
    assert out is same and report == NOTHING
    nameless = _upload(("NCAAB", DAY, "South Florida", "Tennessee State"))
    nameless["away_team"] = pd.NA
    blank_feed = live.copy()
    blank_feed.loc[0, "away_team"] = pd.NA
    out, report = nk.reconcile(nameless, blank_feed)
    assert out is nameless and report == NOTHING
    # An index that repeats (frames joined end to end) is read by position.
    repeated = pd.concat([_upload(("NCAAB", DAY, "South Florida", "Tennessee State")),
                          _upload(("NCAAB", DAY, "California Baptist", "Seattle U"))])
    assert list(repeated.index) == [0, 1, 0, 1]
    out, report = nk.reconcile(repeated, live)
    assert list(out["away_team"]) == ["Tennessee St", "Tennessee St", "Seattle", "Seattle"] and len(report["reconciled"]) == 2
    # Nothing to work with.
    for upload, feed in ((None, live), (both, None), (both.iloc[0:0], live), (both, live.iloc[0:0]),
                         (both.drop(columns=["matchup_id"]), live), (both, live.drop(columns=["game_date"]))):
        out, report = nk.reconcile(upload, feed)
        assert out is upload and report == NOTHING


def test_the_day_is_read_in_one_form_for_both_frames():
    upload = _upload(("NCAAB", DAY, "South Florida", "Tennessee State"))
    upload["game_date"] = pd.Timestamp(DAY, tz="UTC")                       # the upload keeps a timestamp, the feed a day
    live = _live(("NCAAB", DAY, "South Florida", "Tennessee St"))
    assert nk.reconcile(upload, live)[1] == NOTHING                         # as they stand the two days read differently
    out, report = nk.reconcile(upload, live,
                               day_of=lambda frame: pd.to_datetime(frame["game_date"], utc=True).dt.strftime("%Y-%m-%d"))
    assert len(report["reconciled"]) == 1 and report["reconciled"][0]["game_date"] == DAY
    assert set(out["away_team"]) == {"Tennessee St"} and set(out["matchup_id"]) == {_id("South Florida", "Tennessee St", DAY)}


# ---- through the production pipeline ----

def _feed_row(home, away, total=164.5):
    row = {"league": "NCAAB", "home_team": home, "away_team": away, "commence_time_raw": "2031-01-15T00:00:00Z",
           "game_date": "2031-01-14"}
    for book in ("novig", "fanduel", "draftkings", "betmgm"):
        row.update({f"{book}_home_price": -110, f"{book}_home_point": -11.5, f"{book}_away_price": -110,
                    f"{book}_away_point": 11.5, f"{book}_over_price": -112, f"{book}_over_point": total,
                    f"{book}_under_price": -108, f"{book}_under_point": total,
                    f"{book}_h2h_home_price": -650, f"{book}_h2h_away_price": 475})
    return row


def _totals(away):
    return pd.DataFrame({"league": ["NCAAB"], "home_team": ["South Florida"], "away_team": [away], "game_date": ["2031-01-14"],
                         "line": [164.5], "pick": ["UNDER"], "pickcode": ["OU_UNDER"], "winprobability": [None],
                         "winprobsource": ["not_provided"], "modelhitrate": [0.632], "odds_american": [-108]})


def _run(monkeypatch, away):
    import core.streamlit_pipeline as sp
    monkeypatch.setattr(sp, "load_base_data", lambda: pd.DataFrame())
    monkeypatch.setattr(sp, "fetch_live_odds_dataframe",
                        lambda *args, **kwargs: pd.DataFrame([_feed_row("South Florida Bulls", "Tennessee St Tigers")]))
    analysis, _best, diagnostics = sp.run_analysis_pipeline(sports=["NCAAB"], max_rows=40, use_ml=False, spreads_df=None,
                                                             totals_df=_totals(away))
    return analysis.set_index("market_type"), diagnostics["candidate_generation_diagnostics"]


def test_through_the_pipeline_the_game_is_an_exact_match_and_keeps_theovers_pick(monkeypatch):
    rows, counts = _run(monkeypatch, "Tennessee State")
    assert len(rows) == 4 and rows["orientation_source"].str.startswith("exact_match|college_name_key").all()
    assert not rows["orientation_source"].str.contains("fuzzy").any()
    under = rows.loc["total_under"]
    assert under["best_pick"] == "Under 164.5" and (under["home_team"], under["away_team"]) == ("South Florida", "Tennessee St")
    assert under["theover_pick_agrees"] == 1.0 and under["theover_model_hit_rate"] == pytest.approx(0.632)
    assert under["theover_model_in_blend"] == 1.0 and under["uploaded_total_line"] == 164.5
    assert "after the college name key (Tennessee State -> Tennessee St)" in under["upload_match_reason"]
    assert rows.loc["total_over", "theover_pick_agrees"] == 0.0
    assert [entry["changed"] for entry in counts["college_name_key"]["reconciled"]] == ["Tennessee State -> Tennessee St"]
    assert (counts["games_matched_exact"], counts["games_matched_fuzzy"]) == (1, 0)


def test_a_game_spelled_the_same_on_both_sides_reads_as_before(monkeypatch):
    rows, counts = _run(monkeypatch, "Tennessee St")
    assert rows["orientation_source"].str.startswith("exact_match").all()
    assert not rows["orientation_source"].str.contains("college_name_key").any()
    assert rows.loc["total_under", "upload_match_reason"] == "Matched by exact matchup_id"
    assert counts["college_name_key"] == NOTHING and counts["games_matched_exact"] == 1


def test_without_the_key_or_with_a_typo_it_is_still_the_fuzzy_match_the_app_rejects(monkeypatch):
    rows, counts = _run(monkeypatch, "Tennesee State")                      # a typo: similar, not the same name
    assert counts["college_name_key"] == NOTHING and counts["games_matched_exact"] == 0
    assert not rows["orientation_source"].str.contains("exact_match").any()  # at best a fuzzy match, never an exact one
    monkeypatch.setattr(nk, "LEAGUES", frozenset())                         # the key switched off: what happened before
    rows, counts = _run(monkeypatch, "Tennessee State")
    assert set(rows["orientation_source"]) == {"fuzzy_match"} and counts["games_matched_fuzzy"] == 1
