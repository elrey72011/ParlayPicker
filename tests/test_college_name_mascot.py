"""College name key, second rule: a nickname left on the end of a name, and the feed's short forms (owner 2026-11-18). SYNTHETIC."""
import pandas as pd
import pytest

from app_core import college_name_key as nk

DAY = "2026-11-18"
NOTHING = {"reconciled": [], "ambiguous": []}


def _id(home, away, day=DAY):
    return "|".join(sorted((home.upper().replace(" ", ""), away.upper().replace(" ", "")))) + "|" + day


def _upload(*games):
    """Uploaded rows, an Over and an Under for each (league, home, away)."""
    return pd.DataFrame([{"league": league, "game_date": DAY, "home_team": home, "away_team": away, "market_type": market,
                          "matchup_id": _id(home, away), "total_line": 158.5,
                          "game_key": f"{league}|{home.upper()}|{away.upper()}"}
                         for league, home, away in games for market in ("total_over", "total_under")])


def _live(*games):
    return pd.DataFrame([{"league": league, "game_date": DAY, "home_team": home, "away_team": away,
                          "matchup_id": _id(home, away)} for league, home, away in games])


def test_a_nickname_is_dropped_from_the_end_of_a_name_and_nothing_else_is():
    same = [("San Diego St Aztecs", "San Diego State"), ("Creighton Bluejays", "Creighton"),
            ("Boston Univ Terriers", "Boston University"), ("Boston Univ Terriers", "Boston U"),
            ("Texas Tech Red Raiders", "Texas Tech"), ("Tennessee Tech Golden Eagles", "Tennessee Tech"),
            ("Delaware Fightin' Blue Hens", "Delaware"), ("Louisiana Ragin' Cajuns", "Louisiana"),
            ("Long Beach St 49ers", "Long Beach State"), ("Cornell Big Red", "Cornell"), ("North Texas Mean Green", "North Texas"),
            ("St. John's Red Storm", "Saint Johns"), ("Mount St. Mary's Mountaineers", "Mt St Marys"),
            ("William & Mary Tribe", "William and Mary"), ("Seattle U Redhawks", "Seattle"), ("American Eagles", "American University")]
    for left, right in same:
        assert nk.school_key(left) == nk.school_key(right) != "", (left, right)
    # With no nickname on it a name has one key, and the first key never drops a nickname.
    for name in ("San Diego State", "Illinois State", "Bowling Green", "Texas Tech", "Boston College", "Long Beach", "Saint Louis U"):
        assert nk.school_key(name) == nk.team_key(name), name
    assert nk.team_key("San Diego St Aztecs") == "sandiegostateaztecs" != nk.team_key("San Diego State")
    # A school is never cut down to another school: "State", "Tech", "College" and a place are not nicknames.
    different = [("Illinois State", "Illinois"), ("Michigan State Spartans", "Michigan"), ("Boston College Eagles", "Boston U"),
                 ("Texas Tech Red Raiders", "Texas"), ("Bowling Green Falcons", "Bowling"), ("Loyola Marymount Lions", "Loyola Chicago"),
                 ("Southern Jaguars", "Southern Utah Thunderbirds"), ("Tennessee Tech Golden Eagles", "Tennessee"),
                 ("Long Beach St 49ers", "Long Beach")]
    for left, right in different:
        assert nk.school_key(left) != nk.school_key(right), (left, right)
    # "Red Raiders" goes whole, not "Raiders" alone; a name that is a nickname and nothing else stays whole.
    assert nk.school_key("Texas Tech Red Raiders") == "texastech" and nk.school_key("Colgate Raiders") == "colgate"
    assert nk.school_key("Aztecs") == "aztecs" and nk.school_key("Golden Eagles") == "goldeneagles"
    # Half a nickname, left when the shared table has taken the other half ("Golden Panthers", "Screaming Eagles").
    for left, right in (("Florida Intl Golden", "Florida International"), ("Southern Indiana Screaming", "Southern Indiana"),
                        ("Texas Tech Red", "Texas Tech"), ("Marquette Golden", "Marquette"), ("Maine Black", "Maine")):
        assert nk.school_key(left) == nk.school_key(right) != "", (left, right)
        assert nk.team_key(left) != nk.team_key(right), (left, right)
    assert nk.school_key("Golden") == "golden" and nk.school_key("Bowling Green") == "bowlinggreen"
    assert nk.school_key("Cornell Big Red") == "cornell"                      # the whole nickname first, never "Cornell Big"
    assert nk.school_key(None) == "" and nk.school_key(float("nan")) == "" and nk.school_key(pd.NA) == ""


def test_the_list_of_nicknames_holds_no_word_of_a_schools_own_name():
    school_words = {"state", "saint", "mount", "tech", "college", "university", "univ", "u", "am", "city", "beach", "southern",
                    "northern", "eastern", "western", "central", "international", "christian", "baptist", "valley"}
    assert not {word for nickname in nk.MASCOTS for word in nickname} & school_words
    # A colour is a nickname only as part of one ("Big Green", "Mean Green", "Big Red"), so Bowling Green keeps its name.
    assert not {("green",), ("red",), ("blue",), ("golden",), ("big",), ("black",), ("purple",)} & nk.MASCOTS
    assert all(nickname and all(word == word.lower() and word.isalnum() for word in nickname) for nickname in nk.MASCOTS)
    assert {("red", "raiders"), ("raiders",), ("fightin", "blue", "hens"), ("aztecs",), ("bluejays",), ("terriers",)} <= nk.MASCOTS
    assert max(len(nickname) for nickname in nk.MASCOTS) == 3
    # The half nicknames are first words of a listed nickname, and none is a word a school's name ends in.
    firsts = {nickname[0] for nickname in nk.MASCOTS if len(nickname) > 1}
    assert nk.HALF_NICKNAMES <= firsts and not nk.HALF_NICKNAMES & school_words
    assert not nk.HALF_NICKNAMES & {"green", "big", "state", "great", "mean"}
    assert not {(word,) for word in nk.HALF_NICKNAMES} & nk.MASCOTS


def test_the_feeds_short_forms_read_as_the_school_written_out():
    same = [("Csu Fullerton", "Cal State Fullerton"), ("Cs Fullerton", "Cal St Fullerton"), ("CSU Bakersfield", "Cal State Bakersfield"),
            ("Csu Northridge", "Cal State Northridge"), ("Loyola Chi", "Loyola Chicago"), ("Loyola (Chi)", "Loyola-Chicago"),
            ("Loyola Md", "Loyola Maryland"), ("Miami Oh", "Miami Ohio"), ("Miami (OH)", "Miami Ohio"),
            ("St Thomas Mn", "St. Thomas"), ("Purdue Fort Wayne", "Fort Wayne"), ("Florida Intl", "Florida International")]
    for left, right in same:
        assert nk.team_key(left) == nk.team_key(right) != "", (left, right)
    # Each short form counts only where it belongs.
    different = [("Csu", "Cal State"), ("Csu Pueblo", "Cal State Pueblo"), ("Miami Oh", "Miami"), ("Loyola Chicago", "Loyola Maryland"),
                 ("Purdue", "Purdue Fort Wayne"), ("Ohio", "Oh"), ("Chi State", "Chicago State")]
    for left, right in different:
        assert nk.team_key(left) != nk.team_key(right), (left, right)
    assert nk.team_key("Csu Fullerton") == "calstatefullerton" and nk.team_key("Purdue Fort Wayne") == "fortwayne"
    # With a nickname on the end as well.
    assert nk.school_key("Loyola (Chi) Ramblers") == nk.school_key("Loyola Chicago") == "loyolachicago"
    assert nk.school_key("Fort Wayne Mastodons") == nk.school_key("Purdue Fort Wayne") == "fortwayne"
    assert nk.school_key("St Thomas Mn Tommies") == nk.school_key("St. Thomas") == "saintthomas"
    assert nk.school_key("Csu Northridge Matadors") == nk.school_key("Cal State Northridge")
    assert nk.school_key("Loyola Md Greyhounds") == nk.school_key("Loyola Maryland")


def test_the_nickname_rule_is_tried_after_the_spelling_rule_and_each_says_which_it_was():
    assert [name for name, _ in nk.RULES] == ["spelling", "mascot"]
    upload = _upload(("NCAAB", "San Diego State", "Texas Southern"), ("NCAAB", "Creighton Bluejays", "Bethunecookman"),
                     ("NCAAB", "South Florida", "Tennessee State"), ("NCAAB", "Duke", "Illinois"))
    live = _live(("NCAAB", "San Diego St Aztecs", "Texas Southern"),
                 ("NCAAB", "Bethunecookman", "Creighton"),                     # the feed lists this one the other way round
                 ("NCAAB", "South Florida", "Tennessee St"), ("NCAAB", "Duke", "Illinois"),
                 ("NCAAB", "Illinois State", "Michigan State"))
    before = upload.copy()
    out, report = nk.reconcile(upload, live)
    pd.testing.assert_frame_equal(upload, before)                            # the caller's frame is not touched
    assert [(e["rule"], e["uploaded"], e["feed"], e["changed"], e["rows"]) for e in report["reconciled"]] == [
        ("spelling", "Tennessee State at South Florida", "Tennessee St at South Florida", "Tennessee State -> Tennessee St", 2),
        ("mascot", "Texas Southern at San Diego State", "Texas Southern at San Diego St Aztecs",
         "San Diego State -> San Diego St Aztecs", 2),
        ("mascot", "Bethunecookman at Creighton Bluejays", "Creighton at Bethunecookman", "Creighton Bluejays -> Creighton", 2)]
    assert report["ambiguous"] == []
    aztecs = out[out["matchup_id"].eq(_id("San Diego St Aztecs", "Texas Southern"))]
    assert len(aztecs) == 2 and set(aztecs["home_team"]) == {"San Diego St Aztecs"} and set(aztecs["away_team"]) == {"Texas Southern"}
    assert set(aztecs["game_key"]) == {"NCAAB|SAN DIEGO ST AZTECS|TEXAS SOUTHERN"} and set(aztecs["total_line"]) == {158.5}
    # The other way round in the feed: the upload keeps its own home and away, in the feed's spelling.
    creighton = out[out["matchup_id"].eq(_id("Bethunecookman", "Creighton"))]
    assert len(creighton) == 2 and set(creighton["home_team"]) == {"Creighton"} and set(creighton["away_team"]) == {"Bethunecookman"}
    assert out[out["home_team"].eq("Duke")].equals(upload[upload["home_team"].eq("Duke")])
    assert nk.notes(report) == {_id("South Florida", "Tennessee St"): "Tennessee State -> Tennessee St",
                                _id("San Diego St Aztecs", "Texas Southern"): "San Diego State -> San Diego St Aztecs",
                                _id("Bethunecookman", "Creighton"): "Creighton Bluejays -> Creighton"}


def test_under_the_nickname_rule_too_anything_less_certain_is_left_as_it_was():
    live = _live(("NCAAB", "San Diego St Aztecs", "Texas Southern"), ("NCAAB", "Illinois State", "Michigan State"))
    cases = {"a school is not its longer namesake": _upload(("NCAAB", "Illinois", "Michigan")),
             "only one of the two teams": _upload(("NCAAB", "San Diego State", "Texas State")),
             "another league": _upload(("NCAAF", "San Diego State", "Texas Southern")),
             "a typo is still a typo": _upload(("NCAAB", "San Deigo State", "Texas Southern")),
             "a nickname is not a school": _upload(("NCAAB", "Aztecs", "Texas Southern"))}
    for label, upload in cases.items():
        out, report = nk.reconcile(upload, live)
        assert out is upload and report == NOTHING, label
    # Two uploaded games that read as the one feed game: neither is renamed, and it is said once.
    twice = _upload(("NCAAB", "San Diego State", "Texas Southern"), ("NCAAB", "San Diego State University", "Texas Southern"))
    out, report = nk.reconcile(twice, live)
    assert out is twice and report["reconciled"] == []
    assert report["ambiguous"] == [{"league": "NCAAB", "game_date": DAY, "feed": ["Texas Southern at San Diego St Aztecs"],
                                    "uploaded": ["Texas Southern at San Diego State", "Texas Southern at San Diego State University"]}]
    # Two feed games that read as the one uploaded game: the same.
    doubled = pd.concat([live, _live(("NCAAB", "San Diego State Aztecs", "Texas Southern"))], ignore_index=True)
    out, report = nk.reconcile(_upload(("NCAAB", "San Diego State", "Texas Southern")), doubled)
    assert report["reconciled"] == [] and len(report["ambiguous"]) == 1 and len(report["ambiguous"][0]["feed"]) == 2
    # A feed game the spelling rule has already given to one uploaded game is not offered to another under this rule.
    both = _upload(("NCAAB", "San Diego State Aztecs", "Texas Southern"), ("NCAAB", "San Diego State", "Texas Southern"))
    out, report = nk.reconcile(both, live)
    assert [(e["rule"], e["uploaded"]) for e in report["reconciled"]] == [("spelling", "Texas Southern at San Diego State Aztecs")]
    assert report["ambiguous"] == [] and set(out.loc[out["home_team"].eq("San Diego State"), "matchup_id"]) == {
        _id("San Diego State", "Texas Southern")}
    # A game with a nameless side beside a doubtful one: the doubtful one is reported, and nothing fails.
    mixed = pd.concat([twice, _upload(("NCAAB", "Duke", "Illinois"))], ignore_index=True)
    mixed.loc[mixed["home_team"].eq("Duke"), "away_team"] = pd.NA
    out, report = nk.reconcile(mixed, live)
    assert out is mixed and len(report["ambiguous"]) == 1 and report["reconciled"] == []


# ---- through the production pipeline ----

def _feed_row(home, away, total=158.5):
    row = {"league": "NCAAB", "home_team": home, "away_team": away, "commence_time_raw": "2031-01-15T00:00:00Z",
           "game_date": "2031-01-14"}
    for book in ("novig", "fanduel", "draftkings", "betmgm"):
        row.update({f"{book}_home_price": -110, f"{book}_home_point": -11.5, f"{book}_away_price": -110,
                    f"{book}_away_point": 11.5, f"{book}_over_price": -112, f"{book}_over_point": total,
                    f"{book}_under_price": -108, f"{book}_under_point": total,
                    f"{book}_h2h_home_price": -650, f"{book}_h2h_away_price": 475})
    return row


def _totals(home, away):
    return pd.DataFrame({"league": ["NCAAB"], "home_team": [home], "away_team": [away], "game_date": ["2031-01-14"],
                         "line": [158.5], "pick": ["UNDER"], "pickcode": ["OU_UNDER"], "winprobability": [None],
                         "winprobsource": ["not_provided"], "modelhitrate": [0.632], "odds_american": [-108]})


def _run(monkeypatch, feed, upload):
    import core.streamlit_pipeline as sp
    monkeypatch.setattr(sp, "load_base_data", lambda: pd.DataFrame())
    monkeypatch.setattr(sp, "fetch_live_odds_dataframe", lambda *args, **kwargs: pd.DataFrame([_feed_row(*feed)]))
    analysis, _best, diagnostics = sp.run_analysis_pipeline(sports=["NCAAB"], max_rows=40, use_ml=False, spreads_df=None,
                                                             totals_df=_totals(*upload))
    return analysis.set_index("market_type"), diagnostics["candidate_generation_diagnostics"]


@pytest.mark.parametrize("feed, upload", [
    (("San Diego St Aztecs", "Texas Southern Tigers"), ("San Diego State", "Texas Southern")),       # 18 November 2026
    (("Creighton Bluejays", "Bethune-Cookman Wildcats"), ("Creighton", "Bethune-Cookman")),          # read two ways by the table
    (("Boston Univ. Terriers", "Merrimack Warriors"), ("Boston University", "Merrimack")),
    (("Fort Wayne Mastodons", "Bradley Braves"), ("Purdue Fort Wayne", "Bradley")),
    (("Loyola (Chi) Ramblers", "Wichita St Shockers"), ("Loyola Chicago", "Wichita State")),
    (("CSU Fullerton Titans", "UCLA Bruins"), ("Cal State Fullerton", "UCLA"))])
def test_through_the_pipeline_the_game_is_an_exact_match_and_keeps_theovers_pick(monkeypatch, feed, upload):
    rows, counts = _run(monkeypatch, feed, upload)
    # However the names come to agree (the shared table may learn one of them), the game is matched exactly.
    assert len(rows) == 4 and rows["orientation_source"].str.startswith("exact_match").all()
    assert (counts["games_matched_exact"], counts["games_matched_fuzzy"]) == (1, 0)
    under = rows.loc["total_under"]
    assert under["best_pick"] == "Under 158.5" and under["uploaded_total_line"] == 158.5
    assert under["theover_pick_agrees"] == 1.0 and under["theover_model_hit_rate"] == pytest.approx(0.632)
    assert under["theover_model_in_blend"] == 1.0 and under["blend_in_theover"] == pytest.approx(0.632)
    assert under["calibrated_probability"] > under["blend_in_market"]
    assert rows.loc["total_over", "theover_pick_agrees"] == 0.0


def test_through_the_pipeline_with_the_key_switched_off_the_game_is_not_matched(monkeypatch):
    # A school no table can know, spelled two ways with a nickname on one: matched with the key, not without it.
    feed, upload = ("Wobegon St Lumberjacks", "Mist County Tigers"), ("Wobegon State", "Mist County")
    rows, counts = _run(monkeypatch, feed, upload)
    assert rows["orientation_source"].str.startswith("exact_match|college_name_key").all() and counts["games_matched_exact"] == 1
    changed = [entry["changed"] for entry in counts["college_name_key"]["reconciled"]]
    assert len(changed) == 1 and changed[0].startswith("Wobegon State -> Wobegon St")
    monkeypatch.setattr(nk, "LEAGUES", frozenset())                         # what happened before 18 November 2026
    rows, counts = _run(monkeypatch, feed, upload)
    assert counts["college_name_key"] == NOTHING and counts["games_matched_exact"] == 0
    assert not rows["orientation_source"].str.contains("exact_match").any()
