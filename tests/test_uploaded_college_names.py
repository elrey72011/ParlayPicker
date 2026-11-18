"""Uploaded college names stay as TheOver wrote them (owner 2026-11-18). SYNTHETIC schedule files and prices.

The app carries a schedule file of old games and completes short team names from it. That suits a city ("Oklahoma" for
Oklahoma City) and not a school, whose last word is part of its name: it turned Houston into Houston Christian and Ohio
into Ohio State, so on 18 November Houston at Florida and Morehead State at Ohio found no match in TheOver's files.
"""
import pandas as pd
import pytest

import core.streamlit_pipeline as sp

# Old games as a schedule file holds them. The longer-named school is there and its shorter-named neighbour is not.
OLD_GAMES = [("NCAAB", "Houston Christian Huskies", "Ohio State Buckeyes"), ("NCAAB", "Penn State Nittany Lions", "Loyola Marymount Lions"),
             ("NCAAB", "Hawaii Rainbow", "Morehead State Eagles"), ("NCAAF", "San Diego State Aztecs", "Boise State Broncos"),
             ("NBA", "Oklahoma City Thunder", "Golden State Warriors")]


@pytest.fixture
def schedule_file(tmp_path, monkeypatch):
    """The schedule as the app's own loader reads it from data/master_all_sports.csv."""
    (tmp_path / "data").mkdir()
    pd.DataFrame([{"sport": league, "home_team": home, "away_team": away, "commence_time": "2030-03-20T23:00:00Z"}
                  for league, home, away in OLD_GAMES]).to_csv(tmp_path / "data" / "master_all_sports.csv", index=False)
    # The loader keeps its first answer for the life of the process. Reading through to the function itself means this
    # file neither takes another test's answer nor leaves its own behind for the next test.
    kept = sp.load_base_data.cache_info()
    with monkeypatch.context() as inside:
        inside.chdir(tmp_path)
        frame = sp.load_base_data.__wrapped__()
    assert sp.load_base_data.cache_info() == kept
    assert len(frame) == len(OLD_GAMES) and set(frame["league"]) == {"NCAAB", "NCAAF", "NBA"}
    return frame


def _uploaded(names):
    return pd.DataFrame({"league": [league for league, _ in names], "home_team": [name for _, name in names],
                         "away_team": ["Rutgers"] * len(names)})


def test_a_school_is_not_completed_into_another_school(schedule_file):
    assert sp.SCHOOL_NAME_LEAGUES == {"NCAAB", "NCAAF"}
    names = [("NCAAB", "Houston"), ("NCAAB", "Ohio"), ("NCAAB", "Penn"), ("NCAAB", "Loyola"), ("NCAAB", "Hawaii"),
             ("NCAAB", "Morehead"), ("NCAAF", "San Diego"), ("NCAAF", "Boise")]
    out = sp._resolve_team_names_from_base(_uploaded(names), schedule_file)
    assert out["home_team"].tolist() == ["Houston", "Ohio", "Penn", "Loyola", "Hawaii", "Morehead", "San Diego", "Boise"]
    assert out["away_team"].eq("Rutgers").all() and out["league"].tolist() == [league for league, _ in names]
    # The same names with no old games to complete them from come out the same: the file adds nothing for a school.
    no_colleges = schedule_file[schedule_file["league"].eq("NBA")]
    assert out[["home_team", "away_team"]].equals(sp._resolve_team_names_from_base(_uploaded(names), no_colleges)[["home_team", "away_team"]])
    # A name the file spells in full is left alone too, in the upload's own letters.
    full = sp._resolve_team_names_from_base(_uploaded([("NCAAB", "Houston Christian"), ("NCAAB", "ohio state")]), schedule_file)
    assert [name.lower() for name in full["home_team"]] == ["houston christian", "ohio state"]


def test_a_city_is_still_completed_for_a_professional_league(schedule_file):
    out = sp._resolve_team_names_from_base(_uploaded([("NBA", "Oklahoma"), ("NBA", "Golden")]), schedule_file)
    assert out["home_team"].str.startswith("Oklahoma City").tolist() == [True, False]
    assert out["home_team"].str.startswith("Golden State").tolist() == [False, True]
    # The rule that does it is the one a college league no longer uses.
    as_college = sp._resolve_team_names_from_base(_uploaded([("NCAAB", "Oklahoma"), ("NCAAB", "Golden")]), schedule_file)
    assert as_college["home_team"].tolist() == ["Oklahoma", "Golden"]


# ---- through the production pipeline ----

def _feed_row(home, away, total=145.5):
    row = {"league": "NCAAB", "home_team": home, "away_team": away, "commence_time_raw": "2031-01-15T00:00:00Z",
           "game_date": "2031-01-14"}
    for book in ("novig", "fanduel", "draftkings", "betmgm"):
        row.update({f"{book}_home_price": -110, f"{book}_home_point": -1.5, f"{book}_away_price": -110,
                    f"{book}_away_point": 1.5, f"{book}_over_price": -112, f"{book}_over_point": total,
                    f"{book}_under_price": -108, f"{book}_under_point": total,
                    f"{book}_h2h_home_price": -125, f"{book}_h2h_away_price": 105})
    return row


def _totals(home, away):
    return pd.DataFrame({"league": ["NCAAB"], "home_team": [home], "away_team": [away], "game_date": ["2031-01-14"],
                         "line": [145.5], "pick": ["UNDER"], "pickcode": ["OU_UNDER"], "winprobability": [None],
                         "winprobsource": ["not_provided"], "modelhitrate": [None], "odds_american": [-108]})


@pytest.mark.parametrize("feed, upload", [
    (("Florida Gators", "Houston Cougars"), ("Florida", "Houston")),                # 18 November 2026
    (("Ohio Bobcats", "Morehead St Eagles"), ("Ohio", "Morehead State")),           # 18 November 2026
    (("Houston Cougars", "Rutgers Scarlet Knights"), ("Houston", "Rutgers"))])      # 17 November 2026
def test_through_the_pipeline_the_game_finds_theovers_row_whatever_old_games_the_schedule_file_holds(monkeypatch, schedule_file, feed, upload):
    monkeypatch.setattr(sp, "load_base_data", lambda: schedule_file.copy())
    monkeypatch.setattr(sp, "fetch_live_odds_dataframe", lambda *args, **kwargs: pd.DataFrame([_feed_row(*feed)]))
    analysis, _best, diagnostics = sp.run_analysis_pipeline(sports=["NCAAB"], max_rows=40, use_ml=False, spreads_df=None,
                                                             totals_df=_totals(*upload))
    rows, counts = analysis.set_index("market_type"), diagnostics["candidate_generation_diagnostics"]
    assert len(rows) == 4 and rows["orientation_source"].str.startswith("exact_match").all()
    # The upload holds a total for the game and no side, so the two totals rows are the uploaded ones.
    assert rows.loc[["total_over", "total_under"], "candidate_source"].eq("upload_matched").all()
    assert (counts["games_matched_exact"], counts["games_matched_fuzzy"], counts["games_unmatched"]) == (1, 0, 0)
    assert not diagnostics.get("unmatched_live_games")
    # TheOver's Under is on the Under row and nowhere else.
    assert rows.loc["total_under", "theover_pick_agrees"] == 1 and rows.loc["total_over", "theover_pick_agrees"] == 0
    assert rows.loc["total_under", "theover_pick_line"] == 145.5
