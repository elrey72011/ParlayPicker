"""TheOver's college totals plan: the nightly tally (owner-approved 2026-11-19). SYNTHETIC games throughout.

The rules are the committed file ``docs/roadmap/theover-college-totals-plan-v1.json``; these tests read it
and never restate its fingerprint. The tally is read-only: nothing in the pipeline imports it.
"""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from core import theover_plan as plan

ROOT = Path(__file__).resolve().parents[1]
RULES, FINGERPRINT = plan.load_rules()
ET = plan.EASTERN


def _at(day, hour=19, minute=0):
    """An Eastern wall time on an ISO date, as an aware UTC datetime."""
    return datetime.combine(date.fromisoformat(day), datetime.min.time(), ET).replace(hour=hour, minute=minute).astimezone(timezone.utc)


def _total(game, run, saved, start, side="total_under", picked=1.0, *, hit=0.73, mark=1.0, line=150.5, price=-110,
           best=None, outcome=None, sport="NCAAB", book="novig", sheet=None, rating=0.56):
    """One totals candidate of a saved run, with the columns the app saves."""
    return {"matchup_id": game, "snapshot_id": run, "capture_recorded_at": saved.isoformat(),
            "game_start_utc": start.isoformat(), "sport": sport, "league": sport, "market_type": side,
            "best_pick": ("Under " if side == "total_under" else "Over ") + str(line), "candidate_id": f"{run}|{game}|{side}",
            "theover_pick_agrees": picked, "theover_model_hit_rate": hit, "theover_model_in_blend": mark,
            "total_line": line, "theover_pick_line": sheet, "odds_american": price, "best_line_price": best,
            "quote_bookmaker": book, "provider_event_id": "event-" + game, "calibrated_probability": rating,
            "candidate_outcome": outcome}


def _game(game, run, saved, start, pick="total_under", **fields):
    """Both sides of a game's total in one run: TheOver's side picked, the other not."""
    other = "total_over" if pick == "total_under" else "total_under"
    return [_total(game, run, saved, start, pick, 1.0, **fields),
            _total(game, run, saved, start, other, 0.0, **{**fields, "outcome": None})]


def _close(game, side, line, *, book="Novig", run="run-1", at="2026-11-21T23:50:00+00:00", verified=True):
    return {"snapshot_id": run, "candidate_id": f"{run}|{game}|{side}", "closing_capture_at": at, "quote_verified": verified,
            "quote": {"provider_event_id": "event-" + game, "market_type": side, "sportsbook": book, "line": line, "price": -110}}


# ---- what a bet is ----

def test_the_bet_is_theovers_side_in_the_last_run_saved_before_the_game():
    start = _at("2026-11-21")
    early, late, after = _at("2026-11-21", 15), _at("2026-11-21", 18), _at("2026-11-21", 19, 5)
    rows = (_game("g", "run-1", early, start, "total_under") + _game("g", "run-2", late, start, "total_over", line=151.5)
            + _game("g", "run-3", after, start, "total_under")                            # saved after the start: not a pick
            + _game("h", "run-1", early, start) + _game("h", "run-2", late, start, hit=None)   # the last run lists no model
            + _game("n", "run-2", late, start, sport="NBA", mark=0.0)                      # a record-only league
            + _game("s", "run-2", late, start, mark=0.0)                                  # matched by similar names only
            + [dict(_total("g", "run-2", late, start, "spread_home", None), total_line=None)])
    (bet,) = plan.bets(rows)
    assert (bet["game"], bet["snapshot_id"], bet["market_type"], bet["line"]) == ("g", "run-2", "total_over", 151.5)
    assert bet["slate"] == "2026-11-21" and bet["book"] == "Novig" and bet["event"] == "event-g"
    assert plan.bets([]) == [] and plan.bets([{"matchup_id": "x"}]) == []


def test_the_price_is_the_boards_best_at_the_line_and_the_picks_own_where_none_is_listed():
    start, saved = _at("2026-11-21"), _at("2026-11-21", 18)
    best, own = plan.bets(_game("a", "run-1", saved, start, price=-112, best=-104) + _game("b", "run-1", saved, start, price=-112))
    assert best["decimal"] == pytest.approx(1 + 100 / 104) and own["decimal"] == pytest.approx(1 + 100 / 112)


def test_a_start_after_midnight_belongs_to_the_night_before():
    assert plan.slate_date(_at("2026-11-22", 0, 30)) == date(2026, 11, 21)
    assert plan.slate_date(_at("2026-11-21", 12)) == date(2026, 11, 21)
    assert plan.slate_date(_at("2026-11-22", 5)) == date(2026, 11, 22)


# ---- its close ----

def test_a_close_read_for_an_earlier_runs_candidate_measures_the_last_runs_bet_from_its_own_line():
    start = _at("2026-11-21")
    rows = (_game("over", "run-2", _at("2026-11-21", 18), start, "total_over", line=150.5)
            + _game("under", "run-2", _at("2026-11-21", 18), start, "total_under", line=150.5)
            + _game("none", "run-2", _at("2026-11-21", 18), start, "total_under"))
    observations = [
        _close("over", "total_over", 151.5, at="2026-11-21T23:40:00+00:00"),
        _close("over", "total_over", 152.5, at="2026-11-21T23:50:00+00:00"),         # the later capture is the close
        _close("under", "total_under", 148.5),
        _close("none", "total_over", 149.5),                                         # the other side of that total
        _close("none", "total_under", 149.5, book="FanDuel"),                        # another book
        _close("none", "total_under", 149.5, verified=False)]                        # not a verified close
    by_game = {bet["game"]: bet for bet in plan.with_closes(plan.bets(rows), observations)}
    assert (by_game["over"]["close_line"], by_game["over"]["move"]) == (152.5, 2.0)     # an Over wants the number to rise
    assert (by_game["under"]["close_line"], by_game["under"]["move"]) == (148.5, 2.0)   # an Under wants it to fall
    assert by_game["none"]["move"] is None and by_game["none"]["close_line"] is None


# ---- the measures ----

def test_the_summary_counts_units_closes_and_the_move_with_its_standard_error():
    bets = [dict(outcome="WIN", decimal=2.0, move=1.0, rating=0.6, market_type="total_under"),
            dict(outcome="WIN", decimal=1.5, move=0.0, rating=0.6, market_type="total_under"),
            dict(outcome="LOSS", decimal=2.0, move=2.0, rating=0.6, market_type="total_over"),
            dict(outcome="PUSH", decimal=2.0, move=None, rating=0.6, market_type="total_over"),
            dict(outcome=None, decimal=2.0, move=None, rating=0.6, market_type="total_over")]
    s = plan.summarize(bets)
    assert (s["bets"], s["wins"], s["losses"], s["pushes"], s["waiting_for_a_result"]) == (5, 2, 1, 1, 1)
    assert s["units"] == pytest.approx(1.0 + 0.5 - 1.0 + 0.0) and s["units_per_bet"] == pytest.approx(0.5 / 4)   # a push is a bet
    assert s["hit_rate"] == pytest.approx(2 / 3) and s["break_even_hit_rate"] == pytest.approx(4 / 7.5)
    assert (s["closes"], s["close_share"]) == (3, pytest.approx(0.6))
    assert s["mean_move"] == pytest.approx(1.0) and s["move_standard_error"] == pytest.approx(1 / 3 ** 0.5)
    assert s["move_t"] == pytest.approx(3 ** 0.5)
    assert s["brier"] == pytest.approx((0.16 + 0.16 + 0.36) / 3) and s["calibration_error"] == pytest.approx(abs(0.6 - 2 / 3))
    empty = plan.summarize([])
    assert empty["bets"] == 0 and empty["move_t"] is None and empty["units"] is None


# ---- the windows ----

def _nights(first, nights, per_night, *, outcome="WIN", move=1.0, graded=True, with_close=True, prefix="v"):
    """Rows and closes for ``per_night`` plan bets on each of ``nights`` nights from ``first``."""
    rows, observations = [], []
    for night in range(nights):
        day = (date.fromisoformat(first) + timedelta(days=night)).isoformat()
        for number in range(per_night):
            game = f"{prefix}-{day}-{number}"
            # Alternate the result so the sample is not one-sided: three wins in five.
            result = (outcome if number % 5 < 3 else "LOSS") if graded else None
            rows += _game(game, "run-" + day, _at(day, 18), _at(day), "total_under", outcome=result, line=150.5)
            if with_close:
                # A move that varies around its mean, so it has a standard error.
                observations.append(_close(game, "total_under", 150.5 - move - (0.5 if number % 2 else -0.5), run="run-" + day))
    return rows, observations


def _tally(rows, observations, day, hour=10):
    return plan.report(rows, observations, RULES, FINGERPRINT, now=_at(day, hour))


def test_the_rules_file_names_two_windows_of_fourteen_nights_back_to_back():
    dates = RULES["windows"]["slate_dates_eastern"]
    (v0, v1), (h0, h1) = (tuple(date.fromisoformat(day) for day in dates[name]) for name in plan.WINDOWS)
    assert (v1 - v0).days == 13 and (h1 - h0).days == 13 and h0 == v1 + timedelta(days=1)
    assert RULES["bar_in_each_window"]["min_bets"] == 100 and RULES["both_windows_must_pass"] is True
    assert RULES["scope"]["sport"] == "NCAAB" and len(FINGERPRINT) == 64


def test_before_the_first_night_nothing_is_judged_and_earlier_bets_are_only_reported():
    rows, observations = _nights("2026-11-19", 2, 5, prefix="shakedown")
    tally = _tally(rows, observations, "2026-11-20")
    assert tally["verdict"] == "COLLECTING"
    assert [tally["windows"][name]["state"] for name in plan.WINDOWS] == ["not started", "not started"]
    assert tally["before_the_windows"]["bets"] == 10 and tally["windows"]["validation"]["bets"] == 0
    assert tally["rules_sha256"] == FINGERPRINT and tally["changes_nothing"] is True


def test_a_window_is_collecting_until_its_last_night_is_past_and_then_judged():
    rows, observations = _nights("2026-11-21", 14, 10)
    during = _tally(rows, observations, "2026-11-25")["windows"]["validation"]
    assert during["state"] == "collecting" and during["fails"] is None and during["bets"] == 140     # shown, never judged early
    last_night = _tally(rows, observations, "2026-12-04", 23)["windows"]["validation"]
    assert last_night["state"] == "collecting"
    done = _tally(rows, observations, "2026-12-05")
    window = done["windows"]["validation"]
    assert (window["state"], window["fails"], window["bets"], window["closes"]) == ("complete", [], 140, 140)
    assert window["mean_move"] == pytest.approx(1.0) and window["move_t"] > 2
    assert done["verdict"] == "COLLECTING" and done["windows"]["holdout"]["state"] == "collecting"   # the holdout decides too


def test_both_windows_passing_is_a_pass_and_a_failed_validation_is_a_failure():
    v_rows, v_obs = _nights("2026-11-21", 14, 10)
    h_rows, h_obs = _nights("2026-12-05", 14, 10, prefix="h")
    assert _tally(v_rows + h_rows, v_obs + h_obs, "2026-12-19")["verdict"] == "PASSED"
    flat_rows, flat_obs = _nights("2026-11-21", 14, 10, move=0.0)
    failed = _tally(flat_rows, flat_obs, "2026-12-05")
    assert failed["verdict"] == "FAILED"
    assert failed["windows"]["validation"]["fails"] == ["min_standard_errors_of_mean_line_move_above_zero"]
    lost_rows, lost_obs = _nights("2026-12-05", 14, 10, move=0.0, prefix="h")
    assert _tally(v_rows + lost_rows, v_obs + lost_obs, "2026-12-19")["verdict"] == "FAILED"   # a holdout failure is a failure


@pytest.mark.parametrize("change, fails", [
    (dict(with_close=False), ["min_share_of_bets_with_a_close", "min_standard_errors_of_mean_line_move_above_zero"]),
    (dict(outcome="LOSS"), ["min_units_per_bet", "max_brier", "max_calibration_error"]),
    (dict(move=-1.0), ["min_standard_errors_of_mean_line_move_above_zero"]),
])
def test_each_rule_of_the_bar_is_named_when_a_complete_window_misses_it(change, fails):
    rows, observations = _nights("2026-11-21", 14, 10, **change)
    assert _tally(rows, observations, "2026-12-05")["windows"]["validation"]["fails"] == fails


def test_a_window_short_of_its_bets_is_extended_a_night_at_a_time_and_the_holdout_follows():
    rows, observations = _nights("2026-11-21", 20, 6)                      # 84 bets in fourteen nights, 102 in seventeen
    short = _tally(rows, observations, "2026-12-06")["windows"]["validation"]
    assert (short["state"], short["last_night"], short["bets"]) == ("extending: short of its bets", "2026-12-05", 90)
    tally = _tally(rows, observations, "2026-12-09")
    window, holdout = tally["windows"]["validation"], tally["windows"]["holdout"]
    assert (window["state"], window["last_night"], window["extended_nights"], window["bets"]) == ("complete", "2026-12-07", 3, 102)
    assert (holdout["first_night"], holdout["last_night"]) == ("2026-12-08", "2026-12-21")       # still fourteen nights


def test_a_window_with_a_result_still_missing_waits_and_is_not_judged():
    rows, observations = _nights("2026-11-21", 14, 10)
    for row in rows:
        if row["matchup_id"] == "v-2026-12-04-0":
            row["candidate_outcome"] = None
    window = _tally(rows, observations, "2026-12-05")["windows"]["validation"]
    assert (window["state"], window["fails"], window["waiting_for_a_result"]) == ("waiting for results", None, 1)


def test_sides_line_gaps_and_college_football_are_reported_and_decide_nothing():
    start, saved = _at("2026-11-21"), _at("2026-11-21", 18)
    rows = (_game("worse", "r", saved, start, "total_under", line=149.5, sheet=150.5, outcome="WIN")     # an Under at a lower number
            + _game("same", "r", saved, start, "total_over", line=150.5, sheet=150.5, outcome="LOSS")
            + _game("better", "r", saved, start, "total_over", line=149.5, sheet=150.5, outcome="WIN")
            + _game("football", "r", saved, start, "total_under", sport="NCAAF", outcome="WIN"))
    tally = _tally(rows, [], "2026-11-22")
    gap = tally["line_against_theovers_own"]
    assert [gap[key]["bets"] for key in ("board_line_worse_for_the_pick", "same_line", "board_line_better_for_the_pick")] == [1, 1, 1]
    assert (tally["by_side"]["unders"]["bets"], tally["by_side"]["overs"]["bets"]) == (1, 2)
    assert tally["reported_without_a_bar"]["NCAAF"]["bets"] == 1 and tally["windows"]["validation"]["bets"] == 3


# ---- how it is shown and where it runs ----

def test_the_text_carries_the_fingerprint_the_verdict_and_each_window():
    rows, observations = _nights("2026-11-21", 14, 10)
    text = plan.markdown(_tally(rows, observations, "2026-12-05"))
    assert FINGERPRINT[:16] in text and "**COLLECTING**" in text and "changes no rating, pick or stake" in text
    assert "| validation | 2026-11-21 to 2026-12-04 | complete | 140 |" in text and "| none |" in text
    assert "| holdout | 2026-12-05 to 2026-12-18 | collecting | 0 |" in text
    assert "140 of 140 (100.0%)" in text and "+1.00 pts (t " in text


def test_build_reads_the_saved_rows_and_the_closes_of_one_database(monkeypatch, tmp_path):
    rows, observations = _nights("2026-11-21", 1, 3)
    monkeypatch.setattr("core.research_performance.saved_rows", lambda database: rows)
    monkeypatch.setattr("app_core.activation_closing.observations", lambda database: observations)
    tally = plan.build(tmp_path / "evidence.sqlite3", now=_at("2026-11-22"))
    assert tally["schema"] == plan.SCHEMA and tally["windows"]["validation"]["bets"] == 3


def test_the_morning_job_prints_the_tally_and_nothing_in_the_pipeline_reads_it():
    script = (ROOT / "scripts" / "track_record.py").read_text(encoding="utf-8")
    assert "theover_plan.build(args.database)" in script and "theover_plan.markdown(" in script
    for name in ("core/streamlit_pipeline.py", "streamlit_app.py", "app_core/activation_closing.py"):
        assert "theover_plan" not in (ROOT / name).read_text(encoding="utf-8")


def test_the_tally_reads_the_best_price_so_only_the_morning_report_may_load_it_and_it_saves_nothing():
    """The best price is a pick note, kept out of everything that picks or stakes (``tests/test_pick_notes.py``).

    The tally is the one reader outside the board and the exports (owner-approved 2026-11-19). It stays
    harmless only while nothing but the morning report loads it and it saves nothing a later run could read.
    """
    named = set()
    for folder in ("app", "app_core", "core", "services", "scripts"):
        for path in (ROOT / folder).rglob("*.py"):
            if "theover_plan" in path.read_text(encoding="utf-8", errors="replace"):
                named.add(path.relative_to(ROOT).as_posix())
    assert "theover_plan" not in (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    assert named == {"scripts/track_record.py"}
    source = (ROOT / "core" / "theover_plan.py").read_text(encoding="utf-8")
    for writer in (".write(", "write_text", "write_bytes", "to_csv", "to_sql", "to_json(", ".execute(", ".commit(", "backup("):
        assert writer not in source, writer
    assert source.count("best_line_price") == 2         # the docstring, and the one read in ``bets``
