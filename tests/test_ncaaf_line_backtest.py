"""NCAAF archived-line backtest (roadmap D18 step 1). SYNTHETIC model and games; no network."""
from app_core import ncaaf_line_backtest as bt


def _artifact(margin, total):
    model = lambda mu: {"kind": "constant", "intercept": mu, "bias": 0.0, "sigma": 3.0}
    return {"models": {"constant": {"margin": model(margin), "total": model(total)}}}


def _row(game_id, margin, total):
    return {"game_id": game_id, "y_margin": margin, "y_total": total}


def test_select_lines_prefers_consensus_and_keeps_home_perspective():
    games = [{"id": 1, "lines": [{"provider": "Bovada", "spread": -6.5, "overUnder": 50.5},
                                 {"provider": "consensus", "spread": -7.0, "overUnder": 51.0}]},
             {"id": 2, "lines": [{"provider": "Bovada", "spread": None, "overUnder": 44.5}]},
             {"id": 3, "lines": []}]
    lines = bt.select_lines(games)
    assert lines[1] == {"spread": -7.0, "total": 51.0, "provider": "consensus"}
    assert lines[2]["spread"] is None and lines[2]["total"] == 44.5
    assert 3 not in lines


def test_spread_and_total_grading_follow_the_live_pricing():
    # Model: home by 10, total 60. Line: home -7, total 50.5.
    artifact = _artifact(10.0, 60.0)
    lines = {1: {"spread": -7.0, "total": 50.5, "provider": "consensus"},
             2: {"spread": -7.0, "total": 50.5, "provider": "consensus"},
             3: {"spread": -7.0, "total": 50.5, "provider": "consensus"}}
    holdout = [_row(1, 10, 61), _row(2, 3, 40), _row(3, 7, 55)]  # cover, no cover, push
    report = bt.backtest(artifact, holdout, lines)["constant"]
    spread = report["SPREAD"]["by_threshold"]["0.50"]
    assert (spread["picks"], spread["wins"], spread["losses"], spread["pushes"]) == (3, 1, 1, 1)
    assert abs(spread["roi_per_pick"] - (100 / 110 - 1) / 3) < 1e-9
    total = report["TOTAL"]["by_threshold"]["0.50"]
    assert (total["wins"], total["losses"]) == (2, 1)  # model says over; 61 and 55 go over, 40 under
    assert report["SPREAD"]["brier_side_a"] is not None


def test_games_without_a_line_are_skipped_and_markdown_renders():
    artifact = _artifact(1.0, 50.0)
    report = bt.backtest(artifact, [_row(9, 3, 50)], {})
    assert report["constant"]["SPREAD"] == {"games": 0}
    text = bt.markdown({"artifact_hash": "a" * 64, "artifact_matches_frozen": True, "holdout_games": 1,
                        "games_with_lines": 0, "backtest": report})
    assert "matches" in text and "52.4%" in text
