from __future__ import annotations

import subprocess

import pandas as pd

from scripts import refresh_calibration


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _best_row(run_id: str, home: str, away: str) -> dict:
    return {
        "export_run_id": run_id,
        "league": "MLB",
        "Home": home,
        "Away": away,
        "best_pick": f"{home} +1.5",
        "effective_win_probability": 0.56,
    }


def test_classify_prefers_full_card_over_later_scoped_exports(tmp_path):
    run_id = "20260821T204933Z"
    full_export = tmp_path / "best_picks_export - run.csv"
    _write_csv(
        full_export,
        [
            _best_row(run_id, "Home A", "Away A"),
            _best_row(run_id, "Home B", "Away B"),
        ],
    )
    _write_csv(
        tmp_path / "production_game_bets.csv",
        [_best_row(run_id, "Home A", "Away A")],
    )
    _write_csv(
        tmp_path / "precision_game_card.csv",
        [_best_row(run_id, "Home B", "Away B")],
    )
    _write_csv(
        tmp_path / "player_props_all_export.csv",
        [_best_row(run_id, "Home C", "Away C")],
    )
    recap = tmp_path / "Performance Recap.csv"
    _write_csv(
        recap,
        [{"export_run_id": run_id, "Pick Taken": "Home A +1.5", "Outcome": "WIN"}],
    )

    exports, recaps = refresh_calibration._classify(tmp_path)

    assert exports == {run_id: full_export}
    assert recaps == [(recap, run_id)]


def test_classify_rejects_scoped_exports_when_full_card_is_missing(tmp_path):
    run_id = "20260821T204933Z"
    _write_csv(
        tmp_path / "production_game_bets.csv",
        [_best_row(run_id, "Home A", "Away A")],
    )
    _write_csv(
        tmp_path / "precision_game_card.csv",
        [_best_row(run_id, "Home B", "Away B")],
    )

    exports, recaps = refresh_calibration._classify(tmp_path)

    assert exports == {}
    assert recaps == []


def test_bucket_refit_continues_when_global_calibration_promotion_is_blocked(
    monkeypatch, tmp_path
):
    calls: list[str] = []
    results = iter(
        [
            subprocess.CompletedProcess([], 2),
            subprocess.CompletedProcess([], 0),
        ]
    )

    def fake_run(command, *, check):
        calls.append(str(command[1]))
        result = next(results)
        if check:
            result.check_returncode()
        return result

    monkeypatch.setattr(refresh_calibration.subprocess, "run", fake_run)

    promoted = refresh_calibration._run_refits(tmp_path)

    assert promoted is False
    assert calls[0].endswith("fit_calibration.py")
    assert calls[1].endswith("fit_bucket_stats.py")
