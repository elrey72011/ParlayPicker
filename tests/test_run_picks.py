"""Headless pick runner. SYNTHETIC pipeline; no network."""
import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_picks", ROOT / "scripts" / "run_picks.py")
run_picks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_picks)


def test_run_writes_board_analysis_and_a_labelled_summary(tmp_path):
    calls = {}

    def pipeline(sports, max_rows, use_ml):
        calls.update(sports=sports, max_rows=max_rows, use_ml=use_ml)
        best = pd.DataFrame([{"league": "NCAAB", "matchup": "A @ B", "pick": "B -3.5", "odds": -110,
                              "probability": 0.55, "secret_column": "x|y"}])
        return pd.DataFrame([{"game": 1}, {"game": 2}]), best, {"rows": 2, "nested": {"skip": True}}

    text, picks, games = run_picks.run(["NCAAB"], tmp_path / "out", pipeline=pipeline)
    assert calls == {"sports": ["NCAAB"], "max_rows": 10_000, "use_ml": True}
    assert (picks, games) == (1, 2)
    assert "NO_BET with zero stake" in text and "| NCAAB | A @ B | B -3.5 | -110 | 0.550 |" in text
    assert json.loads((tmp_path / "out" / "diagnostics.json").read_text()) == {"rows": 2}
    assert len(pd.read_csv(tmp_path / "out" / "best-picks.csv")) == 1


def test_empty_board_says_so(tmp_path):
    text = run_picks.summary_markdown(pd.DataFrame(), ["NBA"])
    assert "No picks were produced." in text
