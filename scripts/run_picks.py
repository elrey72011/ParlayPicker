"""Run today's game-pick pipeline headlessly, the same pipeline as the app's Refresh.

    python scripts/run_picks.py --output-dir OUT [--sports NBA,NCAAB,NHL,NFL,NCAAF]

Needs ODDS_API_KEY (and CFBD_API_KEY for NCAAF). It calls
``core.streamlit_pipeline.run_analysis_pipeline`` and writes the best-picks board and the
full analysis as CSV, a short markdown summary, and scalar diagnostics. It writes nothing
to Drive or the evidence store and places no wagers.

These picks are research-grade. No market has passed its frozen validation plan, so the
qualification chain marks every pick NO_BET with zero stake (``app_core.selection``).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_SPORTS = "NBA,NCAAB,NHL,NFL,NCAAF"
SHOW = ("league", "game_date", "commence_time", "matchup", "home_team", "away_team", "market", "market_type",
        "pick", "selected_pick", "line", "odds", "american_odds", "book", "best_book", "probability",
        "model_probability", "ml_probability", "calibrated_probability", "ev", "edge", "confidence",
        "wager_decision", "recommended_bet")
NOTICE = ("Research-grade picks from the app's pipeline. No market has passed its frozen validation plan, "
          "so the qualification chain treats every pick as NO_BET with zero stake.")


def _scalars(value):
    plain = (str, int, float, bool, type(None))
    return {k: v for k, v in (value or {}).items() if isinstance(v, plain)}


def summary_markdown(best, sports, *, limit=40):
    lines = [f"## Picks for {', '.join(sports)}", "", NOTICE, ""]
    if best is None or best.empty:
        return "\n".join(lines + ["No picks were produced.", ""])
    columns = [c for c in SHOW if c in best.columns] or list(best.columns[:12])
    lines.append(f"{len(best)} picks; showing up to {limit}.")
    lines.append("")
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "---|" * len(columns))
    for _, row in best.head(limit).iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = f"{value:.3f}"
            cells.append("" if value is None or str(value) == "nan" else str(value).replace("|", "/"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def run(sports, output_dir, *, pipeline=None):
    if pipeline is None:
        from core.streamlit_pipeline import run_analysis_pipeline as pipeline
    analysis, best, diagnostics = pipeline(sports=list(sports), max_rows=10_000, use_ml=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    best.to_csv(output_dir / "best-picks.csv", index=False)
    analysis.to_csv(output_dir / "analysis.csv", index=False)
    (output_dir / "diagnostics.json").write_text(
        json.dumps(_scalars(diagnostics), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    text = summary_markdown(best, sports)
    (output_dir / "picks.md").write_text(text, encoding="utf-8")
    return text, len(best), len(analysis)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sports", default=DEFAULT_SPORTS)
    args = parser.parse_args(argv)
    if not os.getenv("ODDS_API_KEY", "").strip():
        raise SystemExit("MISSING_CREDENTIALS: ODDS_API_KEY")
    sports = [s.strip().upper() for s in args.sports.split(",") if s.strip()]
    text, picks, games = run(sports, args.output_dir)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    print(json.dumps({"picks": picks, "analysis_rows": games}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
