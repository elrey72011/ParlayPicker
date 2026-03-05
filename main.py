"""Production pipeline entrypoint."""
from __future__ import annotations

from datetime import datetime
import pandas as pd

from parlaypicker.core.probability_engine import calibrated_probability, american_to_prob
from parlaypicker.utils.config import MODEL_WEIGHTS
from parlaypicker.core.ev_engine import expected_value, bet_signal
from parlaypicker.core.bankroll_manager import kelly_fraction
from parlaypicker.core.parlay_engine import build_parlays
from parlaypicker.data_pipeline.sports_data_pipeline import run_pipeline
from parlaypicker.simulation.bankroll_simulation import run_bankroll_projection
from parlaypicker.utils.helpers import export_csv


def run(sport: str = "NBA", date_key: str | None = None) -> pd.DataFrame:
    date_key = date_key or datetime.now().strftime("%Y-%m-%d")
    df = run_pipeline(sport, date_key, workers=2)
    if df.empty:
        return df

    df["market_prob"] = american_to_prob(df["home_odds"].to_numpy())
    df["true_probability"] = (
        df.get("market_prob", 0) * MODEL_WEIGHTS["market"]
        + df.get("ml_prob", 0) * MODEL_WEIGHTS["ml"]
        + df.get("kalshi_prob", 0) * MODEL_WEIGHTS["kalshi"]
        + df.get("theover_prob", 0) * MODEL_WEIGHTS["theover"]
        + df.get("sentiment_prob", 0) * MODEL_WEIGHTS["sentiment"]
    ).clip(0.01, 0.99).map(calibrated_probability)
    df["expected_value"] = [expected_value(p, o) for p, o in zip(df["true_probability"], df["home_odds"])]
    df["kelly_fraction"] = [kelly_fraction(p, o) for p, o in zip(df["true_probability"], df["home_odds"])]
    df["bet_signal"] = df["expected_value"].map(bet_signal)

    parlays = build_parlays(df[["game_id", "home_odds", "true_probability"]].rename(columns={"home_odds": "odds"}), leg_count=2)
    export_csv(df.rename(columns={"home_odds": "odds"}), "data/exports/single_bets.csv")
    export_csv(parlays, "data/exports/parlays.csv")
    run_bankroll_projection(1000, float(df["true_probability"].mean()), -110)
    return df


if __name__ == "__main__":
    run()
