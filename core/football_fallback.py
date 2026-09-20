"""Football research fallback boundaries. No synthetic opposing prices."""
import pandas as pd


def market_input(frame, existing):
    out = existing.copy()
    sport = frame.get("league", pd.Series("", index=frame.index)).astype(str).str.upper()
    football = sport.isin(["NFL", "NCAAF"])
    def implied(column):
        odds = pd.to_numeric(frame.get(column, pd.Series(float("nan"), index=frame.index)), errors="coerce")
        valid = odds.abs().ge(100) & odds.abs().lt(float("inf"))
        return (odds.abs() / (odds.abs() + 100)).where(odds < 0, 100 / (odds + 100)).where(valid)
    selected, opposing = implied("odds_american"), implied("opposing_odds_american")
    paired = selected / (selected + opposing)
    out.loc[football] = paired.loc[football]
    return out


def selection_sources(frame):
    sources = pd.Series("calibrated_probability", index=frame.index)
    sport = frame.get("league", pd.Series("", index=frame.index)).astype(str).str.upper()
    model = pd.to_numeric(frame.get("ml_probability", pd.Series(float("nan"), index=frame.index)), errors="coerce")
    absent = ~model.between(0, 1, inclusive="neither")
    sources.loc[sport.isin(["NFL", "NCAAF"]) & absent] = "football_research_blend_no_independent_model"
    return sources
