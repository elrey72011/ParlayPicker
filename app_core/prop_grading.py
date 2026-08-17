"""App-facing MLB prop-export grading and cumulative ledger assembly."""
from __future__ import annotations

import inspect
from typing import Callable

import pandas as pd


REQUIRED_PROP_EXPORT_COLUMNS = frozenset({
    "market_type", "line", "odds_american",
})

PROP_AUDIT_COLUMNS = (
    "pipeline_build",
    "export_run_id",
    "directional_market",
    "MarketProbability",
    "CalibratedProbability",
    "ConservativeWinProbability",
    "CalibrationSource",
    "CalibrationSampleSize",
    "CalibrationProfileSampleSize",
    "DirectionalCalibrationSampleSize",
    "expected_count",
    "production_eligible",
    "production_gate_reason",
    "production_price_allowed",
    "production_min_american_odds",
    "production_projection_cushion",
    "Prop_Tier",
    "Market_Probation",
    "Status_Reason",
)


def _pick_column(card: pd.DataFrame) -> str | None:
    return next((name for name in ("best_pick", "pick") if name in card.columns), None)


def validate_prop_export(card: pd.DataFrame | None) -> tuple[bool, str]:
    if card is None or card.empty:
        return False, "The uploaded prop export is empty."
    missing = sorted(REQUIRED_PROP_EXPORT_COLUMNS.difference(card.columns))
    if _pick_column(card) is None:
        missing.append("best_pick")
    if missing:
        return False, "Missing required prop columns: " + ", ".join(missing)
    return True, ""


def _player_name(row: pd.Series) -> str:
    return str(
        row.get("player") or row.get("batter") or row.get("pitcher") or ""
    ).strip()


def _participant_type(row: pd.Series) -> str:
    explicit = str(row.get("participant_type") or "").strip().lower()
    if explicit in {"batter", "pitcher", "nfl_player"}:
        return explicit
    if str(row.get("league") or "").strip().upper() == "NFL":
        return "nfl_player"
    return "batter" if str(row.get("market_type", "")).startswith("batter_") else "pitcher"


def _default_name_resolver(card: pd.DataFrame, game_date: str) -> dict[str, object]:
    from scripts.grade_props import _build_name_to_id
    from app_core.mlb_batter_stats import resolve_batter_id

    league = card.get(
        "league", pd.Series("MLB", index=card.index)
    ).fillna("MLB").astype(str).str.upper()
    mlb_card = card[~league.eq("NFL")]
    ids = _build_name_to_id(game_date, mlb_card) if not mlb_card.empty else {}
    for _, row in mlb_card.iterrows():
        name = _player_name(row)
        if not name or name.lower() in ids:
            continue
        player_id = resolve_batter_id(name)
        if player_id is not None:
            ids[name.lower()] = player_id
    for _, row in card[league.eq("NFL")].iterrows():
        name = _player_name(row)
        if name:
            ids[name.lower()] = f"nfl:{name.lower()}"
    return ids


def _default_actual_fetcher(game_date: str, card: pd.DataFrame | None = None):
    from scripts.grade_props import fetch_actual_batter, fetch_actual_ks

    season = int(str(game_date)[:4])
    nfl_actuals: dict[str, dict[str, float]] = {}
    if isinstance(card, pd.DataFrame) and not card.empty:
        league = card.get(
            "league", pd.Series("MLB", index=card.index)
        ).fillna("MLB").astype(str).str.upper()
        if league.eq("NFL").any():
            from app_core.nfl_prop_pipeline import fetch_nfl_actuals

            nfl_actuals = fetch_nfl_actuals(game_date)

    def fetch(player_id: object, participant_type: str):
        if participant_type == "nfl_player":
            key = str(player_id).removeprefix("nfl:").strip().lower()
            return nfl_actuals.get(key)
        if participant_type == "batter":
            return fetch_actual_batter(player_id, game_date, season)
        return fetch_actual_ks(player_id, game_date, season)

    return fetch


def grade_prop_export(
    card: pd.DataFrame,
    game_date: str,
    *,
    name_resolver: Callable[[pd.DataFrame, str], dict[str, int]] | None = None,
    actual_fetcher: Callable | None = None,
) -> pd.DataFrame:
    """Grade every valid row in a normal prop export, including research rows."""
    from scripts.grade_props import (
        _decimal,
        _side_from_market,
        _stat_for_market,
        grade_side,
    )

    valid, error = validate_prop_export(card)
    if not valid:
        raise ValueError(error)
    game_date = str(game_date)[:10]
    resolver = name_resolver or _default_name_resolver
    name_to_id = resolver(card, game_date)
    fetch_actual = actual_fetcher or _default_actual_fetcher(game_date, card)
    try:
        accepts_participant_type = len(inspect.signature(fetch_actual).parameters) >= 2
    except (TypeError, ValueError):
        accepts_participant_type = True

    actual_cache: dict[tuple[str, str], object] = {}
    rows: list[dict] = []
    for _, source in card.iterrows():
        name = _player_name(source)
        market_type = str(source.get("market_type") or "")
        pick = source.get("best_pick") or source.get("pick")
        participant_type = _participant_type(source)
        side = _side_from_market(market_type, pick)
        stat = _stat_for_market(market_type, pick)
        line = pd.to_numeric(pd.Series([source.get("line")]), errors="coerce").iloc[0]
        odds = pd.to_numeric(
            pd.Series([source.get("odds_american")]), errors="coerce"
        ).iloc[0]
        if not name or pd.isna(line) or pd.isna(odds) or not str(pick or "").strip():
            continue
        player_id = name_to_id.get(name.lower())
        actual = None
        if player_id is not None:
            cache_key = (str(player_id), participant_type)
            if cache_key not in actual_cache:
                actual_cache[cache_key] = (
                    fetch_actual(player_id, participant_type)
                    if accepts_participant_type
                    else fetch_actual(player_id)
                )
            actual = actual_cache[cache_key]
        actual_value = actual.get(stat) if isinstance(actual, dict) else actual
        try:
            actual_missing = actual_value is None or bool(pd.isna(actual_value))
        except (TypeError, ValueError):
            actual_missing = actual_value is None
        result = (
            "PENDING"
            if actual_missing
            else grade_side(side, float(line), actual_value)
        )
        if str(result or "").upper() not in {"WIN", "LOSS", "PUSH"}:
            result = "PENDING"
        stake = float(pd.to_numeric(
            pd.Series([source.get("Kelly_Bet_Size", 0.0)]), errors="coerce"
        ).fillna(0.0).iloc[0])
        if result == "WIN":
            profit = stake * (_decimal(float(odds)) - 1.0)
        elif result == "LOSS":
            profit = -stake
        elif result == "PUSH":
            profit = 0.0
        else:
            profit = None
        raw_probability = source.get("RawWinProbability")
        if pd.isna(raw_probability) if raw_probability is not None else True:
            raw_probability = source.get("WinProbability")
        graded_row = {
            "league": str(source.get("league") or "MLB").strip().upper(),
            "game_date": game_date,
            "date": game_date,
            "player": name,
            "participant_type": participant_type,
            "pick": pick,
            "best_pick": pick,
            "side": side,
            "market_type": market_type,
            "stat": stat,
            "line": float(line),
            "matchup": source.get("matchup"),
            "book": source.get("book"),
            "RawWinProbability": raw_probability,
            "WinProbability": source.get("WinProbability"),
            "expected_value": source.get("expected_value"),
            "edge": source.get("edge"),
            "odds_american": float(odds),
            "stake": round(stake, 2),
            "actual_value": actual_value,
            "result": result,
            "profit": round(profit, 2) if profit is not None else None,
            "source_stake_status": source.get("Stake_Status"),
            "source_pick_status": source.get("Pick_Status"),
        }
        for column in PROP_AUDIT_COLUMNS:
            if column in source.index:
                graded_row[column] = source.get(column)
        rows.append(graded_row)
    return pd.DataFrame(rows)


def merge_prop_ledgers(
    existing: pd.DataFrame | None,
    newly_graded: pd.DataFrame | None,
) -> pd.DataFrame:
    """Append and dedupe uploaded results without losing prior settled rows."""
    frames = [
        frame.copy()
        for frame in (existing, newly_graded)
        if isinstance(frame, pd.DataFrame) and not frame.empty
    ]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    date = combined.get("game_date", combined.get("date", pd.Series("", index=combined.index)))
    player = combined.get("player", combined.get("pitcher", pd.Series("", index=combined.index)))
    pick = combined.get("pick", combined.get("best_pick", pd.Series("", index=combined.index)))
    league = combined.get(
        "league", pd.Series("MLB", index=combined.index)
    ).fillna("MLB").astype(str).str.upper().str.strip()
    combined["_ledger_key"] = (
        league
        + "|" + date.astype(str).str[:10].str.strip()
        + "|" + player.astype(str).str.lower().str.strip()
        + "|" + pick.astype(str).str.lower().str.strip()
    )
    combined = combined.drop_duplicates("_ledger_key", keep="last")
    return combined.drop(columns="_ledger_key").reset_index(drop=True)


def assemble_prop_ledgers(
    bundled: pd.DataFrame | None,
    uploaded: pd.DataFrame | None,
    generated: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build active history with newer sources overriding the repo baseline."""
    combined = merge_prop_ledgers(bundled, uploaded)
    return merge_prop_ledgers(combined, generated)


def grading_summary(ledger: pd.DataFrame | None) -> dict[str, int | float]:
    if ledger is None or ledger.empty or "result" not in ledger.columns:
        return {"graded": 0, "wins": 0, "losses": 0, "pushes": 0, "unresolved": 0}
    result = ledger["result"].fillna("").astype(str).str.upper()
    wins = int(result.eq("WIN").sum())
    losses = int(result.eq("LOSS").sum())
    pushes = int(result.eq("PUSH").sum())
    unresolved = int((~result.isin(["WIN", "LOSS", "PUSH"])).sum())
    return {
        "graded": wins + losses,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "unresolved": unresolved,
        "win_rate": wins / (wins + losses) if wins + losses else 0.0,
    }


def ledger_coverage_summary(ledger: pd.DataFrame | None) -> dict[str, object]:
    """Describe whether a graded prop ledger is genuinely cumulative."""
    if ledger is None or ledger.empty:
        return {
            "rows": 0,
            "settled": 0,
            "date_count": 0,
            "start_date": None,
            "end_date": None,
        }
    result = ledger.get(
        "result", pd.Series("", index=ledger.index)
    ).fillna("").astype(str).str.upper().str.strip()
    date_col = next(
        (
            column
            for column in ("game_date", "date", "graded_at", "settled_at")
            if column in ledger.columns
        ),
        None,
    )
    dates = pd.Series(dtype="datetime64[ns, UTC]")
    if date_col is not None:
        dates = pd.to_datetime(ledger[date_col], errors="coerce", utc=True).dropna()
    normalized_dates = dates.dt.strftime("%Y-%m-%d") if not dates.empty else dates
    unique_dates = sorted(set(normalized_dates.tolist()))
    return {
        "rows": int(len(ledger)),
        "settled": int(result.isin(["WIN", "LOSS"]).sum()),
        "date_count": int(len(unique_dates)),
        "start_date": unique_dates[0] if unique_dates else None,
        "end_date": unique_dates[-1] if unique_dates else None,
    }


def ledger_history_gap_summary(
    ledger: pd.DataFrame | None,
    target_date: object,
    *,
    max_gap_days: int = 1,
) -> dict[str, object]:
    """Flag grading that could silently skip a newer cumulative ledger.

    Streamlit's local recovery file is not durable across every deployment. If
    the bundled/uploaded ledger ends several days before the slate being graded,
    require an explicit acknowledgement so a fresh deployment cannot quietly
    replace recent calibration history with only the newly graded slate.
    """
    coverage = ledger_coverage_summary(ledger)
    latest = pd.to_datetime(coverage.get("end_date"), errors="coerce", utc=True)
    target = pd.to_datetime(target_date, errors="coerce", utc=True)
    gap_days = None
    if pd.notna(latest) and pd.notna(target):
        gap_days = int((target.normalize() - latest.normalize()).days)
    requires_confirmation = bool(
        gap_days is not None and gap_days > max(0, int(max_gap_days))
    )
    return {
        "latest_date": coverage.get("end_date"),
        "target_date": (
            target.strftime("%Y-%m-%d") if pd.notna(target) else None
        ),
        "gap_days": gap_days,
        "requires_confirmation": requires_confirmation,
    }
