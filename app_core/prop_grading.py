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


def prop_funded_mask(card: pd.DataFrame | None) -> pd.Series:
    """Identify rows that were actually authorized for a production stake.

    Normal exports carry ``Stake_Status``/``Kelly_Bet_Size`` while graded rows
    preserve those fields as ``source_stake_status``/``stake``. Use the same
    fail-closed rule for both representations so research rows can be graded
    without ever leaking into wager P&L.
    """
    if card is None:
        return pd.Series(dtype=bool)
    if card.empty:
        return pd.Series(False, index=card.index, dtype=bool)

    if "source_funded" in card.columns:
        values = card["source_funded"]
        if pd.api.types.is_bool_dtype(values.dtype):
            return values.fillna(False).astype(bool)
        normalized = values.astype("string").fillna("").str.strip().str.casefold()
        return normalized.isin({"true", "1", "yes", "y"})

    for column in ("Stake_Status", "source_stake_status"):
        if column not in card.columns:
            continue
        status = card[column].astype("string").fillna("").str.strip()
        if status.ne("").any():
            return status.str.casefold().eq("funded")

    pick_status = pd.Series("", index=card.index, dtype="object")
    for column in ("Pick_Status", "source_pick_status"):
        if column in card.columns:
            pick_status = card[column]
            break
    stake = pd.Series(0.0, index=card.index, dtype="float64")
    for column in ("Kelly_Bet_Size", "stake"):
        if column in card.columns:
            stake = pd.to_numeric(card[column], errors="coerce").fillna(0.0)
            break
    actionable = (
        pick_status.astype("string")
        .fillna("")
        .str.strip()
        .str.casefold()
        .eq("actionable")
    )
    return actionable & stake.gt(0.0)


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


def _default_actual_fetcher(
    game_date: str,
    card: pd.DataFrame | None = None,
    name_to_id: dict[str, object] | None = None,
):
    from scripts.grade_props import (
        fetch_actual_batter,
        fetch_actual_ks,
        fetch_boxscore_actuals,
    )

    season = int(str(game_date)[:4])
    nfl_actuals: dict[str, dict[str, float]] = {}
    mlb_boxscore_actuals: dict[tuple[str, str], dict[str, object]] = {}
    if isinstance(card, pd.DataFrame) and not card.empty:
        league = card.get(
            "league", pd.Series("MLB", index=card.index)
        ).fillna("MLB").astype(str).str.upper()
        if league.eq("NFL").any():
            from app_core.nfl_prop_pipeline import fetch_nfl_actuals

            nfl_actuals = fetch_nfl_actuals(game_date)
        mlb_rows = card[~league.eq("NFL")]
        if not mlb_rows.empty and name_to_id:
            expected_players: dict[tuple[str, str], str] = {}
            for _, row in mlb_rows.iterrows():
                name = _player_name(row)
                player_id = name_to_id.get(name.lower()) if name else None
                if player_id is None:
                    continue
                expected_players[(str(player_id), _participant_type(row))] = str(
                    row.get("matchup") or ""
                )
            if expected_players:
                mlb_boxscore_actuals = fetch_boxscore_actuals(
                    game_date, expected_players
                )

    def fetch(player_id: object, participant_type: str):
        if participant_type == "nfl_player":
            key = str(player_id).removeprefix("nfl:").strip().lower()
            return nfl_actuals.get(key)
        if participant_type == "batter":
            game_log = fetch_actual_batter(player_id, game_date, season)
        else:
            game_log = fetch_actual_ks(player_id, game_date, season)
        if game_log is not None:
            return game_log
        return mlb_boxscore_actuals.get((str(player_id), participant_type))

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
    fetch_actual = actual_fetcher or _default_actual_fetcher(
        game_date, card, name_to_id
    )
    funded_mask = prop_funded_mask(card)
    try:
        accepts_participant_type = len(inspect.signature(fetch_actual).parameters) >= 2
    except (TypeError, ValueError):
        accepts_participant_type = True

    actual_cache: dict[tuple[str, str], object] = {}
    rows: list[dict] = []
    for position, (_, source) in enumerate(card.iterrows()):
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
        grading_override = (
            str(actual.get("_grading_result") or "").strip().upper()
            if isinstance(actual, dict)
            else ""
        )
        grading_source = (
            actual.get("_grading_source") if isinstance(actual, dict) else None
        )
        void_reason = actual.get("_void_reason") if isinstance(actual, dict) else None
        actual_value = actual.get(stat) if isinstance(actual, dict) else actual
        try:
            actual_missing = actual_value is None or bool(pd.isna(actual_value))
        except (TypeError, ValueError):
            actual_missing = actual_value is None
        if grading_override == "VOID":
            result = "VOID"
        elif actual_missing:
            result = "PENDING"
        else:
            result = grade_side(side, float(line), actual_value)
        if str(result or "").upper() not in {"WIN", "LOSS", "PUSH", "VOID"}:
            result = "PENDING"
        recommended_stake = float(pd.to_numeric(
            pd.Series([source.get("Kelly_Bet_Size", 0.0)]), errors="coerce"
        ).fillna(0.0).iloc[0])
        source_funded = bool(funded_mask.iloc[position])
        stake = recommended_stake if source_funded else 0.0
        if result == "WIN":
            profit = stake * (_decimal(float(odds)) - 1.0)
        elif result == "LOSS":
            profit = -stake
        elif result in {"PUSH", "VOID"}:
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
            "source_recommended_stake": round(recommended_stake, 2),
            "source_funded": source_funded,
            "evaluation_scope": (
                "PRODUCTION-APPROVED WAGER"
                if source_funded
                else "RESEARCH / CALIBRATION"
            ),
            "actual_value": actual_value,
            "result": result,
            "profit": round(profit, 2) if profit is not None else None,
            "grading_source": grading_source,
            "void_reason": void_reason,
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
        return {
            "graded": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "voids": 0,
            "unresolved": 0,
        }
    result = ledger["result"].fillna("").astype(str).str.upper()
    wins = int(result.eq("WIN").sum())
    losses = int(result.eq("LOSS").sum())
    pushes = int(result.eq("PUSH").sum())
    voids = int(result.eq("VOID").sum())
    unresolved = int((~result.isin(["WIN", "LOSS", "PUSH", "VOID"])).sum())
    return {
        "graded": wins + losses,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "voids": voids,
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
    """Warn when a cumulative ledger has a calendar gap before the target slate.

    A calendar gap does not prove that prop rows are missing: a slate can have no
    supported markets, and the ledger merge is additive and deduplicated. The
    old hard block made it impossible to resume grading after such a day. Keep
    the gap visible, but allow the current export to be appended without
    deleting any loaded history; missing ledgers can still be backfilled later.
    """
    coverage = ledger_coverage_summary(ledger)
    latest = pd.to_datetime(coverage.get("end_date"), errors="coerce", utc=True)
    target = pd.to_datetime(target_date, errors="coerce", utc=True)
    gap_days = None
    if pd.notna(latest) and pd.notna(target):
        gap_days = int((target.normalize() - latest.normalize()).days)
    gap_detected = bool(
        gap_days is not None and gap_days > max(0, int(max_gap_days))
    )
    return {
        "latest_date": coverage.get("end_date"),
        "target_date": (
            target.strftime("%Y-%m-%d") if pd.notna(target) else None
        ),
        "gap_days": gap_days,
        "gap_detected": gap_detected,
        "requires_confirmation": False,
        "grading_blocked": False,
    }
