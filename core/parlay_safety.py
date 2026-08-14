"""Conservative safety and pricing helpers for production parlays.

This module is deliberately pure and dependency-light. It does not claim to
make a model predictive; it prevents the parlay layer from treating missing,
fallback, stale, or weakly priced rows as proven value.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

import pandas as pd


def american_to_decimal(odds: Any) -> float | None:
    try:
        value = float(odds)
    except (TypeError, ValueError):
        return None
    if value == 0:
        return None
    return 1.0 + (value / 100.0 if value > 0 else 100.0 / abs(value))


def break_even_probability(odds: Any) -> float | None:
    decimal = american_to_decimal(odds)
    return 1.0 / decimal if decimal and decimal > 1.0 else None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def row_is_untrusted(row: Any) -> bool:
    """Return True when the row is not safe for production parlay use."""
    get = row.get if hasattr(row, "get") else lambda key, default=None: default

    for key in (
        "is_fallback",
        "fallback_used",
        "fallback_triggered",
        "degraded_feature_subset_flag",
        "model_unavailable",
        "stale_data_flag",
        "game_already_started_flag",
    ):
        if _truthy(get(key)):
            return True

    for key in ("prediction_note", "model_note", "run_health_warning", "status_blocker_stage"):
        value = str(get(key, "") or "").strip().lower()
        if any(token in value for token in ("fallback", "model unavailable", "error fallback", "stale", "started")):
            return True

    status = str(get("Pick_Status", "") or "").strip()
    if status in {"Missing Line", "No Play", "Fallback / Low Confidence"}:
        return True

    if "production_eligible" in row and not _truthy(get("production_eligible")):
        return True

    return False


def production_candidate_mask(df: pd.DataFrame, *, min_probability: float = 0.55,
                              min_edge: float = 0.02) -> pd.Series:
    """Strict candidate mask for parlay legs.

    The mask intentionally requires explicit evidence. Missing columns are treated
    as unsafe instead of being silently synthesized.
    """
    if df is None or df.empty:
        return pd.Series(dtype=bool)

    mask = pd.Series(True, index=df.index)
    if "Pick_Status" in df.columns:
        mask &= df["Pick_Status"].astype(str).eq("Actionable")
    if "odds_american" in df.columns:
        mask &= pd.to_numeric(df["odds_american"], errors="coerce").ne(0)
        mask &= pd.to_numeric(df["odds_american"], errors="coerce").notna()
    elif "decimal_odds" in df.columns:
        mask &= pd.to_numeric(df["decimal_odds"], errors="coerce").gt(1.0)
    else:
        return pd.Series(False, index=df.index)

    prob_col = next((c for c in (
        "empirical_win_probability",
        "effective_win_probability",
        "calibrated_probability",
        "WinProbability",
    ) if c in df.columns), None)
    if prob_col is None:
        return pd.Series(False, index=df.index)

    probs = pd.to_numeric(df[prob_col], errors="coerce")
    mask &= probs.ge(float(min_probability))
    if "edge" in df.columns:
        mask &= pd.to_numeric(df["edge"], errors="coerce").ge(float(min_edge))
    elif "empirical_edge" in df.columns:
        mask &= pd.to_numeric(df["empirical_edge"], errors="coerce").ge(float(min_edge))
    else:
        return pd.Series(False, index=df.index)

    mask &= ~df.apply(row_is_untrusted, axis=1)
    return mask


def _normal_tokens(value: Any) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", str(value or "").lower())
    generic = {"new", "los", "las", "san", "st", "saint", "city", "blue", "red", "white", "sox", "moneyline", "spread", "total", "over", "under", "point", "points", "run", "runs", "goal", "goals", "game", "team"}
    return {t for t in tokens if len(t) >= 4 and t not in generic}


def _first_present(row: Any, keys: tuple[str, ...]) -> str:
    get = row.get if hasattr(row, "get") else lambda key, default=None: default
    for key in keys:
        value = get(key, "")
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            text = str(value).strip()
            if text and text.casefold() not in {"nan", "none", "<na>"}:
                return text
    return ""


def _team_key(value: Any) -> str:
    return "".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _matchup_team_key(row: Any) -> tuple[str, str] | None:
    home = _first_present(row, ("home_team", "Home", "home"))
    away = _first_present(row, ("away_team", "Away", "away"))
    if not home or not away:
        return None
    # Team order is intentionally ignored. Provider rows occasionally reverse
    # home/away labels while still describing the same event.
    return tuple(sorted((_team_key(home), _team_key(away))))


def shares_game(left: Any, right: Any) -> bool:
    """Conservative same-game detector for the one-leg-per-game invariant.

    Exact event IDs are strongest, but matching team pairs also count as the same
    game when provider-specific IDs differ. Text matching remains the fallback for
    mixed game/prop boards that only retain a human-readable matchup.
    """
    lget = left.get if hasattr(left, "get") else lambda key, default=None: default
    rget = right.get if hasattr(right, "get") else lambda key, default=None: default

    id_keys = ("matchup_id", "game_id", "GameID", "event_id", "Game", "game")
    left_id = _first_present(left, id_keys)
    right_id = _first_present(right, id_keys)
    if left_id and right_id and left_id.casefold() == right_id.casefold():
        return True

    left_teams = _matchup_team_key(left)
    right_teams = _matchup_team_key(right)
    if left_teams is not None and right_teams is not None:
        return left_teams == right_teams

    # Distinct explicit IDs override generic market vocabulary when neither row
    # contains enough team identity to prove an overlap.
    if left_id and right_id:
        return False

    text_keys = (
        "matchup", "detail", "home_team", "away_team", "Home", "Away",
        "best_pick", "pick", "pitcher", "player",
    )
    left_text = " ".join(str(lget(k, "") or "") for k in text_keys)
    right_text = " ".join(str(rget(k, "") or "") for k in text_keys)
    return bool(_normal_tokens(left_text) & _normal_tokens(right_text))


def has_unique_games(legs: Any) -> bool:
    """Return True only when every parlay leg belongs to a different game."""
    if isinstance(legs, pd.DataFrame):
        rows = [legs.iloc[index] for index in range(len(legs))]
    else:
        rows = [] if legs is None else list(legs)
    return not any(
        shares_game(rows[left], rows[right])
        for left in range(len(rows))
        for right in range(left + 1, len(rows))
    )


def conservative_joint_probability(probabilities: Iterable[float], *, haircut: float = 0.97) -> float:
    """Independent product with a conservative model-risk haircut.

    Correlation is not inferred as a made-up probability boost. Same-game
    combinations should be rejected by the caller; unknown dependence is paid
    for with a haircut.
    """
    result = 1.0
    for probability in probabilities:
        result *= float(probability)
    return max(0.0, min(1.0, result * float(haircut)))


def parlay_expected_value(joint_probability: float, decimal_odds: Any) -> float | None:
    try:
        decimal = float(decimal_odds)
    except (TypeError, ValueError):
        return None
    if decimal <= 1.0:
        return None
    return float(joint_probability * decimal - 1.0)
