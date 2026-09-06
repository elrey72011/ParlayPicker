"""Grade every best-pick candidate against final scores and summarize rank quality.

The selected best-picks export contains only one row per game.  The candidate audit
contains every side/total alternative that survived the data-integrity gates.  Grading
that wider audit is what lets the app learn whether rank 1 is actually beating rank 2,
and whether a market family is carrying or hurting the card.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

import pandas as pd


_OUTCOMES = {"WIN", "LOSS", "PUSH"}
SELECTED_CANDIDATE_SCOPE = "SELECTED BEST AVAILABLE / SCORECARD"
ALTERNATIVE_CANDIDATE_SCOPE = "ALTERNATIVE CANDIDATE / DIAGNOSTIC"
MIN_SELECTED_DECISIONS_FOR_TREND = 50
MIN_SELECTED_SLATES_FOR_TREND = 5
_TREND_ALPHA_Z = 1.959963984540054
DEFAULT_CANDIDATE_RESULTS_RUNTIME_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "candidate_results"
    / "candidate_results_runtime.csv"
)


def _series(frame: pd.DataFrame, names: Iterable[str], default: object = pd.NA) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series(default, index=frame.index, dtype="object")


def _canonical_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _strict_bool(values: pd.Series) -> pd.Series:
    """Parse exported booleans without treating the string ``"False"`` as true."""

    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    normalized = values.astype("string").fillna("").str.strip().str.casefold()
    return normalized.isin({"true", "1", "yes", "y"})


def annotate_candidate_evaluation_scope(ledger: pd.DataFrame) -> pd.DataFrame:
    """Make selected-pick performance unambiguous in downloaded candidate data."""

    if ledger is None or ledger.empty:
        return pd.DataFrame() if ledger is None else ledger.copy()
    out = ledger.copy()
    selected = _strict_bool(_series(out, ("best_available_selected",), False))
    outcomes = _series(out, ("candidate_outcome", "Outcome"), "N/A")
    outcomes = outcomes.fillna("N/A").astype(str).str.upper()
    out["candidate_evaluation_scope"] = ALTERNATIVE_CANDIDATE_SCOPE
    out.loc[selected, "candidate_evaluation_scope"] = SELECTED_CANDIDATE_SCOPE
    out["selected_scorecard_row"] = selected
    out["selected_scorecard_decision"] = selected & outcomes.isin({"WIN", "LOSS"})
    out["selected_scorecard_outcome"] = outcomes.where(selected, "NOT SELECTED")
    return out


def selected_candidate_results(ledger: pd.DataFrame) -> pd.DataFrame:
    """Return the one-row-per-game Best Available scorecard export."""

    annotated = annotate_candidate_evaluation_scope(ledger)
    if annotated.empty:
        return annotated
    return annotated.loc[annotated["selected_scorecard_row"]].reset_index(drop=True)


def _event_key(frame: pd.DataFrame) -> pd.Series:
    league = _series(frame, ("league", "League"), "").map(_canonical_text)
    home = _series(frame, ("home_team", "Home Team", "Home"), "").map(_canonical_text)
    away = _series(frame, ("away_team", "Away Team", "Away"), "").map(_canonical_text)
    return league + "|" + away + "|" + home


def _slate_day(frame: pd.DataFrame) -> pd.Series:
    """Return a stable calendar day for candidate-ledger grouping."""

    raw = _series(frame, ("game_date", "slate_date"), "").fillna("").astype(str)
    parsed = pd.to_datetime(raw, errors="coerce", utc=True)
    day = parsed.dt.strftime("%Y-%m-%d")
    matchup = _series(frame, ("matchup_id",), "").fillna("").astype(str)
    matchup_day = matchup.str.extract(r"^(\d{4}-\d{2}-\d{2})", expand=False)
    raw_day = raw.str.extract(r"^(\d{4}-\d{2}-\d{2})", expand=False)
    return day.fillna(matchup_day).fillna(raw_day).fillna("").astype(str)


def _latest_event_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one auditable pregame snapshot per event when run IDs are available.

    A cumulative ledger may be assembled from several intraday downloads.  If a
    line or selected direction changed between runs, candidate-key de-duplication
    alone retained both versions and counted the same final score repeatedly.
    Prefer the latest export run for each event while preserving legacy rows that
    have no run ID at all.
    """

    if frame is None or frame.empty or "export_run_id" not in frame.columns:
        return frame.copy()
    out = frame.copy()
    event = _slate_day(out) + "|" + _event_key(out)
    run = _series(out, ("export_run_id",), "").fillna("").map(_canonical_text)
    has_run = run.ne("")
    event_has_run = has_run.groupby(event, dropna=False).transform("any")
    latest_run = run.groupby(event, dropna=False).transform("max")
    keep = (~event_has_run) | (has_run & run.eq(latest_run))
    return out.loc[keep].copy()


def _parse_number(text: object, pattern: str) -> float | None:
    match = re.search(pattern, str(text or ""), flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _grade_candidate(row: pd.Series) -> str:
    home_score = pd.to_numeric(pd.Series([row.get("actual_home_score")]), errors="coerce").iloc[0]
    away_score = pd.to_numeric(pd.Series([row.get("actual_away_score")]), errors="coerce").iloc[0]
    if pd.isna(home_score) or pd.isna(away_score):
        return "N/A"

    home_score = float(home_score)
    away_score = float(away_score)
    if home_score == 0 and away_score == 0:
        return "N/A"

    market_type = str(row.get("market_type", "") or "").strip().casefold()
    pick = str(row.get("best_pick", "") or "").strip()
    pick_folded = pick.casefold()

    if market_type.startswith("total_") or "over" in pick_folded or "under" in pick_folded:
        line = _parse_number(pick, r"(?:over|under)\s*(\d+(?:\.\d+)?)")
        if line is None:
            return "N/A"
        total = home_score + away_score
        side = "over" if market_type == "total_over" or "over" in pick_folded else "under"
        if total == line:
            return "PUSH"
        if side == "over":
            return "WIN" if total > line else "LOSS"
        return "WIN" if total < line else "LOSS"

    if market_type.startswith("spread_") or re.search(r"[+-]\d", pick):
        line = _parse_number(pick, r"([+-]\d+(?:\.\d+)?)")
        if line is None:
            return "N/A"
        if market_type == "spread_home":
            margin = home_score - away_score
        elif market_type == "spread_away":
            margin = away_score - home_score
        else:
            pick_team = re.sub(r"\s*[+-]\d+(?:\.\d+)?\s*$", "", pick).strip()
            pick_key = _canonical_text(pick_team)
            home_key = _canonical_text(row.get("home_team", row.get("Home", "")))
            away_key = _canonical_text(row.get("away_team", row.get("Away", "")))
            if pick_key == home_key:
                margin = home_score - away_score
            elif pick_key == away_key:
                margin = away_score - home_score
            else:
                return "N/A"
        result = margin + line
        return "PUSH" if result == 0 else ("WIN" if result > 0 else "LOSS")

    if market_type in {"moneyline_home", "ml_home"}:
        margin = home_score - away_score
    elif market_type in {"moneyline_away", "ml_away"}:
        margin = away_score - home_score
    else:
        pick_team = re.sub(r"\s+ml\s*$", "", pick, flags=re.IGNORECASE).strip()
        pick_key = _canonical_text(pick_team)
        home_key = _canonical_text(row.get("home_team", row.get("Home", "")))
        away_key = _canonical_text(row.get("away_team", row.get("Away", "")))
        if pick_key == home_key:
            margin = home_score - away_score
        elif pick_key == away_key:
            margin = away_score - home_score
        else:
            return "N/A"
    return "PUSH" if margin == 0 else ("WIN" if margin > 0 else "LOSS")


def _ledger_key(frame: pd.DataFrame) -> pd.Series:
    date_value = _series(frame, ("game_date", "slate_date"), "").fillna("").astype(str)
    matchup = _series(frame, ("matchup_id",), "").fillna("").astype(str)
    slate_identity = date_value.where(date_value.str.strip().ne(""), matchup)
    slate_identity = slate_identity.map(_canonical_text)
    market = _series(frame, ("market_type",), "").map(_canonical_text)
    pick = _series(frame, ("best_pick", "Pick Taken", "Best Pick"), "").map(_canonical_text)
    return slate_identity + "|" + _event_key(frame) + "|" + market + "|" + pick


def grade_candidate_audit(
    candidate_audit: pd.DataFrame,
    scored_picks: pd.DataFrame,
) -> pd.DataFrame:
    """Attach final scores to every candidate row and grade its offered line."""

    if candidate_audit is None or candidate_audit.empty:
        return pd.DataFrame() if candidate_audit is None else candidate_audit.copy()
    if scored_picks is None or scored_picks.empty:
        out = candidate_audit.copy()
        out["actual_home_score"] = pd.NA
        out["actual_away_score"] = pd.NA
        out["candidate_outcome"] = "N/A"
        out["candidate_graded"] = False
        out["candidate_ledger_key"] = _ledger_key(out)
        return annotate_candidate_evaluation_scope(out)

    out = candidate_audit.copy()
    rename = {}
    for source, target in (
        ("League", "league"),
        ("Home Team", "home_team"),
        ("Home", "home_team"),
        ("Away Team", "away_team"),
        ("Away", "away_team"),
        ("Pick Taken", "best_pick"),
        ("Best Pick", "best_pick"),
    ):
        if source in out.columns and target not in out.columns:
            rename[source] = target
    out = out.rename(columns=rename)
    out = out.drop(
        columns=[
            "actual_home_score",
            "actual_away_score",
            "candidate_outcome",
            "candidate_graded",
            "candidate_ledger_key",
        ],
        errors="ignore",
    )
    out["_candidate_event_key"] = _event_key(out)

    score_frame = scored_picks.copy()
    score_frame["_candidate_event_key"] = _event_key(score_frame)
    score_frame["actual_home_score"] = pd.to_numeric(
        _series(score_frame, ("actual_home_score", "Home Score", "home_score")),
        errors="coerce",
    )
    score_frame["actual_away_score"] = pd.to_numeric(
        _series(score_frame, ("actual_away_score", "Away Score", "away_score")),
        errors="coerce",
    )
    score_lookup = (
        score_frame[
            ["_candidate_event_key", "actual_home_score", "actual_away_score"]
        ]
        .dropna(subset=["actual_home_score", "actual_away_score"])
        .drop_duplicates("_candidate_event_key", keep="last")
    )

    out = out.merge(score_lookup, on="_candidate_event_key", how="left")
    out["candidate_outcome"] = out.apply(_grade_candidate, axis=1)
    out["candidate_graded"] = out["candidate_outcome"].isin(_OUTCOMES)
    out["candidate_ledger_key"] = _ledger_key(out)
    out = out.drop(columns=["_candidate_event_key"], errors="ignore")
    return annotate_candidate_evaluation_scope(out)


def merge_candidate_ledgers(
    current: pd.DataFrame,
    prior: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Append current grades to a prior ledger, replacing duplicate candidate rows."""

    frames = []
    if prior is not None and not prior.empty:
        frames.append(prior.copy())
    if current is not None and not current.empty:
        frames.append(current.copy())
    if not frames:
        return pd.DataFrame()

    ledger = pd.concat(frames, ignore_index=True, sort=False)
    ledger = _latest_event_snapshots(ledger)
    ledger["candidate_outcome"] = _series(
        ledger, ("candidate_outcome", "Outcome"), "N/A"
    ).fillna("N/A").astype(str).str.upper()
    ledger["candidate_graded"] = ledger["candidate_outcome"].isin(_OUTCOMES)
    ledger["candidate_ledger_key"] = _ledger_key(ledger)
    ledger = ledger.drop_duplicates(
        "candidate_ledger_key", keep="last"
    ).reset_index(drop=True)
    return annotate_candidate_evaluation_scope(ledger)


def load_candidate_results_ledger(
    path: Path | str = DEFAULT_CANDIDATE_RESULTS_RUNTIME_PATH,
    uploaded: object = None,
) -> pd.DataFrame | None:
    """Restore persisted history and layer one or more downloaded ledgers over it.

    Candidate results are monitoring evidence, so losing earlier slates can make a
    normal one-day result look like a selector regression.  The runtime copy keeps
    history across local app restarts, while accepting multiple uploads recovers it
    after a deployment or on another machine.
    """

    merged = pd.DataFrame()
    try:
        file_path = Path(path)
        if file_path.exists():
            merged = merge_candidate_ledgers(pd.read_csv(file_path), merged)

        uploads = (
            list(uploaded)
            if isinstance(uploaded, (list, tuple))
            else [uploaded] if uploaded is not None else []
        )
        for item in uploads:
            if item is None:
                continue
            if hasattr(item, "seek"):
                item.seek(0)
            frame = pd.read_csv(item)
            if hasattr(item, "seek"):
                item.seek(0)
            merged = merge_candidate_ledgers(frame, merged)
    except (OSError, ValueError, TypeError, pd.errors.ParserError):
        return None
    return merged if not merged.empty else None


def persist_candidate_results_ledger(
    ledger: pd.DataFrame | None,
    path: Path | str = DEFAULT_CANDIDATE_RESULTS_RUNTIME_PATH,
) -> bool:
    """Atomically save cumulative candidate history for restart recovery."""

    if not isinstance(ledger, pd.DataFrame) or ledger.empty:
        return False
    destination = Path(path)
    temporary: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".tmp",
            prefix=f".{destination.stem}-",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            ledger.to_csv(handle, index=False)
        temporary.replace(destination)
        return True
    except (OSError, ValueError, TypeError):
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        return False


def _summarize(ledger: pd.DataFrame, group_column: str, output_column: str) -> pd.DataFrame:
    columns = [
        output_column,
        "Candidates",
        "Graded",
        "Wins",
        "Losses",
        "Pushes",
        "Hit Rate",
        "Selected Rows",
        "Avg Ranking Score",
        "Avg EV",
    ]
    if ledger is None or ledger.empty or group_column not in ledger.columns:
        return pd.DataFrame(columns=columns)

    work = ledger.copy()
    work["candidate_outcome"] = _series(
        work, ("candidate_outcome", "Outcome"), "N/A"
    ).fillna("N/A").astype(str).str.upper()
    work["_group"] = work[group_column].where(work[group_column].notna(), "Unknown")
    work["_probability"] = pd.to_numeric(
        _series(work, ("selection_probability_used",)), errors="coerce"
    )
    work["_ev"] = pd.to_numeric(_series(work, ("expected_value",)), errors="coerce")
    selected = _series(work, ("best_available_selected",), False)
    work["_selected"] = _strict_bool(selected)

    rows = []
    for value, group in work.groupby("_group", dropna=False, sort=False):
        outcomes = group["candidate_outcome"]
        wins = int(outcomes.eq("WIN").sum())
        losses = int(outcomes.eq("LOSS").sum())
        pushes = int(outcomes.eq("PUSH").sum())
        decisions = wins + losses
        rows.append(
            {
                output_column: value,
                "Candidates": int(len(group)),
                "Graded": wins + losses + pushes,
                "Wins": wins,
                "Losses": losses,
                "Pushes": pushes,
                "Hit Rate": (wins / decisions) if decisions else 0.0,
                "Selected Rows": int(group["_selected"].sum()),
                "Avg Ranking Score": group["_probability"].mean(),
                "Avg EV": group["_ev"].mean(),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_candidate_performance(ledger: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return cumulative performance views by global rank, family rank, and market."""

    rank = _summarize(ledger, "best_available_rank", "Candidate Rank")
    if not rank.empty:
        rank["_sort"] = pd.to_numeric(rank["Candidate Rank"], errors="coerce")
        rank = rank.sort_values("_sort", na_position="last").drop(columns="_sort").reset_index(drop=True)

    family_rank = _summarize(ledger, "best_available_family_rank", "Family Rank")
    if not family_rank.empty:
        family_rank["_sort"] = pd.to_numeric(family_rank["Family Rank"], errors="coerce")
        family_rank = (
            family_rank.sort_values("_sort", na_position="last")
            .drop(columns="_sort")
            .reset_index(drop=True)
        )

    market_family = _summarize(ledger, "market_family", "Market Family")
    if not market_family.empty:
        market_family = market_family.sort_values("Market Family").reset_index(drop=True)

    return {
        "rank": rank,
        "family_rank": family_rank,
        "market_family": market_family,
    }


def _wilson_interval(wins: int, decisions: int) -> tuple[float | None, float | None]:
    if decisions <= 0:
        return None, None
    rate = wins / decisions
    z = _TREND_ALPHA_Z
    denominator = 1.0 + z * z / decisions
    center = (rate + z * z / (2.0 * decisions)) / denominator
    half_width = (
        z
        * math.sqrt(rate * (1.0 - rate) / decisions + z * z / (4.0 * decisions**2))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _poisson_binomial_lower_tail(probabilities: list[float], wins: int) -> float | None:
    """Exact P(X <= wins) for independent, non-identical pick probabilities."""

    if not probabilities:
        return None
    mass = [1.0]
    for probability in probabilities:
        next_mass = [0.0] * (len(mass) + 1)
        for count, value in enumerate(mass):
            next_mass[count] += value * (1.0 - probability)
            next_mass[count + 1] += value * probability
        mass = next_mass
    return float(sum(mass[: min(max(0, wins), len(mass) - 1) + 1]))


def summarize_selected_trend(ledger: pd.DataFrame) -> dict[str, object]:
    """Summarize whether selected-pick results support a regression claim.

    The status deliberately requires multiple slates and at least 50 decisions.
    A one-day record remains visible, but it cannot trigger a selector rewrite.
    """

    work = _latest_event_snapshots(ledger) if ledger is not None else pd.DataFrame()
    if work.empty:
        return {
            "status": "NO_DECISIONS",
            "decisions": 0,
            "wins": 0,
            "losses": 0,
            "slates": 0,
            "minimum_decisions": MIN_SELECTED_DECISIONS_FOR_TREND,
            "minimum_slates": MIN_SELECTED_SLATES_FOR_TREND,
        }

    outcomes = _series(work, ("candidate_outcome", "Outcome"), "N/A").fillna("N/A")
    outcomes = outcomes.astype(str).str.upper()
    selected = _strict_bool(_series(work, ("best_available_selected",), False))
    settled = selected & outcomes.isin({"WIN", "LOSS"})
    decisions = int(settled.sum())
    wins = int((settled & outcomes.eq("WIN")).sum())
    losses = decisions - wins
    slate_days = _slate_day(work).loc[settled]
    slates = int(slate_days[slate_days.ne("")].nunique())
    hit_rate = wins / decisions if decisions else None
    interval_low, interval_high = _wilson_interval(wins, decisions)

    probability = pd.to_numeric(
        _series(work, ("selection_probability_used",)), errors="coerce"
    ).loc[settled]
    # The selector's empirical blend ranks alternatives; it is not a calibrated
    # forecast and must not power an expected-wins significance test.
    ranking_only = _series(work, ("selection_probability_source",), "").astype(
        "string"
    ).str.strip().str.casefold().eq("empirical_bucket_blend").fillna(False)
    probability = probability.mask(ranking_only.loc[settled])
    usable_probability = probability[probability.between(0.0, 1.0, inclusive="both")]
    expected_decisions = int(usable_probability.count())
    expected_wins = float(usable_probability.sum()) if expected_decisions else None
    expected_hit_rate = (
        float(usable_probability.mean()) if expected_decisions else None
    )
    lower_tail = (
        _poisson_binomial_lower_tail(usable_probability.tolist(), wins)
        if expected_decisions == decisions and decisions
        else None
    )

    if decisions == 0:
        status = "NO_DECISIONS"
        reason = "No selected candidate decisions are graded."
    elif (
        decisions < MIN_SELECTED_DECISIONS_FOR_TREND
        or slates < MIN_SELECTED_SLATES_FOR_TREND
    ):
        status = "INSUFFICIENT_HISTORY"
        reason = (
            "Current results are a monitoring sample, not enough independent "
            "history to diagnose a selector regression."
        )
    elif expected_decisions != decisions:
        status = "INSUFFICIENT_EXPECTATION_DATA"
        reason = (
            "Some selected decisions have ranking-only scores or lack a valid final selection probability, "
            "so expected-versus-observed regression testing is unavailable."
        )
    elif (
        expected_hit_rate is not None
        and hit_rate is not None
        and hit_rate < expected_hit_rate
        and lower_tail is not None
        and lower_tail < 0.05
    ):
        status = "REGRESSION_SIGNAL"
        reason = (
            "Observed selected-pick accuracy is below the model expectation with "
            "a lower-tail probability under 5%."
        )
    else:
        status = "WITHIN_EXPECTED_VARIANCE"
        reason = (
            "Observed selected-pick accuracy remains within ordinary sampling "
            "variation around the exported probabilities."
        )

    return {
        "status": status,
        "reason": reason,
        "decisions": decisions,
        "wins": wins,
        "losses": losses,
        "slates": slates,
        "hit_rate": hit_rate,
        "confidence_interval_low": interval_low,
        "confidence_interval_high": interval_high,
        "expected_decisions": expected_decisions,
        "expected_wins": expected_wins,
        "expected_hit_rate": expected_hit_rate,
        "observed_minus_expected": (
            hit_rate - expected_hit_rate
            if hit_rate is not None and expected_hit_rate is not None
            else None
        ),
        "lower_tail_probability": lower_tail,
        "minimum_decisions": MIN_SELECTED_DECISIONS_FOR_TREND,
        "minimum_slates": MIN_SELECTED_SLATES_FOR_TREND,
    }
