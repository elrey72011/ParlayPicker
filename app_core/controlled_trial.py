"""Owner-authorized, tightly capped wager trials for unvalidated models.

Controlled trials are deliberately separate from the validated live-wager
contract. They may create a small, explicit recommendation while prospective
validation is accumulated, but they never set ``production_eligible`` or
``wager_approved`` and they never place a sportsbook order.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from app_core.quote_freshness import QUOTE_MAX_AGE_SECONDS
from core.market_policy import production_market
from core.wager_decisions import aware, decimal_price, finite
from core.price_value import price_value
from core.exposure_ledger import digest
from app_core.trial_authority import consent_status, exposure_status, reserve_recommendation


VERSION = "controlled-trial-v2"
LEGACY_VERSION = "controlled-trial-v1"
AUTHORITY = "OWNER_AUTHORIZED_UNVALIDATED_TRIAL"
MIN_PRICE_EDGE = 0.02
MIN_EXPECTED_VALUE = 0.0
MAX_BANKROLL_FRACTION_PER_PICK = 0.0025
MAX_DOLLARS_PER_PICK = 5.0
MAX_PICKS_PER_SLATE = 2
MAX_REVIEW_CANDIDATES_PER_SLATE = 6
FRACTIONAL_KELLY = 0.125
MIN_AMERICAN_ODDS = -200.0
MAX_AMERICAN_ODDS = 200.0

PUBLIC_FIELDS = tuple(
    "version authority candidate_id game_id sport market_type selection line odds "
    "sportsbook quote_timestamp start probability_source source_probability_semantics probability_semantics "
    "estimated_probability push_probability loss_probability break_even_probability "
    "estimated_price_edge estimated_expected_value gemini_review_status "
    "gemini_stake_multiplier identity_verified quote_verified quote_fresh "
    "data_quality_status trial_eligible bankroll_fraction recommended_bet_amount "
    "reason".split()
)
LEGACY_PUBLIC_FIELDS = tuple(
    "version authority candidate_id game_id sport market_type selection line odds "
    "sportsbook quote_timestamp start estimated_probability break_even_probability "
    "estimated_price_edge estimated_expected_value gemini_review_status "
    "gemini_stake_multiplier identity_verified quote_verified quote_fresh "
    "data_quality_status trial_eligible bankroll_fraction recommended_bet_amount "
    "reason".split()
)


def public_fields_for_version(contract: dict[str, Any]) -> tuple[str, ...]:
    """Only for schema/read compatibility; v1 cannot be issued by this runtime."""
    version = contract.get("version") if isinstance(contract, dict) else None
    if version == VERSION:
        return PUBLIC_FIELDS
    if version == LEGACY_VERSION:
        return LEGACY_PUBLIC_FIELDS
    raise ValueError("Unknown controlled-trial contract version")


def enabled(*, now: datetime | None = None) -> bool:
    """An environment switch alone never establishes trial authority."""
    switch = str(os.getenv("PARLAYPICKER_CONTROLLED_TRIALS_ENABLED", "0")).strip().lower()
    return switch in {"1", "true", "yes", "on"} and consent_status(now=now)[1] == "AUTHORIZED"


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _text(row: dict | pd.Series, *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        if str(value).strip():
            return str(value).strip()
    return ""


def _number(row: dict | pd.Series, *names: str) -> float | None:
    for name in names:
        value = finite(row.get(name))
        if value is not None:
            return value
    return None


def _now(value: datetime | pd.Timestamp | None = None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if value.tzinfo is None:
        raise ValueError("Controlled-trial decision time requires a timezone")
    return value.astimezone(timezone.utc)


def _price_allowed(odds: float | None) -> bool:
    if odds is None:
        return False
    return MIN_AMERICAN_ODDS <= odds <= -100.0 or 100.0 <= odds <= MAX_AMERICAN_ODDS


def _optional_truth(row: dict | pd.Series, name: str) -> bool | None:
    """Return an explicit boolean without treating a missing fact as false."""
    if name not in row:
        return None
    value = row.get(name)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return None


def _expected_pick(row: dict | pd.Series, market: str, line: float | None) -> str:
    if line is None:
        return ""
    if market == "spread_home":
        return f"{_text(row, 'home_team')} {line:+.1f}".strip()
    if market == "spread_away":
        return f"{_text(row, 'away_team')} {line:+.1f}".strip()
    if market == "total_over":
        return f"Over {line:.1f}"
    if market == "total_under":
        return f"Under {line:.1f}"
    return ""


def attest_candidate_integrity(frame: pd.DataFrame) -> pd.DataFrame:
    """Certify exact trial rows from their immutable provider quote evidence.

    The expanded ranking pool is created before the display winner receives its
    final line/event flags. Trial alternatives therefore need the equivalent
    checks against their own exact quote. Explicit upstream failures remain
    vetoes; this function only fills facts that were absent from the pool.
    """
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()

    from app_core.prediction_evidence import matching_quotes

    out = frame.copy()
    line_results: list[bool] = []
    identity_results: list[bool] = []
    for _, row in out.iterrows():
        market = _text(row, "market_type").lower()
        odds = _number(row, "odds_american", "american_odds")
        line = _number(
            row,
            "line",
            "market_line_used",
            "total_line" if market.startswith("total") else "spread_line",
        )
        matches = matching_quotes(row)
        quote = matches[0] if len(matches) == 1 else None
        quote_line = finite(quote.get("point")) if quote else None
        quote_price = finite(quote.get("price")) if quote else None
        quote_bound = _truth(row.get("quote_binding_verified"))
        expected_pick = _expected_pick(row, market, line)
        actual_pick = _text(row, "best_pick", "selection")
        rejected_source = any(
            _text(row, name).lower().startswith("rejected")
            for name in ("line_source", "market_line_source", "odds_source")
        )
        fuzzy_identity = "fuzzy" in _text(row, "orientation_source").lower()

        computed_line = bool(
            production_market(market)
            and quote_bound
            and quote is not None
            and line is not None
            and quote_line is not None
            and odds is not None
            and quote_price is not None
            and math.isclose(line, quote_line, abs_tol=1e-8)
            and math.isclose(odds, quote_price, abs_tol=1e-8)
            and expected_pick
            and actual_pick.casefold() == expected_pick.casefold()
            and not rejected_source
        )

        provider_event_id = _text(quote or {}, "provider_event_id")
        provider_namespace = _text(quote or {}, "provider_namespace")
        bound_event_id = _text(row, "provider_event_id")
        bound_namespace = _text(row, "provider_namespace")
        try:
            supplied_quotes = json.loads(row.get("provider_quotes") or "[]")
        except (TypeError, ValueError):
            supplied_quotes = []
        provider_conflict = any(
            isinstance(item, dict)
            and _text(item, "provider_namespace") == provider_namespace
            and _text(item, "provider_event_id")
            and _text(item, "provider_event_id") != provider_event_id
            for item in supplied_quotes
        ) if isinstance(supplied_quotes, list) and provider_namespace else True
        computed_identity = bool(
            computed_line
            and provider_event_id
            and provider_namespace
            and bound_event_id == provider_event_id
            and bound_namespace == provider_namespace
            and _text(row, "game_id", "matchup_id")
            and _text(row, "home_team")
            and _text(row, "away_team")
            and not provider_conflict
            and not fuzzy_identity
        )

        explicit_line = _optional_truth(row, "line_consistency_flag")
        explicit_identity = [
            value for value in (
                _optional_truth(row, "identity_verified"),
                _optional_truth(row, "line_event_identity_match_flag"),
            ) if value is not None
        ]
        line_results.append(computed_line and explicit_line is not False)
        identity_results.append(computed_identity and False not in explicit_identity)

    out["line_consistency_flag"] = line_results
    out["line_event_identity_match_flag"] = identity_results
    out["identity_verified"] = identity_results
    return out


def _metrics(row: dict | pd.Series) -> tuple[float | None, float | None, float | None, float | None, float | None, float | None, str, str]:
    source = _text(row, "calibration_probability_field") or "calibrated_probability"
    if source not in {"calibrated_probability", "selection_probability_used", "sport_calibrated_probability"}:
        source = ""
    probability = _number(row, source) if source else None
    odds = _number(row, "odds_american", "american_odds")
    decimal = decimal_price(odds)
    semantics = _text(row, "probability_semantics")
    line = _number(row, "line", "market_line_used", "total_line" if _text(row, "market_type").startswith("total") else "spread_line")
    explicit_push = _number(row, "push_probability")
    no_push_line = line is not None and math.isclose(abs(line) % 1, 0.5, abs_tol=1e-8)
    if semantics == "win_conditional_on_decision":
        push = explicit_push if explicit_push is not None else (0.0 if no_push_line else None)
        win = probability * (1 - push) if probability is not None and push is not None else None
    elif semantics == "win_unconditional_with_push":
        push = explicit_push
        win = probability
    else:
        push = None
        win = None
    pricing = price_value(win, push, decimal) if decimal is not None else None
    supplied_loss = _number(row, "loss_probability_unconditional")
    if pricing and supplied_loss is not None and not math.isclose(supplied_loss, pricing["p_loss"], abs_tol=1e-8):
        pricing = None
    if not pricing or win is None or not 0 < win < 1:
        return None, None, None, None, None, None, semantics, source
    return (pricing["p_win"], pricing["p_push"], pricing["p_loss"],
            pricing["break_even"], pricing["edge"], pricing["expected_value"], semantics, source)


def _trial_input_hash(row: dict | pd.Series) -> str:
    """Bind reviewed probability to the exact event, line, quote and source."""
    market = _text(row, "market_type")
    source = _text(row, "calibration_probability_field") or "calibrated_probability"
    return digest(dict(
        candidate_id=_text(row, "candidate_id"), game_id=_text(row, "game_id", "matchup_id"),
        sport=_text(row, "sport", "league"), market_type=market,
        selection=_text(row, "selection", "best_pick"),
        line=_number(row, "line", "market_line_used", "total_line" if market.startswith("total") else "spread_line"),
        odds=_number(row, "odds_american", "american_odds"),
        sportsbook=_text(row, "quote_bookmaker", "book", "odds_source", "sportsbook"),
        quote_timestamp=_text(row, "quote_time", "odds_recorded_at", "quote_timestamp"),
        probability_source=source, source_probability=_number(row, source),
        probability_semantics=_text(row, "probability_semantics"),
        push_probability=_number(row, "push_probability"),
        loss_probability_unconditional=_number(row, "loss_probability_unconditional"),
    ))


def evaluate_candidates(frame: pd.DataFrame, *, now: datetime | pd.Timestamp | None = None) -> pd.DataFrame:
    """Annotate every candidate with the controlled-trial deterministic gate."""
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    decision_time = _now(now)
    out = frame.copy()
    eligible: list[bool] = []
    reasons: list[str] = []
    probabilities: list[float | None] = []
    pushes: list[float | None] = []
    losses: list[float | None] = []
    semantics_values: list[str] = []
    probability_sources: list[str] = []
    input_hashes: list[str] = []
    break_evens: list[float | None] = []
    edges: list[float | None] = []
    evs: list[float | None] = []

    for _, row in out.iterrows():
        probability, push, loss, break_even, edge, calculated_ev, source_semantics, probability_source = _metrics(row)
        producer_ev = _number(row, "expected_value", "estimated_expected_value")
        odds = _number(row, "odds_american", "american_odds")
        market = _text(row, "market_type").lower()
        line = _number(
            row,
            "line",
            "market_line_used",
            "total_line" if market.startswith("total") else "spread_line",
        )
        start = aware(_text(row, "start", "game_start_utc", "commence_time", "game_time_est"))
        quote = aware(_text(row, "quote_time", "odds_recorded_at", "quote_timestamp"))
        quote_facts = [_optional_truth(row, name) for name in
                       ("quote_verified", "exact_quote_verified", "quote_binding_verified")]
        quote_verified = True in quote_facts and False not in quote_facts
        identity_facts = [_optional_truth(row, name) for name in
                          ("identity_verified", "line_event_identity_match_flag")]
        identity_verified = (True in identity_facts and False not in identity_facts
                             and bool(_text(row, "game_id", "matchup_id")))
        line_verified = _truth(row.get("line_consistency_flag"))
        model_status = _text(row, "model_status").casefold()
        stats_quality = _text(row, "stats_quality").upper()
        data_quality = _text(row, "data_quality_status").upper()
        degraded = (
            _truth(row.get("degraded_feature_subset_flag"))
            or _truth(row.get("used_stale_features"))
            or _truth(row.get("suspicious_data_flag"))
            or _truth(row.get("critical_feature_error"))
            or model_status in {"statistical fallback", "neutral fallback", "model failure"}
            or stats_quality in {"FALLBACK", "MISSING"}
            or data_quality in {"DEGRADED", "UNVERIFIED", "MISSING", "ERROR"}
            or (market.startswith("total") and _text(row, "total_input_status").upper() == "DEGRADED")
        )
        league = _text(row, "league", "sport").upper()
        nfl_context_complete = (
            league != "NFL"
            or (
                _text(row, "nfl_context_status").casefold() == "complete"
                and _text(row, "ml_probability_source").casefold() == "score-distribution-v1:nfl"
                and _truth(row.get("nfl_context_model_used"))
            )
        )

        blockers: list[str] = []
        if not production_market(market):
            blockers.append("unsupported market")
        if line is None:
            blockers.append("missing exact line")
        if not _price_allowed(odds):
            blockers.append("price outside controlled-trial range")
        if not identity_verified:
            blockers.append("event identity not verified")
        if not line_verified:
            blockers.append("line identity not verified")
        if not quote_verified:
            blockers.append("exact quote not verified")
        if start is None or start <= decision_time:
            blockers.append("game started or start unavailable")
        if quote is None or not 0 <= (decision_time - quote).total_seconds() <= QUOTE_MAX_AGE_SECONDS:
            blockers.append("quote stale or timestamp unavailable")
        if degraded:
            blockers.append("degraded or critical input state")
        if not nfl_context_complete:
            blockers.append("NFL recent-result and injury context incomplete")
        if probability is None or break_even is None or edge is None or calculated_ev is None:
            blockers.append("priced probability or push semantics unavailable")
        elif edge < MIN_PRICE_EDGE:
            blockers.append(f"price edge below {MIN_PRICE_EDGE:.1%}")
        if producer_ev is None or producer_ev <= MIN_EXPECTED_VALUE:
            blockers.append("producer expected value is not positive")
        if calculated_ev is None or calculated_ev <= MIN_EXPECTED_VALUE:
            blockers.append("recomputed expected value is not positive")

        eligible.append(not blockers)
        reasons.append("Controlled-trial deterministic checks passed" if not blockers else "; ".join(blockers))
        probabilities.append(probability)
        pushes.append(push)
        losses.append(loss)
        semantics_values.append(source_semantics)
        probability_sources.append(probability_source)
        input_hashes.append(_trial_input_hash(row))
        break_evens.append(break_even)
        edges.append(edge)
        evs.append(calculated_ev)

    out["controlled_trial_deterministic_eligible"] = eligible
    out["controlled_trial_gate_reason"] = reasons
    out["controlled_trial_estimated_probability"] = probabilities
    out["controlled_trial_push_probability"] = pushes
    out["controlled_trial_loss_probability"] = losses
    out["controlled_trial_source_probability_semantics"] = semantics_values
    out["controlled_trial_probability_source"] = probability_sources
    out["controlled_trial_input_hash"] = input_hashes
    out["controlled_trial_break_even_probability"] = break_evens
    out["controlled_trial_estimated_price_edge"] = edges
    out["controlled_trial_estimated_expected_value"] = evs
    return out


def select_review_candidates(
    frame: pd.DataFrame,
    *,
    now: datetime | pd.Timestamp | None = None,
    max_picks: int = MAX_REVIEW_CANDIDATES_PER_SLATE,
) -> pd.DataFrame:
    """Return a bounded review pool; final picks have a separate lower cap."""
    evaluated = evaluate_candidates(frame, now=now)
    if evaluated.empty or not enabled(now=_now(now)):
        return evaluated.iloc[0:0].copy()
    eligible = evaluated[evaluated["controlled_trial_deterministic_eligible"]].copy()
    if eligible.empty:
        return eligible
    eligible["_trial_game"] = eligible.apply(
        lambda row: _text(row, "game_id", "matchup_id")
        or "|".join((_text(row, "league", "sport"), _text(row, "away_team"), _text(row, "home_team"))),
        axis=1,
    )
    eligible["_trial_candidate"] = eligible.apply(
        lambda row: _text(row, "candidate_id", "best_pick"), axis=1
    )
    eligible = eligible.sort_values(
        ["controlled_trial_estimated_expected_value", "controlled_trial_estimated_price_edge", "controlled_trial_estimated_probability", "_trial_candidate"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    eligible = eligible.drop_duplicates("_trial_game", keep="first").head(max(0, int(max_picks)))
    eligible["controlled_trial_candidate"] = True
    return eligible.drop(columns=["_trial_game", "_trial_candidate"])


def _contract(row: dict | pd.Series, bankroll: float, now: datetime) -> dict[str, Any]:
    probability = _number(row, "controlled_trial_estimated_probability")
    push = _number(row, "controlled_trial_push_probability")
    loss = _number(row, "controlled_trial_loss_probability")
    break_even = _number(row, "controlled_trial_break_even_probability")
    edge = _number(row, "controlled_trial_estimated_price_edge")
    ev = _number(row, "controlled_trial_estimated_expected_value")
    odds = _number(row, "odds_american", "american_odds")
    decimal = decimal_price(odds)
    multiplier = _number(row, "gemini_stake_multiplier") or 0.0
    pricing = price_value(probability, push, decimal) if decimal is not None else None
    fraction = min(
        MAX_BANKROLL_FRACTION_PER_PICK,
        (pricing["full_kelly"] if pricing else 0.0) * FRACTIONAL_KELLY,
    ) * multiplier
    amount = math.floor(max(0.0, min(bankroll * fraction, MAX_DOLLARS_PER_PICK)) * 100 + 1e-9) / 100
    fraction = amount / bankroll if bankroll > 0 else 0.0
    quote = aware(_text(row, "quote_time", "odds_recorded_at", "quote_timestamp"))
    start = aware(_text(row, "start", "game_start_utc", "commence_time", "game_time_est"))
    eligible = bool(
        row.get("controlled_trial_deterministic_eligible") is True
        and _text(row, "controlled_trial_input_hash") == _trial_input_hash(row)
        and _text(row, "gemini_review_status").upper() == "APPROVE"
        and probability is not None
        and pricing is not None and loss is not None
        and math.isclose(loss, pricing["p_loss"], abs_tol=1e-9)
        and break_even is not None and math.isclose(break_even, pricing["break_even"], abs_tol=1e-9)
        and edge is not None and math.isclose(edge, pricing["edge"], abs_tol=1e-9)
        and ev is not None and math.isclose(ev, pricing["expected_value"], abs_tol=1e-9)
        and edge is not None and edge >= MIN_PRICE_EDGE
        and ev is not None and ev > 0
        and quote is not None and 0 <= (now - quote).total_seconds() <= QUOTE_MAX_AGE_SECONDS
        and start is not None and start > now
        and amount > 0
    )
    if not eligible:
        amount = 0.0
        fraction = 0.0
    line = _number(
        row,
        "line",
        "market_line_used",
        "total_line" if _text(row, "market_type").startswith("total") else "spread_line",
    )
    book = _text(row, "quote_bookmaker", "book", "odds_source", "sportsbook")
    reason = (
        "Owner-authorized controlled trial: unvalidated model, exact live price, positive value, "
        "Gemini approved, and stake capped at 0.25% of bankroll / $5"
        if eligible else _text(row, "gemini_gate_reason", "controlled_trial_gate_reason") or "Controlled trial held"
    )
    return {
        "version": VERSION,
        "authority": AUTHORITY,
        "candidate_id": _text(row, "candidate_id"),
        "game_id": _text(row, "game_id", "matchup_id"),
        "sport": _text(row, "sport", "league"),
        "market_type": _text(row, "market_type"),
        "selection": _text(row, "selection", "best_pick"),
        "line": line,
        "odds": odds,
        "sportsbook": book,
        "quote_timestamp": quote.isoformat() if quote else None,
        "start": start.isoformat() if start else None,
        "probability_source": _text(row, "controlled_trial_probability_source"),
        "source_probability_semantics": _text(row, "controlled_trial_source_probability_semantics"),
        "probability_semantics": "win_unconditional_with_push",
        "estimated_probability": probability,
        "push_probability": push,
        "loss_probability": loss,
        "break_even_probability": break_even,
        "estimated_price_edge": edge,
        "estimated_expected_value": ev,
        "gemini_review_status": _text(row, "gemini_review_status").upper(),
        "gemini_stake_multiplier": multiplier,
        "identity_verified": (True in [_optional_truth(row, k) for k in ("identity_verified", "line_event_identity_match_flag")]
                              and False not in [_optional_truth(row, k) for k in ("identity_verified", "line_event_identity_match_flag")]),
        "quote_verified": (True in [_optional_truth(row, k) for k in ("quote_verified", "exact_quote_verified", "quote_binding_verified")]
                           and False not in [_optional_truth(row, k) for k in ("quote_verified", "exact_quote_verified", "quote_binding_verified")]),
        "quote_fresh": bool(quote and 0 <= (now - quote).total_seconds() <= QUOTE_MAX_AGE_SECONDS),
        "data_quality_status": "NO_DEGRADED_FLAGS" if not _truth(row.get("degraded_feature_subset_flag")) else "DEGRADED",
        "trial_eligible": eligible,
        "bankroll_fraction": fraction,
        "recommended_bet_amount": amount,
        "reason": reason,
    }


def _clear_prior_trial(final: pd.DataFrame) -> pd.DataFrame:
    """A saved trial label never survives a fresh authority decision by itself."""
    out = final.copy()
    if "controlled_trial_contract" not in out:
        return out
    for idx, row in out.iterrows():
        if not isinstance(row.get("controlled_trial_contract"), dict):
            continue
        out.at[idx, "controlled_trial_contract"] = None
        out.at[idx, "controlled_trial_eligible"] = False
        out.at[idx, "controlled_trial_stake"] = 0.0
        if not (isinstance(row.get("wager_contract"), dict)
                and row["wager_contract"].get("production_eligible") is True):
            out.at[idx, "Pick_Status"] = "No Play"
            out.at[idx, "commercial_tier"] = "RESEARCH"
            out.at[idx, "Export_Scope"] = "RESEARCH ONLY"
            out.at[idx, "Wager_Instruction"] = "PASS — controlled-trial authority must be rechecked"
            out.at[idx, "Status_Reason"] = "Controlled-trial authority must be rechecked"
            out.at[idx, "qualification_reason"] = "Controlled-trial authority must be rechecked"
    return out


def apply_trials(
    final: pd.DataFrame,
    reviewed: pd.DataFrame,
    bankroll: float,
    *,
    now: datetime | pd.Timestamp | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Overlay eligible trials only where no validated wager already exists."""
    if final is None or final.empty:
        return final
    final = _clear_prior_trial(final)
    if reviewed is None or reviewed.empty:
        return final
    decision_time = _now(now)
    if not str(os.getenv("PARLAYPICKER_CONTROLLED_TRIALS_ENABLED", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        if diagnostics is not None:
            diagnostics["controlled_trial_authorization_status"] = "TRIALS_DISABLED"
        return final
    consent, consent_reason = consent_status(now=decision_time)
    if diagnostics is not None:
        diagnostics["controlled_trial_authorization_status"] = consent_reason
    if consent is None:
        return final
    bank = finite(bankroll)
    if bank is None or bank <= 0:
        if diagnostics is not None:
            diagnostics["controlled_trial_allocation_status"] = "BANKROLL_UNAVAILABLE"
        return final
    exposure, external_recommended, exposure_reason = exposure_status(bank, now=decision_time)
    if diagnostics is not None:
        diagnostics["controlled_trial_allocation_status"] = exposure_reason
        diagnostics["controlled_trial_exposure_snapshot_id"] = exposure.get("snapshot_hash") if exposure else None
    if exposure is None:
        return final
    out = final.copy()
    # Pandas cannot safely assign a dict into a column that does not exist: it
    # tries to align the dict as a Series.  Declare the object-valued contract
    # slot before overlaying an eligible trial.
    if "controlled_trial_contract" not in out.columns:
        out["controlled_trial_contract"] = pd.Series(
            [None] * len(out), index=out.index, dtype=object
        )
    used: set[str] = set()
    candidate_flags = pd.Series(
        reviewed.get("controlled_trial_candidate", False), index=reviewed.index
    ).astype("string").str.strip().str.lower().isin({"true", "1", "yes"})
    review_status = pd.Series(
        reviewed.get("gemini_review_status", ""), index=reviewed.index
    ).astype(str).str.upper()
    approved_reviews = reviewed[candidate_flags & review_status.eq("APPROVE")].copy()
    approved_reviews = approved_reviews.sort_values(
        ["controlled_trial_estimated_expected_value", "controlled_trial_estimated_price_edge"],
        ascending=[False, False],
        kind="mergesort",
    )
    review_columns = [name for name in approved_reviews.columns if name.startswith("gemini_")]
    allocation_reasons: dict[str, int] = {}

    for _, candidate in approved_reviews.iterrows():
        if len(used) >= MAX_PICKS_PER_SLATE:
            allocation_reasons["FINAL_SELECTION_CAP"] = allocation_reasons.get("FINAL_SELECTION_CAP", 0) + 1
            continue
        game = _text(candidate, "game_id", "matchup_id")
        if not game or game in used:
            continue
        matches = out.index[
            out.apply(lambda row: _text(row, "game_id", "matchup_id") == game, axis=1)
        ]
        if len(matches) != 1:
            continue
        idx = matches[0]
        strict_contract = out.at[idx, "wager_contract"] if "wager_contract" in out.columns else None
        if isinstance(strict_contract, dict) and strict_contract.get("production_eligible") is True:
            continue
        contract = _contract(candidate, bank, decision_time)
        if not contract["trial_eligible"]:
            continue
        try:
            validate_contract(contract)
        except ValueError:
            allocation_reasons["INVALID_TRIAL_CONTRACT"] = allocation_reasons.get("INVALID_TRIAL_CONTRACT", 0) + 1
            continue
        raw_team_ids = candidate.get("team_ids")
        if isinstance(raw_team_ids, str):
            try:
                raw_team_ids = json.loads(raw_team_ids)
            except (TypeError, ValueError):
                raw_team_ids = None
        team_ids = raw_team_ids if isinstance(raw_team_ids, (list, tuple)) else [
            _text(candidate, "home_team_id"), _text(candidate, "away_team_id")
        ]
        start = aware(contract["start"])
        slate_key = _text(candidate, "slate_id") or (
            start.astimezone(ZoneInfo("America/New_York")).date().isoformat() if start else ""
        )
        exact_identity = {
            "candidate_id": contract["candidate_id"], "game_id": game,
            "sport": contract["sport"], "market_type": contract["market_type"],
            "selection": contract["selection"], "line": contract["line"],
            "odds": contract["odds"], "sportsbook": contract["sportsbook"],
            "quote_timestamp": contract["quote_timestamp"],
            "estimated_probability": contract["estimated_probability"],
            "push_probability": contract["push_probability"],
            "team_ids": team_ids,
        }
        amount, reservation_id, allocation_status = reserve_recommendation(
            exact_identity, contract["recommended_bet_amount"], consent=consent,
            exposure=exposure, external_recommended=external_recommended,
            slate_key=slate_key, sport=contract["sport"], game_id=game,
            team_ids=team_ids, now=decision_time, max_picks=MAX_PICKS_PER_SLATE,
        )
        allocation_reasons[allocation_status] = allocation_reasons.get(allocation_status, 0) + 1
        if amount <= 0:
            continue
        contract["recommended_bet_amount"] = amount
        contract["bankroll_fraction"] = amount / bank
        validate_contract(contract)
        for column in (
            "candidate_id", "best_pick", "selection", "market_type", "odds_american", "american_odds",
            "odds_source", "quote_bookmaker", "quote_source", "quote_time", "odds_recorded_at",
            "market_line_used", "spread_line", "total_line", "calibrated_probability",
            "effective_win_probability", "production_win_probability", "expected_value",
            "effective_expected_value", "production_expected_value", "edge", "effective_edge",
            "production_edge", "line_consistency_flag", "line_event_identity_match_flag",
            "quote_binding_verified", "exact_quote_verified", "identity_verified",
        ):
            if column in candidate.index:
                out.at[idx, column] = candidate.get(column)
        for column in review_columns:
            out.at[idx, column] = candidate.get(column)
        out.at[idx, "best_pick"] = contract["selection"]
        out.at[idx, "controlled_trial_contract"] = contract
        out.at[idx, "controlled_trial_candidate"] = True
        out.at[idx, "controlled_trial_eligible"] = True
        out.at[idx, "controlled_trial_stake"] = contract["recommended_bet_amount"]
        out.at[idx, "production_eligible"] = False
        out.at[idx, "wager_approved"] = False
        out.at[idx, "Bettable"] = False
        out.at[idx, "Play_Stake"] = 0.0
        out.at[idx, "Kelly_Bet_Size"] = 0.0
        out.at[idx, "Pick_Status"] = "Controlled Trial"
        out.at[idx, "commercial_tier"] = "CONTROLLED_TRIAL"
        out.at[idx, "Export_Scope"] = "CONTROLLED TRIAL WAGER"
        out.at[idx, "Wager_Instruction"] = (
            f"CONTROLLED TRIAL — VERIFY EXACT LINE/PRICE — MAX ${contract['recommended_bet_amount']:.2f}"
        )
        out.at[idx, "Status_Reason"] = contract["reason"]
        out.at[idx, "qualification_reason"] = contract["reason"]
        if "wager_contract" in out.columns:
            out.at[idx, "wager_contract"] = None
        used.add(game)
    if diagnostics is not None:
        diagnostics["controlled_trial_allocation_reasons"] = allocation_reasons
    return out


def attach_reviews(candidates: pd.DataFrame, reviewed: pd.DataFrame) -> pd.DataFrame:
    """Persist exact-candidate trial decisions and Gemini provenance."""
    if candidates is None or candidates.empty or reviewed is None or reviewed.empty:
        return candidates
    if "candidate_id" not in candidates or "candidate_id" not in reviewed:
        return candidates
    out = candidates.copy()
    fields = [
        name for name in reviewed.columns
        if name.startswith("gemini_") or name.startswith("controlled_trial_")
    ]
    for _, source in reviewed.iterrows():
        matches = out.index[out["candidate_id"].astype(str).eq(str(source.get("candidate_id")))]
        if len(matches) != 1:
            continue
        for field in fields:
            out.at[matches[0], field] = source.get(field)
    return out


def validate_contract(contract: dict[str, Any], *, read_only_legacy: bool = False) -> None:
    if not isinstance(contract, dict) or set(contract) != set(public_fields_for_version(contract)):
        raise ValueError("Invalid controlled-trial contract schema")
    if contract.get("authority") != AUTHORITY or (contract.get("version") == LEGACY_VERSION and not read_only_legacy):
        raise ValueError("Invalid controlled-trial authority")
    for _, value in contract.items():
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ValueError("Invalid controlled-trial field")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite controlled-trial metric")
    amount = finite(contract.get("recommended_bet_amount")) or 0.0
    fraction = finite(contract.get("bankroll_fraction")) or 0.0
    if contract.get("version") == LEGACY_VERSION:
        odds = finite(contract.get("odds"))
        decimal = decimal_price(odds)
        probability = finite(contract.get("estimated_probability"))
        break_even = finite(contract.get("break_even_probability"))
        edge = finite(contract.get("estimated_price_edge"))
        ev = finite(contract.get("estimated_expected_value"))
        if (decimal is None or probability is None or not 0 < probability < 1
                or break_even is None or edge is None or ev is None
                or abs(break_even - 1 / decimal) > 1e-9
                or abs(edge - (probability - break_even)) > 1e-9
                or abs(ev - (probability * decimal - 1)) > 1e-9
                or amount < 0 or amount > MAX_DOLLARS_PER_PICK
                or fraction < 0 or fraction > MAX_BANKROLL_FRACTION_PER_PICK
                or (contract.get("trial_eligible") is True and (
                    amount <= 0 or fraction <= 0 or edge < MIN_PRICE_EDGE or ev <= 0
                    or contract.get("gemini_review_status") != "APPROVE"
                    or contract.get("identity_verified") is not True
                    or contract.get("quote_verified") is not True
                    or contract.get("quote_fresh") is not True))
                or (contract.get("trial_eligible") is not True and (amount != 0 or fraction != 0))):
            raise ValueError("Invalid historical controlled-trial evidence")
        return
    required_text = (
        "candidate_id", "game_id", "sport", "market_type", "selection",
        "sportsbook", "quote_timestamp", "start", "data_quality_status", "reason",
    )
    odds = finite(contract.get("odds"))
    line = finite(contract.get("line"))
    probability = finite(contract.get("estimated_probability"))
    push = finite(contract.get("push_probability"))
    loss = finite(contract.get("loss_probability"))
    break_even = finite(contract.get("break_even_probability"))
    edge = finite(contract.get("estimated_price_edge"))
    expected_value = finite(contract.get("estimated_expected_value"))
    decimal = decimal_price(odds)
    pricing = price_value(probability, push, decimal) if decimal is not None else None
    quote = aware(contract.get("quote_timestamp"))
    start = aware(contract.get("start"))
    if (
        any(not str(contract.get(field) or "").strip() for field in required_text)
        or not _price_allowed(odds)
        or line is None
        or probability is None or not 0 < probability < 1
        or pricing is None or loss is None
        or contract.get("probability_semantics") != "win_unconditional_with_push"
        or contract.get("probability_source") not in {
            "calibrated_probability", "selection_probability_used", "sport_calibrated_probability"}
        or contract.get("source_probability_semantics") not in {
            "win_conditional_on_decision", "win_unconditional_with_push"}
        or break_even is None or edge is None or expected_value is None
        or abs(loss - pricing["p_loss"]) > 1e-9
        or abs(break_even - pricing["break_even"]) > 1e-9
        or abs(edge - pricing["edge"]) > 1e-9
        or abs(expected_value - pricing["expected_value"]) > 1e-9
        or quote is None or start is None or quote >= start
        or contract.get("data_quality_status") != "NO_DEGRADED_FLAGS"
    ):
        raise ValueError("Invalid controlled-trial evidence")
    if contract.get("trial_eligible") is True:
        if (
            not production_market(contract.get("market_type"))
            or contract.get("gemini_review_status") != "APPROVE"
            or contract.get("identity_verified") is not True
            or contract.get("quote_verified") is not True
            or contract.get("quote_fresh") is not True
            or (finite(contract.get("estimated_price_edge")) or 0.0) < MIN_PRICE_EDGE
            or (finite(contract.get("estimated_expected_value")) or 0.0) <= 0
            or not 0 < fraction <= MAX_BANKROLL_FRACTION_PER_PICK
            or not 0 < amount <= MAX_DOLLARS_PER_PICK
            or finite(contract.get("gemini_stake_multiplier")) not in {0.75, 1.0}
        ):
            raise ValueError("Invalid eligible controlled trial")
    elif amount != 0 or fraction != 0:
        raise ValueError("Ineligible controlled trial must have zero stake")
