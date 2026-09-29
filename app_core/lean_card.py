"""All-games lean view: the model's read on EVERY game, tiered honestly.

The games card stakes only proven +EV picks, so on an efficient slate it looks empty even
though the model has a directional read on every game. This view re-presents the SAME card
(it adds no staking and changes no guard) so a bettor who wants action across the board can
see, per game: the model's side, its confidence, and an honest tier -

  * BET   - a strict Premium pick or an explicitly labelled, small-stake
            Controlled Value pick that clears the exact calibrated price gate.
  * LEAN  - the model has a positive-EV side but below the stake bar, and it is not fading
            Kalshi. A read worth knowing; NOT a proven +EV bet. Bet at your own risk.
  * AVOID - negative EV at the price, or the model is fading consensus (Disagrees). The
            board the math says to stay off.

The tiers separate a best-available directional read from a production wager. Every game
remains visible, but only a BET clears the absolute price gate and receives money.
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from core.production_gate import (
    MIN_PRODUCTION_CALIBRATED_EDGE,
    evaluate_absolute_production_gate,
)


def _first_col(df: pd.DataFrame, *names: str):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([None] * len(df), index=df.index)


def _strict_bool_col(df: pd.DataFrame, name: str, *, default: bool = False) -> pd.Series:
    """Return a fail-closed boolean Series for an optional authorization column."""

    if name not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    values = df[name]
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(default).astype(bool)
    normalized = values.astype("string").fillna("").str.strip().str.casefold()
    return normalized.isin({"true", "1", "1.0", "yes", "y"})


def classify_lean_tier(status: object, eff_ev: object, consensus: object,
                       *, calibrated_win: object = None, break_even: object = None) -> str:
    """BET / LEAN / AVOID for one row (see module docstring).

    When ``calibrated_win`` and ``break_even`` are supplied, a would-be LEAN is demoted to
    AVOID if its CALIBRATED win probability fails to beat the break-even price - the model's
    raw EV is overconfident in the 0.50-0.55 band (327 graded picks: predicted .53, realized
    .43), so a positive *raw* EV there is usually a negative *calibrated* EV.
    """
    ev = pd.to_numeric(pd.Series([eff_ev]), errors="coerce").iloc[0]
    cons = str(consensus or "").strip()
    cw = pd.to_numeric(pd.Series([calibrated_win]), errors="coerce").iloc[0]
    be = pd.to_numeric(pd.Series([break_even]), errors="coerce").iloc[0]
    absolute_edge = (float(cw) - float(be)) if pd.notna(cw) and pd.notna(be) else None

    # "Actionable" is necessary but no longer sufficient. A production BET must
    # also clear the offered price by an absolute safety margin and retain
    # positive model EV. This prevents a relative best-in-game candidate from
    # being funded merely because every alternative was worse.
    if (
        str(status).strip() == "Actionable"
        and pd.notna(ev)
        and float(ev) > 0.0
        and absolute_edge is not None
        and absolute_edge >= MIN_PRODUCTION_CALIBRATED_EDGE
    ):
        return "BET"
    if not (pd.notna(ev) and ev > 0 and cons != "Disagrees"):
        return "AVOID"
    if absolute_edge is not None and absolute_edge <= 0.0:
        return "AVOID"
    return "LEAN"


_TIER_ORDER = {"BET": 0, "LEAN": 1, "AVOID": 2}


def _american_breakeven(odds: object):
    o = pd.to_numeric(pd.Series([odds]), errors="coerce").iloc[0]
    if pd.isna(o):
        return None
    o = float(o)
    return abs(o) / (abs(o) + 100.0) if o < 0 else 100.0 / (o + 100.0)


def _american_decimal(odds: object):
    value = pd.to_numeric(pd.Series([odds]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    value = float(value)
    if value == 0.0:
        return None
    return 1.0 + (100.0 / abs(value) if value < 0.0 else value / 100.0)


def _bucket_transform_identity(bucket_stats: object) -> str:
    """Identify supporting bucket evidence separately from calibration authority."""

    if not isinstance(bucket_stats, dict) or not bucket_stats.get("buckets"):
        return ""
    try:
        payload = json.dumps(
            bucket_stats, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError):
        return "bucket-conditional-tilt-v1:unserializable-supporting-evidence"
    return f"bucket-conditional-tilt-v1:{hashlib.sha256(payload).hexdigest()}"


_UNSET = object()


def _production_context_calibration(
    df: pd.DataFrame,
    probabilities: pd.Series,
    consensus: pd.Series,
    bucket_stats: object,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Apply the default artifact only to rows whose provenance matches it.

    The production artifact is exact-sport, exact-market, and predictor scoped.
    Mixed frames are therefore evaluated one context at a time.  Missing or
    rejected context leaves the upstream probability unchanged and is labelled
    explicitly; it never borrows the first row's accepted curve.
    """
    from core.empirical_tiers import bucket_key
    from core.market_policy import sport_market_family
    from core.probability_calibration import (
        CONDITIONAL_PROBABILITY_SEMANTICS,
        apply_calibration,
        apply_bucket_calibration,
        calibration_provenance,
        load_calibration,
    )
    from core.probability_semantics import unconditional_from_conditional

    sports = _first_col(df, "exact_sport", "sport", "league", "League").fillna("").astype(str).str.strip().str.upper()
    raw_families = _first_col(df, "exact_market_family", "market_family").fillna("").astype(str).str.strip().str.upper()
    market_types = _first_col(df, "market_type").fillna("")
    families = pd.Series(
        [
            family if family else (sport_market_family(sport, market_type) or "")
            for sport, family, market_type in zip(sports, raw_families, market_types)
        ],
        index=df.index,
        dtype="object",
    )
    predictors = _first_col(
        df,
        "source_predictor_version",
        "predictor_version",
        "model_version",
        "ensemble_version",
    ).fillna("").astype(str).str.strip()
    semantics = _first_col(df, "probability_semantics").fillna("").astype(str).str.strip()
    pushes = pd.to_numeric(_first_col(df, "push_probability"), errors="coerce")
    lines = pd.to_numeric(_first_col(df, "line", "selected_line", "spread_line", "total_line"), errors="coerce")
    buckets = pd.Series(
        [bucket_key(sport, market_type, agreement) for sport, market_type, agreement in zip(sports, market_types, consensus)],
        index=df.index,
        dtype="object",
    )

    calibrated = pd.to_numeric(probabilities, errors="coerce").copy()
    status = pd.Series("CONSUMER_CONTEXT_MISSING", index=df.index, dtype="object")
    version = pd.Series("", index=df.index, dtype="object")
    raw_sha256 = pd.Series("", index=df.index, dtype="object")
    post_transform = pd.Series("NONE", index=df.index, dtype="object")
    post_transform_identity = pd.Series("", index=df.index, dtype="object")
    contexts = pd.DataFrame(
        {"sport": sports, "family": families, "predictor": predictors}, index=df.index
    )
    semantics_ok = semantics.eq(CONDITIONAL_PROBABILITY_SEMANTICS)
    status.loc[semantics.ne("") & ~semantics_ok] = "PROBABILITY_SEMANTICS_INCOMPATIBLE"
    push_ok = pushes.notna() & pushes.ge(0.0) & pushes.lt(1.0)
    half_point_with_push = (
        lines.notna()
        & lines.sub(lines.round()).abs().gt(1e-9)
        & pushes.fillna(0.0).gt(0.0)
    )
    status.loc[semantics_ok & (~push_ok | half_point_with_push)] = (
        "PUSH_SUPPORT_MISSING_OR_INVALID"
    )
    complete = contexts.ne("").all(axis=1) & semantics_ok & push_ok & ~half_point_with_push
    for (sport, family, predictor), indexes in contexts.loc[complete].groupby(
        ["sport", "family", "predictor"], sort=False
    ).groups.items():
        table = load_calibration(
            expected_predictor_version=str(predictor),
            expected_training_scope={
                "exact_sport": str(sport),
                "exact_market_family": str(family),
            },
        )
        if table is None or not bool(table):
            status.loc[indexes] = "CALIBRATION_REJECTED"
            continue
        try:
            artifact_only_values = apply_calibration(
                probabilities.loc[indexes], table
            )
            conditional_values = apply_bucket_calibration(
                probabilities.loc[indexes], buckets.loc[indexes], table, bucket_stats
            )
            artifact_only_numeric = pd.to_numeric(
                artifact_only_values, errors="coerce"
            )
            conditional_numeric = pd.to_numeric(
                conditional_values, errors="coerce"
            )
            tilted_indexes = conditional_numeric.index[
                ~conditional_numeric.sub(artifact_only_numeric).abs().le(1e-12)
            ]
        except (TypeError, ValueError, IndexError):
            status.loc[indexes] = "CALIBRATION_REJECTED"
            continue
        values = pd.Series(index=indexes, dtype=float)
        for row_index, conditional_value in conditional_values.items():
            mass = unconditional_from_conditional(
                conditional_value, pushes.loc[row_index]
            )
            if mass is None:
                status.loc[row_index] = "PUSH_SUPPORT_MISSING_OR_INVALID"
                continue
            values.loc[row_index] = mass["p_win"]
        facts = calibration_provenance(table)
        if not facts:
            status.loc[indexes] = "CALIBRATION_REJECTED"
            continue
        numeric_values = pd.to_numeric(values, errors="coerce")
        valid_indexes = numeric_values[numeric_values.notna()].index
        tilted_indexes = tilted_indexes.intersection(valid_indexes)
        calibrated.loc[valid_indexes] = numeric_values.loc[valid_indexes]
        status.loc[valid_indexes] = "PRODUCTION_CALIBRATION_APPLIED"
        if len(tilted_indexes):
            status.loc[tilted_indexes] = "CALIBRATION_POST_TRANSFORM_UNAUTHORIZED"
            post_transform.loc[tilted_indexes] = "BUCKET_TILT_RESEARCH_ONLY"
            post_transform_identity.loc[tilted_indexes] = _bucket_transform_identity(
                bucket_stats
            )
        version.loc[valid_indexes] = str(facts.get("calibration_version", ""))
        raw_sha256.loc[valid_indexes] = str(facts.get("artifact_raw_sha256", ""))
    return (
        calibrated,
        status,
        version,
        raw_sha256,
        post_transform,
        post_transform_identity,
    )


def score_best_picks_rows(best_picks_df: pd.DataFrame, *, calibration: object = _UNSET,
                          bucket_stats: object = _UNSET) -> pd.DataFrame:
    """Per-row lean scoring, INDEX-ALIGNED to the input frame (no sorting).

    Shared core of the Play Card and the main-card tier columns: computes the
    bucket-calibrated win probability, break-even, Emp_Edge, and BET/LEAN/AVOID
    tier for every row of a best-picks frame. build_all_games_lean_card sorts
    and formats this; the main Best Picks card joins it back by index so every
    displayed row carries a tier and a playable stake.
    """
    if best_picks_df is None or best_picks_df.empty:
        return pd.DataFrame()

    if bucket_stats is _UNSET:
        try:
            from core.empirical_tiers import bucket_stats_are_fresh, load_bucket_stats
            bucket_stats = load_bucket_stats()
            if not bucket_stats_are_fresh(bucket_stats):
                bucket_stats = None
        except Exception:
            bucket_stats = None

    df = best_picks_df
    home = _first_col(df, "Home", "home_team").astype(str)
    away = _first_col(df, "Away", "away_team").astype(str)
    status = _first_col(df, "Pick_Status")
    eff_ev = _first_col(df, "effective_expected_value", "expected_value")
    consensus = _first_col(df, "consensus_agreement")
    win = pd.to_numeric(_first_col(df, "effective_win_probability", "WinProbability"), errors="coerce")
    controlled_value = _strict_bool_col(df, "controlled_card_recovery")
    controlled_empirical_win = pd.to_numeric(
        _first_col(df, "empirical_win_probability"), errors="coerce"
    )
    edge = _first_col(df, "effective_edge", "edge")
    odds = _first_col(df, "odds_american")
    kelly = pd.to_numeric(_first_col(df, "Kelly_Bet_Size"), errors="coerce").fillna(0.0)

    # Calibrated win probability + break-even, for the LEAN gate and transparency. Bucket-
    # conditional (global curve + per-bucket realized tilt) so a proven bucket (e.g.
    # under:Agrees ~61%) isn't crushed below break-even by the pooled curve - same number the
    # staking gate uses, so the view and the card agree.
    calibration_status = pd.Series("CALIBRATION_DISABLED", index=df.index, dtype="object")
    calibration_version = pd.Series("", index=df.index, dtype="object")
    calibration_raw_sha256 = pd.Series("", index=df.index, dtype="object")
    calibration_post_transform = pd.Series("NONE", index=df.index, dtype="object")
    calibration_post_transform_identity = pd.Series(
        "", index=df.index, dtype="object"
    )
    if calibration is _UNSET:
        try:
            (
                calib_win,
                calibration_status,
                calibration_version,
                calibration_raw_sha256,
                calibration_post_transform,
                calibration_post_transform_identity,
            ) = _production_context_calibration(df, win, consensus, bucket_stats)
        except Exception:
            calib_win = win.copy()
            calibration_status[:] = "CALIBRATION_CONSUMER_ERROR"
    elif calibration is not None:
        try:
            from core.probability_calibration import (
                CalibrationTable,
                apply_bucket_calibration,
                calibration_provenance,
            )
            from core.empirical_tiers import bucket_key
            league = _first_col(df, "league", "League")
            market_type = _first_col(df, "market_type")
            buckets = [bucket_key(l, m, c) for l, m, c in zip(league, market_type, consensus)]
            calib_win = pd.to_numeric(
                apply_bucket_calibration(win, buckets, calibration, bucket_stats), errors="coerce"
            )
            calibration_status[:] = "EXPLICIT_CALIBRATION_APPLIED"
            facts = calibration_provenance(calibration)
            if facts:
                calibration_version[:] = str(
                    facts.get("calibration_version", "")
                )
                calibration_raw_sha256[:] = str(
                    facts.get("artifact_raw_sha256", "")
                )
            elif (
                isinstance(calibration, CalibrationTable)
                and calibration.trusted_snapshot_valid()
            ):
                calibration_version[:] = str(
                    (calibration.payload.get("meta") or {}).get(
                        "calibration_version", ""
                    )
                )
                calibration_raw_sha256[:] = str(calibration.raw_sha256 or "")
        except Exception:
            calib_win = win
            calibration_status[:] = "CALIBRATION_REJECTED"
    else:
        # The upstream effective probability is already the best available
        # calibrated value on legacy/no-artifact runs. Using it as the explicit
        # fallback keeps the gate price-aware instead of silently funding blind.
        calib_win = win.copy()
    # Controlled Value recovery is approved against the empirical probability at
    # the exact offered price.  Keep that same authority through the public card
    # and exports; re-gating a recovered row with the legacy effective probability
    # could turn an approved $5 wager into a contradictory $0 pass.
    controlled_empirical_available = controlled_value & controlled_empirical_win.notna()
    calib_win = pd.Series(calib_win, index=df.index, dtype=float).where(
        ~controlled_empirical_available,
        controlled_empirical_win,
    )
    calibration_status = calibration_status.where(
        ~controlled_empirical_available, "CONTROLLED_EMPIRICAL_OVERRIDE"
    )
    calibration_post_transform = calibration_post_transform.where(
        ~controlled_empirical_available, "CONTROLLED_EMPIRICAL_OVERRIDE"
    )

    # One final probability/price contract.  Production calibration returns
    # unconditional win mass; explicit/legacy conditional inputs are converted
    # with the same per-candidate push mass before pricing.  Research values may
    # remain visible when authority is absent, but they cannot pass the gate.
    calib_num = pd.to_numeric(calib_win, errors="coerce")
    source_semantics = _first_col(df, "probability_semantics").fillna("").astype(str).str.strip()
    push_input = pd.to_numeric(_first_col(df, "push_probability"), errors="coerce")
    explicit_contract = any(
        column in df.columns
        for column in ("probability_semantics", "push_probability", "decimal_odds")
    )
    pricing_push = push_input.copy()
    if not explicit_contract:
        pricing_push = pricing_push.fillna(0.0)
    line = pd.to_numeric(
        _first_col(df, "line", "selected_line", "spread_line", "total_line"),
        errors="coerce",
    )
    half_point_with_push = (
        line.notna()
        & line.sub(line.round()).abs().gt(1e-9)
        & pricing_push.fillna(0.0).gt(0.0)
    )
    pricing_push = pricing_push.mask(half_point_with_push)

    american_decimal = pd.Series(
        [_american_decimal(value) for value in odds], index=df.index, dtype=float
    )
    supplied_decimal = pd.to_numeric(_first_col(df, "decimal_odds"), errors="coerce")
    pricing_decimal = supplied_decimal.where(supplied_decimal.notna(), american_decimal)
    quote_consistent = pd.Series(True, index=df.index, dtype=bool)
    both_prices = supplied_decimal.notna() & american_decimal.notna()
    quote_consistent.loc[both_prices] = (
        supplied_decimal.loc[both_prices]
        .sub(american_decimal.loc[both_prices])
        .abs()
        .le(1e-8)
    )
    pricing_decimal = pricing_decimal.mask(~quote_consistent)

    conditional_semantics = "win_conditional_on_decision"
    already_unconditional = calibration_status.isin(
        {
            "PRODUCTION_CALIBRATION_APPLIED",
            "CALIBRATION_POST_TRANSFORM_UNAUTHORIZED",
            "CONTROLLED_EMPIRICAL_OVERRIDE",
        }
    )
    pricing_win = calib_num.copy()
    convert_mean = source_semantics.eq(conditional_semantics) & ~already_unconditional
    pricing_win.loc[convert_mean] = (
        calib_num.loc[convert_mean] * (1.0 - pricing_push.loc[convert_mean])
    )

    conservative_raw = pd.to_numeric(
        _first_col(df, "conservative_probability", "p_win_conservative"),
        errors="coerce",
    )
    conservative_semantics = _first_col(
        df, "conservative_probability_semantics"
    ).fillna("").astype(str).str.strip().str.casefold()
    conservative_final = pd.Series(float("nan"), index=df.index, dtype=float)
    conservative_conditional = conservative_semantics.eq(conditional_semantics)
    conservative_unconditional = conservative_semantics.isin(
        {
            "unconditional",
            "unconditional_win_push_loss",
            "win_unconditional_with_push",
        }
    )
    conservative_final.loc[conservative_conditional] = (
        conservative_raw.loc[conservative_conditional]
        * (1.0 - pricing_push.loc[conservative_conditional])
    )
    conservative_final.loc[conservative_unconditional] = conservative_raw.loc[
        conservative_unconditional
    ]
    conservative_for_gate: object = (
        conservative_final if conservative_raw.notna().any() else None
    )

    if explicit_contract:
        gate = evaluate_absolute_production_gate(
            pricing_win,
            model_expected_value=eff_ev,
            push_probability=pricing_push,
            decimal_odds=pricing_decimal,
            conservative_probability=conservative_for_gate,
        )
    else:
        legacy_break_even = pd.Series(
            [_american_breakeven(value) for value in odds], index=df.index
        )
        gate = evaluate_absolute_production_gate(
            pricing_win,
            legacy_break_even,
            eff_ev,
        )
    breakeven = gate["sportsbook_break_even_probability"]

    calibration_authoritative = pd.Series(True, index=df.index, dtype=bool)
    if calibration is _UNSET:
        calibration_authoritative = calibration_status.isin(
            {"PRODUCTION_CALIBRATION_APPLIED", "CONTROLLED_EMPIRICAL_OVERRIDE"}
        )
    value_contract_authoritative = (
        gate["pricing_contract_status"].eq("PUSH_AWARE_VERIFIED")
        & quote_consistent
        & ~half_point_with_push
        & calibration_authoritative
    )
    if not explicit_contract:
        value_contract_authoritative = gate["pricing_contract_status"].eq(
            "LEGACY_NO_PUSH_COMPATIBILITY"
        )
    value_contract_status = gate["pricing_contract_status"].copy()
    value_contract_status.loc[~quote_consistent] = "QUOTE_PRICE_MISMATCH"
    value_contract_status.loc[half_point_with_push] = "ILLEGAL_LINE_PUSH_PAIRING"
    value_contract_status.loc[
        gate["pricing_contract_status"].ne("INVALID") & ~calibration_authoritative
    ] = "CALIBRATION_OR_TRANSFORM_NOT_AUTHORIZED"

    tier = [
        classify_lean_tier(s, e, c, calibrated_win=cw, break_even=be)
        for s, e, c, cw, be in zip(status, eff_ev, consensus, pricing_win, breakeven)
    ]
    qualified_pick = (
        _strict_bool_col(df, "qualified_pick")
        if "qualified_pick" in df.columns
        else pd.Series(True, index=df.index, dtype=bool)
    )
    qualification_known = pd.Series(
        "qualified_pick" in df.columns, index=df.index, dtype=bool
    )
    tier = pd.Series(tier, index=df.index).where(qualified_pick, "AVOID")

    # Empirical edge = bucket-aware calibrated win minus break-even. This - NOT model EV - is
    # the predictive ranking signal: across 63 graded picks the model's EV ranking was
    # inverted (its highest-EV/most-contrarian picks lost), while bucket-realized performance
    # held up. So the card is ordered by Emp_Edge, best first, within each tier.
    emp_edge = gate["absolute_production_edge"]

    # A started game is never playable at ANY size - pre-game lines are stale
    # and the shown odds may be in-game. attach_play_stakes zeroes these rows.
    started = _strict_bool_col(df, "game_already_started_flag")
    if "status_blocker_stage" in df.columns:
        started = started | df["status_blocker_stage"].astype(str).eq("game_already_started")
    if "Play_Tier" in df.columns:
        started = started | df["Play_Tier"].astype(str).str.strip().str.upper().eq("STARTED")
    pick_text = _first_col(df, "best_pick", "display_pick").fillna("").astype(str)
    unavailable_line = pick_text.str.lower().str.contains(
        r"unresolved|\(no line\)|missing line|rejected", regex=True, na=False
    )
    unsafe_line_identity = pd.Series(False, index=df.index)
    if "line_consistency_flag" in df.columns:
        unsafe_line_identity = unsafe_line_identity | ~_strict_bool_col(df, "line_consistency_flag")
    if "line_event_identity_match_flag" in df.columns:
        unsafe_line_identity = unsafe_line_identity | ~_strict_bool_col(df, "line_event_identity_match_flag")
    if "market_line_source_detail" in df.columns:
        unsafe_line_identity = unsafe_line_identity | df["market_line_source_detail"].astype(str).eq(
            "upload_total_fallback_after_rejected_live"
        )
    playable = ~(started | unavailable_line | unsafe_line_identity)
    public_qualified_pick = qualified_pick.copy()
    if "qualified_pick" in df.columns:
        public_qualified_pick = qualified_pick & tier.isin({"BET", "LEAN"}) & playable
    qualification_reason = _first_col(df, "qualification_reason").fillna("").astype(str).copy()
    final_tier_downgrade = qualified_pick & ~public_qualified_pick
    qualification_reason.loc[final_tier_downgrade & playable] = (
        "PASS: final empirical tier is AVOID at the offered price."
    )
    qualification_reason.loc[final_tier_downgrade & ~playable] = (
        "PASS: final line is unavailable or failed identity validation."
    )
    # A mathematical BET label is not authorization. When the classified pipeline
    # supplies an approval/eligibility field, require that explicit flag plus a funded
    # amount. Legacy callers without authorization columns retain the historical gate.
    authorization_known = "wager_approved" in df.columns or "production_eligible" in df.columns
    explicitly_authorized = pd.Series(True, index=df.index, dtype=bool)
    if "wager_approved" in df.columns:
        explicitly_authorized &= _strict_bool_col(df, "wager_approved")
    if "production_eligible" in df.columns:
        explicitly_authorized &= _strict_bool_col(df, "production_eligible")
    if authorization_known:
        funded_amounts = [
            pd.to_numeric(df[column], errors="coerce").fillna(0.0)
            for column in ("production_bet_amount", "Kelly_Bet_Size", "Play_Stake")
            if column in df.columns
        ]
        if funded_amounts:
            explicitly_authorized &= pd.concat(funded_amounts, axis=1).max(axis=1).gt(0.0)
        else:
            explicitly_authorized &= False

    mathematical_bet = tier.eq("BET")
    production_gate_pass = (
        mathematical_bet
        & gate["production_gate_pass"]
        & value_contract_authoritative
        & playable
        & explicitly_authorized
    )
    # Keep an otherwise-qualified, unfunded direction visible as a LEAN, never BET.
    tier = tier.where(~(mathematical_bet & ~production_gate_pass), "LEAN")
    production_gate_reason = gate["production_gate_reason"].copy()
    non_actionable = ~status.astype(str).str.strip().eq("Actionable")
    production_gate_reason.loc[non_actionable & gate["production_gate_pass"]] = (
        "upstream status is not Actionable"
    )
    production_gate_reason.loc[
        mathematical_bet
        & gate["production_gate_pass"]
        & ~value_contract_authoritative
    ] = "final probability/price contract is not production-authoritative"
    production_gate_reason.loc[
        mathematical_bet
        & gate["production_gate_pass"]
        & value_contract_authoritative
        & playable
        & ~explicitly_authorized
    ] = "explicit wager approval or funded production stake is missing"

    selection_mode = public_qualified_pick.map(
        {True: "Qualified Pick / Pass", False: "Best Available Pick / Pass"}
    )
    selection_mode.loc[production_gate_pass & controlled_value] = (
        "Controlled Value Pick"
    )
    selection_mode.loc[production_gate_pass & ~controlled_value] = "Premium Pick"

    return pd.DataFrame({
        "League": _first_col(df, "league", "League"),
        "Matchup": (away + " @ " + home).str.strip(" @"),
        "Pick": _first_col(df, "best_pick", "display_pick"),
        "Win%": win,
        "Calib_Win%": calib_num,
        "Calibration_Consumer_Status": calibration_status,
        "Calibration_Version": calibration_version,
        "Calibration_Artifact_SHA256": calibration_raw_sha256,
        "Calibration_Post_Transform": calibration_post_transform,
        "Calibration_Post_Transform_Identity": calibration_post_transform_identity,
        "Candidate_ID": _first_col(df, "canonical_event_id", "matchup_id", "game_id"),
        "Quote_ID": _first_col(df, "quote_id"),
        "Quote_Observed_At": _first_col(df, "quote_observed_at", "odds_recorded_at"),
        "Odds_American": pd.to_numeric(odds, errors="coerce"),
        "Odds_Decimal": pricing_decimal,
        "Probability_Semantics": pd.Series(
            "win_unconditional_with_push", index=df.index, dtype="object"
        ).where(gate["pricing_contract_status"].ne("INVALID"), "UNRESOLVED"),
        "Push_Source": _first_col(df, "push_probability_source").fillna("").astype(str).where(
            _first_col(df, "push_probability_source").fillna("").astype(str).str.strip().ne(""),
            "candidate_push_probability" if explicit_contract else "legacy_implicit_zero",
        ),
        "Value_Contract_Version": "unconditional-refunded-push-v1",
        "Value_Contract_Status": value_contract_status,
        "Final_P_Win": gate["final_p_win"],
        "Final_P_Push": gate["final_p_push"],
        "Final_P_Loss": gate["final_p_loss"],
        "Price_Break_Even": gate["sportsbook_break_even_probability"],
        "Mean_EV_Per_Unit": gate["mean_expected_value_per_unit"],
        "P_Win_Conservative": conservative_final,
        "Conservative_EV_Per_Unit": gate["conservative_expected_value_per_unit"],
        "Minimum_Acceptable_Decimal_Odds": gate[
            "minimum_acceptable_decimal_odds"
        ],
        "Emp_Edge": emp_edge,
        "Edge": pd.to_numeric(edge, errors="coerce"),
        "Upstream_Model_EV": pd.to_numeric(eff_ev, errors="coerce"),
        "EV_Field_Semantics": "legacy alias of upstream model EV; not final priced EV",
        "EV": pd.to_numeric(eff_ev, errors="coerce"),
        "Consensus": consensus,
        "Tier": tier,
        "Started": started,
        # This only states that a valid pregame line is present; wager approval
        # remains exclusively in Production_Gate_Pass / Wager_Approved.
        "Line_Available": playable,
        "Selection_Mode": selection_mode,
        "Bet_Decision": pd.Series("BEST AVAILABLE - PASS", index=df.index)
        .where(~(qualification_known & public_qualified_pick), "QUALIFIED LEAN - PASS")
        .where(~production_gate_pass, "BET"),
        "Qualified_Pick": public_qualified_pick,
        "Qualification_Reason": qualification_reason,
        "Production_Gate_Pass": production_gate_pass,
        "Production_Gate_Reason": production_gate_reason,
        "Absolute_Edge": gate["absolute_production_edge"],
        "Calibrated_EV": gate["calibrated_expected_value"],
        "Controlled_Value_Card": controlled_value & production_gate_pass,
        # Stake only on rows that pass the independent absolute price gate.
        "Suggested_Stake": kelly.where(production_gate_pass, 0.0),
    }, index=df.index)


def build_all_games_lean_card(best_picks_df: pd.DataFrame, *, calibration: object = _UNSET,
                              bucket_stats: object = _UNSET) -> pd.DataFrame:
    """Derive the all-games lean card from the games best-picks frame.

    Pure and side-effect-free: reads the existing per-game pick/probability/EV columns,
    assigns a tier, and returns a compact view RANKED BY EMPIRICAL EDGE (bucket-realized),
    not model EV. Tolerant of the pre- and post-export column names (home_team/Home, etc.).

    The LEAN tier is gated by bucket-conditional calibration: a pick stays LEAN only if its
    calibrated win beats break-even. ``calibration`` / ``bucket_stats`` default to the fitted
    tables on disk; pass ``None`` to disable (raw behavior) or explicit values to inject
    (tests).
    """
    out = score_best_picks_rows(best_picks_df, calibration=calibration, bucket_stats=bucket_stats)
    if out.empty:
        return out

    # Preserve deployment and run provenance in the all-games CSV without
    # coupling this presentation module back to the main pipeline (and without
    # inventing stamps for legacy callers that did not supply them). The run ID
    # proves this card and its graded best-picks export share one snapshot.
    provenance_position = 0
    for column in ("pipeline_build", "export_run_id"):
        if column not in best_picks_df.columns or column in out.columns:
            continue
        values = best_picks_df[column].reindex(out.index).fillna("").astype(str)
        if values.str.strip().ne("").any():
            out.insert(provenance_position, column, values)
            provenance_position += 1

    out["_t"] = out["Tier"].map(_TIER_ORDER).fillna(3)
    # Rank by empirical edge (bucket-proven), not model Win%/EV - the latter is anti-informative.
    # Win% is the tiebreaker and the fallback when there's no calibration (Emp_Edge all NaN).
    out = out.sort_values(
        ["_t", "Emp_Edge", "Win%"], ascending=[True, False, False], na_position="last"
    ).drop(columns="_t").reset_index(drop=True)
    return out


# Display units by tier for attach_play_stakes. Only a genuine BET receives a
# stake. LEAN and AVOID remain useful reads, but assigning dollars to them
# contradicts the production gate and makes an abstaining card look bettable.
PLAY_UNITS_BET = 2.0
PLAY_UNITS_CONTROLLED_VALUE = 0.5
PLAY_UNITS_LEAN = 0.0
PLAY_UNITS_AVOID_NEAR = 0.0
PLAY_UNITS_AVOID_FAR = 0.0
AVOID_NEAR_EDGE = -0.05
# Hopeless-price floor: below this calibrated-win-vs-break-even edge, even the
# minimum action unit is money on fire (4 Jul: CWS +5.5 at -1718, Emp_Edge
# -0.35, drew a $2.50 play stake). No recreational stake at any size.
PLAY_NO_PRICE_EDGE = -0.20


def attach_play_stakes(card: pd.DataFrame, unit: float = 1.0) -> pd.DataFrame:
    """Attach a dollar stake only to approved BET rows.

    LEAN and AVOID rows remain visible so the model's full-board read is useful,
    but they receive zero dollars. This keeps Play_Stake aligned with the app's
    positive-EV production decision:

      Premium BET     2.0u (or the pick's own Kelly stake if larger)
      Controlled BET  0.5u (or its tightly capped production stake if larger)
      LEAN   0u
      AVOID  0u

    Pure: returns a copy with Play_Stake ($) and Play_Units columns.
    """
    if card is None or card.empty:
        return pd.DataFrame() if card is None else card.copy()
    out = card.copy()
    tier = out.get("Tier", pd.Series("", index=out.index)).astype(str)
    emp_edge = pd.to_numeric(out.get("Emp_Edge"), errors="coerce")

    units = pd.Series(0.0, index=out.index)
    units[tier.eq("BET")] = PLAY_UNITS_BET
    controlled_value = _strict_bool_col(out, "Controlled_Value_Card")
    units[tier.eq("BET") & controlled_value] = PLAY_UNITS_CONTROLLED_VALUE
    # Hopeless prices (deeply below break-even) get $0 at any tier except a
    # genuine BET: paying -1700 juice for "action" isn't recreation, it's a fee.
    units[emp_edge.lt(PLAY_NO_PRICE_EDGE) & ~tier.eq("BET")] = 0.0

    stake = units * float(unit)
    if "Suggested_Stake" in out.columns:
        kelly = pd.to_numeric(out["Suggested_Stake"], errors="coerce").fillna(0.0)
        stake = stake.where(~(tier.eq("BET") & kelly.gt(stake)), kelly)

    # Prefer the explicit availability name while accepting legacy cards that
    # still carry Playable. Availability is not itself wager approval.
    availability_column = (
        "Line_Available" if "Line_Available" in out.columns
        else "Playable" if "Playable" in out.columns
        else None
    )
    if availability_column is not None:
        line_available = _strict_bool_col(out, availability_column)
        units = units.where(line_available, 0.0)
        stake = stake.where(line_available, 0.0)
        out.loc[~line_available, "Tier"] = "UNAVAILABLE"
        if "Bet_Decision" in out.columns:
            out.loc[~line_available, "Bet_Decision"] = "UNAVAILABLE"

    # Started games are unplayable at any size: the pre-game line is gone.
    if "Started" in out.columns:
        started = _strict_bool_col(out, "Started")
        units = units.where(~started, 0.0)
        stake = stake.where(~started, 0.0)
        out.loc[started, "Tier"] = "STARTED"
        if "Bet_Decision" in out.columns:
            out.loc[started, "Bet_Decision"] = "STARTED"

    out["Play_Units"] = units
    out["Play_Stake"] = stake.round(2)
    out["All_Row_Bet"] = stake.gt(0)
    out["Wager_Approved"] = out["Play_Stake"].gt(0)
    qualified = (
        _strict_bool_col(out, "Qualified_Pick")
        if "Qualified_Pick" in out.columns
        else pd.Series(True, index=out.index, dtype=bool)
    )
    qualification_known = "Qualified_Pick" in out.columns
    out["Export_Role"] = (
        "BEST AVAILABLE PICK - PASS / RESEARCH"
        if qualification_known
        else "COVERAGE PICK - PASS"
    )
    if qualification_known:
        out.loc[qualified, "Export_Role"] = "QUALIFIED LEAN - PASS"
    out.loc[out["Wager_Approved"], "Export_Role"] = "PRODUCTION WAGER"
    out.loc[out["Wager_Approved"] & controlled_value, "Export_Role"] = (
        "CONTROLLED VALUE WAGER"
    )
    out["Wager_Instruction"] = (
        "DO NOT BET: best available pick does not clear the wager qualification gate."
        if qualification_known
        else "DO NOT BET: coverage row without explicit wager qualification."
    )
    if qualification_known:
        out.loc[qualified, "Wager_Instruction"] = "DO NOT BET: qualified research lean without a funded edge."
    out.loc[out["Wager_Approved"], "Wager_Instruction"] = (
        "APPROVED: wager the exported Play_Stake amount."
    )
    out.loc[out["Wager_Approved"] & controlled_value, "Wager_Instruction"] = (
        "APPROVED CONTROLLED VALUE: use the exported small stake; not a Premium pick."
    )
    return out
