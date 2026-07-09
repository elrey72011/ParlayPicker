from __future__ import annotations

from itertools import combinations

import pandas as pd
import hashlib
import json

from core.probability_calibration import apply_calibration
from core.parlay_safety import (
    conservative_joint_probability,
    production_candidate_mask,
    shares_game,
)

# Only picks with these statuses qualify as parlay legs.
# Below Threshold added 11 Jun: across the graded Jun 5-10 recaps the staked tiers
# went Actionable 1-4 and HV/Spec 3-11 (~21%) while Below Threshold went 29-20 (59%).
# Status reflects the EV/edge promotion gates — which select for maximum model-vs-
# market disagreement, not realized accuracy — so BT legs are admitted and every leg
# must instead clear the probability/edge/consensus floors below on its own merits.
PARLAY_ELIGIBLE_STATUSES = {"Actionable", "High Variance/Speculative", "Below Threshold"}

# Maximum legs per parlay. Held at 2 until the realized single-leg hit rate clears
# ~57-58%: at the ~50-55% the Jun 5-10 recaps show, 3-leg parlays are -EV after vig
# regardless of construction (0.55**3 = 16.6% vs ~14.3% breakeven at +600 only with
# perfectly fair odds; real books pay less). Restore 3 when calibrated legs earn it.
MAX_PARLAY_LEGS = 2

# At most this many totals legs sharing the same (league, over/under direction) per
# parlay. Same-night same-direction totals are driven by shared factors (weather,
# league-wide run/scoring environment, a shared model bias on Overs) and lose
# together — 6 Jun: the HV tier's four MLB Overs went 0-4 as a block. The production
# card already has concentration caps; this is the parlay-level equivalent.
#
# Exception: when EVERY leg in the bucket is Kalshi-Agrees, up to
# MAX_SAME_DIRECTION_AGREES_TOTAL_LEGS may pair. The block-loss failure mode is
# shared MODEL bias against the market (the 6 Jun 0-4 block was fade-forced Overs
# where TheOver disagreed); Agrees legs have the market on their side and hit 61.4%
# on graded MLB totals (n=182). Such combos are flagged is_high_correlation and
# their Kelly stake is cut by CORRELATED_PARLAY_KELLY_MULTIPLIER — the correlation
# is paid for in stake, not ignored.
MAX_SAME_DIRECTION_TOTAL_LEGS = 1
MAX_SAME_DIRECTION_AGREES_TOTAL_LEGS = 2

# Stake multiplier for parlays flagged is_high_correlation (same-game legs or a
# same-direction Agrees pair). Joint outcomes are positively correlated, so these
# boom/bust as a unit; half stake keeps the EV exposure with half the block variance.
CORRELATED_PARLAY_KELLY_MULTIPLIER = 0.5

# When a calibration table is active, each leg must clear its own break-even price
# by this margin on the CALIBRATED probability: p_cal >= 1/decimal_odds + margin.
# This supersedes the raw-space MLB totals floor (MLB_TOTAL_HV_PARLAY_MIN_PROB)
# in calibrated mode — that floor was a proxy for "real win rate high enough",
# tuned on uncalibrated data; calibrated-EV-vs-price is the honest version of the
# same protection (10 Jun replay: the floor blocked legs that were calibrated +EV
# at their actual odds).
MIN_LEG_EV_MARGIN = 0.03

# Minimum effective_win_probability per leg. Uses effective_win_probability (shrinkage-
# adjusted) rather than calibrated_probability — for Overs this is lower than calibrated,
# making it a better discriminator. Threshold of 0.58 corresponds to ~0.68 calibrated for
# Overs (0.58 = 0.5 + 0.85*(0.68-0.5)), keeping quality high while allowing good Unders.
MIN_LEG_PROBABILITY = 0.58

# Minimum edge per leg — ensures each leg genuinely beats the market.
MIN_LEG_EDGE = 0.02

# Actionable picks are force-included in the candidate pool (they still must pass
# the per-leg floors). They no longer get sort priority: Actionable went 1-4 across
# the graded Jun 5-10 recaps, so an Actionable anchor is not evidence a combo is
# better — parlays rank purely on EV computed from calibrated leg probabilities.
MAX_ACTIONABLE_ANCHORS = 5

# Non-Actionable MLB total picks must clear a stricter floor for parlay legs.
# May 27: MLB HV/Spec totals went 0-6 while Actionable MLB totals went 2-2 (100%).
# Require effective_win_probability >= 0.62 — the Actionable-tier entry point.
# Applied to Below Threshold MLB totals as well now that BT legs are eligible.
MLB_TOTAL_HV_PARLAY_MIN_PROB = 0.62

# Production mode is enabled only when the pipeline supplies explicit production
# metadata. Legacy/unit-test frames without that metadata retain compatibility;
# real cards are held to the strict path below.
STRICT_PRODUCTION_PARLAYS = True
PARLAY_PROBABILITY_HAIRCUT = 0.97


def _amer_to_dec(odds: float) -> float:
    if pd.isna(odds) or odds == 0:
        return 1.0
    if odds > 0:
        return 1 + (odds / 100.0)
    return 1 + (100.0 / abs(odds))


def _best_book_odds(legs: pd.DataFrame) -> tuple[float, str]:
    """Return (best combined decimal odds, book name) across available bookmakers."""
    best_odds = float(legs["decimal_odds"].prod())
    best_book = "novig"
    for book in ("fanduel", "draftkings", "betmgm"):
        col = f"odds_american_{book}"
        if col in legs.columns and legs[col].notna().all():
            book_odds = float(legs[col].apply(_amer_to_dec).prod())
            if book_odds > best_odds:
                best_odds = book_odds
                best_book = book
    return best_odds, best_book


def _adj_prob(legs: pd.DataFrame, prob_col: str, rho: float = 0.8) -> float:
    """Combined probability with power-law correlation penalty for same-game legs."""
    result = 1.0
    for matchup_id in legs["matchup_id"].unique():
        m_legs = legs[legs["matchup_id"] == matchup_id]
        if len(m_legs) > 1:
            p_prod = m_legs[prob_col].prod()
            p_min = m_legs[prob_col].min()
            result *= (p_prod ** (1 - rho)) * (p_min ** rho)
        else:
            result *= float(m_legs[prob_col].iloc[0])
    return float(result)


def _direction_buckets(legs: pd.DataFrame) -> pd.Series:
    """(league, over/under direction) bucket per totals leg; NA for non-totals."""
    if "best_pick" not in legs.columns or "league" not in legs.columns:
        return pd.Series(pd.NA, index=legs.index)
    pick = legs["best_pick"].astype(str).str.lower()
    direction = pick.str.extract(r"\b(over|under)\b", expand=False)
    return (legs["league"].astype(str).str.upper() + ":" + direction).where(direction.notna())


def _violates_direction_cap(legs: pd.DataFrame) -> bool:
    """True when a (league, over/under direction) totals bucket exceeds its cap.
    Same-night same-direction totals are correlated through weather/run environment
    and shared model bias — they lose as a block (6 Jun: four MLB Overs in HV went
    0-4 together). All-Agrees buckets get the higher cap (see constants above)."""
    buckets = _direction_buckets(legs)
    if buckets.isna().all():
        return False
    consensus = (
        legs["consensus_agreement"].astype(str)
        if "consensus_agreement" in legs.columns
        else pd.Series("", index=legs.index)
    )
    for bucket, count in buckets.dropna().value_counts().items():
        all_agrees = consensus[buckets == bucket].eq("Agrees").all()
        cap = MAX_SAME_DIRECTION_AGREES_TOTAL_LEGS if all_agrees else MAX_SAME_DIRECTION_TOTAL_LEGS
        if count > cap:
            return True
    return False


def _has_same_direction_pair(legs: pd.DataFrame) -> bool:
    counts = _direction_buckets(legs).dropna().value_counts()
    return bool((counts > 1).any())


def downweight_correlated_parlay_kelly(
    parlays: pd.DataFrame,
    columns: tuple[str, ...] = ("kelly_fraction", "recommended_bet"),
) -> pd.DataFrame:
    """Halve the Kelly sizing of parlays flagged is_high_correlation. Positive
    leg correlation concentrates outcomes (the combo booms or busts as a unit),
    so the stake — not the EV estimate — absorbs the extra variance. Pure helper
    so the policy is unit-testable outside the pipeline."""
    if parlays is None or parlays.empty or "is_high_correlation" not in parlays.columns:
        return parlays
    out = parlays.copy()
    mask = out["is_high_correlation"].fillna(False).astype(bool)
    for col in columns:
        if col in out.columns:
            out.loc[mask, col] = out.loc[mask, col] * CORRELATED_PARLAY_KELLY_MULTIPLIER
    return out


def _leg_labels(legs: pd.DataFrame, label_cols: list[str]) -> list[str]:
    labels: list[str] = []
    has_teams = {"away_team", "home_team"}.issubset(set(label_cols))
    for _, row in legs.iterrows():
        game_ctx = ""
        if has_teams and pd.notna(row.get("away_team")) and pd.notna(row.get("home_team")):
            game_ctx = f"{row['away_team']} @ {row['home_team']}"

        if "best_pick" in label_cols and pd.notna(row.get("best_pick")) and str(row.get("best_pick")).strip():
            pick = str(row["best_pick"])
            labels.append(f"{game_ctx}: {pick}" if game_ctx else pick)
        elif "team" in label_cols and pd.notna(row.get("team")):
            labels.append(str(row["team"]))
        elif game_ctx:
            labels.append(game_ctx)
        elif "away_team" in label_cols:
            labels.append(str(row["away_team"]))
        else:
            labels.append("leg")
    return labels


def _build_record(legs: pd.DataFrame, label_cols: list[str], leg_count: int,
                  risk_tier: str, group_id=pd.NA, prob_col: str = "calibrated_probability",
                  strict_mode: bool = False) -> dict | None:
    """Build a parlay record dict; returns None if EV is non-positive.

    ``prob_col`` is the per-leg probability the combined win prob and EV are built
    from. The engine passes ``effective_win_probability`` (shrinkage-adjusted) here:
    for Overs that field is de-biased downward (calibrated is overconfident — predicts
    ~62%, hits ~52% in the 0.60-0.65 band), and overconfidence COMPOUNDS multiplicatively
    across legs (0.62**3 = 23.8% shown vs ~0.52**3 = 14.1% true). Gating, the EV>0 drop,
    the EV-desc sort, and the displayed win prob must all use the same de-biased field
    the rest of the pipeline trusts; ``calibrated_probability`` is the legacy fallback.
    The market baseline stays on ``market_probability`` (no model bias to remove).
    """
    combined_probability = _adj_prob(legs, prob_col)
    if strict_mode:
        # Unknown dependence and model error are paid for conservatively. We do
        # not manufacture a correlation boost from a heuristic discount.
        combined_probability = conservative_joint_probability(
            [float(v) for v in legs[prob_col].tolist()],
            haircut=PARLAY_PROBABILITY_HAIRCUT,
        )
    combined_market_prob = _adj_prob(legs, "market_probability")
    combined_decimal_odds, best_book = _best_book_odds(legs)

    parlay_ev = (combined_probability * (combined_decimal_odds - 1)) - (1 - combined_probability)
    if parlay_ev <= 0:
        return None

    ev_boost_pct = (
        (combined_probability - combined_market_prob) / combined_market_prob
        if combined_market_prob > 0 else 0.0
    )
    is_high_correlation = (
        len(legs["matchup_id"].unique()) < leg_count or _has_same_direction_pair(legs)
    )
    parlay_conviction = float(legs["Conviction_Score"].mean()) if "Conviction_Score" in legs.columns else pd.NA
    min_leg_prob = float(legs[prob_col].min())

    has_actionable = (
        legs["Pick_Status"].astype(str).eq("Actionable").any()
        if "Pick_Status" in legs.columns else False
    )

    return {
        "parlay_legs": " | ".join(_leg_labels(legs, label_cols)),
        "combined_probability": combined_probability,
        "combined_decimal_odds": combined_decimal_odds,
        "parlay_ev": parlay_ev,
        "legs": leg_count,
        "combined_market_prob": combined_market_prob,
        "ev_boost_pct": ev_boost_pct,
        "is_high_correlation": is_high_correlation,
        "risk_tier": risk_tier,
        "group_id": group_id,
        "best_payout_book": best_book.capitalize() if best_book != "novig" else "Novig",
        "Conviction_Score": parlay_conviction,
        "min_leg_prob": min_leg_prob,
        "has_actionable_anchor": has_actionable,
        "production_safety_mode": bool(strict_mode),
        "model_risk_haircut": PARLAY_PROBABILITY_HAIRCUT if strict_mode else 1.0,
    }


def generate_smart_parlays(
    df: pd.DataFrame,
    num_rr_candidates: int = 5,
    calibration: list | None = None,
) -> pd.DataFrame:
    """Generate +EV parlays (2-leg by default; see MAX_PARLAY_LEGS) from
    quality-filtered candidates.

    Candidate selection ranks by effective_win_probability (shrinkage-adjusted),
    falling back to calibrated_probability if the column is absent. For Overs,
    effective_win_probability < calibrated_probability, which makes it a better
    discriminator of actual win rate. Edge governs single-game sizing; probability
    governs parlay construction.

    ``calibration`` is an optional recap-fitted isotonic table (see
    scripts/fit_calibration.py / core.probability_calibration.load_calibration).
    When provided, every leg probability is mapped predicted->realized BEFORE the
    floors, ranking, combination math, and the EV>0 gate. This is the systemic fix
    for the overconfidence that compounds multiplicatively across legs — fitted on
    all graded slates instead of per-slate shrink knobs.

    Consensus gate: only "Agrees" picks qualify as legs. Falls back to
    Agrees+Neutral when fewer than 3 Agrees candidates are available.

    Cross-game concentration: same-night totals legs sharing a league and direction
    (e.g. two MLB Overs) win and lose together — shared weather/run environment and
    shared model bias. Rather than pretending they are independent (or rewarding the
    positive correlation with a higher joint probability), combos exceeding
    MAX_SAME_DIRECTION_TOTAL_LEGS in any (league, direction) bucket are skipped.

    Combined probability + EV: ``_build_record`` builds these from the same
    (calibrated) ``rank_col`` used for ranking/gating, NOT the raw
    ``calibrated_probability``, so the EV>0 drop and the EV-desc sort run on the
    de-biased number and inflated all-Over parlays no longer pass the gate.
    """
    columns = [
        "parlay_legs", "combined_probability", "combined_decimal_odds", "parlay_ev",
        "legs", "combined_market_prob", "ev_boost_pct", "is_high_correlation",
        "risk_tier", "group_id", "best_payout_book", "Conviction_Score", "min_leg_prob",
        "has_actionable_anchor", "production_safety_mode", "model_risk_haircut",
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    needed = {"edge", "calibrated_probability", "decimal_odds", "market_probability", "matchup_id"}
    if not needed.issubset(df.columns):
        return pd.DataFrame(columns=columns)

    # Use effective_win_probability as the ranking signal when available — it is
    # shrinkage-adjusted for Overs, making it a better predictor of parlay leg win rate.
    # Falls back to calibrated_probability for compatibility with older data.
    rank_col = "effective_win_probability" if "effective_win_probability" in df.columns else "calibrated_probability"

    candidates = df.copy()

    # In production, only explicit Actionable rows with complete pricing and
    # eligibility evidence may become parlay legs. Fallback, stale, degraded,
    # or merely informational rows are not silently upgraded by this layer.
    strict_mode = bool(
        STRICT_PRODUCTION_PARLAYS
        and "Pick_Status" in candidates.columns
        and any(c in candidates.columns for c in (
            "production_eligible", "is_fallback", "fallback_used",
            "degraded_feature_subset_flag", "market_line_source",
        ))
    )
    if strict_mode:
        candidates = candidates[production_candidate_mask(candidates)].copy()
        if candidates.empty:
            return pd.DataFrame(columns=columns)

    # Recap-fitted isotonic calibration: map predicted -> realized win rate before
    # any floor/gate/EV math. Opt-in via the caller so tests and raw consumers see
    # unchanged behavior when no table is passed.
    #
    # The probability floors below are defined in RAW effective_win_probability
    # space (0.58 raw ~ 0.54 realized on the fitted table), so when legs are
    # calibrated the floors must be mapped through the same table — otherwise a
    # 0.58 floor applied to calibrated values silently demands ~0.65 raw.
    min_leg_probability = MIN_LEG_PROBABILITY
    mlb_total_min_prob = MLB_TOTAL_HV_PARLAY_MIN_PROB
    if calibration:
        candidates["_parlay_leg_prob"] = apply_calibration(candidates[rank_col], calibration)
        min_leg_probability = float(
            apply_calibration(pd.Series([MIN_LEG_PROBABILITY]), calibration).iloc[0]
        )
        rank_col = "_parlay_leg_prob"

        # Honest per-leg EV gate (calibrated mode only): the calibrated probability
        # must beat the leg's own break-even price by MIN_LEG_EV_MARGIN. This
        # replaces the raw-space MLB totals floor below — same protection, priced
        # against the leg's actual odds instead of a hand-tuned constant.
        mlb_total_min_prob = None
        if "decimal_odds" in candidates.columns:
            dec = pd.to_numeric(candidates["decimal_odds"], errors="coerce")
            breakeven = 1.0 / dec.where(dec > 1.0)
            candidates = candidates[candidates[rank_col].ge(breakeven + MIN_LEG_EV_MARGIN)]

    # Quality gate on pick status (see PARLAY_ELIGIBLE_STATUSES)
    if "Pick_Status" in candidates.columns:
        candidates = candidates[
            candidates["Pick_Status"].astype(str).isin(PARLAY_ELIGIBLE_STATUSES)
        ]

    # Exclude picks the empirical overlay flagged as belonging to a proven-losing
    # bucket. The consensus gate below already keeps Disagrees legs out, but Neutral
    # legs are admitted in the <3-Agrees fallback, and a proven-losing Neutral bucket
    # (e.g. MLB:under:Neutral ~43%) would otherwise slip in on the model's inflated
    # raw edge. The overlay tags these rows so the parlay path can drop them too.
    if "status_blocker_stage" in candidates.columns:
        candidates = candidates[
            candidates["status_blocker_stage"].astype(str) != "empirical_proven_losing_bucket"
        ]

    # Consensus gate: only "Agrees" legs (Kalshi and model agree direction).
    # Graded 20 May-7 Jun MLB totals (n=182): Agrees hit 61.4% vs Neutral 48.2%.
    # "No Kalshi" is treated as Neutral — Kalshi simply has no market (all NBA totals,
    # many niche games); absence of Kalshi is not a conflicting signal.
    # Falls back to all candidates if consensus_agreement column is absent.
    if "consensus_agreement" in candidates.columns:
        agrees_mask = candidates["consensus_agreement"].astype(str).isin(["Agrees"])
        agrees_candidates = candidates[agrees_mask]
        # Soft fallback: if fewer than 3 Agrees candidates remain, widen to include Neutral/No Kalshi
        if len(agrees_candidates) < 3:
            candidates = candidates[
                candidates["consensus_agreement"].astype(str).isin(["Agrees", "Neutral", "No Kalshi"])
            ]
        else:
            candidates = agrees_candidates

    # Non-Actionable MLB total picks require a stricter floor for parlay legs.
    # May 27: MLB HV/Spec totals 0-6; only Actionable MLB totals reliably hit (2-2).
    # Below Threshold MLB totals are held to the same stricter floor.
    # In calibrated mode this raw-space floor is superseded by the per-leg EV gate
    # above (mlb_total_min_prob is None).
    if (
        mlb_total_min_prob is not None
        and "league" in candidates.columns
        and "Pick_Status" in candidates.columns
        and "best_pick" in candidates.columns
    ):
        is_mlb_nonactionable_total = (
            candidates["league"].astype(str).str.upper().eq("MLB")
            & candidates["best_pick"].astype(str).str.lower().str.contains(r"over|under", regex=True, na=False)
            & candidates["Pick_Status"].astype(str).ne("Actionable")
        )
        candidates = candidates[
            ~is_mlb_nonactionable_total | candidates[rank_col].ge(mlb_total_min_prob)
        ]

    # Per-leg effective_win_probability and edge floor
    candidates = candidates[
        candidates[rank_col].ge(min_leg_probability)
        & candidates["edge"].ge(MIN_LEG_EDGE)
    ]

    if candidates.empty:
        return pd.DataFrame(columns=columns)

    # Actionable-first anchoring: force Actionable picks into the pool so they
    # always appear as legs, then fill remaining slots with best High Variance
    # picks ranked by effective_win_probability (not edge — legs multiply in parlays).
    if "Pick_Status" in candidates.columns:
        actionable = candidates[candidates["Pick_Status"].astype(str).eq("Actionable")]
        speculative = candidates[candidates["Pick_Status"].astype(str).ne("Actionable")]
    else:
        actionable = pd.DataFrame()
        speculative = candidates

    top_actionable = (
        actionable.nlargest(MAX_ACTIONABLE_ANCHORS, rank_col)
        if not actionable.empty else pd.DataFrame()
    )
    remaining_slots = max(0, 20 - len(top_actionable))
    top_speculative = (
        speculative.nlargest(remaining_slots, rank_col)
        if not speculative.empty and remaining_slots > 0 else pd.DataFrame()
    )

    candidate_bets = pd.concat([top_actionable, top_speculative]).dropna(
        subset=["calibrated_probability", "decimal_odds"]
    ).copy()

    if candidate_bets.empty:
        return pd.DataFrame(columns=columns)

    # RR pool: top candidates by effective_win_probability (or fallback)
    rr_candidates = candidates.nlargest(num_rr_candidates, rank_col).dropna(
        subset=["calibrated_probability", "decimal_odds"]
    ).copy()

    label_cols = [c for c in ["best_pick", "team", "away_team", "home_team"] if c in candidate_bets.columns]
    records: list[dict] = []

    leg_counts = tuple(range(2, MAX_PARLAY_LEGS + 1))

    for leg_count in leg_counts:
        if len(candidate_bets) < leg_count:
            continue
        for combo in combinations(candidate_bets.index, leg_count):
            legs = candidate_bets.loc[list(combo)]
            if _violates_direction_cap(legs):
                continue
            if strict_mode and any(
                shares_game(legs.iloc[a], legs.iloc[b])
                for a, b in combinations(range(len(legs)), 2)
            ):
                continue
            risk_tier = "Bankroll Builder" if leg_count == 2 else "Standard"
            rec = _build_record(legs, label_cols, leg_count, risk_tier, prob_col=rank_col, strict_mode=strict_mode)
            if rec is not None:
                records.append(rec)

    # Round-Robin packages from the top quality candidates
    if not rr_candidates.empty and len(rr_candidates) >= 2:
        leg_names = _leg_labels(rr_candidates, label_cols)
        group_hash = hashlib.md5(json.dumps(sorted(leg_names)).encode()).hexdigest()[:8]
        rr_group_id = f"RR_{group_hash}"

        for leg_count in leg_counts:
            if len(rr_candidates) < leg_count:
                continue
            for combo in combinations(rr_candidates.index, leg_count):
                legs = rr_candidates.loc[list(combo)]
                if _violates_direction_cap(legs):
                    continue
                if strict_mode and any(
                    shares_game(legs.iloc[a], legs.iloc[b])
                    for a, b in combinations(range(len(legs)), 2)
                ):
                    continue
                rec = _build_record(legs, label_cols, leg_count, "Round Robin", rr_group_id, prob_col=rank_col, strict_mode=strict_mode)
                if rec is not None:
                    records.append(rec)

    if not records:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(records)

    # Drop duplicate combinations (same legs can appear in both standard and RR pools,
    # potentially in different leg order). Deduplicate on a sorted canonical key.
    result["_canonical_legs"] = result["parlay_legs"].apply(
        lambda s: " | ".join(sorted(str(s).split(" | ")))
    )
    result = result.drop_duplicates(subset=["_canonical_legs"]).drop(columns=["_canonical_legs"]).copy()

    # Sort: highest EV first, then strongest weakest leg. Actionable anchoring no
    # longer grants sort priority (Actionable went 1-4 across Jun 5-10 recaps);
    # has_actionable_anchor remains a display column only.
    result = result.sort_values(
        ["parlay_ev", "min_leg_prob"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return result
