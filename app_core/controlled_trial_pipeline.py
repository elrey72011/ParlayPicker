"""Small orchestration helpers for controlled-trial candidate review."""

from __future__ import annotations

from datetime import datetime
from typing import Any, MutableMapping

import pandas as pd

from app_core.controlled_trial import attach_reviews, select_review_candidates


def prepare_review_candidates(
    diagnostics: MutableMapping[str, Any], *, now: datetime
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bind the exact candidate pool and identify trial rows before Gemini."""
    pool = diagnostics.get("candidate_authority_df")
    if not isinstance(pool, pd.DataFrame):
        pool = pd.DataFrame()
    from app_core.prediction_evidence import bind_authoritative_candidates
    from app_core.controlled_trial import attest_candidate_integrity
    from core.prospective_uncertainty import prepare_live

    pool = bind_authoritative_candidates(pool)
    pool = attest_candidate_integrity(pool)
    pool = prepare_live(pool)
    trials = select_review_candidates(pool, now=now)
    return pool, trials


def _target_key(row: pd.Series, prefix: str, position: int) -> str:
    candidate = str(row.get("candidate_id") or "").strip()
    return f"candidate:{candidate}" if candidate else f"{prefix}:{position}"


def review_candidates(
    best: pd.DataFrame,
    trials: pd.DataFrame,
    *,
    enabled: bool,
    session_state: Any,
    analysis: pd.DataFrame,
    diagnostics: MutableMapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str | None]:
    """Review the union once, then project verdicts back to both exact frames."""
    best = best.copy()
    trials = trials.copy()
    # A strict best pick may also be the selected trial candidate. Put its trial
    # flags on that row before deduplication so it remains Gemini-eligible.
    best = attach_reviews(best, trials)
    frames = []
    if not best.empty:
        part = best.copy()
        part["_review_target_key"] = [
            _target_key(row, "best", pos) for pos, (_, row) in enumerate(part.iterrows())
        ]
        frames.append(part)
    if not trials.empty:
        part = trials.copy()
        part["_review_target_key"] = [
            _target_key(row, "trial", pos) for pos, (_, row) in enumerate(part.iterrows())
        ]
        frames.append(part)
    targets = (
        pd.concat(frames, ignore_index=True, sort=False)
        .drop_duplicates("_review_target_key", keep="first")
        if frames else pd.DataFrame()
    )
    error = None
    if enabled and not targets.empty:
        try:
            from integrations.gemini_client import run_gemini_analysis

            targets = run_gemini_analysis(
                targets,
                session_state,
                analysis_df=analysis,
                eligible_only=True,
            )
        except Exception as exc:  # fail closed; caller gets one concise warning
            error = str(exc)
            for column, value in {
                "gemini_error": "GEMINI_REVIEW_FAILED",
                "gemini_explanation": "Gemini analysis unavailable",
                "gemini_risk_notes": "Gemini analysis unavailable",
                "gemini_pick": "No Gemini pick",
                "gemini_confidence": "",
                "gemini_flags": "",
                "gemini_reviewed": False,
            }.items():
                targets[column] = value

    if not targets.empty:
        from app_core.gemini_bet_gate import apply_gemini_bet_gate

        targets = apply_gemini_bet_gate(
            targets,
            enabled=enabled,
            product="best_pick",
            diagnostics=diagnostics,
        )
        best = attach_reviews(best, targets)
        trials = attach_reviews(trials, targets)
    pool_reviews = targets.drop(columns=["_review_target_key"], errors="ignore")
    return best, trials, pool_reviews, error
