"""Validation gates for promoting probability models into production."""
from __future__ import annotations

from math import isfinite
from typing import Any, Mapping


def select_best_candidate(candidates: Mapping[str, Mapping[str, Any]]) -> str:
    """Choose by proper scoring rules, never by prediction uniqueness."""
    if not candidates:
        raise ValueError("at least one candidate is required")

    def rank(item: tuple[str, Mapping[str, Any]]) -> tuple[float, float, float, str]:
        name, payload = item
        metrics = payload.get("metrics", payload)
        try:
            log_loss = float(metrics["ll"])
            brier = float(metrics["brier"])
            auc = float(metrics["auc"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"candidate {name!r} is missing valid ll/brier/auc metrics") from exc
        if not all(isfinite(value) for value in (log_loss, brier, auc)):
            raise ValueError(f"candidate {name!r} contains non-finite metrics")
        return log_loss, brier, -auc, name

    return min(candidates.items(), key=rank)[0]


def compare_candidate_to_market(
    candidate: Mapping[str, Any],
    market: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an explicit promotion decision against the market baseline."""
    reasons: list[str] = []
    required = ("ll", "brier", "auc")
    try:
        candidate_values = {key: float(candidate[key]) for key in required}
        market_values = {key: float(market[key]) for key in required}
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "promotable": False,
            "reasons": [f"missing or invalid benchmark metric: {exc}"],
        }

    if not all(isfinite(value) for value in (*candidate_values.values(), *market_values.values())):
        reasons.append("candidate or market benchmark contains non-finite metrics")
    if candidate_values["ll"] >= market_values["ll"]:
        reasons.append("candidate log loss does not beat market implied probability")
    if candidate_values["brier"] >= market_values["brier"]:
        reasons.append("candidate Brier score does not beat market implied probability")

    return {
        "promotable": not reasons,
        "reasons": reasons,
        "log_loss_improvement": market_values["ll"] - candidate_values["ll"],
        "brier_improvement": market_values["brier"] - candidate_values["brier"],
        "auc_change": candidate_values["auc"] - market_values["auc"],
    }
