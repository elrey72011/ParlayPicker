"""Versioned sport-specific policies; unvalidated research settings cannot fund bets."""
from dataclasses import dataclass
from typing import Mapping
import math

SPORTS = ("NFL", "NCAAF", "NBA", "NCAAB", "MLB", "NHL")

@dataclass(frozen=True)
class SportPolicy:
    sport: str
    version: str
    validation_id: str = ""
    update_frequency: str = "daily"
    historical_prior_strength: float = 0.0
    historical_prior_decay: float = 0.0
    historical_effective_sample_cap: float = 0.0
    current_season_weighting: float = 1.0
    minimum_evidence: int = 0
    calibration_policy: str = "separate-sport-market"
    calibration_refresh_frequency: str = "daily"
    provisional_allowed: bool = False
    provisional_stake_cap: float = 0.0
    standard_stake_cap: float = 0.0
    premium_stake_cap: float = 0.0
    premium_requirements: str = "prospective sport/market validation required"
    clv_requirements: str = "prior slates only; paired entry/close quotes"
    gemini_policy: str = "confirm_reduce_hold"
    parlay_eligibility: tuple[str, ...] = ("STANDARD", "PREMIUM")
    regime_detection_policy: str = "same-sport-market only"
    kelly_fraction: float = 0.0
    min_conservative_edge: float = 0.0
    uncertainty_quantile: float = 0.25
    sport_exposure_cap: float = 0.0

    def __post_init__(self):
        if not self.version or not isinstance(self.minimum_evidence, int) or self.minimum_evidence < 0:
            raise ValueError("Version and nonnegative evidence minimum required")
        if self.sport not in SPORTS:
            raise ValueError("Unsupported sport policy")
        for name in ("historical_prior_strength", "historical_effective_sample_cap", "current_season_weighting"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"Invalid {name}")
        for name in ("historical_prior_decay", "provisional_stake_cap", "standard_stake_cap", "premium_stake_cap", "kelly_fraction", "min_conservative_edge", "sport_exposure_cap"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"Invalid {name}")
        if not 0 < self.uncertainty_quantile < 0.5:
            raise ValueError("Conservative quantile must be below the median")
        if self.provisional_allowed and self.sport not in {"NFL", "NCAAF"}:
            raise ValueError("Football provisional policy cannot alter other sports")


def research_policies() -> Mapping[str, SportPolicy]:
    # Cadences are hypotheses, not optimized settings. Zero caps deliberately
    # prevent this unvalidated configuration from changing live stakes.
    return {sport: SportPolicy(sport, "research-v1", update_frequency="weekly" if sport in {"NFL", "NCAAF"} else "daily",
                              calibration_refresh_frequency="weekly" if sport in {"NFL", "NCAAF"} else "daily")
            for sport in SPORTS}
