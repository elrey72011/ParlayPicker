"""Versioned API and release contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EngineeringStatus(StrEnum):
    NOT_VERIFIED = "NOT_VERIFIED"
    STAGING_VERIFIED = "STAGING_VERIFIED"
    FAILED = "FAILED"


class CommercialStatus(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REVOKED = "REVOKED"


class MarketStatus(StrEnum):
    UNKNOWN = "UNKNOWN"
    RESEARCH = "RESEARCH"
    QUALIFIED = "QUALIFIED"
    SUSPENDED = "SUSPENDED"


class SalesStatus(StrEnum):
    DISABLED = "DISABLED"
    OWNER_ENABLED = "OWNER_ENABLED"
    PAUSED = "PAUSED"


class PublicationStatus(StrEnum):
    DRAFT = "DRAFT"
    REVIEWED = "REVIEWED"
    VERIFIED = "VERIFIED"
    WITHDRAWN = "WITHDRAWN"


class RecommendationStatus(StrEnum):
    CURRENT = "CURRENT"
    EXPIRED = "EXPIRED"
    PRICE_MOVED = "PRICE_MOVED"
    WITHDRAWN = "WITHDRAWN"
    SETTLED = "SETTLED"


class ProbabilitySemantics(StrEnum):
    """Supported subscriber probability mass is unconditional and push-aware."""

    WIN_UNCONDITIONAL_WITH_PUSH = "win_unconditional_with_push"


class UncertaintyMethod(StrEnum):
    """Conservative EV lowers win mass while holding recorded push mass fixed."""

    FIXED_PUSH_LOWER_WIN_BOUND = "fixed_push_lower_win_bound"


SUPPORTED_MARKETS = frozenset(
    {
        ("NFL", "SPREAD"), ("NFL", "TOTAL"),
        ("NCAAF", "SPREAD"), ("NCAAF", "TOTAL"),
        ("NBA", "SPREAD"), ("NBA", "TOTAL"),
        ("NCAAB", "SPREAD"), ("NCAAB", "TOTAL"),
        ("MLB", "RUN_LINE"), ("MLB", "TOTAL"),
        ("NHL", "PUCK_LINE"), ("NHL", "TOTAL"),
    }
)


class AuthorityBinding(StrictModel):
    schema_version: int = Field(ge=1)
    authority_id: str = Field(min_length=1, max_length=200)
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    upstream_decision_reference: str = Field(min_length=1, max_length=500)
    upstream_gate_result: str
    market_status: MarketStatus
    activation_reference: str = Field(min_length=1, max_length=500)
    validation_artifact_id: str = Field(min_length=1, max_length=500)
    calibration_id: str = Field(min_length=1, max_length=500)
    content_rights_references: list[str] = Field(min_length=1)
    effective_at: datetime
    expires_at: datetime
    revoked: bool = False

    @model_validator(mode="after")
    def current_qualified_authority(self) -> "AuthorityBinding":
        if self.effective_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("authority timestamps must be timezone-aware")
        if self.expires_at <= self.effective_at:
            raise ValueError("authority expiry must follow effective time")
        return self


class Recommendation(StrictModel):
    schema_version: Literal[2] = 2
    recommendation_id: str = Field(min_length=1, max_length=200)
    exact_sport: str
    exact_market_family: str
    canonical_event_id: str = Field(min_length=1, max_length=250)
    selection: str = Field(min_length=1, max_length=250)
    line: float
    sportsbook_id: str = Field(min_length=1, max_length=100)
    odds_american: int
    odds_decimal: float = Field(gt=1.0)
    quote_id: str = Field(min_length=1, max_length=250)
    quote_observed_at: datetime
    provider_updated_at: datetime | None = None
    analysis_generated_at: datetime
    event_start_utc: datetime
    expiry_at: datetime
    model_id: str
    model_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_target_semantics: str
    calibration_id: str
    validation_artifact_id: str
    policy_id: str
    activation_reference: str
    probability_semantics: ProbabilitySemantics
    p_win: float = Field(ge=0.0, le=1.0)
    p_push: float = Field(ge=0.0, le=1.0)
    p_loss: float = Field(ge=0.0, le=1.0)
    mean_ev_per_unit: float
    p_win_conservative: float = Field(ge=0.0, le=1.0)
    conservative_ev_per_unit: float
    uncertainty_method: UncertaintyMethod
    minimum_acceptable_decimal_odds: float = Field(gt=1.0)
    minimum_acceptable_line: float | None = None
    disclosure_version: str

    @field_validator("probability_semantics", mode="before")
    @classmethod
    def normalize_probability_semantics(cls, value: object) -> object:
        if isinstance(value, ProbabilitySemantics):
            return value
        token = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
        aliases = {
            "unconditional",
            "unconditional_win_push_loss",
            "win_unconditional_with_push",
        }
        if token in aliases:
            return ProbabilitySemantics.WIN_UNCONDITIONAL_WITH_PUSH
        raise ValueError("unsupported probability semantics")

    @field_validator("uncertainty_method", mode="before")
    @classmethod
    def normalize_uncertainty_method(cls, value: object) -> object:
        if isinstance(value, UncertaintyMethod):
            return value
        token = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
        if token == UncertaintyMethod.FIXED_PUSH_LOWER_WIN_BOUND.value:
            return UncertaintyMethod.FIXED_PUSH_LOWER_WIN_BOUND
        raise ValueError("unsupported uncertainty method")

    @model_validator(mode="after")
    def valid_recommendation(self) -> "Recommendation":
        if (self.exact_sport, self.exact_market_family) not in SUPPORTED_MARKETS:
            raise ValueError("unsupported exact market")
        numeric = (
            self.line, self.odds_decimal, self.p_win, self.p_push, self.p_loss,
            self.mean_ev_per_unit, self.p_win_conservative,
            self.conservative_ev_per_unit,
            self.minimum_acceptable_decimal_odds,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("recommendation numbers must be finite")
        if abs((self.p_win + self.p_push + self.p_loss) - 1.0) > 1e-6:
            raise ValueError("p_win + p_push + p_loss must equal 1")
        if self.p_win_conservative > self.p_win + 1e-12:
            raise ValueError("p_win_conservative cannot exceed mean p_win")
        mean_expected = self.p_win * self.odds_decimal + self.p_push - 1.0
        if abs(mean_expected - self.mean_ev_per_unit) > 1e-6:
            raise ValueError("mean EV does not match unconditional win/push/loss semantics")
        conservative_expected = (
            self.p_win_conservative * self.odds_decimal + self.p_push - 1.0
        )
        if abs(conservative_expected - self.conservative_ev_per_unit) > 1e-6:
            raise ValueError("conservative EV does not match fixed-push lower win bound")
        times = (self.quote_observed_at, self.analysis_generated_at, self.event_start_utc, self.expiry_at)
        if any(value.tzinfo is None for value in times):
            raise ValueError("recommendation timestamps must be timezone-aware")
        if self.expiry_at > self.event_start_utc:
            raise ValueError("expiry cannot follow event start")
        return self

    def customer_projection(self) -> dict[str, Any]:
        fields = {
            "schema_version", "recommendation_id", "exact_sport", "exact_market_family", "canonical_event_id",
            "selection", "line", "sportsbook_id", "odds_american", "odds_decimal",
            "quote_id", "quote_observed_at", "analysis_generated_at", "event_start_utc", "expiry_at",
            "probability_semantics", "p_win", "p_push", "p_loss", "mean_ev_per_unit",
            "p_win_conservative", "conservative_ev_per_unit", "uncertainty_method",
            "minimum_acceptable_decimal_odds",
            "minimum_acceptable_line", "disclosure_version",
        }
        raw = self.model_dump(mode="json")
        return {key: raw[key] for key in fields}


class ReleaseSubmission(StrictModel):
    schema_version: Literal[2] = 2
    release_id: str = Field(min_length=1, max_length=200)
    revision_id: str = Field(min_length=1, max_length=200)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    environment: str
    reviewed_payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    operator_review_id: str = Field(min_length=1, max_length=200)
    operator_reviewed_at: datetime
    product_code: str = Field(min_length=1, max_length=100)
    recommendations: list[Recommendation] = Field(min_length=1)
    authority: AuthorityBinding

    @model_validator(mode="after")
    def coherent_scope(self) -> "ReleaseSubmission":
        if self.operator_reviewed_at.tzinfo is None:
            raise ValueError("review timestamp must be timezone-aware")
        for item in self.recommendations:
            if item.calibration_id != self.authority.calibration_id:
                raise ValueError("authority calibration does not match recommendation")
            if item.validation_artifact_id != self.authority.validation_artifact_id:
                raise ValueError("authority validation does not match recommendation")
            if item.activation_reference != self.authority.activation_reference:
                raise ValueError("authority activation does not match recommendation")
        return self

    def review_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "release_id": self.release_id,
            "revision_id": self.revision_id,
            "source_commit": self.source_commit,
            "environment": self.environment,
            "product_code": self.product_code,
            "recommendations": [item.model_dump(mode="json") for item in self.recommendations],
            "authority": self.authority.model_dump(mode="json"),
        }

    def customer_projection(self) -> dict[str, Any]:
        expiry = min(item.expiry_at for item in self.recommendations)
        return {
            "schema_version": 2,
            "release_id": self.release_id,
            "revision_id": self.revision_id,
            "published_at": None,
            "expiry_at": expiry.isoformat(),
            "recommendations": [item.customer_projection() for item in self.recommendations],
        }


class LaunchDecision(StrictModel):
    allowed: bool
    reason_codes: list[str]
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
