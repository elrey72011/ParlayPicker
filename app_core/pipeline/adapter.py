"""Sport adapter interface for the shared pipeline (roadmap M1.3).

An adapter is the only sport-specific code the orchestrator calls. It must:
- declare its markets and their settlement rules (``markets``). A market that
  is not supported returns a specific reason, never a guessed probability;
- build point-in-time features from completed games strictly before ``as_of``,
  where ``as_of`` is no later than kickoff (``features``);
- produce a market-specific forecast at the exact signed line (``forecast``);
- grade a forecast from a final score under the declared rules (``grade``).

Adapters perform no provider requests themselves. History and quotes are passed
in, so every network call stays inside an operation with its own one-use claim.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
import math

from app_core.pipeline.store import digest


class AdapterError(ValueError):
    """Developer-authored adapter error code."""


def require(ok, code):
    if not ok:
        raise AdapterError(code)


def instant(value, code='ADAPTER_CLOCK_INVALID'):
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise AdapterError(code) from None
    require(moment.tzinfo is not None, code)
    return moment


@dataclass(frozen=True)
class Market:
    """One market type and how it settles."""
    market_type: str            # e.g. spread_home, total_over
    family: str                 # SPREAD, TOTAL, MONEYLINE, RUN_LINE, PUCK_LINE
    side: str                   # home, away, over, under
    period: str = 'full_game'
    overtime: str = 'included'
    push: str = 'refund'
    supported: bool = True
    unsupported_reason: str = ''


@dataclass(frozen=True)
class FeatureSet:
    sport: str
    event_id: str
    home: str
    away: str
    kickoff_utc: str
    as_of_utc: str
    values: dict
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        require(all(isinstance(v, str) and v for v in (self.sport, self.event_id, self.home, self.away)),
                'ADAPTER_FEATURE_IDENTITY')
        require(self.home != self.away, 'ADAPTER_FEATURE_IDENTITY')
        require(instant(self.as_of_utc) <= instant(self.kickoff_utc), 'ADAPTER_FEATURES_AFTER_KICKOFF')
        require(isinstance(self.values, dict) and self.values
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                        for v in self.values.values()), 'ADAPTER_FEATURE_VALUES')

    @property
    def sha256(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class Forecast:
    sport: str
    event_id: str
    market_type: str
    line: float | None
    probability: float | None
    status: str                 # AVAILABLE or UNAVAILABLE
    reason: str                 # AVAILABLE, or the specific reason it is unavailable
    model_id: str
    target: str
    semantics: str
    features_sha256: str
    projection: float | None = None

    def record(self):
        return asdict(self)


class SportAdapter(ABC):
    sport: str = ''

    @abstractmethod
    def markets(self) -> dict[str, Market]:
        """All market types the adapter knows, including unsupported ones."""

    @abstractmethod
    def features(self, event: dict, history: list[dict], *, as_of_utc: str) -> FeatureSet:
        """Point-in-time features from completed ``history`` games before ``as_of_utc``."""

    @abstractmethod
    def forecast(self, features: FeatureSet, market_type: str, line: float | None) -> Forecast:
        """Market-specific probability at the exact signed line, or a specific reason."""

    def market(self, market_type):
        markets = self.markets()
        require(market_type in markets, 'ADAPTER_MARKET_UNKNOWN')
        return markets[market_type]

    def unavailable(self, features, market_type, line, reason, *, model_id='', target='', semantics=''):
        return Forecast(self.sport, features.event_id, market_type, line, None, 'UNAVAILABLE', reason,
                        model_id, target, semantics, features.sha256)

    def grade(self, forecast: Forecast, *, home_score: int, away_score: int) -> str:
        """WIN, LOSS or PUSH for an available forecast under the declared settlement rules."""
        require(forecast.status == 'AVAILABLE', 'ADAPTER_GRADE_UNAVAILABLE_FORECAST')
        require(all(type(s) is int and s >= 0 for s in (home_score, away_score)), 'ADAPTER_GRADE_SCORE')
        market = self.market(forecast.market_type)
        if market.family in {'SPREAD', 'RUN_LINE', 'PUCK_LINE'}:
            margin = home_score - away_score if market.side == 'home' else away_score - home_score
            value = margin + forecast.line
        elif market.family == 'TOTAL':
            total = home_score + away_score
            value = total - forecast.line if market.side == 'over' else forecast.line - total
        elif market.family == 'MONEYLINE':
            margin = home_score - away_score if market.side == 'home' else away_score - home_score
            value = margin
        else:
            raise AdapterError('ADAPTER_GRADE_FAMILY_UNSUPPORTED')
        return 'WIN' if value > 0 else 'LOSS' if value < 0 else 'PUSH'


_REGISTRY: dict[str, SportAdapter] = {}


def register(adapter: SportAdapter) -> SportAdapter:
    require(isinstance(adapter, SportAdapter) and adapter.sport, 'ADAPTER_INVALID')
    require(adapter.sport not in _REGISTRY or type(_REGISTRY[adapter.sport]) is type(adapter), 'ADAPTER_DUPLICATE')
    _REGISTRY[adapter.sport] = adapter
    return adapter


def adapter_for(sport: str) -> SportAdapter:
    require(sport in _REGISTRY, 'ADAPTER_SPORT_NOT_SUPPORTED')
    return _REGISTRY[sport]
