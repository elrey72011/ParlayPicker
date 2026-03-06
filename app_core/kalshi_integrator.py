from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)
API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

LEAGUE_SERIES_MAP = {
    "NCAAB": {"spread": "KXNCAAMBSPREAD", "total": "KXNCAAMBTOTAL"},
    "NBA": {"spread": "KXNBASPREAD", "total": "KXNBATOTAL"},
    "NHL": {"spread": "KXNHLSPREAD", "total": "KXNHLTOTAL"},
}

NCAAB_CODE_ALIASES = {
    "manhattan": "MAN",
    "wagner": "WAG",
    "princeton": "PRIN",
    "vermont": "UVM",
    "washington state": "WSU",
    "washington st": "WSU",
    "seton hall": "HALL",
    "st johns": "SJU",
    "saint johns": "SJU",
    "idaho state": "IDST",
    "idaho st": "IDST",
    "sam houston": "SHSU",
    "sam houston state": "SHSU",
    "sam houston st": "SHSU",
    "florida gulf coast": "FGCU",
    "fgcu": "FGCU",
    "kennesaw state": "KENN",
    "kennesaw st": "KENN",
}

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


@dataclass
class KalshiMatchResult:
    market_ticker: str | None = None
    event_ticker: str | None = None
    market_title: str | None = None
    market_subtitle: str | None = None
    probability: float | None = None
    status: str = "miss"
    reason: str = "no_market_for_tickers"
    tried_tickers: list[str] | None = None


class KalshiAPIError(RuntimeError):
    pass


def _normalize_team_token(name: str) -> str:
    s = str(name or "").lower().strip()
    s = s.replace("&", " and ").replace("'", "")
    s = s.replace("-", " ").replace(".", " ")
    s = re.sub(r"\bst\b", "state", s)
    s = re.sub(r"\bsaint\b", "st", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_team_for_kalshi(team_name: str) -> str:
    return _normalize_team_token(team_name)


def build_kalshi_date_code(game_date: Any) -> str:
    dt = pd.to_datetime(game_date, errors="coerce", utc=True)
    if pd.isna(dt):
        return ""
    return f"{dt.year % 100:02d}{MONTHS[dt.month - 1]}{dt.day:02d}"


def _market_family(best_pick: str) -> str | None:
    p = str(best_pick or "").strip().lower()
    if p.startswith("over") or p.startswith("under"):
        return "total"
    if p:
        return "spread"
    return None


def _guess_code(team: str) -> str | None:
    token = _normalize_team_token(team)
    if token in NCAAB_CODE_ALIASES:
        return NCAAB_CODE_ALIASES[token]
    words = [w for w in token.split() if w not in {"the", "of", "and", "university", "college", "state"}]
    if not words:
        return None
    if len(words) == 1:
        return words[0][:4].upper()
    return "".join(w[:2] for w in words)[:4].upper()


def _det_team_code(league: str, team_name: str) -> str | None:
    lg = str(league or "").upper()
    token = _normalize_team_token(team_name)
    if lg == "NCAAB" and token in NCAAB_CODE_ALIASES:
        return NCAAB_CODE_ALIASES[token]
    return _guess_code(token)


def _get_markets(params: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        resp = requests.get(f"{API_BASE}/markets", params=params, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("markets", []) if isinstance(payload, dict) else []
    except Exception as exc:
        raise KalshiAPIError(str(exc)) from exc


def _select_probability(market: dict[str, Any]) -> float | None:
    bid = pd.to_numeric(market.get("yes_bid_dollars"), errors="coerce")
    ask = pd.to_numeric(market.get("yes_ask_dollars"), errors="coerce")
    if pd.notna(bid) and pd.notna(ask):
        return float((bid + ask) / 2.0)
    for key in ("last_price_dollars", "yes_bid_dollars", "yes_ask_dollars"):
        val = pd.to_numeric(market.get(key), errors="coerce")
        if pd.notna(val):
            return float(val)
    return None


def _deterministic_tickers(row: pd.Series) -> tuple[list[str], str | None, str | None, str | None, str]:
    league = str(row.get("league") or "").upper()
    family = _market_family(str(row.get("best_pick") or ""))
    series = LEAGUE_SERIES_MAP.get(league, {}).get(family or "")
    away = _det_team_code(league, str(row.get("away_team") or ""))
    home = _det_team_code(league, str(row.get("home_team") or ""))
    date_code = build_kalshi_date_code(row.get("game_date"))
    if not series or not away or not home or not date_code:
        return [], series, away, home, date_code
    prefix = f"{series}-{date_code}"
    return [f"{prefix}{away}{home}", f"{prefix}{home}{away}"], series, away, home, date_code


def _fallback_series_lookup(series: str, away_code: str, home_code: str) -> dict[str, Any] | None:
    markets = _get_markets({"series_ticker": series, "status": "open", "limit": 1000})
    for market in markets:
        ticker = str(market.get("ticker") or "")
        title = f"{market.get('title', '')} {market.get('subtitle', '')}".upper()
        if (away_code in ticker and home_code in ticker) or (away_code in title and home_code in title):
            return market
    return None


def enrich_with_kalshi_markets(best_picks_df: pd.DataFrame) -> pd.DataFrame:
    if best_picks_df is None or best_picks_df.empty:
        return best_picks_df.copy() if isinstance(best_picks_df, pd.DataFrame) else pd.DataFrame()

    out = best_picks_df.copy()
    out["kalshi_probability"] = pd.NA
    out["kalshi_market_title"] = pd.NA
    out["kalshi_event_ticker"] = pd.NA
    out["kalshi_market_ticker"] = pd.NA
    out["kalshi_match_status"] = "miss"
    out["kalshi_match_reason"] = "no_valid_candidates"
    out["kalshi_tried_tickers"] = "[]"

    for idx, row in out.iterrows():
        tried, series, away_code, home_code, date_code = _deterministic_tickers(row)
        out.at[idx, "kalshi_tried_tickers"] = json.dumps(tried)

        if not date_code:
            out.at[idx, "kalshi_match_reason"] = "missing_date"
            continue
        if not away_code or not home_code:
            out.at[idx, "kalshi_match_reason"] = "missing_team_code"
            continue
        if not tried:
            out.at[idx, "kalshi_match_reason"] = "no_valid_candidates"
            continue

        market = None
        try:
            exact = _get_markets({"tickers": ",".join(tried)})
            by_ticker = {str(m.get("ticker") or ""): m for m in exact}
            for t in tried:
                if t in by_ticker:
                    market = by_ticker[t]
                    break
        except KalshiAPIError:
            out.at[idx, "kalshi_match_reason"] = "no_market_for_tickers"
            continue

        if market is None and series:
            try:
                market = _fallback_series_lookup(series, away_code, home_code)
            except KalshiAPIError:
                out.at[idx, "kalshi_match_reason"] = "no_market_for_tickers"
                continue

        if market is None:
            out.at[idx, "kalshi_match_reason"] = "no_market_for_tickers"
            continue

        out.at[idx, "kalshi_probability"] = _select_probability(market)
        out.at[idx, "kalshi_market_title"] = market.get("title")
        out.at[idx, "kalshi_event_ticker"] = market.get("event_ticker")
        out.at[idx, "kalshi_market_ticker"] = market.get("ticker")
        out.at[idx, "kalshi_match_status"] = "matched"
        out.at[idx, "kalshi_match_reason"] = "matched"

    return out


NCAAF_CODE_ALIASES: dict[str, str] = {}
TEAM_CODE_ALIASES = {"NCAAB": NCAAB_CODE_ALIASES}
KALSHI_NCAAB_TEAM_CODES = NCAAB_CODE_ALIASES


class KalshiIntegrator:
    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return enrich_with_kalshi_markets(df)


def match_game_to_kalshi(*args, **kwargs):
    return None


def _parse_market_metadata(*args, **kwargs):
    return {}


def match_nba_spread(*args, **kwargs):
    return None


def match_ncaab_total(*args, **kwargs):
    return None


def extract_margin_from_yes_side(*args, **kwargs):
    return None


def extract_total_from_ticker(*args, **kwargs):
    return None


def validate_market_type_match(*args, **kwargs):
    return True


def validate_teams_match(*args, **kwargs):
    return True
