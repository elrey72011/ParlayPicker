from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import timedelta
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
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

NCAAB_CODE_ALIASES = {
    "manhattan": "MAN",
    "wagner": "WAG",
    "princeton": "PRIN",
    "vermont": "UVM",
    "washington state": "WSU",
    "seton hall": "HALL",
    "st johns": "SJU",
    "saint johns": "SJU",
    "idaho state": "IDST",
    "sam houston": "SHSU",
    "sam houston state": "SHSU",
    "florida gulf coast": "FGCU",
    "kennesaw state": "KENN",
    "fairfield": "FAIR",
    "columbia": "CLMB",
    "pepperdine": "PEPP",
    "unc wilmington": "UNCW",
    "se louisiana": "SELA",
    "louisiana tech": "LT",
    "indiana state": "INST",
    "temple": "TEM",
    "rhode island": "URI",
    "saint marys": "SMC",
    "wichita state": "WICH",
    "memphis": "MEM",
    "southern illinois": "SIU",
    "idaho st": "IDST",
    "sam houston st": "SHSU",
    "fgcu": "FGCU",
    "kennesaw st": "KENN",
    "washington st": "WSU",
}
TEAM_CODE_ALIASES = {"NCAAB": NCAAB_CODE_ALIASES}
KALSHI_NCAAB_TEAM_CODES = NCAAB_CODE_ALIASES


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


class KalshiRateLimitError(KalshiAPIError):
    pass


def _normalize_team_token(name: str) -> str:
    s = str(name or "").lower().strip()
    s = s.replace("&", " and ")
    s = s.replace("'", "")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^st\s+", "saint ", s)
    s = re.sub(r"\bst$", "state", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_team_for_kalshi(team_name: str) -> str:
    return _normalize_team_token(team_name)


def canonical_team_name(team_name: str) -> str:
    return _normalize_team_token(team_name).title()


def build_kalshi_date_code(game_date: Any) -> str:
    ts = pd.to_datetime(game_date, errors="coerce", utc=True)
    if pd.isna(ts):
        return ""
    return f"{ts.year % 100:02d}{MONTHS[ts.month - 1]}{ts.day:02d}"


def _market_family(best_pick: str) -> str | None:
    p = str(best_pick or "").lower()
    if p.startswith("over") or p.startswith("under"):
        return "total"
    if p:
        return "spread"
    return None


def _guess_code(name: str) -> str | None:
    token = _normalize_team_token(name)
    mapped = NCAAB_CODE_ALIASES.get(token)
    if mapped:
        return mapped
    words = [w for w in token.split() if w not in {"the", "of", "and", "university", "college", "saint", "state"}]
    if not words:
        return None
    if len(words) == 1:
        return words[0][:4].upper()
    return "".join(w[:2] for w in words)[:4].upper()


def _det_team_code(arg1: str, arg2: str) -> str | None:
    # backward compatible signature: _det_team_code(team, league) or _det_team_code(league, team)
    if str(arg1).upper() in LEAGUE_SERIES_MAP:
        league, team_name = str(arg1).upper(), arg2
    else:
        team_name, league = arg1, str(arg2).upper()
    token = _normalize_team_token(team_name)
    if league == "NCAAB":
        return NCAAB_CODE_ALIASES.get(token) or _guess_code(token)
    return _guess_code(token)


def team_code_for_league(team_name: str, league: str) -> str | None:
    return _det_team_code(league, team_name)


def league_series_ticker(league: str, family: str = "spread") -> str | None:
    return LEAGUE_SERIES_MAP.get(str(league or "").upper(), {}).get(family)


def league_game_prefix(league: str) -> str:
    return str(league_series_ticker(league, "spread") or "")


def build_kalshi_ticker(series_ticker: str, game_date: Any, away_code: str, home_code: str) -> str:
    return f"{series_ticker}-{build_kalshi_date_code(game_date)}{away_code}{home_code}"


def parse_event_ticker_codes(event_ticker: str) -> tuple[str, str] | tuple[None, None]:
    m = re.search(r"-\d{2}[A-Z]{3}\d{2}([A-Z0-9]+)$", str(event_ticker or ""))
    if not m:
        return None, None
    code = m.group(1)
    return code[: len(code) // 2], code[len(code) // 2 :]


def resolve_team_code(team_name: str, league: str = "NCAAB") -> str | None:
    return _det_team_code(league, team_name)


def _select_probability(market: dict[str, Any]) -> float | None:
    yes_bid = pd.to_numeric(market.get("yes_bid_dollars"), errors="coerce")
    yes_ask = pd.to_numeric(market.get("yes_ask_dollars"), errors="coerce")
    if pd.notna(yes_bid) and pd.notna(yes_ask):
        return float((yes_bid + yes_ask) / 2.0)
    for field in ("last_price_dollars", "yes_bid_dollars", "yes_ask_dollars"):
        val = pd.to_numeric(market.get(field), errors="coerce")
        if pd.notna(val):
            return float(val)
    return None


def _get_markets(params: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        resp = requests.get(f"{API_BASE}/markets", params=params, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("markets", []) if isinstance(payload, dict) else []
    except Exception as exc:
        raise KalshiAPIError(str(exc)) from exc


def _fallback_match(series: str, row: pd.Series, away_code: str, home_code: str) -> dict[str, Any] | None:
    params: dict[str, Any] = {"series_ticker": series, "status": "open", "limit": 1000}
    markets = _get_markets(params)
    candidates = []
    for m in markets:
        ticker = str(m.get("ticker") or "")
        text = f"{m.get('title','')} {m.get('subtitle','')}".upper()
        has_codes = away_code in ticker and home_code in ticker
        token_match = away_code in text and home_code in text
        if has_codes or token_match:
            candidates.append(m)
    return candidates[0] if len(candidates) == 1 else None


def _deterministic_tickers(row: pd.Series) -> tuple[list[str], str | None, str | None, str | None]:
    league = str(row.get("league") or "").upper()
    family = _market_family(str(row.get("best_pick") or ""))
    series = league_series_ticker(league, family) if family else None
    away_code = _det_team_code(league, str(row.get("away_team") or ""))
    home_code = _det_team_code(league, str(row.get("home_team") or ""))
    if not series or not away_code or not home_code:
        return [], series, away_code, home_code
    date_code = build_kalshi_date_code(row.get("game_date"))
    if not date_code:
        return [], series, away_code, home_code
    base = f"{series}-{date_code}"
    return [f"{base}{away_code}{home_code}", f"{base}{home_code}{away_code}"], series, away_code, home_code


def enrich_with_kalshi_markets(best_picks_df: pd.DataFrame) -> pd.DataFrame:
    if best_picks_df is None or best_picks_df.empty:
        return best_picks_df.copy() if isinstance(best_picks_df, pd.DataFrame) else pd.DataFrame()

    out = best_picks_df.copy()
    out["kalshi_market_ticker"] = pd.NA
    out["kalshi_event_ticker"] = pd.NA
    out["kalshi_market_title"] = pd.NA
    out["kalshi_market_subtitle"] = pd.NA
    out["kalshi_probability"] = pd.NA
    out["kalshi_match_status"] = "miss"
    out["kalshi_match_reason"] = "no_market_for_tickers"
    out["kalshi_tried_tickers"] = "[]"

    for idx, row in out.iterrows():
        date_code = build_kalshi_date_code(row.get("game_date"))
        tried, series, away_code, home_code = _deterministic_tickers(row)
        out.at[idx, "kalshi_tried_tickers"] = json.dumps(tried)

        if not date_code:
            out.at[idx, "kalshi_match_reason"] = "missing_date"
            continue
        if not away_code or not home_code:
            out.at[idx, "kalshi_match_reason"] = "missing_team_code"
            continue
        if not tried:
            out.at[idx, "kalshi_match_reason"] = "no_market_for_tickers"
            continue

        try:
            deterministic = _get_markets({"tickers": ",".join(tried)})
        except KalshiAPIError:
            out.at[idx, "kalshi_match_status"] = "error"
            out.at[idx, "kalshi_match_reason"] = "api_error"
            continue

        ticker_lookup = {str(m.get("ticker") or ""): m for m in deterministic}
        market = next((ticker_lookup[t] for t in tried if t in ticker_lookup), None)
        reason = "no_market_for_tickers"
        if market is None and series:
            try:
                market = _fallback_match(series, row, away_code, home_code)
                reason = "no_market_for_tickers" if market is None else "matched"
            except KalshiAPIError:
                out.at[idx, "kalshi_match_status"] = "error"
                out.at[idx, "kalshi_match_reason"] = "api_error"
                continue

        if market is None:
            out.at[idx, "kalshi_match_reason"] = reason
            continue

        out.at[idx, "kalshi_market_ticker"] = market.get("ticker")
        out.at[idx, "kalshi_event_ticker"] = market.get("event_ticker")
        out.at[idx, "kalshi_market_title"] = market.get("title")
        out.at[idx, "kalshi_market_subtitle"] = market.get("subtitle")
        out.at[idx, "kalshi_probability"] = _select_probability(market)
        out.at[idx, "kalshi_match_status"] = "matched"
        out.at[idx, "kalshi_match_reason"] = "matched"

    return out


# Compatibility helpers used by legacy modules/scripts.
NCAAF_CODE_ALIASES: dict[str, str] = {}


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


def generate_comprehensive_team_variants(team_name: str):
    token = _normalize_team_token(team_name)
    code = _guess_code(token)
    return [team_name, token, code] if code else [team_name, token]


# Unit-test style examples (pure functions only; no live API calls).
def _example_assertions() -> None:
    assert _det_team_code("NCAAB", "Manhattan") == "MAN"
    assert _det_team_code("NCAAB", "Wagner") == "WAG"
    assert build_kalshi_ticker("KXNCAAMBSPREAD", "2026-01-12", "MAN", "WAG") == "KXNCAAMBSPREAD-26JAN12MANWAG"
    assert _det_team_code("NCAAB", "Princeton") == "PRIN"
    assert _det_team_code("NCAAB", "Vermont") == "UVM"
    assert _det_team_code("NCAAB", "Washington St") == "WSU"
    assert _det_team_code("NCAAB", "Seton Hall") == "HALL"
    assert _det_team_code("NCAAB", "Idaho St") == "IDST"
    assert _det_team_code("NCAAB", "Sam Houston St") == "SHSU"
    assert _det_team_code("NCAAB", "FGCU") == "FGCU"
    assert _det_team_code("NCAAB", "Kennesaw St") == "KENN"
