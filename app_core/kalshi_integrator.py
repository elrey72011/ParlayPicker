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

TEAM_CODE_ALIASES = {
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
    "sam houston": "SHSU",
    "sam houston state": "SHSU",
    "florida gulf coast": "FGCU",
    "kennesaw state": "KENN",
    "fairfield": "FAIR",
    "columbia": "COL",
    "pepperdine": "PEPP",
    "unc wilmington": "UNCW",
    "se louisiana": "SELA",
    "louisiana tech": "LT",
    "indiana state": "INST",
    "temple": "TEM",
    "rhode island": "URI",
    "saint marys": "SMC",
    "st marys": "SMC",
    "wichita state": "WICH",
    "memphis": "MEM",
    "southern illinois": "SIU",
}

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


@dataclass
class KalshiMatchResult:
    market_ticker: str | None = None
    event_ticker: str | None = None
    market_title: str | None = None
    market_subtitle: str | None = None
    probability: float | None = None
    status: str = "no_match"
    reason: str = "no_market_for_tickers"
    tried_tickers: list[str] | None = None


class KalshiAPIError(RuntimeError):
    pass


def _normalize_team_token(name: str) -> str:
    s = str(name or "").lower().strip()
    s = s.replace("\u2019", "'").replace("&", " and ")
    s = s.replace("-", " ").replace(".", " ").replace("'", "")
    s = re.sub(r"\bst\b", "state", s)
    s = re.sub(r"\bsaint\b", "st", s)
    s = re.sub(r"\bsoutheastern\b", "se", s)
    s = re.sub(r"\bnorthwestern\b", "nw", s)
    s = re.sub(r"\bnortheastern\b", "ne", s)
    s = re.sub(r"\bsouthwestern\b", "sw", s)
    s = re.sub(r"\bunc\b", "unc", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_team_for_kalshi(team_name: str) -> str:
    return _normalize_team_token(team_name)


def build_kalshi_date_code(game_date: Any) -> str:
    dt = pd.to_datetime(game_date, errors="coerce", utc=True)
    if pd.isna(dt):
        return ""
    return f"{dt.year % 100:02d}{MONTHS[dt.month - 1]}{dt.day:02d}"


def _market_family(row: pd.Series) -> str | None:
    market_type = str(row.get("market_type") or "").strip().lower()
    if market_type.startswith("spread"):
        return "spread"
    if market_type.startswith("total"):
        return "total"
    best_pick = str(row.get("best_pick") or "").strip().lower()
    if best_pick.startswith("over") or best_pick.startswith("under"):
        return "total"
    if best_pick:
        return "spread"
    return None


def _guess_code(team: str) -> str | None:
    token = _normalize_team_token(team)
    if token in TEAM_CODE_ALIASES:
        return TEAM_CODE_ALIASES[token]
    words = [w for w in token.split() if w not in {"the", "of", "and", "university", "college", "state"}]
    if not words:
        return None
    if len(words) == 1:
        return words[0][:4].upper()
    return "".join(w[0] for w in words)[:4].upper()


def _get_markets(params: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        response = requests.get(f"{API_BASE}/markets", params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise KalshiAPIError(str(exc)) from exc
    if not isinstance(payload, dict):
        return []
    return payload.get("markets", [])


def _select_probability(market: dict[str, Any]) -> float | None:
    bid = pd.to_numeric(market.get("yes_bid_dollars"), errors="coerce")
    ask = pd.to_numeric(market.get("yes_ask_dollars"), errors="coerce")
    if pd.notna(bid) and pd.notna(ask):
        return float((bid + ask) / 2.0)

    last = pd.to_numeric(market.get("last_price_dollars"), errors="coerce")
    if pd.notna(last):
        return float(last)

    for key in ("yes_bid_dollars", "yes_ask_dollars"):
        val = pd.to_numeric(market.get(key), errors="coerce")
        if pd.notna(val):
            return float(val)
    return None


def _deterministic_tickers(row: pd.Series) -> tuple[list[str], str | None, str | None, str | None, str, str | None]:
    league = str(row.get("league") or "").upper()
    family = _market_family(row)
    series = LEAGUE_SERIES_MAP.get(league, {}).get(family or "")
    away_code = _guess_code(str(row.get("away_team") or ""))
    home_code = _guess_code(str(row.get("home_team") or ""))
    date_code = build_kalshi_date_code(row.get("game_date"))

    if not date_code:
        return [], series, away_code, home_code, date_code, family
    if not away_code or not home_code:
        return [], series, away_code, home_code, date_code, family
    if not series:
        return [], series, away_code, home_code, date_code, family

    prefix = f"{series}-{date_code}"
    candidates = [f"{prefix}{away_code}{home_code}", f"{prefix}{home_code}{away_code}"]
    return list(dict.fromkeys(candidates)), series, away_code, home_code, date_code, family


def _fetch_series_cache(series_set: set[str]) -> dict[str, dict[str, Any]]:
    """Fetch all open markets for each unique series in one call per series.
    Returns a dict of {ticker: market} for fast lookup.
    """
    cache: dict[str, dict[str, Any]] = {}
    for series in series_set:
        try:
            markets = _get_markets({"series_ticker": series, "status": "open", "limit": 1000})
            for m in markets:
                ticker = str(m.get("ticker") or "")
                if ticker:
                    cache[ticker] = m
        except KalshiAPIError as exc:
            logger.warning("Kalshi series fetch failed for %s: %s", series, exc)
    return cache


def enrich_with_kalshi_markets(best_picks_df: pd.DataFrame) -> pd.DataFrame:
    if best_picks_df is None or best_picks_df.empty:
        return best_picks_df.copy() if isinstance(best_picks_df, pd.DataFrame) else pd.DataFrame()

    if "game_date" not in best_picks_df.columns:
        raise ValueError("best_picks_df missing game_date before Kalshi enrichment")

    out = best_picks_df.copy()
    out["game_date"] = pd.to_datetime(out.get("game_date"), errors="coerce", utc=True)

    for col in [
        "kalshi_probability",
        "kalshi_market_title",
        "kalshi_event_ticker",
        "kalshi_market_ticker",
    ]:
        if col not in out.columns:
            out[col] = pd.NA
    out["kalshi_match_status"] = "no_match"
    out["kalshi_match_reason"] = "no_valid_candidates"
    out["kalshi_tried_tickers"] = "[]"

    # --- BATCH STEP: determine all needed series upfront, fetch once per series ---
    needed_series: set[str] = set()
    row_meta: dict[Any, tuple[list[str], str | None, str | None, str | None, str, str | None]] = {}
    for idx, row in out.iterrows():
        tried, series, away_code, home_code, date_code, family = _deterministic_tickers(row)
        row_meta[idx] = (tried, series, away_code, home_code, date_code, family)
        if series and date_code and away_code and home_code:
            needed_series.add(series)

    # Single API call per unique series (e.g. KXNCAAMBSPREAD, KXNCAAMBTOTAL)
    series_cache = _fetch_series_cache(needed_series)
    logger.info("Kalshi series cache loaded: %d markets across %d series", len(series_cache), len(needed_series))

    # --- MATCH STEP: lookup from cache, no per-row API calls ---
    for idx, row in out.iterrows():
        tried, series, away_code, home_code, date_code, family = row_meta[idx]

        if not date_code:
            out.at[idx, "kalshi_match_reason"] = "missing_date"
            continue
        if not away_code or not home_code:
            out.at[idx, "kalshi_match_reason"] = "missing_team_code"
            continue
        if not series or family not in {"spread", "total"}:
            out.at[idx, "kalshi_match_reason"] = "no_valid_candidates"
            continue

        out.at[idx, "kalshi_tried_tickers"] = json.dumps(tried)
        if not tried:
            out.at[idx, "kalshi_match_reason"] = "no_valid_candidates"
            continue

        # Direct ticker lookup from cache
        market = None
        for ticker in tried:
            if ticker in series_cache:
                market = series_cache[ticker]
                break

        # Fuzzy fallback: scan cache for matching team codes in ticker or title
        if market is None:
            for ticker, m in series_cache.items():
                if not (series and ticker.startswith(series)):
                    continue
                title = f"{m.get('title', '')} {m.get('subtitle', '')}".upper()
                if (away_code in ticker and home_code in ticker) or (away_code in title and home_code in title):
                    market = m
                    break

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


class KalshiIntegrator:
    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return enrich_with_kalshi_markets(df)
