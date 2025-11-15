# ParlayDesk_AI_Enhanced.py - v9.1 FIXED
# AI-Enhanced parlay finder with sentiment analysis, ML predictions, and live market data
import os, io, json, itertools, re, copy, logging, hashlib, math
from html import escape
from dataclasses import asdict
from typing import Dict, Any, List, Tuple, Optional, Iterable
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
import pytz
from pathlib import Path
import re
from collections import defaultdict

from app_core import (
    APISportsBasketballClient,
    APISportsFootballClient,
    APISportsHockeyClient,
    HistoricalDataBuilder,
    HistoricalMLPredictor,
    MLPredictor,
    RealSentimentAnalyzer,
    SentimentAnalyzer,
    SportsDataNCAABClient,
    SportsDataNCAAFClient,
    SportsDataNBAClient,
    SportsDataNFLClient,
    SportsDataNHLClient,
)

logger = logging.getLogger(__name__)

# ============ HELPER FUNCTIONS ============
def american_to_decimal_safe(odds) -> Optional[float]:
    """
    Safe American→Decimal conversion.
    Returns None for None/0/invalid odds in (-100, 100) or on parsing errors.
    """
    try:
        if odds is None:
            return None
        o = float(odds)
        if abs(o) < 100:
            return None
        if o >= 100:
            return 1.0 + o/100.0
        else:
            return 1.0 + 100.0/abs(o)
    except Exception:
        return None

APP_CFG: Dict[str, Any] = {
    "title": "ParlayDesk - AI-Enhanced Odds Finder",
    "sports_common": [
        "americanfootball_nfl","americanfootball_ncaaf",
        "basketball_nba","basketball_ncaab",
        "baseball_mlb","icehockey_nhl","mma_mixed_martial_arts",
        "soccer_epl","soccer_uefa_champs_league","tennis_atp_singles"
    ]
}


SPORTSDATA_CONFIG: Dict[str, Dict[str, Any]] = {
    "americanfootball_nfl": {
        "label": "NFL",
        "emoji": "🏈",
        "client_class": SportsDataNFLClient,
        "secret_names": (
            "NFL_SPORTSDATA_API_KEY",
            "SPORTSDATA_NFL_KEY",
            "SPORTSDATA_API_KEY",
            "SPORTSDATA_KEY",
        ),
        "help": "Set the NFL_SPORTSDATA_API_KEY secret or request an NFL token from https://sportsdata.io/",
    },
    "basketball_nba": {
        "label": "NBA",
        "emoji": "🏀",
        "client_class": SportsDataNBAClient,
        "secret_names": (
            "NBA_SPORTSDATA_API_KEY",
            "SPORTSDATA_NBA_KEY",
            "SPORTSDATA_API_KEY",
            "SPORTSDATA_KEY",
        ),
        "help": "Set the NBA_SPORTSDATA_API_KEY secret or provide your SportsData.io universal key.",
    },
    "icehockey_nhl": {
        "label": "NHL",
        "emoji": "🏒",
        "client_class": SportsDataNHLClient,
        "secret_names": (
            "NHL_SPORTSDATA_API_KEY",
            "SPORTSDATA_NHL_KEY",
            "SPORTSDATA_API_KEY",
            "SPORTSDATA_KEY",
        ),
        "help": "Set the NHL_SPORTSDATA_API_KEY secret or reuse your SportsData.io master key.",
    },
    "americanfootball_ncaaf": {
        "label": "NCAAF",
        "emoji": "🎓🏈",
        "client_class": SportsDataNCAAFClient,
        "secret_names": (
            "NCAAF_SPORTSDATA_API_KEY",
            "SPORTSDATA_NCAAF_KEY",
            "SPORTSDATA_API_KEY",
            "SPORTSDATA_KEY",
        ),
        "help": "Set the NCAAF_SPORTSDATA_API_KEY secret or reuse your SportsData.io key for college football.",
    },
    "basketball_ncaab": {
        "label": "NCAAB",
        "emoji": "🎓🏀",
        "client_class": SportsDataNCAABClient,
        "secret_names": (
            "NCAAB_SPORTSDATA_API_KEY",
            "SPORTSDATA_NCAAB_KEY",
            "SPORTSDATA_API_KEY",
            "SPORTSDATA_KEY",
        ),
        "help": "Set the NCAAB_SPORTSDATA_API_KEY secret or reuse your SportsData.io key for college hoops.",
    },
}


TRACKED_PARLAYS_FILE = Path(__file__).resolve().parent / "tracked_parlays.json"


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_tracked_parlays_from_disk() -> List[Dict[str, Any]]:
    if not TRACKED_PARLAYS_FILE.exists():
        return []
    try:
        with TRACKED_PARLAYS_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return data
    except Exception:
        logger.debug("Failed to load tracked parlays from disk", exc_info=True)
    return []


def _write_tracked_parlays_to_disk(parlays: List[Dict[str, Any]]) -> None:
    try:
        TRACKED_PARLAYS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with TRACKED_PARLAYS_FILE.open("w", encoding="utf-8") as handle:
            json.dump(parlays, handle, indent=2)
    except Exception:
        logger.debug("Failed to persist tracked parlays", exc_info=True)


def get_tracked_parlays_state() -> List[Dict[str, Any]]:
    tracked = st.session_state.get('tracked_parlays')
    if tracked is None:
        tracked = _load_tracked_parlays_from_disk()
        st.session_state['tracked_parlays'] = tracked
    return tracked


def _parlay_signature(legs: List[Dict[str, Any]]) -> str:
    tokens: List[str] = []
    for leg in legs or []:
        token = "|".join(
            str(leg.get(key, ""))
            for key in ("event_id", "market", "side", "point", "team")
        )
        tokens.append(token)
    base = "||".join(sorted(tokens)) if tokens else str(datetime.utcnow().timestamp())
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def save_parlay_for_tracking(
    parlay_row: Dict[str, Any],
    title: str,
    index: int,
    timezone_label: Optional[str] = None,
) -> Tuple[bool, str]:
    legs_payload: List[Dict[str, Any]] = []
    commence_candidates: List[datetime] = []

    for leg in parlay_row.get('legs', []):
        point_val = _safe_float(leg.get('point'))
        commence_iso: Optional[str] = None
        raw_commence = leg.get('commence_time') or leg.get('kickoff')
        dt_obj = None
        if raw_commence is not None:
            dt_obj = _parse_commence_time(raw_commence)
        if dt_obj is not None:
            commence_candidates.append(dt_obj)
            commence_iso = dt_obj.isoformat()
        elif isinstance(raw_commence, str):
            commence_iso = raw_commence

        legs_payload.append({
            'event_id': leg.get('event_id'),
            'label': leg.get('label'),
            'market': leg.get('market'),
            'type': leg.get('type'),
            'team': leg.get('team'),
            'side': leg.get('side'),
            'point': point_val,
            'sport_key': leg.get('sport_key'),
            'home_team': leg.get('home_team'),
            'away_team': leg.get('away_team'),
            'commence_time': commence_iso,
            'decimal_odds': _safe_float(leg.get('d')),
        })

    if not legs_payload:
        return False, "No legs available to save."

    signature = _parlay_signature(legs_payload)
    tracked = get_tracked_parlays_state()
    timezone_name = timezone_label or st.session_state.get('user_timezone') or 'UTC'
    now_utc = datetime.utcnow().replace(tzinfo=pytz.UTC).isoformat()
    target_commence = min(commence_candidates).isoformat() if commence_candidates else None

    analysis_payload = {
        'decimal_odds': _safe_float(parlay_row.get('d')),
        'ai_probability': _safe_float(parlay_row.get('p_ai')),
        'market_probability': _safe_float(parlay_row.get('p')),
        'ai_ev': _safe_float(parlay_row.get('ev_ai')),
        'market_ev': _safe_float(parlay_row.get('ev_market')),
        'ai_score': _safe_float(parlay_row.get('ai_score')),
        'kalshi_factor': _safe_float(parlay_row.get('kalshi_factor')),
    }
    analysis_payload = {k: v for k, v in analysis_payload.items() if v is not None}

    record: Dict[str, Any] = {
        'parlay_id': signature,
        'name': f"{title} #{index}",
        'saved_at_utc': now_utc,
        'analysis_timezone': timezone_name,
        'source_title': title,
        'leg_count': len(legs_payload),
        'legs': legs_payload,
        'analysis': analysis_payload,
    }
    if target_commence:
        record['target_commence'] = target_commence

    existing_idx = next((i for i, entry in enumerate(tracked) if entry.get('parlay_id') == signature), None)
    if existing_idx is not None:
        existing = tracked[existing_idx]
        record['created_at_utc'] = existing.get('created_at_utc', existing.get('saved_at_utc', now_utc))
        if 'evaluation' in existing:
            record['evaluation'] = existing['evaluation']
        tracked[existing_idx] = record
        message = "Updated the existing tracked parlay."
    else:
        record['created_at_utc'] = now_utc
        tracked.append(record)
        message = "Parlay saved for next-day tracking."

    _write_tracked_parlays_to_disk(tracked)
    st.session_state['tracked_parlays'] = tracked
    return True, message


def remove_tracked_parlay(parlay_id: str) -> bool:
    if not parlay_id:
        return False
    tracked = get_tracked_parlays_state()
    new_list = [entry for entry in tracked if entry.get('parlay_id') != parlay_id]
    if len(new_list) == len(tracked):
        return False
    _write_tracked_parlays_to_disk(new_list)
    st.session_state['tracked_parlays'] = new_list
    return True


def _parse_commence_time(raw_value: Any) -> Optional[datetime]:
    if raw_value is None:
        return None
    if isinstance(raw_value, datetime):
        return raw_value.astimezone(pytz.UTC)
    if isinstance(raw_value, (int, float)):
        try:
            return datetime.fromtimestamp(raw_value, tz=pytz.UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(raw_value, str):
        candidate = raw_value.strip()
        if not candidate:
            return None
        try:
            return datetime.fromisoformat(candidate.replace("Z", "+00:00")).astimezone(pytz.UTC)
        except ValueError:
            try:
                return datetime.fromtimestamp(float(candidate), tz=pytz.UTC)
            except (TypeError, ValueError, OverflowError, OSError):
                return None
    return None


def _normalize_team_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return re.sub(r"[^A-Z]", "", name.upper())


def _extract_score_value(container: Any) -> Optional[float]:
    if container is None:
        return None
    if isinstance(container, (int, float)):
        return float(container)
    if isinstance(container, str):
        try:
            return float(container)
        except ValueError:
            return None
    if isinstance(container, dict):
        for key in ("total", "points", "score", "value", "runs", "goals"):
            if key in container:
                return _extract_score_value(container.get(key))
    return None


FINAL_STATUS_TOKENS = {
    "finished",
    "final",
    "after extra time",
    "after overtime",
    "completed",
    "ended",
    "ft",
    "aet",
    "aot",
}


def _status_text(status_field: Any) -> Optional[str]:
    if isinstance(status_field, dict):
        for key in ("long", "short", "type", "description"):
            value = status_field.get(key)
            if value:
                return str(value)
    elif status_field:
        return str(status_field)
    return None


def _is_final_status(status_text: Optional[str]) -> bool:
    if not status_text:
        return False
    lowered = status_text.lower()
    return any(token in lowered for token in FINAL_STATUS_TOKENS)


def _aggregate_leg_statuses(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "pending"
    statuses = [res.get('status') for res in results]
    if any(status == 'loss' for status in statuses):
        return "miss"
    if any(status in {"pending", "no_data", "missing_key"} for status in statuses):
        return "pending"
    if all(status == 'win' for status in statuses):
        return "hit"
    if all(status in {'win', 'push'} for status in statuses):
        return "push"
    return "pending"


def _evaluate_leg_with_client(
    leg: Dict[str, Any],
    client: Any,
    timezone_label: str,
    games_cache: Dict[Tuple[str, str], List[Dict[str, Any]]],
) -> Dict[str, Any]:
    sport_key = leg.get('sport_key')
    result: Dict[str, Any] = {
        'status': 'pending',
        'game_status': None,
        'home_score': None,
        'away_score': None,
        'reason': None,
    }

    if client is None or not getattr(client, 'is_configured', lambda: False)():
        result['reason'] = 'Missing API-Sports key'
        result['warning'] = 'Provide the appropriate API-Sports key to evaluate saved parlays.'
        result['status'] = 'missing_key'
        return result

    commence_dt = _parse_commence_time(leg.get('commence_time'))
    if commence_dt is None:
        commence_dt = _parse_commence_time(leg.get('kickoff'))

    if commence_dt is None:
        result['reason'] = 'Kickoff time unavailable'
        return result

    base_date = commence_dt.date()
    date_candidates = [base_date]
    for offset in (-1, 1):
        alt_date = base_date + timedelta(days=offset)
        if alt_date not in date_candidates:
            date_candidates.append(alt_date)

    matched_game: Optional[Dict[str, Any]] = None
    for candidate in date_candidates:
        cache_key = (client.SPORT_KEY or sport_key or "unknown", candidate.isoformat())
        if cache_key not in games_cache:
            try:
                games_cache[cache_key] = client.get_games_by_date(candidate, timezone='UTC')
            except Exception:
                games_cache[cache_key] = []
        games = games_cache.get(cache_key, [])
        matched = client.match_game(games, leg.get('home_team'), leg.get('away_team'))
        if matched:
            matched_game = matched
            break

    if not matched_game:
        result['reason'] = 'Game not yet available from API-Sports'
        return result

    status_text = _status_text(matched_game.get('status'))
    result['game_status'] = status_text

    scores = (matched_game.get('scores') or {})
    home_score = _extract_score_value(scores.get('home'))
    away_score = _extract_score_value(scores.get('away'))
    result['home_score'] = home_score
    result['away_score'] = away_score

    if not _is_final_status(status_text):
        result['reason'] = status_text or 'Game in progress'
        return result

    if home_score is None or away_score is None:
        result['status'] = 'no_data'
        result['reason'] = 'Final score unavailable'
        return result

    leg_type = (leg.get('type') or leg.get('market') or '').lower()
    side = (leg.get('side') or '').lower()

    if leg_type == 'moneyline':
        if leg.get('side') == 'home':
            if home_score > away_score:
                result['status'] = 'win'
            elif home_score < away_score:
                result['status'] = 'loss'
            else:
                result['status'] = 'push'
        else:
            if away_score > home_score:
                result['status'] = 'win'
            elif away_score < home_score:
                result['status'] = 'loss'
            else:
                result['status'] = 'push'

    elif leg_type == 'spread':
        point = _safe_float(leg.get('point'))
        if point is None:
            result['status'] = 'no_data'
            result['reason'] = 'Spread point unavailable'
        else:
            if leg.get('side') == 'home':
                adjusted_home = home_score + point
                adjusted_away = away_score
            else:
                adjusted_home = home_score
                adjusted_away = away_score + point
            if adjusted_home > adjusted_away:
                result['status'] = 'win'
            elif adjusted_home < adjusted_away:
                result['status'] = 'loss'
            else:
                result['status'] = 'push'

    elif leg_type == 'total':
        point = _safe_float(leg.get('point'))
        if point is None:
            result['status'] = 'no_data'
            result['reason'] = 'Total point unavailable'
        else:
            total_points = home_score + away_score
            if side.startswith('over'):
                if total_points > point:
                    result['status'] = 'win'
                elif total_points < point:
                    result['status'] = 'loss'
                else:
                    result['status'] = 'push'
            elif side.startswith('under'):
                if total_points < point:
                    result['status'] = 'win'
                elif total_points > point:
                    result['status'] = 'loss'
                else:
                    result['status'] = 'push'
            else:
                result['status'] = 'no_data'
                result['reason'] = 'Unknown totals side'

    else:
        result['status'] = 'no_data'
        result['reason'] = f"Unsupported leg type: {leg.get('type')}"

    return result


def evaluate_tracked_parlays(
    parlays: List[Dict[str, Any]],
    clients: Dict[str, Any],
    timezone_label: str,
) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
    if not parlays:
        return parlays, False, None

    games_cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    warnings: List[str] = []
    changed = False

    for entry in parlays:
        leg_results: List[Dict[str, Any]] = []
        for leg in entry.get('legs', []):
            sport_key = leg.get('sport_key')
            client = clients.get(sport_key)
            result = _evaluate_leg_with_client(leg, client, timezone_label, games_cache)
            if 'warning' in result and result['warning']:
                warnings.append(result['warning'])
            leg_results.append({k: v for k, v in result.items() if k != 'warning'})

        evaluation = {
            'status': _aggregate_leg_statuses(leg_results),
            'legs': leg_results,
            'checked_at': datetime.utcnow().replace(tzinfo=pytz.UTC).isoformat(),
        }

        if entry.get('evaluation') != evaluation:
            entry['evaluation'] = evaluation
            changed = True

    if changed:
        _write_tracked_parlays_to_disk(parlays)
        st.session_state['tracked_parlays'] = parlays

    warning_message = None
    if warnings:
        unique_warnings = sorted(set(warnings))
        warning_message = "\n".join(unique_warnings)

    return parlays, changed, warning_message


def render_saved_parlay_tracker(clients: Dict[str, Any], timezone_label: str) -> None:
    st.markdown("### 🧾 Saved Parlay Tracker")
    tracked = get_tracked_parlays_state()

    with st.expander("View saved parlays and outcomes", expanded=False):
        if not tracked:
            st.info("Save a parlay to track its result after the games conclude.")
            return

        actions_col1, actions_col2 = st.columns([1, 1])
        with actions_col1:
            refresh_clicked = st.button("🔁 Refresh tracked results", key="refresh_tracked_parlays")
        with actions_col2:
            clear_clicked = st.button("🗑️ Clear all tracked parlays", key="clear_tracked_parlays")

        if clear_clicked:
            _write_tracked_parlays_to_disk([])
            st.session_state['tracked_parlays'] = []
            st.success("Cleared all saved parlays.")
            tracked = []
        elif refresh_clicked:
            _, changed, warning_message = evaluate_tracked_parlays(tracked, clients, timezone_label)
            if changed:
                st.success("Updated tracked parlays with the latest results.")
            if warning_message:
                st.warning(warning_message)
            tracked = get_tracked_parlays_state()

        if not tracked:
            st.info("No parlays are currently being tracked.")
            return

        summary_rows: List[Dict[str, Any]] = []
        status_emojis = {
            'hit': '✅ Hit',
            'miss': '❌ Miss',
            'push': '⚖️ Push',
            'pending': '⏳ Pending',
        }

        for entry in tracked:
            evaluation = entry.get('evaluation', {})
            leg_results = evaluation.get('legs', []) or []
            wins = sum(1 for res in leg_results if res.get('status') == 'win')
            losses = sum(1 for res in leg_results if res.get('status') == 'loss')
            pushes = sum(1 for res in leg_results if res.get('status') == 'push')
            pending = sum(1 for res in leg_results if res.get('status') in {'pending', 'no_data', 'missing_key'})

            summary_rows.append({
                'Parlay': entry.get('name') or entry.get('parlay_id'),
                'Legs': entry.get('leg_count', len(entry.get('legs', []))),
                'Status': status_emojis.get(evaluation.get('status'), evaluation.get('status', 'pending').title()),
                'Wins': wins,
                'Losses': losses,
                'Pushes': pushes,
                'Pending': pending,
                'Decimal Odds': entry.get('analysis', {}).get('decimal_odds'),
                'Saved (UTC)': entry.get('created_at_utc'),
                'Last Checked': evaluation.get('checked_at'),
            })

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        for entry in tracked:
            evaluation = entry.get('evaluation', {})
            leg_results = evaluation.get('legs', []) or []
            parlay_label = entry.get('name') or entry.get('parlay_id')
            status_display = status_emojis.get(evaluation.get('status'), evaluation.get('status', 'pending').title())
            with st.expander(f"{parlay_label} — {status_display}"):
                st.caption(f"Created: {entry.get('created_at_utc', 'N/A')} | Last checked: {evaluation.get('checked_at', 'N/A')}")
                detail_rows: List[Dict[str, Any]] = []
                for leg, result in zip(entry.get('legs', []), leg_results):
                    score_display = None
                    if result.get('home_score') is not None and result.get('away_score') is not None:
                        score_display = f"{result['home_score']} - {result['away_score']}"
                    detail_rows.append({
                        'Selection': leg.get('label'),
                        'Type': leg.get('type'),
                        'Result': result.get('status', 'pending').title(),
                        'Score': score_display or '—',
                        'Game Status': result.get('game_status') or '—',
                        'Reason': result.get('reason') or '—',
                    })

                if detail_rows:
                    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No leg details available.")

                if st.button("🗑️ Remove from tracker", key=f"remove_tracked_{entry.get('parlay_id')}"):
                    if remove_tracked_parlay(entry.get('parlay_id')):
                        st.success("Removed parlay from tracker.")
                    else:
                        st.warning("Unable to remove parlay. Please try again.")


def resolve_odds_api_key_with_source() -> Tuple[str, Optional[str]]:
    """Return the active Odds API key and where it was sourced from."""

    # Prefer Streamlit secrets if available so hosted deployments can supply
    # credentials without exposing them in the UI.
    secret_container = getattr(st, "secrets", None)
    if secret_container is not None:
        for secret_name in ("ODDS_API_KEY", "THE_ODDS_API_KEY"):
            try:
                secret_value = secret_container.get(secret_name)
            except Exception:
                secret_value = None
            if secret_value:
                return str(secret_value), f"secret:{secret_name}"

    for env_name in ("ODDS_API_KEY", "THE_ODDS_API_KEY"):
        env_value = os.environ.get(env_name)
        if env_value:
            return env_value, f"env:{env_name}"

    # Fall back to whatever is already in session state (if accessible).
    try:
        session_key = st.session_state.get('api_key', "")
    except Exception:
        session_key = ""
    if session_key:
        return session_key, "session:api_key"

    return "", None


def resolve_odds_api_key() -> str:
    """Lightweight helper so background threads can safely fetch the Odds key."""

    key, _ = resolve_odds_api_key_with_source()
    return key


def render_sidebar_controls() -> Dict[str, Any]:
    """Render configuration controls in the Streamlit sidebar."""

    sidebar = st.sidebar
    sidebar.header("⚙️ Control Center")

    # --------------------- Odds API key ---------------------
    default_odds_key, odds_key_source = resolve_odds_api_key_with_source()
    st.session_state.setdefault('api_key', default_odds_key)
    st.session_state.setdefault('odds_key_source', odds_key_source)
    odds_api_input = sidebar.text_input(
        "The Odds API key",
        value=st.session_state.get('api_key', ""),
        type="password",
        help="Stored for this session so live odds and historical snapshots can load.",
    ).strip()
    if odds_api_input != st.session_state.get('api_key', ""):
        st.session_state['api_key'] = odds_api_input
    if st.session_state.get('api_key'):
        sidebar.caption("✅ The Odds API key configured")
    else:
        sidebar.caption("❌ Enter your The Odds API key to fetch odds data")

    # --------------------- News API key ---------------------
    st.session_state.setdefault('news_api_key', os.environ.get("NEWS_API_KEY", ""))
    news_api_input = sidebar.text_input(
        "NewsAPI key (sentiment)",
        value=st.session_state.get('news_api_key', ""),
        type="password",
        help="Optional. Enables real news sentiment analysis when provided.",
    ).strip()
    if news_api_input != st.session_state.get('news_api_key', ""):
        st.session_state['news_api_key'] = news_api_input
        st.session_state['sentiment_analyzer'] = RealSentimentAnalyzer(news_api_input or None)
    if st.session_state.get('news_api_key'):
        sidebar.caption("📰 Live sentiment enabled")
    else:
        sidebar.caption("ℹ️ Using neutral fallback sentiment")

    # --------------------- API-Sports keys ---------------------
    nfl_key_default, nfl_source_default = resolve_nfl_apisports_key()
    st.session_state.setdefault('nfl_apisports_api_key', nfl_key_default)
    st.session_state.setdefault('nfl_apisports_key_source', nfl_source_default)
    nfl_key_input = sidebar.text_input(
        "NFL API-Sports key",
        value=st.session_state.get('nfl_apisports_api_key', ""),
        type="password",
        help="Used for live NFL context and historical model training.",
    ).strip()
    if nfl_key_input != st.session_state.get('nfl_apisports_api_key', ""):
        st.session_state['nfl_apisports_api_key'] = nfl_key_input
        st.session_state['nfl_apisports_key_source'] = "user"

    nhl_key_default, nhl_source_default = resolve_nhl_apisports_key()
    st.session_state.setdefault('nhl_apisports_api_key', nhl_key_default)
    st.session_state.setdefault('nhl_apisports_key_source', nhl_source_default)
    nhl_key_input = sidebar.text_input(
        "NHL API-Sports key",
        value=st.session_state.get('nhl_apisports_api_key', ""),
        type="password",
        help="Used for live NHL context and historical model training.",
    ).strip()
    if nhl_key_input != st.session_state.get('nhl_apisports_api_key', ""):
        st.session_state['nhl_apisports_api_key'] = nhl_key_input
        st.session_state['nhl_apisports_key_source'] = "user"

    nba_key_default, nba_source_default = resolve_nba_apisports_key()
    st.session_state.setdefault('nba_apisports_api_key', nba_key_default)
    st.session_state.setdefault('nba_apisports_key_source', nba_source_default)
    nba_key_input = sidebar.text_input(
        "NBA API-Sports key",
        value=st.session_state.get('nba_apisports_api_key', ""),
        type="password",
        help="Used for live NBA context and historical model training.",
    ).strip()
    if nba_key_input != st.session_state.get('nba_apisports_api_key', ""):
        st.session_state['nba_apisports_api_key'] = nba_key_input
        st.session_state['nba_apisports_key_source'] = "user"

    sidebar.subheader("📈 SportsData.io keys")
    for sport_key, cfg in SPORTSDATA_CONFIG.items():
        session_key = f"{sport_key}_sportsdata_api_key"
        source_session_key = f"{sport_key}_sportsdata_key_source"
        widget_key = f"{sport_key}_sportsdata_widget"

        if session_key not in st.session_state:
            default_key, default_source = resolve_sportsdata_key(sport_key)
            st.session_state[session_key] = default_key
            st.session_state[source_session_key] = default_source

        key_input = sidebar.text_input(
            f"{cfg['label']} SportsData.io key",
            value=st.session_state.get(session_key, ""),
            type="password",
            help=cfg.get('help', ""),
            key=widget_key,
        ).strip()

        if key_input != st.session_state.get(session_key, ""):
            st.session_state[session_key] = key_input
            st.session_state[source_session_key] = "user" if key_input else None

        if st.session_state.get(session_key):
            sidebar.caption(f"{cfg['emoji']} SportsData.io {cfg['label']} key detected")
        else:
            sidebar.caption(
                f"ℹ️ Add your SportsData.io {cfg['label']} key to enrich live metrics and ML features"
            )

    # --------------------- Time & sport filters ---------------------
    sidebar.subheader("📅 Filters")
    default_tz_name = st.session_state.get('user_timezone', 'America/New_York')
    tz_input = sidebar.text_input(
        "Timezone (IANA)",
        value=default_tz_name,
        help="Controls how kickoff times and date filters are interpreted.",
    ).strip() or default_tz_name
    try:
        tz_obj = pytz.timezone(tz_input)
        tz_name = getattr(tz_obj, 'zone', tz_input) or tz_input
    except Exception:
        tz_obj = pytz.timezone('UTC')
        tz_name = 'UTC'
        sidebar.warning("Invalid timezone entered. Defaulting to UTC.")
    st.session_state['user_timezone'] = tz_name

    default_date = st.session_state.get('selected_date')
    if not default_date:
        default_date = datetime.now(tz_obj).date()
    sel_date = sidebar.date_input(
        "Focus date",
        value=default_date,
        help="Only bets within the selected window around this date are shown.",
    )
    st.session_state['selected_date'] = sel_date

    day_window = sidebar.slider(
        "Include events within ±N days",
        0,
        7,
        int(st.session_state.get('day_window', 0) or 0),
        1,
    )
    st.session_state['day_window'] = day_window

    default_sports = st.session_state.setdefault('selected_sports', APP_CFG["sports_common"][:6])
    sports = sidebar.multiselect(
        "Sports",
        options=APP_CFG["sports_common"],
        default=default_sports,
        format_func=format_sport_label,
        key="selected_sports",
    )

    # --------------------- AI settings ---------------------
    ai_expander = sidebar.expander("🤖 AI Settings", expanded=False)
    with ai_expander:
        use_sentiment = ai_expander.checkbox(
            "Enable Sentiment Analysis",
            value=st.session_state.get('use_sentiment', True),
            help="Analyze news sentiment for each team when computing edges.",
        )

        current_ml_state = bool(st.session_state.get('use_ml_predictions', True))
        use_ml_predictions = ai_expander.checkbox(
            "Enable ML Predictions",
            value=current_ml_state,
            help="Blend trained historical models into probability estimates.",
        )

        toggle_label = "🔌 Disable ML for this session" if use_ml_predictions else "⚡ Re-enable ML predictions"
        toggle_help = (
            "Temporarily turn the historical machine-learning models off. "
            "When disabled, the app falls back to odds + sentiment without training datasets."
            if use_ml_predictions
            else "Turn the historical machine-learning models back on for eligible sports."
        )
        if ai_expander.button(
            toggle_label,
            key="toggle_ml_predictions_button",
            use_container_width=True,
            help=toggle_help,
        ):
            use_ml_predictions = not use_ml_predictions
            st.session_state['use_ml_predictions'] = use_ml_predictions

        if not use_ml_predictions:
            ai_expander.info(
                "ML predictions are disabled. Odds, sentiment, Kalshi, and live data signals still run as usual."
            )
        min_ai_confidence = ai_expander.slider(
            "Minimum AI Confidence",
            0.0,
            1.0,
            float(st.session_state.get('min_ai_confidence', 0.60) or 0.60),
            0.05,
        )
        min_parlay_probability = ai_expander.slider(
            "Minimum Parlay Probability",
            0.20,
            0.60,
            float(st.session_state.get('min_parlay_probability', 0.30) or 0.30),
            0.05,
        )
        max_parlay_probability = ai_expander.slider(
            "Maximum Parlay Probability",
            0.45,
            0.85,
            float(st.session_state.get('max_parlay_probability', 0.65) or 0.65),
            0.05,
        )

    st.session_state['use_sentiment'] = use_sentiment
    st.session_state['use_ml_predictions'] = use_ml_predictions
    st.session_state['min_ai_confidence'] = min_ai_confidence
    st.session_state['min_parlay_probability'] = min_parlay_probability
    st.session_state['max_parlay_probability'] = max_parlay_probability

    return {
        "tz": tz_obj,
        "timezone_name": tz_name,
        "selected_date": sel_date,
        "day_window": day_window,
        "sports": sports,
        "use_sentiment": use_sentiment,
        "use_ml_predictions": use_ml_predictions,
        "min_ai_confidence": min_ai_confidence,
        "min_parlay_probability": min_parlay_probability,
        "max_parlay_probability": max_parlay_probability,
    }


def resolve_nfl_apisports_key() -> Tuple[str, Optional[str]]:
    """Locate the NFL API-Sports key from Streamlit secrets or the environment."""

    secret_container = getattr(st, "secrets", None)
    if secret_container is not None:
        for secret_name in ("NFL_APISPORTS_API_KEY", "APISPORTS_API_KEY", "API_SPORTS_KEY"):
            try:
                secret_value = secret_container.get(secret_name)
            except Exception:
                secret_value = None
            if secret_value:
                return str(secret_value), f"secret:{secret_name}"

    for env_name in ("NFL_APISPORTS_API_KEY", "APISPORTS_API_KEY", "API_SPORTS_KEY"):
        env_value = os.environ.get(env_name)
        if env_value:
            return env_value, f"env:{env_name}"

    return "", None


def resolve_nhl_apisports_key() -> Tuple[str, Optional[str]]:
    """Locate the NHL API-Sports key from Streamlit secrets or the environment."""

    secret_container = getattr(st, "secrets", None)
    if secret_container is not None:
        for secret_name in ("NHL_APISPORTS_API_KEY", "APISPORTS_API_KEY", "API_SPORTS_KEY"):
            try:
                secret_value = secret_container.get(secret_name)
            except Exception:
                secret_value = None
            if secret_value:
                return str(secret_value), f"secret:{secret_name}"

    for env_name in ("NHL_APISPORTS_API_KEY", "APISPORTS_API_KEY", "API_SPORTS_KEY"):
        env_value = os.environ.get(env_name)
        if env_value:
            return env_value, f"env:{env_name}"

    return "", None


def resolve_nba_apisports_key() -> Tuple[str, Optional[str]]:
    """Locate the NBA API-Sports key from Streamlit secrets or the environment."""

    secret_container = getattr(st, "secrets", None)
    if secret_container is not None:
        for secret_name in ("NBA_APISPORTS_API_KEY", "APISPORTS_API_KEY", "API_SPORTS_KEY"):
            try:
                secret_value = secret_container.get(secret_name)
            except Exception:
                secret_value = None
            if secret_value:
                return str(secret_value), f"secret:{secret_name}"

    for env_name in ("NBA_APISPORTS_API_KEY", "APISPORTS_API_KEY", "API_SPORTS_KEY"):
        env_value = os.environ.get(env_name)
        if env_value:
            return env_value, f"env:{env_name}"

    return "", None


def resolve_sportsdata_key(sport_key: str) -> Tuple[str, Optional[str]]:
    """Locate the SportsData.io key for a specific sport."""

    config = SPORTSDATA_CONFIG.get(sport_key, {})
    secret_priority: List[str] = list(dict.fromkeys(config.get("secret_names", ())))
    for fallback_name in ("SPORTSDATA_API_KEY", "SPORTSDATA_KEY"):
        if fallback_name not in secret_priority:
            secret_priority.append(fallback_name)

    secret_container = getattr(st, "secrets", None)
    if secret_container is not None:
        for secret_name in secret_priority:
            try:
                secret_value = secret_container.get(secret_name)
            except Exception:
                secret_value = None
            if secret_value:
                return str(secret_value), f"secret:{secret_name}"

    for env_name in secret_priority:
        env_value = os.environ.get(env_name)
        if env_value:
            return env_value, f"env:{env_name}"

    return "", None


def ensure_sportsdata_clients() -> Dict[str, Any]:
    """Instantiate and synchronize SportsData.io clients for all configured sports."""

    clients: Dict[str, Any] = st.session_state.get('sportsdata_clients', {}) or {}

    for sport_key, cfg in SPORTSDATA_CONFIG.items():
        session_key = f"{sport_key}_sportsdata_api_key"
        source_session_key = f"{sport_key}_sportsdata_key_source"
        api_key = st.session_state.get(session_key, "")
        source = st.session_state.get(source_session_key)

        client = clients.get(sport_key)
        if client is None:
            client = cfg["client_class"](api_key or None, key_source=source)
            clients[sport_key] = client
        else:
            if getattr(client, "api_key", "") != (api_key or ""):
                client.update_api_key(api_key or None, source=source or "user")

    st.session_state['sportsdata_clients'] = clients
    return clients

# Comprehensive mapping of Kalshi team abbreviations → canonical team names.
# The Kalshi markets often reference tickers like "NBA.LAL_GSW" or subtitles using
# short-hands. By centralizing these variations we can translate between
# sportsbook-style names ("Los Angeles Lakers") and Kalshi identifiers ("LAL").
# This dramatically increases the likelihood that we locate the correct Kalshi
# market when validating a parlay leg.
KALSHI_TEAM_ABBREVIATIONS: Dict[str, List[str]] = {
    # ========================= NFL =========================
    "ARIZONA CARDINALS": ["ARI", "ARZ", "AZ"],
    "ATLANTA FALCONS": ["ATL"],
    "BALTIMORE RAVENS": ["BAL"],
    "BUFFALO BILLS": ["BUF"],
    "CAROLINA PANTHERS": ["CAR", "CLT"],
    "CHICAGO BEARS": ["CHI", "CHB"],
    "CINCINNATI BENGALS": ["CIN", "CINC"],
    "CLEVELAND BROWNS": ["CLE"],
    "DALLAS COWBOYS": ["DAL"],
    "DENVER BRONCOS": ["DEN"],
    "DETROIT LIONS": ["DET"],
    "GREEN BAY PACKERS": ["GB", "GBP", "GBE"],
    "HOUSTON TEXANS": ["HOU", "HTX"],
    "INDIANAPOLIS COLTS": ["IND"],
    "JACKSONVILLE JAGUARS": ["JAX", "JAC"],
    "KANSAS CITY CHIEFS": ["KC", "KCC"],
    "LAS VEGAS RAIDERS": ["LV", "LVR"],
    "LOS ANGELES CHARGERS": ["LAC", "LA CHARGERS"],
    "LOS ANGELES RAMS": ["LAR", "LA RAMS"],
    "MIAMI DOLPHINS": ["MIA"],
    "MINNESOTA VIKINGS": ["MIN", "MINN"],
    "NEW ENGLAND PATRIOTS": ["NE", "NEP"],
    "NEW ORLEANS SAINTS": ["NO", "NOS"],
    "NEW YORK GIANTS": ["NYG", "NY GIANTS"],
    "NEW YORK JETS": ["NYJ", "NY JETS"],
    "PHILADELPHIA EAGLES": ["PHI", "PHL", "PHI EAGLES"],
    "PITTSBURGH STEELERS": ["PIT", "PITTSBURGH"],
    "SAN FRANCISCO 49ERS": ["SF", "SFO", "SF 49ERS"],
    "SEATTLE SEAHAWKS": ["SEA", "SEA HAWKS"],
    "TAMPA BAY BUCCANEERS": ["TB", "TBB"],
    "TENNESSEE TITANS": ["TEN", "TENN"],
    "WASHINGTON COMMANDERS": ["WAS", "WSH"],

    # ========================= NBA =========================
    "ATLANTA HAWKS": ["ATL"],
    "BOSTON CELTICS": ["BOS"],
    "BROOKLYN NETS": ["BKN", "BRK"],
    "CHARLOTTE HORNETS": ["CHA", "CHH", "CLT"],
    "CHICAGO BULLS": ["CHI"],
    "CLEVELAND CAVALIERS": ["CLE", "CAVS"],
    "DALLAS MAVERICKS": ["DAL", "MAVS"],
    "DENVER NUGGETS": ["DEN"],
    "DETROIT PISTONS": ["DET"],
    "GOLDEN STATE WARRIORS": ["GSW", "GS"],
    "HOUSTON ROCKETS": ["HOU"],
    "INDIANA PACERS": ["IND"],
    "LOS ANGELES CLIPPERS": ["LAC", "LA CLIPPERS"],
    "LOS ANGELES LAKERS": ["LAL", "LA LAKERS"],
    "MEMPHIS GRIZZLIES": ["MEM"],
    "MIAMI HEAT": ["MIA"],
    "MILWAUKEE BUCKS": ["MIL"],
    "MINNESOTA TIMBERWOLVES": ["MIN", "MINN"],
    "NEW ORLEANS PELICANS": ["NOP", "NO PELICANS"],
    "NEW YORK KNICKS": ["NYK", "NY KNICKS"],
    "OKLAHOMA CITY THUNDER": ["OKC"],
    "ORLANDO MAGIC": ["ORL"],
    "PHILADELPHIA 76ERS": ["PHI", "PHL", "SIXERS"],
    "PHOENIX SUNS": ["PHX"],
    "PORTLAND TRAIL BLAZERS": ["POR", "PTB", "PDX"],
    "SACRAMENTO KINGS": ["SAC"],
    "SAN ANTONIO SPURS": ["SAS", "SA SPURS"],
    "TORONTO RAPTORS": ["TOR"],
    "UTAH JAZZ": ["UTA"],
    "WASHINGTON WIZARDS": ["WAS", "WSH"],

    # ========================= MLB =========================
    "ARIZONA DIAMONDBACKS": ["ARI", "ARZ", "AZ"],
    "ATLANTA BRAVES": ["ATL"],
    "BALTIMORE ORIOLES": ["BAL"],
    "BOSTON RED SOX": ["BOS"],
    "CHICAGO CUBS": ["CHC"],
    "CHICAGO WHITE SOX": ["CWS", "CHW"],
    "CINCINNATI REDS": ["CIN", "CINC"],
    "CLEVELAND GUARDIANS": ["CLE", "CLV"],
    "COLORADO ROCKIES": ["COL"],
    "DETROIT TIGERS": ["DET"],
    "HOUSTON ASTROS": ["HOU"],
    "KANSAS CITY ROYALS": ["KC", "KCR"],
    "LOS ANGELES ANGELS": ["LAA", "LA ANGELS"],
    "LOS ANGELES DODGERS": ["LAD", "LA DODGERS"],
    "MIAMI MARLINS": ["MIA"],
    "MILWAUKEE BREWERS": ["MIL"],
    "MINNESOTA TWINS": ["MIN", "MINN"],
    "NEW YORK METS": ["NYM", "NY METS"],
    "NEW YORK YANKEES": ["NYY", "NY YANKEES"],
    "OAKLAND ATHLETICS": ["OAK"],
    "PHILADELPHIA PHILLIES": ["PHI", "PHL", "PHILS"],
    "PITTSBURGH PIRATES": ["PIT"],
    "SAN DIEGO PADRES": ["SD", "SDP"],
    "SAN FRANCISCO GIANTS": ["SF", "SFG"],
    "SEATTLE MARINERS": ["SEA"],
    "ST. LOUIS CARDINALS": ["STL", "SLC"],
    "TAMPA BAY RAYS": ["TB", "TBR"],
    "TEXAS RANGERS": ["TEX"],
    "TORONTO BLUE JAYS": ["TOR"],
    "WASHINGTON NATIONALS": ["WSH", "WAS"],

    # ========================= NHL =========================
    "ANAHEIM DUCKS": ["ANA"],
    "ARIZONA COYOTES": ["ARI", "ARZ", "AZ"],
    "BOSTON BRUINS": ["BOS"],
    "BUFFALO SABRES": ["BUF"],
    "CALGARY FLAMES": ["CGY"],
    "CAROLINA HURRICANES": ["CAR", "CLT"],
    "CHICAGO BLACKHAWKS": ["CHI"],
    "COLORADO AVALANCHE": ["COL"],
    "COLUMBUS BLUE JACKETS": ["CBJ"],
    "DALLAS STARS": ["DAL"],
    "DETROIT RED WINGS": ["DET"],
    "EDMONTON OILERS": ["EDM"],
    "FLORIDA PANTHERS": ["FLA"],
    "LOS ANGELES KINGS": ["LAK", "LA KINGS"],
    "MINNESOTA WILD": ["MIN", "MINN"],
    "MONTREAL CANADIENS": ["MTL"],
    "NASHVILLE PREDATORS": ["NSH"],
    "NEW JERSEY DEVILS": ["NJD"],
    "NEW YORK ISLANDERS": ["NYI"],
    "NEW YORK RANGERS": ["NYR"],
    "OTTAWA SENATORS": ["OTT"],
    "PHILADELPHIA FLYERS": ["PHI", "PHL"],
    "PITTSBURGH PENGUINS": ["PIT"],
    "SAN JOSE SHARKS": ["SJS"],
    "SEATTLE KRAKEN": ["SEA"],
    "ST. LOUIS BLUES": ["STL"],
    "TAMPA BAY LIGHTNING": ["TB", "TBL"],
    "TORONTO MAPLE LEAFS": ["TOR"],
    "VANCOUVER CANUCKS": ["VAN"],
    "VEGAS GOLDEN KNIGHTS": ["VGK", "VEGAS"],
    "WASHINGTON CAPITALS": ["WSH", "WAS"],
    "WINNIPEG JETS": ["WPG"],
}

# Map teams back to their primary league so we can build league-aware fallbacks
KALSHI_LEAGUE_TEAM_SETS: Dict[str, List[str]] = {
    "NFL": [
        "ARIZONA CARDINALS", "ATLANTA FALCONS", "BALTIMORE RAVENS", "BUFFALO BILLS",
        "CAROLINA PANTHERS", "CHICAGO BEARS", "CINCINNATI BENGALS", "CLEVELAND BROWNS",
        "DALLAS COWBOYS", "DENVER BRONCOS", "DETROIT LIONS", "GREEN BAY PACKERS",
        "HOUSTON TEXANS", "INDIANAPOLIS COLTS", "JACKSONVILLE JAGUARS", "KANSAS CITY CHIEFS",
        "LAS VEGAS RAIDERS", "LOS ANGELES CHARGERS", "LOS ANGELES RAMS", "MIAMI DOLPHINS",
        "MINNESOTA VIKINGS", "NEW ENGLAND PATRIOTS", "NEW ORLEANS SAINTS", "NEW YORK GIANTS",
        "NEW YORK JETS", "PHILADELPHIA EAGLES", "PITTSBURGH STEELERS", "SAN FRANCISCO 49ERS",
        "SEATTLE SEAHAWKS", "TAMPA BAY BUCCANEERS", "TENNESSEE TITANS", "WASHINGTON COMMANDERS"
    ],
    "NBA": [
        "ATLANTA HAWKS", "BOSTON CELTICS", "BROOKLYN NETS", "CHARLOTTE HORNETS",
        "CHICAGO BULLS", "CLEVELAND CAVALIERS", "DALLAS MAVERICKS", "DENVER NUGGETS",
        "DETROIT PISTONS", "GOLDEN STATE WARRIORS", "HOUSTON ROCKETS", "INDIANA PACERS",
        "LOS ANGELES CLIPPERS", "LOS ANGELES LAKERS", "MEMPHIS GRIZZLIES", "MIAMI HEAT",
        "MILWAUKEE BUCKS", "MINNESOTA TIMBERWOLVES", "NEW ORLEANS PELICANS", "NEW YORK KNICKS",
        "OKLAHOMA CITY THUNDER", "ORLANDO MAGIC", "PHILADELPHIA 76ERS", "PHOENIX SUNS",
        "PORTLAND TRAIL BLAZERS", "SACRAMENTO KINGS", "SAN ANTONIO SPURS", "TORONTO RAPTORS",
        "UTAH JAZZ", "WASHINGTON WIZARDS"
    ],
    "MLB": [
        "ARIZONA DIAMONDBACKS", "ATLANTA BRAVES", "BALTIMORE ORIOLES", "BOSTON RED SOX",
        "CHICAGO CUBS", "CHICAGO WHITE SOX", "CINCINNATI REDS", "CLEVELAND GUARDIANS",
        "COLORADO ROCKIES", "DETROIT TIGERS", "HOUSTON ASTROS", "KANSAS CITY ROYALS",
        "LOS ANGELES ANGELS", "LOS ANGELES DODGERS", "MIAMI MARLINS", "MILWAUKEE BREWERS",
        "MINNESOTA TWINS", "NEW YORK METS", "NEW YORK YANKEES", "OAKLAND ATHLETICS",
        "PHILADELPHIA PHILLIES", "PITTSBURGH PIRATES", "SAN DIEGO PADRES", "SAN FRANCISCO GIANTS",
        "SEATTLE MARINERS", "ST. LOUIS CARDINALS", "TAMPA BAY RAYS", "TEXAS RANGERS",
        "TORONTO BLUE JAYS", "WASHINGTON NATIONALS"
    ],
    "NHL": [
        "ANAHEIM DUCKS", "ARIZONA COYOTES", "BOSTON BRUINS", "BUFFALO SABRES",
        "CALGARY FLAMES", "CAROLINA HURRICANES", "CHICAGO BLACKHAWKS", "COLORADO AVALANCHE",
        "COLUMBUS BLUE JACKETS", "DALLAS STARS", "DETROIT RED WINGS", "EDMONTON OILERS",
        "FLORIDA PANTHERS", "LOS ANGELES KINGS", "MINNESOTA WILD", "MONTREAL CANADIENS",
        "NASHVILLE PREDATORS", "NEW JERSEY DEVILS", "NEW YORK ISLANDERS", "NEW YORK RANGERS",
        "OTTAWA SENATORS", "PHILADELPHIA FLYERS", "PITTSBURGH PENGUINS", "SAN JOSE SHARKS",
        "SEATTLE KRAKEN", "ST. LOUIS BLUES", "TAMPA BAY LIGHTNING", "TORONTO MAPLE LEAFS",
        "VANCOUVER CANUCKS", "VEGAS GOLDEN KNIGHTS", "WASHINGTON CAPITALS", "WINNIPEG JETS"
    ],
}

KALSHI_TEAM_LEAGUE_MAP: Dict[str, str] = {
    team: league
    for league, teams in KALSHI_LEAGUE_TEAM_SETS.items()
    for team in teams
}

SPORT_KEY_TO_LEAGUE: Dict[str, str] = {
    "americanfootball_nfl": "NFL",
    "americanfootball_ncaaf": "NCAAF",
    "basketball_nba": "NBA",
    "basketball_ncaab": "NCAAB",
    "baseball_mlb": "MLB",
    "icehockey_nhl": "NHL",
    "mma_mixed_martial_arts": "MMA",
    "soccer_epl": "EPL",
    "soccer_uefa_champs_league": "UEFA",
    "tennis_atp_singles": "TENNIS",
}


def format_sport_label(sport_key: Any) -> str:
    """Return a user-friendly league label for an Odds API sport key."""

    if not isinstance(sport_key, str):
        return str(sport_key)

    if sport_key in SPORT_KEY_TO_LEAGUE:
        return SPORT_KEY_TO_LEAGUE[sport_key]

    if "_" in sport_key:
        return sport_key.split("_")[-1].upper()

    return sport_key.upper()

# ============ REAL SENTIMENT ANALYSIS ENGINE ============
# (moved to app_core.sentiment so it can be reused without importing the
# Streamlit UI. RealSentimentAnalyzer and SentimentAnalyzer are imported above.)

# ============ AI PARLAY OPTIMIZER ============
class AIOptimizer:
    """Optimizes parlay selection using AI insights"""

    def __init__(
        self,
        sentiment_analyzer: SentimentAnalyzer,
        ml_predictor: Optional[MLPredictor],
    ):
        self.sentiment = sentiment_analyzer
        self.ml = ml_predictor
    
    def score_parlay(self, legs: List[Dict]) -> Dict[str, float]:
        """
        Score a parlay combination using AI
        Higher scores = better opportunity
        NOW INCLUDES KALSHI VALIDATION
        """
        def _empty_score() -> Dict[str, Any]:
            return {
                'score': 0.0,
                'ai_ev': 0.0,
                'confidence': 0.0,
                'edge': 0.0,
                'correlation_factor': 1.0,
                'kalshi_factor': 1.0,
                'kalshi_legs': 0,
                'kalshi_boost': 0.0,
                'kalshi_alignment_avg': 0.0,
                'kalshi_alignment_abs_avg': 0.0,
                'kalshi_alignment_positive': 0,
                'kalshi_alignment_negative': 0,
                'kalshi_alignment_count': 0,
                'live_data_factor': 1.0,
                'live_data_boost': 0.0,
                'live_data_legs': 0,
                'live_data_sports': [],
                'apisports_factor': 1.0,
                'apisports_legs': 0,
                'apisports_boost': 0.0,
                'apisports_sports': [],
                'sportsdata_factor': 1.0,
                'sportsdata_legs': 0,
                'sportsdata_boost': 0.0,
                'sportsdata_sports': [],
            }

        if not legs:
            return _empty_score()

        valid_legs: List[Dict[str, Any]] = []
        for leg in legs:
            if not isinstance(leg, dict):
                logger.debug("Skipping non-dict parlay leg: %r", leg)
                continue
            decimal_odds = _safe_float(leg.get('d'))
            base_prob = _safe_float(leg.get('p'))
            if decimal_odds is None or decimal_odds <= 1 or base_prob is None:
                logger.debug(
                    "Skipping parlay leg missing odds/probability: %s",
                    leg.get('label', leg.get('team', 'unknown')),
                )
                continue
            valid_legs.append(leg)

        if not valid_legs:
            return _empty_score()

        legs = valid_legs

        # Calculate combined probability (AI-adjusted)
        combined_prob = 1.0
        combined_confidence = 1.0
        total_edge = 0
        kalshi_boost = 0
        kalshi_legs = 0
        kalshi_alignment_total = 0.0
        kalshi_alignment_abs_total = 0.0
        kalshi_alignment_positive = 0
        kalshi_alignment_negative = 0
        kalshi_alignment_count = 0
        apisports_boost = 0
        apisports_legs = 0
        apisports_sports: set[str] = set()
        sportsdata_boost = 0
        sportsdata_legs = 0
        sportsdata_sports: set[str] = set()
        live_data_legs = 0
        live_data_sports: set[str] = set()

        for leg in legs:
            ai_prob_val = _safe_float(leg.get('ai_prob'))
            if ai_prob_val is None:
                ai_prob_val = _safe_float(leg.get('p'))
            if ai_prob_val is None:
                logger.debug(
                    "Skipping AI probability blend for leg with missing prob: %s",
                    leg.get('label', leg.get('team', 'unknown')),
                )
                continue
            combined_prob *= ai_prob_val

            confidence_val = _safe_float(leg.get('ai_confidence'))
            if confidence_val is None:
                confidence_val = 0.5
            combined_confidence *= max(0.01, min(1.0, confidence_val))

            total_edge += _safe_float(leg.get('ai_edge')) or 0.0

            has_live_context = False

            # KALSHI INTEGRATION: Add Kalshi influence
            if 'kalshi_validation' in leg:
                kv = leg['kalshi_validation']
                if kv.get('kalshi_available'):
                    kalshi_legs += 1
                    # Kalshi provides additional probability estimate
                    kalshi_prob = kv.get('kalshi_prob', 0)
                    sportsbook_prob = leg.get('p', 0)
                    data_source = kv.get('data_source', 'kalshi')
                    source_weight = 0.6 if data_source and 'synthetic' in data_source else 1.0

                    # Track Kalshi vs. model alignment before blending probabilities.
                    ai_pre_kalshi = leg.get('ai_prob_before_kalshi')
                    if ai_pre_kalshi is None:
                        delta_hint = leg.get('kalshi_alignment_delta')
                        if isinstance(delta_hint, (int, float)):
                            ai_pre_kalshi = leg.get('ai_prob', sportsbook_prob) - delta_hint
                    if ai_pre_kalshi is None:
                        ai_pre_kalshi = leg.get('ai_prob', sportsbook_prob)
                    alignment_delta = kalshi_prob - ai_pre_kalshi
                    kalshi_alignment_total += alignment_delta
                    kalshi_alignment_abs_total += abs(alignment_delta)
                    kalshi_alignment_count += 1
                    if alignment_delta >= 0.01:
                        kalshi_alignment_positive += 1
                    elif alignment_delta <= -0.01:
                        kalshi_alignment_negative += 1

                    # If Kalshi and AI both disagree with sportsbook in same direction
                    # that's a strong signal
                    ai_prob = leg.get('ai_prob', sportsbook_prob)

                    if kalshi_prob > sportsbook_prob and ai_prob > sportsbook_prob:
                        # Both Kalshi and AI see value
                        kalshi_boost += 15 * source_weight  # Strong boost
                    elif kalshi_prob < sportsbook_prob and ai_prob < sportsbook_prob:
                        # Both Kalshi and AI skeptical
                        kalshi_boost -= 10 * source_weight  # Penalty
                    elif abs(kalshi_prob - ai_prob) < 0.05:
                        # Kalshi and AI agree (regardless of sportsbook)
                        kalshi_boost += 10 * source_weight  # Agreement boost
                    elif abs(kalshi_prob - sportsbook_prob) < 0.03:
                        # Kalshi confirms market
                        kalshi_boost += 5 * source_weight  # Small boost for confirmation
                    else:
                        # Kalshi contradicts both AI and market
                        kalshi_boost -= 5 * source_weight  # Small penalty for confusion

            apisports_info = leg.get('apisports')
            if apisports_info:
                has_live_context = True
                apisports_legs += 1
                sport_key = apisports_info.get('sport_key') if isinstance(apisports_info, dict) else None
                if sport_key:
                    apisports_sports.add(sport_key)
                    live_data_sports.add(sport_key)
                trend = apisports_info.get('trend')
                if trend == 'hot':
                    apisports_boost += 5
                elif trend == 'cold':
                    apisports_boost -= 5

            sportsdata_info = leg.get('sportsdata')
            if sportsdata_info:
                has_live_context = True
                sportsdata_legs += 1
                sport_key = sportsdata_info.get('sport_key') if isinstance(sportsdata_info, dict) else None
                if sport_key:
                    sportsdata_sports.add(sport_key)
                    live_data_sports.add(sport_key)
                trend = sportsdata_info.get('trend')
                if trend == 'hot':
                    sportsdata_boost += 4
                elif trend == 'cold':
                    sportsdata_boost -= 4
                strength_delta = sportsdata_info.get('strength_delta')
                if isinstance(strength_delta, (int, float)):
                    if strength_delta >= 2.0:
                        sportsdata_boost += 6
                    elif strength_delta <= -2.0:
                        sportsdata_boost -= 6
                turnover_margin = sportsdata_info.get('turnover_margin')
                if isinstance(turnover_margin, (int, float)):
                    if turnover_margin >= 0.5:
                        sportsdata_boost += 2
                    elif turnover_margin <= -0.5:
                        sportsdata_boost -= 2

            if has_live_context:
                live_data_legs += 1

        # Calculate combined decimal odds
        combined_odds = 1.0
        for leg in legs:
            decimal_val = _safe_float(leg.get('d'))
            if decimal_val is None or decimal_val <= 1:
                logger.debug(
                    "Skipping leg with invalid decimal odds in multiplier: %s",
                    leg.get('label', leg.get('team', 'unknown')),
                )
                continue
            combined_odds *= decimal_val

        # AI-enhanced EV
        ai_ev = (combined_prob * combined_odds) - 1.0

        # Correlation penalty (same-game parlays are correlated)
        unique_event_ids = [leg.get('event_id') for leg in legs if leg.get('event_id')]
        unique_games = len(set(unique_event_ids)) if unique_event_ids else len(legs)
        correlation_factor = unique_games / len(legs)
        
        # Kalshi validation factor (0.8 to 1.2 multiplier)
        if kalshi_legs > 0:
            kalshi_factor = 1.0 + (kalshi_boost / 100)
            kalshi_factor = max(0.8, min(1.2, kalshi_factor))  # Clamp between 0.8 and 1.2
        else:
            kalshi_factor = 1.0  # No Kalshi data = neutral
        
        # Final score components with KALSHI integration
        ev_score = ai_ev * 100  # EV contribution
        confidence_score = combined_confidence * 50  # Confidence contribution
        edge_score = total_edge * 150  # Edge is most important
        
        # Calculate base score
        base_score = (edge_score * 0.45 +      # 45% edge
                     ev_score * 0.30 +          # 30% EV
                     confidence_score * 0.25)    # 25% confidence

        # Apply Kalshi factor, correlation factor, and live data adjustments
        apisports_factor = 1.0
        if apisports_legs:
            apisports_factor += apisports_boost / 100.0
            apisports_factor = max(0.9, min(1.1, apisports_factor))

        sportsdata_factor = 1.0
        if sportsdata_legs:
            sportsdata_factor += sportsdata_boost / 100.0
            sportsdata_factor = max(0.9, min(1.1, sportsdata_factor))

        combined_boost = apisports_boost + sportsdata_boost
        live_data_factor = 1.0
        if live_data_legs and combined_boost:
            live_data_factor += combined_boost / 100.0
            live_data_factor = max(0.85, min(1.15, live_data_factor))

        final_score = base_score * correlation_factor * kalshi_factor * live_data_factor

        return {
            'score': final_score,
            'ai_ev': ai_ev,
            'confidence': combined_confidence,
            'edge': total_edge,
            'correlation_factor': correlation_factor,
            'kalshi_factor': kalshi_factor,
            'kalshi_legs': kalshi_legs,
            'kalshi_boost': kalshi_boost,
            'kalshi_alignment_avg': (kalshi_alignment_total / kalshi_alignment_count)
            if kalshi_alignment_count
            else 0.0,
            'kalshi_alignment_abs_avg': (kalshi_alignment_abs_total / kalshi_alignment_count)
            if kalshi_alignment_count
            else 0.0,
            'kalshi_alignment_positive': kalshi_alignment_positive,
            'kalshi_alignment_negative': kalshi_alignment_negative,
            'kalshi_alignment_count': kalshi_alignment_count,
            'live_data_factor': live_data_factor,
            'live_data_boost': combined_boost,
            'live_data_legs': live_data_legs,
            'live_data_sports': sorted(live_data_sports),
            'apisports_factor': apisports_factor,
            'apisports_legs': apisports_legs,
            'apisports_boost': apisports_boost,
            'apisports_sports': sorted(apisports_sports),
            'sportsdata_factor': sportsdata_factor,
            'sportsdata_legs': sportsdata_legs,
            'sportsdata_boost': sportsdata_boost,
            'sportsdata_sports': sorted(sportsdata_sports),
        }

# ============ KALSHI INTEGRATION ============

class KalshiIntegrator:
    """Integrates Kalshi prediction market odds and analysis"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or os.environ.get("KALSHI_API_KEY")
        self.api_secret = api_secret or os.environ.get("KALSHI_API_SECRET")
        self.base_url = "https://api.elections.kalshi.com/trade-api/v2"
        self.demo_url = "https://demo-api.elections.kalshi.com/trade-api/v2"

        # Use demo for testing, production for live
        self.api_url = self.base_url if self.api_key else self.demo_url

        self.headers = {
            "Content-Type": "application/json"
        }

        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

        # Synthetic fallback cache when Kalshi API is unavailable (e.g., network blocks)
        self._using_synthetic_data = False
        self._synthetic_markets: List[Dict[str, Any]] = []
        self._synthetic_orderbooks: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self._synthetic_market_by_team: Dict[str, Dict[str, Any]] = {}
        self.last_error: Optional[str] = None

    # -------------------- Synthetic helpers --------------------
    def _synthetic_probability(self, team: str, sport_key: Optional[str] = None,
                               sportsbook_prob: Optional[float] = None) -> float:
        """Generate a deterministic synthetic probability for a team."""

        team_upper = team.upper()
        base_prob = sportsbook_prob if sportsbook_prob is not None else 0.52

        # Deterministic offset based on team characters (stable pseudo-random)
        ordinal_sum = sum(ord(c) for c in team_upper if c.isalpha())
        offset = ((ordinal_sum % 21) - 10) / 200.0  # -0.05 .. +0.05

        league = KALSHI_TEAM_LEAGUE_MAP.get(team_upper)
        if sport_key and sport_key in SPORT_KEY_TO_LEAGUE and not league:
            league = SPORT_KEY_TO_LEAGUE[sport_key]

        league_bias = {
            "NFL": 0.015,
            "NBA": 0.010,
            "MLB": 0.005,
            "NHL": 0.005,
        }.get(league, 0.0)

        synthetic = base_prob + offset + league_bias
        return max(0.05, min(0.95, synthetic))

    def _synthetic_ticker_for_team(self, team: str, league: Optional[str]) -> str:
        team_key = re.sub(r"[^A-Z0-9]", "", team.upper())
        league_key = league or "SPORTS"
        return f"SIM.{league_key}.{team_key[:8]}"

    def _ensure_synthetic_data(self) -> None:
        if self._synthetic_markets:
            return

        # Build synthetic markets for every team we know about so UI has coverage
        now = datetime.utcnow()
        expiry = (now + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")

        for team, abbrs in KALSHI_TEAM_ABBREVIATIONS.items():
            league = KALSHI_TEAM_LEAGUE_MAP.get(team)
            ticker = self._synthetic_ticker_for_team(team, league)
            prob = self._synthetic_probability(team)
            price = int(round(prob * 100))

            market = {
                "ticker": ticker,
                "title": f"{team.title()} confidence (synthetic Kalshi)",
                "subtitle": "Synthetic fallback market generated locally",
                "series_ticker": "SPORTS",
                "status": "open",
                "close_time": expiry,
                "league": league,
                "synthetic": True,
                "team": team,
                "abbreviation": abbrs[0] if abbrs else None,
            }

            self._synthetic_markets.append(market)
            self._synthetic_orderbooks[ticker] = {
                "yes": [{"price": price, "contracts": 100}],
                "no": [{"price": 100 - price, "contracts": 100}],
            }
            self._synthetic_market_by_team[team] = market

    def using_synthetic_data(self) -> bool:
        return self._using_synthetic_data

    def get_synthetic_market_for_team(self, team: str) -> Optional[Dict[str, Any]]:
        self._ensure_synthetic_data()
        return self._synthetic_market_by_team.get(team.upper())

    def synthetic_probability(self, team: str, sport_key: Optional[str] = None,
                               sportsbook_prob: Optional[float] = None) -> float:
        """Public helper to compute synthetic probabilities for validation."""
        self._ensure_synthetic_data()
        return self._synthetic_probability(team, sport_key, sportsbook_prob)
    
    def get_markets(self, category: str = "sports", status: str = "open") -> List[Dict]:
        """Fetch available Kalshi markets.

        Args:
            category: 'sports', 'politics', 'economics', etc.
            status: 'open', 'closed', 'settled'

        Returns:
            List of market dictionaries.
        """
        if self._using_synthetic_data:
            self._ensure_synthetic_data()
            return copy.deepcopy(self._synthetic_markets)

        try:
            endpoint = f"{self.api_url}/markets"
            params = {
                "limit": 100,
                "status": status
            }

            if category:
                params["series_ticker"] = category.upper()

            response = requests.get(endpoint, headers=self.headers, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                markets = data.get("markets", [])
                if markets:
                    self.last_error = None
                    return markets
                else:
                    self.last_error = "Kalshi API returned no markets"
            else:
                self.last_error = f"Kalshi API responded with status {response.status_code}"

        except Exception as e:
            self.last_error = str(e)
            st.warning(f"Error fetching Kalshi markets: {str(e)}")

        # Fallback to synthetic data when API fails or returns nothing
        self._using_synthetic_data = True
        self._ensure_synthetic_data()
        return copy.deepcopy(self._synthetic_markets)
    
    def get_sports_markets(self) -> List[Dict]:
        """Get all active sports betting markets"""
        all_markets = self.get_markets()
        
        # Filter for sports-related markets
        sports_keywords = ['NFL', 'NBA', 'MLB', 'NHL', 'UFC', 'SOCCER', 'TENNIS', 
                          'GOLF', 'FOOTBALL', 'BASKETBALL', 'BASEBALL', 'HOCKEY']
        
        sports_markets = []
        for market in all_markets:
            title = market.get('title', '').upper()
            ticker = market.get('ticker', '').upper()
            
            if any(keyword in title or keyword in ticker for keyword in sports_keywords):
                sports_markets.append(market)
        
        return sports_markets
    
    def get_market_details(self, market_ticker: str) -> Dict:
        """Get detailed information about a specific market"""
        if self._using_synthetic_data:
            self._ensure_synthetic_data()
            market = next((m for m in self._synthetic_markets if m.get('ticker') == market_ticker), None)
            return copy.deepcopy(market) if market else {}

        try:
            endpoint = f"{self.api_url}/markets/{market_ticker}"
            response = requests.get(endpoint, headers=self.headers, timeout=10)

            if response.status_code == 200:
                return response.json().get("market", {})
            else:
                return {}

        except Exception as e:
            st.warning(f"Error fetching market details: {str(e)}")
            return {}

    def get_orderbook(self, market_ticker: str) -> Dict:
        """Get current orderbook (bids/asks) for a market"""
        if self._using_synthetic_data:
            self._ensure_synthetic_data()
            orderbook = self._synthetic_orderbooks.get(market_ticker)
            return copy.deepcopy(orderbook) if orderbook else {}

        try:
            endpoint = f"{self.api_url}/markets/{market_ticker}/orderbook"
            response = requests.get(endpoint, headers=self.headers, timeout=10)

            if response.status_code == 200:
                return response.json().get("orderbook", {})
            else:
                return {}

        except Exception as e:
            st.warning(f"Error fetching orderbook: {str(e)}")
            return {}
    
    def compare_with_sportsbook(
        self,
        kalshi_market: Dict,
        sportsbook_odds: Dict,
    ) -> Dict:
        """Compare Kalshi prediction prices with a sportsbook listing."""

        result: Dict[str, Any] = {
            "kalshi_prob": None,
            "sportsbook_prob": None,
            "discrepancy": 0.0,
            "edge": 0.0,
            "recommendation": "⚪ Insufficient data for comparison",
            "has_arbitrage": False,
        }

        if not kalshi_market:
            return result

        kalshi_raw = kalshi_market.get("yes_bid")
        if kalshi_raw is None:
            return result

        kalshi_yes_price = float(kalshi_raw) / 100.0
        result["kalshi_prob"] = kalshi_yes_price

        sb_prob: Optional[float] = None
        if sportsbook_odds and sportsbook_odds.get("price") is not None:
            sb_prob = implied_p_from_american(sportsbook_odds["price"])
            result["sportsbook_prob"] = sb_prob

        if sb_prob is None or kalshi_yes_price <= 0:
            return result

        discrepancy = abs(kalshi_yes_price - sb_prob)
        edge = discrepancy
        recommendation = "🟡 Prices aligned (no significant edge)"

        if kalshi_yes_price < sb_prob - 0.05:
            edge = sb_prob - kalshi_yes_price
            recommendation = "🟢 BUY YES on Kalshi (underpriced vs sportsbook)"
        elif kalshi_yes_price > sb_prob + 0.05:
            edge = kalshi_yes_price - sb_prob
            recommendation = "🟢 BUY NO on Kalshi (or take sportsbook)"

        result["discrepancy"] = discrepancy
        result["edge"] = edge
        result["recommendation"] = recommendation
        result["has_arbitrage"] = discrepancy > 0.10
        return result
    
    def find_arbitrage_opportunities(self, kalshi_markets: List[Dict], 
                                     sportsbook_events: List[Dict]) -> List[Dict]:
        """
        Find arbitrage opportunities between Kalshi and traditional sportsbooks
        
        Returns list of arbitrage opportunities with expected profit
        """
        arbitrage_opps = []
        
        for kalshi_market in kalshi_markets:
            title = kalshi_market.get('title', '')
            
            # Try to match with sportsbook events
            for sb_event in sportsbook_events:
                # Simple matching logic (can be enhanced)
                home_team = sb_event.get('home_team', '')
                away_team = sb_event.get('away_team', '')
                
                if home_team in title or away_team in title:
                    comparison = self.compare_with_sportsbook(kalshi_market, sb_event)
                    
                    if comparison['has_arbitrage']:
                        arbitrage_opps.append({
                            'kalshi_market': title,
                            'kalshi_ticker': kalshi_market.get('ticker'),
                            'sportsbook_game': f"{away_team} @ {home_team}",
                            'comparison': comparison
                        })
        
        return arbitrage_opps
    
    def analyze_kalshi_market(self, market: Dict, sentiment_score: float = 0, 
                             ml_probability: float = None) -> Dict:
        """
        Comprehensive analysis of a Kalshi market
        
        Combines:
        - Kalshi orderbook data
        - Sentiment analysis
        - ML predictions
        - Value assessment
        """
        yes_bid = market.get('yes_bid', 0) / 100
        yes_ask = market.get('yes_ask', 100) / 100
        no_bid = market.get('no_bid', 0) / 100
        no_ask = market.get('no_ask', 100) / 100
        
        volume = market.get('volume', 0)
        open_interest = market.get('open_interest', 0)
        
        # Market efficiency (tight spread = efficient)
        yes_spread = yes_ask - yes_bid
        no_spread = no_ask - no_bid
        avg_spread = (yes_spread + no_spread) / 2
        
        efficiency = 1 - avg_spread  # Higher = more efficient
        
        # Compare with AI prediction
        kalshi_implied = yes_bid  # Using bid as market consensus
        ai_edge = 0
        ai_recommendation = "⚪️ No AI prediction available"
        
        if ml_probability:
            ai_edge = ml_probability - kalshi_implied
            
            if ai_edge > 0.10:
                ai_recommendation = f"🟢 STRONG BUY YES - AI sees {ai_edge*100:.1f}% edge"
            elif ai_edge < -0.10:
                ai_recommendation = f"🟢 STRONG BUY NO - AI sees {abs(ai_edge)*100:.1f}% edge"
            elif abs(ai_edge) < 0.05:
                ai_recommendation = "🟡 FAIR PRICE - AI agrees with market"
            else:
                ai_recommendation = f"🟡 MODERATE EDGE - AI sees {ai_edge*100:.1f}% edge"
        
        # Calculate overall score (0-100)
        liquidity_score = min(100, (volume + open_interest) / 100)
        efficiency_score = efficiency * 100
        edge_score = max(0, abs(ai_edge) * 100) if ml_probability else 0
        sentiment_factor = (sentiment_score + 1) / 2 * 100  # Normalize -1 to 1 range to 0-100
        
        overall_score = (
            liquidity_score * 0.3 +
            efficiency_score * 0.2 +
            edge_score * 0.4 +
            sentiment_factor * 0.1
        )
        
        return {
            'yes_bid': yes_bid,
            'yes_ask': yes_ask,
            'no_bid': no_bid,
            'no_ask': no_ask,
            'yes_spread': yes_spread,
            'no_spread': no_spread,
            'efficiency': efficiency,
            'volume': volume,
            'open_interest': open_interest,
            'kalshi_implied': kalshi_implied,
            'ai_edge': ai_edge,
            'ai_recommendation': ai_recommendation,
            'overall_score': overall_score,
            'liquidity_score': liquidity_score,
            'efficiency_score': efficiency_score,
            'edge_score': edge_score,
            'sentiment_score': sentiment_score,
        }


def evaluate_tracked_parlays(
    parlays: List[Dict[str, Any]],
    clients: Dict[str, Any],
    timezone_label: str,
) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
    if not parlays:
        return parlays, False, None

    games_cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    warnings: List[str] = []
    changed = False

    for entry in parlays:
        leg_results: List[Dict[str, Any]] = []
        for leg in entry.get('legs', []):
            sport_key = leg.get('sport_key')
            client = clients.get(sport_key)
            result = _evaluate_leg_with_client(leg, client, timezone_label, games_cache)
            if 'warning' in result and result['warning']:
                warnings.append(result['warning'])
            leg_results.append({k: v for k, v in result.items() if k != 'warning'})

        evaluation = {
            'status': _aggregate_leg_statuses(leg_results),
            'legs': leg_results,
            'checked_at': datetime.utcnow().replace(tzinfo=pytz.UTC).isoformat(),
        }
    
    def _calculate_overall_score(self, ai_edge: float, sentiment: float, 
                                 efficiency: float, volume: int) -> float:
        """Calculate overall opportunity score 0-100"""
        # Weight different factors
        edge_score = min(abs(ai_edge) * 200, 50)  # Max 50 points
        sentiment_score = min(abs(sentiment) * 50, 20)  # Max 20 points
        efficiency_score = efficiency * 15  # Max 15 points
        liquidity_score = min(volume / 100, 15)  # Max 15 points
        
        total = edge_score + sentiment_score + efficiency_score + liquidity_score
        return min(total, 100)
    
    def get_best_opportunities(self, min_score: float = 60) -> List[Dict]:
        """
        Find the best Kalshi betting opportunities
        
        Returns markets with high overall scores
        """
        markets = self.get_sports_markets()
        opportunities = []

        for market in markets:
            # Get orderbook
            ticker = market.get('ticker')
            orderbook = self.get_orderbook(ticker)

            if orderbook:
                # Enhance market data with orderbook
                market['yes_bid'] = orderbook.get('yes', [{}])[0].get('price', 0)
                market['yes_ask'] = orderbook.get('yes', [{}])[-1].get('price', 100)
                market['no_bid'] = orderbook.get('no', [{}])[0].get('price', 0)
                market['no_ask'] = orderbook.get('no', [{}])[-1].get('price', 100)

                # Analyze
                analysis = self.analyze_kalshi_market(market)

                if analysis['overall_score'] >= min_score:
                    opportunity = {
                        'market': market,
                        'analysis': analysis,
                    }
                    opportunities.append(opportunity)
        
        # Sort by score
        opportunities.sort(key=lambda x: x['analysis']['overall_score'], reverse=True)
        
        return opportunities
    
    def calculate_kelly_for_kalshi(self, kalshi_prob: float, ai_prob: float, 
                                   bankroll: float = 1000) -> Dict:
        """
        Calculate Kelly Criterion for Kalshi market
        
        Kalshi uses probability pricing (0-100 cents), not traditional odds
        """
        if ai_prob <= kalshi_prob:
            return {
                'kelly_percentage': 0,
                'recommended_stake': 0,
                'expected_value': 0,
                'recommendation': '🔴 NO EDGE - AI probability not better than Kalshi price'
            }
        
        # Edge calculation for binary market
        edge = ai_prob - kalshi_prob
        
        # Kelly formula for binary outcome: f = edge / (1 - kalshi_prob)
        if kalshi_prob < 0.95:  # Avoid division issues
            kelly_fraction = edge / (1 - kalshi_prob)
        else:
            sidebar.caption(
                f"ℹ️ Add your SportsData.io {cfg['label']} key to enrich live metrics and ML features"
            )

    # --------------------- Time & sport filters ---------------------
    sidebar.subheader("📅 Filters")
    default_tz_name = st.session_state.get('user_timezone', 'America/New_York')
    tz_input = sidebar.text_input(
        "Timezone (IANA)",
        value=default_tz_name,
        help="Controls how kickoff times and date filters are interpreted.",
    ).strip() or default_tz_name
    try:
        tz_obj = pytz.timezone(tz_input)
        tz_name = getattr(tz_obj, 'zone', tz_input) or tz_input
    except Exception:
        tz_obj = pytz.timezone('UTC')
        tz_name = 'UTC'
        sidebar.warning("Invalid timezone entered. Defaulting to UTC.")
    st.session_state['user_timezone'] = tz_name

    default_date = st.session_state.get('selected_date')
    if not default_date:
        default_date = datetime.now(tz_obj).date()
    sel_date = sidebar.date_input(
        "Focus date",
        value=default_date,
        help="Only bets within the selected window around this date are shown.",
    )
    st.session_state['selected_date'] = sel_date

    day_window = sidebar.slider(
        "Include events within ±N days",
        0,
        7,
        int(st.session_state.get('day_window', 0) or 0),
        1,
    )
    st.session_state['day_window'] = day_window

    default_sports = st.session_state.setdefault('selected_sports', APP_CFG["sports_common"][:6])
    sports = sidebar.multiselect(
        "Sports",
        options=APP_CFG["sports_common"],
        default=default_sports,
        format_func=format_sport_label,
        key="selected_sports",
    )

    # --------------------- AI settings ---------------------
    ai_expander = sidebar.expander("🤖 AI Settings", expanded=False)
    with ai_expander:
        use_sentiment = ai_expander.checkbox(
            "Enable Sentiment Analysis",
            value=st.session_state.get('use_sentiment', True),
            help="Analyze news sentiment for each team when computing edges.",
        )

        current_ml_state = bool(st.session_state.get('use_ml_predictions', True))
        use_ml_predictions = ai_expander.checkbox(
            "Enable ML Predictions",
            value=current_ml_state,
            help="Blend trained historical models into probability estimates.",
        )

        toggle_label = "🔌 Disable ML for this session" if use_ml_predictions else "⚡ Re-enable ML predictions"
        toggle_help = (
            "Temporarily turn the historical machine-learning models off. "
            "When disabled, the app falls back to odds + sentiment without training datasets."
            if use_ml_predictions
            else "Turn the historical machine-learning models back on for eligible sports."
        )
        if ai_expander.button(
            toggle_label,
            key="toggle_ml_predictions_button",
            use_container_width=True,
            help=toggle_help,
        ):
            use_ml_predictions = not use_ml_predictions
            st.session_state['use_ml_predictions'] = use_ml_predictions

        if not use_ml_predictions:
            ai_expander.info(
                "ML predictions are disabled. Odds, sentiment, Kalshi, and live data signals still run as usual."
            )
        min_ai_confidence = ai_expander.slider(
            "Minimum AI Confidence",
            0.0,
            1.0,
            float(st.session_state.get('min_ai_confidence', 0.60) or 0.60),
            0.05,
        )
        min_parlay_probability = ai_expander.slider(
            "Minimum Parlay Probability",
            0.20,
            0.60,
            float(st.session_state.get('min_parlay_probability', 0.30) or 0.30),
            0.05,
        )
        max_parlay_probability = ai_expander.slider(
            "Maximum Parlay Probability",
            0.45,
            0.85,
            float(st.session_state.get('max_parlay_probability', 0.65) or 0.65),
            0.05,
        )

    st.session_state['use_sentiment'] = use_sentiment
    st.session_state['use_ml_predictions'] = use_ml_predictions
    st.session_state['min_ai_confidence'] = min_ai_confidence
    st.session_state['min_parlay_probability'] = min_parlay_probability
    st.session_state['max_parlay_probability'] = max_parlay_probability

    return {
        "tz": tz_obj,
        "timezone_name": tz_name,
        "selected_date": sel_date,
        "day_window": day_window,
        "sports": sports,
        "use_sentiment": use_sentiment,
        "use_ml_predictions": use_ml_predictions,
        "min_ai_confidence": min_ai_confidence,
        "min_parlay_probability": min_parlay_probability,
        "max_parlay_probability": max_parlay_probability,
    }


def resolve_nfl_apisports_key() -> Tuple[str, Optional[str]]:
    """Locate the NFL API-Sports key from Streamlit secrets or the environment."""

    secret_container = getattr(st, "secrets", None)
    if secret_container is not None:
        for secret_name in ("NFL_APISPORTS_API_KEY", "APISPORTS_API_KEY", "API_SPORTS_KEY"):
            try:
                secret_value = secret_container.get(secret_name)
            except Exception:
                secret_value = None
            if secret_value:
                return str(secret_value), f"secret:{secret_name}"

    for env_name in ("NFL_APISPORTS_API_KEY", "APISPORTS_API_KEY", "API_SPORTS_KEY"):
        env_value = os.environ.get(env_name)
        if env_value:
            return env_value, f"env:{env_name}"

    return "", None


def resolve_nhl_apisports_key() -> Tuple[str, Optional[str]]:
    """Locate the NHL API-Sports key from Streamlit secrets or the environment."""

    secret_container = getattr(st, "secrets", None)
    if secret_container is not None:
        for secret_name in ("NHL_APISPORTS_API_KEY", "APISPORTS_API_KEY", "API_SPORTS_KEY"):
            try:
                secret_value = secret_container.get(secret_name)
            except Exception:
                secret_value = None
            if secret_value:
                return str(secret_value), f"secret:{secret_name}"

    for env_name in ("NHL_APISPORTS_API_KEY", "APISPORTS_API_KEY", "API_SPORTS_KEY"):
        env_value = os.environ.get(env_name)
        if env_value:
            return env_value, f"env:{env_name}"

    return "", None


def resolve_nba_apisports_key() -> Tuple[str, Optional[str]]:
    """Locate the NBA API-Sports key from Streamlit secrets or the environment."""

    secret_container = getattr(st, "secrets", None)
    if secret_container is not None:
        for secret_name in ("NBA_APISPORTS_API_KEY", "APISPORTS_API_KEY", "API_SPORTS_KEY"):
            try:
                secret_value = secret_container.get(secret_name)
            except Exception:
                secret_value = None
            if secret_value:
                return str(secret_value), f"secret:{secret_name}"

    for env_name in ("NBA_APISPORTS_API_KEY", "APISPORTS_API_KEY", "API_SPORTS_KEY"):
        env_value = os.environ.get(env_name)
        if env_value:
            return env_value, f"env:{env_name}"

    return "", None


def resolve_sportsdata_key(sport_key: str) -> Tuple[str, Optional[str]]:
    """Locate the SportsData.io key for a specific sport."""

    config = SPORTSDATA_CONFIG.get(sport_key, {})
    secret_priority: List[str] = list(dict.fromkeys(config.get("secret_names", ())))
    for fallback_name in ("SPORTSDATA_API_KEY", "SPORTSDATA_KEY"):
        if fallback_name not in secret_priority:
            secret_priority.append(fallback_name)

    secret_container = getattr(st, "secrets", None)
    if secret_container is not None:
        for secret_name in secret_priority:
            try:
                secret_value = secret_container.get(secret_name)
            except Exception:
                secret_value = None
            if secret_value:
                return str(secret_value), f"secret:{secret_name}"

    for env_name in secret_priority:
        env_value = os.environ.get(env_name)
        if env_value:
            return env_value, f"env:{env_name}"

    return "", None


def ensure_sportsdata_clients() -> Dict[str, Any]:
    """Instantiate and synchronize SportsData.io clients for all configured sports."""

    clients: Dict[str, Any] = st.session_state.get('sportsdata_clients', {}) or {}

    for sport_key, cfg in SPORTSDATA_CONFIG.items():
        session_key = f"{sport_key}_sportsdata_api_key"
        source_session_key = f"{sport_key}_sportsdata_key_source"
        api_key = st.session_state.get(session_key, "")
        source = st.session_state.get(source_session_key)

        client = clients.get(sport_key)
        if client is None:
            client = cfg["client_class"](api_key or None, key_source=source)
            clients[sport_key] = client
        else:
            if getattr(client, "api_key", "") != (api_key or ""):
                client.update_api_key(api_key or None, source=source or "user")

    st.session_state['sportsdata_clients'] = clients
    return clients

# Comprehensive mapping of Kalshi team abbreviations → canonical team names.
# The Kalshi markets often reference tickers like "NBA.LAL_GSW" or subtitles using
# short-hands. By centralizing these variations we can translate between
# sportsbook-style names ("Los Angeles Lakers") and Kalshi identifiers ("LAL").
# This dramatically increases the likelihood that we locate the correct Kalshi
# market when validating a parlay leg.
KALSHI_TEAM_ABBREVIATIONS: Dict[str, List[str]] = {
    # ========================= NFL =========================
    "ARIZONA CARDINALS": ["ARI", "ARZ", "AZ"],
    "ATLANTA FALCONS": ["ATL"],
    "BALTIMORE RAVENS": ["BAL"],
    "BUFFALO BILLS": ["BUF"],
    "CAROLINA PANTHERS": ["CAR", "CLT"],
    "CHICAGO BEARS": ["CHI", "CHB"],
    "CINCINNATI BENGALS": ["CIN", "CINC"],
    "CLEVELAND BROWNS": ["CLE"],
    "DALLAS COWBOYS": ["DAL"],
    "DENVER BRONCOS": ["DEN"],
    "DETROIT LIONS": ["DET"],
    "GREEN BAY PACKERS": ["GB", "GBP", "GBE"],
    "HOUSTON TEXANS": ["HOU", "HTX"],
    "INDIANAPOLIS COLTS": ["IND"],
    "JACKSONVILLE JAGUARS": ["JAX", "JAC"],
    "KANSAS CITY CHIEFS": ["KC", "KCC"],
    "LAS VEGAS RAIDERS": ["LV", "LVR"],
    "LOS ANGELES CHARGERS": ["LAC", "LA CHARGERS"],
    "LOS ANGELES RAMS": ["LAR", "LA RAMS"],
    "MIAMI DOLPHINS": ["MIA"],
    "MINNESOTA VIKINGS": ["MIN", "MINN"],
    "NEW ENGLAND PATRIOTS": ["NE", "NEP"],
    "NEW ORLEANS SAINTS": ["NO", "NOS"],
    "NEW YORK GIANTS": ["NYG", "NY GIANTS"],
    "NEW YORK JETS": ["NYJ", "NY JETS"],
    "PHILADELPHIA EAGLES": ["PHI", "PHL", "PHI EAGLES"],
    "PITTSBURGH STEELERS": ["PIT", "PITTSBURGH"],
    "SAN FRANCISCO 49ERS": ["SF", "SFO", "SF 49ERS"],
    "SEATTLE SEAHAWKS": ["SEA", "SEA HAWKS"],
    "TAMPA BAY BUCCANEERS": ["TB", "TBB"],
    "TENNESSEE TITANS": ["TEN", "TENN"],
    "WASHINGTON COMMANDERS": ["WAS", "WSH"],

    # ========================= NBA =========================
    "ATLANTA HAWKS": ["ATL"],
    "BOSTON CELTICS": ["BOS"],
    "BROOKLYN NETS": ["BKN", "BRK"],
    "CHARLOTTE HORNETS": ["CHA", "CHH", "CLT"],
    "CHICAGO BULLS": ["CHI"],
    "CLEVELAND CAVALIERS": ["CLE", "CAVS"],
    "DALLAS MAVERICKS": ["DAL", "MAVS"],
    "DENVER NUGGETS": ["DEN"],
    "DETROIT PISTONS": ["DET"],
    "GOLDEN STATE WARRIORS": ["GSW", "GS"],
    "HOUSTON ROCKETS": ["HOU"],
    "INDIANA PACERS": ["IND"],
    "LOS ANGELES CLIPPERS": ["LAC", "LA CLIPPERS"],
    "LOS ANGELES LAKERS": ["LAL", "LA LAKERS"],
    "MEMPHIS GRIZZLIES": ["MEM"],
    "MIAMI HEAT": ["MIA"],
    "MILWAUKEE BUCKS": ["MIL"],
    "MINNESOTA TIMBERWOLVES": ["MIN", "MINN"],
    "NEW ORLEANS PELICANS": ["NOP", "NO PELICANS"],
    "NEW YORK KNICKS": ["NYK", "NY KNICKS"],
    "OKLAHOMA CITY THUNDER": ["OKC"],
    "ORLANDO MAGIC": ["ORL"],
    "PHILADELPHIA 76ERS": ["PHI", "PHL", "SIXERS"],
    "PHOENIX SUNS": ["PHX"],
    "PORTLAND TRAIL BLAZERS": ["POR", "PTB", "PDX"],
    "SACRAMENTO KINGS": ["SAC"],
    "SAN ANTONIO SPURS": ["SAS", "SA SPURS"],
    "TORONTO RAPTORS": ["TOR"],
    "UTAH JAZZ": ["UTA"],
    "WASHINGTON WIZARDS": ["WAS", "WSH"],

    # ========================= MLB =========================
    "ARIZONA DIAMONDBACKS": ["ARI", "ARZ", "AZ"],
    "ATLANTA BRAVES": ["ATL"],
    "BALTIMORE ORIOLES": ["BAL"],
    "BOSTON RED SOX": ["BOS"],
    "CHICAGO CUBS": ["CHC"],
    "CHICAGO WHITE SOX": ["CWS", "CHW"],
    "CINCINNATI REDS": ["CIN", "CINC"],
    "CLEVELAND GUARDIANS": ["CLE", "CLV"],
    "COLORADO ROCKIES": ["COL"],
    "DETROIT TIGERS": ["DET"],
    "HOUSTON ASTROS": ["HOU"],
    "KANSAS CITY ROYALS": ["KC", "KCR"],
    "LOS ANGELES ANGELS": ["LAA", "LA ANGELS"],
    "LOS ANGELES DODGERS": ["LAD", "LA DODGERS"],
    "MIAMI MARLINS": ["MIA"],
    "MILWAUKEE BREWERS": ["MIL"],
    "MINNESOTA TWINS": ["MIN", "MINN"],
    "NEW YORK METS": ["NYM", "NY METS"],
    "NEW YORK YANKEES": ["NYY", "NY YANKEES"],
    "OAKLAND ATHLETICS": ["OAK"],
    "PHILADELPHIA PHILLIES": ["PHI", "PHL", "PHILS"],
    "PITTSBURGH PIRATES": ["PIT"],
    "SAN DIEGO PADRES": ["SD", "SDP"],
    "SAN FRANCISCO GIANTS": ["SF", "SFG"],
    "SEATTLE MARINERS": ["SEA"],
    "ST. LOUIS CARDINALS": ["STL", "SLC"],
    "TAMPA BAY RAYS": ["TB", "TBR"],
    "TEXAS RANGERS": ["TEX"],
    "TORONTO BLUE JAYS": ["TOR"],
    "WASHINGTON NATIONALS": ["WSH", "WAS"],

    # ========================= NHL =========================
    "ANAHEIM DUCKS": ["ANA"],
    "ARIZONA COYOTES": ["ARI", "ARZ", "AZ"],
    "BOSTON BRUINS": ["BOS"],
    "BUFFALO SABRES": ["BUF"],
    "CALGARY FLAMES": ["CGY"],
    "CAROLINA HURRICANES": ["CAR", "CLT"],
    "CHICAGO BLACKHAWKS": ["CHI"],
    "COLORADO AVALANCHE": ["COL"],
    "COLUMBUS BLUE JACKETS": ["CBJ"],
    "DALLAS STARS": ["DAL"],
    "DETROIT RED WINGS": ["DET"],
    "EDMONTON OILERS": ["EDM"],
    "FLORIDA PANTHERS": ["FLA"],
    "LOS ANGELES KINGS": ["LAK", "LA KINGS"],
    "MINNESOTA WILD": ["MIN", "MINN"],
    "MONTREAL CANADIENS": ["MTL"],
    "NASHVILLE PREDATORS": ["NSH"],
    "NEW JERSEY DEVILS": ["NJD"],
    "NEW YORK ISLANDERS": ["NYI"],
    "NEW YORK RANGERS": ["NYR"],
    "OTTAWA SENATORS": ["OTT"],
    "PHILADELPHIA FLYERS": ["PHI", "PHL"],
    "PITTSBURGH PENGUINS": ["PIT"],
    "SAN JOSE SHARKS": ["SJS"],
    "SEATTLE KRAKEN": ["SEA"],
    "ST. LOUIS BLUES": ["STL"],
    "TAMPA BAY LIGHTNING": ["TB", "TBL"],
    "TORONTO MAPLE LEAFS": ["TOR"],
    "VANCOUVER CANUCKS": ["VAN"],
    "VEGAS GOLDEN KNIGHTS": ["VGK", "VEGAS"],
    "WASHINGTON CAPITALS": ["WSH", "WAS"],
    "WINNIPEG JETS": ["WPG"],
}

# Map teams back to their primary league so we can build league-aware fallbacks
KALSHI_LEAGUE_TEAM_SETS: Dict[str, List[str]] = {
    "NFL": [
        "ARIZONA CARDINALS", "ATLANTA FALCONS", "BALTIMORE RAVENS", "BUFFALO BILLS",
        "CAROLINA PANTHERS", "CHICAGO BEARS", "CINCINNATI BENGALS", "CLEVELAND BROWNS",
        "DALLAS COWBOYS", "DENVER BRONCOS", "DETROIT LIONS", "GREEN BAY PACKERS",
        "HOUSTON TEXANS", "INDIANAPOLIS COLTS", "JACKSONVILLE JAGUARS", "KANSAS CITY CHIEFS",
        "LAS VEGAS RAIDERS", "LOS ANGELES CHARGERS", "LOS ANGELES RAMS", "MIAMI DOLPHINS",
        "MINNESOTA VIKINGS", "NEW ENGLAND PATRIOTS", "NEW ORLEANS SAINTS", "NEW YORK GIANTS",
        "NEW YORK JETS", "PHILADELPHIA EAGLES", "PITTSBURGH STEELERS", "SAN FRANCISCO 49ERS",
        "SEATTLE SEAHAWKS", "TAMPA BAY BUCCANEERS", "TENNESSEE TITANS", "WASHINGTON COMMANDERS"
    ],
    "NBA": [
        "ATLANTA HAWKS", "BOSTON CELTICS", "BROOKLYN NETS", "CHARLOTTE HORNETS",
        "CHICAGO BULLS", "CLEVELAND CAVALIERS", "DALLAS MAVERICKS", "DENVER NUGGETS",
        "DETROIT PISTONS", "GOLDEN STATE WARRIORS", "HOUSTON ROCKETS", "INDIANA PACERS",
        "LOS ANGELES CLIPPERS", "LOS ANGELES LAKERS", "MEMPHIS GRIZZLIES", "MIAMI HEAT",
        "MILWAUKEE BUCKS", "MINNESOTA TIMBERWOLVES", "NEW ORLEANS PELICANS", "NEW YORK KNICKS",
        "OKLAHOMA CITY THUNDER", "ORLANDO MAGIC", "PHILADELPHIA 76ERS", "PHOENIX SUNS",
        "PORTLAND TRAIL BLAZERS", "SACRAMENTO KINGS", "SAN ANTONIO SPURS", "TORONTO RAPTORS",
        "UTAH JAZZ", "WASHINGTON WIZARDS"
    ],
    "MLB": [
        "ARIZONA DIAMONDBACKS", "ATLANTA BRAVES", "BALTIMORE ORIOLES", "BOSTON RED SOX",
        "CHICAGO CUBS", "CHICAGO WHITE SOX", "CINCINNATI REDS", "CLEVELAND GUARDIANS",
        "COLORADO ROCKIES", "DETROIT TIGERS", "HOUSTON ASTROS", "KANSAS CITY ROYALS",
        "LOS ANGELES ANGELS", "LOS ANGELES DODGERS", "MIAMI MARLINS", "MILWAUKEE BREWERS",
        "MINNESOTA TWINS", "NEW YORK METS", "NEW YORK YANKEES", "OAKLAND ATHLETICS",
        "PHILADELPHIA PHILLIES", "PITTSBURGH PIRATES", "SAN DIEGO PADRES", "SAN FRANCISCO GIANTS",
        "SEATTLE MARINERS", "ST. LOUIS CARDINALS", "TAMPA BAY RAYS", "TEXAS RANGERS",
        "TORONTO BLUE JAYS", "WASHINGTON NATIONALS"
    ],
    "NHL": [
        "ANAHEIM DUCKS", "ARIZONA COYOTES", "BOSTON BRUINS", "BUFFALO SABRES",
        "CALGARY FLAMES", "CAROLINA HURRICANES", "CHICAGO BLACKHAWKS", "COLORADO AVALANCHE",
        "COLUMBUS BLUE JACKETS", "DALLAS STARS", "DETROIT RED WINGS", "EDMONTON OILERS",
        "FLORIDA PANTHERS", "LOS ANGELES KINGS", "MINNESOTA WILD", "MONTREAL CANADIENS",
        "NASHVILLE PREDATORS", "NEW JERSEY DEVILS", "NEW YORK ISLANDERS", "NEW YORK RANGERS",
        "OTTAWA SENATORS", "PHILADELPHIA FLYERS", "PITTSBURGH PENGUINS", "SAN JOSE SHARKS",
        "SEATTLE KRAKEN", "ST. LOUIS BLUES", "TAMPA BAY LIGHTNING", "TORONTO MAPLE LEAFS",
        "VANCOUVER CANUCKS", "VEGAS GOLDEN KNIGHTS", "WASHINGTON CAPITALS", "WINNIPEG JETS"
    ],
}

KALSHI_TEAM_LEAGUE_MAP: Dict[str, str] = {
    team: league
    for league, teams in KALSHI_LEAGUE_TEAM_SETS.items()
    for team in teams
}

SPORT_KEY_TO_LEAGUE: Dict[str, str] = {
    "americanfootball_nfl": "NFL",
    "americanfootball_ncaaf": "NCAAF",
    "basketball_nba": "NBA",
    "basketball_ncaab": "NCAAB",
    "baseball_mlb": "MLB",
    "icehockey_nhl": "NHL",
    "mma_mixed_martial_arts": "MMA",
    "soccer_epl": "EPL",
    "soccer_uefa_champs_league": "UEFA",
    "tennis_atp_singles": "TENNIS",
}


def format_sport_label(sport_key: Any) -> str:
    """Return a user-friendly league label for an Odds API sport key."""

    if not isinstance(sport_key, str):
        return str(sport_key)

    if sport_key in SPORT_KEY_TO_LEAGUE:
        return SPORT_KEY_TO_LEAGUE[sport_key]

    if "_" in sport_key:
        return sport_key.split("_")[-1].upper()

    return sport_key.upper()

# ============ REAL SENTIMENT ANALYSIS ENGINE ============
# (moved to app_core.sentiment so it can be reused without importing the
# Streamlit UI. RealSentimentAnalyzer and SentimentAnalyzer are imported above.)

# ============ AI PARLAY OPTIMIZER ============
class AIOptimizer:
    """Optimizes parlay selection using AI insights"""

    def __init__(
        self,
        sentiment_analyzer: SentimentAnalyzer,
        ml_predictor: Optional[MLPredictor],
    ):
        self.sentiment = sentiment_analyzer
        self.ml = ml_predictor
    
    def score_parlay(self, legs: List[Dict]) -> Dict[str, float]:
        """
        Score a parlay combination using AI
        Higher scores = better opportunity
        NOW INCLUDES KALSHI VALIDATION
        """
        def _empty_score() -> Dict[str, Any]:
            return {
                'score': 0.0,
                'ai_ev': 0.0,
                'confidence': 0.0,
                'edge': 0.0,
                'correlation_factor': 1.0,
                'kalshi_factor': 1.0,
                'kalshi_legs': 0,
                'kalshi_boost': 0.0,
                'kalshi_alignment_avg': 0.0,
                'kalshi_alignment_abs_avg': 0.0,
                'kalshi_alignment_positive': 0,
                'kalshi_alignment_negative': 0,
                'kalshi_alignment_count': 0,
                'live_data_factor': 1.0,
                'live_data_boost': 0.0,
                'live_data_legs': 0,
                'live_data_sports': [],
                'apisports_factor': 1.0,
                'apisports_legs': 0,
                'apisports_boost': 0.0,
                'apisports_sports': [],
                'sportsdata_factor': 1.0,
                'sportsdata_legs': 0,
                'sportsdata_boost': 0.0,
                'sportsdata_sports': [],
            }

        if not legs:
            return _empty_score()

        valid_legs: List[Dict[str, Any]] = []
        for leg in legs:
            if not isinstance(leg, dict):
                logger.debug("Skipping non-dict parlay leg: %r", leg)
                continue
            decimal_odds = _safe_float(leg.get('d'))
            base_prob = _safe_float(leg.get('p'))
            if decimal_odds is None or decimal_odds <= 1 or base_prob is None:
                logger.debug(
                    "Skipping parlay leg missing odds/probability: %s",
                    leg.get('label', leg.get('team', 'unknown')),
                )
                continue
            valid_legs.append(leg)

        if not valid_legs:
            return _empty_score()

        legs = valid_legs

        # Calculate combined probability (AI-adjusted)
        combined_prob = 1.0
        combined_confidence = 1.0
        total_edge = 0
        kalshi_boost = 0
        kalshi_legs = 0
        kalshi_alignment_total = 0.0
        kalshi_alignment_abs_total = 0.0
        kalshi_alignment_positive = 0
        kalshi_alignment_negative = 0
        kalshi_alignment_count = 0
        apisports_boost = 0
        apisports_legs = 0
        apisports_sports: set[str] = set()
        sportsdata_boost = 0
        sportsdata_legs = 0
        sportsdata_sports: set[str] = set()
        live_data_legs = 0
        live_data_sports: set[str] = set()

        for leg in legs:
            ai_prob_val = _safe_float(leg.get('ai_prob'))
            if ai_prob_val is None:
                ai_prob_val = _safe_float(leg.get('p'))
            if ai_prob_val is None:
                logger.debug(
                    "Skipping AI probability blend for leg with missing prob: %s",
                    leg.get('label', leg.get('team', 'unknown')),
                )
                continue
            combined_prob *= ai_prob_val

            confidence_val = _safe_float(leg.get('ai_confidence'))
            if confidence_val is None:
                confidence_val = 0.5
            combined_confidence *= max(0.01, min(1.0, confidence_val))

            total_edge += _safe_float(leg.get('ai_edge')) or 0.0

            has_live_context = False

            # KALSHI INTEGRATION: Add Kalshi influence
            if 'kalshi_validation' in leg:
                kv = leg['kalshi_validation']
                if kv.get('kalshi_available'):
                    kalshi_legs += 1
                    # Kalshi provides additional probability estimate
                    kalshi_prob = kv.get('kalshi_prob', 0)
                    sportsbook_prob = leg.get('p', 0)
                    data_source = kv.get('data_source', 'kalshi')
                    source_weight = 0.6 if data_source and 'synthetic' in data_source else 1.0

                    # Track Kalshi vs. model alignment before blending probabilities.
                    ai_pre_kalshi = leg.get('ai_prob_before_kalshi')
                    if ai_pre_kalshi is None:
                        delta_hint = leg.get('kalshi_alignment_delta')
                        if isinstance(delta_hint, (int, float)):
                            ai_pre_kalshi = leg.get('ai_prob', sportsbook_prob) - delta_hint
                    if ai_pre_kalshi is None:
                        ai_pre_kalshi = leg.get('ai_prob', sportsbook_prob)
                    alignment_delta = kalshi_prob - ai_pre_kalshi
                    kalshi_alignment_total += alignment_delta
                    kalshi_alignment_abs_total += abs(alignment_delta)
                    kalshi_alignment_count += 1
                    if alignment_delta >= 0.01:
                        kalshi_alignment_positive += 1
                    elif alignment_delta <= -0.01:
                        kalshi_alignment_negative += 1

                    # If Kalshi and AI both disagree with sportsbook in same direction
                    # that's a strong signal
                    ai_prob = leg.get('ai_prob', sportsbook_prob)

                    if kalshi_prob > sportsbook_prob and ai_prob > sportsbook_prob:
                        # Both Kalshi and AI see value
                        kalshi_boost += 15 * source_weight  # Strong boost
                    elif kalshi_prob < sportsbook_prob and ai_prob < sportsbook_prob:
                        # Both Kalshi and AI skeptical
                        kalshi_boost -= 10 * source_weight  # Penalty
                    elif abs(kalshi_prob - ai_prob) < 0.05:
                        # Kalshi and AI agree (regardless of sportsbook)
                        kalshi_boost += 10 * source_weight  # Agreement boost
                    elif abs(kalshi_prob - sportsbook_prob) < 0.03:
                        # Kalshi confirms market
                        kalshi_boost += 5 * source_weight  # Small boost for confirmation
                    else:
                        # Kalshi contradicts both AI and market
                        kalshi_boost -= 5 * source_weight  # Small penalty for confusion

            apisports_info = leg.get('apisports')
            if apisports_info:
                has_live_context = True
                apisports_legs += 1
                sport_key = apisports_info.get('sport_key') if isinstance(apisports_info, dict) else None
                if sport_key:
                    apisports_sports.add(sport_key)
                    live_data_sports.add(sport_key)
                trend = apisports_info.get('trend')
                if trend == 'hot':
                    apisports_boost += 5
                elif trend == 'cold':
                    apisports_boost -= 5

            sportsdata_info = leg.get('sportsdata')
            if sportsdata_info:
                has_live_context = True
                sportsdata_legs += 1
                sport_key = sportsdata_info.get('sport_key') if isinstance(sportsdata_info, dict) else None
                if sport_key:
                    sportsdata_sports.add(sport_key)
                    live_data_sports.add(sport_key)
                trend = sportsdata_info.get('trend')
                if trend == 'hot':
                    sportsdata_boost += 4
                elif trend == 'cold':
                    sportsdata_boost -= 4
                strength_delta = sportsdata_info.get('strength_delta')
                if isinstance(strength_delta, (int, float)):
                    if strength_delta >= 2.0:
                        sportsdata_boost += 6
                    elif strength_delta <= -2.0:
                        sportsdata_boost -= 6
                turnover_margin = sportsdata_info.get('turnover_margin')
                if isinstance(turnover_margin, (int, float)):
                    if turnover_margin >= 0.5:
                        sportsdata_boost += 2
                    elif turnover_margin <= -0.5:
                        sportsdata_boost -= 2

            if has_live_context:
                live_data_legs += 1

        # Calculate combined decimal odds
        combined_odds = 1.0
        for leg in legs:
            decimal_val = _safe_float(leg.get('d'))
            if decimal_val is None or decimal_val <= 1:
                logger.debug(
                    "Skipping leg with invalid decimal odds in multiplier: %s",
                    leg.get('label', leg.get('team', 'unknown')),
                )
                continue
            combined_odds *= decimal_val

        # AI-enhanced EV
        ai_ev = (combined_prob * combined_odds) - 1.0

        # Correlation penalty (same-game parlays are correlated)
        unique_event_ids = [leg.get('event_id') for leg in legs if leg.get('event_id')]
        unique_games = len(set(unique_event_ids)) if unique_event_ids else len(legs)
        correlation_factor = unique_games / len(legs)
        
        # Kalshi validation factor (0.8 to 1.2 multiplier)
        if kalshi_legs > 0:
            kalshi_factor = 1.0 + (kalshi_boost / 100)
            kalshi_factor = max(0.8, min(1.2, kalshi_factor))  # Clamp between 0.8 and 1.2
        else:
            kalshi_factor = 1.0  # No Kalshi data = neutral
        
        # Final score components with KALSHI integration
        ev_score = ai_ev * 100  # EV contribution
        confidence_score = combined_confidence * 50  # Confidence contribution
        edge_score = total_edge * 150  # Edge is most important
        
        # Calculate base score
        base_score = (edge_score * 0.45 +      # 45% edge
                     ev_score * 0.30 +          # 30% EV
                     confidence_score * 0.25)    # 25% confidence

        # Apply Kalshi factor, correlation factor, and live data adjustments
        apisports_factor = 1.0
        if apisports_legs:
            apisports_factor += apisports_boost / 100.0
            apisports_factor = max(0.9, min(1.1, apisports_factor))

        sportsdata_factor = 1.0
        if sportsdata_legs:
            sportsdata_factor += sportsdata_boost / 100.0
            sportsdata_factor = max(0.9, min(1.1, sportsdata_factor))

        combined_boost = apisports_boost + sportsdata_boost
        live_data_factor = 1.0
        if live_data_legs and combined_boost:
            live_data_factor += combined_boost / 100.0
            live_data_factor = max(0.85, min(1.15, live_data_factor))

        final_score = base_score * correlation_factor * kalshi_factor * live_data_factor

        return {
            'score': final_score,
            'ai_ev': ai_ev,
            'confidence': combined_confidence,
            'edge': total_edge,
            'correlation_factor': correlation_factor,
            'kalshi_factor': kalshi_factor,
            'kalshi_legs': kalshi_legs,
            'kalshi_boost': kalshi_boost,
            'kalshi_alignment_avg': (kalshi_alignment_total / kalshi_alignment_count)
            if kalshi_alignment_count
            else 0.0,
            'kalshi_alignment_abs_avg': (kalshi_alignment_abs_total / kalshi_alignment_count)
            if kalshi_alignment_count
            else 0.0,
            'kalshi_alignment_positive': kalshi_alignment_positive,
            'kalshi_alignment_negative': kalshi_alignment_negative,
            'kalshi_alignment_count': kalshi_alignment_count,
            'live_data_factor': live_data_factor,
            'live_data_boost': combined_boost,
            'live_data_legs': live_data_legs,
            'live_data_sports': sorted(live_data_sports),
            'apisports_factor': apisports_factor,
            'apisports_legs': apisports_legs,
            'apisports_boost': apisports_boost,
            'apisports_sports': sorted(apisports_sports),
            'sportsdata_factor': sportsdata_factor,
            'sportsdata_legs': sportsdata_legs,
            'sportsdata_boost': sportsdata_boost,
            'sportsdata_sports': sorted(sportsdata_sports),
        }


class KalshiIntegrator:
    """Integrates Kalshi prediction market odds and analysis"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or os.environ.get("KALSHI_API_KEY")
        self.api_secret = api_secret or os.environ.get("KALSHI_API_SECRET")
        self.base_url = "https://api.elections.kalshi.com/trade-api/v2"
        self.demo_url = "https://demo-api.elections.kalshi.com/trade-api/v2"

        # Use demo for testing, production for live
        self.api_url = self.base_url if self.api_key else self.demo_url

        self.headers = {
            "Content-Type": "application/json"
        }

        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

        # Synthetic fallback cache when Kalshi API is unavailable (e.g., network blocks)
        self._using_synthetic_data = False
        self._synthetic_markets: List[Dict[str, Any]] = []
        self._synthetic_orderbooks: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self._synthetic_market_by_team: Dict[str, Dict[str, Any]] = {}
        self.last_error: Optional[str] = None

    # -------------------- Synthetic helpers --------------------
    def _synthetic_probability(self, team: str, sport_key: Optional[str] = None,
                               sportsbook_prob: Optional[float] = None) -> float:
        """Generate a deterministic synthetic probability for a team."""

        team_upper = team.upper()
        base_prob = sportsbook_prob if sportsbook_prob is not None else 0.52

        # Deterministic offset based on team characters (stable pseudo-random)
        ordinal_sum = sum(ord(c) for c in team_upper if c.isalpha())
        offset = ((ordinal_sum % 21) - 10) / 200.0  # -0.05 .. +0.05

        league = KALSHI_TEAM_LEAGUE_MAP.get(team_upper)
        if sport_key and sport_key in SPORT_KEY_TO_LEAGUE and not league:
            league = SPORT_KEY_TO_LEAGUE[sport_key]

        league_bias = {
            "NFL": 0.015,
            "NBA": 0.010,
            "MLB": 0.005,
            "NHL": 0.005,
        }.get(league, 0.0)

        synthetic = base_prob + offset + league_bias
        return max(0.05, min(0.95, synthetic))

    def _synthetic_ticker_for_team(self, team: str, league: Optional[str]) -> str:
        team_key = re.sub(r"[^A-Z0-9]", "", team.upper())
        league_key = league or "SPORTS"
        return f"SIM.{league_key}.{team_key[:8]}"

    def _ensure_synthetic_data(self) -> None:
        if self._synthetic_markets:
            return

        # Build synthetic markets for every team we know about so UI has coverage
        now = datetime.utcnow()
        expiry = (now + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")

        for team, abbrs in KALSHI_TEAM_ABBREVIATIONS.items():
            league = KALSHI_TEAM_LEAGUE_MAP.get(team)
            ticker = self._synthetic_ticker_for_team(team, league)
            prob = self._synthetic_probability(team)
            price = int(round(prob * 100))

            market = {
                "ticker": ticker,
                "title": f"{team.title()} confidence (synthetic Kalshi)",
                "subtitle": "Synthetic fallback market generated locally",
                "series_ticker": "SPORTS",
                "status": "open",
                "close_time": expiry,
                "league": league,
                "synthetic": True,
                "team": team,
                "abbreviation": abbrs[0] if abbrs else None,
            }

            self._synthetic_markets.append(market)
            self._synthetic_orderbooks[ticker] = {
                "yes": [{"price": price, "contracts": 100}],
                "no": [{"price": 100 - price, "contracts": 100}],
            }
            self._synthetic_market_by_team[team] = market

    def using_synthetic_data(self) -> bool:
        return self._using_synthetic_data

    def get_synthetic_market_for_team(self, team: str) -> Optional[Dict[str, Any]]:
        self._ensure_synthetic_data()
        return self._synthetic_market_by_team.get(team.upper())

    def synthetic_probability(self, team: str, sport_key: Optional[str] = None,
                               sportsbook_prob: Optional[float] = None) -> float:
        """Public helper to compute synthetic probabilities for validation."""
        self._ensure_synthetic_data()
        return self._synthetic_probability(team, sport_key, sportsbook_prob)
    
    def get_markets(self, category: str = "sports", status: str = "open") -> List[Dict]:
        """
        Fetch available Kalshi markets

        Args:
            category: 'sports', 'politics', 'economics', etc.
            status: 'open', 'closed', 'settled'
        
        Returns:
            List of market dictionaries
        """
        if self._using_synthetic_data:
            self._ensure_synthetic_data()
            return copy.deepcopy(self._synthetic_markets)

        try:
            endpoint = f"{self.api_url}/markets"
            params = {
                "limit": 100,
                "status": status
            }

            if category:
                params["series_ticker"] = category.upper()

            response = requests.get(endpoint, headers=self.headers, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                markets = data.get("markets", [])
                if markets:
                    self.last_error = None
                    return markets
                else:
                    self.last_error = "Kalshi API returned no markets"
            else:
                self.last_error = f"Kalshi API responded with status {response.status_code}"

        except Exception as e:
            self.last_error = str(e)
            st.warning(f"Error fetching Kalshi markets: {str(e)}")

        # Fallback to synthetic data when API fails or returns nothing
        self._using_synthetic_data = True
        self._ensure_synthetic_data()
        return copy.deepcopy(self._synthetic_markets)
    
    def get_sports_markets(self) -> List[Dict]:
        """Get all active sports betting markets"""
        all_markets = self.get_markets()
        
        # Filter for sports-related markets
        sports_keywords = ['NFL', 'NBA', 'MLB', 'NHL', 'UFC', 'SOCCER', 'TENNIS', 
                          'GOLF', 'FOOTBALL', 'BASKETBALL', 'BASEBALL', 'HOCKEY']
        
        sports_markets = []
        for market in all_markets:
            title = market.get('title', '').upper()
            ticker = market.get('ticker', '').upper()
            
            if any(keyword in title or keyword in ticker for keyword in sports_keywords):
                sports_markets.append(market)
        
        return sports_markets
    
    def get_market_details(self, market_ticker: str) -> Dict:
        """Get detailed information about a specific market"""
        if self._using_synthetic_data:
            self._ensure_synthetic_data()
            market = next((m for m in self._synthetic_markets if m.get('ticker') == market_ticker), None)
            return copy.deepcopy(market) if market else {}

        try:
            endpoint = f"{self.api_url}/markets/{market_ticker}"
            response = requests.get(endpoint, headers=self.headers, timeout=10)

            if response.status_code == 200:
                return response.json().get("market", {})
            else:
                return {}

        except Exception as e:
            st.warning(f"Error fetching market details: {str(e)}")
            return {}

    def get_orderbook(self, market_ticker: str) -> Dict:
        """Get current orderbook (bids/asks) for a market"""
        if self._using_synthetic_data:
            self._ensure_synthetic_data()
            orderbook = self._synthetic_orderbooks.get(market_ticker)
            return copy.deepcopy(orderbook) if orderbook else {}

        try:
            endpoint = f"{self.api_url}/markets/{market_ticker}/orderbook"
            response = requests.get(endpoint, headers=self.headers, timeout=10)

            if response.status_code == 200:
                return response.json().get("orderbook", {})
            else:
                return {}

        except Exception as e:
            st.warning(f"Error fetching orderbook: {str(e)}")
            return {}
    
    def compare_with_sportsbook(self, kalshi_market: Dict, sportsbook_odds: Dict) -> Dict:
        """
        Compare Kalshi prediction market with traditional sportsbook
        
        This finds arbitrage and value opportunities
        """
        kalshi_yes_price = kalshi_market.get('yes_bid', 0) / 100  # Convert cents to probability
        kalshi_no_price = kalshi_market.get('no_bid', 0) / 100
        
        # Get implied probability from sportsbook
        sb_odds = sportsbook_odds.get('price')
        if sb_odds:
            sb_prob = implied_p_from_american(sb_odds)
        else:
            if away_score > home_score:
                result['status'] = 'win'
            elif away_score < home_score:
                result['status'] = 'loss'
            else:
                result['status'] = 'push'
        return result

# 7. SOCIAL MEDIA ANALYZER (Simplified)
class SocialMediaAnalyzer:
    """Analyzes Twitter/Reddit for real-time sentiment"""
    
    def __init__(self, twitter_api_key: str = None):
        self.twitter_key = twitter_api_key or os.environ.get("TWITTER_API_KEY")
        self.keywords = {
            'positive': ['win', 'great', 'amazing', 'clutch', 'dominant', 'beast'],
            'negative': ['lose', 'bad', 'terrible', 'bench', 'cut', 'injured']
        }
    
    def analyze_social_sentiment(self, team: str, player: str = None) -> Dict:
        """
        Analyze social media sentiment
        
        Note: This is a simplified placeholder. Full implementation requires:
        - Twitter API v2 access
        - Reddit API (PRAW) integration
        - Real-time data scraping
        """
        # Placeholder return
        return {
            'sentiment_score': 0.0,
            'tweet_volume': 0,
            'trending': False,
            'confidence': 0.2,
            'recommendation': '⚪ Social media analysis unavailable (API key needed)'
        }
    
    def detect_breaking_news(self, team: str) -> Dict:
        """Detect breaking news that might not be priced in yet"""
        # Placeholder - would monitor Twitter for breaking news keywords
        return {
            'has_breaking_news': False,
            'news_type': None,
            'urgency': 'low'
        }

# ============ KALSHI VALIDATION HELPER ============
def validate_with_kalshi(kalshi_integrator, home_team: str, away_team: str,
                        side: str, sportsbook_prob: float, sport: str) -> Dict:
    """
    IMPROVED: Validate sportsbook odds with Kalshi prediction market
    
    Now handles team name variations like:
    - "Memphis Grizzlies" matches "Memphis"
    - "New York Knicks" matches "New York K"  
    - "Los Angeles Lakers" matches "LA Lakers"
    
    Returns:
        'kalshi_prob': Kalshi market probability
        'discrepancy': Difference between markets
        'validation': 'confirms', 'contradicts', or 'unavailable'
        'edge': Additional edge from Kalshi vs sportsbook
        'confidence_boost': How much to boost confidence (0-0.20)
    """
    
    def normalize_team_name(team: str) -> List[str]:
        """Generate multiple variations of a team name for flexible matching"""
        team_upper = team.upper()
        variations = [team_upper, team_upper.replace(" ", "")]

        # Split into parts and add individual words
        parts = team_upper.split()
        for part in parts:
            if len(part) > 2:  # Skip very short words
                variations.append(part)

        # Special handling for common abbreviations
        abbreviations = {
            'NEW YORK': ['NY', 'NEW YORK K', 'N.Y.'],
            'LOS ANGELES': ['LA', 'L.A.'],
            'SAN FRANCISCO': ['SF', 'S.F.'],
            'GOLDEN STATE': ['GS'],
            'OKLAHOMA CITY': ['OKC'],
            'WASHINGTON': ['WSH'],
        }

        for city, abbrevs in abbreviations.items():
            if team_upper.startswith(city):
                variations.extend(abbrevs)

        # Kalshi-specific abbreviation support
        for canonical, abbrs in KALSHI_TEAM_ABBREVIATIONS.items():
            canonical_upper = canonical.upper()
            canonical_words = canonical_upper.split()

            # Direct matches (exact, contains, or shared keywords)
            if (
                team_upper == canonical_upper
                or canonical_upper in team_upper
                or team_upper in canonical_upper
                or any(word in canonical_words for word in parts if len(word) > 2)
            ):
                variations.extend(abbrs)

        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for variation in variations:
            if variation not in seen:
                seen.add(variation)
                unique_variations.append(variation)

        return unique_variations
    
    def teams_match(bet_team: str, market_text: str) -> bool:
        """Check if a bet team matches text in a market without short false-positives."""
        bet_variations = normalize_team_name(bet_team)
        market_upper = re.sub(r"[^A-Z0-9 ]", " ", market_text.upper().replace('_', ' '))
        market_compact = market_upper.replace(' ', '')
        market_tokens = set(re.findall(r"[A-Z0-9]+", market_upper))

        for variation in bet_variations:
            variation_upper = variation.upper()
            variation_clean = re.sub(r"[^A-Z0-9 ]", " ", variation_upper).strip()
            variation_compact = variation_clean.replace(' ', '')

            if not variation_compact:
                continue

            # Longer variations (team names, extended abbreviations) can match anywhere in the text
            if len(variation_compact) >= 4 and variation_compact in market_compact:
                return True

            # Compare token-by-token to avoid matching "LA" with "ATLANTA"
            variation_tokens = re.findall(r"[A-Z0-9]+", variation_clean)
            if variation_tokens and all(token in market_tokens for token in variation_tokens):
                return True

            # Allow short tokens (NY, LA, SF) only on whole-word matches
            if len(variation_compact) <= 3 and variation_clean in market_tokens:
                return True

        return False
    
    def extract_probability(orderbook: Dict[str, Any]) -> Optional[float]:
        if not orderbook:
            return None

        yes_bids = orderbook.get('yes', [])
        no_bids = orderbook.get('no', [])

        if yes_bids:
            price = yes_bids[0].get('price')
            if price is not None:
                return price / 100.0

        if no_bids:
            price = no_bids[0].get('price')
            if price is not None:
                return 1.0 - (price / 100.0)

        return None

    def find_canonical_team_name(team: str) -> Optional[str]:
        team_upper = team.upper()
        squeezed = team_upper.replace(" ", "")

        for canonical, abbrs in KALSHI_TEAM_ABBREVIATIONS.items():
            if canonical == team_upper or canonical.replace(" ", "") == squeezed:
                return canonical
            if canonical in team_upper or team_upper in canonical:
                return canonical
            for abbr in abbrs:
                abbr_clean = abbr.upper().replace(" ", "")
                if abbr_clean and abbr_clean in squeezed:
                    return canonical
        return None

    def build_market_validation(market: Dict[str, Any], scope: str) -> Optional[Dict[str, Any]]:
        if not market:
            return None

        orderbook = kalshi_integrator.get_orderbook(market.get('ticker', '')) if market.get('ticker') else {}
        kalshi_prob = extract_probability(orderbook)

        if kalshi_prob is None:
            return None

        diff = kalshi_prob - sportsbook_prob
        synthetic_market = market.get('synthetic', False)

        mild_threshold = 0.05 if scope == 'head_to_head' else 0.04
        strong_threshold = 0.10 if scope == 'head_to_head' else 0.08
        base_boost = 0.08 if scope == 'head_to_head' else 0.05
        boost_multiplier = 0.6 if synthetic_market else 1.0

        if diff >= strong_threshold:
            validation = 'strong_kalshi_higher'
            confidence_boost = base_boost * 1.2 * boost_multiplier
            edge = diff
        elif diff >= mild_threshold:
            validation = 'kalshi_higher'
            confidence_boost = base_boost * boost_multiplier
            edge = diff
        elif diff <= -strong_threshold:
            validation = 'strong_contradiction'
            confidence_boost = -base_boost * boost_multiplier
            edge = abs(diff)
        elif diff <= -mild_threshold:
            validation = 'kalshi_lower'
            confidence_boost = -base_boost * 0.6 * boost_multiplier
            edge = abs(diff)
        else:
            validation = 'confirms'
            confidence_boost = base_boost * 0.5 * boost_multiplier
            edge = max(diff, 0)

        return {
            'kalshi_prob': kalshi_prob,
            'kalshi_available': True,
            'discrepancy': abs(diff),
            'validation': validation,
            'edge': edge,
            'confidence_boost': confidence_boost,
            'market_ticker': market.get('ticker'),
            'market_title': market.get('title'),
            'market_scope': scope,
            'data_source': 'synthetic' if synthetic_market else 'kalshi'
        }

    try:
        markets = kalshi_integrator.get_sports_markets()

        bet_team = home_team if side == 'home' else away_team
        other_team = away_team if side == 'home' else home_team

        canonical_team = find_canonical_team_name(bet_team) or bet_team.upper()

        matching_market = None
        fallback_market = None

        for market in markets:
            title = market.get('title', '')
            ticker = market.get('ticker', '')
            subtitle = market.get('subtitle', '')
            market_text = f"{title} {ticker} {subtitle}"

            has_bet_team = teams_match(bet_team, market_text)
            has_other_team = teams_match(other_team, market_text)
            is_synthetic_market = market.get('synthetic', False)

            if has_bet_team and has_other_team and not is_synthetic_market:
                matching_market = market
                break

            if has_bet_team and fallback_market is None:
                fallback_market = market

        if matching_market:
            result = build_market_validation(matching_market, 'head_to_head')
            if result:
                return result

        if fallback_market:
            scope = 'head_to_head' if teams_match(other_team, f"{fallback_market.get('title', '')} {fallback_market.get('subtitle', '')}") else 'team_future'
            result = build_market_validation(fallback_market, scope)
            if result:
                return result

        synthetic_market = kalshi_integrator.get_synthetic_market_for_team(canonical_team)
        if synthetic_market:
            result = build_market_validation(synthetic_market, 'synthetic')
            if result:
                return result

        synthetic_prob = kalshi_integrator.synthetic_probability(canonical_team, sport, sportsbook_prob)
        diff = synthetic_prob - sportsbook_prob

        if diff >= 0.06:
            validation = 'kalshi_higher'
            confidence_boost = 0.03
            edge = diff
        elif diff <= -0.06:
            validation = 'strong_contradiction'
            confidence_boost = -0.03
            edge = abs(diff)
        else:
            validation = 'confirms'
            confidence_boost = 0.02
            edge = max(diff, 0)

        return {
            'kalshi_prob': synthetic_prob,
            'kalshi_available': True,
            'discrepancy': abs(diff),
            'validation': validation,
            'edge': edge,
            'confidence_boost': confidence_boost,
            'market_ticker': None,
            'market_title': f"Synthetic confidence for {bet_team}",
            'market_scope': 'synthetic_estimate',
            'data_source': 'synthetic_estimate'
        }

    except Exception as e:
        # Error fetching Kalshi data
        return {
            'kalshi_prob': None,
            'kalshi_available': False,
            'discrepancy': 0,
            'validation': 'error',
            'edge': 0,
            'confidence_boost': 0,
            'market_ticker': None,
            'market_title': None,
            'market_scope': 'error',
            'data_source': 'error'
        }

# Helper to apply Kalshi validation to a betting leg in-place
def integrate_kalshi_into_leg(
    leg_data: Dict[str, Any],
    home_team: str,
    away_team: str,
    side: str,
    base_prob: float,
    sport: str,
    use_kalshi: bool,
) -> None:
    """Mutate a leg dictionary with Kalshi validation + probability blending."""

    # Ensure downstream code sees the reason when Kalshi is not active
    if not use_kalshi:
        leg_data.setdefault('kalshi_validation', {
            'kalshi_available': False,
            'validation': 'disabled',
            'edge': 0,
            'confidence_boost': 0,
            'market_scope': 'disabled',
            'data_source': 'disabled'
        })
        return

    kalshi = None
    try:
        kalshi = st.session_state.get('kalshi_integrator')
    except Exception:
        # When Streamlit session state isn't available (e.g. testing), skip gracefully
        pass

    if not kalshi:
        leg_data['kalshi_validation'] = {
            'kalshi_available': False,
            'validation': 'unavailable',
            'edge': 0,
            'confidence_boost': 0,
            'market_scope': 'not_initialized',
            'data_source': 'unavailable'
        }
        return

    try:
        kalshi_data = validate_with_kalshi(kalshi, home_team, away_team, side, base_prob, sport)
    except Exception:
        leg_data['kalshi_validation'] = {
            'kalshi_available': False,
            'validation': 'error',
            'edge': 0,
            'confidence_boost': 0,
            'market_scope': 'error',
            'data_source': 'error'
        }
        return

    leg_data['kalshi_validation'] = kalshi_data

    if not kalshi_data.get('kalshi_available'):
        return

    original_ai_prob = leg_data.get('ai_prob', base_prob)
    kalshi_prob = kalshi_data.get('kalshi_prob', base_prob)

    blended_prob = (
        original_ai_prob * 0.50 +  # AI model
        kalshi_prob * 0.30 +       # Kalshi market
        base_prob * 0.20           # Sportsbook baseline
    )

    alignment_delta = kalshi_prob - original_ai_prob

    leg_data['ai_prob_before_kalshi'] = original_ai_prob
    leg_data['ai_prob'] = blended_prob
    leg_data['kalshi_influence'] = blended_prob - original_ai_prob
    leg_data['kalshi_alignment_delta'] = alignment_delta
    leg_data['kalshi_alignment_abs'] = abs(alignment_delta)
    leg_data['kalshi_prob_raw'] = kalshi_prob
    leg_data['kalshi_edge'] = kalshi_data.get('edge', 0)
    leg_data['ai_confidence'] = min(
        leg_data.get('ai_confidence', 0.5) + kalshi_data.get('confidence_boost', 0),
        0.95
    )

    leg_data['kalshi_validation'] = kalshi_data

    if not kalshi_data.get('kalshi_available'):
        return

    original_ai_prob = leg_data.get('ai_prob', base_prob)
    kalshi_prob = kalshi_data.get('kalshi_prob', base_prob)

    blended_prob = (
        original_ai_prob * 0.50 +  # AI model
        kalshi_prob * 0.30 +       # Kalshi market
        base_prob * 0.20           # Sportsbook baseline
    )

    alignment_delta = kalshi_prob - original_ai_prob

    leg_data['ai_prob_before_kalshi'] = original_ai_prob
    leg_data['ai_prob'] = blended_prob
    leg_data['kalshi_influence'] = blended_prob - original_ai_prob
    leg_data['kalshi_alignment_delta'] = alignment_delta
    leg_data['kalshi_alignment_abs'] = abs(alignment_delta)
    leg_data['kalshi_prob_raw'] = kalshi_prob
    leg_data['kalshi_edge'] = kalshi_data.get('edge', 0)
    leg_data['ai_confidence'] = min(
        leg_data.get('ai_confidence', 0.5) + kalshi_data.get('confidence_boost', 0),
        0.95
    )

# Helper to apply Kalshi validation to a betting leg in-place
def integrate_kalshi_into_leg(
    leg_data: Dict[str, Any],
    home_team: str,
    away_team: str,
    side: str,
    base_prob: float,
    sport: str,
    use_kalshi: bool,
) -> None:
    """Mutate a leg dictionary with Kalshi validation + probability blending."""

    # Ensure downstream code sees the reason when Kalshi is not active
    if not use_kalshi:
        leg_data.setdefault('kalshi_validation', {
            'kalshi_available': False,
            'validation': 'disabled',
            'edge': 0,
            'confidence_boost': 0,
            'market_scope': 'disabled',
            'data_source': 'disabled'
        })
        return

    kalshi = None
    try:
        kalshi = st.session_state.get('kalshi_integrator')
    except Exception:
        # When Streamlit session state isn't available (e.g. testing), skip gracefully
        pass

    if not kalshi:
        leg_data['kalshi_validation'] = {
            'kalshi_available': False,
            'validation': 'unavailable',
            'edge': 0,
            'confidence_boost': 0,
            'market_scope': 'not_initialized',
            'data_source': 'unavailable'
        }
        return

    try:
        kalshi_data = validate_with_kalshi(kalshi, home_team, away_team, side, base_prob, sport)
    except Exception:
        leg_data['kalshi_validation'] = {
            'kalshi_available': False,
            'validation': 'error',
            'edge': 0,
            'confidence_boost': 0,
            'market_scope': 'error',
            'data_source': 'error'
        }
        return

    leg_data['kalshi_validation'] = kalshi_data

    if not kalshi_data.get('kalshi_available'):
        return

    original_ai_prob = leg_data.get('ai_prob', base_prob)
    kalshi_prob = kalshi_data.get('kalshi_prob', base_prob)

    blended_prob = (
        original_ai_prob * 0.50 +  # AI model
        kalshi_prob * 0.30 +       # Kalshi market
        base_prob * 0.20           # Sportsbook baseline
    )

    alignment_delta = kalshi_prob - original_ai_prob

    leg_data['ai_prob_before_kalshi'] = original_ai_prob
    leg_data['ai_prob'] = blended_prob
    leg_data['kalshi_influence'] = blended_prob - original_ai_prob
    leg_data['kalshi_alignment_delta'] = alignment_delta
    leg_data['kalshi_alignment_abs'] = abs(alignment_delta)
    leg_data['kalshi_prob_raw'] = kalshi_prob
    leg_data['kalshi_edge'] = kalshi_data.get('edge', 0)
    leg_data['ai_confidence'] = min(
        leg_data.get('ai_confidence', 0.5) + kalshi_data.get('confidence_boost', 0),
        0.95
    )

    leg_data['kalshi_validation'] = kalshi_data

    if not kalshi_data.get('kalshi_available'):
        return

    original_ai_prob = leg_data.get('ai_prob', base_prob)
    kalshi_prob = kalshi_data.get('kalshi_prob', base_prob)

    blended_prob = (
        original_ai_prob * 0.50 +  # AI model
        kalshi_prob * 0.30 +       # Kalshi market
        base_prob * 0.20           # Sportsbook baseline
    )

    alignment_delta = kalshi_prob - original_ai_prob

    leg_data['ai_prob_before_kalshi'] = original_ai_prob
    leg_data['ai_prob'] = blended_prob
    leg_data['kalshi_influence'] = blended_prob - original_ai_prob
    leg_data['kalshi_alignment_delta'] = alignment_delta
    leg_data['kalshi_alignment_abs'] = abs(alignment_delta)
    leg_data['kalshi_prob_raw'] = kalshi_prob
    leg_data['kalshi_edge'] = kalshi_data.get('edge', 0)
    leg_data['ai_confidence'] = min(
        leg_data.get('ai_confidence', 0.5) + kalshi_data.get('confidence_boost', 0),
        0.95
    )

# Helper to apply Kalshi validation to a betting leg in-place
def integrate_kalshi_into_leg(
    leg_data: Dict[str, Any],
    home_team: str,
    away_team: str,
    side: str,
    base_prob: float,
    sport: str,
    use_kalshi: bool,
) -> None:
    """Mutate a leg dictionary with Kalshi validation + probability blending."""

    # Ensure downstream code sees the reason when Kalshi is not active
    if not use_kalshi:
        leg_data.setdefault('kalshi_validation', {
            'kalshi_available': False,
            'validation': 'disabled',
            'edge': 0,
            'confidence_boost': 0,
            'market_scope': 'disabled',
            'data_source': 'disabled'
        })
        return

    kalshi = None
    try:
        kalshi = st.session_state.get('kalshi_integrator')
    except Exception:
        # When Streamlit session state isn't available (e.g. testing), skip gracefully
        pass

    if not kalshi:
        leg_data['kalshi_validation'] = {
            'kalshi_available': False,
            'validation': 'unavailable',
            'edge': 0,
            'confidence_boost': 0,
            'market_scope': 'not_initialized',
            'data_source': 'unavailable'
        }
        return

    try:
        kalshi_data = validate_with_kalshi(kalshi, home_team, away_team, side, base_prob, sport)
    except Exception:
        leg_data['kalshi_validation'] = {
            'kalshi_available': False,
            'validation': 'error',
            'edge': 0,
            'confidence_boost': 0,
            'market_scope': 'error',
            'data_source': 'error'
        }
        return

    leg_data['kalshi_validation'] = kalshi_data

    if not kalshi_data.get('kalshi_available'):
        return

    original_ai_prob = leg_data.get('ai_prob', base_prob)
    kalshi_prob = kalshi_data.get('kalshi_prob', base_prob)

    blended_prob = (
        original_ai_prob * 0.50 +  # AI model
        kalshi_prob * 0.30 +       # Kalshi market
        base_prob * 0.20           # Sportsbook baseline
    )

    alignment_delta = kalshi_prob - original_ai_prob

    leg_data['ai_prob_before_kalshi'] = original_ai_prob
    leg_data['ai_prob'] = blended_prob
    leg_data['kalshi_influence'] = blended_prob - original_ai_prob
    leg_data['kalshi_alignment_delta'] = alignment_delta
    leg_data['kalshi_alignment_abs'] = abs(alignment_delta)
    leg_data['kalshi_prob_raw'] = kalshi_prob
    leg_data['kalshi_edge'] = kalshi_data.get('edge', 0)
    leg_data['ai_confidence'] = min(
        leg_data.get('ai_confidence', 0.5) + kalshi_data.get('confidence_boost', 0),
        0.95
    )

def _odds_api_base():
    return "https://api.the-odds-api.com"


def build_leg_apisports_payload(summary: Any, side: str, sport_key: Optional[str] = None) -> Dict[str, Any]:
    """Return a compact snapshot of API-Sports data for a parlay leg."""

    if not summary:
        return {}

    def _get(container, attr):
        if container is None:
            return None
        if isinstance(container, dict):
            return container.get(attr)
        return getattr(container, attr, None)

    team_obj = _get(summary, 'home' if side == 'home' else 'away')
    opponent_obj = _get(summary, 'away' if side == 'home' else 'home')

    payload = {
        'game_id': _get(summary, 'id'),
        'league': _get(summary, 'league'),
        'season': _get(summary, 'season'),
        'status': _get(summary, 'status'),
        'kickoff': _get(summary, 'kickoff_local'),
        'venue': _get(summary, 'venue'),
        'sport_key': sport_key or _get(summary, 'sport_key'),
        'sport_name': _get(summary, 'sport_name'),
        'scoring_metric': _get(summary, 'scoring_metric'),
        'team_record': _get(team_obj, 'record'),
        'team_form': _get(team_obj, 'form'),
        'trend': _get(team_obj, 'trend'),
        'team_avg_points_for': _get(team_obj, 'average_points_for'),
        'team_avg_points_against': _get(team_obj, 'average_points_against'),
        'opponent_record': _get(opponent_obj, 'record'),
        'opponent_form': _get(opponent_obj, 'form'),
        'opponent_avg_points_for': _get(opponent_obj, 'average_points_for'),
        'opponent_avg_points_against': _get(opponent_obj, 'average_points_against'),
    }

    return {k: v for k, v in payload.items() if v not in (None, '')}


def build_leg_sportsdata_payload(summary: Any, side: str, sport_key: Optional[str] = None) -> Dict[str, Any]:
    """Return a compact snapshot of SportsData.io metrics for a leg."""

    if not summary:
        return {}

    def _get(obj: Any, attr: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(attr)
        return getattr(obj, attr, None)

    if side == 'total':
        team_obj = None
        opponent_obj = None
    else:
        team_obj = _get(summary, 'home' if side == 'home' else 'away')
        opponent_obj = _get(summary, 'away' if side == 'home' else 'home')

    game_payload = {
        'game_key': _get(summary, 'game_key'),
        'season': _get(summary, 'season'),
        'season_type': _get(summary, 'season_type'),
        'week': _get(summary, 'week'),
        'kickoff': _get(summary, 'kickoff'),
        'stadium': _get(summary, 'stadium'),
        'status': _get(summary, 'status'),
        'sport_key': sport_key or _get(summary, 'sport_key') or 'americanfootball_nfl',
        'sport_name': _get(summary, 'sport_name') or SPORTSDATA_CONFIG.get(
            sport_key or '',
            {
                'label': 'SportsData.io',
            },
        ).get('label', 'SportsData.io'),
    }

    payload: Dict[str, Any] = dict(game_payload)

    def _team_payload(team: Any, opponent: Any) -> Dict[str, Any]:
        if not team:
            return {}
        record = _get(team, 'record')
        streak = _get(team, 'streak')
        trend = _get(team, 'trend')
        net_ppg = _get(team, 'net_points_per_game')
        turnover = _get(team, 'turnover_margin')
        power_index = _get(team, 'power_index')
        opp_power = _get(opponent, 'power_index') if opponent else None
        strength_delta = None
        if isinstance(power_index, (int, float)) and isinstance(opp_power, (int, float)):
            strength_delta = float(power_index) - float(opp_power)

        return {
            'team_record': record,
            'streak': streak,
            'trend': trend,
            'points_for_per_game': _get(team, 'points_for_per_game'),
            'points_against_per_game': _get(team, 'points_against_per_game'),
            'net_points_per_game': net_ppg,
            'turnover_margin': turnover,
            'power_index': power_index,
            'strength_delta': strength_delta,
        }

    if side == 'total':
        home_payload = _team_payload(_get(summary, 'home'), _get(summary, 'away'))
        away_payload = _team_payload(_get(summary, 'away'), _get(summary, 'home'))
        combined_delta = None
        if isinstance(home_payload.get('strength_delta'), (int, float)) and isinstance(
            away_payload.get('strength_delta'), (int, float)
        ):
            combined_delta = float(home_payload['strength_delta']) - float(away_payload['strength_delta'])
        home_pf = _safe_float(home_payload.get('points_for_per_game'))
        away_pf = _safe_float(away_payload.get('points_for_per_game'))
        home_pa = _safe_float(home_payload.get('points_against_per_game'))
        away_pa = _safe_float(away_payload.get('points_against_per_game'))
        avg_points_total = None
        if home_pf is not None and away_pf is not None:
            avg_points_total = home_pf + away_pf
        avg_points_allowed = None
        if home_pa is not None and away_pa is not None:
            avg_points_allowed = home_pa + away_pa
        payload.update(
            {
                'home_team': _get(summary, 'home').name if hasattr(_get(summary, 'home'), 'name') else _get(summary, 'home'),
                'away_team': _get(summary, 'away').name if hasattr(_get(summary, 'away'), 'name') else _get(summary, 'away'),
                'trend': home_payload.get('trend') or away_payload.get('trend'),
                'combined_strength_delta': combined_delta,
                'home_points_for_per_game': home_pf,
                'away_points_for_per_game': away_pf,
                'home_points_against_per_game': home_pa,
                'away_points_against_per_game': away_pa,
                'avg_points_total': avg_points_total,
                'avg_points_allowed': avg_points_allowed,
            }
        )
    else:
        team_payload = _team_payload(team_obj, opponent_obj)
        payload.update(team_payload)

    return {k: v for k, v in payload.items() if v not in (None, '')}


def format_timestamp_utc(ts: Optional[datetime]) -> Optional[str]:
    """Format a naive UTC timestamp for display."""

    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d %H:%M UTC")
    return None

def fetch_oddsapi_snapshot(api_key: str, sport_key: str) -> Dict[str, Any]:
    url = f"{_odds_api_base()}/v4/sports/{sport_key}/odds"
    params = {"apiKey": api_key, "regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american"}
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        
        # Check if we got valid data
        if not data or not isinstance(data, list):
            st.warning(f"No data returned for {sport_key}")
            return {"events": []}
            
    except requests.exceptions.Timeout:
        st.warning(f"Request timeout for {sport_key} - skipping")
        return {"events": []}
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed for {sport_key}: {str(e)}")
        return {"events": []}
    except Exception as e:
        st.error(f"Unexpected error fetching {sport_key}: {str(e)}")
        return {"events": []}
    
    events = []
    for ev in (data or []):
        home, away = ev.get("home_team"), ev.get("away_team")
        markets = {"h2h": {}, "spreads": [], "totals": []}
        for b in ev.get("bookmakers") or []:
            for m in b.get("markets") or []:
                key = m.get("key")
                if key == "h2h":
                    for o in m.get("outcomes") or []:
                        if o.get("name") == home:
                            markets["h2h"]["home"] = {"price": o.get("price")}
                        elif o.get("name") == away:
                            markets["h2h"]["away"] = {"price": o.get("price")}
                elif key == "spreads":
                    for o in m.get("outcomes") or []:
                        markets["spreads"].append(
                            {
                                "name": o.get("name"),
                                "point": o.get("point"),
                                "price": o.get("price"),
                                "bookmaker": b.get("title") or b.get("key"),
                            }
                        )
                elif key == "totals":
                    for o in m.get("outcomes") or []:
                        markets["totals"].append(
                            {
                                "name": o.get("name"),
                                "point": o.get("point"),
                                "price": o.get("price"),
                                "bookmaker": b.get("title") or b.get("key"),
                            }
                        )
        events.append(
            {
                "id": ev.get("id"),
                "commence_time": ev.get("commence_time"),
                "home_team": home,
                "away_team": away,
                "markets": markets,
                "sport_key": sport_key,
                "league": format_sport_label(sport_key),
                "bookmakers": [
                    {
                        "key": b.get("key"),
                        "title": b.get("title") or b.get("key"),
                        "last_update": b.get("last_update"),
                        "markets": b.get("markets") or [],
                    }
                    for b in ev.get("bookmakers") or []
                ],
            }
        )
    return {"events": events}


def convert_american_to_decimal(odds: Any) -> Optional[float]:
    """Convert American odds to decimal odds with safety guards."""

    try:
        decimal = american_to_decimal_safe(odds)
        if decimal is None:
            return None
        return float(decimal)
    except Exception:
        return None


def filter_events_by_date_range(
    events: Optional[List[Dict[str, Any]]],
    start_date: Optional[datetime.date],
    end_date: Optional[datetime.date],
    tz_name: str = "UTC",
) -> List[Dict[str, Any]]:
    """Return events whose commence_time falls between the provided dates."""

    if events is None:
        return []

    if start_date and end_date and start_date > end_date:
        start_date, end_date = end_date, start_date

    try:
        tz_info = pytz.timezone(tz_name)
    except Exception:
        tz_info = pytz.timezone("UTC")

    filtered: List[Dict[str, Any]] = []
    for event in events:
        commence = event.get("commence_time")
        if not commence:
            continue

        ts = pd.to_datetime(commence, utc=True, errors="coerce")
        if pd.isna(ts):
            continue

        local_ts = ts.tz_convert(tz_info)
        event_date = local_ts.date()

        if start_date and event_date < start_date:
            continue
        if end_date and event_date > end_date:
            continue

        event_copy = dict(event)
        event_copy["event_date"] = event_date
        event_copy["commence_local"] = local_ts
        filtered.append(event_copy)

    return filtered


def compute_best_overall_odds(
    events: Optional[List[Dict[str, Any]]],
    tz_name: str = "UTC",
) -> pd.DataFrame:
    """Calculate the best available odds per game/side across bookmakers."""

    if not events:
        return pd.DataFrame(
            columns=
            [
                "event_id",
                "league",
                "event_date",
                "home_team",
                "away_team",
                "market",
                "side",
                "line",
                "bookmaker",
                "american_odds",
                "decimal_odds",
            ]
        )

    try:
        tz_info = pytz.timezone(tz_name)
    except Exception:
        tz_info = pytz.timezone("UTC")

    records: List[Dict[str, Any]] = []

    for event in events:
        event_id = event.get("id")
        home = event.get("home_team")
        away = event.get("away_team")
        if not (event_id and home and away):
            continue

        league = event.get("league") or format_sport_label(event.get("sport_key"))
        event_date = event.get("event_date")
        if event_date is None:
            commence = event.get("commence_time")
            ts = pd.to_datetime(commence, utc=True, errors="coerce")
            if not pd.isna(ts):
                event_date = ts.tz_convert(tz_info).date()

        bookmakers = event.get("bookmakers") or []
        for bookmaker in bookmakers:
            book_name = bookmaker.get("title") or bookmaker.get("key") or "Unknown"
            for market in bookmaker.get("markets") or []:
                market_key = (market.get("key") or "").lower()
                if market_key not in {"h2h", "spreads", "totals"}:
                    continue

                outcomes = market.get("outcomes") or []
                for outcome in outcomes:
                    price = outcome.get("price")
                    decimal_odds = convert_american_to_decimal(price)
                    if decimal_odds is None:
                        continue

                    market_label = {
                        "h2h": "Moneyline",
                        "spreads": "Spread",
                        "totals": "Total",
                    }[market_key]

                    side = None
                    line_value: Optional[float] = None
                    name = outcome.get("name")

                    if market_key == "h2h":
                        if _names_match(name, home):
                            side = "home"
                        elif _names_match(name, away):
                            side = "away"
                        else:
                            continue
                    elif market_key == "spreads":
                        if _names_match(name, home):
                            side = "home"
                        elif _names_match(name, away):
                            side = "away"
                        else:
                            continue
                        line_value = _safe_float(outcome.get("point"))
                    elif market_key == "totals":
                        side = (name or "").strip().lower()
                        if side not in {"over", "under"}:
                            continue
                        line_value = _safe_float(outcome.get("point"))

                    records.append(
                        {
                            "event_id": event_id,
                            "league": league,
                            "event_date": event_date,
                            "home_team": home,
                            "away_team": away,
                            "market": market_label,
                            "side": side,
                            "line": line_value if market_key != "h2h" else None,
                            "bookmaker": book_name,
                            "american_odds": price,
                            "decimal_odds": decimal_odds,
                        }
                    )

    if not records:
        return pd.DataFrame(
            columns=
            [
                "event_id",
                "league",
                "event_date",
                "home_team",
                "away_team",
                "market",
                "side",
                "line",
                "bookmaker",
                "american_odds",
                "decimal_odds",
            ]
        )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["decimal_odds"] = pd.to_numeric(df["decimal_odds"], errors="coerce")
    df["american_odds"] = pd.to_numeric(df["american_odds"], errors="coerce")
    df = df.dropna(subset=["decimal_odds"])
    if df.empty:
        return df

    def _normalize_line_value(row: pd.Series) -> Optional[float]:
        """Convert spread/total lines to a comparable float if possible."""

        market = row.get("market")
        if market not in {"Spread", "Total"}:
            return None

        raw_line = row.get("line")
        if raw_line is None or (isinstance(raw_line, float) and pd.isna(raw_line)):
            return None

        try:
            if isinstance(raw_line, (int, float)):
                return round(float(raw_line), 3)

            text = str(raw_line).strip()
            if not text:
                return None

            lowered = text.lower()
            if lowered in {"pk", "pick", "pick'em", "pickem"}:
                return 0.0

            cleaned = (
                text.replace("½", ".5")
                .replace("–", "-")
                .replace("—", "-")
                .replace("−", "-")
            )
            return round(float(cleaned), 3)
        except Exception:
            return None

    df["line_key"] = df.apply(_normalize_line_value, axis=1)

    sort_cols = ["decimal_odds", "bookmaker"]
    df_sorted = df.sort_values(sort_cols, ascending=[False, True])
    group_cols = [
        "event_id",
        "league",
        "event_date",
        "home_team",
        "away_team",
        "market",
        "side",
        "line_key",
    ]

    best = df_sorted.groupby(group_cols, as_index=False).first()
    if "line" in best.columns:
        best.drop(columns=["line"], inplace=True)
    best.rename(columns={"line_key": "line"}, inplace=True)
    best.sort_values(["event_date", "league", "event_id", "market", "side"], inplace=True)

    column_order = [
        "event_id",
        "league",
        "event_date",
        "home_team",
        "away_team",
        "market",
        "side",
        "line",
        "bookmaker",
        "american_odds",
        "decimal_odds",
    ]

    for col in column_order:
        if col not in best.columns:
            best[col] = None

    return best[column_order]


def build_best_odds_report(
    api_key: str,
    sport_keys: List[str],
    start_date: Optional[date],
    end_date: Optional[date],
    tz_name: str,
) -> pd.DataFrame:
    """Fetch odds snapshots and return the best book per market across sports."""

    aggregated_events: List[Dict[str, Any]] = []
    for sport in sport_keys:
        snapshot = fetch_oddsapi_snapshot(api_key, sport)
        events = snapshot.get("events", [])
        if not events:
            continue

        filtered = filter_events_by_date_range(events, start_date, end_date, tz_name)
        aggregated_events.extend(filtered)

    return compute_best_overall_odds(aggregated_events, tz_name)

def calculate_profit(decimal_odds: float, stake: float = 100) -> float:
    return (decimal_odds - 1.0) * stake


def decimal_to_american(decimal_odds: Optional[float]) -> Optional[int]:
    """Convert decimal odds back to American format for display."""

    try:
        if decimal_odds is None:
            return None
        dec = float(decimal_odds)
        if dec <= 1.0:
            return None
        if dec >= 2.0:
            return int(round((dec - 1.0) * 100))
        return int(round(-100 / (dec - 1.0)))
    except Exception:
        return None


def _blend_probability(base: float, new_value: float, weight: float) -> float:
    """Blend two probability estimates using the provided weight."""

    weight = max(0.0, min(weight, 1.0))
    return max(0.0, min(1.0, base * (1 - weight) + new_value * weight))


def _record_pct_from_text(record: Any) -> Optional[float]:
    if not record:
        return None
    tokens = [tok for tok in re.split(r"[^0-9]", str(record)) if tok]
    if not tokens:
        return None
    try:
        wins = float(tokens[0])
        losses = float(tokens[1]) if len(tokens) > 1 else 0.0
        draws = float(tokens[2]) if len(tokens) > 2 else 0.0
    except ValueError:
        return None
    total = wins + losses + draws
    if total <= 0:
        return None
    return wins / total


def _logistic_probability(score: float, *, scale: float = 1.0) -> float:
    if scale <= 0:
        scale = 1.0
    scaled = max(-8.0, min(8.0, score / scale))
    return 1.0 / (1.0 + math.exp(-scaled))


def _sportsdata_probability_for_leg(leg: Dict[str, Any]) -> Optional[float]:
    """Derive a probability estimate for a leg using SportsData.io metrics."""

    payload = leg.get('sportsdata') or {}
    if not payload:
        return None

    market = (leg.get('type') or leg.get('market') or '').lower()
    if not market:
        return None

    side = str(leg.get('side') or '').lower()
    record_pct = _record_pct_from_text(payload.get('team_record'))
    strength_delta = _safe_float(payload.get('strength_delta'))
    net_ppg = _safe_float(payload.get('net_points_per_game'))
    turnover_margin = _safe_float(payload.get('turnover_margin'))
    trend = str(payload.get('trend') or '').lower()

    trend_boost = 0.0
    if 'hot' in trend:
        trend_boost = 0.18
    elif 'cold' in trend:
        trend_boost = -0.18

    def _home_edge() -> float:
        if side == 'home':
            return 0.25
        if side == 'away':
            return -0.25
        return 0.0

    def _common_score_components() -> float:
        score = 0.0
        if strength_delta is not None:
            score += max(-1.5, min(1.5, strength_delta / 10.0))
        if net_ppg is not None:
            score += max(-1.2, min(1.2, net_ppg / 6.0))
        if turnover_margin is not None:
            score += max(-0.9, min(0.9, turnover_margin / 5.0))
        if record_pct is not None:
            score += max(-1.0, min(1.0, (record_pct - 0.5) * 3.6))
        score += trend_boost
        score += _home_edge()
        return score

    if 'moneyline' in market or market == 'ml':
        if all(val is None for val in (strength_delta, net_ppg, record_pct, turnover_margin)):
            return None
        score = _common_score_components()
        return _logistic_probability(score, scale=1.25)

    if 'spread' in market:
        line_val = _safe_float(leg.get('point'))
        if line_val is None:
            line_val = _safe_float(leg.get('line'))
        score = _common_score_components()
        if line_val is not None:
            effective_line = line_val
            if side == 'away':
                effective_line = -line_val
            score += max(-1.5, min(1.5, -effective_line / 6.5))
        return _logistic_probability(score, scale=1.4)

    if 'total' in market:
        line_val = _safe_float(leg.get('point'))
        if line_val is None:
            line_val = _safe_float(leg.get('line'))
        if line_val is None:
            return None

        avg_total = _safe_float(payload.get('avg_points_total'))
        if avg_total is None:
            home_pf = _safe_float(payload.get('home_points_for_per_game'))
            away_pf = _safe_float(payload.get('away_points_for_per_game'))
            if home_pf is not None and away_pf is not None:
                avg_total = home_pf + away_pf
        if avg_total is None:
            avg_allowed = _safe_float(payload.get('avg_points_allowed'))
            if avg_allowed is not None:
                avg_total = avg_allowed
        if avg_total is None:
            combined_delta = _safe_float(payload.get('combined_strength_delta'))
            if combined_delta is not None:
                avg_total = line_val + combined_delta
        if avg_total is None:
            return None

        diff = avg_total - line_val
        if side == 'under':
            diff = -diff
        if trend_boost:
            diff += trend_boost * 1.2
        return _logistic_probability(diff, scale=5.5)

    return None


def build_best_bets_per_game(
    api_key: str,
    sport_keys: List[str],
    start_date: Optional[date],
    end_date: Optional[date],
    tz_name: str,
    *,
    sentiment_analyzer: Optional[SentimentAnalyzer],
    ml_predictor: Optional[MLPredictor],
    use_sentiment: bool,
    use_ml_predictions: bool,
    min_ai_confidence: float,
    use_kalshi: bool,
    theover_ml_data: Optional[pd.DataFrame] = None,
    theover_spreads_data: Optional[pd.DataFrame] = None,
    theover_totals_data: Optional[pd.DataFrame] = None,
    sportsdata_clients: Optional[Dict[str, Any]] = None,
    apisports_clients: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Return one best-value leg per game across the provided sports and date range."""

    sportsdata_clients = sportsdata_clients or {}
    apisports_clients = apisports_clients or {}

    timezone_label = tz_name or 'UTC'
    try:
        tz_info = pytz.timezone(timezone_label)
    except Exception:
        tz_info = pytz.UTC
        timezone_label = 'UTC'

    aggregated_events: List[Dict[str, Any]] = []
    legs: List[Dict[str, Any]] = []
    enriched_legs: List[Dict[str, Any]] = []
    event_meta: Dict[str, Dict[str, Any]] = {}
    apisports_games_cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    default_sentiment = {'score': 0, 'trend': 'neutral'}

    for sport_key in sport_keys:
        snapshot = fetch_oddsapi_snapshot(api_key, sport_key)
        events = snapshot.get("events", [])
        if not events:
            continue

        filtered_events = filter_events_by_date_range(events, start_date, end_date, timezone_label)
        if not filtered_events:
            continue

        aggregated_events.extend(filtered_events)

        apisports_client = apisports_clients.get(sport_key)
        sportsdata_client = sportsdata_clients.get(sport_key)

        for event in filtered_events:
            event_id = event.get("id")
            home = event.get("home_team")
            away = event.get("away_team")
            if not event_id or not home or not away:
                continue

            commence_time = event.get("commence_time")
            commence_dt = pd.to_datetime(commence_time, utc=True, errors="coerce")
            commence_local_dt = None
            commence_display = commence_time
            if not pd.isna(commence_dt):
                try:
                    commence_local_dt = commence_dt.tz_convert(tz_info)
                    commence_display = commence_local_dt.strftime("%Y-%m-%d %H:%M %Z")
                except Exception:
                    commence_local_dt = None

            event_meta[event_id] = {
                'home': home,
                'away': away,
                'league': format_sport_label(sport_key),
                'sport_key': sport_key,
                'commence_time': commence_time,
                'commence_local_dt': commence_local_dt,
                'commence_local_display': commence_display,
            }

            apisports_payload_home = None
            apisports_payload_away = None
            apisports_payload_total = None

            if (
                apisports_client
                and getattr(apisports_client, 'is_configured', lambda: False)()
                and not pd.isna(commence_dt)
            ):
                try:
                    local_date = commence_dt.tz_convert(tz_info).date()
                    cache_key = (sport_key, local_date.isoformat())
                    if cache_key not in apisports_games_cache:
                        apisports_games_cache[cache_key] = apisports_client.get_games_by_date(
                            local_date,
                            timezone=timezone_label,
                        )
                    matched_game = apisports_client.match_game(
                        apisports_games_cache.get(cache_key, []),
                        home,
                        away,
                    )
                    if matched_game:
                        summary = apisports_client.build_game_summary(
                            matched_game,
                            tz_name=timezone_label,
                        )
                        apisports_payload_home = build_leg_apisports_payload(
                            summary,
                            'home',
                            sport_key=sport_key,
                        )
                        apisports_payload_away = build_leg_apisports_payload(
                            summary,
                            'away',
                            sport_key=sport_key,
                        )
                        total_trend = (
                            (apisports_payload_home or {}).get('trend')
                            or (apisports_payload_away or {}).get('trend')
                        )
                        apisports_payload_total = {
                            key: getattr(summary, attr)
                            for key, attr in [
                                ('game_id', 'id'),
                                ('league', 'league'),
                                ('season', 'season'),
                                ('status', 'status'),
                                ('kickoff', 'kickoff_local'),
                                ('venue', 'venue'),
                            ]
                            if getattr(summary, attr, None)
                        }
                        apisports_payload_total['sport_key'] = sport_key
                        apisports_payload_total['sport_name'] = getattr(summary, 'sport_name', None)
                        apisports_payload_total['scoring_metric'] = getattr(summary, 'scoring_metric', None)
                        if total_trend:
                            apisports_payload_total['trend'] = total_trend
                except Exception:
                    apisports_payload_home = None
                    apisports_payload_away = None
                    apisports_payload_total = None

            sportsdata_payload_home = None
            sportsdata_payload_away = None
            sportsdata_payload_total = None

            if (
                sportsdata_client
                and getattr(sportsdata_client, 'is_configured', lambda: False)()
                and not pd.isna(commence_dt)
            ):
                try:
                    local_date_sd = commence_dt.tz_convert(tz_info).date()
                    summary_sd = sportsdata_client.find_game_insight(
                        local_date_sd,
                        home,
                        away,
                    )
                    if summary_sd:
                        sportsdata_payload_home = build_leg_sportsdata_payload(
                            summary_sd,
                            'home',
                            sport_key=sport_key,
                        )
                        sportsdata_payload_away = build_leg_sportsdata_payload(
                            summary_sd,
                            'away',
                            sport_key=sport_key,
                        )
                        sportsdata_payload_total = build_leg_sportsdata_payload(
                            summary_sd,
                            'total',
                            sport_key=sport_key,
                        )
                except Exception:
                    sportsdata_payload_home = None
                    sportsdata_payload_away = None
                    sportsdata_payload_total = None

            if use_sentiment and sentiment_analyzer:
                try:
                    home_sentiment = dict(sentiment_analyzer.get_team_sentiment(home, sport_key))
                except Exception:
                    home_sentiment = default_sentiment.copy()
                try:
                    away_sentiment = dict(sentiment_analyzer.get_team_sentiment(away, sport_key))
                except Exception:
                    away_sentiment = default_sentiment.copy()
            else:
                home_sentiment = default_sentiment.copy()
                away_sentiment = default_sentiment.copy()

            mkts = event.get("markets") or {}

            ml_prediction_result = None
            if (
                use_ml_predictions
                and ml_predictor is not None
                and "h2h" in mkts
            ):
                home_price = _dig(mkts["h2h"], "home.price")
                away_price = _dig(mkts["h2h"], "away.price")
                if home_price is not None and away_price is not None:
                    ml_context = {
                        'sport_key': sport_key,
                        'event_id': event_id,
                        'apisports_home': apisports_payload_home,
                        'apisports_away': apisports_payload_away,
                        'sportsdata_home': sportsdata_payload_home,
                        'sportsdata_away': sportsdata_payload_away,
                    }
                    try:
                        ml_prediction_result = ml_predictor.predict_game_outcome(
                            home,
                            away,
                            home_price,
                            away_price,
                            home_sentiment['score'],
                            away_sentiment['score'],
                            context=ml_context,
                        )
                    except Exception:
                        ml_prediction_result = None

            # Moneyline legs
            if "h2h" in mkts:
                home_price = _dig(mkts["h2h"], "home.price")
                away_price = _dig(mkts["h2h"], "away.price")

                if home_price is not None and -750 <= home_price <= 750:
                    base_prob = implied_p_from_american(home_price)
                    ai_prob = base_prob
                    ai_confidence = 0.5
                    ai_edge = 0.0
                    ml_prob = None
                    if ml_prediction_result:
                        ai_prob = ml_prediction_result.get('home_prob', base_prob)
                        ai_confidence = ml_prediction_result.get('confidence', 0.5)
                        ai_edge = ml_prediction_result.get('edge', 0.0)
                        ml_prob = ml_prediction_result.get('home_prob')
                    if ai_confidence >= min_ai_confidence:
                        decimal_odds = american_to_decimal_safe(home_price)
                        if decimal_odds is not None:
                            leg_data = {
                                "event_id": event_id,
                                "type": "Moneyline",
                                "team": home,
                                "side": "home",
                                "market": "ML",
                                "label": f"{away} @ {home} — {home} ML @{home_price}",
                                "p": base_prob,
                                "ai_prob": ai_prob,
                                "ai_confidence": ai_confidence,
                                "ai_edge": ai_edge,
                                "d": decimal_odds,
                                "sentiment_trend": home_sentiment.get('trend', 'neutral'),
                                "sport_key": sport_key,
                                "home_team": home,
                                "away_team": away,
                                "commence_time": commence_time,
                                "league": event_meta[event_id]['league'],
                                "ml_probability": ml_prob,
                            }
                            if ml_prediction_result:
                                leg_data['ai_model_source'] = ml_prediction_result.get('model_used')
                                leg_data['ai_training_rows'] = ml_prediction_result.get('training_rows')
                                component_breakdown = ml_prediction_result.get('component_probabilities')
                                if isinstance(component_breakdown, dict) and component_breakdown:
                                    leg_data['ai_component_probabilities'] = component_breakdown
                            if apisports_payload_home:
                                leg_data['apisports'] = apisports_payload_home
                            if sportsdata_payload_home:
                                leg_data['sportsdata'] = sportsdata_payload_home

                            integrate_kalshi_into_leg(
                                leg_data,
                                home,
                                away,
                                'home',
                                base_prob,
                                sport_key,
                                use_kalshi,
                            )

                            legs.append(leg_data)

                if away_price is not None and -750 <= away_price <= 750:
                    base_prob = implied_p_from_american(away_price)
                    ai_prob = base_prob
                    ai_confidence = 0.5
                    ai_edge = 0.0
                    ml_prob = None
                    if ml_prediction_result:
                        ai_prob = ml_prediction_result.get('away_prob', base_prob)
                        ai_confidence = ml_prediction_result.get('confidence', 0.5)
                        ai_edge = ml_prediction_result.get('edge', 0.0)
                        ml_prob = ml_prediction_result.get('away_prob')
                    if ai_confidence >= min_ai_confidence:
                        decimal_odds = american_to_decimal_safe(away_price)
                        if decimal_odds is not None:
                            leg_data = {
                                "event_id": event_id,
                                "type": "Moneyline",
                                "team": away,
                                "side": "away",
                                "market": "ML",
                                "label": f"{away} @ {home} — {away} ML @{away_price}",
                                "p": base_prob,
                                "ai_prob": ai_prob,
                                "ai_confidence": ai_confidence,
                                "ai_edge": ai_edge,
                                "d": decimal_odds,
                                "sentiment_trend": away_sentiment.get('trend', 'neutral'),
                                "sport_key": sport_key,
                                "home_team": home,
                                "away_team": away,
                                "commence_time": commence_time,
                                "league": event_meta[event_id]['league'],
                                "ml_probability": ml_prob,
                            }
                            if ml_prediction_result:
                                leg_data['ai_model_source'] = ml_prediction_result.get('model_used')
                                leg_data['ai_training_rows'] = ml_prediction_result.get('training_rows')
                                component_breakdown = ml_prediction_result.get('component_probabilities')
                                if isinstance(component_breakdown, dict) and component_breakdown:
                                    leg_data['ai_component_probabilities'] = component_breakdown
                            if apisports_payload_away:
                                leg_data['apisports'] = apisports_payload_away
                            if sportsdata_payload_away:
                                leg_data['sportsdata'] = sportsdata_payload_away

                            integrate_kalshi_into_leg(
                                leg_data,
                                home,
                                away,
                                'away',
                                base_prob,
                                sport_key,
                                use_kalshi,
                            )

                            legs.append(leg_data)

            # Spreads
            if "spreads" in mkts:
                for outcome in mkts["spreads"][:6]:
                    team_name = outcome.get("name")
                    point = outcome.get("point")
                    price = outcome.get("price")
                    if team_name is None or point is None or price is None:
                        continue

                    base_prob = implied_p_from_american(price)
                    sentiment = home_sentiment if team_name == home else away_sentiment
                    ai_prob = base_prob * (1 + sentiment.get('score', 0) * 0.40)
                    ai_prob = max(0.1, min(0.9, ai_prob))
                    ai_confidence = 0.65
                    decimal_odds = american_to_decimal_safe(price)
                    if decimal_odds is None or ai_confidence < min_ai_confidence:
                        continue

                    leg_data = {
                        "event_id": event_id,
                        "type": "Spread",
                        "team": team_name,
                        "side": "home" if team_name == home else "away",
                        "point": point,
                        "market": "Spread",
                        "label": f"{away} @ {home} — {team_name} {point:+.1f} @{price}",
                        "p": base_prob,
                        "ai_prob": ai_prob,
                        "ai_confidence": ai_confidence,
                        "ai_edge": abs(ai_prob - base_prob),
                        "d": decimal_odds,
                        "sentiment_trend": sentiment.get('trend', 'neutral'),
                        "sport_key": sport_key,
                        "home_team": home,
                        "away_team": away,
                        "commence_time": commence_time,
                        "league": event_meta[event_id]['league'],
                        "ml_probability": None,
                    }
                    if team_name == home and apisports_payload_home:
                        leg_data['apisports'] = apisports_payload_home
                    elif team_name == away and apisports_payload_away:
                        leg_data['apisports'] = apisports_payload_away
                    if team_name == home and sportsdata_payload_home:
                        leg_data['sportsdata'] = sportsdata_payload_home
                    elif team_name == away and sportsdata_payload_away:
                        leg_data['sportsdata'] = sportsdata_payload_away

                    integrate_kalshi_into_leg(
                        leg_data,
                        home,
                        away,
                        'home' if team_name == home else 'away',
                        base_prob,
                        sport_key,
                        use_kalshi,
                    )

                    legs.append(leg_data)

            # Totals
            if "totals" in mkts:
                for outcome in mkts["totals"][:6]:
                    side = outcome.get("name")
                    point = outcome.get("point")
                    price = outcome.get("price")
                    if side is None or point is None or price is None:
                        continue

                    base_prob = implied_p_from_american(price)
                    combined_sentiment = (home_sentiment.get('score', 0) + away_sentiment.get('score', 0)) / 2
                    ai_prob = base_prob * (1 + combined_sentiment * 0.40 * 0.5)
                    ai_prob = max(0.1, min(0.9, ai_prob))
                    ai_confidence = 0.60
                    decimal_odds = american_to_decimal_safe(price)
                    if decimal_odds is None or ai_confidence < min_ai_confidence:
                        continue

                    leg_data = {
                        "event_id": event_id,
                        "type": "Total",
                        "team": f"{home} vs {away}",
                        "side": side,
                        "point": point,
                        "market": "Total",
                        "label": f"{away} @ {home} — {side} {point} @{price}",
                        "p": base_prob,
                        "ai_prob": ai_prob,
                        "ai_confidence": ai_confidence,
                        "ai_edge": abs(ai_prob - base_prob),
                        "d": decimal_odds,
                        "sentiment_trend": 'neutral',
                        "sport_key": sport_key,
                        "home_team": home,
                        "away_team": away,
                        "commence_time": commence_time,
                        "league": event_meta[event_id]['league'],
                        "ml_probability": None,
                    }
                    if apisports_payload_total:
                        leg_data['apisports'] = apisports_payload_total
                    if sportsdata_payload_total:
                        leg_data['sportsdata'] = sportsdata_payload_total

                    if use_kalshi:
                        leg_data['kalshi_validation'] = {
                            'kalshi_available': False,
                            'validation': 'unsupported',
                            'edge': 0,
                            'confidence_boost': 0,
                            'market_scope': 'totals',
                            'data_source': 'unsupported',
                        }

                    legs.append(leg_data)

    if not legs:
        return pd.DataFrame(), []

    apply_theover_probabilities_to_legs(
        legs,
        theover_ml_data=theover_ml_data,
        theover_spreads_data=theover_spreads_data,
        theover_totals_data=theover_totals_data,
    )

    best_odds_df = compute_best_overall_odds(aggregated_events, timezone_label) if aggregated_events else pd.DataFrame()

    odds_lookup: Dict[Tuple[Any, str, str, Optional[float]], Dict[str, Any]] = {}
    if not best_odds_df.empty:
        for record in best_odds_df.to_dict(orient='records'):
            event_id = record.get('event_id')
            if not event_id:
                continue
            market_label = record.get('market')
            side_value = str(record.get('side') or '').lower()
            line_val = record.get('line')
            try:
                line_key = round(float(line_val), 3) if line_val is not None and not pd.isna(line_val) else None
            except Exception:
                line_key = None
            odds_lookup[(event_id, market_label, side_value, line_key)] = record
            odds_lookup.setdefault((event_id, market_label, side_value, None), record)

    for leg in legs:
        event_id = leg.get('event_id')
        if not event_id:
            continue

        leg_type = (leg.get('type') or leg.get('market') or '').lower()
        if 'total' in leg_type:
            market_label = 'Total'
        elif 'spread' in leg_type:
            market_label = 'Spread'
        else:
            market_label = 'Moneyline'

        raw_side = leg.get('side')
        side_key = str(raw_side).lower() if raw_side is not None else ''
        if market_label == 'Moneyline' and side_key not in {'home', 'away'}:
            if leg.get('team') == leg.get('home_team'):
                side_key = 'home'
            elif leg.get('team') == leg.get('away_team'):
                side_key = 'away'

        point_val = leg.get('point')
        try:
            line_key = round(float(point_val), 3) if point_val is not None else None
        except Exception:
            line_key = None

        odds_record = odds_lookup.get((event_id, market_label, side_key, line_key))
        if odds_record is None:
            odds_record = odds_lookup.get((event_id, market_label, side_key, None))

        best_decimal = odds_record.get('decimal_odds') if odds_record else None
        best_american = odds_record.get('american_odds') if odds_record else None
        best_book = odds_record.get('bookmaker') if odds_record else None
        if odds_record and market_label != 'Moneyline' and odds_record.get('line') is not None:
            point_val = odds_record.get('line')

        if best_decimal is None:
            best_decimal = leg.get('d')
        decimal_float = _safe_float(best_decimal)
        if decimal_float is None or decimal_float <= 1:
            continue
        best_decimal = decimal_float

        if best_american is None and best_decimal is not None:
            best_american = decimal_to_american(best_decimal)

        market_implied_prob = None
        try:
            market_implied_prob = 1.0 / float(best_decimal)
        except Exception:
            market_implied_prob = None

        ai_prob_post_kalshi = leg.get('ai_prob', leg.get('p'))
        if ai_prob_post_kalshi is None:
            continue

        ai_prob_pre_kalshi = leg.get('ai_prob_before_kalshi', ai_prob_post_kalshi)
        ml_prob = leg.get('ml_probability')
        theover_prob = leg.get('theover_probability')
        sportsdata_prob = _sportsdata_probability_for_leg(leg)
        kalshi_validation = leg.get('kalshi_validation') or {}
        kalshi_prob = _safe_float(
            kalshi_validation.get('kalshi_prob')
            if kalshi_validation.get('kalshi_available')
            else None
        )

        ai_prob_effective = _safe_float(ai_prob_pre_kalshi)
        if ai_prob_effective is None:
            ai_prob_effective = _safe_float(ai_prob_post_kalshi)
        if ai_prob_effective is None:
            continue

        if ml_prob is not None and abs(ml_prob - ai_prob_effective) > 1e-3:
            ai_prob_effective = _blend_probability(ai_prob_effective, ml_prob, 0.45)

        if theover_prob is not None and abs(theover_prob - ai_prob_effective) > 1e-3:
            ai_prob_effective = _blend_probability(ai_prob_effective, theover_prob, 0.25)

        if sportsdata_prob is not None and abs(sportsdata_prob - ai_prob_effective) > 1e-3:
            ai_prob_effective = _blend_probability(ai_prob_effective, sportsdata_prob, 0.30)

        if kalshi_prob is not None and abs(kalshi_prob - ai_prob_effective) > 1e-3:
            ai_prob_effective = _blend_probability(ai_prob_effective, kalshi_prob, 0.35)
            alignment_delta = kalshi_prob - (ai_prob_pre_kalshi if ai_prob_pre_kalshi is not None else ai_prob_effective)
            if alignment_delta < -0.08:
                ai_prob_effective -= min(0.12, abs(alignment_delta) * 0.6)
            elif alignment_delta > 0.08:
                ai_prob_effective += min(0.08, alignment_delta * 0.5)

        ai_prob_effective = max(0.01, min(0.99, ai_prob_effective))

        probability_candidates: List[Tuple[str, float]] = []

        def _append_prob(label: str, value: Any) -> None:
            coerced = _safe_float(value)
            if coerced is not None and not math.isnan(coerced):
                probability_candidates.append((label, float(coerced)))

        _append_prob("Consensus", ai_prob_effective)
        _append_prob("AI (pre-Kalshi)", ai_prob_pre_kalshi)
        _append_prob("AI (post-Kalshi)", ai_prob_post_kalshi)
        _append_prob("ML", ml_prob)
        _append_prob("theover.ai", theover_prob)
        _append_prob("SportsData", sportsdata_prob)
        _append_prob("Kalshi", kalshi_prob)

        primary_source: Optional[str] = None
        if probability_candidates:
            primary_source, _ = max(
                probability_candidates, key=lambda item: item[1]
            )

        win_metric: Optional[float] = float(ai_prob_effective) if ai_prob_effective is not None else None

        if win_metric is None:
            for fallback_prob in (
                ai_prob_post_kalshi,
                ml_prob,
                theover_prob,
                sportsdata_prob,
                kalshi_prob,
            ):
                coerced = _safe_float(fallback_prob)
                if coerced is not None and not math.isnan(coerced):
                    win_metric = float(coerced)
                    break

        best_win_source = (
            f"Consensus ({primary_source})"
            if primary_source is not None and primary_source != "Consensus"
            else "Consensus"
        )

        ai_ev = (
            ev_rate(ai_prob_effective, float(best_decimal))
            if market_implied_prob is not None
            else None
        )
        ai_edge = (
            ai_prob_effective - market_implied_prob
            if market_implied_prob is not None
            else None
        )

        sportsdata_delta = (
            sportsdata_prob - market_implied_prob
            if sportsdata_prob is not None and market_implied_prob is not None
            else None
        )
        sportsdata_ev = (
            ev_rate(sportsdata_prob, float(best_decimal))
            if sportsdata_prob is not None
            else None
        )
        theover_ev = (
            ev_rate(theover_prob, float(best_decimal))
            if theover_prob is not None
            else None
        )

        edge_candidates: List[Tuple[str, float]] = []
        if ai_ev is not None:
            edge_candidates.append(("AI EV", ai_ev))
        if sportsdata_ev is not None:
            edge_candidates.append(("SportsData EV", sportsdata_ev))
        if theover_ev is not None:
            edge_candidates.append(("theover.ai EV", theover_ev))
        if not edge_candidates and ai_edge is not None:
            edge_candidates.append(("AI Edge", ai_edge))
        if not edge_candidates and sportsdata_delta is not None:
            edge_candidates.append(("SportsData Δ", sportsdata_delta))

        best_edge = None
        best_edge_source = None
        if edge_candidates:
            best_edge_source, best_edge = max(edge_candidates, key=lambda item: item[1])

        event_info = event_meta.get(event_id, {})
        league = event_info.get('league', format_sport_label(leg.get('sport_key')))
        home_team = leg.get('home_team') or event_info.get('home')
        away_team = leg.get('away_team') or event_info.get('away')
        commence_display = event_info.get('commence_local_display')
        commence_sort = event_info.get('commence_local_dt')

        if market_label == 'Moneyline':
            selection = leg.get('team')
        elif market_label == 'Spread':
            coerced_point = _safe_float(point_val)
            if coerced_point is not None:
                selection = f"{leg.get('team')} {coerced_point:+g}"
            else:
                selection = f"{leg.get('team')} {point_val}"
        else:
            coerced_point = _safe_float(point_val)
            if coerced_point is not None:
                selection = f"{str(raw_side).title()} {coerced_point:g}"
            else:
                selection = f"{str(raw_side).title()} {point_val}"

        leg_metrics = {
            'event_id': event_id,
            'sport_key': leg.get('sport_key'),
            'league': league,
            'home_team': home_team,
            'away_team': away_team,
            'market': market_label,
            'side': side_key,
            'selection': selection,
            'line': point_val,
            'best_book': best_book,
            'best_decimal': best_decimal,
            'best_american': best_american,
            'implied_prob': market_implied_prob,
            'ai_prob_raw': ai_prob_post_kalshi,
            'ai_prob_pre_kalshi': ai_prob_pre_kalshi,
            'ai_prob_effective': ai_prob_effective,
            'ai_ev': ai_ev,
            'ai_edge': ai_edge,
            'ai_confidence': leg.get('ai_confidence'),
            'ml_probability': leg.get('ml_probability'),
            'ml_model': leg.get('ai_model_source'),
            'theover_probability': leg.get('theover_probability'),
            'theover_delta': leg.get('theover_probability_delta'),
            'theover_source': leg.get('theover_probability_source'),
            'sportsdata_probability': sportsdata_prob,
            'sportsdata_delta': sportsdata_delta,
            'sportsdata_ev': sportsdata_ev,
            'kalshi_prob': kalshi_prob,
            'kalshi_delta': leg.get('kalshi_alignment_delta'),
            'kalshi_edge': leg.get('kalshi_edge'),
            'kalshi_verdict': (leg.get('kalshi_validation') or {}).get('validation'),
            'best_edge': best_edge,
            'best_edge_source': best_edge_source,
            'win_probability': win_metric,
            'win_prob_source': best_win_source,
            'win_metric': win_metric,
            'commence_display': commence_display,
            'commence_sort': commence_sort,
            'commence_time': event_info.get('commence_time'),
        }
        enriched_legs.append(leg_metrics)

    if not enriched_legs:
        return pd.DataFrame(), []

    legs_by_event: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for metrics in enriched_legs:
        legs_by_event[metrics['event_id']].append(metrics)

    best_rows: List[Dict[str, Any]] = []
    for event_id, options in legs_by_event.items():
        ranked = [opt for opt in options if opt.get('win_metric') is not None]
        if not ranked:
            continue
        best_option = max(
            ranked,
            key=lambda item: (
                item.get('win_metric', 0.0),
                item.get('ai_prob_effective', item.get('ai_prob_raw', 0.0)),
                item.get('best_edge', float('-inf')) if item.get('best_edge') is not None else float('-inf'),
            ),
        )

        row = {
            'League': best_option['league'],
            'Game': f"{best_option['away_team']} @ {best_option['home_team']}",
            'Commence (Local)': best_option['commence_display'],
            'Market': best_option['market'],
            'Side': best_option['side'],
            'Selection': best_option['selection'],
            'Line': best_option['line'],
            'Best Book': best_option['best_book'],
            'Best American': best_option['best_american'],
            'Best Decimal': best_option['best_decimal'],
            'Implied Prob %': best_option['implied_prob'] * 100 if best_option['implied_prob'] is not None else None,
            'AI Prob %': best_option['ai_prob_effective'] * 100 if best_option['ai_prob_effective'] is not None else None,
            'AI Raw %': best_option['ai_prob_raw'] * 100 if best_option['ai_prob_raw'] is not None else None,
            'AI EV %': best_option['ai_ev'] * 100 if best_option['ai_ev'] is not None else None,
            'AI Edge pp': best_option['ai_edge'] * 100 if best_option['ai_edge'] is not None else None,
            'AI Confidence %': best_option['ai_confidence'] * 100 if best_option['ai_confidence'] is not None else None,
            'ML Prob %': best_option['ml_probability'] * 100 if best_option['ml_probability'] is not None else None,
            'ML Model': best_option['ml_model'],
            'theover.ai %': best_option['theover_probability'] * 100 if best_option['theover_probability'] is not None else None,
            'theover Δ pp': best_option['theover_delta'] * 100 if best_option['theover_delta'] is not None else None,
            'theover Source': best_option['theover_source'],
            'SportsData Prob %': best_option['sportsdata_probability'] * 100 if best_option['sportsdata_probability'] is not None else None,
            'SportsData Δ pp': best_option['sportsdata_delta'] * 100 if best_option['sportsdata_delta'] is not None else None,
            'Kalshi Prob %': best_option['kalshi_prob'] * 100 if best_option['kalshi_prob'] is not None else None,
            'Kalshi Δ pp': best_option['kalshi_delta'] * 100 if best_option['kalshi_delta'] is not None else None,
            'Kalshi Edge %': best_option['kalshi_edge'] * 100 if best_option['kalshi_edge'] is not None else None,
            'Kalshi Verdict': best_option['kalshi_verdict'],
            'Best Edge %': best_option['best_edge'] * 100 if best_option['best_edge'] is not None else None,
            'Best Edge Source': best_option['best_edge_source'],
            'Best Win Prob %': best_option['win_metric'] * 100 if best_option['win_metric'] is not None else None,
            'Win Prob Source': best_option['win_prob_source'],
            'Event ID': event_id,
            'Sport Key': best_option['sport_key'],
            'Commence (UTC)': best_option['commence_time'],
            'Commence Sort': best_option['commence_sort'],
        }
        best_rows.append(row)

    if not best_rows:
        return pd.DataFrame(), enriched_legs

    best_df = pd.DataFrame(best_rows)
    if 'Commence Sort' in best_df.columns:
        try:
            best_df.sort_values(
                by=['Best Win Prob %', 'Best Edge %', 'Commence Sort'],
                ascending=[False, False, True],
                inplace=True,
                na_position='last',
            )
        except Exception:
            best_df.sort_values(
                by=['Best Win Prob %', 'Best Edge %'],
                ascending=[False, False],
                inplace=True,
                na_position='last',
            )
        best_df.drop(columns=['Commence Sort'], inplace=True)
    else:
        best_df.sort_values(
            by=['Best Win Prob %', 'Best Edge %'],
            ascending=[False, False],
            inplace=True,
            na_position='last',
        )

    return best_df.reset_index(drop=True), enriched_legs

def _tokenize_name(name: str) -> List[str]:
    return [token for token in re.split(r"[^a-z0-9]+", (name or "").lower()) if token]


def _names_match(candidate: str, *targets: str) -> bool:
    candidate = (candidate or "").lower().strip()
    if not candidate:
        return False
    candidate_tokens = set(_tokenize_name(candidate))
    for target in targets:
        target_clean = (target or "").lower()
        if not target_clean:
            continue
        if candidate in target_clean or target_clean in candidate:
            return True
        target_tokens = set(_tokenize_name(target_clean))
        if candidate_tokens and candidate_tokens.issubset(target_tokens):
            return True
    return False


LEAGUE_KEYWORDS: Dict[str, List[str]] = {
    "nfl": ["nfl", "national football league"],
    "nba": ["nba", "national basketball association"],
    "nhl": ["nhl", "national hockey league"],
    "mlb": ["mlb", "major league baseball"],
    "ncaaf": [
        "ncaaf",
        "ncaa football",
        "college football",
        "ncaa football fbs",
        "ncaa fbs",
        "college football fbs",
        "fbs",
        "cfb",
    ],
    "ncaab": [
        "ncaab",
        "ncaa basketball",
        "college basketball",
        "ncaa men's basketball",
        "ncaa hoops",
        "cbb",
    ],
}


def _league_matches(leg_league: str, candidate_league: str) -> bool:
    if not leg_league or not candidate_league:
        return True

    leg_norm = leg_league.lower().strip()
    cand_norm = candidate_league.lower().strip()
    if not cand_norm:
        return True

    if leg_norm in cand_norm:
        return True

    tokens = set(_tokenize_name(cand_norm))
    keywords = LEAGUE_KEYWORDS.get(leg_norm, [leg_norm])
    for keyword in keywords:
        keyword_norm = keyword.lower()
        if keyword_norm in cand_norm:
            return True
        keyword_tokens = set(_tokenize_name(keyword_norm))
        if keyword_tokens and keyword_tokens.issubset(tokens):
            return True
    return False


def _normalize_probability_value(value: Any) -> Optional[float]:
    if value is None:
        return None

    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "null"}:
        return None

    raw = raw.replace("%", "")

    try:
        prob = float(raw)
    except ValueError:
        match = re.search(r"[-+]?\d*\.?\d+", raw)
        if not match:
            return None
        try:
            prob = float(match.group())
        except ValueError:
            return None

    if prob < 0:
        return None

    if prob > 1:
        prob = prob / 100.0
    if prob > 1:
        return None

    return max(0.0, min(prob, 1.0))


def _parse_moneyline_value(value: Any) -> Optional[int]:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        try:
            return int(value)
        except Exception:
            return None

    match = re.search(r"[-+]?\d+", str(value))
    if not match:
        return None
    try:
        return int(match.group())
    except Exception:
        return None


def _implied_probability_from_moneyline(odds: Optional[int]) -> Optional[float]:
    if odds is None:
        return None
    if odds >= 0:
        prob = 100.0 / (odds + 100.0)
    else:
        prob = (-odds) / ((-odds) + 100.0)
    return max(0.0, min(prob, 1.0))


THEOVER_HOME_PROB_CANDIDATES = [
    'ml_home_prob',
    'home_win_probability',
    'home_probability',
    'home_prob',
    'prob_home',
]
THEOVER_AWAY_PROB_CANDIDATES = [
    'ml_away_prob',
    'away_win_probability',
    'away_probability',
    'away_prob',
    'prob_away',
]
THEOVER_GENERIC_PROB_CANDIDATES = [
    'model_probability',
    'win_probability',
    'ai_probability',
    'probability',
    'pick_probability',
]
THEOVER_SPREAD_PROB_CANDIDATES = [
    'prob_cover',
    'cover_probability',
    'spread_probability',
]
THEOVER_TOTAL_OVER_CANDIDATES = [
    'prob_over',
    'over_probability',
]
THEOVER_TOTAL_UNDER_CANDIDATES = [
    'prob_under',
    'under_probability',
]
THEOVER_LINE_CANDIDATES = [
    'line',
    'spread',
    'spread_line',
    'handicap',
    'point',
    'total',
    'total_line',
]
THEOVER_HOME_ODDS_CANDIDATES = [
    'home_moneyline',
    'home_odds',
    'ml_home_odds',
    'home_price',
    'moneyline_home',
    'moneylineods',
]
THEOVER_AWAY_ODDS_CANDIDATES = [
    'away_moneyline',
    'away_odds',
    'ml_away_odds',
    'away_price',
    'moneyline_away',
]


def _find_first_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    for candidate in candidates:
        for column in columns:
            if candidate in column:
                return column
    return None


def _coerce_probability(row: pd.Series, candidates: Iterable[str]) -> Tuple[Optional[float], Optional[str]]:
    column = _find_first_column(row.index, candidates)
    if column is None:
        return None, None
    value = _normalize_probability_value(row.get(column))
    if value is None:
        return None, None
    return value, column


def _extract_line_from_pick(pick_text: Any) -> Optional[float]:
    if not isinstance(pick_text, str):
        return None
    match = re.search(r"[-+]?\d*\.?\d+", pick_text)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def _coerce_line(row: pd.Series, fallback_text: Any = None) -> Tuple[Optional[float], Optional[str]]:
    column = _find_first_column(row.index, THEOVER_LINE_CANDIDATES)
    if column:
        value = _safe_float(row.get(column))
        if value is not None:
            return value, column
    parsed = _extract_line_from_pick(fallback_text)
    if parsed is not None:
        return parsed, 'pick'
    return None, None


def _resolve_theover_entry(
    records: List[Dict[str, Any]],
    league: str,
    home: str,
    away: str,
) -> Tuple[Dict[str, Any], bool]:
    league_norm = (league or '').lower().strip()
    for entry in records:
        if not _league_matches(league_norm, entry.get('league_norm', '')):
            continue
        if _names_match(entry.get('home'), home) and _names_match(entry.get('away'), away):
            return entry, False
        if _names_match(entry.get('home'), away) and _names_match(entry.get('away'), home):
            return entry, True

    entry = {
        'league': league,
        'league_norm': league_norm,
        'home': home,
        'away': away,
        'home_tokens': set(_tokenize_name(home)),
        'away_tokens': set(_tokenize_name(away)),
        'ml': {'home': None, 'away': None},
        'spreads': {},
        'totals': {},
    }
    records.append(entry)
    return entry, False


def _map_side(side: str, swapped: bool) -> str:
    if not swapped:
        return side
    if side == 'home':
        return 'away'
    if side == 'away':
        return 'home'
    return side


def _find_line_bucket(
    buckets: Dict[Optional[float], Dict[str, Any]],
    target_line: Optional[float],
) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    if not buckets:
        return None, None
    if target_line is None:
        key, value = next(iter(buckets.items()))
        return key, value

    try:
        target = round(abs(float(target_line)), 3)
    except Exception:
        return None, None

    best_key = None
    best_diff = None
    for key in buckets.keys():
        if key is None:
            continue
        diff = abs(key - target)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_key = key

    tolerance = 0.25
    if best_key is not None and (best_diff is None or best_diff <= tolerance):
        return best_key, buckets.get(best_key)

    if None in buckets:
        return None, buckets.get(None)
    return None, None


def _ingest_theover_ml_row(
    entry: Dict[str, Any],
    row: pd.Series,
    swapped: bool,
    idx: int,
    row_home: str,
    row_away: str,
) -> None:
    section = entry.setdefault('ml', {'home': None, 'away': None})
    home_key = _map_side('home', swapped)
    away_key = _map_side('away', swapped)

    home_prob, home_source = _coerce_probability(row, THEOVER_HOME_PROB_CANDIDATES)
    away_prob, away_source = _coerce_probability(row, THEOVER_AWAY_PROB_CANDIDATES)

    if home_prob is None or away_prob is None:
        generic_prob, generic_source = _coerce_probability(row, THEOVER_GENERIC_PROB_CANDIDATES)
        pick_val = str(row.get('pick', '')).strip()
        if generic_prob is not None:
            if _names_match(pick_val, row_home) and home_prob is None:
                home_prob = generic_prob
                home_source = generic_source
            elif _names_match(pick_val, row_away) and away_prob is None:
                away_prob = generic_prob
                away_source = generic_source

    home_odds_col = _find_first_column(row.index, THEOVER_HOME_ODDS_CANDIDATES)
    away_odds_col = _find_first_column(row.index, THEOVER_AWAY_ODDS_CANDIDATES)

    if home_prob is None and home_odds_col:
        ml_value = _parse_moneyline_value(row.get(home_odds_col))
        implied = _implied_probability_from_moneyline(ml_value)
        if implied is not None:
            home_prob = implied
            home_source = f"moneyline:{home_odds_col}"

    if away_prob is None and away_odds_col:
        ml_value = _parse_moneyline_value(row.get(away_odds_col))
        implied = _implied_probability_from_moneyline(ml_value)
        if implied is not None:
            away_prob = implied
            away_source = f"moneyline:{away_odds_col}"

    if home_prob is not None:
        section[home_key] = {
            'prob': home_prob,
            'source': home_source,
            'row_index': idx,
            'moneyline': _parse_moneyline_value(row.get(home_odds_col)) if home_odds_col else None,
        }
    if away_prob is not None:
        section[away_key] = {
            'prob': away_prob,
            'source': away_source,
            'row_index': idx,
            'moneyline': _parse_moneyline_value(row.get(away_odds_col)) if away_odds_col else None,
        }


def _ingest_theover_spread_row(
    entry: Dict[str, Any],
    row: pd.Series,
    swapped: bool,
    idx: int,
    row_home: str,
    row_away: str,
) -> None:
    line_value, line_source = _coerce_line(row, row.get('pick'))
    line_key = round(abs(line_value), 3) if line_value is not None else None
    bucket = entry.setdefault('spreads', {}).setdefault(line_key, {'home': None, 'away': None})

    prob_value, prob_source = _coerce_probability(row, THEOVER_SPREAD_PROB_CANDIDATES)
    if prob_value is None:
        generic_prob, generic_source = _coerce_probability(row, THEOVER_GENERIC_PROB_CANDIDATES)
        if generic_prob is not None:
            prob_value = generic_prob
            prob_source = generic_source

    if prob_value is None:
        return

    pick_val = str(row.get('pick', '')).strip()
    side = None
    if pick_val:
        if _names_match(pick_val, row_home):
            side = 'home'
        elif _names_match(pick_val, row_away):
            side = 'away'
    if side is None:
        team_val = row.get('team')
        if isinstance(team_val, str):
            if _names_match(team_val, row_home):
                side = 'home'
            elif _names_match(team_val, row_away):
                side = 'away'
    if side is None:
        pick_lower = pick_val.lower()
        if 'home' in pick_lower:
            side = 'home'
        elif 'away' in pick_lower:
            side = 'away'
    if side is None:
        return

    mapped_side = _map_side(side, swapped)
    bucket[mapped_side] = {
        'prob': prob_value,
        'source': prob_source or line_source,
        'line': line_value,
        'row_index': idx,
    }


def _ingest_theover_total_row(
    entry: Dict[str, Any],
    row: pd.Series,
    idx: int,
) -> None:
    line_value, line_source = _coerce_line(row, row.get('pick'))
    line_key = round(abs(line_value), 3) if line_value is not None else None
    bucket = entry.setdefault('totals', {}).setdefault(line_key, {'over': None, 'under': None})

    prob_over, source_over = _coerce_probability(row, THEOVER_TOTAL_OVER_CANDIDATES)
    prob_under, source_under = _coerce_probability(row, THEOVER_TOTAL_UNDER_CANDIDATES)

    if prob_over is None or prob_under is None:
        generic_prob, generic_source = _coerce_probability(row, THEOVER_GENERIC_PROB_CANDIDATES)
        pick_lower = str(row.get('pick', '')).lower()
        if generic_prob is not None:
            if 'over' in pick_lower and prob_over is None:
                prob_over = generic_prob
                source_over = generic_source
            elif 'under' in pick_lower and prob_under is None:
                prob_under = generic_prob
                source_under = generic_source

    if prob_over is not None:
        bucket['over'] = {
            'prob': prob_over,
            'source': source_over or line_source,
            'line': line_value,
            'row_index': idx,
        }
    if prob_under is not None:
        bucket['under'] = {
            'prob': prob_under,
            'source': source_under or line_source,
            'line': line_value,
            'row_index': idx,
        }


def _infer_theover_market(row: pd.Series, default: Optional[str] = None) -> str:
    market_hint = str(row.get('market', row.get('markettype', row.get('picktype', '')))).lower()
    pick_lower = str(row.get('pick', '')).lower()
    if 'spread' in market_hint:
        return 'spread'
    if 'total' in market_hint or any(token in pick_lower for token in ('over', 'under')):
        return 'total'
    if 'ml' in market_hint or 'moneyline' in market_hint:
        return 'ml'
    if default:
        return default
    return 'ml'


def prepare_theover_dataset(
    theover_data: Optional[pd.DataFrame],
    market_hint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if theover_data is None:
        return None

    if isinstance(theover_data, dict) and theover_data.get('_prepared_theover'):
        return theover_data

    try:
        df = theover_data.copy()
    except Exception:
        return None

    try:
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    except Exception:
        return None

    records: List[Dict[str, Any]] = []
    explicit_market = (market_hint or '').lower() or None

    for idx, row in df.iterrows():
        league_raw = str(row.get('league', row.get('sport', ''))).strip()
        home_raw = str(row.get('home_team', row.get('hometeam', row.get('home', '')))).strip()
        away_raw = str(row.get('away_team', row.get('awayteam', row.get('away', '')))).strip()
        if not (home_raw and away_raw):
            continue

        entry, swapped = _resolve_theover_entry(records, league_raw, home_raw, away_raw)
        market_type = _infer_theover_market(row, explicit_market)

        if market_type == 'spread':
            _ingest_theover_spread_row(entry, row, swapped, idx, home_raw, away_raw)
        elif market_type == 'total':
            _ingest_theover_total_row(entry, row, idx)
        else:
            _ingest_theover_ml_row(entry, row, swapped, idx, home_raw, away_raw)

    return {
        '_prepared_theover': True,
        'dataframe': df,
        'records': records,
        'market_type': explicit_market,
    }


def _format_theover_source(source: Optional[str]) -> str:
    if not source:
        return 'model output'

    source_norm = source.lower()
    if source_norm.startswith('moneyline:'):
        return 'moneyline odds'
    if 'moneyline' in source_norm:
        return 'moneyline odds'
    if 'model' in source_norm or 'win_prob' in source_norm:
        return 'model output'
    if 'prob' in source_norm:
        return 'model output'
    return source.replace('_', ' ')


def _match_theover_ml_leg(leg: Dict[str, Any], entry: Dict[str, Any], swapped: bool) -> Optional[Dict[str, Any]]:
    section = entry.get('ml') or {}
    side = (leg.get('side') or '').lower()
    leg_team = leg.get('team')
    if side not in {'home', 'away'}:
        if leg_team and _names_match(leg_team, entry.get('home')):
            side = 'home'
        elif leg_team and _names_match(leg_team, entry.get('away')):
            side = 'away'
    mapped_side = _map_side(side or 'home', swapped)
    payload = section.get(mapped_side)
    if not payload or payload.get('prob') is None:
        return None

    probability = payload.get('prob')
    home_payload = section.get('home')
    away_payload = section.get('away')
    entry_home_name = entry.get('home')
    entry_away_name = entry.get('away')
    if swapped:
        home_prob_val = away_payload.get('prob') if isinstance(away_payload, dict) else None
        away_prob_val = home_payload.get('prob') if isinstance(home_payload, dict) else None
        entry_home_name, entry_away_name = entry_away_name, entry_home_name
    else:
        home_prob_val = home_payload.get('prob') if isinstance(home_payload, dict) else None
        away_prob_val = away_payload.get('prob') if isinstance(away_payload, dict) else None

    predicted_team = None
    matches = None
    if home_prob_val is not None and away_prob_val is not None:
        if home_prob_val > away_prob_val:
            predicted_team = entry_home_name
            matches = _names_match(leg_team, entry_home_name)
        elif away_prob_val > home_prob_val:
            predicted_team = entry_away_name
            matches = _names_match(leg_team, entry_away_name)
        else:
            matches = None
    elif leg_team:
        target_name = entry_home_name if (side or 'home') == 'home' else entry_away_name
        matches = _names_match(leg_team, target_name)

    signal = '🎯'
    if matches is True:
        signal = '✅'
    elif matches is False:
        signal = '⚠️'

    return {
        'pick': leg_team or predicted_team or entry_home_name,
        'matches': matches,
        'signal': signal,
        'league': entry.get('league'),
        'model_probability': probability,
        'probability_source': payload.get('source'),
        'moneyline_odds': payload.get('moneyline'),
        'predicted_team': predicted_team,
        'row_index': payload.get('row_index'),
    }


def _match_theover_spread_leg(leg: Dict[str, Any], entry: Dict[str, Any], swapped: bool) -> Optional[Dict[str, Any]]:
    section = entry.get('spreads') or {}
    target_line = _safe_float(leg.get('point'))
    line_key, bucket = _find_line_bucket(section, target_line)
    if bucket is None:
        return None

    side = (leg.get('side') or '').lower()
    if side not in {'home', 'away'}:
        team = leg.get('team')
        if team and _names_match(team, entry.get('home')):
            side = 'home'
        elif team and _names_match(team, entry.get('away')):
            side = 'away'
    mapped_side = _map_side(side, swapped)
    payload = bucket.get(mapped_side)
    if not payload or payload.get('prob') is None:
        return None

    probability = payload.get('prob')
    home_payload = bucket.get('home')
    away_payload = bucket.get('away')
    if swapped:
        home_prob_val = away_payload.get('prob') if isinstance(away_payload, dict) else None
        away_prob_val = home_payload.get('prob') if isinstance(home_payload, dict) else None
    else:
        home_prob_val = home_payload.get('prob') if isinstance(home_payload, dict) else None
        away_prob_val = away_payload.get('prob') if isinstance(away_payload, dict) else None

    recommended_side = None
    if home_prob_val is not None and away_prob_val is not None:
        if home_prob_val > away_prob_val:
            recommended_side = 'home'
        elif away_prob_val > home_prob_val:
            recommended_side = 'away'

    matches = None
    if recommended_side is not None:
        matches = (side == recommended_side)

    signal = '🎯'
    if matches is True:
        signal = '✅'
    elif matches is False:
        signal = '⚠️'

    return {
        'pick': leg.get('team'),
        'matches': matches,
        'signal': signal,
        'league': entry.get('league'),
        'model_probability': probability,
        'probability_source': payload.get('source'),
        'predicted_team': entry.get('home') if recommended_side == 'home' else (entry.get('away') if recommended_side == 'away' else None),
        'spread_line': payload.get('line') if payload.get('line') is not None else target_line,
        'row_index': payload.get('row_index'),
    }


def _match_theover_total_leg(leg: Dict[str, Any], entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    section = entry.get('totals') or {}
    target_line = _safe_float(leg.get('point'))
    line_key, bucket = _find_line_bucket(section, target_line)
    if bucket is None:
        return None

    direction = (leg.get('side') or '').strip().lower()
    if direction not in {'over', 'under'}:
        label_lower = (leg.get('label') or '').lower()
        if 'over' in label_lower:
            direction = 'over'
        elif 'under' in label_lower:
            direction = 'under'
    payload = bucket.get(direction)
    if not payload or payload.get('prob') is None:
        return None

    probability = payload.get('prob')
    over_payload = bucket.get('over')
    under_payload = bucket.get('under')
    recommended = None
    if isinstance(over_payload, dict) and isinstance(under_payload, dict):
        over_prob = over_payload.get('prob')
        under_prob = under_payload.get('prob')
        if over_prob is not None and under_prob is not None:
            if over_prob > under_prob:
                recommended = 'over'
            elif under_prob > over_prob:
                recommended = 'under'

    matches = None
    if recommended is not None:
        matches = (direction == recommended)

    signal = '🎯'
    if matches is True:
        signal = '✅'
    elif matches is False:
        signal = '⚠️'

    return {
        'pick': f"{direction.title()} {payload.get('line') if payload.get('line') is not None else leg.get('point')}",
        'matches': matches,
        'signal': signal,
        'league': entry.get('league'),
        'model_probability': probability,
        'probability_source': payload.get('source'),
        'predicted_team': direction.title(),
        'row_index': payload.get('row_index'),
    }


def match_theover_to_leg(
    leg: Dict[str, Any],
    theover_data,
    prepared: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    dataset = prepared
    if dataset is None:
        if theover_data is None:
            return None
        leg_market_type = (leg.get('type') or leg.get('market') or '').lower()
        hint = 'ml'
        if 'total' in leg_market_type:
            hint = 'total'
        elif 'spread' in leg_market_type:
            hint = 'spread'
        dataset = prepare_theover_dataset(theover_data, hint)

    if not dataset:
        return None

    records = dataset.get('records') if isinstance(dataset, dict) else None
    if not records:
        return None

    leg_home = leg.get('home_team')
    leg_away = leg.get('away_team')
    if not (leg_home and leg_away):
        return None

    leg_league = SPORT_KEY_TO_LEAGUE.get(leg.get('sport_key'), '').lower()

    selected_entry = None
    swapped = False
    for entry in records:
        if not _league_matches(leg_league, entry.get('league_norm', '')):
            continue
        if _names_match(entry.get('home'), leg_home) and _names_match(entry.get('away'), leg_away):
            selected_entry = entry
            swapped = False
            break
        if _names_match(entry.get('home'), leg_away) and _names_match(entry.get('away'), leg_home):
            selected_entry = entry
            swapped = True
            break

    if not selected_entry:
        return None

    market_type = (leg.get('type') or leg.get('market') or '').lower()
    if 'total' in market_type:
        return _match_theover_total_leg(leg, selected_entry)
    if 'spread' in market_type:
        return _match_theover_spread_leg(leg, selected_entry, swapped)
    return _match_theover_ml_leg(leg, selected_entry, swapped)


def apply_theover_probabilities_to_legs(
    legs: List[Dict[str, Any]],
    theover_ml_data: Optional[pd.DataFrame] = None,
    theover_spreads_data: Optional[pd.DataFrame] = None,
    theover_totals_data: Optional[pd.DataFrame] = None,
    blend_weight: float = 0.35,
) -> Dict[str, Any]:
    """Normalize theover.ai datasets and blend their probabilities into legs."""

    prepared_theover_ml = (
        prepare_theover_dataset(theover_ml_data, 'ml') if theover_ml_data is not None else None
    )
    prepared_theover_spreads = (
        prepare_theover_dataset(theover_spreads_data, 'spread')
        if theover_spreads_data is not None
        else None
    )
    prepared_theover_totals = (
        prepare_theover_dataset(theover_totals_data, 'total')
        if theover_totals_data is not None
        else None
    )

    def _dataset_for_leg(leg_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        leg_type = (leg_dict.get('type') or leg_dict.get('market') or '').lower()
        if 'total' in leg_type:
            return prepared_theover_totals
        if 'spread' in leg_type:
            return prepared_theover_spreads or prepared_theover_ml
        return prepared_theover_ml

    theover_cache: Dict[int, Dict[str, Any]] = {}

    if prepared_theover_ml or prepared_theover_spreads or prepared_theover_totals:
        for leg in legs:
            dataset = _dataset_for_leg(leg)
            if not dataset:
                if 'ai_prob_pre_theover' in leg:
                    leg['ai_prob'] = leg['ai_prob_pre_theover']
                leg.pop('theover_probability', None)
                leg.pop('theover_probability_delta', None)
                leg.pop('theover_match', None)
                leg.pop('theover_predicted_team', None)
                continue

            try:
                match_info = match_theover_to_leg(leg, None, dataset)
            except Exception:
                match_info = None

            if match_info:
                theover_cache[id(leg)] = match_info
                leg['theover_match'] = match_info
                if match_info.get('predicted_team'):
                    leg['theover_predicted_team'] = match_info.get('predicted_team')

                base_ai_prob = leg.get('ai_prob_pre_theover', leg.get('ai_prob', leg.get('p', 0.5)))
                leg['ai_prob_pre_theover'] = base_ai_prob

                theover_prob = match_info.get('model_probability')
                if theover_prob is None:
                    theover_prob = match_info.get('implied_probability')

                if theover_prob is not None and base_ai_prob is not None:
                    blended_prob = base_ai_prob * (1 - blend_weight) + theover_prob * blend_weight
                    leg['ai_prob'] = blended_prob
                    leg['ai_prob_with_theover'] = blended_prob
                    leg['theover_probability'] = theover_prob
                    leg['theover_probability_source'] = match_info.get('probability_source') or 'moneyline_odds'
                    leg['theover_probability_delta'] = theover_prob - base_ai_prob
                    leg['theover_probability_weight'] = blend_weight
                else:
                    leg['ai_prob'] = base_ai_prob
                    leg.pop('theover_probability', None)
                    leg.pop('theover_probability_delta', None)
            else:
                if 'ai_prob_pre_theover' in leg:
                    leg['ai_prob'] = leg['ai_prob_pre_theover']
                leg.pop('theover_probability', None)
                leg.pop('theover_probability_delta', None)
                leg.pop('theover_match', None)
                leg.pop('theover_predicted_team', None)
    else:
        for leg in legs:
            if 'ai_prob_pre_theover' in leg:
                leg['ai_prob'] = leg['ai_prob_pre_theover']
            leg.pop('theover_probability', None)
            leg.pop('theover_probability_delta', None)
            leg.pop('theover_match', None)
            leg.pop('theover_predicted_team', None)

    return {
        'prepared_ml': prepared_theover_ml,
        'prepared_spreads': prepared_theover_spreads,
        'prepared_totals': prepared_theover_totals,
        'cache': theover_cache,
        'dataset_for_leg': _dataset_for_leg,
    }
def build_combos_ai(
    legs,
    k,
    allow_sgp,
    optimizer,
    theover_ml_data=None,
    theover_spreads_data=None,
    theover_totals_data=None,
    min_probability=0.25,
    max_probability=0.70,
):
    """Build parlay combinations with AI scoring - deduplicates and keeps best odds
    Now filters parlays to realistic probability range (default: 25-70%)"""

    theover_context = apply_theover_probabilities_to_legs(
        legs,
        theover_ml_data=theover_ml_data,
        theover_spreads_data=theover_spreads_data,
        theover_totals_data=theover_totals_data,
    )
    prepared_theover_ml = theover_context['prepared_ml']
    prepared_theover_spreads = theover_context['prepared_spreads']
    prepared_theover_totals = theover_context['prepared_totals']
    theover_cache: Dict[int, Dict[str, Any]] = theover_context['cache']
    dataset_for_leg = theover_context['dataset_for_leg']

    parlay_map = {}  # Maps parlay_key -> best parlay so far

    for combo in itertools.combinations(legs, k):
        if not allow_sgp and len({c["event_id"] for c in combo}) < k:
            continue
        
        # Skip combos with missing required fields
        try:
            # IMPROVED: Create a unique key that treats different spreads for same team as duplicates
            # This prevents parlays like "Bucs -2.5 + Bucs -3.0" which are redundant
            parlay_key_parts = []
            for c in combo:
                event_id = c.get('event_id', '')
                bet_type = c.get('type', '')
                team = c.get('team', '')
                side = c.get('side', '')
                
                # For spreads/totals, don't include exact point in key
                # This way Bucs -2.5 and Bucs -3.0 are treated as same bet
                if bet_type in ['Spread', 'Total']:
                    # Key without specific point value
                    key_part = f"{event_id}_{bet_type}_{team}_{side}"
                else:
                    # For ML, include everything
                    key_part = f"{event_id}_{bet_type}_{team}_{side}"
                
                parlay_key_parts.append(key_part)
            
            parlay_key = tuple(sorted(parlay_key_parts))
            
            # Check if this combination has duplicate legs (same team, same game, same bet type)
            # This catches cases like "Bucs -2.5 + Bucs -3.0" in the same parlay
            unique_bets = set(parlay_key_parts)
            if len(unique_bets) < len(combo):
                continue  # Skip parlays with duplicate bets
                
        except Exception:
            continue  # Skip this combo if we can't create a key
        
        d = 1.0
        p_market = 1.0
        p_ai = 1.0
        
        # Safety check: Skip if any leg has invalid decimal odds
        skip_combo = False
        for c in combo:
            leg_d = c.get("d")
            if leg_d is None or leg_d <= 0:
                skip_combo = True
                break
            d *= leg_d
            p_market *= c.get("p", 0.5)
            p_ai *= c.get("ai_prob", c.get("p", 0.5))
        
        if skip_combo or d <= 0:
            continue  # Skip this combo if odds are invalid
        
        # Get AI score for this parlay
        ai_metrics = optimizer.score_parlay(list(combo))
        
        # Calculate theover.ai validation bonus and probability deltas
        theover_bonus = 0.0
        theover_matches = 0
        theover_conflicts = 0
        theover_prob_deltas: List[float] = []
        theover_prob_sources: set[str] = set()
        theover_prob_count = 0

        if prepared_theover_ml or prepared_theover_spreads or prepared_theover_totals:
            for leg in combo:
                dataset = dataset_for_leg(leg)
                if not dataset:
                    continue

                result = theover_cache.get(id(leg))
                if result is None:
                    try:
                        result = match_theover_to_leg(leg, None, dataset)
                    except Exception:
                        result = None
                    if result:
                        theover_cache[id(leg)] = result
                        leg['theover_match'] = result
                if not result or not isinstance(result, dict):
                    continue

                matches = result.get('matches')
                if matches is True:
                    theover_matches += 1
                elif matches is False:
                    theover_conflicts += 1

                theover_prob = result.get('model_probability')
                if theover_prob is None:
                    theover_prob = result.get('implied_probability')

                base_prob = leg.get('ai_prob_pre_theover', leg.get('ai_prob', leg.get('p', 0.5)))
                if theover_prob is not None and base_prob is not None:
                    theover_prob_deltas.append(theover_prob - base_prob)
                    theover_prob_count += 1
                    raw_source = result.get('probability_source')
                    if not raw_source and result.get('implied_probability') is not None:
                        raw_source = 'moneyline_odds'
                    theover_prob_sources.add(_format_theover_source(raw_source))

        avg_delta = sum(theover_prob_deltas) / len(theover_prob_deltas) if theover_prob_deltas else 0.0
        consensus_bonus = 0.02 * (theover_matches - theover_conflicts)
        raw_bonus = avg_delta + consensus_bonus
        theover_bonus = max(-0.15, min(0.15, raw_bonus))
        theover_probability_sources = sorted(theover_prob_sources)

        profit = calculate_profit(d, 100)
        market_ev = ev_rate(p_market, d)

        # Apply theover.ai bonus to AI score
        base_ai_score = ai_metrics['score']
        boosted_ai_score = base_ai_score * (1.0 + theover_bonus)
        
        parlay_data = {
            "legs": combo,
            "d": d,
            "p": p_market,
            "p_ai": p_ai,
            "ev_market": market_ev,
            "ev_ai": ai_metrics['ai_ev'],
            "profit": profit,
            "ai_score": boosted_ai_score,  # Boosted score with theover.ai
            "base_ai_score": base_ai_score,  # Original score
            "theover_bonus": theover_bonus,
            "theover_matches": theover_matches,
            "theover_conflicts": theover_conflicts,
            "theover_avg_delta": avg_delta,
            "theover_probability_count": theover_prob_count,
            "theover_probability_sources": theover_probability_sources,
            "ai_confidence": ai_metrics['confidence'],
            "ai_edge": ai_metrics['edge'],
            "kalshi_factor": ai_metrics.get('kalshi_factor', 1.0),
            "kalshi_boost": ai_metrics.get('kalshi_boost', 0),
            "kalshi_legs": ai_metrics.get('kalshi_legs', 0),
            "kalshi_alignment_avg": ai_metrics.get('kalshi_alignment_avg', 0.0),
            "kalshi_alignment_abs_avg": ai_metrics.get('kalshi_alignment_abs_avg', 0.0),
            "kalshi_alignment_positive": ai_metrics.get('kalshi_alignment_positive', 0),
            "kalshi_alignment_negative": ai_metrics.get('kalshi_alignment_negative', 0),
            "kalshi_alignment_count": ai_metrics.get('kalshi_alignment_count', 0),
            "live_data_factor": ai_metrics.get('live_data_factor', ai_metrics.get('apisports_factor', 1.0)),
            "live_data_boost": ai_metrics.get('live_data_boost', 0),
            "live_data_legs": ai_metrics.get('live_data_legs', 0),
            "live_data_sports": ai_metrics.get('live_data_sports', []),
            "apisports_factor": ai_metrics.get('apisports_factor', 1.0),
            "apisports_boost": ai_metrics.get('apisports_boost', 0),
            "apisports_legs": ai_metrics.get('apisports_legs', 0),
            "apisports_sports": ai_metrics.get('apisports_sports', []),
            "sportsdata_factor": ai_metrics.get('sportsdata_factor', 1.0),
            "sportsdata_boost": ai_metrics.get('sportsdata_boost', 0),
            "sportsdata_legs": ai_metrics.get('sportsdata_legs', 0),
            "sportsdata_sports": ai_metrics.get('sportsdata_sports', []),
        }
        
        # Keep only the version with best combined odds (highest decimal odds = best payout)
        if parlay_key not in parlay_map or d > parlay_map[parlay_key]["d"]:
            parlay_map[parlay_key] = parlay_data
    
    # Convert back to list
    out = list(parlay_map.values())
    
    # FILTER: Keep only realistic parlays in the value zone
    # Remove longshots AND heavy favorites (chalk parlays)
    out = [p for p in out if min_probability <= p["p_ai"] <= max_probability]
    
    # SIMPLE & EFFECTIVE: Sort by AI Expected Value
    # This finds bets where AI thinks you have an edge
    # Positive EV = long-term profitable
    out.sort(key=lambda x: (x["ev_ai"], x["p_ai"]), reverse=True)
    
    # FINAL DEDUPLICATION: Remove parlays that are too similar
    # This catches edge cases where same game appears multiple times with slightly different bets
    final_parlays = []
    seen_game_combos = set()
    
    for parlay in out:
        # Create a signature of which games are in this parlay
        game_ids = tuple(sorted(set(leg['event_id'] for leg in parlay['legs'])))
        
        # For same-game combinations, also check the specific bets
        bet_signature = tuple(sorted([
            f"{leg['event_id']}_{leg['type']}_{leg.get('team', '')}_{leg.get('side', '')}"
            for leg in parlay['legs']
        ]))
        
        if bet_signature not in seen_game_combos:
            final_parlays.append(parlay)
            seen_game_combos.add(bet_signature)
    
    return final_parlays

def render_parlay_section_ai(
    title,
    rows,
    theover_ml_data=None,
    theover_spreads_data=None,
    theover_totals_data=None,
    timezone_label: Optional[str] = None,
):
    """Render parlays with AI insights"""
    st.markdown(f"### {title}")
    if not rows:
        st.info("No combinations found with current filters")
        return

    prepared_theover_ml = (
        prepare_theover_dataset(theover_ml_data, 'ml') if theover_ml_data is not None else None
    )
    prepared_theover_spreads = (
        prepare_theover_dataset(theover_spreads_data, 'spread')
        if theover_spreads_data is not None
        else None
    )
    prepared_theover_totals = (
        prepare_theover_dataset(theover_totals_data, 'total')
        if theover_totals_data is not None
        else None
    )

    def _dataset_for_leg(leg_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        leg_type = (leg_dict.get('type') or '').lower()
        if 'total' in leg_type:
            return prepared_theover_totals
        if 'spread' in leg_type:
            return prepared_theover_spreads or prepared_theover_ml
        return prepared_theover_ml

    for i, row in enumerate(rows, start=1):
        # AI confidence indicator
        conf = row['ai_confidence']
        if conf > 0.7:
            conf_icon = "🟢"
        elif conf > 0.5:
            conf_icon = "🟡"
        else:
            conf_icon = "🟠"
        
        # EV indicator
        ai_ev_pct = row['ev_ai'] * 100
        if ai_ev_pct > 5:
            ev_icon = "💰"
        elif ai_ev_pct > 0:
            ev_icon = "📈"
        else:
            ev_icon = "📉"
        
        # Probability warning
        prob = row['p_ai']
        if prob < 0.25:
            prob_warning = "⚠️"  # Warning for low probability
        elif prob < 0.35:
            prob_warning = "⚡"  # Caution
        else:
            prob_warning = ""  # Good probability
        
        # theover.ai boost indicator with ML delta context
        theover_boost = ""
        theover_segments: List[str] = []
        match_count = row.get('theover_matches', 0)
        conflict_count = row.get('theover_conflicts', 0)
        if match_count:
            label = "match" if match_count == 1 else "matches"
            theover_segments.append(f"{match_count} {label}")
        if conflict_count:
            label = "conflict" if conflict_count == 1 else "conflicts"
            theover_segments.append(f"⚠️ {conflict_count} {label}")
        prob_count = row.get('theover_probability_count', 0)
        if prob_count:
            delta_pct = row.get('theover_avg_delta', 0.0) * 100
            legs_label = "leg" if prob_count == 1 else "legs"
            theover_segments.append(f"ML Δ{delta_pct:+.1f}pp ({prob_count} {legs_label})")
            sources = row.get('theover_probability_sources') or []
            if sources:
                theover_segments.append("source: " + "/".join(sources))

        if theover_segments:
            details = " • ".join(theover_segments)
            theover_boost = f" | 🎯 theover.ai {details}"

        # Kalshi validation indicator with INFLUENCE
        kalshi_boost = ""
        kalshi_legs = sum(
            1
            for leg in row.get('legs', [])
            if leg.get('kalshi_validation', {}).get('kalshi_available', False)
        )
        kalshi_factor = row.get('kalshi_factor', 1.0)

        if kalshi_legs > 0:
            # Show if Kalshi boosted or reduced score
            if kalshi_factor > 1.05:
                kalshi_boost = f" | 📊 {kalshi_legs} Kalshi✓ ↗️+{(kalshi_factor-1)*100:.0f}%"
            elif kalshi_factor < 0.95:
                kalshi_boost = f" | 📊 {kalshi_legs} Kalshi✓ ↘️{(kalshi_factor-1)*100:.0f}%"
            else:
                kalshi_boost = f" | 📊 {kalshi_legs} Kalshi✓"

            align_avg = row.get('kalshi_alignment_avg', 0.0)
            if align_avg:
                kalshi_boost += f" Δ{align_avg*100:+.1f}pp vs ML"

        apisports_info = ""
        apisports_legs = row.get('apisports_legs', 0)
        apisports_factor = row.get('apisports_factor', 1.0)
        apisports_sports = row.get('apisports_sports', []) or []
        if apisports_legs:
            sport_icon_lookup = {
                "americanfootball_nfl": "🏈",
                "basketball_nba": "🏀",
                "icehockey_nhl": "🏒",
            }
            icon_sequence = "".join(
                sport_icon_lookup.get(sport, "🛰️") for sport in sorted(set(apisports_sports))
            ) or "🛰️"
            label = f"{icon_sequence} API-Sports {apisports_legs}"
            if apisports_factor > 1.02:
                apisports_info = f" | {label} ↗️"
            elif apisports_factor < 0.98:
                apisports_info = f" | {label} ↘️"
            else:
                apisports_info = f" | {label}"

        prob_pct = row['p_ai'] * 100
        with st.expander(
            f"{conf_icon}{ev_icon}{prob_warning} #{i} - AI Score: {row['ai_score']:.1f}{theover_boost}{kalshi_boost}{apisports_info} | Odds: {row['d']:.2f} | AI Prob: {prob_pct:.1f}% | Profit: ${row['profit']:.2f}"
        ):
            # Metrics
            col_a, col_b, col_c, col_d, col_e, col_f = st.columns(6)
            with col_a:
                st.metric("Decimal Odds", f"{row['d']:.3f}")
            with col_b:
                st.metric("AI Probability", f"{prob_pct:.2f}%")
            with col_c:
                st.metric("AI Confidence", f"{conf*100:.1f}%")
            with col_d:
                st.metric("Profit on $100", f"${row['profit']:.2f}")
            with col_e:
                delta_color = "normal" if row['ev_ai'] > 0 else "inverse"
                st.metric("AI Expected Value", f"{ai_ev_pct:.2f}%")
            with col_f:
                if apisports_legs:
                    st.metric(
                        "API-Sports Legs",
                        f"{apisports_legs}/{len(row['legs'])}",
                        help="Number of legs with live API-Sports context",
                    )
                else:
                    st.metric(
                        "API-Sports Legs",
                        "0",
                        help="Provide an API-Sports key (NFL, NBA, or NHL) to enrich supported legs",
                    )
            
            # KALSHI STATUS - ALWAYS SHOW (whether data exists or not)
            st.markdown("---")
            kalshi_legs_with_data = row.get('kalshi_legs', 0)
            total_legs = len(row.get('legs', []))
            
            if kalshi_legs_with_data > 0:
                # HAS KALSHI DATA - Show influence
                st.markdown("### 📊 Kalshi Prediction Market Influence:")

                synthetic_legs = sum(
                    1 for leg in row.get('legs', [])
                    if 'synthetic' in leg.get('kalshi_validation', {}).get('data_source', '')
                )

                if synthetic_legs:
                    st.info(
                        f"🧪 Using simulated Kalshi fallback for {synthetic_legs} leg(s) "
                        "because live market data was unavailable."
                    )


                col_k1, col_k2, col_k3, col_k4 = st.columns(4)
                
                with col_k1:
                    st.metric(
                        "Kalshi Legs",
                        f"{row.get('kalshi_legs', 0)}/{len(row.get('legs', []))}",
                        help="How many legs have Kalshi market data"
                    )
                
                with col_k2:
                    kalshi_boost_val = row.get('kalshi_boost', 0)
                    delta_boost = float(kalshi_boost_val) if kalshi_boost_val else None
                    delta_color = "normal" if kalshi_boost_val >= 0 else "inverse"
                    st.metric(
                        "Kalshi Boost Points",
                        f"{kalshi_boost_val:+.0f}",
                        delta=delta_boost,
                        delta_color=delta_color,
                        help="Raw boost points from Kalshi validation (+15 = strong confirmation, -10 = contradiction)"
                    )
                
                with col_k3:
                    kalshi_factor_val = row.get('kalshi_factor', 1.0)
                    st.metric(
                        "Score Multiplier",
                        f"{kalshi_factor_val:.2f}x",
                        delta=f"{(kalshi_factor_val-1)*100:+.0f}%" if kalshi_factor_val != 1.0 else None,
                        help="How much Kalshi adjusted the AI score (1.0 = no change, >1.0 = boosted, <1.0 = reduced)"
                    )
                
                with col_k4:
                    # Calculate score change from Kalshi
                    base_score = row['ai_score'] / kalshi_factor_val if kalshi_factor_val != 0 else row['ai_score']
                    score_change = row['ai_score'] - base_score
                    st.metric(
                        "Score Impact",
                        f"{score_change:+.1f} pts",
                        help="How many points Kalshi added/subtracted from AI score"
                    )

                align_avg = row.get('kalshi_alignment_avg', 0.0)
                align_abs = row.get('kalshi_alignment_abs_avg', 0.0)
                align_pos = row.get('kalshi_alignment_positive', 0)
                align_neg = row.get('kalshi_alignment_negative', 0)
                align_count = row.get('kalshi_alignment_count', 0)
                disagreement_display = f"{align_neg}/{align_count}" if align_count else "—"

                col_align1, col_align2, col_align3 = st.columns(3)

                with col_align1:
                    st.metric(
                        "Avg Kalshi vs ML",
                        f"{align_avg*100:+.1f} pp",
                        help="Average percentage-point difference between Kalshi pricing and the trained model before blending"
                    )

                with col_align2:
                    st.metric(
                        "Avg Absolute Gap",
                        f"{align_abs*100:.1f} pp",
                        help="Typical gap size between Kalshi prices and the model regardless of direction"
                    )

                with col_align3:
                    st.metric(
                        "Disagreement Legs",
                        disagreement_display,
                        help="Legs where Kalshi is ≥1 percentage point more bearish than the model"
                    )

                # Explanation of Kalshi influence
                if kalshi_factor_val > 1.05:
                    st.success(f"🟢 **Kalshi BOOSTED this parlay by {(kalshi_factor_val-1)*100:.0f}%** - Prediction markets confirm AI analysis!")
                elif kalshi_factor_val < 0.95:
                    st.warning(f"🟠 **Kalshi REDUCED this parlay by {(1-kalshi_factor_val)*100:.0f}%** - Prediction markets skeptical of AI picks.")
                else:
                    st.info("🟡 **Kalshi NEUTRAL** - Prediction markets neither strongly confirm nor contradict AI.")

                if align_count:
                    if align_avg <= -0.05:
                        st.warning(
                            f"⚠️ Kalshi is on average {abs(align_avg)*100:.1f} percentage points more bearish than the model. "
                            "Final AI probabilities are being pulled toward Kalshi's price."
                        )
                    elif align_avg >= 0.05:
                        st.success(
                            f"🟢 Kalshi is on average {align_avg*100:.1f} percentage points more bullish than the model, giving "
                            "the blended AI number an extra push."
                        )

                if align_neg > 0:
                    st.warning(
                        f"⚠️ Kalshi is more bearish than the model on {align_neg} of {align_count} leg(s)."
                        " Expect the blended AI probability to drift toward the market price."
                    )
                elif align_pos > 0:
                    st.success(
                        f"✅ Kalshi prices {align_pos} leg(s) richer than the model, reinforcing the AI edge before blending."
                    )
            else:
                # NO KALSHI DATA - Explain why
                st.markdown("### 📊 Kalshi Prediction Market Status:")
                legs = row.get('legs', [])
                scopes = [leg.get('kalshi_validation', {}).get('market_scope') for leg in legs]
                unsupported_labels = [
                    leg.get('label')
                    for leg in legs
                    if leg.get('kalshi_validation', {}).get('market_scope') in {
                        'total_market', 'unsupported_market', 'totals_not_supported'
                    }
                ]
                error_labels = [
                    leg.get('label')
                    for leg in legs
                    if leg.get('kalshi_validation', {}).get('market_scope') == 'error'
                ]
                not_initialized = any(scope == 'not_initialized' for scope in scopes)
                disabled = (not st.session_state.get('kalshi_enabled', False)) or any(scope == 'disabled' for scope in scopes)

                if disabled:
                    st.info("Kalshi validation is turned off. Toggle the Kalshi checkbox above to blend prediction markets into the analysis.")
                elif unsupported_labels:
                    st.info("Kalshi does not publish totals/prop markets, so these leg(s) rely on AI + sentiment only:")
                    for label in unsupported_labels:
                        st.caption(f"• {label}")
                    st.caption("Moneyline and spread legs will include Kalshi coverage whenever a market is available.")
                elif not_initialized:
                    st.info("Kalshi markets have not loaded yet. Add your Kalshi API key or retry to use the live/synthetic market data.")
                elif error_labels:
                    st.warning("Kalshi validation encountered an error for these legs (falling back to AI + sentiment):")
                    for label in error_labels:
                        st.caption(f"• {label}")
                else:
                    st.warning(f"""
                    **⚠️ No Kalshi Data Available for this Parlay** ({kalshi_legs_with_data}/{total_legs} legs)

                    **This means:**
                    - ✅ Analysis still uses AI + Sentiment (2 of 3 sources)
                    - ⚠️ Missing prediction market validation
                    - 🔄 Kalshi Factor = 1.0x (neutral, no impact)
                    - 📊 AI Score unchanged by Kalshi

                    **Why no data?**
                    - Kalshi doesn't have markets for these specific games
                    - Kalshi focuses on season-long outcomes (playoffs, championships)
                    - Individual game spreads/totals rarely have Kalshi markets

                    **What this means:**
                    - Bet based on AI + Sentiment confidence
                    - Higher risk without 3rd source validation
                    - Consider checking Tab 4 for available Kalshi markets

                    💡 **Tip:** For Kalshi validation, focus on season futures, playoff odds, or major championships.
                    """)

            live_data_legs_with_data = row.get('live_data_legs', row.get('apisports_legs', 0))
            live_data_factor = row.get('live_data_factor', row.get('apisports_factor', 1.0))
            live_data_boost = row.get('live_data_boost', row.get('apisports_boost', 0))
            apisports_legs = row.get('apisports_legs', 0)
            apisports_boost = row.get('apisports_boost', 0)
            sportsdata_legs = row.get('sportsdata_legs', 0)
            sportsdata_boost = row.get('sportsdata_boost', 0)
            live_data_sports = row.get('live_data_sports', row.get('apisports_sports', [])) or []

            sport_icon_lookup = {
                'americanfootball_nfl': '🏈',
                'basketball_nba': '🏀',
                'icehockey_nhl': '🏒',
            }

            if live_data_legs_with_data:
                st.markdown("### 🛰️ Live Data Influence (API-Sports + SportsData.io):")

                if live_data_sports:
                    icons = " ".join(
                        sport_icon_lookup.get(sport, '🛰️') for sport in sorted(set(live_data_sports))
                    )
                    st.caption(
                        f"Live data applied from: {icons} {', '.join(sorted(set(live_data_sports)))}"
                    )

                col_a1, col_a2, col_a3, col_a4, col_a5 = st.columns(5)

                with col_a1:
                    st.metric(
                        "Live Data Legs",
                        f"{live_data_legs_with_data}/{len(row.get('legs', []))}",
                        help="How many legs include live team context",
                    )

                with col_a2:
                    delta_color = "normal" if apisports_boost >= 0 else "inverse"
                    display_val = f"{apisports_boost:+.0f}" if apisports_legs else "—"
                    st.metric(
                        "API-Sports Points",
                        display_val,
                        delta=float(apisports_boost) if apisports_boost else None,
                        delta_color=delta_color,
                        help="Boost or penalty applied from API-Sports hot/cold team trends",
                    )

                with col_a3:
                    delta_color_sd = "normal" if sportsdata_boost >= 0 else "inverse"
                    display_sd = f"{sportsdata_boost:+.0f}" if sportsdata_legs else "—"
                    st.metric(
                        "SportsData.io Points",
                        display_sd,
                        delta=float(sportsdata_boost) if sportsdata_boost else None,
                        delta_color=delta_color_sd,
                        help="Power index and turnover margin boost from SportsData.io",
                    )

                with col_a4:
                    st.metric(
                        "Score Multiplier",
                        f"{live_data_factor:.2f}x",
                        delta=f"{(live_data_factor-1)*100:+.0f}%" if live_data_factor != 1.0 else None,
                        help="Overall adjustment to the AI score from live data feeds",
                    )

                with col_a5:
                    baseline = row['ai_score'] / live_data_factor if live_data_factor else row['ai_score']
                    live_delta = row['ai_score'] - baseline
                    st.metric(
                        "Score Impact",
                        f"{live_delta:+.1f} pts",
                        help="How many points live data added or removed",
                    )

                if live_data_factor >= 1.02:
                    st.success(
                        f"🟢 **Live data boosted this parlay by {(live_data_factor-1)*100:.0f}%** thanks to favorable team trends."
                    )
                elif live_data_factor <= 0.98:
                    st.warning(
                        f"🟠 **Live data reduced this parlay by {(1-live_data_factor)*100:.0f}%** due to cold or negative trends."
                    )
                else:
                    st.info("🟡 **Live data neutral** – trends across API-Sports and SportsData.io are balanced.")

                if apisports_legs:
                    st.caption(
                        f"API-Sports coverage: {apisports_legs} leg(s), points {apisports_boost:+.0f}"
                    )
                if sportsdata_legs:
                    st.caption(
                        f"SportsData.io coverage: {sportsdata_legs} leg(s), points {sportsdata_boost:+.0f}"
                    )
            else:
                st.markdown("### 🛰️ Live Data Status:")
                apisports_client = st.session_state.get('apisports_nfl_client')
                hockey_client = st.session_state.get('apisports_hockey_client')
                configured_apisports = any(
                    client and client.is_configured()
                    for client in (
                        apisports_client,
                        basketball_client,
                        hockey_client,
                    )
                )
                configured_sportsdata = any(
                    client and getattr(client, "is_configured", lambda: False)()
                    for client in sportsdata_clients.values()
                )
                if not (configured_apisports or configured_sportsdata):
                    st.info(
                        "Add your API-Sports and SportsData.io keys across the leagues you follow to blend live team trends into scoring."
                    )
                else:
                    st.info("Live data feeds are configured but no matching games were found for this parlay.")

        save_key_suffix = hashlib.sha1(title.encode('utf-8')).hexdigest()[:6] if isinstance(title, str) else 'parlay'

        if st.button(
            "📌 Save this parlay for tracking",
            key=f"save_parlay_{save_key_suffix}_{i}",
        ):
            tz_for_save = timezone_label or st.session_state.get('user_timezone', 'UTC') or 'UTC'
            success, message = save_parlay_for_tracking(row, title, i, tz_for_save)
            if success:
                st.success(message)
            else:
                st.info(message)

        st.caption("Saved parlays appear in the tracker above for next-day result checks.")

        # theover.ai boost info if available
        theover_bonus = row.get('theover_bonus', 0)
        theover_prob_count = row.get('theover_probability_count', 0)
        theover_avg_delta = row.get('theover_avg_delta', 0.0)

        if theover_prob_count:
            delta_pct = theover_avg_delta * 100
            legs_label = "leg" if theover_prob_count == 1 else "legs"
            base_msg = f"ML probabilities blended for {theover_prob_count} {legs_label} (avg Δ{delta_pct:+.1f}pp vs AI)."
            bonus_pct = theover_bonus * 100
            if bonus_pct > 0.1:
                st.success(
                    f"🎯 **theover.ai Boost:** +{bonus_pct:.0f}% to AI score — {base_msg}"
                )
            elif bonus_pct < -0.1:
                st.warning(
                    f"⚠️ **theover.ai Conflict:** {bonus_pct:.0f}% penalty — {base_msg}"
                )
            else:
                st.info(f"🎯 **theover.ai ML data applied:** {base_msg}")
        elif theover_bonus:
            bonus_pct = theover_bonus * 100
            if bonus_pct > 0:
                st.success(
                    f"🎯 **theover.ai Boost:** +{bonus_pct:.0f}% to AI score "
                    f"({row.get('theover_matches', 0)} matching picks)"
                )
            else:
                st.warning(
                    f"⚠️ **theover.ai Conflict:** {bonus_pct:.0f}% penalty "
                    f"({row.get('theover_conflicts', 0)} conflicting picks)"
                )

        # Market vs AI comparison
        st.markdown("**📊 Market vs AI Analysis:**")
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            st.write(f"Market EV: {row['ev_market']*100:.2f}%")
            st.write(f"Market Prob: {row['p']*100:.2f}%")
        with comp_col2:
            st.write(f"AI EV: {ai_ev_pct:.2f}%")
            st.write(f"AI Edge: {row['ai_edge']*100:.2f}%")

            # Show sentiment impact
            prob_diff = (row['p_ai'] - row['p']) * 100
            if abs(prob_diff) > 1:
                if prob_diff > 0:
                    st.success(f"↗️ Sentiment boosted by {prob_diff:.1f}%")
                else:
                    st.warning(f"↘️ Sentiment reduced by {abs(prob_diff):.1f}%")

        # Legs breakdown with theover.ai integration
        st.markdown("**🎯 Parlay Legs:**")
        legs_data = []
        leg_summaries: List[str] = []
        has_theover = False
        theover_matches = 0
        theover_conflicts = 0

        for j, leg in enumerate(row["legs"], start=1):
            # Summaries for quick leg glance
            label_text = leg.get('label') or leg.get('selection')
            if not label_text:
                home = leg.get('home_team')
                away = leg.get('away_team')
                if home and away:
                    label_text = f"{away} @ {home}"
            market_name = leg.get('market')
            if label_text and market_name:
                leg_summaries.append(f"- **{label_text}** ({market_name})")
            elif label_text:
                leg_summaries.append(f"- **{label_text}**")
            elif market_name:
                leg_summaries.append(f"- **{market_name}**")

            # Try to match with theover.ai data
            theover_result = leg.get('theover_match')
            dataset = _dataset_for_leg(leg)
            if theover_result is None and dataset is not None:
                try:
                    theover_result = match_theover_to_leg(leg, None, dataset)
                except Exception:
                    theover_result = None
                if theover_result:
                    leg['theover_match'] = theover_result

            theover_display = "—"
            theover_prob_display = '—'
            theover_delta_display = '—'

            if theover_result is not None:
                has_theover = True

                if isinstance(theover_result, dict):
                    pick = theover_result.get('pick', '')
                    matches = theover_result.get('matches')
                    signal = theover_result.get('signal', '🎯')

                    if matches is True:
                        theover_matches += 1
                    elif matches is False:
                        theover_conflicts += 1

                    probability = leg.get('theover_probability')
                    if probability is None:
                        probability = theover_result.get('model_probability') or theover_result.get('implied_probability')
                    delta = leg.get('theover_probability_delta')
                    if delta is None and isinstance(probability, (int, float)):
                        base_prob = leg.get('ai_prob_pre_theover', leg.get('ai_prob', leg.get('p')))
                        if isinstance(base_prob, (int, float)):
                            delta = probability - base_prob

                    if isinstance(probability, (int, float)):
                        prob_pct = probability * 100
                        theover_prob_display = f"{prob_pct:.1f}%"
                        if isinstance(delta, (int, float)):
                            theover_delta_display = f"{delta*100:+.1f}pp"
                            theover_display = f"{signal} {pick} ({prob_pct:.1f}% | Δ{delta*100:+.1f}pp)"
                        else:
                            theover_display = f"{signal} {pick} ({prob_pct:.1f}%)"
                    else:
                        theover_display = f"{signal} {pick}" if pick else signal
                else:
                    if isinstance(theover_result, (int, float)):
                        value = float(theover_result)
                        theover_display = f"🎯 {value:.2f}"
                        theover_prob_display = f"{value:.2f}"
                    else:
                        theover_display = f"🎯 {theover_result}"
            # Kalshi validation display
            kalshi_display = "—"
            kalshi_influence_display = ""

            if 'kalshi_validation' in leg:
                kv = leg['kalshi_validation']
                if kv.get('kalshi_available'):
                    kalshi_prob = kv.get('kalshi_prob', 0) * 100
                    validation = kv.get('validation', 'unavailable')
                    data_source = kv.get('data_source', 'kalshi')

                    source_prefix = "🧪 " if data_source and 'synthetic' in data_source else ""

                    if validation == 'confirms':
                        kalshi_display = f"{source_prefix}✅ {kalshi_prob:.1f}%"
                    elif validation == 'kalshi_higher':
                        kalshi_display = f"{source_prefix}📈 {kalshi_prob:.1f}%"
                    elif validation == 'strong_kalshi_higher':
                        kalshi_display = f"{source_prefix}🟢 {kalshi_prob:.1f}%"
                    elif validation == 'kalshi_lower':
                        kalshi_display = f"{source_prefix}📉 {kalshi_prob:.1f}%"
                    elif validation == 'strong_contradiction':
                        kalshi_display = f"{source_prefix}⚠️ {kalshi_prob:.1f}%"
                    else:
                        kalshi_display = f"{source_prefix}{kalshi_prob:.1f}%"

                    influence = leg.get('kalshi_influence')
                    if isinstance(influence, (int, float)):
                        influence_pct = influence * 100
                        kalshi_influence_display = f"{influence_pct:+.1f}%" if abs(influence_pct) > 0.1 else "—"
                    else:
                        kalshi_influence_display = "—"
                else:
                    kalshi_influence_display = "—"

            if not kalshi_influence_display:
                kalshi_influence_display = "—"

            apisports_display = "—"
            apisports_details = leg.get('apisports')
            if isinstance(apisports_details, dict) and apisports_details:
                parts = []
                sport_name = apisports_details.get('sport_name')
                if sport_name:
                    parts.append(f"{sport_name}")
                record = apisports_details.get('team_record')
                if record:
                    parts.append(f"Record {record}")
                trend = apisports_details.get('trend')
                if trend:
                    icon = {'hot': '🔥', 'cold': '🥶', 'neutral': '⚪️'}.get(trend, '📊')
                    parts.append(f"{icon} {trend.capitalize()}")
                avg_for = apisports_details.get('team_avg_points_for')
                metric_label = apisports_details.get('scoring_metric') or 'points'
                metric_text = 'pts'
                if isinstance(metric_label, str):
                    if metric_label.lower() in ('points', 'point'):
                        metric_text = 'pts'
                    else:
                        metric_text = metric_label.lower()
                if isinstance(avg_for, (int, float)):
                    parts.append(f"{avg_for:.1f} {metric_text} for")
                avg_against = apisports_details.get('team_avg_points_against')
                if isinstance(avg_against, (int, float)):
                    parts.append(f"{avg_against:.1f} {metric_text} allowed")
                status = apisports_details.get('status')
                if status:
                    parts.append(status)
                kickoff = apisports_details.get('kickoff')
                if kickoff:
                    parts.append(kickoff)
                apisports_display = " | ".join(parts) if parts else "Live data"

            sportsdata_display = "—"
            sportsdata_details = leg.get('sportsdata')
            if isinstance(sportsdata_details, dict) and sportsdata_details:
                sd_parts = []
                record_sd = sportsdata_details.get('team_record')
                if record_sd:
                    sd_parts.append(f"Record {record_sd}")
                streak_sd = sportsdata_details.get('streak')
                if streak_sd:
                    sd_parts.append(streak_sd)
                trend_sd = sportsdata_details.get('trend')
                if trend_sd:
                    icon_sd = {'hot': '🔥', 'cold': '🥶', 'neutral': '⚪️'}.get(trend_sd, '📊')
                    sd_parts.append(f"{icon_sd} {trend_sd.capitalize()}")
                net_ppg = sportsdata_details.get('net_points_per_game')
                if isinstance(net_ppg, (int, float)):
                    sd_parts.append(f"Net {net_ppg:+.1f} ppg")
                turnover_sd = sportsdata_details.get('turnover_margin')
                if isinstance(turnover_sd, (int, float)):
                    sd_parts.append(f"TO {turnover_sd:+.1f}")
                strength_delta_sd = sportsdata_details.get('strength_delta')
                if strength_delta_sd is None:
                    strength_delta_sd = sportsdata_details.get('combined_strength_delta')
                if isinstance(strength_delta_sd, (int, float)):
                    sd_parts.append(f"Edge {strength_delta_sd:+.1f}")
                sportsdata_display = " | ".join(sd_parts) if sd_parts else "SportsData.io"

            model_used = leg.get('ai_model_source')
            if isinstance(model_used, str) and model_used:
                if model_used.startswith('historical-ensemble'):
                    suffix = model_used[len('historical-ensemble'):].lstrip('-')
                    model_display = f"Historical Ensemble ({suffix.upper()})" if suffix else 'Historical Ensemble'
                elif model_used.startswith('historical-logistic'):
                    suffix = model_used[len('historical-logistic'):].lstrip('-')
                    model_display = f"Historical Logistic ({suffix.upper()})" if suffix else 'Historical Logistic'
                else:
                    model_display = model_used.replace('-', ' ').title()
            else:
                model_display = '—'

            component_display = '—'
            component_payload = leg.get('ai_component_probabilities')
            if isinstance(component_payload, dict) and component_payload:
                breakdown = []
                for key, val in sorted(component_payload.items()):
                    try:
                        breakdown.append(f"{key.replace('_', ' ').title()}: {float(val)*100:.1f}%")
                    except Exception:
                        continue
                if breakdown:
                    component_display = "; ".join(breakdown)

            training_rows_val = leg.get('ai_training_rows')
            if isinstance(training_rows_val, (int, float)) and training_rows_val:
                training_display = f"{int(training_rows_val)}"
            else:
                training_display = '—'

            ai_pre_kalshi = leg.get('ai_prob_before_kalshi')
            if isinstance(ai_pre_kalshi, (int, float)):
                ai_pre_display = f"{ai_pre_kalshi*100:.1f}%"
            else:
                ai_pre_display = '—'

            alignment_delta = leg.get('kalshi_alignment_delta')
            if isinstance(alignment_delta, (int, float)):
                alignment_display = f"{alignment_delta*100:+.1f}pp"
            else:
                alignment_display = '—'

            leg_entry = {
                "Leg": j,
                "Type": leg.get("market", "—"),
                "Selection": leg.get("label", label_text or "—"),
                "Odds": f"{leg['d']:.3f}",
                "Market %": f"{leg['p']*100:.1f}%",
                "AI % (pre-Kalshi)": ai_pre_display,
                "AI % (final)": f"{leg.get('ai_prob', leg['p'])*100:.1f}%",
                "ML Model": model_display,
                "ML Breakdown": component_display,
                "Training Rows": training_display,
                "Kalshi": kalshi_display,
                "K Impact": kalshi_influence_display,
                "Kalshi vs ML": alignment_display,
                "Sentiment": leg.get('sentiment_trend', 'N/A'),
                "API-Sports": apisports_display,
                "SportsData.io": sportsdata_display,
                "theover.ai": theover_display,
                "theover.ai %": theover_prob_display,
                "theover Δ": theover_delta_display,
            }
            legs_data.append(leg_entry)

            # Persist common displays for downstream tables (e.g., Kalshi panel)
            leg['ai_model_display'] = model_display
            leg['ai_component_display'] = component_display
            leg['ai_training_display'] = training_display
            leg['ai_pre_prob_display'] = ai_pre_display

        if leg_summaries:
            st.markdown("**📝 Selected Legs:**")
            st.markdown("\n".join(leg_summaries))

        st.dataframe(pd.DataFrame(legs_data), use_container_width=True, hide_index=True)

        if has_theover:
            st.caption(
                "**theover.ai %** = ML probability from theover.ai; **theover Δ** = percentage-point change versus the AI model before blending."
            )

        if any(leg.get('ai_model_source') for leg in row.get("legs", [])):
            st.caption(
                "**ML Model** = Historical ML blends logistic regression with gradient boosting when available;"
                " **ML Breakdown** = component probabilities from each estimator;"
                " **Training Rows** = number of historical games in the most recent fit."
            )

        if any(leg.get('kalshi_validation', {}).get('kalshi_available') for leg in row.get("legs", [])):
            st.caption(
                "**AI % (pre-Kalshi)** = model probability before blending with Kalshi; "
                "**Kalshi vs ML** = percentage-point gap between Kalshi pricing and the model."
            )

        # Kalshi impact legend
        if any(leg.get('kalshi_validation', {}).get('kalshi_available') for leg in row.get("legs", [])):
            st.caption("**K Impact** = How much Kalshi adjusted AI probability (blended 50% AI + 30% Kalshi + 20% Market)")
        if any(leg.get('apisports') for leg in row.get("legs", [])):
            st.caption("**API-Sports** = Live NFL/NHL data (records, form, kickoff) from api-sports.io")
        if any(leg.get('sportsdata') for leg in row.get("legs", [])):
            st.caption("**SportsData.io** = NFL power index, turnover margin, and streak context")

        # Show legend and summary
        if has_theover:
            col_legend1, col_legend2 = st.columns(2)
            with col_legend1:
                st.caption("✅ = Matches theover.ai pick | ⚠️ = Conflicts with theover.ai | 🎯 = theover.ai data available")
            with col_legend2:
                if theover_matches > 0:
                    st.success(f"✅ {theover_matches} leg(s) match theover.ai recommendations")
                if theover_conflicts > 0:
                    st.warning(f"⚠️ {theover_conflicts} leg(s) conflict with theover.ai recommendations")
        
        # Kalshi validation legend and detailed influence
        kalshi_available = sum(1 for leg in row.get("legs", []) if leg.get('kalshi_validation', {}).get('kalshi_available', False))
        
        # Show info if Kalshi enabled but no data found
        if 'kalshi_validation' in row.get("legs", [{}])[0] and kalshi_available == 0:
            st.info("""
            **📊 Kalshi Validation Enabled** but no matching markets found for these games.
            
            **This is normal because:**
            - Kalshi may not have markets for all games
            - Markets might be for season-long outcomes (playoffs, championships) rather than individual games
            - Try checking Tab 4 to see what Kalshi markets are actually available
            
            **Note:** Parlay analysis still uses AI + Sentiment, just without Kalshi cross-validation.
            """)
        
        if kalshi_available > 0:
            kalshi_confirmed = sum(1 for leg in row.get("legs", []) if leg.get('kalshi_validation', {}).get('validation') == 'confirms')
            kalshi_higher = sum(1 for leg in row.get("legs", []) if 'higher' in leg.get('kalshi_validation', {}).get('validation', ''))
            kalshi_contradicts = sum(1 for leg in row.get("legs", []) if 'contradiction' in leg.get('kalshi_validation', {}).get('validation', ''))
            
            st.markdown("---")
            st.markdown("**📊 Kalshi Prediction Market Validation:**")
            
            col_k1, col_k2 = st.columns(2)
            with col_k1:
                st.caption("**Legend:** ✅ = Confirms | 📈 = Kalshi higher | 📉 = Kalshi lower | 🟢 = Strong value | ⚠️ = Contradiction")
            with col_k2:
                if kalshi_confirmed > 0:
                    st.success(f"✅ {kalshi_confirmed} leg(s) confirmed by Kalshi")
                if kalshi_higher > 0:
                    st.info(f"📈 {kalshi_higher} leg(s) show Kalshi value")
                if kalshi_contradicts > 0:
                    st.warning(f"⚠️ {kalshi_contradicts} leg(s) contradicted by Kalshi")
            
            # Detailed Kalshi Influence Analysis
            st.markdown("**🔍 How Kalshi Influenced This Parlay:**")
            
            total_confidence_boost = 0
            total_kalshi_edge = 0
            kalshi_details = []
            
            for j, leg in enumerate(row.get("legs", []), start=1):
                kv = leg.get('kalshi_validation', {})
                if kv.get('kalshi_available'):
                    kalshi_prob = kv.get('kalshi_prob', 0)
                    validation = kv.get('validation', 'unavailable')
                    confidence_boost = kv.get('confidence_boost', 0)
                    edge = kv.get('edge', 0)
                    market_ticker = kv.get('market_ticker', 'N/A')
                    
                    total_confidence_boost += confidence_boost
                    total_kalshi_edge += edge
                    
                    # Create status icon
                    if validation == 'confirms':
                        status_icon = "✅"
                        status_text = "CONFIRMS"
                        status_color = "green"
                    elif validation == 'strong_kalshi_higher':
                        status_icon = "🟢"
                        status_text = "STRONG VALUE"
                        status_color = "green"
                    elif 'higher' in validation:
                        status_icon = "📈"
                        status_text = "KALSHI HIGHER"
                        status_color = "blue"
                    elif 'contradiction' in validation:
                        status_icon = "⚠️"
                        status_text = "CONTRADICTION"
                        status_color = "red"
                    else:
                        status_icon = "📉"
                        status_text = "KALSHI LOWER"
                        status_color = "orange"
                    
                    sportsbook_prob = leg.get('p', 0) * 100
                    kalshi_prob_pct = kalshi_prob * 100
                    discrepancy = abs(kalshi_prob - leg.get('p', 0)) * 100
                    
                    model_display = leg.get('ai_model_display')
                    if not model_display:
                        model_used = leg.get('ai_model_source')
                        if isinstance(model_used, str) and model_used:
                            if model_used.startswith('historical-ensemble'):
                                suffix = model_used[len('historical-ensemble'):].lstrip('-')
                                model_display = f"Historical Ensemble ({suffix.upper()})" if suffix else 'Historical Ensemble'
                            elif model_used.startswith('historical-logistic'):
                                suffix = model_used[len('historical-logistic'):].lstrip('-')
                                model_display = f"Historical Logistic ({suffix.upper()})" if suffix else 'Historical Logistic'
                            else:
                                model_display = model_used.replace('-', ' ').title()
                        else:
                            model_display = '—'

                    ml_prob_pre = leg.get('ai_prob_before_kalshi')
                    if isinstance(ml_prob_pre, (int, float)):
                        ml_prob_display = f"{ml_prob_pre*100:.1f}%"
                    else:
                        ml_prob_display = leg.get('ai_pre_prob_display', "—")

                    alignment_delta = leg.get('kalshi_alignment_delta')
                    if isinstance(alignment_delta, (int, float)):
                        alignment_display = f"{alignment_delta*100:+.1f}pp"
                    else:
                        alignment_display = "—"

                    kalshi_details.append({
                        'Leg': j,
                        'Pick': leg.get('team', 'N/A'),
                        'Status': f"{status_icon} {status_text}",
                        'Sportsbook': f"{sportsbook_prob:.1f}%",
                        'AI % (pre-Kalshi)': ml_prob_display,
                        'Kalshi': f"{kalshi_prob_pct:.1f}%",
                        'Kalshi vs ML': alignment_display,
                        'Discrepancy vs Market': f"{discrepancy:.1f}%",
                        'Confidence Boost': f"{confidence_boost*100:+.0f}%",
                        'Edge': f"{edge*100:+.1f}%",
                        'ML Model': model_display,
                        'ML Breakdown': leg.get('ai_component_display', '—'),
                        'Training Rows': leg.get('ai_training_display', '—'),
                        'Market': market_ticker[:20]
                    })
            
            if kalshi_details:
                st.dataframe(pd.DataFrame(kalshi_details), use_container_width=True, hide_index=True)
                
                # Summary metrics
                st.markdown("**📈 Kalshi Impact Summary:**")
                col_impact1, col_impact2, col_impact3, col_impact4 = st.columns(4)
                
                with col_impact1:
                    st.metric(
                        "Legs Validated",
                        f"{kalshi_available}/{len(row.get('legs', []))}",
                        help="How many legs have Kalshi market data"
                    )
                
                with col_impact2:
                    st.metric(
                        "Total Confidence Boost",
                        f"{total_confidence_boost*100:+.0f}%",
                        help="How much Kalshi boosted overall confidence"
                    )
                
                with col_impact3:
                    st.metric(
                        "Additional Edge",
                        f"{total_kalshi_edge*100:+.1f}%",
                        help="Extra edge identified from Kalshi vs sportsbook"
                    )
                
                with col_impact4:
                    avg_discrepancy = sum(abs(leg.get('kalshi_validation', {}).get('discrepancy', 0)) 
                                        for leg in row.get("legs", []) 
                                        if leg.get('kalshi_validation', {}).get('kalshi_available')) / max(kalshi_available, 1)
                    st.metric(
                        "Avg Discrepancy",
                        f"{avg_discrepancy*100:.1f}%",
                        help="Average difference between Kalshi and sportsbook"
                    )
                
                # Interpretation
                st.markdown("**💡 Interpretation:**")
                
                if total_confidence_boost >= 0.15:
                    st.success("🟢 **STRONG KALSHI CONFIRMATION** - All sources strongly agree. High confidence bet!")
                elif total_confidence_boost >= 0.05:
                    st.info("🟡 **MODERATE CONFIRMATION** - Kalshi generally agrees. Good bet with decent validation.")
                elif total_confidence_boost >= -0.05:
                    st.warning("🟠 **NEUTRAL VALIDATION** - Kalshi shows mixed signals. Proceed with caution.")
                else:
                    st.error("🔴 **KALSHI DISAGREES** - Prediction market contradicts this parlay. Consider skipping or investigating further.")
                
                if total_kalshi_edge > 0.10:
                    st.success(f"💰 **VALUE DETECTED**: Kalshi shows {total_kalshi_edge*100:.1f}% additional edge! This parlay may be underpriced by sportsbooks.")
                elif total_kalshi_edge < -0.10:
                    st.warning(f"⚠️ **OVERPRICED WARNING**: Kalshi thinks this parlay is overpriced. Sportsbooks may be offering poor value.")
                
                # Recommendation based on Kalshi
                kalshi_score = (total_confidence_boost * 50) + (total_kalshi_edge * 30) + (kalshi_confirmed / max(kalshi_available, 1) * 20)
                
                st.markdown("**🎯 Kalshi-Based Recommendation:**")
                if kalshi_score > 15:
                    st.success("✅ **KALSHI APPROVES** - Strong validation from prediction markets. Excellent bet!")
                elif kalshi_score > 5:
                    st.info("🟡 **KALSHI CAUTIOUS** - Some validation but mixed signals. Decent bet if AI score is high.")
                else:
                    st.warning("⚠️ **KALSHI SKEPTICAL** - Prediction market doesn't support this parlay. Bet with caution or skip.")
        
            # Betting scenarios
            st.markdown("**💵 Betting Scenarios:**")
            for stake in [50, 100, 250, 500]:
                profit_amt = calculate_profit(row['d'], stake)
                payout = stake + profit_amt
                exp_return = stake * (1 + row['ev_ai'])
                st.write(f"${stake} bet → ${payout:.2f} payout | Expected return: ${exp_return:.2f}")
            
            # CSV export
            csv_buf = io.StringIO()
            df_export = pd.DataFrame(legs_data)
            df_export.to_csv(csv_buf, index=False)
            st.download_button(
                "💾 Download CSV",
                data=csv_buf.getvalue(),
                file_name=f"ai_parlay_{i}.csv",
                mime="text/csv",
                key=f"download_ai_{title}_{i}"
            )

# ============ STREAMLIT UI ============
st.set_page_config(page_title=APP_CFG["title"], layout="wide")
st.title("🤖 " + APP_CFG["title"])
st.caption("AI-powered parlay finder with sentiment analysis and machine learning predictions")

# Initialize AI components
if 'sentiment_analyzer' not in st.session_state:
    news_key = os.environ.get("NEWS_API_KEY", "")
    st.session_state['sentiment_analyzer'] = RealSentimentAnalyzer(news_key)
    st.session_state['news_api_key'] = news_key
sidebar_state = render_sidebar_controls()
tz = sidebar_state["tz"]
sel_date = sidebar_state["selected_date"]
_day_window = sidebar_state["day_window"]
sports = sidebar_state["sports"]
active_sports_list = sports if sports else APP_CFG["sports_common"]
active_sport_keys = set(active_sports_list)
use_sentiment = sidebar_state["use_sentiment"]
use_ml_predictions = sidebar_state["use_ml_predictions"]
min_ai_confidence = sidebar_state["min_ai_confidence"]
min_parlay_probability = sidebar_state["min_parlay_probability"]
max_parlay_probability = sidebar_state["max_parlay_probability"]

# Manage historical ML components lazily so resource-heavy datasets are only
# built when machine-learning predictions are enabled.
builder_error = st.session_state.get('historical_builder_error')
if use_ml_predictions:
    builder = st.session_state.get('historical_data_builder')
    if builder is None:
        try:
            builder = HistoricalDataBuilder(
                resolve_odds_api_key,
                days_back=120,
                max_days_back=540,
                min_rows_target=30,
            )
            st.session_state['historical_data_builder'] = builder
            st.session_state.pop('historical_builder_error', None)
            builder_error = None
        except TypeError as builder_init_error:  # pragma: no cover - defensive guard
            logger.exception("Failed to initialize HistoricalDataBuilder", exc_info=True)
            builder = HistoricalDataBuilder(resolve_odds_api_key)
            st.session_state['historical_data_builder'] = builder
            st.session_state['historical_builder_error'] = str(builder_init_error)
            builder_error = str(builder_init_error)

    if st.session_state.get('ml_predictor') is None and builder is not None:
        st.session_state['ml_predictor'] = HistoricalMLPredictor(builder)
else:
    builder = st.session_state.get('historical_data_builder')
    if builder and hasattr(builder, 'reset_cache'):
        try:
            builder.reset_cache()
        except Exception:  # pragma: no cover - defensive cache clear
            logger.debug("Failed to reset historical dataset cache", exc_info=True)
    st.session_state.pop('ml_predictor', None)
    builder_error = None
    st.session_state['historical_builder_error'] = None

ml_predictor_state = st.session_state.get('ml_predictor')
ai_optimizer = st.session_state.get('ai_optimizer')
if (
    ai_optimizer is None
    or getattr(ai_optimizer, 'ml', None) is not ml_predictor_state
    or getattr(ai_optimizer, 'sentiment', None) is not st.session_state['sentiment_analyzer']
):
    st.session_state['ai_optimizer'] = AIOptimizer(
        st.session_state['sentiment_analyzer'],
        ml_predictor_state,
    )

# Initialize advanced analyzers
if 'sharp_detector' not in st.session_state:
    st.session_state['sharp_detector'] = SharpMoneyDetector()
if 'player_impact' not in st.session_state:
    st.session_state['player_impact'] = PlayerImpactAnalyzer()
if 'weather_analyzer' not in st.session_state:
    weather_key = os.environ.get("WEATHER_API_KEY", "")
    st.session_state['weather_analyzer'] = WeatherAnalyzer(weather_key)
if 'kelly_calculator' not in st.session_state:
    st.session_state['kelly_calculator'] = KellyCalculator()
if 'matchup_analyzer' not in st.session_state:
    st.session_state['matchup_analyzer'] = MatchupAnalyzer()
if 'advanced_stats' not in st.session_state:
    st.session_state['advanced_stats'] = AdvancedStatsIntegrator()
if 'social_analyzer' not in st.session_state:
    twitter_key = os.environ.get("TWITTER_API_KEY", "")
    st.session_state['social_analyzer'] = SocialMediaAnalyzer(twitter_key)
if 'kalshi_integrator' not in st.session_state:
    kalshi_key = os.environ.get("KALSHI_API_KEY", "")
    kalshi_secret = os.environ.get("KALSHI_API_SECRET", "")
    st.session_state['kalshi_integrator'] = KalshiIntegrator(kalshi_key, kalshi_secret)
if 'apisports_nfl_client' not in st.session_state:
    stored_key = st.session_state.get('nfl_apisports_api_key')
    stored_source = st.session_state.get('nfl_apisports_key_source')
    if not stored_key:
        stored_key, stored_source = resolve_nfl_apisports_key()
        st.session_state['nfl_apisports_api_key'] = stored_key
        st.session_state['nfl_apisports_key_source'] = stored_source
    st.session_state['apisports_nfl_client'] = APISportsFootballClient(
        stored_key or None,
        key_source=stored_source,
    )
elif 'nfl_apisports_api_key' not in st.session_state:
    apisports_client = st.session_state.get('apisports_nfl_client')
    st.session_state['nfl_apisports_api_key'] = (
        apisports_client.api_key if apisports_client else ""
    )
if 'apisports_hockey_client' not in st.session_state:
    stored_hockey_key = st.session_state.get('nhl_apisports_api_key')
    stored_hockey_source = st.session_state.get('nhl_apisports_key_source')
    if not stored_hockey_key:
        stored_hockey_key, stored_hockey_source = resolve_nhl_apisports_key()
        st.session_state['nhl_apisports_api_key'] = stored_hockey_key
        st.session_state['nhl_apisports_key_source'] = stored_hockey_source
    st.session_state['apisports_hockey_client'] = APISportsHockeyClient(
        stored_hockey_key or None,
        key_source=stored_hockey_source,
    )
elif 'nhl_apisports_api_key' not in st.session_state:
    hockey_client = st.session_state.get('apisports_hockey_client')
    st.session_state['nhl_apisports_api_key'] = (
        hockey_client.api_key if hockey_client else ""
    )

if 'apisports_basketball_client' not in st.session_state:
    stored_nba_key = st.session_state.get('nba_apisports_api_key')
    stored_nba_source = st.session_state.get('nba_apisports_key_source')
    if not stored_nba_key:
        stored_nba_key, stored_nba_source = resolve_nba_apisports_key()
        st.session_state['nba_apisports_api_key'] = stored_nba_key
        st.session_state['nba_apisports_key_source'] = stored_nba_source
    st.session_state['apisports_basketball_client'] = APISportsBasketballClient(
        stored_nba_key or None,
        key_source=stored_nba_source,
    )
elif 'nba_apisports_api_key' not in st.session_state:
    basketball_client = st.session_state.get('apisports_basketball_client')
    st.session_state['nba_apisports_api_key'] = (
        basketball_client.api_key if basketball_client else ""
    )

# Main navigation tabs (fallback to containers if tabs are unavailable)
tab_labels = [
    "🎯 Sports Betting Parlays",
    "🔍 Sentiment & AI Analysis",
    "🎨 Custom Parlay Builder",
    "📊 Kalshi Prediction Markets",
    "🛰️ API-Sports Live Data",
]
tabs = []
try:
    potential_tabs = st.tabs(tab_labels)
    if len(potential_tabs) != len(tab_labels):
        raise ValueError("Streamlit returned an unexpected number of tabs")
    tabs = list(potential_tabs)
except Exception as tab_error:  # pragma: no cover - defensive fallback
    st.warning(
        "Tab layout is unavailable in this Streamlit runtime."
        " Showing sections sequentially instead.",
        icon="⚠️",
    )
    tabs = [st.container() for _ in tab_labels]
    logger.debug(
        "Falling back to container-based layout for tabs: %s",
        tab_error,
        exc_info=True,
    )

if len(tabs) != len(tab_labels):  # pragma: no cover - ultra-defensive guard
    tabs = [st.container() for _ in tab_labels]

main_tab1, main_tab2, main_tab3, main_tab4, main_tab5 = tabs

# ===== TAB 1: SPORTS BETTING PARLAYS =====
with main_tab1:
    apisports_client = st.session_state.get('apisports_nfl_client')
    nfl_key = st.session_state.get('nfl_apisports_api_key', "")
    nfl_source = st.session_state.get('nfl_apisports_key_source')
    if apisports_client is None:
        apisports_client = APISportsFootballClient(
            nfl_key or None,
            key_source=nfl_source,
        )
        st.session_state['apisports_nfl_client'] = apisports_client
    else:
        if apisports_client.api_key != (nfl_key or ""):
            apisports_client.update_api_key(nfl_key or None, source=nfl_source or "user")

    sportsdata_clients = ensure_sportsdata_clients()

    hockey_client = st.session_state.get('apisports_hockey_client')
    nhl_key = st.session_state.get('nhl_apisports_api_key', "")
    nhl_source = st.session_state.get('nhl_apisports_key_source')
    if hockey_client is None:
        hockey_client = APISportsHockeyClient(
            nhl_key or None,
            key_source=nhl_source,
        )
        st.session_state['apisports_hockey_client'] = hockey_client
    else:
        if hockey_client.api_key != (nhl_key or ""):
            hockey_client.update_api_key(nhl_key or None, source=nhl_source or "user")

    basketball_client = st.session_state.get('apisports_basketball_client')
    nba_key = st.session_state.get('nba_apisports_api_key', "")
    nba_source = st.session_state.get('nba_apisports_key_source')
    if basketball_client is None:
        basketball_client = APISportsBasketballClient(
            nba_key or None,
            key_source=nba_source,
        )
        st.session_state['apisports_basketball_client'] = basketball_client
    else:
        if basketball_client.api_key != (nba_key or ""):
            basketball_client.update_api_key(nba_key or None, source=nba_source or "user")

    ml_predictor = st.session_state.get('ml_predictor')
    if ml_predictor and use_ml_predictions:
        def _register_sportsdata(sport_key: str) -> None:
            client = sportsdata_clients.get(sport_key)
            if client is not None:
                ml_predictor.register_sportsdata_client(sport_key, client)

        if 'americanfootball_nfl' in active_sport_keys:
            ml_predictor.register_client('americanfootball_nfl', apisports_client)
        if 'icehockey_nhl' in active_sport_keys:
            ml_predictor.register_client('icehockey_nhl', hockey_client)
        if 'basketball_nba' in active_sport_keys:
            ml_predictor.register_client('basketball_nba', basketball_client)

        # SportsData.io-driven leagues (including college football) feed the ML datasets
        for sd_sport in (
            'americanfootball_nfl',
            'icehockey_nhl',
            'basketball_nba',
            'americanfootball_ncaaf',
            'basketball_ncaab',
        ):
            _register_sportsdata(sd_sport)

    # Quick configuration summary to reinforce sidebar selections
    config_cols = st.columns(3)
    with config_cols[0]:
        st.metric("Odds API", "Configured" if st.session_state.get('api_key') else "Missing")
    with config_cols[1]:
        st.metric("Sentiment", "Live" if st.session_state.get('news_api_key') else "Neutral")
    with config_cols[2]:
        timezone_label = st.session_state.get('user_timezone', 'UTC')
        st.metric("Timezone", timezone_label)

    st.markdown("---")

    # theover.ai Integration Section
    st.markdown("### 📊 theover.ai Integration (Optional)")
    st.caption("Upload separate datasets for ML picks and totals so each leg type can match correctly.")

    def _collect_theover_dataset(title: str, key_prefix: str) -> Optional[pd.DataFrame]:
        st.markdown(title)
        st.caption(
            "Include `League`, `Away Team`, and `Home Team` columns so matchups align across datasets."
        )

        dataset: Optional[pd.DataFrame] = None

        uploaded_file = st.file_uploader(
            "Upload theover.ai CSV export",
            type=["csv"],
            key=f"{key_prefix}_upload",
        )
        if uploaded_file is not None:
            try:
                dataset = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(dataset)} rows from theover.ai")
                with st.expander("📋 Preview uploaded data", expanded=False):
                    st.dataframe(dataset.head(10), use_container_width=True)
            except Exception as exc:
                st.error(f"Error loading CSV: {exc}")

        with st.expander("📋 Or paste theover.ai data", expanded=False):
            st.info(
                """
                **Paste Format (comma or tab-separated)**
                ```
                League,AwayTeam,HomeTeam,Pick,WinProbability
                NHL,Maple Leafs,Canadiens,Over,0.57
                ```
                """
            )
            pasted_data = st.text_area(
                "Paste data here",
                height=150,
                key=f"{key_prefix}_paste",
            )
            if dataset is None and pasted_data.strip():
                try:
                    from io import StringIO

                    if "\t" in pasted_data and "," not in pasted_data:
                        dataset = pd.read_csv(StringIO(pasted_data), sep="\t")
                    else:
                        dataset = pd.read_csv(StringIO(pasted_data))

                    st.success(f"✅ Loaded {len(dataset)} rows from pasted theover.ai data")
                    st.dataframe(dataset.head(10), use_container_width=True)
                except Exception as exc:
                    st.error(f"Error parsing data: {exc}")

        return dataset

    theover_ml_data = _collect_theover_dataset("#### 🤖 Moneyline ML projections", "theover_ml")
    theover_spreads_data = _collect_theover_dataset("#### 📐 Spread projections", "theover_spreads")
    theover_totals_data = _collect_theover_dataset("#### 📈 Totals (Over/Under) projections", "theover_totals")
    
    st.markdown("---")

    st.caption(
        "AI filters applied: sentiment {sentiment_state}, ML {ml_state}, confidence ≥ {conf:.0%}, parlay probability {min_prob:.0%}-{max_prob:.0%}".format(
            sentiment_state="on" if use_sentiment else "off",
            ml_state="on" if use_ml_predictions else "off",
            conf=min_ai_confidence,
            min_prob=min_parlay_probability,
            max_prob=max_parlay_probability,
        )
    )

    builder = st.session_state.get('historical_data_builder')
    builder_error = st.session_state.get('historical_builder_error')
    if builder_error:
        st.error(
            "Historical dataset builder unavailable: "
            f"{builder_error}. Using default settings until resolved."
        )
    ml_predictor_state = st.session_state.get('ml_predictor')
    if not use_ml_predictions:
        st.info(
            "Machine-learning predictions are turned off. Re-enable them in the sidebar when you're ready to blend historical models."
        )
    elif builder and ml_predictor_state:
        st.markdown("#### 🤖 Historical ML Training Status")
        show_training = st.checkbox(
            "Show ML training diagnostics (may trigger large API downloads)",
            key="show_ml_training_status",
            help="Enabling this fetches API-Sports history to update the historical model status."
        )
        if not show_training:
            st.info(
                "Enable the checkbox above to refresh ML training metrics only when you need them."
            )
        else:
            ml_capable_rows = [
                {
                    "label": "NFL",
                    "sport_key": "americanfootball_nfl",
                    "icon": "🏈",
                    "apisports_client": apisports_client,
                    "sportsdata_client": sportsdata_clients.get('americanfootball_nfl'),
                    "no_key_message": "Provide an NFL API-Sports key to enable historical ML training.",
                },
                {
                    "label": "NBA",
                    "sport_key": "basketball_nba",
                    "icon": "🏀",
                    "apisports_client": basketball_client,
                    "sportsdata_client": sportsdata_clients.get('basketball_nba'),
                    "no_key_message": "Provide an NBA API-Sports key to enable historical ML training.",
                },
                {
                    "label": "NHL",
                    "sport_key": "icehockey_nhl",
                    "icon": "🏒",
                    "apisports_client": hockey_client,
                    "sportsdata_client": sportsdata_clients.get('icehockey_nhl'),
                    "no_key_message": "Provide an NHL API-Sports key to enable historical ML training.",
                },
                {
                    "label": "NCAAF",
                    "sport_key": "americanfootball_ncaaf",
                    "icon": "🎓🏈",
                    "apisports_client": None,
                    "sportsdata_client": sportsdata_clients.get('americanfootball_ncaaf'),
                    "no_key_message": "Provide your SportsData.io NCAAF key to enable historical ML training.",
                },
            ]
            active_ml_rows = [
                row for row in ml_capable_rows if row["sport_key"] in active_sport_keys
            ]

            if not active_ml_rows:
                st.info("Select an NFL, NBA, NHL, or NCAAF sport to enable historical ML training.")
            else:
                status_cols = st.columns(min(2, len(active_ml_rows)))

                for idx, row_info in enumerate(active_ml_rows):
                    sport_label = row_info["label"]
                    sport_key = row_info["sport_key"]
                    sport_icon = row_info["icon"]
                    apisports_client_for_row = row_info.get("apisports_client")
                    sportsdata_client_for_row = row_info.get("sportsdata_client")
                    fallback_message = row_info.get("no_key_message") or "Provide an API key to enable historical ML training."

                    with status_cols[idx % len(status_cols)]:
                        st.markdown(f"**{sport_icon} {sport_label} Historical Model**")

                        default_metadata = {
                            "sport_key": sport_key,
                            "dataset_rows": 0,
                            "training_rows": 0,
                            "model_ready": False,
                            "last_dataset_build": None,
                            "last_trained": None,
                            "min_rows": getattr(ml_predictor_state, "min_rows", 25),
                            "error": None,
                            "model_engine": None,
                        }

                        metadata = default_metadata.copy()
                        if hasattr(ml_predictor_state, "training_metadata"):
                            try:
                                fetched_metadata = ml_predictor_state.training_metadata(sport_key) or {}
                                if isinstance(fetched_metadata, dict):
                                    metadata.update(fetched_metadata)
                                else:
                                    metadata["error"] = "invalid_metadata_payload"
                            except Exception as metadata_error:  # pragma: no cover - defensive guard
                                logger.exception(
                                    "Failed to load ML training metadata for %s", sport_key, exc_info=True
                                )
                                metadata["error"] = "metadata_unavailable"
                        else:
                            metadata["error"] = "predictor_missing"

                        st.metric(
                            "Historical games",
                            int(metadata.get('dataset_rows', 0)),
                            help="Joined API-Sports summaries with historical odds. Rebuilt every 6 hours.",
                        )
                        st.metric(
                            "Training rows used",
                            int(metadata.get('training_rows', 0)),
                            help="Rows consumed by the historical models during the last training run.",
                        )

                        engine_label = metadata.get('model_engine')
                        if isinstance(engine_label, str):
                            engine_lower = engine_label.lower()
                            if engine_lower == 'blend':
                                st.caption("Engine: scikit-learn ensemble (logistic + gradient boosting)")
                            elif engine_lower == 'logistic':
                                st.caption("Engine: scikit-learn logistic regression")
                            elif engine_lower == 'simple':
                                st.caption("Engine: NumPy logistic fallback")
                            else:
                                st.caption(f"Engine: {engine_label}")

                        rows_needed = int(metadata.get('min_rows_target') or metadata.get('min_rows', 0) or 0)
                        if metadata.get('model_ready'):
                            st.success("Model trained on recent history ✅")
                        else:
                            if rows_needed and metadata.get('dataset_rows', 0) < rows_needed:
                                st.warning(
                                    f"Collecting more games… need {rows_needed}+ rows for training.",
                                )
                            else:
                                st.info("Training will kick in once enough balanced outcomes are available.")

                        last_built = format_timestamp_utc(metadata.get('last_dataset_build'))
                        last_trained = format_timestamp_utc(metadata.get('last_trained'))
                        status_lines: List[str] = []
                        if last_built:
                            status_lines.append(f"Data refreshed: {last_built}")
                        if last_trained:
                            status_lines.append(f"Model trained: {last_trained}")
                        error_code = metadata.get('error')
                        if error_code and metadata.get('dataset_rows', 0) == 0:
                            friendly = {
                                'missing_api_key': 'Add your API key (API-Sports or SportsData.io) to fetch team history.',
                                'unregistered_client': 'Register this league with the ML builder.',
                                'games_fetch_failed': 'API-Sports schedule request failed. Retry shortly.',
                                'summary_build_failed': 'Could not assemble team summaries from API-Sports.',
                                'no_historical_rows': 'No completed games found in the selected window yet.',
                                'season_fetch_failed': 'Season backfill request failed. Retry after checking your API-Sports quota.',
                                'insufficient_rows': 'Need more completed games from API-Sports. Try expanding the season window.',
                                'metadata_unavailable': 'Model metadata is unavailable right now. Please retry shortly.',
                                'predictor_missing': 'Machine learning module is not ready. Refresh after initialization.',
                                'invalid_metadata_payload': 'Received unexpected metadata payload. Check server logs.',
                            }.get(error_code, error_code.replace('_', ' '))
                            st.error(friendly)
                        elif status_lines:
                            for line in status_lines:
                                st.caption(line)
                        else:
                            configured = False
                            for candidate_client in (
                                apisports_client_for_row,
                                sportsdata_client_for_row,
                            ):
                                if candidate_client and getattr(
                                    candidate_client, 'is_configured', lambda: False
                                )():
                                    configured = True
                                    break
                            if not configured:
                                st.info(fallback_message)

                        seasons = metadata.get('dataset_seasons') or metadata.get('seasons')
                        if seasons:
                            season_str = ", ".join(str(season) for season in seasons)
                            st.caption(f"Seasons in training set: {season_str}")
                        max_days = metadata.get('dataset_max_days_back')
                        if max_days:
                            st.caption(f"Historical lookback window: {int(max_days)} days")
                        backfills = metadata.get('season_backfills')
                        if backfills:
                            st.caption("Season backfills attempted: " + ", ".join(str(b) for b in backfills))

                        sample_rows = metadata.get('sample_rows_added')
                        if sample_rows:
                            real_rows = metadata.get('real_rows')
                            if isinstance(real_rows, (int, float)) and real_rows:
                                st.caption(
                                    f"Synthetic booster rows: {int(sample_rows)} (real games: {int(real_rows)})"
                                )
                            else:
                                st.caption(f"Synthetic booster rows added: {int(sample_rows)}")
    col3, col4, col5 = st.columns(3)
    with col3:
        per_sport_events = st.slider("Max events per sport", 3, 50, 12, 1)
    with col4:
        show_top = st.slider("Show top N combos", 1, 50, 15, 1)
    with col5:
        allow_sgp = st.checkbox("Allow same-game legs", value=False)
    
    st.info("💡 **Duplicate Removal:** Identical bets from different bookmakers are automatically deduplicated, keeping only the best odds.")

    # Bet type filters
    st.subheader("Include Bet Types")
    col_bet1, col_bet2, col_bet3 = st.columns(3)
    with col_bet1:
        inc_ml = st.checkbox("Moneyline", value=True)
    with col_bet2:
        inc_spread = st.checkbox("Spreads", value=True)
    with col_bet3:
        inc_total = st.checkbox("Totals (O/U)", value=True)

    # Kalshi Validation Option
    st.markdown("---")
    st.subheader("📊 Kalshi Prediction Market Validation")
    use_kalshi = st.checkbox(
        "✅ Validate with Kalshi (Prediction Market Cross-Check)",
        value=False,
        help="Compare sportsbook odds with Kalshi prediction markets to find discrepancies and boost confidence"
    )

    st.session_state['kalshi_enabled'] = use_kalshi

    if use_kalshi:
        st.info("""
        **Kalshi Validation Benefits:**
        - 🎯 Cross-validates odds with prediction markets
        - 📈 Boosts confidence when markets agree (up to +15%)
        - ⚠️ Flags contradictions when markets disagree
        - 💎 Identifies arbitrage opportunities
        - 🔍 Shows additional edge from market discrepancies
        
        **Important Note:** Kalshi typically has markets for:
        - Season-long outcomes (playoffs, championships, win totals)
        - Major events and marquee matchups
        - NOT every individual regular season game
        
        If you see "—" in the Kalshi column, it means no matching market was found for that specific game.
        This is normal and expected. The parlay analysis will still work using AI + Sentiment.
        
        **Tip:** Check Tab 4 to see what Kalshi markets are actually available right now.
        """)
        
        with st.expander("ℹ️ Understanding Kalshi Market Coverage"):
            st.markdown("""
            **What Kalshi Markets Are Available:**
            
            ✅ **Common Kalshi Markets:**
            - "Will [Team] make the playoffs?"
            - "Will [Team] win the Super Bowl/Championship?"
            - "Will [Team] win more than X games?"
            - "Will [Player] win MVP?"
            - Major rivalry games or primetime matchups
            
            ❌ **NOT Usually Available:**
            - Individual regular season game outcomes
            - Week-to-week game spreads
            - Game totals (over/under)
            - Every team's games
            
            **Why This Matters:**
            - Your parlay might have 0 Kalshi matches → That's OK!
            - Kalshi is a bonus validation source, not required
            - AI + Sentiment still provide strong analysis
            
            **When Kalshi DOES Match:**
            - You get extra confidence boost
            - Can spot market inefficiencies
            - Valuable cross-validation
            
            **Recommendation:**
            - Leave Kalshi validation ON (no harm if no matches)
            - When it finds matches, that's a bonus!
            - Don't expect matches for every game
            """)

    st.markdown("---")

    def render_api_sports_key_section(
        header: str,
        label: str,
        session_key: str,
        source_session_key: Optional[str],
        client,
        help_text: str,
        success_message: str,
        empty_message: str,
        widget_suffix: str,
    ) -> None:
        st.subheader(header)
        client_key = client.api_key if client else ""
        stored_value = st.session_state.get(session_key)
        if stored_value is None:
            stored_value = client_key or ""
            st.session_state[session_key] = stored_value

        if source_session_key and source_session_key not in st.session_state:
            origin = client.key_origin() if client else None
            st.session_state[source_session_key] = origin

        widget_key = f"{session_key}_{widget_suffix}"
        st.session_state.setdefault(widget_key, stored_value)

        new_value_raw = st.text_input(
            label,
            key=widget_key,
            type="password",
            help=help_text,
        )
        new_value = (new_value_raw or "").strip()
        if new_value != st.session_state.get(session_key, ""):
            st.session_state[session_key] = new_value
            if source_session_key:
                st.session_state[source_session_key] = "manual-entry" if new_value else None
            if client:
                client.update_api_key(new_value or None, source="manual-entry" if new_value else None)
            if new_value:
                st.success(success_message)
            else:
                st.info(empty_message)
        elif not st.session_state.get(session_key):
            st.caption(empty_message)

    render_api_sports_key_section(
        header="🏈 API-Sports NFL Data Integration",
        label="NFL API-Sports Key",
        session_key='nfl_apisports_api_key',
        source_session_key='nfl_apisports_key_source',
        client=apisports_client,
        help_text="Set the NFL_APISPORTS_API_KEY secret or request an NFL token from https://api-sports.io/",
        success_message="✅ NFL API-Sports key saved for this session.",
        empty_message="API-Sports integration disabled until an NFL key is provided.",
        widget_suffix="main",
    )

    render_api_sports_key_section(
        header="🏀 API-Sports NBA Data Integration",
        label="NBA API-Sports Key",
        session_key='nba_apisports_api_key',
        source_session_key='nba_apisports_key_source',
        client=basketball_client,
        help_text="Set the NBA_APISPORTS_API_KEY secret or request an NBA token from https://api-sports.io/",
        success_message="✅ NBA API-Sports key saved for this session.",
        empty_message="NBA live data disabled until an API-Sports key is provided.",
        widget_suffix="main",
    )

    def describe_key_origin(origin: Optional[str]) -> str:
        if not origin:
            return "no configured source"
        if origin.startswith("secret:"):
            return f"Streamlit secret `{origin.split(':', 1)[1]}`"
        if origin.startswith("env:"):
            return f"environment variable `{origin.split(':', 1)[1]}`"
        if origin == "manual-entry":
            return "manual entry"
        if origin == "runtime":
            return "runtime configuration"
        return origin

    if apisports_client and apisports_client.is_configured():
        st.caption(
            f"Using NFL API-Sports key from {describe_key_origin(apisports_client.key_origin())}."
        )
    else:
        st.caption("No NFL API-Sports key detected; live data calls will be skipped.")

    for sport_key, cfg in SPORTSDATA_CONFIG.items():
        client = sportsdata_clients.get(sport_key)
        if client and client.is_configured():
            st.caption(
                f"SportsData.io {cfg['label']} key from {describe_key_origin(client.key_origin())}."
            )
        else:
            st.caption(
                f"No SportsData.io {cfg['label']} key detected; {cfg['label']} power metrics fall back to sportsbook + sentiment only."
            )

    if basketball_client and basketball_client.is_configured():
        st.caption(
            f"Using NBA API-Sports key from {describe_key_origin(basketball_client.key_origin())}."
        )
    else:
        st.caption("No NBA API-Sports key detected; NBA live data will be skipped.")

    render_api_sports_key_section(
        header="🏒 API-Sports NHL Data Integration",
        label="NHL API-Sports Key",
        session_key='nhl_apisports_api_key',
        source_session_key='nhl_apisports_key_source',
        client=hockey_client,
        help_text="Set the NHL_APISPORTS_API_KEY secret or request an NHL token from https://api-sports.io/",
        success_message="✅ NHL API-Sports key saved for this session.",
        empty_message="NHL live data integration disabled until a key is provided.",
        widget_suffix="main",
    )

    if hockey_client and hockey_client.is_configured():
        st.caption(
            f"Using NHL API-Sports key from {describe_key_origin(hockey_client.key_origin())}."
        )
    else:
        st.caption("No NHL API-Sports key detected; NHL live data will be skipped.")

    tracker_clients = {
        'americanfootball_nfl': apisports_client,
        'basketball_nba': basketball_client,
        'icehockey_nhl': hockey_client,
    }
    user_timezone_label = st.session_state.get('user_timezone', 'UTC')
    render_saved_parlay_tracker(tracker_clients, user_timezone_label)

    st.markdown("---")
    st.subheader("🏆 Best Overall Odds for Date Range")

    default_start = sel_date - timedelta(days=_day_window or 0)
    default_end = sel_date + timedelta(days=_day_window or 0)

    col_start, col_end, col_action = st.columns([1, 1, 1])
    with col_start:
        best_odds_start = st.date_input(
            "Start date",
            value=default_start,
            key="best_odds_start",
        )
    with col_end:
        best_odds_end = st.date_input(
            "End date",
            value=default_end,
            key="best_odds_end",
        )
    with col_action:
        fetch_best_odds = st.button(
            "Show Best Odds",
            key="best_odds_button",
            use_container_width=True,
        )

    if fetch_best_odds:
        odds_key = st.session_state.get('api_key', "") or os.environ.get("ODDS_API_KEY", "")
        if not odds_key:
            st.error("Configure your The Odds API key to pull live odds data.")
        else:
            with st.spinner("Calculating top prices across sportsbooks..."):
                best_odds_df = build_best_odds_report(
                    odds_key,
                    active_sports_list,
                    best_odds_start,
                    best_odds_end,
                    sidebar_state["timezone_name"],
                )

            if best_odds_df.empty:
                st.info("No qualifying odds found for the selected range.")
            else:
                display_df = best_odds_df.copy()
                if "decimal_odds" in display_df.columns:
                    display_df["decimal_odds"] = display_df["decimal_odds"].map(
                        lambda x: f"{float(x):.3f}" if pd.notna(x) else "—"
                    )
                if "line" in display_df.columns:
                    display_df["line"] = display_df["line"].map(
                        lambda x: f"{float(x):.1f}" if pd.notna(x) else "—"
                    )

                st.dataframe(display_df, use_container_width=True, hide_index=True)

                csv_export = best_odds_df.to_csv(index=False)
                file_name = (
                    f"best_odds_{best_odds_start.isoformat()}_{best_odds_end.isoformat()}.csv"
                )
                st.download_button(
                    "💾 Download best odds CSV",
                    data=csv_export,
                    file_name=file_name,
                    mime="text/csv",
                    key="best_odds_download",
                )


    st.subheader("🤖 AI/ML Best Bet Per Game (Parlay View)")

    leg_start_col, leg_end_col, leg_sport_col = st.columns([1, 1, 1])
    with leg_start_col:
        best_leg_start = st.date_input(
            "Start date (best bet)",
            value=default_start,
            key="best_leg_start",
        )
    with leg_end_col:
        best_leg_end = st.date_input(
            "End date (best bet)",
            value=default_end,
            key="best_leg_end",
        )

    sport_display_map = {format_sport_label(key): key for key in APP_CFG["sports_common"]}
    default_sport_labels = [
        label for label, key in sport_display_map.items() if key in active_sports_list
    ]
    with leg_sport_col:
        selected_sport_labels = st.multiselect(
            "Sports",
            options=list(sport_display_map.keys()),
            default=default_sport_labels or list(sport_display_map.keys()),
            key="best_leg_sports",
        )

    selected_sport_keys = [
        sport_display_map[label]
        for label in selected_sport_labels
        if label in sport_display_map
    ]

    compute_best_bets = st.button(
        "Compute Best Bets",
        key="compute_best_bets",
        use_container_width=True,
    )

    if compute_best_bets:
        odds_key = st.session_state.get('api_key', "") or os.environ.get("ODDS_API_KEY", "")
        if not odds_key:
            st.error("Configure your The Odds API key to evaluate best bets.")
        else:
            sentiment_analyzer = st.session_state.get('sentiment_analyzer')
            ml_predictor_state = st.session_state.get('ml_predictor') if use_ml_predictions else None
            apisports_map = {
                'americanfootball_nfl': apisports_client,
                'basketball_nba': basketball_client,
                'icehockey_nhl': hockey_client,
            }

            target_sports = selected_sport_keys or active_sports_list

            with st.spinner("Blending AI, ML, and market data across each game..."):
                best_bets_df, _ = build_best_bets_per_game(
                    odds_key,
                    target_sports,
                    best_leg_start,
                    best_leg_end,
                    sidebar_state["timezone_name"],
                    sentiment_analyzer=sentiment_analyzer,
                    ml_predictor=ml_predictor_state,
                    use_sentiment=use_sentiment,
                    use_ml_predictions=use_ml_predictions,
                    min_ai_confidence=min_ai_confidence,
                    use_kalshi=use_kalshi,
                    theover_ml_data=theover_ml_data,
                    theover_spreads_data=theover_spreads_data,
                    theover_totals_data=theover_totals_data,
                    sportsdata_clients=sportsdata_clients,
                    apisports_clients=apisports_map,
                )

            if best_bets_df.empty:
                st.info("No qualifying legs found for the selected range and sports.")
            else:
                display_df = best_bets_df.copy()

                percent_columns = [
                    "Implied Prob %",
                    "AI Prob %",
                    "AI Raw %",
                    "AI EV %",
                    "AI Edge pp",
                    "AI Confidence %",
                    "ML Prob %",
                    "theover.ai %",
                    "theover Δ pp",
                    "SportsData Prob %",
                    "SportsData Δ pp",
                    "Kalshi Prob %",
                    "Kalshi Δ pp",
                    "Kalshi Edge %",
                    "Best Edge %",
                    "Best Win Prob %",
                ]
                for col in percent_columns:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].map(
                            lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
                        )

                if 'Best Decimal' in display_df.columns:
                    display_df['Best Decimal'] = pd.to_numeric(
                        display_df['Best Decimal'], errors='coerce'
                    )
                    display_df['Best Decimal'] = display_df['Best Decimal'].map(
                        lambda x: f"{float(x):.3f}" if pd.notna(x) else "—"
                    )

                if 'Best American' in display_df.columns:
                    display_df['Best American'] = pd.to_numeric(
                        display_df['Best American'], errors='coerce'
                    )
                    display_df['Best American'] = display_df['Best American'].map(
                        lambda x: f"{int(round(float(x))):+d}" if pd.notna(x) else "—"
                    )

                if 'Line' in display_df.columns:
                    display_df['Line'] = pd.to_numeric(display_df['Line'], errors='coerce')
                    display_df['Line'] = display_df['Line'].map(
                        lambda x: f"{float(x):g}" if pd.notna(x) else "—"
                    )

                st.dataframe(display_df, use_container_width=True, hide_index=True)

                csv_export = best_bets_df.to_csv(index=False)
                download_name = (
                    f"best_bets_{best_leg_start.isoformat()}_{best_leg_end.isoformat()}.csv"
                )
                st.download_button(
                    "💾 Download best bet CSV",
                    data=csv_export,
                    file_name=download_name,
                    mime="text/csv",
                    key="best_bets_download",
                )


    def is_within_date_window(iso_str) -> bool:
        """Return True when an event falls within the selected day ± window."""
        try:
            ts_local = pd.to_datetime(iso_str, utc=True).tz_convert(tz)
            event_date = ts_local.date()
            delta_days = (event_date - sel_date).days
            return -_day_window <= delta_days <= _day_window
        except Exception:
            return False

    if st.button("🤖 Find AI-Optimized Parlays", type="primary"):
        # Get API key from session state or environment only
        api_key = st.session_state.get('api_key', "") or os.environ.get("ODDS_API_KEY", "")
        
        if not api_key:
            st.error("No API key provided. Please enter your API key above.")
            st.stop()
        
        if not (inc_ml or inc_spread or inc_total):
            st.error("Please select at least one bet type")
            st.stop()
        
        try:
            # Safely get AI components
            try:
                sentiment_analyzer = st.session_state.get('sentiment_analyzer')
                ml_predictor = st.session_state.get('ml_predictor')
                ai_optimizer = st.session_state.get('ai_optimizer')
                
                if not sentiment_analyzer or not ml_predictor or not ai_optimizer:
                    st.error("AI components not initialized. Please refresh the page.")
                    st.stop()
            except Exception as e:
                st.error(f"Error accessing AI components: {str(e)}")
                st.stop()
        
            with st.spinner("🧠 Analyzing markets with AI..."):
                progress_bar = st.progress(0)
                all_legs = []
                apisports_games_cache: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
                total_sports = len(active_sports_list)

                for sport_idx, skey in enumerate(active_sports_list):
                    try:
                        progress_bar.progress((sport_idx) / total_sports)
                        snap = fetch_oddsapi_snapshot(api_key, skey)
                        
                        if not snap or not snap.get("events"):
                            continue  # Skip if no events
                        
                        for ev in (snap.get("events") or [])[:per_sport_events]:
                            try:
                                if not is_within_date_window(ev.get("commence_time")):
                                    continue
                                
                                eid = ev.get("id")
                                home = ev.get("home_team", "?")
                                away = ev.get("away_team", "?")
                                mkts = ev.get("markets") or {}
                                
                                if not eid or not home or not away:
                                    continue  # Skip invalid events

                                apisports_summary = None
                                apisports_payload_home = None
                                apisports_payload_away = None
                                apisports_payload_total = None
                                sportsdata_summary = None
                                sportsdata_payload_home = None
                                sportsdata_payload_away = None
                                sportsdata_payload_total = None

                                client_for_leg = None
                                if (
                                    skey == "americanfootball_nfl"
                                    and apisports_client
                                    and apisports_client.is_configured()
                                ):
                                    client_for_leg = apisports_client
                                elif (
                                    skey == "basketball_nba"
                                    and basketball_client
                                    and basketball_client.is_configured()
                                ):
                                    client_for_leg = basketball_client
                                elif (
                                    skey == "icehockey_nhl"
                                    and hockey_client
                                    and hockey_client.is_configured()
                                ):
                                    client_for_leg = hockey_client

                                if client_for_leg:
                                    try:
                                        event_ts = pd.to_datetime(ev.get("commence_time"), utc=True)
                                        tz_label = st.session_state.get('user_timezone') or getattr(tz, 'zone', 'UTC') or 'UTC'
                                        try:
                                            target_tz = pytz.timezone(tz_label)
                                        except Exception:
                                            target_tz = pytz.timezone('UTC')
                                        local_date = event_ts.tz_convert(target_tz).date()
                                        cache_key = (skey, local_date.isoformat(), tz_label)

                                        if cache_key not in apisports_games_cache:
                                            apisports_games_cache[cache_key] = client_for_leg.get_games_by_date(
                                                local_date,
                                                timezone=tz_label,
                                            )

                                        matched_game = client_for_leg.match_game(
                                            apisports_games_cache.get(cache_key, []),
                                            home,
                                            away,
                                        )

                                        if matched_game:
                                            apisports_summary = client_for_leg.build_game_summary(
                                                matched_game,
                                                tz_name=tz_label,
                                            )
                                            apisports_payload_home = build_leg_apisports_payload(
                                                apisports_summary,
                                                'home',
                                                sport_key=skey,
                                            )
                                            apisports_payload_away = build_leg_apisports_payload(
                                                apisports_summary,
                                                'away',
                                                sport_key=skey,
                                            )
                                            total_trend = apisports_payload_home.get('trend') or apisports_payload_away.get('trend')
                                            apisports_payload_total = {
                                                key: getattr(apisports_summary, attr)
                                                for key, attr in [
                                                    ('game_id', 'id'),
                                                    ('league', 'league'),
                                                    ('season', 'season'),
                                                    ('status', 'status'),
                                                    ('kickoff', 'kickoff_local'),
                                                    ('venue', 'venue'),
                                                ]
                                                if getattr(apisports_summary, attr, None)
                                            }
                                            apisports_payload_total['sport_key'] = skey
                                            apisports_payload_total['sport_name'] = getattr(apisports_summary, 'sport_name', None)
                                            apisports_payload_total['scoring_metric'] = getattr(apisports_summary, 'scoring_metric', None)
                                            if total_trend:
                                                apisports_payload_total['trend'] = total_trend
                                    except Exception:
                                        apisports_summary = None
                                        apisports_payload_home = None
                                        apisports_payload_away = None
                                        apisports_payload_total = None

                                client_for_sd = sportsdata_clients.get(skey)
                                if client_for_sd and client_for_sd.is_configured():
                                    try:
                                        event_ts = pd.to_datetime(ev.get("commence_time"), utc=True)
                                        tz_label = st.session_state.get('user_timezone') or getattr(tz, 'zone', 'UTC') or 'UTC'
                                        try:
                                            target_tz = pytz.timezone(tz_label)
                                        except Exception:
                                            target_tz = pytz.timezone('UTC')
                                        local_date_sd = event_ts.tz_convert(target_tz).date()
                                        sportsdata_summary = client_for_sd.find_game_insight(
                                            local_date_sd,
                                            home,
                                            away,
                                        )
                                        if sportsdata_summary:
                                            sportsdata_payload_home = build_leg_sportsdata_payload(
                                                sportsdata_summary,
                                                'home',
                                                sport_key=skey,
                                            )
                                            sportsdata_payload_away = build_leg_sportsdata_payload(
                                                sportsdata_summary,
                                                'away',
                                                sport_key=skey,
                                            )
                                            sportsdata_payload_total = build_leg_sportsdata_payload(
                                                sportsdata_summary,
                                                'total',
                                                sport_key=skey,
                                            )
                                    except Exception:
                                        sportsdata_summary = None
                                        sportsdata_payload_home = None
                                        sportsdata_payload_away = None
                                        sportsdata_payload_total = None

                                # Get sentiment for both teams
                                try:
                                    home_sentiment = sentiment_analyzer.get_team_sentiment(home, skey) if use_sentiment else {'score': 0, 'trend': 'neutral'}
                                    away_sentiment = sentiment_analyzer.get_team_sentiment(away, skey) if use_sentiment else {'score': 0, 'trend': 'neutral'}
                                except Exception:
                                    home_sentiment = {'score': 0, 'trend': 'neutral'}
                                    away_sentiment = {'score': 0, 'trend': 'neutral'}
                                
                                # Moneyline with AI
                                if inc_ml and "h2h" in mkts:
                                    hp = _dig(mkts["h2h"], "home.price")
                                    ap = _dig(mkts["h2h"], "away.price")

                                    ml_context = {
                                        "sport_key": skey,
                                        "event_id": eid,
                                        "apisports_home": apisports_payload_home,
                                        "apisports_away": apisports_payload_away,
                                        "sportsdata_home": sportsdata_payload_home,
                                        "sportsdata_away": sportsdata_payload_away,
                                    }
                                    ml_prediction_result = None
                                    if use_ml_predictions and hp is not None and ap is not None:
                                        try:
                                            ml_prediction_result = ml_predictor.predict_game_outcome(
                                                home,
                                                away,
                                                hp,
                                                ap,
                                                home_sentiment['score'],
                                                away_sentiment['score'],
                                                context=ml_context,
                                            )
                                        except Exception:
                                            ml_prediction_result = None

                                    if hp is not None and -750 <= hp <= 750:
                                        base_prob = implied_p_from_american(hp)
                                        ai_prob = base_prob

                                        if use_ml_predictions and ap is not None:
                                            if ml_prediction_result is None:
                                                try:
                                                    ml_prediction_result = ml_predictor.predict_game_outcome(
                                                        home,
                                                        away,
                                                        hp,
                                                        ap,
                                                        home_sentiment['score'],
                                                        away_sentiment['score'],
                                                        context=ml_context,
                                                    )
                                                except Exception:
                                                    ml_prediction_result = None
                                            if ml_prediction_result:
                                                ai_prob = ml_prediction_result['home_prob']
                                                ai_confidence = ml_prediction_result['confidence']
                                                ai_edge = ml_prediction_result['edge']
                                            else:
                                                ai_confidence = 0.5
                                                ai_edge = 0
                                        else:
                                            ai_confidence = 0.5
                                            ai_edge = 0

                                        if ai_confidence >= min_ai_confidence:
                                            decimal_odds = american_to_decimal_safe(hp)
                                            if decimal_odds is not None:  # Safety check
                                                leg_data = {
                                                    "event_id": eid,
                                                    "type": "Moneyline",
                                                    "team": home,
                                                    "side": "home",
                                                    "market": "ML",
                                                    "label": f"{away} @ {home} — {home} ML @{hp}",
                                                    "p": base_prob,
                                                    "ai_prob": ai_prob,
                                                    "ai_confidence": ai_confidence,
                                                    "ai_edge": ai_edge,
                                                    "d": decimal_odds,
                                                    "sentiment_trend": home_sentiment['trend'],
                                                    "sport_key": skey,
                                                    "home_team": home,
                                                    "away_team": away,
                                                    "commence_time": ev.get('commence_time'),
                                                }

                                                if ml_prediction_result:
                                                    leg_data['ai_model_source'] = ml_prediction_result.get('model_used')
                                                    leg_data['ai_training_rows'] = ml_prediction_result.get('training_rows')
                                                    component_breakdown = ml_prediction_result.get('component_probabilities')
                                                    if isinstance(component_breakdown, dict) and component_breakdown:
                                                        leg_data['ai_component_probabilities'] = component_breakdown

                                                if apisports_payload_home:
                                                    leg_data['apisports'] = apisports_payload_home
                                                if sportsdata_payload_home:
                                                    leg_data['sportsdata'] = sportsdata_payload_home

                                                integrate_kalshi_into_leg(
                                                    leg_data,
                                                    home,
                                                    away,
                                                    'home',
                                                    base_prob,
                                                    skey,
                                                    use_kalshi
                                                )

                                                all_legs.append(leg_data)

                                    if ap is not None and -750 <= ap <= 750:
                                        base_prob = implied_p_from_american(ap)
                                        ai_prob = base_prob

                                        if use_ml_predictions and hp is not None:
                                            if ml_prediction_result is None:
                                                try:
                                                    ml_prediction_result = ml_predictor.predict_game_outcome(
                                                        home,
                                                        away,
                                                        hp,
                                                        ap,
                                                        home_sentiment['score'],
                                                        away_sentiment['score'],
                                                        context=ml_context,
                                                    )
                                                except Exception:
                                                    ml_prediction_result = None
                                            if ml_prediction_result:
                                                ai_prob = ml_prediction_result['away_prob']
                                                ai_confidence = ml_prediction_result['confidence']
                                                ai_edge = ml_prediction_result['edge']
                                            else:
                                                ai_confidence = 0.5
                                                ai_edge = 0
                                        else:
                                            ai_confidence = 0.5
                                            ai_edge = 0

                                        if ai_confidence >= min_ai_confidence:
                                            decimal_odds = american_to_decimal_safe(ap)
                                            if decimal_odds is not None:  # Safety check
                                                leg_data = {
                                                    "event_id": eid,
                                                    "type": "Moneyline",
                                                    "team": away,
                                                    "side": "away",
                                                    "market": "ML",
                                                    "label": f"{away} @ {home} — {away} ML @{ap}",
                                                    "p": base_prob,
                                                    "ai_prob": ai_prob,
                                                    "ai_confidence": ai_confidence,
                                                    "ai_edge": ai_edge,
                                                    "d": decimal_odds,
                                                    "sentiment_trend": away_sentiment['trend'],
                                                    "sport_key": skey,
                                                    "home_team": home,
                                                    "away_team": away,
                                                    "commence_time": ev.get('commence_time'),
                                                }

                                                if ml_prediction_result:
                                                    leg_data['ai_model_source'] = ml_prediction_result.get('model_used')
                                                    leg_data['ai_training_rows'] = ml_prediction_result.get('training_rows')
                                                    component_breakdown = ml_prediction_result.get('component_probabilities')
                                                    if isinstance(component_breakdown, dict) and component_breakdown:
                                                        leg_data['ai_component_probabilities'] = component_breakdown

                                                if apisports_payload_away:
                                                    leg_data['apisports'] = apisports_payload_away
                                                if sportsdata_payload_away:
                                                    leg_data['sportsdata'] = sportsdata_payload_away

                                                integrate_kalshi_into_leg(
                                                    leg_data,
                                                    home,
                                                    away,
                                                    'away',
                                                    base_prob,
                                                    skey,
                                                    use_kalshi
                                                )

                                                all_legs.append(leg_data)
                                
                                # Spreads
                                if inc_spread and "spreads" in mkts:
                                    for o in mkts["spreads"][:4]:
                                        nm, pt, pr = o.get("name"), o.get("point"), o.get("price")
                                        if nm is None or pt is None or pr is None: 
                                            continue
                                        
                                        base_prob = implied_p_from_american(pr)
                                        sentiment = home_sentiment if nm == home else away_sentiment
                                        
                                        # Simple sentiment adjustment for spreads (40% weight)
                                        ai_prob = base_prob * (1 + sentiment['score'] * 0.40)
                                        ai_prob = max(0.1, min(0.9, ai_prob))  # Clamp
                                        
                                        # Higher confidence for spread bets (adjusted from 0.6 to 0.65)
                                        ai_confidence = 0.65
                                        
                                        decimal_odds = american_to_decimal_safe(pr)
                                        if decimal_odds is not None and ai_confidence >= min_ai_confidence:  # Safety check
                                            leg_data = {
                                                "event_id": eid,
                                                "type": "Spread",
                                                "team": nm,
                                                "side": "home" if nm == home else "away",
                                                "point": pt,
                                                "market": "Spread",
                                                "label": f"{away} @ {home} — {nm} {pt:+.1f} @{pr}",
                                                "p": base_prob,
                                                "ai_prob": ai_prob,
                                                "ai_confidence": ai_confidence,
                                                "ai_edge": abs(ai_prob - base_prob),
                                                "d": decimal_odds,
                                                "sentiment_trend": sentiment['trend'],
                                                "sport_key": skey,
                                                "home_team": home,
                                                "away_team": away,
                                                "commence_time": ev.get('commence_time'),
                                            }

                                            if nm == home and apisports_payload_home:
                                                leg_data['apisports'] = apisports_payload_home
                                            elif nm == away and apisports_payload_away:
                                                leg_data['apisports'] = apisports_payload_away

                                            if nm == home and sportsdata_payload_home:
                                                leg_data['sportsdata'] = sportsdata_payload_home
                                            elif nm == away and sportsdata_payload_away:
                                                leg_data['sportsdata'] = sportsdata_payload_away

                                            integrate_kalshi_into_leg(
                                                leg_data,
                                                home,
                                                away,
                                                'home' if nm == home else 'away',
                                                base_prob,
                                                skey,
                                                use_kalshi
                                            )

                                            all_legs.append(leg_data)
                                
                                # Totals
                                if inc_total and "totals" in mkts:
                                    for o in mkts["totals"][:4]:
                                        nm, pt, pr = o.get("name"), o.get("point"), o.get("price")
                                        if nm is None or pt is None or pr is None: 
                                            continue
                                        
                                        base_prob = implied_p_from_american(pr)
                                        
                                        # For totals, combine both teams' offensive sentiment (40% weight)
                                        combined_sentiment = (home_sentiment['score'] + away_sentiment['score']) / 2
                                        ai_prob = base_prob * (1 + combined_sentiment * 0.40 * 0.5)
                                        ai_prob = max(0.1, min(0.9, ai_prob))
                                        
                                        # Higher confidence for totals (adjusted from 0.55 to 0.60)
                                        ai_confidence = 0.60
                                        
                                        decimal_odds = american_to_decimal_safe(pr)
                                        if decimal_odds is not None and ai_confidence >= min_ai_confidence:  # Safety check
                                            leg_data = {
                                                "event_id": eid,
                                                "type": "Total",
                                                "team": f"{home} vs {away}",
                                                "side": nm,  # Over or Under
                                                "point": pt,
                                                "market": "Total",
                                                "label": f"{away} @ {home} — {nm} {pt} @{pr}",
                                                "p": base_prob,
                                                "ai_prob": ai_prob,
                                                "ai_confidence": ai_confidence,
                                                "ai_edge": abs(ai_prob - base_prob),
                                                "d": decimal_odds,
                                                "sentiment_trend": "neutral",
                                                "sport_key": skey,
                                                "home_team": home,
                                                "away_team": away,
                                                "commence_time": ev.get('commence_time'),
                                            }

                                            if apisports_payload_total:
                                                leg_data['apisports'] = apisports_payload_total
                                            if sportsdata_payload_total:
                                                leg_data['sportsdata'] = sportsdata_payload_total

                                            if use_kalshi:
                                                leg_data['kalshi_validation'] = {
                                                    'kalshi_available': False,
                                                    'validation': 'unsupported',
                                                    'edge': 0,
                                                    'confidence_boost': 0,
                                                    'market_scope': 'total_market',
                                                    'data_source': 'unsupported',
                                                    'reason': 'Kalshi does not list totals/over-under style markets'
                                                }
                                            else:
                                                leg_data['kalshi_validation'] = {
                                                    'kalshi_available': False,
                                                    'validation': 'disabled',
                                                    'edge': 0,
                                                    'confidence_boost': 0,
                                                    'market_scope': 'disabled',
                                                    'data_source': 'disabled'
                                                }

                                            all_legs.append(leg_data)
                            
                            except Exception as e:
                                # Skip this event if there's an error processing it
                                continue
                    
                    except Exception as e:
                        # Skip this sport if there's an error
                        st.warning(f"Error processing {skey}: {str(e)[:100]}")
                        continue
                
                progress_bar.progress(1.0)
                
                if not all_legs:
                    st.warning("No bets found for selected date/sports/filters")
                    st.stop()
                
                # AI Summary
                st.success(f"🤖 AI Analysis Complete: Found {len(all_legs)} betting opportunities")
                
                # Show AI insights
                with st.expander("📊 AI Market Analysis", expanded=True):
                    col_insight1, col_insight2, col_insight3 = st.columns(3)

                    def _is_high_confidence(leg: Dict[str, Any]) -> bool:
                        try:
                            return float(leg.get('ai_confidence', 0)) > 0.7
                        except (TypeError, ValueError):
                            return False

                    def _is_positive_ev(leg: Dict[str, Any]) -> bool:
                        try:
                            prob = float(leg.get('ai_prob', 0))
                            decimal = float(leg.get('d', 0))
                        except (TypeError, ValueError):
                            return False
                        return prob * decimal > 1.05

                    high_confidence = [leg for leg in all_legs if _is_high_confidence(leg)]
                    positive_ev = [leg for leg in all_legs if _is_positive_ev(leg)]
                    sentiment_edge = [leg for leg in all_legs if leg.get('sentiment_trend') == 'positive']
                    
                    with col_insight1:
                        st.metric(
                            "High Confidence Bets",
                            len(high_confidence),
                            help="Bets with >70% AI confidence",
                        )
                    with col_insight2:
                        st.metric(
                            "Positive AI EV Bets",
                            len(positive_ev),
                            help="Bets with >5% expected value",
                        )
                    with col_insight3:
                        st.metric(
                            "Positive Sentiment",
                            len(sentiment_edge),
                            help="Teams with positive news sentiment",
                        )
                
                # Create tabs for different parlay sizes
                tab_2, tab_3, tab_4, tab_5 = st.tabs([
                    "🎯 2-Leg Parlays", 
                    "🎲 3-Leg Parlays", 
                    "🚀 4-Leg Parlays", 
                    "💎 5-Leg Parlays"
                ])
                
                with tab_2:
                    st.subheader("Best 2-Leg AI-Optimized Parlays")
                    try:
                        with st.spinner("Calculating optimal 2-leg combinations..."):
                            combos_2 = build_combos_ai(
                                all_legs,
                                2,
                                allow_sgp,
                                ai_optimizer,
                                theover_ml_data,
                                theover_spreads_data,
                                theover_totals_data,
                                min_parlay_probability,
                                max_parlay_probability,
                            )[:show_top]
                            render_parlay_section_ai(
                                "2-Leg AI Parlays",
                                combos_2,
                                theover_ml_data,
                                theover_spreads_data,
                                theover_totals_data,
                                timezone_label=user_timezone_label,
                            )
                    except Exception as e:
                        st.error(f"Error building 2-leg parlays: {str(e)}")
                
                with tab_3:
                    st.subheader("Best 3-Leg AI-Optimized Parlays")
                    try:
                        with st.spinner("Calculating optimal 3-leg combinations..."):
                            combos_3 = build_combos_ai(
                                all_legs,
                                3,
                                allow_sgp,
                                ai_optimizer,
                                theover_ml_data,
                                theover_spreads_data,
                                theover_totals_data,
                                min_parlay_probability,
                                max_parlay_probability,
                            )[:show_top]
                            render_parlay_section_ai(
                                "3-Leg AI Parlays",
                                combos_3,
                                theover_ml_data,
                                theover_spreads_data,
                                theover_totals_data,
                                timezone_label=user_timezone_label,
                            )
                    except Exception as e:
                        st.error(f"Error building 3-leg parlays: {str(e)}")
                
                with tab_4:
                    st.subheader("Best 4-Leg AI-Optimized Parlays")
                    try:
                        with st.spinner("Calculating optimal 4-leg combinations..."):
                            combos_4 = build_combos_ai(
                                all_legs,
                                4,
                                allow_sgp,
                                ai_optimizer,
                                theover_ml_data,
                                theover_spreads_data,
                                theover_totals_data,
                                min_parlay_probability,
                                max_parlay_probability,
                            )[:show_top]
                            render_parlay_section_ai(
                                "4-Leg AI Parlays",
                                combos_4,
                                theover_ml_data,
                                theover_spreads_data,
                                theover_totals_data,
                                timezone_label=user_timezone_label,
                            )
                    except Exception as e:
                        st.error(f"Error building 4-leg parlays: {str(e)}")
                
                with tab_5:
                    st.subheader("Best 5-Leg AI-Optimized Parlays")
                    try:
                        with st.spinner("Calculating optimal 5-leg combinations..."):
                            combos_5 = build_combos_ai(
                                all_legs,
                                5,
                                allow_sgp,
                                ai_optimizer,
                                theover_ml_data,
                                theover_spreads_data,
                                theover_totals_data,
                                min_parlay_probability,
                                max_parlay_probability,
                            )[:show_top]
                            render_parlay_section_ai(
                                "5-Leg AI Parlays",
                                combos_5,
                                theover_ml_data,
                                theover_spreads_data,
                                theover_totals_data,
                                timezone_label=user_timezone_label,
                            )
                    except Exception as e:
                        st.error(f"Error building 5-leg parlays: {str(e)}")
        
        except KeyError as e:
            st.error(f"Configuration error: Missing key {str(e)}. Please refresh the page.")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            import traceback
            with st.expander("Full Error Details (for debugging)"):
                st.code(traceback.format_exc())
            st.info("Please try:")
            st.markdown("""
            1. Refresh the page
            2. Check your API key is valid
            3. Try selecting fewer sports
            4. Try a different date
            5. Disable sentiment analysis if enabled
            """)
# ---- Global safe odds helper (always available) ------------------------------
def american_to_decimal_safe(odds) -> Optional[float]:
    """
    Safe American→Decimal conversion.
    Returns None for None/0/invalid odds in (-100, 100) or on parsing errors.
    """
    try:
        if odds is None:
            return None
        o = float(odds)
        if abs(o) < 100:
            return None
        if o >= 100:
            return 1.0 + o/100.0
        else:
            return 1.0 + 100.0/abs(o)
    except Exception:
        return None
# -----------------------------------------------------------------------------
# ---- Safety shim to guarantee robust odds conversion ----
try:
    american_to_decimal_safe
except NameError:
    pass
# ---------------------------------------------------------
    
st.markdown("---")
st.markdown("""
    ### 🤖 AI Features Explained:

    **Sentiment Analysis** 🎭
    - Analyzes team news, social media, and expert opinions
    - Positive sentiment = team trending up, negative = trending down
    - Integrated into probability adjustments

    **ML Predictions** 🧠
    - Ensemble machine learning model analyzes odds patterns
    - Adjusts probabilities based on multiple factors
    - Higher confidence when ML and market agree

    **AI Confidence Score** 📊
    - Green (🟢): High confidence (>70%) - Strong AI signal
    - Yellow (🟡): Moderate confidence (50-70%) - Good opportunity
    - Orange (🟠): Lower confidence (<50%) - Higher risk

    **AI Expected Value** 💰
    - Compares AI probability vs market odds
    - Positive EV = mathematical edge over the market
    - Higher EV = better long-term profitability

    **AI Score** ⭐
    - Combined metric: EV × Confidence × Edge
    - Higher scores = best overall opportunities
    - Accounts for correlation in same-game parlays
    """)
    
st.caption("🟢 High Confidence | 💰 High +EV | 📈 Positive EV | 📉 Negative EV | Powered by AI & ML")

# ===== TAB 2: PRIZEPICKS =====

# ===== TAB 2: SENTIMENT & AI ANALYSIS =====
with main_tab2:
    st.header("🔍 Sentiment & AI Analysis Dashboard")
    st.markdown("**Advanced sentiment analysis using web scraping, news APIs, and AI-powered insights**")
    st.caption("Get deep insights into team performance, news sentiment, betting trends, and AI predictions")
    
    # API Configuration
    st.markdown("---")
    st.subheader("⚙️ Configuration")
    
    col_api1, col_api2 = st.columns(2)
    with col_api1:
        odds_key = st.session_state.get('api_key', "") or os.environ.get("ODDS_API_KEY", "")
        if not odds_key:
            st.warning("⚠️ Odds API key not configured. Please set it in the Sports Betting tab.")
    
    with col_api2:
        news_key = st.session_state.get('news_api_key', "") or os.environ.get("NEWS_API_KEY", "")
        news_key_input = st.text_input(
            "NewsAPI Key (optional - enhances sentiment)",
            value=news_key,
            type="password",
            help="Get free key at https://newsapi.org"
        )
        if news_key_input != news_key:
            st.session_state['news_api_key'] = news_key_input
            st.session_state['sentiment_analyzer'] = RealSentimentAnalyzer(news_key_input)
    
    st.markdown("---")
    
    # Team Selection
    st.subheader("🎯 Select Teams to Analyze")
    
    col_sport, col_num = st.columns(2)
    with col_sport:
        analysis_sport = st.selectbox(
            "Sport",
            options=APP_CFG["sports_common"],
            key="analysis_sport",
            format_func=format_sport_label,
        )
    
    with col_num:
        num_teams = st.slider("Number of teams to analyze", 2, 10, 5)
    
    # Fetch games and extract teams
    if st.button("🔍 Load Teams", type="primary"):
        if not odds_key:
            st.error("Please configure Odds API key first")
        else:
            sport_label = format_sport_label(analysis_sport)
            with st.spinner(f"Loading {sport_label} teams..."):
                try:
                    snap = fetch_oddsapi_snapshot(odds_key, analysis_sport)
                    events = snap.get("events", [])
                    
                    if not events:
                        st.warning("No games found for this sport")
                    else:
                        # Extract unique teams
                        teams = set()
                        team_games = {}  # Map team to their games
                        
                        for ev in events:
                            home = ev.get("home_team")
                            away = ev.get("away_team")
                            
                            if home:
                                teams.add(home)
                                if home not in team_games:
                                    team_games[home] = []
                                team_games[home].append({
                                    'opponent': away,
                                    'location': 'home',
                                    'game_id': ev.get('id'),
                                    'commence_time': ev.get('commence_time')
                                })
                            
                            if away:
                                teams.add(away)
                                if away not in team_games:
                                    team_games[away] = []
                                team_games[away].append({
                                    'opponent': home,
                                    'location': 'away',
                                    'game_id': ev.get('id'),
                                    'commence_time': ev.get('commence_time')
                                })
                        
                        st.session_state['analysis_teams'] = sorted(list(teams))
                        st.session_state['team_games'] = team_games
                        st.success(f"✅ Found {len(teams)} teams with upcoming games")
                
                except Exception as e:
                    st.error(f"Error loading teams: {str(e)}")
    
    # Team Analysis
    if 'analysis_teams' in st.session_state and st.session_state['analysis_teams']:
        st.markdown("---")
        st.subheader("📊 Team Sentiment Analysis")
        
        teams = st.session_state['analysis_teams']
        selected_teams = st.multiselect(
            "Select teams to analyze",
            options=teams,
            default=teams[:min(num_teams, len(teams))],
            max_selections=10
        )
        
        if st.button("🤖 Run Deep Analysis", type="primary"):
            if not selected_teams:
                st.warning("Please select at least one team")
            else:
                sentiment_analyzer = st.session_state.get('sentiment_analyzer')
                
                if not sentiment_analyzer:
                    st.error("Sentiment analyzer not initialized")
                else:
                    progress_bar = st.progress(0)
                    analysis_results = []
                    
                    for idx, team in enumerate(selected_teams):
                        progress_bar.progress((idx + 1) / len(selected_teams))
                        
                        with st.spinner(f"Analyzing {team}..."):
                            try:
                                # Get sentiment
                                sentiment = sentiment_analyzer.get_team_sentiment(team, analysis_sport)
                                
                                # Get game info
                                games = st.session_state['team_games'].get(team, [])
                                next_game = games[0] if games else None
                                
                                # Get odds if available
                                odds_data = None
                                if next_game:
                                    for ev in st.session_state.get('available_games', []):
                                        if ev.get('id') == next_game['game_id']:
                                            h2h = ev.get('markets', {}).get('h2h', {})
                                            if next_game['location'] == 'home':
                                                odds_data = _dig(h2h, 'home.price')
                                            else:
                                                odds_data = _dig(h2h, 'away.price')
                                            break
                                
                                analysis_results.append({
                                    'team': team,
                                    'sentiment': sentiment,
                                    'next_game': next_game,
                                    'odds': odds_data
                                })
                            
                            except Exception as e:
                                st.warning(f"Error analyzing {team}: {str(e)}")
                    
                    progress_bar.progress(1.0)
                    
                    # Display Results
                    if analysis_results:
                        st.markdown("---")
                        st.markdown("### 📈 Analysis Results")
                        
                        # Summary metrics
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        
                        positive_teams = sum(1 for r in analysis_results if r['sentiment']['trend'] == 'positive')
                        negative_teams = sum(1 for r in analysis_results if r['sentiment']['trend'] == 'negative')
                        avg_score = sum(r['sentiment']['score'] for r in analysis_results) / len(analysis_results)
                        avg_confidence = sum(r['sentiment']['confidence'] for r in analysis_results) / len(analysis_results)
                        
                        with col_m1:
                            st.metric("Positive Sentiment", positive_teams)
                        with col_m2:
                            st.metric("Negative Sentiment", negative_teams)
                        with col_m3:
                            st.metric("Avg Sentiment Score", f"{avg_score:+.2f}")
                        with col_m4:
                            st.metric("Avg Confidence", f"{avg_confidence*100:.0f}%")
                        
                        # Sort by sentiment score
                        analysis_results.sort(key=lambda x: x['sentiment']['score'], reverse=True)
                        
                        # Display each team
                        st.markdown("### 🏈 Team-by-Team Breakdown")
                        
                        for result in analysis_results:
                            team = result['team']
                            sentiment = result['sentiment']
                            next_game = result['next_game']
                            odds = result['odds']
                            
                            # Determine sentiment icon and color
                            if sentiment['trend'] == 'positive':
                                trend_icon = "🟢"
                                trend_color = "green"
                            elif sentiment['trend'] == 'negative':
                                trend_icon = "🔴"
                                trend_color = "red"
                            else:
                                trend_icon = "🟡"
                                trend_color = "orange"
                            
                            with st.expander(f"{trend_icon} {team} - {sentiment['trend'].upper()} ({sentiment['score']:+.2f})"):
                                col_info1, col_info2 = st.columns(2)
                                
                                with col_info1:
                                    st.markdown("**Sentiment Analysis:**")
                                    st.write(f"- **Score:** {sentiment['score']:+.2f}")
                                    st.write(f"- **Trend:** {sentiment['trend'].upper()}")
                                    st.write(f"- **Confidence:** {sentiment['confidence']*100:.0f}%")
                                    st.write(f"- **Sources:** {sentiment['sources']} articles")
                                    st.write(f"- **Method:** {sentiment['method']}")
                                
                                with col_info2:
                                    st.markdown("**Next Game:**")
                                    if next_game:
                                        location = "🏠 Home" if next_game['location'] == 'home' else "✈️ Away"
                                        st.write(f"- **vs {next_game['opponent']}** ({location})")
                                        if odds:
                                            prob = implied_p_from_american(odds)
                                            st.write(f"- **Odds:** {odds:+.0f}")
                                            st.write(f"- **Implied Prob:** {prob*100:.1f}%")
                                        
                                        # Try to parse time
                                        try:
                                            import datetime
                                            game_time = datetime.datetime.fromtimestamp(next_game['commence_time'])
                                            st.write(f"- **Time:** {game_time.strftime('%m/%d %I:%M %p')}")
                                        except:
                                            pass
                                    else:
                                        st.write("No upcoming game found")
                                
                                # Betting recommendation based on sentiment + odds
                                st.markdown("**💡 AI Betting Insight:**")
                                
                                if sentiment['score'] > 0.3 and sentiment['confidence'] > 0.6:
                                    if odds and odds > 0:  # Underdog with positive sentiment
                                        st.success("🟢 **STRONG VALUE** - Positive sentiment underdog. Market may be undervaluing this team.")
                                    elif odds and odds < -200:  # Heavy favorite with positive sentiment
                                        st.info("🟡 **GOOD SPOT** - Sentiment confirms favorite status, but odds may be steep.")
                                    else:
                                        st.success("🟢 **FAVORABLE** - Strong positive sentiment. Consider backing this team.")
                                
                                elif sentiment['score'] < -0.3 and sentiment['confidence'] > 0.6:
                                    if odds and odds < 0:  # Favorite with negative sentiment
                                        st.warning("🟠 **FADE CANDIDATE** - Negative sentiment on a favorite. Public may be overvaluing.")
                                    else:
                                        st.error("🔴 **AVOID** - Strong negative sentiment. Look elsewhere.")
                                
                                else:
                                    st.info("🟡 **NEUTRAL** - No strong sentiment signal. Rely on other factors.")
                        
                        # Export option
                        st.markdown("---")
                        export_data = []
                        for result in analysis_results:
                            export_data.append({
                                'Team': result['team'],
                                'Sentiment Score': f"{result['sentiment']['score']:+.2f}",
                                'Trend': result['sentiment']['trend'],
                                'Confidence': f"{result['sentiment']['confidence']*100:.0f}%",
                                'Sources': result['sentiment']['sources'],
                                'Next Opponent': result['next_game']['opponent'] if result['next_game'] else 'N/A',
                                'Location': result['next_game']['location'] if result['next_game'] else 'N/A',
                                'Odds': f"{result['odds']:+.0f}" if result['odds'] else 'N/A'
                            })
                        
                        df_export = pd.DataFrame(export_data)
                        csv_buf = io.StringIO()
                        df_export.to_csv(csv_buf, index=False)
                        
                        st.download_button(
                            "💾 Download Analysis CSV",
                            data=csv_buf.getvalue(),
                            file_name=f"sentiment_analysis_{format_sport_label(analysis_sport)}.csv",
                            mime="text/csv"
                        )
    
    else:
        st.info("👆 Click 'Load Teams' to start sentiment analysis")
    
    # Advanced Features Section
    st.markdown("---")
    st.markdown("### 🚀 Advanced Analysis Features")
    
    with st.expander("📰 News Sentiment Analysis"):
        st.markdown("""
        **How it works:**
        - Scrapes recent news articles about each team (last 3 days)
        - Uses NewsAPI.org for reliable news sources
        - Analyzes headlines and descriptions using NLP
        - Identifies positive words (win, dominant, stellar) vs negative words (lose, injury, struggle)
        - Calculates sentiment score from -1.0 (very negative) to +1.0 (very positive)
        - Provides confidence score based on number of sources and consistency
        
        **Positive Indicators:**
        - Winning streak, dominant performance, star player excelling
        - Record-breaking stats, momentum, clutch plays
        - Positive coaching changes, key player returns
        
        **Negative Indicators:**
        - Losing streak, injuries to key players, poor performance
        - Internal conflicts, coaching issues, suspensions
        - Defensive/offensive struggles, blown leads
        """)
    
    with st.expander("🤖 AI Prediction Model"):
        st.markdown("""
        **Machine Learning Components:**
        - **Input Features:** API-Sports team trends, The Odds API closing prices, sentiment deltas, home/away context
        - **Model Type:** Logistic regression pipeline (imputer + scaler + balanced solver) retrained every 6 hours
        - **Historical Window:** Most recent 90 days of completed games per league (25+ rows required)
        - **Output:** Win probability for each team, confidence score, edge calculation

        **How AI Adjusts Probabilities:**
        1. Collects API-Sports matchup summaries (record, form, points for/against) and sportsbook odds
        2. Trains a balanced logistic regression on recent outcomes once enough data is available
        3. Blends model output with market odds and sentiment (65% model • 25% market • 10% sentiment)
        4. Applies home-field baselines and API-Sports trend boosts
        5. Outputs adjusted probability + confidence and tracks the training sample size

        **Confidence Scoring:**
        - High (70%+): Strong signal from multiple factors
        - Medium (50-70%): Moderate signals, some uncertainty
        - Low (<50%): Conflicting signals or limited data

        **Fallback Mode:**
        - When fewer than 25 historical games exist or the dataset lacks both outcomes, the app reverts to the odds + sentiment heuristic and flags the "Historical ML" column accordingly.
        """)
    
    with st.expander("📊 Betting Trend Analysis"):
        st.markdown("""
        **Market Movements:**
        - Track line movements throughout the day
        - Identify sharp vs public money
        - Detect reverse line movement (line moves opposite to betting percentages)
        
        **Value Detection:**
        - Compare sentiment score to market odds
        - Find positive sentiment underdogs (best value)
        - Identify overvalued favorites with negative sentiment
        
        **Correlation Analysis:**
        - Sentiment vs actual outcomes (historical accuracy)
        - Best sports/leagues for sentiment analysis
        - Optimal bet types for sentiment-based betting
        """)
    
    with st.expander("🎯 How to Use This Analysis"):
        st.markdown("""
        **Step 1: Load & Analyze**
        1. Select sport and load current teams
        2. Choose 3-5 teams you're interested in
        3. Run deep analysis
        
        **Step 2: Interpret Results**
        - **🟢 Green (Positive)**: Team has favorable news/momentum
        - **🔴 Red (Negative)**: Team has concerning news/struggles  
        - **🟡 Yellow (Neutral)**: No strong sentiment signal
        
        **Step 3: Find Value**
        - Look for positive sentiment underdogs (market undervaluing)
        - Fade negative sentiment favorites (market overvaluing)
        - Combine with AI probability for best bets
        
        **Step 4: Validate**
        - Use Custom Parlay Builder to test specific picks
        - Compare sentiment analysis to AI expected value
        - Make informed decision based on multiple factors
        
        **Best Practices:**
        - Don't bet on sentiment alone - combine with AI analysis
        - Higher confidence scores are more reliable
        - More news sources = more reliable sentiment
        - Recent news (1-3 days) is most relevant
        """)
    
    # Tips
    st.markdown("---")
    st.markdown("""
    ### 💡 Sentiment Analysis Tips:
    
    **What Makes Strong Sentiment:**
    - ✅ Multiple news sources (5+ articles)
    - ✅ High confidence score (70%+)
    - ✅ Recent news (last 24-48 hours)
    - ✅ Consistent trend across sources
    
    **Red Flags:**
    - ⚠️ Low confidence (<40%)
    - ⚠️ Few news sources (<3 articles)
    - ⚠️ Mixed signals (some positive, some negative)
    - ⚠️ Old news (4+ days ago)
    
    **Best Use Cases:**
    - 🎯 Finding undervalued underdogs with positive momentum
    - 🎯 Fading overvalued favorites with negative news
    - 🎯 Validating your own analysis with AI/sentiment
    - 🎯 Identifying injury/coaching impacts quickly
    
    **Combine With:**
    - Use sentiment for initial screening
    - Use AI analysis for probability adjustment
    - Use Custom Parlay Builder to test combinations
    - Compare multiple data points before betting
    """)

with main_tab3:
    st.header("🎨 Custom Parlay Builder")
    st.markdown("**Build your own parlay and get AI-powered analysis**")
    st.caption("Select 2-4 legs, then get comprehensive AI/ML analysis with sentiment, probability, and edge calculations")
    
    # API key check
    api_key = st.session_state.get('api_key', "") or os.environ.get("ODDS_API_KEY", "")
    
    if not api_key:
        st.warning("⚠️ Please enter your Odds API key in the 'Sports Betting Parlays' tab first")
        st.stop()
    
    # Initialize session state for custom parlay legs
    if 'custom_legs' not in st.session_state:
        st.session_state['custom_legs'] = []
    
    st.markdown("---")
    
    # Step 1: Fetch Available Games
    st.subheader("📋 Step 1: Select Sport & Date")
    
    col_sport, col_date = st.columns(2)
    with col_sport:
        custom_sport = st.selectbox(
            "Sport",
            options=APP_CFG["sports_common"],
            key="custom_sport",
            format_func=format_sport_label,
        )
    with col_date:
        custom_date = st.date_input(
            "Game Date",
            value=pd.Timestamp.now().date(),
            key="custom_date"
        )
    
    if st.button("🔄 Load Games", type="primary"):
        sport_label = format_sport_label(custom_sport)
        with st.spinner(f"Loading {sport_label} games..."):
            try:
                snap = fetch_oddsapi_snapshot(api_key, custom_sport)
                st.session_state['available_games'] = snap.get("events", [])
                st.success(f"✅ Loaded {len(st.session_state.get('available_games', []))} games")
            except Exception as e:
                st.error(f"Error loading games: {str(e)}")
    
    # Step 2: Add Legs to Custom Parlay
    if 'available_games' in st.session_state and st.session_state['available_games']:
        st.markdown("---")
        st.subheader("⚡ Step 2: Build Your Parlay")
        
        games = st.session_state['available_games']
        
        # Game selector
        game_options = [f"{g['away_team']} @ {g['home_team']}" for g in games]
        selected_game_idx = st.selectbox(
            "Select Game",
            options=range(len(game_options)),
            format_func=lambda x: game_options[x],
            key="game_selector"
        )
        
        if selected_game_idx is not None:
            selected_game = games[selected_game_idx]
            home_team = selected_game['home_team']
            away_team = selected_game['away_team']
            markets = selected_game.get('markets', {})
            
            # Bet type selector
            col_bet1, col_bet2 = st.columns(2)
            with col_bet1:
                bet_type = st.selectbox(
                    "Bet Type",
                    options=["Moneyline", "Spread", "Total"],
                    key="bet_type_selector"
                )
            
            with col_bet2:
                if bet_type == "Moneyline":
                    h2h = markets.get('h2h', {})
                    home_price = _dig(h2h, 'home.price')
                    away_price = _dig(h2h, 'away.price')
                    
                    if home_price and away_price:
                        selection = st.selectbox(
                            "Selection",
                            options=[
                                f"{home_team} ML @{home_price:+.0f}",
                                f"{away_team} ML @{away_price:+.0f}"
                            ],
                            key="ml_selector"
                        )
                        
                        # Parse selection
                        if home_team in selection:
                            pick_team = home_team
                            pick_price = home_price
                            pick_side = "home"
                        else:
                            pick_team = away_team
                            pick_price = away_price
                            pick_side = "away"
                    else:
                        st.warning("Moneyline odds not available for this game")
                        selection = None
                
                elif bet_type == "Spread":
                    spreads = markets.get('spreads', [])
                    if spreads:
                        spread_options = []
                        for s in spreads[:8]:  # Show up to 8 spread options
                            team = s.get('name')
                            point = s.get('point')
                            price = s.get('price')
                            if team and point is not None and price:
                                spread_options.append({
                                    'label': f"{team} {point:+.1f} @{price:+.0f}",
                                    'team': team,
                                    'point': point,
                                    'price': price,
                                    'side': 'home' if team == home_team else 'away'
                                })
                        
                        if spread_options:
                            selection = st.selectbox(
                                "Selection",
                                options=range(len(spread_options)),
                                format_func=lambda x: spread_options[x]['label'],
                                key="spread_selector"
                            )
                            
                            if selection is not None:
                                pick_data = spread_options[selection]
                                pick_team = pick_data['team']
                                pick_price = pick_data['price']
                                pick_side = pick_data['side']
                                pick_point = pick_data['point']
                        else:
                            st.warning("Spread odds not available")
                            selection = None
                    else:
                        st.warning("Spread odds not available")
                        selection = None
                
                elif bet_type == "Total":
                    totals = markets.get('totals', [])
                    if totals:
                        total_options = []
                        for t in totals[:8]:  # Show up to 8 total options
                            name = t.get('name', '')
                            point = t.get('point')
                            price = t.get('price')
                            if point is not None and price:
                                total_options.append({
                                    'label': f"{name} {point} @{price:+.0f}",
                                    'name': name,
                                    'point': point,
                                    'price': price
                                })
                        
                        if total_options:
                            selection = st.selectbox(
                                "Selection",
                                options=range(len(total_options)),
                                format_func=lambda x: total_options[x]['label'],
                                key="total_selector"
                            )
                            
                            if selection is not None:
                                pick_data = total_options[selection]
                                pick_team = f"{home_team} vs {away_team}"
                                pick_price = pick_data['price']
                                pick_side = pick_data['name']
                                pick_point = pick_data['point']
                        else:
                            st.warning("Total odds not available")
                            selection = None
                    else:
                        st.warning("Total odds not available")
                        selection = None
            
            # Add leg button
            if selection is not None and st.button("➕ Add to Parlay", type="secondary"):
                # Create leg data
                leg = {
                    'event_id': selected_game['id'],
                    'game': f"{away_team} @ {home_team}",
                    'home_team': home_team,
                    'away_team': away_team,
                    'type': bet_type,
                    'team': pick_team,
                    'side': pick_side,
                    'price': pick_price,
                    'd': american_to_decimal_safe(pick_price)
                }
                
                if bet_type in ['Spread', 'Total']:
                    leg['point'] = pick_point
                
                # Check if already at max legs
                if len(st.session_state['custom_legs']) >= 4:
                    st.warning("⚠️ Maximum 4 legs allowed. Remove a leg to add another.")
                else:
                    # Check for duplicates (same game + same bet type)
                    duplicate = False
                    for existing_leg in st.session_state['custom_legs']:
                        if (existing_leg['event_id'] == leg['event_id'] and 
                            existing_leg['type'] == leg['type'] and
                            existing_leg['team'] == leg['team']):
                            duplicate = True
                            break
                    
                    if duplicate:
                        st.warning("⚠️ This leg is already in your parlay")
                    else:
                        st.session_state['custom_legs'].append(leg)
                        st.success(f"✅ Added to parlay!")
                        st.rerun()
    
    # Step 3: Show Current Parlay
    if st.session_state['custom_legs']:
        st.markdown("---")
        st.subheader("🎯 Your Custom Parlay")
        
        for i, leg in enumerate(st.session_state['custom_legs'], 1):
            col_leg, col_remove = st.columns([5, 1])
            with col_leg:
                label = f"{leg['game']} — {leg['team']}"
                if leg['type'] == 'Spread':
                    label += f" {leg['point']:+.1f}"
                elif leg['type'] == 'Total':
                    label += f" {leg['side']} {leg['point']}"
                label += f" @{leg['price']:+.0f}"
                
                st.write(f"**Leg {i}:** {label}")
            
            with col_remove:
                if st.button("🗑️", key=f"remove_{i}"):
                    st.session_state['custom_legs'].pop(i-1)
                    st.rerun()
        
        # Step 4: Analyze Parlay
        if len(st.session_state['custom_legs']) >= 2:
            st.markdown("---")
            st.subheader("🤖 AI Analysis")
            
            if st.button("🔍 Analyze My Parlay", type="primary"):
                with st.spinner("🧠 Running AI/ML analysis..."):
                    try:
                        sentiment_analyzer = st.session_state.get('sentiment_analyzer')
                        ml_predictor = st.session_state.get('ml_predictor')
                        ai_optimizer = st.session_state.get('ai_optimizer')
                        
                        # Enhance legs with AI analysis
                        analyzed_legs = []
                        for leg in st.session_state['custom_legs']:
                            # Get sentiment for teams
                            try:
                                home_sentiment = sentiment_analyzer.get_team_sentiment(
                                    leg['home_team'], custom_sport
                                ) if sentiment_analyzer else {'score': 0, 'trend': 'neutral'}
                                away_sentiment = sentiment_analyzer.get_team_sentiment(
                                    leg['away_team'], custom_sport
                                ) if sentiment_analyzer else {'score': 0, 'trend': 'neutral'}
                            except Exception:
                                home_sentiment = {'score': 0, 'trend': 'neutral'}
                                away_sentiment = {'score': 0, 'trend': 'neutral'}
                            
                            # Calculate probabilities
                            base_prob = implied_p_from_american(leg['price'])
                            
                            # Get AI probability
                            if leg['type'] == 'Moneyline' and ml_predictor:
                                # Get opponent price for ML prediction
                                opp_price = None
                                home_price = None
                                away_price = None
                                for g in st.session_state['available_games']:
                                    if g['id'] == leg['event_id']:
                                        h2h = g.get('markets', {}).get('h2h', {}) or {}
                                        home_price = _dig(h2h, 'home.price')
                                        away_price = _dig(h2h, 'away.price')
                                        if leg['side'] == 'home':
                                            opp_price = away_price
                                        else:
                                            opp_price = home_price
                                        break

                                if opp_price:
                                    ml_context = {
                                        "sport_key": custom_sport,
                                        "event_id": leg['event_id'],
                                    }
                                    apisports_payload = leg.get('apisports')
                                    if isinstance(apisports_payload, dict):
                                        if leg['side'] == 'home':
                                            ml_context['apisports_home'] = apisports_payload
                                        elif leg['side'] == 'away':
                                            ml_context['apisports_away'] = apisports_payload
                                    sportsdata_payload = leg.get('sportsdata')
                                    if isinstance(sportsdata_payload, dict):
                                        if leg['side'] == 'home':
                                            ml_context['sportsdata_home'] = sportsdata_payload
                                        elif leg['side'] == 'away':
                                            ml_context['sportsdata_away'] = sportsdata_payload
                                    ml_prediction = ml_predictor.predict_game_outcome(
                                        leg['home_team'],
                                        leg['away_team'],
                                        home_price,
                                        away_price,
                                        home_sentiment['score'],
                                        away_sentiment['score'],
                                        context=ml_context,
                                    )
                                    ai_prob = ml_prediction[f"{leg['side']}_prob"]
                                    ai_confidence = ml_prediction['confidence']
                                    ai_edge = ml_prediction['edge']
                                    ai_model_source = ml_prediction.get('model_used')
                                    ai_training_rows = ml_prediction.get('training_rows')
                                else:
                                    ai_prob = base_prob
                                    ai_confidence = 0.5
                                    ai_edge = 0
                                    ai_model_source = None
                                    ai_training_rows = None
                            else:
                                # For spreads/totals, use sentiment adjustment
                                sentiment = home_sentiment if leg['team'] == leg['home_team'] else away_sentiment
                                if leg['type'] == 'Total':
                                    sentiment = {'score': (home_sentiment['score'] + away_sentiment['score']) / 2}
                                
                                ai_prob = base_prob * (1 + sentiment['score'] * 0.40)
                                ai_prob = max(0.1, min(0.9, ai_prob))
                                ai_confidence = 0.65 if leg['type'] == 'Spread' else 0.60
                                ai_edge = abs(ai_prob - base_prob)
                            
                            # Add analysis to leg
                            analyzed_leg = leg.copy()
                            analyzed_leg.update({
                                'p': base_prob,
                                'ai_prob': ai_prob,
                                'ai_confidence': ai_confidence,
                                'ai_edge': ai_edge,
                                'ai_model_source': ai_model_source,
                                'ai_training_rows': ai_training_rows,
                                'sentiment_trend': home_sentiment['trend'] if leg['side'] == 'home' else away_sentiment['trend'],
                                'home_sentiment': home_sentiment,
                                'away_sentiment': away_sentiment
                            })
                            analyzed_legs.append(analyzed_leg)
                        
                        # Calculate parlay metrics
                        combined_odds = 1.0
                        market_prob = 1.0
                        ai_prob = 1.0
                        
                        for leg in analyzed_legs:
                            combined_odds *= leg['d']
                            market_prob *= leg['p']
                            ai_prob *= leg['ai_prob']
                        
                        # Get AI optimizer score
                        if ai_optimizer:
                            ai_metrics = ai_optimizer.score_parlay(analyzed_legs)
                        else:
                            ai_metrics = {
                                'score': 50,
                                'ai_ev': (ai_prob * combined_odds) - 1.0,
                                'confidence': sum(l['ai_confidence'] for l in analyzed_legs) / len(analyzed_legs),
                                'edge': sum(l['ai_edge'] for l in analyzed_legs)
                            }
                        
                        # Calculate payouts
                        stake = 100
                        potential_payout = stake * combined_odds
                        potential_profit = potential_payout - stake
                        
                        # Market EV
                        market_ev = (market_prob * combined_odds) - 1.0
                        market_expected_return = stake * market_ev
                        
                        # AI EV
                        ai_expected_return = stake * ai_metrics['ai_ev']
                        
                        # Display Results
                        st.markdown("### 📊 Analysis Results")
                        
                        # Main metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Combined Odds", f"{combined_odds:.2f}x")
                        with col2:
                            st.metric("Market Probability", f"{market_prob*100:.1f}%")
                        with col3:
                            st.metric("AI Probability", f"{ai_prob*100:.1f}%")
                        with col4:
                            confidence_color = "🟢" if ai_metrics['confidence'] > 0.7 else ("🟡" if ai_metrics['confidence'] > 0.5 else "🟠")
                            st.metric("AI Confidence", f"{confidence_color} {ai_metrics['confidence']*100:.0f}%")
                        
                        # EV metrics
                        col5, col6, col7 = st.columns(3)
                        with col5:
                            st.metric(
                                "Market EV",
                                f"${market_expected_return:+.2f}",
                                help="Expected value based on market (no-vig) probabilities"
                            )
                        with col6:
                            ev_delta = ai_expected_return - market_expected_return
                            st.metric(
                                "AI EV",
                                f"${ai_expected_return:+.2f}",
                                delta=ev_delta,
                                help="Expected value based on AI-adjusted probabilities"
                            )
                        with col7:
                            st.metric("AI Score", f"{ai_metrics['score']:.1f}")
                        
                        # Sentiment Analysis
                        st.markdown("### 🎭 Sentiment Analysis")
                        for i, leg in enumerate(analyzed_legs, 1):
                            with st.expander(f"Leg {i}: {leg['game']}"):
                                col_sent1, col_sent2 = st.columns(2)
                                with col_sent1:
                                    st.write(f"**{leg['home_team']}**")
                                    st.write(f"Sentiment: {leg['home_sentiment']['trend'].upper()}")
                                    st.write(f"Score: {leg['home_sentiment']['score']:+.2f}")
                                with col_sent2:
                                    st.write(f"**{leg['away_team']}**")
                                    st.write(f"Sentiment: {leg['away_sentiment']['trend'].upper()}")
                                    st.write(f"Score: {leg['away_sentiment']['score']:+.2f}")
                        
                        # Detailed Leg Analysis
                        st.markdown("### 📋 Leg-by-Leg Breakdown")
                        leg_data = []
                        for i, leg in enumerate(analyzed_legs, 1):
                            leg_data.append({
                                "Leg": i,
                                "Game": leg['game'],
                                "Pick": f"{leg['team']} {leg['type']}",
                                "Odds": f"{leg['price']:+.0f}",
                                "Market %": f"{leg['p']*100:.1f}%",
                                "AI %": f"{leg['ai_prob']*100:.1f}%",
                                "AI Confidence": f"{leg['ai_confidence']*100:.0f}%",
                                "Edge": f"{leg['ai_edge']*100:+.1f}%",
                                "Sentiment": leg['sentiment_trend']
                            })
                        
                        st.dataframe(pd.DataFrame(leg_data), use_container_width=True, hide_index=True)
                        
                        # Payout Scenarios
                        st.markdown("### 💰 Payout Scenarios")
                        st.write(f"**On a $100 bet:**")
                        st.write(f"- Total Payout: **${potential_payout:.2f}**")
                        st.write(f"- Profit: **${potential_profit:.2f}**")
                        st.write(f"- ROI: **{(potential_profit/stake)*100:.1f}%**")
                        
                        st.markdown("**Other Stakes:**")
                        for bet_amount in [25, 50, 100, 250, 500]:
                            payout = bet_amount * combined_odds
                            profit = payout - bet_amount
                            st.write(f"${bet_amount} → ${payout:.2f} (${profit:+.2f} profit)")
                        
                        # Recommendation
                        st.markdown("### 💡 AI Recommendation")
                        
                        # Decision logic
                        if ai_expected_return > 5 and ai_metrics['confidence'] > 0.65:
                            st.success("🟢 **STRONG PLAY** - Positive AI EV with high confidence")
                        elif ai_expected_return > 0 and ai_metrics['confidence'] > 0.55:
                            st.info("🟡 **CONSIDER** - Slight positive AI EV with moderate confidence")
                        elif market_expected_return > 0:
                            st.warning("🟠 **CAUTION** - Market EV positive but AI less confident")
                        else:
                            st.error("🔴 **AVOID** - Negative expected value")
                        
                        # Key insights
                        st.markdown("**Key Insights:**")
                        insights = []
                        
                        if ai_prob > market_prob * 1.1:
                            insights.append(f"✅ AI sees {((ai_prob/market_prob-1)*100):.0f}% better chance than market")
                        elif ai_prob < market_prob * 0.9:
                            insights.append(f"⚠️ AI sees {((1-ai_prob/market_prob)*100):.0f}% worse chance than market")
                        
                        if ai_metrics['confidence'] > 0.7:
                            insights.append("✅ High AI confidence across all legs")
                        elif ai_metrics['confidence'] < 0.5:
                            insights.append("⚠️ Low AI confidence - consider alternative picks")
                        
                        positive_sentiment = sum(1 for leg in analyzed_legs if leg['sentiment_trend'] == 'positive')
                        if positive_sentiment == len(analyzed_legs):
                            insights.append("✅ All picks have positive sentiment")
                        elif positive_sentiment == 0:
                            insights.append("⚠️ No picks have positive sentiment")
                        
                        for insight in insights:
                            st.write(insight)
                        
                        # Download option
                        st.markdown("---")
                        analysis_data = {
                            "Parlay": [f"Custom {len(analyzed_legs)}-Leg"],
                            "Combined Odds": [f"{combined_odds:.2f}"],
                            "Market Probability": [f"{market_prob*100:.1f}%"],
                            "AI Probability": [f"{ai_prob*100:.1f}%"],
                            "AI Confidence": [f"{ai_metrics['confidence']*100:.0f}%"],
                            "Market EV": [f"${market_expected_return:+.2f}"],
                            "AI EV": [f"${ai_expected_return:+.2f}"],
                            "AI Score": [f"{ai_metrics['score']:.1f}"]
                        }
                        
                        csv_buf = io.StringIO()
                        pd.DataFrame(analysis_data).to_csv(csv_buf, index=False)
                        st.download_button(
                            "💾 Download Analysis",
                            data=csv_buf.getvalue(),
                            file_name="custom_parlay_analysis.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error analyzing parlay: {str(e)}")
                        with st.expander("Error Details"):
                            import traceback
                            st.code(traceback.format_exc())
        else:
            st.info("ℹ️ Add at least 2 legs to analyze your parlay")
        
        # Clear parlay button
        if st.button("🗑️ Clear All Legs", type="secondary"):
            st.session_state['custom_legs'] = []
            st.rerun()
    else:
        st.info("👆 Load games and start building your parlay above")
    
    # Tips section
    st.markdown("---")
    st.markdown("""
    ### 💡 Custom Parlay Tips:
    
    **Building Your Parlay:**
    - Start with 2-3 legs for better win probability
    - Mix different bet types (ML, Spread, Total) for variety
    - Avoid same-game parlays unless you have strong correlation thesis
    
    **Using AI Analysis:**
    - **Positive AI EV** = AI thinks you have an edge
    - **High Confidence (70%+)** = AI very sure about probabilities
    - **Positive Sentiment** = Recent news favors this pick
    - **Green AI Score** = Strong overall recommendation
    
    **Making Decisions:**
    - ✅ Target: Positive AI EV + High Confidence + Good Sentiment
    - ⚠️ Caution: Negative AI EV or Low Confidence
    - 🔴 Avoid: Multiple red flags (negative EV, low confidence, bad sentiment)
    """)

# ===== TAB 4: KALSHI PREDICTION MARKETS =====
with main_tab4:
    st.header("📊 Kalshi Prediction Markets")
    st.markdown("**Compare prediction market odds with traditional sportsbooks and AI analysis**")
    st.caption("Find arbitrage opportunities and value bets by comparing Kalshi's wisdom-of-crowds pricing with sportsbook odds")
    
    # API Configuration
    st.markdown("---")
    st.subheader("⚙️ Kalshi Configuration")
    
    col_kalshi1, col_kalshi2 = st.columns(2)
    with col_kalshi1:
        kalshi_key = st.text_input(
            "Kalshi API Key",
            value=st.session_state.get('kalshi_api_key', os.environ.get("KALSHI_API_KEY", "")),
            type="password",
            help="Get your API key from https://kalshi.com/account/api"
        )
        if kalshi_key != st.session_state.get('kalshi_api_key', ''):
            st.session_state['kalshi_api_key'] = kalshi_key
            st.session_state['kalshi_integrator'] = KalshiIntegrator(
                kalshi_key, 
                st.session_state.get('kalshi_api_secret', '')
            )
    
    with col_kalshi2:
        kalshi_secret = st.text_input(
            "Kalshi API Secret (optional)",
            value=st.session_state.get('kalshi_api_secret', os.environ.get("KALSHI_API_SECRET", "")),
            type="password",
            help="API secret for authenticated requests"
        )
        if kalshi_secret != st.session_state.get('kalshi_api_secret', ''):
            st.session_state['kalshi_api_secret'] = kalshi_secret
            st.session_state['kalshi_integrator'] = KalshiIntegrator(
                st.session_state.get('kalshi_api_key', ''),
                kalshi_secret
            )
    
    if not kalshi_key:
        st.info("💡 **Demo Mode:** You can explore Kalshi without API keys. For live trading, get your API key at [kalshi.com](https://kalshi.com)")
    
    st.markdown("---")
    
    # Main Analysis Sections
    analysis_mode = st.radio(
        "Select Analysis Mode:",
        ["🔍 Browse Kalshi Sports Markets", "⚖️ Compare with Sportsbooks", "💎 Find Arbitrage Opportunities"],
        horizontal=True
    )
    
    kalshi = st.session_state.get('kalshi_integrator')
    
    if analysis_mode == "🔍 Browse Kalshi Sports Markets":
        st.subheader("🏈 Available Sports Betting Markets")
        
        if st.button("🔄 Load Kalshi Markets", type="primary"):
            with st.spinner("Fetching Kalshi markets..."):
                try:
                    markets = kalshi.get_sports_markets()
                    st.session_state['kalshi_markets'] = markets
                    st.success(f"✅ Loaded {len(markets)} sports markets")
                    if kalshi.using_synthetic_data():
                        st.warning("🧪 Live Kalshi API unavailable – showing synthetic demo markets instead.")
                        if kalshi.last_error:
                            st.caption(f"Last API error: {kalshi.last_error}")
                except Exception as e:
                    st.error(f"Error loading markets: {str(e)}")
                    st.info("💡 Try demo mode without API keys to explore sample markets")

        if 'kalshi_markets' in st.session_state and st.session_state['kalshi_markets']:
            markets = st.session_state['kalshi_markets']

            if kalshi and kalshi.using_synthetic_data():
                st.info("🧪 Displaying locally generated Kalshi fallback data for exploration.")

            st.markdown(f"### 📋 {len(markets)} Markets Available")
            
            # Filter options
            col_filter1, col_filter2 = st.columns(2)
            with col_filter1:
                sport_filter = st.selectbox(
                    "Filter by Sport",
                    options=["All"] + ["NFL", "NBA", "MLB", "NHL", "UFC", "Soccer"],
                    key="kalshi_sport_filter"
                )
            
            with col_filter2:
                sort_by = st.selectbox(
                    "Sort by",
                    options=["Volume (High to Low)", "Close Date (Soonest)", "Title (A-Z)"],
                    key="kalshi_sort"
                )
            
            # Filter markets
            filtered_markets = markets
            if sport_filter != "All":
                filtered_markets = [m for m in markets if sport_filter.upper() in m.get('title', '').upper()]
            
            # Sort markets
            if sort_by == "Volume (High to Low)":
                filtered_markets.sort(key=lambda x: x.get('volume', 0), reverse=True)
            elif sort_by == "Close Date (Soonest)":
                filtered_markets.sort(key=lambda x: x.get('close_time', ''))
            else:
                filtered_markets.sort(key=lambda x: x.get('title', ''))
            
            # Display markets
            for i, market in enumerate(filtered_markets[:20]):  # Show top 20
                title = market.get('title', 'Unknown Market')
                ticker = market.get('ticker', '')
                volume = market.get('volume', 0)
                
                with st.expander(f"**{i+1}. {title}**"):
                    col_m1, col_m2 = st.columns(2)
                    
                    with col_m1:
                        st.write(f"**Ticker:** {ticker}")
                        st.write(f"**Volume:** {volume:,} contracts")
                        st.write(f"**Status:** {market.get('status', 'unknown')}")
                    
                    with col_m2:
                        # Get orderbook for this market
                        button_key = f"analyze_{ticker}_{i}"
                        if st.button(f"📊 Analyze {ticker[:15]}...", key=button_key):
                            with st.spinner("Fetching market details..."):
                                try:
                                    orderbook = kalshi.get_orderbook(ticker)
                                    
                                    if orderbook:
                                        yes_bids = orderbook.get('yes', [])
                                        no_bids = orderbook.get('no', [])
                                        
                                        if yes_bids:
                                            best_yes_bid = yes_bids[0].get('price', 0) / 100
                                            st.success(f"**YES Price:** {best_yes_bid*100:.1f}¢ ({best_yes_bid*100:.1f}% probability)")
                                        
                                        if no_bids:
                                            best_no_bid = no_bids[0].get('price', 0) / 100
                                            st.info(f"**NO Price:** {best_no_bid*100:.1f}¢ ({best_no_bid*100:.1f}% probability)")
                                        
                                        # Kelly recommendation
                                        if yes_bids:
                                            st.markdown("**💰 Kelly Sizing:**")
                                            # Assume user has 55% confidence (can be adjusted)
                                            user_prob = 0.55
                                            kelly_result = kalshi.calculate_kelly_for_kalshi(
                                                best_yes_bid, user_prob, 1000
                                            )
                                            st.write(kelly_result['recommendation'])
                                    else:
                                        st.warning("No orderbook data available")
                                
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
            
            if len(filtered_markets) > 20:
                st.info(f"Showing top 20 of {len(filtered_markets)} markets. Adjust filters to see others.")
        
        else:
            st.info("👆 Click 'Load Kalshi Markets' to see available betting opportunities")
    
    elif analysis_mode == "⚖️ Compare with Sportsbooks":
        st.subheader("⚖️ Kalshi vs Sportsbook Odds Comparison")
        st.markdown("Compare prediction market pricing with traditional sportsbook odds to find value")
        
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            st.markdown("#### Kalshi Market")
            kalshi_market_title = st.text_input(
                "Market Title",
                placeholder="e.g., Will Chiefs win Super Bowl?",
                key="kalshi_market_compare"
            )
            kalshi_yes_price = st.number_input(
                "YES Price (cents)",
                min_value=0.0,
                max_value=100.0,
                value=65.0,
                step=0.1,
                help="Price in cents (e.g., 65 = 65% probability)"
            ) / 100
        
        with col_comp2:
            st.markdown("#### Sportsbook")
            sb_selection = st.text_input(
                "Corresponding Bet",
                placeholder="e.g., Chiefs Super Bowl Winner",
                key="sb_compare"
            )
            sb_odds = st.number_input(
                "Sportsbook Odds (American)",
                min_value=-10000,
                max_value=10000,
                value=-150,
                step=5,
                help="American odds (e.g., -150, +200)"
            )
        
        if st.button("🔍 Compare Markets", type="primary"):
            # Calculate sportsbook implied probability
            sb_prob = implied_p_from_american(sb_odds)
            
            # Comparison
            discrepancy = abs(kalshi_yes_price - sb_prob)
            edge = kalshi_yes_price - sb_prob
            
            st.markdown("---")
            st.markdown("### 📊 Comparison Results")
            
            col_r1, col_r2, col_r3 = st.columns(3)
            
            with col_r1:
                st.metric("Kalshi Probability", f"{kalshi_yes_price*100:.1f}%")
            with col_r2:
                st.metric("Sportsbook Probability", f"{sb_prob*100:.1f}%")
            with col_r3:
                st.metric("Discrepancy", f"{discrepancy*100:.1f}%", 
                         delta=f"{edge*100:+.1f}%" if edge != 0 else None)
            
            # Recommendation
            st.markdown("### 💡 Recommendation")
            
            if discrepancy > 0.10:  # 10%+ difference
                if kalshi_yes_price < sb_prob:
                    st.success(f"🟢 **STRONG VALUE on Kalshi YES**")
                    st.write(f"- Kalshi is pricing YES at {kalshi_yes_price*100:.1f}%")
                    st.write(f"- Sportsbook implies {sb_prob*100:.1f}%")
                    st.write(f"- **Edge: {(sb_prob - kalshi_yes_price)*100:.1f}% in your favor**")
                    st.write(f"- **Action:** Buy YES on Kalshi")
                else:
                    st.success(f"🟢 **STRONG VALUE on Sportsbook**")
                    st.write(f"- Kalshi is overpricing at {kalshi_yes_price*100:.1f}%")
                    st.write(f"- Sportsbook implies {sb_prob*100:.1f}%")
                    st.write(f"- **Edge: {(kalshi_yes_price - sb_prob)*100:.1f}%**")
                    st.write(f"- **Action:** Take sportsbook bet OR buy NO on Kalshi")
            
            elif discrepancy > 0.05:  # 5-10% difference
                if kalshi_yes_price < sb_prob:
                    st.info(f"🟡 **MODERATE VALUE on Kalshi YES**")
                    st.write(f"- Small edge of {(sb_prob - kalshi_yes_price)*100:.1f}%")
                    st.write(f"- Consider Kalshi YES if you agree with sportsbook assessment")
                else:
                    st.info(f"🟡 **MODERATE VALUE on Sportsbook**")
                    st.write(f"- Small edge of {(kalshi_yes_price - sb_prob)*100:.1f}%")
                    st.write(f"- Consider sportsbook if you agree with that probability")
            
            else:
                st.success("✅ **MARKETS IN AGREEMENT**")
                st.write(f"- Both markets pricing very similarly")
                st.write(f"- Difference of only {discrepancy*100:.1f}%")
                st.write(f"- No significant arbitrage or value opportunity")
                st.write(f"- Bet on either if you have additional information/analysis")
            
            # Kelly calculation for Kalshi
            st.markdown("### 💰 Optimal Bet Sizing (Kalshi)")
            
            user_confidence = st.slider(
                "Your Confidence Level",
                min_value=0.0,
                max_value=1.0,
                value=sb_prob,
                step=0.01,
                format="%.1f%%",
                help="Your personal assessment of the probability"
            )
            
            bankroll = st.number_input(
                "Your Bankroll",
                min_value=100,
                max_value=1000000,
                value=1000,
                step=100
            )
            
            kelly = kalshi.calculate_kelly_for_kalshi(kalshi_yes_price, user_confidence, bankroll)
            
            col_k1, col_k2, col_k3 = st.columns(3)
            with col_k1:
                st.metric("Kelly %", f"{kelly['kelly_percentage']:.2f}%")
            with col_k2:
                st.metric("Recommended Stake", f"${kelly['recommended_stake']:.0f}")
            with col_k3:
                st.metric("Expected Value", f"{kelly['expected_value']*100:+.1f}%")
            
            st.write(kelly['recommendation'])
    
    elif analysis_mode == "💎 Find Arbitrage Opportunities":
        st.subheader("💎 Arbitrage Opportunity Scanner")
        st.markdown("Automatically find discrepancies between Kalshi and traditional sportsbooks")
        
        st.info("🔧 **Coming Soon:** This feature will automatically scan all markets and identify arbitrage opportunities where you can profit regardless of outcome by betting both sides.")
        
        st.markdown("""
        **How Arbitrage Works:**
        
        1. **Find Discrepancy:** Kalshi prices YES at 40% but sportsbook implies 50%
        2. **Bet Both Sides:** 
           - Bet YES on Kalshi (40¢)
           - Bet NO equivalent on sportsbook
        3. **Lock Profit:** Guaranteed profit regardless of outcome
        
        **Requirements:**
        - Discrepancy must be > 10% (to cover fees)
        - Sufficient liquidity on both sides
        - Fast execution (prices move quickly)
        
        **Manual Search:**
        Use the "Compare with Sportsbooks" tab above to manually check for arbitrage opportunities.
        """)
    
    # Educational Section
    st.markdown("---")
    st.markdown("### 📚 Understanding Kalshi Prediction Markets")
    
    with st.expander("🤔 What is Kalshi?"):
        st.markdown("""
        **Kalshi** is a CFTC-regulated prediction market where you can trade on real-world events:
        
        - **Legal & Regulated:** First CFTC-regulated event contract exchange in the US
        - **Binary Outcomes:** Markets settle to either 0¢ or 100¢
        - **Pricing:** Prices represent probability (65¢ = 65% chance)
        - **Liquidity:** Limit orderbook like stocks
        
        **Example:**
        - Market: "Will Chiefs win their next game?"
        - YES trading at 70¢ = Market thinks 70% chance
        - If you buy YES at 70¢ and Chiefs win, you get 100¢ (30¢ profit)
        - If they lose, you get 0¢ (lose your 70¢)
        """)
    
    with st.expander("💡 Why Compare with Sportsbooks?"):
        st.markdown("""
        **Different Pricing Mechanisms:**
        
        **Kalshi (Prediction Market):**
        - Wisdom of crowds pricing
        - Real money at stake
        - Efficient market hypothesis
        - Can be slower to react to news
        
        **Sportsbooks:**
        - Built-in vig (~5-10%)
        - Designed to balance book
        - React quickly to sharp money
        - Public bias can skew lines
        
        **Opportunities:**
        - ✅ Kalshi often has better prices (lower vig)
        - ✅ Arbitrage when markets disagree significantly
        - ✅ Value bets when you trust one pricing over the other
        - ✅ Hedge existing positions
        """)
    
    with st.expander("🎯 How to Use This Tool"):
        st.markdown("""
        **Step 1: Browse Markets**
        - Load Kalshi sports markets
        - Find events you're interested in
        - Check current pricing
        
        **Step 2: Compare Prices**
        - Find the same event on a sportsbook
        - Use "Compare with Sportsbooks" tab
        - Look for 5%+ discrepancies
        
        **Step 3: Make Decision**
        - If Kalshi < Sportsbook: Buy YES on Kalshi
        - If Kalshi > Sportsbook: Take sportsbook OR buy NO on Kalshi
        - If similar: Use your own analysis to decide
        
        **Step 4: Size Optimally**
        - Use Kelly Calculator
        - Input your confidence level
        - Bet recommended amount
        
        **Step 5: Track & Learn**
        - Monitor your positions
        - See which market was more accurate
        - Refine your strategy
        """)
    
    with st.expander("⚠️ Important Considerations"):
        st.markdown("""
        **Advantages of Kalshi:**
        - ✅ Lower fees than sportsbooks
        - ✅ Can exit position early (sell before event)
        - ✅ Legal in most US states
        - ✅ No betting limits (unlike sportsbooks)
        
        **Disadvantages:**
        - ⚠️ Lower liquidity than sportsbooks
        - ⚠️ Spreads can be wide on low-volume markets
        - ⚠️ Fewer markets available
        - ⚠️ Funds take time to withdraw
        
        **Best Practices:**
        - Only trade on high-volume markets
        - Check the spread before entering
        - Start small to learn the platform
        - Compare fees with sportsbook vig
        - Consider exit liquidity
        """)
    
    # Tips section
    st.markdown("---")
    st.markdown("""
    ### 💡 Kalshi Trading Tips:
    
    **Finding Value:**
    - ✅ Look for 5%+ discrepancies with sportsbooks
    - ✅ Check multiple sportsbooks for best comparison
    - ✅ Use AI analysis from other tabs to inform your view
    - ✅ Focus on high-volume markets (easier exit)
    
    **Risk Management:**
    - ✅ Use Kelly Criterion for position sizing
    - ✅ Don't tie up too much capital (lower liquidity)
    - ✅ Set maximum position sizes
    - ✅ Consider exit strategy before entering
    
    **Advanced Strategies:**
    - 📈 Arbitrage between Kalshi and sportsbooks
    - 📈 Hedge existing sportsbook bets on Kalshi
    - 📈 Take early profits by selling before event
    - 📈 Buy when news breaks before market adjusts

    **Combining with AI:**
    - Use Tab 2 sentiment analysis to validate Kalshi prices
    - Use Tab 3 custom builder to calculate fair value
    - Compare AI probability with Kalshi pricing
    - Bet when AI and Kalshi agree on value
    """)

# ===== TAB 5: API-SPORTS LIVE DATA =====
with main_tab5:
    apisports_client = st.session_state.get('apisports_nfl_client')
    if apisports_client is None:
        fallback_key, fallback_source = resolve_nfl_apisports_key()
        apisports_client = APISportsFootballClient(
            fallback_key or None,
            key_source=fallback_source,
        )
        st.session_state['apisports_nfl_client'] = apisports_client

    sportsdata_clients = ensure_sportsdata_clients()

    hockey_client = st.session_state.get('apisports_hockey_client')
    if hockey_client is None:
        fallback_key, fallback_source = resolve_nhl_apisports_key()
        hockey_client = APISportsHockeyClient(
            fallback_key or None,
            key_source=fallback_source,
        )
        st.session_state['apisports_hockey_client'] = hockey_client

    basketball_client = st.session_state.get('apisports_basketball_client')
    if basketball_client is None:
        fallback_key, fallback_source = resolve_nba_apisports_key()
        basketball_client = APISportsBasketballClient(
            fallback_key or None,
            key_source=fallback_source,
        )
        st.session_state['apisports_basketball_client'] = basketball_client

    st.header("🛰️ Live Data Feeds")
    st.markdown("**Blend API-Sports schedules with SportsData.io power metrics for richer context across supported leagues.**")

    league_choice = st.selectbox(
        "League",
        options=["NFL", "NBA", "NHL"],
        key="apisports_live_league",
    )

    league_config = {
        "NFL": {
            "client": apisports_client,
            "caption": "Provide your NFL API-Sports key in the Sports Betting tab to enable these insights.",
            "warning": "Add your NFL API-Sports key in the Sports Betting tab to load live NFL data.",
            "button": "Fetch NFL games",
            "spinner": "Loading NFL schedule from API-Sports...",
            "no_games": "No NFL games found for this date.",
            "emoji": "🏈",
        },
        "NBA": {
            "client": basketball_client,
            "caption": "Provide your NBA API-Sports key in the Sports Betting tab to enable these insights.",
            "warning": "Add your NBA API-Sports key in the Sports Betting tab to load live NBA data.",
            "button": "Fetch NBA games",
            "spinner": "Loading NBA schedule from API-Sports...",
            "no_games": "No NBA games found for this date.",
            "emoji": "🏀",
        },
        "NHL": {
            "client": hockey_client,
            "caption": "Provide your NHL API-Sports key in the Sports Betting tab to enable these insights.",
            "warning": "Add your NHL API-Sports key in the Sports Betting tab to load live NHL data.",
            "button": "Fetch NHL games",
            "spinner": "Loading NHL schedule from API-Sports...",
            "no_games": "No NHL games found for this date.",
            "emoji": "🏒",
        },
    }

    config = league_config.get(league_choice, league_config["NFL"])
    selected_client = config["client"]
    st.caption(config["caption"])

    if not selected_client or not selected_client.is_configured():
        st.warning(config["warning"])
    else:
        default_tz = st.session_state.get('user_timezone', 'America/New_York')
        tz_key = f"apisports_live_tz_{league_choice.lower()}"
        tz_input = st.text_input("Timezone (IANA)", value=default_tz, key=tz_key)
        try:
            tz_obj = pytz.timezone(tz_input)
        except Exception:
            tz_obj = pytz.timezone('UTC')
            st.warning("Invalid timezone. Using UTC for display.")

        date_key = f"apisports_live_date_{league_choice.lower()}"
        game_date = st.date_input(
            "Game date",
            value=pd.Timestamp.now(tz_obj).date(),
            key=date_key,
        )

        if st.button(config["button"], key=f"fetch_apisports_games_{league_choice.lower()}"):
            with st.spinner(config["spinner"]):
                games = selected_client.get_games_by_date(game_date, timezone=tz_input)

            if not games:
                if selected_client.last_error:
                    st.error(f"API-Sports error: {selected_client.last_error}")
                else:
                    st.info(config["no_games"])
            else:
                st.success(f"✅ Loaded {len(games)} games")
                for raw_game in games:
                    summary = selected_client.build_game_summary(raw_game, tz_name=tz_input)
                    home = summary.home
                    away = summary.away

                    st.markdown("---")
                    st.subheader(f"{config['emoji']} {away.name} @ {home.name}")

                    metric_label = summary.scoring_metric or 'points'
                    if isinstance(metric_label, str):
                        metric_lower = metric_label.lower()
                        metric_text = 'Pts' if metric_lower in ('points', 'point') else metric_label.title()
                    else:
                        metric_text = 'Pts'

                    col_meta, col_home, col_away = st.columns([2, 2, 2])
                    with col_meta:
                        st.write(f"Start: {summary.kickoff_local or 'TBD'}")
                        st.write(f"Status: {summary.status or 'Scheduled'}")
                        st.write(f"Venue: {summary.venue or 'TBD'}")
                        st.write(f"Stage: {summary.stage or 'Regular Season'}")
                    with col_home:
                        st.write(f"**Home: {home.name}**")
                        st.write(f"Record: {home.record or '—'}")
                        st.write(f"Form: {home.form or '—'}")
                        if home.average_points_for is not None:
                            st.write(f"{metric_text} For: {home.average_points_for:.1f}")
                        if home.average_points_against is not None:
                            st.write(f"{metric_text} Allowed: {home.average_points_against:.1f}")
                        if home.trend:
                            icon = {'hot': '🔥', 'cold': '🥶', 'neutral': '⚪️'}.get(home.trend, '📊')
                            st.write(f"Trend: {icon} {home.trend.capitalize()}")
                    with col_away:
                        st.write(f"**Away: {away.name}**")
                        st.write(f"Record: {away.record or '—'}")
                        st.write(f"Form: {away.form or '—'}")
                        if away.average_points_for is not None:
                            st.write(f"{metric_text} For: {away.average_points_for:.1f}")
                        if away.average_points_against is not None:
                            st.write(f"{metric_text} Allowed: {away.average_points_against:.1f}")
                        if away.trend:
                            icon = {'hot': '🔥', 'cold': '🥶', 'neutral': '⚪️'}.get(away.trend, '📊')
                            st.write(f"Trend: {icon} {away.trend.capitalize()}")

                if selected_client.last_error:
                    st.info(f"API-Sports notice: {selected_client.last_error}")

        st.markdown("---")
        st.subheader("📡 SportsData.io Snapshot")
        st.caption("Review SportsData.io power metrics, streaks, and turnover margins across supported leagues.")

        sd_options = [(cfg['label'], sport_key, cfg) for sport_key, cfg in SPORTSDATA_CONFIG.items()]
        sd_labels = [label for label, _, _ in sd_options]
        default_sd_index = 0
        selected_sd_label = st.selectbox(
            "SportsData.io league",
            options=sd_labels,
            index=default_sd_index,
            key="sportsdata_snapshot_league",
        )
        selected_sd_key = next(key for label, key, _ in sd_options if label == selected_sd_label)
        selected_sd_cfg = SPORTSDATA_CONFIG[selected_sd_key]
        sd_client = sportsdata_clients.get(selected_sd_key)

        if not sd_client or not sd_client.is_configured():
            st.warning(
                f"Add your SportsData.io {selected_sd_cfg['label']} key in the Sports Betting tab to enable this snapshot."
            )
        else:
            sd_tz_default = st.session_state.get('user_timezone', 'America/New_York')
            sd_tz_input = st.text_input(
                "Timezone (IANA)",
                value=sd_tz_default,
                key=f"sportsdata_live_tz_{selected_sd_key}",
            )
            try:
                sd_tz_obj = pytz.timezone(sd_tz_input)
            except Exception:
                sd_tz_obj = pytz.timezone('UTC')
                st.warning("Invalid timezone. Using UTC for SportsData.io snapshot display.")

            sd_date = st.date_input(
                "Game date (SportsData.io)",
                value=pd.Timestamp.now(sd_tz_obj).date(),
                key=f"sportsdata_live_date_{selected_sd_key}",
            )

            button_key = f"fetch_sportsdata_snapshot_{selected_sd_key}"
            if st.button(
                f"Fetch SportsData.io {selected_sd_cfg['label']} snapshot",
                key=button_key,
            ):
                with st.spinner(f"Loading {selected_sd_cfg['label']} data from SportsData.io..."):
                    sd_games = sd_client.get_scores_by_date(sd_date)

                if not sd_games:
                    if sd_client.last_error:
                        st.error(f"SportsData.io error: {sd_client.last_error}")
                    else:
                        st.info(f"No {selected_sd_cfg['label']} games returned for this date.")
                else:
                    st.success(f"✅ Loaded {len(sd_games)} game(s)")
                    for raw_game in sd_games:
                        insight = sd_client.build_game_insight(raw_game)
                        if not insight:
                            continue

                        st.markdown("---")
                        st.subheader(f"{selected_sd_cfg['emoji']} {insight.away.name} @ {insight.home.name}")
                        st.write(f"Kickoff: {insight.kickoff or 'TBD'}")
                        st.write(f"Status: {insight.status or 'Scheduled'}")
                        season_bits = [bit for bit in [insight.season, insight.season_type, insight.week] if bit]
                        if season_bits:
                            st.write("Season info: " + " • ".join(str(bit) for bit in season_bits))
                        if insight.stadium:
                            st.write(f"Venue: {insight.stadium}")

                        col_sd_home, col_sd_away = st.columns(2)

                        def _render_sd_team(column, label, team):
                            with column:
                                st.write(f"**{label}: {team.name}**")
                                if team.record:
                                    st.write(f"Record: {team.record}")
                                if team.streak:
                                    st.write(f"Streak: {team.streak}")
                                if team.trend:
                                    icon = {'hot': '🔥', 'cold': '🥶', 'neutral': '⚪️'}.get(team.trend, '📊')
                                    st.write(f"Trend: {icon} {team.trend.capitalize()}")
                                if team.net_points_per_game is not None:
                                    st.write(f"Net {selected_sd_cfg['label']} PPG: {team.net_points_per_game:+.1f}")
                                if team.turnover_margin is not None:
                                    st.write(f"TO Margin: {team.turnover_margin:+.1f}")
                                if team.power_index is not None:
                                    st.write(f"Power Index: {team.power_index:.2f}")

                        _render_sd_team(col_sd_home, "Home", insight.home)
                        _render_sd_team(col_sd_away, "Away", insight.away)

                    if sd_client.last_error:
                        st.info(f"SportsData.io notice: {sd_client.last_error}")

        st.markdown("---")
        st.subheader("🌐 API-Sports League Widget")
        st.caption(
            "Embed the official API-Sports widget to explore leagues beyond the configured sport using your key."
        )

        widget_key = (
            st.session_state.get('nfl_apisports_api_key')
            or st.session_state.get('nba_apisports_api_key')
            or st.session_state.get('nhl_apisports_api_key')
            or (apisports_client.api_key if apisports_client else "")
            or (basketball_client.api_key if basketball_client else "")
            or (hockey_client.api_key if hockey_client else "")
        )
        if not widget_key:
            st.info("Provide an NFL, NBA, or NHL API-Sports key in the Sports Betting tab to load the widget.")
        else:
            sport_labels = {
                "NFL": "nfl",
                "NBA": "nba",
                "MLB": "mlb",
                "NHL": "nhl",
                "NCAAB": "ncaab",
                "NCAAF": "ncaaf",
                "WNBA": "wnba",
                "MLS": "mls",
            }
            default_index = list(sport_labels.keys()).index(league_choice) if league_choice in sport_labels else 0
            selected_label = st.selectbox(
                "Widget sport",
                options=list(sport_labels.keys()),
                index=default_index,
                key="apisports_widget_sport",
                help="Choose which league the API-Sports widget should highlight.",
            )
            theme_label = st.selectbox(
                "Widget theme",
                options=["Light", "Dark"],
                index=0,
                key="apisports_widget_theme",
            )
            show_widget_errors = st.checkbox(
                "Show API error messages in widget",
                value=True,
                key="apisports_widget_errors",
            )
            widget_height = st.slider(
                "Widget height (px)",
                400,
                900,
                600,
                50,
                key="apisports_widget_height",
            )

            widget_theme = "white" if theme_label == "Light" else "dark"
            widget_sport = sport_labels.get(selected_label, "nfl")

            widget_html = f"""
            <div class=\"apisports-widget\">
              <script type=\"module\" src=\"https://widgets.api-sports.io/2.0.3/widgets.js\"></script>
              <api-sports-widget data-type=\"config\"
                data-key=\"{escape(widget_key)}\"
                data-sport=\"{escape(widget_sport)}\"
                data-lang=\"en\"
                data-theme=\"{escape(widget_theme)}\"
                data-show-errors=\"{str(show_widget_errors).lower()}\"
              ></api-sports-widget>
              <api-sports-widget data-type=\"leagues\"></api-sports-widget>
            </div>
            """
            components.html(widget_html, height=widget_height, scrolling=True)
