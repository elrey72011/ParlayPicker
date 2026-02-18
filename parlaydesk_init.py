"""Initialization helpers for ParlayDesk Streamlit app.

Responsible for safe, single-source setup of clients and session state.
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

# Ensure repo root on path for Streamlit Cloud execution
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # Lightweight import guards
    from app_core.odds_api import TheOddsAPIClient
except Exception:  # pragma: no cover - fallback for missing dependency
    TheOddsAPIClient = None  # type: ignore

try:
    from app_core import (
        APISportsBasketballClient,
        APISportsFootballClient,
        APISportsHockeyClient,
        RealSentimentAnalyzer,
        SentimentAnalyzer,
        SportsDataNCAABClient,
        SportsDataNCAAFClient,
        SportsDataNBAClient,
        SportsDataNFLClient,
        SportsDataNHLClient,
        KalshiIntegrator,
    )
except Exception:
    APISportsBasketballClient = None  # type: ignore
    APISportsFootballClient = None  # type: ignore
    APISportsHockeyClient = None  # type: ignore
    SportsDataNCAABClient = None  # type: ignore
    SportsDataNCAAFClient = None  # type: ignore
    SportsDataNBAClient = None  # type: ignore
    SportsDataNFLClient = None  # type: ignore
    SportsDataNHLClient = None  # type: ignore
    RealSentimentAnalyzer = None  # type: ignore
    SentimentAnalyzer = None  # type: ignore
    KalshiIntegrator = None  # type: ignore

try:
    from google.oauth2 import service_account
    import vertexai
except Exception:
    service_account = None  # type: ignore
    vertexai = None  # type: ignore


def make_key(*parts: Any) -> str:
    """Create deterministic widget keys with minimal collision risk."""
    cleaned = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, (int, float)):
            cleaned.append(str(part))
        else:
            cleaned.append(str(part).strip().replace(" ", "_"))
    if not cleaned:
        return "key"
    return "-".join(cleaned)


def init_logger() -> logging.Logger:
    logger = logging.getLogger("parlaydesk")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _build_odds_client(logger: logging.Logger) -> Optional[Any]:
    api_key = st.secrets.get("ODDS_API_KEY") or st.secrets.get("odds_api_key")
    if not api_key or TheOddsAPIClient is None:
        logger.info("Odds API not configured; operating without live odds.")
        st.session_state["api_config"]["odds_api"] = False
        return None
    try:
        client = TheOddsAPIClient(api_key)
        st.session_state["api_config"]["odds_api"] = True
        return client
    except Exception as exc:  # pragma: no cover - runtime safety
        logger.warning(f"Failed to init TheOddsAPIClient: {exc}")
        st.session_state["api_config"]["odds_api"] = False
        return None


def _build_apisports_clients() -> Dict[str, Any]:
    clients: Dict[str, Any] = {}
    if not APISportsBasketballClient or not APISportsFootballClient:
        st.session_state["api_config"]["apisports"] = False
        return clients
    try:
        api_key = st.secrets.get("APISPORTS_KEY") or st.secrets.get("apisports_key")
        if not api_key:
            st.session_state["api_config"]["apisports"] = False
            return clients
        clients = {
            "basketball": APISportsBasketballClient(api_key),
            "football": APISportsFootballClient(api_key),
            "hockey": APISportsHockeyClient(api_key) if APISportsHockeyClient else None,
        }
        st.session_state["api_config"]["apisports"] = True
    except Exception:
        st.session_state["api_config"]["apisports"] = False
    return clients


def _build_sportsdata_clients() -> Dict[str, Any]:
    clients: Dict[str, Any] = {}
    key = st.secrets.get("SPORTSDATA_API_KEY") or st.secrets.get("sportsdata_api_key")
    if not key or not SportsDataNBAClient:
        st.session_state["api_config"]["sportsdata"] = False
        return clients
    try:
        clients = {
            "nba": SportsDataNBAClient(key),
            "nfl": SportsDataNFLClient(key) if SportsDataNFLClient else None,
            "nhl": SportsDataNHLClient(key) if SportsDataNHLClient else None,
            "ncaab": SportsDataNCAABClient(key) if SportsDataNCAABClient else None,
            "ncaaf": SportsDataNCAAFClient(key) if SportsDataNCAAFClient else None,
        }
        st.session_state["api_config"]["sportsdata"] = True
    except Exception:
        st.session_state["api_config"]["sportsdata"] = False
    return clients


def _build_sentiment_analyzer() -> Optional[Any]:
    if RealSentimentAnalyzer:
        try:
            st.session_state["api_config"]["news"] = True
            return RealSentimentAnalyzer()
        except Exception:
            pass
    if SentimentAnalyzer:
        try:
            st.session_state["api_config"]["news"] = True
            return SentimentAnalyzer()
        except Exception:
            pass
    st.session_state["api_config"]["news"] = False
    return None


def init_vertex_from_secrets(logger: logging.Logger) -> Dict[str, Any]:
    """Initialize Vertex AI from st.secrets; returns status dict."""
    if st.session_state.get("vertex_initialized"):
        return st.session_state["vertex_status"]

    status = {"enabled": False, "message": "Vertex not configured."}
    sa_info = st.secrets.get("gcp_service_account")
    if not sa_info or service_account is None or vertexai is None:
        st.session_state["vertex_initialized"] = False
        st.session_state["vertex_status"] = status
        st.session_state["api_config"]["vertex"] = False
        return status

    try:
        creds = service_account.Credentials.from_service_account_info(
            dict(sa_info), scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        project = st.secrets.get("GCP_PROJECT_ID") or st.secrets.get("gcp_project_id")
        region = st.secrets.get("GCP_REGION") or "us-central1"
        vertexai.init(project=project, location=region, credentials=creds)
        status = {"enabled": True, "message": f"Vertex initialized for {project} ({region})"}
        st.session_state["vertex_initialized"] = True
        st.session_state["api_config"]["vertex"] = True
    except Exception as exc:  # pragma: no cover
        status = {"enabled": False, "message": f"Vertex init failed: {exc}"}
        st.session_state["vertex_initialized"] = False
        st.session_state["api_config"]["vertex"] = False
        logger.warning(status["message"])

    st.session_state["vertex_status"] = status
    return status


def get_kalshi_integrator(logger: logging.Logger) -> Optional[Any]:
    """Initialize KalshiIntegrator once; returns None if not configured."""
    if "kalshi_integrator" in st.session_state:
        return st.session_state["kalshi_integrator"]

    # Legacy fallback: check for old key
    if "kalshi_obj" in st.session_state:
        obj = st.session_state.pop("kalshi_obj")
        if obj:
            st.session_state["kalshi_integrator"] = obj
            return obj

    # Get API credentials (NOT email/password)
    api_key = (
        st.secrets.get("kalshi_api_key")
        or st.secrets.get("KALSHI_API_KEY")
        or (st.secrets.get("kalshi") or {}).get("api_key")
    )
    api_secret = (
        st.secrets.get("kalshi_secret_key")
        or st.secrets.get("kalshi_api_secret")
        or st.secrets.get("KALSHI_API_SECRET")
        or (st.secrets.get("kalshi") or {}).get("api_secret")
    )

    if not api_key or not api_secret or KalshiIntegrator is None:
        st.session_state["kalshi_enabled"] = False
        st.session_state["kalshi_integrator"] = None
        st.session_state["api_config"]["kalshi"] = False
        return None

    try:
        integrator = KalshiIntegrator(api_key=api_key, api_secret=api_secret)
        st.session_state["kalshi_enabled"] = True
        st.session_state["kalshi_integrator"] = integrator
        st.session_state["api_config"]["kalshi"] = True
        return integrator
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Kalshi init failed: {exc}")
        st.session_state["kalshi_enabled"] = False
        st.session_state["kalshi_integrator"] = None
        st.session_state["api_config"]["kalshi"] = False
        return None


def init_app_state() -> Dict[str, Any]:
    if not st.session_state.get("_parlaydesk_init"):
        st.session_state["api_config"] = {
            "odds_api": False,
            "apisports": False,
            "sportsdata": False,
            "news": False,
            "kalshi": False,
            "vertex": False,
        }
        st.session_state["last_exception"] = None
        st.session_state["_parlaydesk_init"] = True

    logger = init_logger()

    odds_client = _build_odds_client(logger)
    apisports_clients = _build_apisports_clients()
    sportsdata_clients = _build_sportsdata_clients()
    sentiment = _build_sentiment_analyzer()

    config = {
        "logger": logger,
        "odds_client": odds_client,
        "apisports_clients": apisports_clients,
        "sportsdata_clients": sportsdata_clients,
        "sentiment_analyzer": sentiment,
    }
    return config

__all__ = [
    "make_key",
    "init_logger",
    "init_app_state",
    "init_vertex_from_secrets",
    "get_kalshi_integrator",
]
