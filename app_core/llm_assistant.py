# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import json
import logging
from typing import Any, Dict, List, Optional
import streamlit as st

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# GEMINI (GOOGLE GENERATIVE AI) SETUP
# -------------------------------------------------------------------
try:
    # Use google-genai package
    from google import genai
    from google.genai import types
    _GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    _GEMINI_AVAILABLE = False
    logger.warning("google-genai not found. Gemini features disabled.")


# Global holding the currently active model name
ACTIVE_MODEL = "gemini-1.5-flash-001"

# Fallback list as requested (Updated for stability)
MODEL_FALLBACKS = ["gemini-1.5-flash-001", "gemini-1.5-pro", "gemini-1.5-flash"]

_GEMINI_CLIENT = None

def initialize_gemini():
    """Initializes Google Generative AI Client with API Key."""
    global _GEMINI_CLIENT
    if not _GEMINI_AVAILABLE:
        return

    if _GEMINI_CLIENT is not None:
        return

    # Try to find API key in environment or Streamlit secrets
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    # Fallback to st.secrets if not in env
    if not api_key:
        try:
            api_key = st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass

    if api_key:
        try:
            _GEMINI_CLIENT = genai.Client(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Gemini Client: {e}")
            if hasattr(st, "session_state"):
                st.session_state["gemini_disabled_reason"] = f"INIT_FAILED: {str(e)}"
    else:
        logger.warning("GEMINI_API_KEY or GOOGLE_API_KEY not found in env or secrets.")
        if hasattr(st, "session_state"):
            st.session_state["gemini_disabled_reason"] = "MISSING_KEY"

def _safe_json_extract(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}

    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        chunk = text[start : end + 1]
        try:
            return json.loads(chunk)
        except Exception:
            return {}
    return {}

def analyze_kalshi_context_with_llm(context_markdown: str, session_state: Optional[Any] = None) -> List[Dict[str, Any]]:
    """
    Returns a list of contract dicts, or [] on any error.

    Args:
        context_markdown: Context about the game/market
        session_state: Optional Streamlit session_state to check/set gemini_disabled_reason

    Returns:
        List of contract recommendations, or empty list on error
    """
    if not _GEMINI_AVAILABLE:
        return []
    if not context_markdown or not context_markdown.strip():
        return []

    # Check session-level disable flag
    if session_state is not None:
        disabled_reason = getattr(session_state, "gemini_disabled_reason", None) or session_state.get("gemini_disabled_reason")
        if disabled_reason:
            # Already disabled - skip silently
            return []

    initialize_gemini()
    if _GEMINI_CLIENT is None:
        return []

    system_instructions = """You are a prediction market assistant that evaluates current prices for event contracts on Kalshi.

You will receive a description of a single game, including:
- League and teams (e.g., NBA, NFL)
- Game time
- Sportsbook moneyline and spread odds
- Kalshi market ticker, label, and implied probability

Your job is to identify one or more underpriced contracts and explain why.

CRITICAL OUTPUT REQUIREMENTS:
1) Output JSON ONLY (no commentary, no markdown).
2) Use this exact structure:
{
  "contracts": [
    {
      "ticker": "<Kalshi ticker or descriptive id>",
      "side": "yes" | "no" | "home" | "away",
      "bid_price": 0,
      "reason": "short explanation",
      "confidence": 0
    }
  ]
}
3) confidence must be an integer 0-100.
4) bid_price must be an integer 0-100 (cents).
5) If you see no clear edge, return {"contracts": []}.
"""

    prompt = f"""{system_instructions}

CONTEXT:
{context_markdown}
"""

    try:
        text = ""
        # Try models in order
        for model_name in MODEL_FALLBACKS:
            try:
                # Rate limit protection (Increased to 3.5s)
                time.sleep(3.5)  # Verified 3.5s rate limit

                resp = _GEMINI_CLIENT.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                text = getattr(resp, "text", "") or ""
                break
            except Exception as e:
                exc_str = str(e)

                # Check for APIKEYINVALID error
                if "API_KEY_INVALID" in exc_str or "API key not valid" in exc_str or "INVALID_ARGUMENT" in exc_str or "400" in exc_str: # 400 often means bad request/key
                    # Disable Gemini for the rest of this session
                    if session_state is not None:
                        if hasattr(session_state, "gemini_disabled_reason"):
                            session_state.gemini_disabled_reason = "APIKEYINVALID"
                        else:
                            session_state["gemini_disabled_reason"] = "APIKEYINVALID"
                    # Log ONE warning and return
                    logger.warning(f"⚠️ Gemini API key invalid/error. Disabling Gemini for this session. Error: {exc_str}")
                    return []

                logger.warning(f"Kalshi LLM analysis failed with {model_name}: {e}")
                continue

        if not text:
            return []

        payload = _safe_json_extract(text)
        contracts = payload.get("contracts", [])
        if not isinstance(contracts, list):
            return []

        cleaned: List[Dict[str, Any]] = []
        for c in contracts:
            if not isinstance(c, dict):
                continue

            # sanitize types
            try:
                c["bid_price"] = int(c.get("bid_price", 0))
                c["confidence"] = int(c.get("confidence", 0))
            except Exception:
                continue

            cleaned.append(c)

        return cleaned

    except Exception as e:
        logger.warning(f"LLM assistant call failed: {e}")
        return []

def generate_confidence_explanation(prompt: str, session_state: Optional[Any] = None) -> Dict[str, Any]:
    """
    Lightweight Gemini call for qualitative confidence/explanation metadata.
    Returns an empty dict if Gemini is unavailable or any error occurs.

    Args:
        prompt: The prompt to send to Gemini
        session_state: Optional Streamlit session_state to check/set gemini_disabled_reason

    Returns:
        Dictionary with confidence explanation, or empty dict on error
    """
    # Check if Gemini is globally unavailable
    if not _GEMINI_AVAILABLE:
        return {}
    if not prompt:
        return {}

    # Check session-level disable flag (prevents repeated API calls with invalid key)
    if session_state is not None:
        disabled_reason = getattr(session_state, "gemini_disabled_reason", None) or session_state.get("gemini_disabled_reason")
        if disabled_reason:
            # Already disabled - skip silently (no repeated warnings)
            return {}

    initialize_gemini()
    if _GEMINI_CLIENT is None:
        return {}

    # Try models in order
    errors = []
    for model_name in MODEL_FALLBACKS:
        try:
            # Rate limit protection
            time.sleep(3.5)

            resp = _GEMINI_CLIENT.models.generate_content(
                model=model_name,
                contents=prompt
            )
            text = getattr(resp, "text", "") or ""
            return _safe_json_extract(text)
        except Exception as exc:
            exc_str = str(exc)
            errors.append(f"{model_name}: {exc_str}")

            # Check for APIKEYINVALID error (Google API returns 400 with this message)
            if "API_KEY_INVALID" in exc_str or "API key not valid" in exc_str or "INVALID_ARGUMENT" in exc_str or "400" in exc_str:
                # Disable Gemini for the rest of this session
                if session_state is not None:
                    if hasattr(session_state, "gemini_disabled_reason"):
                        session_state.gemini_disabled_reason = "APIKEYINVALID"
                    else:
                        session_state["gemini_disabled_reason"] = "APIKEYINVALID"
                # Log ONE warning and return
                logger.warning(f"⚠️ Gemini API key invalid. Disabling Gemini for this session. Error: {exc_str}")
                return {}

            continue

    # All models failed (but not due to invalid key)
    logger.warning(f"Gemini confidence call failed on all fallbacks: {'; '.join(errors)}")
    return {}
