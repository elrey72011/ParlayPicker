# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# GEMINI (GOOGLE GENERATIVE AI) SETUP
# -------------------------------------------------------------------
try:
    import google.generativeai as genai
    from google.generativeai import GenerativeModel
    _GEMINI_AVAILABLE = True
except ImportError:
    GenerativeModel = None
    _GEMINI_AVAILABLE = False
    logger.warning("google.generativeai not found. Gemini features disabled.")


# Global holding the currently active model name
ACTIVE_MODEL = "gemini-1.5-flash-001"

# Fallback list as requested (Updated for stability)
MODEL_FALLBACKS = ["gemini-1.5-flash-001", "gemini-1.5-pro", "gemini-1.5-flash"]

def initialize_gemini():
    """Initializes Google Generative AI with API Key."""
    if not _GEMINI_AVAILABLE:
        return

    # Try to find API key in environment or Streamlit secrets (if available via env var injection)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        # Check if we can get it from streamlit secrets via a helper if it was injected into env
        # Note: In Streamlit Cloud, secrets are often loaded into env or accessed via st.secrets.
        # Here we assume the calling app might have set the env var from st.secrets.
        pass

    if api_key:
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
    else:
        # logger.warning("GEMINI_API_KEY or GOOGLE_API_KEY not found.")
        pass

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

def analyze_kalshi_context_with_llm(context_markdown: str) -> List[Dict[str, Any]]:
    """
    Returns a list of contract dicts, or [] on any error.
    """
    if not _GEMINI_AVAILABLE or GenerativeModel is None:
        return []
    if not context_markdown or not context_markdown.strip():
        return []

    initialize_gemini()

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
                model = GenerativeModel(model_name)
                resp = model.generate_content(prompt)
                text = getattr(resp, "text", "") or ""
                break
            except Exception as e:
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

def generate_confidence_explanation(prompt: str) -> Dict[str, Any]:
    """
    Lightweight Gemini call for qualitative confidence/explanation metadata.
    Returns an empty dict if Gemini is unavailable or any error occurs.
    """
    if not _GEMINI_AVAILABLE or GenerativeModel is None:
        return {}
    if not prompt:
        return {}
    
    initialize_gemini()
    
    # Try models in order
    errors = []
    for model_name in MODEL_FALLBACKS:
        try:
            # Rate limit protection
            time.sleep(3.5)
            model = GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", "") or ""
            return _safe_json_extract(text)
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")
            continue
            
    logger.warning(f"Gemini confidence call failed on all fallbacks: {'; '.join(errors)}")
    return {}
