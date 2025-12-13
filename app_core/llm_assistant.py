"""
app_core.llm_assistant

Lightweight LLM assistant wrapper for ParlayPicker.

Provides:
    - analyze_kalshi_context_with_llm(context_markdown: str) -> List[dict]

This is used as a SECOND-OPINION reasoning engine (Gemini via Vertex AI),
NOT as a replacement for Vertex Master Analysis. If Gemini/Vertex isn't
available, the function returns an empty list.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# GEMINI (VERTEX AI) SETUP
# -------------------------------------------------------------------
try:
    # Vertex AI Gemini (preferred). Do NOT use google-generativeai here.
    from vertexai.generative_models import GenerativeModel  # type: ignore

    _GEMINI_AVAILABLE = True
except Exception as e:
    GenerativeModel = None  # type: ignore
    _GEMINI_AVAILABLE = False
    logger.warning(f"Vertex Gemini not available: {e}")

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

# -------------------------------------------------------------------
# Pydantic schema for contract recommendations
# -------------------------------------------------------------------
class ContractRecommendation(BaseModel):
    ticker: str
    side: str          # "yes"/"no" or "home"/"away"
    bid_price: int     # 0–100 (cents)
    reason: str
    confidence: int    # 0–100


def _safe_json_extract(text: str) -> Dict[str, Any]:
    """
    Best-effort extraction of a JSON object from model output.
    We demand JSON-only, but models can still wrap in text; this tries to recover.
    """
    text = (text or "").strip()
    if not text:
        return {}

    # Already JSON?
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass

    # Try to find the first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        chunk = text[start : end + 1]
        try:
            return json.loads(chunk)
        except Exception:
            return {}

    return {}

# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------
def analyze_kalshi_context_with_llm(context_markdown: str) -> List[Dict[str, Any]]:
    """
    Use Gemini via Vertex AI to analyze a Kalshi-style context string and return
    contract recommendations.

    Returns [] if Gemini/Vertex is unavailable or any error occurs.
    """
    if not _GEMINI_AVAILABLE or GenerativeModel is None:
        logger.debug("LLM assistant disabled (Vertex Gemini not available).")
        return []

    if not context_markdown or not context_markdown.strip():
        return []

    # Keep this as ONE triple-quoted string to avoid quote/f-string bugs.
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
        model = GenerativeModel(GEMINI_MODEL_NAME)
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""

        payload = _safe_json_extract(text)
        contracts_raw = payload.get("contracts", [])
        if not isinstance(contracts_raw, list):
            return []

        cleaned: List[Dict[str, Any]] = []
        for item in contracts_raw:
            if not isinstance(item, dict):
                continue
            try:
                cr = ContractRecommendation(**item)
                cleaned.append(cr.model_dump())
            except ValidationError:
                continue

        return cleaned

    except Exception as e:
        logger.warning(f"LLM assistant call failed: {e}")
        return []