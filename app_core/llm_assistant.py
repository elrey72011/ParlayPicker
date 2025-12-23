# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# GEMINI (VERTEX AI) SETUP
# -------------------------------------------------------------------
try:
    from vertexai.generative_models import GenerativeModel  # type: ignore
    _GEMINI_AVAILABLE = True
except Exception as e:
    GenerativeModel = None  # type: ignore
    _GEMINI_AVAILABLE = False
    logger.warning(f"Vertex Gemini not available: {e}")

GEMINI_MODEL_NAME = "gemini-3-flash-preview"

def _ensure_vertex_init() -> None:
    """
    Initialize Vertex AI with env-provided project/location if available.
    Safe to call multiple times; silently no-ops on error.
    """
    try:
        import vertexai  # type: ignore

        project = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GCP_REGION") or os.getenv("GCP_LOCATION") or "us-central1"
        if project:
            vertexai.init(project=project, location=location)
    except Exception:
        return


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

    _ensure_vertex_init()
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
    try:
        _ensure_vertex_init()
        model = GenerativeModel(GEMINI_MODEL_NAME)
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        return _safe_json_extract(text)
    except Exception as exc:
        logger.warning(f"Gemini confidence call failed: {exc}")
        return {}
