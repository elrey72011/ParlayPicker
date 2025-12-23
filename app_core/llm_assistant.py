# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import logging
import vertexai
from typing import Any, Dict, List
from vertexai.generative_models import GenerativeModel

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# UNIFIED GEMINI (VERTEX AI) SETUP
# -------------------------------------------------------------------

# 1. Define the Fallbacks (Order of preference for Dec 2025)
MODEL_FALLBACKS = [
    "gemini-2.0-flash-exp",   # Modern experimental stable
    "gemini-1.5-flash-002",   # Stable workhorse
    "gemini-3-flash-preview"  # Newest preview (if enabled in Model Garden)
]

# 2. Global State Variables
_ACTIVE_MODEL_ID = None
_GEMINI_AVAILABLE = False

def initialize_gemini():
    """Finds the first working model and sets global availability."""
    global _ACTIVE_MODEL_ID, _GEMINI_AVAILABLE
    
    project = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GCP_REGION") or "us-central1"
    
    if not project:
        logger.warning("GCP_PROJECT_ID not found. Gemini disabled.")
        return None

    try:
        vertexai.init(project=project, location=location)
        
        # Try fallbacks until one works
        for model_id in MODEL_FALLBACKS:
            try:
                test_model = GenerativeModel(model_id)
                # Quick test call to verify availability
                test_model.generate_content("ping") 
                _ACTIVE_MODEL_ID = model_id
                _GEMINI_AVAILABLE = True
                logger.info(f"✅ Gemini Initialized with: {model_id}")
                return test_model
            except Exception:
                continue
    except Exception as e:
        logger.error(f"Critical Vertex AI Init Error: {e}")
    
    return None

# Initialize once on module load
ACTIVE_MODEL = initialize_gemini()

# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------

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
    Returns a list of contract dicts using the verified ACTIVE_MODEL.
    """
    if not _GEMINI_AVAILABLE or ACTIVE_MODEL is None:
        return []
    if not context_markdown or not context_markdown.strip():
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

    prompt = f"{system_instructions}\n\nCONTEXT:\n{context_markdown}"

    try:
        resp = ACTIVE_MODEL.generate_content(prompt)
        text = getattr(resp, "text", "") or ""

        payload = _safe_json_extract(text)
        contracts = payload.get("contracts", [])
        if not isinstance(contracts, list):
            return []

        cleaned: List[Dict[str, Any]] = []
        for c in contracts:
            if not isinstance(c, dict):
                continue
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
    Qualitative confidence metadata using the verified ACTIVE_MODEL.
    """
    if not _GEMINI_AVAILABLE or ACTIVE_MODEL is None:
        return {}
    if not prompt:
        return {}
    try:
        resp = ACTIVE_MODEL.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        return _safe_json_extract(text)
    except Exception as exc:
        logger.warning(f"Gemini confidence call failed: {exc}")
        return {}
