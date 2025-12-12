"""
app_core.llm_assistant

Safe, optional Gemini (Vertex AI) second-opinion engine.
This module MUST NEVER crash the app.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# -------------------------------------------------
# Vertex Gemini setup (safe)
# -------------------------------------------------

_GEMINI_AVAILABLE = False
GenerativeModel = None  # type: ignore

try:
    from vertexai.generative_models import GenerativeModel as _GenerativeModel  # type: ignore
    GenerativeModel = _GenerativeModel  # type: ignore
    _GEMINI_AVAILABLE = True
except Exception as e:
    logger.warning(f"Vertex Gemini unavailable: {e}")

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")


# -------------------------------------------------
# Schema
# -------------------------------------------------

class ContractRecommendation(BaseModel):
    ticker: str
    side: str
    bid_price: int
    reason: str
    confidence: int


# -------------------------------------------------
# Public API
# -------------------------------------------------

def analyze_kalshi_context_with_llm(context_markdown: str) -> List[Dict[str, Any]]:
    """
    Analyze Kalshi context using Gemini via Vertex AI.

    Returns [] on ANY failure.
    """
    if not _GEMINI_AVAILABLE or GenerativeModel is None:
        return []

    if not context_markdown or not context_markdown.strip():
        return []

    system_prompt = """You are a prediction market assistant.

Return JSON ONLY in this format:
{
  "contracts": [
    {
      "ticker": "...",
      "side": "yes|no|home|away",
      "bid_price": 0,
      "reason": "...",
      "confidence": 0
    }
  ]
}

If no edge exists, return {"contracts": []}.
"""

    prompt = f"""{system_prompt}

CONTEXT:
{context_markdown}
"""

    try:
        model = GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(prompt)
        text = getattr(response, "text", "") or ""

        if not text.strip():
            return []

        # Strip markdown fences if present
        if "```" in text:
            text = text.split("```")[1]

        data = json.loads(text)
        contracts = data.get("contracts", [])

        results: List[Dict[str, Any]] = []
        for c in contracts:
            try:
                rec = ContractRecommendation(**c)
                results.append(rec.model_dump())
            except ValidationError:
                continue

        return results

    except Exception as e:
        logger.warning(f"LLM assistant failed safely: {e}")
        return []
