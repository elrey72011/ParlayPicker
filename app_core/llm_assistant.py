# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import streamlit as st
from streamlit.runtime.secrets import StreamlitSecretNotFoundError

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# GEMINI (GOOGLE GENERATIVE AI) SETUP
# -------------------------------------------------------------------
try:
    # Use google-generativeai package (V1) as requested
    from google import genai
    _GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    _GEMINI_AVAILABLE = False
    logger.warning("google-genai not found. Gemini features disabled.")


# Global holding the currently active model name
ACTIVE_MODEL = "gemini-2.5-flash"

# Fallback list (still useful for internal tracking, though implementation focuses on ACTIVE_MODEL)
MODEL_FALLBACKS = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

_GEMINI_CLIENT = None

def initialize_gemini():
    """Initialize Gemini with updated model"""
    global _GEMINI_CLIENT
    if not _GEMINI_AVAILABLE:
        return None, "Library not available"

    if _GEMINI_CLIENT is not None:
        return _GEMINI_CLIENT, None

    # Gather candidate keys from secrets and environment
    candidates = []

    # Check secrets (prioritize GOOGLE_API_KEY if user migrated)
    if hasattr(st, "secrets"):
        try:
            if "GOOGLE_API_KEY" in st.secrets: candidates.append(st.secrets["GOOGLE_API_KEY"])
            if "GEMINI_API_KEY" in st.secrets: candidates.append(st.secrets["GEMINI_API_KEY"])
        except (FileNotFoundError, StreamlitSecretNotFoundError, KeyError):
            pass  # Secrets not available

    # Check env vars
    if "GOOGLE_API_KEY" in os.environ: candidates.append(os.environ["GOOGLE_API_KEY"])
    if "GEMINI_API_KEY" in os.environ: candidates.append(os.environ["GEMINI_API_KEY"])

    # Filter valid keys (non-empty, non-dummy)
    valid_candidates = []
    seen = set()
    for k in candidates:
        k_str = str(k).strip()
        # Filter out known dummy keys or empty strings
        if k_str and k_str not in seen and not k_str.startswith("AIzaSyBIDJgxLuUouiBQrslV") and not k_str == "None":
             valid_candidates.append(k_str)
             seen.add(k_str)

    if not valid_candidates:
         error_msg = "Gemini initialization failed: No valid API keys found in secrets or environment."
         logger.warning(error_msg)
         if hasattr(st, "session_state"):
            st.session_state["gemini_disabled_reason"] = "NO_VALID_KEYS"
         return None, error_msg

    last_err = None
    # Try each key until one works
    for i, api_key in enumerate(valid_candidates):
        try:
            # Configure with API key
            client = genai.Client(api_key=api_key)

            # Test with simple call to verify key and model validity
            # Using a very small token limit to keep it fast
            client.models.generate_content(
                model=ACTIVE_MODEL,
                contents="test",
                config=genai.types.GenerateContentConfig(max_output_tokens=5)
            )

            _GEMINI_CLIENT = client
            logger.info(f"✓ Gemini 2.5 Flash initialized successfully (Key index {i})")
            return client, None

        except Exception as e:
            last_err = e
            masked_key = f"...{str(api_key)[-4:]}" if len(str(api_key)) > 4 else "INVALID"
            logger.warning(f"Gemini key attempt {i+1} failed ({masked_key}): {str(e)}")
            continue

    # If we get here, all keys failed
    error_msg = f"Gemini initialization failed: All {len(valid_candidates)} keys failed. Last error: {str(last_err)}"
    logger.error(error_msg)
    if hasattr(st, "session_state"):
        st.session_state["gemini_disabled_reason"] = f"ALL_KEYS_FAILED: {str(last_err)}"
    return None, error_msg

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

    client, err = initialize_gemini()
    if client is None:
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
        # Rate limit protection (Increased to 3.5s)
        time.sleep(3.5)

        resp = client.models.generate_content(
            model=ACTIVE_MODEL,
            contents=prompt
        )
        text = getattr(resp, "text", "") or ""

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
        exc_str = str(e)

        # Check for APIKEYINVALID error
        if "API_KEY_INVALID" in exc_str or "API key not valid" in exc_str or "INVALID_ARGUMENT" in exc_str or "400" in exc_str:
            # Disable Gemini for the rest of this session
            if session_state is not None:
                if hasattr(session_state, "gemini_disabled_reason"):
                    session_state.gemini_disabled_reason = "APIKEYINVALID"
                else:
                    session_state["gemini_disabled_reason"] = "APIKEYINVALID"
            # Log ONE warning and return
            logger.warning(f"⚠️ Gemini API key invalid/error. Disabling Gemini for this session. Error: {exc_str}")
            return []

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

    client, err = initialize_gemini()
    if client is None:
        return {}

    try:
        # Rate limit protection
        time.sleep(3.5)

        resp = client.models.generate_content(
            model=ACTIVE_MODEL,
            contents=prompt
        )
        text = getattr(resp, "text", "") or ""
        return _safe_json_extract(text)

    except Exception as exc:
        exc_str = str(exc)

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

        logger.warning(f"Gemini confidence call failed: {exc_str}")
        return {}


def _complete_batch_review(payload: Any) -> bool:
    """Return whether Gemini supplied every field required by the wager gate."""
    if not isinstance(payload, dict):
        return False
    required_text = ("recommended_bet", "confidence", "explanation", "risk_notes")
    if not all(str(payload.get(field) or "").strip() for field in required_text):
        return False
    if str(payload.get("confidence") or "").strip().upper() not in {"HIGH", "MEDIUM", "LOW"}:
        return False
    return "flags" in payload and isinstance(payload.get("flags"), list)


def generate_batch_confidence_explanation(
    games_data: List[Dict[str, Any]],
    session_state: Optional[Any] = None,
    *,
    _retry_incomplete: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Single Gemini call for all games at once (batch processing).

    Args:
        games_data: List of game dicts with pick info. Each dict must have a 'game_id' or unique identifier.
        session_state: Optional Streamlit session_state to check/set gemini_disabled_reason

    Returns:
        Dict mapping game_id -> confidence assessment dict
    """
    # Check if Gemini is globally unavailable
    if not _GEMINI_AVAILABLE or not games_data:
        return {}

    # Check session-level disable flag
    if session_state is not None:
        disabled_reason = getattr(session_state, "gemini_disabled_reason", None) or session_state.get("gemini_disabled_reason")
        if disabled_reason:
            return {}

    client, err = initialize_gemini()
    if client is None:
        return {}

    # Limit batch size to avoid token limits.
    BATCH_SIZE = 50
    all_results = {}

    logger.info(f"Processing {len(games_data)} games in {max(1, (len(games_data) + BATCH_SIZE - 1) // BATCH_SIZE)} batches")

    for i in range(0, len(games_data), BATCH_SIZE):
        batch = games_data[i:i+BATCH_SIZE]

        # Build prompt
        current_time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        prompt = f"""Current Time: {current_time_str}
You are an independent sports betting risk reviewer. Do not assume any side has already
been approved or rejected — some games here are ones the system declined, others
are ones it bet, and you are not told which is which.

Each game has a 'side_a' object: the market line, de-vigged market probability,
and this system's model probabilities (Kalshi/ML/TheOver), odds, and edge/EV for
one candidate side. Many games also have a 'side_b' object with the SAME fields
computed independently for the opposing side of that same market (e.g. side_a is
total_under, side_b is total_over for the same total line). When side_b is
present, do a genuine head-to-head comparison of side_a vs side_b using both
sides' full data and pick whichever you believe is the better bet — your pick is
NOT required to be side_a just because it's listed first. When side_b is null,
evaluate side_a on its own. Rows with is_player_prop=true are player propositions;
use their projection, exact line/price, market probability, form sample, and
calibration evidence. Do not invent or claim injuries, lineups, weather, news,
or any other fact not present in the payload. You are a bounded reviewer, not a
source of new odds or probabilities.

For each game, return a JSON object with:
- game_id: The identifier provided in input
- recommended_bet: ALWAYS name a side, using its 'best_pick' label verbatim (e.g. "Houston +1.5", "Under 7.5") — the side you believe is the better bet after comparing side_a and side_b, even when the edge at this price is thin or negative for both. Only use 'none' if the game data given is genuinely too incomplete to form any opinion (e.g. odds or probabilities are missing for both sides).
- confidence: HIGH/MEDIUM/LOW — and LOW (not 'none') is how you flag a side with little or no betting value at the given price
- explanation: Brief 1-sentence rationale for the pick, grounded in the probabilities/edge given for both sides considered (max 240 chars)
- risk_notes: Specific risks or reasons for caution (max 240 chars) — if the price doesn't justify a bet, say so here (e.g. "edge does not clear the vig at this price"). If the 'is_live_data' flag is false, you must output the exact risk note: "Analysis used league-average fallbacks due to missing live stats."
- flags: Array of short flag strings (e.g. "missing_odds", "missing_live_stats",
  "contrarian", "no_value_at_price", "picked_opposing_side"). Use
  "no_value_at_price" whenever the recommended side does not have positive value
  at the exact offered price. Use "picked_opposing_side" whenever recommended_bet
  differs from side_a.best_pick.

Games to analyze:
{json.dumps(batch, indent=2, default=str)}

Return ONLY a JSON array of objects. No markdown formatting.
"""

        try:
            # Rate limit protection - slight delay between batches
            time.sleep(2.0)

            resp = client.models.generate_content(
                model=ACTIVE_MODEL,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            text = getattr(resp, "text", "") or ""

            # Parse response
            # Sometimes it might be wrapped in ```json ... ```
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            batch_results = []
            try:
                batch_results = json.loads(text)
            except json.JSONDecodeError:
                # Try _safe_json_extract logic for lists (roughly)
                start = text.find("[")
                end = text.rfind("]")
                if start >= 0 and end > start:
                    try:
                        batch_results = json.loads(text[start:end+1])
                    except:
                        pass

            if isinstance(batch_results, list):
                for res in batch_results:
                    if isinstance(res, dict):
                        g_id = str(res.get('game_id', ''))
                        if g_id:
                            all_results[g_id] = res
            else:
                 logger.warning(f"Gemini batch response was not a list: {str(text)[:100]}")

        except Exception as exc:
             exc_str = str(exc)
             # Check for APIKEYINVALID error
             if "API_KEY_INVALID" in exc_str or "API key not valid" in exc_str or "INVALID_ARGUMENT" in exc_str or "400" in exc_str:
                if session_state is not None:
                    if hasattr(session_state, "gemini_disabled_reason"):
                        session_state.gemini_disabled_reason = "APIKEYINVALID"
                    else:
                         session_state["gemini_disabled_reason"] = "APIKEYINVALID"
                logger.warning(f"⚠️ Gemini API key invalid. Disabling. Error: {exc_str}")
                return all_results # Return what we have

             logger.warning(f"Gemini batch call failed: {exc_str}")
             # We continue to next batch

    if _retry_incomplete:
        incomplete = [
            game
            for game in games_data
            if not _complete_batch_review(all_results.get(str(game.get("game_id", ""))))
        ]
        if incomplete:
            logger.warning(
                "Gemini returned %s incomplete review(s); retrying those rows once",
                len(incomplete),
            )
            retried = generate_batch_confidence_explanation(
                incomplete,
                session_state,
                _retry_incomplete=False,
            )
            all_results.update(retried)

    return all_results


def generate_pick_rationale(pick, home_team, away_team, market, prob, edge, session_state=None):
    """
    Calls Gemini to explain why this pick has value.
    Returns dict with 'confidence' and 'rationale' keys.
    """
    if not _GEMINI_AVAILABLE:
        return {'confidence': '', 'rationale': '', 'error': 'Gemini not available'}

    # Check session-level disable flag
    if session_state is not None:
        disabled_reason = getattr(session_state, "gemini_disabled_reason", None) or session_state.get("gemini_disabled_reason")
        if disabled_reason:
            return {'confidence': '', 'rationale': '', 'error': f'Gemini disabled: {disabled_reason}'}

    client, error = initialize_gemini()
    if not client:
        return {'confidence': '', 'rationale': '', 'error': error}

    # Format edge/prob as percentage string if float, otherwise use as string
    try:
        edge_val = float(edge)
        edge_str = f"{edge_val:.1%}"
    except (ValueError, TypeError):
        edge_str = str(edge)

    try:
        prob_val = float(prob)
        prob_str = f"{prob_val:.1%}"
    except (ValueError, TypeError):
        prob_str = str(prob)

    prompt = f"""
    You are a sports betting analyst. Explain why this pick has value:

    Game: {away_team} @ {home_team}
    Market: {market}
    Pick: {pick}
    Model Probability: {prob_str}
    Edge vs Market: {edge_str}

    Provide:
    1. A confidence rating (LOW/MEDIUM/HIGH)
    2. A 1-2 sentence rationale focusing on the edge driver

    Format:
    CONFIDENCE: [rating]
    RATIONALE: [explanation]
    """

    try:
        # Rate limit protection
        time.sleep(1.0)

        response = client.models.generate_content(
            model=ACTIVE_MODEL,
            contents=prompt
        )
        text = getattr(response, "text", "") or ""

        # Parse response
        lines = text.split('\n')
        conf_line = [l for l in lines if 'CONFIDENCE:' in l]
        rat_line = [l for l in lines if 'RATIONALE:' in l]

        confidence = conf_line[0].split(':', 1)[1].strip() if conf_line else ''
        rationale = rat_line[0].split(':', 1)[1].strip() if rat_line else text[:200]

        # Fallback cleanup
        confidence = confidence.upper()
        if confidence not in ["HIGH", "MEDIUM", "LOW"]:
            # Try to infer if possible or leave as is
            pass

        return {
            'confidence': confidence,
            'rationale': rationale,
            'error': ''
        }
    except Exception as e:
        exc_str = str(e)
        logger.warning(f"generate_pick_rationale failed: {exc_str}")
        return {'confidence': '', 'rationale': '', 'error': exc_str[:100]}
