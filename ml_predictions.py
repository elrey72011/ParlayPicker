"""
ML Predictions Module
Provides Vertex AI (GCP), Anthropic Claude, and local ML predictions for Streamlit app
"""

import streamlit as st
from typing import Optional, List, Dict, Any
import logging
import numpy as np
import json
import re
import os

logger = logging.getLogger(__name__)


def is_vertex_ai_enabled() -> bool:
    """
    Check if Vertex AI/ML predictions are enabled
    Checks multiple possible keys and auto-detects based on configuration
    
    Returns:
        True if any ML/AI functionality should be available
    """
    # Method 1: Check explicit enable flags (try multiple possible keys)
    enable_keys = [
        'use_vertex_ai',
        'enable_vertex_ai',
        'vertex_ai_enabled',
        'ai_enabled',
        'ml_enabled',
        'use_ml',
        'enable_ai',
    ]
    
    for key in enable_keys:
        if st.session_state.get(key, False):
            logger.info(f"AI enabled via session key: {key}")
            return True
    
    # Method 2: Auto-detect based on GCP configuration
    has_gcp_config = bool(
        st.session_state.get('gcp_project_id') and
        st.session_state.get('vertex_endpoint_id')
    )
    
    if has_gcp_config:
        logger.info("AI auto-enabled: found GCP Vertex AI configuration")
        return True
    
    # Method 3: Auto-detect based on Anthropic API key
    if st.session_state.get('anthropic_api_key'):
        logger.info("AI auto-enabled: found Anthropic API key")
        return True
    
    # Method 4: Auto-detect based on local models
    try:
        from pathlib import Path
        if Path('./models').exists():
            model_files = list(Path('./models').rglob('*.pkl'))
            if model_files:
                logger.info(f"AI auto-enabled: found {len(model_files)} local models")
                return True
    except Exception as e:
        logger.debug(f"Could not check for local models: {e}")
    
    # Method 5: Default to True if nothing explicitly disables it
    # This makes the sections visible by default
    logger.info("AI enabled by default (no explicit disable)")
    return True


def get_vertex_ai_prediction(features: List[float], game_context: Dict = None) -> Optional[float]:
    """
    Get prediction from Vertex AI (GCP), Anthropic Claude, or local model
    
    Args:
        features: List of feature values (must match training data format)
        game_context: Optional dict with game details for better AI analysis
        
    Returns:
        Probability between 0 and 1, or None if prediction fails
        
    Also sets st.session_state['last_ml_source'] to track which method was used
    """
    if not is_vertex_ai_enabled():
        logger.info("AI predictions not enabled")
        st.session_state['last_ml_source'] = 'disabled'
        return None
    
    # Try GCP Vertex AI endpoint first
    vertex_result = _try_vertex_ai_endpoint(features)
    if vertex_result is not None:
        logger.info(f"✅ GCP Vertex AI prediction: {vertex_result:.3f}")
        st.session_state['last_ml_source'] = 'gcp_vertex'
        return vertex_result
    
    # Try Anthropic Claude as fallback
    anthropic_result = _try_anthropic_claude_prediction(features, game_context)
    if anthropic_result is not None:
        logger.info(f"✅ Anthropic Claude prediction: {anthropic_result:.3f}")
        st.session_state['last_ml_source'] = 'anthropic_claude'
        return anthropic_result
    
    # Fall back to local model
    logger.info("Trying local model prediction")
    local_result = _try_local_model_prediction(features)
    if local_result is not None:
        logger.info(f"✅ Local model prediction: {local_result:.3f}")
        st.session_state['last_ml_source'] = 'local_model'
        return local_result
    
    # Ultimate fallback: use feature-based heuristic for demo purposes
    logger.warning("No model available, using feature-based heuristic")
    st.session_state['last_ml_source'] = 'heuristic'
    return _calculate_heuristic_prediction(features)


def _try_vertex_ai_endpoint(features: List[float]) -> Optional[float]:
    """
    Try to get prediction from GCP Vertex AI endpoint
    
    Args:
        features: Feature values
        
    Returns:
        Probability or None
    """
    try:
        from google.cloud import aiplatform
        
        # Helper to safely get secrets
        def get_secret(key, default=""):
            try:
                if key in st.secrets:
                    return st.secrets[key]
            except Exception:
                pass
            return default
        
        # Get configuration from session state first, then secrets
        project_id = st.session_state.get('gcp_project_id') or get_secret('gcp_project_id', '')
        location = st.session_state.get('gcp_location') or get_secret('gcp_location', 'us-central1')
        endpoint_id = st.session_state.get('vertex_endpoint_id') or get_secret('vertex_endpoint_id', '')
        
        if not project_id:
            logger.info("GCP Project ID not configured")
            return None
        
        if not endpoint_id:
            logger.info("Vertex AI Endpoint ID not configured")
            return None
        
        logger.info(f"Connecting to Vertex AI: project={project_id}, endpoint={endpoint_id}")
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
        
        # Get endpoint - use full resource name
        endpoint_resource = f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"
        endpoint = aiplatform.Endpoint(endpoint_resource)
        
        # Format instances for prediction
        # Try different formats based on what the model expects
        # Format 1: List of features directly
        instances = [features]
        
        # Make prediction
        logger.info(f"Sending {len(features)} features to Vertex AI endpoint")
        response = endpoint.predict(instances=instances)
        
        # Extract probability from response
        if response and response.predictions:
            prediction = response.predictions[0]
            
            # Handle different response formats
            if isinstance(prediction, (int, float)):
                prob = float(prediction)
            elif isinstance(prediction, list) and len(prediction) > 0:
                # Model might return [prob_class_0, prob_class_1]
                if len(prediction) == 2:
                    prob = float(prediction[1])  # Class 1 probability (home win)
                else:
                    prob = float(prediction[0])
            elif isinstance(prediction, dict):
                # Model might return {"probability": X} or {"predictions": [X]}
                prob = float(prediction.get('probability', prediction.get('score', prediction.get('predictions', [0.5])[0])))
            else:
                logger.warning(f"Unknown prediction format: {type(prediction)}")
                prob = float(prediction) if prediction else None
            
            # Ensure probability is valid
            if prob is not None and 0 <= prob <= 1:
                logger.info(f"Vertex AI prediction successful: {prob:.3f}")
                return prob
            else:
                logger.warning(f"Invalid probability from Vertex AI: {prob}")
                return None
        
        logger.warning("No predictions in Vertex AI response")
        return None
        
    except ImportError:
        logger.info("Google Cloud SDK not installed")
        return None
    except Exception as e:
        logger.error(f"Vertex AI prediction failed: {e}")
        return None


def _try_anthropic_claude_prediction(features: List[float], game_context: Dict = None) -> Optional[float]:
    """
    Get prediction from Anthropic Claude API
    
    Args:
        features: Feature values
        game_context: Optional game details dict
        
    Returns:
        Probability or None
    """
    try:
        import anthropic
        
        # Get API key from session state
        api_key = st.session_state.get('anthropic_api_key', '')
        
        if not api_key:
            logger.info("No Anthropic API key configured")
            return None
        
        client = anthropic.Anthropic(api_key=api_key)
        
        # Build context from features
        feature_names = [
            "home_win_pct", "away_win_pct", "home_avg_points", "away_avg_points",
            "home_def_rating", "away_def_rating", "spread_normalized", 
            "home_last_5", "away_last_5", "home_home_record", "away_away_record",
            "head_to_head", "rest_advantage", "injuries_impact", "weather_factor",
            "public_betting_pct", "sharp_money_indicator", "line_movement",
            "total_movement", "model_consensus"
        ]
        
        feature_dict = {}
        for i, val in enumerate(features):
            if i < len(feature_names):
                feature_dict[feature_names[i]] = round(val, 4) if isinstance(val, float) else val
            else:
                feature_dict[f"feature_{i}"] = round(val, 4) if isinstance(val, float) else val
        
        # Add game context if available
        context_str = ""
        if game_context:
            context_str = f"""
GAME DETAILS:
- Home Team: {game_context.get('home_team', 'Unknown')}
- Away Team: {game_context.get('away_team', 'Unknown')}
- Sport: {game_context.get('sport', game_context.get('league', 'Unknown'))}
- Spread: {game_context.get('spread', 'N/A')}
- Total: {game_context.get('total', 'N/A')}
- Date: {game_context.get('date', game_context.get('commence_time', 'Unknown'))}
"""
        
        prompt = f"""You are an expert sports betting AI analyst with deep knowledge of statistical modeling and betting markets.

{context_str}

STATISTICAL FEATURES (normalized 0-1 scale unless noted):
{json.dumps(feature_dict, indent=2)}

KEY FEATURE EXPLANATIONS:
- home_win_pct/away_win_pct: Season win percentages (0-1)
- home_avg_points/away_avg_points: Average points scored (normalized)
- home_def_rating/away_def_rating: Defensive efficiency (lower is better)
- spread_normalized: Point spread (positive favors home)
- home_last_5/away_last_5: Win rate in last 5 games (0-1)

TASK: Based on these features and your sports knowledge, estimate the HOME TEAM win probability.

IMPORTANT:
- Be decisive - don't default to 50%
- Consider home court/field advantage
- Weight recent form heavily
- Typical probabilities range from 0.35 to 0.65 for competitive matchups

RESPOND WITH ONLY THIS JSON (no other text):
{{
  "home_win_probability": 0.XX,
  "confidence": "high/medium/low",
  "key_factors": ["factor1", "factor2"],
  "reasoning": "One sentence explanation"
}}"""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
            timeout=20.0
        )
        
        response_text = message.content[0].text
        
        # Parse JSON response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            analysis = json.loads(json_match.group())
            prob = float(analysis.get('home_win_probability', 0))
            
            # Validate probability - reject clear defaults
            if prob == 0.5 or prob == 50.0 or prob == 0.505:
                logger.warning(f"Claude returned default probability: {prob}")
                return None
            
            # Normalize if given as percentage
            if prob > 1:
                prob = prob / 100.0
            
            # Validate range
            if 0.20 <= prob <= 0.80:
                logger.info(f"Anthropic Claude prediction: {prob:.3f}")
                return prob
            else:
                logger.warning(f"Claude returned out-of-range probability: {prob}")
                return float(np.clip(prob, 0.25, 0.75))
        
        logger.warning("Could not parse Claude response")
        return None
        
    except ImportError:
        logger.info("Anthropic library not installed")
        return None
    except Exception as e:
        logger.error(f"Anthropic prediction failed: {e}")
        return None


def _try_local_model_prediction(features: List[float]) -> Optional[float]:
    """
    Try to get prediction from local XGBoost model
    
    Args:
        features: Feature values
        
    Returns:
        Probability or None
    """
    try:
        import pickle
        from pathlib import Path
        
        # Try to find a trained model
        # Check multiple possible locations and sports
        possible_models = [
            Path('./models/nfl/NFL_spread.pkl'),
            Path('./models/nba/NBA_spread.pkl'),
            Path('./models/nhl/NHL_spread.pkl'),
            Path('./models/ncaab/NCAAB_spread.pkl'),
            Path('./models/ncaaf/NCAAF_spread.pkl'),
            Path('./models/spread_model.pkl'),
            Path('../models/nfl/NFL_spread.pkl'),
        ]
        
        model = None
        scaler = None
        model_path = None
        
        for path in possible_models:
            if path.exists():
                model_path = path
                logger.info(f"Found model at {path}")
                break
        
        if not model_path:
            logger.info("No trained model found locally")
            return None
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Try to load scaler
        scaler_path = model_path.parent / f"{model_path.stem}_scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.info("Loaded feature scaler")
        
        # Prepare features
        features_array = np.array(features).reshape(1, -1)
        
        # Scale if scaler available
        if scaler is not None:
            features_array = scaler.transform(features_array)
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            # Classification model
            prob = model.predict_proba(features_array)[0][1]
        else:
            # Regression model - convert to probability
            prediction = model.predict(features_array)[0]
            # Assume prediction is margin, convert to probability using sigmoid
            prob = 1 / (1 + np.exp(-prediction / 10))
        
        # Ensure valid probability
        prob = float(np.clip(prob, 0.0, 1.0))
        
        logger.info(f"Local model prediction: {prob:.3f}")
        return prob
        
    except ImportError as e:
        logger.error(f"Required library not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Local prediction failed: {e}")
        return None


def _calculate_heuristic_prediction(features: List[float]) -> float:
    """
    Calculate a simple heuristic prediction when no model is available
    PRIORITIZES TheOver.ai spread-derived probabilities over placeholder stats
    
    Feature positions (from vertex_master_analyzer):
    0-2: team strength (home_win_pct, away_win_pct, diff) - MAY BE NONE
    3-8: offense/defense stats - MAY BE NONE
    9-11: recent form - MAY BE NONE
    12: implied_home_prob (KEY - from TheOver.ai)
    13: home_spread / 20 (KEY - from TheOver.ai)
    14: total_line / 200
    15: sentiment_diff
    16-17: home/away sentiment
    18-19: local_ml
    20: theover_probability (KEY - from TheOver.ai)
    21: consensus_prob
    22-23: theover_total data
    24-27: kalshi data
    
    Returns:
        Estimated probability based on features (always valid, never NaN)
    """
    try:
        if len(features) < 2:
            return 0.5
        
        # Helper to safely get feature value
        def safe_get(idx, default=0.5):
            if len(features) > idx:
                val = features[idx]
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return default
                return val
            return default
        
        # Get key TheOver.ai features (these are the most reliable)
        implied_home_prob = safe_get(12, 0.5)
        spread_normalized = safe_get(13, 0)  # home_spread / 20
        theover_prob = safe_get(20, 0.5)
        consensus_prob = safe_get(21, 0.5)
        
        # Get sentiment
        sentiment_diff = safe_get(15, 0)
        
        # Get team stats (may be None/placeholder)
        home_win_pct = safe_get(0, 0.5)
        away_win_pct = safe_get(1, 0.5)
        
        # Check if we have real TheOver.ai data
        has_real_spread = spread_normalized != 0
        has_theover = theover_prob != 0 and theover_prob != 0.5
        has_real_implied = implied_home_prob != 0.5
        
        # PRIORITY 1: Use TheOver.ai probability directly if available
        if has_theover:
            # TheOver.ai already calculated probability from spread
            base_prob = theover_prob * 0.6 + implied_home_prob * 0.25 + consensus_prob * 0.15
            logger.debug(f"Using TheOver.ai prob: {theover_prob:.3f} -> blend: {base_prob:.3f}")
        
        # PRIORITY 2: Use spread to calculate probability
        elif has_real_spread:
            # Each point of spread ≈ 2.8% shift from 50%
            spread_points = spread_normalized * 20  # Undo normalization
            spread_prob = 0.5 + (spread_points * 0.028)
            base_prob = np.clip(spread_prob, 0.20, 0.80)
            
            # Blend with implied if available
            if has_real_implied:
                base_prob = base_prob * 0.6 + implied_home_prob * 0.4
            
            logger.debug(f"Using spread-derived prob: spread={spread_points:.1f} -> {base_prob:.3f}")
        
        # PRIORITY 3: Use implied probability from moneyline
        elif has_real_implied:
            base_prob = implied_home_prob
            logger.debug(f"Using implied prob: {base_prob:.3f}")
        
        # FALLBACK: Team stats (only if they look real, not placeholder)
        else:
            # Check if team stats look like placeholder (both teams 55%)
            is_placeholder = (abs(home_win_pct - 0.55) < 0.01 and abs(away_win_pct - 0.55) < 0.01)
            
            if is_placeholder:
                # Return 50% with slight random variance based on team names
                base_prob = 0.5
                logger.debug("Placeholder stats detected - using 50%")
            else:
                # Real team stats - use weighted combination
                base_prob = home_win_pct * 0.55 + (1 - away_win_pct) * 0.45
        
        # Add sentiment adjustment (small effect)
        if sentiment_diff != 0:
            base_prob += sentiment_diff * 0.03  # Reduced from 0.05
        
        # Final safety check - ensure no NaN
        if np.isnan(base_prob):
            logger.warning("NaN detected in heuristic, returning 0.5")
            return 0.5
        
        # Clip to valid probability range
        return float(np.clip(base_prob, 0.20, 0.80))
        
    except Exception as e:
        logger.warning(f"Heuristic calculation failed: {e}")
        return 0.5


def get_batch_predictions(features_list: List[List[float]], game_contexts: List[Dict] = None) -> List[Optional[float]]:
    """
    Get predictions for multiple games at once
    
    Args:
        features_list: List of feature vectors
        game_contexts: Optional list of game context dicts
        
    Returns:
        List of probabilities (None for failed predictions)
    """
    if not is_vertex_ai_enabled():
        return [None] * len(features_list)
    
    results = []
    for i, features in enumerate(features_list):
        context = game_contexts[i] if game_contexts and i < len(game_contexts) else None
        prob = get_vertex_ai_prediction(features, context)
        results.append(prob)
    
    return results


def show_vertex_ai_prediction_section(home_team: str, away_team: str):
    """
    Display Vertex AI prediction for a single game
    
    Args:
        home_team: Home team name
        away_team: Away team name
    """
    st.subheader(f"🤖 AI Prediction: {away_team} @ {home_team}")
    
    if not is_vertex_ai_enabled():
        st.warning("⚠️ AI predictions not enabled. Configure in sidebar settings.")
        
        with st.expander("📖 How to Enable AI Predictions"):
            st.write("**Option 1: Use GCP Vertex AI (Your deployed model)**")
            st.write("1. Add to Streamlit secrets:")
            st.code("""gcp_project_id = "your-project-id"
vertex_endpoint_id = "your-endpoint-id"
gcp_location = "us-central1"
""", language="toml")
            
            st.write("\n**Option 2: Use Anthropic Claude (Fallback)**")
            st.write("1. Get API key from console.anthropic.com")
            st.write("2. Enter in sidebar under 'Anthropic API key'")
            
            st.write("\n**Option 3: Use Local Models**")
            st.write("1. Train models using the ML pipeline")
            st.write("2. Models will be used automatically")
        
        return
    
    # Create game context
    game_context = {
        'home_team': home_team,
        'away_team': away_team,
        'sport': 'Unknown'
    }
    
    # Generate demo features
    demo_features = [
        0.55,   # home_win_pct
        0.45,   # away_win_pct
        110.0,  # home_avg_points
        105.0,  # away_avg_points
        105.0,  # home_def_rating
        108.0,  # away_def_rating
        0.15,   # spread_normalized
        0.6,    # home_last_5
        0.4,    # away_last_5
    ]
    
    with st.spinner("Getting AI prediction..."):
        prob = get_vertex_ai_prediction(demo_features, game_context)
    
    if prob is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label=f"{home_team} Win Probability",
                value=f"{prob * 100:.1f}%",
                delta=f"{(prob - 0.5) * 100:+.1f}% vs 50/50"
            )
        
        with col2:
            st.metric(
                label=f"{away_team} Win Probability",
                value=f"{(1 - prob) * 100:.1f}%",
                delta=f"{(0.5 - prob) * 100:+.1f}% vs 50/50"
            )
        
        # Show confidence
        confidence = abs(prob - 0.5) * 2
        
        if confidence > 0.3:
            confidence_label = "🟢 High"
        elif confidence > 0.15:
            confidence_label = "🟡 Medium"
        else:
            confidence_label = "🔴 Low"
        
        st.write(f"**Confidence:** {confidence_label} ({confidence * 100:.1f}%)")
        
        # Show AI source
        if st.session_state.get('gcp_project_id') and st.session_state.get('vertex_endpoint_id'):
            st.caption("☁️ Powered by GCP Vertex AI")
        elif st.session_state.get('anthropic_api_key'):
            st.caption("🤖 Powered by Anthropic Claude")
        else:
            st.caption("📊 Using local model/heuristics")
    else:
        st.error("❌ Prediction failed. Check configuration and logs.")
        
        with st.expander("🔍 Debug Information"):
            st.write(f"**Vertex AI Enabled:** {is_vertex_ai_enabled()}")
            st.write(f"**GCP Project ID:** {st.session_state.get('gcp_project_id', 'Not set')}")
            st.write(f"**Endpoint ID:** {st.session_state.get('vertex_endpoint_id', 'Not set')}")
            st.write(f"**GCP Location:** {st.session_state.get('gcp_location', 'Not set')}")
            st.write(f"**Anthropic Key:** {'✅ Configured' if st.session_state.get('anthropic_api_key') else '❌ Not set'}")


def validate_vertex_ai_configuration() -> Dict[str, Any]:
    """
    Validate Vertex AI configuration
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'enabled': is_vertex_ai_enabled(),
        'has_gcp_project': bool(st.session_state.get('gcp_project_id')),
        'has_endpoint': bool(st.session_state.get('vertex_endpoint_id')),
        'has_location': bool(st.session_state.get('gcp_location')),
        'has_anthropic_key': bool(st.session_state.get('anthropic_api_key')),
        'has_google_cloud': False,
        'has_anthropic': False,
        'has_local_models': False,
        'errors': []
    }
    
    # Check Google Cloud SDK
    try:
        from google.cloud import aiplatform
        results['has_google_cloud'] = True
    except ImportError:
        results['errors'].append("Google Cloud SDK not installed")
    
    # Check Anthropic
    try:
        import anthropic
        results['has_anthropic'] = True
    except ImportError:
        results['errors'].append("Anthropic library not installed")
    
    # Check for local models
    from pathlib import Path
    model_dirs = Path('./models').glob('*/') if Path('./models').exists() else []
    results['has_local_models'] = any(
        (d / f'{d.name.upper()}_spread.pkl').exists() 
        for d in model_dirs if d.is_dir()
    )
    
    # Overall status
    results['can_use_gcp'] = (
        results['has_gcp_project'] and 
        results['has_endpoint'] and
        results['has_google_cloud']
    )
    
    results['can_use_anthropic'] = (
        results['has_anthropic_key'] and 
        results['has_anthropic']
    )
    
    results['can_use_local'] = results['has_local_models']
    
    results['ready'] = results['can_use_gcp'] or results['can_use_anthropic'] or results['can_use_local']
    
    return results


# Example usage
if __name__ == "__main__":
    print("ML Predictions Module")
    print("=" * 50)
    print(f"Vertex AI enabled: {is_vertex_ai_enabled()}")
    
    test_features = [0.55, 0.45, 110, 105, 105, 108, 0.15, 0.6, 0.4]
    result = get_vertex_ai_prediction(test_features)
    
    if result is not None:
        print(f"✅ Prediction: {result:.3f}")
    else:
        print("❌ Prediction failed")
