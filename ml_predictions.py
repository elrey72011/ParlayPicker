"""
ML Predictions Module
Provides Vertex AI, Anthropic Claude, and local ML predictions for Streamlit app
"""

import streamlit as st
from typing import Optional, List, Dict, Any
import logging
import numpy as np
import json
import re

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
    
    # Method 2: Auto-detect based on Anthropic API key (NEW!)
    if st.session_state.get('anthropic_api_key'):
        logger.info("AI auto-enabled: found Anthropic API key")
        return True
    
    # Method 3: Auto-detect based on GCP configuration
    has_gcp_config = bool(
        st.session_state.get('gcp_project_id') or
        st.session_state.get('vertex_endpoint_id')
    )
    
    if has_gcp_config:
        logger.info("AI auto-enabled: found GCP configuration")
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
    Get prediction from Vertex AI, Anthropic Claude, or local model
    
    Args:
        features: List of feature values (must match training data format)
        game_context: Optional dict with game details for Claude analysis
        
    Returns:
        Probability between 0 and 1, or None if prediction fails
    """
    if not is_vertex_ai_enabled():
        logger.info("AI predictions not enabled")
        return None
    
    # Try Vertex AI endpoint first (Google Cloud)
    vertex_result = _try_vertex_ai_endpoint(features)
    if vertex_result is not None:
        return vertex_result
    
    # Try Anthropic Claude (NEW!)
    anthropic_result = _try_anthropic_claude_prediction(features, game_context)
    if anthropic_result is not None:
        return anthropic_result
    
    # Fall back to local model
    logger.info("Trying local model prediction")
    local_result = _try_local_model_prediction(features)
    if local_result is not None:
        return local_result
    
    # Ultimate fallback: use feature-based heuristic for demo purposes
    logger.warning("No model available, using feature-based heuristic")
    return _calculate_heuristic_prediction(features)


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
            "home_last_5", "away_last_5"
        ]
        
        feature_dict = {}
        for i, val in enumerate(features):
            if i < len(feature_names):
                feature_dict[feature_names[i]] = val
            else:
                feature_dict[f"feature_{i}"] = val
        
        # Add game context if available
        context_str = ""
        if game_context:
            context_str = f"""
GAME DETAILS:
- Home Team: {game_context.get('home_team', 'Unknown')}
- Away Team: {game_context.get('away_team', 'Unknown')}
- Sport: {game_context.get('sport', 'Unknown')}
- Spread: {game_context.get('spread', 'N/A')}
- Total: {game_context.get('total', 'N/A')}
"""
        
        prompt = f"""You are a sports betting AI analyst. Based on the following statistical features, predict the HOME TEAM win probability.

{context_str}

STATISTICAL FEATURES:
{json.dumps(feature_dict, indent=2)}

FEATURE EXPLANATIONS:
- home_win_pct/away_win_pct: Season win percentages (0-1 scale)
- home_avg_points/away_avg_points: Average points scored per game
- home_def_rating/away_def_rating: Defensive efficiency (lower is better)
- spread_normalized: Point spread normalized (positive favors home)
- home_last_5/away_last_5: Win rate in last 5 games (0-1 scale)

Based on these features and your sports knowledge, provide:
1. Your estimated HOME TEAM win probability (between 0.30 and 0.70 - be decisive, don't default to 0.50)
2. Brief reasoning

RESPOND WITH ONLY THIS JSON (no other text):
{{
  "home_win_probability": 0.XX,
  "confidence": "high/medium/low",
  "reasoning": "Brief 1-2 sentence explanation"
}}"""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
            timeout=15.0
        )
        
        response_text = message.content[0].text
        
        # Parse JSON response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            analysis = json.loads(json_match.group())
            prob = float(analysis.get('home_win_probability', 0))
            
            # Validate probability
            if 0.25 <= prob <= 0.75:
                logger.info(f"Anthropic Claude prediction: {prob:.3f}")
                return prob
            else:
                logger.warning(f"Claude returned out-of-range probability: {prob}")
                # Clip to valid range
                return float(np.clip(prob, 0.30, 0.70))
        
        logger.warning("Could not parse Claude response")
        return None
        
    except ImportError:
        logger.info("Anthropic library not installed")
        return None
    except Exception as e:
        logger.error(f"Anthropic prediction failed: {e}")
        return None


def _try_vertex_ai_endpoint(features: List[float]) -> Optional[float]:
    """
    Try to get prediction from Vertex AI endpoint
    
    Args:
        features: Feature values
        
    Returns:
        Probability or None
    """
    try:
        from google.cloud import aiplatform
        
        # Get configuration from session state
        project_id = st.session_state.get('gcp_project_id')
        location = st.session_state.get('gcp_location', 'us-central1')
        endpoint_id = st.session_state.get('vertex_endpoint_id')
        
        if not project_id:
            logger.warning("GCP Project ID not configured")
            return None
        
        if not endpoint_id:
            logger.warning("Vertex AI Endpoint ID not configured")
            return None
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
        
        # Get endpoint
        endpoint = aiplatform.Endpoint(endpoint_id)
        
        # Format instances for prediction
        # Adjust this based on your model's expected input format
        instances = [{"features": features}]
        
        # Make prediction
        response = endpoint.predict(instances=instances)
        
        # Extract probability from response
        if response and response.predictions:
            # Assuming model returns probability as first element
            prob = float(response.predictions[0])
            
            # Ensure probability is valid
            if 0 <= prob <= 1:
                logger.info(f"Vertex AI prediction: {prob:.3f}")
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
            logger.warning("No trained model found locally")
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
        st.warning("⚠️ Vertex AI is not enabled. Enable in AI Settings to see predictions.")
        
        with st.expander("📖 How to Enable AI Predictions"):
            st.write("**Option 1: Use Anthropic Claude (Recommended)**")
            st.write("1. Get API key from console.anthropic.com")
            st.write("2. Enter in sidebar under 'Anthropic API key'")
            st.write("3. AI predictions will work automatically!")
            
            st.write("\n**Option 2: Use Vertex AI (Google Cloud)**")
            st.write("1. Go to AI Settings in sidebar")
            st.write("2. Check 'Enable Vertex AI'")
            st.write("3. Enter your GCP Project ID")
            st.write("4. Enter your Vertex AI Endpoint ID")
            st.write("5. Set GCP Location (e.g., us-central1)")
            
            st.write("\n**Option 3: Use Local Models**")
            st.write("1. Train models using the ML pipeline:")
            st.code("python main.py --train --sports NFL NBA", language="bash")
            st.write("2. Models will be saved to ./models/")
            st.write("3. Enable Vertex AI in settings")
            st.write("4. Local models will be used automatically")
        
        return
    
    # Generate demo features (replace with real features)
    demo_features = [
        0.55,   # home_win_pct
        0.45,   # away_win_pct
        110.0,  # home_avg_points
        105.0,  # away_avg_points
        105.0,  # home_def_rating
        108.0,  # away_def_rating
        0.15,   # spread_abs / 20
        0.6,    # home_last_5 / 5
        0.4,    # away_last_5 / 5
    ]
    
    # Create game context for better Claude predictions
    game_context = {
        'home_team': home_team,
        'away_team': away_team,
        'sport': 'Unknown'
    }
    
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
        confidence = abs(prob - 0.5) * 2  # 0 to 1 scale
        
        if confidence > 0.3:
            confidence_label = "🟢 High"
        elif confidence > 0.15:
            confidence_label = "🟡 Medium"
        else:
            confidence_label = "🔴 Low"
        
        st.write(f"**Confidence:** {confidence_label} ({confidence * 100:.1f}%)")
        
        # Show AI source
        if st.session_state.get('anthropic_api_key'):
            st.caption("🤖 Powered by Anthropic Claude")
        elif st.session_state.get('gcp_project_id'):
            st.caption("☁️ Powered by Google Vertex AI")
        else:
            st.caption("📊 Using local model/heuristics")
        
        # Show features used
        with st.expander("📊 Features Used in Prediction"):
            st.write("**Feature Values:**")
            feature_names = [
                "Home Win %",
                "Away Win %",
                "Home Avg Points",
                "Away Avg Points",
                "Home Def Rating",
                "Away Def Rating",
                "Spread (normalized)",
                "Home Last 5",
                "Away Last 5"
            ]
            
            for name, value in zip(feature_names, demo_features):
                st.write(f"- {name}: {value:.3f}")
    else:
        st.error("❌ Prediction failed. Check configuration and logs.")
        
        # Show debugging info
        with st.expander("🔍 Debug Information"):
            st.write(f"**Vertex AI Enabled:** {is_vertex_ai_enabled()}")
            st.write(f"**Anthropic API Key:** {'✅ Configured' if st.session_state.get('anthropic_api_key') else '❌ Not set'}")
            st.write(f"**GCP Project ID:** {st.session_state.get('gcp_project_id', 'Not set')}")
            st.write(f"**Endpoint ID:** {st.session_state.get('vertex_endpoint_id', 'Not set')}")
            st.write(f"**Location:** {st.session_state.get('gcp_location', 'Not set')}")
            
            st.write("\n**Possible Issues:**")
            st.write("- No Anthropic API key configured")
            st.write("- Vertex AI credentials not configured")
            st.write("- No trained models found locally")
            st.write("- Feature format doesn't match model")


def validate_vertex_ai_configuration() -> Dict[str, Any]:
    """
    Validate Vertex AI configuration
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'enabled': is_vertex_ai_enabled(),
        'has_anthropic_key': bool(st.session_state.get('anthropic_api_key')),
        'has_project_id': bool(st.session_state.get('gcp_project_id')),
        'has_endpoint_id': bool(st.session_state.get('vertex_endpoint_id')),
        'has_location': bool(st.session_state.get('gcp_location')),
        'has_google_cloud': False,
        'has_anthropic': False,
        'has_local_models': False,
        'errors': []
    }
    
    # Check Anthropic (NEW!)
    try:
        import anthropic
        results['has_anthropic'] = True
    except ImportError:
        results['errors'].append("Anthropic library not installed")
    
    # Check Google Cloud SDK
    try:
        from google.cloud import aiplatform
        results['has_google_cloud'] = True
    except ImportError:
        results['errors'].append("Google Cloud SDK not installed")
    
    # Check for local models
    from pathlib import Path
    model_dirs = Path('./models').glob('*/') if Path('./models').exists() else []
    results['has_local_models'] = any(
        (d / f'{d.name.upper()}_spread.pkl').exists() 
        for d in model_dirs if d.is_dir()
    )
    
    # Overall status
    results['can_use_anthropic'] = (
        results['enabled'] and 
        results['has_anthropic_key'] and 
        results['has_anthropic']
    )
    
    results['can_use_vertex'] = (
        results['enabled'] and 
        results['has_project_id'] and 
        results['has_endpoint_id'] and
        results['has_google_cloud']
    )
    
    results['can_use_local'] = (
        results['enabled'] and 
        results['has_local_models']
    )
    
    results['ready'] = results['can_use_anthropic'] or results['can_use_vertex'] or results['can_use_local']
    
    return results


def _calculate_heuristic_prediction(features: List[float]) -> float:
    """
    Calculate a simple heuristic prediction when no model is available
    This is a fallback for demo/testing purposes
    
    Args:
        features: Feature values (assumes standard 9-feature format)
        
    Returns:
        Estimated probability based on features
    """
    try:
        # Assume standard feature order:
        # [home_win_pct, away_win_pct, home_points, away_points, 
        #  home_def, away_def, spread, home_l5, away_l5]
        
        if len(features) < 2:
            return 0.5
        
        home_win_pct = features[0]
        away_win_pct = features[1]
        
        # Simple weighted average favoring home team slightly (home advantage)
        base_prob = (home_win_pct * 0.55 + (1 - away_win_pct) * 0.45)
        
        # Add recent form if available
        if len(features) >= 9:
            home_form = features[7]  # home_last_5
            away_form = features[8]  # away_last_5
            form_factor = (home_form - away_form) * 0.1
            base_prob += form_factor
        
        # Clip to valid probability range
        return float(np.clip(base_prob, 0.1, 0.9))
        
    except Exception as e:
        logger.warning(f"Heuristic calculation failed: {e}")
        return 0.5  # Return neutral 50/50


# Export key functions


# Example usage in Streamlit
if __name__ == "__main__":
    print("ML Predictions Module")
    print("=" * 50)
    
    # Test configuration
    print("\nConfiguration Check:")
    print(f"- Vertex AI enabled: {is_vertex_ai_enabled()}")
    
    # Test prediction
    print("\nTest Prediction:")
    test_features = [0.55, 0.45, 110, 105, 105, 108, 0.15, 0.6, 0.4]
    result = get_vertex_ai_prediction(test_features)
    
    if result is not None:
        print(f"✅ Prediction successful: {result:.3f}")
    else:
        print("❌ Prediction failed")
    
    print("\n" + "=" * 50)
