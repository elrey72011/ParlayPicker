# Quick Fix Patch for ML Prediction Issue
# File: ml_prediction_quickfix.py
# Date: November 27, 2025
#
# This file contains the corrected code to fix the critical ML prediction issue
# Replace the broken section in streamlit_app__23_.py (lines ~12000-12200)

"""
ISSUE: ML predictions never execute, causing all games to show 58% probability

LOCATION: streamlit_app__23_.py, around lines 12034-12200
          (in the main game processing loop, moneyline section)

SYMPTOMS:
- All games show identical probabilities
- ml_prediction_result is always None
- No game-specific AI analysis

ROOT CAUSE:
The code creates games_to_predict list but never calls the prediction method.
ml_prediction_result stays None, triggering default baseline probabilities.
"""

# =============================================================================
# CORRECTED CODE - Replace your existing moneyline ML section with this
# =============================================================================

def process_moneyline_with_ml_predictions(
    home: str,
    away: str,
    skey: str,
    home_price: float,
    away_price: float,
    ml_predictor,
    use_ml_predictions: bool,
    baseline_confidence: float = 0.55,
    min_ai_confidence: float = 0.50,
):
    """
    Process moneyline bets with ML predictions.
    
    This is the CORRECTED version that actually calls the ML predictor!
    
    Args:
        home: Home team name
        away: Away team name
        skey: Sport key (e.g., 'americanfootball_nfl')
        home_price: Home team moneyline odds
        away_price: Away team moneyline odds
        ml_predictor: MLPredictor instance from session state
        use_ml_predictions: Whether ML is enabled
        baseline_confidence: Default confidence if ML unavailable
        min_ai_confidence: Minimum confidence threshold
    
    Returns:
        dict: {
            'home_prob': float,
            'away_prob': float,
            'confidence': float,
            'edge': float,
            'model_used': str,
            'prediction_successful': bool
        }
    """
    
    # Calculate baseline probabilities from odds
    home_base_prob = implied_p_from_american(home_price)
    away_base_prob = implied_p_from_american(away_price)
    
    result = {
        'home_prob': home_base_prob,
        'away_prob': away_base_prob,
        'confidence': baseline_confidence,
        'edge': 0.0,
        'model_used': 'baseline_odds',
        'prediction_successful': False,
        'training_rows': None,
        'component_probabilities': None
    }
    
    # Skip ML if not enabled or predictor not available
    if not use_ml_predictions or not ml_predictor:
        logger.info(f"ML predictions disabled or unavailable for {away} @ {home}")
        return result
    
    # Skip if odds are invalid
    if home_price is None or away_price is None:
        logger.warning(f"Invalid odds for {away} @ {home}: home={home_price}, away={away_price}")
        return result
    
    # =============================================================================
    # THIS IS THE CRITICAL FIX: Actually call the ML predictor!
    # =============================================================================
    try:
        logger.info(f"Calling ML predictor for {away} @ {home} ({skey})")
        
        # Call the predictor
        ml_prediction_result = ml_predictor.predict_game_outcome(
            home_team=home,
            away_team=away,
            sport_key=skey
        )
        
        # Check if prediction was successful
        if ml_prediction_result and isinstance(ml_prediction_result, dict):
            # Extract probabilities
            home_prob = ml_prediction_result.get('home_win_prob')
            away_prob = ml_prediction_result.get('away_win_prob')
            confidence = ml_prediction_result.get('confidence')
            
            # Validate probabilities
            if home_prob is not None and away_prob is not None and confidence is not None:
                # Calculate edge (difference from market)
                edge = abs(home_prob - home_base_prob)
                
                result.update({
                    'home_prob': home_prob,
                    'away_prob': away_prob,
                    'confidence': confidence,
                    'edge': edge,
                    'model_used': ml_prediction_result.get('model_used', 'ml_model'),
                    'prediction_successful': True,
                    'training_rows': ml_prediction_result.get('training_rows'),
                    'component_probabilities': ml_prediction_result.get('component_probabilities'),
                })
                
                logger.info(
                    f"✅ ML prediction successful for {away} @ {home}: "
                    f"home={home_prob:.1%}, away={away_prob:.1%}, conf={confidence:.1%}, edge={edge:.1%}"
                )
            else:
                logger.warning(f"ML prediction returned None values for {away} @ {home}")
        else:
            logger.warning(f"ML prediction returned invalid result for {away} @ {home}")
    
    except AttributeError as e:
        logger.error(f"ML predictor missing predict_game_outcome method: {e}")
    
    except Exception as e:
        logger.warning(f"ML prediction failed for {away} @ {home}: {e}")
        # Keep baseline values in result
    
    return result


# =============================================================================
# USAGE IN YOUR MAIN LOOP
# =============================================================================

"""
Replace your existing code (around line 12020-12075) with this:

# Get ML predictor from session state
ml_predictor = st.session_state.get('ml_predictor')

# Get sentiment for both teams
try:
    home_sentiment = sentiment_analyzer.get_team_sentiment(home, skey) if use_sentiment else {'score': 0, 'trend': 'neutral'}
    away_sentiment = sentiment_analyzer.get_team_sentiment(away, skey) if use_sentiment else {'score': 0, 'trend': 'neutral'}
except Exception:
    home_sentiment = {'score': 0, 'trend': 'neutral'}
    away_sentiment = {'score': 0, 'trend': 'neutral'}

baseline_confidence = max(min_ai_confidence, 0.55)

# Moneyline with AI - HOME TEAM
if inc_ml and "h2h" in mkts:
    hp = _dig(mkts["h2h"], "home.price")
    ap = _dig(mkts["h2h"], "away.price")
    
    if _is_reasonable_moneyline(hp):
        # Get ML prediction using corrected function
        ml_result = process_moneyline_with_ml_predictions(
            home=home,
            away=away,
            skey=skey,
            home_price=hp,
            away_price=ap,
            ml_predictor=ml_predictor,
            use_ml_predictions=use_ml_predictions,
            baseline_confidence=baseline_confidence,
            min_ai_confidence=min_ai_confidence
        )
        
        # Use ML probabilities if successful
        ai_prob = ml_result['home_prob']
        ai_confidence = ml_result['confidence']
        ai_edge = ml_result['edge']
        
        # Only create leg if confidence meets threshold
        if ai_confidence >= min_ai_confidence:
            hp_raw = hp
            hp = _safe_float(hp)
            base_prob = implied_p_from_american(hp)
            decimal_odds = american_to_decimal_safe(hp)
            
            if decimal_odds is not None:
                leg_data = {
                    "event_id": eid,
                    "type": "Moneyline",
                    "team": home,
                    "side": "home",
                    "market": "ML",
                    "label": f"{away} @ {home} — {home} ML @{hp_raw}",
                    "p": base_prob,
                    "ai_prob": ai_prob,
                    "ai_confidence": ai_confidence,
                    "ai_edge": ai_edge,
                    "d": decimal_odds,
                    "sentiment_trend": home_sentiment['trend'],
                    "sport_key": skey,
                    "home_team": home,
                    "away_team": away,
                    "commence_time": commence_time,
                }
                
                # Add ML metadata if prediction was successful
                if ml_result['prediction_successful']:
                    leg_data['ai_model_source'] = ml_result['model_used']
                    leg_data['ai_training_rows'] = ml_result['training_rows']
                    if ml_result['component_probabilities']:
                        leg_data['ai_component_probabilities'] = ml_result['component_probabilities']
                
                # Add context data
                if apisports_payload_home:
                    leg_data['apisports'] = apisports_payload_home
                if sportsdata_payload_home:
                    leg_data['sportsdata'] = sportsdata_payload_home
                
                # Integrate Kalshi validation
                integrate_kalshi_into_leg(
                    leg_data,
                    home,
                    away,
                    'home',
                    base_prob,
                    skey,
                    use_kalshi
                )
                
                all_legs.append(leg_data)
    
    # Moneyline with AI - AWAY TEAM
    if _is_reasonable_moneyline(ap):
        # Get ML prediction (reuse the same result from above!)
        if 'ml_result' not in locals():
            ml_result = process_moneyline_with_ml_predictions(
                home=home,
                away=away,
                skey=skey,
                home_price=hp,
                away_price=ap,
                ml_predictor=ml_predictor,
                use_ml_predictions=use_ml_predictions,
                baseline_confidence=baseline_confidence,
                min_ai_confidence=min_ai_confidence
            )
        
        # Use ML probabilities if successful
        ai_prob = ml_result['away_prob']
        ai_confidence = ml_result['confidence']
        ai_edge = ml_result['edge']
        
        # Only create leg if confidence meets threshold
        if ai_confidence >= min_ai_confidence:
            ap_raw = ap
            ap = _safe_float(ap)
            base_prob = implied_p_from_american(ap)
            decimal_odds = american_to_decimal_safe(ap)
            
            if decimal_odds is not None:
                leg_data = {
                    "event_id": eid,
                    "type": "Moneyline",
                    "team": away,
                    "side": "away",
                    "market": "ML",
                    "label": f"{away} @ {home} — {away} ML @{ap_raw}",
                    "p": base_prob,
                    "ai_prob": ai_prob,
                    "ai_confidence": ai_confidence,
                    "ai_edge": ai_edge,
                    "d": decimal_odds,
                    "sentiment_trend": away_sentiment['trend'],
                    "sport_key": skey,
                    "home_team": home,
                    "away_team": away,
                    "commence_time": commence_time,
                }
                
                # Add ML metadata if prediction was successful
                if ml_result['prediction_successful']:
                    leg_data['ai_model_source'] = ml_result['model_used']
                    leg_data['ai_training_rows'] = ml_result['training_rows']
                    if ml_result['component_probabilities']:
                        leg_data['ai_component_probabilities'] = ml_result['component_probabilities']
                
                # Add context data
                if apisports_payload_away:
                    leg_data['apisports'] = apisports_payload_away
                if sportsdata_payload_away:
                    leg_data['sportsdata'] = sportsdata_payload_away
                
                # Integrate Kalshi validation
                integrate_kalshi_into_leg(
                    leg_data,
                    home,
                    away,
                    'away',
                    base_prob,
                    skey,
                    use_kalshi
                )
                
                all_legs.append(leg_data)
"""


# =============================================================================
# TESTING THE FIX
# =============================================================================

def test_ml_prediction_fix():
    """
    Test function to verify the ML prediction fix works.
    Run this after implementing the fix to verify it's working.
    """
    
    print("🧪 Testing ML Prediction Fix...")
    print("=" * 60)
    
    # Check 1: Verify ml_predictor exists
    ml_predictor = st.session_state.get('ml_predictor')
    if ml_predictor is None:
        print("❌ FAIL: ml_predictor not found in session state")
        print("   → Train a model first in the ML Training tab")
        return False
    else:
        print("✅ PASS: ml_predictor found in session state")
    
    # Check 2: Verify predictor has predict_game_outcome method
    if not hasattr(ml_predictor, 'predict_game_outcome'):
        print("❌ FAIL: ml_predictor missing predict_game_outcome method")
        return False
    else:
        print("✅ PASS: predict_game_outcome method exists")
    
    # Check 3: Test prediction on a real game
    try:
        test_result = ml_predictor.predict_game_outcome(
            home_team="Kansas City Chiefs",
            away_team="Las Vegas Raiders",
            sport_key="americanfootball_nfl"
        )
        
        if test_result is None:
            print("❌ FAIL: Prediction returned None")
            return False
        
        if not isinstance(test_result, dict):
            print("❌ FAIL: Prediction didn't return a dict")
            return False
        
        # Check required fields
        required_fields = ['home_win_prob', 'away_win_prob', 'confidence']
        missing_fields = [f for f in required_fields if f not in test_result]
        
        if missing_fields:
            print(f"❌ FAIL: Missing required fields: {missing_fields}")
            return False
        
        # Check probability values are valid
        home_prob = test_result['home_win_prob']
        away_prob = test_result['away_win_prob']
        confidence = test_result['confidence']
        
        if not (0 <= home_prob <= 1 and 0 <= away_prob <= 1 and 0 <= confidence <= 1):
            print("❌ FAIL: Probabilities out of range [0, 1]")
            print(f"   home_prob={home_prob}, away_prob={away_prob}, confidence={confidence}")
            return False
        
        print("✅ PASS: Test prediction successful")
        print(f"   Home Win Prob: {home_prob:.1%}")
        print(f"   Away Win Prob: {away_prob:.1%}")
        print(f"   Confidence: {confidence:.1%}")
        
        # Check 4: Verify probabilities are different from 58%
        if abs(home_prob - 0.58) < 0.01:
            print("⚠️  WARNING: Probability is suspiciously close to 58%")
            print("   This might indicate the model isn't trained properly")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Prediction raised exception: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# VERIFICATION CHECKLIST
# =============================================================================

"""
After implementing this fix, verify:

1. ✅ Different games show different probabilities
   - Check 5-10 games
   - Each should have unique home/away probabilities
   - Not all games showing 58%

2. ✅ Console logs show ML predictions being called
   - Look for "Calling ML predictor for..." messages
   - Look for "✅ ML prediction successful..." messages
   - If you see warnings, investigate

3. ✅ Leg data includes ML metadata
   - Check all_legs list
   - Each leg should have 'ai_model_source'
   - Each leg should have 'ai_training_rows'
   - 'prediction_successful' should be True

4. ✅ Parlays are ranked differently
   - Generate 2-leg parlays
   - Verify AI scores vary widely
   - Not all showing same score

5. ✅ Expected Value calculations are working
   - Check 'ai_edge' field in legs
   - Should be > 0 for value bets
   - Should be < 0 for poor bets

If any of these checks fail, review the logs and verify:
- ML model is properly trained
- Model has data for the sport you're testing
- The sport_key matches your training data
"""


# =============================================================================
# ADDITIONAL OPTIMIZATIONS (Optional)
# =============================================================================

"""
Once the basic fix is working, consider these optimizations:

1. BATCH PREDICTION (5-10x speedup):
   - Predict all games at once before the loop
   - Store in a lookup dictionary
   - Retrieve predictions as needed

2. CACHING:
   - Cache predictions for 30 minutes
   - Avoid re-predicting same games
   - Use st.cache_data decorator

3. LOGGING:
   - Add detailed logging for debugging
   - Track prediction success rate
   - Monitor model performance

4. ERROR HANDLING:
   - Add fallbacks for prediction failures
   - Track which games fail
   - Alert if failure rate > 10%

5. MONITORING:
   - Track average probabilities per sport
   - Alert if all games show similar probabilities
   - Dashboard for model health
"""

# =============================================================================
# COMMON ERRORS AND SOLUTIONS
# =============================================================================

"""
ERROR 1: "ml_predictor has no attribute 'predict_game_outcome'"
SOLUTION: Your MLPredictor class needs this method. Check ml_predictions.py

ERROR 2: "Prediction returns None"
SOLUTION: Model might not be trained, or training data missing for this sport

ERROR 3: "All probabilities still showing 58%"
SOLUTION: Verify process_moneyline_with_ml_predictions is actually being called

ERROR 4: "KeyError: 'home_win_prob'"
SOLUTION: Prediction method needs to return dict with these exact keys

ERROR 5: "Model not found for sport_key"
SOLUTION: Train model for this specific sport first
"""


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "=" * 60)
    print("Use the functions above to fix your ML prediction issue")
    print("=" * 60)
