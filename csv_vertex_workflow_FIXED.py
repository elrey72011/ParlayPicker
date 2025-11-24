# ═══════════════════════════════════════════════════════════════
# CSV UPLOAD → VERTEX-POWERED ANALYSIS (FIXED WITH DEBUGGING)
# Now with proper error handling and visible status messages
# ═══════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import re

# ═══════════════════════════════════════════════════════════════
# IMPROVED VERTEX FUNCTIONS WITH DEBUGGING
# ═══════════════════════════════════════════════════════════════

def vertex_analyze_historical_data_debug(game, anthropic_api_key):
    """
    Use Vertex AI to analyze historical game data - WITH DEBUGGING
    """
    if not anthropic_api_key:
        return None, "❌ No API key provided"
    
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        
        prompt = f"""You are an expert sports analyst. Analyze this upcoming game:

GAME: {game['away_team']} @ {game['home_team']}
Sport: {game['sport']}
Date: {game['date']}
Home Odds: {game.get('home_odds', 'N/A')}
Away Odds: {game.get('away_odds', 'N/A')}

Analyze:
1. Recent head-to-head history
2. Home/away performance trends  
3. Injuries or roster changes you know about
4. Betting line value assessment
5. Sport-specific factors (rest, travel, etc)

RESPOND WITH ONLY THIS JSON (no other text):
{{
  "home_win_probability": 65.0,
  "away_win_probability": 35.0,
  "confidence_level": 78,
  "risk_assessment": "Low",
  "key_factors": ["Factor 1", "Factor 2", "Factor 3"],
  "historical_edge": "Brief summary",
  "betting_recommendation": "Specific bet",
  "value_rating": 7.5,
  "sentiment": "Bullish on home",
  "concerns": ["Concern 1", "Concern 2"]
}}"""
        
        # Call API with timeout
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
            timeout=30.0
        )
        
        response_text = message.content[0].text
        
        # Try to parse JSON
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            analysis = json.loads(json_match.group())
            
            # Validate we got actual probabilities (not defaults)
            home_prob = float(analysis.get('home_win_probability', 50.0))
            if home_prob == 50.0 or home_prob == 50.5:
                return None, "⚠️ Got default probability (50%), likely API issue"
            
            return analysis, "✅ Success"
        else:
            return None, f"❌ JSON parse failed. Response: {response_text[:200]}"
    
    except anthropic.APIError as e:
        return None, f"❌ API Error: {str(e)[:100]}"
    except anthropic.APITimeoutError:
        return None, "❌ API Timeout (>30s)"
    except json.JSONDecodeError as e:
        return None, f"❌ JSON decode error: {str(e)[:100]}"
    except Exception as e:
        return None, f"❌ Unexpected error: {str(e)[:100]}"


def comprehensive_vertex_analysis_debug(game, ml_result, sentiment_result, anthropic_api_key):
    """
    Vertex AI synthesizes ALL data for final winning probability - WITH DEBUGGING
    """
    if not anthropic_api_key:
        return None, "❌ No API key"
    
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        
        prompt = f"""FINAL ANALYSIS - Synthesize all data to make best betting decision:

GAME: {game['away_team']} @ {game['home_team']}
Sport: {game['sport']}
Date: {game['date']}

MACHINE LEARNING:
- ML Home Win: {ml_result['ml_home_prob']:.1f}%
- ML Away Win: {ml_result['ml_away_prob']:.1f}%
- ML Confidence: {ml_result['ml_confidence']:.1f}%

SENTIMENT:
- Sentiment: {sentiment_result['sentiment_label']}
- Score: {sentiment_result['sentiment_score']:.1f}

BETTING LINES:
- Home Odds: {game.get('home_odds', 'N/A')}
- Away Odds: {game.get('away_odds', 'N/A')}

YOUR TASK: Produce FINAL winning probability combining ML, sentiment, lines, and YOUR sports knowledge.

RESPOND WITH ONLY THIS JSON (no other text):
{{
  "final_home_win_probability": 62.5,
  "final_away_win_probability": 37.5,
  "overall_confidence": 82,
  "risk_level": "Low",
  "recommended_bet": "Home Team -5.5",
  "bet_confidence": "High",
  "value_assessment": 8.0,
  "key_insights": ["Insight 1", "Insight 2", "Insight 3"],
  "edge_analysis": "Edge description",
  "final_verdict": "STRONG HOME PLAY",
  "stars": 4
}}"""
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
            timeout=30.0
        )
        
        response_text = message.content[0].text
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        
        if json_match:
            analysis = json.loads(json_match.group())
            
            # Validate actual analysis happened
            final_prob = float(analysis.get('final_home_win_probability', 50.0))
            if final_prob == 50.0 or final_prob == 50.5:
                return None, "⚠️ Got default probability, API issue"
            
            return analysis, "✅ Success"
        else:
            return None, f"❌ Parse failed: {response_text[:200]}"
    
    except Exception as e:
        return None, f"❌ Error: {str(e)[:100]}"


def run_complete_analysis_with_debug(games, ml_predictor, sentiment_analyzer, anthropic_api_key):
    """
    Run complete analysis with DETAILED STATUS REPORTING
    """
    results = []
    errors_log = []
    
    progress_bar = st.progress(0)
    status_container = st.container()
    
    for idx, game in enumerate(games):
        game_name = f"{game['away_team']} @ {game['home_team']}"
        
        with status_container:
            st.write(f"### 🔍 Game {idx+1}/{len(games)}: {game_name}")
            
            # Step 1: Vertex Historical
            hist_col, hist_status = st.columns([3, 1])
            with hist_col:
                st.write("🧠 Vertex Historical Analysis...")
            
            vertex_historical, hist_error = vertex_analyze_historical_data_debug(
                game, anthropic_api_key
            )
            
            with hist_status:
                if vertex_historical:
                    st.success("✅")
                else:
                    st.error("❌")
                    st.caption(hist_error)
                    errors_log.append(f"Game {idx+1} Historical: {hist_error}")
            
            # Step 2: ML
            ml_col, ml_status = st.columns([3, 1])
            with ml_col:
                st.write("🤖 ML Predictions...")
            
            if ml_predictor:
                try:
                    prediction = ml_predictor.predict_game(game['home_team'], game['away_team'])
                    ml_result = {
                        'ml_home_prob': prediction.get('home_win_prob', 50.0),
                        'ml_away_prob': prediction.get('away_win_prob', 50.0),
                        'ml_confidence': prediction.get('confidence', 50.0)
                    }
                    with ml_status:
                        st.success("✅")
                except Exception as e:
                    ml_result = {'ml_home_prob': 50.0, 'ml_away_prob': 50.0, 'ml_confidence': 50.0}
                    with ml_status:
                        st.warning("⚠️")
                    errors_log.append(f"Game {idx+1} ML: {str(e)[:50]}")
            else:
                ml_result = {'ml_home_prob': 50.0, 'ml_away_prob': 50.0, 'ml_confidence': 50.0}
                with ml_status:
                    st.info("⊘")
            
            # Step 3: Sentiment
            sent_col, sent_status = st.columns([3, 1])
            with sent_col:
                st.write("📰 Sentiment Analysis...")
            
            if sentiment_analyzer:
                try:
                    result = sentiment_analyzer.analyze_game(
                        game['home_team'], game['away_team'], game['sport']
                    )
                    sentiment_result = {
                        'sentiment_score': result.get('score', 50.0),
                        'sentiment_label': result.get('label', 'Neutral')
                    }
                    with sent_status:
                        st.success("✅")
                except Exception as e:
                    sentiment_result = {'sentiment_score': 50.0, 'sentiment_label': 'Neutral'}
                    with sent_status:
                        st.warning("⚠️")
                    errors_log.append(f"Game {idx+1} Sentiment: {str(e)[:50]}")
            else:
                sentiment_result = {'sentiment_score': 50.0, 'sentiment_label': 'Neutral'}
                with sent_status:
                    st.info("⊘")
            
            # Step 4: Vertex Final Synthesis
            final_col, final_status = st.columns([3, 1])
            with final_col:
                st.write("🎯 Vertex Final Synthesis...")
            
            final_analysis, final_error = comprehensive_vertex_analysis_debug(
                game, ml_result, sentiment_result, anthropic_api_key
            )
            
            with final_status:
                if final_analysis:
                    st.success("✅")
                else:
                    st.error("❌")
                    st.caption(final_error)
                    errors_log.append(f"Game {idx+1} Final: {final_error}")
            
            st.write("---")
        
        # Compile result
        result = {
            'Game': game_name,
            'Sport': game['sport'],
            'Date': game['date'],
            
            # Historical
            'Vertex Historical': "✅" if vertex_historical else "❌",
            'Historical Confidence': f"{vertex_historical.get('confidence_level', 0):.0f}%" if vertex_historical else "N/A",
            
            # ML
            'ML Home %': f"{ml_result['ml_home_prob']:.1f}%",
            'ML Away %': f"{ml_result['ml_away_prob']:.1f}%",
            
            # Sentiment
            'Sentiment': sentiment_result['sentiment_label'],
            'Sentiment Score': f"{sentiment_result['sentiment_score']:.1f}",
            
            # Final
            'Final Analysis': "✅" if final_analysis else "❌",
            'FINAL Home %': f"{final_analysis.get('final_home_win_probability', 0):.1f}%" if final_analysis else "N/A",
            'FINAL Away %': f"{final_analysis.get('final_away_win_probability', 0):.1f}%" if final_analysis else "N/A",
            'Confidence': f"{final_analysis.get('overall_confidence', 0):.0f}%" if final_analysis else "N/A",
            'Risk': final_analysis.get('risk_level', 'N/A') if final_analysis else "N/A",
            'Recommended Bet': final_analysis.get('recommended_bet', 'N/A') if final_analysis else "N/A",
            'Stars': final_analysis.get('stars', 0) if final_analysis else 0,
            'Verdict': final_analysis.get('final_verdict', 'N/A') if final_analysis else "N/A",
            
            # Errors
            'Status': "✅ Complete" if (vertex_historical and final_analysis) else "⚠️ Partial" if (vertex_historical or final_analysis) else "❌ Failed"
        }
        
        results.append(result)
        progress_bar.progress((idx + 1) / len(games))
    
    progress_bar.empty()
    
    # Show error log
    if errors_log:
        with st.expander("⚠️ Errors Encountered", expanded=True):
            for error in errors_log:
                st.error(error)
    
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# PARSE CSV FUNCTION (unchanged)
# ═══════════════════════════════════════════════════════════════

def parse_theover_csv(uploaded_file):
    """Parse uploaded CSV"""
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.lower()
        
        games = []
        for idx, row in df.iterrows():
            # Try different column name variations
            home = row.get('home_team') or row.get('home') or row.get('home team') or ''
            away = row.get('away_team') or row.get('away') or row.get('away team') or ''
            sport = row.get('sport') or row.get('league') or 'unknown'
            date = row.get('date') or row.get('game_date') or row.get('game date') or ''
            
            if not home or not away:
                continue  # Skip rows without teams
            
            games.append({
                'game_id': f"game_{idx}",
                'home_team': str(home).strip(),
                'away_team': str(away).strip(),
                'sport': str(sport).strip().lower(),
                'date': str(date),
                'home_odds': row.get('home_odds') or row.get('home odds'),
                'away_odds': row.get('away_odds') or row.get('away odds'),
                'spread': row.get('spread'),
                'total': row.get('total')
            })
        
        return games, df
    
    except Exception as e:
        st.error(f"CSV parsing error: {e}")
        return None, None


# ═══════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════

st.write("---")
st.write("---")
st.header("📊 CSV Upload → Vertex Analysis (DEBUG MODE)")

st.info("""
**This version shows EXACTLY what's happening:**
- ✅ = Step succeeded
- ❌ = Step failed (with error message)
- ⚠️ = Step partially worked
- ⊘ = Step skipped (optional)

Upload CSV and watch each step's status in real-time!
""")

# API Key Check
st.subheader("🔑 Step 1: Verify API Key")
anthropic_key = st.session_state.get('anthropic_api_key', '')

if anthropic_key:
    # Test the API key
    try:
        import anthropic
        test_client = anthropic.Anthropic(api_key=anthropic_key)
        st.success(f"✅ Anthropic API key detected ({anthropic_key[:8]}...)")
    except Exception as e:
        st.error(f"❌ API key invalid: {e}")
        anthropic_key = None
else:
    st.error("❌ No Anthropic API key found")
    st.info("Add your API key in the sidebar under 'Anthropic API Key'")

st.write("---")

# CSV Upload
st.subheader("📤 Step 2: Upload CSV")
uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=['csv'],
    help="Columns needed: Home Team, Away Team, Sport, Date"
)

if uploaded_file is not None:
    with st.spinner("📊 Parsing CSV..."):
        games, raw_df = parse_theover_csv(uploaded_file)
    
    if games:
        st.success(f"✅ Parsed {len(games)} games")
        
        # Preview
        with st.expander("📋 View Parsed Games"):
            preview = pd.DataFrame([{
                'Away': g['away_team'],
                'Home': g['home_team'],
                'Sport': g['sport'],
                'Date': g['date']
            } for g in games])
            st.dataframe(preview, use_container_width=True)
        
        st.write("---")
        
        # Analysis
        st.subheader("🚀 Step 3: Run Analysis")
        
        ml_predictor = st.session_state.get('ml_predictor')
        sentiment_analyzer = st.session_state.get('sentiment_analyzer')
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"ML Predictor: {'✅ Available' if ml_predictor else '⊘ Not available'}")
        with col2:
            st.info(f"Sentiment: {'✅ Available' if sentiment_analyzer else '⊘ Not available'}")
        
        # Limit games for testing
        max_games = st.slider("Max games to analyze (for testing)", 1, len(games), min(5, len(games)))
        games_to_analyze = games[:max_games]
        
        if not anthropic_key:
            st.error("❌ Cannot proceed without Anthropic API key")
        else:
            if st.button("🚀 Run Analysis with Debug Mode", type="primary"):
                st.write("### 📊 Analysis Progress")
                
                results_df = run_complete_analysis_with_debug(
                    games_to_analyze,
                    ml_predictor,
                    sentiment_analyzer,
                    anthropic_key
                )
                
                st.write("---")
                st.success(f"✅ Analysis complete for {len(results_df)} games")
                
                # Summary
                st.write("### 📊 Results Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    complete = len(results_df[results_df['Status'] == "✅ Complete"])
                    st.metric("Fully Analyzed", complete)
                with col2:
                    partial = len(results_df[results_df['Status'] == "⚠️ Partial"])
                    st.metric("Partial", partial)
                with col3:
                    failed = len(results_df[results_df['Status'] == "❌ Failed"])
                    st.metric("Failed", failed)
                
                # Results table
                st.write("### 📊 Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Download
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    "vertex_analysis_debug.csv",
                    "text/csv"
                )
                
                # Analysis of problems
                if failed > 0 or partial > 0:
                    st.write("### 🔍 Problem Analysis")
                    
                    # Check what's failing
                    hist_fails = len(results_df[results_df['Vertex Historical'] == "❌"])
                    final_fails = len(results_df[results_df['Final Analysis'] == "❌"])
                    
                    if hist_fails > 0:
                        st.error(f"❌ Historical analysis failed {hist_fails} times")
                        st.info("Possible causes: API timeout, rate limit, JSON parse error")
                    
                    if final_fails > 0:
                        st.error(f"❌ Final synthesis failed {final_fails} times")
                        st.info("Possible causes: API timeout, rate limit, JSON parse error")
                    
                    # Recommendations
                    st.write("**Recommendations:**")
                    st.write("- Try analyzing fewer games at once")
                    st.write("- Check your API key has sufficient credits")
                    st.write("- Look at error messages in the expander above")
                    st.write("- Try again with a delay between games")

st.write("---")
