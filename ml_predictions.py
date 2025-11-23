def show_vertex_ai_prediction_section(home_team, away_team, home_stats=None, away_stats=None):
    """
    Display Vertex AI prediction with REAL features
    
    Args:
        home_team: Home team name
        away_team: Away team name
        home_stats: Dict with home team stats (win%, avg points, etc.)
        away_stats: Dict with away team stats
    """
    if not is_vertex_ai_enabled():
        return
    
    st.markdown("---")
    st.subheader("🤖 Google Cloud Vertex AI Prediction")
    st.caption("Advanced ML prediction powered by Google Cloud")
    
    # BUILD REAL FEATURES
    # Example: Use win percentage and average points
    if home_stats and away_stats:
        features = [
            home_stats.get('win_pct', 0.5),     # Home team win %
            away_stats.get('win_pct', 0.5),     # Away team win %
            home_stats.get('avg_points', 100),  # Home team avg points
            away_stats.get('avg_points', 100),  # Away team avg points
            # Add more features as needed
        ]
    else:
        # Fallback to dummy features
        features = [1, 1]
        st.warning("⚠️ Using dummy features - pass real stats for accurate predictions")
    
    if st.button("Get Vertex AI Prediction", key=f"vertex_ai_predict_{home_team}_{away_team}"):
        import time
        start_time = time.time()
        
        with st.spinner("Getting prediction from Vertex AI..."):
            prediction = get_vertex_ai_prediction(features)
        
        elapsed_time = time.time() - start_time
        
        if prediction is not None:
            st.success(f"✅ Prediction received in {elapsed_time:.2f}s")
            
            # Show if cached
            if elapsed_time < 0.5:
                st.info("⚡ Cached result (instant)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Win Probability",
                    f"{prediction * 100:.1f}%",
                    help="Vertex AI predicted win probability"
                )
            
            with col2:
                confidence = abs(prediction - 0.5) * 2
                st.metric(
                    "Confidence",
                    f"{confidence * 100:.1f}%",
                    help="How confident the model is"
                )
            
            if prediction > 0.5:
                st.info(f"📈 Model favors: **{home_team}**")
            else:
                st.info(f"📈 Model favors: **{away_team}**")
            
            # Show features used
            with st.expander("📊 Features Used"):
                st.write(features)
                
        else:
            st.error("❌ Vertex AI prediction failed. Check logs.")
