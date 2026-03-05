"""
Complete Analysis Workflow Implementation
Add this to your streamlit_app__13_.py to implement the proper pipeline
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional
import logging

from core.probability_utils import ensure_probability_column

logger = logging.getLogger(__name__)


def show_complete_analysis_workflow():
    """
    Complete analysis workflow with proper order:
    1. Upload Data
    2. Vertex AI Analyzer
    3. ML Predictions
    4. Master Analysis
    5. Best Bets
    6. Parlays
    """
    
    st.title("🎯 Complete Betting Analysis Pipeline")
    st.caption("Follow the steps in order for best results")
    
    # Initialize session state for results
    if 'workflow_results' not in st.session_state:
        st.session_state['workflow_results'] = {
            'vertex': None,
            'ml': None,
            'sentiment': None,
            'master': None,
            'best_bets': None,
            'parlays': None
        }
    
    # ============================================================================
    # STEP 1: UPLOAD DATA
    # ============================================================================
    
    st.header("📊 Step 1: Upload Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        spreads_file = st.file_uploader(
            "Upload Spreads CSV",
            type=['csv'],
            key='spreads_upload_workflow'
        )
    
    with col2:
        ml_file = st.file_uploader(
            "Upload Moneyline CSV", 
            type=['csv'],
            key='ml_upload_workflow'
        )
    
    with col3:
        totals_file = st.file_uploader(
            "Upload Totals CSV",
            type=['csv'],
            key='totals_upload_workflow'
        )
    
    # Load data
    spreads_df = None
    ml_df = None
    totals_df = None
    
    if spreads_file:
        spreads_df = pd.read_csv(spreads_file)
        st.success(f"✅ Loaded {len(spreads_df)} spread picks")
    
    if ml_file:
        ml_df = pd.read_csv(ml_file)
        st.success(f"✅ Loaded {len(ml_df)} moneyline picks")
    
    if totals_file:
        totals_df = pd.read_csv(totals_file)
        st.success(f"✅ Loaded {len(totals_df)} totals picks")
    
    # Must have at least spreads to continue
    if spreads_df is None:
        st.info("👆 Upload at least the Spreads CSV to begin analysis")
        return
    
    st.markdown("---")
    
    # ============================================================================
    # STEP 2: VERTEX AI ANALYZER
    # ============================================================================
    
    st.header("🤖 Step 2: Vertex AI Analyzer")
    st.caption("Get AI predictions for each game")
    
    vertex_status = st.session_state['workflow_results']['vertex']
    
    if vertex_status is not None:
        st.success(f"✅ Vertex AI Complete - Analyzed {len(vertex_status)} games")
        
        if st.button("🔄 Re-run Vertex AI Analysis"):
            st.session_state['workflow_results']['vertex'] = None
            st.rerun()
    else:
        if st.button("🎯 Run Vertex AI Analyzer", type="primary"):
            with st.spinner("Running Vertex AI analysis..."):
                try:
                    from theover_vertex_analyzer import analyze_theover_spreads_with_vertex
                    from ml_predictions import is_vertex_ai_enabled
                    
                    if not is_vertex_ai_enabled():
                        st.error("❌ Vertex AI not enabled. Enable in settings.")
                    else:
                        # Run analysis
                        results = analyze_theover_spreads_with_vertex(
                            spreads_df,
                            sportsdata_client=None,
                            apisports_client=None
                        )
                        
                        if not results.empty:
                            st.session_state['workflow_results']['vertex'] = results
                            st.success(f"✅ Analyzed {len(results)} games!")
                            st.rerun()
                        else:
                            st.warning("No results from Vertex AI")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Vertex AI error: {e}", exc_info=True)
    
    # Show preview if available
    if vertex_status is not None:
        with st.expander("📊 Preview Vertex AI Results"):
            st.dataframe(
                vertex_status[['home_team', 'away_team', 'ai_probability', 'ev_percentage', 'confidence_level']].head(10),
                width="stretch"
            )
    
    st.markdown("---")
    
    # ============================================================================
    # STEP 3: ML PREDICTIONS (Optional but recommended)
    # ============================================================================
    
    st.header("🧠 Step 3: ML Predictions (Optional)")
    st.caption("Add trained model predictions for better accuracy")
    
    ml_status = st.session_state['workflow_results']['ml']
    
    if ml_status is not None:
        st.success(f"✅ ML Predictions Complete - {len(ml_status)} games")
        
        if st.button("🔄 Re-run ML Predictions"):
            st.session_state['workflow_results']['ml'] = None
            st.rerun()
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info("💡 Optional: Run ML predictions for additional confidence")
        
        with col2:
            if st.button("🤖 Run ML", key="run_ml_btn"):
                with st.spinner("Running ML predictions..."):
                    try:
                        # Placeholder - replace with your ML predictor
                        ml_results = run_ml_predictions(spreads_df)
                        st.session_state['workflow_results']['ml'] = ml_results
                        st.success("✅ ML Complete!")
                        st.rerun()
                    except Exception as e:
                        st.warning(f"ML predictions not available: {e}")
    
    if ml_status is not None:
        with st.expander("📊 Preview ML Results"):
            st.write("ML predictions available for Master Analysis")
    
    st.markdown("---")
    
    # ============================================================================
    # STEP 4: MASTER ANALYSIS
    # ============================================================================
    
    st.header("🌟 Step 4: Master Analysis")
    st.caption("Combines all analyzers for consensus predictions")
    
    # Check prerequisites
    can_run_master = vertex_status is not None
    
    if not can_run_master:
        st.warning("⚠️ Complete Step 2 (Vertex AI) first")
    else:
        master_status = st.session_state['workflow_results']['master']
        
        if master_status is not None:
            st.success(f"✅ Master Analysis Complete - {len(master_status)} games analyzed")
            
            if st.button("🔄 Re-run Master Analysis"):
                st.session_state['workflow_results']['master'] = None
                st.rerun()
        else:
            # Show what will be included
            st.write("**Data sources available:**")
            if vertex_status is not None:
                st.write("✅ Vertex AI predictions")
            if ml_status is not None:
                st.write("✅ ML model predictions")
            else:
                st.write("⚪ ML predictions (optional - not run)")
            
            if st.button("🌟 Run Master Analysis", type="primary"):
                with st.spinner("Running master analysis..."):
                    try:
                        master_results = run_master_analysis(
                            vertex_results=vertex_status,
                            ml_results=ml_status,
                            spreads_df=spreads_df
                        )
                        
                        st.session_state['workflow_results']['master'] = master_results
                        st.success(f"✅ Master Analysis Complete!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        logger.error(f"Master analysis error: {e}", exc_info=True)
        
        if master_status is not None:
            with st.expander("📊 Preview Master Analysis Results"):
                preview_cols = ['home_team', 'away_team', 'consensus_prob', 'confidence_level', 'expected_value']
                available_cols = [col for col in preview_cols if col in master_status.columns]
                st.dataframe(master_status[available_cols].head(10), width="stretch")
    
    st.markdown("---")
    
    # ============================================================================
    # STEP 5: BEST BETS COMPUTATION
    # ============================================================================
    
    st.header("🎯 Step 5: Best Bet Recommendations")
    st.caption("Top opportunities ranked by expected value")
    
    can_compute_bets = st.session_state['workflow_results']['master'] is not None
    
    if not can_compute_bets:
        st.warning("⚠️ Complete Step 4 (Master Analysis) first")
    else:
        best_bets = st.session_state['workflow_results']['best_bets']
        
        if best_bets is not None:
            st.success(f"✅ Found {len(best_bets)} best bet opportunities")
            
            # Display best bets
            display_best_bets(best_bets)
            
            if st.button("🔄 Recompute Best Bets"):
                st.session_state['workflow_results']['best_bets'] = None
                st.rerun()
        else:
            # Configuration
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_confidence = st.slider(
                    "Minimum Confidence %",
                    min_value=50,
                    max_value=70,
                    value=53,
                    key="min_conf_best_bets"
                )
            
            with col2:
                min_edge = st.slider(
                    "Minimum Edge %",
                    min_value=0,
                    max_value=10,
                    value=3,
                    key="min_edge_best_bets"
                )
            
            with col3:
                max_bets = st.number_input(
                    "Max Bets to Show",
                    min_value=5,
                    max_value=25,
                    value=10,
                    key="max_bets_show"
                )
            
            if st.button("🎯 Compute Best Bets", type="primary"):
                with st.spinner("Computing best bets..."):
                    master_results = st.session_state['workflow_results']['master']
                    
                    best_bets = compute_best_bets(
                        master_results,
                        min_confidence=min_confidence / 100,
                        min_edge=min_edge / 100,
                        max_count=max_bets
                    )
                    
                    st.session_state['workflow_results']['best_bets'] = best_bets
                    st.rerun()
    
    st.markdown("---")
    
    # ============================================================================
    # STEP 6: PARLAY BUILDER
    # ============================================================================
    
    st.header("🎲 Step 6: Parlay Builder")
    st.caption("Build optimal parlay combinations")
    
    can_build_parlays = st.session_state['workflow_results']['best_bets'] is not None
    
    if not can_build_parlays:
        st.warning("⚠️ Complete Step 5 (Best Bets) first")
    else:
        parlays = st.session_state['workflow_results']['parlays']
        
        if parlays is not None:
            st.success(f"✅ Built {sum(len(v) for v in parlays.values())} optimal parlays")
            
            # Display parlays
            display_parlays(parlays)
            
            if st.button("🔄 Rebuild Parlays"):
                st.session_state['workflow_results']['parlays'] = None
                st.rerun()
        else:
            # Configuration
            col1, col2, col3 = st.columns(3)
            
            with col1:
                parlay_sizes = st.multiselect(
                    "Parlay Sizes",
                    options=[2, 3, 4, 5, 6],
                    default=[2, 3, 4],
                    key="parlay_sizes_select"
                )
            
            with col2:
                max_parlays_per_size = st.number_input(
                    "Max per Size",
                    min_value=5,
                    max_value=20,
                    value=10,
                    key="max_parlays_per"
                )
            
            with col3:
                check_correlation = st.checkbox(
                    "Check Correlation",
                    value=True,
                    key="check_corr",
                    help="Avoid same-game parlays"
                )
            
            if st.button("🎲 Build Parlays", type="primary"):
                with st.spinner("Building parlays..."):
                    best_bets = st.session_state['workflow_results']['best_bets']
                    
                    parlays = build_optimal_parlays(
                        best_bets,
                        parlay_sizes=parlay_sizes,
                        max_per_size=max_parlays_per_size,
                        check_correlation=check_correlation
                    )
                    
                    st.session_state['workflow_results']['parlays'] = parlays
                    st.rerun()


def run_ml_predictions(spreads_df: pd.DataFrame) -> pd.DataFrame:
    """Run ML predictions on spreads"""
    # Placeholder - implement your ML predictor
    results = spreads_df.copy()
    results['ml_probability'] = 0.55  # Placeholder
    return results


def run_master_analysis(
    vertex_results: pd.DataFrame,
    ml_results: Optional[pd.DataFrame],
    spreads_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine all analyzer results into master analysis
    """
    master = vertex_results.copy()
    master = ensure_probability_column(master, "ai_probability")

    logger.info(f"Master columns: {list(master.columns)}")

    if ml_results is not None:
        ml_results = ml_results.copy()
        ml_results = ensure_probability_column(ml_results, "ml_probability")
        logger.info(f"ML result columns: {list(ml_results.columns)}")

        if 'game_id' in master.columns and 'game_id' in ml_results.columns:
            master = master.merge(
                ml_results[['game_id', 'ml_probability']],
                on='game_id',
                how='left'
            )
        else:
            if len(master) == len(ml_results):
                master['ml_probability'] = ml_results['ml_probability'].values
            else:
                master['ml_probability'] = 0.5

        master['ml_probability'] = master['ml_probability'].fillna(0.5)
    else:
        logger.info("ML result columns: []")
        master['ml_probability'] = 0.5

    required_cols = ["ai_probability", "ml_probability"]
    for col in required_cols:
        if col not in master.columns:
            raise ValueError(f"Missing required column: {col}")

    master["combined_probability"] = (
        master["ai_probability"] + master["ml_probability"]
    ) / 2
    master['consensus_prob'] = master['combined_probability']

    # Recalculate confidence levels
    master['edge'] = abs(master['consensus_prob'] - 0.5)
    master['confidence_level'] = master['edge'].apply(
        lambda x: 'High' if x > 0.15 else ('Medium' if x > 0.08 else 'Low')
    )

    # Recalculate expected value
    master['expected_value'] = master.apply(
        lambda row: calculate_ev(row['consensus_prob'], row['line']),
        axis=1
    )

    return master


def calculate_ev(probability: float, line: float) -> float:
    """Calculate expected value"""
    bet_to_win = 100
    bet_to_risk = 110
    ev = (probability * bet_to_win) - ((1 - probability) * bet_to_risk)
    return (ev / bet_to_risk) * 100


def compute_best_bets(
    master_results: pd.DataFrame,
    min_confidence: float = 0.53,
    min_edge: float = 0.03,
    max_count: int = 10
) -> pd.DataFrame:
    """
    Compute best betting opportunities
    """
    # Filter by confidence
    filtered = master_results[
        master_results['consensus_prob'] >= min_confidence
    ].copy()
    
    # Filter by edge
    filtered = filtered[
        abs(filtered['consensus_prob'] - 0.5) >= min_edge
    ]
    
    # Sort by expected value
    filtered = filtered.sort_values('expected_value', ascending=False)
    
    # Return top N
    return filtered.head(max_count)


def build_optimal_parlays(
    best_bets: pd.DataFrame,
    parlay_sizes: List[int] = [2, 3, 4],
    max_per_size: int = 10,
    check_correlation: bool = True
) -> Dict[str, List[Dict]]:
    """
    Build optimal parlay combinations
    """
    from itertools import combinations
    
    parlays = {f'{size}-leg': [] for size in parlay_sizes}
    
    for size in parlay_sizes:
        for combo in combinations(best_bets.iterrows(), size):
            # Extract rows
            bets = [row for idx, row in combo]
            
            # Check correlation if enabled
            if check_correlation:
                if are_correlated(bets):
                    continue
            
            # Calculate parlay probability
            parlay_prob = 1.0
            for bet in bets:
                parlay_prob *= bet['consensus_prob']
            
            # Calculate parlay odds (assume -110 for each)
            parlay_odds = (2.0 / 1.1) ** size  # Simplified
            
            # Calculate EV
            parlay_ev = (parlay_prob * (parlay_odds - 1) * 100) - \
                       ((1 - parlay_prob) * 100)
            
            parlays[f'{size}-leg'].append({
                'legs': bets,
                'probability': parlay_prob,
                'odds': parlay_odds,
                'expected_value': parlay_ev
            })
        
        # Sort by EV and keep top N
        parlays[f'{size}-leg'].sort(key=lambda x: x['expected_value'], reverse=True)
        parlays[f'{size}-leg'] = parlays[f'{size}-leg'][:max_per_size]
    
    return parlays


def are_correlated(bets: List) -> bool:
    """Check if bets are from same game (correlated)"""
    # Check if any two bets involve same teams
    games = set()
    for bet in bets:
        game_key = tuple(sorted([bet['home_team'], bet['away_team']]))
        if game_key in games:
            return True
        games.add(game_key)
    return False


def display_best_bets(best_bets: pd.DataFrame):
    """Display best bets in nice format"""
    st.subheader(f"🎯 Top {len(best_bets)} Best Bets")
    
    for idx, bet in best_bets.iterrows():
        # Color code
        if bet['expected_value'] > 5:
            icon = '🟢'
        elif bet['expected_value'] > 2:
            icon = '🟡'
        else:
            icon = '⚪'
        
        col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])
        
        with col1:
            st.write(f"{icon} **{bet['away_team']} @ {bet['home_team']}**")
            st.caption(f"{bet['pick']} {bet['line']:+.1f}")
        
        with col2:
            st.metric("Win Prob", f"{bet['consensus_prob']*100:.1f}%")
        
        with col3:
            st.metric("EV", f"{bet['expected_value']:+.2f}%")
        
        with col4:
            st.metric("Confidence", bet['confidence_level'])
        
        with col5:
            st.write("")  # Spacer


def display_parlays(parlays: Dict[str, List[Dict]]):
    """Display parlays in tabs"""
    tabs = st.tabs([f"{size} Parlays" for size in parlays.keys()])
    
    for tab, (size, parlay_list) in zip(tabs, parlays.items()):
        with tab:
            if not parlay_list:
                st.info(f"No {size} parlays generated")
                continue
            
            st.write(f"**Top {len(parlay_list)} {size} Parlays**")
            
            for i, parlay in enumerate(parlay_list, 1):
                with st.expander(
                    f"#{i} - Prob: {parlay['probability']*100:.1f}% - "
                    f"EV: {parlay['expected_value']:+.2f}%",
                    expanded=(i <= 3)
                ):
                    st.write(f"**Combined Probability:** {parlay['probability']*100:.1f}%")
                    st.write(f"**Parlay Odds:** {parlay['odds']:.2f}x")
                    st.write(f"**Expected Value:** {parlay['expected_value']:+.2f}%")
                    
                    st.write("**Legs:**")
                    for j, bet in enumerate(parlay['legs'], 1):
                        st.write(
                            f"{j}. {bet['away_team']} @ {bet['home_team']} - "
                            f"{bet['pick']} {bet['line']:+.1f} "
                            f"({bet['consensus_prob']*100:.1f}%)"
                        )


# Add this to your streamlit_app__13_.py:
"""
# In your main app, add a new section or page:

if __name__ == "__main__":
    # ... your existing code ...
    
    # Add new workflow section
    st.markdown("---")
    show_complete_analysis_workflow()
"""
