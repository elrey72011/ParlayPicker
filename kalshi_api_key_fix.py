"""
Kalshi Integration Fix - For API Key/Secret Authentication
Replace the Kalshi section in parlaypicker_integration_patch.py with this
"""

import streamlit as st
import logging

logger = logging.getLogger(__name__)

def initialize_kalshi_with_api_key():
    """
    Initialize Kalshi using API Key and Secret Key (not email/password)
    """
    if 'kalshi_integrator' not in st.session_state or st.session_state.get('kalshi_integrator') is None:
        try:
            # Get Kalshi API credentials (KEY and SECRET, not email/password!)
            kalshi_api_key = st.secrets.get('kalshi_api_key') or st.session_state.get('kalshi_api_key')
            kalshi_secret_key = st.secrets.get('kalshi_secret_key') or st.session_state.get('kalshi_secret_key')
            
            if kalshi_api_key and kalshi_secret_key:
                from app_core import KalshiIntegrator
                
                # Initialize with API key/secret instead of email/password
                st.session_state['kalshi_integrator'] = KalshiIntegrator(
                    api_key=kalshi_api_key,
                    secret_key=kalshi_secret_key
                )
                st.sidebar.success("✅ Kalshi: Connected (API Key)")
                logger.info("✅ Kalshi integrator initialized with API key")
            else:
                st.session_state['kalshi_integrator'] = None
                st.sidebar.warning("⚠️ Kalshi: No credentials (using synthetic)")
                logger.warning("Kalshi API credentials not found - will use synthetic data")
                
        except Exception as e:
            st.session_state['kalshi_integrator'] = None
            st.sidebar.error(f"❌ Kalshi: {str(e)[:50]}")
            logger.error(f"Kalshi initialization error: {e}")
    else:
        st.sidebar.info("✅ Kalshi: Already initialized")


# ============================================================
# COMPLETE CORRECTED INTEGRATION FUNCTION
# ============================================================

def initialize_all_integrations():
    """
    Initialize ALL data source integrations - CORRECTED for API Key authentication
    Call this ONCE at app startup, BEFORE analyzing games
    """
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔌 Data Source Status")
    
    # ============================================================
    # 1. KALSHI INTEGRATION (CORRECTED - API KEY METHOD)
    # ============================================================
    
    if 'kalshi_integrator' not in st.session_state or st.session_state.get('kalshi_integrator') is None:
        try:
            # CORRECTED: Use API key and secret key (not email/password)
            kalshi_api_key = st.secrets.get('kalshi_api_key') or st.session_state.get('kalshi_api_key')
            kalshi_secret_key = st.secrets.get('kalshi_secret_key') or st.session_state.get('kalshi_secret_key')
            
            if kalshi_api_key and kalshi_secret_key:
                from app_core import KalshiIntegrator
                
                # Try both possible initialization methods
                try:
                    # Method 1: Direct API key/secret
                    st.session_state['kalshi_integrator'] = KalshiIntegrator(
                        api_key=kalshi_api_key,
                        secret_key=kalshi_secret_key
                    )
                except TypeError:
                    # Method 2: Maybe it expects 'api_secret' instead of 'secret_key'
                    st.session_state['kalshi_integrator'] = KalshiIntegrator(
                        api_key=kalshi_api_key,
                        api_secret=kalshi_secret_key
                    )
                
                st.sidebar.success("✅ Kalshi: Connected (API Key)")
                logger.info("✅ Kalshi integrator initialized with API key/secret")
            else:
                st.session_state['kalshi_integrator'] = None
                st.sidebar.warning("⚠️ Kalshi: No credentials (using synthetic)")
                logger.warning("Kalshi API credentials not found - will use synthetic data")
                
        except Exception as e:
            st.session_state['kalshi_integrator'] = None
            st.sidebar.error(f"❌ Kalshi: {str(e)[:50]}")
            logger.error(f"Kalshi initialization error: {e}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        st.sidebar.info("✅ Kalshi: Already initialized")
    
    # ============================================================
    # 2. API-SPORTS INTEGRATION
    # ============================================================
    
    if 'apisports_clients' not in st.session_state or not st.session_state.get('apisports_clients'):
        try:
            apisports_key = st.secrets.get('apisports_key') or st.session_state.get('apisports_key')
            
            if apisports_key:
                from app_core import (
                    APISportsBasketballClient,
                    APISportsFootballClient,
                    APISportsHockeyClient
                )
                
                st.session_state['apisports_clients'] = {
                    'nba': APISportsBasketballClient(api_key=apisports_key),
                    'basketball_nba': APISportsBasketballClient(api_key=apisports_key),
                    'nfl': APISportsFootballClient(api_key=apisports_key),
                    'americanfootball_nfl': APISportsFootballClient(api_key=apisports_key),
                    'nhl': APISportsHockeyClient(api_key=apisports_key),
                    'icehockey_nhl': APISportsHockeyClient(api_key=apisports_key),
                }
                st.sidebar.success("✅ API-Sports: 3 sports")
                logger.info("✅ API-Sports clients initialized")
            else:
                st.session_state['apisports_clients'] = {}
                st.sidebar.info("⚠️ API-Sports: Not configured")
                
        except Exception as e:
            st.session_state['apisports_clients'] = {}
            st.sidebar.error(f"❌ API-Sports: {str(e)[:50]}")
            logger.error(f"API-Sports initialization error: {e}")
    else:
        st.sidebar.info(f"✅ API-Sports: {len(st.session_state['apisports_clients'])} sports")
    
    # ============================================================
    # 3. SPORTSDATA.IO INTEGRATION
    # ============================================================
    
    if 'sportsdata_clients' not in st.session_state or not st.session_state.get('sportsdata_clients'):
        try:
            sportsdata_clients = {}
            
            # NBA
            nba_key = st.secrets.get('sportsdata_nba_key') or st.session_state.get('sportsdata_nba_key')
            if nba_key:
                from app_core import SportsDataNBAClient
                sportsdata_clients['nba'] = SportsDataNBAClient(api_key=nba_key)
                sportsdata_clients['basketball_nba'] = SportsDataNBAClient(api_key=nba_key)
            
            # NFL
            nfl_key = st.secrets.get('sportsdata_nfl_key') or st.session_state.get('sportsdata_nfl_key')
            if nfl_key:
                from app_core import SportsDataNFLClient
                sportsdata_clients['nfl'] = SportsDataNFLClient(api_key=nfl_key)
                sportsdata_clients['americanfootball_nfl'] = SportsDataNFLClient(api_key=nfl_key)
            
            # NHL
            nhl_key = st.secrets.get('sportsdata_nhl_key') or st.session_state.get('sportsdata_nhl_key')
            if nhl_key:
                from app_core import SportsDataNHLClient
                sportsdata_clients['nhl'] = SportsDataNHLClient(api_key=nhl_key)
                sportsdata_clients['icehockey_nhl'] = SportsDataNHLClient(api_key=nhl_key)
            
            # NCAAB
            ncaab_key = st.secrets.get('sportsdata_ncaab_key') or st.session_state.get('sportsdata_ncaab_key')
            if ncaab_key:
                from app_core import SportsDataNCAABClient
                sportsdata_clients['ncaab'] = SportsDataNCAABClient(api_key=ncaab_key)
                sportsdata_clients['basketball_ncaab'] = SportsDataNCAABClient(api_key=ncaab_key)
            
            # NCAAF
            ncaaf_key = st.secrets.get('sportsdata_ncaaf_key') or st.session_state.get('sportsdata_ncaaf_key')
            if ncaaf_key:
                from app_core import SportsDataNCAAFClient
                sportsdata_clients['ncaaf'] = SportsDataNCAAFClient(api_key=ncaaf_key)
                sportsdata_clients['americanfootball_ncaaf'] = SportsDataNCAAFClient(api_key=ncaaf_key)
            
            if sportsdata_clients:
                st.session_state['sportsdata_clients'] = sportsdata_clients
                unique_count = len(set(sportsdata_clients.values()))
                st.sidebar.success(f"✅ SportsData.io: {unique_count} sports")
                logger.info(f"✅ SportsData.io clients initialized: {unique_count} sports")
            else:
                st.session_state['sportsdata_clients'] = {}
                st.sidebar.info("⚠️ SportsData.io: Not configured")
                
        except Exception as e:
            st.session_state['sportsdata_clients'] = {}
            st.sidebar.error(f"❌ SportsData.io: {str(e)[:50]}")
            logger.error(f"SportsData.io initialization error: {e}")
    else:
        unique_count = len(set(st.session_state['sportsdata_clients'].values()))
        st.sidebar.info(f"✅ SportsData.io: {unique_count} sports")
    
    # ============================================================
    # 4. SENTIMENT ANALYZER
    # ============================================================
    
    if 'sentiment_analyzer' not in st.session_state or st.session_state.get('sentiment_analyzer') is None:
        try:
            from app_core import RealSentimentAnalyzer
            
            st.session_state['sentiment_analyzer'] = RealSentimentAnalyzer()
            st.sidebar.success("✅ Sentiment: Active")
            logger.info("✅ Sentiment analyzer initialized")
            
        except Exception as e:
            st.session_state['sentiment_analyzer'] = None
            st.sidebar.warning("⚠️ Sentiment: Using synthetic")
            logger.warning(f"Sentiment analyzer error: {e}")
    else:
        st.sidebar.info("✅ Sentiment: Already initialized")
    
    # ============================================================
    # 5. ML PREDICTOR
    # ============================================================
    
    if 'ml_predictor' not in st.session_state or st.session_state.get('ml_predictor') is None:
        try:
            from app_core import MLPredictor
            
            st.session_state['ml_predictor'] = MLPredictor()
            st.sidebar.success("✅ ML Predictor: Ready")
            logger.info("✅ ML Predictor initialized")
            
        except Exception as e:
            st.session_state['ml_predictor'] = None
            st.sidebar.warning("⚠️ ML: Not available")
            logger.warning(f"ML Predictor error: {e}")
    else:
        st.sidebar.info("✅ ML: Already initialized")
    
    st.sidebar.markdown("---")
    
    # Return summary
    return {
        'kalshi': st.session_state.get('kalshi_integrator') is not None,
        'apisports': len(st.session_state.get('apisports_clients', {})) > 0,
        'sportsdata': len(st.session_state.get('sportsdata_clients', {})) > 0,
        'sentiment': st.session_state.get('sentiment_analyzer') is not None,
        'ml': st.session_state.get('ml_predictor') is not None,
    }


def create_vertex_master_analyzer():
    """
    Create VertexMasterAnalyzer with ALL initialized clients
    Call this BEFORE analyzing each game
    """
    from app_core.vertex_master_analyzer import VertexMasterAnalyzer
    
    # Get theover data if available
    theover_data = st.session_state.get('theover_data', {})
    
    # Create analyzer with ALL clients
    analyzer = VertexMasterAnalyzer(
        odds_api_client=None,  # Not needed - odds come from game dict
        sportsdata_clients=st.session_state.get('sportsdata_clients', {}),
        apisports_clients=st.session_state.get('apisports_clients', {}),
        sentiment_analyzer=st.session_state.get('sentiment_analyzer'),
        local_ml_predictor=st.session_state.get('ml_predictor'),
        theover_data=theover_data,
        kalshi_integrator=st.session_state.get('kalshi_integrator'),
        use_kalshi=st.session_state.get('kalshi_enabled', True),
    )
    
    # Log what's available
    logger.info("="*60)
    logger.info("VertexMasterAnalyzer created with:")
    logger.info(f"  Kalshi: {analyzer.kalshi is not None}")
    logger.info(f"  SportsData clients: {list(analyzer.sportsdata.keys())}")
    logger.info(f"  APISports clients: {list(analyzer.apisports.keys())}")
    logger.info(f"  Sentiment: {analyzer.sentiment is not None}")
    logger.info(f"  ML Predictor: {analyzer.local_ml is not None}")
    logger.info(f"  TheOver data: {list(analyzer.theover.keys())}")
    logger.info("="*60)
    
    return analyzer
