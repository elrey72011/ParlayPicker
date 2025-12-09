# vertex_ai_integration.py
"""
Google Vertex AI Integration for ParlayDesk
Uses Claude through Google Cloud Vertex AI (Anthropic on Vertex)
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import Vertex AI dependencies
try:
    from anthropic import AnthropicVertex
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    logger.warning("AnthropicVertex not available. Install with: pip install anthropic[vertex]")

try:
    from google.cloud import aiplatform
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    logger.warning("Google Cloud AI Platform not available. Install with: pip install google-cloud-aiplatform")


class VertexAIAnalyzer:
    """
    Analyzes sports betting games using Claude through Google Vertex AI.
    
    This uses the proper Anthropic on Vertex integration, not direct API calls.
    """
    
    def __init__(self, project_id: Optional[str] = None, region: str = "us-east5"):
        """
        Initialize Vertex AI Analyzer.
        
        Args:
            project_id: Your Google Cloud project ID. If None, will try to use
                        GCP_PROJECT_ID or GOOGLE_CLOUD_PROJECT from environment.
            region: GCP region (us-east5 has Claude 3.5/4.5 Sonnet)
        
        Raises:
            ValueError: If Vertex AI dependencies not available
            ValueError: If project_id not provided and not found in env
        """
        if not VERTEX_AI_AVAILABLE:
            raise ValueError(
                "Vertex AI dependencies not installed. "
                "Install with: pip install anthropic[vertex] google-cloud-aiplatform"
            )
        
        # Fallback to environment variables if project_id not explicitly provided
        if not project_id:
            env_project = (
                os.environ.get("GCP_PROJECT_ID")
                or os.environ.get("GOOGLE_CLOUD_PROJECT")
            )
            project_id = env_project
        
        if not project_id:
            raise ValueError(
                "GCP project_id is required for Vertex AI. "
                "Set it explicitly or via GCP_PROJECT_ID / GOOGLE_CLOUD_PROJECT env vars."
            )
        
        self.project_id = project_id
        self.region = region
        self.client: Optional[AnthropicVertex] = None
        
        logger.info(f"Initializing Vertex AI with project={project_id}, region={region}")
        
    def _get_client(self) -> AnthropicVertex:
        """Get or create Vertex AI client."""
        if self.client is None:
            try:
                # Initialize Google Cloud AI Platform
                if GOOGLE_CLOUD_AVAILABLE:
                    aiplatform.init(project=self.project_id, location=self.region)
                
                # Create Anthropic Vertex client
                self.client = AnthropicVertex(
                    project_id=self.project_id,
                    region=self.region
                )
                logger.info("✅ Vertex AI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Vertex AI client: {e}")
                raise
        
        return self.client
    
    def analyze_game(
        self,
        home_team: str,
        away_team: str,
        sport_key: str,
        commence_time: str,
        best_moneyline: Optional[float] = None,
        best_spread: Optional[float] = None,
        context_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single game using Vertex AI.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            sport_key: Sport identifier
            commence_time: Game start time
            best_moneyline: Best available moneyline odds
            best_spread: Best available spread
            context_data: Additional data (ML predictions, TheOver.ai, etc.)
        
        Returns:
            Dict with analysis results including probabilities and recommendations
        """
        context_data = context_data or {}
        
        # Build comprehensive prompt
        prompt = self._build_analysis_prompt(
            home_team=home_team,
            away_team=away_team,
            sport_key=sport_key,
            commence_time=commence_time,
            best_moneyline=best_moneyline,
            best_spread=best_spread,
            context_data=context_data
        )
        
        try:
            client = self._get_client()
            
            # Call Vertex AI with Claude
            logger.info(f"Calling Vertex AI for {away_team} @ {home_team}")
            
            message = client.messages.create(
                model="claude-sonnet-4-5@20250514",  # Claude 4.5 Sonnet (latest)
                max_tokens=2000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            response_text = message.content[0].text.strip()
            
            # Extract JSON from response
            response_text = self._extract_json(response_text)
            analysis = json.loads(response_text)
            
            logger.info(
                f"✅ Vertex AI analysis complete: "
                f"{away_team} @ {home_team} = {analysis.get('home_win_probability')}%"
            )
            
            return {
                'game_id': f"{away_team}_{home_team}_{sport_key}",
                'home_team': home_team,
                'away_team': away_team,
                'sport': sport_key,
                'commence_time': commence_time,
                'vertex_probability': analysis.get('home_win_probability', 50.0),
                'away_probability': 100.0 - analysis.get('home_win_probability', 50.0),
                'confidence': analysis.get('confidence', 50.0),
                'key_factors': analysis.get('key_factors', []),
                'has_edge': analysis.get('has_edge', False),
                'edge_explanation': analysis.get('edge_explanation', ''),
                'recommended_bet': analysis.get('recommended_bet', 'none'),
                'bet_type': analysis.get('bet_type', 'none'),
                'risk_level': analysis.get('risk_level', 'medium'),
                'best_moneyline': best_moneyline,
                'best_spread': best_spread,
                'sources_used': self._get_sources_used(context_data),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Vertex AI response: {e}")
            logger.error(f"Response text: {response_text[:500]}")
            return self._fallback_result(home_team, away_team, sport_key, commence_time, 
                                        best_moneyline, best_spread, f"JSON parse error: {e}")
        
        except Exception as e:
            logger.error(f"Vertex AI analysis failed for {away_team} @ {home_team}: {e}")
            return self._fallback_result(home_team, away_team, sport_key, commence_time,
                                        best_moneyline, best_spread, str(e))
    
    def analyze_games_batch(
        self,
        games: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple games using Vertex AI.
        
        Args:
            games: List of game dicts with required fields
            progress_callback: Optional function(current, total) for progress updates
        
        Returns:
            List of analysis results
        """
        results = []
        total = len(games)
        
        for idx, game in enumerate(games):
            if progress_callback:
                progress_callback(idx + 1, total)
            
            result = self.analyze_game(
                home_team=game.get('home_team'),
                away_team=game.get('away_team'),
                sport_key=game.get('sport_key'),
                commence_time=game.get('commence_time'),
                best_moneyline=game.get('best_moneyline'),
                best_spread=game.get('best_spread'),
                context_data=game.get('context_data', {})
            )
            
            results.append(result)
        
        return results
    
    def _build_analysis_prompt(
        self,
        home_team: str,
        away_team: str,
        sport_key: str,
        commence_time: str,
        best_moneyline: Optional[float],
        best_spread: Optional[float],
        context_data: Dict[str, Any]
    ) -> str:
        """Build comprehensive analysis prompt."""
        
        prompt = f"""You are an expert sports betting analyst. Analyze this game comprehensively and provide accurate probability assessments.

**Game Details:**
- Away Team: {away_team}
- Home Team: {home_team}
- Sport: {sport_key}
- Date: {commence_time}

**Market Odds:**
- Home Moneyline: {best_moneyline if best_moneyline else 'N/A'}
- Home Spread: {best_spread if best_spread else 'N/A'}
"""
        
        # Add ML predictions if available
        if context_data.get('ml'):
            ml = context_data['ml']
            prompt += f"""
**ML Model Prediction:**
- Home Win Probability: {ml.get('home_win_prob', 0):.1%}
- Away Win Probability: {ml.get('away_win_prob', 0):.1%}
- Confidence: {ml.get('confidence', 0):.1%}
- Model: {ml.get('model_used', 'Unknown')}
- Edge: {ml.get('edge', 0):.2%}
"""
        
        # Add TheOver.ai if available
        if context_data.get('theover'):
            theover = context_data['theover']
            prompt += f"""
**TheOver.ai Analysis:**
- Pick: {theover.get('Pick')}
- Line: {theover.get('Line')}
- Market: {theover.get('Market')}
"""
            if theover.get('Probability'):
                prompt += f"- Model Probability: {theover.get('Probability')}\n"
        
        # Add SportsData if available
        if context_data.get('sportsdata'):
            prompt += f"""
**SportsData:**
- Available for {context_data['sportsdata'].get('sport', 'this sport')}
"""
        
        # Add API-Sports if available
        if context_data.get('apisports'):
            prompt += f"""
**API-Sports Live Data:**
- Real-time stats available
"""
        
        prompt += """

**Your Task:**
Synthesize ALL available data sources and provide a comprehensive analysis.

Consider:
1. Historical matchups between these teams
2. Current form and momentum
3. Home field advantage
4. Market inefficiencies (odds vs reality)
5. ML model insights (if provided)
6. Expert analysis (if provided)

**Output Format:**
Return ONLY a valid JSON object with this EXACT structure:

{
    "home_win_probability": <float 0-100>,
    "confidence": <float 0-100>,
    "key_factors": ["<factor1>", "<factor2>", "<factor3>"],
    "has_edge": <true|false>,
    "edge_explanation": "<brief explanation>",
    "recommended_bet": "<home|away|none>",
    "bet_type": "<moneyline|spread|none>",
    "risk_level": "<low|medium|high>"
}

CRITICAL RULES:
- Output ONLY valid JSON, no additional text
- All probabilities must be 0-100
- Include exactly 3 key factors
- Be specific in edge_explanation
- Recommended bet should be "none" if no clear edge
"""
        
        return prompt
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from response text."""
        # Remove markdown code blocks if present
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        
        # Remove any leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _get_sources_used(self, context_data: Dict[str, Any]) -> str:
        """Get list of data sources used."""
        sources = []
        
        if context_data.get('ml'):
            sources.append('ML Model')
        if context_data.get('theover'):
            sources.append('TheOver.ai')
        if context_data.get('sportsdata'):
            sources.append('SportsData')
        if context_data.get('apisports'):
            sources.append('API-Sports')
        
        if not sources:
            return 'Market odds only'
        
        return ', '.join(sources)
    
    def _fallback_result(
        self,
        home_team: str,
        away_team: str,
        sport_key: str,
        commence_time: str,
        best_moneyline: Optional[float],
        best_spread: Optional[float],
        error_msg: str
    ) -> Dict[str, Any]:
        """Return fallback result when analysis fails."""
        return {
            'game_id': f"{away_team}_{home_team}_{sport_key}",
            'home_team': home_team,
            'away_team': away_team,
            'sport': sport_key,
            'commence_time': commence_time,
            'vertex_probability': 50.0,
            'away_probability': 50.0,
            'confidence': 0.0,
            'key_factors': ['Analysis failed', 'Using default values', 'See error log'],
            'has_edge': False,
            'edge_explanation': f'Error: {error_msg[:100]}',
            'recommended_bet': 'none',
            'bet_type': 'none',
            'risk_level': 'high',
            'best_moneyline': best_moneyline,
            'best_spread': best_spread,
            'sources_used': 'error',
            'analysis_timestamp': datetime.now().isoformat(),
            'error': error_msg
        }


def test_vertex_ai_connection(project_id: str, region: str = "us-east5") -> bool:
    """
    Test Vertex AI connection.
    
    Args:
        project_id: GCP project ID
        region: GCP region
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        analyzer = VertexAIAnalyzer(project_id=project_id, region=region)
        
        # Test with a simple game
        result = analyzer.analyze_game(
            home_team="Kansas City Chiefs",
            away_team="Las Vegas Raiders",
            sport_key="americanfootball_nfl",
            commence_time="2024-01-01T13:00:00Z",
            best_moneyline=-200,
            best_spread=-3.5
        )
        
        if result and result.get('vertex_probability'):
            logger.info("✅ Vertex AI connection test successful!")
            return True
        else:
            logger.error("❌ Vertex AI returned invalid result")
            return False
            
    except Exception as e:
        logger.error(f"❌ Vertex AI connection test failed: {e}")
        return False


# Streamlit UI helper functions
def get_vertex_ai_config():
    """
    Get Vertex AI configuration from Streamlit session state or environment.
    
    Returns:
        Dict with project_id and region, or None if not configured
    """
    import streamlit as st
    
    # Check session state first
    project_id = st.session_state.get('gcp_project_id')
    region = st.session_state.get('gcp_region', 'us-east5')
    
    # Fall back to environment variables
    if not project_id:
        project_id = (
            os.environ.get('GCP_PROJECT_ID')
            or os.environ.get('GOOGLE_CLOUD_PROJECT')
        )
    
    if not project_id:
        logger.error(
            "Vertex AI config missing: no project id in session_state['gcp_project_id'], "
            "GCP_PROJECT_ID, or GOOGLE_CLOUD_PROJECT."
        )
        return None
    
    return {
        'project_id': project_id,
        'region': region
    }


def show_vertex_ai_config_ui():
    """Show Vertex AI configuration UI in Streamlit sidebar."""
    import streamlit as st
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Google Vertex AI Config")
    
    # Check if dependencies installed
    if not VERTEX_AI_AVAILABLE:
        st.sidebar.error("❌ Vertex AI not installed")
        st.sidebar.code("pip install anthropic[vertex] google-cloud-aiplatform")
        return False
    
    # GCP Project ID
    project_id = st.sidebar.text_input(
        "GCP Project ID",
        value=st.session_state.get('gcp_project_id', ''),
        type="password",
        help="Your Google Cloud project ID",
        key="vertex_project_id_input"
    )
    
    if project_id != st.session_state.get('gcp_project_id', ''):
        st.session_state['gcp_project_id'] = project_id
    
    # GCP Region
    region = st.sidebar.selectbox(
        "GCP Region",
        options=["us-east5", "us-central1", "europe-west4"],
        index=0,
        help="Region for Vertex AI (us-east5 recommended for Claude 4.5 Sonnet)",
        key="vertex_region_select"
    )
    
    st.session_state['gcp_region'] = region
    
    # Service Account Key Upload
    st.sidebar.markdown("**Authentication:**")
    uploaded_key = st.sidebar.file_uploader(
        "Upload Service Account Key (JSON)",
        type=['json'],
        help="Download from GCP Console → IAM & Admin → Service Accounts",
        key="vertex_sa_key_upload"
    )
    
    if uploaded_key:
        try:
            # Save service account key
            key_path = "/tmp/gcp_service_account_key.json"
            with open(key_path, "wb") as f:
                f.write(uploaded_key.read())
            
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path
            st.sidebar.success("✅ Service account key loaded")
            
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load key: {e}")
    
    # Test connection button
    if project_id:
        if st.sidebar.button("🧪 Test Vertex AI Connection", key="test_vertex_btn"):
            with st.sidebar:
                with st.spinner("Testing connection..."):
                    success = test_vertex_ai_connection(project_id, region)
                    if success:
                        st.success("✅ Connection successful!")
                    else:
                        st.error("❌ Connection failed - check logs")
        
        st.sidebar.success("✅ Vertex AI Configured")
        return True
    else:
        st.sidebar.warning("⚠️ Enter GCP Project ID")
        return False


if __name__ == "__main__":
    # Test the module
    import sys
    
    if len(sys.argv) > 1:
        project_id = sys.argv[1]
        print(f"Testing Vertex AI with project: {project_id}")
        success = test_vertex_ai_connection(project_id)
        sys.exit(0 if success else 1)
    else:
        print("Usage: python vertex_ai_integration.py <gcp_project_id>")
        sys.exit(1)
