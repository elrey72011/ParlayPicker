# gemini_integration.py
"""
Google Gemini Integration for ParlayDesk
Uses native Google Gemini models on Vertex AI.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import streamlit as st

logger = logging.getLogger(__name__)

# Try to import Google Generative AI dependencies
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Generative AI not available. Install with: pip install google-genai")


def _normalize_flags(flags: Any) -> List[str]:
    """Normalize risk flags into a short list of strings."""
    if isinstance(flags, list):
        return [str(f) for f in flags if f][:8]
    if isinstance(flags, str):
        return [flags[:120]] if flags else []
    return []


class GeminiAnalyzer:
    """
    Analyzes sports betting games using Google Gemini via google-genai SDK.
    
    Cost-effective with strong quality.
    """
    
    def __init__(self, project_id: str, region: str = "us-central1"):
        """
        Initialize Gemini Analyzer.
        
        Args:
            project_id: Your Google Cloud project ID (for Vertex AI compatibility via genai)
            region: GCP region (us-central1 recommended for Gemini)
        
        Raises:
            ValueError: If Gemini dependencies not available
        """
        if not GEMINI_AVAILABLE:
            raise ValueError(
                "Gemini dependencies not installed. "
                "Install with: pip install google-genai"
            )

        self.project_id = project_id
        self.region = region
        self.client = None
        
        logger.info(f"Initializing Gemini Client")
        
    def _get_client(self):
        """Get or create Gemini client."""
        if self.client is None:
            import streamlit as st

            # Check for direct API key first (most secure, bypasses Vertex)
            api_key = os.environ.get("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))

            if not api_key and 'gemini_api_key' in st.session_state and st.session_state['gemini_api_key']:
                api_key = st.session_state['gemini_api_key']

            try:
                if api_key:
                    # Direct Gemini Developer API
                    self.client = genai.Client(api_key=api_key)
                    logger.info("✅ Gemini client initialized successfully (using API Key)")
                else:
                    # Ensure credentials are set (restore from session state if needed)
                    self._ensure_credentials()

                    # Create GenAI client
                    # If project_id and location are provided, google-genai can route through Vertex
                    # depending on the environment and credentials
                    if self.project_id:
                        self.client = genai.Client(vertexai=True, project=self.project_id, location=self.region)
                    else:
                        self.client = genai.Client()

                    logger.info("✅ Gemini client initialized successfully (using Vertex/Service Account)")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                raise
        
        return self.client
    
    def _ensure_credentials(self):
        """Ensure Google credentials are available."""
        import streamlit as st
        
        # Check if credentials file exists
        if os.path.exists("/tmp/gcp_service_account_key.json"):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/tmp/gcp_service_account_key.json"
            return
        
        # Try to restore from session state
        if 'gcp_service_account_key' in st.session_state:
            try:
                key_path = "/tmp/gcp_service_account_key.json"
                with open(key_path, "wb") as f:
                    f.write(st.session_state['gcp_service_account_key'])
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path
                logger.info("✅ Credentials restored from session state")
            except Exception as e:
                logger.error(f"Failed to restore credentials: {e}")
                raise ValueError("Google credentials not available. Please upload service account key in sidebar.")
        else:
            raise ValueError("Google credentials not found. Please provide GEMINI_API_KEY or upload service account key in sidebar.")
    
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
        Analyze a single game using Gemini.
        
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
        
        prompt = (
            "You are a reviewer. The pick is already chosen elsewhere. "
            "Provide ONLY confidence_explanation and risk_notes in short text. "
            "Do NOT output probabilities, percentages, or pick a side. "
            f"Context: home={home_team}, away={away_team}, sport={sport_key}, commence={commence_time}. "
            f"Moneyline={best_moneyline}, spread={best_spread}, extra={json.dumps(context_data or {}, default=str)}. "
            'Return JSON: {"confidence_explanation": "...", "risk_notes": "..."}'
        )
        
        try:
            client = self._get_client()
            
            # Configure generation
            config = types.GenerateContentConfig(
                temperature=0.2,
                top_p=0.8,
                max_output_tokens=2048,
            )
            
            # Call Gemini
            logger.info(f"Calling Gemini for {away_team} @ {home_team}")
            
            response = client.models.generate_content(
                model='gemini-2.0-flash-001',
                contents=prompt,
                config=config
            )
            
            # Parse response
            response_text = response.text.strip()
            
            # Extract JSON from response
            response_text = self._extract_json(response_text)
            analysis = json.loads(response_text)
            
            return {
                'game_id': f"{away_team}_{home_team}_{sport_key}",
                'home_team': home_team,
                'away_team': away_team,
                'sport': sport_key,
                'commence_time': commence_time,
                'gemini_probability': None,
                'away_probability': None,
                'confidence_explanation': analysis.get('confidence_explanation'),
                'risk_notes': analysis.get('risk_notes'),
                'key_factors': [],
                'has_edge': False,
                'edge_explanation': '',
                'recommended_bet': analysis.get('recommended_bet') or 'none',
                'confidence': (str(analysis.get('confidence')).upper() if analysis.get('confidence') else 'MEDIUM'),
                'flags': _normalize_flags(analysis.get('risk_notes')),
                'bet_type': 'none',
                'risk_level': 'informational',
                'best_moneyline': best_moneyline,
                'best_spread': best_spread,
                'sources_used': self._get_sources_used(context_data),
                'analysis_timestamp': datetime.now().isoformat(),
                'model': 'gemini-2.0-flash-001',
                'explanation': analysis.get('confidence_explanation') or analysis.get('edge_explanation') or '',
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            logger.error(f"Response text: {response_text[:500]}")
            return self._fallback_result(home_team, away_team, sport_key, commence_time, 
                                        best_moneyline, best_spread, f"JSON parse error: {e}")
        
        except Exception as e:
            logger.error(f"Gemini analysis failed for {away_team} @ {home_team}: {e}")
            return self._fallback_result(home_team, away_team, sport_key, commence_time,
                                        best_moneyline, best_spread, str(e))
    
    def analyze_games_batch(
        self,
        games: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple games using Gemini.
        
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
- Output ONLY valid JSON, no additional text before or after
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
            'gemini_probability': None,
            'away_probability': None,
            'confidence_explanation': "Analysis failed; no probabilities returned.",
            'risk_notes': error_msg[:120],
            'key_factors': ['Analysis failed', 'See error log'],
            'has_edge': False,
            'edge_explanation': f'Error: {error_msg[:100]}',
            'recommended_bet': 'none',
            'bet_type': 'none',
            'risk_level': 'high',
            'best_moneyline': best_moneyline,
            'best_spread': best_spread,
            'sources_used': 'error',
            'analysis_timestamp': datetime.now().isoformat(),
            'model': 'gemini-2.0-flash-001',
            'error': error_msg
        }


def test_gemini_connection(project_id: str, region: str = "us-central1") -> bool:
    """
    Test Gemini connection.
    
    Args:
        project_id: GCP project ID
        region: GCP region
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        analyzer = GeminiAnalyzer(project_id=project_id, region=region)
        
        # Test with a simple game
        result = analyzer.analyze_game(
            home_team="Kansas City Chiefs",
            away_team="Las Vegas Raiders",
            sport_key="americanfootball_nfl",
            commence_time="2024-01-01T13:00:00Z",
            best_moneyline=-200,
            best_spread=-3.5
        )
        
        if result and result.get('gemini_probability'):
            logger.info("✅ Gemini connection test successful!")
            return True
        else:
            logger.error("❌ Gemini returned invalid result")
            return False
            
    except Exception as e:
        logger.error(f"❌ Gemini connection test failed: {e}")
        return False


@st.cache_data(ttl=600)
def summarize_gemini_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map a Gemini analysis result into the lightweight metadata expected by the app.
    """
    recommended = result.get("recommended_bet") or "none"
    explanation = result.get("explanation") or result.get("confidence_explanation") or ""
    confidence = str(result.get("confidence") or "").upper()
    if confidence not in {"HIGH", "MEDIUM", "LOW"}:
        confidence = "MEDIUM"
    flags = _normalize_flags(result.get("flags") or result.get("risk_notes") or [])
    return {
        "recommended_bet": str(recommended),
        "confidence": confidence,
        "explanation": str(explanation)[:240],
        "flags": flags,
    }


# Streamlit UI helper functions
def show_gemini_config_ui():
    """Show Gemini configuration UI in Streamlit sidebar."""
    import streamlit as st
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Google Gemini Config")
    st.sidebar.caption("💰 Low-cost Gemini analysis")
    
    # Check if dependencies installed
    if not GEMINI_AVAILABLE:
        st.sidebar.error("❌ Gemini SDK not installed")
        st.sidebar.code("pip install google-genai")
        return False
    
    # API Key Input (Preferred/Lightweight)
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        value=st.session_state.get('gemini_api_key', st.secrets.get("GEMINI_API_KEY", "")),
        type="password",
        help="Your Google Gemini API Key. Can also be set in st.secrets.",
        key="gemini_api_key_input"
    )
    
    if api_key != st.session_state.get('gemini_api_key', ''):
        st.session_state['gemini_api_key'] = api_key

    # Advanced options toggle for Vertex AI
    with st.sidebar.expander("Advanced: Vertex AI via Service Account"):
        # GCP Project ID
        project_id = st.text_input(
            "GCP Project ID",
            value=st.session_state.get('gcp_project_id', ''),
            type="password",
            help="Your Google Cloud project ID",
            key="gemini_project_id_input"
        )

        if project_id != st.session_state.get('gcp_project_id', ''):
            st.session_state['gcp_project_id'] = project_id

        # GCP Region
        region = st.selectbox(
            "GCP Region",
            options=["us-central1", "us-east4", "europe-west4"],
            index=0,
            help="Region for Gemini (us-central1 recommended)",
            key="gemini_region_select"
        )

        st.session_state['gcp_region'] = region

        # Service Account Key Upload
        st.markdown("**Authentication:**")
        uploaded_key = st.file_uploader(
            "Upload Service Account Key (JSON)",
            type=['json'],
            help="Download from GCP Console → IAM & Admin → Service Accounts",
            key="gemini_sa_key_upload"
        )

        # Store key in session state for persistence across reruns
        if uploaded_key:
            try:
                # Read the key content
                key_content = uploaded_key.read()

                # Store in session state so it persists
                st.session_state['gcp_service_account_key'] = key_content

                # Save to file
                key_path = "/tmp/gcp_service_account_key.json"
                with open(key_path, "wb") as f:
                    f.write(key_content)

                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path
                st.success("✅ Service account key loaded")

            except Exception as e:
                st.error(f"❌ Failed to load key: {e}")

        # If key is in session state but not uploaded this time, restore it
        elif 'gcp_service_account_key' in st.session_state:
            try:
                key_path = "/tmp/gcp_service_account_key.json"
                with open(key_path, "wb") as f:
                    f.write(st.session_state['gcp_service_account_key'])
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path
                st.info("🔑 Using saved credentials")
            except Exception as e:
                st.warning(f"⚠️ Could not restore credentials: {e}")
    
    # Test connection button
    can_test = bool(api_key or project_id)
    if can_test:
        if st.sidebar.button("🧪 Test Gemini Connection", key="test_gemini_btn"):
            with st.sidebar:
                with st.spinner("Testing connection..."):
                    success = test_gemini_connection(project_id, region) if not api_key else test_gemini_connection("")
                    if success:
                        st.success("✅ Connection successful!")
                        st.info(f"💰 Estimated cost: ~$0.001/game")
                    else:
                        st.error("❌ Connection failed - check logs")
        
        # Show status based on whether we have credentials
        has_vertex_creds = 'gcp_service_account_key' in st.session_state or uploaded_key
        if api_key or has_vertex_creds:
            st.sidebar.success("✅ Gemini Configured")
            st.sidebar.caption("💡 Gemini keeps analysis affordable")
            return True
        else:
            st.sidebar.warning("⚠️ Upload service account key or provide API Key")
            return False
    else:
        st.sidebar.warning("⚠️ Enter API Key or GCP Project ID")
        return False


if __name__ == "__main__":
    # Test the module
    import sys
    
    if len(sys.argv) > 1:
        project_id = sys.argv[1]
        print(f"Testing Gemini with project: {project_id}")
        success = test_gemini_connection(project_id)
        sys.exit(0 if success else 1)
    else:
        print("Usage: python gemini_integration.py <gcp_project_id>")
        sys.exit(1)
