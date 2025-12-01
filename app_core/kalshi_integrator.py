"""
Kalshi Integrator with Proper RSA-PSS Authentication
This file goes in: app_core/kalshi_integrator.py or app_core/__init__.py
"""

import os
import copy
import time
import logging
import requests
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class KalshiIntegrator:
    """Integrates Kalshi prediction market odds and analysis
    
    Kalshi uses RSA signature authentication for API requests.
    The API key is a UUID and the secret is an RSA private key.
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or os.environ.get("KALSHI_API_KEY")
        self.api_secret = api_secret or os.environ.get("KALSHI_API_SECRET")
        
        # Kalshi API URLs - try production first
        self.base_url = "https://api.kalshi.com/trade-api/v2"
        self.demo_url = "https://demo-api.kalshi.co/trade-api/v2"
        
        # Use production API if we have credentials
        self.api_url = self.base_url if self.api_key else self.demo_url
        
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        # RSA key for signing (parsed from api_secret)
        self._private_key = None
        self._auth_ready = False
        
        if self.api_key and self.api_secret:
            try:
                from cryptography.hazmat.primitives import serialization
                from cryptography.hazmat.backends import default_backend
                
                # Clean up the key if needed
                key_data = self.api_secret.strip()
                if not key_data.startswith('-----BEGIN'):
                    key_data = f"-----BEGIN RSA PRIVATE KEY-----\n{key_data}\n-----END RSA PRIVATE KEY-----"
                
                self._private_key = serialization.load_pem_private_key(
                    key_data.encode(),
                    password=None,
                    backend=default_backend()
                )
                self._auth_ready = True
                logger.info(f"✅ Kalshi RSA key loaded successfully (key: {self.api_key[:8]}...)")
            except ImportError:
                logger.warning("cryptography library not installed - Kalshi auth disabled")
                self._private_key = None
            except Exception as e:
                logger.warning(f"Could not load Kalshi RSA key: {e}")
                self._private_key = None

        # Synthetic fallback cache when Kalshi API is unavailable
        self._using_synthetic_data = False
        self._synthetic_markets: List[Dict[str, Any]] = []
        self._synthetic_orderbooks: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self._synthetic_market_by_team: Dict[str, Dict[str, Any]] = {}
        self.last_error: Optional[str] = None
        
        # Cache for API responses
        self._markets_cache = None
        self._cache_time = None
        self._cache_duration = 300  # 5 minutes
        
    def _sign_request(self, method: str, path: str, timestamp: str) -> str:
        """Create RSA signature for Kalshi API request"""
        if not self._private_key:
            return ""
        
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            import base64
            
            # Message format: timestamp + method + path
            message = f"{timestamp}{method}{path}"
            
            # Kalshi requires PSS padding, not PKCS1v15
            signature = self._private_key.sign(
                message.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH
                ),
                hashes.SHA256()
            )
            
            return base64.b64encode(signature).decode('utf-8')
        except Exception as e:
            logger.warning(f"Error signing Kalshi request: {e}")
            return ""
    
    def _make_authenticated_request(self, method: str, endpoint: str, params: dict = None) -> Optional[dict]:
        """Make authenticated request to Kalshi API
        
        Important: Per Kalshi docs, we sign ONLY the path (without query params),
        but send the full URL with query params in the actual request.
        """
        import time as time_module
        
        url = f"{self.api_url}{endpoint}"
        timestamp = str(int(time_module.time() * 1000))
        
        headers = self.headers.copy()
        
        if self._auth_ready and self._private_key:
            # CRITICAL: Strip query parameters from endpoint before signing
            # Per Kalshi docs: "Strip query parameters from path before signing"
            path_without_query = endpoint.split('?')[0]
            
            signature = self._sign_request(method.upper(), path_without_query, timestamp)
            headers["KALSHI-ACCESS-KEY"] = self.api_key
            headers["KALSHI-ACCESS-SIGNATURE"] = signature
            headers["KALSHI-ACCESS-TIMESTAMP"] = timestamp
            
            logger.debug(f"Signing: {timestamp}{method.upper()}{path_without_query}")
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, params=params, timeout=15)
            else:
                response = requests.post(url, headers=headers, json=params, timeout=15)
            
            logger.debug(f"Kalshi API response: {response.status_code}")
            
            if response.status_code == 200:
                self.last_error = None
                return response.json()
            elif response.status_code == 401:
                logger.warning(f"Kalshi API authentication failed - check API key and secret. Response: {response.text[:200]}")
                self.last_error = "Authentication failed"
            elif response.status_code == 403:
                logger.warning(f"Kalshi API access forbidden. Response: {response.text[:200]}")
                self.last_error = "Access forbidden"
            else:
                logger.warning(f"Kalshi API error: {response.status_code} - {response.text[:200]}")
                self.last_error = f"API error: {response.status_code}"
                
        except requests.exceptions.Timeout:
            logger.warning("Kalshi API timeout")
            self.last_error = "Request timeout"
        except Exception as e:
            logger.warning(f"Kalshi API request failed: {e}")
            self.last_error = str(e)
        
        return None
    
    def is_configured(self) -> bool:
        """Check if Kalshi is properly configured"""
        return bool(self.api_key and self._auth_ready)

    def get_markets(self, category: str = "sports", status: str = "open") -> List[Dict]:
        """Fetch available Kalshi markets.

        Args:
            category: 'sports', 'politics', 'economics', etc.
            status: 'open', 'closed', 'settled'

        Returns:
            List of market dictionaries.
        """
        if self._using_synthetic_data:
            return []

        try:
            endpoint = "/markets"
            params = {
                "limit": 200,  # Increased limit
                "status": status
            }

            # Try without any category filter first
            logger.info(f"Fetching Kalshi markets with params: {params}")
            
            # Use authenticated request method with RSA signature
            response_data = self._make_authenticated_request("GET", endpoint, params=params)

            if response_data:
                markets = response_data.get("markets", [])
                cursor = response_data.get("cursor")
                
                logger.info(f"Kalshi API returned {len(markets)} markets, cursor: {cursor}")
                
                if markets:
                    self.last_error = None
                    logger.info(f"✅ Loaded {len(markets)} Kalshi markets")
                    
                    # Log first few market tickers for debugging
                    sample_tickers = [m.get('ticker', 'NO_TICKER') for m in markets[:5]]
                    logger.info(f"Sample tickers: {sample_tickers}")
                    
                    return markets
                else:
                    self.last_error = "Kalshi API returned no markets"
                    logger.warning(f"Kalshi API returned empty markets list. Response keys: {list(response_data.keys())}")
                    logger.warning(f"Full response: {response_data}")
            else:
                logger.warning(f"Kalshi API failed: {self.last_error}")

        except Exception as e:
            self.last_error = str(e)
            logger.warning(f"Error fetching Kalshi markets: {str(e)}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")

        # Return empty list if API fails (don't fallback to synthetic)
        self._using_synthetic_data = True
        return []
    
    def get_sports_markets(self) -> List[Dict]:
        """Get all active sports betting markets"""
        all_markets = self.get_markets()
        
        # Filter for sports-related markets
        sports_keywords = ['NFL', 'NBA', 'MLB', 'NHL', 'UFC', 'SOCCER', 'TENNIS', 
                          'GOLF', 'FOOTBALL', 'BASKETBALL', 'BASEBALL', 'HOCKEY']
        
        sports_markets = []
        for market in all_markets:
            title = market.get('title', '').upper()
            ticker = market.get('ticker', '').upper()
            
            if any(keyword in title or keyword in ticker for keyword in sports_keywords):
                sports_markets.append(market)
        
        return sports_markets
