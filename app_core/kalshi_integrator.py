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
        
        # Kalshi API URLs - use elections subdomain (verified working)
        self.base_url = "https://api.elections.kalshi.com/trade-api/v2"
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
        
        logger.info(f"Kalshi API request: {method} {url} with params: {params}")
        
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
            logger.info(f"Using API URL: {self.api_url}")
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, params=params, timeout=15)
            else:
                response = requests.post(url, headers=headers, json=params, timeout=15)
            
            logger.info(f"Kalshi API response: Status {response.status_code}, URL: {response.url}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    logger.info(f"Response data keys: {list(data.keys())}")
                    if 'markets' in data:
                        logger.info(f"Markets in response: {len(data.get('markets', []))}")
                    self.last_error = None
                    return data
                except Exception as json_error:
                    logger.error(f"Failed to parse JSON response: {json_error}")
                    logger.error(f"Response text: {response.text[:500]}")
                    self.last_error = f"JSON parse error: {json_error}"
                    return None
            elif response.status_code == 401:
                logger.warning(f"Kalshi API authentication failed - Response: {response.text[:200]}")
                self.last_error = "Authentication failed"
            elif response.status_code == 403:
                logger.warning(f"Kalshi API access forbidden - Response: {response.text[:200]}")
                self.last_error = "Access forbidden"
            else:
                logger.warning(f"Kalshi API error: {response.status_code} - {response.text[:200]}")
                self.last_error = f"API error: {response.status_code}"
                
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Kalshi API connection error (network may be blocked): {e}")
            self.last_error = f"Connection blocked - check network settings"
        except requests.exceptions.Timeout:
            logger.warning("Kalshi API timeout")
            self.last_error = "Request timeout"
        except Exception as e:
            logger.error(f"Kalshi API request failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.last_error = str(e)
        
        return None
    
    def is_configured(self) -> bool:
        """Check if Kalshi is properly configured"""
        return bool(self.api_key and self._auth_ready)

    def get_sports_series(self) -> List[Dict]:
        """Get all available sports series tickers from Kalshi
        
        Returns list of series with tickers like:
        - KXNFL (NFL markets)
        - KXNBA (NBA markets)  
        - KXMLB (MLB markets)
        etc.
        """
        try:
            endpoint = "/series"
            params = {"limit": 200}
            
            logger.info("Fetching Kalshi series list...")
            response_data = self._make_authenticated_request("GET", endpoint, params=params)
            
            if response_data:
                all_series = response_data.get("series", [])
                logger.info(f"Found {len(all_series)} total series")
                
                # Filter for sports-related series
                sports_keywords = ['NFL', 'NBA', 'MLB', 'NHL', 'UFC', 'SOCCER', 'TENNIS', 
                                  'GOLF', 'FOOTBALL', 'BASKETBALL', 'BASEBALL', 'HOCKEY',
                                  'SPORT', 'GAME']
                
                sports_series = []
                for series in all_series:
                    ticker = series.get('ticker', '').upper()
                    title = series.get('title', '').upper()
                    category = series.get('category', '').upper()
                    
                    if any(keyword in ticker or keyword in title or keyword in category 
                           for keyword in sports_keywords):
                        sports_series.append(series)
                        logger.info(f"Found sports series: {series.get('ticker')} - {series.get('title')}")
                
                return sports_series
            
            return []
            
        except Exception as e:
            logger.error(f"Error fetching sports series: {e}")
            return []
    
    def get_markets(self, category: str = "sports", status: str = "open") -> List[Dict]:
        """Fetch available Kalshi markets.

        Args:
            category: 'sports', 'politics', 'economics', etc. (used to filter series)
            status: 'open', 'closed', 'settled'

        Returns:
            List of market dictionaries.
        """
        if self._using_synthetic_data:
            return []

        try:
            # First, get sports series tickers
            sports_series = self.get_sports_series()
            
            if not sports_series:
                logger.warning("No sports series found")
                self.last_error = "No sports series available"
                self._using_synthetic_data = True
                return []
            
            logger.info(f"Found {len(sports_series)} sports series")
            
            # Collect markets from all sports series
            all_markets = []
            
            for series in sports_series[:10]:  # Limit to first 10 series to avoid rate limits
                series_ticker = series.get('ticker')
                if not series_ticker:
                    continue
                
                logger.info(f"Fetching markets for series: {series_ticker}")
                
                endpoint = "/markets"
                params = {
                    "series_ticker": series_ticker,  # CRITICAL: Use series_ticker parameter
                    "limit": 200,
                    "status": status
                }
                
                response_data = self._make_authenticated_request("GET", endpoint, params=params)
                
                if response_data:
                    markets = response_data.get("markets", [])
                    logger.info(f"Got {len(markets)} markets from {series_ticker}")
                    all_markets.extend(markets)
                    
                    if len(all_markets) >= 100:
                        # We have enough markets, stop querying
                        break
            
            if all_markets:
                self.last_error = None
                logger.info(f"✅ Loaded {len(all_markets)} total Kalshi sports markets")
                
                # Log sample tickers
                sample_tickers = [m.get('ticker', 'NO_TICKER') for m in all_markets[:5]]
                logger.info(f"Sample tickers: {sample_tickers}")
                
                return all_markets
            else:
                self.last_error = "Kalshi API returned no markets"
                logger.warning("No markets found in any sports series")

        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error fetching Kalshi markets: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

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
