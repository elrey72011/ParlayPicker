# Real Sentiment Analysis Implementation
# This module provides ACTUAL sentiment analysis using news APIs and NLP

import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import re
from collections import Counter

class RealSentimentAnalyzer:
    """
    Real sentiment analysis using actual news sources and NLP
    
    Data Sources:
    - NewsAPI.org (free tier: 100 requests/day)
    - Web search results for team news
    - Basic NLP sentiment classification
    """
    
    def __init__(self, news_api_key: Optional[str] = None):
        """
        Initialize with API keys
        
        Args:
            news_api_key: NewsAPI.org key (get free at https://newsapi.org/)
        """
        self.news_api_key = news_api_key or os.environ.get("NEWS_API_KEY")
        self.sentiment_cache = {}
        self.cache_duration = 1800  # 30 minutes
        
        # Simple sentiment word lists (can be expanded)
        self.positive_words = {
            'win', 'wins', 'won', 'winning', 'victory', 'beat', 'beats', 
            'dominant', 'strong', 'excellent', 'best', 'great', 'hot', 
            'streak', 'momentum', 'comeback', 'champion', 'star', 'explosive',
            'impressive', 'outstanding', 'stellar', 'clutch', 'elite',
            'record-breaking', 'unstoppable', 'phenomenal', 'surging', 'rolling'
        }
        
        self.negative_words = {
            'lose', 'loses', 'lost', 'losing', 'defeat', 'beaten',
            'weak', 'poor', 'worst', 'bad', 'cold', 'slump', 'struggle',
            'injury', 'injured', 'hurt', 'out', 'questionable', 'doubtful',
            'blow', 'collapse', 'disaster', 'awful', 'terrible', 'embarrassing',
            'turnover', 'frustrated', 'disappointing', 'concerning', 'worry'
        }
        
    def get_team_sentiment(self, team_name: str, sport: str) -> Dict[str, float]:
        """
        Get real sentiment analysis for a team
        
        Returns:
            {
                'score': float (-1 to +1),
                'confidence': float (0 to 1),
                'sources': int,
                'trend': str ('positive', 'negative', 'neutral'),
                'articles_analyzed': int,
                'method': str (which analysis method was used)
            }
        """
        cache_key = f"{team_name}_{sport}_{datetime.now().date()}"
        
        # Check cache
        if cache_key in self.sentiment_cache:
            cached = self.sentiment_cache[cache_key]
            age = (datetime.now() - cached['timestamp']).seconds
            if age < self.cache_duration:
                return cached['data']
        
        # Try NewsAPI first (best source)
        if self.news_api_key:
            result = self._analyze_with_newsapi(team_name, sport)
            if result['sources'] > 0:
                self.sentiment_cache[cache_key] = {
                    'data': result,
                    'timestamp': datetime.now()
                }
                return result
        
        # Fallback to web search sentiment
        result = self._analyze_with_web_search(team_name, sport)
        
        self.sentiment_cache[cache_key] = {
            'data': result,
            'timestamp': datetime.now()
        }
        
        return result
    
    def _analyze_with_newsapi(self, team_name: str, sport: str) -> Dict:
        """
        Analyze sentiment using NewsAPI.org
        """
        try:
            # Calculate date range (last 3 days)
            to_date = datetime.now()
            from_date = to_date - timedelta(days=3)
            
            # Build search query
            query = f'"{team_name}"'
            if 'nba' in sport.lower():
                query += ' NBA basketball'
            elif 'nfl' in sport.lower():
                query += ' NFL football'
            elif 'mlb' in sport.lower():
                query += ' MLB baseball'
            elif 'nhl' in sport.lower():
                query += ' NHL hockey'
            
            # Call NewsAPI
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'relevancy',
                'pageSize': 20,
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                return self._fallback_result("NewsAPI error")
            
            data = response.json()
            articles = data.get('articles', [])
            
            if not articles:
                return self._fallback_result("No articles found")
            
            # Analyze sentiment from articles
            sentiment_scores = []
            
            for article in articles[:20]:  # Analyze up to 20 articles
                title = article.get('title', '').lower()
                description = article.get('description', '').lower()
                text = f"{title} {description}"
                
                score = self._calculate_text_sentiment(text)
                sentiment_scores.append(score)
            
            # Calculate aggregate sentiment
            if sentiment_scores:
                avg_score = sum(sentiment_scores) / len(sentiment_scores)
                
                # Calculate confidence based on consistency
                score_variance = sum((s - avg_score) ** 2 for s in sentiment_scores) / len(sentiment_scores)
                confidence = max(0.3, 1.0 - score_variance)  # Lower variance = higher confidence
                
                trend = 'positive' if avg_score > 0.15 else ('negative' if avg_score < -0.15 else 'neutral')
                
                return {
                    'score': avg_score,
                    'confidence': min(confidence, 0.95),
                    'sources': len(sentiment_scores),
                    'trend': trend,
                    'articles_analyzed': len(articles),
                    'method': 'NewsAPI + NLP'
                }
            
            return self._fallback_result("No sentiment data")
            
        except requests.exceptions.Timeout:
            return self._fallback_result("NewsAPI timeout")
        except Exception as e:
            return self._fallback_result(f"Error: {str(e)}")
    
    def _analyze_with_web_search(self, team_name: str, sport: str) -> Dict:
        """
        Fallback: Use web search results for sentiment
        This is called when NewsAPI is not available
        """
        # This would use the web_search tool if available in Streamlit
        # For now, return neutral with low confidence
        return {
            'score': 0.0,
            'confidence': 0.3,
            'sources': 0,
            'trend': 'neutral',
            'articles_analyzed': 0,
            'method': 'No API key - using neutral'
        }
    
    def _calculate_text_sentiment(self, text: str) -> float:
        """
        Calculate sentiment score for text using word matching
        
        Returns:
            float: Sentiment score from -1.0 (very negative) to +1.0 (very positive)
        """
        # Tokenize and clean text
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Count positive and negative words
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return 0.0  # Neutral
        
        # Calculate sentiment score
        sentiment_score = (positive_count - negative_count) / total_sentiment_words
        
        # Apply dampening to avoid extreme scores
        sentiment_score = sentiment_score * 0.7
        
        return max(-1.0, min(1.0, sentiment_score))
    
    def _fallback_result(self, reason: str) -> Dict:
        """Return neutral sentiment when analysis fails"""
        return {
            'score': 0.0,
            'confidence': 0.2,
            'sources': 0,
            'trend': 'neutral',
            'articles_analyzed': 0,
            'method': f'Fallback ({reason})'
        }
    
    def get_api_status(self) -> Dict:
        """Check if APIs are configured and working"""
        status = {
            'newsapi_configured': bool(self.news_api_key),
            'newsapi_working': False
        }
        
        if self.news_api_key:
            try:
                # Test API with simple query
                response = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={'q': 'test', 'apiKey': self.news_api_key},
                    timeout=5
                )
                status['newsapi_working'] = response.status_code == 200
            except:
                pass
        
        return status


# Example usage:
if __name__ == "__main__":
    # Get your free API key at https://newsapi.org/
    analyzer = RealSentimentAnalyzer(news_api_key="YOUR_API_KEY_HERE")
    
    # Analyze a team
    sentiment = analyzer.get_team_sentiment("Lakers", "nba")
    
    print(f"Sentiment Score: {sentiment['score']:.2f}")
    print(f"Confidence: {sentiment['confidence']:.1%}")
    print(f"Trend: {sentiment['trend']}")
    print(f"Articles Analyzed: {sentiment['articles_analyzed']}")
    print(f"Method: {sentiment['method']}")
