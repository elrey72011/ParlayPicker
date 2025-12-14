"""Core modules shared between the Streamlit app and auxiliary scripts."""

from .apisports import (
    APISportsBasketballClient,
    APISportsFootballClient,
    APISportsHockeyClient,
    GameSummary,
    TeamSummary,
)
from .sportsdata import (
    SportsDataGameInsight,
    SportsDataNCAABClient,
    SportsDataNCAAFClient,
    SportsDataNBAClient,
    SportsDataNFLClient,
    SportsDataNHLClient,
    SportsDataTeamInsight,
)
from .ml import HistoricalDataBuilder, HistoricalMLPredictor, MLPredictor
from .kalshi_integrator import KalshiIntegrator
from .sentiment import RealSentimentAnalyzer, SentimentAnalyzer
from .odds_api import TheOddsAPIClient

__all__ = [
    "APISportsBasketballClient",
    "APISportsFootballClient",
    "APISportsHockeyClient",
    "GameSummary",
    "TeamSummary",
    "SportsDataNFLClient",
    "SportsDataNBAClient",
    "SportsDataNHLClient",
    "SportsDataNCAAFClient",
    "SportsDataNCAABClient",
    "SportsDataGameInsight",
    "SportsDataTeamInsight",
    "KalshiIntegrator",
    "TheOddsAPIClient",
    "RealSentimentAnalyzer",
    "SentimentAnalyzer",
]
