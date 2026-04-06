from abc import ABC, abstractmethod
from typing import List, Dict
from datetime import datetime

class BaseNewsFetcher(ABC):
    """Abstract base class for any news source (FT sitemap, Investing.com, NewsAPI, etc.)"""

    @abstractmethod
    def fetch(self, hours_back: int = 168) -> List[Dict]:
        """
        Return list of articles in this format:
        {
            'title': str,
            'text': str,           # full text or title if body not available
            'published_at': datetime,
            'url': str
        }
        """
        pass

    @abstractmethod
    def is_live_mode(self) -> bool:
        """True for live sources (Investing.com), False for historical (FT)"""
        pass