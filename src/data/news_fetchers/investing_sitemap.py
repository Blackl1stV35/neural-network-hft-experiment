import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Dict
from .base import BaseNewsFetcher

class InvestingSitemapFetcher(BaseNewsFetcher):
    """Investing.com news sitemap – excellent for live trading"""

    def __init__(self):
        self.sitemap_url = "https://www.investing.com/news_sitemap.xml"

    def fetch(self, hours_back: int = 24) -> List[Dict]:
        """Fetch recent news from Investing.com sitemap"""
        articles = []
        cutoff = datetime.utcnow() - timedelta(hours=hours_back)

        try:
            resp = requests.get(self.sitemap_url, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 (compatible; NeuralHFTBot/1.0)"
            })
            resp.raise_for_status()

            root = ET.fromstring(resp.content)

            for url in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                loc = url.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                news = url.find('{http://www.google.com/schemas/sitemap-news/0.9}news')

                if not loc or not news:
                    continue

                title_elem = news.find('{http://www.google.com/schemas/sitemap-news/0.9}title')
                pub_date_elem = news.find('{http://www.google.com/schemas/sitemap-news/0.9}publication_date')

                if not title_elem or not pub_date_elem:
                    continue

                pub_date = datetime.fromisoformat(pub_date_elem.text.replace('Z', '+00:00'))

                if pub_date < cutoff:
                    continue

                articles.append({
                    'title': title_elem.text,
                    'text': title_elem.text,          # full article text would require scraping
                    'published_at': pub_date,
                    'url': loc.text
                })

            return articles

        except Exception as e:
            print(f"[Investing Sitemap] Error: {e}")
            return []

    def is_live_mode(self) -> bool:
        return True