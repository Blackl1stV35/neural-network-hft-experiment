# src/data/news_fetchers/ft_sitemap.py
import requests
import xml.etree.ElementTree as ET
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict
from bs4 import BeautifulSoup
from .base import BaseNewsFetcher


class FTSitemapFetcher(BaseNewsFetcher):
    """Robust FT sitemap fetcher – handles index + monthly archives + real headline scraping"""

    def __init__(self):
        self.index_url = "https://www.ft.com/sitemaps/index.xml"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })

    def _scrape_headline(self, article_url: str) -> str:
        """Extract real headline using multiple reliable methods"""
        try:
            resp = self.session.get(article_url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            # Method 1: <title> tag (most reliable)
            title_tag = soup.find("title")
            if title_tag and title_tag.text.strip():
                title = title_tag.text.strip()
                if " | Financial Times" in title:
                    title = title.split(" | Financial Times")[0].strip()
                return title

            # Method 2: schema.org JSON-LD (exactly what you showed in the HTML)
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict) and data.get("@type") == "NewsArticle":
                        headline = data.get("headline") or data.get("alternativeHeadline")
                        if headline:
                            return headline
                except:
                    continue

            # Fallback
            return article_url.split("/")[-1].replace("-", " ").title()

        except Exception as e:
            print(f"    [Scrape failed for {article_url}] {e}")
            return article_url.split("/")[-1].replace("-", " ").title()

    def fetch(self, hours_back: int = 365*24) -> List[Dict]:
        articles = []
        cutoff = datetime.utcnow() - timedelta(hours=hours_back)

        try:
            print("📡 1. Fetching FT sitemap index...")
            resp = self.session.get(self.index_url, timeout=20)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)

            sitemap_count = 0
            article_count = 0

            for sitemap in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
                loc_elem = sitemap.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                lastmod_elem = sitemap.find('{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod')

                if not loc_elem:
                    continue

                sitemap_url = loc_elem.text
                lastmod_str = lastmod_elem.text if lastmod_elem is not None else ""

                # More forgiving date parsing
                try:
                    lastmod = datetime.fromisoformat(lastmod_str.replace("Z", "+00:00"))
                except:
                    lastmod = datetime.utcnow()

                if lastmod < cutoff:
                    continue

                sitemap_count += 1
                print(f"   → Processing monthly archive #{sitemap_count}: {sitemap_url}")

                # Fetch monthly sitemap
                monthly_resp = self.session.get(sitemap_url, timeout=20)
                monthly_root = ET.fromstring(monthly_resp.content)

                for url_entry in monthly_root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                    loc = url_entry.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                    lastmod_elem = url_entry.find('{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod')

                    if not loc:
                        continue

                    article_url = loc.text

                    # Get publication date
                    try:
                        pub_str = lastmod_elem.text if lastmod_elem is not None else lastmod_str
                        pub_date = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                    except:
                        pub_date = datetime.utcnow()

                    if pub_date < cutoff:
                        continue

                    # Scrape real headline
                    title = self._scrape_headline(article_url)
                    article_count += 1

                    articles.append({
                        'title': title,
                        'text': title,
                        'published_at': pub_date,
                        'url': article_url
                    })

                    if article_count % 20 == 0:
                        print(f"      → Collected {article_count} articles so far...")

                    time.sleep(1.0)  # polite delay

            print(f"\n✅ Finished! Processed {sitemap_count} archives → {len(articles)} articles with real headlines")
            return articles

        except Exception as e:
            print(f"[FT Sitemap] Critical Error: {e}")
            return []

    def is_live_mode(self) -> bool:
        return False