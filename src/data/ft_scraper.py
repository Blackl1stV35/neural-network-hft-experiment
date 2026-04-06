"""Revised FT.com Scraper using Playwright (2026-compatible)
Bypasses paywall using real browser rendering.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests
from loguru import logger
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# Gold-related keywords for relevance scoring
GOLD_MACRO_KEYWORDS = [
    "gold", "xauusd", "bullion", "precious metal", "federal reserve", "fomc",
    "interest rate", "monetary policy", "inflation", "cpi", "dollar index",
    "dxy", "safe haven", "nonfarm", "payroll", "gdp", "pmi", "commodity",
    "treasury yield", "central bank"
]

_KEYWORD_PATTERNS = [re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in GOLD_MACRO_KEYWORDS]


@dataclass
class FTArticle:
    url: str
    published_at: str
    headline: str = ""
    summary: str = ""
    body_text: str = ""
    image_caption: str = ""
    fetch_timestamp: str = ""
    relevance_score: float = 0.0

    @property
    def text_for_embedding(self) -> str:
        parts = [self.headline]
        if self.summary:
            parts.append(self.summary)
        if self.body_text:
            parts.append(self.body_text[:600])
        return ". ".join(p for p in parts if p)

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "published_at": self.published_at,
            "headline": self.headline,
            "summary": self.summary,
            "body_text": self.body_text[:1200],
            "image_caption": self.image_caption,
            "fetch_timestamp": self.fetch_timestamp,
            "relevance_score": self.relevance_score,
        }


class FTSitemapScraper:
    INDEX_URL = "https://www.ft.com/sitemaps/index.xml"

    def __init__(self, cache_dir: str = "data/ft_cache", headless: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "sitemaps").mkdir(exist_ok=True)
        (self.cache_dir / "articles").mkdir(exist_ok=True)
        (self.cache_dir / "processed").mkdir(exist_ok=True)

        self.headless = headless
        self.playwright = None
        self.browser = None

    def _init_browser(self):
        if not self.playwright:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=self.headless)

    def _polite_delay(self):
        time.sleep(random.uniform(2.0, 5.0))

    def discover_archives(self) -> List[dict]:
        """Parse index sitemap."""
        cache_path = self.cache_dir / "sitemaps" / "index.xml"
        if cache_path.exists():
            xml_text = cache_path.read_text(encoding="utf-8")
        else:
            resp = requests.get(self.INDEX_URL, timeout=20)
            xml_text = resp.text
            cache_path.write_text(xml_text, encoding="utf-8")

        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_text)
        NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"

        archives = []
        for sitemap in root.findall(f"{NS}sitemap"):
            loc = sitemap.findtext(f"{NS}loc", "")
            if loc:
                archives.append({"url": loc})
        logger.info(f"Discovered {len(archives)} monthly archives")
        return archives

    def filter_archives_by_date(self, archives: List[dict], start_month: str, end_month: Optional[str] = None) -> List[dict]:
        if end_month is None:
            end_month = datetime.utcnow().strftime("%Y-%m")
        filtered = []
        for arch in archives:
            match = re.search(r"archive-(\d{4})-(\d{2})", arch["url"])
            if match:
                month_str = f"{match.group(1)}-{match.group(2)}"
                if start_month <= month_str <= end_month:
                    filtered.append(arch)
        return filtered

    def parse_archive(self, archive_url: str) -> List[dict]:
        """Parse monthly archive."""
        cache_key = hashlib.md5(archive_url.encode()).hexdigest()
        cache_path = self.cache_dir / "sitemaps" / f"{cache_key}.xml"

        if cache_path.exists():
            xml_text = cache_path.read_text(encoding="utf-8")
        else:
            resp = requests.get(archive_url, timeout=15)
            xml_text = resp.text
            cache_path.write_text(xml_text, encoding="utf-8")

        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_text)
        NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"

        articles = []
        for url_elem in root.findall(f"{NS}url"):
            loc = url_elem.findtext(f"{NS}loc", "")
            lastmod = url_elem.findtext(f"{NS}lastmod", "")
            if loc and "/content/" in loc:
                articles.append({"url": loc, "lastmod": lastmod})
        return articles

    def extract_article_content(self, url: str) -> Optional[dict]:
        """Extract content using Playwright."""
        self._init_browser()
        page = self.browser.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=25000)
            self._polite_delay()

            # Robust headline extraction
            headline = page.locator("h1").first.inner_text(timeout=8000).strip() or ""

            # Summary
            summary = ""
            for sel in ["[data-trackable='standfirst']", ".article-standfirst", "main p:first-of-type"]:
                try:
                    summary = page.locator(sel).first.inner_text(timeout=5000).strip()
                    if summary:
                        break
                except:
                    continue

            # Body (first 10-12 paragraphs)
            body_text = ""
            try:
                paragraphs = page.locator("article p").all()
                body_text = " ".join([p.inner_text().strip() for p in paragraphs[:12]])
            except:
                pass

            return {
                "headline": headline,
                "summary": summary,
                "body_text": body_text,
            }

        except Exception as e:
            logger.warning(f"Failed to extract content from {url}: {e}")
            return None
        finally:
            page.close()

    def fetch_articles(
        self,
        start_month: str = "2025-01",
        end_month: Optional[str] = None,
        min_relevance: float = 0.01,
        max_articles_per_month: int = 300,
    ) -> List[FTArticle]:
        archives = self.discover_archives()
        archives = self.filter_archives_by_date(archives, start_month, end_month)

        all_articles: List[FTArticle] = []

        for archive in archives:
            logger.info(f"Processing archive: {archive['url']}")
            entries = self.parse_archive(archive["url"])

            if len(entries) > max_articles_per_month:
                entries = entries[:max_articles_per_month]

            for entry in entries:
                content = self.extract_article_content(entry["url"])
                if not content or not content["headline"]:
                    continue

                article = FTArticle(
                    url=entry["url"],
                    published_at=entry.get("lastmod", ""),
                    headline=content["headline"],
                    summary=content["summary"],
                    body_text=content["body_text"],
                    fetch_timestamp=datetime.utcnow().isoformat(),
                )

                article.relevance_score = sum(
                    1 for pat in _KEYWORD_PATTERNS if pat.search(article.text_for_embedding.lower())
                ) / len(GOLD_MACRO_KEYWORDS)

                if article.relevance_score >= min_relevance:
                    all_articles.append(article)

            logger.info(f"Archive done: {len([a for a in all_articles if a.url.startswith(archive['url'][:60])])} relevant")

        # Save cache
        cache_path = self.cache_dir / "processed" / f"articles_{start_month}_{end_month or 'current'}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump([a.to_dict() for a in all_articles], f, indent=2)

        logger.info(f"Total relevant articles: {len(all_articles)}")
        return all_articles

    def close(self):
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()


# Simple test
if __name__ == "__main__":
    scraper = FTSitemapScraper(headless=True)
    try:
        articles = scraper.fetch_articles(start_month="2025-04", min_relevance=0.01)
        print(f"Scraped {len(articles)} articles.")
    finally:
        scraper.close()