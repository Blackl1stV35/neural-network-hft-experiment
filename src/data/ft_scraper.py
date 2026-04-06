"""FT.com sitemap-based news scraper for training pipeline.

Architecture:
    1. Parse index sitemap → discover monthly archive URLs
    2. Parse monthly archives → extract article URLs + lastmod dates
    3. Filter articles by date range and gold/macro keywords
    4. Fetch individual articles → extract headline + body via DOM xpath
    5. Cache everything to disk to respect rate limits

This is for TRAINING DATA ONLY. For live inference, use the GDELT/RSS pipeline.
FT.com will rate-limit aggressive scraping — use polite delays and caching.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from loguru import logger


# Sitemap XML namespaces
NS_SITEMAP = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
NS_IMAGE = "{http://www.google.com/schemas/sitemap-image/1.1}"

# Gold/macro keywords for filtering relevant articles
GOLD_MACRO_KEYWORDS = [
    # Direct gold
    "gold", "xauusd", "bullion", "precious metal",
    # Central banks / monetary policy
    "federal reserve", "fomc", "interest rate", "rate decision", "rate cut",
    "rate hike", "monetary policy", "quantitative", "tightening", "easing",
    "central bank", "ecb", "boj", "pboc",
    # Inflation / yields
    "inflation", "cpi", "ppi", "treasury yield", "bond yield", "real yield",
    "deflation", "stagflation",
    # Dollar / FX
    "dollar index", "dxy", "dollar strength", "dollar weakness",
    "currency", "forex", "usd",
    # Geopolitical / safe haven
    "safe haven", "geopolitical", "war", "conflict", "sanctions",
    "trade war", "tariff",
    # Economic indicators
    "nonfarm", "payroll", "unemployment", "gdp", "retail sales",
    "manufacturing", "pmi", "consumer confidence",
    # Commodities context
    "commodity", "oil price", "silver", "copper",
]

# Compile keyword patterns for fast matching
_KEYWORD_PATTERNS = [re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in GOLD_MACRO_KEYWORDS]


@dataclass
class FTArticle:
    """Parsed FT.com article."""

    url: str
    published_at: str  # ISO format
    headline: str = ""
    summary: str = ""
    body_text: str = ""
    image_caption: str = ""
    fetch_timestamp: str = ""
    relevance_score: float = 0.0  # keyword match density

    @property
    def text_for_embedding(self) -> str:
        """Combined text for FinBERT embedding."""
        parts = [self.headline]
        if self.summary:
            parts.append(self.summary)
        if self.body_text:
            # Use first 500 chars of body to stay within BERT token limits
            parts.append(self.body_text[:500])
        return ". ".join(p for p in parts if p)

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "published_at": self.published_at,
            "headline": self.headline,
            "summary": self.summary,
            "body_text": self.body_text[:1000],  # truncate for storage
            "image_caption": self.image_caption,
            "fetch_timestamp": self.fetch_timestamp,
            "relevance_score": self.relevance_score,
        }


def _relevance_score(text: str) -> float:
    """Score text by keyword match density. Higher = more relevant to gold trading."""
    if not text:
        return 0.0
    text_lower = text.lower()
    matches = sum(1 for pat in _KEYWORD_PATTERNS if pat.search(text_lower))
    return matches / len(GOLD_MACRO_KEYWORDS)


class FTSitemapScraper:
    """Scrape FT.com articles via their public sitemap XML feeds.

    Sitemap structure:
        index.xml → lists monthly archives (archive-YYYY-MM.xml)
        archive-YYYY-MM.xml → lists article URLs with lastmod dates
        article page → headline at specific DOM xpath

    Usage:
        scraper = FTSitemapScraper(cache_dir="data/ft_cache")
        articles = scraper.fetch_articles(
            start_month="2024-01",
            end_month="2025-03",
            min_relevance=0.02,
        )
    """

    INDEX_URL = "https://www.ft.com/sitemaps/index.xml"
    ARCHIVE_URL_TEMPLATE = "https://www.ft.com/sitemaps/archive-{year}-{month:02d}.xml"

    # DOM xpath for headline extraction (as provided by user)
    # /html/body/div[1]/div[2]/div/main/div/div[1]/div/div/div[1]/div/section/h1/blockquote
    # We'll use multiple CSS selector strategies for robustness
    HEADLINE_SELECTORS = [
        "h1 blockquote",                          # simplest match
        "main h1",                                  # fallback: any h1 in main
        "[data-trackable='header'] h1",             # FT data attribute
        ".article-header h1",                       # class-based
        "section h1",                               # generic section h1
    ]

    SUMMARY_SELECTORS = [
        "[data-trackable='standfirst']",
        ".article-standfirst",
        "main p:first-of-type",
    ]

    BODY_SELECTORS = [
        "[data-trackable='article-body']",
        ".article-body",
        "main article",
        "main .body-content",
    ]

    def __init__(
        self,
        cache_dir: str = "data/ft_cache",
        request_delay: tuple[float, float] = (1.0, 3.0),
        max_retries: int = 3,
        user_agent: str = "Mozilla/5.0 (compatible; ResearchBot/1.0; academic research)",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "sitemaps").mkdir(exist_ok=True)
        (self.cache_dir / "articles").mkdir(exist_ok=True)
        (self.cache_dir / "processed").mkdir(exist_ok=True)

        self.request_delay = request_delay
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })

    def _polite_delay(self) -> None:
        """Random delay between requests to be respectful."""
        delay = random.uniform(*self.request_delay)
        time.sleep(delay)

    def _fetch_url(self, url: str, timeout: int = 15) -> Optional[str]:
        """Fetch URL with retries and caching."""
        # Check cache first
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_path = self.cache_dir / "articles" / f"{cache_key}.html"

        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8", errors="replace")

        for attempt in range(self.max_retries):
            try:
                self._polite_delay()
                resp = self.session.get(url, timeout=timeout, allow_redirects=True)

                if resp.status_code == 200:
                    content = resp.text
                    cache_path.write_text(content, encoding="utf-8")
                    return content

                elif resp.status_code == 429:
                    wait = 30 * (attempt + 1)
                    logger.warning(f"Rate limited on {url}, waiting {wait}s")
                    time.sleep(wait)

                elif resp.status_code == 403:
                    logger.debug(f"403 on {url} (paywall/blocked)")
                    return None

                else:
                    logger.warning(f"HTTP {resp.status_code} on {url}")

            except requests.RequestException as e:
                logger.warning(f"Request failed ({attempt+1}/{self.max_retries}): {e}")
                time.sleep(5 * (attempt + 1))

        return None

    def _fetch_xml(self, url: str) -> Optional[str]:
        """Fetch XML content (sitemaps)."""
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_path = self.cache_dir / "sitemaps" / f"{cache_key}.xml"

        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

        try:
            self._polite_delay()
            resp = self.session.get(url, timeout=15)
            if resp.status_code == 200:
                cache_path.write_text(resp.text, encoding="utf-8")
                return resp.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch sitemap {url}: {e}")

        return None

    # ------------------------------------------------------------------
    # Step 1: Parse index sitemap → discover monthly archives
    # ------------------------------------------------------------------

    def discover_archives(self) -> list[dict]:
        """Parse the index sitemap to find all monthly archive URLs.

        Returns:
            List of {"url": str, "lastmod": str} dicts.
        """
        xml_text = self._fetch_xml(self.INDEX_URL)
        if not xml_text:
            logger.error("Failed to fetch index sitemap")
            return []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"Failed to parse index sitemap: {e}")
            return []

        archives = []
        for sitemap_elem in root.findall(f"{NS_SITEMAP}sitemap"):
            loc = sitemap_elem.findtext(f"{NS_SITEMAP}loc", "")
            lastmod = sitemap_elem.findtext(f"{NS_SITEMAP}lastmod", "")
            if loc:
                archives.append({"url": loc, "lastmod": lastmod})

        logger.info(f"Discovered {len(archives)} monthly archives from index sitemap")
        return archives

    def filter_archives_by_date(
        self,
        archives: list[dict],
        start_month: str = "2020-01",
        end_month: Optional[str] = None,
    ) -> list[dict]:
        """Filter archives to a date range.

        Args:
            start_month: "YYYY-MM" format
            end_month: "YYYY-MM" format (default: current month)
        """
        if end_month is None:
            end_month = datetime.utcnow().strftime("%Y-%m")

        filtered = []
        for arch in archives:
            # Extract YYYY-MM from URL like archive-2024-03.xml
            match = re.search(r"archive-(\d{4})-(\d{2})", arch["url"])
            if match:
                archive_month = f"{match.group(1)}-{match.group(2)}"
                if start_month <= archive_month <= end_month:
                    filtered.append(arch)

        logger.info(f"Filtered to {len(filtered)} archives in [{start_month} → {end_month}]")
        return filtered

    # ------------------------------------------------------------------
    # Step 2: Parse monthly archives → extract article URLs
    # ------------------------------------------------------------------

    def parse_archive(self, archive_url: str) -> list[dict]:
        """Parse a monthly archive sitemap to extract article URLs.

        Returns:
            List of {"url": str, "lastmod": str, "image_caption": str}
        """
        xml_text = self._fetch_xml(archive_url)
        if not xml_text:
            return []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"Failed to parse archive {archive_url}: {e}")
            return []

        articles = []
        for url_elem in root.findall(f"{NS_SITEMAP}url"):
            loc = url_elem.findtext(f"{NS_SITEMAP}loc", "")
            lastmod = url_elem.findtext(f"{NS_SITEMAP}lastmod", "")

            # Extract image caption if present (often contains headline hint)
            image_caption = ""
            image_elem = url_elem.find(f"{NS_IMAGE}image")
            if image_elem is not None:
                caption_elem = image_elem.find(f"{NS_IMAGE}caption")
                if caption_elem is not None and caption_elem.text:
                    image_caption = caption_elem.text.strip()

            if loc and "/content/" in loc:
                articles.append({
                    "url": loc,
                    "lastmod": lastmod,
                    "image_caption": image_caption,
                })

        logger.info(f"Parsed {len(articles)} article URLs from {archive_url}")
        return articles

    # ------------------------------------------------------------------
    # Step 3: Fetch individual articles → extract content via DOM
    # ------------------------------------------------------------------

    def extract_article_content(self, url: str) -> Optional[dict]:
        """Fetch an article page and extract headline + body.

        Uses BeautifulSoup with multiple CSS selector strategies
        since FT.com DOM structure may vary across article types.
        """
        html = self._fetch_url(url)
        if not html:
            return None

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("beautifulsoup4 not installed. pip install beautifulsoup4 lxml")
            return None

        soup = BeautifulSoup(html, "lxml")

        # Extract headline — try multiple selectors
        headline = ""
        for selector in self.HEADLINE_SELECTORS:
            elem = soup.select_one(selector)
            if elem and elem.get_text(strip=True):
                headline = elem.get_text(strip=True)
                break

        # Fallback: use <title> tag
        if not headline:
            title_tag = soup.find("title")
            if title_tag:
                headline = title_tag.get_text(strip=True)
                # Remove " | Financial Times" suffix
                headline = re.sub(r"\s*\|\s*Financial Times\s*$", "", headline)

        # Fallback: og:title meta tag
        if not headline:
            og_title = soup.find("meta", property="og:title")
            if og_title:
                headline = og_title.get("content", "")

        # Extract summary/standfirst
        summary = ""
        for selector in self.SUMMARY_SELECTORS:
            elem = soup.select_one(selector)
            if elem and elem.get_text(strip=True):
                summary = elem.get_text(strip=True)
                break

        # Fallback: og:description
        if not summary:
            og_desc = soup.find("meta", property="og:description")
            if og_desc:
                summary = og_desc.get("content", "")

        # Extract body text (may be limited by paywall)
        body_text = ""
        for selector in self.BODY_SELECTORS:
            elem = soup.select_one(selector)
            if elem:
                paragraphs = elem.find_all("p")
                if paragraphs:
                    body_text = " ".join(p.get_text(strip=True) for p in paragraphs[:10])
                    break

        return {
            "headline": headline,
            "summary": summary,
            "body_text": body_text,
        }

    # ------------------------------------------------------------------
    # Step 4: Full pipeline — discover → filter → fetch → score
    # ------------------------------------------------------------------

    def fetch_articles(
        self,
        start_month: str = "2024-01",
        end_month: Optional[str] = None,
        min_relevance: float = 0.01,
        max_articles_per_month: int = 500,
        keywords_prefilter: bool = True,
    ) -> list[FTArticle]:
        """Full pipeline: fetch gold/macro-relevant articles from FT.com sitemaps.

        Args:
            start_month: Start date "YYYY-MM"
            end_month: End date "YYYY-MM" (default: current month)
            min_relevance: Minimum keyword relevance score to keep
            max_articles_per_month: Cap articles per archive to manage load
            keywords_prefilter: If True, pre-filter by image_caption keywords
                              before fetching full articles (saves bandwidth)

        Returns:
            List of FTArticle objects with headline, summary, body, relevance score.
        """
        # Check for processed cache
        cache_key = f"{start_month}_{end_month or 'now'}_{min_relevance}"
        processed_path = self.cache_dir / "processed" / f"articles_{cache_key}.json"
        if processed_path.exists():
            logger.info(f"Loading cached articles from {processed_path}")
            with open(processed_path) as f:
                data = json.load(f)
            return [FTArticle(**d) for d in data]

        # Discover archives
        all_archives = self.discover_archives()
        archives = self.filter_archives_by_date(all_archives, start_month, end_month)

        all_articles: list[FTArticle] = []

        for archive in archives:
            logger.info(f"Processing archive: {archive['url']}")
            article_entries = self.parse_archive(archive["url"])

            # Optional pre-filter by image caption keywords
            if keywords_prefilter:
                article_entries = [
                    a for a in article_entries
                    if _relevance_score(a.get("image_caption", "")) > 0
                    or not a.get("image_caption")  # keep articles without captions
                ]

            # Cap per month
            if len(article_entries) > max_articles_per_month:
                article_entries = article_entries[:max_articles_per_month]

            fetched_count = 0
            for entry in article_entries:
                content = self.extract_article_content(entry["url"])
                if content is None:
                    continue

                article = FTArticle(
                    url=entry["url"],
                    published_at=entry.get("lastmod", ""),
                    headline=content["headline"],
                    summary=content["summary"],
                    body_text=content["body_text"],
                    image_caption=entry.get("image_caption", ""),
                    fetch_timestamp=datetime.utcnow().isoformat(),
                )

                # Score relevance
                combined_text = f"{article.headline} {article.summary} {article.image_caption}"
                article.relevance_score = _relevance_score(combined_text)

                if article.relevance_score >= min_relevance:
                    all_articles.append(article)
                    fetched_count += 1

            logger.info(
                f"Archive complete: {fetched_count} relevant articles "
                f"(of {len(article_entries)} total)"
            )

        # Sort by date
        all_articles.sort(key=lambda a: a.published_at)

        # Cache processed results
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        with open(processed_path, "w") as f:
            json.dump([a.to_dict() for a in all_articles], f, indent=2)

        logger.info(
            f"Pipeline complete: {len(all_articles)} relevant articles "
            f"from {start_month} to {end_month or 'now'}"
        )
        return all_articles

    def fetch_month(self, year: int, month: int, min_relevance: float = 0.01) -> list[FTArticle]:
        """Convenience: fetch articles for a single month."""
        month_str = f"{year}-{month:02d}"
        return self.fetch_articles(
            start_month=month_str,
            end_month=month_str,
            min_relevance=min_relevance,
        )


class FTArticleCache:
    """Manage cached FT articles for fast access during training."""

    def __init__(self, cache_dir: str = "data/ft_cache/processed"):
        self.cache_dir = Path(cache_dir)

    def load_date_range(
        self,
        start_date: str,
        end_date: str,
    ) -> list[FTArticle]:
        """Load cached articles within a date range.

        Args:
            start_date: "YYYY-MM-DD" format
            end_date: "YYYY-MM-DD" format
        """
        all_articles = []

        for json_file in sorted(self.cache_dir.glob("articles_*.json")):
            with open(json_file) as f:
                data = json.load(f)

            for d in data:
                pub = d.get("published_at", "")[:10]  # YYYY-MM-DD
                if start_date <= pub <= end_date:
                    all_articles.append(FTArticle(**d))

        logger.info(f"Loaded {len(all_articles)} cached articles for [{start_date} → {end_date}]")
        return all_articles

    def get_articles_for_timestamp(
        self,
        target_time: datetime,
        lookback_hours: int = 24,
        max_articles: int = 20,
    ) -> list[FTArticle]:
        """Get articles published within lookback window of a timestamp.

        Useful for aligning news with specific trading bars during training.
        """
        cutoff = target_time - timedelta(hours=lookback_hours)

        all_articles = []
        for json_file in sorted(self.cache_dir.glob("articles_*.json")):
            with open(json_file) as f:
                data = json.load(f)

            for d in data:
                try:
                    pub_time = datetime.fromisoformat(d["published_at"].replace("Z", "+00:00"))
                    pub_time = pub_time.replace(tzinfo=None)
                    if cutoff <= pub_time <= target_time:
                        all_articles.append(FTArticle(**d))
                except (ValueError, KeyError):
                    continue

        # Sort by relevance, take top N
        all_articles.sort(key=lambda a: a.relevance_score, reverse=True)
        return all_articles[:max_articles]
