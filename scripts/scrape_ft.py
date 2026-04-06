# scripts/scrape_ft.py
#!/usr/bin/env python3
"""
Production-ready FT.com News Scraper
- curl_cffi for fast sitemap crawling
- Playwright CDP for reliable headline extraction (bypasses paywall)
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import polars as pl
from curl_cffi.requests import Session
from playwright.async_api import async_playwright

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HybridFTScraper:
    def __init__(self, cookie_string: str):
        self.cookie_string = cookie_string
        self.curl = Session(impersonate="chrome")
        self.browser = None
        self.context = None
        self.playwright = None

    async def setup(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36 Edg/146.0.0.0"
        )
        # Add cookies
        cookie_list = []
        for c in self.cookie_string.split(";"):
            if "=" in c:
                name, value = c.strip().split("=", 1)
                cookie_list.append({"name": name.strip(), "value": value.strip(), "domain": ".ft.com", "path": "/"})
        await self.context.add_cookies(cookie_list)
        print("✅ Browser + cookies initialized")

    async def get_headline(self, url: str, max_retries: int = 2):
        """Extract real headline using CDP with retry logic"""
        for attempt in range(max_retries):
            page = None
            try:
                page = await self.context.new_page()
                # Use domcontentloaded instead of networkidle for heavy JS sites
                # FT.com has continuous ads/tracking so networkidle never completes
                await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                
                # Wait a bit for JSON-LD to render
                await asyncio.sleep(2)
                
                json_ld = await page.evaluate("""
                    () => {
                        const scripts = document.querySelectorAll('script[type="application/ld+json"]');
                        for (let s of scripts) {
                            try {
                                const data = JSON.parse(s.textContent);
                                if (data["@type"] === "NewsArticle" || data["@type"] === "Article") {
                                    return data;
                                }
                            } catch(e) {}
                        }
                        return null;
                    }
                """)
                if json_ld:
                    headline = json_ld.get("headline") or json_ld.get("alternativeHeadline")
                    if headline:
                        return headline
                
                logger.warning(f"No headline found in JSON-LD for {url}")
                return None
                
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries} for {url}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch {url} after {max_retries} attempts")
                    return None
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                return None
            finally:
                if page:
                    await page.close()

    def fetch_sitemap(self, url: str):
        """Fast sitemap fetch using curl_cffi"""
        try:
            resp = self.curl.get(url, timeout=15)
            return resp.text if resp.status_code == 200 else None
        except Exception as e:
            print(f"    [curl_cffi failed for {url}] {e}")
            return None

    async def close(self):
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="Start month YYYY-MM (e.g. 2025-04)")
    parser.add_argument("--end", required=True, help="End month YYYY-MM (e.g. 2026-04)")
    args = parser.parse_args()

    # ←←← PASTE YOUR FULL COOKIE STRING HERE (from curl -b) ←←←
    COOKIE_STRING = "FTClientSessionId=880b2d28-f266-42df-8d2e-b3adbfe3bf97; spoor-id=880b2d28-f266-42df-8d2e-b3adbfe3bf97; __cf_bm=lrXBQ63_Zvx3cEw2_87rGviYecnpxYBpDX4796UzJeA-1775499541.168267-1.0.1.1-bMRDCu52SofwFZlk05NECd8SC.B4q2eO5URZBhZLk1DgCTleW5ngyXMC7sDbvx8jpo7RVCHL4IzHoDWoR4hsu39gjrOPmI58uFvNd8m30N6Jck2so._PXWSi18rjb5Ve"

    scraper = HybridFTScraper(COOKIE_STRING)
    try:
        await scraper.setup()

        print(f"🔄 Full scraping FT news from {args.start} to {args.end}...")

        articles = []
        start_date = datetime.strptime(args.start, "%Y-%m")
        end_date = datetime.strptime(args.end, "%Y-%m")

        # 1. Fetch sitemap index
        index_url = "https://www.ft.com/sitemaps/index.xml"
        print("📡 Fetching sitemap index...")
        index_xml = scraper.fetch_sitemap(index_url)

        if not index_xml:
            print("❌ Failed to fetch sitemap index")
            return

        # 2. Parse monthly archives
        from xml.etree import ElementTree as ET
        root = ET.fromstring(index_xml)

        for sitemap in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
            loc = sitemap.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
            lastmod = sitemap.find('{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod')

            if loc is None or loc.text is None:
                continue

            archive_url = loc.text
            # Check if archive is within date range
            try:
                # Extract date from URL like: https://www.ft.com/sitemaps/archive-2025-04.xml
                date_str = archive_url.split("archive-")[-1].split(".xml")[0]
                archive_date = datetime.strptime(date_str, "%Y-%m")
            except ValueError:
                continue

            if not (start_date <= archive_date <= end_date):
                continue

            print(f"   → Processing monthly archive: {archive_url}")

            # 3. Fetch monthly archive
            monthly_xml = scraper.fetch_sitemap(archive_url)
            if not monthly_xml:
                continue

            monthly_root = ET.fromstring(monthly_xml)

            for url_entry in monthly_root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                loc = url_entry.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if loc is None or loc.text is None:
                    continue

                article_url = loc.text

                # 4. Extract headline using CDP
                logger.info(f"Processing: {article_url}")
                headline = await scraper.get_headline(article_url)

                if headline:
                    articles.append({
                        "title": headline,
                        "url": article_url,
                        "published_at": datetime.now()  # will be improved later
                    })
                    print(f"      ✅ Added: {headline[:80]}...")
                    # Small delay between requests to avoid overwhelming server
                    await asyncio.sleep(0.5)
                else:
                    logger.info(f"Skipped (no headline): {article_url}")
                    await asyncio.sleep(0.3)

        print(f"\n✅ Total relevant articles found: {len(articles)}")

        # 5. Save raw dataset
        if articles:
            output_dir = Path("data/news")
            output_dir.mkdir(parents=True, exist_ok=True)
            df = pl.DataFrame(articles)
            output_file = output_dir / f"ft_raw_{args.start}_to_{args.end}.parquet"
            df.write_parquet(output_file)
            print(f"💾 Saved raw news dataset → {output_file}")

        print("🎉 Full scraping completed!")
    finally:
        await scraper.close()


if __name__ == "__main__":
    asyncio.run(main())