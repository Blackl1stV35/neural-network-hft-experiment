"""CLI for FT.com Scraper"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.data.ft_scraper import FTSitemapScraper


def main():
    parser = argparse.ArgumentParser(description="FT.com News Scraper for Gold Sentiment")
    parser.add_argument("--start", required=True, help="Start month YYYY-MM")
    parser.add_argument("--end", default=None, help="End month YYYY-MM")
    parser.add_argument("--min-relevance", type=float, default=0.01)
    parser.add_argument("--max-per-month", type=int, default=300)
    parser.add_argument("--headless", action="store_true", default=True)

    args = parser.parse_args()

    scraper = FTSitemapScraper(headless=args.headless)
    try:
        articles = scraper.fetch_articles(
            start_month=args.start,
            end_month=args.end,
            min_relevance=args.min_relevance,
            max_articles_per_month=args.max_per_month,
        )
        logger.info(f"\n{'='*60}")
        logger.info(f"Scraping complete: {len(articles)} relevant articles")
        logger.info(f"Date range: {args.start} → {args.end or 'now'}")
    finally:
        scraper.close()


if __name__ == "__main__":
    main()