"""
Download historical news from FT sitemap for the exact same period as your price data.
Saves raw articles ready for later FinBERT embedding.
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from src.data.news_fetchers.ft_sitemap import FTSitemapFetcher
from src.data.tick_store import TickStore


def get_price_date_range(duckdb_path: str = "data/ticks.duckdb"):
    """Automatically detect the exact date range from your existing price data"""
    if not os.path.exists(duckdb_path):
        print(f"⚠️  DuckDB not found at {duckdb_path}. Using default 365 days.")
        end = datetime.utcnow()
        start = end - timedelta(days=365)
        return start, end

    store = TickStore(duckdb_path)
    # ✅ CORRECT METHOD: query_ohlcv
    df = store.query_ohlcv("XAUUSD", "M1", limit=None)
    store.close()

    if df.is_empty():
        raise ValueError("No data in ticks.duckdb")

    start = df["timestamp"].min()
    end = df["timestamp"].max()
    print(f"📅 Detected price data range: {start.date()} → {end.date()} ({(end-start).days} days)")
    return start, end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365,
                        help="Number of days to fetch (default: 365)")
    parser.add_argument("--output", type=str, default="data/news",
                        help="Folder to save raw news dataset")
    args = parser.parse_args()

    # Create output folder
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get exact date range from price data (updated today)
    start_date, end_date = get_price_date_range()

    # Override with --days if user wants
    if args.days != 365:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=args.days)

    print(f"🔄 Fetching FT news from {start_date.date()} to {end_date.date()}...")

    fetcher = FTSitemapFetcher()

    articles = fetcher.fetch(hours_back=365*24)   # wide window

    # Filter to match price data range
    filtered = []
    for article in articles:
        pub_date = article.get("published_at")
        if isinstance(pub_date, str):
            pub_date = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
        if start_date <= pub_date <= end_date:
            filtered.append(article)

    print(f"✅ Downloaded {len(filtered)} relevant news articles")

    # Save raw dataset (ready for embedding)
    output_file = output_dir / f"ft_raw_{start_date.date()}_to_{end_date.date()}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, default=str)

    # Also save Parquet (fast loading later)
    if filtered:
        df = pl.DataFrame(filtered)
        parquet_file = output_dir / f"ft_raw_{start_date.date()}_to_{end_date.date()}.parquet"
        df.write_parquet(parquet_file)
        print(f"💾 Saved raw dataset → {output_file} and {parquet_file}")

    print(f"\n🎉 Done! Raw news dataset is ready in {output_dir}")


if __name__ == "__main__":
    main()