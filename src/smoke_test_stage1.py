"""Stage 1 Smoke Test Script.

Validates end-to-end ingestion on 1 ticker per exchange:
- HKEX: 0700.HK (Tencent)
- KRX: 005930.KS (Samsung Electronics)
Demonstrates:
- Hourly and daily OHLCV prices (yfinance) converted to UTC
- Recent relevant news articles (Exa API) with UTC dates
- Social posts normalization and caching (Apify)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ingest_apify import ingest_social_pipeline
from src.ingest_exa import ingest_news_pipeline
from src.prices import ingest_prices_pipeline
from src.utils import load_parquet, setup_logging


def run_stage1_smoke_test() -> None:
    """Execute smoke test for Stage 1."""
    setup_logging()
    print("=" * 70)
    print("STAGE 1 SMOKE TEST: 1 TICKER PER EXCHANGE (0700.HK & 005930.KS)")
    print("=" * 70)

    target_tickers = ["0700.HK", "005930.KS"]

    # 1. Prices Ingestion
    print("\n--- [1/3] Ingesting OHLCV Prices (yfinance) ---")
    df_h, df_d = ingest_prices_pipeline(target_tickers=target_tickers)
    print(f"[Prices Hourly] Total records: {len(df_h)}")
    if not df_h.empty:
        print("Sample Hourly Prices:")
        print(df_h.groupby("ticker").tail(2)[["ts_utc", "exchange", "ticker", "open", "high", "low", "close", "volume"]])

    print(f"\n[Prices Daily] Total records: {len(df_d)}")
    if not df_d.empty:
        print("Sample Daily Prices:")
        print(df_d.groupby("ticker").tail(2)[["ts_utc", "exchange", "ticker", "close", "volume"]])

    # 2. News Ingestion (Exa)
    print("\n--- [2/3] Ingesting News (Exa API) ---")
    df_news = ingest_news_pipeline(target_tickers=target_tickers)
    print(f"[News] Total records: {len(df_news)}")
    if not df_news.empty:
        print("Sample News Articles:")
        for _, row in df_news.groupby("ticker").head(1).iterrows():
            print(f"[{row['ticker']}] {row['title']} ({row['date']})")
            print(f"URL: {row['url']}")
            snippet = row['snippet'][:120] + "..." if len(row['snippet']) > 120 else row['snippet']
            print(f"Snippet: {snippet}\n")

    # 3. Social Posts Ingestion (Apify)
    print("\n--- [3/3] Ingesting Social Posts (Apify Client) ---")
    df_posts = ingest_social_pipeline(target_tickers=target_tickers)
    print(f"[Posts] Total records: {len(df_posts)}")
    if not df_posts.empty:
        print("Sample Normalized Posts:")
        print(df_posts.head(2)[["ts_utc", "exchange", "source", "ticker", "text", "likes", "views"]])

    # 4. Parquet File Verification
    print("\n--- [4/4] Processed Parquet Storage Verification ---")
    proc_files = [
        "data/processed/prices_hourly.parquet",
        "data/processed/prices_daily.parquet",
        "data/processed/news.parquet",
        "data/processed/posts.parquet",
    ]
    for p in proc_files:
        path = Path(p)
        exists = path.exists()
        size_kb = path.stat().st_size / 1024 if exists else 0
        rows = len(load_parquet(path)) if exists else 0
        print(f"  {p:<38} | Exists: {str(exists):<5} | Size: {size_kb:6.1f} KB | Rows: {rows}")

    print("\n" + "=" * 70)
    print("STAGE 1 SMOKE TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_stage1_smoke_test()
