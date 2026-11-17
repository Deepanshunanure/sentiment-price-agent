"""Price ingestion module using yfinance.

Fetches hourly and daily OHLCV bars, normalizes timestamps to UTC, handles
retries/rate limits, and caches responses to disk.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yfinance as yf
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.utils import (
    get_cache_path,
    load_settings_config,
    load_tickers_config,
    normalize_to_utc,
    read_cache,
    save_parquet,
    setup_logging,
    write_cache,
)

logger = logging.getLogger("sentiment_price_agent.prices")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _fetch_from_yfinance_with_retry(
    ticker: str,
    period: str,
    interval: str,
) -> pd.DataFrame:
    """Internal helper to fetch OHLCV with exponential backoff retries."""
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval, auto_adjust=True)
    return df


def fetch_ticker_prices(
    ticker: str,
    exchange: str,
    period: str = "10d",
    interval: str = "1h",
    use_cache: bool = True,
    cache_ttl_hours: int = 4,
) -> pd.DataFrame:
    """Fetch OHLCV price series for a single ticker and normalize to UTC.

    Args:
        ticker: Ticker symbol (e.g. '0700.HK', '005930.KS').
        exchange: Exchange code ('HKEX' or 'KRX').
        period: yfinance period string (e.g. '10d', '60d').
        interval: yfinance interval string ('1h' or '1d').
        use_cache: Whether to use disk cache.
        cache_ttl_hours: Max cache age in hours.

    Returns:
        DataFrame with columns: ts_utc, exchange, ticker, open, high, low, close, volume.
    """
    cache_key = f"prices_{ticker}_{interval}_{period}"
    max_age_seconds = cache_ttl_hours * 3600

    if use_cache:
        cached_data = read_cache("prices", cache_key, max_age_seconds=max_age_seconds)
        if cached_data is not None:
            logger.info("Loaded cached prices for %s (%s)", ticker, interval)
            df_cached = pd.DataFrame(cached_data)
            if not df_cached.empty and "ts_utc" in df_cached.columns:
                df_cached["ts_utc"] = df_cached["ts_utc"].apply(normalize_to_utc)
                return df_cached

    logger.info("Fetching %s prices for %s (period=%s, interval=%s)", exchange, ticker, period, interval)
    try:
        raw_df = _fetch_from_yfinance_with_retry(ticker, period=period, interval=interval)
    except Exception as exc:
        logger.error("Failed to fetch price data for %s: %s", ticker, exc)
        return pd.DataFrame(columns=["ts_utc", "exchange", "ticker", "open", "high", "low", "close", "volume"])

    if raw_df.empty:
        logger.warning("Empty price dataframe returned for %s", ticker)
        return pd.DataFrame(columns=["ts_utc", "exchange", "ticker", "open", "high", "low", "close", "volume"])

    # Reset index to extract Datetime
    df = raw_df.reset_index()
    # Normalize timestamp column name
    dt_col = "Datetime" if "Datetime" in df.columns else "Date"
    df["ts_utc"] = df[dt_col].apply(normalize_to_utc)

    # Standardize column names
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)
    df["exchange"] = exchange
    df["ticker"] = ticker

    cols = ["ts_utc", "exchange", "ticker", "open", "high", "low", "close", "volume"]
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values("ts_utc").reset_index(drop=True)

    # Cache results
    if use_cache and not df.empty:
        records = df.to_dict(orient="records")
        write_cache("prices", cache_key, records)

    return df


def ingest_prices_pipeline(
    tickers_config_path: str = "config/tickers.yaml",
    settings_config_path: str = "config/settings.yaml",
    target_tickers: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ingest both hourly and daily prices for all configured tickers.

    Args:
        tickers_config_path: Path to tickers.yaml.
        settings_config_path: Path to settings.yaml.
        target_tickers: Optional list to filter specific tickers (for smoke testing).

    Returns:
        Tuple of (df_hourly, df_daily).
    """
    tickers_cfg = load_tickers_config(tickers_config_path)
    settings_cfg = load_settings_config(settings_config_path)

    hourly_period = f"{settings_cfg.get('prices', {}).get('hourly_period_days', 10)}d"
    daily_period = f"{settings_cfg.get('prices', {}).get('daily_period_days', 60)}d"
    use_cache = settings_cfg.get("cache", {}).get("enabled", True)
    ttl_hours = settings_cfg.get("cache", {}).get("ttl_hours", 24)

    all_hourly: list[pd.DataFrame] = []
    all_daily: list[pd.DataFrame] = []

    for exchange, ex_data in tickers_cfg.get("exchanges", {}).items():
        for item in ex_data.get("tickers", []):
            ticker = item["ticker"]
            if target_tickers and ticker not in target_tickers:
                continue

            # Fetch hourly
            df_h = fetch_ticker_prices(
                ticker=ticker,
                exchange=exchange,
                period=hourly_period,
                interval="1h",
                use_cache=use_cache,
                cache_ttl_hours=ttl_hours,
            )
            if not df_h.empty:
                all_hourly.append(df_h)

            # Fetch daily
            df_d = fetch_ticker_prices(
                ticker=ticker,
                exchange=exchange,
                period=daily_period,
                interval="1d",
                use_cache=use_cache,
                cache_ttl_hours=ttl_hours,
            )
            if not df_d.empty:
                all_daily.append(df_d)

    df_hourly = pd.concat(all_hourly, ignore_index=True) if all_hourly else pd.DataFrame()
    df_daily = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()

    proc_dir = Path(settings_cfg.get("storage", {}).get("processed_dir", "data/processed"))
    if not df_hourly.empty:
        save_parquet(df_hourly, proc_dir / "prices_hourly.parquet")
    if not df_daily.empty:
        save_parquet(df_daily, proc_dir / "prices_daily.parquet")

    return df_hourly, df_daily


if __name__ == "__main__":
    setup_logging()
    h, d = ingest_prices_pipeline()
    print(f"Hourly rows: {len(h)}, Daily rows: {len(d)}")
