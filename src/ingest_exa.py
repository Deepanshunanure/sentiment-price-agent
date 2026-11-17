"""News ingestion module using Exa API.

Performs semantic search for recent news articles per ticker, normalizes dates to UTC,
caches responses under data/cache/exa/, and saves structured articles to Parquet.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.utils import (
    get_env_var,
    load_settings_config,
    load_tickers_config,
    normalize_to_utc,
    read_cache,
    save_parquet,
    setup_logging,
    write_cache,
)

logger = logging.getLogger("sentiment_price_agent.ingest_exa")


class ExaNewsIngester:
    """Ingests news articles per ticker using the Exa API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "data/cache",
        use_cache: bool = True,
        cache_ttl_hours: int = 12,
    ) -> None:
        self.api_key = api_key or get_env_var("EXA_API_KEY", required=False)
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.cache_ttl_hours = cache_ttl_hours
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazy initialization of Exa client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("EXA_API_KEY is not configured in environment or .env file.")
            from exa_py import Exa

            self._client = Exa(api_key=self.api_key)
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _search_with_retry(self, query: str, num_results: int, objective: str) -> list[dict[str, Any]]:
        """Execute Exa search with retries and exponential backoff."""
        response = self.client.search(
            query=query,
            type="auto",
            num_results=num_results,
            contents={"highlights": True},
            objective=objective,
        )

        results = []
        for r in getattr(response, "results", []):
            highlights = getattr(r, "highlights", [])
            snippet = "\n".join(highlights) if highlights else getattr(r, "text", "") or ""
            # Truncate overly long text snippets if needed
            if len(snippet) > 1500:
                snippet = snippet[:1500] + "..."

            results.append(
                {
                    "title": getattr(r, "title", "") or "",
                    "date": getattr(r, "published_date", "") or "",
                    "url": getattr(r, "url", "") or "",
                    "snippet": snippet,
                }
            )
        return results

    def fetch_ticker_news(
        self,
        ticker: str,
        name: str,
        keywords: Optional[list[str]] = None,
        num_results: int = 10,
    ) -> pd.DataFrame:
        """Fetch news articles for a single ticker.

        Args:
            ticker: Ticker symbol (e.g. '0700.HK').
            name: Full company name.
            keywords: Optional native language keywords.
            num_results: Max articles to fetch.

        Returns:
            DataFrame with columns: title, date, url, snippet, ticker.
        """
        cache_key = f"exa_news_{ticker}_{num_results}"
        max_age_seconds = self.cache_ttl_hours * 3600

        if self.use_cache:
            cached_data = read_cache("exa", cache_key, self.cache_dir, max_age_seconds)
            if cached_data is not None:
                logger.info("Loaded cached Exa news for %s (%d articles)", ticker, len(cached_data))
                df_cached = pd.DataFrame(cached_data)
                if not df_cached.empty and "date" in df_cached.columns:
                    df_cached["date"] = pd.to_datetime(df_cached["date"], utc=True)
                    return df_cached

        if not self.api_key:
            logger.warning("No EXA_API_KEY provided; skipping news fetch for %s", ticker)
            return pd.DataFrame(columns=["title", "date", "url", "snippet", "ticker"])

        # Construct concise targeted query
        kw_str = f" ({keywords[0]})" if keywords else ""
        query = f"recent financial news {name}{kw_str} {ticker}"
        objective = f"Retrieve latest market news, financial analysis, and earnings sentiment for {name} ({ticker})"

        logger.info("Querying Exa for %s: %r", ticker, query)
        try:
            raw_results = self._search_with_retry(query, num_results=num_results, objective=objective)
        except Exception as exc:
            logger.error("Failed to query Exa for %s: %s", ticker, exc)
            return pd.DataFrame(columns=["title", "date", "url", "snippet", "ticker"])

        records = []
        for item in raw_results:
            raw_date = item.get("date")
            utc_date = normalize_to_utc(raw_date) if raw_date else pd.Timestamp.now(tz=timezone.utc)
            records.append(
                {
                    "title": item.get("title", ""),
                    "date": utc_date,
                    "url": item.get("url", ""),
                    "snippet": item.get("snippet", ""),
                    "ticker": ticker,
                }
            )

        df = pd.DataFrame(records)
        if not df.empty and self.use_cache:
            # Cache serializable records
            serializable = []
            for r in records:
                r_copy = dict(r)
                r_copy["date"] = r_copy["date"].isoformat()
                serializable.append(r_copy)
            write_cache("exa", cache_key, serializable, self.cache_dir)

        logger.info("Retrieved %d news articles for %s", len(df), ticker)
        return df


def ingest_news_pipeline(
    tickers_config_path: str = "config/tickers.yaml",
    settings_config_path: str = "config/settings.yaml",
    target_tickers: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Run news ingestion pipeline for configured tickers.

    Args:
        tickers_config_path: Path to tickers.yaml.
        settings_config_path: Path to settings.yaml.
        target_tickers: Optional list to limit tickers (for smoke tests).

    Returns:
        Consolidated DataFrame of news articles.
    """
    tickers_cfg = load_tickers_config(tickers_config_path)
    settings_cfg = load_settings_config(settings_config_path)

    use_cache = settings_cfg.get("cache", {}).get("enabled", True)
    cache_dir = settings_cfg.get("cache", {}).get("cache_dir", "data/cache")
    ttl_hours = settings_cfg.get("cache", {}).get("ttl_hours", 24)
    num_results = settings_cfg.get("exa", {}).get("num_results_per_ticker", 10)

    ingester = ExaNewsIngester(cache_dir=cache_dir, use_cache=use_cache, cache_ttl_hours=ttl_hours)
    all_news: list[pd.DataFrame] = []

    for exchange, ex_data in tickers_cfg.get("exchanges", {}).items():
        for item in ex_data.get("tickers", []):
            ticker = item["ticker"]
            if target_tickers and ticker not in target_tickers:
                continue

            df = ingester.fetch_ticker_news(
                ticker=ticker,
                name=item.get("name", ticker),
                keywords=item.get("keywords", []),
                num_results=num_results,
            )
            if not df.empty:
                all_news.append(df)

    if all_news:
        final_df = pd.concat(all_news, ignore_index=True)
    else:
        final_df = pd.DataFrame(columns=["title", "date", "url", "snippet", "ticker"])

    proc_dir = Path(settings_cfg.get("storage", {}).get("processed_dir", "data/processed"))
    save_parquet(final_df, proc_dir / "news.parquet")

    return final_df


if __name__ == "__main__":
    setup_logging()
    res = ingest_news_pipeline(target_tickers=["0700.HK", "005930.KS"])
    print(f"Total news articles: {len(res)}")
