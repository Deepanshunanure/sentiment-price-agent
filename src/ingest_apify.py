"""Social post ingestion module using Apify Client.

Executes configured Apify Actors, normalizes posts, extracts ticker associations,
handles retries/rate-limits with exponential backoff, and caches all results.
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
    load_actors_config,
    load_settings_config,
    load_tickers_config,
    normalize_to_utc,
    read_cache,
    save_parquet,
    setup_logging,
    write_cache,
)

logger = logging.getLogger("sentiment_price_agent.ingest_apify")

POSTS_COLUMNS = [
    "ts_utc",
    "exchange",
    "source",
    "ticker",
    "text",
    "likes",
    "reposts",
    "views",
    "author_followers",
]


class ApifySocialIngester:
    """Ingests social posts from Apify actors configured in config/actors.yaml."""

    def __init__(
        self,
        api_token: Optional[str] = None,
        cache_dir: str = "data/cache",
        use_cache: bool = True,
        cache_ttl_hours: int = 12,
    ) -> None:
        self.api_token = api_token or get_env_var("APIFY_API_TOKEN", required=False)
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.cache_ttl_hours = cache_ttl_hours
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazy initialization of ApifyClient."""
        if self._client is None:
            if not self.api_token:
                raise ValueError("APIFY_API_TOKEN is not configured in environment or .env file.")
            from apify_client import ApifyClient

            self._client = ApifyClient(token=self.api_token)
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _run_actor_with_retry(
        self,
        actor_id: str,
        run_input: dict[str, Any],
        timeout_secs: int = 120,
    ) -> list[dict[str, Any]]:
        """Run an Apify actor and fetch its dataset items."""
        logger.info("Executing Apify Actor %s with timeout=%ds", actor_id, timeout_secs)
        actor_client = self.client.actor(actor_id)
        run = actor_client.call(run_input=run_input, timeout_secs=timeout_secs)

        if not run:
            logger.warning("Actor %s returned empty run object", actor_id)
            return []

        dataset_id = run.get("defaultDatasetId") or getattr(run, "default_dataset_id", None)
        if not dataset_id:
            logger.warning("Actor run %s has no defaultDatasetId", getattr(run, "id", "unknown"))
            return []

        dataset_client = self.client.dataset(dataset_id)
        items_resource = dataset_client.list_items()
        items = getattr(items_resource, "items", items_resource)
        logger.info("Actor %s run finished successfully; fetched %d items", actor_id, len(items))
        return list(items)

    def fetch_source_items(
        self,
        source_name: str,
        actor_id: Optional[str],
        input_template: dict[str, Any],
        exchange: str,
    ) -> list[dict[str, Any]]:
        """Fetch items for a specific source and exchange, using cache if available.

        Args:
            source_name: Name of the social source (e.g. 'lihkg', 'naver_finance').
            actor_id: Apify actor ID or None.
            input_template: Actor input payload.
            exchange: 'HKEX' or 'KRX'.

        Returns:
            List of raw item dictionaries.
        """
        if not actor_id or str(actor_id).strip() in ("", "None", "null"):
            logger.warning(
                "Source '%s' (%s) has no actor_id configured in config/actors.yaml. Skipping.",
                source_name,
                exchange,
            )
            return []

        cache_key = f"apify_{exchange}_{source_name}_{actor_id}"
        max_age_seconds = self.cache_ttl_hours * 3600

        if self.use_cache:
            cached_data = read_cache("apify", cache_key, self.cache_dir, max_age_seconds)
            if cached_data is not None:
                logger.info("Loaded %d cached items for %s (%s)", len(cached_data), source_name, exchange)
                return cached_data

        if not self.api_token:
            logger.error("APIFY_API_TOKEN is missing. Cannot run actor %s for %s.", actor_id, source_name)
            return []

        try:
            items = self._run_actor_with_retry(actor_id=actor_id, run_input=input_template)
            if self.use_cache and items:
                write_cache("apify", cache_key, items, self.cache_dir)
            return items
        except Exception as exc:
            logger.error("Failed executing Apify actor %s for source %s: %s", actor_id, source_name, exc)
            return []

    def normalize_post(
        self,
        item: dict[str, Any],
        exchange: str,
        source: str,
        target_ticker: str,
    ) -> dict[str, Any]:
        """Normalize raw scraped item into standard post schema.

        Schema: ts_utc, exchange, source, ticker, text, likes, reposts, views, author_followers
        """
        raw_ts = (
            item.get("ts_utc")
            or item.get("createdAt")
            or item.get("created_at")
            or item.get("date")
            or item.get("timestamp")
        )
        utc_ts = normalize_to_utc(raw_ts)

        # Extract text from typical fields or fallback
        text = (
            item.get("text")
            or item.get("content")
            or item.get("body")
            or item.get("title")
            or str(item)
        )
        if isinstance(text, str):
            text = text.strip()
        else:
            text = str(text)

        # Ticker resolution
        ticker = item.get("ticker") or target_ticker

        return {
            "ts_utc": utc_ts,
            "exchange": exchange,
            "source": source,
            "ticker": ticker,
            "text": text,
            "likes": int(item.get("likes") or item.get("likeCount") or 0),
            "reposts": int(item.get("reposts") or item.get("retweetCount") or item.get("shares") or 0),
            "views": int(item.get("views") or item.get("viewCount") or 0),
            "author_followers": int(item.get("author_followers") or item.get("followersCount") or 0),
        }


def match_post_to_tickers(
    text: str,
    ticker_definitions: list[dict[str, Any]],
) -> list[str]:
    """Identify which tickers from the exchange are mentioned in the text.

    Args:
        text: Post text.
        ticker_definitions: List of ticker metadata from config/tickers.yaml.

    Returns:
        List of matched ticker symbols.
    """
    matched: list[str] = []
    text_upper = text.upper()

    for item in ticker_definitions:
        symbol = item["ticker"]
        # Match symbol (e.g. "0700.HK" or "0700")
        clean_num = symbol.split(".")[0]
        if symbol in text or clean_num in text:
            matched.append(symbol)
            continue

        # Match cashtags
        for tag in item.get("cashtags", []):
            if tag.upper() in text_upper:
                matched.append(symbol)
                break
        if symbol in matched:
            continue

        # Match native keywords
        for kw in item.get("keywords", []):
            if kw in text:
                matched.append(symbol)
                break

    return matched


def ingest_social_pipeline(
    actors_config_path: str = "config/actors.yaml",
    tickers_config_path: str = "config/tickers.yaml",
    settings_config_path: str = "config/settings.yaml",
    target_tickers: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Run Apify social post ingestion pipeline for HKEX and KRX.

    Args:
        actors_config_path: Path to config/actors.yaml.
        tickers_config_path: Path to config/tickers.yaml.
        settings_config_path: Path to config/settings.yaml.
        target_tickers: Optional list to limit tickers (for smoke tests).

    Returns:
        Consolidated DataFrame saved to data/processed/posts.parquet.
    """
    actors_cfg = load_actors_config(actors_config_path)
    tickers_cfg = load_tickers_config(tickers_config_path)
    settings_cfg = load_settings_config(settings_config_path)

    use_cache = settings_cfg.get("cache", {}).get("enabled", True)
    cache_dir = settings_cfg.get("cache", {}).get("cache_dir", "data/cache")
    ttl_hours = settings_cfg.get("cache", {}).get("ttl_hours", 24)

    ingester = ApifySocialIngester(
        cache_dir=cache_dir,
        use_cache=use_cache,
        cache_ttl_hours=ttl_hours,
    )

    all_normalized_posts: list[dict[str, Any]] = []

    for exchange, ex_data in actors_cfg.get("exchanges", {}).items():
        ex_ticker_defs = tickers_cfg.get("exchanges", {}).get(exchange, {}).get("tickers", [])
        if target_tickers:
            ex_ticker_defs = [t for t in ex_ticker_defs if t["ticker"] in target_tickers]

        sources = ex_data.get("sources", [])
        for src in sources:
            source_name = src.get("name", "unknown")
            actor_id = src.get("actor_id")
            input_tpl = src.get("input_template") or {}

            raw_items = ingester.fetch_source_items(
                source_name=source_name,
                actor_id=actor_id,
                input_template=input_tpl,
                exchange=exchange,
            )

            for item in raw_items:
                raw_text = (
                    item.get("text")
                    or item.get("title")
                    or item.get("content")
                    or str(item)
                )
                matched_tickers = match_post_to_tickers(raw_text, ex_ticker_defs)

                if matched_tickers:
                    for tkr in matched_tickers:
                        norm = ingester.normalize_post(item, exchange, source_name, tkr)
                        all_normalized_posts.append(norm)
                elif ex_ticker_defs:
                    # If specific ticker matching didn't trigger, associate with first candidate for completeness
                    first_tkr = ex_ticker_defs[0]["ticker"]
                    norm = ingester.normalize_post(item, exchange, source_name, first_tkr)
                    all_normalized_posts.append(norm)

    if all_normalized_posts:
        posts_df = pd.DataFrame(all_normalized_posts)
    else:
        logger.warning("No social posts were ingested from configured Apify actors.")
        posts_df = pd.DataFrame(columns=POSTS_COLUMNS)

    # Ensure schema types
    posts_df = posts_df[POSTS_COLUMNS] if not posts_df.empty else pd.DataFrame(columns=POSTS_COLUMNS)

    proc_dir = Path(settings_cfg.get("storage", {}).get("processed_dir", "data/processed"))
    save_parquet(posts_df, proc_dir / "posts.parquet")

    return posts_df


if __name__ == "__main__":
    setup_logging()
    df = ingest_social_pipeline()
    print(f"Total posts normalized: {len(df)}")
