"""Utility functions for sentiment-price-agent.

Includes configuration loaders, UTC timestamp normalization, API result caching,
and logging setup.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv(override=False)

logger = logging.getLogger("sentiment_price_agent")


def setup_logging(level: Optional[str] = None) -> logging.Logger:
    """Configure unified logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR). Defaults to LOG_LEVEL env var or INFO.

    Returns:
        Configured logger instance.
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    numeric_level = getattr(logging, level, logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    return logger


def get_env_var(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Retrieve an environment variable.

    Args:
        name: Name of the environment variable.
        default: Default value if not set.
        required: If True, raise ValueError when not found or empty.

    Returns:
        Value of the environment variable.

    Raises:
        ValueError: If required is True and the variable is missing/empty.
    """
    val = os.getenv(name, default)
    if required and (val is None or val.strip() == ""):
        raise ValueError(f"Required environment variable '{name}' is missing or empty.")
    return val


def load_yaml(file_path: Union[str, Path]) -> dict[str, Any]:
    """Load and parse a YAML file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Dictionary parsed from YAML.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid YAML.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data or {}


def load_tickers_config(config_path: Union[str, Path] = "config/tickers.yaml") -> dict[str, Any]:
    """Load tickers configuration."""
    return load_yaml(config_path)


def load_actors_config(config_path: Union[str, Path] = "config/actors.yaml") -> dict[str, Any]:
    """Load actors configuration."""
    return load_yaml(config_path)


def load_settings_config(config_path: Union[str, Path] = "config/settings.yaml") -> dict[str, Any]:
    """Load general application settings."""
    return load_yaml(config_path)


def normalize_to_utc(
    ts: Union[str, int, float, datetime, pd.Timestamp],
    default_tz: Optional[str] = None,
) -> pd.Timestamp:
    """Normalize any timestamp input to a timezone-aware UTC pandas.Timestamp.

    Handles Unix timestamps (seconds or milliseconds), ISO-8601 strings,
    naive datetimes, and pandas Timestamps.

    Args:
        ts: Input timestamp in various formats.
        default_tz: Timezone to assume if input is naive (e.g. 'Asia/Hong_Kong').
                    If None, UTC is assumed.

    Returns:
        UTC pandas.Timestamp.
    """
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return pd.Timestamp.now(tz=timezone.utc)

    # Numeric timestamp (seconds or milliseconds)
    if isinstance(ts, (int, float)):
        # If timestamp is likely milliseconds (e.g., > 1e11)
        unit = "ms" if ts > 1e11 else "s"
        return pd.to_datetime(ts, unit=unit, utc=True)

    # String, datetime, or Timestamp
    parsed = pd.to_datetime(ts)
    if parsed.tzinfo is None:
        if default_tz:
            parsed = parsed.tz_localize(default_tz).tz_convert(timezone.utc)
        else:
            parsed = parsed.tz_localize(timezone.utc)
    else:
        parsed = parsed.tz_convert(timezone.utc)

    return parsed


def get_cache_path(category: str, key: str, cache_dir: Union[str, Path] = "data/cache") -> Path:
    """Generate a file path for a cached item.

    Args:
        category: Subfolder for the cache (e.g. 'apify', 'exa', 'prices').
        key: Unique string identifier for the cache entry.
        cache_dir: Root cache directory.

    Returns:
        Path object pointing to cache JSON file.
    """
    key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    folder = Path(cache_dir) / category
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{key_hash}.json"


def read_cache(
    category: str,
    key: str,
    cache_dir: Union[str, Path] = "data/cache",
    max_age_seconds: Optional[int] = None,
) -> Optional[Any]:
    """Read cached data if available and fresh.

    Args:
        category: Cache category subfolder.
        key: Cache key.
        cache_dir: Root cache directory.
        max_age_seconds: Maximum allowed age in seconds before invalidation.

    Returns:
        Cached payload or None if missing/stale.
    """
    cache_file = get_cache_path(category, key, cache_dir)
    if not cache_file.is_file():
        return None

    try:
        stat = cache_file.stat()
        if max_age_seconds is not None:
            age = (datetime.now(timezone.utc).timestamp()) - stat.st_mtime
            if age > max_age_seconds:
                return None

        with cache_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to read cache file %s: %s", cache_file, e)
        return None


def write_cache(
    category: str,
    key: str,
    data: Any,
    cache_dir: Union[str, Path] = "data/cache",
) -> None:
    """Write data to cache.

    Args:
        category: Cache category subfolder.
        key: Cache key.
        data: JSON-serializable data.
        cache_dir: Root cache directory.
    """
    cache_file = get_cache_path(category, key, cache_dir)
    try:
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        logger.warning("Failed to write cache file %s: %s", cache_file, e)


def save_parquet(df: pd.DataFrame, file_path: Union[str, Path]) -> Path:
    """Save DataFrame to Parquet, creating parent directories if needed.

    Args:
        df: DataFrame to save.
        file_path: Destination path.

    Returns:
        Path of saved file.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
    logger.info("Saved %d rows to %s", len(df), path)
    return path


def load_parquet(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load DataFrame from Parquet file.

    Args:
        file_path: Path to Parquet file.

    Returns:
        pd.DataFrame or empty DataFrame if file not found.
    """
    path = Path(file_path)
    if not path.is_file():
        logger.warning("Parquet file not found: %s", path)
        return pd.DataFrame()
    return pd.read_parquet(path, engine="pyarrow")
