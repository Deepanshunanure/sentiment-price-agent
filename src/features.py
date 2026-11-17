"""Feature engineering and per-ticker extraction module.

Constructs hourly aligned financial and sentiment features per ticker:
- mention count
- engagement-weighted sentiment (weighted by likes, reposts, views, followers)
- bull/bear ratio
- news sentiment
- sentiment change and exponential decay
- lagged returns, volume ratios, and next-bar targets (n+1)
- builds per-ticker summary table with technical and crowd support/resistance levels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from src.levels import calculate_support_resistance
from src.utils import (
    load_parquet,
    load_settings_config,
    load_tickers_config,
    normalize_to_utc,
    save_parquet,
    setup_logging,
)

logger = logging.getLogger("sentiment_price_agent.features")


def compute_engagement_weight(
    likes: int,
    reposts: int,
    views: int,
    followers: int,
) -> float:
    """Calculate engagement weight for a social post.

    Weight formula: 1 + ln(1 + likes) + 0.5*ln(1 + reposts) + 0.1*ln(1 + views) + 0.1*ln(1 + followers)
    """
    w = 1.0
    w += np.log1p(max(0, likes))
    w += 0.5 * np.log1p(max(0, reposts))
    w += 0.1 * np.log1p(max(0, views))
    w += 0.1 * np.log1p(max(0, followers))
    return float(w)


def build_hourly_sentiment_aggregates(
    df_scored: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate scored items into hourly bins per ticker.

    Calculates: mention_count, engagement_weighted_sentiment, bull_bear_ratio, news_sentiment.
    """
    if df_scored.empty:
        return pd.DataFrame(
            columns=[
                "ts_hour",
                "ticker",
                "mention_count",
                "engagement_weighted_sentiment",
                "bull_bear_ratio",
                "news_sentiment",
                "crowd_levels",
            ]
        )

    df = df_scored.copy()
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["ts_hour"] = df["ts_utc"].dt.floor("h")

    # Compute engagement weights
    df["weight"] = df.apply(
        lambda r: compute_engagement_weight(
            likes=int(r.get("likes") or 0),
            reposts=int(r.get("reposts") or 0),
            views=int(r.get("views") or 0),
            followers=int(r.get("author_followers") or 0),
        ),
        axis=1,
    )
    df["weighted_sentiment"] = df["sentiment"] * df["weight"]

    grouped = df.groupby(["ts_hour", "ticker"])
    records = []

    for (hour, ticker), grp in grouped:
        posts_grp = grp[grp["item_type"] == "post"]
        news_grp = grp[grp["item_type"] == "news"]

        mention_count = len(posts_grp)
        if mention_count > 0:
            total_weight = posts_grp["weight"].sum()
            eng_sent = posts_grp["weighted_sentiment"].sum() / (total_weight if total_weight > 0 else 1.0)
            bulls = (posts_grp["direction"] == "bullish").sum()
            bears = (posts_grp["direction"] == "bearish").sum()
            bb_ratio = float(bulls) / float(bears + 1.0)
        else:
            eng_sent = 0.0
            bb_ratio = 1.0

        news_sent = float(news_grp["sentiment"].mean()) if len(news_grp) > 0 else 0.0

        # Collect crowd levels stated
        levels: list[float] = []
        for l_list in grp["support_resistance_levels"]:
            if isinstance(l_list, (list, np.ndarray)):
                for val in l_list:
                    try:
                        levels.append(float(val))
                    except (ValueError, TypeError):
                        pass

        records.append(
            {
                "ts_hour": hour,
                "ticker": ticker,
                "mention_count": int(mention_count),
                "engagement_weighted_sentiment": round(float(eng_sent), 4),
                "bull_bear_ratio": round(float(bb_ratio), 4),
                "news_sentiment": round(float(news_sent), 4),
                "crowd_levels": levels,
            }
        )

    return pd.DataFrame(records)


def build_hourly_features(
    prices_path: Union[str, Path] = "data/processed/prices_hourly.parquet",
    scored_path: Union[str, Path] = "data/processed/scored.parquet",
    output_path: Union[str, Path] = "data/processed/features_hourly.parquet",
) -> pd.DataFrame:
    """Build hourly feature matrix aligned with price bars and sentiment metrics.

    Features generated per ticker per hour:
    - mention_count
    - engagement_weighted_sentiment
    - bull_bear_ratio
    - news_sentiment
    - sentiment_change
    - sentiment_ema (exponential decay smoothing)
    - return_1h, return_lag1, return_lag2, return_lag3
    - volume, volume_log, volume_ratio
    - price_range (high - low) / close
    - target_return (next-bar return n+1)
    - target_direction (sign of next-bar return: 1 if positive, 0 otherwise)

    Returns:
        DataFrame with complete feature set.
    """
    df_prices = load_parquet(prices_path)
    df_scored = load_parquet(scored_path)

    if df_prices.empty:
        logger.error("Price dataset %s is empty; cannot construct feature matrix.", prices_path)
        return pd.DataFrame()

    df_prices["ts_utc"] = pd.to_datetime(df_prices["ts_utc"], utc=True)
    df_prices["ts_hour"] = df_prices["ts_utc"].dt.floor("h")

    # Aggregate sentiment
    df_sent_hourly = build_hourly_sentiment_aggregates(df_scored)

    feature_dfs: list[pd.DataFrame] = []

    for ticker, grp in df_prices.groupby("ticker"):
        grp = grp.sort_values("ts_utc").copy()

        # Merge with sentiment on hour
        if not df_sent_hourly.empty:
            ticker_sent = df_sent_hourly[df_sent_hourly["ticker"] == ticker]
            merged = pd.merge(grp, ticker_sent, on=["ts_hour", "ticker"], how="left")
        else:
            merged = grp.copy()
            merged["mention_count"] = 0
            merged["engagement_weighted_sentiment"] = 0.0
            merged["bull_bear_ratio"] = 1.0
            merged["news_sentiment"] = 0.0
            merged["crowd_levels"] = [[] for _ in range(len(merged))]

        # Fill sentiment nulls
        merged["mention_count"] = merged["mention_count"].fillna(0).astype(int)
        merged["engagement_weighted_sentiment"] = merged["engagement_weighted_sentiment"].fillna(0.0)
        merged["bull_bear_ratio"] = merged["bull_bear_ratio"].fillna(1.0)
        merged["news_sentiment"] = merged["news_sentiment"].fillna(0.0)

        # Sentiment dynamics
        merged["sentiment_change"] = merged["engagement_weighted_sentiment"].diff().fillna(0.0)
        merged["sentiment_ema"] = (
            merged["engagement_weighted_sentiment"]
            .ewm(halflife=6, min_periods=1)
            .mean()
        )
        merged["sentiment_ema_change"] = merged["sentiment_ema"].diff().fillna(0.0)

        # Price return features
        merged["return_1h"] = merged["close"].pct_change(1).fillna(0.0)
        merged["return_lag1"] = merged["return_1h"].shift(1).fillna(0.0)
        merged["return_lag2"] = merged["return_1h"].shift(2).fillna(0.0)
        merged["return_lag3"] = merged["return_1h"].shift(3).fillna(0.0)

        # Volume and range features
        merged["volume_log"] = np.log1p(merged["volume"].clip(lower=0))
        roll_vol = merged["volume"].rolling(24, min_periods=1).mean().replace(0, 1.0)
        merged["volume_ratio"] = merged["volume"] / roll_vol
        merged["price_range"] = (merged["high"] - merged["low"]) / merged["close"]

        # Target definitions for Stage 4 (next-bar n+1 return)
        merged["target_return"] = merged["close"].pct_change(1).shift(-1)
        merged["target_price"] = merged["close"].shift(-1)
        merged["target_direction"] = np.where(merged["target_return"] > 0, 1, 0)

        feature_dfs.append(merged)

    full_features = pd.concat(feature_dfs, ignore_index=True)
    save_parquet(full_features, output_path)
    logger.info("Saved hourly features matrix with %d rows and %d columns to %s", len(full_features), len(full_features.columns), output_path)
    return full_features


def build_ticker_summary_table(
    prices_path: Union[str, Path] = "data/processed/prices_hourly.parquet",
    scored_path: Union[str, Path] = "data/processed/scored.parquet",
    tickers_config_path: Union[str, Path] = "config/tickers.yaml",
    output_path: Union[str, Path] = "data/processed/ticker_summary.parquet",
) -> pd.DataFrame:
    """Build per-ticker summary table: ticker, asset name, current price,

    sentiment, direction, position, support, resistance, mention count.
    """
    tickers_cfg = load_tickers_config(tickers_config_path)
    df_prices = load_parquet(prices_path)
    df_scored = load_parquet(scored_path)

    ticker_meta: dict[str, dict[str, Any]] = {}
    for ex_name, ex_data in tickers_cfg.get("exchanges", {}).items():
        for t in ex_data.get("tickers", []):
            ticker_meta[t["ticker"]] = {
                "name": t.get("name", t["ticker"]),
                "exchange": ex_name,
            }

    summary_rows = []

    # Iterate over all tickers present in prices or config
    all_tickers = sorted(list(set(list(df_prices["ticker"].unique()) + list(ticker_meta.keys()))))

    for tkr in all_tickers:
        meta = ticker_meta.get(tkr, {"name": tkr, "exchange": "UNKNOWN"})
        tkr_prices = df_prices[df_prices["ticker"] == tkr].sort_values("ts_utc") if not df_prices.empty else pd.DataFrame()
        tkr_scored = df_scored[df_scored["ticker"] == tkr] if not df_scored.empty else pd.DataFrame()

        current_price = float(tkr_prices["close"].iloc[-1]) if not tkr_prices.empty else 0.0

        # Crowd levels
        crowd_levels = []
        if not tkr_scored.empty:
            for l_list in tkr_scored["support_resistance_levels"]:
                if isinstance(l_list, (list, np.ndarray)):
                    for v in l_list:
                        try:
                            crowd_levels.append(float(v))
                        except (ValueError, TypeError):
                            pass

        # Support & Resistance levels from levels.py
        levels_data = calculate_support_resistance(tkr_prices, crowd_levels=crowd_levels)

        # Sentiment summary
        mention_count = len(tkr_scored[tkr_scored["item_type"] == "post"]) if not tkr_scored.empty else 0
        mean_sentiment = float(tkr_scored["sentiment"].mean()) if not tkr_scored.empty else 0.0

        # Mode direction and position
        if not tkr_scored.empty:
            dir_mode = tkr_scored["direction"].mode()
            direction = str(dir_mode.iloc[0]) if not dir_mode.empty else "neutral"
            pos_non_none = tkr_scored[tkr_scored["position"] != "none"]["position"]
            if not pos_non_none.empty:
                position = str(pos_non_none.mode().iloc[0])
            else:
                position = "none"
        else:
            direction = "neutral"
            position = "none"

        summary_rows.append(
            {
                "ticker": tkr,
                "asset_name": meta["name"],
                "exchange": meta["exchange"],
                "current_price": round(current_price, 2),
                "sentiment": round(mean_sentiment, 3),
                "direction": direction,
                "position": position,
                "support": levels_data["primary_support"],
                "resistance": levels_data["primary_resistance"],
                "mention_count": int(mention_count),
                "pivot_point": levels_data["pivot_point"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    save_parquet(summary_df, output_path)
    logger.info("Saved per-ticker summary table with %d tickers to %s", len(summary_df), output_path)
    return summary_df


def run_stage3_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute Stage 3 end-to-end feature extraction and summary table build."""
    features_df = build_hourly_features()
    summary_df = build_ticker_summary_table()
    return features_df, summary_df


if __name__ == "__main__":
    setup_logging()
    feat_df, sum_df = run_stage3_pipeline()
    print("\n--- PER-TICKER SUMMARY TABLE ---")
    print(sum_df.to_string(index=False))
