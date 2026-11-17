"""Support and resistance level detection module.

Computes technical support/resistance from rolling highs/lows and pivot points
(Standard and Fibonacci), and merges them with crowd-mentioned price levels
extracted from social discussions.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("sentiment_price_agent.levels")


def calculate_pivot_points(
    high: float,
    low: float,
    close: float,
) -> dict[str, float]:
    """Calculate standard and Fibonacci pivot points from reference OHLC.

    Args:
        high: Period high.
        low: Period low.
        close: Period close.

    Returns:
        Dictionary with pivot point, support, and resistance levels.
    """
    pivot = (high + low + close) / 3.0
    rng = high - low

    # Standard pivots
    r1_std = 2.0 * pivot - low
    s1_std = 2.0 * pivot - high
    r2_std = pivot + rng
    s2_std = pivot - rng

    # Fibonacci pivots
    r1_fib = pivot + 0.382 * rng
    r2_fib = pivot + 0.618 * rng
    r3_fib = pivot + 1.000 * rng
    s1_fib = pivot - 0.382 * rng
    s2_fib = pivot - 0.618 * rng
    s3_fib = pivot - 1.000 * rng

    return {
        "pivot": round(pivot, 2),
        "r1_standard": round(r1_std, 2),
        "s1_standard": round(s1_std, 2),
        "r2_standard": round(r2_std, 2),
        "s2_standard": round(s2_std, 2),
        "r1_fib": round(r1_fib, 2),
        "r2_fib": round(r2_fib, 2),
        "r3_fib": round(r3_fib, 2),
        "s1_fib": round(s1_fib, 2),
        "s2_fib": round(s2_fib, 2),
        "s3_fib": round(s3_fib, 2),
    }


def cluster_price_levels(
    levels: list[float],
    current_price: float,
    tolerance_pct: float = 0.015,
) -> list[float]:
    """Filter outlier levels and cluster proximate price points.

    Args:
        levels: List of candidate price levels.
        current_price: Current market price.
        tolerance_pct: Relative percentage threshold for grouping (default 1.5%).

    Returns:
        Deduplicated, clustered list of price levels sorted ascending.
    """
    if not levels or current_price <= 0:
        return []

    # Filter levels within plausible range [0.5 * current_price, 1.8 * current_price]
    valid_levels = [
        lvl for lvl in levels
        if isinstance(lvl, (int, float)) and not np.isnan(lvl)
        and (0.5 * current_price <= lvl <= 1.8 * current_price)
    ]
    if not valid_levels:
        return []

    valid_levels.sort()
    clusters: list[list[float]] = []

    for lvl in valid_levels:
        if not clusters:
            clusters.append([lvl])
        else:
            prev_cluster = clusters[-1]
            cluster_avg = sum(prev_cluster) / len(prev_cluster)
            if abs(lvl - cluster_avg) / cluster_avg <= tolerance_pct:
                prev_cluster.append(lvl)
            else:
                clusters.append([lvl])

    # Return cluster means rounded
    return [round(float(np.mean(c)), 2) for c in clusters]


def calculate_support_resistance(
    df_prices: pd.DataFrame,
    crowd_levels: Optional[list[float]] = None,
    lookback_bars: int = 40,
) -> dict[str, Any]:
    """Calculate merged technical and crowd support/resistance levels.

    Args:
        df_prices: DataFrame of price bars with columns 'high', 'low', 'close'.
        crowd_levels: Optional price levels extracted from social posts.
        lookback_bars: Number of past bars to consider for rolling extremes.

    Returns:
        Dictionary containing current_price, primary_support, primary_resistance,
        support_levels, resistance_levels, and pivot points.
    """
    if df_prices.empty or "close" not in df_prices.columns:
        return {
            "current_price": 0.0,
            "primary_support": 0.0,
            "primary_resistance": 0.0,
            "support_levels": [],
            "resistance_levels": [],
            "pivot_point": 0.0,
            "crowd_levels": [],
        }

    current_price = float(df_prices["close"].iloc[-1])
    recent_bars = df_prices.tail(lookback_bars)

    period_high = float(recent_bars["high"].max())
    period_low = float(recent_bars["low"].min())
    period_close = current_price

    # 1. Pivot calculations
    pivots = calculate_pivot_points(period_high, period_low, period_close)

    # 2. Rolling technical swing levels
    rolling_lows = [float(recent_bars["low"].min()), float(recent_bars["low"].quantile(0.25))]
    rolling_highs = [float(recent_bars["high"].max()), float(recent_bars["high"].quantile(0.75))]

    # 3. Gather candidate levels
    technical_levels = [
        pivots["s1_standard"],
        pivots["s2_standard"],
        pivots["s1_fib"],
        pivots["s2_fib"],
        pivots["r1_standard"],
        pivots["r2_standard"],
        pivots["r1_fib"],
        pivots["r2_fib"],
        *rolling_lows,
        *rolling_highs,
    ]

    all_candidates = technical_levels.copy()
    valid_crowd = cluster_price_levels(crowd_levels or [], current_price)
    all_candidates.extend(valid_crowd)

    # Separate into support (below current price) and resistance (above current price)
    raw_support = [lvl for lvl in all_candidates if lvl < current_price * 0.999]
    raw_resistance = [lvl for lvl in all_candidates if lvl > current_price * 1.001]

    clustered_support = cluster_price_levels(raw_support, current_price)
    clustered_resistance = cluster_price_levels(raw_resistance, current_price)

    # Primary support: highest level below current price
    primary_support = max(clustered_support) if clustered_support else round(period_low, 2)
    # Primary resistance: lowest level above current price
    primary_resistance = min(clustered_resistance) if clustered_resistance else round(period_high, 2)

    return {
        "current_price": round(current_price, 2),
        "primary_support": round(primary_support, 2),
        "primary_resistance": round(primary_resistance, 2),
        "support_levels": clustered_support,
        "resistance_levels": clustered_resistance,
        "pivot_point": pivots["pivot"],
        "crowd_levels": valid_crowd,
        "period_high": round(period_high, 2),
        "period_low": round(period_low, 2),
    }
