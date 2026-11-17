"""Unit tests for levels.py and features.py."""

import numpy as np
import pandas as pd
import pytest

from src.features import compute_engagement_weight
from src.levels import (
    calculate_pivot_points,
    calculate_support_resistance,
    cluster_price_levels,
)


def test_calculate_pivot_points():
    """Verify standard and Fibonacci pivot point formulas."""
    high = 100.0
    low = 90.0
    close = 95.0
    pivots = calculate_pivot_points(high, low, close)

    # Pivot = (100 + 90 + 95) / 3 = 95.0
    assert pivots["pivot"] == 95.0
    # Standard R1 = 2*P - L = 2*95 - 90 = 100.0
    assert pivots["r1_standard"] == 100.0
    # Standard S1 = 2*P - H = 2*95 - 100 = 90.0
    assert pivots["s1_standard"] == 90.0

    # Fibonacci R1 = P + 0.382 * 10 = 98.82
    assert pivots["r1_fib"] == 98.82
    # Fibonacci S1 = P - 0.382 * 10 = 91.18
    assert pivots["s1_fib"] == 91.18


def test_cluster_price_levels():
    """Verify clustering of nearby price points and elimination of absurd outliers."""
    current_price = 100.0
    raw_levels = [98.5, 99.0, 99.2, 101.5, 102.0, 25.0, 500.0]  # 25 and 500 are outliers

    clustered = cluster_price_levels(raw_levels, current_price, tolerance_pct=0.015)
    assert 25.0 not in clustered
    assert 500.0 not in clustered
    # 98.5, 99.0, 99.2 should cluster together around ~98.9
    assert any(98.5 <= lvl <= 99.5 for lvl in clustered)


def test_support_resistance_merging():
    """Verify support is below current price and resistance is above current price."""
    prices = pd.DataFrame(
        {
            "high": [102.0, 105.0, 104.0],
            "low": [98.0, 99.0, 97.0],
            "close": [100.0, 103.0, 101.0],
        }
    )
    crowd = [98.0, 106.0]
    res = calculate_support_resistance(prices, crowd_levels=crowd)

    assert res["current_price"] == 101.0
    assert res["primary_support"] <= 101.0
    assert res["primary_resistance"] >= 101.0


def test_compute_engagement_weight():
    """Verify engagement weighting increases monotonically with interactions."""
    w0 = compute_engagement_weight(likes=0, reposts=0, views=0, followers=0)
    w_likes = compute_engagement_weight(likes=50, reposts=0, views=0, followers=0)
    w_viral = compute_engagement_weight(likes=1000, reposts=200, views=50000, followers=10000)

    assert w0 == 1.0
    assert w_likes > w0
    assert w_viral > w_likes
