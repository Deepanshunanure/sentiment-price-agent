"""Feature engineering module.

Calculates hourly per-ticker features: mention counts, engagement-weighted sentiment,
bull/bear ratios, news sentiment, sentiment change, lagged returns, and volume.
(Full implementation in Stage 3).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger("sentiment_price_agent.features")


def build_hourly_features(
    prices_df: pd.DataFrame,
    posts_df: pd.DataFrame,
    news_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build hourly feature matrix for next-bar return modeling."""
    raise NotImplementedError("Implemented in Stage 3")
