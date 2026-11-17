"""Integration test for end-to-end components and artifact generation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import build_hourly_features, build_ticker_summary_table
from src.model import train_and_validate_exchange_models
from src.report import generate_html_report


def test_pipeline_integration(tmp_path: Path):
    """Test full flow: features -> model validation -> report generation on slice."""
    # 1. Create minimal synthetic hourly prices
    dates = pd.date_range("2026-11-01", periods=40, freq="1h", tz="UTC")
    prices_df = pd.DataFrame(
        {
            "ts_utc": dates,
            "exchange": ["HKEX"] * 40,
            "ticker": ["0700.HK"] * 40,
            "open": np.linspace(370, 380, 40),
            "high": np.linspace(372, 382, 40),
            "low": np.linspace(368, 378, 40),
            "close": np.linspace(371, 381, 40),
            "volume": [100000] * 40,
        }
    )
    prices_file = tmp_path / "prices_hourly.parquet"
    prices_df.to_parquet(prices_file, engine="pyarrow", index=False)

    # 2. Minimal scored data
    scored_df = pd.DataFrame(
        {
            "id": ["p1", "p2"],
            "item_type": ["post", "news"],
            "ts_utc": [dates[5], dates[10]],
            "exchange": ["HKEX", "HKEX"],
            "source": ["lihkg", "exa"],
            "ticker": ["0700.HK", "0700.HK"],
            "text": ["Great rally", "Strong earnings"],
            "likes": [10, 0],
            "reposts": [2, 0],
            "views": [500, 0],
            "author_followers": [1000, 0],
            "sentiment": [0.6, 0.4],
            "direction": ["bullish", "bullish"],
            "confidence": [0.8, 0.9],
            "position": ["long", "none"],
            "support_resistance_levels": [[370.0, 385.0], [380.0]],
            "reasoning": ["Positive", "Strong"],
        }
    )
    scored_file = tmp_path / "scored.parquet"
    scored_df.to_parquet(scored_file, engine="pyarrow", index=False)

    # 3. Build features
    features_file = tmp_path / "features_hourly.parquet"
    feat_df = build_hourly_features(
        prices_path=prices_file,
        scored_path=scored_file,
        output_path=features_file,
    )
    assert not feat_df.empty
    assert "target_return" in feat_df.columns
    assert "engagement_weighted_sentiment" in feat_df.columns

    # 4. Run walk-forward model validation
    preds_df, metrics = train_and_validate_exchange_models(feat_df, exchange="HKEX", n_splits=2)
    assert not preds_df.empty
    assert "actual_return" in preds_df.columns
    assert "lgbm_pred" in preds_df.columns
    assert "overall_lgbm" in metrics

    # 5. Build summary table and report
    summary_file = tmp_path / "ticker_summary.parquet"
    summary_df = build_ticker_summary_table(
        prices_path=prices_file,
        scored_path=scored_file,
        output_path=summary_file,
    )
    assert not summary_df.empty

    preds_file = tmp_path / "predictions.parquet"
    preds_df.to_parquet(preds_file, engine="pyarrow", index=False)

    metrics_file = tmp_path / "metrics.json"
    import json
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({"HKEX": metrics}, f)

    report_file = tmp_path / "report.html"
    out_rep = generate_html_report(
        summary_path=summary_file,
        features_path=features_file,
        predictions_path=preds_file,
        news_path=scored_file,
        metrics_json_path=metrics_file,
        output_path=report_file,
    )
    assert out_rep.is_file()
    assert out_rep.stat().st_size > 5000
