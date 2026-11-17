"""Unit tests for model walk-forward validation and anti-leakage splitter."""

import numpy as np
import pandas as pd
import pytest

from src.model import (
    TimeOrderedWalkForwardSplitter,
    compute_directional_accuracy,
    evaluate_predictions,
)


def test_time_ordered_walk_forward_splitter():
    """Verify that train set strictly precedes test set in time with zero leakage."""
    dates = pd.date_range("2026-11-01", periods=50, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "ts_utc": dates,
            "feature": np.random.randn(50),
            "target_return": np.random.randn(50) * 0.01,
        }
    )

    splitter = TimeOrderedWalkForwardSplitter(n_splits=3, min_train_ratio=0.5)
    splits = list(splitter.split(df, time_col="ts_utc"))

    assert len(splits) == 3

    for fold_num, train_idx, test_idx in splits:
        train_times = df.iloc[train_idx]["ts_utc"]
        test_times = df.iloc[test_idx]["ts_utc"]

        # Strict chronological order check: max(train) < min(test)
        assert train_times.max() < test_times.min()

        # No overlapping indices
        assert len(set(train_idx).intersection(set(test_idx))) == 0


def test_compute_directional_accuracy():
    """Verify directional sign accuracy calculation."""
    y_true = np.array([0.02, -0.01, 0.03, -0.04])
    y_pred = np.array([0.01, -0.02, -0.01, -0.05])  # 3 match, 1 mismatch

    acc = compute_directional_accuracy(y_true, y_pred)
    assert acc == 0.75


def test_evaluate_predictions_metrics():
    """Verify calculation of MAE, RMSE, and accuracy."""
    y_true = np.array([0.01, -0.01])
    y_pred = np.array([0.01, -0.01])

    metrics = evaluate_predictions(y_true, y_pred)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["directional_accuracy"] == 1.0
