"""Prediction and walk-forward validation module.

Predicts next-bar (n+1) returns using pooled exchange data.
Implements strictly time-ordered walk-forward validation with zero leakage.
Evaluates Naive Baseline, Ridge Regression, and LightGBM models.
Saves predictions and performance metrics honestly against the baseline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.utils import (
    load_parquet,
    load_settings_config,
    save_parquet,
    setup_logging,
)

logger = logging.getLogger("sentiment_price_agent.model")

FEATURE_COLUMNS = [
    # Price technicals
    "return_1h",
    "return_lag1",
    "return_lag2",
    "return_lag3",
    "volume_log",
    "volume_ratio",
    "price_range",
    # Sentiment signals
    "mention_count",
    "engagement_weighted_sentiment",
    "bull_bear_ratio",
    "news_sentiment",
    "sentiment_change",
    "sentiment_ema",
    "sentiment_ema_change",
]


class TimeOrderedWalkForwardSplitter:
    """Strict time-ordered walk-forward splitter on timestamps.

    Ensures zero lookahead bias by partitioning unique timestamps chronologically
    so that training sets strictly precede testing sets across all pooled tickers.
    """

    def __init__(self, n_splits: int = 4, min_train_ratio: float = 0.5) -> None:
        self.n_splits = n_splits
        self.min_train_ratio = min_train_ratio

    def split(self, df: pd.DataFrame, time_col: str = "ts_utc"):
        """Yield (train_indices, test_indices) for each walk-forward fold."""
        unique_times = np.sort(df[time_col].unique())
        n_times = len(unique_times)
        min_train_idx = int(n_times * self.min_train_ratio)

        test_times = unique_times[min_train_idx:]
        test_chunk_size = max(1, len(test_times) // self.n_splits)

        for fold in range(self.n_splits):
            start_test_idx = min_train_idx + fold * test_chunk_size
            end_test_idx = (
                min_train_idx + (fold + 1) * test_chunk_size
                if fold < self.n_splits - 1
                else n_times
            )

            train_timestamps = set(unique_times[:start_test_idx])
            test_timestamps = set(unique_times[start_test_idx:end_test_idx])

            train_idx = df.index[df[time_col].isin(train_timestamps)].to_numpy()
            test_idx = df.index[df[time_col].isin(test_timestamps)].to_numpy()

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield fold + 1, train_idx, test_idx


def compute_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute directional sign accuracy: fraction of bars where sign(pred) matches sign(actual)."""
    # Exclude exact zeros or treat zero as positive
    signs_match = np.sign(y_pred) == np.sign(y_true)
    return float(np.mean(signs_match))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate MAE, RMSE, and Directional Accuracy."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    dir_acc = compute_directional_accuracy(y_true, y_pred)
    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "directional_accuracy": round(dir_acc, 4),
    }


def train_and_validate_exchange_models(
    df_features: pd.DataFrame,
    exchange: Optional[str] = None,
    n_splits: int = 4,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run walk-forward validation for a given exchange subset or pooled dataset.

    Args:
        df_features: Hourly feature dataset.
        exchange: 'HKEX', 'KRX', or None for overall pooled.
        n_splits: Number of walk-forward folds.

    Returns:
        Tuple of (predictions_dataframe, metrics_dictionary).
    """
    df = df_features.copy()
    if exchange:
        df = df[df["exchange"] == exchange].copy()

    # Drop bars with missing target (e.g. final bar)
    df = df.dropna(subset=["target_return"]).reset_index(drop=True)

    if len(df) < 30:
        logger.warning("Insufficient samples (%d) for exchange %s; returning empty results.", len(df), exchange)
        return pd.DataFrame(), {}

    splitter = TimeOrderedWalkForwardSplitter(n_splits=n_splits)

    fold_predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []

    for fold_num, train_idx, test_idx in splitter.split(df, time_col="ts_utc"):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx].copy()

        X_train = train_df[FEATURE_COLUMNS].fillna(0.0).to_numpy()
        y_train = train_df["target_return"].to_numpy()

        X_test = test_df[FEATURE_COLUMNS].fillna(0.0).to_numpy()
        y_test = test_df["target_return"].to_numpy()

        # 1. Naive Baseline (predicts 0.0 or train mean return)
        baseline_pred_val = float(np.mean(y_train))
        y_pred_naive = np.full_like(y_test, baseline_pred_val)

        # 2. Ridge Regression with StandardScaler fitted on train only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        ridge = Ridge(alpha=10.0, random_state=42)
        ridge.fit(X_train_scaled, y_train)
        y_pred_ridge = ridge.predict(X_test_scaled)

        # 3. LightGBM Regressor
        lgbm = lgb.LGBMRegressor(
            n_estimators=60,
            max_depth=3,
            learning_rate=0.03,
            min_child_samples=5,
            random_state=42,
            verbosity=-1,
        )
        lgbm.fit(X_train, y_train)
        y_pred_lgbm = lgbm.predict(X_test)

        # Record predictions
        test_df["fold"] = fold_num
        test_df["actual_return"] = y_test
        test_df["naive_pred"] = y_pred_naive
        test_df["ridge_pred"] = y_pred_ridge
        test_df["lgbm_pred"] = y_pred_lgbm

        # Directional prediction
        test_df["naive_dir_pred"] = np.where(y_pred_naive >= 0, 1, 0)
        test_df["ridge_dir_pred"] = np.where(y_pred_ridge >= 0, 1, 0)
        test_df["lgbm_dir_pred"] = np.where(y_pred_lgbm >= 0, 1, 0)

        fold_predictions.append(
            test_df[
                [
                    "ts_utc",
                    "exchange",
                    "ticker",
                    "fold",
                    "close",
                    "actual_return",
                    "target_direction",
                    "naive_pred",
                    "ridge_pred",
                    "lgbm_pred",
                    "naive_dir_pred",
                    "ridge_dir_pred",
                    "lgbm_dir_pred",
                ]
            ]
        )

        # Fold metrics
        m_naive = evaluate_predictions(y_test, y_pred_naive)
        m_ridge = evaluate_predictions(y_test, y_pred_ridge)
        m_lgbm = evaluate_predictions(y_test, y_pred_lgbm)

        fold_metrics.append(
            {
                "fold": fold_num,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "naive": m_naive,
                "ridge": m_ridge,
                "lgbm": m_lgbm,
            }
        )

    if not fold_predictions:
        return pd.DataFrame(), {}

    df_all_preds = pd.concat(fold_predictions, ignore_index=True)

    # Calculate overall aggregate metrics
    y_actual = df_all_preds["actual_return"].to_numpy()
    overall_naive = evaluate_predictions(y_actual, df_all_preds["naive_pred"].to_numpy())
    overall_ridge = evaluate_predictions(y_actual, df_all_preds["ridge_pred"].to_numpy())
    overall_lgbm = evaluate_predictions(y_actual, df_all_preds["lgbm_pred"].to_numpy())

    # Comparison vs Baseline
    def compute_vs_baseline(model_m: dict[str, float], base_m: dict[str, float]) -> dict[str, float]:
        mae_imp = ((base_m["mae"] - model_m["mae"]) / base_m["mae"]) * 100.0 if base_m["mae"] > 0 else 0.0
        rmse_imp = ((base_m["rmse"] - model_m["rmse"]) / base_m["rmse"]) * 100.0 if base_m["rmse"] > 0 else 0.0
        acc_diff = (model_m["directional_accuracy"] - base_m["directional_accuracy"]) * 100.0
        return {
            "mae_improvement_pct": round(mae_imp, 2),
            "rmse_improvement_pct": round(rmse_imp, 2),
            "directional_accuracy_diff_pct": round(acc_diff, 2),
            "beats_baseline_mae": model_m["mae"] < base_m["mae"],
            "beats_baseline_accuracy": model_m["directional_accuracy"] > base_m["directional_accuracy"],
        }

    summary = {
        "exchange": exchange or "POOLED_ALL",
        "total_test_samples": len(df_all_preds),
        "overall_naive": overall_naive,
        "overall_ridge": overall_ridge,
        "overall_lgbm": overall_lgbm,
        "ridge_vs_baseline": compute_vs_baseline(overall_ridge, overall_naive),
        "lgbm_vs_baseline": compute_vs_baseline(overall_lgbm, overall_naive),
        "fold_details": fold_metrics,
    }

    return df_all_preds, summary


def run_modeling_pipeline(
    features_path: Union[str, Path] = "data/processed/features_hourly.parquet",
    predictions_path: Union[str, Path] = "data/processed/predictions.parquet",
    metrics_json_path: Union[str, Path] = "reports/metrics.json",
    metrics_csv_path: Union[str, Path] = "reports/metrics.csv",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run full walk-forward prediction and validation pipeline across HKEX, KRX, and pooled datasets.

    Saves predictions to predictions.parquet and metrics to reports/.
    """
    df_features = load_parquet(features_path)
    if df_features.empty:
        raise ValueError(f"Features file {features_path} is empty or missing.")

    all_predictions_list = []
    final_metrics_report: dict[str, Any] = {}
    csv_rows = []

    # Run for HKEX, KRX, and POOLED
    scopes = ["HKEX", "KRX", None]

    for scope in scopes:
        scope_name = scope or "POOLED_ALL"
        logger.info("Running walk-forward validation for scope: %s", scope_name)
        preds, metrics = train_and_validate_exchange_models(df_features, exchange=scope, n_splits=4)

        if not preds.empty:
            all_predictions_list.append(preds)
            final_metrics_report[scope_name] = metrics

            # Build row for CSV
            for m_name in ["naive", "ridge", "lgbm"]:
                m_data = metrics.get(f"overall_{m_name}", {})
                vs_data = metrics.get(f"{m_name}_vs_baseline", {})
                csv_rows.append(
                    {
                        "scope": scope_name,
                        "model": m_name,
                        "mae": m_data.get("mae"),
                        "rmse": m_data.get("rmse"),
                        "directional_accuracy": m_data.get("directional_accuracy"),
                        "mae_improvement_pct": vs_data.get("mae_improvement_pct", 0.0),
                        "beats_baseline_mae": vs_data.get("beats_baseline_mae", False),
                        "beats_baseline_accuracy": vs_data.get("beats_baseline_accuracy", False),
                    }
                )

    # Save consolidated predictions (deduplicated by keeping pooled or per-exchange)
    consolidated_preds = all_predictions_list[0] if all_predictions_list else pd.DataFrame()
    if len(all_predictions_list) > 1:
        # Save exchange-specific predictions
        ex_preds = [p for p in all_predictions_list[:-1] if not p.empty]
        consolidated_preds = pd.concat(ex_preds, ignore_index=True) if ex_preds else all_predictions_list[-1]

    save_parquet(consolidated_preds, predictions_path)

    # Save JSON metrics
    metrics_path = Path(metrics_json_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(final_metrics_report, f, indent=2, ensure_ascii=False)
    logger.info("Saved validation metrics to %s", metrics_path)

    # Save CSV metrics
    df_metrics = pd.DataFrame(csv_rows)
    df_metrics.to_csv(metrics_csv_path, index=False)
    logger.info("Saved metrics CSV to %s", metrics_csv_path)

    return consolidated_preds, final_metrics_report


if __name__ == "__main__":
    setup_logging()
    preds, metrics = run_modeling_pipeline()
    print("\n" + "=" * 75)
    print("STAGE 4 WALK-FORWARD VALIDATION RESULTS")
    print("=" * 75)
    for scope, m in metrics.items():
        print(f"\n--- Scope: {scope} (Test samples: {m['total_test_samples']}) ---")
        print(f"Naive Baseline : MAE={m['overall_naive']['mae']:.6f} | RMSE={m['overall_naive']['rmse']:.6f} | DirAcc={m['overall_naive']['directional_accuracy']*100:.1f}%")
        print(f"Ridge Model    : MAE={m['overall_ridge']['mae']:.6f} | RMSE={m['overall_ridge']['rmse']:.6f} | DirAcc={m['overall_ridge']['directional_accuracy']*100:.1f}% | Beats Baseline: {m['ridge_vs_baseline']['beats_baseline_mae']}")
        print(f"LightGBM Model : MAE={m['overall_lgbm']['mae']:.6f} | RMSE={m['overall_lgbm']['rmse']:.6f} | DirAcc={m['overall_lgbm']['directional_accuracy']*100:.1f}% | Beats Baseline: {m['lgbm_vs_baseline']['beats_baseline_mae']}")
