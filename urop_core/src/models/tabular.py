"""Tabular ML models (Linear Regression, Random Forest, XGBoost) with validation tuning and timing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.models.metrics import regression_metrics


@dataclass
class TrainedTabularModel:
    model_name: str
    estimator: Any
    scaler: StandardScaler | None
    config: dict
    train_time_s: float
    infer_time_ms: float
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    clip_bounds: tuple[float, float]

    def predict(self, X: np.ndarray, already_scaled: bool = False) -> np.ndarray:
        Xin = X if (already_scaled or self.scaler is None) else self.scaler.transform(X)
        preds = self.estimator.predict(Xin)
        return np.clip(preds, self.clip_bounds[0], self.clip_bounds[1])


def train_tabular_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    clip_bounds: tuple[float, float] = (0.0, 1.0),
    seed: int = 42,
    already_scaled: bool = False,
    fast_mode: bool = False,
) -> dict[str, TrainedTabularModel]:
    """Train Linear Regression, Random Forest, and XGBoost with val-based hyperparameter selection."""
    scaler = None if already_scaled else StandardScaler().fit(X_train)
    Xtr = X_train if already_scaled else scaler.transform(X_train)
    Xva = X_val if already_scaled else scaler.transform(X_val)
    Xte = X_test if already_scaled else scaler.transform(X_test)

    trained: dict[str, TrainedTabularModel] = {}

    # 1. Linear Regression (Ridge baseline tuned over alpha on validation set)
    t0 = time.perf_counter()
    best_lr, best_lr_cfg, best_lr_rmse = None, {}, float("inf")
    for alpha in (1e-4, 1e-2, 0.1, 1.0, 10.0):
        est = Ridge(alpha=alpha)
        est.fit(Xtr, y_train)
        pred_v = np.clip(est.predict(Xva), *clip_bounds)
        rmse_v = regression_metrics(y_val, pred_v)["rmse"]
        if rmse_v < best_lr_rmse:
            best_lr_rmse = rmse_v
            best_lr = est
            best_lr_cfg = {"alpha": alpha}
    t_train = time.perf_counter() - t0
    ti0 = time.perf_counter()
    pred_te = np.clip(best_lr.predict(Xte), *clip_bounds)
    t_inf_ms = (time.perf_counter() - ti0) * 1000.0
    trained["LinearRegression"] = TrainedTabularModel(
        model_name="LinearRegression",
        estimator=best_lr,
        scaler=scaler,
        config=best_lr_cfg,
        train_time_s=round(t_train, 4),
        infer_time_ms=round(t_inf_ms, 3),
        val_metrics=regression_metrics(y_val, np.clip(best_lr.predict(Xva), *clip_bounds)),
        test_metrics=regression_metrics(y_test, pred_te),
        clip_bounds=clip_bounds,
    )

    # 2. Random Forest (tuned on validation set)
    t0 = time.perf_counter()
    rf_grid = (
        [{"n_estimators": 60, "max_depth": 10, "min_samples_leaf": 3}]
        if fast_mode
        else [
            {"n_estimators": 120, "max_depth": 10, "min_samples_leaf": 3},
            {"n_estimators": 160, "max_depth": 16, "min_samples_leaf": 2},
        ]
    )
    best_rf, best_rf_cfg, best_rf_rmse = None, {}, float("inf")
    for params in rf_grid:
        est = RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
        est.fit(Xtr, y_train)
        pred_v = np.clip(est.predict(Xva), *clip_bounds)
        rmse_v = regression_metrics(y_val, pred_v)["rmse"]
        if rmse_v < best_rf_rmse:
            best_rf_rmse = rmse_v
            best_rf = est
            best_rf_cfg = params
    t_train = time.perf_counter() - t0
    ti0 = time.perf_counter()
    pred_te = np.clip(best_rf.predict(Xte), *clip_bounds)
    t_inf_ms = (time.perf_counter() - ti0) * 1000.0
    trained["RandomForest"] = TrainedTabularModel(
        model_name="RandomForest",
        estimator=best_rf,
        scaler=scaler,
        config=best_rf_cfg,
        train_time_s=round(t_train, 4),
        infer_time_ms=round(t_inf_ms, 3),
        val_metrics=regression_metrics(y_val, np.clip(best_rf.predict(Xva), *clip_bounds)),
        test_metrics=regression_metrics(y_test, pred_te),
        clip_bounds=clip_bounds,
    )

    # 3. XGBoost (tuned on validation set)
    t0 = time.perf_counter()
    xgb_grid = (
        [{"n_estimators": 100, "max_depth": 4, "learning_rate": 0.08, "subsample": 0.85, "colsample_bytree": 0.85}]
        if fast_mode
        else [
            {"n_estimators": 180, "max_depth": 4, "learning_rate": 0.06, "subsample": 0.85, "colsample_bytree": 0.85},
            {"n_estimators": 240, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85},
        ]
    )
    best_xgb, best_xgb_cfg, best_xgb_rmse = None, {}, float("inf")
    for params in xgb_grid:
        est = XGBRegressor(
            objective="reg:squarederror",
            reg_lambda=1.5,
            random_state=seed,
            n_jobs=2,
            verbosity=0,
            **params,
        )
        est.fit(Xtr, y_train)
        pred_v = np.clip(est.predict(Xva), *clip_bounds)
        rmse_v = regression_metrics(y_val, pred_v)["rmse"]
        if rmse_v < best_xgb_rmse:
            best_xgb_rmse = rmse_v
            best_xgb = est
            best_xgb_cfg = params
    t_train = time.perf_counter() - t0
    ti0 = time.perf_counter()
    pred_te = np.clip(best_xgb.predict(Xte), *clip_bounds)
    t_inf_ms = (time.perf_counter() - ti0) * 1000.0
    trained["XGBoost"] = TrainedTabularModel(
        model_name="XGBoost",
        estimator=best_xgb,
        scaler=scaler,
        config=best_xgb_cfg,
        train_time_s=round(t_train, 4),
        infer_time_ms=round(t_inf_ms, 3),
        val_metrics=regression_metrics(y_val, np.clip(best_xgb.predict(Xva), *clip_bounds)),
        test_metrics=regression_metrics(y_test, pred_te),
        clip_bounds=clip_bounds,
    )

    return trained
