"""Standard quantitative regression evaluation metrics (MAE, RMSE, R^2) per Section 7."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and R^2 with clean rounding."""
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()
    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(root_mean_squared_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p))
    return {
        "mae": round(mae, 5),
        "rmse": round(rmse, 5),
        "r2": round(r2, 5),
    }
