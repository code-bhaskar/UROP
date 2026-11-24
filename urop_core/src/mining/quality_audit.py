"""Automated Data Mining Quality Audit: missingness, duplicates, physical bounds, and IsolationForest anomaly detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest

from src.config import CHEMISTRY_SPECS, PipelineConfig


def audit_and_clean_bronze_to_silver(
    cfg: PipelineConfig,
    bronze_ts_path: Path,
    bronze_cycles_path: Path,
) -> tuple[Path, Path, dict]:
    """Audit Bronze Parquet tables, clean invalid records, and write Silver Parquet tables."""
    ts_raw = pl.read_parquet(bronze_ts_path)
    cyc_raw = pl.read_parquet(bronze_cycles_path)

    n_raw_ts = ts_raw.height
    null_v = int(ts_raw["voltage_v"].null_count() + ts_raw["voltage_v"].is_nan().sum())
    null_t = int(ts_raw["temp_c"].null_count() + ts_raw["temp_c"].is_nan().sum())
    null_i = int(ts_raw["current_a"].null_count() + ts_raw["current_a"].is_nan().sum())

    # 1. Deduplicate by natural measurement key
    ts_dedup = ts_raw.unique(
        subset=["chemistry", "cell_id", "protocol", "cycle_number", "time_s"],
        keep="first",
        maintain_order=True,
    )
    n_dups_removed = n_raw_ts - ts_dedup.height

    # 2. Enforce chemistry-specific physical voltage/temperature bounds & drop NaNs/nulls
    li = CHEMISTRY_SPECS["li_ion"]
    na = CHEMISTRY_SPECS["na_ion"]

    valid_voltage = (
        (pl.col("chemistry") == "li_ion")
        & pl.col("voltage_v").is_between(li.v_min - 0.35, li.v_max + 0.35)
    ) | (
        (pl.col("chemistry") == "na_ion")
        & pl.col("voltage_v").is_between(na.v_min - 0.35, na.v_max + 0.35)
    )
    valid_temp = pl.col("temp_c").is_between(-25.0, 85.0)
    valid_curr = (pl.col("current_a").abs() / pl.col("q_nom")) <= 6.0

    ts_bounded = ts_dedup.filter(
        pl.col("voltage_v").is_not_null()
        & pl.col("voltage_v").is_not_nan()
        & pl.col("current_a").is_not_null()
        & pl.col("current_a").is_not_nan()
        & pl.col("temp_c").is_not_null()
        & pl.col("temp_c").is_not_nan()
        & valid_voltage
        & valid_temp
        & valid_curr
    )
    n_bounds_or_null_removed = ts_dedup.height - ts_bounded.height

    # 3. Unsupervised Data Mining: IsolationForest multivariate anomaly scoring per chemistry
    # Flags extreme multivariate anomalies (kept as diagnostic column; extreme 0.2% dropped)
    clean_parts: list[pl.DataFrame] = []
    n_iforest_flagged = 0

    for chem in sorted(ts_bounded["chemistry"].unique().to_list()):
        sub = ts_bounded.filter(pl.col("chemistry") == chem)
        spec = CHEMISTRY_SPECS[chem]
        X_anom = np.column_stack(
            [
                (sub["voltage_v"].to_numpy() - spec.v_min) / (spec.v_max - spec.v_min),
                sub["current_a"].to_numpy() / sub["q_nom"].to_numpy(),
                (sub["temp_c"].to_numpy() - sub["ambient_temp_c"].to_numpy()),
            ]
        )
        iso = IsolationForest(
            n_estimators=80,
            contamination=0.002,
            random_state=cfg.seed,
            n_jobs=-1,
        )
        preds = iso.fit_predict(X_anom)
        scores = iso.decision_function(X_anom)
        n_iforest_flagged += int(np.sum(preds == -1))

        sub = sub.with_columns(
            pl.Series("anomaly_score", scores.astype(np.float64)),
            pl.Series("is_anomaly", (preds == -1)),
        ).filter(~pl.col("is_anomaly"))
        clean_parts.append(sub)

    ts_silver = pl.concat(clean_parts, how="vertical").sort(
        ["chemistry", "cell_id", "protocol", "cycle_number", "time_s"]
    )

    # Clean cycle table
    n_raw_cyc = cyc_raw.height
    cyc_silver = (
        cyc_raw.unique(subset=["chemistry", "cell_id", "cycle_number"], keep="first")
        .filter(
            pl.col("soh").is_between(0.45, 1.05)
            & pl.col("discharge_capacity_ah").gt(0.1)
            & pl.col("r_int_est_ohm").is_between(0.001, 2.0)
        )
        .sort(["chemistry", "cell_id", "cycle_number"])
    )

    silver_ts_path = cfg.lake_dir / "silver" / "timeseries.parquet"
    silver_cyc_path = cfg.lake_dir / "silver" / "cycles.parquet"
    ts_silver.write_parquet(silver_ts_path, compression="zstd")
    cyc_silver.write_parquet(silver_cyc_path, compression="zstd")

    audit_report = {
        "timeseries": {
            "bronze_rows": n_raw_ts,
            "silver_rows": ts_silver.height,
            "null_voltage_count": null_v,
            "null_temp_count": null_t,
            "null_current_count": null_i,
            "duplicates_removed": n_dups_removed,
            "bounds_and_null_removed": n_bounds_or_null_removed,
            "isolation_forest_anomalies_removed": n_iforest_flagged,
            "retention_rate_pct": round(100.0 * ts_silver.height / max(n_raw_ts, 1), 2),
        },
        "cycles": {
            "bronze_rows": n_raw_cyc,
            "silver_rows": cyc_silver.height,
            "removed": n_raw_cyc - cyc_silver.height,
        },
    }
    return silver_ts_path, silver_cyc_path, audit_report
