"""Physics-informed feature engineering & reference SoC/SoH calculation (Silver -> Gold Parquet).

Produces two parallel feature representations per task so cross-chemistry ablation
(Raw vs Chemistry-Invariant Physics-Normalized) is completely rigorous and leak-free:
- SoC: instantaneous + short-window electrical/thermal observables (NO cumulative Ah leakage).
- SoH: indirect cycle-level health indicators (NO discharge_capacity_ah leakage).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from src.config import CHEMISTRY_SPECS, PipelineConfig

RAW_SOC_FEATURES: list[str] = [
    "voltage_v",
    "current_a",
    "temp_c",
    "power_w",
    "v_roll_mean",
    "v_roll_std",
    "i_roll_mean",
    "i_roll_std",
    "dv_dt",
    "di_dt",
    "temp_rise_c",
]

PHYSICS_SOC_FEATURES: list[str] = [
    "v_norm",
    "c_rate",
    "p_norm",
    "temp_norm",
    "temp_rise_norm",
    "v_norm_roll_mean",
    "v_norm_roll_std",
    "c_rate_roll_mean",
    "c_rate_roll_std",
    "dv_norm_dt",
    "dc_rate_dt",
    "v_ir_comp_norm",
    "v_ir_comp_roll_mean",
    "v_norm_sq",
    "v_norm_cu",
]

RAW_SOH_FEATURES: list[str] = [
    "discharge_duration_s",
    "cc_duration_s",
    "cv_duration_s",
    "mean_voltage_v",
    "mean_temp_c",
    "temp_rise_c",
    "r_int_est_ohm",
    "ica_peak",
    "ica_v_norm_peak",
    "v_norm_integral",
]

PHYSICS_SOH_FEATURES: list[str] = [
    "dis_dur_ratio",
    "cc_dur_ratio",
    "cv_dur_ratio",
    "cv_cc_ratio",
    "r_int_ratio",
    "ica_peak_ratio",
    "ica_v_shift",
    "temp_rise_ratio",
    "v_norm_int_ratio",
    "temp_norm",
]


def verify_coulomb_counting_soc(
    current_a: np.ndarray,
    time_s: np.ndarray,
    usable_capacity_ah: float,
    soc_init: float = 1.0,
    coulombic_eff: float = 0.995,
) -> np.ndarray:
    """Discrete Coulomb counting (Section 5.2): SoC(k) = SoC(k-1) + eta * I_k * dt_k / (3600 * Q_usable)."""
    dt = np.diff(time_s, prepend=time_s[0])
    dt = np.clip(dt, 0.0, 600.0)
    dsoc = coulombic_eff * current_a * dt / (3600.0 * max(usable_capacity_ah, 1e-6))
    return np.clip(soc_init + np.cumsum(dsoc), 0.0, 1.0)


def compute_capacity_soh(discharge_capacity_ah: np.ndarray, ref_capacity_ah: float) -> np.ndarray:
    """Capacity-based SoH (Section 5.3): SoH = Q_measured / Q_ref."""
    return np.clip(discharge_capacity_ah / max(ref_capacity_ah, 1e-6), 0.0, 1.05)


def build_gold_features(
    cfg: PipelineConfig,
    silver_ts_path: Path,
    silver_cycles_path: Path,
) -> tuple[Path, Path]:
    """Engineer both Raw and Chemistry-Invariant Physics features into Gold Parquet tables."""
    ts = pl.read_parquet(silver_ts_path)
    cyc = pl.read_parquet(silver_cycles_path)

    li = CHEMISTRY_SPECS["li_ion"]
    na = CHEMISTRY_SPECS["na_ion"]

    # Map chemistry physical constants via Polars expressions
    v_min_expr = pl.when(pl.col("chemistry") == "li_ion").then(li.v_min).otherwise(na.v_min)
    v_max_expr = pl.when(pl.col("chemistry") == "li_ion").then(li.v_max).otherwise(na.v_max)
    r_ref_expr = (
        pl.when(pl.col("chemistry") == "li_ion")
        .then(li.r_int_bol * (li.default_q_nom / pl.col("q_nom")).pow(0.72))
        .otherwise(na.r_int_bol * (na.default_q_nom / pl.col("q_nom")).pow(0.72))
    )

    seg_keys = ["chemistry", "cell_id", "protocol", "cycle_number"]

    ts_gold = (
        ts.sort(seg_keys + ["time_s"])
        .with_columns(
            (v_max_expr - v_min_expr).alias("_v_span"),
            v_min_expr.alias("_v_min"),
            r_ref_expr.alias("_r_ref"),
        )
        .with_columns(
            # Basic raw & dimensionless physics signals
            (pl.col("voltage_v") * pl.col("current_a")).alias("power_w"),
            (pl.col("temp_c") - pl.col("ambient_temp_c")).alias("temp_rise_c"),
            ((pl.col("voltage_v") - pl.col("_v_min")) / pl.col("_v_span")).alias("v_norm"),
            (pl.col("current_a") / pl.col("q_nom")).alias("c_rate"),
            ((pl.col("temp_c") - 25.0) / 25.0).alias("temp_norm"),
            ((pl.col("temp_c") - pl.col("ambient_temp_c")) / 20.0).alias("temp_rise_norm"),
        )
        .with_columns(
            (pl.col("v_norm") * pl.col("c_rate")).alias("p_norm"),
            # IR-drop compensated normalized OCV proxy (accounts for ohmic drop & Arrhenius T-scaling)
            (
                (
                    pl.col("voltage_v")
                    - pl.col("current_a")
                    * pl.col("_r_ref")
                    * ((25.0 - pl.col("temp_c")) * 0.020).exp()
                    - pl.col("_v_min")
                )
                / pl.col("_v_span")
            )
            .clip(-0.15, 1.15)
            .alias("v_ir_comp_norm"),
            pl.col("v_norm").pow(2).alias("v_norm_sq"),
            pl.col("v_norm").pow(3).alias("v_norm_cu"),
            pl.col("time_s").diff().over(seg_keys).fill_null(10.0).clip(0.5, 600.0).alias("_dt"),
        )
        .with_columns(
            # Short-window rolling dynamics (strictly causal: past 6 samples, no future leakage)
            pl.col("voltage_v").rolling_mean(window_size=6, min_samples=1).over(seg_keys).alias("v_roll_mean"),
            pl.col("voltage_v")
            .rolling_std(window_size=6, min_samples=1)
            .over(seg_keys)
            .fill_null(0.0)
            .alias("v_roll_std"),
            pl.col("current_a").rolling_mean(window_size=6, min_samples=1).over(seg_keys).alias("i_roll_mean"),
            pl.col("current_a")
            .rolling_std(window_size=6, min_samples=1)
            .over(seg_keys)
            .fill_null(0.0)
            .alias("i_roll_std"),
            (pl.col("voltage_v").diff().over(seg_keys).fill_null(0.0) / pl.col("_dt")).alias("dv_dt"),
            (pl.col("current_a").diff().over(seg_keys).fill_null(0.0) / pl.col("_dt")).alias("di_dt"),
            # Physics-normalized rolling dynamics
            pl.col("v_norm").rolling_mean(window_size=6, min_samples=1).over(seg_keys).alias("v_norm_roll_mean"),
            pl.col("v_norm")
            .rolling_std(window_size=6, min_samples=1)
            .over(seg_keys)
            .fill_null(0.0)
            .alias("v_norm_roll_std"),
            pl.col("c_rate").rolling_mean(window_size=6, min_samples=1).over(seg_keys).alias("c_rate_roll_mean"),
            pl.col("c_rate")
            .rolling_std(window_size=6, min_samples=1)
            .over(seg_keys)
            .fill_null(0.0)
            .alias("c_rate_roll_std"),
            (pl.col("v_norm").diff().over(seg_keys).fill_null(0.0) / pl.col("_dt")).alias("dv_norm_dt"),
            (pl.col("c_rate").diff().over(seg_keys).fill_null(0.0) / pl.col("_dt")).alias("dc_rate_dt"),
            pl.col("v_ir_comp_norm")
            .rolling_mean(window_size=6, min_samples=1)
            .over(seg_keys)
            .alias("v_ir_comp_roll_mean"),
        )
        .drop(["_v_span", "_v_min", "_r_ref", "_dt"])
    )

    # --- Cycle/Curve-level SoH Gold Features ---
    # Ordered cells (NASA): normalize to genuine BoL cycles (cycle_number <= 3).
    # Pooled unordered workbooks (Mendeley): normalize using purely unsupervised input quantiles
    # (no SoH label use, no row-order dependence).
    if "data_provenance" not in cyc.columns:
        cyc = cyc.with_columns(pl.lit("ordered_per_cell").alias("data_provenance"))
    if "sample_idx" not in cyc.columns:
        cyc = cyc.with_columns(pl.col("cycle_number").fill_null(0).alias("sample_idx"))

    ordered_cyc = cyc.filter(pl.col("data_provenance") == "ordered_per_cell")
    pooled_cyc = cyc.filter(pl.col("data_provenance") != "ordered_per_cell")

    bol_parts: list[pl.DataFrame] = []
    if ordered_cyc.height > 0:
        bol_parts.append(
            ordered_cyc.filter(pl.col("cycle_number") <= 3)
            .group_by(["chemistry", "cell_id"])
            .agg(
                pl.col("discharge_duration_s").mean().alias("_dis_dur_0"),
                pl.col("cc_duration_s").mean().alias("_cc_dur_0"),
                pl.col("cv_duration_s").mean().alias("_cv_dur_0"),
                pl.col("r_int_est_ohm").mean().alias("_r_int_0"),
                pl.col("ica_peak").mean().alias("_ica_peak_0"),
                pl.col("ica_v_norm_peak").mean().alias("_ica_v_0"),
                pl.col("temp_rise_c").mean().alias("_temp_rise_0"),
                pl.col("v_norm_integral").mean().alias("_v_int_0"),
            )
        )
    if pooled_cyc.height > 0:
        bol_parts.append(
            pooled_cyc.group_by(["chemistry", "cell_id"]).agg(
                pl.col("discharge_duration_s").quantile(0.95).alias("_dis_dur_0"),
                pl.col("cc_duration_s").quantile(0.95).alias("_cc_dur_0"),
                pl.col("cv_duration_s").quantile(0.95).alias("_cv_dur_0"),
                pl.col("r_int_est_ohm").quantile(0.05).alias("_r_int_0"),
                pl.col("ica_peak").quantile(0.95).alias("_ica_peak_0"),
                pl.col("ica_v_norm_peak").median().alias("_ica_v_0"),
                pl.col("temp_rise_c").quantile(0.05).alias("_temp_rise_0"),
                pl.col("v_norm_integral").quantile(0.95).alias("_v_int_0"),
            )
        )

    bol_baselines = pl.concat(bol_parts, how="vertical_relaxed") if bol_parts else pl.DataFrame()

    cyc_gold = (
        cyc.join(bol_baselines, on=["chemistry", "cell_id"], how="left")
        .with_columns(
            (pl.col("discharge_duration_s") / pl.col("_dis_dur_0").clip(1.0, None)).alias("dis_dur_ratio"),
            (pl.col("cc_duration_s") / pl.col("_cc_dur_0").clip(1.0, None)).alias("cc_dur_ratio"),
            (pl.col("cv_duration_s") / pl.col("_cv_dur_0").clip(1.0, None)).alias("cv_dur_ratio"),
            (
                (pl.col("cv_duration_s") / pl.col("cc_duration_s").clip(1.0, None))
                / (pl.col("_cv_dur_0") / pl.col("_cc_dur_0").clip(1.0, None)).clip(0.01, None)
            ).alias("cv_cc_ratio"),
            (pl.col("r_int_est_ohm") / pl.col("_r_int_0").clip(1e-4, None)).alias("r_int_ratio"),
            (pl.col("ica_peak") / pl.col("_ica_peak_0").clip(1e-3, None)).alias("ica_peak_ratio"),
            (pl.col("ica_v_norm_peak") - pl.col("_ica_v_0")).alias("ica_v_shift"),
            (pl.col("temp_rise_c") / pl.col("_temp_rise_0").clip(0.1, None)).alias("temp_rise_ratio"),
            (pl.col("v_norm_integral") / pl.col("_v_int_0").clip(0.05, None)).alias("v_norm_int_ratio"),
            ((pl.col("mean_temp_c") - 25.0) / 25.0).alias("temp_norm"),
        )
        .drop(
            [
                "_dis_dur_0",
                "_cc_dur_0",
                "_cv_dur_0",
                "_r_int_0",
                "_ica_peak_0",
                "_ica_v_0",
                "_temp_rise_0",
                "_v_int_0",
            ]
        )
        .sort(["chemistry", "cell_id", "sample_idx"])
    )

    gold_soc_path = cfg.lake_dir / "gold" / "soc_features.parquet"
    gold_soh_path = cfg.lake_dir / "gold" / "soh_features.parquet"
    ts_gold.write_parquet(gold_soc_path, compression="zstd")
    cyc_gold.write_parquet(gold_soh_path, compression="zstd")
    return gold_soc_path, gold_soh_path
