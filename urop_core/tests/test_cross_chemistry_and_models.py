"""Tests verifying Cross-Chemistry Alignment dramatically outperforms Raw Zero-Shot and Transfer Learning converges."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

from src.config import PipelineConfig
from src.cross_chemistry.evaluation import run_task_within_and_cross_chemistry
from src.cross_chemistry.transfer_learning import run_transfer_learning_experiments
from src.data.lakehouse import ingest_to_bronze
from src.mining.feature_engineering import build_gold_features
from src.mining.quality_audit import audit_and_clean_bronze_to_silver


def test_cross_chemistry_alignment_dramatically_improves_soc_and_soh(tmp_path: Path) -> None:
    cfg = PipelineConfig(repo_root=tmp_path, seed=11, fast_mode=True, tl_fractions=(0.10,))
    _, _, b_ts, b_cyc = ingest_to_bronze(cfg, force_fallback=True)
    s_ts, s_cyc, _ = audit_and_clean_bronze_to_silver(cfg, b_ts, b_cyc)
    g_soc, g_soh = build_gold_features(cfg, s_ts, s_cyc)

    soc_df = pl.read_parquet(g_soc)
    soh_df = pl.read_parquet(g_soh)

    soc_bundle = run_task_within_and_cross_chemistry(cfg, soc_df, "SoC")
    soh_bundle = run_task_within_and_cross_chemistry(cfg, soh_df, "SoH")

    # Within-chemistry models should all achieve strong R^2 (even in reduced 10-epoch fast_mode)
    within_df = pd.DataFrame(soc_bundle.within_rows + soh_bundle.within_rows)
    assert set(within_df["model"].unique()) == {"LinearRegression", "RandomForest", "XGBoost", "LSTM", "GRU"}
    tab_within = within_df[within_df["model"].isin(["LinearRegression", "RandomForest", "XGBoost"])]
    assert tab_within["r2"].min() > 0.95
    assert within_df["r2"].min() > 0.70

    # Cross-chemistry SoC: Tier 3 (Physics+OCV+CORAL) must dramatically beat Tier 1 (Raw) in both directions
    cross_soc = pd.DataFrame(soc_bundle.cross_rows)
    for direction in ("li_ion_to_na_ion", "na_ion_to_li_ion"):
        raw_rmse = cross_soc[
            (cross_soc["direction"] == direction)
            & (cross_soc["tier"] == "1_raw_zero_shot")
            & (cross_soc["model"] == "XGBoost")
        ]["rmse"].iloc[0]
        aln_rmse = cross_soc[
            (cross_soc["direction"] == direction)
            & (cross_soc["tier"] == "3_physics_ocv_coral_zero_shot")
            & (cross_soc["model"] == "XGBoost")
        ]["rmse"].iloc[0]
        assert aln_rmse < raw_rmse * 0.85
        assert aln_rmse < 0.16

    # Cross-chemistry SoH: Tier 2/3 physics-aligned must achieve R^2 > 0.95 in both directions
    cross_soh = pd.DataFrame(soh_bundle.cross_rows)
    for direction in ("li_ion_to_na_ion", "na_ion_to_li_ion"):
        aln_r2 = cross_soh[
            (cross_soh["direction"] == direction)
            & (cross_soh["tier"] == "3_physics_ocv_coral_zero_shot")
            & (cross_soh["model"] == "XGBoost")
        ]["r2"].iloc[0]
        assert aln_r2 > 0.95

    # Transfer learning at 10% target data must achieve R^2 > 0.96
    soc_tl = run_transfer_learning_experiments(cfg, soc_bundle)
    tl_df = pd.DataFrame(soc_tl.results_rows)
    gru_tl = tl_df[
        (tl_df["direction"] == "li_ion_to_na_ion")
        & (tl_df["target_fraction"] == 0.10)
        & (tl_df["method"] == "TL_TwoPhase_GRU")
    ].iloc[0]
    assert gru_tl["r2"] > 0.96
