"""Tests verifying Cross-Chemistry Alignment outperforms Raw Zero-Shot and Transfer Learning converges on real-format data."""

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


def test_cross_chemistry_alignment_improves_soc_and_handles_single_chem_soh(
    tmp_path: Path, real_format_dataset_root: Path
) -> None:
    out_root = tmp_path / "run_out"
    cfg = PipelineConfig(
        repo_root=out_root,
        dataset_root_override=real_format_dataset_root,
        seed=11,
        fast_mode=True,
        tl_fractions=(0.25,),
    )
    _, mode, b_ts, b_cyc = ingest_to_bronze(cfg)
    assert mode == "real_dataset"
    s_ts, s_cyc, _ = audit_and_clean_bronze_to_silver(cfg, b_ts, b_cyc)
    g_soc, g_soh = build_gold_features(cfg, s_ts, s_cyc)

    soc_df = pl.read_parquet(g_soc)
    soh_df = pl.read_parquet(g_soh)

    soc_bundle = run_task_within_and_cross_chemistry(cfg, soc_df, "SoC")
    soh_bundle = run_task_within_and_cross_chemistry(cfg, soh_df, "SoH")

    # Within-chemistry models report validation & test metrics across all 5 model families
    within_df = pd.DataFrame(soc_bundle.within_rows + soh_bundle.within_rows)
    assert {"val_mae", "val_rmse", "val_r2", "mae", "rmse", "r2"}.issubset(set(within_df.columns))
    assert set(within_df["model"].unique()) == {"LinearRegression", "RandomForest", "XGBoost", "LSTM", "GRU"}
    soc_within_tab = within_df[
        (within_df["task"] == "SoC") & (within_df["model"].isin(["RandomForest", "XGBoost"]))
    ]
    assert soc_within_tab["r2"].min() > 0.90

    # Single-chemistry real SoH skips cross-chemistry cleanly without crashing or fabricating Na-ion aging
    assert soh_bundle.cross_chemistry_available is False
    assert bool(soh_bundle.compatibility_note)
    assert len(soh_bundle.cross_rows) == 0

    # Cross-chemistry SoC: Tier 3 (Physics+OCV+CORAL) must strongly beat Tier 1 (Raw) in both directions
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
        assert aln_rmse < raw_rmse * 0.75

    # Transfer learning at 25% target data converges strongly across LOCO folds
    soc_tl = run_transfer_learning_experiments(cfg, soc_bundle)
    tl_df = pd.DataFrame(soc_tl.results_rows)
    gru_tl = tl_df[
        (tl_df["direction"] == "li_ion_to_na_ion")
        & (tl_df["target_fraction"] == 0.25)
        & (tl_df["method"] == "TL_TwoPhase_GRU")
    ].iloc[0]
    assert gru_tl["r2"] > 0.80

    # Verify Na-ion SoC LOCO evaluated every Na-ion cell with zero overlap between test_cell and val_cell
    assert len(soc_bundle.loco_per_cell_rows) > 0
    na_cells = set(soc_df.filter(pl.col("chemistry") == "na_ion")["cell_id"].unique().to_list())
    tested_cells = {r["test_cell"] for r in soc_bundle.loco_per_cell_rows}
    assert tested_cells == na_cells
    for r in soc_bundle.loco_per_cell_rows:
        assert r["test_cell"] != r["val_cell"]


def test_assert_no_cell_overlap_catches_leakage() -> None:
    """Hard runtime guard catches any test cell/workbook appearing in train or validation."""
    import pytest
    from src.cross_chemistry.evaluation import assert_no_cell_overlap

    assert_no_cell_overlap(["c1", "c2"], ["c3"], ["c4"], context="clean_split")
    with pytest.raises(AssertionError, match="Leakage detected"):
        assert_no_cell_overlap(["c1", "c4"], ["c3"], ["c4"], context="train_leak")
    with pytest.raises(AssertionError, match="Leakage detected"):
        assert_no_cell_overlap(["c1", "c2"], ["c4"], ["c4"], context="val_leak")
