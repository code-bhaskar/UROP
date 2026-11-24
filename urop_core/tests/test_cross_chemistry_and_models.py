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


def test_leave_one_cell_out_tiny_fixture() -> None:
    """Verify 4-cell Leave-One-Cell-Out holds out each cell once with zero train/val overlap."""
    import numpy as np
    from src.cross_chemistry.evaluation import run_na_soc_leave_one_cell_out
    from src.mining.feature_engineering import PHYSICS_SOC_FEATURES

    rng = np.random.default_rng(7)
    rows: list[dict] = []
    for c_idx, cell_id in enumerate(["na_3.2Ah_bat1", "na_3.2Ah_bat2", "na_10.0Ah_bat3", "na_10.0Ah_bat4"]):
        q_nom = 3.2 if "3.2Ah" in cell_id else 10.0
        for step in range(40):
            soc = float(1.0 - step / 39.0)
            rec = {
                "chemistry": "na_ion",
                "cell_id": cell_id,
                "protocol": "pulse_T25",
                "cycle_number": 1,
                "q_nom": q_nom,
                "soc": soc,
                "data_provenance": "ordered_per_cell",
            }
            for j, feat in enumerate(PHYSICS_SOC_FEATURES):
                rec[feat] = float(soc * (0.8 + 0.05 * j) + rng.normal(0.0, 0.02))
            rows.append(rec)

    tiny_df = pl.DataFrame(rows)
    cfg = PipelineConfig(seed=42, fast_mode=True)
    per_cell, summary = run_na_soc_leave_one_cell_out(cfg, tiny_df, include_seq=False)

    assert len({r["test_cell"] for r in per_cell}) == 4
    assert all(r["test_cell"] != r["val_cell"] for r in per_cell)
    assert len(summary) == 3  # LinearRegression, RandomForest, XGBoost
    assert all(r["rmse_std"] >= 0.0 for r in summary)


def test_no_test_unit_in_any_fit_and_exploratory_pooled_soh_tabular_only(
    tmp_path: Path, real_format_dataset_root: Path, monkeypatch
) -> None:
    """Spy on assert_no_cell_overlap across every experiment function + verify pooled SoH skips LSTM/GRU."""
    import numpy as np
    import src.cross_chemistry.evaluation as eval_mod
    import src.cross_chemistry.transfer_learning as tl_mod
    from src.cross_chemistry.evaluation import run_repeated_cell_cv

    checked_contexts: list[str] = []
    orig_assert = eval_mod.assert_no_cell_overlap

    def spy_assert(train_ids, val_ids, test_ids, context=""):
        checked_contexts.append(context)
        orig_assert(train_ids, val_ids, test_ids, context=context)

    monkeypatch.setattr(eval_mod, "assert_no_cell_overlap", spy_assert)
    monkeypatch.setattr(tl_mod, "assert_no_cell_overlap", spy_assert)

    cfg = PipelineConfig(
        repo_root=tmp_path / "spy_out",
        dataset_root_override=real_format_dataset_root,
        seed=17,
        fast_mode=True,
        tl_fractions=(0.25,),
    )
    _, _, b_ts, b_cyc = ingest_to_bronze(cfg)
    s_ts, s_cyc, _ = audit_and_clean_bronze_to_silver(cfg, b_ts, b_cyc)
    g_soc, g_soh = build_gold_features(cfg, s_ts, s_cyc)

    soc_df = pl.read_parquet(g_soc)
    soh_df = pl.read_parquet(g_soh)

    # Append 2 pooled Mendeley-style Na-ion workbooks to test exploratory LOWO + cross-workbook TL
    li_template = soh_df.filter(pl.col("chemistry") == "li_ion").head(120)
    na_wb_a = li_template.head(60).with_columns(
        pl.lit("na_ion").alias("chemistry"),
        pl.lit("na_nfm_25C").alias("cell_id"),
        pl.lit(None, dtype=pl.Int64).alias("cycle_number"),
        pl.lit("pooled_unordered").alias("data_provenance"),
    )
    na_wb_b = li_template.tail(60).with_columns(
        pl.lit("na_ion").alias("chemistry"),
        pl.lit("na_nfm_m15C").alias("cell_id"),
        pl.lit(None, dtype=pl.Int64).alias("cycle_number"),
        pl.lit("pooled_unordered").alias("data_provenance"),
    )
    soh_with_pooled = pl.concat([soh_df, na_wb_a, na_wb_b], how="diagonal_relaxed")

    soc_bundle = eval_mod.run_task_within_and_cross_chemistry(cfg, soc_df, "SoC")
    soh_bundle = eval_mod.run_task_within_and_cross_chemistry(cfg, soh_with_pooled, "SoH")
    _ = run_repeated_cell_cv(cfg, soc_df, "SoC", seeds=(17,))
    _ = tl_mod.run_transfer_learning_experiments(cfg, soc_bundle)
    soh_tl = tl_mod.run_transfer_learning_experiments(cfg, soh_bundle)

    # Every experiment family invoked assert_no_cell_overlap with zero leakage
    assert any(c.startswith("na_soc_loco_") for c in checked_contexts)
    assert any(c.startswith("na_soh_lowo_") for c in checked_contexts)
    assert any(c.startswith("cross_SoC_") for c in checked_contexts)
    assert any(c.startswith("cross_SoH_") for c in checked_contexts)
    assert any(c.startswith("tl_soc_loco_") for c in checked_contexts)
    assert any(c.startswith("tl_soh_xwb_") for c in checked_contexts)
    assert any(c.startswith("cv_SoC_") for c in checked_contexts)

    # Pooled exploratory SoH strictly skips sequence models (LSTM/GRU)
    assert soh_bundle.cross_chemistry_mode == "exploratory_pooled"
    na_soh_models = {r["model"] for r in soh_bundle.within_rows if r["chemistry"] == "na_ion"}
    assert "LSTM" not in na_soh_models and "GRU" not in na_soh_models
    soh_cross_models = {r["model"] for r in soh_bundle.cross_rows}
    assert "LSTM" not in soh_cross_models and "GRU" not in soh_cross_models
    soh_tl_methods = {r["method"] for r in soh_tl.results_rows}
    assert all("GRU" not in m and "LSTM" not in m for m in soh_tl_methods)
