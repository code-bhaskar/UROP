"""Tests for Columnar Lakehouse (Polars/Parquet/DuckDB), Quality Audit, Feature Engineering, and Leak-Free Splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.config import PipelineConfig
from src.data.lakehouse import ingest_to_bronze, run_duckdb_bigdata_analytics
from src.mining.domain_mining import compute_rbf_mmd, quantify_domain_shift
from src.mining.feature_engineering import (
    PHYSICS_SOC_FEATURES,
    PHYSICS_SOH_FEATURES,
    RAW_SOC_FEATURES,
    RAW_SOH_FEATURES,
    build_gold_features,
    compute_capacity_soh,
    verify_coulomb_counting_soc,
)
from src.mining.quality_audit import audit_and_clean_bronze_to_silver
from src.mining.splits import split_by_cell


def test_coulomb_counting_and_soh_formulas() -> None:
    t = np.linspace(0.0, 1800.0, 181)
    i = np.full_like(t, -1.0)
    soc = verify_coulomb_counting_soc(i, t, usable_capacity_ah=2.0, soc_init=1.0, coulombic_eff=1.0)
    assert np.isclose(soc[0], 1.0, atol=1e-6)
    assert np.isclose(soc[-1], 0.75, atol=1e-3)

    soh = compute_capacity_soh(np.array([2.0, 1.8, 1.6]), ref_capacity_ah=2.0)
    assert np.allclose(soh, [1.0, 0.9, 0.8])


def test_ingest_raises_when_real_datasets_missing(tmp_path: Path) -> None:
    """Ensure NO fake fallback exists: missing real datasets must raise FileNotFoundError."""
    cfg = PipelineConfig(repo_root=tmp_path, dataset_root_override=tmp_path, fast_mode=True)
    with pytest.raises(FileNotFoundError, match="Real battery datasets"):
        ingest_to_bronze(cfg)


def test_medallion_lakehouse_quality_audit_and_leak_free_splits(
    tmp_path: Path, real_format_dataset_root: Path
) -> None:
    out_root = tmp_path / "run_out"
    cfg = PipelineConfig(
        repo_root=out_root,
        dataset_root_override=real_format_dataset_root,
        seed=7,
        fast_mode=True,
    )
    _, mode, b_ts, b_cyc = ingest_to_bronze(cfg)
    assert mode == "real_dataset"
    assert b_ts.exists() and b_cyc.exists()

    s_ts, s_cyc, audit = audit_and_clean_bronze_to_silver(cfg, b_ts, b_cyc)
    assert audit["timeseries"]["silver_rows"] > 500
    assert audit["cycles"]["silver_rows"] > 50

    analytics = run_duckdb_bigdata_analytics(s_ts, s_cyc, cfg.tables_dir)
    assert len(analytics["chemistry_summary"]) == 2
    assert len(analytics["aging_summary"]) == 1  # Real Li-ion aging only (no fabricated Na-ion SoH)

    g_soc, g_soh = build_gold_features(cfg, s_ts, s_cyc)
    soc_df = pl.read_parquet(g_soc)
    soh_df = pl.read_parquet(g_soh)

    for col in RAW_SOC_FEATURES + PHYSICS_SOC_FEATURES:
        assert col in soc_df.columns
        assert soc_df[col].null_count() == 0
    for col in RAW_SOH_FEATURES + PHYSICS_SOH_FEATURES:
        assert col in soh_df.columns
        assert soh_df[col].null_count() == 0

    for chem in ("li_ion", "na_ion"):
        sp = split_by_cell(soc_df, chem, seed=7)
        assert set(sp.train_cells).isdisjoint(sp.val_cells)
        assert set(sp.train_cells).isdisjoint(sp.test_cells)
        assert set(sp.val_cells).isdisjoint(sp.test_cells)


def test_domain_shift_metrics_detect_distribution_gap() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(300, 4))
    y_same = rng.normal(0.0, 1.0, size=(300, 4))
    y_shift = rng.normal(3.0, 1.0, size=(300, 4))
    assert compute_rbf_mmd(x, y_shift) > compute_rbf_mmd(x, y_same) * 5.0
    m = quantify_domain_shift(x, y_shift)
    assert m["mean_wasserstein_1"] > 1.0
