"""End-to-end CLI orchestrator for the UROP Cross-Chemistry BMS framework (Objectives 1-6, Real Data Only).

Usage:
    python -m src.pipeline [--fast] [--dataset-root /path/to/datasets] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import polars as pl

from src.charging.adaptive_charging import run_adaptive_charging_benchmark
from src.config import PipelineConfig
from src.cross_chemistry.evaluation import run_repeated_cell_cv, run_task_within_and_cross_chemistry
from src.cross_chemistry.transfer_learning import run_transfer_learning_experiments
from src.data.lakehouse import ingest_to_bronze, run_duckdb_bigdata_analytics
from src.mining.domain_mining import run_data_mining_suite
from src.mining.feature_engineering import build_gold_features
from src.mining.quality_audit import audit_and_clean_bronze_to_silver
from src.reporting.plots import (
    plot_adaptive_charging_comparison,
    plot_cross_chemistry_ablation,
    plot_cross_chemistry_parity,
    plot_domain_alignment_pca,
    plot_na_soc_loco_per_cell,
    plot_transfer_learning_curves,
    plot_within_chemistry_benchmarks,
)
from src.reporting.report_generator import generate_experiment_report


def run_full_pipeline(cfg: PipelineConfig) -> dict:
    """Execute the complete Columnar Lakehouse -> Data Mining -> ML/DL -> Cross-Chemistry -> TL -> Adaptive Charging pipeline on real datasets."""
    t_start = time.perf_counter()
    cfg.ensure_dirs()

    print("[1/7] Columnar Lakehouse: Ingesting REAL NASA Li-ion & Wang et al. Na-ion datasets to Bronze Parquet...")
    inv, data_mode, bronze_ts, bronze_cyc = ingest_to_bronze(cfg)
    print(f"      Mode={data_mode} | Dataset inventory: {inv.pulled_files}/{inv.total_files} real files ({inv.to_dict()['total_declared_mb']} MB)")

    print("[2/7] Data Mining Quality Audit: Bronze -> Silver cleaning (nulls, duplicates, bounds, IsolationForest)...")
    silver_ts, silver_cyc, audit_report = audit_and_clean_bronze_to_silver(cfg, bronze_ts, bronze_cyc)
    duckdb_analytics = run_duckdb_bigdata_analytics(silver_ts, silver_cyc, cfg.tables_dir)
    print(
        f"      Silver TS={audit_report['timeseries']['silver_rows']:,} rows "
        f"(retention {audit_report['timeseries']['retention_rate_pct']}%) | Silver Cycles={audit_report['cycles']['silver_rows']:,}"
    )

    print("[3/7] Physics Feature Engineering & Unsupervised Domain Mining (Silver -> Gold Parquet)...")
    gold_soc_path, gold_soh_path = build_gold_features(cfg, silver_ts, silver_cyc)
    soc_df = pl.read_parquet(gold_soc_path)
    soh_df = pl.read_parquet(gold_soh_path)
    mining_results = run_data_mining_suite(cfg, soc_df, soh_df, cfg.tables_dir)
    print(f"      K-Means operating-regime Silhouette={mining_results['kmeans_silhouette']:.4f}")

    print("[4/7] Within-Chemistry, 3-Tier Cross-Chemistry & Repeated Cell-Level CV (SoC & SoH)...")
    soc_bundle = run_task_within_and_cross_chemistry(cfg, soc_df, task_name="SoC")
    soh_bundle = run_task_within_and_cross_chemistry(cfg, soh_df, task_name="SoH")

    within_df = pd.DataFrame(soc_bundle.within_rows + soh_bundle.within_rows)
    cross_df = pd.DataFrame(soc_bundle.cross_rows + soh_bundle.cross_rows)
    within_df.to_csv(cfg.tables_dir / "within_chemistry_results.csv", index=False)
    cross_df.to_csv(cfg.tables_dir / "cross_chemistry_ablation_results.csv", index=False)

    # Export Problem 1 & Problem 2 granular audit tables
    na_soc_loco_df = pd.DataFrame(soc_bundle.loco_per_cell_rows)
    if not na_soc_loco_df.empty:
        na_soc_loco_df.to_csv(cfg.tables_dir / "na_soc_loco_per_cell.csv", index=False)

    na_soh_lowo_df = pd.DataFrame(soh_bundle.loco_per_cell_rows)
    if not na_soh_lowo_df.empty:
        na_soh_lowo_df.to_csv(cfg.tables_dir / "na_soh_lowo_folds.csv", index=False)

    cross_per_unit_df = pd.DataFrame(soc_bundle.cross_per_unit_rows + soh_bundle.cross_per_unit_rows)
    if not cross_per_unit_df.empty:
        cross_per_unit_df.to_csv(cfg.tables_dir / "cross_chemistry_per_unit.csv", index=False)

    external_ncm_df = pd.DataFrame(soh_bundle.external_test_rows)
    if not external_ncm_df.empty:
        external_ncm_df.to_csv(cfg.tables_dir / "external_ncm_soh_test.csv", index=False)

    cv_seeds = (cfg.seed, cfg.seed + 59) if cfg.fast_mode else (cfg.seed, cfg.seed + 59, cfg.seed + 160)
    cv_rows = run_repeated_cell_cv(cfg, soc_df, "SoC", seeds=cv_seeds) + run_repeated_cell_cv(
        cfg, soh_df, "SoH", seeds=cv_seeds
    )
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(cfg.tables_dir / "repeated_cell_cv_results.csv", index=False)

    print("[5/7] Few-Shot Cross-Chemistry Transfer Learning (6-Fold LOCO SoC & Cross-Workbook Exploratory SoH)...")
    soc_tl = run_transfer_learning_experiments(cfg, soc_bundle)
    soh_tl = run_transfer_learning_experiments(cfg, soh_bundle)
    tl_df = pd.DataFrame(soc_tl.results_rows + soh_tl.results_rows)
    tl_df.to_csv(cfg.tables_dir / "transfer_learning_results.csv", index=False)

    print("[6/7] Simulated Adaptive Charging Strategy vs Conventional CC-CV Baseline (2-RC Mismatched Plant)...")
    li_soc_model = soc_bundle.within_tabular_phys["li_ion"]["XGBoost"]
    na_tl_soc_model = soc_tl.best_tl_li_to_na_predictor or soc_bundle.within_tabular_phys["na_ion"]["XGBoost"]
    li_soh_mae = (
        soh_bundle.within_tabular_phys["li_ion"]["XGBoost"].test_metrics["mae"]
        if "li_ion" in soh_bundle.within_tabular_phys
        else 0.01
    )
    na_soh_xgb_rows = [r for r in soh_bundle.within_rows if r["chemistry"] == "na_ion" and r["model"] == "XGBoost"]
    na_soh_mae = float(na_soh_xgb_rows[0]["mae"]) if na_soh_xgb_rows else 0.025

    charging_rows, charging_traces = run_adaptive_charging_benchmark(
        li_soc_predictor=lambda X: li_soc_model.predict(X),
        na_tl_soc_predictor=lambda X: na_tl_soc_model.predict(X),
        li_soh_mae=li_soh_mae,
        na_soh_mae=na_soh_mae,
        empirical_ocv_curves=soc_bundle.empirical_ocv_curves,
    )
    charging_df = pd.DataFrame(charging_rows)
    charging_df.to_csv(cfg.tables_dir / "adaptive_charging_results.csv", index=False)

    # Export structured JSON time-series traces & 3D PCA manifold coordinates for downstream viewers
    traces_payload = {}
    for key, tr in charging_traces.items():
        # Downsample very long traces gently to <= 240 smooth frames
        stride = max(1, len(tr.time_min) // 220)
        sl = slice(None, None, stride)
        t_arr = tr.time_min[sl].tolist()
        if t_arr[-1] != float(tr.time_min[-1]):
            t_arr.append(float(tr.time_min[-1]))
            soc_t = tr.soc_true[sl].tolist() + [float(tr.soc_true[-1])]
            soc_e = tr.soc_est[sl].tolist() + [float(tr.soc_est[-1])]
            vol = tr.voltage_v[sl].tolist() + [float(tr.voltage_v[-1])]
            cr = tr.current_c_rate[sl].tolist() + [float(tr.current_c_rate[-1])]
            tmp = tr.temp_c[sl].tolist() + [float(tr.temp_c[-1])]
        else:
            soc_t = tr.soc_true[sl].tolist()
            soc_e = tr.soc_est[sl].tolist()
            vol = tr.voltage_v[sl].tolist()
            cr = tr.current_c_rate[sl].tolist()
            tmp = tr.temp_c[sl].tolist()

        traces_payload[key] = {
            "chemistry": tr.chemistry,
            "scenario": tr.scenario,
            "strategy": tr.strategy,
            "metrics": tr.metrics,
            "series": {
                "time_min": [round(x, 3) for x in t_arr],
                "soc_true": [round(x, 4) for x in soc_t],
                "soc_est": [round(x, 4) for x in soc_e],
                "voltage_v": [round(x, 4) for x in vol],
                "current_c_rate": [round(x, 4) for x in cr],
                "temp_c": [round(x, 3) for x in tmp],
            },
        }
    (cfg.output_dir / "charging_traces.json").write_text(
        json.dumps(traces_payload, indent=2), encoding="utf-8"
    )

    pca_export = {}
    for task_key, payload in mining_results["pca_payload"].items():
        # Scale each 3D cloud symmetrically into [-1, 1] cube for clean 3D visualization
        raw_all = np.vstack([payload["raw_li"], payload["raw_na"]])
        raw_scale = float(np.percentile(np.abs(raw_all), 98)) or 1.0
        aln_all = np.vstack([payload["aligned_li"], payload["aligned_na"]])
        aln_scale = float(np.percentile(np.abs(aln_all), 98)) or 1.0

        pca_export[task_key] = {
            "raw_li": np.round(np.clip(payload["raw_li"] / raw_scale, -1.4, 1.4), 4).tolist(),
            "raw_na": np.round(np.clip(payload["raw_na"] / raw_scale, -1.4, 1.4), 4).tolist(),
            "aligned_li": np.round(np.clip(payload["aligned_li"] / aln_scale, -1.4, 1.4), 4).tolist(),
            "aligned_na": np.round(np.clip(payload["aligned_na"] / aln_scale, -1.4, 1.4), 4).tolist(),
            "raw_explained_var": payload.get("raw_explained_var", []),
            "aligned_explained_var": payload.get("aligned_explained_var", []),
        }
    (cfg.output_dir / "pca_manifold.json").write_text(
        json.dumps(pca_export, indent=2), encoding="utf-8"
    )

    # Export data-driven empirical OCV curves fitted strictly on real training cells (consumed by fullstack_3d live sim)
    ocv_export = {
        chem: {
            "chemistry": curve.chemistry,
            "estimation_method": curve.estimation_method,
            "n_samples_used": curve.n_samples_used,
            "soc_grid": [round(float(x), 5) for x in curve.soc_grid],
            "v_norm_grid": [round(float(x), 5) for x in curve.v_norm_grid],
        }
        for chem, curve in soc_bundle.empirical_ocv_curves.items()
    }
    (cfg.output_dir / "empirical_ocv_curves.json").write_text(
        json.dumps(ocv_export, indent=2), encoding="utf-8"
    )

    print("[7/7] Generating Publication Figures & Experiment Report...")
    figs = [
        plot_domain_alignment_pca(mining_results, cfg.figures_dir),
        plot_within_chemistry_benchmarks(soc_bundle, soh_bundle, cfg.figures_dir),
        plot_cross_chemistry_ablation(soc_bundle, soh_bundle, cfg.figures_dir),
        plot_transfer_learning_curves(soc_tl, soh_tl, cfg.figures_dir),
        plot_cross_chemistry_parity(soc_bundle, soh_bundle, cfg.figures_dir),
        plot_adaptive_charging_comparison(charging_rows, charging_traces, cfg.figures_dir),
    ]
    fig7 = plot_na_soc_loco_per_cell(soc_bundle.loco_per_cell_rows, cfg.figures_dir)
    if fig7 is not None:
        figs.append(fig7)

    elapsed_s = round(time.perf_counter() - t_start, 2)
    summary = {
        "data_mode": data_mode,
        "elapsed_seconds": elapsed_s,
        "dataset_inventory": inv.to_dict(),
        "quality_audit": audit_report,
        "duckdb_analytics": duckdb_analytics,
        "data_mining": {k: v for k, v in mining_results.items() if k != "pca_payload"},
        "unit_counts": {
            "SoC": soc_bundle.unit_counts,
            "SoH": soh_bundle.unit_counts,
        },
        "evaluation_modes": {
            "SoC_cross_chemistry": soc_bundle.cross_chemistry_mode,
            "SoH_cross_chemistry": soh_bundle.cross_chemistry_mode,
            "na_soc_within": f"leave_one_cell_out_{soc_bundle.unit_counts.get('na_ion', 0)}_folds",
            "na_soh_within": "exploratory_lowo_temp_confounded",
        },
        "external_ncm_soh_test": soh_bundle.external_test_rows,
        "na_soh_lowo_folds": soh_bundle.loco_per_cell_rows,
        "cross_chemistry_per_unit": soc_bundle.cross_per_unit_rows + soh_bundle.cross_per_unit_rows,
        "empirical_ocv_audit": {
            chem: {
                "estimation_method": curve.estimation_method,
                "n_training_samples_used": curve.n_samples_used,
                "fitted_on": "training_cells_only",
                "uses_simulator_formula": False,
            }
            for chem, curve in soc_bundle.empirical_ocv_curves.items()
        },
        "compatibility_notes": [
            n for n in [soc_bundle.compatibility_note, soh_bundle.compatibility_note] if n
        ],
        "splits": {
            "SoC": {
                chem: {
                    "train_cells": sp.train_cells,
                    "val_cells": sp.val_cells,
                    "test_cells": sp.test_cells,
                }
                for chem, sp in soc_bundle.splits.items()
            },
            "SoH": {
                chem: {
                    "train_cells": sp.train_cells,
                    "val_cells": sp.val_cells,
                    "test_cells": sp.test_cells,
                }
                for chem, sp in soh_bundle.splits.items()
            },
        },
        "figures": [str(f.relative_to(cfg.repo_root)) for f in figs],
    }

    report_path = generate_experiment_report(
        output_dir=cfg.output_dir,
        summary=summary,
        within_df=within_df,
        cross_df=cross_df,
        tl_df=tl_df,
        charging_df=charging_df,
        cv_df=cv_df,
        na_soc_loco_df=na_soc_loco_df,
        na_soh_lowo_df=na_soh_lowo_df,
        cross_per_unit_df=cross_per_unit_df,
        external_ncm_df=external_ncm_df,
    )
    print(f"Done in {elapsed_s:.2f}s! Report written to: {report_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UROP Cross-Chemistry BMS Full Pipeline (Real Data Only)")
    parser.add_argument("--fast", action="store_true", help="Run reduced-epoch fast mode")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Directory containing real NASA & Na-ion folders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = PipelineConfig(seed=args.seed, fast_mode=args.fast, dataset_root_override=args.dataset_root)
    run_full_pipeline(cfg)


if __name__ == "__main__":
    main()
