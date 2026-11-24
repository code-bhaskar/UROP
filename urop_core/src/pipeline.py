"""End-to-end CLI orchestrator for the UROP Cross-Chemistry BMS framework (Objectives 1-6).

Usage:
    python -m src.pipeline [--fast] [--force-fallback] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import polars as pl

from src.charging.adaptive_charging import run_adaptive_charging_benchmark
from src.config import PipelineConfig
from src.cross_chemistry.evaluation import run_task_within_and_cross_chemistry
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
    plot_transfer_learning_curves,
    plot_within_chemistry_benchmarks,
)
from src.reporting.report_generator import generate_experiment_report


def run_full_pipeline(cfg: PipelineConfig, force_fallback: bool = False) -> dict:
    """Execute the complete Big Data -> Data Mining -> ML/DL -> Cross-Chemistry -> TL -> Adaptive Charging pipeline."""
    t_start = time.perf_counter()
    cfg.ensure_dirs()

    print("[1/7] Big Data Lakehouse: Ingesting battery datasets to Bronze Parquet...")
    inv, data_mode, bronze_ts, bronze_cyc = ingest_to_bronze(cfg, force_fallback=force_fallback)
    print(f"      Mode={data_mode} | LFS inventory: {inv.pulled_files}/{inv.total_files} pulled ({inv.to_dict()['total_declared_mb']} MB)")

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

    print("[4/7] Within-Chemistry & 3-Tier Bidirectional Cross-Chemistry Evaluation (SoC & SoH)...")
    soc_bundle = run_task_within_and_cross_chemistry(cfg, soc_df, task_name="SoC")
    soh_bundle = run_task_within_and_cross_chemistry(cfg, soh_df, task_name="SoH")

    within_df = pd.DataFrame(soc_bundle.within_rows + soh_bundle.within_rows)
    cross_df = pd.DataFrame(soc_bundle.cross_rows + soh_bundle.cross_rows)
    within_df.to_csv(cfg.tables_dir / "within_chemistry_results.csv", index=False)
    cross_df.to_csv(cfg.tables_dir / "cross_chemistry_ablation_results.csv", index=False)

    print("[5/7] Few-Shot Cross-Chemistry Transfer Learning (XGBoost Residual/Warm-Start & Two-Phase GRU)...")
    soc_tl = run_transfer_learning_experiments(cfg, soc_bundle)
    soh_tl = run_transfer_learning_experiments(cfg, soh_bundle)
    tl_df = pd.DataFrame(soc_tl.results_rows + soh_tl.results_rows)
    tl_df.to_csv(cfg.tables_dir / "transfer_learning_results.csv", index=False)

    print("[6/7] Simulated Adaptive Charging Strategy vs Conventional CC-CV Baseline...")
    li_soc_model = soc_bundle.within_tabular_phys["li_ion"]["XGBoost"]
    na_tl_soc_model = soc_tl.best_tl_li_to_na_predictor or soc_bundle.within_tabular_phys["na_ion"]["XGBoost"]
    li_soh_mae = soh_bundle.within_tabular_phys["li_ion"]["XGBoost"].test_metrics["mae"]
    na_soh_mae = soh_bundle.within_tabular_phys["na_ion"]["XGBoost"].test_metrics["mae"]

    charging_rows, charging_traces = run_adaptive_charging_benchmark(
        li_soc_predictor=lambda X: li_soc_model.predict(X),
        na_tl_soc_predictor=lambda X: na_tl_soc_model.predict(X),
        li_soh_mae=li_soh_mae,
        na_soh_mae=na_soh_mae,
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

    print("[7/7] Generating Publication Figures & Experiment Report...")
    figs = [
        plot_domain_alignment_pca(mining_results, cfg.figures_dir),
        plot_within_chemistry_benchmarks(soc_bundle, soh_bundle, cfg.figures_dir),
        plot_cross_chemistry_ablation(soc_bundle, soh_bundle, cfg.figures_dir),
        plot_transfer_learning_curves(soc_tl, soh_tl, cfg.figures_dir),
        plot_cross_chemistry_parity(soc_bundle, soh_bundle, cfg.figures_dir),
        plot_adaptive_charging_comparison(charging_rows, charging_traces, cfg.figures_dir),
    ]

    elapsed_s = round(time.perf_counter() - t_start, 2)
    summary = {
        "data_mode": data_mode,
        "elapsed_seconds": elapsed_s,
        "dataset_inventory": inv.to_dict(),
        "quality_audit": audit_report,
        "duckdb_analytics": duckdb_analytics,
        "data_mining": {k: v for k, v in mining_results.items() if k != "pca_payload"},
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
    )
    print(f"Done in {elapsed_s:.2f}s! Report written to: {report_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UROP Cross-Chemistry BMS Full Pipeline")
    parser.add_argument("--fast", action="store_true", help="Run reduced-size fast mode")
    parser.add_argument("--force-fallback", action="store_true", help="Force physics simulator fallback")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = PipelineConfig(seed=args.seed, fast_mode=args.fast)
    run_full_pipeline(cfg, force_fallback=args.force_fallback)


if __name__ == "__main__":
    main()
