"""Automated Markdown & JSON report generator mapping all outputs to UROP PDF Objectives 1-6."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a clean GitHub-flavored Markdown table without extra dependencies."""
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def generate_experiment_report(
    output_dir: Path,
    summary: dict,
    within_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    tl_df: pd.DataFrame,
    charging_df: pd.DataFrame,
) -> Path:
    """Write outputs/results_summary.json and outputs/EXPERIMENT_REPORT.md."""
    summary_path = output_dir / "results_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    data_mode = summary["data_mode"]
    inv = summary["dataset_inventory"]
    audit = summary["quality_audit"]
    mining = summary["data_mining"]

    # Highlight cross-chemistry improvement (XGBoost & GRU)
    cross_highlights = cross_df[cross_df["model"].isin(["XGBoost", "RandomForest", "GRU", "LSTM"])].copy()
    tl_highlights = tl_df[
        (tl_df["direction"] == "li_ion_to_na_ion") & (tl_df["target_fraction"].isin([0.0, 0.05, 0.10, 0.25]))
    ].copy()

    shift_df = pd.DataFrame(mining["domain_shift"])

    mode_banner = (
        "> **Data Mode:** `real_lfs_dataset` — Executed directly on pulled Git LFS NASA PCoE Li-ion and Wang et al. Na-ion datasets."
        if data_mode == "real_lfs_dataset"
        else (
            "> **Data Mode:** `physics_simulated_fallback` — The 42 Git LFS dataset files (~1.34 GB) in this checkout "
            "are currently 133-byte LFS pointer stubs (sandbox network cannot reach `github-cloud.githubusercontent.com`). "
            "All models and experiments below ran end-to-end on the repo's physics-informed electro-thermal-aging simulator "
            "matching the exact NASA 18650 (2.0 Ah, 2.5–4.2 V) and Transimage/HiNa Na-ion (3.2/10 Ah, 1.5–3.9 V) envelopes. "
            "Run `git lfs pull && python -m src.pipeline` locally to automatically switch to the real 1.34 GB datasets."
        )
    )

    sections = [
        "# UROP Cross-Chemistry BMS Experiment Report",
        "",
        "**Project:** *Data-Driven Battery State Estimation and Adaptive Charging Optimization Across Lithium-Ion and Sodium-Ion Batteries*",
        "",
        mode_banner,
        "",
        "---",
        "",
        "## 1. Big Data Lakehouse & Data Quality Audit (Objective 1 — Section 5.1)",
        "",
        f"- **Storage & Query Stack:** Polars (lazy streaming) + PyArrow/Apache Parquet (ZSTD Medallion Lakehouse: Bronze → Silver → Gold) + DuckDB SQL Analytics.",
        f"- **Repository LFS Inventory:** {inv['total_files']} dataset files ({inv['total_declared_mb']} MB declared; {inv['pulled_files']} pulled, {inv['pointer_files']} LFS pointers).",
        f"- **Time-Series Cleaning:** {audit['timeseries']['bronze_rows']:,} Bronze rows → {audit['timeseries']['silver_rows']:,} Silver rows "
        f"(removed {audit['timeseries']['duplicates_removed']} duplicates, {audit['timeseries']['bounds_and_null_removed']} null/out-of-bounds spikes, "
        f"{audit['timeseries']['isolation_forest_anomalies_removed']} IsolationForest anomalies; **{audit['timeseries']['retention_rate_pct']}%** retention).",
        f"- **Cycle-Aging Cleaning:** {audit['cycles']['bronze_rows']:,} Bronze cycles → {audit['cycles']['silver_rows']:,} Silver cycles.",
        "",
        "### DuckDB Big Data Chemistry Summary",
        _df_to_markdown(pd.DataFrame(summary["duckdb_analytics"]["chemistry_summary"])),
        "",
        "### DuckDB Aging Summary",
        _df_to_markdown(pd.DataFrame(summary["duckdb_analytics"]["aging_summary"])),
        "",
        "---",
        "",
        "## 2. Data Mining & Cross-Chemistry Domain-Shift Diagnostics",
        "",
        f"- **Unsupervised Operating-Regime Discovery (K-Means, k=4):** Silhouette Score = **{mining['kmeans_silhouette']}** across shared physics-normalized operating states.",
        "- **Quantitative Domain-Shift Reduction (Li-ion vs Na-ion):**",
        "",
        _df_to_markdown(shift_df),
        "",
        "![PCA Domain Alignment](figures/fig1_domain_alignment_pca.png)",
        "",
        "---",
        "",
        "## 3. Within-Chemistry Baselines (Objectives 2 & 3 — Sections 5.2–5.4)",
        "",
        "Strict **cell-level, temporally-ordered train/validation/test splits** (zero cell leakage). All 5 PDF model families evaluated on held-out test cells:",
        "",
        _df_to_markdown(
            within_df[["task", "chemistry", "model", "mae", "rmse", "r2", "train_time_s", "infer_time_ms"]]
        ),
        "",
        "![Within-Chemistry Benchmarks](figures/fig2_within_chemistry_benchmarks.png)",
        "",
        "---",
        "",
        "## 4. Making Cross-Chemistry Applicable: 3-Tier Unsupervised Alignment (Objective 4 — Section 5.5)",
        "",
        "Evaluated bidirectionally (**Li-ion → Na-ion** and **Na-ion → Li-ion**) with **ZERO target labels**:",
        "1. **Tier 1 (`1_raw_zero_shot`):** Raw voltage/current/resistance features (demonstrates severe voltage-window & capacity-scale domain shift).",
        "2. **Tier 2 (`2_physics_normalized_zero_shot`):** Dimensionless electrochemical features (`v_norm`, `c_rate`, IR-compensated `v_ir_comp_norm` for SoC; BoL-normalized health ratios for SoH).",
        "3. **Tier 3 (`3_physics_ocv_coral_zero_shot`):** Physics-normalized features + **Physics-Anchored OCV-to-OCV Equilibrium Warping** (mapping Na-ion's sloping OCV to Li-ion's flat plateau via static 1D OCV curves) + **Shrinkage CORAL Covariance Alignment**.",
        "",
        _df_to_markdown(cross_highlights[["task", "direction", "tier", "model", "mae", "rmse", "r2"]]),
        "",
        "![Cross-Chemistry Ablation](figures/fig3_cross_chemistry_ablation.png)",
        "",
        "![Cross-Chemistry Parity](figures/fig5_cross_chemistry_parity.png)",
        "",
        "---",
        "",
        "## 5. Few-Shot Cross-Chemistry Transfer Learning (Objective 4 — Section 5.6)",
        "",
        "Comparing **Zero-Shot Aligned**, **Target-Only**, **TL-Residual XGBoost**, **TL-WarmStart XGBoost**, and **Two-Phase Fine-Tuned GRU** on the exact same held-out Na-ion test cells:",
        "",
        _df_to_markdown(
            tl_highlights[
                ["task", "direction", "target_fraction", "n_target_samples", "method", "mae", "rmse", "r2"]
            ]
        ),
        "",
        "![Transfer Learning Curves](figures/fig4_transfer_learning_curves.png)",
        "",
        "---",
        "",
        "## 6. Simulated Adaptive Charging Strategy vs Conventional CC-CV (Objective 5 — Section 5.7)",
        "",
        "> *Disclaimer (Section 5.7):* Evaluated via coupled electro-thermal-aging simulation under chemistry-specific voltage and thermal safety constraints; not presented as physical hardware validation.",
        "",
        "For **Na-ion**, the adaptive controller is driven in closed loop by our **Li-ion → Na-ion Transfer-Learned SoC estimator** (connecting Objective 4 directly to Objective 5).",
        "",
        _df_to_markdown(
            charging_df[
                [
                    "chemistry",
                    "soh_true",
                    "strategy",
                    "charge_time_min",
                    "peak_temp_rise_c",
                    "energy_efficiency_pct",
                    "estimated_degradation_pct",
                    "soc_observer_mae",
                    "safety_limits_respected",
                ]
            ]
        ),
        "",
        "![Adaptive Charging vs CC-CV](figures/fig6_adaptive_vs_cccv_charging.png)",
        "",
    ]

    report_path = output_dir / "EXPERIMENT_REPORT.md"
    report_path.write_text("\n".join(sections), encoding="utf-8")
    return report_path
