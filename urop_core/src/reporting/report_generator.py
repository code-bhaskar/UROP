"""Auto-generate a strictly data-synchronized, audit-honest EXPERIMENT_REPORT.md and results_summary.json."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _df_to_md(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df is None or df.empty:
        return "_No records available._"
    sub = df.head(max_rows)
    cols = list(sub.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in sub.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
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
    cv_df: pd.DataFrame | None = None,
    na_soc_loco_df: pd.DataFrame | None = None,
    na_soh_lowo_df: pd.DataFrame | None = None,
    cross_per_unit_df: pd.DataFrame | None = None,
    external_ncm_df: pd.DataFrame | None = None,
) -> Path:
    """Write outputs/EXPERIMENT_REPORT.md and outputs/results_summary.json."""
    summary_path = output_dir / "results_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    inv = summary["dataset_inventory"]
    audit = summary["quality_audit"]
    mining = summary["data_mining"]
    unit_counts = summary.get("unit_counts", {})
    li_soc_n = unit_counts.get("SoC", {}).get("li_ion", 0)
    na_soc_n = unit_counts.get("SoC", {}).get("na_ion", 0)
    li_soh_n = unit_counts.get("SoH", {}).get("li_ion", 0)
    na_soh_n = unit_counts.get("SoH", {}).get("na_ion", 0)

    compat_notes = summary.get("compatibility_notes", [])
    compat_md = (
        "\n".join(f"> **Data-Scope & Compatibility Note:** {n}" for n in compat_notes)
        if compat_notes
        else ""
    )

    within_cols = [
        c
        for c in [
            "task",
            "chemistry",
            "evaluation_mode",
            "model",
            "val_rmse",
            "mae",
            "rmse",
            "rmse_std",
            "r2",
            "r2_std",
            "train_time_s",
        ]
        if c in within_df.columns
    ]
    within_view = within_df[within_cols].copy()

    cross_cols = [
        c
        for c in ["task", "direction", "evaluation_mode", "tier", "model", "mae", "rmse", "r2"]
        if c in cross_df.columns
    ]
    cross_view = cross_df[cross_cols].copy() if not cross_df.empty else pd.DataFrame()

    tl_cols = [
        c
        for c in [
            "task",
            "direction",
            "evaluation_mode",
            "target_fraction",
            "n_folds",
            "n_target_samples",
            "method",
            "rmse",
            "rmse_std",
            "r2",
            "adapt_time_s",
        ]
        if c in tl_df.columns
    ]
    tl_view = tl_df[tl_cols].copy() if not tl_df.empty else pd.DataFrame()

    chg_view = charging_df[
        [
            "chemistry",
            "scenario",
            "strategy",
            "charge_time_min",
            "peak_temp_c",
            "peak_temp_rise_c",
            "energy_efficiency_pct",
            "estimated_degradation_pct",
            "soc_observer_mae",
            "safety_limits_respected",
        ]
    ].copy()

    shift_df = pd.DataFrame(mining["domain_shift"])
    aging_df = pd.DataFrame(summary["duckdb_analytics"].get("aging_summary", []))

    soc_cross_per_unit = (
        cross_per_unit_df[cross_per_unit_df["task"] == "SoC"]
        if cross_per_unit_df is not None and not cross_per_unit_df.empty
        else pd.DataFrame()
    )
    soh_cross_per_unit = (
        cross_per_unit_df[cross_per_unit_df["task"] == "SoH"]
        if cross_per_unit_df is not None and not cross_per_unit_df.empty
        else pd.DataFrame()
    )

    md = f"""# UROP Cross-Chemistry BMS Framework — Experimental Report

**Project:** Machine Learning-Based State Estimation and Cross-Chemistry Transfer Learning for Battery Management Systems (Li-ion ↔ Na-ion)
**Pipeline Runtime:** `{summary['elapsed_seconds']} s` | **Data Mode:** `{summary['data_mode']}`
**Real Laboratory Corpus:** `{inv['pulled_files']}/{inv['total_files']}` files present (`{inv['total_declared_mb']} MB`)

{compat_md}

---

## 1. Columnar Out-of-Core Data Pipeline, Provenance & Quality Audit (Objectives 1 & 2)

All raw laboratory files are ingested into a local **Bronze → Silver → Gold Parquet** lakehouse using Polars and DuckDB out-of-core queries. See [`urop_core/DATA_NOTES.md`](../DATA_NOTES.md) for raw file inspections and provenance proof.

- **SoC Time-Series Sources (`ordered_per_cell`):**
  - **NASA Prognostics Li-ion (`5. Battery Data Set/`):** `{li_soc_n}` ordered cells (`B0005`–`B0056`).
  - **Wang et al. (2025) Na-ion (`3.2Ah na-ion/` & `10Ah na-ion/`):** `{na_soc_n}` ordered cells across 6 ambient temperatures (`0 °C` to `50 °C`): `bat1`/`bat2` (3.2 Ah pulse/stepped) and `bat5` (3.2 Ah driving cycles); `bat3`/`bat4` (10 Ah pulse/stepped) and `bat6` (10 Ah driving cycles).
- **SoH Degradation Sources:**
  - **NASA Li-ion (`ordered_per_cell`):** `{li_soh_n}` independent cells with genuine sequential `cycle_number`. Used exclusively for within-chemistry Li-ion SoH training/validation/testing (tabular + LSTM/GRU).
  - **Mendeley `10.17632/4mztcdc4gt.1` (`pooled_unordered`):** 3 workbooks (`Dataset 1-1.xlsx` → `na_nfm_25C`, `Dataset 1-2.xlsx` → `na_nfm_m15C`, `Dataset 2.xlsx` → `li_ncm_25C`). Raw file diagnostics confirmed **no cell IDs**, **no cycle numbers**, strictly non-increasing capacity sorting in `Dataset 1-2` and `Dataset 2` (`0` upward steps), and **non-aligned Discharge vs Charge sheets** (`corr(diff(Q_max)) ≈ 0`). Each workbook is therefore loaded as **one Discharge-only unit** with `cycle_number = None` and `data_provenance = "pooled_unordered"`. Sequence models (LSTM/GRU) are strictly refused on pooled workbooks; `li_ncm_25C` is held out as an external pooled Li-ion test set and never mixed into NASA training.

### Lakehouse Quality Audit & Aggregation Summary
- **Time-Series Layer:** `{audit['timeseries']['bronze_rows']:,}` Bronze rows → `{audit['timeseries']['silver_rows']:,}` Silver rows (`{audit['timeseries']['retention_rate_pct']}%` retention; `{audit['timeseries'].get('bounds_and_null_removed', 0)}` bounds/null-filtered, `{audit['timeseries'].get('isolation_forest_anomalies_removed', 0)}` IsolationForest-filtered, `{audit['timeseries'].get('duplicates_removed', 0)}` duplicates removed).
- **Cycle / Degradation Layer:** `{audit['cycles']['bronze_rows']:,}` Bronze records → `{audit['cycles']['silver_rows']:,}` Silver records (`{round(audit['cycles']['silver_rows'] / max(audit['cycles']['bronze_rows'], 1) * 100.0, 2)}%` retention; `{audit['cycles'].get('removed', 0)}` removed).

{_df_to_md(aging_df)}

### SoH Feature Scope & Cross-Dataset Mismatch Note
- **Partial-window discharge durations (`t_vnorm_upper_s`, `t_vnorm_mid_s`, `t_vnorm_lower_s`):** On Mendeley 1C constant-current discharge curves, time spent traversing fixed normalized voltage windows is proportional to incremental capacity $\\Delta Q / I$ in those windows — an **indirect health indicator** (similar to ICA/DVA sub-window features), not a leakage-free independent proxy.
- **NASA vs Mendeley protocol mismatch:** NASA discharge cycles are measured at 2 A with varying cell cutoff voltages (2.0–2.7 V) and real temperature sensors, whereas Mendeley workbooks provide 1C fixed-window Discharge `Q(V)` curves without cell temperature telemetry. Cross-chemistry SoH transfer therefore combines both electrochemistry shift and test-protocol mismatch.

---

## 2. Unsupervised Data Mining & Domain-Shift Analysis (Objective 2)

- **Operating-Regime K-Means Clustering (4 regimes, physics-normalized SoC space):** Silhouette score = `{mining['kmeans_silhouette']}`.
- **Domain Shift Reduction (NASA Li-ion vs Na-ion; SoH comparison is exploratory against pooled Mendeley NFM):**

{_df_to_md(shift_df)}

![Domain Alignment PCA](figures/fig1_domain_alignment_pca.png)

---

## 3. Within-Chemistry Benchmarks & 6-Fold Na-ion SoC LOCO (Objective 3)

- **Li-ion SoC & SoH:** Cell-level train/val/test split over the `{li_soc_n}` ordered NASA cells (no cell ever crosses train/val/test).
- **Na-ion SoC (`6-fold Leave-One-Cell-Out`):** Each of the `{na_soc_n}` Wang et al. cells (all 6 temperatures kept together) is held out once as the test cell while training on 4 cells and validating on 1 cell. Sequence models (LSTM/GRU) use reduced epochs (`6` epochs per LOCO fold; `4` head + `5` full epochs in 6-fold LOCO Transfer Learning) for tractable CPU runtime across all 6 folds. Note protocol dependence: `bat1`/`bat2` (3.2 Ah) and `bat3`/`bat4` (10 Ah) share pulse/stepped profiles, whereas `bat5` and `bat6` carry driving-cycle profiles.
- **Na-ion SoH (`exploratory_lowo_temp_confounded`):** 2-fold Leave-One-Workbook-Out (`na_nfm_25C` at `25 °C` ↔ `na_nfm_m15C` at `-15 °C`), **tabular models only** (LSTM/GRU disabled on pooled unordered data). Temperature and workbook identity are confounded.

### 3.1 Summary Within-Chemistry Benchmarks

{_df_to_md(within_view)}

### 3.2 Na-ion SoC 6-Fold Leave-One-Cell-Out — Per-Cell Breakdown

{_df_to_md(na_soc_loco_df)}

![Na-ion SoC 6-Fold LOCO Per-Cell RMSE](figures/fig7_na_soc_loco_per_cell.png)

### 3.3 External Pooled Li-ion SoH Test Set (NASA-Trained Models → Mendeley NCM `li_ncm_25C`)

{_df_to_md(external_ncm_df)}

### 3.4 Exploratory Na-ion SoH 2-Fold Leave-One-Workbook-Out (Temperature-Confounded, Tabular Only)

{_df_to_md(na_soh_lowo_df)}

### 3.5 Repeated Cell-Level Split CV (Ordered Cells Only, Mean ± Std)

{_df_to_md(cv_df)}

![Within-Chemistry Benchmarks](figures/fig2_within_chemistry_benchmarks.png)

---

## 4. Bidirectional Cross-Chemistry Zero-Shot Evaluation (Objective 4)

- **SoC `Li-ion → Na-ion`:** Evaluated jointly across **all `{na_soc_n}` held-out Na-ion cells** as one target test set. Tier 3 unsupervised CORAL alignment uses unlabeled target input statistics (**transductive** alignment; zero target labels are used).
- **SoH `Li-ion ↔ Na-ion` (`exploratory_pooled`, tabular only):** Evaluated strictly with tabular models across ordered NASA Li-ion cells and the 2 pooled Mendeley NFM workbooks (sequence models omitted because pooled rows have no temporal order).

### 4.1 3-Tier Cross-Chemistry Ablation Table

{_df_to_md(cross_view)}

### 4.2 Per-Cell Breakdown: `Li-ion → Na-ion` SoC (XGBoost across all {na_soc_n} held-out cells)

{_df_to_md(soc_cross_per_unit)}

### 4.3 Per-Workbook Breakdown: `Li-ion → Na-ion` SoH (Exploratory XGBoost across both NFM workbooks)

{_df_to_md(soh_cross_per_unit)}

![Cross-Chemistry 3-Tier Ablation](figures/fig3_cross_chemistry_ablation.png)
![Cross-Chemistry Parity](figures/fig5_cross_chemistry_parity.png)

---

## 5. Few-Shot Cross-Chemistry Transfer Learning (`Li-ion → Na-ion`, Objective 4)

- **SoC Transfer Learning (`6-fold LOCO`):** For each held-out Na-ion test cell `c`, adaptation uses `5%, 10%, 25%, 50%` of the *other 5* Na-ion cells (strictly asserting `c` never enters adaptation or empirical OCV estimation), averaged across all 6 folds.
- **SoH Transfer Learning (`exploratory_cross_workbook_temp_confounded`, tabular only):** Adaptation samples `5%, 10%, 25%, 50%` strictly from Workbook A (`25 °C` or `-15 °C`) and evaluates on 100% of Workbook B (opposite temperature), averaged across both directions. No intra-workbook random splits and no sequence models are used.

{_df_to_md(tl_view)}

![Transfer Learning Curves](figures/fig4_transfer_learning_curves.png)

---

## 6. Adaptive Charging Simulation Study vs CC-CV Baseline (Proposal §5.7, Objective 5)

> **Simulation Study Note (Proposal §5.7):** These results come from a closed-loop **simulation study** on an independent 2-RC electro-thermal plant with deliberate controller-plant mismatch (+12% resistance offset, unmodeled slow diffusion RC polarization, charging hysteresis, and 0.5% current bias) using empirical training-cell OCV curves. They are **not** physical hardware bench measurements. The closed-loop `soc_observer_mae` is reported beside every metric so observer error is transparent.

{_df_to_md(chg_view)}

![Adaptive vs CC-CV Charging](figures/fig6_adaptive_vs_cccv_charging.png)

---

## 7. Limitations

1. **Mendeley SoH Workbooks Lack Cell Identities and Cycle Order (`pooled_unordered`):** Raw file inspection proved `Dataset 1-1.xlsx`, `Dataset 1-2.xlsx`, and `Dataset 2.xlsx` contain no cell IDs or cycle numbers, have non-aligned Discharge/Charge sheets, and are sorted by capacity in `Dataset 1-2` and `Dataset 2`. Consequently, Na-ion SoH has only **2 independent workbook units** (`25 °C` vs `-15 °C`, confounded with temperature), preventing cell-level CV, sequence modeling (LSTM/GRU), or cycle-rate fade claims. All Na-ion SoH and cross-chemistry SoH results must be interpreted as **exploratory**.
2. **NASA vs Mendeley SoH Feature & Protocol Mismatch:** NASA discharge cycles use 2 A constant current across varying cell cutoff voltages with real cell temperature sensors, whereas Mendeley workbooks provide 1C fixed-window discharge `Q(V)` curves without temperature telemetry. Partial-window 1C durations act as indirect capacity-correlated health indicators rather than protocol-invariant physical constants.
3. **Small Na-ion SoC Cell Count (`N = 6`) & Protocol Coupling:** Although 6-fold Leave-One-Cell-Out eliminates the variance of a single 2-cell split, 6 cells remain a small sample where `bat1`/`bat2` (3.2 Ah) and `bat3`/`bat4` (10 Ah) share stepped-pulse protocols while `bat5`/`bat6` cover driving cycles; generalization to unseen Na-ion form factors or cathode chemistries (beyond Wang et al.) is not established here.
4. **Transductive Nature of Unsupervised CORAL Alignment:** In Tier 3 zero-shot `Li-ion → Na-ion` evaluation over all 6 held-out Na-ion cells, unsupervised CORAL computes covariance alignment using unlabeled target-domain inputs. While zero target labels are used, online single-cell deployment before observing a full target voltage/current distribution would require streaming or batch warmup.
5. **Simulation-Only Adaptive Charging (Proposal §5.7):** Objective 5 evaluates ML/TL-informed charging inside an electro-thermal 2-RC simulation plant rather than on physical cycler hardware; thermal reductions, degradation index improvements, and zero safety-limit violations are conditional on the 2-RC plant model and must be interpreted alongside each scenario's closed-loop `soc_observer_mae`.
"""

    report_path = output_dir / "EXPERIMENT_REPORT.md"
    report_path.write_text(md, encoding="utf-8")
    return report_path
