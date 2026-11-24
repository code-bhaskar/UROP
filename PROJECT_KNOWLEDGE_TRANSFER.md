# UROP Cross-Chemistry BMS Framework — Complete Project Knowledge Transfer & Technical Documentation

**Repository:** `code-bhaskar/UROP`
**Project Title:** Machine Learning-Based State Estimation and Cross-Chemistry Transfer Learning for Battery Management Systems (Li-ion ↔ Na-ion)
**Proposal Reference:** `AP23110011334_24230_UROP_CLA.pdf`
**Last Verified Pipeline Run:** `cd urop_core && python -m src.pipeline` (`744.14 s` on 2 vCPUs, 45/45 real laboratory files, `16/16` pytest passed, `tsc -b && vite build` passed)

---

## 1. Executive Summary & Research Purpose

This repository implements an end-to-end, reproducible research software platform investigating **State-of-Charge (SoC)** estimation, **State-of-Health (SoH)** estimation, **cross-chemistry generalization (Li-ion ↔ Na-ion)**, **few-shot transfer learning**, and **ML-informed adaptive charging simulation** for Battery Management Systems (BMS).

A core challenge in next-generation Sodium-ion (Na-ion) battery deployment is **data scarcity** compared to mature Lithium-ion (Li-ion) datasets, compounded by fundamental electrochemical differences:
- **Operating voltage windows differ:** Li-ion operates around `2.5–4.2 V` with a relatively flat mid-SoC Open-Circuit Voltage (OCV) plateau, whereas layered-oxide Na-ion operates around `1.5–3.9 V` (Wang et al.) or `2.0–4.0 V` (Mendeley NFM) with a steeper, sloping OCV curve and multi-phase transitions.
- **Cross-chemistry models fail naively:** Applying a model trained on raw Li-ion electrical measurements directly to Na-ion cells yields negative $R^2$ (`XGBoost RMSE = 0.3765`, $R^2 = -0.6222$).
- **Physics + Unsupervised Alignment + Few-Shot Transfer recovers performance:** Physics-normalized representations, data-driven empirical OCV-to-OCV warping, shrinkage CORAL covariance alignment, and two-phase GRU transfer learning recover zero-shot `Li → Na` SoC to `RMSE = 0.1598–0.1642` ($R^2 = +0.69–0.71$) and 5%–50% few-shot `TL_TwoPhase_GRU` to `RMSE = 0.0454 → 0.0348` across 6-fold Leave-One-Cell-Out (LOCO) validation.

### Non-Negotiable Scientific Principles Built Into This Repo
1. **Real Laboratory Data Only (No Synthetic Training Data):** There is zero synthetic dataset generator anywhere in the repository (`test_no_synthetic_data_generators_in_repo` enforces this via AST inspection).
2. **Strict Provenance Honesty (No Fabricated Structure):** Every dataset record is tagged with `data_provenance` (`"ordered_per_cell"` vs `"pooled_unordered"`). Where raw files lack cell IDs or cycle order (Mendeley SoH workbooks), the pipeline refuses to invent sub-cells or cycle indices and blocks sequence models (`LSTM`/`GRU`) at runtime.
3. **Cell/Workbook-Level Split Integrity:** A hard runtime guard (`assert_no_cell_overlap`) verifies before every fit call that zero test cell or workbook ID ever enters training, validation, scaling, or empirical OCV estimation.
4. **Decoupled Architecture:** `urop_core/` is a standalone Python research package with zero web dependencies. `fullstack_3d/` is strictly a read-only presentation layer consuming `urop_core/outputs/` plus an interactive Proposal §5.7 charging simulator.

---

## 2. Repository Architecture & Directory Map

```text
UROP/
├── 5. Battery Data Set/                  # NASA PCoE Li-ion (6 nested ZIP archives, 33 ordered cells)
├── 3.2Ah na-ion/                         # Wang et al. (2025) 3.2 Ah Na-ion (18 Arbin CSVs, 3 cells × 6 temps)
├── 10Ah na-ion/                          # Wang et al. (2025) 10 Ah Na-ion (18 Arbin CSVs, 3 cells × 6 temps)
├── Dataset 1-1.xlsx                      # Mendeley 10.17632/4mztcdc4gt.1 NFM Na-ion 25 °C (1,528 rows)
├── Dataset 1-2.xlsx                      # Mendeley 10.17632/4mztcdc4gt.1 NFM Na-ion -15 °C (1,211 rows)
├── Dataset 2.xlsx                        # Mendeley 10.17632/4mztcdc4gt.1 NCM Li-ion 25 °C (3,351 rows)
├── AP23110011334_24230_UROP_CLA.pdf      # Original UROP proposal document (compiled binary PDF)
├── README.md                             # Entry-point documentation & quickstart
├── PROJECT_KNOWLEDGE_TRANSFER.md         # This complete handoff handbook
├── .gitattributes                        # Binary markers for .zip/.xlsx (plain Git, NO Git LFS)
├── .github/workflows/ci.yml              # CI: pytest in urop_core + npm ci && npm run build in fullstack_3d
│
├── urop_core/                            # Standalone ML & Data Engineering Package
│   ├── DATA_NOTES.md                     # Verbatim Step 1 raw file diagnostics & provenance proof
│   ├── requirements.txt                  # Python dependencies (polars, duckdb, pyarrow, torch, xgboost, etc.)
│   ├── pyproject.toml                    # Pytest configuration
│   ├── data_lake/                        # Generated local Parquet lakehouse (Bronze/Silver/Gold + .npz cache)
│   ├── outputs/                          # Committed, synchronized outputs from the latest real run
│   │   ├── EXPERIMENT_REPORT.md          # Auto-generated full scientific report (including Limitations)
│   │   ├── results_summary.json          # Single source of truth consumed by report + fullstack_3d
│   │   ├── charging_traces.json          # Downsampled closed-loop charging trajectories
│   │   ├── empirical_ocv_curves.json     # Training-cell-fitted empirical OCV(SoC) curves
│   │   ├── pca_manifold.json             # 3D PCA point clouds (Raw vs Aligned) for WebGL morph
│   │   ├── tables/                       # 15 structured CSV tables
│   │   └── figures/                      # 7 publication figures (fig1–fig7)
│   ├── src/
│   │   ├── config.py                     # PipelineConfig, ChemistrySpec, path resolution
│   │   ├── pipeline.py                   # 7-stage CLI orchestrator (`python -m src.pipeline`)
│   │   ├── data/
│   │   │   ├── lfs_check.py              # Dataset discovery & completeness audit (45 files, 1,357.11 MB)
│   │   │   ├── real_loaders.py           # NASA .mat, Wang Arbin CSV, and Mendeley Discharge-only loaders
│   │   │   └── lakehouse.py              # Bronze/Silver Parquet ingestion + DuckDB SQL aggregations
│   │   ├── mining/
│   │   │   ├── quality_audit.py          # Bronze -> Silver cleaning (nulls, dedup, bounds, IsolationForest)
│   │   │   ├── feature_engineering.py    # Silver -> Gold physics-normalized SoC & SoH features
│   │   │   ├── domain_mining.py          # K-Means regimes, Mutual Information, PCA, MMD^2/W1/PSI
│   │   │   └── splits.py                 # Strict cell-level train/val/test partitioning
│   │   ├── models/
│   │   │   ├── metrics.py                # MAE, RMSE, R^2 regression metrics
│   │   │   ├── tabular.py                # LinearRegression (Ridge), RandomForest, XGBoost
│   │   │   └── sequence.py               # PyTorch LSTM/GRU, segmented windows (refuses pooled_unordered), 2-phase TL
│   │   ├── cross_chemistry/
│   │   │   ├── alignment.py              # Empirical OCV estimation, OCV-to-OCV warp, Shrinkage CORAL
│   │   │   ├── evaluation.py             # Within-chem, 6-fold LOCO, 2-fold LOWO, 3-tier cross-chem, Repeated CV
│   │   │   └── transfer_learning.py      # 6-fold LOCO SoC TL + exploratory cross-workbook SoH TL
│   │   ├── charging/
│   │   │   └── adaptive_charging.py      # Proposal §5.7 2-RC mismatched electro-thermal charging simulation
│   │   └── reporting/
│   │       ├── plots.py                  # Matplotlib/Seaborn generator for fig1–fig7
│   │       └── report_generator.py       # Auto-generates EXPERIMENT_REPORT.md + results_summary.json
│   └── tests/                            # 16 automated pytest tests (no external downloads needed)
│       ├── conftest.py                   # Real-format fixture generators (NASA ZIP, Arbin CSV)
│       ├── test_real_loaders.py          # Loader correctness, Mendeley pooled guard, sorted-order detector, AST audit
│       ├── test_lakehouse_and_mining.py  # Medallion lakehouse, quality audit, domain-shift metrics
│       ├── test_cross_chemistry_and_models.py # Alignment, LOCO fixture, all-experiment leakage spy, pooled tabular guard
│       └── test_adaptive_charging.py     # Charging convergence, safety limits, degradation reduction
│
└── fullstack_3d/                         # Decoupled Read-Only 3D Web Dashboard
    ├── backend/
    │   ├── requirements.txt              # FastAPI, Uvicorn, Pydantic
    │   └── app/
    │       ├── main.py                   # REST API endpoints (/api/*)
    │       ├── artifacts.py              # Read-only loader for urop_core/outputs/
    │       └── live_simulator.py         # Live interactive Objective 5 2-RC simulation endpoint
    └── frontend/
        ├── package.json                  # React 19, TypeScript, Vite, Three.js, Recharts, Lucide
        └── src/
            ├── App.tsx                   # Main 3-tab layout & backend data loader
            ├── api.ts                    # Typed fetch client
            ├── types.ts                  # TypeScript interfaces matching results_summary.json
            ├── styles.css                # Dark scientific dashboard styling
            └── components/
                ├── AdaptiveChargingLab.tsx    # Tab 1: 3D battery cell + live 2-RC simulator
                ├── BatteryCell3D.tsx          # Three.js interactive thermal/SoC battery cell mesh
                ├── CrossChemistryView.tsx     # Tab 2: 3D PCA morph + 3-tier ablation + TL curves + SoH caveats
                ├── CrossChemManifold3D.tsx    # Three.js 1,000-point morphing PCA point cloud
                └── ModelsAndDataView.tsx      # Tab 3: Lakehouse KPIs, benchmarks, MI rankings, fig1–fig7 gallery
```

---

## 3. Laboratory Datasets & Provenance Audit

All 45 real laboratory files (`1,357.11 MB`) are committed directly as plain Git files (approximately `1.8 GB` full clone; **Git LFS is not used**). Custom external dataset locations are supported via `--dataset-root /path/to/datasets` or `export UROP_DATASET_ROOT=/path/to/datasets`.

| Dataset | Location | Files | Independent Units | Provenance Tag | Scientific Role & Restrictions |
| --- | --- | --- | --- | --- | --- |
| **NASA PCoE Li-ion** | `5. Battery Data Set/` | 6 ZIPs (`B0005`–`B0056`) | **33 ordered cells** | `ordered_per_cell` | Li-ion SoC (28,220 Silver rows) and Li-ion SoH (2,410 ordered cycles). Full tabular + LSTM/GRU modeling. |
| **Wang et al. (2025) Na-ion** (Zenodo `13836819`) | `3.2Ah na-ion/`, `10Ah na-ion/` | 36 Arbin CSVs (6 cells × 6 temps: `0, 10, 20, 30, 40, 50 °C`) | **6 ordered cells** (`bat1`, `bat2`, `bat5` @ 3.2 Ah; `bat3`, `bat4`, `bat6` @ 10 Ah) | `ordered_per_cell` | Na-ion SoC (21,525 Silver rows). Single-cycle characterization (no multi-cycle aging). Evaluated via **6-fold LOCO** (all 6 temperatures of a cell kept together). |
| **Mendeley NFM Na-ion** (`10.17632/4mztcdc4gt.1`) | `Dataset 1-1.xlsx` (25 °C), `Dataset 1-2.xlsx` (-15 °C) | 2 Excel workbooks | **2 pooled workbooks** (`na_nfm_25C`, `na_nfm_m15C`, 2,739 rows) | `pooled_unordered` | Exploratory Na-ion SoH only. Discharge sheet only. `cycle_number = None`. **Tabular models only** (LSTM/GRU disabled). Confounded with temperature (`25 °C` vs `-15 °C`). |
| **Mendeley NCM Li-ion** (`10.17632/4mztcdc4gt.1`) | `Dataset 2.xlsx` (25 °C) | 1 Excel workbook | **1 pooled workbook** (`li_ncm_25C`, 3,351 rows) | `pooled_unordered` | Held out strictly as an **external pooled Li-ion SoH test set**; never mixed into the 33-cell NASA training set. |

### Key Mendeley Provenance Findings (Full Tables in `urop_core/DATA_NOTES.md`)
Raw `openpyxl(read_only=True, data_only=True)` inspection established **Case A (`pooled_unordered`)**:
1. **No cell IDs and no cycle index columns** exist in any of the three workbooks.
2. **Capacity/SoH sorting destroys cycle order:** `Dataset 1-2.xlsx` (`1,211` rows, `SOH` `0.9998 → 0.8944`) and `Dataset 2.xlsx` (`3,351` rows, `Q_max` `1.4878 → 1.1246 Ah`) have **`0` upward steps (`0.00%`)** — strictly non-increasing, proving post-hoc capacity sorting rather than noisy per-cell cycling.
3. **Discharge and Charge sheets are not row-aligned:** First-difference correlations across Discharge vs Charge sheets are near zero (`corr(diff(Q_max)) = +0.0287` in `Dataset 1-1` and `-0.0350` in `Dataset 1-2`), and Discharge vs Charge `SOH` columns in `Dataset 1-2` disagree row-by-row. Therefore, **all Charge-sheet and cross-sheet hysteresis features were deleted**, and the loader reads **Discharge sheets exclusively**.
4. **Caching:** Parsed Mendeley Discharge arrays are cached under `urop_core/data_lake/cache/*.npz` to avoid repeated slow `openpyxl` XML parsing across runs.

---

## 4. Detailed 7-Stage Pipeline Walkthrough (`urop_core/src/pipeline.py`)

Running `cd urop_core && python -m src.pipeline` executes seven sequential stages:

### Stage 1 — Columnar Out-of-Core Medallion Lakehouse (`data/lakehouse.py`, `data/real_loaders.py`, `data/lfs_check.py`)
- Audits all 45 dataset files (`1,357.11 MB`) via `inspect_dataset_inventory()`.
- **NASA loader (`load_real_nasa_mat_zips`):** Reads nested `.zip` archives and `.mat` structs via `scipy.io.loadmat`; trims post-cutoff `I = 0` rebound tails using `last(|I| > 0.15 A)`; computes exact discharge Coulomb-counted SoC, initial load-step internal resistance, smoothed ICA peak (`dQ/dV`), and Discharge-only partial-voltage-window transit durations (mid 25–75%, upper 45–90%, tail 75–100%).
- **Wang et al. Na-ion loader (`load_real_na_ion_csvs`):** Uses Polars with `infer_schema_length=0` (preventing schema nullification after 900 s rest steps), computes temperature-calibrated Coulomb-counted SoC from the built-in 0.2C full calibration discharge before even downsampling (~600 rows/file), and emits zero fake SoH records for single-cycle tests.
- **Mendeley loader (`load_real_mendeley_degradation_xlsx`):** Uses `detect_pooled_sorted_order()`, loads Discharge sheets only, assigns 1 unit per workbook (`cycle_number = None`, `data_provenance = "pooled_unordered"`), and computes matching Discharge-only partial-voltage-window transit durations + Discharge-only polarization proxies.
- Writes ZSTD-compressed **Bronze Parquet** tables (`timeseries.parquet`, `cycles.parquet`).

### Stage 2 — Data Mining Quality Audit & DuckDB SQL Analytics (`mining/quality_audit.py`, `data/lakehouse.py`)
- Deduplicates time-series on `(chemistry, cell_id, protocol, cycle_number, time_s)` and cycles on `(chemistry, cell_id, sample_idx)` (preserving `cycle_number = None` pooled rows).
- Removes out-of-bounds voltages/temperatures/SoC and applies `IsolationForest` outlier detection per chemistry (`49,938 → 49,745` Silver time-series rows, `99.61%` retention; `8,500` Silver degradation records).
- Runs DuckDB SQL aggregations directly on Silver Parquet to produce `bigdata_chemistry_summary.csv`, `bigdata_protocol_temp_summary.csv`, and `bigdata_aging_summary.csv` (grouped by `(chemistry, data_provenance)`, reporting `n_units`, `n_records`, `min/mean/max_soh`, with zero invented fade-per-cycle stats).

### Stage 3 — Physics-Normalized Feature Engineering & Unsupervised Domain Mining (`mining/feature_engineering.py`, `mining/domain_mining.py`)
- **Chemistry-Invariant SoC Features (`PHYSICS_SOC_FEATURES`, 11 features):**
  - `v_norm = (V - V_min) / (V_max - V_min)`, `c_rate = I / Q_nom`, `temp_norm = (T - 25) / 25`, `temp_rise_norm`, `dv_dt_norm`, `di_dt_norm`, `v_norm_roll_mean`, `v_norm_roll_std`, `c_rate_roll_mean`, `power_norm`, `dv_over_dc`.
- **Chemistry-Invariant SoH Features (`PHYSICS_SOH_FEATURES`, 10 features — `log_cycle_norm` removed):**
  - `dis_dur_ratio`, `cc_dur_ratio`, `cv_dur_ratio`, `cv_cc_ratio`, `r_int_ratio`, `ica_peak_ratio`, `ica_v_shift`, `temp_rise_ratio`, `v_norm_int_ratio`, `temp_norm`.
  - **Provenance-safe Beginning-of-Life (BoL) baselines:** For `ordered_per_cell` cells (NASA), BoL medians come from `cycle_number <= 3`. For `pooled_unordered` workbooks (Mendeley), baselines use **purely unsupervised input feature quantiles** (e.g., 95th percentile of input durations, 5th percentile of resistance) — zero target (`soh`) or row-order dependence!
  - **Important feature note:** Partial-window 1C durations act as **indirect health indicators** proportional to incremental sub-window capacity, and NASA (2 A varying cutoffs + real temp telemetry) vs Mendeley (1C fixed windows, no cell temp sensors) introduce cross-protocol shift.
- **Unsupervised Domain Mining:**
  - 4-cluster K-Means operating-regime clustering (`Silhouette = 0.3501`).
  - Mutual Information feature rankings per chemistry & task.
  - Quantitative domain-shift metrics (**RBF-kernel $\text{MMD}^2$**, **Mean Wasserstein-1**, **Mean PSI**) across Raw → Physics-Normalized → Physics + Empirical OCV + CORAL, plus 3D PCA projections exported to `pca_manifold.json`.

### Stage 4 — Within-Chemistry, 6-Fold LOCO, Exploratory LOWO, 3-Tier Cross-Chemistry & Repeated CV (`cross_chemistry/evaluation.py`, `cross_chemistry/alignment.py`, `models/`)
- **Five model families:** `LinearRegression` (Ridge), `RandomForest`, `XGBoost`, PyTorch `LSTM`, and PyTorch `GRU`.
- **Sequence-model provenance guard (`models/sequence.py`):** `build_segmented_sequences()` inspects `data_provenance` and raises `ValueError` if `"pooled_unordered"` is present.
- **Data-Driven Empirical OCV Alignment (`cross_chemistry/alignment.py`):**
  - `estimate_empirical_ocv_curve()` fits monotonic 1D `OCV(SoC)` curves strictly on **training cells only** (preferring low-current rest points when spanning $\ge 0.55$ SoC, otherwise IR-compensated binned voltages + isotonic regression).
  - `CrossChemistryDomainAdapter` applies (1) linear physics normalization, (2) empirical OCV-to-OCV warping, and (3) shrinkage CORAL covariance alignment (Sun et al., AAAI 2016).
- **Evaluation protocols:**
  - **Li-ion SoC & SoH:** Cell-level train/val/test splits over the 33 ordered NASA cells (both tabular & LSTM/GRU) + 3-seed Repeated Cell-Level CV. Pooled Mendeley NCM (`li_ncm_25C`) is evaluated strictly out-of-sample as an external test set (`external_ncm_soh_test.csv`).
  - **Na-ion SoC:** **6-fold Leave-One-Cell-Out (LOCO)** across all 6 Wang et al. cells (all 6 temperatures of a held-out cell kept together; 4 train + 1 val; sequence models run `6` epochs per LOCO fold for tractable CPU runtime), emitting `na_soc_loco_per_cell.csv` and `fig7_na_soc_loco_per_cell.png`.
  - **Na-ion SoH (`exploratory_pooled`):** **2-fold Leave-One-Workbook-Out (LOWO)** (`na_nfm_25C` at `25 °C` ↔ `na_nfm_m15C` at `-15 °C`), **tabular models only** (`na_soh_lowo_folds.csv`).
  - **Bidirectional 3-Tier Cross-Chemistry Zero-Shot:**
    - `Li → Na` SoC evaluates **all 6 Na-ion cells jointly** (transductive unsupervised CORAL) AND reports per-cell **single-unit transductive CORAL** in `cross_chemistry_per_unit.csv`.
    - `Li ↔ Na` SoH runs strictly in **exploratory tabular-only mode** across NASA ordered cells and the 2 Mendeley NFM workbooks.

### Stage 5 — Few-Shot Cross-Chemistry Transfer Learning (`cross_chemistry/transfer_learning.py`)
- **SoC (`Li-ion → Na-ion`, 6-fold LOCO):** For each held-out Na-ion test cell `c`, adapts on `5%, 10%, 25%, 50%` of the *other 5* cells (fitting fold-specific empirical target OCV and CORAL strictly on the 5 adaptation cells, asserting `c` never enters adaptation). Compares `ZeroShot_Aligned`, `TargetOnly_XGBoost`, `TL_Residual_XGBoost`, `TL_WarmStart_XGBoost`, `TargetOnly_GRU`, and `TL_TwoPhase_GRU` (`4` head + `5` full epochs per LOCO fold).
- **SoH (`Li-ion → Na-ion`, Exploratory Cross-Workbook, Tabular Only):** Adapts on `5%, 10%, 25%, 50%` of Workbook A and tests on 100% of Workbook B across both directions (`25 °C ↔ -15 °C`). Zero same-workbook random splits and zero sequence models.

### Stage 6 — Proposal §5.7 Adaptive Charging Simulation Study (`charging/adaptive_charging.py`)
- Compares a multi-stage health/thermal-aware ML/TL adaptive charging controller against conventional 1.0C CC-CV inside an **independent 2-RC electro-thermal plant** with deliberate plant-to-controller mismatch (`+12%` resistance mismatch, slow diffusion RC polarization, charging hysteresis, `0.5%` current sensor bias) driven by training-cell-fitted empirical OCV curves.
- Explicitly framed as a **simulation study** (not physical cycler bench hardware), reporting closed-loop `soc_observer_mae` alongside every thermal, efficiency, and degradation metric.

### Stage 7 — Publication Figures & Synchronized Report (`reporting/plots.py`, `reporting/report_generator.py`)
- Generates all 7 figures (`fig1`–`fig7`), writes `results_summary.json`, `charging_traces.json`, `empirical_ocv_curves.json`, `pca_manifold.json`, and compiles `EXPERIMENT_REPORT.md` including a dedicated **Limitations** section.

---

## 5. Complete Synchronized Experimental Results (`744.14 s` Full Run)

### 5.1 Data Corpus & Domain-Shift Reduction
- **Time-Series Silver Layer:** `49,745` rows (`99.61%` retention; 33 NASA Li-ion + 6 Wang Na-ion cells).
- **Degradation Silver Layer:** `8,500` records (`2,410` NASA ordered cycles + `2,739` pooled Mendeley NFM rows + `3,351` external pooled Mendeley NCM rows).
- **Domain-Shift Metrics (Raw → Physics-Normalized → Physics + Empirical OCV + CORAL):**
  - **SoC:** $\text{MMD}^2$ `0.1767 → 0.0852` (`-51.8%`), Mean W1 `0.5656 → 0.3981`, Mean PSI `2.0260 → 1.5183`.
  - **SoH (Exploratory Pooled):** $\text{MMD}^2$ `0.5616 → 0.2852`, Mean W1 `1.0772 → 0.6598`, Mean PSI `5.1072 → 3.4375`.

### 5.2 Within-Chemistry Benchmarks
| Task | Chemistry | Evaluation Protocol | Best Model | Test RMSE | Test $R^2$ | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| **SoC** | Li-ion | 33 NASA cells (8 held-out test cells) | **GRU** | `0.0633` | `0.9570` | LSTM `0.0662`, RF `0.0693`, XGB `0.0698`, LR `0.1010` |
| **SoC** | Na-ion | **6-fold LOCO** (all 6 Wang cells) | **GRU** | `0.0341 ± 0.0079` | `0.9780 ± 0.0190` | RF `0.0352 ± 0.0099`, XGB `0.0362 ± 0.0092`, LSTM `0.0367 ± 0.0074`, LR `0.0427 ± 0.0096` |
| **SoH** | Li-ion | 33 ordered NASA cells only | **RandomForest** | `0.0343` | `0.8762` | XGB `0.0418` ($R^2 = 0.8163$); external pooled NCM test: `0.0905–0.1304` ($R^2 < 0$) |
| **SoH** | Na-ion | **Exploratory 2-fold LOWO** (`25 °C ↔ -15 °C`, tabular only) | **RandomForest** | `0.0297 ± 0.0136` | `-0.0888 ± 0.7290` | Fold `25 °C → -15 °C`: RF `0.0161` ($R^2 = 0.6402$); Fold `-15 °C → 25 °C`: RF `0.0432` ($R^2 = -0.8178$) |

**Na-ion SoC Per-Cell LOCO Insight (Protocol Dependence):**
- Pulse-protocol twins (`na_10.0Ah_bat4` RF `0.0219`, `na_3.2Ah_bat2` RF `0.0224`) achieve ~2.2% RMSE when held out because their same-capacity pulse twin remains in training.
- Driving-cycle cells (`na_3.2Ah_bat5` GRU `0.0473` / RF `0.0484`; `na_10.0Ah_bat6` GRU `0.0357` / RF `0.0405`) have roughly 2× higher error when held out.

### 5.3 Cross-Chemistry Zero-Shot & Few-Shot Transfer Learning (`Li-ion → Na-ion`)
- **Zero-Shot `Li → Na` SoC (all 6 Na-ion cells held out, transductive CORAL):**
  - Raw XGBoost `RMSE = 0.3765` ($R^2 = -0.6222$) → Physics-Normalized `0.2121` ($R^2 = +0.4849$) → Physics + OCV + CORAL **`0.1642`** ($R^2 = +0.6913$; LinearRegression **`0.1598`**, $R^2 = +0.7079$).
  - Single-cell transductive CORAL XGBoost ranges `0.1544–0.1774` across all 6 cells (`cross_chemistry_per_unit.csv`).
- **6-Fold LOCO Transfer Learning `Li → Na` SoC (`TL_TwoPhase_GRU` vs `TargetOnly_GRU`):**
  - **5% target (~880 seq):** `0.0454 ± 0.0070` vs `0.0716 ± 0.0297` (**-36.6% RMSE**)
  - **10% target (~1,761 seq):** `0.0421 ± 0.0073` vs `0.0588 ± 0.0107` (**-28.4% RMSE**)
  - **25% target (~4,402 seq):** `0.0365 ± 0.0092` vs `0.0539 ± 0.0210` (**-32.3% RMSE**)
  - **50% target (~8,804 seq):** `0.0348 ± 0.0071` vs `0.0399 ± 0.0085` (**-12.8% RMSE**)
  - *(Tabular note: `TargetOnly_XGBoost` achieves `0.0376` at 5% vs `TL_Residual_XGBoost` `0.0526`, showing neural GRU benefits strongly from cross-chemistry weight initialization while trees already fit 897 target samples directly.)*
- **Exploratory Zero-Shot & Cross-Workbook TL `Li → Na` SoH (Tabular Only):**
  - Zero-shot XGBoost improves from Raw `0.2286` → Aligned **`0.0691`**; single-workbook transductive CORAL gives **`0.0304`** ($R^2 = +0.0987$) at room temperature `25 °C` (`na_nfm_25C`) vs **`0.0981`** at `-15 °C` (`na_nfm_m15C`).
  - Cross-workbook SoH TL (`25 °C ↔ -15 °C`) exhibits **negative transfer** (`TL_Residual_XGBoost = 0.0808–0.1028` vs `ZeroShot = 0.0649` vs `TargetOnly_XGBoost = 0.0345–0.0409`) due to the 40 °C temperature shift confounded across the two workbooks.

### 5.4 Adaptive Charging Simulation Study (Proposal §5.7)
| Scenario | Strategy | Charge Time (min) | Peak Temp Rise (°C) | Energy Eff. (%) | Est. Degradation (%) | Closed-Loop `soc_observer_mae` | Safety Respected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Fresh Li-ion (SoH 0.98)** | CC-CV / Adaptive ML | `61.17` / `71.17` | `5.61` → `4.71` (`-16.0%`) | `92.55` → `93.44` | `0.0185` → `0.0157` (`-15.1%`) | `0.0168` / `0.0318` | `True` / `True` |
| **Aged Li-ion (SoH 0.80)** | CC-CV / Adaptive ML | `56.50` / `71.00` | `7.12` → `4.43` (`-37.8%`) | `90.93` → `92.73` | `0.0238` → `0.0178` (`-25.2%`) | `0.0177` / `0.0381` | `True` / `True` |
| **Fresh Na-ion (SoH 0.98, TL)** | CC-CV / Adaptive ML | `52.83` / `80.17` | `4.75` → `4.03` (`-15.2%`) | `92.15` → `93.74` | `0.0221` → `0.0168` (`-24.0%`) | `0.0689` / `0.0599` | `True` / `True` |
| **Aged Na-ion (SoH 0.80, TL)** | CC-CV / Adaptive ML | `43.83` / `79.00` | `6.13` → `3.76` (`-38.7%`) | `90.27` → `93.14` | `0.0292` → `0.0189` (`-35.3%`) | `0.0502` / `0.0567` | `True` / `True` |

---

## 6. Decoupled 3D Web Dashboard (`fullstack_3d/`)

### FastAPI Read-Only Artifact & Live Simulator API (`fullstack_3d/backend/app/main.py`)
| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/health` | `GET` | Service health + list of all 15 allowed tables & 7 figures |
| `/api/overview` | `GET` | Serves `results_summary.json` (unit counts, audit metrics, exploratory flags) + figure list |
| `/api/tables/{table_key}` | `GET` | Serves any of the 15 CSV tables as typed JSON (`within_chemistry`, `cross_chemistry`, `na_soc_loco_per_cell`, `na_soh_lowo_folds`, `cross_chemistry_per_unit`, `external_ncm_soh_test`, `transfer_learning`, `adaptive_charging`, `repeated_cell_cv`, `domain_shift`, `mutual_information`, `operating_regimes`, `bigdata_*`) |
| `/api/charging/traces` | `GET` | Precomputed Objective 5 adaptive vs CC-CV trajectories |
| `/api/charging/simulate` | `POST` | Live interactive 2-RC electro-thermal charging comparison using empirical OCV curves |
| `/api/pca-manifold` | `GET` | 3D PCA point clouds (Raw vs Aligned) + domain-shift metrics |
| `/api/report` | `GET` | Full `EXPERIMENT_REPORT.md` markdown text |
| `/api/figures/{filename}` | `GET` | Serves `fig1`–`fig7` PNGs |

### React + TypeScript + Three.js Frontend (`fullstack_3d/frontend/src/`)
- **Tab 1 — 3D Adaptive Charging (`AdaptiveChargingLab.tsx`, `BatteryCell3D.tsx`):** Interactive Three.js 3D thermal/SoC battery cell, playback scrubber, live 2-RC parameter sliders, and transparent simulation disclaimer.
- **Tab 2 — 3D Cross-Chemistry & Transfer Learning (`CrossChemistryView.tsx`, `CrossChemManifold3D.tsx`):** Interactive 1,000-point 3D PCA morph slider, live MMD²/W1/PSI gauges, 3-tier ablation chart, and few-shot TL curves. Dynamically detects exploratory SoH (`evaluation_modes.SoH_cross_chemistry === "exploratory_pooled"` or `< 3` Na-ion SoH units), displays prominent **Exploratory · Pooled Workbooks** caveat banners, labels the SoH manifold accordingly, and restricts SoH charts to tabular models only.
- **Tab 3 — Lakehouse, Models & Figures (`ModelsAndDataView.tsx`):** Reads all cell/unit counts dynamically from `results_summary.json`, shows columnar out-of-core audit KPIs, 5-model RMSE comparisons, Mutual Information rankings, and the full 7-figure gallery (`fig1`–`fig7`).

---

## 7. Developer Onboarding, Commands & Troubleshooting

### Setup & Full Reproduction
```bash
# 1. Clone repository (~1.8 GB, plain Git files — do NOT run git lfs pull)
git clone https://github.com/code-bhaskar/UROP.git
cd UROP

# 2. Create Python 3.11+ virtual environment
python3 -m venv .venv
source .venv/bin/activate
# Optional: CPU-only PyTorch wheel first to save ~2 GB disk
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r urop_core/requirements.txt pytest

# 3. Run automated test suite (16 tests, ~1 minute)
cd urop_core && pytest -v

# 4. Run full real-data pipeline (~12 min on 2 vCPUs; use --fast for ~2-3 min smoke test)
python -m src.pipeline
# Or point to datasets stored outside the repo:
python -m src.pipeline --dataset-root /path/to/datasets

# 5. Launch 3D Web Dashboard (two terminals)
pip install -r ../fullstack_3d/backend/requirements.txt
cd ../fullstack_3d/backend && uvicorn app.main:app --host 0.0.0.0 --port 8000

cd ../frontend && npm ci && npm run dev
```

### Key Implementation Caveats & Known Gotchas
- **DuckDB 2.0 API:** Use `.fetchall()[0][0]` rather than `.fetchone()`.
- **Polars concatenation:** Use `how="diagonal_relaxed"` when concatenating schemas with optional nullable columns (such as `cycle_number = None` on pooled Mendeley rows).
- **Arbin CSV parsing:** Always keep `infer_schema_length=0` in `real_loaders.py` so initial 900 s zero-current rest steps do not force integer typing and nullify fractional float currents later in the file.
- **Never re-enable Git LFS:** `.gitattributes` marks `.zip` and `.xlsx` as `binary` only; datasets are plain committed files.
- **Never run sequence models on pooled data:** `build_segmented_sequences()` will intentionally raise `ValueError` if `data_provenance == "pooled_unordered"` is present.

---

## 8. Limitations & Recommended Next Steps

1. **Na-ion Multi-Cycle Aging Data (Highest Priority):** Mendeley `10.17632/4mztcdc4gt.1` lacks cell IDs and cycle order, leaving only 2 temperature-confounded pooled workbooks (`25 °C` vs `-15 °C`). Future work should ingest a public Na-ion multi-cycle aging dataset with explicit per-cell IDs and sequential cycle histories (verified via `detect_pooled_sorted_order`) to unlock non-exploratory SoH cell-level CV and sequence TL.
2. **Na-ion SoC Sample Size (`N = 6`):** Adding more Na-ion cells across diverse cathode chemistries (polyanionic, Prussian Blue analogues, layered oxides) and mixed drive cycles will decouple cell identity from test protocol (`bat1/bat2` & `bat3/bat4` pulse twins vs `bat5/bat6` drive cycles).
3. **Online / Streaming Adaptation:** Tier 3 unsupervised CORAL is transductive (requires unlabeled target feature statistics); streaming covariance updates should be explored for cold-start BMS deployment.
4. **Hardware-in-the-Loop Validation:** Objective 5 adaptive charging results come from a mismatched 2-RC simulation plant (Proposal §5.7) and should be validated on physical battery cycler hardware before real BMS deployment.
5. **Proposal PDF Edits (`AP23110011334_24230_UROP_CLA.pdf`):** Because only the binary PDF is committed (no `.tex`/`.docx` source), apply three manual edits in the original manuscript source: (a) fix skipped Section 9 numbering, (b) replace `"his"` → `"their"`, and (c) add a brief **"Datasets Actually Used"** paragraph listing NASA PCoE (33 cells), Wang et al. Zenodo `13836819` (6 cells), and Mendeley `10.17632/4mztcdc4gt.1` (pooled exploratory SoH).
