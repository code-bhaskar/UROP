# Machine Learning-Based State Estimation and Cross-Chemistry Transfer Learning for Battery Management Systems (Li-ion ↔ Na-ion)

A standalone, reproducible Python ML/data-engineering pipeline (`urop_core/`) plus a decoupled 3D web dashboard (`fullstack_3d/`) implementing all six objectives of the UROP proposal on **real laboratory battery datasets** (45 files, 1,357.11 MB).

---

## 1. Repository Layout & Separation

```text
UROP/
├── 5. Battery Data Set/          # NASA Prognostics Li-ion (6 ZIPs, 33 ordered cells)
├── 3.2Ah na-ion/                 # Wang et al. (2025) 3.2 Ah Na-ion (18 CSVs, 3 ordered cells × 6 temps)
├── 10Ah na-ion/                  # Wang et al. (2025) 10 Ah Na-ion (18 CSVs, 3 ordered cells × 6 temps)
├── Dataset 1-1.xlsx              # Mendeley 10.17632/4mztcdc4gt.1 NFM Na-ion 25 °C (pooled unordered)
├── Dataset 1-2.xlsx              # Mendeley 10.17632/4mztcdc4gt.1 NFM Na-ion -15 °C (pooled unordered)
├── Dataset 2.xlsx                # Mendeley 10.17632/4mztcdc4gt.1 NCM Li-ion 25 °C (pooled unordered)
├── urop_core/                    # Standalone pipeline (NO web dependencies)
│   ├── DATA_NOTES.md             # Raw dataset inspection, provenance & Mendeley diagnostic tables
│   ├── src/                      # Polars/DuckDB lakehouse, feature eng, models, alignment, TL, charging
│   ├── outputs/                  # EXPERIMENT_REPORT.md, results_summary.json, tables/, figures/
│   └── tests/                    # Automated pytest suite
└── fullstack_3d/                 # Decoupled read-only FastAPI + React/TypeScript/Three.js viewer
```

### Cloning & Dataset Paths (Plain Committed Files — No Git LFS)
All 45 dataset files are committed directly in Git as plain files (~1.8 GB total clone; **Git LFS is not used** and `git lfs pull` is not required). If you store the datasets in an external folder instead of the repo root, pass `--dataset-root /path/to/datasets` or export `UROP_DATASET_ROOT=/path/to/datasets`.

---

## 2. Quickstart

### Run the Standalone ML Pipeline (`urop_core/`)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r urop_core/requirements.txt

cd urop_core
python -m src.pipeline                # Full real-data run (all 45 files)
# Or pass custom dataset path:
python -m src.pipeline --dataset-root /path/to/datasets
# Or run tests:
pytest -v
```

### Run the Read-Only 3D Web Dashboard (`fullstack_3d/`)
```bash
pip install -r fullstack_3d/backend/requirements.txt
cd fullstack_3d/backend && uvicorn app.main:app --host 0.0.0.0 --port 8000

cd fullstack_3d/frontend && npm ci && npm run dev
```

---

## 3. Dataset Provenance & Scientific Scope

See [`urop_core/DATA_NOTES.md`](urop_core/DATA_NOTES.md) for full raw-file diagnostics.

| Dataset | Chemistry & Role | Independent Units | Provenance | Models Permitted |
| --- | --- | --- | --- | --- |
| NASA Prognostics (`5. Battery Data Set/`) | Li-ion SoC & SoH | **33 ordered cells** (`B0005`–`B0056`) | `ordered_per_cell` | Tabular + LSTM/GRU |
| Wang et al. (`3.2Ah na-ion/`, `10Ah na-ion/`) | Na-ion SoC (`0 °C`–`50 °C`) | **6 ordered cells** (`bat1`, `bat2`, `bat5`, `bat3`, `bat4`, `bat6`) | `ordered_per_cell` | Tabular + LSTM/GRU (6-fold LOCO) |
| Mendeley `Dataset 1-1.xlsx`, `Dataset 1-2.xlsx` | Na-ion NFM SoH (`25 °C`, `-15 °C`) | **2 pooled workbooks** (`na_nfm_25C`, `na_nfm_m15C`, 2,739 rows) | `pooled_unordered` (Discharge-only; no cell IDs or cycle order) | **Tabular only** (exploratory; temp-confounded) |
| Mendeley `Dataset 2.xlsx` | Li-ion NCM SoH (`25 °C`) | **1 pooled workbook** (`li_ncm_25C`, 3,351 rows) | `pooled_unordered` (Discharge-only) | External pooled Li-ion test set only (never mixed into NASA training) |

---

## 4. Key Experimental Results (Synchronized from `urop_core/outputs/EXPERIMENT_REPORT.md`)

### Columnar Out-of-Core Medallion Lakehouse (Polars + DuckDB + ZSTD Parquet)
- **Time-Series Layer:** `49,938` Bronze rows → **`49,745` Silver rows** (`99.61%` retention; 3 duplicates, 90 bounds/null, 100 IsolationForest-filtered).
- **Degradation Layer:** **`8,500` Silver records** (`2,410` NASA ordered cycles across 33 cells + `2,739` pooled Mendeley NFM rows across 2 workbooks + `3,351` external pooled Mendeley NCM rows).
- **Operating-Regime K-Means Silhouette (4 clusters):** `0.3501`.

### Within-Chemistry Estimation
- **Li-ion SoC (33 NASA cells, held-out 8-cell test split):** Best **GRU** `RMSE = 0.0633` ($R^2 = 0.9570$); RandomForest `0.0693`, XGBoost `0.0698`, LSTM `0.0662`.
- **Na-ion SoC (6-fold Leave-One-Cell-Out across all 6 Wang cells):** Best **GRU** `RMSE = 0.0341 ± 0.0079` ($R^2 = 0.9780 \pm 0.0190$); RandomForest `0.0352 ± 0.0099`, XGBoost `0.0362 ± 0.0092`, LSTM `0.0367 ± 0.0074`, LinearRegression `0.0427 ± 0.0096`. Per-cell LOCO confirms protocol dependence: pulse twins (`bat4`, `bat2`) score ~`0.022`, while driving-cycle cells (`bat5`, `bat6`) score `0.036–0.048`.
- **Li-ion SoH (33 ordered NASA cells only, NCM excluded from training):** Best **RandomForest** `RMSE = 0.0343` ($R^2 = 0.8762$); XGBoost `0.0418` ($R^2 = 0.8163$). Evaluated strictly out-of-sample on external pooled Mendeley NCM (`li_ncm_25C`, 1C fixed window vs NASA 2 A), RMSE is `0.0905–0.1304` ($R^2 < 0$), documenting cross-protocol mismatch.
- **Na-ion SoH (Exploratory 2-fold Leave-One-Workbook-Out, `25 °C ↔ -15 °C`, tabular only):** Mean RandomForest `RMSE = 0.0297 ± 0.0136` (`0.0161` when testing `-15 °C`, `0.0432` when extrapolating `-15 °C → 25 °C`); XGBoost `0.0339 ± 0.0152`.

### Cross-Chemistry Generalization & Transfer Learning (`Li-ion → Na-ion`)
- **SoC Zero-Shot (all 6 Na-ion cells held out; Tier 3 transductive unsupervised CORAL):**
  - Raw XGBoost `RMSE = 0.3765` ($R^2 = -0.6222$) → Physics-Normalized `0.2121` ($R^2 = +0.4849$) → Physics+OCV+CORAL **`0.1642`** ($R^2 = +0.6913$; LinearRegression **`0.1598`**, $R^2 = +0.7079$).
- **SoC Transfer Learning (6-fold LOCO):**
  - **`TL_TwoPhase_GRU`** consistently outperforms `TargetOnly_GRU` at every budget: `5%` (`0.0454 ± 0.0070` vs `0.0716 ± 0.0297`), `10%` (`0.0421 ± 0.0073` vs `0.0588 ± 0.0107`), `25%` (`0.0365 ± 0.0092` vs `0.0539 ± 0.0210`), `50%` (`0.0348 ± 0.0071` vs `0.0399 ± 0.0085`).
- **SoH Zero-Shot & Transfer Learning (Exploratory · Pooled Workbooks · Tabular Only):**
  - Zero-shot `Li → Na` XGBoost improves from Raw `0.2286` → Aligned **`0.0691`** (`0.0298` at room temperature `25 °C` vs `0.0984` at `-15 °C`).
  - Cross-workbook SoH TL (`25 °C ↔ -15 °C`, tabular only) exhibits **negative transfer** (`TL_Residual_XGBoost = 0.0808–0.1028` vs `ZeroShot = 0.0649` vs `TargetOnly = 0.0345–0.0409`) due to the 40 °C temperature shift confounded across the two workbooks.

### Proposal §5.7 Adaptive Charging Simulation Study (2-RC Mismatched Electro-Thermal Plant)
Simulation study comparing ML/TL-informed adaptive charging against CC-CV (not physical bench hardware; `soc_observer_mae` reported alongside every scenario):
- **Aged Li-ion (`SoH 0.80`):** Peak $\Delta T$ `7.12 → 4.43 °C` (`-37.8%`), degradation index `0.0238 → 0.0178%` (`-25.2%`), efficiency `90.93 → 92.73%`, closed-loop `soc_observer_mae = 0.0381` (vs `0.0177` CC-CV), `100%` safety limits respected.
- **Aged Na-ion (`SoH 0.80`, TL-SoC):** Peak $\Delta T$ `6.13 → 3.76 °C` (`-38.7%`), degradation index `0.0292 → 0.0189%` (`-35.3%`), efficiency `90.27 → 93.14%`, closed-loop `soc_observer_mae = 0.0567` (vs `0.0502` CC-CV), `100%` safety limits respected.

Full tables, all 7 figures (`fig1`–`fig7`), and the **Limitations** section are in [`urop_core/outputs/EXPERIMENT_REPORT.md`](urop_core/outputs/EXPERIMENT_REPORT.md).
