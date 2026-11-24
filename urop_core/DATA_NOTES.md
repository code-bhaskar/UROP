# Dataset Diagnostics & Provenance Notes (`DATA_NOTES.md`)

This document records raw-file structural diagnostics executed directly on the committed datasets in this repository before modeling, ensuring no cell identities, cycle orders, or cross-sheet pairings are ever invented.

---

## 1. NASA PCoE Li-ion Battery Aging Dataset (`5. Battery Data Set/*.zip`)
- **Provenance:** `ordered_per_cell` (genuine per-cell `.mat` files `B0005`–`B0056`, real cycle timestamps and per-cycle capacity measurements).
- **Files:** 6 `.zip` archives (`200 MB`), containing **34 unique cells** (`B0025`–`B0028` appear in both archives 2 and 3 and are deduplicated by cell ID; 33 active aged cells pass minimum cycle count and physical bounds).
- **Test Protocol:** Constant-current discharge at `2.0 A` (with varying cell-group cutoffs `2.0–2.7 V`) and CC-CV charging at `1.5 A` to `4.2 V` across `4 °C`, `24 °C`, and `43 °C`.
- **Preprocessing Safeguards:**
  - `B0033`, `B0034`, and `B0036` begin with anomalous low-capacity partial cycles; Beginning-of-Life reference capacity uses `max(valid_caps[:8])`.
  - Post-cutoff zero-current voltage rebound tails (`|I| <= 0.15 A` after discharge termination) are trimmed so `SoC = 0%` is not confounded with open-circuit rebound voltages.

---

## 2. Wang et al. (2024, Zenodo 13836819) Na-ion Dataset (`3.2Ah na-ion/`, `10Ah na-ion/`)
- **Provenance:** `ordered_per_cell` for time-series SoC (`Cycle Index = [1]` across all 36 Arbin `.CSV` files; single-cycle characterization — **zero multi-cycle aging data**).
- **Files:** 36 Arbin `.CSV` files (`1,145 MB`) across **6 physical cells** (`bat1`, `bat2`, `bat5` at `3.2 Ah`; `bat3`, `bat4`, `bat6` at `10.0 Ah`) tested at 6 ambient temperatures (`-5`, `5`, `15`, `25`, `35`, `45 °C`) under two protocols:
  - **Enhanced Pulse (`bat1`, `bat2`, `bat3`, `bat4`):** 4 cells × 6 temperatures.
  - **Driving Cycle (`bat5`, `bat6`):** 2 cells × 6 temperatures.
- **Note on Cell-Type / Protocol Coupling:** `bat1`/`bat2` (`3.2 Ah`) and `bat3`/`bat4` (`10 Ah`) share the pulse protocol across 6 temperatures, while `bat5` (`3.2 Ah`) and `bat6` (`10 Ah`) are the only driving-cycle cells. All temperatures from a given cell stay strictly inside the same fold (never split across train/test), and per-cell Leave-One-Cell-Out (LOCO) results are reported explicitly because accuracy depends on cell type and protocol.
- **Known Source Duplicate Files:** `Pulse_data_temp-5_bat1.CSV` & `bat2.CSV` share identical SHA-256 hashes, and `Pulse_data_temp-5_bat3.CSV` & `bat4.CSV` share identical SHA-256 hashes (source repository artifact; noted here as a data limitation).
- **SoC Ground Truth:** Computed at full native 1 Hz resolution prior to downsampling via `0.2 C` calibration-capacity-anchored Coulomb counting (`Step 2` `3.90 V` full-charge anchor + `Step 4/5` full `0.2 C` discharge capacity measurement), with `infer_schema_length=0` to prevent `Int64` coercion on initial 900 s zero-current rest rows.

---

## 3. Mendeley Battery Degradation Datasets (`Dataset 1-1.xlsx`, `Dataset 1-2.xlsx`, `Dataset 2.xlsx`)
- **Citation:** Shen Siyuan, Xinyang Yue, Junjian Zhao, Zhengyu Chen, Wei Hao, Yuanmao Chen, Yu Zhang, Yakun Liu, Zheng Liang (2026), *"Battery Degradation Datasets (NFM Sodium-ion Batteries and NCM Lithium-ion Batteries)"*, Mendeley Data, v1, DOI: `10.17632/4mztcdc4gt.1` (Shanghai Jiao Tong University; published 21 April 2026).
- **Published Description:**
  - `Dataset 1-1.xlsx`: NFM Na-ion, `4.5 Ah` nominal, `25 °C`, `2.0–4.0 V`, `1 C` CC charge/discharge.
  - `Dataset 1-2.xlsx`: NFM Na-ion, `4.5 Ah` nominal, `-15 °C`, `2.0–4.0 V`, `1 C` CC charge/discharge.
  - `Dataset 2.xlsx`: NCM Li-ion, `1.5 Ah` nominal, `25 °C`, `2.5–4.2 V`, `1 C` CC discharge.
  - The Mendeley description does not state the exact number of physical cells per workbook, and none of the Excel sheets include a cell ID column or a cycle index column.

### Raw File Structural Diagnostics (Run via `openpyxl(..., read_only=True, data_only=True)`)

| File | Sheet | Rows | Numeric V Cols | Non-Numeric Cols | `Q_max` First → Last (Ah) | `Q_max` Upward Steps | Max Upward Jump (Ah) | Strictly Non-Increasing? |
| :--- | :--- | ---: | ---: | :--- | :--- | ---: | ---: | :--- |
| `Dataset 1-1.xlsx` | `NFM \| 1C \| 25℃ \| Discharge` | 1,528 | 201 (`2.00–4.00 V`) | None | `4.376593 → 3.743480` | 324 (`21.22%`) | `+0.109249` | No (`Q_max` oscillates across pooled cells) |
| `Dataset 1-1.xlsx` | `NFM \| 1C \| 25℃ \| Charge` | 1,528 | 201 (`2.00–4.00 V`) | None | `4.211638 → 3.694033` | 717 (`46.95%`) | `+0.302900` | No |
| `Dataset 1-2.xlsx` | `NFM \| 1C \| -15℃ \| Discharge` | 1,211 | 201 (`2.00–4.00 V`) | `SOH` (col 201) | `3.628833 → 3.229000` | **0 (`0.00%`)** | `+0.000000` | **Yes** (`SOH`: `0.999752 → 0.894408`, 0 increases) |
| `Dataset 1-2.xlsx` | `NFM \| 1C \| -15℃ \| Charge` | 1,211 | 191 (`2.10–4.00 V`) | `SOH` (col 191) | `3.548700 → 3.149033` | 591 (`48.84%`) | `+0.154111` | `Q_max`: No; **`SOH` col: Yes** (`1.000000 → 0.893385`, 0 increases) |
| `Dataset 2.xlsx` | `NCM \| 1C \| 25℃ \| Discharge` | 3,351 | 171 (`2.50–4.20 V`) | None | `1.487788 → 1.124600` | **0 (`0.00%`)** | `+0.000000` | **Yes** |

### Cross-Sheet Row Alignment Diagnostics (Discharge vs Charge)

| File | Same Row Count? | `corr(Q_max_dis, Q_max_chg)` | `corr(Q_mid_dis, Q_mid_chg)` | `corr(diff(Q_max_dis), diff(Q_max_chg))` | `corr(diff(Q_mid_dis), diff(Q_mid_chg))` | Aligned? |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| `Dataset 1-1.xlsx` | Yes (`1,528`) | `+0.911500` | `+0.705586` | **`+0.028740`** | **`+0.045666`** | **No** (global fade trend only; row-level deltas uncorrelated) |
| `Dataset 1-2.xlsx` | Yes (`1,211`) | `+0.988065` | `+0.692392` | **`-0.035018`** | **`+0.010656`** | **No** (plus Discharge `SOH` `0.999752→0.894408` ≠ Charge `SOH` `1.000000→0.893385`) |

### Engineering & Scientific Conclusions (Case A — `pooled_unordered`)
1. **No Cell IDs or Cycle Order Exist:** `Dataset 1-2` (Discharge & Charge `SOH`) and `Dataset 2` (`Q_max`) are strictly monotonically non-increasing with `0` upward steps, proving the rows were sorted by capacity/SoH after pooling. `Dataset 1-1` also lacks cell IDs and cycle indices and exhibits `±0.109 Ah` oscillations typical of pooled cells sorted by normalized health.
2. **Discharge & Charge Sheets Are NOT Row-Aligned:** First-difference correlations between Discharge and Charge sheets are near zero (`+0.029` and `-0.035`), and `Dataset 1-2` carries distinct `SOH` columns on its two sheets. Therefore, **all cross-sheet features (hysteresis resistance `|V50_chg - V50_dis|/2I` and Charge-sheet durations) are dropped**, and features are computed exclusively from each workbook's Discharge sheet.
3. **Mandatory Modeling Constraints Enforced in Code:**
   - Every workbook is treated as **one indivisible unit** (`na_nfm_25C`, `na_nfm_m15C`, `li_ncm_25C`) tagged with `data_provenance = "pooled_unordered"`.
   - **Zero invented cell clusters** (KMeans pseudo-cell clustering is completely deleted).
   - **Zero invented `cycle_number`** (`cycle_number = None` / null for pooled rows; no fade-per-cycle statistic computed for pooled workbooks).
   - **Tabular models only** (`LinearRegression`, `RandomForest`, `XGBoost`) for Mendeley SoH; **GRU/LSTM sequence models refuse/skip `pooled_unordered` data** because sorting rows by SoH would leak the target through sequence order.
   - **Reference Capacity Definition:** `Dataset 1-2` uses its real provided `SOH` column (`implied Q_ref ≈ 3.63 Ah` at `-15 °C`); `Dataset 1-1` (`na_nfm_25C`) and `Dataset 2` (`li_ncm_25C`) have no `SOH` column, so `SoH = Q_max / max(Q_max)` within each workbook (explicitly documented as workbook-maximum capacity normalization, NOT nominal `4.5 Ah` / `1.5 Ah` normalization).
   - **Unit / Protocol Mismatch Note (NASA vs Mendeley):** NASA PCoE Li-ion cells discharge at constant `2.0 A` (`~1 C` on `2.0 Ah` 18650 cells) across varying cutoffs (`2.0–2.7 V`), whereas Mendeley NFM/NCM sheets report `Q(V)` curves at constant `1 C` across fixed windows (`2.0–4.0 V` for NFM; `2.5–4.2 V` for NCM). Both loaders now compute the **exact same Discharge-only indirect health indicator definitions** (fixed mid-voltage window `[25%–75%]` partial transit time, upper-window `[45%–90%]` transit time, upper-tail `[75%–100%]` transit time, smoothed `dQ/dV` ICA peak & normalized peak voltage, energy-weighted normalized mean voltage, and discharge polarization proxy) — while noting that remaining C-rate/cutoff/temperature differences are an inherent real-world cross-chemistry domain shift.
