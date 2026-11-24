# Cross-Chemistry Battery Management System & 3D Adaptive Charging Digital Twin (UROP)

**Project:** *Data-Driven Battery State Estimation and Adaptive Charging Optimization Across Lithium-Ion and Sodium-Ion Batteries* (`AP23110011334_24230_UROP_CLA.pdf`)

This repository is cleanly separated into **two independent folders** so the core 6-objective ML / Big Data research project can run completely standalone without any web stack, while the full-stack 3D Three.js application consumes the generated results of the core project:

```
UROP/
├── 10Ah na-ion/                     # Git LFS Sodium-Ion 10Ah datasets (Wang et al., 2024)
├── 3.2Ah na-ion/                    # Git LFS Sodium-Ion 3.2Ah datasets (Wang et al., 2024)
├── 5. Battery Data Set/             # Git LFS Lithium-Ion NASA PCoE aging datasets
├── AP23110011334_24230_UROP_CLA.pdf # UROP Proposal Document
│
├── urop_core/                       # [FOLDER 1] Standalone 6-Objective Research Pipeline (ZERO web deps)
│   ├── src/                         # Polars/Parquet/DuckDB Lakehouse, Data Mining, ML/DL, Cross-Chem, TL, Charging
│   ├── tests/                       # Standalone pytest suite
│   └── outputs/                     # Generated JSON, CSV tables, PNG figures, charging traces, 3D PCA manifolds, Markdown report
│
└── fullstack_3d/                    # [FOLDER 2] Full-Stack 3D Web Application (Consumes urop_core/outputs/)
    ├── backend/                     # FastAPI REST API + Live Objective 5 Electro-Thermal Simulator
    └── frontend/                    # React + TypeScript + Vite + Three.js (3D Battery Digital Twin & 3D PCA Morph)
```

---

## Folder 1: Run the Standalone 6-Objective Research Project (`urop_core/`)

Zero Node.js or FastAPI dependencies required:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r urop_core/requirements.txt

# Optional: pull 1.34 GB real datasets on a machine with Git LFS access
git lfs install && git lfs pull

# Run all 6 objectives end-to-end
cd urop_core
python -m src.pipeline

# Run standalone automated tests
pytest -v
```

Outputs are written to `urop_core/outputs/` (`EXPERIMENT_REPORT.md`, `results_summary.json`, `charging_traces.json`, `pca_manifold.json`, `tables/*.csv`, `figures/*.png`).

---

## Folder 2: Run the Full-Stack 3D Digital Twin (`fullstack_3d/`)

Consumes the generated results in `urop_core/outputs/` without mixing web code into `urop_core/`:

```bash
# Terminal 1 — Start FastAPI Backend (port 8000)
pip install -r fullstack_3d/backend/requirements.txt
cd fullstack_3d/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Start React + Three.js Frontend (port 5173)
cd fullstack_3d/frontend
npm install
npm run dev
```

Open `http://localhost:5173` to explore:
1. **3D Adaptive Charging Lab (Objective 5):** Interactive Three.js dual cutaway **18650 Li-ion** & **Prismatic Na-ion** digital twin comparing **CC-CV vs. Adaptive ML** side by side with live C-rate ion flux particles, real-time thermal core glow, high-SoC plating stress halo, timeline playback, synchronized telemetry charts, and a live what-if simulator (`POST /api/charging/simulate`).
2. **3D Cross-Chemistry & Transfer Learning (Objective 4):** Interactive Three.js **3D PCA Feature Manifold Morph** (morph 1,000 Li-ion/Na-ion samples smoothly from Raw Features $\to$ Physics + OCV-to-OCV + CORAL Aligned), 3-tier zero-shot ablation charts, and few-shot transfer learning curves.
3. **Big Data Lakehouse, Models & Figures (Objectives 1–3, 6):** Polars/Parquet/DuckDB metrics, IsolationForest audit stats, all 5 ML/DL models (LR, RF, XGBoost, LSTM, GRU), Mutual Information rankings, and all 6 publication PNG figures.
