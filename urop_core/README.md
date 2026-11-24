# `urop_core/` — Standalone 6-Objective UROP Research Pipeline

Completely independent of `fullstack_3d/` (zero web, Node.js, or FastAPI dependencies).

## Run All 6 Objectives
```bash
pip install -r requirements.txt
python -m src.pipeline            # Full run (Objectives 1-6)
python -m src.pipeline --fast     # Fast verification mode
pytest -v                         # Run 6 unit & integration tests
```

## Outputs Generated (`outputs/`)
- `EXPERIMENT_REPORT.md` — Full quantitative report mapped to Proposal Sections 5.1–5.7 & 7
- `results_summary.json` — Complete structured summary
- `charging_traces.json` — Time-stepped CC-CV & Adaptive ML trajectories (Li-ion & Na-ion, fresh & aged)
- `pca_manifold.json` — 3D PCA manifold coordinates (Raw vs Physics + OCV-to-OCV + CORAL Aligned)
- `tables/*.csv` — 10 CSV benchmark, Big Data DuckDB, Data Mining, Cross-Chemistry, Transfer Learning, and Charging tables
- `figures/*.png` — 6 publication-ready PNG figures
