# `fullstack_3d/` — 3D Cross-Chemistry BMS Digital Twin (FastAPI + React + Three.js)

Consumes the generated outputs of `../urop_core/outputs/` cleanly without modifying or mixing into the core 6-objective ML project.

## Stack
- **Backend (`backend/`):** FastAPI + Uvicorn + Pydantic (serves `../urop_core/outputs/` artifacts + runs live on-demand Objective 5 charging simulations via `POST /api/charging/simulate`)
- **Frontend (`frontend/`):** React 18 + TypeScript + Vite + **Three.js** (`OrbitControls`, dual cutaway 18650 & Prismatic cell digital twin, live ion-flux particles, thermal & plating risk shaders, interactive 3D PCA manifold morph) + Recharts

## Local Launch
```bash
# 1. Backend API (port 8000)
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 2. Frontend 3D App (port 5173, proxies /api -> 127.0.0.1:8000)
cd ../frontend
npm install
npm run dev
```
