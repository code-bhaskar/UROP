"""FastAPI backend serving UROP 6-Objective outputs and live Objective 5 Adaptive Charging simulations."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .artifacts import (
    ALLOWED_TABLES,
    FIGURES_DIR,
    list_figures,
    load_csv_table,
    load_json_artifact,
    load_report_markdown,
)
from .live_simulator import SimulationRequest, run_live_charging_comparison

app = FastAPI(
    title="UROP Cross-Chemistry BMS 3D Digital Twin API",
    version="1.0.0",
    description="Read-only artifact server for urop_core/outputs + live Objective 5 Adaptive Charging simulation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "urop-cross-chemistry-bms-api",
        "available_tables": sorted(ALLOWED_TABLES.keys()),
        "available_figures": list_figures(),
    }


@app.get("/api/overview")
def get_overview() -> dict:
    try:
        summary = load_json_artifact("results_summary.json")
        return {
            "summary": summary,
            "figures": list_figures(),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/tables/{table_key}")
def get_table(table_key: str) -> dict:
    try:
        rows = load_csv_table(table_key)
        return {"table": table_key, "count": len(rows), "rows": rows}
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/charging/traces")
def get_charging_traces() -> dict:
    try:
        traces = load_json_artifact("charging_traces.json")
        summary_rows = load_csv_table("adaptive_charging")
        return {"traces": traces, "summary": summary_rows}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/charging/simulate")
def simulate_charging(req: SimulationRequest) -> dict:
    return run_live_charging_comparison(req)


@app.get("/api/pca-manifold")
def get_pca_manifold() -> dict:
    try:
        manifold = load_json_artifact("pca_manifold.json")
        domain_shift = load_csv_table("domain_shift")
        return {"manifold": manifold, "domain_shift": domain_shift}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/report")
def get_report() -> dict:
    return {"markdown": load_report_markdown()}


@app.get("/api/figures/{filename}")
def get_figure(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = FIGURES_DIR / filename
    if not path.exists() or path.suffix.lower() != ".png":
        raise HTTPException(status_code=404, detail=f"Figure '{filename}' not found")
    return FileResponse(path, media_type="image/png")
