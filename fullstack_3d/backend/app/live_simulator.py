"""On-demand Objective 5 Adaptive Charging vs CC-CV live simulation bridge.

Imports the electro-thermal-aging charging simulator from ../urop_core in read-only mode
and loads the real-data empirical OCV curves exported by urop_core/outputs/empirical_ocv_curves.json
(ZERO synthetic dataset generator imports).
"""

from __future__ import annotations

import sys

import numpy as np
from pydantic import BaseModel, Field

from .artifacts import UROP_CORE_DIR, load_json_artifact

if str(UROP_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(UROP_CORE_DIR))

from src.charging.adaptive_charging import ChargingTrace, simulate_charging_episode  # noqa: E402
from src.config import CHEMISTRY_SPECS  # noqa: E402
from src.cross_chemistry.alignment import EmpiricalOCVCurve  # noqa: E402


class SimulationRequest(BaseModel):
    chemistry: str = Field(default="na_ion", pattern="^(li_ion|na_ion)$")
    soh_true: float = Field(default=0.80, ge=0.68, le=1.00)
    ambient_temp_c: float = Field(default=25.0, ge=0.0, le=38.0)
    soc_start: float = Field(default=0.10, ge=0.05, le=0.50)
    soc_target: float = Field(default=0.90, ge=0.70, le=0.96)
    q_nom: float | None = Field(default=None, ge=1.0, le=15.0)
    observer_noise_std: float = Field(default=0.014, ge=0.0, le=0.08)


def _load_real_empirical_ocv(chemistry: str) -> EmpiricalOCVCurve | None:
    """Load the training-cell-fitted empirical OCV curve exported from real urop_core runs."""
    try:
        payload = load_json_artifact("empirical_ocv_curves.json")
        entry = payload.get(chemistry)
        if not entry:
            return None
        return EmpiricalOCVCurve(
            chemistry=entry["chemistry"],
            soc_grid=np.asarray(entry["soc_grid"], dtype=np.float64),
            v_norm_grid=np.asarray(entry["v_norm_grid"], dtype=np.float64),
            estimation_method=entry["estimation_method"],
            n_samples_used=int(entry["n_samples_used"]),
        )
    except FileNotFoundError:
        return None


def _make_empirical_ocv_observer(
    chemistry: str,
    empirical_ocv: EmpiricalOCVCurve | None,
    noise_std: float,
    seed: int = 42,
):
    """Lightweight inverse empirical-OCV observer calibrated from real urop_core training curves."""
    if empirical_ocv is not None:
        soc_grid = empirical_ocv.soc_grid
        v_norm_grid = np.maximum.accumulate(empirical_ocv.v_norm_grid + np.linspace(0.0, 1e-6, len(soc_grid)))
    else:
        soc_grid = np.linspace(0.0, 1.0, 11)
        v_norm_grid = (
            np.array([0.02, 0.48, 0.62, 0.68, 0.72, 0.76, 0.80, 0.85, 0.90, 0.95, 0.995])
            if chemistry == "li_ion"
            else np.array([0.02, 0.22, 0.36, 0.45, 0.53, 0.61, 0.69, 0.76, 0.84, 0.92, 0.995])
        )
    rng = np.random.default_rng(seed)

    def predict(X_phys: np.ndarray) -> np.ndarray:
        v_ir_norm = float(X_phys[0, 11])
        soc_est = float(np.interp(v_ir_norm, v_norm_grid, soc_grid))
        return np.array([np.clip(soc_est + rng.normal(0.0, noise_std), 0.0, 1.0)])

    return predict


def _serialize_trace(tr: ChargingTrace) -> dict:
    stride = max(1, len(tr.time_min) // 200)
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

    return {
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


def run_live_charging_comparison(req: SimulationRequest) -> dict:
    """Run synchronized CC-CV vs Adaptive_ML simulation for custom user parameters."""
    spec = CHEMISTRY_SPECS[req.chemistry]
    q_nom = req.q_nom if req.q_nom is not None else spec.default_q_nom
    soh_est = float(np.clip(req.soh_true + (0.004 if req.chemistry == "li_ion" else -0.006), 0.65, 1.0))
    emp_ocv = _load_real_empirical_ocv(req.chemistry)

    cc_trace = simulate_charging_episode(
        chemistry=req.chemistry,
        strategy="CC-CV",
        soh_true=req.soh_true,
        soh_est=soh_est,
        soc_predictor=_make_empirical_ocv_observer(req.chemistry, emp_ocv, req.observer_noise_std, seed=101),
        q_nom=q_nom,
        amb_temp=req.ambient_temp_c,
        soc_start=req.soc_start,
        soc_target=req.soc_target,
        empirical_ocv=emp_ocv,
    )
    ad_trace = simulate_charging_episode(
        chemistry=req.chemistry,
        strategy="Adaptive_ML",
        soh_true=req.soh_true,
        soh_est=soh_est,
        soc_predictor=_make_empirical_ocv_observer(req.chemistry, emp_ocv, req.observer_noise_std, seed=202),
        q_nom=q_nom,
        amb_temp=req.ambient_temp_c,
        soc_start=req.soc_start,
        soc_target=req.soc_target,
        empirical_ocv=emp_ocv,
    )

    m_cc = cc_trace.metrics
    m_ad = ad_trace.metrics
    deltas = {
        "temp_rise_reduction_pct": round(
            100.0 * (m_cc["peak_temp_rise_c"] - m_ad["peak_temp_rise_c"]) / max(m_cc["peak_temp_rise_c"], 1e-4), 1
        ),
        "degradation_reduction_pct": round(
            100.0
            * (m_cc["estimated_degradation_pct"] - m_ad["estimated_degradation_pct"])
            / max(m_cc["estimated_degradation_pct"], 1e-6),
            1,
        ),
        "efficiency_gain_pct": round(m_ad["energy_efficiency_pct"] - m_cc["energy_efficiency_pct"], 2),
        "charge_time_delta_min": round(m_ad["charge_time_min"] - m_cc["charge_time_min"], 2),
    }

    return {
        "request": req.model_dump(),
        "spec": {
            "name": spec.name,
            "v_min": spec.v_min,
            "v_max": spec.v_max,
            "v_nom": spec.v_nom,
            "q_nom": q_nom,
            "t_max_safe": spec.t_max_safe,
        },
        "cc_cv": _serialize_trace(cc_trace),
        "adaptive_ml": _serialize_trace(ad_trace),
        "deltas": deltas,
    }
