"""Coupled 2-RC Electro-Thermal-Aging Battery Charging Simulator: ML-Informed Adaptive Charging vs CC-CV Baseline.

Note (per Proposal Section 5.7): This module performs a physics-based closed-loop simulation
evaluating state-informed adaptive charging against CC-CV under defined voltage, current,
and thermal safety constraints. It is a simulation study, not physical hardware validation.

Plant-to-controller separation:
- The charging plant uses an independent 2-RC electro-thermal model (fast charge-transfer RC +
  slow solid-state diffusion RC + charging hysteresis + +12% unmodeled resistance mismatch)
  that intentionally differs from the controller's 1-RC online feature extractor.
- When data-driven `EmpiricalOCVCurve` objects (fitted strictly on training cells) are supplied,
  the plant uses those empirical OCV curves plus realistic charge-direction hysteresis.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from src.config import CHEMISTRY_SPECS
from src.cross_chemistry.alignment import EmpiricalOCVCurve

# Independent empirical 11-point literature datasheet OCV lookup anchors (used only when no
# training-estimated EmpiricalOCVCurve is supplied, e.g. standalone live what-if API calls).
_SOC_TABLE = np.linspace(0.0, 1.0, 11)
_DATASHEET_V_NORM = {
    "li_ion": np.array([0.02, 0.48, 0.62, 0.68, 0.72, 0.76, 0.80, 0.85, 0.90, 0.95, 0.995]),
    "na_ion": np.array([0.02, 0.22, 0.36, 0.45, 0.53, 0.61, 0.69, 0.76, 0.84, 0.92, 0.995]),
}


def _plant_ocv(chemistry: str, soc: float, empirical_ocv: EmpiricalOCVCurve | None = None) -> float:
    """Evaluate charging plant equilibrium OCV(SoC) with slight nonlinear charge-knee mismatch."""
    spec = CHEMISTRY_SPECS[chemistry]
    s = float(np.clip(soc, 0.0, 1.0))
    if empirical_ocv is not None:
        v_norm = float(empirical_ocv.eval_v_norm(np.array([s]))[0])
    else:
        v_norm = float(np.interp(s, _SOC_TABLE, _DATASHEET_V_NORM[chemistry]))
    # Subtle unmodeled charge-direction OCV hysteresis & high-SoC diffusion knee (+12 mV)
    v_ocv = spec.v_min + v_norm * (spec.v_max - spec.v_min) + 0.012 * np.sin(np.pi * s)
    return float(np.clip(v_ocv, spec.v_min, spec.v_max - 0.01))


@dataclass
class ChargingTrace:
    chemistry: str
    scenario: str
    strategy: str
    time_min: np.ndarray
    soc_true: np.ndarray
    soc_est: np.ndarray
    voltage_v: np.ndarray
    current_c_rate: np.ndarray
    temp_c: np.ndarray
    metrics: dict


def _build_realtime_physics_feature_vec(
    chemistry: str,
    q_nom: float,
    soh_est: float,
    v: float,
    curr: float,
    temp: float,
    amb_temp: float,
    hist_v_norm: deque,
    hist_c_rate: deque,
    hist_v_ir: deque,
    prev_v_norm: float,
    prev_c_rate: float,
    dt_s: float,
) -> np.ndarray:
    """Build the 15-element PHYSICS_SOC_FEATURES vector online inside the charging loop."""
    spec = CHEMISTRY_SPECS[chemistry]
    v_span = spec.v_max - spec.v_min
    # Controller's nominal 1-RC resistance estimate (intentionally differs from 2-RC plant resistance!)
    r_ref = spec.r_int_bol * (spec.default_q_nom / q_nom) ** 0.72 * (1.0 + 1.50 * (1.0 - soh_est))

    v_norm = (v - spec.v_min) / v_span
    c_rate = curr / q_nom
    p_norm = v_norm * c_rate
    temp_norm = (temp - 25.0) / 25.0
    temp_rise_norm = (temp - amb_temp) / 20.0
    v_ir_comp_norm = float(
        np.clip(
            (v - curr * r_ref * np.exp(0.020 * (25.0 - temp)) - spec.v_min) / v_span,
            -0.15,
            1.15,
        )
    )

    hist_v_norm.append(v_norm)
    hist_c_rate.append(c_rate)
    hist_v_ir.append(v_ir_comp_norm)

    v_rm = float(np.mean(hist_v_norm))
    v_rs = float(np.std(hist_v_norm)) if len(hist_v_norm) > 1 else 0.0
    c_rm = float(np.mean(hist_c_rate))
    c_rs = float(np.std(hist_c_rate)) if len(hist_c_rate) > 1 else 0.0
    vir_rm = float(np.mean(hist_v_ir))

    dv_dt = (v_norm - prev_v_norm) / max(dt_s, 1.0)
    dc_dt = (c_rate - prev_c_rate) / max(dt_s, 1.0)

    return np.array(
        [
            [
                v_norm,
                c_rate,
                p_norm,
                temp_norm,
                temp_rise_norm,
                v_rm,
                v_rs,
                c_rm,
                c_rs,
                dv_dt,
                dc_dt,
                v_ir_comp_norm,
                vir_rm,
                v_norm ** 2,
                v_norm ** 3,
            ]
        ],
        dtype=np.float64,
    )


def simulate_charging_episode(
    chemistry: str,
    strategy: str,  # "CC-CV" | "Adaptive_ML"
    soh_true: float,
    soh_est: float,
    soc_predictor,  # callable(X_phys_1x15) -> array([soc_hat])
    q_nom: float | None = None,
    amb_temp: float = 25.0,
    soc_start: float = 0.10,
    soc_target: float = 0.90,
    dt_s: float = 10.0,
    max_time_s: float = 10800.0,
    empirical_ocv: EmpiricalOCVCurve | None = None,
    plant_mismatch_pct: float = 0.12,
) -> ChargingTrace:
    """Simulate a charging episode on an independent 2-RC electro-thermal plant with plant–model mismatch."""
    spec = CHEMISTRY_SPECS[chemistry]
    q_nom = q_nom if q_nom is not None else spec.default_q_nom
    usable_cap = q_nom * soh_true

    # Independent 2-RC plant parameters (intentionally mismatched by +plant_mismatch_pct vs nominal spec!)
    r0_base = spec.r_int_bol * (1.0 + plant_mismatch_pct) * (spec.default_q_nom / q_nom) ** 0.72
    aging_mult = 1.0 + 1.95 * (1.0 - soh_true)
    c_th = 42.0 * (1.0 - 0.06 * plant_mismatch_pct) * (q_nom / 2.0) ** 0.75
    h_a = 0.082 * (q_nom / 2.0) ** 0.60

    # Two-branch polarization (fast charge-transfer + slow solid-state diffusion)
    tau1 = 32.0 if chemistry == "li_ion" else 26.0
    tau2 = 175.0 if chemistry == "li_ion" else 145.0
    v_hyst_charge = 0.018 if chemistry == "li_ion" else 0.024

    n_max = int(max_time_s / dt_s) + 1
    times: list[float] = []
    soc_trues: list[float] = []
    soc_ests: list[float] = []
    volts: list[float] = []
    c_rates: list[float] = []
    temps: list[float] = []

    soc = soc_start
    # Realistic initial observer offset (+1.5% SoC initial error to test observer convergence!)
    soc_hat = float(np.clip(soc_start + 0.015, 0.0, 1.0))
    temp = amb_temp
    v_rc1 = 0.0
    v_rc2 = 0.0

    e_in_wh = 0.0
    e_stored_wh = 0.0
    deg_accum = 0.0

    hist_v = deque(maxlen=6)
    hist_c = deque(maxlen=6)
    hist_vir = deque(maxlen=6)
    v_init = _plant_ocv(chemistry, soc, empirical_ocv)
    prev_v_norm = (v_init - spec.v_min) / (spec.v_max - spec.v_min)
    prev_c_rate = 0.0

    cc_base_c_rate = 1.0 if chemistry == "li_ion" else 0.9

    for step in range(n_max):
        t_now = step * dt_s
        r0 = r0_base * aging_mult * np.exp(0.022 * (25.0 - temp))
        r1 = r0 * 0.42
        r2 = r0 * 0.28  # Unmodeled slow diffusion resistance
        ocv = _plant_ocv(chemistry, soc, empirical_ocv)

        i_v_limit = max(0.0, (spec.v_max - ocv - v_rc1 - v_rc2 - v_hyst_charge) / max(r0, 1e-4))
        c_v_limit = i_v_limit / q_nom

        if strategy == "CC-CV":
            c_cmd = min(cc_base_c_rate, c_v_limit)
            c_cmd = max(0.05, c_cmd) if soc < soc_target else 0.0
        else:
            if soc_hat < 0.45:
                c_soc = 1.00 if chemistry == "li_ion" else 0.90
            elif soc_hat < 0.65:
                c_soc = 0.85 if chemistry == "li_ion" else 0.78
            elif soc_hat < 0.80:
                c_soc = 0.62 if chemistry == "li_ion" else 0.58
            else:
                c_soc = 0.40 if chemistry == "li_ion" else 0.38

            soh_factor = float(np.clip(0.68 + 0.32 * ((soh_est - 0.70) / 0.30), 0.70, 1.00))

            t_warn = 28.5
            if temp <= t_warn:
                t_factor = 1.0
            elif temp < spec.t_max_safe:
                t_factor = max(0.35, 1.0 - 0.65 * (temp - t_warn) / (spec.t_max_safe - t_warn))
            else:
                t_factor = 0.20

            c_cmd = min(c_soc * soh_factor * t_factor, spec.max_c_rate_charge, c_v_limit)
            c_cmd = max(0.05, c_cmd)

        curr = c_cmd * q_nom
        alpha1 = np.exp(-dt_s / tau1)
        alpha2 = np.exp(-dt_s / tau2)
        v_rc1 = alpha1 * v_rc1 + r1 * (1.0 - alpha1) * curr
        v_rc2 = alpha2 * v_rc2 + r2 * (1.0 - alpha2) * curr
        v_term = min(spec.v_max, ocv + curr * r0 + v_rc1 + v_rc2 + v_hyst_charge)

        # Innovation-gated adaptive complementary BMS observer (EKF-inspired polarization gating):
        # Under heavy polarization (high C-rate or voltage-clamped aged CV), downweight raw voltage-ML
        # innovation and clamp outliers so unmodeled 2-RC polarization cannot cause observer windup.
        if soc_predictor is not None:
            feat_vec = _build_realtime_physics_feature_vec(
                chemistry=chemistry,
                q_nom=q_nom,
                soh_est=soh_est,
                v=v_term,
                curr=curr,
                temp=temp,
                amb_temp=amb_temp,
                hist_v_norm=hist_v,
                hist_c_rate=hist_c,
                hist_v_ir=hist_vir,
                prev_v_norm=prev_v_norm,
                prev_c_rate=prev_c_rate,
                dt_s=dt_s,
            )
            prev_v_norm = float(feat_vec[0, 0])
            prev_c_rate = float(feat_vec[0, 1])
            ml_soc = float(soc_predictor(feat_vec)[0])

            # Mild realistic current-sensor drift (0.5%) in Coulomb propagation
            i_meas = curr * 1.005
            soc_prop = float(np.clip(soc_hat + spec.coulombic_eff * i_meas * dt_s / (3600.0 * q_nom * soh_est), 0.0, 1.0))

            # Dynamic polarization-aware gain + Huber-style innovation clamp
            innovation = float(np.clip(ml_soc - soc_prop, -0.06, 0.06))
            pol_damping = 1.0 / (1.0 + 4.5 * (c_cmd ** 2) + 3.0 * (1.0 - soh_est))
            gain = 0.025 * pol_damping
            soc_hat = float(np.clip(soc_prop + gain * innovation, 0.0, 1.0))
        else:
            soc_hat = soc

        times.append(t_now / 60.0)
        soc_trues.append(soc)
        soc_ests.append(soc_hat)
        volts.append(v_term)
        c_rates.append(c_cmd)
        temps.append(temp)

        e_in_wh += (v_term * curr * dt_s) / 3600.0
        e_stored_wh += (ocv * curr * dt_s) / 3600.0

        arrh_thermal = np.exp(0.055 * max(0.0, temp - 25.0))
        plating_risk = 1.0 + 4.2 * (max(0.0, soc - 0.68) ** 1.6) * (c_cmd ** 1.7)
        aging_vuln = 1.0 + 1.5 * (1.0 - soh_true)
        base_deg_rate = 0.018 if chemistry == "li_ion" else 0.022
        dsoc = spec.coulombic_eff * curr * dt_s / (3600.0 * usable_cap)
        deg_accum += base_deg_rate * dsoc * arrh_thermal * (c_cmd ** 0.55) * plating_risk * aging_vuln

        if soc >= soc_target:
            break

        # 2-RC Joule heat dissipation
        q_gen = (curr ** 2) * (r0 + 0.5 * r1 + 0.4 * r2)
        temp = temp + (q_gen - h_a * (temp - amb_temp)) * (dt_s / c_th)
        soc = float(np.clip(soc + dsoc, 0.0, 1.0))

    soc_true_arr = np.array(soc_trues)
    soc_est_arr = np.array(soc_ests)
    temp_arr = np.array(temps)
    eff_pct = 100.0 * e_stored_wh / max(e_in_wh, 1e-9)

    scenario_label = f"{chemistry}_soh{int(round(soh_true * 100))}"
    metrics = {
        "chemistry": chemistry,
        "scenario": scenario_label,
        "soh_true": round(soh_true, 2),
        "soh_est": round(soh_est, 3),
        "strategy": strategy,
        "plant_model": f"2RC_mismatched_+{int(round(plant_mismatch_pct * 100))}pct",
        "charge_time_min": round(float(times[-1]), 2),
        "peak_temp_c": round(float(np.max(temp_arr)), 2),
        "peak_temp_rise_c": round(float(np.max(temp_arr) - amb_temp), 2),
        "mean_temp_rise_c": round(float(np.mean(temp_arr) - amb_temp), 2),
        "energy_efficiency_pct": round(float(eff_pct), 2),
        "estimated_degradation_pct": round(float(deg_accum), 4),
        "soc_observer_mae": round(float(np.mean(np.abs(soc_true_arr - soc_est_arr))), 4),
        "max_voltage_v": round(float(np.max(volts)), 3),
        "safety_limits_respected": bool(
            np.max(volts) <= spec.v_max + 0.01 and np.max(temp_arr) <= spec.t_max_safe + 0.1
        ),
    }

    return ChargingTrace(
        chemistry=chemistry,
        scenario=scenario_label,
        strategy=strategy,
        time_min=np.array(times),
        soc_true=soc_true_arr,
        soc_est=soc_est_arr,
        voltage_v=np.array(volts),
        current_c_rate=np.array(c_rates),
        temp_c=temp_arr,
        metrics=metrics,
    )


def run_adaptive_charging_benchmark(
    li_soc_predictor,
    na_tl_soc_predictor,
    li_soh_mae: float = 0.01,
    na_soh_mae: float = 0.015,
    empirical_ocv_curves: dict[str, EmpiricalOCVCurve] | None = None,
) -> tuple[list[dict], dict[str, ChargingTrace]]:
    """Compare Conventional CC-CV vs ML-State-Informed Adaptive Charging across Li-ion & Na-ion (Fresh & Aged)."""
    ocvs = empirical_ocv_curves or {}
    scenarios = [
        ("li_ion", 0.98, min(1.0, 0.98 - li_soh_mae * 0.4), li_soc_predictor, 2.0),
        ("li_ion", 0.80, max(0.70, 0.80 + li_soh_mae * 0.5), li_soc_predictor, 2.0),
        ("na_ion", 0.98, min(1.0, 0.98 - na_soh_mae * 0.4), na_tl_soc_predictor, 3.2),
        ("na_ion", 0.80, max(0.70, 0.80 + na_soh_mae * 0.5), na_tl_soc_predictor, 3.2),
    ]

    summary_rows: list[dict] = []
    traces: dict[str, ChargingTrace] = {}

    for chem, soh_true, soh_est, predictor, q_nom in scenarios:
        for strat in ("CC-CV", "Adaptive_ML"):
            tr = simulate_charging_episode(
                chemistry=chem,
                strategy=strat,
                soh_true=soh_true,
                soh_est=soh_est,
                soc_predictor=predictor,
                q_nom=q_nom,
                empirical_ocv=ocvs.get(chem),
            )
            summary_rows.append(tr.metrics)
            traces[f"{tr.scenario}_{strat}"] = tr

    return summary_rows, traces
