"""Coupled Electro-Thermal-Aging Battery Charging Simulator: ML-Informed Adaptive Charging vs CC-CV Baseline.

Note (per Proposal Section 5.7): This module performs a physics-based closed-loop simulation
evaluating state-informed adaptive charging against CC-CV under defined voltage, current,
and thermal safety constraints. It is a simulation study, not physical hardware validation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from src.config import CHEMISTRY_SPECS
from src.data.physics_simulator import ocv_curve


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
    r_ref = spec.r_int_bol * (spec.default_q_nom / q_nom) ** 0.72 * (1.0 + 1.5 * (1.0 - soh_est))

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
) -> ChargingTrace:
    """Simulate a single charging episode from soc_start to soc_target under strict safety constraints."""
    spec = CHEMISTRY_SPECS[chemistry]
    q_nom = q_nom if q_nom is not None else spec.default_q_nom
    usable_cap = q_nom * soh_true

    # Coupled electro-thermal parameters
    r0_base = spec.r_int_bol * (spec.default_q_nom / q_nom) ** 0.72
    aging_mult = 1.0 + 1.85 * (1.0 - soh_true)
    c_th = 42.0 * (q_nom / 2.0) ** 0.75
    h_a = 0.082 * (q_nom / 2.0) ** 0.60

    n_max = int(max_time_s / dt_s) + 1
    times = []
    soc_trues = []
    soc_ests = []
    volts = []
    c_rates = []
    temps = []

    soc = soc_start
    soc_hat = soc_start
    temp = amb_temp
    v_rc = 0.0
    tau1 = 35.0 if chemistry == "li_ion" else 28.0

    # Energy & degradation accumulators
    e_in_wh = 0.0
    e_stored_wh = 0.0
    deg_accum = 0.0  # Semi-empirical Arrhenius + high-SoC plating + C-rate stress (%)

    # Online feature rolling buffers for ML SoC observer
    hist_v = deque(maxlen=6)
    hist_c = deque(maxlen=6)
    hist_vir = deque(maxlen=6)
    v_init = float(ocv_curve(chemistry, np.array([soc]))[0])
    prev_v_norm = (v_init - spec.v_min) / (spec.v_max - spec.v_min)
    prev_c_rate = 0.0

    # Conventional CC-CV rate vs Adaptive policy envelope
    cc_base_c_rate = 1.0 if chemistry == "li_ion" else 0.9

    for step in range(n_max):
        t_now = step * dt_s
        r0 = r0_base * aging_mult * np.exp(0.022 * (25.0 - temp))
        r1 = r0 * 0.45
        ocv = float(ocv_curve(chemistry, np.array([soc]))[0])

        # Maximum current that respects upper cut-off voltage V_max
        i_v_limit = max(0.0, (spec.v_max - ocv - v_rc) / max(r0, 1e-4))
        c_v_limit = i_v_limit / q_nom

        if strategy == "CC-CV":
            # Fixed CC up to V_max, then CV taper
            c_cmd = min(cc_base_c_rate, c_v_limit)
            c_cmd = max(0.05, c_cmd) if soc < soc_target else 0.0
        else:
            # --- ML-State-Informed Multi-Stage Adaptive Charging Controller ---
            # 1. SoC-stage profile (respects CC-CV nominal ceiling at low SoC, smoothly tapers at high SoC
            #    to prevent Li/Na plating and excessive Joule heating)
            if soc_hat < 0.45:
                c_soc = 1.00 if chemistry == "li_ion" else 0.90
            elif soc_hat < 0.65:
                c_soc = 0.85 if chemistry == "li_ion" else 0.78
            elif soc_hat < 0.80:
                c_soc = 0.62 if chemistry == "li_ion" else 0.58
            else:
                c_soc = 0.40 if chemistry == "li_ion" else 0.38

            # 2. SoH-aware health derating (protects aged high-resistance cells from Joule heating & fade)
            soh_factor = float(np.clip(0.68 + 0.32 * ((soh_est - 0.70) / 0.30), 0.70, 1.00))

            # 3. Thermal governor (smooth proportional derating above 28.5 C, strict safety cap at T_max)
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
        alpha = np.exp(-dt_s / tau1)
        v_rc = alpha * v_rc + r1 * (1.0 - alpha) * curr
        v_term = min(spec.v_max, ocv + curr * r0 + v_rc)

        # Online ML/TL SoC prediction fused with short-horizon Coulomb consistency (standard BMS observer)
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
            # Complementary filter (standard BMS observer gain: 0.96 Coulomb propagation + 0.04 ML drift correction)
            soc_prop = float(np.clip(soc_hat + spec.coulombic_eff * curr * dt_s / (3600.0 * q_nom * soh_est), 0.0, 1.0))
            soc_hat = float(np.clip(0.96 * soc_prop + 0.04 * ml_soc, 0.0, 1.0))
        else:
            soc_hat = soc

        times.append(t_now / 60.0)
        soc_trues.append(soc)
        soc_ests.append(soc_hat)
        volts.append(v_term)
        c_rates.append(c_cmd)
        temps.append(temp)

        # Energy accounting
        e_in_wh += (v_term * curr * dt_s) / 3600.0
        e_stored_wh += (ocv * curr * dt_s) / 3600.0

        # Coupled degradation stress model:
        # Combines Arrhenius thermal stress + C-rate^1.5 stress + high-SoC (>0.70) Li/Na plating penalty
        arrh_thermal = np.exp(0.055 * max(0.0, temp - 25.0))
        plating_risk = 1.0 + 4.2 * (max(0.0, soc - 0.68) ** 1.6) * (c_cmd ** 1.7)
        aging_vuln = 1.0 + 1.5 * (1.0 - soh_true)
        base_deg_rate = 0.018 if chemistry == "li_ion" else 0.022  # % capacity fade per nominal full charge
        dsoc = spec.coulombic_eff * curr * dt_s / (3600.0 * usable_cap)
        deg_accum += base_deg_rate * dsoc * arrh_thermal * (c_cmd ** 0.55) * plating_risk * aging_vuln

        if soc >= soc_target:
            break

        # Advance physical state
        q_gen = (curr ** 2) * (r0 + 0.5 * r1)
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
) -> tuple[list[dict], dict[str, ChargingTrace]]:
    """Compare Conventional CC-CV vs ML-State-Informed Adaptive Charging across Li-ion & Na-ion (Fresh & Aged)."""
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
            )
            summary_rows.append(tr.metrics)
            traces[f"{tr.scenario}_{strat}"] = tr

    return summary_rows, traces
