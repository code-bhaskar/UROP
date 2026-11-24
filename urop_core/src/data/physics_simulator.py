"""Physics-informed electro-thermal-aging battery dataset simulator (fallback when Git LFS blobs are unavailable).

Models realistic electrochemical differences between:
- Lithium-ion (NASA PCoE 18650: 2.0 Ah, 2.5-4.2 V, flatter plateau OCV)
- Sodium-ion (Transimage 3.2 Ah & HiNa 10.0 Ah: 1.5-3.9 V, steeper sloping OCV)

All records are explicitly tagged with data_source='physics_simulated_fallback'.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.config import CHEMISTRY_SPECS, PipelineConfig


def ocv_li_ion(soc: np.ndarray) -> np.ndarray:
    """Characteristic LCO/NMC 18650 OCV(SoC) curve: 2.5 V to 4.2 V with flat mid-SoC plateau."""
    s = np.clip(soc, 0.0, 1.0)
    # Plateau around 3.65-3.90 V, steep knees below 0.12 and above 0.90
    v = (
        2.55
        + 1.02 * (1.0 - np.exp(-14.0 * s))
        + 0.36 * s
        + 0.18 * np.power(s, 3.2)
        + 0.09 * np.sin(np.pi * s)
    )
    return np.clip(v, 2.50, 4.20)


def ocv_na_ion(soc: np.ndarray) -> np.ndarray:
    """Characteristic layered-oxide / hard-carbon Na-ion OCV(SoC): 1.5 V to 3.9 V, steeper sloping curve."""
    s = np.clip(soc, 0.0, 1.0)
    # Hard-carbon low-voltage plateau below 0.20 + steep sloping layered-oxide region 0.20-0.95
    v = (
        1.55
        + 0.52 * (1.0 - np.exp(-9.0 * s))
        + 1.48 * np.power(s, 0.88)
        + 0.35 * np.power(s, 2.6)
    )
    return np.clip(v, 1.50, 3.90)


def ocv_curve(chemistry: str, soc: np.ndarray) -> np.ndarray:
    if chemistry == "li_ion":
        return ocv_li_ion(soc)
    return ocv_na_ion(soc)


def _simulate_dynamic_profile(
    rng: np.random.Generator,
    chemistry: str,
    cell_id: str,
    protocol: str,
    q_nom: float,
    amb_temp: float,
    soh: float,
    cycle_num: int,
    n_steps: int,
    dt_s: float = 10.0,
) -> pl.DataFrame:
    """Simulate a 1-RC electro-thermal dynamic discharge/pulse/driving-cycle time series."""
    spec = CHEMISTRY_SPECS[chemistry]
    usable_cap = q_nom * soh * (1.0 - 0.0025 * max(0.0, 25.0 - amb_temp))

    # Arrhenius temperature scaling for internal resistance
    r0_base = spec.r_int_bol * (spec.default_q_nom / q_nom) ** 0.72
    temp_factor = np.exp(0.022 * (25.0 - amb_temp))
    aging_factor = 1.0 + 1.65 * (1.0 - soh)
    r0 = r0_base * temp_factor * aging_factor * rng.uniform(0.93, 1.07)
    r1 = r0 * 0.48
    tau1 = 35.0 if chemistry == "li_ion" else 28.0

    # Thermal parameters
    c_th = 45.0 * (q_nom / 2.0) ** 0.75
    h_a = 0.085 * (q_nom / 2.0) ** 0.60

    soc = np.zeros(n_steps, dtype=np.float64)
    v_meas = np.zeros(n_steps, dtype=np.float64)
    i_meas = np.zeros(n_steps, dtype=np.float64)
    t_meas = np.zeros(n_steps, dtype=np.float64)
    time_s = np.arange(n_steps, dtype=np.float64) * dt_s

    soc[0] = rng.uniform(0.96, 1.00)
    t_meas[0] = amb_temp + rng.normal(0.0, 0.2)
    v_rc = 0.0

    for k in range(n_steps):
        if protocol == "pulse":
            # Enhanced pulse test: alternating rest and 0.25C to 2.5C discharge pulses
            phase = k % 16
            if phase < 4:
                c_rate = -rng.choice([0.5, 1.0, 1.5, 2.0, 2.5])
            elif phase < 9:
                c_rate = -0.35
            else:
                c_rate = 0.0
        elif protocol == "driving_cycle":
            # FUDS/UDDS/DST-inspired dynamic load with brief regenerative braking pulses
            base = -0.65 - 0.45 * np.sin(2.0 * np.pi * k / 28.0) - 0.30 * np.cos(2.0 * np.pi * k / 11.0)
            burst = rng.normal(0.0, 0.45)
            c_rate = float(np.clip(base + burst, -2.5, 0.8))
        elif protocol == "cc_cv_charge":
            # Standard CC-CV charging trajectory (mirrors NASA charge cycles & Wang et al. 0.5C Na-ion pre-charge)
            if k == 0:
                soc[0] = rng.uniform(0.05, 0.14)
            ocv_now = float(ocv_curve(chemistry, np.array([soc[k]]))[0])
            c_cc = 0.90 if chemistry == "li_ion" else 0.75
            c_vlim = max(0.04, (spec.v_max - ocv_now - v_rc) / max(r0 * q_nom, 1e-4))
            c_rate = float(min(c_cc, c_vlim)) if soc[k] < 0.985 else 0.0
        else:
            # Constant-current discharge with mild load variations (NASA aging discharge)
            c_rate = -1.0 + rng.normal(0.0, 0.04)

        # Keep cell inside safe SoC bounds (except cc_cv_charge which monotonically charges to ~99%)
        if protocol != "cc_cv_charge":
            if soc[k] <= 0.04 and c_rate < 0:
                c_rate = 0.65  # brief recharge segment
            elif soc[k] >= 0.99 and c_rate > 0:
                c_rate = -0.5

        curr = c_rate * q_nom
        i_meas[k] = curr

        # 1-RC polarization update
        alpha = np.exp(-dt_s / tau1)
        v_rc = alpha * v_rc + r1 * (1.0 - alpha) * curr

        # Hysteresis + OCV + ohmic/RC polarization
        ocv = float(ocv_curve(chemistry, np.array([soc[k]]))[0])
        hyst = 0.015 * np.sign(curr) * (1.0 if chemistry == "li_ion" else 1.45)
        v_term = ocv + curr * r0 + v_rc + hyst + rng.normal(0.0, 0.004)
        v_meas[k] = float(np.clip(v_term, spec.v_min - 0.05, spec.v_max + 0.05))

        # Thermal ODE step
        q_gen = (curr ** 2) * (r0 + 0.5 * r1)
        dt_cell = (q_gen - h_a * (t_meas[k] - amb_temp)) * (dt_s / c_th)

        if k + 1 < n_steps:
            t_meas[k + 1] = t_meas[k] + dt_cell + rng.normal(0.0, 0.05)
            dsoc = spec.coulombic_eff * curr * dt_s / (3600.0 * usable_cap)
            soc[k + 1] = float(np.clip(soc[k] + dsoc, 0.0, 1.0))

    # Recompute exact discrete Coulomb-counting reference SoC with tiny sensor noise on I
    i_noisy = i_meas + rng.normal(0.0, 0.004 * q_nom, size=n_steps)
    v_noisy = v_meas + rng.normal(0.0, 0.003, size=n_steps)
    t_noisy = t_meas + rng.normal(0.0, 0.08, size=n_steps)

    return pl.DataFrame(
        {
            "chemistry": [chemistry] * n_steps,
            "cell_id": [cell_id] * n_steps,
            "protocol": [protocol] * n_steps,
            "cycle_number": np.full(n_steps, cycle_num, dtype=np.int32),
            "q_nom": np.full(n_steps, q_nom, dtype=np.float64),
            "ambient_temp_c": np.full(n_steps, amb_temp, dtype=np.float64),
            "time_s": time_s,
            "voltage_v": v_noisy,
            "current_a": i_noisy,
            "temp_c": t_noisy,
            "usable_capacity_ah": np.full(n_steps, usable_cap, dtype=np.float64),
            "soc": soc,
            "soh": np.full(n_steps, soh, dtype=np.float64),
            "data_source": ["physics_simulated_fallback"] * n_steps,
        }
    )


def _simulate_aging_cycles(
    rng: np.random.Generator,
    chemistry: str,
    cell_id: str,
    q_nom: float,
    amb_temp: float,
    n_cycles: int,
) -> list[dict]:
    """Simulate realistic cycle-level degradation records (SoH 1.00 -> ~0.72) with chemistry-specific physics."""
    spec = CHEMISTRY_SPECS[chemistry]
    rows: list[dict] = []

    ref_cap = q_nom * rng.uniform(0.95, 1.02)
    r0_bol = spec.r_int_bol * (spec.default_q_nom / q_nom) ** 0.72 * np.exp(0.018 * (25.0 - amb_temp))
    fade_rate = (0.00145 if chemistry == "li_ion" else 0.00175) * rng.uniform(0.85, 1.18)
    knee_cycle = int(n_cycles * rng.uniform(0.68, 0.85))
    temp_stress = 1.0 + 0.012 * abs(amb_temp - 25.0)

    # Initial reference discharge / charge timings (1C discharge, 0.75C CC-CV charge)
    t_dis_0 = 3600.0 * (ref_cap / q_nom) * rng.uniform(0.97, 1.03)
    t_cc_0 = 3400.0 * rng.uniform(0.95, 1.05)
    t_cv_0 = 1100.0 * rng.uniform(0.92, 1.08)
    ica_peak_0 = (3.8 if chemistry == "li_ion" else 2.1) * rng.uniform(0.94, 1.06)
    ica_v_peak_0 = (0.68 if chemistry == "li_ion" else 0.54) + rng.normal(0.0, 0.015)
    temp_rise_0 = (5.2 if chemistry == "li_ion" else 4.1) * (q_nom / spec.default_q_nom) ** 0.3

    for c in range(1, n_cycles + 1):
        # Semi-empirical SEI sqrt(n) + linear + late-life knee fade + occasional capacity regeneration
        sei_fade = 0.0085 * np.sqrt(c) * temp_stress
        lin_fade = fade_rate * c * temp_stress
        knee_fade = 0.0018 * max(0, c - knee_cycle) ** 1.25 / 10.0
        regen = 0.014 if (c % 22 == 0) else (0.007 if (c % 22 == 1) else 0.0)
        soh = float(np.clip(1.0 - sei_fade - lin_fade - knee_fade + regen + rng.normal(0.0, 0.0035), 0.65, 1.01))
        cap = ref_cap * soh

        deg = 1.0 - soh  # degradation fraction [0, 0.35]
        # Shared physical degradation signatures (modulated by chemistry-specific constants)
        r_mult = 1.0 + (2.15 if chemistry == "li_ion" else 2.35) * deg + rng.normal(0.0, 0.025)
        r_est = r0_bol * max(0.95, r_mult)

        t_dis = t_dis_0 * (soh * (1.0 - 0.10 * deg) + rng.normal(0.0, 0.008))
        t_cc = t_cc_0 * (1.0 - 1.38 * deg + rng.normal(0.0, 0.015))
        t_cv = t_cv_0 * (1.0 + 2.05 * deg + rng.normal(0.0, 0.025))
        t_chg = max(600.0, t_cc + t_cv)

        mean_v = (
            spec.v_nom
            - (0.14 if chemistry == "li_ion" else 0.21) * deg
            + rng.normal(0.0, 0.008)
        )
        v_norm_int = (mean_v - spec.v_min) / (spec.v_max - spec.v_min)
        energy_wh = cap * mean_v * rng.uniform(0.985, 1.015)

        ica_peak = ica_peak_0 * (1.0 - 1.55 * deg + rng.normal(0.0, 0.02))
        ica_v_peak = ica_v_peak_0 + 0.16 * deg + rng.normal(0.0, 0.006)
        temp_rise = temp_rise_0 * (1.0 + 1.80 * deg + rng.normal(0.0, 0.03))
        mean_temp = amb_temp + 0.55 * temp_rise
        max_temp = amb_temp + temp_rise

        rows.append(
            {
                "chemistry": chemistry,
                "cell_id": cell_id,
                "cycle_number": c,
                "q_nom": q_nom,
                "ambient_temp_c": amb_temp,
                "discharge_capacity_ah": float(cap),
                "ref_capacity_ah": float(ref_cap),
                "soh": soh,
                "discharge_duration_s": float(max(300.0, t_dis)),
                "charge_duration_s": float(t_chg),
                "cc_duration_s": float(max(200.0, t_cc)),
                "cv_duration_s": float(max(150.0, t_cv)),
                "mean_voltage_v": float(mean_v),
                "min_voltage_v": float(spec.v_min + rng.uniform(0.0, 0.12)),
                "max_voltage_v": float(spec.v_max - rng.uniform(0.0, 0.04)),
                "mean_current_a": float(q_nom * rng.uniform(0.95, 1.05)),
                "mean_temp_c": float(mean_temp),
                "max_temp_c": float(max_temp),
                "temp_rise_c": float(max(0.5, temp_rise)),
                "r_int_est_ohm": float(r_est),
                "energy_wh": float(energy_wh),
                "ica_peak": float(max(0.2, ica_peak)),
                "ica_v_norm_peak": float(np.clip(ica_v_peak, 0.1, 0.95)),
                "v_norm_integral": float(np.clip(v_norm_int, 0.1, 0.95)),
                "data_source": "physics_simulated_fallback",
            }
        )

    return rows


def _inject_realistic_raw_imperfections(df: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Inject a small realistic fraction (~0.4%) of nulls, duplicates, and sensor spikes for the Quality Audit."""
    n = df.height
    if n < 200:
        return df
    pdf = df.to_pandas()
    # 0.15% missing voltage/temp readings
    n_null = max(2, int(n * 0.0015))
    null_idx = rng.choice(n, size=n_null, replace=False)
    pdf.loc[null_idx[: n_null // 2], "voltage_v"] = np.nan
    pdf.loc[null_idx[n_null // 2 :], "temp_c"] = np.nan

    # 0.15% physically impossible sensor spikes (to be flagged by quality audit)
    n_spike = max(2, int(n * 0.0015))
    spike_idx = rng.choice(n, size=n_spike, replace=False)
    pdf.loc[spike_idx[: n_spike // 2], "voltage_v"] = 9.99
    pdf.loc[spike_idx[n_spike // 2 :], "temp_c"] = 145.0

    # 0.10% duplicate rows
    n_dup = max(2, int(n * 0.001))
    dup_rows = pdf.sample(n=n_dup, random_state=int(rng.integers(0, 10000)))
    pdf = pl.concat([pl.from_pandas(pdf), pl.from_pandas(dup_rows)], how="vertical")
    return pdf


def generate_physics_fallback_datasets(cfg: PipelineConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate both Li-ion and Na-ion time-series (SoC) and cycle-aging (SoH) Polars DataFrames."""
    rng = np.random.default_rng(cfg.seed)
    steps_per_profile = 180 if cfg.fast_mode else 340
    n_cycles = 70 if cfg.fast_mode else 140

    ts_frames: list[pl.DataFrame] = []
    cycle_rows: list[dict] = []

    # --- 1. NASA PCoE Lithium-ion cells (8 cells: train/val/test cleanly separable by cell_id) ---
    li_cells = [
        ("B0005", 2.0, 24.0),
        ("B0006", 2.0, 24.0),
        ("B0007", 2.0, 24.0),
        ("B0018", 2.0, 24.0),
        ("B0025", 2.0, 15.0),
        ("B0026", 2.0, 35.0),
        ("B0027", 2.0, 24.0),
        ("B0028", 2.0, 40.0),
    ]
    for cell_id, q_nom, amb_t in li_cells:
        # Dynamic, discharge, and CC-CV charge profiles at multiple SoH states
        for c_idx, (proto, soh_val) in enumerate(
            [
                ("discharge_aging", 0.99),
                ("cc_cv_charge", 0.97),
                ("driving_cycle", 0.95),
                ("pulse", 0.90),
                ("driving_cycle", 0.85),
                ("cc_cv_charge", 0.81),
            ],
            start=1,
        ):
            ts_frames.append(
                _simulate_dynamic_profile(
                    rng=rng,
                    chemistry="li_ion",
                    cell_id=cell_id,
                    protocol=proto,
                    q_nom=q_nom,
                    amb_temp=amb_t,
                    soh=soh_val,
                    cycle_num=c_idx * 25,
                    n_steps=steps_per_profile,
                )
            )
        cycle_rows.extend(_simulate_aging_cycles(rng, "li_ion", cell_id, q_nom, amb_t, n_cycles))

    # --- 2. Wang et al. Sodium-ion cells (3.2 Ah & 10.0 Ah across -5 to 45 C) ---
    na_cells = [
        ("na_3.2Ah_bat1", 3.2, [5.0, 25.0, 45.0], "pulse"),
        ("na_3.2Ah_bat2", 3.2, [-5.0, 15.0, 35.0], "pulse"),
        ("na_3.2Ah_bat5", 3.2, [-5.0, 15.0, 25.0, 45.0], "driving_cycle"),
        ("na_10.0Ah_bat3", 10.0, [5.0, 25.0, 45.0], "pulse"),
        ("na_10.0Ah_bat4", 10.0, [-5.0, 15.0, 35.0], "pulse"),
        ("na_10.0Ah_bat6", 10.0, [-5.0, 15.0, 25.0, 35.0], "driving_cycle"),
        ("na_3.2Ah_bat7", 3.2, [15.0, 25.0, 35.0], "driving_cycle"),
        ("na_10.0Ah_bat8", 10.0, [5.0, 25.0, 45.0], "driving_cycle"),
    ]
    for cell_id, q_nom, temps, primary_proto in na_cells:
        for t_idx, amb_t in enumerate(temps):
            soh_val = float(np.clip(0.99 - 0.04 * t_idx, 0.80, 1.0))
            proto = primary_proto if t_idx % 2 == 0 else ("pulse" if primary_proto == "driving_cycle" else "driving_cycle")
            ts_frames.append(
                _simulate_dynamic_profile(
                    rng=rng,
                    chemistry="na_ion",
                    cell_id=cell_id,
                    protocol=proto,
                    q_nom=q_nom,
                    amb_temp=amb_t,
                    soh=soh_val,
                    cycle_num=(t_idx + 1) * 15,
                    n_steps=steps_per_profile,
                )
            )
        # Include CC-CV charge protocol (per Zenodo 13836819 0.5C CC-CV pre-charge)
        ts_frames.append(
            _simulate_dynamic_profile(
                rng=rng,
                chemistry="na_ion",
                cell_id=cell_id,
                protocol="cc_cv_charge",
                q_nom=q_nom,
                amb_temp=temps[1],
                soh=0.95,
                cycle_num=99,
                n_steps=steps_per_profile,
            )
        )
        cycle_rows.extend(_simulate_aging_cycles(rng, "na_ion", cell_id, q_nom, temps[1], n_cycles))

    ts_df = pl.concat(ts_frames, how="vertical")
    ts_df = _inject_realistic_raw_imperfections(ts_df, rng)
    cycle_df = pl.DataFrame(cycle_rows)
    return ts_df, cycle_df
