"""Production Big Data loaders for real NASA PCoE Li-ion (.zip/.mat) and Wang et al. Na-ion (.CSV) datasets."""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
from scipy.io import loadmat

from src.config import CHEMISTRY_SPECS, PipelineConfig

_FILENAME_RE = re.compile(
    r"(?P<protocol>Pulse|Driving_cycle)_data_temp(?P<temp>-?\d+)_bat(?P<bat>\d+)",
    re.IGNORECASE,
)


def _match_column(columns: list[str], patterns: list[str]) -> str | None:
    """Find first column matching any case-insensitive regex pattern."""
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for col in columns:
            if rx.search(col.strip()):
                return col
    return None


def load_real_nasa_li_ion(
    cfg: PipelineConfig,
    zip_paths: list[Path],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load NASA PCoE Li-ion .mat archives into unified timeseries & cycle-level Polars DataFrames."""
    spec = CHEMISTRY_SPECS["li_ion"]
    ts_frames: list[pl.DataFrame] = []
    cycle_rows: list[dict] = []

    for zp in zip_paths:
        with zipfile.ZipFile(zp, "r") as zf:
            mat_names = [n for n in zf.namelist() if n.lower().endswith(".mat") and "__MACOSX" not in n]
            for mat_name in sorted(mat_names):
                cell_id = Path(mat_name).stem
                raw_bytes = zf.read(mat_name)
                mat = loadmat(io.BytesIO(raw_bytes), simplify_cells=True)
                if cell_id not in mat:
                    keys = [k for k in mat.keys() if not k.startswith("__")]
                    if not keys:
                        continue
                    cell_struct = mat[keys[0]]
                else:
                    cell_struct = mat[cell_id]

                cycles = cell_struct.get("cycle", [])
                if isinstance(cycles, dict):
                    cycles = [cycles]

                discharge_idx = 0
                ref_capacity: float | None = None
                prev_charge_dur = np.nan
                prev_cc_dur = np.nan
                prev_cv_dur = np.nan

                for cyc in cycles:
                    ctype = str(cyc.get("type", "")).strip().lower()
                    amb_temp = float(cyc.get("ambient_temperature", 24.0))
                    data = cyc.get("data", {})
                    if not isinstance(data, dict):
                        continue

                    if ctype == "charge":
                        t = np.asarray(data.get("Time", []), dtype=np.float64).ravel()
                        v = np.asarray(data.get("Voltage_measured", []), dtype=np.float64).ravel()
                        i = np.asarray(data.get("Current_measured", []), dtype=np.float64).ravel()
                        if len(t) >= 5:
                            prev_charge_dur = float(t[-1] - t[0])
                            cc_mask = v < (spec.v_max - 0.03)
                            prev_cc_dur = float(t[cc_mask][-1] - t[0]) if np.any(cc_mask) else prev_charge_dur * 0.5
                            prev_cv_dur = max(0.0, prev_charge_dur - prev_cc_dur)
                        continue

                    if ctype != "discharge":
                        continue

                    v = np.asarray(data.get("Voltage_measured", []), dtype=np.float64).ravel()
                    i = np.asarray(data.get("Current_measured", []), dtype=np.float64).ravel()
                    temp = np.asarray(data.get("Temperature_measured", []), dtype=np.float64).ravel()
                    t = np.asarray(data.get("Time", []), dtype=np.float64).ravel()
                    cap_val = data.get("Capacity", np.nan)
                    cap = float(np.asarray(cap_val).ravel()[0]) if np.size(cap_val) > 0 else np.nan

                    n = min(len(v), len(i), len(temp), len(t))
                    if n < 10 or not np.isfinite(cap) or cap <= 0.2:
                        continue

                    discharge_idx += 1
                    v, i, temp, t = v[:n], i[:n], temp[:n], t[:n]
                    if ref_capacity is None:
                        ref_capacity = cap
                    soh = float(np.clip(cap / ref_capacity, 0.4, 1.05))

                    # Discrete Coulomb counting for discharge SoC
                    dt = np.diff(t, prepend=t[0])
                    dt = np.clip(dt, 0.0, 600.0)
                    # Discharge current is negative in NASA dataset
                    i_signed = -np.abs(i) if np.nanmean(i) > 0 else i
                    dq_ah = np.cumsum(spec.coulombic_eff * i_signed * dt / 3600.0)
                    soc = np.clip(1.0 + dq_ah / max(cap, 1e-6), 0.0, 1.0)

                    # Internal resistance estimate from initial voltage drop
                    dv_init = abs(v[0] - v[min(3, n - 1)])
                    di_init = max(abs(i_signed[min(3, n - 1)] - i_signed[0]), abs(i_signed[min(3, n - 1)]), 0.2)
                    r_est = float(np.clip(dv_init / di_init, 0.01, 1.0))

                    # Incremental capacity peak (dQ/dV)
                    v_norm = (v - spec.v_min) / (spec.v_max - spec.v_min)
                    q_cum = np.abs(dq_ah)
                    dv_step = np.gradient(v_norm)
                    dq_step = np.gradient(q_cum / spec.default_q_nom)
                    valid_ica = np.abs(dv_step) > 1e-4
                    if np.any(valid_ica):
                        ica_vals = np.abs(dq_step[valid_ica] / dv_step[valid_ica])
                        ica_peak = float(np.percentile(ica_vals, 95))
                        ica_v_peak = float(v_norm[valid_ica][int(np.argmax(ica_vals))])
                    else:
                        ica_peak, ica_v_peak = 1.0, 0.5

                    energy_wh = float(np.trapezoid(np.abs(v * i_signed), t) / 3600.0)
                    t_dur = float(t[-1] - t[0])

                    cycle_rows.append(
                        {
                            "chemistry": "li_ion",
                            "cell_id": cell_id,
                            "cycle_number": discharge_idx,
                            "q_nom": spec.default_q_nom,
                            "ambient_temp_c": amb_temp,
                            "discharge_capacity_ah": cap,
                            "ref_capacity_ah": ref_capacity,
                            "soh": soh,
                            "discharge_duration_s": t_dur,
                            "charge_duration_s": prev_charge_dur if np.isfinite(prev_charge_dur) else t_dur * 1.4,
                            "cc_duration_s": prev_cc_dur if np.isfinite(prev_cc_dur) else t_dur * 0.9,
                            "cv_duration_s": prev_cv_dur if np.isfinite(prev_cv_dur) else t_dur * 0.5,
                            "mean_voltage_v": float(np.mean(v)),
                            "min_voltage_v": float(np.min(v)),
                            "max_voltage_v": float(np.max(v)),
                            "mean_current_a": float(np.mean(np.abs(i_signed))),
                            "mean_temp_c": float(np.mean(temp)),
                            "max_temp_c": float(np.max(temp)),
                            "temp_rise_c": float(np.max(temp) - temp[0]),
                            "r_int_est_ohm": r_est,
                            "energy_wh": energy_wh,
                            "ica_peak": ica_peak,
                            "ica_v_norm_peak": ica_v_peak,
                            "v_norm_integral": float(np.trapezoid(v_norm, t) / max(t_dur, 1.0)),
                            "data_source": "real_lfs_dataset",
                        }
                    )

                    # Keep representative cycles for time-series SoC dataset
                    stride = max(1, n // 120)
                    idx = np.arange(0, n, stride)
                    ts_frames.append(
                        pl.DataFrame(
                            {
                                "chemistry": ["li_ion"] * len(idx),
                                "cell_id": [cell_id] * len(idx),
                                "protocol": ["discharge_aging"] * len(idx),
                                "cycle_number": np.full(len(idx), discharge_idx, dtype=np.int32),
                                "q_nom": np.full(len(idx), spec.default_q_nom, dtype=np.float64),
                                "ambient_temp_c": np.full(len(idx), amb_temp, dtype=np.float64),
                                "time_s": t[idx].astype(np.float64),
                                "voltage_v": v[idx].astype(np.float64),
                                "current_a": i_signed[idx].astype(np.float64),
                                "temp_c": temp[idx].astype(np.float64),
                                "usable_capacity_ah": np.full(len(idx), cap, dtype=np.float64),
                                "soc": soc[idx].astype(np.float64),
                                "soh": np.full(len(idx), soh, dtype=np.float64),
                                "data_source": ["real_lfs_dataset"] * len(idx),
                            }
                        )
                    )

    ts_df = pl.concat(ts_frames, how="vertical") if ts_frames else pl.DataFrame()
    cycle_df = pl.DataFrame(cycle_rows) if cycle_rows else pl.DataFrame()
    return ts_df, cycle_df


def load_real_na_ion_csvs(
    cfg: PipelineConfig,
    csv_paths_3_2ah: list[Path],
    csv_paths_10ah: list[Path],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Stream-ingest 3.2 Ah & 10 Ah Arbin Na-ion CSV files using Polars lazy engine."""
    spec = CHEMISTRY_SPECS["na_ion"]
    all_specs = [(p, 3.2) for p in csv_paths_3_2ah] + [(p, 10.0) for p in csv_paths_10ah]

    ts_frames: list[pl.DataFrame] = []
    cycle_rows: list[dict] = []

    for csv_path, q_nom in all_specs:
        m = _FILENAME_RE.search(csv_path.stem)
        protocol = m.group("protocol").lower() if m else "pulse"
        amb_temp = float(m.group("temp")) if m else 25.0
        bat_num = m.group("bat") if m else csv_path.stem
        cell_id = f"na_{q_nom:.1f}Ah_bat{bat_num}"

        # Peek schema lazily via Polars
        lf = pl.scan_csv(csv_path, ignore_errors=True, infer_schema_length=500)
        cols = lf.collect_schema().names()

        c_time = _match_column(cols, [r"test_?time", r"^time", r"elapsed", r"sec"])
        c_volt = _match_column(cols, [r"volt", r"^v\b", r"^u\b"])
        c_curr = _match_column(cols, [r"curr", r"^i\b"])
        c_temp = _match_column(cols, [r"temp", r"^t_?cell", r"^aux_?t"])
        c_soc = _match_column(cols, [r"^soc", r"state_?of_?charge"])
        c_cyc = _match_column(cols, [r"cycle"])

        if not c_time or not c_volt or not c_curr:
            continue

        select_exprs = [
            pl.col(c_time).cast(pl.Float64, strict=False).alias("time_s"),
            pl.col(c_volt).cast(pl.Float64, strict=False).alias("voltage_v"),
            pl.col(c_curr).cast(pl.Float64, strict=False).alias("current_a"),
        ]
        if c_temp:
            select_exprs.append(pl.col(c_temp).cast(pl.Float64, strict=False).alias("temp_c"))
        else:
            select_exprs.append(pl.lit(amb_temp).cast(pl.Float64).alias("temp_c"))
        if c_soc:
            select_exprs.append(pl.col(c_soc).cast(pl.Float64, strict=False).alias("soc_raw"))
        if c_cyc:
            select_exprs.append(pl.col(c_cyc).cast(pl.Int32, strict=False).alias("cycle_number"))
        else:
            select_exprs.append(pl.lit(1, dtype=pl.Int32).alias("cycle_number"))

        df = (
            lf.select(select_exprs)
            .drop_nulls(subset=["time_s", "voltage_v", "current_a"])
            .gather_every(cfg.real_csv_downsample)
            .collect(engine="streaming")
        )
        if df.height < 20:
            continue

        t = df["time_s"].to_numpy()
        v = df["voltage_v"].to_numpy()
        i = df["current_a"].to_numpy()
        temp = df["temp_c"].to_numpy()
        dt = np.clip(np.diff(t, prepend=t[0]), 0.0, 600.0)

        if "soc_raw" in df.columns and df["soc_raw"].null_count() < df.height * 0.5:
            soc = df["soc_raw"].to_numpy()
            if np.nanmax(soc) > 1.5:
                soc = soc / 100.0
            soc = np.clip(soc, 0.0, 1.0)
        else:
            # Discrete Coulomb counting anchored at full charge (3.9 V prior to test per Zenodo protocol)
            dq = np.cumsum(spec.coulombic_eff * i * dt / 3600.0)
            usable = max(q_nom, float(np.max(dq) - np.min(dq)))
            soc = np.clip(1.0 + (dq - np.max(dq)) / usable, 0.0, 1.0)

        usable_cap = max(0.5 * q_nom, float(np.sum(np.abs(i) * dt / 3600.0)))
        # Approximate health proxy for characterization test (fresh cell near 1.0)
        soh_est = float(np.clip(min(usable_cap, q_nom) / q_nom, 0.75, 1.0))

        ts_frames.append(
            df.select(["cycle_number", "time_s", "voltage_v", "current_a", "temp_c"]).with_columns(
                pl.lit("na_ion").alias("chemistry"),
                pl.lit(cell_id).alias("cell_id"),
                pl.lit(protocol).alias("protocol"),
                pl.lit(q_nom).alias("q_nom"),
                pl.lit(amb_temp).alias("ambient_temp_c"),
                pl.lit(usable_cap).alias("usable_capacity_ah"),
                pl.Series("soc", soc.astype(np.float64)),
                pl.lit(soh_est).alias("soh"),
                pl.lit("real_lfs_dataset").alias("data_source"),
            )
        )

        # Cycle-segment health record
        dv = np.diff(v, prepend=v[0])
        di = np.diff(i, prepend=i[0])
        pulse_mask = np.abs(di) > 0.25 * q_nom
        r_est = float(np.median(np.abs(dv[pulse_mask] / di[pulse_mask]))) if np.any(pulse_mask) else spec.r_int_bol
        v_norm = (v - spec.v_min) / (spec.v_max - spec.v_min)
        dur = float(max(t[-1] - t[0], 1.0))
        cycle_rows.append(
            {
                "chemistry": "na_ion",
                "cell_id": cell_id,
                "cycle_number": 1,
                "q_nom": q_nom,
                "ambient_temp_c": amb_temp,
                "discharge_capacity_ah": min(usable_cap, q_nom),
                "ref_capacity_ah": q_nom,
                "soh": soh_est,
                "discharge_duration_s": dur,
                "charge_duration_s": dur * 1.2,
                "cc_duration_s": dur * 0.85,
                "cv_duration_s": dur * 0.35,
                "mean_voltage_v": float(np.mean(v)),
                "min_voltage_v": float(np.min(v)),
                "max_voltage_v": float(np.max(v)),
                "mean_current_a": float(np.mean(np.abs(i))),
                "mean_temp_c": float(np.mean(temp)),
                "max_temp_c": float(np.max(temp)),
                "temp_rise_c": float(np.max(temp) - temp[0]),
                "r_int_est_ohm": float(np.clip(r_est, 0.005, 0.5)),
                "energy_wh": float(np.trapezoid(np.abs(v * i), t) / 3600.0),
                "ica_peak": 1.0,
                "ica_v_norm_peak": 0.5,
                "v_norm_integral": float(np.trapezoid(v_norm, t) / dur),
                "data_source": "real_lfs_dataset",
            }
        )

    ts_df = pl.concat(ts_frames, how="vertical") if ts_frames else pl.DataFrame()
    cycle_df = pl.DataFrame(cycle_rows) if cycle_rows else pl.DataFrame()
    return ts_df, cycle_df
