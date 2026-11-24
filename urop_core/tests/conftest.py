"""Shared real-format test fixtures (minimal NASA .zip/.mat & Wang et al. Arbin .CSV files)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat


def _write_nasa_cell_mat(cell_name: str, seed: int) -> bytes:
    """Write a minimal MATLAB .mat file matching the exact NASA PCoE cell.cycle schema."""
    rng = np.random.default_rng(seed)
    cycle_list = []
    q_init = 1.88 + rng.normal(0.0, 0.04)
    for k in range(1, 15):
        cap = max(1.25, q_init * (1.0 - 0.018 * (k - 1)) + rng.normal(0.0, 0.005))
        t_dis = np.linspace(0.0, 3200.0 * (cap / 1.88), 45)
        soc = 1.0 - t_dis / t_dis[-1]
        v_ocv = 2.65 + 1.50 * (0.28 + 0.72 * soc) + 0.14 * np.tanh(7.0 * (soc - 0.12)) - 0.14
        i_dis = np.full_like(t_dis, -2.0)
        r_int = 0.065 * (1.0 + 0.55 * (1.0 - cap / q_init))
        v_dis = np.clip(v_ocv + i_dis * r_int + rng.normal(0.0, 0.006, len(t_dis)), 2.4, 4.25)
        temp_dis = 24.0 + (4.5 + 3.0 * (1.0 - cap / q_init)) * (1.0 - soc) + rng.normal(0.0, 0.15, len(t_dis))
        cycle_list.append(
            {
                "type": "discharge",
                "ambient_temperature": 24.0,
                "data": {
                    "Time": t_dis,
                    "Voltage_measured": v_dis,
                    "Current_measured": i_dis,
                    "Temperature_measured": temp_dis,
                    "Capacity": np.array([[cap]]),
                },
            }
        )
    buf = io.BytesIO()
    savemat(buf, {cell_name: {"cycle": cycle_list}})
    return buf.getvalue()


def _write_arbin_na_csv(path: Path, q_nom: float, seed: int) -> None:
    """Write a minimal Arbin .CSV file matching the exact Wang et al. Zenodo 13836819 schema."""
    rng = np.random.default_rng(seed)
    # Step 2: 0.5C charge to 3.9V; Step 4: 0.2C full discharge (calibration); Step 8: 0.5C recharge; Step 10+: pulses
    rows = ["Test Time (s),Step Index,Cycle Index,Current (A),Voltage (V),Aux_Temperature_1 (C)"]
    t = 0.0

    def add_seg(step_idx: int, curr: float, n_pts: int, dt: float, soc_start: float) -> float:
        nonlocal t
        soc = soc_start
        for _ in range(n_pts):
            soc = float(np.clip(soc + curr * dt / (3600.0 * q_nom), 0.0, 1.0))
            v_ocv = 1.65 + 2.20 * soc + 0.06 * np.sin(np.pi * soc)
            v = float(np.clip(v_ocv + curr * (0.045 / (q_nom / 3.2)) + rng.normal(0.0, 0.005), 1.5, 3.95))
            temp = 25.0 + 0.35 * abs(curr / q_nom) * 5.0 + rng.normal(0.0, 0.1)
            rows.append(f"{t:.1f},{step_idx},1,{curr:.4f},{v:.4f},{temp:.2f}")
            t += dt
        return soc

    soc = add_seg(2, 0.5 * q_nom, 25, 144.0, 0.5)
    soc = add_seg(3, 0.0, 8, 20.0, 1.0)
    soc = add_seg(4, -0.2 * q_nom, 120, 150.0, 1.0)  # exact 1.0 -> 0.0 full 0.2C calibration discharge
    soc = add_seg(8, 0.5 * q_nom, 60, 120.0, 0.0)
    for k in range(8):
        soc = add_seg(10 + 2 * k, -0.75 * q_nom, 18, 40.0, soc)
        soc = add_seg(11 + 2 * k, 0.0, 8, 25.0, soc)
    path.write_text("\n".join(rows), encoding="utf-8")


@pytest.fixture()
def real_format_dataset_root(tmp_path: Path) -> Path:
    """Build a minimal real-format dataset directory (6 NASA cells in .zip + 6 Arbin Na-ion CSVs)."""
    nasa_dir = tmp_path / "5. Battery Data Set"
    na32_dir = tmp_path / "3.2Ah na-ion"
    na10_dir = tmp_path / "10Ah na-ion"
    nasa_dir.mkdir(parents=True)
    na32_dir.mkdir(parents=True)
    na10_dir.mkdir(parents=True)

    zip_path = nasa_dir / "1. BatteryAgingARC-FY08Q4.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for idx, cid in enumerate(["B0005", "B0006", "B0007", "B0018", "B0025", "B0026"], start=1):
            zf.writestr(f"{cid}.mat", _write_nasa_cell_mat(cid, seed=100 + idx))

    for idx, fname in enumerate(
        [
            "Pulse_data_temp25_bat1.CSV",
            "Pulse_data_temp25_bat2.CSV",
            "Driving_cycle_data_temp25_bat5.CSV",
        ],
        start=1,
    ):
        _write_arbin_na_csv(na32_dir / fname, q_nom=3.2, seed=200 + idx)

    for idx, fname in enumerate(
        [
            "Pulse_data_temp25_bat3.CSV",
            "Pulse_data_temp25_bat4.CSV",
            "Driving_cycle_data_temp25_bat6.CSV",
        ],
        start=1,
    ):
        _write_arbin_na_csv(na10_dir / fname, q_nom=10.0, seed=300 + idx)

    return tmp_path
