"""Tests for real-format NASA (.zip/.mat) & Wang et al. Arbin (.CSV) loaders + zero fake-simulator repo audit."""

from __future__ import annotations

import ast
import io
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import savemat

from src.config import PipelineConfig
from src.data.real_loaders import (
    load_real_mendeley_degradation_xlsx,
    load_real_na_ion_csvs,
    load_real_nasa_li_ion,
)


def _make_nasa_mat_bytes(cell_name: str = "B0005") -> bytes:
    """Create a real MATLAB .mat file matching NASA PCoE cell.cycle struct schema."""
    t_chg = np.linspace(0.0, 3600.0, 60)
    v_chg = np.clip(3.2 + 1.0 * (t_chg / 2400.0), 3.2, 4.2)
    i_chg = np.where(v_chg < 4.18, 1.5, 1.5 * np.exp(-(t_chg - 2400.0) / 800.0))
    temp_chg = 24.0 + 3.0 * (i_chg / 1.5)

    t_dis = np.linspace(0.0, 3200.0, 80)
    v_dis = 4.18 - 1.48 * (t_dis / 3200.0) ** 1.15
    i_dis = np.full_like(t_dis, -2.0)
    temp_dis = 24.0 + 6.0 * (t_dis / 3200.0)

    cycle_list = []
    for cap in [1.85, 1.81, 1.76]:
        cycle_list.append(
            {
                "type": "charge",
                "ambient_temperature": 24.0,
                "data": {
                    "Time": t_chg,
                    "Voltage_measured": v_chg,
                    "Current_measured": i_chg,
                    "Temperature_measured": temp_chg,
                },
            }
        )
        cycle_list.append(
            {
                "type": "discharge",
                "ambient_temperature": 24.0,
                "data": {
                    "Time": t_dis * (cap / 1.85),
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


def test_real_nasa_zip_loader_handles_nested_zip_and_mat(tmp_path: Path) -> None:
    """Verify load_real_nasa_li_ion reads .mat inside nested .zip archives and extracts valid SoC/SoH."""
    mat_bytes = _make_nasa_mat_bytes("B0005")

    inner_buf = io.BytesIO()
    with zipfile.ZipFile(inner_buf, "w") as inner_zf:
        inner_zf.writestr("subdir/B0005.mat", mat_bytes)

    outer_zip = tmp_path / "1. BatteryAgingARC-FY08Q4.zip"
    with zipfile.ZipFile(outer_zip, "w") as outer_zf:
        outer_zf.writestr("nested_archive.zip", inner_buf.getvalue())

    cfg = PipelineConfig(repo_root=tmp_path, fast_mode=True)
    ts_df, cyc_df = load_real_nasa_li_ion(cfg, [outer_zip])

    assert ts_df.height > 100
    assert cyc_df.height == 3
    assert set(ts_df["data_source"].unique().to_list()) == {"real_dataset"}
    assert set(cyc_df["cell_id"].unique().to_list()) == {"B0005"}
    assert float(cyc_df["soh"].max()) <= 1.05
    assert float(cyc_df["soh"].min()) < 0.98
    assert float(ts_df["soc"].min()) >= 0.0
    assert float(ts_df["soc"].max()) <= 1.0


def test_real_na_ion_csv_loader_arbin_schema_and_no_fake_soh(tmp_path: Path) -> None:
    """Verify Arbin CSV parsing (with UTF-8 BOM & rest zeros) emits ZERO fake SoH rows for single-cycle tests."""
    csv_32 = tmp_path / "Pulse_data_temp25_bat1.CSV"
    t = np.arange(0, 600, 5, dtype=float)
    curr = np.where((np.arange(len(t)) % 12) < 4, -3.2, 0.0)
    soc_true = np.clip(1.0 + np.cumsum(curr * 5.0 / (3600.0 * 3.2)), 0.0, 1.0)
    volt = 1.6 + 2.25 * soc_true + curr * 0.03
    temp = 25.0 + 0.4 * np.abs(curr)

    lines = ["\ufeffTest_Time(s),Step Index,Voltage(V),Current(A),Aux_Temperature_1 (C),Cycle_Index"]
    for k in range(len(t)):
        lines.append(f"{t[k]:.1f},1,{volt[k]:.4f},{curr[k]:.4f},{temp[k]:.2f},1")
    csv_32.write_text("\n".join(lines), encoding="utf-8")

    cfg = PipelineConfig(repo_root=tmp_path, fast_mode=True, real_csv_downsample=1)
    ts_df, cyc_df = load_real_na_ion_csvs(cfg, [csv_32], [])

    assert ts_df.height == len(t)
    assert set(ts_df["cell_id"].unique().to_list()) == {"na_3.2Ah_bat1"}
    assert set(ts_df["protocol"].unique().to_list()) == {"pulse_T25"}
    assert set(ts_df["data_source"].unique().to_list()) == {"real_dataset"}
    assert cyc_df.height == 0


def test_real_na_ion_csv_loader_extracts_genuine_multicycle_aging(tmp_path: Path) -> None:
    """Verify load_real_na_ion_csvs extracts real SoH cycles when a CSV genuinely has >= 5 aging cycles."""
    csv_aged = tmp_path / "Driving_cycle_data_temp25_bat5.CSV"
    lines = ["Test_Time(s),Voltage(V),Current(A),Aux_Temperature_1 (C),Cycle_Index,Discharge_Capacity(Ah)"]
    t_now = 0.0
    for cyc in range(1, 7):
        cap = 3.2 * (1.0 - 0.03 * (cyc - 1))
        for step in range(30):
            frac = step / 29.0
            v = 3.85 - 2.2 * frac
            i = -3.2
            lines.append(f"{t_now:.1f},{v:.4f},{i:.4f},27.5,{cyc},{cap * frac:.4f}")
            t_now += 100.0
    csv_aged.write_text("\n".join(lines), encoding="utf-8")

    cfg = PipelineConfig(repo_root=tmp_path, fast_mode=True, real_csv_downsample=1)
    ts_df, cyc_df = load_real_na_ion_csvs(cfg, [csv_aged], [])

    assert ts_df.height == 180
    assert cyc_df.height == 6
    assert float(cyc_df["soh"].max()) > 0.99
    assert float(cyc_df["soh"].min()) < 0.90


def test_real_mendeley_xlsx_loader_extracts_non_leaky_degradation(tmp_path: Path) -> None:
    """Verify load_real_mendeley_degradation_xlsx parses Shen et al. Q(V) Excel sheets cleanly."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws_d = wb.active
    ws_d.title = "NFM | 1C | 25℃ | Discharge"
    ws_c = wb.create_sheet("NFM | 1C | 25℃ | Charge")

    v_grid = np.round(np.linspace(2.0, 4.0, 41), 2)
    ws_d.append([float(v) for v in v_grid] + ["SOH"])
    ws_c.append([float(v) for v in v_grid])

    rng = np.random.default_rng(7)
    for k in range(120):
        soh = 1.0 - 0.12 * (k / 119.0)
        q_max = 4.4 * soh
        shift = (k % 3) * 0.04
        x_norm = np.clip((v_grid - 2.0 - shift) / 2.0, 0.0, 1.0)
        q_d = q_max * (x_norm ** (1.05 + 0.25 * (1.0 - soh)))
        q_c = q_max * np.clip(((v_grid - 2.15 - shift) / 1.85), 0.0, 1.0) ** 1.1
        ws_d.append([round(float(val), 5) for val in q_d] + [round(float(soh), 5)])
        ws_c.append([round(float(val), 5) for val in q_c])

    xlsx_path = tmp_path / "Dataset 1-1.xlsx"
    wb.save(xlsx_path)

    cfg = PipelineConfig(repo_root=tmp_path, fast_mode=True)
    cyc_df = load_real_mendeley_degradation_xlsx(cfg, [xlsx_path])

    assert cyc_df.height >= 100
    assert set(cyc_df["chemistry"].unique().to_list()) == {"na_ion"}
    assert cyc_df["cell_id"].n_unique() == 3
    assert set(cyc_df["data_source"].unique().to_list()) == {"real_dataset"}
    assert float(cyc_df["soh"].max()) <= 1.05
    assert float(cyc_df["soh"].min()) < 0.92
    assert float(cyc_df["r_int_est_ohm"].min()) >= 0.005


def test_zero_fake_simulator_anywhere_in_repo() -> None:
    """Strict repo-wide audit: physics_simulator.py is deleted and ZERO source files import it."""
    repo_root = Path(__file__).resolve().parents[2]
    assert not (repo_root / "urop_core" / "src" / "data" / "physics_simulator.py").exists()

    for scan_dir in [repo_root / "urop_core" / "src", repo_root / "fullstack_3d" / "backend" / "app"]:
        for py_file in scan_dir.rglob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert "physics_simulator" not in node.module, f"Forbidden import in {py_file}: {node.module}"
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert "physics_simulator" not in alias.name, f"Forbidden import in {py_file}: {alias.name}"
