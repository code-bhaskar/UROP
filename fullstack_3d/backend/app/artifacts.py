"""Read-only artifact loader consuming generated results from ../urop_core/outputs/."""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
UROP_CORE_DIR = REPO_ROOT / "urop_core"
OUTPUTS_DIR = UROP_CORE_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"

ALLOWED_TABLES: dict[str, str] = {
    "within_chemistry": "within_chemistry_results.csv",
    "cross_chemistry": "cross_chemistry_ablation_results.csv",
    "repeated_cell_cv": "repeated_cell_cv_results.csv",
    "transfer_learning": "transfer_learning_results.csv",
    "adaptive_charging": "adaptive_charging_results.csv",
    "domain_shift": "mining_domain_shift_reduction.csv",
    "mutual_information": "mining_mutual_information.csv",
    "operating_regimes": "mining_operating_regimes.csv",
    "bigdata_chemistry": "bigdata_chemistry_summary.csv",
    "bigdata_aging": "bigdata_aging_summary.csv",
    "bigdata_protocol": "bigdata_protocol_temp_summary.csv",
}


def _coerce_value(val: str):
    if val == "" or val is None:
        return None
    low = val.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        if "." not in val and "e" not in low:
            return int(val)
        return float(val)
    except ValueError:
        return val


def load_json_artifact(filename: str) -> dict:
    path = OUTPUTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact {path} not found. Run `python -m src.pipeline` inside urop_core/ first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_table(table_key: str) -> list[dict]:
    if table_key not in ALLOWED_TABLES:
        raise KeyError(f"Unknown table '{table_key}'. Valid keys: {sorted(ALLOWED_TABLES)}")
    path = TABLES_DIR / ALLOWED_TABLES[table_key]
    if not path.exists():
        raise FileNotFoundError(f"Table {path} not found. Run urop_core pipeline first.")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({k: _coerce_value(v) for k, v in row.items()})
    return rows


def load_report_markdown() -> str:
    path = OUTPUTS_DIR / "EXPERIMENT_REPORT.md"
    if not path.exists():
        return "# Report not found\nRun `python -m src.pipeline` inside `urop_core/` first."
    return path.read_text(encoding="utf-8")


def list_figures() -> list[str]:
    if not FIGURES_DIR.exists():
        return []
    return sorted(p.name for p in FIGURES_DIR.glob("*.png"))
