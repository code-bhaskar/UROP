"""Big Data Medallion Lakehouse built on Polars, Apache Parquet, and DuckDB SQL analytics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import polars as pl

from src.config import PipelineConfig
from src.data.lfs_check import DatasetInventory, inspect_datasets
from src.data.physics_simulator import generate_physics_fallback_datasets
from src.data.real_loaders import load_real_na_ion_csvs, load_real_nasa_li_ion


@dataclass
class LakehouseArtifacts:
    inventory: DatasetInventory
    data_mode: str  # "real_lfs_dataset" | "physics_simulated_fallback" | "hybrid"
    bronze_ts_path: Path
    bronze_cycles_path: Path
    silver_ts_path: Path
    silver_cycles_path: Path
    gold_soc_path: Path
    gold_soh_path: Path
    duckdb_analytics: dict


def ingest_to_bronze(cfg: PipelineConfig, force_fallback: bool = False) -> tuple[DatasetInventory, str, Path, Path]:
    """Ingest raw battery datasets into Bronze Apache Parquet files."""
    cfg.ensure_dirs()
    inv = inspect_datasets(cfg)

    bronze_ts = cfg.lake_dir / "bronze" / "timeseries.parquet"
    bronze_cyc = cfg.lake_dir / "bronze" / "cycles.parquet"

    if inv.all_real_available and not force_fallback:
        li_ts, li_cyc = load_real_nasa_li_ion(cfg, inv.li_ion_files)
        na_ts, na_cyc = load_real_na_ion_csvs(cfg, inv.na_ion_3_2ah_files, inv.na_ion_10ah_files)
        ts_df = pl.concat([li_ts, na_ts], how="vertical_relaxed")
        cyc_df = pl.concat([li_cyc, na_cyc], how="vertical_relaxed")
        mode = "real_lfs_dataset"
    else:
        ts_df, cyc_df = generate_physics_fallback_datasets(cfg)
        mode = "physics_simulated_fallback"

    # Write columnar compressed Parquet (ZSTD) for Big Data out-of-core querying
    ts_df.write_parquet(bronze_ts, compression="zstd")
    cyc_df.write_parquet(bronze_cyc, compression="zstd")
    return inv, mode, bronze_ts, bronze_cyc


def run_duckdb_bigdata_analytics(
    silver_ts_path: Path,
    silver_cycles_path: Path,
    tables_dir: Path,
) -> dict:
    """Execute Big Data SQL analytical queries over Silver Parquet tables using DuckDB."""
    con = duckdb.connect(database=":memory:")
    ts_p = str(silver_ts_path)
    cyc_p = str(silver_cycles_path)

    # 1. Chemistry-level time-series statistics
    chem_summary = pl.from_arrow(
        con.sql(
            f"""
        SELECT
            chemistry,
            COUNT(*) AS n_records,
            COUNT(DISTINCT cell_id) AS n_cells,
            ROUND(MIN(voltage_v), 3) AS min_voltage_v,
            ROUND(AVG(voltage_v), 3) AS mean_voltage_v,
            ROUND(MAX(voltage_v), 3) AS max_voltage_v,
            ROUND(AVG(ABS(current_a)), 3) AS mean_abs_current_a,
            ROUND(MIN(temp_c), 2) AS min_temp_c,
            ROUND(AVG(temp_c), 2) AS mean_temp_c,
            ROUND(MAX(temp_c), 2) AS max_temp_c,
            ROUND(AVG(soc), 4) AS mean_soc
        FROM read_parquet('{ts_p}')
        GROUP BY chemistry
        ORDER BY chemistry
        """
        ).to_arrow_table()
    )

    # 2. Protocol & temperature breakdown
    protocol_summary = pl.from_arrow(
        con.sql(
            f"""
        SELECT
            chemistry,
            protocol,
            ROUND(ambient_temp_c, 0) AS ambient_temp_c,
            COUNT(*) AS n_rows,
            ROUND(AVG(voltage_v), 3) AS avg_v,
            ROUND(STDDEV(voltage_v), 3) AS std_v,
            ROUND(AVG(ABS(current_a) / q_nom), 3) AS avg_c_rate
        FROM read_parquet('{ts_p}')
        GROUP BY chemistry, protocol, ROUND(ambient_temp_c, 0)
        ORDER BY chemistry, protocol, ambient_temp_c
        """
        ).to_arrow_table()
    )

    # 3. Cycle aging degradation analytics using SQL window functions
    aging_summary = pl.from_arrow(
        con.sql(
            f"""
        WITH ranked AS (
            SELECT
                chemistry,
                cell_id,
                cycle_number,
                soh,
                r_int_est_ohm,
                temp_rise_c,
                FIRST_VALUE(soh) OVER (PARTITION BY cell_id ORDER BY cycle_number) AS initial_soh,
                LAST_VALUE(soh) OVER (
                    PARTITION BY cell_id ORDER BY cycle_number
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) AS final_soh,
                MAX(cycle_number) OVER (PARTITION BY cell_id) AS max_cycle
            FROM read_parquet('{cyc_p}')
        )
        SELECT
            chemistry,
            COUNT(DISTINCT cell_id) AS n_aged_cells,
            MAX(max_cycle) AS max_cycles,
            ROUND(AVG(initial_soh), 4) AS avg_initial_soh,
            ROUND(AVG(final_soh), 4) AS avg_final_soh,
            ROUND(AVG((initial_soh - final_soh) / NULLIF(max_cycle, 0) * 100.0), 4) AS avg_fade_pct_per_cycle,
            ROUND(AVG(r_int_est_ohm), 4) AS avg_r_int_ohm,
            ROUND(AVG(temp_rise_c), 2) AS avg_temp_rise_c
        FROM ranked
        GROUP BY chemistry
        ORDER BY chemistry
        """
        ).to_arrow_table()
    )

    con.close()

    chem_summary.write_csv(tables_dir / "bigdata_chemistry_summary.csv")
    protocol_summary.write_csv(tables_dir / "bigdata_protocol_temp_summary.csv")
    aging_summary.write_csv(tables_dir / "bigdata_aging_summary.csv")

    return {
        "chemistry_summary": chem_summary.to_dicts(),
        "aging_summary": aging_summary.to_dicts(),
        "protocol_rows": protocol_summary.height,
    }
