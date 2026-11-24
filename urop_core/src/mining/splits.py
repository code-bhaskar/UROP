"""Strict battery-cell-level, temporally-ordered train / validation / test splitting (zero leakage)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class CellSplit:
    train_df: pl.DataFrame
    val_df: pl.DataFrame
    test_df: pl.DataFrame
    train_cells: list[str]
    val_cells: list[str]
    test_cells: list[str]


def split_by_cell(
    df: pl.DataFrame,
    chemistry: str,
    train_frac: float = 0.60,
    val_frac: float = 0.15,
    seed: int = 42,
) -> CellSplit:
    """Split a single chemistry's DataFrame strictly by cell_id (never splitting a cell across train/test).

    Preserves temporal order within every cell. Guarantees disjoint cell sets.
    """
    sub = df.filter(pl.col("chemistry") == chemistry)
    cells = sorted(sub["cell_id"].unique().to_list())
    if len(cells) < 3:
        raise ValueError(f"Need at least 3 cells for train/val/test split in {chemistry}, got {cells}")

    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(cells))

    n_total = len(perm)
    n_test = max(2 if n_total >= 6 else 1, int(round(n_total * (1.0 - train_frac - val_frac))))
    n_val = max(1, int(round(n_total * val_frac)))
    n_train = max(1, n_total - n_val - n_test)
    while n_train + n_val + n_test > n_total:
        if n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            n_train -= 1

    train_cells = sorted(perm[:n_train])
    val_cells = sorted(perm[n_train : n_train + n_val])
    test_cells = sorted(perm[n_train + n_val :])

    sort_cols = [c for c in ["cell_id", "protocol", "cycle_number", "time_s"] if c in sub.columns]
    train_df = sub.filter(pl.col("cell_id").is_in(train_cells)).sort(sort_cols)
    val_df = sub.filter(pl.col("cell_id").is_in(val_cells)).sort(sort_cols)
    test_df = sub.filter(pl.col("cell_id").is_in(test_cells)).sort(sort_cols)

    return CellSplit(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_cells=train_cells,
        val_cells=val_cells,
        test_cells=test_cells,
    )
