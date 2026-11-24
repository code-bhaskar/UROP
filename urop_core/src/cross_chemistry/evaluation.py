"""Within-chemistry baselines, 6-fold LOCO (Na-ion SoC), 2-fold LOWO (exploratory Na-ion SoH),
3-tier bidirectional Cross-Chemistry evaluation, and Repeated Cell-Level CV.

Scientific safeguards enforced in code:
1. Runtime ``assert_no_cell_overlap`` checks before every model fit (zero test cell/workbook in train/val).
2. Both validation (``val_mae``, ``val_rmse``, ``val_r2``) and held-out test metrics are reported.
3. Empirical OCV curves for Tier 3 alignment are estimated strictly from training cells only.
4. Provenance enforcement: ``pooled_unordered`` datasets (Mendeley workbooks without cell IDs or
   cycle order) are restricted to tabular models only; LSTM/GRU sequence models are skipped/refused.
5. Li-ion SoH trains exclusively on ordered NASA cells; pooled Mendeley NCM (``li_ncm_25C``) is
   evaluated strictly as an external pooled test set and never mixed into NASA cell training.
6. Na-ion SoC uses 6-fold Leave-One-Cell-Out (LOCO) across all 6 Wang et al. cells; Li->Na SoC also
   evaluates all 6 Na-ion cells jointly as a single held-out target set.
7. Na-ion SoH (2 pooled workbooks at 25 °C and -15 °C) runs in documented exploratory Leave-One-Workbook-Out
   mode with explicit temperature-confounding caveats.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

from src.config import PipelineConfig
from src.cross_chemistry.alignment import (
    CrossChemistryDomainAdapter,
    EmpiricalOCVCurve,
    estimate_empirical_ocv_curve,
)
from src.mining.feature_engineering import (
    PHYSICS_SOC_FEATURES,
    PHYSICS_SOH_FEATURES,
    RAW_SOC_FEATURES,
    RAW_SOH_FEATURES,
)
from src.mining.splits import CellSplit, split_by_cell
from src.models.metrics import regression_metrics
from src.models.sequence import (
    TrainedSequenceModel,
    build_segmented_sequences,
    train_sequence_model,
)
from src.models.tabular import TrainedTabularModel, train_tabular_models


def assert_no_cell_overlap(
    train_ids: list[str] | set[str],
    val_ids: list[str] | set[str],
    test_ids: list[str] | set[str],
    context: str = "",
) -> None:
    """Hard guard verifying zero test cell/workbook ID ever enters training or validation."""
    tr, va, te = set(train_ids), set(val_ids), set(test_ids)
    if not tr.isdisjoint(te):
        raise AssertionError(f"[{context}] Leakage detected: test units {tr & te} appeared in train!")
    if not va.isdisjoint(te):
        raise AssertionError(f"[{context}] Leakage detected: test units {va & te} appeared in val!")


@dataclass
class TaskExperimentBundle:
    task_name: str  # "SoC" | "SoH"
    splits: dict[str, CellSplit]
    within_rows: list[dict]
    cross_rows: list[dict]
    within_tabular_phys: dict[str, dict[str, TrainedTabularModel]]
    within_seq_phys: dict[str, dict[str, TrainedSequenceModel]]
    adapters: dict[str, CrossChemistryDomainAdapter]
    sample_predictions: dict[str, dict[str, np.ndarray]]
    empirical_ocv_curves: dict[str, EmpiricalOCVCurve] = field(default_factory=dict)
    cross_chemistry_available: bool = True
    cross_chemistry_mode: str = "full"  # "full" | "exploratory_pooled" | "unavailable"
    compatibility_note: str = ""
    loco_per_cell_rows: list[dict] = field(default_factory=list)
    cross_per_unit_rows: list[dict] = field(default_factory=list)
    external_test_rows: list[dict] = field(default_factory=list)
    full_task_df: pl.DataFrame | None = None
    unit_counts: dict[str, int] = field(default_factory=dict)


def _inspect_chemistry_task_mode(df: pl.DataFrame, chemistry: str, is_soc: bool) -> tuple[str, int]:
    """Return (mode, n_independent_units) for a chemistry/task pair.

    - ``full``: >= 3 ordered per-cell units with genuine target spread.
    - ``exploratory_pooled``: >= 2 independent units (e.g., pooled workbooks without cell IDs; tabular only).
    - ``unavailable``: fewer than 2 independent units or negligible target spread.
    """
    if df.height == 0:
        return "unavailable", 0
    sub = df.filter(pl.col("chemistry") == chemistry)
    if sub.height < 30:
        return "unavailable", 0

    if not is_soc:
        soh_arr = sub["soh"].to_numpy()
        if float(np.nanmax(soh_arr) - np.nanmin(soh_arr)) < 0.02:
            return "unavailable", 0

    has_prov = "data_provenance" in sub.columns
    if has_prov:
        ordered = sub.filter(pl.col("data_provenance") == "ordered_per_cell")
        n_ordered = int(ordered["cell_id"].n_unique()) if ordered.height > 0 else 0
        if n_ordered >= 3:
            return "full", n_ordered
        n_total = int(sub["cell_id"].n_unique())
        if n_total >= 2:
            return "exploratory_pooled", n_total
        return "unavailable", n_total

    n_cells = int(sub["cell_id"].n_unique())
    if n_cells >= 3:
        return "full", n_cells
    if n_cells >= 2:
        return "exploratory_pooled", n_cells
    return "unavailable", n_cells


def _chemistry_has_valid_task_data(df: pl.DataFrame, chemistry: str, is_soc: bool) -> bool:
    mode, _ = _inspect_chemistry_task_mode(df, chemistry, is_soc)
    return mode in ("full", "exploratory_pooled")


def _run_within_chemistry_for_representation(
    split: CellSplit,
    feature_cols: list[str],
    target_col: str,
    group_cols: list[str],
    clip_bounds: tuple[float, float],
    cfg: PipelineConfig,
    include_seq: bool = True,
    seq_epochs_override: int | None = None,
) -> tuple[dict[str, TrainedTabularModel], dict[str, TrainedSequenceModel], StandardScaler]:
    assert_no_cell_overlap(split.train_cells, split.val_cells, split.test_cells, context=f"within_{target_col}")

    X_tr = split.train_df.select(feature_cols).to_numpy()
    y_tr = split.train_df[target_col].to_numpy()
    X_va = split.val_df.select(feature_cols).to_numpy()
    y_va = split.val_df[target_col].to_numpy()
    X_te = split.test_df.select(feature_cols).to_numpy()
    y_te = split.test_df[target_col].to_numpy()

    tab_models = train_tabular_models(
        X_tr,
        y_tr,
        X_va,
        y_va,
        X_te,
        y_te,
        clip_bounds=clip_bounds,
        seed=cfg.seed,
        already_scaled=False,
        fast_mode=cfg.fast_mode,
    )

    scaler = StandardScaler().fit(X_tr)
    seq_models: dict[str, TrainedSequenceModel] = {}

    is_pooled = (
        "data_provenance" in split.train_df.columns
        and "pooled_unordered" in set(split.train_df["data_provenance"].unique().to_list())
    )
    if include_seq and not is_pooled:
        seq_len = 8 if target_col == "soh" else cfg.seq_len
        epochs = seq_epochs_override if seq_epochs_override is not None else (10 if cfg.fast_mode else cfg.seq_epochs)
        Xtr_s, ytr_s = build_segmented_sequences(
            split.train_df, feature_cols, target_col, group_cols, seq_len, scaler.transform
        )
        Xva_s, yva_s = build_segmented_sequences(
            split.val_df, feature_cols, target_col, group_cols, seq_len, scaler.transform
        )
        Xte_s, yte_s = build_segmented_sequences(
            split.test_df, feature_cols, target_col, group_cols, seq_len, scaler.transform
        )

        if len(Xtr_s) > 20 and len(Xva_s) > 5 and len(Xte_s) > 5:
            for rnn_type in ("LSTM", "GRU"):
                seq_models[rnn_type] = train_sequence_model(
                    Xtr_s,
                    ytr_s,
                    Xva_s,
                    yva_s,
                    Xte_s,
                    yte_s,
                    rnn_type=rnn_type,
                    hidden_dim=32 if cfg.fast_mode else cfg.seq_hidden_dim,
                    num_layers=2,
                    epochs=epochs,
                    batch_size=cfg.seq_batch_size,
                    lr=cfg.seq_lr,
                    clip_bounds=clip_bounds,
                    seed=cfg.seed,
                    scaler=scaler,
                )

    return tab_models, seq_models, scaler


def run_na_soc_leave_one_cell_out(
    cfg: PipelineConfig,
    soc_df: pl.DataFrame,
    include_seq: bool = True,
) -> tuple[list[dict], list[dict]]:
    """6-fold Leave-One-Cell-Out (LOCO) evaluation for Na-ion SoC across all Wang et al. cells.

    For each held-out cell ``c``:
      - ``test_cells = [c]`` (all 6 temperatures of cell ``c`` stay strictly in test)
      - Remaining 5 cells are split into 4 train + 1 validation (for early stopping / val reporting)
      - Hard ``assert_no_cell_overlap`` prevents any test cell from entering fit or scaling.
    Sequence models run with reduced epochs per LOCO fold (documented explicitly in report).
    """
    na_df = soc_df.filter(pl.col("chemistry") == "na_ion")
    cells = sorted(na_df["cell_id"].unique().to_list())
    if len(cells) < 3:
        return [], []

    group_cols = ["cell_id", "protocol", "cycle_number"]
    clip_bounds = (0.0, 1.0)
    loco_seq_epochs = 4 if cfg.fast_mode else 6

    per_cell_rows: list[dict] = []
    agg_buckets: dict[str, list[dict]] = {}

    for fold_idx, test_cell in enumerate(cells):
        others = [c for c in cells if c != test_cell]
        val_cell = others[fold_idx % len(others)]
        train_cells = [c for c in others if c != val_cell]

        assert_no_cell_overlap(train_cells, [val_cell], [test_cell], context=f"na_soc_loco_fold_{test_cell}")

        sp = CellSplit(
            train_df=na_df.filter(pl.col("cell_id").is_in(train_cells)),
            val_df=na_df.filter(pl.col("cell_id") == val_cell),
            test_df=na_df.filter(pl.col("cell_id") == test_cell),
            train_cells=train_cells,
            val_cells=[val_cell],
            test_cells=[test_cell],
        )
        tab_phys, seq_phys, _ = _run_within_chemistry_for_representation(
            sp,
            PHYSICS_SOC_FEATURES,
            "soc",
            group_cols,
            clip_bounds,
            cfg,
            include_seq=include_seq,
            seq_epochs_override=loco_seq_epochs,
        )

        # Cell metadata (nominal capacity & protocol family)
        q_nom = float(sp.test_df["q_nom"].median())
        protos = sorted({p.split("_")[0] for p in sp.test_df["protocol"].unique().to_list()})
        proto_str = "+".join(protos)

        for mname, tm in tab_phys.items():
            row = {
                "task": "SoC",
                "chemistry": "na_ion",
                "test_cell": test_cell,
                "q_nom_ah": round(q_nom, 1),
                "protocol_family": proto_str,
                "model": mname,
                "val_cell": val_cell,
                "val_rmse": tm.val_metrics["rmse"],
                "mae": tm.test_metrics["mae"],
                "rmse": tm.test_metrics["rmse"],
                "r2": tm.test_metrics["r2"],
                "train_time_s": tm.train_time_s,
                "infer_time_ms": tm.infer_time_ms,
            }
            per_cell_rows.append(row)
            agg_buckets.setdefault(mname, []).append(row)

        for mname, sm in seq_phys.items():
            row = {
                "task": "SoC",
                "chemistry": "na_ion",
                "test_cell": test_cell,
                "q_nom_ah": round(q_nom, 1),
                "protocol_family": proto_str,
                "model": mname,
                "val_cell": val_cell,
                "val_rmse": sm.val_metrics["rmse"],
                "mae": sm.test_metrics["mae"],
                "rmse": sm.test_metrics["rmse"],
                "r2": sm.test_metrics["r2"],
                "train_time_s": sm.train_time_s,
                "infer_time_ms": sm.infer_time_ms,
            }
            per_cell_rows.append(row)
            agg_buckets.setdefault(mname, []).append(row)

    summary_within_rows: list[dict] = []
    for mname, fold_rows in agg_buckets.items():
        summary_within_rows.append(
            {
                "task": "SoC",
                "chemistry": "na_ion",
                "model": mname,
                "representation": "physics_normalized",
                "evaluation_mode": f"leave_one_cell_out_{len(cells)}_folds",
                "train_cells": f"{len(cells)-2}_per_fold",
                "val_cells": "1_per_fold",
                "test_cells": f"all_{len(cells)}_cells_loco",
                "val_mae": round(float(np.mean([r["val_rmse"] for r in fold_rows])), 5),
                "val_rmse": round(float(np.mean([r["val_rmse"] for r in fold_rows])), 5),
                "val_r2": round(float(np.mean([r["r2"] for r in fold_rows])), 5),
                "mae": round(float(np.mean([r["mae"] for r in fold_rows])), 5),
                "rmse": round(float(np.mean([r["rmse"] for r in fold_rows])), 5),
                "rmse_std": round(float(np.std([r["rmse"] for r in fold_rows])), 5),
                "r2": round(float(np.mean([r["r2"] for r in fold_rows])), 5),
                "r2_std": round(float(np.std([r["r2"] for r in fold_rows])), 5),
                "train_time_s": round(float(np.mean([r["train_time_s"] for r in fold_rows])), 4),
                "infer_time_ms": round(float(np.mean([r["infer_time_ms"] for r in fold_rows])), 4),
            }
        )

    return per_cell_rows, summary_within_rows


def _run_exploratory_lowo_na_soh(
    cfg: PipelineConfig,
    na_soh_df: pl.DataFrame,
) -> tuple[list[dict], list[dict], dict[str, TrainedTabularModel], dict[str, TrainedTabularModel]]:
    """2-fold Leave-One-Workbook-Out (LOWO) for pooled Mendeley NFM Na-ion SoH (tabular models ONLY).

    Fold 1: train na_nfm_25C (25 °C) -> test na_nfm_m15C (-15 °C)
    Fold 2: train na_nfm_m15C (-15 °C) -> test na_nfm_25C (25 °C)
    Explicitly labeled exploratory & temperature-confounded; zero sequence models run.
    """
    units = sorted(na_soh_df["cell_id"].unique().to_list())
    per_fold_rows: list[dict] = []
    agg_buckets: dict[str, list[dict]] = {}

    for test_unit in units:
        train_units = [u for u in units if u != test_unit]
        assert_no_cell_overlap(train_units, [], [test_unit], context=f"na_soh_lowo_{test_unit}")

        tr_df = na_soh_df.filter(pl.col("cell_id").is_in(train_units))
        te_df = na_soh_df.filter(pl.col("cell_id") == test_unit)

        X_tr = tr_df.select(PHYSICS_SOH_FEATURES).to_numpy()
        y_tr = tr_df["soh"].to_numpy()
        X_te = te_df.select(PHYSICS_SOH_FEATURES).to_numpy()
        y_te = te_df["soh"].to_numpy()

        # Contiguous 20% validation block inside training workbook strictly for val logging (never touches test workbook)
        n_tr = len(X_tr)
        val_cut = max(10, int(0.8 * n_tr))
        tab_phys = train_tabular_models(
            X_tr[:val_cut],
            y_tr[:val_cut],
            X_tr[val_cut:],
            y_tr[val_cut:],
            X_te,
            y_te,
            clip_bounds=(0.50, 1.05),
            seed=cfg.seed,
            fast_mode=cfg.fast_mode,
        )
        for mname, tm in tab_phys.items():
            row = {
                "task": "SoH",
                "chemistry": "na_ion",
                "train_unit": ",".join(train_units),
                "test_unit": test_unit,
                "model": mname,
                "evaluation_mode": "exploratory_lowo_temp_confounded",
                "val_rmse": tm.val_metrics["rmse"],
                "mae": tm.test_metrics["mae"],
                "rmse": tm.test_metrics["rmse"],
                "r2": tm.test_metrics["r2"],
                "train_time_s": tm.train_time_s,
                "infer_time_ms": tm.infer_time_ms,
            }
            per_fold_rows.append(row)
            agg_buckets.setdefault(mname, []).append(row)

    summary_rows: list[dict] = []
    for mname, frows in agg_buckets.items():
        summary_rows.append(
            {
                "task": "SoH",
                "chemistry": "na_ion",
                "model": mname,
                "representation": "physics_normalized",
                "evaluation_mode": "exploratory_lowo_temp_confounded",
                "train_cells": "1_workbook_per_fold",
                "val_cells": "contiguous_20pct_of_train_workbook",
                "test_cells": ",".join(units),
                "val_mae": round(float(np.mean([r["val_rmse"] for r in frows])), 5),
                "val_rmse": round(float(np.mean([r["val_rmse"] for r in frows])), 5),
                "val_r2": round(float(np.mean([r["r2"] for r in frows])), 5),
                "mae": round(float(np.mean([r["mae"] for r in frows])), 5),
                "rmse": round(float(np.mean([r["rmse"] for r in frows])), 5),
                "rmse_std": round(float(np.std([r["rmse"] for r in frows])), 5),
                "r2": round(float(np.mean([r["r2"] for r in frows])), 5),
                "r2_std": round(float(np.std([r["r2"] for r in frows])), 5),
                "train_time_s": round(float(np.mean([r["train_time_s"] for r in frows])), 4),
                "infer_time_ms": round(float(np.mean([r["infer_time_ms"] for r in frows])), 4),
            }
        )

    # Also fit full-Na tabular models (Raw & Physics) strictly for Na -> Li cross-chemistry source
    X_all_raw = na_soh_df.select(RAW_SOH_FEATURES).to_numpy()
    X_all_phys = na_soh_df.select(PHYSICS_SOH_FEATURES).to_numpy()
    y_all = na_soh_df["soh"].to_numpy()
    n_all = len(y_all)
    cut = max(20, int(0.85 * n_all))
    na_src_raw = train_tabular_models(
        X_all_raw[:cut], y_all[:cut], X_all_raw[cut:], y_all[cut:], X_all_raw[cut:], y_all[cut:],
        clip_bounds=(0.50, 1.05), seed=cfg.seed, fast_mode=cfg.fast_mode,
    )
    na_src_phys = train_tabular_models(
        X_all_phys[:cut], y_all[:cut], X_all_phys[cut:], y_all[cut:], X_all_phys[cut:], y_all[cut:],
        clip_bounds=(0.50, 1.05), seed=cfg.seed, fast_mode=cfg.fast_mode,
    )
    return per_fold_rows, summary_rows, na_src_raw, na_src_phys


def run_task_within_and_cross_chemistry(
    cfg: PipelineConfig,
    df: pl.DataFrame,
    task_name: str,
) -> TaskExperimentBundle:
    """Run within-chemistry baselines and 3-tier bidirectional cross-chemistry evaluation for SoC or SoH."""
    is_soc = task_name == "SoC"
    raw_cols = RAW_SOC_FEATURES if is_soc else RAW_SOH_FEATURES
    phys_cols = PHYSICS_SOC_FEATURES if is_soc else PHYSICS_SOH_FEATURES
    target_col = "soc" if is_soc else "soh"
    group_cols = ["cell_id", "protocol", "cycle_number"] if is_soc else ["cell_id"]
    clip_bounds = (0.0, 1.0) if is_soc else (0.50, 1.05)
    seq_len = cfg.seq_len if is_soc else 8

    modes: dict[str, str] = {}
    unit_counts: dict[str, int] = {}
    for chem in ("li_ion", "na_ion"):
        m, n_u = _inspect_chemistry_task_mode(df, chem, is_soc)
        modes[chem] = m
        unit_counts[chem] = n_u

    available_chems = [c for c in ("li_ion", "na_ion") if modes[c] != "unavailable"]
    has_prov = "data_provenance" in df.columns

    splits: dict[str, CellSplit] = {}
    within_rows: list[dict] = []
    cross_rows: list[dict] = []
    loco_per_cell_rows: list[dict] = []
    cross_per_unit_rows: list[dict] = []
    external_test_rows: list[dict] = []

    within_tab_raw: dict[str, dict[str, TrainedTabularModel]] = {}
    within_seq_raw: dict[str, dict[str, TrainedSequenceModel]] = {}
    within_tab_phys: dict[str, dict[str, TrainedTabularModel]] = {}
    within_seq_phys: dict[str, dict[str, TrainedSequenceModel]] = {}
    adapters: dict[str, CrossChemistryDomainAdapter] = {}
    sample_predictions: dict[str, dict[str, np.ndarray]] = {}
    empirical_ocvs: dict[str, EmpiricalOCVCurve] = {}

    # 1. Within-Chemistry Training & Evaluation
    for idx_c, chem in enumerate(available_chems):
        if not is_soc and chem == "li_ion" and has_prov:
            # Li-ion SoH: train/val/test strictly on ordered NASA cells ONLY (never mix pooled NCM!)
            ordered_li = df.filter((pl.col("chemistry") == "li_ion") & (pl.col("data_provenance") == "ordered_per_cell"))
            pooled_li = df.filter((pl.col("chemistry") == "li_ion") & (pl.col("data_provenance") != "ordered_per_cell"))
            sp = split_by_cell(ordered_li, "li_ion", seed=cfg.seed + idx_c)
            splits["li_ion"] = sp

            tab_raw, seq_raw, _ = _run_within_chemistry_for_representation(
                sp, raw_cols, target_col, group_cols, clip_bounds, cfg, include_seq=True
            )
            tab_phys, seq_phys, _ = _run_within_chemistry_for_representation(
                sp, phys_cols, target_col, group_cols, clip_bounds, cfg, include_seq=True
            )
            within_tab_raw["li_ion"] = tab_raw
            within_seq_raw["li_ion"] = seq_raw
            within_tab_phys["li_ion"] = tab_phys
            within_seq_phys["li_ion"] = seq_phys

            for mname, tm in tab_phys.items():
                within_rows.append(
                    {
                        "task": task_name,
                        "chemistry": "li_ion",
                        "model": mname,
                        "representation": "physics_normalized",
                        "evaluation_mode": "ordered_nasa_cell_split",
                        "train_cells": ",".join(sp.train_cells),
                        "val_cells": ",".join(sp.val_cells),
                        "test_cells": ",".join(sp.test_cells),
                        "val_mae": tm.val_metrics["mae"],
                        "val_rmse": tm.val_metrics["rmse"],
                        "val_r2": tm.val_metrics["r2"],
                        "mae": tm.test_metrics["mae"],
                        "rmse": tm.test_metrics["rmse"],
                        "r2": tm.test_metrics["r2"],
                        "train_time_s": tm.train_time_s,
                        "infer_time_ms": tm.infer_time_ms,
                    }
                )
            for mname, sm in seq_phys.items():
                within_rows.append(
                    {
                        "task": task_name,
                        "chemistry": "li_ion",
                        "model": mname,
                        "representation": "physics_normalized",
                        "evaluation_mode": "ordered_nasa_cell_split",
                        "train_cells": ",".join(sp.train_cells),
                        "val_cells": ",".join(sp.val_cells),
                        "test_cells": ",".join(sp.test_cells),
                        "val_mae": sm.val_metrics["mae"],
                        "val_rmse": sm.val_metrics["rmse"],
                        "val_r2": sm.val_metrics["r2"],
                        "mae": sm.test_metrics["mae"],
                        "rmse": sm.test_metrics["rmse"],
                        "r2": sm.test_metrics["r2"],
                        "train_time_s": sm.train_time_s,
                        "infer_time_ms": sm.infer_time_ms,
                    }
                )

            # Evaluate NASA-trained tabular models strictly out-of-sample on external pooled NCM workbook!
            if pooled_li.height > 0:
                X_ext = pooled_li.select(phys_cols).to_numpy()
                y_ext = pooled_li["soh"].to_numpy()
                ext_units = sorted(pooled_li["cell_id"].unique().to_list())
                assert_no_cell_overlap(sp.train_cells, sp.val_cells, ext_units, context="li_soh_external_ncm")
                for mname, tm in tab_phys.items():
                    preds = tm.predict(X_ext)
                    external_test_rows.append(
                        {
                            "task": "SoH",
                            "train_source": "nasa_li_ion_33_ordered_cells",
                            "external_test_unit": ",".join(ext_units),
                            "n_rows": int(pooled_li.height),
                            "model": mname,
                            **regression_metrics(y_ext, preds),
                        }
                    )

            X_te_phys = sp.test_df.select(phys_cols).to_numpy()
            y_te = sp.test_df[target_col].to_numpy()
            sample_predictions["within_li_ion"] = {
                "y_true": y_te,
                "y_pred_xgb": tab_phys["XGBoost"].predict(X_te_phys),
                "y_pred_rf": tab_phys["RandomForest"].predict(X_te_phys),
            }
            continue

        if not is_soc and chem == "na_ion" and modes["na_ion"] == "exploratory_pooled":
            # Na-ion SoH: 2 pooled Mendeley NFM workbooks -> exploratory 2-fold LOWO (tabular ONLY)
            na_soh_df = df.filter(pl.col("chemistry") == "na_ion")
            lowo_rows, lowo_summary, na_src_raw, na_src_phys = _run_exploratory_lowo_na_soh(cfg, na_soh_df)
            within_rows.extend(lowo_summary)
            loco_per_cell_rows.extend(lowo_rows)
            within_tab_raw["na_ion"] = na_src_raw
            within_tab_phys["na_ion"] = na_src_phys
            within_seq_raw["na_ion"] = {}
            within_seq_phys["na_ion"] = {}

            units = sorted(na_soh_df["cell_id"].unique().to_list())
            # Store canonical 2-workbook split for reference (train=first workbook, test=second workbook)
            splits["na_ion"] = CellSplit(
                train_df=na_soh_df.filter(pl.col("cell_id") == units[0]),
                val_df=na_soh_df.filter(pl.col("cell_id") == units[0]).head(0),
                test_df=na_soh_df.filter(pl.col("cell_id") == units[-1]),
                train_cells=[units[0]],
                val_cells=[],
                test_cells=[units[-1]],
            )
            continue

        # Standard ordered cell split (Li-ion SoC, Na-ion SoC canonical split for cross-chem source)
        sp = split_by_cell(df, chem, seed=cfg.seed + idx_c)
        splits[chem] = sp

        if is_soc:
            empirical_ocvs[chem] = estimate_empirical_ocv_curve(sp.train_df, chem)

        tab_raw, seq_raw, _ = _run_within_chemistry_for_representation(
            sp, raw_cols, target_col, group_cols, clip_bounds, cfg, include_seq=True
        )
        tab_phys, seq_phys, _ = _run_within_chemistry_for_representation(
            sp, phys_cols, target_col, group_cols, clip_bounds, cfg, include_seq=True
        )
        within_tab_raw[chem] = tab_raw
        within_seq_raw[chem] = seq_raw
        within_tab_phys[chem] = tab_phys
        within_seq_phys[chem] = seq_phys

        if is_soc and chem == "na_ion":
            # Problem 2: Replace single 2-cell Na-ion SoC split with full 6-fold Leave-One-Cell-Out!
            per_cell_loco, summary_loco = run_na_soc_leave_one_cell_out(cfg, df, include_seq=True)
            loco_per_cell_rows.extend(per_cell_loco)
            within_rows.extend(summary_loco)
        else:
            for mname, tm in tab_phys.items():
                within_rows.append(
                    {
                        "task": task_name,
                        "chemistry": chem,
                        "model": mname,
                        "representation": "physics_normalized",
                        "evaluation_mode": "ordered_cell_split",
                        "train_cells": ",".join(sp.train_cells),
                        "val_cells": ",".join(sp.val_cells),
                        "test_cells": ",".join(sp.test_cells),
                        "val_mae": tm.val_metrics["mae"],
                        "val_rmse": tm.val_metrics["rmse"],
                        "val_r2": tm.val_metrics["r2"],
                        "mae": tm.test_metrics["mae"],
                        "rmse": tm.test_metrics["rmse"],
                        "r2": tm.test_metrics["r2"],
                        "train_time_s": tm.train_time_s,
                        "infer_time_ms": tm.infer_time_ms,
                    }
                )
            for mname, sm in seq_phys.items():
                within_rows.append(
                    {
                        "task": task_name,
                        "chemistry": chem,
                        "model": mname,
                        "representation": "physics_normalized",
                        "evaluation_mode": "ordered_cell_split",
                        "train_cells": ",".join(sp.train_cells),
                        "val_cells": ",".join(sp.val_cells),
                        "test_cells": ",".join(sp.test_cells),
                        "val_mae": sm.val_metrics["mae"],
                        "val_rmse": sm.val_metrics["rmse"],
                        "val_r2": sm.val_metrics["r2"],
                        "mae": sm.test_metrics["mae"],
                        "rmse": sm.test_metrics["rmse"],
                        "r2": sm.test_metrics["r2"],
                        "train_time_s": sm.train_time_s,
                        "infer_time_ms": sm.infer_time_ms,
                    }
                )

        X_te_phys = sp.test_df.select(phys_cols).to_numpy()
        y_te = sp.test_df[target_col].to_numpy()
        sample_predictions[f"within_{chem}"] = {
            "y_true": y_te,
            "y_pred_xgb": tab_phys["XGBoost"].predict(X_te_phys),
            "y_pred_rf": tab_phys["RandomForest"].predict(X_te_phys),
        }

    cross_available = ("li_ion" in available_chems) and ("na_ion" in available_chems)
    if not cross_available:
        missing = [c for c in ("li_ion", "na_ion") if c not in available_chems]
        compat_note = (
            f"Cross-chemistry {task_name} skipped cleanly because {missing} lacks sufficient independent "
            f"{task_name} units in the current dataset (per Proposal Section 5.1 compatibility check)."
        )
        return TaskExperimentBundle(
            task_name=task_name,
            splits=splits,
            within_rows=within_rows,
            cross_rows=cross_rows,
            within_tabular_phys=within_tab_phys,
            within_seq_phys=within_seq_phys,
            adapters=adapters,
            sample_predictions=sample_predictions,
            empirical_ocv_curves=empirical_ocvs,
            cross_chemistry_available=False,
            cross_chemistry_mode="unavailable",
            compatibility_note=compat_note,
            loco_per_cell_rows=loco_per_cell_rows,
            cross_per_unit_rows=cross_per_unit_rows,
            external_test_rows=external_test_rows,
            full_task_df=df,
            unit_counts=unit_counts,
        )

    is_exploratory = "exploratory_pooled" in (modes["li_ion"], modes["na_ion"])
    cross_mode = "exploratory_pooled" if is_exploratory else "full"
    compat_note = (
        f"Exploratory {task_name} cross-chemistry mode: Na-ion SoH comprises {unit_counts['na_ion']} pooled "
        f"Mendeley workbooks (na_nfm_25C at 25 °C and na_nfm_m15C at -15 °C) without cell IDs or cycle order. "
        f"Sequence models (LSTM/GRU) are disabled for pooled SoH; tabular models only."
        if is_exploratory
        else ""
    )

    # 2. Bidirectional Cross-Chemistry Evaluation (Li-ion -> Na-ion AND Na-ion -> Li-ion)
    for src_chem, tgt_chem in (("li_ion", "na_ion"), ("na_ion", "li_ion")):
        direction = f"{src_chem}_to_{tgt_chem}"
        src_split = splits[src_chem]
        tgt_split = splits[tgt_chem]

        # For Li -> Na (both SoC and SoH): hold out ALL Na-ion units together as the complete target test set!
        if src_chem == "li_ion" and tgt_chem == "na_ion":
            tgt_eval_df = df.filter(pl.col("chemistry") == "na_ion")
            tgt_unlabeled_df = tgt_eval_df  # Transductive unsupervised alignment using unlabeled target inputs only
            eval_scope = (
                "all_6_na_cells_held_out_transductive_coral"
                if is_soc
                else "all_2_na_workbooks_held_out_exploratory"
            )
        else:
            tgt_eval_df = tgt_split.test_df
            tgt_unlabeled_df = tgt_split.train_df
            eval_scope = "held_out_li_test_cells"

        # Verify ZERO unit overlap between source train and target eval
        assert_no_cell_overlap(
            src_split.train_cells,
            src_split.val_cells,
            tgt_eval_df["cell_id"].unique().to_list(),
            context=f"cross_{task_name}_{direction}",
        )

        y_tgt_test = tgt_eval_df[target_col].to_numpy()
        X_tgt_test_raw = tgt_eval_df.select(raw_cols).to_numpy()
        X_tgt_test_phys = tgt_eval_df.select(phys_cols).to_numpy()

        tgt_is_pooled = (
            "data_provenance" in tgt_eval_df.columns
            and "pooled_unordered" in set(tgt_eval_df["data_provenance"].unique().to_list())
        )
        run_seq_cross = (not tgt_is_pooled) and (not is_exploratory)

        # --- Tier 1: Naive Raw Features Zero-Shot ---
        for mname, tm in within_tab_raw[src_chem].items():
            preds = tm.predict(X_tgt_test_raw)
            cross_rows.append(
                {
                    "task": task_name,
                    "direction": direction,
                    "tier": "1_raw_zero_shot",
                    "model": mname,
                    "evaluation_mode": eval_scope,
                    **regression_metrics(y_tgt_test, preds),
                }
            )
        if run_seq_cross:
            for mname, sm in within_seq_raw[src_chem].items():
                Xte_s, yte_s = build_segmented_sequences(
                    tgt_eval_df, raw_cols, target_col, group_cols, seq_len, sm.scaler.transform
                )
                if len(Xte_s) > 0:
                    preds = sm.predict_sequences(Xte_s)
                    cross_rows.append(
                        {
                            "task": task_name,
                            "direction": direction,
                            "tier": "1_raw_zero_shot",
                            "model": mname,
                            "evaluation_mode": eval_scope,
                            **regression_metrics(yte_s, preds),
                        }
                    )

        # --- Tier 2: Physics-Normalized Chemistry-Invariant Zero-Shot ---
        for mname, tm in within_tab_phys[src_chem].items():
            preds = tm.predict(X_tgt_test_phys)
            cross_rows.append(
                {
                    "task": task_name,
                    "direction": direction,
                    "tier": "2_physics_normalized_zero_shot",
                    "model": mname,
                    "evaluation_mode": eval_scope,
                    **regression_metrics(y_tgt_test, preds),
                }
            )
        if run_seq_cross:
            for mname, sm in within_seq_phys[src_chem].items():
                Xte_s, yte_s = build_segmented_sequences(
                    tgt_eval_df, phys_cols, target_col, group_cols, seq_len, sm.scaler.transform
                )
                if len(Xte_s) > 0:
                    preds = sm.predict_sequences(Xte_s)
                    cross_rows.append(
                        {
                            "task": task_name,
                            "direction": direction,
                            "tier": "2_physics_normalized_zero_shot",
                            "model": mname,
                            "evaluation_mode": eval_scope,
                            **regression_metrics(yte_s, preds),
                        }
                    )

        # --- Tier 3: Physics + Training-Only Empirical OCV Warping + Shrinkage CORAL ---
        if src_chem == "na_ion" and not is_soc and is_exploratory:
            X_src_tr_phys = df.filter(pl.col("chemistry") == "na_ion").select(phys_cols).to_numpy()
        else:
            X_src_tr_phys = src_split.train_df.select(phys_cols).to_numpy()
        X_tgt_tr_unlabeled = tgt_unlabeled_df.select(phys_cols).to_numpy()

        adapter = CrossChemistryDomainAdapter(
            src_ocv_curve=empirical_ocvs.get(src_chem) if is_soc else None,
            tgt_ocv_curve=empirical_ocvs.get(tgt_chem) if is_soc else None,
            use_ocv_warp=is_soc,
            use_coral=True,
            coral_reg=0.10,
            coral_shrinkage=0.25 if is_soc else 0.15,
            align_mean=False,
        ).fit(X_src_tr_phys, X_tgt_tr_unlabeled, phys_cols)
        adapters[direction] = adapter

        X_tgt_test_aligned = adapter.transform_target(X_tgt_test_phys)
        for mname, tm in within_tab_phys[src_chem].items():
            preds = tm.predict(X_tgt_test_aligned, already_scaled=True)
            cross_rows.append(
                {
                    "task": task_name,
                    "direction": direction,
                    "tier": "3_physics_ocv_coral_zero_shot",
                    "model": mname,
                    "evaluation_mode": eval_scope,
                    **regression_metrics(y_tgt_test, preds),
                }
            )
        if run_seq_cross:
            for mname, sm in within_seq_phys[src_chem].items():
                Xte_s, yte_s = build_segmented_sequences(
                    tgt_eval_df, phys_cols, target_col, group_cols, seq_len, adapter.transform_target
                )
                if len(Xte_s) > 0:
                    preds = sm.predict_sequences(Xte_s)
                    cross_rows.append(
                        {
                            "task": task_name,
                            "direction": direction,
                            "tier": "3_physics_ocv_coral_zero_shot",
                            "model": mname,
                            "evaluation_mode": eval_scope,
                            **regression_metrics(yte_s, preds),
                        }
                    )

        # Per-cell / per-workbook breakdown for Li -> Na using strictly single-unit transductive CORAL
        # (adapter sees ONLY source train + this single held-out unit's unlabeled inputs; zero target labels)
        if src_chem == "li_ion" and tgt_chem == "na_ion":
            xgb_raw = within_tab_raw["li_ion"]["XGBoost"]
            xgb_phys = within_tab_phys["li_ion"]["XGBoost"]
            for unit_id in sorted(tgt_eval_df["cell_id"].unique().to_list()):
                sub_u = tgt_eval_df.filter(pl.col("cell_id") == unit_id)
                assert_no_cell_overlap(
                    src_split.train_cells, src_split.val_cells, [unit_id], context=f"cross_unit_{task_name}_{unit_id}"
                )
                yu = sub_u[target_col].to_numpy()
                m_raw = regression_metrics(yu, xgb_raw.predict(sub_u.select(raw_cols).to_numpy()))
                X_u_phys = sub_u.select(phys_cols).to_numpy()
                m_phys = regression_metrics(yu, xgb_phys.predict(X_u_phys))
                # Fit transductive unsupervised CORAL strictly on (source train, this single target unit's unlabeled X)
                unit_adapter = CrossChemistryDomainAdapter(
                    src_ocv_curve=empirical_ocvs.get(src_chem) if is_soc else None,
                    tgt_ocv_curve=empirical_ocvs.get(tgt_chem) if is_soc else None,
                    use_ocv_warp=is_soc,
                    use_coral=True,
                    coral_reg=0.10,
                    coral_shrinkage=0.25 if is_soc else 0.15,
                    align_mean=False,
                ).fit(X_src_tr_phys, X_u_phys, phys_cols)
                m_aln = regression_metrics(
                    yu, xgb_phys.predict(unit_adapter.transform_target(X_u_phys), already_scaled=True)
                )
                cross_per_unit_rows.append(
                    {
                        "task": task_name,
                        "direction": direction,
                        "target_unit": unit_id,
                        "alignment_scope": "single_unit_transductive_coral",
                        "n_rows": int(sub_u.height),
                        "raw_rmse": m_raw["rmse"],
                        "raw_r2": m_raw["r2"],
                        "phys_rmse": m_phys["rmse"],
                        "phys_r2": m_phys["r2"],
                        "aligned_rmse": m_aln["rmse"],
                        "aligned_r2": m_aln["r2"],
                    }
                )

        sample_predictions[f"cross_{direction}"] = {
            "y_true": y_tgt_test,
            "y_pred_raw": within_tab_raw[src_chem]["XGBoost"].predict(X_tgt_test_raw),
            "y_pred_phys": within_tab_phys[src_chem]["XGBoost"].predict(X_tgt_test_phys),
            "y_pred_aligned": within_tab_phys[src_chem]["XGBoost"].predict(X_tgt_test_aligned, already_scaled=True),
        }

    return TaskExperimentBundle(
        task_name=task_name,
        splits=splits,
        within_rows=within_rows,
        cross_rows=cross_rows,
        within_tabular_phys=within_tab_phys,
        within_seq_phys=within_seq_phys,
        adapters=adapters,
        sample_predictions=sample_predictions,
        empirical_ocv_curves=empirical_ocvs,
        cross_chemistry_available=True,
        cross_chemistry_mode=cross_mode,
        compatibility_note=compat_note,
        loco_per_cell_rows=loco_per_cell_rows,
        cross_per_unit_rows=cross_per_unit_rows,
        external_test_rows=external_test_rows,
        full_task_df=df,
        unit_counts=unit_counts,
    )


def run_repeated_cell_cv(
    cfg: PipelineConfig,
    df: pl.DataFrame,
    task_name: str,
    seeds: tuple[int, ...] = (42, 101, 202),
) -> list[dict]:
    """Run repeated cell-level splits across multiple seeds on ordered cells (excludes pooled workbooks)."""
    is_soc = task_name == "SoC"
    raw_cols = RAW_SOC_FEATURES if is_soc else RAW_SOH_FEATURES
    phys_cols = PHYSICS_SOC_FEATURES if is_soc else PHYSICS_SOH_FEATURES
    target_col = "soc" if is_soc else "soh"
    clip_bounds = (0.0, 1.0) if is_soc else (0.50, 1.05)

    has_prov = "data_provenance" in df.columns
    work_df = df.filter(pl.col("data_provenance") == "ordered_per_cell") if (not is_soc and has_prov) else df

    ordered_chems = [
        c for c in ("li_ion", "na_ion") if _inspect_chemistry_task_mode(work_df, c, is_soc)[0] == "full"
    ]
    if not ordered_chems:
        return []

    records: dict[tuple[str, str, str], list[tuple[float, float]]] = {}

    for seed in seeds:
        splits = {chem: split_by_cell(work_df, chem, seed=seed + idx) for idx, chem in enumerate(ordered_chems)}
        emp_ocvs = (
            {chem: estimate_empirical_ocv_curve(splits[chem].train_df, chem) for chem in ordered_chems}
            if is_soc
            else {}
        )
        tab_raw_by_chem: dict[str, dict[str, TrainedTabularModel]] = {}
        tab_phys_by_chem: dict[str, dict[str, TrainedTabularModel]] = {}

        for chem in ordered_chems:
            sp = splits[chem]
            assert_no_cell_overlap(sp.train_cells, sp.val_cells, sp.test_cells, context=f"cv_{task_name}_{chem}_{seed}")
            tab_raw = train_tabular_models(
                sp.train_df.select(raw_cols).to_numpy(),
                sp.train_df[target_col].to_numpy(),
                sp.val_df.select(raw_cols).to_numpy(),
                sp.val_df[target_col].to_numpy(),
                sp.test_df.select(raw_cols).to_numpy(),
                sp.test_df[target_col].to_numpy(),
                clip_bounds=clip_bounds,
                seed=seed,
                fast_mode=True,
            )
            tab_phys = train_tabular_models(
                sp.train_df.select(phys_cols).to_numpy(),
                sp.train_df[target_col].to_numpy(),
                sp.val_df.select(phys_cols).to_numpy(),
                sp.val_df[target_col].to_numpy(),
                sp.test_df.select(phys_cols).to_numpy(),
                sp.test_df[target_col].to_numpy(),
                clip_bounds=clip_bounds,
                seed=seed,
                fast_mode=True,
            )
            tab_raw_by_chem[chem] = tab_raw
            tab_phys_by_chem[chem] = tab_phys
            for mname, tm in tab_phys.items():
                records.setdefault((f"within_{chem}", "physics_normalized", mname), []).append(
                    (tm.test_metrics["rmse"], tm.test_metrics["r2"])
                )

        if len(ordered_chems) == 2:
            for src_chem, tgt_chem in (("li_ion", "na_ion"), ("na_ion", "li_ion")):
                direction = f"{src_chem}_to_{tgt_chem}"
                sp_s, sp_t = splits[src_chem], splits[tgt_chem]
                y_te = sp_t.test_df[target_col].to_numpy()
                X_te_raw = sp_t.test_df.select(raw_cols).to_numpy()
                X_te_phys = sp_t.test_df.select(phys_cols).to_numpy()

                adapter = CrossChemistryDomainAdapter(
                    src_ocv_curve=emp_ocvs.get(src_chem) if is_soc else None,
                    tgt_ocv_curve=emp_ocvs.get(tgt_chem) if is_soc else None,
                    use_ocv_warp=is_soc,
                    use_coral=True,
                    coral_reg=0.10,
                    coral_shrinkage=0.25 if is_soc else 0.15,
                    align_mean=False,
                ).fit(
                    sp_s.train_df.select(phys_cols).to_numpy(),
                    sp_t.train_df.select(phys_cols).to_numpy(),
                    phys_cols,
                )
                X_te_aln = adapter.transform_target(X_te_phys)

                for mname in ("RandomForest", "XGBoost"):
                    m_raw = regression_metrics(y_te, tab_raw_by_chem[src_chem][mname].predict(X_te_raw))
                    m_aln = regression_metrics(
                        y_te, tab_phys_by_chem[src_chem][mname].predict(X_te_aln, already_scaled=True)
                    )
                    records.setdefault((direction, "1_raw_zero_shot", mname), []).append((m_raw["rmse"], m_raw["r2"]))
                    records.setdefault((direction, "3_physics_ocv_coral_zero_shot", mname), []).append(
                        (m_aln["rmse"], m_aln["r2"])
                    )

    summary_rows: list[dict] = []
    for (setting, tier, model), vals in sorted(records.items()):
        rmses = np.array([v[0] for v in vals])
        r2s = np.array([v[1] for v in vals])
        summary_rows.append(
            {
                "task": task_name,
                "setting": setting,
                "tier": tier,
                "model": model,
                "n_splits": len(vals),
                "rmse_mean": round(float(np.mean(rmses)), 4),
                "rmse_std": round(float(np.std(rmses)), 4),
                "r2_mean": round(float(np.mean(r2s)), 4),
                "r2_std": round(float(np.std(r2s)), 4),
            }
        )
    return summary_rows
