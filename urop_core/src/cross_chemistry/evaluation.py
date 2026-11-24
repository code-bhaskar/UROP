"""Within-chemistry baselines and 3-tier bidirectional Cross-Chemistry evaluation (Li-ion <-> Na-ion)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

from src.config import PipelineConfig
from src.cross_chemistry.alignment import CrossChemistryDomainAdapter
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


def _run_within_chemistry_for_representation(
    split: CellSplit,
    feature_cols: list[str],
    target_col: str,
    group_cols: list[str],
    clip_bounds: tuple[float, float],
    cfg: PipelineConfig,
    include_seq: bool = True,
) -> tuple[dict[str, TrainedTabularModel], dict[str, TrainedSequenceModel], StandardScaler]:
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

    if include_seq:
        seq_len = 8 if target_col == "soh" else cfg.seq_len
        epochs = (10 if cfg.fast_mode else cfg.seq_epochs)
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

    splits = {
        "li_ion": split_by_cell(df, "li_ion", seed=cfg.seed),
        "na_ion": split_by_cell(df, "na_ion", seed=cfg.seed + 1),
    }

    within_rows: list[dict] = []
    cross_rows: list[dict] = []
    within_tab_raw: dict[str, dict[str, TrainedTabularModel]] = {}
    within_seq_raw: dict[str, dict[str, TrainedSequenceModel]] = {}
    within_tab_phys: dict[str, dict[str, TrainedTabularModel]] = {}
    within_seq_phys: dict[str, dict[str, TrainedSequenceModel]] = {}
    adapters: dict[str, CrossChemistryDomainAdapter] = {}
    sample_predictions: dict[str, dict[str, np.ndarray]] = {}

    # 1. Train Within-Chemistry Models on BOTH Raw and Physics-Normalized Representations
    for chem in ("li_ion", "na_ion"):
        tab_raw, seq_raw, _ = _run_within_chemistry_for_representation(
            splits[chem], raw_cols, target_col, group_cols, clip_bounds, cfg, include_seq=True
        )
        tab_phys, seq_phys, _ = _run_within_chemistry_for_representation(
            splits[chem], phys_cols, target_col, group_cols, clip_bounds, cfg, include_seq=True
        )
        within_tab_raw[chem] = tab_raw
        within_seq_raw[chem] = seq_raw
        within_tab_phys[chem] = tab_phys
        within_seq_phys[chem] = seq_phys

        for mname, tm in tab_phys.items():
            within_rows.append(
                {
                    "task": task_name,
                    "chemistry": chem,
                    "model": mname,
                    "representation": "physics_normalized",
                    "train_cells": ",".join(splits[chem].train_cells),
                    "test_cells": ",".join(splits[chem].test_cells),
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
                    "train_cells": ",".join(splits[chem].train_cells),
                    "test_cells": ",".join(splits[chem].test_cells),
                    "mae": sm.test_metrics["mae"],
                    "rmse": sm.test_metrics["rmse"],
                    "r2": sm.test_metrics["r2"],
                    "train_time_s": sm.train_time_s,
                    "infer_time_ms": sm.infer_time_ms,
                }
            )

        # Save test parity sample for best within-chemistry tabular model (XGBoost)
        X_te_phys = splits[chem].test_df.select(phys_cols).to_numpy()
        y_te = splits[chem].test_df[target_col].to_numpy()
        sample_predictions[f"within_{chem}"] = {
            "y_true": y_te,
            "y_pred_xgb": tab_phys["XGBoost"].predict(X_te_phys),
            "y_pred_rf": tab_phys["RandomForest"].predict(X_te_phys),
        }

    # 2. Bidirectional Cross-Chemistry Evaluation (Li-ion -> Na-ion AND Na-ion -> Li-ion)
    for src_chem, tgt_chem in (("li_ion", "na_ion"), ("na_ion", "li_ion")):
        direction = f"{src_chem}_to_{tgt_chem}"
        src_split = splits[src_chem]
        tgt_split = splits[tgt_chem]

        y_tgt_test = tgt_split.test_df[target_col].to_numpy()

        # --- Tier 1: Naive Raw Features Zero-Shot ---
        X_tgt_test_raw = tgt_split.test_df.select(raw_cols).to_numpy()
        for mname, tm in within_tab_raw[src_chem].items():
            preds = tm.predict(X_tgt_test_raw)
            met = regression_metrics(y_tgt_test, preds)
            cross_rows.append(
                {
                    "task": task_name,
                    "direction": direction,
                    "tier": "1_raw_zero_shot",
                    "model": mname,
                    **met,
                }
            )
        for mname, sm in within_seq_raw[src_chem].items():
            Xte_s, yte_s = build_segmented_sequences(
                tgt_split.test_df, raw_cols, target_col, group_cols, seq_len, sm.scaler.transform
            )
            if len(Xte_s) > 0:
                preds = sm.predict_sequences(Xte_s)
                met = regression_metrics(yte_s, preds)
                cross_rows.append(
                    {
                        "task": task_name,
                        "direction": direction,
                        "tier": "1_raw_zero_shot",
                        "model": mname,
                        **met,
                    }
                )

        # --- Tier 2: Physics-Normalized Chemistry-Invariant Zero-Shot ---
        X_tgt_test_phys = tgt_split.test_df.select(phys_cols).to_numpy()
        for mname, tm in within_tab_phys[src_chem].items():
            preds = tm.predict(X_tgt_test_phys)
            met = regression_metrics(y_tgt_test, preds)
            cross_rows.append(
                {
                    "task": task_name,
                    "direction": direction,
                    "tier": "2_physics_normalized_zero_shot",
                    "model": mname,
                    **met,
                }
            )
        for mname, sm in within_seq_phys[src_chem].items():
            Xte_s, yte_s = build_segmented_sequences(
                tgt_split.test_df, phys_cols, target_col, group_cols, seq_len, sm.scaler.transform
            )
            if len(Xte_s) > 0:
                preds = sm.predict_sequences(Xte_s)
                met = regression_metrics(yte_s, preds)
                cross_rows.append(
                    {
                        "task": task_name,
                        "direction": direction,
                        "tier": "2_physics_normalized_zero_shot",
                        "model": mname,
                        **met,
                    }
                )

        # --- Tier 3: Physics + Unsupervised Domain Adaptation (OCV Quantile Warping + CORAL) ---
        # Strictly fit adapter ONLY on (source_train, target_train_unlabeled) - ZERO test data or target labels!
        X_src_tr_phys = src_split.train_df.select(phys_cols).to_numpy()
        X_tgt_tr_unlabeled = tgt_split.train_df.select(phys_cols).to_numpy()

        adapter = CrossChemistryDomainAdapter(
            src_chemistry=src_chem,
            tgt_chemistry=tgt_chem,
            use_ocv_warp=is_soc,  # Physics-anchored OCV-to-OCV warping resolves Li/Na OCV curvature mismatch
            use_coral=True,
            coral_reg=0.10,
            coral_shrinkage=0.25 if is_soc else 0.15,
            align_mean=False,
        ).fit(X_src_tr_phys, X_tgt_tr_unlabeled, phys_cols)
        adapters[direction] = adapter

        X_tgt_test_aligned = adapter.transform_target(X_tgt_test_phys)
        for mname, tm in within_tab_phys[src_chem].items():
            # Source tabular models were trained on StandardScaler(X_src_tr_phys), which matches adapter.transform_source!
            preds = tm.predict(X_tgt_test_aligned, already_scaled=True)
            met = regression_metrics(y_tgt_test, preds)
            cross_rows.append(
                {
                    "task": task_name,
                    "direction": direction,
                    "tier": "3_physics_ocv_coral_zero_shot",
                    "model": mname,
                    **met,
                }
            )

        for mname, sm in within_seq_phys[src_chem].items():
            Xte_s, yte_s = build_segmented_sequences(
                tgt_split.test_df, phys_cols, target_col, group_cols, seq_len, adapter.transform_target
            )
            if len(Xte_s) > 0:
                preds = sm.predict_sequences(Xte_s)
                met = regression_metrics(yte_s, preds)
                cross_rows.append(
                    {
                        "task": task_name,
                        "direction": direction,
                        "tier": "3_physics_ocv_coral_zero_shot",
                        "model": mname,
                        **met,
                    }
                )

        # Store sample cross-chemistry predictions (Raw vs Aligned) for plotting
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
    )
