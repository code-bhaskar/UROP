"""Within-chemistry baselines, 3-tier bidirectional Cross-Chemistry evaluation, and Repeated Cell-Level CV.

Key scientific safeguards:
1. Strict battery-cell-level splits (zero cell leakage).
2. Both validation (`val_mae`, `val_rmse`, `val_r2`) and held-out test metrics are reported.
3. Empirical OCV curves for Tier 3 alignment are estimated strictly from training cells only
   (zero simulator imports, zero test-cell leakage).
4. Graceful chemistry compatibility handling: if a task (e.g., real Na-ion SoH without multi-cycle
   aging data) only has genuine multi-cell data in one chemistry, within-chemistry modeling runs
   for that chemistry while cross-chemistry is cleanly flagged as unavailable rather than fabricated.
5. Repeated cell-level split CV reports mean ± std across multiple held-out cell partitions.
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
    compatibility_note: str = ""


def _chemistry_has_valid_task_data(df: pl.DataFrame, chemistry: str, is_soc: bool) -> bool:
    if df.height == 0:
        return False
    sub = df.filter(pl.col("chemistry") == chemistry)
    if sub.height < 30:
        return False
    n_cells = sub["cell_id"].n_unique()
    if n_cells < 3:
        return False
    if not is_soc:
        # Genuine SoH modeling requires meaningful capacity spread across cycles (not flat 1.0 placeholders)
        soh_arr = sub["soh"].to_numpy()
        if float(np.nanmax(soh_arr) - np.nanmin(soh_arr)) < 0.02:
            return False
    return True


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
        epochs = 10 if cfg.fast_mode else cfg.seq_epochs
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

    available_chems = [c for c in ("li_ion", "na_ion") if _chemistry_has_valid_task_data(df, c, is_soc)]
    splits: dict[str, CellSplit] = {}
    for idx_c, chem in enumerate(available_chems):
        splits[chem] = split_by_cell(df, chem, seed=cfg.seed + idx_c)

    within_rows: list[dict] = []
    cross_rows: list[dict] = []
    within_tab_raw: dict[str, dict[str, TrainedTabularModel]] = {}
    within_seq_raw: dict[str, dict[str, TrainedSequenceModel]] = {}
    within_tab_phys: dict[str, dict[str, TrainedTabularModel]] = {}
    within_seq_phys: dict[str, dict[str, TrainedSequenceModel]] = {}
    adapters: dict[str, CrossChemistryDomainAdapter] = {}
    sample_predictions: dict[str, dict[str, np.ndarray]] = {}
    empirical_ocvs: dict[str, EmpiricalOCVCurve] = {}

    # 1. Train Within-Chemistry Models on BOTH Raw and Physics-Normalized Representations
    for chem in available_chems:
        if is_soc:
            # Estimate empirical 1D OCV curve strictly on this chemistry's TRAINING cells only!
            empirical_ocvs[chem] = estimate_empirical_ocv_curve(splits[chem].train_df, chem)

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
                    "val_cells": ",".join(splits[chem].val_cells),
                    "test_cells": ",".join(splits[chem].test_cells),
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
                    "train_cells": ",".join(splits[chem].train_cells),
                    "val_cells": ",".join(splits[chem].val_cells),
                    "test_cells": ",".join(splits[chem].test_cells),
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

        X_te_phys = splits[chem].test_df.select(phys_cols).to_numpy()
        y_te = splits[chem].test_df[target_col].to_numpy()
        sample_predictions[f"within_{chem}"] = {
            "y_true": y_te,
            "y_pred_xgb": tab_phys["XGBoost"].predict(X_te_phys),
            "y_pred_rf": tab_phys["RandomForest"].predict(X_te_phys),
        }

    cross_available = ("li_ion" in available_chems) and ("na_ion" in available_chems)
    compat_note = ""
    if not cross_available:
        missing = [c for c in ("li_ion", "na_ion") if c not in available_chems]
        compat_note = (
            f"Cross-chemistry {task_name} skipped cleanly because {missing} lacks genuine multi-cell "
            f"{task_name} data in the current dataset (per Proposal Section 5.1 compatibility check)."
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
            compatibility_note=compat_note,
        )

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
            cross_rows.append(
                {
                    "task": task_name,
                    "direction": direction,
                    "tier": "1_raw_zero_shot",
                    "model": mname,
                    **regression_metrics(y_tgt_test, preds),
                }
            )
        for mname, sm in within_seq_raw[src_chem].items():
            Xte_s, yte_s = build_segmented_sequences(
                tgt_split.test_df, raw_cols, target_col, group_cols, seq_len, sm.scaler.transform
            )
            if len(Xte_s) > 0:
                preds = sm.predict_sequences(Xte_s)
                cross_rows.append(
                    {
                        "task": task_name,
                        "direction": direction,
                        "tier": "1_raw_zero_shot",
                        "model": mname,
                        **regression_metrics(yte_s, preds),
                    }
                )

        # --- Tier 2: Physics-Normalized Chemistry-Invariant Zero-Shot ---
        X_tgt_test_phys = tgt_split.test_df.select(phys_cols).to_numpy()
        for mname, tm in within_tab_phys[src_chem].items():
            preds = tm.predict(X_tgt_test_phys)
            cross_rows.append(
                {
                    "task": task_name,
                    "direction": direction,
                    "tier": "2_physics_normalized_zero_shot",
                    "model": mname,
                    **regression_metrics(y_tgt_test, preds),
                }
            )
        for mname, sm in within_seq_phys[src_chem].items():
            Xte_s, yte_s = build_segmented_sequences(
                tgt_split.test_df, phys_cols, target_col, group_cols, seq_len, sm.scaler.transform
            )
            if len(Xte_s) > 0:
                preds = sm.predict_sequences(Xte_s)
                cross_rows.append(
                    {
                        "task": task_name,
                        "direction": direction,
                        "tier": "2_physics_normalized_zero_shot",
                        "model": mname,
                        **regression_metrics(yte_s, preds),
                    }
                )

        # --- Tier 3: Physics + Empirical Training-Only OCV Warping + Shrinkage CORAL ---
        # Strictly fit adapter ONLY on (source_train, target_train_unlabeled) + training-only empirical OCV curves!
        X_src_tr_phys = src_split.train_df.select(phys_cols).to_numpy()
        X_tgt_tr_unlabeled = tgt_split.train_df.select(phys_cols).to_numpy()

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
                    **regression_metrics(y_tgt_test, preds),
                }
            )
        for mname, sm in within_seq_phys[src_chem].items():
            Xte_s, yte_s = build_segmented_sequences(
                tgt_split.test_df, phys_cols, target_col, group_cols, seq_len, adapter.transform_target
            )
            if len(Xte_s) > 0:
                preds = sm.predict_sequences(Xte_s)
                cross_rows.append(
                    {
                        "task": task_name,
                        "direction": direction,
                        "tier": "3_physics_ocv_coral_zero_shot",
                        "model": mname,
                        **regression_metrics(yte_s, preds),
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
    )


def run_repeated_cell_cv(
    cfg: PipelineConfig,
    df: pl.DataFrame,
    task_name: str,
    seeds: tuple[int, ...] = (42, 101, 202),
) -> list[dict]:
    """Run repeated cell-level splits across multiple seeds to quantify metric mean ± std across held-out cells."""
    is_soc = task_name == "SoC"
    raw_cols = RAW_SOC_FEATURES if is_soc else RAW_SOH_FEATURES
    phys_cols = PHYSICS_SOC_FEATURES if is_soc else PHYSICS_SOH_FEATURES
    target_col = "soc" if is_soc else "soh"
    clip_bounds = (0.0, 1.0) if is_soc else (0.50, 1.05)

    available_chems = [c for c in ("li_ion", "na_ion") if _chemistry_has_valid_task_data(df, c, is_soc)]
    if not available_chems:
        return []

    records: dict[tuple[str, str, str], list[tuple[float, float]]] = {}

    for seed in seeds:
        splits = {chem: split_by_cell(df, chem, seed=seed + idx) for idx, chem in enumerate(available_chems)}
        emp_ocvs = {chem: estimate_empirical_ocv_curve(splits[chem].train_df, chem) for chem in available_chems} if is_soc else {}
        tab_raw_by_chem: dict[str, dict[str, TrainedTabularModel]] = {}
        tab_phys_by_chem: dict[str, dict[str, TrainedTabularModel]] = {}

        for chem in available_chems:
            sp = splits[chem]
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

        if len(available_chems) == 2:
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
