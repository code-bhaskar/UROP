"""Few-shot Cross-Chemistry Transfer Learning:
- SoC (Li-ion -> Na-ion): 6-fold Leave-One-Cell-Out (LOCO) across all 6 Wang et al. Na-ion cells
  (tabular XGBoost Residual/WarmStart + Two-Phase GRU fine-tuning).
- SoH (Li-ion -> Na-ion, Exploratory): Strictly cross-workbook (train fraction on workbook A ->
  test on 100% of workbook B, averaged across both 25 °C <-> -15 °C directions; tabular ONLY,
  zero same-workbook random splits, zero sequence models on pooled workbooks).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import torch
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.config import PipelineConfig
from src.cross_chemistry.alignment import CrossChemistryDomainAdapter
from src.cross_chemistry.evaluation import TaskExperimentBundle, assert_no_cell_overlap
from src.mining.feature_engineering import PHYSICS_SOC_FEATURES, PHYSICS_SOH_FEATURES
from src.models.metrics import regression_metrics
from src.models.sequence import (
    build_segmented_sequences,
    finetune_sequence_model,
    train_sequence_model,
)


@dataclass
class TransferLearningBundle:
    task_name: str
    results_rows: list[dict]
    best_tl_li_to_na_predictor: object  # callable(X_phys) -> y_hat for adaptive charging integration
    per_fold_rows: list[dict] = field(default_factory=list)


class ResidualTransferXGBoost:
    """Physics-aligned Hypothesis-Transfer Residual Learner: y_hat = y_src(adapter(X)) + delta_f([X, y_src])."""

    def __init__(self, source_model, adapter, clip_bounds: tuple[float, float], seed: int = 42) -> None:
        self.source_model = source_model
        self.adapter = adapter
        self.clip_bounds = clip_bounds
        self.tgt_scaler = StandardScaler()
        self.residual_model = XGBRegressor(
            n_estimators=95,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.5,
            reg_alpha=0.1,
            random_state=seed,
            n_jobs=2,
            verbosity=0,
        )

    def fit(self, X_tgt_phys: np.ndarray, y_tgt: np.ndarray) -> "ResidualTransferXGBoost":
        X_aln = self.adapter.transform_target(X_tgt_phys)
        base_pred = self.source_model.predict(X_aln, already_scaled=True)
        X_meta = np.column_stack([self.tgt_scaler.fit_transform(X_tgt_phys), base_pred])
        residuals = y_tgt - base_pred
        self.residual_model.fit(X_meta, residuals)
        return self

    def predict(self, X_tgt_phys: np.ndarray) -> np.ndarray:
        X_aln = self.adapter.transform_target(X_tgt_phys)
        base_pred = self.source_model.predict(X_aln, already_scaled=True)
        X_meta = np.column_stack([self.tgt_scaler.transform(X_tgt_phys), base_pred])
        corr = self.residual_model.predict(X_meta)
        return np.clip(base_pred + corr, self.clip_bounds[0], self.clip_bounds[1])


def _subsample_stratified(
    X: np.ndarray,
    y: np.ndarray,
    frac: float,
    seed: int,
    min_samples: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic subsampling preserving coverage across target range (low-data regime)."""
    n = len(X)
    k = max(min_samples, int(round(n * frac)))
    k = min(k, n)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=k, replace=False))
    return X[idx], y[idx]


def _fit_and_eval_tabular_tl(
    src_xgb,
    adapter: CrossChemistryDomainAdapter,
    X_adapt: np.ndarray,
    y_adapt: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    clip_bounds: tuple[float, float],
    seed: int,
) -> tuple[dict[str, dict], object]:
    """Run TargetOnly_XGB, TL_Residual_XGB, and TL_WarmStart_XGB on one adapt/test split."""
    out: dict[str, dict] = {}
    X_test_aln = adapter.transform_target(X_test)
    X_adapt_aln = adapter.transform_target(X_adapt)

    # 1. Target-Only XGBoost
    t0 = time.perf_counter()
    sc_only = StandardScaler().fit(X_adapt)
    xgb_only = XGBRegressor(
        n_estimators=110,
        max_depth=4,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        random_state=seed,
        n_jobs=2,
        verbosity=0,
    ).fit(sc_only.transform(X_adapt), y_adapt)
    pred_only = np.clip(xgb_only.predict(sc_only.transform(X_test)), clip_bounds[0], clip_bounds[1])
    out["TargetOnly_XGBoost"] = {**regression_metrics(y_test, pred_only), "adapt_time_s": round(time.perf_counter() - t0, 4)}

    # 2. TL Residual XGBoost
    t0 = time.perf_counter()
    res_tl = ResidualTransferXGBoost(src_xgb, adapter, clip_bounds=clip_bounds, seed=seed).fit(X_adapt, y_adapt)
    pred_res = res_tl.predict(X_test)
    out["TL_Residual_XGBoost"] = {**regression_metrics(y_test, pred_res), "adapt_time_s": round(time.perf_counter() - t0, 4)}

    # 3. TL Warm-Start XGBoost
    t0 = time.perf_counter()
    xgb_ws = XGBRegressor(
        n_estimators=55,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=0,
    )
    xgb_ws.fit(X_adapt_aln, y_adapt, xgb_model=src_xgb.estimator.get_booster())
    pred_ws = np.clip(xgb_ws.predict(X_test_aln), clip_bounds[0], clip_bounds[1])
    out["TL_WarmStart_XGBoost"] = {**regression_metrics(y_test, pred_ws), "adapt_time_s": round(time.perf_counter() - t0, 4)}

    return out, res_tl


def _run_soc_loco_transfer_learning(
    cfg: PipelineConfig,
    bundle: TaskExperimentBundle,
) -> TransferLearningBundle:
    """6-fold Leave-One-Cell-Out Transfer Learning for Li-ion -> Na-ion SoC (plus Na -> Li summary)."""
    df = bundle.full_task_df
    assert df is not None
    phys_cols = PHYSICS_SOC_FEATURES
    group_cols = ["cell_id", "protocol", "cycle_number"]
    clip_bounds = (0.0, 1.0)
    seq_len = cfg.seq_len

    na_df = df.filter(pl.col("chemistry") == "na_ion")
    na_cells = sorted(na_df["cell_id"].unique().to_list())
    src_xgb = bundle.within_tabular_phys["li_ion"]["XGBoost"]
    src_gru = bundle.within_seq_phys["li_ion"].get("GRU")
    src_li_train_phys = bundle.splits["li_ion"].train_df.select(phys_cols).to_numpy()

    per_fold_rows: list[dict] = []
    agg_rows: list[dict] = []
    # (frac, method) -> list of metric dicts across 6 LOCO folds
    buckets: dict[tuple[float, str], list[dict]] = {}

    for fold_idx, test_cell in enumerate(na_cells):
        adapt_cells = [c for c in na_cells if c != test_cell]
        assert_no_cell_overlap(adapt_cells, [], [test_cell], context=f"tl_soc_loco_{test_cell}")

        adapt_df = na_df.filter(pl.col("cell_id").is_in(adapt_cells))
        test_df = na_df.filter(pl.col("cell_id") == test_cell)

        # Fit empirical target OCV strictly on adapt_cells (ZERO test-cell leakage!)
        from src.cross_chemistry.alignment import estimate_empirical_ocv_curve

        tgt_ocv_fold = estimate_empirical_ocv_curve(adapt_df, "na_ion")
        adapter_fold = CrossChemistryDomainAdapter(
            src_ocv_curve=bundle.empirical_ocv_curves.get("li_ion"),
            tgt_ocv_curve=tgt_ocv_fold,
            use_ocv_warp=True,
            use_coral=True,
            coral_reg=0.10,
            coral_shrinkage=0.25,
            align_mean=False,
        ).fit(
            src_li_train_phys,
            adapt_df.select(phys_cols).to_numpy(),
            phys_cols,
        )

        X_adapt_full = adapt_df.select(phys_cols).to_numpy()
        y_adapt_full = adapt_df["soc"].to_numpy()
        X_test = test_df.select(phys_cols).to_numpy()
        y_test = test_df["soc"].to_numpy()
        X_test_aln = adapter_fold.transform_target(X_test)

        # Zero-shot baseline on this held-out cell
        zs_pred = src_xgb.predict(X_test_aln, already_scaled=True)
        m_zs = regression_metrics(y_test, zs_pred)
        row_zs = {
            "task": "SoC",
            "direction": "li_ion_to_na_ion",
            "test_unit": test_cell,
            "target_fraction": 0.0,
            "n_target_samples": 0,
            "method": "ZeroShot_Aligned_XGBoost",
            "evaluation_mode": f"loco_{len(na_cells)}_folds",
            **m_zs,
            "adapt_time_s": 0.0,
        }
        per_fold_rows.append(row_zs)
        buckets.setdefault((0.0, "ZeroShot_Aligned_XGBoost"), []).append(row_zs)

        Xtr_seq_full, ytr_seq_full = build_segmented_sequences(
            adapt_df, phys_cols, "soc", group_cols, seq_len, adapter_fold.transform_target
        )
        Xte_seq, yte_seq = build_segmented_sequences(
            test_df, phys_cols, "soc", group_cols, seq_len, adapter_fold.transform_target
        )
        if src_gru is not None and len(Xte_seq) > 0:
            m_zs_g = regression_metrics(yte_seq, src_gru.predict_sequences(Xte_seq))
            row_zsg = {
                "task": "SoC",
                "direction": "li_ion_to_na_ion",
                "test_unit": test_cell,
                "target_fraction": 0.0,
                "n_target_samples": 0,
                "method": "ZeroShot_Aligned_GRU",
                "evaluation_mode": f"loco_{len(na_cells)}_folds",
                **m_zs_g,
                "adapt_time_s": 0.0,
            }
            per_fold_rows.append(row_zsg)
            buckets.setdefault((0.0, "ZeroShot_Aligned_GRU"), []).append(row_zsg)

        for frac in cfg.tl_fractions:
            seed_f = cfg.seed + fold_idx * 31 + int(frac * 1000)
            X_sub, y_sub = _subsample_stratified(X_adapt_full, y_adapt_full, frac, seed=seed_f)
            tab_res, _ = _fit_and_eval_tabular_tl(
                src_xgb, adapter_fold, X_sub, y_sub, X_test, y_test, clip_bounds, seed=seed_f
            )
            for mname, metrics in tab_res.items():
                r = {
                    "task": "SoC",
                    "direction": "li_ion_to_na_ion",
                    "test_unit": test_cell,
                    "target_fraction": frac,
                    "n_target_samples": len(y_sub),
                    "method": mname,
                    "evaluation_mode": f"loco_{len(na_cells)}_folds",
                    **metrics,
                }
                per_fold_rows.append(r)
                buckets.setdefault((frac, mname), []).append(r)

            # Sequence TL (TargetOnly_GRU & TL_TwoPhase_GRU) with reduced epochs per LOCO fold
            if src_gru is not None and len(Xtr_seq_full) >= 32 and len(Xte_seq) > 0:
                Xs_sub, ys_sub = _subsample_stratified(Xtr_seq_full, ytr_seq_full, frac, seed=seed_f, min_samples=20)
                val_cut = max(8, int(0.85 * len(Xs_sub)))
                Xs_tr, ys_tr = Xs_sub[:val_cut], ys_sub[:val_cut]
                Xs_va, ys_va = (
                    (Xs_sub[val_cut:], ys_sub[val_cut:]) if len(Xs_sub) - val_cut >= 4 else (Xs_sub, ys_sub)
                )

                t0 = time.perf_counter()
                gru_only = train_sequence_model(
                    Xs_tr, ys_tr, Xs_va, ys_va, Xte_seq, yte_seq,
                    rnn_type="GRU",
                    hidden_dim=src_gru.config["hidden_dim"],
                    num_layers=src_gru.config["num_layers"],
                    epochs=4 if cfg.fast_mode else 6,
                    batch_size=min(256, max(32, len(Xs_tr) // 4)),
                    lr=2.5e-3,
                    clip_bounds=clip_bounds,
                    seed=seed_f,
                )
                r_go = {
                    "task": "SoC",
                    "direction": "li_ion_to_na_ion",
                    "test_unit": test_cell,
                    "target_fraction": frac,
                    "n_target_samples": len(ys_sub),
                    "method": "TargetOnly_GRU",
                    "evaluation_mode": f"loco_{len(na_cells)}_folds",
                    **gru_only.test_metrics,
                    "adapt_time_s": round(time.perf_counter() - t0, 4),
                }
                per_fold_rows.append(r_go)
                buckets.setdefault((frac, "TargetOnly_GRU"), []).append(r_go)

                t0 = time.perf_counter()
                gru_tl_net = finetune_sequence_model(
                    src_gru,
                    Xs_tr,
                    ys_tr,
                    Xs_va,
                    ys_va,
                    head_epochs=3 if cfg.fast_mode else 4,
                    full_epochs=3 if cfg.fast_mode else 5,
                    lr_head=3e-3,
                    lr_full=8e-4,
                    seed=seed_f,
                )
                gru_tl_net.eval()
                with torch.no_grad():
                    pred_gt = np.clip(
                        gru_tl_net(torch.tensor(Xte_seq, dtype=torch.float32)).cpu().numpy(),
                        clip_bounds[0],
                        clip_bounds[1],
                    )
                r_gt = {
                    "task": "SoC",
                    "direction": "li_ion_to_na_ion",
                    "test_unit": test_cell,
                    "target_fraction": frac,
                    "n_target_samples": len(ys_sub),
                    "method": "TL_TwoPhase_GRU",
                    "evaluation_mode": f"loco_{len(na_cells)}_folds",
                    **regression_metrics(yte_seq, pred_gt),
                    "adapt_time_s": round(time.perf_counter() - t0, 4),
                }
                per_fold_rows.append(r_gt)
                buckets.setdefault((frac, "TL_TwoPhase_GRU"), []).append(r_gt)

    for (frac, method), frows in sorted(buckets.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        agg_rows.append(
            {
                "task": "SoC",
                "direction": "li_ion_to_na_ion",
                "target_fraction": frac,
                "n_target_samples": int(round(float(np.mean([r["n_target_samples"] for r in frows])))),
                "n_folds": len(frows),
                "method": method,
                "evaluation_mode": f"loco_{len(na_cells)}_folds",
                "mae": round(float(np.mean([r["mae"] for r in frows])), 5),
                "rmse": round(float(np.mean([r["rmse"] for r in frows])), 5),
                "rmse_std": round(float(np.std([r["rmse"] for r in frows])), 5),
                "r2": round(float(np.mean([r["r2"] for r in frows])), 5),
                "r2_std": round(float(np.std([r["r2"] for r in frows])), 5),
                "adapt_time_s": round(float(np.mean([r["adapt_time_s"] for r in frows])), 4),
            }
        )

    # Fit deployable Li -> Na ResidualTransferXGBoost on canonical Na-ion train cells for Objective 5 simulation
    na_train_df = bundle.splits["na_ion"].train_df
    best_predictor = ResidualTransferXGBoost(
        src_xgb, bundle.adapters["li_ion_to_na_ion"], clip_bounds=clip_bounds, seed=cfg.seed
    ).fit(na_train_df.select(phys_cols).to_numpy(), na_train_df["soc"].to_numpy())

    return TransferLearningBundle(
        task_name="SoC",
        results_rows=agg_rows,
        best_tl_li_to_na_predictor=best_predictor,
        per_fold_rows=per_fold_rows,
    )


def _run_soh_exploratory_cross_workbook_tl(
    cfg: PipelineConfig,
    bundle: TaskExperimentBundle,
) -> TransferLearningBundle:
    """Exploratory Li-ion -> Na-ion SoH Transfer Learning: strictly cross-workbook, tabular ONLY.

    Fold A: adapt on fraction of na_nfm_25C (25 °C)  -> test on 100% of na_nfm_m15C (-15 °C).
    Fold B: adapt on fraction of na_nfm_m15C (-15 °C) -> test on 100% of na_nfm_25C (25 °C).
    Never splits rows inside the same workbook; never runs GRU/LSTM on pooled workbooks.
    """
    df = bundle.full_task_df
    assert df is not None
    phys_cols = PHYSICS_SOH_FEATURES
    clip_bounds = (0.50, 1.05)

    na_df = df.filter(pl.col("chemistry") == "na_ion")
    units = sorted(na_df["cell_id"].unique().to_list())
    if len(units) < 2:
        return TransferLearningBundle(task_name="SoH", results_rows=[], best_tl_li_to_na_predictor=None)

    src_xgb = bundle.within_tabular_phys["li_ion"]["XGBoost"]
    src_li_train_phys = bundle.splits["li_ion"].train_df.select(phys_cols).to_numpy()

    per_fold_rows: list[dict] = []
    buckets: dict[tuple[float, str], list[dict]] = {}

    for fold_idx, test_unit in enumerate(units):
        adapt_units = [u for u in units if u != test_unit]
        assert_no_cell_overlap(adapt_units, [], [test_unit], context=f"tl_soh_xwb_{test_unit}")

        adapt_df = na_df.filter(pl.col("cell_id").is_in(adapt_units))
        test_df = na_df.filter(pl.col("cell_id") == test_unit)

        X_adapt_full = adapt_df.select(phys_cols).to_numpy()
        y_adapt_full = adapt_df["soh"].to_numpy()
        X_test = test_df.select(phys_cols).to_numpy()
        y_test = test_df["soh"].to_numpy()

        adapter_fold = CrossChemistryDomainAdapter(
            use_ocv_warp=False,
            use_coral=True,
            coral_reg=0.10,
            coral_shrinkage=0.15,
            align_mean=False,
        ).fit(src_li_train_phys, X_adapt_full, phys_cols)

        m_zs = regression_metrics(y_test, src_xgb.predict(adapter_fold.transform_target(X_test), already_scaled=True))
        row_zs = {
            "task": "SoH",
            "direction": "li_ion_to_na_ion",
            "adapt_unit": ",".join(adapt_units),
            "test_unit": test_unit,
            "target_fraction": 0.0,
            "n_target_samples": 0,
            "method": "ZeroShot_Aligned_XGBoost",
            "evaluation_mode": "exploratory_cross_workbook_temp_confounded",
            **m_zs,
            "adapt_time_s": 0.0,
        }
        per_fold_rows.append(row_zs)
        buckets.setdefault((0.0, "ZeroShot_Aligned_XGBoost"), []).append(row_zs)

        for frac in cfg.tl_fractions:
            seed_f = cfg.seed + fold_idx * 47 + int(frac * 1000)
            X_sub, y_sub = _subsample_stratified(X_adapt_full, y_adapt_full, frac, seed=seed_f)
            tab_res, _ = _fit_and_eval_tabular_tl(
                src_xgb, adapter_fold, X_sub, y_sub, X_test, y_test, clip_bounds, seed=seed_f
            )
            for mname, metrics in tab_res.items():
                r = {
                    "task": "SoH",
                    "direction": "li_ion_to_na_ion",
                    "adapt_unit": ",".join(adapt_units),
                    "test_unit": test_unit,
                    "target_fraction": frac,
                    "n_target_samples": len(y_sub),
                    "method": mname,
                    "evaluation_mode": "exploratory_cross_workbook_temp_confounded",
                    **metrics,
                }
                per_fold_rows.append(r)
                buckets.setdefault((frac, mname), []).append(r)

    agg_rows: list[dict] = []
    for (frac, method), frows in sorted(buckets.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        agg_rows.append(
            {
                "task": "SoH",
                "direction": "li_ion_to_na_ion",
                "target_fraction": frac,
                "n_target_samples": int(round(float(np.mean([r["n_target_samples"] for r in frows])))),
                "n_folds": len(frows),
                "method": method,
                "evaluation_mode": "exploratory_cross_workbook_temp_confounded",
                "mae": round(float(np.mean([r["mae"] for r in frows])), 5),
                "rmse": round(float(np.mean([r["rmse"] for r in frows])), 5),
                "rmse_std": round(float(np.std([r["rmse"] for r in frows])), 5),
                "r2": round(float(np.mean([r["r2"] for r in frows])), 5),
                "r2_std": round(float(np.std([r["r2"] for r in frows])), 5),
                "adapt_time_s": round(float(np.mean([r["adapt_time_s"] for r in frows])), 4),
            }
        )

    return TransferLearningBundle(
        task_name="SoH",
        results_rows=agg_rows,
        best_tl_li_to_na_predictor=None,
        per_fold_rows=per_fold_rows,
    )


def run_transfer_learning_experiments(
    cfg: PipelineConfig,
    bundle: TaskExperimentBundle,
) -> TransferLearningBundle:
    """Dispatch Transfer Learning strictly honoring dataset provenance and unit structure."""
    if not bundle.cross_chemistry_available:
        return TransferLearningBundle(
            task_name=bundle.task_name,
            results_rows=[],
            best_tl_li_to_na_predictor=None,
        )
    if bundle.task_name == "SoC":
        return _run_soc_loco_transfer_learning(cfg, bundle)
    return _run_soh_exploratory_cross_workbook_tl(cfg, bundle)
