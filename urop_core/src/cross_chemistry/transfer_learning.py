"""Few-shot Cross-Chemistry Transfer Learning: Zero-Shot vs Target-Only vs Residual/Warm-Start XGBoost & Two-Phase GRU/LSTM."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.config import PipelineConfig
from src.cross_chemistry.evaluation import TaskExperimentBundle
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


def _subsample_stratified_temporal(
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


def run_transfer_learning_experiments(
    cfg: PipelineConfig,
    bundle: TaskExperimentBundle,
) -> TransferLearningBundle:
    """Run few-shot transfer learning experiments in both directions across target data budgets."""
    is_soc = bundle.task_name == "SoC"
    phys_cols = PHYSICS_SOC_FEATURES if is_soc else PHYSICS_SOH_FEATURES
    target_col = "soc" if is_soc else "soh"
    group_cols = ["cell_id", "protocol", "cycle_number"] if is_soc else ["cell_id"]
    clip_bounds = (0.0, 1.0) if is_soc else (0.50, 1.05)
    seq_len = cfg.seq_len if is_soc else 8

    rows: list[dict] = []
    best_li_to_na_model = None
    best_li_to_na_rmse = float("inf")

    for src_chem, tgt_chem in (("li_ion", "na_ion"), ("na_ion", "li_ion")):
        direction = f"{src_chem}_to_{tgt_chem}"
        tgt_split = bundle.splits[tgt_chem]
        adapter = bundle.adapters[direction]
        src_xgb = bundle.within_tabular_phys[src_chem]["XGBoost"]
        src_gru = bundle.within_seq_phys[src_chem].get("GRU")

        X_tgt_tr_full = tgt_split.train_df.select(phys_cols).to_numpy()
        y_tgt_tr_full = tgt_split.train_df[target_col].to_numpy()
        X_tgt_te = tgt_split.test_df.select(phys_cols).to_numpy()
        y_tgt_te = tgt_split.test_df[target_col].to_numpy()
        X_tgt_te_aln = adapter.transform_target(X_tgt_te)

        # Build aligned sequences for target train, val, and test (for GRU TL comparison)
        Xtr_seq_full, ytr_seq_full = build_segmented_sequences(
            tgt_split.train_df, phys_cols, target_col, group_cols, seq_len, adapter.transform_target
        )
        Xva_seq, yva_seq = build_segmented_sequences(
            tgt_split.val_df, phys_cols, target_col, group_cols, seq_len, adapter.transform_target
        )
        Xte_seq, yte_seq = build_segmented_sequences(
            tgt_split.test_df, phys_cols, target_col, group_cols, seq_len, adapter.transform_target
        )

        # Zero-shot aligned baselines (0% target labels)
        zs_xgb_preds = src_xgb.predict(X_tgt_te_aln, already_scaled=True)
        zs_xgb_met = regression_metrics(y_tgt_te, zs_xgb_preds)
        rows.append(
            {
                "task": bundle.task_name,
                "direction": direction,
                "target_fraction": 0.0,
                "n_target_samples": 0,
                "method": "ZeroShot_Aligned_XGBoost",
                "train_time_s": 0.0,
                **zs_xgb_met,
            }
        )
        if src_gru is not None and len(Xte_seq) > 0:
            zs_gru_preds = src_gru.predict_sequences(Xte_seq)
            zs_gru_met = regression_metrics(yte_seq, zs_gru_preds)
            rows.append(
                {
                    "task": bundle.task_name,
                    "direction": direction,
                    "target_fraction": 0.0,
                    "n_target_samples": 0,
                    "method": "ZeroShot_Aligned_GRU",
                    "train_time_s": 0.0,
                    **zs_gru_met,
                }
            )

        for frac in cfg.tl_fractions:
            X_sub, y_sub = _subsample_stratified_temporal(
                X_tgt_tr_full, y_tgt_tr_full, frac=frac, seed=cfg.seed + int(frac * 100)
            )
            X_sub_aln = adapter.transform_target(X_sub)
            n_sub = len(X_sub)

            # 1. Target-Only XGBoost (trained from scratch on limited target data)
            t0 = time.perf_counter()
            sc_only = StandardScaler().fit(X_sub)
            tgt_only_xgb = XGBRegressor(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.06,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                random_state=cfg.seed,
                n_jobs=2,
                verbosity=0,
            )
            tgt_only_xgb.fit(sc_only.transform(X_sub), y_sub)
            t_only = time.perf_counter() - t0
            pred_only = np.clip(tgt_only_xgb.predict(sc_only.transform(X_tgt_te)), *clip_bounds)
            rows.append(
                {
                    "task": bundle.task_name,
                    "direction": direction,
                    "target_fraction": frac,
                    "n_target_samples": n_sub,
                    "method": "TargetOnly_XGBoost",
                    "train_time_s": round(t_only, 4),
                    **regression_metrics(y_tgt_te, pred_only),
                }
            )

            # 2. Transfer Learning: Source-Residual XGBoost (Physics + OCV + CORAL aligned)
            t0 = time.perf_counter()
            res_tl = ResidualTransferXGBoost(
                source_model=src_xgb,
                adapter=adapter,
                clip_bounds=clip_bounds,
                seed=cfg.seed,
            ).fit(X_sub, y_sub)
            t_res = time.perf_counter() - t0
            pred_res = res_tl.predict(X_tgt_te)
            met_res = regression_metrics(y_tgt_te, pred_res)
            rows.append(
                {
                    "task": bundle.task_name,
                    "direction": direction,
                    "target_fraction": frac,
                    "n_target_samples": n_sub,
                    "method": "TL_Residual_XGBoost",
                    "train_time_s": round(t_res, 4),
                    **met_res,
                }
            )
            if direction == "li_ion_to_na_ion" and frac == 0.25 and met_res["rmse"] < best_li_to_na_rmse:
                best_li_to_na_rmse = met_res["rmse"]
                best_li_to_na_model = res_tl

            # 3. Transfer Learning: XGBoost Warm-Start Tree Continuation
            t0 = time.perf_counter()
            warm_xgb = XGBRegressor(
                n_estimators=55,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=3.0,
                random_state=cfg.seed,
                n_jobs=2,
                verbosity=0,
            )
            warm_xgb.fit(X_sub_aln, y_sub, xgb_model=src_xgb.estimator.get_booster())
            t_warm = time.perf_counter() - t0
            pred_warm = np.clip(warm_xgb.predict(X_tgt_te_aln), *clip_bounds)
            rows.append(
                {
                    "task": bundle.task_name,
                    "direction": direction,
                    "target_fraction": frac,
                    "n_target_samples": n_sub,
                    "method": "TL_WarmStart_XGBoost",
                    "train_time_s": round(t_warm, 4),
                    **regression_metrics(y_tgt_te, pred_warm),
                }
            )

            # 4 & 5. Neural Sequence Comparison: Target-Only GRU vs Two-Phase Transfer-Learning GRU
            if src_gru is not None and len(Xtr_seq_full) >= 30 and len(Xva_seq) >= 5 and len(Xte_seq) >= 5:
                X_seq_sub, y_seq_sub = _subsample_stratified_temporal(
                    Xtr_seq_full, ytr_seq_full, frac=frac, seed=cfg.seed + int(frac * 100), min_samples=20
                )

                # Target-Only GRU
                gru_only = train_sequence_model(
                    X_seq_sub,
                    y_seq_sub,
                    Xva_seq,
                    yva_seq,
                    Xte_seq,
                    yte_seq,
                    rnn_type="GRU",
                    hidden_dim=32 if cfg.fast_mode else cfg.seq_hidden_dim,
                    num_layers=2,
                    epochs=10 if cfg.fast_mode else 14,
                    batch_size=64,
                    lr=2e-3,
                    clip_bounds=clip_bounds,
                    seed=cfg.seed,
                )
                rows.append(
                    {
                        "task": bundle.task_name,
                        "direction": direction,
                        "target_fraction": frac,
                        "n_target_samples": len(X_seq_sub),
                        "method": "TargetOnly_GRU",
                        "train_time_s": gru_only.train_time_s,
                        **gru_only.test_metrics,
                    }
                )

                # Two-Phase Fine-Tuned Transfer-Learning GRU
                t0 = time.perf_counter()
                ft_net = finetune_sequence_model(
                    source_model=src_gru,
                    X_tgt_train=X_seq_sub,
                    y_tgt_train=y_seq_sub,
                    X_tgt_val=Xva_seq,
                    y_tgt_val=yva_seq,
                    head_epochs=5 if cfg.fast_mode else 8,
                    full_epochs=8 if cfg.fast_mode else 12,
                    seed=cfg.seed,
                )
                t_ft = time.perf_counter() - t0
                with torch.no_grad():
                    ft_preds = np.clip(
                        ft_net(torch.tensor(Xte_seq, dtype=torch.float32)).cpu().numpy(),
                        *clip_bounds,
                    )
                rows.append(
                    {
                        "task": bundle.task_name,
                        "direction": direction,
                        "target_fraction": frac,
                        "n_target_samples": len(X_seq_sub),
                        "method": "TL_TwoPhase_GRU",
                        "train_time_s": round(t_ft, 4),
                        **regression_metrics(yte_seq, ft_preds),
                    }
                )

    return TransferLearningBundle(
        task_name=bundle.task_name,
        results_rows=rows,
        best_tl_li_to_na_predictor=best_li_to_na_model,
    )
