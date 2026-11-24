"""Unsupervised Cross-Chemistry Domain Alignment (Li-ion <-> Na-ion).

Implements three complementary physics + statistical domain adaptation layers
(using ZERO dynamic target labels):
1. Physics-Normalized Chemistry-Invariant Representation (computed in Gold layer)
2. Physics-Anchored OCV-to-OCV Equilibrium Warping: maps target normalized voltage
   onto source normalized voltage via each chemistry's 1D static datasheet OCV curve
   v_src(v_tgt^{-1}(v)), immune to sample SoC imbalance across datasets.
3. Shrinkage CORAL Covariance Alignment (Sun et al., AAAI 2016) with optional mean
   preservation so physical zero/BoL reference anchors are never distorted.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.config import CHEMISTRY_SPECS
from src.data.physics_simulator import ocv_curve


def _sym_matrix_power(mat: np.ndarray, power: float, eps: float = 1e-6) -> np.ndarray:
    """Stable symmetric PSD matrix power via eigendecomposition."""
    sym = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, eps, None)
    return (eigvecs * np.power(eigvals, power)) @ eigvecs.T


class CORALAligner:
    """Shrinkage Correlation Alignment (CORAL) mapping target covariance toward source covariance."""

    def __init__(self, reg: float = 0.08, shrinkage: float = 0.65, align_mean: bool = False) -> None:
        self.reg = reg
        self.shrinkage = shrinkage
        self.align_mean = align_mean
        self.mu_src_: np.ndarray | None = None
        self.mu_tgt_: np.ndarray | None = None
        self.transform_t2s_: np.ndarray | None = None

    def fit(self, X_src: np.ndarray, X_tgt_unlabeled: np.ndarray) -> "CORALAligner":
        self.mu_src_ = np.mean(X_src, axis=0)
        self.mu_tgt_ = np.mean(X_tgt_unlabeled, axis=0)

        d = X_src.shape[1]
        cov_s = np.cov(X_src - self.mu_src_, rowvar=False) + self.reg * np.eye(d)
        cov_t = np.cov(X_tgt_unlabeled - self.mu_tgt_, rowvar=False) + self.reg * np.eye(d)

        c_t_inv_sqrt = _sym_matrix_power(cov_t, -0.5, eps=1e-6)
        c_s_sqrt = _sym_matrix_power(cov_s, 0.5, eps=1e-6)
        full_map = c_t_inv_sqrt @ c_s_sqrt
        # Shrinkage interpolation between identity and full CORAL preserves physical axes
        self.transform_t2s_ = (1.0 - self.shrinkage) * np.eye(d) + self.shrinkage * full_map
        return self

    def transform_target(self, X_tgt: np.ndarray) -> np.ndarray:
        assert self.mu_src_ is not None and self.mu_tgt_ is not None and self.transform_t2s_ is not None
        if self.align_mean:
            return (X_tgt - self.mu_tgt_) @ self.transform_t2s_ + self.mu_src_
        return (X_tgt - self.mu_tgt_) @ self.transform_t2s_ + self.mu_tgt_


class OCVPhysicsAligner:
    """Physics-Anchored OCV-to-OCV Warping for voltage-derived features.

    Uses only the 1D static reference OCV curve of each chemistry (datasheet / C/20 OCV table)
    to map target normalized voltage v_tgt -> equilibrium SoC -> source normalized voltage v_src.
    Falls back to monotonic quantile warping if custom/unknown chemistry names are passed.
    """

    VOLTAGE_COLS = (
        "v_norm",
        "v_norm_roll_mean",
        "v_ir_comp_norm",
        "v_ir_comp_roll_mean",
    )

    def __init__(self, src_chemistry: str | None = None, tgt_chemistry: str | None = None, n_grid: int = 401) -> None:
        self.src_chemistry = src_chemistry
        self.tgt_chemistry = tgt_chemistry
        self.n_grid = n_grid
        self.feature_names_: list[str] = []
        self.maps_: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.v_norm_idx_: int | None = None
        self.sq_idx_: int | None = None
        self.cu_idx_: int | None = None
        self.p_norm_idx_: int | None = None
        self.c_rate_idx_: int | None = None
        self.dv_scale_: float = 1.0
        self.dv_idx_: int | None = None

    def fit(
        self,
        X_src: np.ndarray,
        X_tgt_unlabeled: np.ndarray,
        feature_names: list[str],
    ) -> "OCVPhysicsAligner":
        self.feature_names_ = list(feature_names)
        self.maps_ = {}

        use_physics_ocv = (
            self.src_chemistry in CHEMISTRY_SPECS
            and self.tgt_chemistry in CHEMISTRY_SPECS
            and self.src_chemistry != self.tgt_chemistry
        )

        if use_physics_ocv:
            assert self.src_chemistry is not None and self.tgt_chemistry is not None
            soc_grid = np.linspace(0.0, 1.0, self.n_grid)
            sp_s = CHEMISTRY_SPECS[self.src_chemistry]
            sp_t = CHEMISTRY_SPECS[self.tgt_chemistry]

            v_s_norm = (ocv_curve(self.src_chemistry, soc_grid) - sp_s.v_min) / (sp_s.v_max - sp_s.v_min)
            v_t_norm = (ocv_curve(self.tgt_chemistry, soc_grid) - sp_t.v_min) / (sp_t.v_max - sp_t.v_min)
            # Extend linearly slightly beyond [0, 1] for polarization excursions under heavy pulse loads
            v_t_ext = np.concatenate([[-0.20], np.maximum.accumulate(v_t_norm + np.linspace(0, 1e-6, self.n_grid)), [1.20]])
            v_s_ext = np.concatenate([[-0.20], v_s_norm, [1.20]])

            for col in self.VOLTAGE_COLS:
                if col in self.feature_names_:
                    self.maps_[self.feature_names_.index(col)] = (v_t_ext, v_s_ext)

            # Average OCV slope ratio to rescale dv_norm_dt consistently
            slope_s = float(np.median(np.gradient(v_s_norm, soc_grid)))
            slope_t = float(np.median(np.gradient(v_t_norm, soc_grid)))
            self.dv_scale_ = float(np.clip(slope_s / max(slope_t, 1e-3), 0.3, 3.0))
        else:
            q_grid = np.linspace(0.0, 1.0, 201)
            for col in self.VOLTAGE_COLS:
                if col in self.feature_names_:
                    idx = self.feature_names_.index(col)
                    q_t = np.quantile(X_tgt_unlabeled[:, idx], q_grid)
                    q_s = np.quantile(X_src[:, idx], q_grid)
                    q_t = np.maximum.accumulate(q_t + np.linspace(0.0, 1e-6, 201))
                    self.maps_[idx] = (q_t, q_s)

        self.v_norm_idx_ = self.feature_names_.index("v_norm") if "v_norm" in self.feature_names_ else None
        self.sq_idx_ = self.feature_names_.index("v_norm_sq") if "v_norm_sq" in self.feature_names_ else None
        self.cu_idx_ = self.feature_names_.index("v_norm_cu") if "v_norm_cu" in self.feature_names_ else None
        self.p_norm_idx_ = self.feature_names_.index("p_norm") if "p_norm" in self.feature_names_ else None
        self.c_rate_idx_ = self.feature_names_.index("c_rate") if "c_rate" in self.feature_names_ else None
        self.dv_idx_ = self.feature_names_.index("dv_norm_dt") if "dv_norm_dt" in self.feature_names_ else None
        return self

    def transform_target(self, X_tgt: np.ndarray) -> np.ndarray:
        out = X_tgt.copy()
        for idx, (x_knot, y_knot) in self.maps_.items():
            out[:, idx] = np.interp(out[:, idx], x_knot, y_knot)

        if self.v_norm_idx_ is not None:
            v_w = out[:, self.v_norm_idx_]
            if self.sq_idx_ is not None:
                out[:, self.sq_idx_] = v_w ** 2
            if self.cu_idx_ is not None:
                out[:, self.cu_idx_] = v_w ** 3
            if self.p_norm_idx_ is not None and self.c_rate_idx_ is not None:
                out[:, self.p_norm_idx_] = v_w * out[:, self.c_rate_idx_]
        if self.dv_idx_ is not None:
            out[:, self.dv_idx_] = out[:, self.dv_idx_] * self.dv_scale_
        return out


class CrossChemistryDomainAdapter:
    """Unified unsupervised cross-chemistry feature adapter (Physics OCV Warping + Scaling + Shrinkage CORAL)."""

    def __init__(
        self,
        src_chemistry: str | None = None,
        tgt_chemistry: str | None = None,
        use_ocv_warp: bool = True,
        use_coral: bool = True,
        coral_reg: float = 0.08,
        coral_shrinkage: float = 0.40,
        align_mean: bool = False,
    ) -> None:
        self.src_chemistry = src_chemistry
        self.tgt_chemistry = tgt_chemistry
        self.use_ocv_warp = use_ocv_warp
        self.use_coral = use_coral
        self.coral_reg = coral_reg
        self.coral_shrinkage = coral_shrinkage
        self.align_mean = align_mean
        self.ocv_aligner_: OCVPhysicsAligner | None = None
        self.src_scaler_ = StandardScaler()
        self.coral_: CORALAligner | None = None
        self.feature_names_: list[str] = []

    def fit(
        self,
        X_src_train: np.ndarray,
        X_tgt_unlabeled: np.ndarray,
        feature_names: list[str],
    ) -> "CrossChemistryDomainAdapter":
        self.feature_names_ = list(feature_names)
        X_t = X_tgt_unlabeled.copy()

        if self.use_ocv_warp:
            self.ocv_aligner_ = OCVPhysicsAligner(
                src_chemistry=self.src_chemistry,
                tgt_chemistry=self.tgt_chemistry,
            ).fit(X_src_train, X_t, self.feature_names_)
            X_t = self.ocv_aligner_.transform_target(X_t)

        X_s_scaled = self.src_scaler_.fit_transform(X_src_train)
        X_t_scaled = self.src_scaler_.transform(X_t)

        if self.use_coral:
            self.coral_ = CORALAligner(
                reg=self.coral_reg,
                shrinkage=self.coral_shrinkage,
                align_mean=self.align_mean,
            ).fit(X_s_scaled, X_t_scaled)
        return self

    def transform_source(self, X_src: np.ndarray) -> np.ndarray:
        return self.src_scaler_.transform(X_src)

    def transform_target(self, X_tgt: np.ndarray) -> np.ndarray:
        X_t = X_tgt.copy()
        if self.use_ocv_warp and self.ocv_aligner_ is not None:
            X_t = self.ocv_aligner_.transform_target(X_t)
        X_t_scaled = self.src_scaler_.transform(X_t)
        if self.use_coral and self.coral_ is not None:
            X_t_scaled = self.coral_.transform_target(X_t_scaled)
        return X_t_scaled
