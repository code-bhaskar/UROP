"""Cross-Chemistry Domain Alignment (Li-ion <-> Na-ion) — Zero-Circularity, Data-Driven OCV & CORAL.

IMPORTANT SCIENTIFIC GUARANTEE:
This module has ZERO imports from `physics_simulator.py`. All OCV reference curves are
estimated strictly empirically from training cells only (using pulse-rest low-current points
where available, or IR-compensated voltages binned into 10% SoC intervals + isotonic monotonic
regression), never from any analytic simulator formula.

Implements three complementary physics + statistical domain adaptation layers:
1. Physics-Normalized Chemistry-Invariant Representation (computed in Gold layer).
2. Data-Driven Empirical OCV-to-OCV Warping: maps target normalized voltage onto source
   normalized voltage via empirically estimated 1D static OCV(SoC) curves fitted strictly
   on training cells (mirroring a 10%-SoC C/20 or pulse-rest datasheet lookup). Falls back
   to purely unsupervised monotonic quantile warping if no empirical OCV curves are passed.
3. Shrinkage CORAL Covariance Alignment (Sun et al., AAAI 2016) with optional mean
   preservation so physical zero/BoL reference anchors are never distorted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class EmpiricalOCVCurve:
    """Data-driven 1D monotonic normalized OCV(SoC) curve estimated strictly from training cells."""

    chemistry: str
    soc_grid: np.ndarray
    v_norm_grid: np.ndarray
    estimation_method: str  # "rest_points" | "ir_compensated"
    n_samples_used: int

    def eval_v_norm(self, soc: np.ndarray) -> np.ndarray:
        return np.interp(np.clip(soc, 0.0, 1.0), self.soc_grid, self.v_norm_grid)


def estimate_empirical_ocv_curve(
    train_df: pl.DataFrame,
    chemistry: str,
    n_knots: int = 11,
    rest_c_rate_thresh: float = 0.15,
) -> EmpiricalOCVCurve:
    """Estimate a monotonic 1D normalized OCV(SoC) curve strictly from training cells.

    Strategy (matching real battery characterization protocols):
    1. Prefer near-equilibrium rest samples (|c_rate| <= 0.15, e.g. Wang et al. Na-ion 10% SoC
       pulse-rest plateaus) if they cover >= 6 distinct SoC bins with >= 40 total samples.
    2. Otherwise use IR-compensated normalized voltage (`v_ir_comp_norm`) across training samples
       near room temperature (15-35 C), binned into uniform 10% SoC intervals and smoothed via
       monotonic IsotonicRegression.
    """
    sub = train_df.filter(pl.col("chemistry") == chemistry)
    if sub.height < 20:
        raise ValueError(f"Insufficient training rows ({sub.height}) to estimate OCV for {chemistry}")

    soc_all = sub["soc"].to_numpy()
    c_all = np.abs(sub["c_rate"].to_numpy())
    v_norm_all = sub["v_norm"].to_numpy()
    v_ir_all = sub["v_ir_comp_norm"].to_numpy() if "v_ir_comp_norm" in sub.columns else v_norm_all

    # Check if sufficient near-rest equilibrium points genuinely span both low & high SoC
    # (prevents CC-CV taper tails or post-cutoff rebounds from biasing rest-only OCV)
    rest_mask = c_all <= rest_c_rate_thresh
    bin_edges = np.linspace(0.0, 1.0, n_knots)
    bin_centers = bin_edges.copy()

    use_rest = False
    if int(np.sum(rest_mask)) >= 40:
        s_rest = soc_all[rest_mask]
        v_rest = v_norm_all[rest_mask]
        rest_bins = np.digitize(s_rest, bin_edges) - 1
        low_soc_rest = v_rest[(s_rest >= 0.05) & (s_rest <= 0.25)]
        high_soc_rest = v_rest[(s_rest >= 0.75) & (s_rest <= 0.95)]
        if (
            len(np.unique(rest_bins)) >= max(7, n_knots - 3)
            and len(low_soc_rest) >= 15
            and len(high_soc_rest) >= 15
            and float(np.median(low_soc_rest)) < 0.48
            and float(np.median(high_soc_rest)) < 0.97
        ):
            use_rest = True

    if use_rest:
        s_use = soc_all[rest_mask]
        v_use = v_norm_all[rest_mask]
        method = "rest_points"
    else:
        # Filter mild thermal band if enough rows exist, else use all training rows
        if "temp_c" in sub.columns:
            t_all = sub["temp_c"].to_numpy()
            t_mask = (t_all >= 12.0) & (t_all <= 38.0)
            if int(np.sum(t_mask)) >= 50:
                s_use = soc_all[t_mask]
                v_use = v_ir_all[t_mask]
            else:
                s_use = soc_all
                v_use = v_ir_all
        else:
            s_use = soc_all
            v_use = v_ir_all
        method = "ir_compensated"

    # Estimate bin medians at each 10% SoC knot (matching 10% pulse characterization steps)
    half_step = 0.5 / max(n_knots - 1, 1)
    knot_s: list[float] = []
    knot_v: list[float] = []
    for k_soc in bin_centers:
        lo = max(0.0, k_soc - half_step)
        hi = min(1.0, k_soc + half_step)
        m = (s_use >= lo) & (s_use <= hi)
        if int(np.sum(m)) >= 3:
            knot_s.append(float(k_soc))
            knot_v.append(float(np.median(v_use[m])))

    if len(knot_s) < 3:
        # Fallback to quantile bins across observed SoC range
        qs = np.linspace(0.02, 0.98, n_knots)
        order = np.argsort(s_use)
        s_sorted, v_sorted = s_use[order], v_use[order]
        chunks = np.array_split(np.arange(len(s_sorted)), n_knots)
        knot_s = [float(np.median(s_sorted[ch])) for ch in chunks if len(ch) > 0]
        knot_v = [float(np.median(v_sorted[ch])) for ch in chunks if len(ch) > 0]

    # Enforce physical monotonicity via IsotonicRegression
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    v_mono = iso.fit_transform(np.asarray(knot_s), np.asarray(knot_v))

    # Interpolate onto dense [0, 1] grid with gentle linear boundary extrapolation
    dense_soc = np.linspace(0.0, 1.0, 201)
    dense_v = np.interp(dense_soc, np.asarray(knot_s), v_mono)
    # Anchor exact endpoints linearly if outermost knots didn't reach 0.0 / 1.0
    if knot_s[0] > 0.02:
        slope_lo = (v_mono[1] - v_mono[0]) / max(knot_s[1] - knot_s[0], 1e-3)
        lo_mask = dense_soc < knot_s[0]
        dense_v[lo_mask] = np.clip(v_mono[0] - slope_lo * (knot_s[0] - dense_soc[lo_mask]), 0.0, 1.0)
    if knot_s[-1] < 0.98:
        slope_hi = (v_mono[-1] - v_mono[-2]) / max(knot_s[-1] - knot_s[-2], 1e-3)
        hi_mask = dense_soc > knot_s[-1]
        dense_v[hi_mask] = np.clip(v_mono[-1] + slope_hi * (dense_soc[hi_mask] - knot_s[-1]), 0.0, 1.0)

    dense_v = np.maximum.accumulate(dense_v + np.linspace(0.0, 1e-5, len(dense_v)))
    return EmpiricalOCVCurve(
        chemistry=chemistry,
        soc_grid=dense_soc,
        v_norm_grid=np.clip(dense_v, 0.0, 1.0),
        estimation_method=method,
        n_samples_used=int(len(s_use)),
    )


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
        self.transform_t2s_ = (1.0 - self.shrinkage) * np.eye(d) + self.shrinkage * full_map
        return self

    def transform_target(self, X_tgt: np.ndarray) -> np.ndarray:
        assert self.mu_src_ is not None and self.mu_tgt_ is not None and self.transform_t2s_ is not None
        if self.align_mean:
            return (X_tgt - self.mu_tgt_) @ self.transform_t2s_ + self.mu_src_
        return (X_tgt - self.mu_tgt_) @ self.transform_t2s_ + self.mu_tgt_


class OCVPhysicsAligner:
    """Data-Driven Empirical OCV-to-OCV Warping for voltage-derived features (zero simulator imports).

    When empirical source & target 1D OCV curves (fitted strictly on training cells) are supplied,
    warps target normalized voltage v_tgt -> equilibrium SoC -> source normalized voltage v_src.
    When OCV curves are omitted, falls back cleanly to purely unsupervised monotonic quantile warping.
    """

    VOLTAGE_COLS = (
        "v_norm",
        "v_norm_roll_mean",
        "v_ir_comp_norm",
        "v_ir_comp_roll_mean",
    )

    def __init__(
        self,
        src_ocv_curve: EmpiricalOCVCurve | None = None,
        tgt_ocv_curve: EmpiricalOCVCurve | None = None,
        n_grid: int = 201,
    ) -> None:
        self.src_ocv_curve = src_ocv_curve
        self.tgt_ocv_curve = tgt_ocv_curve
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
        self.mode_: str = "quantile_unsupervised"

    def fit(
        self,
        X_src: np.ndarray,
        X_tgt_unlabeled: np.ndarray,
        feature_names: list[str],
    ) -> "OCVPhysicsAligner":
        self.feature_names_ = list(feature_names)
        self.maps_ = {}

        if self.src_ocv_curve is not None and self.tgt_ocv_curve is not None:
            soc_grid = np.linspace(0.0, 1.0, self.n_grid)
            v_s_norm = self.src_ocv_curve.eval_v_norm(soc_grid)
            v_t_norm = self.tgt_ocv_curve.eval_v_norm(soc_grid)
            v_t_mono = np.maximum.accumulate(v_t_norm + np.linspace(0.0, 1e-5, self.n_grid))
            v_t_ext = np.concatenate([[-0.20], v_t_mono, [1.20]])
            v_s_ext = np.concatenate([[-0.20], v_s_norm, [1.20]])

            for col in self.VOLTAGE_COLS:
                if col in self.feature_names_:
                    self.maps_[self.feature_names_.index(col)] = (v_t_ext, v_s_ext)

            slope_s = float(np.median(np.gradient(v_s_norm, soc_grid)))
            slope_t = float(np.median(np.gradient(v_t_norm, soc_grid)))
            self.dv_scale_ = float(np.clip(slope_s / max(slope_t, 1e-3), 0.3, 3.0))
            self.mode_ = f"empirical_ocv({self.src_ocv_curve.estimation_method}->{self.tgt_ocv_curve.estimation_method})"
        else:
            q_grid = np.linspace(0.0, 1.0, 201)
            for col in self.VOLTAGE_COLS:
                if col in self.feature_names_:
                    idx = self.feature_names_.index(col)
                    q_t = np.quantile(X_tgt_unlabeled[:, idx], q_grid)
                    q_s = np.quantile(X_src[:, idx], q_grid)
                    q_t = np.maximum.accumulate(q_t + np.linspace(0.0, 1e-6, 201))
                    self.maps_[idx] = (q_t, q_s)
            self.mode_ = "quantile_unsupervised"

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
    """Unified cross-chemistry feature adapter (Data-Driven Empirical OCV Warping + Scaling + Shrinkage CORAL)."""

    def __init__(
        self,
        src_ocv_curve: EmpiricalOCVCurve | None = None,
        tgt_ocv_curve: EmpiricalOCVCurve | None = None,
        use_ocv_warp: bool = True,
        use_coral: bool = True,
        coral_reg: float = 0.08,
        coral_shrinkage: float = 0.40,
        align_mean: bool = False,
    ) -> None:
        self.src_ocv_curve = src_ocv_curve
        self.tgt_ocv_curve = tgt_ocv_curve
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
                src_ocv_curve=self.src_ocv_curve,
                tgt_ocv_curve=self.tgt_ocv_curve,
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
