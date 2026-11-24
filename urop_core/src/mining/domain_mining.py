"""Data Mining & Domain-Shift Analytics: K-Means regimes, Mutual Information, PCA, MMD, Wasserstein, PSI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import PipelineConfig
from src.cross_chemistry.alignment import CrossChemistryDomainAdapter, estimate_empirical_ocv_curve
from src.mining.feature_engineering import (
    PHYSICS_SOC_FEATURES,
    PHYSICS_SOH_FEATURES,
    RAW_SOC_FEATURES,
    RAW_SOH_FEATURES,
)


def compute_rbf_mmd(X: np.ndarray, Y: np.ndarray, max_samples: int = 800, seed: int = 42) -> float:
    """Unbiased RBF-kernel Maximum Mean Discrepancy (MMD^2) measuring distribution shift."""
    rng = np.random.default_rng(seed)
    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        Y = Y[rng.choice(len(Y), max_samples, replace=False)]

    # Median heuristic bandwidth on pooled samples
    pooled = np.vstack([X, Y])
    d_all = cdist(pooled[:400], pooled[:400], metric="sqeuclidean")
    med = float(np.median(d_all[d_all > 0])) if np.any(d_all > 0) else 1.0
    gamma = 1.0 / max(med, 1e-6)

    k_xx = np.exp(-gamma * cdist(X, X, metric="sqeuclidean"))
    k_yy = np.exp(-gamma * cdist(Y, Y, metric="sqeuclidean"))
    k_xy = np.exp(-gamma * cdist(X, Y, metric="sqeuclidean"))
    mmd2 = float(np.mean(k_xx) + np.mean(k_yy) - 2.0 * np.mean(k_xy))
    return max(0.0, mmd2)


def compute_psi(x_src: np.ndarray, x_tgt: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index (PSI) for a 1D feature."""
    edges = np.quantile(x_src, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    p_s = np.histogram(x_src, bins=edges)[0] / max(len(x_src), 1)
    p_t = np.histogram(x_tgt, bins=edges)[0] / max(len(x_tgt), 1)
    eps = 1e-4
    p_s = np.clip(p_s, eps, None)
    p_t = np.clip(p_t, eps, None)
    return float(np.sum((p_t - p_s) * np.log(p_t / p_s)))


def quantify_domain_shift(
    X_src: np.ndarray,
    X_tgt: np.ndarray,
    seed: int = 42,
    scale_jointly: bool = True,
) -> dict[str, float]:
    """Compute standardized MMD^2, Mean Wasserstein-1, and Mean PSI between two chemistry distributions."""
    if scale_jointly:
        scaler = StandardScaler().fit(np.vstack([X_src, X_tgt]))
        Xs = scaler.transform(X_src)
        Xt = scaler.transform(X_tgt)
    else:
        Xs, Xt = X_src, X_tgt

    mmd2 = compute_rbf_mmd(Xs, Xt, seed=seed)
    w_dists = [float(wasserstein_distance(Xs[:, j], Xt[:, j])) for j in range(Xs.shape[1])]
    psis = [compute_psi(Xs[:, j], Xt[:, j]) for j in range(Xs.shape[1])]
    return {
        "mmd2": round(mmd2, 5),
        "mean_wasserstein_1": round(float(np.mean(w_dists)), 4),
        "mean_psi": round(float(np.mean(psis)), 4),
    }


def run_data_mining_suite(
    cfg: PipelineConfig,
    soc_df: pl.DataFrame,
    soh_df: pl.DataFrame,
    tables_dir: Path,
) -> dict:
    """Run full Data Mining suite: K-Means operating regimes, MI rankings, PCA & Domain-Shift analysis."""
    rng = np.random.default_rng(cfg.seed)

    # 1. Unsupervised Operating-Regime Clustering (K-Means on physics-normalized electrical/thermal space)
    cluster_cols = ["v_norm", "c_rate", "temp_rise_norm", "v_norm_roll_std"]
    X_cl_full = soc_df.select(cluster_cols).to_numpy()
    sample_idx = rng.choice(len(X_cl_full), size=min(4000, len(X_cl_full)), replace=False)
    scaler_cl = StandardScaler()
    X_cl_scaled = scaler_cl.fit_transform(X_cl_full[sample_idx])

    kmeans = KMeans(n_clusters=4, random_state=cfg.seed, n_init=10)
    labels_sample = kmeans.fit_predict(X_cl_scaled)
    sil = float(silhouette_score(X_cl_scaled, labels_sample))

    all_labels = kmeans.predict(scaler_cl.transform(X_cl_full))
    soc_clustered = soc_df.with_columns(pl.Series("operating_regime", all_labels))
    regime_summary = (
        soc_clustered.group_by(["operating_regime", "chemistry"])
        .agg(
            pl.len().alias("n_samples"),
            pl.col("v_norm").mean().round(3).alias("mean_v_norm"),
            pl.col("c_rate").mean().round(3).alias("mean_c_rate"),
            pl.col("temp_rise_c").mean().round(2).alias("mean_temp_rise_c"),
            pl.col("soc").mean().round(3).alias("mean_soc"),
        )
        .sort(["operating_regime", "chemistry"])
    )
    regime_summary.write_csv(tables_dir / "mining_operating_regimes.csv")

    # 2. Mutual Information Feature Importance Ranking (per available chemistry, SoC and SoH)
    mi_rows: list[dict] = []
    for task, df, feat_cols, target_col in [
        ("SoC", soc_df, PHYSICS_SOC_FEATURES, "soc"),
        ("SoH", soh_df, PHYSICS_SOH_FEATURES, "soh"),
    ]:
        if df.height == 0:
            continue
        for chem in ["li_ion", "na_ion"]:
            sub = df.filter(pl.col("chemistry") == chem)
            if sub.height < 20:
                continue
            X = sub.select(feat_cols).to_numpy()
            y = sub[target_col].to_numpy()
            if len(X) > 3500:
                idx = rng.choice(len(X), 3500, replace=False)
                X, y = X[idx], y[idx]
            mi = mutual_info_regression(X, y, random_state=cfg.seed)
            for fname, score in sorted(zip(feat_cols, mi), key=lambda kv: kv[1], reverse=True):
                mi_rows.append(
                    {
                        "task": task,
                        "chemistry": chem,
                        "feature": fname,
                        "mutual_information": round(float(score), 4),
                    }
                )
    mi_df = pl.DataFrame(mi_rows)
    mi_df.write_csv(tables_dir / "mining_mutual_information.csv")

    # 3. Quantitative Domain-Shift Analysis: Raw vs Physics-Normalized vs Physics+EmpiricalOCV+CORAL
    shift_rows: list[dict] = []
    pca_payload: dict[str, dict] = {}

    for task, df, raw_cols, phys_cols, use_ocv in [
        ("SoC", soc_df, RAW_SOC_FEATURES, PHYSICS_SOC_FEATURES, True),
        ("SoH", soh_df, RAW_SOH_FEATURES, PHYSICS_SOH_FEATURES, False),
    ]:
        if df.height == 0:
            continue
        li_sub = df.filter(pl.col("chemistry") == "li_ion")
        if task == "SoH" and "data_provenance" in df.columns:
            li_ord = li_sub.filter(pl.col("data_provenance") == "ordered_per_cell")
            if li_ord.height >= 20:
                li_sub = li_ord
        na_sub = df.filter(pl.col("chemistry") == "na_ion")
        if li_sub.height < 20 or na_sub.height < 20:
            continue

        X_li_raw = li_sub.select(raw_cols).to_numpy()
        X_na_raw = na_sub.select(raw_cols).to_numpy()
        X_li_phys = li_sub.select(phys_cols).to_numpy()
        X_na_phys = na_sub.select(phys_cols).to_numpy()

        src_ocv = estimate_empirical_ocv_curve(li_sub, "li_ion") if use_ocv else None
        tgt_ocv = estimate_empirical_ocv_curve(na_sub, "na_ion") if use_ocv else None

        adapter = CrossChemistryDomainAdapter(
            src_ocv_curve=src_ocv,
            tgt_ocv_curve=tgt_ocv,
            use_ocv_warp=use_ocv,
            use_coral=True,
            coral_reg=0.08,
            coral_shrinkage=0.55 if use_ocv else 0.35,
            align_mean=False,
        )
        adapter.fit(X_li_phys, X_na_phys, phys_cols)
        X_li_aligned = adapter.transform_source(X_li_phys)
        X_na_aligned = adapter.transform_target(X_na_phys)

        m_raw = quantify_domain_shift(X_li_raw, X_na_raw, seed=cfg.seed, scale_jointly=True)
        m_phys = quantify_domain_shift(X_li_phys, X_na_phys, seed=cfg.seed, scale_jointly=True)
        m_aligned = quantify_domain_shift(X_li_aligned, X_na_aligned, seed=cfg.seed, scale_jointly=False)

        for stage, metrics in [
            ("1_raw_features", m_raw),
            ("2_physics_normalized", m_phys),
            ("3_physics_ocv_coral_aligned", m_aligned),
        ]:
            shift_rows.append({"task": task, "representation": stage, **metrics})

        # PCA 3D projections for 2D/3D visualization (subsample for crisp plotting & manifold export)
        n_vis = min(500, len(X_li_raw), len(X_na_raw))
        i_li = rng.choice(len(X_li_raw), n_vis, replace=False)
        i_na = rng.choice(len(X_na_raw), n_vis, replace=False)

        s_raw = StandardScaler().fit_transform(np.vstack([X_li_raw[i_li], X_na_raw[i_na]]))
        pca_raw_model = PCA(n_components=3, random_state=cfg.seed)
        pca_raw = pca_raw_model.fit_transform(s_raw)

        s_aln = np.vstack([X_li_aligned[i_li], X_na_aligned[i_na]])
        pca_aln_model = PCA(n_components=3, random_state=cfg.seed)
        pca_aln = pca_aln_model.fit_transform(s_aln)

        pca_payload[task] = {
            "raw_li": pca_raw[:n_vis],
            "raw_na": pca_raw[n_vis:],
            "aligned_li": pca_aln[:n_vis],
            "aligned_na": pca_aln[n_vis:],
            "raw_explained_var": [round(float(v), 4) for v in pca_raw_model.explained_variance_ratio_],
            "aligned_explained_var": [round(float(v), 4) for v in pca_aln_model.explained_variance_ratio_],
        }

    shift_df = pl.DataFrame(shift_rows)
    shift_df.write_csv(tables_dir / "mining_domain_shift_reduction.csv")

    return {
        "kmeans_silhouette": round(sil, 4),
        "operating_regimes": regime_summary.to_dicts(),
        "domain_shift": shift_rows,
        "pca_payload": pca_payload,
        "top_mi_soc_li": [r for r in mi_rows if r["task"] == "SoC" and r["chemistry"] == "li_ion"][:5],
        "top_mi_soc_na": [r for r in mi_rows if r["task"] == "SoC" and r["chemistry"] == "na_ion"][:5],
        "top_mi_soh_li": [r for r in mi_rows if r["task"] == "SoH" and r["chemistry"] == "li_ion"][:5],
        "top_mi_soh_na": [r for r in mi_rows if r["task"] == "SoH" and r["chemistry"] == "na_ion"][:5],
    }
