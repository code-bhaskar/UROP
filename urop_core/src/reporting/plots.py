"""Publication-ready visualizations covering Big Data Mining, Cross-Chemistry Alignment, Transfer Learning & Charging."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.charging.adaptive_charging import ChargingTrace
from src.cross_chemistry.evaluation import TaskExperimentBundle
from src.cross_chemistry.transfer_learning import TransferLearningBundle

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
PALETTE = {
    "li_ion": "#1f77b4",
    "na_ion": "#d95f02",
    "raw": "#d62728",
    "phys": "#ff7f0e",
    "aligned": "#2ca02c",
}


def plot_domain_alignment_pca(mining_results: dict, figures_dir: Path) -> Path:
    """Fig 1: PCA domain-shift visualization before (Raw) vs after (Physics + OCV + CORAL) alignment."""
    pca_payload = mining_results["pca_payload"]
    shift_df = pd.DataFrame(mining_results["domain_shift"])

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), dpi=160)
    for row_idx, task in enumerate(["SoC", "SoH"]):
        ax0, ax1 = axes[row_idx, 0], axes[row_idx, 1]
        if task not in pca_payload:
            for ax in (ax0, ax1):
                ax.text(0.5, 0.5, f"{task} cross-chemistry manifold\nunavailable (single-chemistry aging)", ha="center", va="center")
                ax.set_axis_off()
            continue
        p = pca_payload[task]
        s_raw = shift_df[(shift_df["task"] == task) & (shift_df["representation"] == "1_raw_features")].iloc[0]
        s_aln = shift_df[
            (shift_df["task"] == task) & (shift_df["representation"] == "3_physics_ocv_coral_aligned")
        ].iloc[0]

        ax0.scatter(p["raw_li"][:, 0], p["raw_li"][:, 1], s=14, alpha=0.45, c=PALETTE["li_ion"], label="Li-ion")
        ax0.scatter(p["raw_na"][:, 0], p["raw_na"][:, 1], s=14, alpha=0.45, c=PALETTE["na_ion"], label="Na-ion")
        ax0.set_title(
            f"{task} — Raw Features (Before Alignment)\nMMD²={s_raw['mmd2']:.4f}, W1={s_raw['mean_wasserstein_1']:.3f}"
        )
        ax0.set_xlabel("PC 1")
        ax0.set_ylabel("PC 2")
        ax0.legend(frameon=True)

        ax1.scatter(p["aligned_li"][:, 0], p["aligned_li"][:, 1], s=14, alpha=0.45, c=PALETTE["li_ion"], label="Li-ion")
        ax1.scatter(p["aligned_na"][:, 0], p["aligned_na"][:, 1], s=14, alpha=0.45, c=PALETTE["na_ion"], label="Na-ion")
        ax1.set_title(
            f"{task} — Physics + Empirical OCV/CORAL Aligned\nMMD²={s_aln['mmd2']:.4f}, W1={s_aln['mean_wasserstein_1']:.3f}"
        )
        ax1.set_xlabel("PC 1")
        ax1.set_ylabel("PC 2")
        ax1.legend(frameon=True)

    fig.suptitle("Cross-Chemistry Feature Manifold Alignment (PCA Projection: Li-ion vs Na-ion)", fontsize=13, y=0.99)
    fig.tight_layout()
    out = figures_dir / "fig1_domain_alignment_pca.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_within_chemistry_benchmarks(
    soc_bundle: TaskExperimentBundle,
    soh_bundle: TaskExperimentBundle,
    figures_dir: Path,
) -> Path:
    """Fig 2: Within-chemistry baseline RMSE across all 5 model families (LR, RF, XGB, LSTM, GRU)."""
    df = pd.DataFrame(soc_bundle.within_rows + soh_bundle.within_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=160)

    for ax, task in zip(axes, ["SoC", "SoH"]):
        sub = df[df["task"] == task]
        sns.barplot(
            data=sub,
            x="model",
            y="rmse",
            hue="chemistry",
            palette={"li_ion": PALETTE["li_ion"], "na_ion": PALETTE["na_ion"]},
            ax=ax,
        )
        ax.set_title(f"Within-Chemistry {task} Estimation (Held-Out Test Cells)")
        ax.set_xlabel("Model")
        ax.set_ylabel("Test RMSE (Lower is Better)")
        ax.tick_params(axis="x", rotation=18)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    fig.suptitle("Objective 2 & 3: Within-Chemistry Baseline Model Comparison", fontsize=13, y=1.01)
    fig.tight_layout()
    out = figures_dir / "fig2_within_chemistry_benchmarks.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cross_chemistry_ablation(
    soc_bundle: TaskExperimentBundle,
    soh_bundle: TaskExperimentBundle,
    figures_dir: Path,
) -> Path:
    """Fig 3: 3-Tier Cross-Chemistry Ablation (Raw -> Physics-Norm -> Physics+OCV+CORAL) in both directions."""
    df = pd.DataFrame(soc_bundle.cross_rows + soh_bundle.cross_rows)
    tier_labels = {
        "1_raw_zero_shot": "1. Raw Zero-Shot",
        "2_physics_normalized_zero_shot": "2. Physics-Norm",
        "3_physics_ocv_coral_zero_shot": "3. Physics+OCV+CORAL",
    }
    df["Tier"] = df["tier"].map(tier_labels)

    # Focus on best representative models (XGBoost, RandomForest, GRU) for clear visual storytelling
    df_vis = df[df["model"].isin(["XGBoost", "RandomForest", "GRU"])].copy()

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.8), dpi=160)
    combos = [
        ("SoC", "li_ion_to_na_ion", "SoC: Li-ion → Na-ion (Zero-Shot)"),
        ("SoC", "na_ion_to_li_ion", "SoC: Na-ion → Li-ion (Zero-Shot)"),
        ("SoH", "li_ion_to_na_ion", "SoH: Li-ion → Na-ion (Zero-Shot)"),
        ("SoH", "na_ion_to_li_ion", "SoH: Na-ion → Li-ion (Zero-Shot)"),
    ]

    for ax, (task, direction, title) in zip(axes.ravel(), combos):
        sub = df_vis[(df_vis["task"] == task) & (df_vis["direction"] == direction)]
        if sub.empty:
            ax.text(0.5, 0.5, f"{title}\n(requires multi-cycle aging in both chemistries)", ha="center", va="center")
            ax.set_axis_off()
            continue
        sns.barplot(
            data=sub,
            x="model",
            y="rmse",
            hue="Tier",
            palette=[PALETTE["raw"], PALETTE["phys"], PALETTE["aligned"]],
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Cross-Chemistry RMSE (Lower is Better)")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.24)
        ax.legend(title="Alignment Tier", fontsize=8, title_fontsize=8, loc="upper right")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=7.5, padding=2)

    fig.suptitle(
        "Objective 4: Making Cross-Chemistry Applicable — 3-Tier Unsupervised Domain Alignment Ablation",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout()
    out = figures_dir / "fig3_cross_chemistry_ablation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_transfer_learning_curves(
    soc_tl: TransferLearningBundle,
    soh_tl: TransferLearningBundle,
    figures_dir: Path,
) -> Path:
    """Fig 4: Few-shot Transfer Learning RMSE vs Target Data Budget (Li-ion -> Na-ion)."""
    df = pd.DataFrame(soc_tl.results_rows + soh_tl.results_rows)
    df = df[df["direction"] == "li_ion_to_na_ion"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=160)
    styles = {
        "TargetOnly_XGBoost": ("#7f7f7f", "--", "o"),
        "TL_Residual_XGBoost": ("#2ca02c", "-", "s"),
        "TL_WarmStart_XGBoost": ("#1f77b4", "-", "^"),
        "TargetOnly_GRU": ("#d62728", ":", "x"),
        "TL_TwoPhase_GRU": ("#9467bd", "-", "D"),
    }

    for ax, task in zip(axes, ["SoC", "SoH"]):
        sub = df[df["task"] == task]
        if sub.empty:
            ax.text(0.5, 0.5, f"{task} Transfer Learning unavailable\n(requires multi-cycle aging in both chemistries)", ha="center", va="center")
            ax.set_axis_off()
            continue
        # Horizontal line for Zero-Shot Aligned XGBoost
        zs = sub[sub["method"] == "ZeroShot_Aligned_XGBoost"]
        if not zs.empty:
            zs_val = float(zs.iloc[0]["rmse"])
            ax.axhline(zs_val, color="#ff7f0e", linestyle="-.", linewidth=1.6, label=f"ZeroShot Aligned XGB ({zs_val:.3f})")

        for method, (color, ls, marker) in styles.items():
            msub = sub[sub["method"] == method].sort_values("target_fraction")
            if msub.empty:
                continue
            ax.plot(
                msub["target_fraction"] * 100.0,
                msub["rmse"],
                color=color,
                linestyle=ls,
                marker=marker,
                linewidth=2.0,
                markersize=6,
                label=method,
            )

        ax.set_title(f"{task} Transfer Learning: Li-ion → Na-ion (Held-Out Na-ion Test Cells)")
        ax.set_xlabel("Target Na-ion Labeled Training Budget (%)")
        ax.set_ylabel("Test RMSE (Lower is Better)")
        ax.legend(fontsize=8, frameon=True)

    fig.suptitle("Objective 4: Low-Data Transfer Learning vs Target-Only & Zero-Shot Baselines", fontsize=13, y=1.01)
    fig.tight_layout()
    out = figures_dir / "fig4_transfer_learning_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cross_chemistry_parity(
    soc_bundle: TaskExperimentBundle,
    soh_bundle: TaskExperimentBundle,
    figures_dir: Path,
) -> Path:
    """Fig 5: Parity plots showing Raw Zero-Shot failure vs Physics+OCV+CORAL recovery on Li-ion -> Na-ion."""
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0), dpi=160)

    for ax, bundle, label, lims in [
        (axes[0], soc_bundle, "SoC", (0.0, 1.02)),
        (axes[1], soh_bundle, "SoH", (0.65, 1.03)),
    ]:
        if "cross_li_ion_to_na_ion" not in bundle.sample_predictions:
            ax.text(0.5, 0.5, f"{label} Cross-Chemistry Parity unavailable\n(single-chemistry aging data)", ha="center", va="center")
            ax.set_axis_off()
            continue
        p = bundle.sample_predictions["cross_li_ion_to_na_ion"]
        yt = p["y_true"]
        idx = np.linspace(0, len(yt) - 1, min(600, len(yt)), dtype=int)
        ax.scatter(yt[idx], p["y_pred_raw"][idx], s=14, alpha=0.35, c=PALETTE["raw"], label="Raw Zero-Shot (XGB)")
        ax.scatter(
            yt[idx],
            p["y_pred_aligned"][idx],
            s=14,
            alpha=0.50,
            c=PALETTE["aligned"],
            label="Physics+OCV+CORAL Zero-Shot (XGB)",
        )
        ax.plot(lims, lims, "k--", linewidth=1.3, label="Ideal Parity (y=x)")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(f"Li-ion → Na-ion {label} Zero-Shot Parity")
        ax.set_xlabel(f"True Na-ion {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.legend(fontsize=8.5, frameon=True)

    fig.suptitle("Cross-Chemistry Parity Recovery via Unsupervised Physics + OCV + CORAL Alignment", fontsize=13, y=1.01)
    fig.tight_layout()
    out = figures_dir / "fig5_cross_chemistry_parity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_adaptive_charging_comparison(
    charging_rows: list[dict],
    traces: dict[str, ChargingTrace],
    figures_dir: Path,
) -> Path:
    """Fig 6: Adaptive ML Charging vs CC-CV Baseline on Aged Li-ion & Aged Na-ion Cells."""
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.2), dpi=160)

    for row_idx, (chem, key_prefix, title_prefix) in enumerate(
        [
            ("li_ion", "li_ion_soh80", "Aged Li-ion (SoH=80%)"),
            ("na_ion", "na_ion_soh80", "Aged Na-ion (SoH=80%, TL-Estimated SoC)"),
        ]
    ):
        tr_cc = traces[f"{key_prefix}_CC-CV"]
        tr_ad = traces[f"{key_prefix}_Adaptive_ML"]

        # Col 0: C-rate & Voltage
        ax0 = axes[row_idx, 0]
        ax0.plot(tr_cc.time_min, tr_cc.current_c_rate, "--", color="#d62728", linewidth=1.8, label="CC-CV C-rate")
        ax0.plot(tr_ad.time_min, tr_ad.current_c_rate, "-", color="#2ca02c", linewidth=2.0, label="Adaptive ML C-rate")
        ax0.set_title(f"{title_prefix}: Charging Current")
        ax0.set_xlabel("Time (min)")
        ax0.set_ylabel("Charging Rate (C)")
        ax0.legend(fontsize=8)

        # Col 1: Cell Temperature
        ax1 = axes[row_idx, 1]
        ax1.plot(tr_cc.time_min, tr_cc.temp_c, "--", color="#d62728", linewidth=1.8, label="CC-CV Temp")
        ax1.plot(tr_ad.time_min, tr_ad.temp_c, "-", color="#2ca02c", linewidth=2.0, label="Adaptive ML Temp")
        ax1.axhline(45.0, color="black", linestyle=":", linewidth=1.0, label="Safety Limit (45 °C)")
        ax1.set_title(f"{title_prefix}: Thermal Trajectory")
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("Cell Temp (°C)")
        ax1.legend(fontsize=8)

        # Col 2: SoC True vs ML/TL Observer
        ax2 = axes[row_idx, 2]
        ax2.plot(tr_ad.time_min, tr_ad.soc_true * 100.0, "-", color="#1f77b4", linewidth=2.0, label="True SoC")
        ax2.plot(tr_ad.time_min, tr_ad.soc_est * 100.0, "--", color="#ff7f0e", linewidth=1.8, label="ML/TL Estimated SoC")
        ax2.set_title(
            f"{title_prefix}: Closed-Loop SoC\n(Observer MAE={tr_ad.metrics['soc_observer_mae'] * 100:.2f}%)"
        )
        ax2.set_xlabel("Time (min)")
        ax2.set_ylabel("State of Charge (%)")
        ax2.legend(fontsize=8)

    fig.suptitle(
        "Objective 5: Simulated ML-Informed Adaptive Charging vs Conventional CC-CV Baseline",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout()
    out = figures_dir / "fig6_adaptive_vs_cccv_charging.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
