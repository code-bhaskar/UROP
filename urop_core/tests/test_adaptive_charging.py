"""Tests for Simulated Adaptive Charging Strategy vs Conventional CC-CV Baseline."""

from __future__ import annotations

import numpy as np

from src.charging.adaptive_charging import run_adaptive_charging_benchmark, simulate_charging_episode


def test_adaptive_charging_safety_and_degradation_reduction() -> None:
    # Use a simple smooth monotonic predictor for deterministic unit test
    dummy_pred = lambda X: np.clip(X[:, 11], 0.0, 1.0)
    rows, traces = run_adaptive_charging_benchmark(
        li_soc_predictor=dummy_pred,
        na_tl_soc_predictor=dummy_pred,
    )
    assert len(rows) == 8
    assert all(r["safety_limits_respected"] for r in rows)

    by_key = {(r["chemistry"], r["soh_true"], r["strategy"]): r for r in rows}
    for chem in ("li_ion", "na_ion"):
        for soh in (0.98, 0.80):
            cc = by_key[(chem, soh, "CC-CV")]
            ad = by_key[(chem, soh, "Adaptive_ML")]
            assert ad["peak_temp_rise_c"] <= cc["peak_temp_rise_c"] + 0.05
            assert ad["energy_efficiency_pct"] >= cc["energy_efficiency_pct"]
            assert ad["estimated_degradation_pct"] < cc["estimated_degradation_pct"]


def test_charging_episode_reaches_target_soc() -> None:
    tr = simulate_charging_episode(
        chemistry="na_ion",
        strategy="Adaptive_ML",
        soh_true=0.85,
        soh_est=0.85,
        soc_predictor=None,
        soc_start=0.10,
        soc_target=0.90,
    )
    assert tr.soc_true[-1] >= 0.90
    assert tr.metrics["safety_limits_respected"]
