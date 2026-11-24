import React, { useEffect, useMemo, useState } from "react";
import { ArrowRightLeft, Layers, Play, Sparkles, TrendingDown } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { CrossRow, DomainShiftRow, OverviewSummary, PcaTaskCloud, TransferRow } from "../types";
import { CrossChemManifold3D } from "./CrossChemManifold3D";

interface CrossChemistryViewProps {
  summary: OverviewSummary;
  manifold: Record<"SoC" | "SoH", PcaTaskCloud>;
  domainShift: DomainShiftRow[];
  crossRows: CrossRow[];
  transferRows: TransferRow[];
}

export const CrossChemistryView: React.FC<CrossChemistryViewProps> = ({
  summary,
  manifold,
  domainShift,
  crossRows,
  transferRows,
}) => {
  const [task, setTask] = useState<"SoC" | "SoH">("SoC");
  const [direction, setDirection] = useState<"li_ion_to_na_ion" | "na_ion_to_li_ion">("li_ion_to_na_ion");
  const [morphFactor, setMorphFactor] = useState<number>(1.0);
  const [animating, setAnimating] = useState<boolean>(false);

  const naSohUnits = summary.unit_counts?.SoH?.na_ion ?? 2;
  const isSohExploratory =
    summary.evaluation_modes?.SoH_cross_chemistry === "exploratory_pooled" || naSohUnits < 3;

  useEffect(() => {
    if (!animating) return;
    let t = 0;
    let raf = 0;
    const tick = () => {
      t += 0.018;
      if (t >= 1) {
        setMorphFactor(1);
        setAnimating(false);
        return;
      }
      setMorphFactor(t);
      raf = requestAnimationFrame(tick);
    };
    setMorphFactor(0);
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [animating]);

  const shiftStats = useMemo(() => {
    const raw = domainShift.find((d) => d.task === task && d.representation === "1_raw_features");
    const aln = domainShift.find((d) => d.task === task && d.representation === "3_physics_ocv_coral_aligned");
    const lerp = (a: number, b: number) => a * (1 - morphFactor) + b * morphFactor;
    return {
      raw,
      aln,
      mmd2: raw && aln ? lerp(raw.mmd2, aln.mmd2) : 0,
      w1: raw && aln ? lerp(raw.mean_wasserstein_1, aln.mean_wasserstein_1) : 0,
      psi: raw && aln ? lerp(raw.mean_psi, aln.mean_psi) : 0,
    };
  }, [domainShift, task, morphFactor]);

  // Build 3-tier grouped bar chart data (tabular only on exploratory pooled SoH)
  const ablationData = useMemo(() => {
    const models =
      task === "SoH" && isSohExploratory
        ? ["LinearRegression", "RandomForest", "XGBoost"]
        : ["RandomForest", "XGBoost", "LSTM", "GRU"];
    return models.map((m) => {
      const find = (tier: string) =>
        crossRows.find((r) => r.task === task && r.direction === direction && r.model === m && r.tier === tier);
      return {
        model: m,
        Raw: find("1_raw_zero_shot")?.rmse ?? 0,
        PhysicsNorm: find("2_physics_normalized_zero_shot")?.rmse ?? 0,
        Aligned: find("3_physics_ocv_coral_zero_shot")?.rmse ?? 0,
        AlignedR2: find("3_physics_ocv_coral_zero_shot")?.r2 ?? 0,
      };
    });
  }, [crossRows, task, direction, isSohExploratory]);

  // Build Few-Shot Transfer Learning chart data
  const tlChartData = useMemo(() => {
    const fracs = [0.05, 0.1, 0.25, 0.5];
    const zs = transferRows.find(
      (r) => r.task === task && r.direction === direction && r.method === "ZeroShot_Aligned_XGBoost"
    )?.rmse;
    return fracs.map((f) => {
      const get = (method: string) =>
        transferRows.find(
          (r) => r.task === task && r.direction === direction && Math.abs(r.target_fraction - f) < 1e-4 && r.method === method
        )?.rmse;
      return {
        budget: `${Math.round(f * 100)}%`,
        ZeroShot: zs,
        TargetOnly_XGB: get("TargetOnly_XGBoost"),
        TL_Residual_XGB: get("TL_Residual_XGBoost"),
        TL_WarmStart_XGB: get("TL_WarmStart_XGBoost"),
        TargetOnly_GRU: get("TargetOnly_GRU"),
        TL_TwoPhase_GRU: get("TL_TwoPhase_GRU"),
      };
    });
  }, [transferRows, task, direction]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      {/* Top Controls */}
      <div className="panel" style={{ padding: "14px 18px" }}>
        <div className="panel-header" style={{ marginBottom: 0 }}>
          <div>
            <h2 className="panel-title">
              <ArrowRightLeft size={18} className="text-cyan" />
              Objective 4: Making Cross-Chemistry Applicable (Li-ion ↔ Na-ion Alignment &amp; Transfer Learning)
            </h2>
            <p className="panel-subtitle">
              Interactive 3D PCA manifold morph + 3-tier unsupervised alignment ablation + few-shot transfer learning curves.
            </p>
          </div>

          <div className="btn-group">
            <button className={`btn ${task === "SoC" ? "active" : ""}`} onClick={() => setTask("SoC")}>
              Task: SoC
            </button>
            {manifold.SoH && (
              <button className={`btn ${task === "SoH" ? "active" : ""}`} onClick={() => setTask("SoH")}>
                Task: SoH{isSohExploratory ? " (Exploratory)" : ""}
              </button>
            )}
            <button
              className={`btn amber ${direction === "li_ion_to_na_ion" ? "active" : ""}`}
              onClick={() => setDirection("li_ion_to_na_ion")}
            >
              Li-ion → Na-ion
            </button>
            <button
              className={`btn ${direction === "na_ion_to_li_ion" ? "active" : ""}`}
              onClick={() => setDirection("na_ion_to_li_ion")}
            >
              Na-ion → Li-ion
            </button>
          </div>
        </div>
      </div>

      {task === "SoH" && isSohExploratory && (
        <div
          className="disclaimer-banner"
          style={{
            borderColor: "rgba(245, 158, 11, 0.55)",
            background: "rgba(245, 158, 11, 0.10)",
            fontSize: 12.5,
          }}
        >
          <strong>Exploratory · Pooled Workbooks Caveat:</strong> Na-ion SoH contains only{" "}
          <strong>{naSohUnits} pooled Mendeley NFM workbooks</strong> (25 °C vs −15 °C, temperature-confounded) without
          cell IDs or cycle order. Sequence models (LSTM/GRU) are disabled; tabular models only; NASA (2 A) vs Mendeley
          (1C) protocols differ; cross-workbook Transfer Learning evaluates strictly across opposite-temperature workbooks.
        </div>
      )}

      {/* 3D Interactive PCA Feature Manifold Morph */}
      <div className="lab-hero-grid">
        <div className="panel" style={{ padding: 0, overflow: "hidden" }}>
          <div style={{ padding: "14px 18px 0", display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
            <div>
              <h3 className="panel-title">
                <Layers size={16} className="text-violet" />
                3D Feature Manifold Morph ({task}
                {task === "SoH" && isSohExploratory ? " — Exploratory, Pooled Workbooks" : ""}: Li-ion Cyan vs Na-ion Amber)
              </h3>
              <p className="panel-subtitle">
                Drag slider or click Animate to morph 1,000 real 3D PCA samples from Raw Features → Physics+OCV+CORAL Aligned.
              </p>
            </div>
            <button className="btn primary" onClick={() => setAnimating(true)}>
              <Play size={13} />
              Animate Alignment
            </button>
          </div>

          {manifold[task] && <CrossChemManifold3D cloud={manifold[task]} morphFactor={morphFactor} />}

          <div style={{ padding: "10px 18px 16px", borderTop: "1px solid var(--border)" }}>
            <div className="control-label" style={{ marginBottom: 6 }}>
              <span className="text-red">0% · Raw Unaligned Features</span>
              <span className="mono text-green">Alignment Blend: {Math.round(morphFactor * 100)}%</span>
              <span className="text-green">100% · Physics + OCV-to-OCV + CORAL</span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={morphFactor}
              onChange={(e) => setMorphFactor(Number(e.target.value))}
            />
          </div>
        </div>

        {/* Live Domain-Shift Discrepancy Gauges */}
        <div className="panel" style={{ display: "flex", flexDirection: "column", justifyContent: "space-between", gap: 12 }}>
          <div>
            <h3 className="panel-title" style={{ marginBottom: 6 }}>
              <Sparkles size={15} className="text-green" />
              Live Domain-Shift Metrics ({task})
            </h3>
            <p className="panel-subtitle" style={{ marginBottom: 14 }}>
              Lower discrepancy = better zero-shot cross-chemistry transferability.
            </p>

            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <div className="kpi-card">
                <div className="kpi-label">RBF-Kernel MMD² Discrepancy</div>
                <div className={`kpi-value mono ${morphFactor > 0.6 ? "text-green" : "text-red"}`}>
                  {shiftStats.mmd2.toFixed(4)}
                </div>
                <div className="kpi-sub mono">
                  Raw: {shiftStats.raw?.mmd2.toFixed(4)} → Aligned: {shiftStats.aln?.mmd2.toFixed(4)}
                </div>
              </div>

              <div className="kpi-card">
                <div className="kpi-label">Mean Wasserstein-1 Distance</div>
                <div className={`kpi-value mono ${morphFactor > 0.6 ? "text-cyan" : "text-amber"}`}>
                  {shiftStats.w1.toFixed(4)}
                </div>
                <div className="kpi-sub mono">
                  Raw: {shiftStats.raw?.mean_wasserstein_1.toFixed(3)} → Aligned:{" "}
                  {shiftStats.aln?.mean_wasserstein_1.toFixed(3)}
                </div>
              </div>

              <div className="kpi-card">
                <div className="kpi-label">Population Stability Index (PSI)</div>
                <div className={`kpi-value mono ${morphFactor > 0.6 ? "text-green" : "text-red"}`}>
                  {shiftStats.psi.toFixed(4)}
                </div>
                <div className="kpi-sub mono">
                  Raw: {shiftStats.raw?.mean_psi.toFixed(3)} → Aligned: {shiftStats.aln?.mean_psi.toFixed(3)}
                </div>
              </div>
            </div>
          </div>

          <div className="disclaimer-banner" style={{ fontSize: 11.5 }}>
            <strong>How alignment works:</strong> Linear voltage scaling maps [2.5, 4.2]V &amp; [1.5, 3.9]V onto [0,1];
            Physics-Anchored OCV-to-OCV warping matches Na-ion&apos;s sloping OCV to Li-ion&apos;s flat plateau; Shrinkage CORAL
            aligns cross-chemistry feature covariances using zero target labels.
          </div>
        </div>
      </div>

      {/* 3-Tier Zero-Shot Ablation + Transfer Learning Curves */}
      <div className="grid-2">
        <div className="panel">
          <div className="panel-header">
            <div>
              <h3 className="panel-title">
                <TrendingDown size={16} className="text-green" />
                3-Tier Zero-Shot Cross-Chemistry Ablation ({task}:{" "}
                {direction === "li_ion_to_na_ion" ? "Li-ion → Na-ion" : "Na-ion → Li-ion"})
              </h3>
              <p className="panel-subtitle">Test RMSE on held-out target chemistry cells (Zero target labels used).</p>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={290}>
            <BarChart data={ablationData}>
              <CartesianGrid stroke="rgba(148,163,184,0.12)" strokeDasharray="3 3" />
              <XAxis dataKey="model" stroke="#94a3b8" fontSize={12} />
              <YAxis stroke="#94a3b8" fontSize={11} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="Raw" name="1. Raw Zero-Shot" fill="#f43f5e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="PhysicsNorm" name="2. Physics-Normalized" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Aligned" name="3. Physics+OCV+CORAL" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="panel">
          <div className="panel-header">
            <div>
              <h3 className="panel-title">
                <Sparkles size={16} className="text-violet" />
                Few-Shot Transfer Learning Curves ({task}:{" "}
                {direction === "li_ion_to_na_ion" ? "Li-ion → Na-ion" : "Na-ion → Li-ion"})
              </h3>
              <p className="panel-subtitle">
                {task === "SoH" && isSohExploratory
                  ? "Exploratory cross-workbook (25 °C ↔ −15 °C, tabular only) RMSE vs adaptation fraction."
                  : "6-fold Leave-One-Cell-Out Test RMSE vs labeled target training budget (5% to 50%)."}
              </p>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={290}>
            <LineChart data={tlChartData}>
              <CartesianGrid stroke="rgba(148,163,184,0.12)" strokeDasharray="3 3" />
              <XAxis dataKey="budget" stroke="#94a3b8" fontSize={12} />
              <YAxis stroke="#94a3b8" fontSize={11} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 11.5 }} />
              <Line type="monotone" dataKey="TargetOnly_XGB" name="TargetOnly XGB" stroke="#94a3b8" strokeDasharray="5 4" strokeWidth={2} />
              <Line type="monotone" dataKey="TL_Residual_XGB" name="TL-Residual XGB" stroke="#10b981" strokeWidth={2.3} />
              <Line type="monotone" dataKey="TL_WarmStart_XGB" name="TL-WarmStart XGB" stroke="#38bdf8" strokeWidth={2} />
              {!(task === "SoH" && isSohExploratory) && (
                <>
                  <Line type="monotone" dataKey="TargetOnly_GRU" name="TargetOnly GRU" stroke="#f43f5e" strokeDasharray="3 3" strokeWidth={2} />
                  <Line type="monotone" dataKey="TL_TwoPhase_GRU" name="TL-TwoPhase GRU" stroke="#a78bfa" strokeWidth={2.6} />
                </>
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
