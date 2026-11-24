import React, { useMemo, useState } from "react";
import { BarChart3, Cpu, Database, FileImage, Filter } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { OverviewSummary, WithinRow } from "../types";

interface ModelsAndDataViewProps {
  summary: OverviewSummary;
  withinRows: WithinRow[];
  figures: string[];
}

const FIGURE_CAPTIONS: Record<string, string> = {
  "fig1_domain_alignment_pca.png": "Fig 1 · PCA Manifold Alignment (Raw vs Physics + OCV-to-OCV + CORAL)",
  "fig2_within_chemistry_benchmarks.png": "Fig 2 · Within-Chemistry Baselines across all 5 Model Families",
  "fig3_cross_chemistry_ablation.png": "Fig 3 · 3-Tier Unsupervised Cross-Chemistry Generalization Ablation",
  "fig4_transfer_learning_curves.png": "Fig 4 · Few-Shot Transfer Learning Curves (Li-ion → Na-ion)",
  "fig5_cross_chemistry_parity.png": "Fig 5 · Cross-Chemistry Parity Recovery (Raw vs Aligned Zero-Shot)",
  "fig6_adaptive_vs_cccv_charging.png": "Fig 6 · Simulated Adaptive ML Charging vs CC-CV Trajectories",
};

export const ModelsAndDataView: React.FC<ModelsAndDataViewProps> = ({ summary, withinRows, figures }) => {
  const [taskFilter, setTaskFilter] = useState<"SoC" | "SoH">("SoC");

  const withinChart = useMemo(() => {
    const models = ["LinearRegression", "RandomForest", "XGBoost", "LSTM", "GRU"];
    return models.map((m) => {
      const li = withinRows.find((r) => r.task === taskFilter && r.chemistry === "li_ion" && r.model === m);
      const na = withinRows.find((r) => r.task === taskFilter && r.chemistry === "na_ion" && r.model === m);
      return {
        model: m,
        LiIon_RMSE: li?.rmse ?? 0,
        NaIon_RMSE: na?.rmse ?? 0,
      };
    });
  }, [withinRows, taskFilter]);

  const audit = summary.quality_audit;
  const inv = summary.dataset_inventory;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      {/* Objective 1: Big Data Lakehouse & Quality Audit KPIs */}
      <div className="panel">
        <div className="panel-header">
          <div>
            <h2 className="panel-title">
              <Database size={18} className="text-cyan" />
              Objective 1 · Big Data Medallion Lakehouse &amp; Data Mining Quality Audit
            </h2>
            <p className="panel-subtitle">
              Polars lazy streaming + ZSTD Apache Parquet (Bronze → Silver → Gold) + DuckDB SQL analytics +
              IsolationForest anomaly removal.
            </p>
          </div>
          <span className="tag tag-li">LFS Inventory: {inv.total_files} files ({inv.total_declared_mb} MB)</span>
        </div>

        <div className="grid-4">
          <div className="kpi-card">
            <div className="kpi-label">Silver Time-Series Rows</div>
            <div className="kpi-value mono text-cyan">{audit.timeseries.silver_rows.toLocaleString()}</div>
            <div className="kpi-sub">
              Retention {audit.timeseries.retention_rate_pct}% (from {audit.timeseries.bronze_rows.toLocaleString()} Bronze)
            </div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">Anomalies Cleaned</div>
            <div className="kpi-value mono text-amber">
              {audit.timeseries.duplicates_removed +
                audit.timeseries.bounds_and_null_removed +
                audit.timeseries.isolation_forest_anomalies_removed}
            </div>
            <div className="kpi-sub">
              {audit.timeseries.duplicates_removed} dups · {audit.timeseries.bounds_and_null_removed} bounds ·{" "}
              {audit.timeseries.isolation_forest_anomalies_removed} IForest
            </div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">Silver Aging Cycles</div>
            <div className="kpi-value mono text-green">{audit.cycles.silver_rows.toLocaleString()}</div>
            <div className="kpi-sub">16 cells (8 Li-ion + 8 Na-ion), strict cell-level splits</div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">K-Means Operating Regimes</div>
            <div className="kpi-value mono text-violet">{summary.data_mining.kmeans_silhouette.toFixed(4)}</div>
            <div className="kpi-sub">4-cluster Silhouette Score in physics-normalized space</div>
          </div>
        </div>
      </div>

      {/* Objectives 2 & 3: Within-Chemistry 5-Model Benchmarks */}
      <div className="grid-2">
        <div className="panel">
          <div className="panel-header">
            <div>
              <h3 className="panel-title">
                <Cpu size={16} className="text-green" />
                Objectives 2 &amp; 3 · Within-Chemistry Model RMSE ({taskFilter})
              </h3>
              <p className="panel-subtitle">Held-out test cells across all 5 model families specified in the PDF.</p>
            </div>
            <div className="btn-group">
              <button className={`btn ${taskFilter === "SoC" ? "active" : ""}`} onClick={() => setTaskFilter("SoC")}>
                SoC
              </button>
              <button className={`btn ${taskFilter === "SoH" ? "active" : ""}`} onClick={() => setTaskFilter("SoH")}>
                SoH
              </button>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={275}>
            <BarChart data={withinChart}>
              <CartesianGrid stroke="rgba(148,163,184,0.12)" strokeDasharray="3 3" />
              <XAxis dataKey="model" stroke="#94a3b8" fontSize={11.5} />
              <YAxis stroke="#94a3b8" fontSize={11} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="LiIon_RMSE" name="Li-ion RMSE" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="NaIon_RMSE" name="Na-ion RMSE" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="panel">
          <div className="panel-header">
            <div>
              <h3 className="panel-title">
                <Filter size={16} className="text-amber" />
                Data Mining · Top Mutual Information Features ({taskFilter})
              </h3>
              <p className="panel-subtitle">Non-parametric relevance of physics-normalized features to {taskFilter}.</p>
            </div>
          </div>
          <div className="grid-2" style={{ gap: 12 }}>
            {[
              {
                label: "Li-ion",
                tag: "tag-li",
                rows:
                  taskFilter === "SoC"
                    ? summary.data_mining.top_mi_soc_li
                    : summary.data_mining.top_mi_soh_li,
              },
              {
                label: "Na-ion",
                tag: "tag-na",
                rows:
                  taskFilter === "SoC"
                    ? summary.data_mining.top_mi_soc_na
                    : summary.data_mining.top_mi_soh_na,
              },
            ].map((grp) => (
              <div key={grp.label} className="kpi-card">
                <div style={{ marginBottom: 10 }}>
                  <span className={`tag ${grp.tag}`}>{grp.label}</span>
                </div>
                {grp.rows.map((r) => (
                  <div
                    key={r.feature}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      fontSize: 12,
                      padding: "5px 0",
                      borderBottom: "1px solid rgba(148,163,184,0.08)",
                    }}
                  >
                    <span className="mono">{r.feature}</span>
                    <span className="mono text-cyan">{r.mutual_information.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Full Quantitative Evaluation Table (Objective 6) */}
      <div className="panel">
        <div className="panel-header">
          <h3 className="panel-title">
            <BarChart3 size={16} className="text-cyan" />
            Objective 6 · Full Within-Chemistry Quantitative Benchmark Table ({taskFilter})
          </h3>
        </div>
        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Chemistry</th>
                <th>Model</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>R²</th>
                <th>Train Time (s)</th>
                <th>Inference (ms)</th>
              </tr>
            </thead>
            <tbody className="mono">
              {withinRows
                .filter((r) => r.task === taskFilter)
                .map((r, i) => (
                  <tr key={i}>
                    <td>
                      <span className={`tag ${r.chemistry === "li_ion" ? "tag-li" : "tag-na"}`}>{r.chemistry}</span>
                    </td>
                    <td>{r.model}</td>
                    <td>{r.mae.toFixed(4)}</td>
                    <td>{r.rmse.toFixed(4)}</td>
                    <td className="text-green">{r.r2.toFixed(4)}</td>
                    <td>{r.train_time_s.toFixed(3)}</td>
                    <td>{r.infer_time_ms.toFixed(2)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Publication Figure Gallery served from urop_core/outputs/figures/ */}
      <div className="panel">
        <div className="panel-header">
          <div>
            <h3 className="panel-title">
              <FileImage size={16} className="text-violet" />
              Generated Publication Figures (Served from urop_core/outputs/figures/)
            </h3>
            <p className="panel-subtitle">
              All artifacts are read directly from the standalone 6-objective UROP pipeline outputs.
            </p>
          </div>
        </div>
        <div className="grid-2">
          {figures.map((fname) => (
            <div key={fname} className="figure-card">
              <img src={`/api/figures/${fname}`} alt={fname} loading="lazy" />
              <div className="figure-caption">{FIGURE_CAPTIONS[fname] ?? fname}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
