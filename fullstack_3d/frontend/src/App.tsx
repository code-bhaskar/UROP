import React, { useEffect, useState } from "react";
import { ArrowRightLeft, BatteryCharging, Cpu, Database } from "lucide-react";
import { api } from "./api";
import { AdaptiveChargingLab } from "./components/AdaptiveChargingLab";
import { CrossChemistryView } from "./components/CrossChemistryView";
import { ModelsAndDataView } from "./components/ModelsAndDataView";
import type {
  ChargingTraceData,
  CrossRow,
  DomainShiftRow,
  OverviewSummary,
  PcaTaskCloud,
  TransferRow,
  WithinRow,
} from "./types";

type TabKey = "charging" | "cross_chem" | "models_data";

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabKey>("charging");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const [summary, setSummary] = useState<OverviewSummary | null>(null);
  const [figures, setFigures] = useState<string[]>([]);
  const [traces, setTraces] = useState<Record<string, ChargingTraceData>>({});
  const [manifold, setManifold] = useState<Record<"SoC" | "SoH", PcaTaskCloud> | null>(null);
  const [domainShift, setDomainShift] = useState<DomainShiftRow[]>([]);
  const [withinRows, setWithinRows] = useState<WithinRow[]>([]);
  const [crossRows, setCrossRows] = useState<CrossRow[]>([]);
  const [transferRows, setTransferRows] = useState<TransferRow[]>([]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [ov, tr, pca, wi, cr, tl] = await Promise.all([
          api.overview(),
          api.chargingTraces(),
          api.pcaManifold(),
          api.withinTable(),
          api.crossTable(),
          api.transferTable(),
        ]);
        if (cancelled) return;
        setSummary(ov.summary);
        setFigures(ov.figures);
        setTraces(tr.traces);
        setManifold(pca.manifold);
        setDomainShift(pca.domain_shift);
        setWithinRows(wi.rows);
        setCrossRows(cr.rows);
        setTransferRows(tl.rows);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const isReal = summary?.data_mode === "real_dataset" || summary?.data_mode === "real_lfs_dataset";

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <BatteryCharging size={20} />
          </div>
          <div>
            <div className="brand-title">Cross-Chemistry BMS 3D Digital Twin</div>
            <div className="brand-sub">Li-Ion &amp; Na-Ion Adaptive Charging · Transfer Learning · Columnar Out-of-Core Lakehouse</div>
          </div>
        </div>

        <nav className="nav-tabs" aria-label="Main sections">
          <button
            className={`nav-tab ${activeTab === "charging" ? "active" : ""}`}
            onClick={() => setActiveTab("charging")}
          >
            <BatteryCharging size={15} />
            3D Adaptive Charging (Obj. 5)
          </button>
          <button
            className={`nav-tab ${activeTab === "cross_chem" ? "active" : ""}`}
            onClick={() => setActiveTab("cross_chem")}
          >
            <ArrowRightLeft size={15} />
            3D Cross-Chemistry &amp; TL (Obj. 4)
          </button>
          <button
            className={`nav-tab ${activeTab === "models_data" ? "active" : ""}`}
            onClick={() => setActiveTab("models_data")}
          >
            <Database size={15} />
            Lakehouse, Models &amp; Figures (Obj. 1–3, 6)
          </button>
        </nav>

        {summary && (
          <div className={`status-pill ${isReal ? "" : "sim"}`}>
            <Cpu size={13} />
            {isReal
              ? `Real Lab Datasets (${summary.dataset_inventory.pulled_files}/${summary.dataset_inventory.total_files} files)`
              : "Consuming urop_core/outputs"}
          </div>
        )}
      </header>

      <main className="main-content">
        {loading && (
          <div className="panel" style={{ textAlign: "center", padding: 48 }}>
            <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 6 }}>
              Loading 6-Objective Artifacts from <code className="mono">urop_core/outputs/</code>...
            </div>
            <div style={{ fontSize: 12.5, color: "#94a3b8" }}>
              Initializing Three.js WebGL Digital Twin &amp; FastAPI Telemetry Bridge
            </div>
          </div>
        )}

        {error && (
          <div className="panel" style={{ borderColor: "rgba(244,63,94,0.5)" }}>
            <h3 className="panel-title text-red">Backend Connection / Artifact Error</h3>
            <p className="panel-subtitle mono" style={{ marginTop: 8 }}>
              {error}
            </p>
            <p className="panel-subtitle" style={{ marginTop: 8 }}>
              Ensure FastAPI is running on port 8000 and <code className="mono">urop_core/outputs/</code> has been
              generated via <code className="mono">cd urop_core &amp;&amp; python -m src.pipeline</code>.
            </p>
          </div>
        )}

        {!loading && !error && summary && (
          <>
            {activeTab === "charging" && <AdaptiveChargingLab precomputedTraces={traces} />}
            {activeTab === "cross_chem" && manifold && (
              <CrossChemistryView
                summary={summary}
                manifold={manifold}
                domainShift={domainShift}
                crossRows={crossRows}
                transferRows={transferRows}
              />
            )}
            {activeTab === "models_data" && (
              <ModelsAndDataView summary={summary} withinRows={withinRows} figures={figures} />
            )}
          </>
        )}
      </main>
    </div>
  );
};

export default App;
