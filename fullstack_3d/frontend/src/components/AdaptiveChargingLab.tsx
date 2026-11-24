import React, { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BatteryCharging,
  Eye,
  Flame,
  Gauge,
  Pause,
  Play,
  RotateCcw,
  ShieldCheck,
  Sliders,
  Sparkles,
  Zap,
} from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api";
import type { ChargingTraceData } from "../types";
import { BatteryCell3D, type LiveCellTelemetry } from "./BatteryCell3D";

interface AdaptiveChargingLabProps {
  precomputedTraces: Record<string, ChargingTraceData>;
}

const PRESETS = [
  { key: "na_ion_soh80", label: "Aged Na-ion (80% SoH · TL-SoC)", chem: "na_ion" as const, soh: 0.8 },
  { key: "li_ion_soh80", label: "Aged Li-ion (80% SoH)", chem: "li_ion" as const, soh: 0.8 },
  { key: "na_ion_soh98", label: "Fresh Na-ion (98% SoH · TL-SoC)", chem: "na_ion" as const, soh: 0.98 },
  { key: "li_ion_soh98", label: "Fresh Li-ion (98% SoH)", chem: "li_ion" as const, soh: 0.98 },
];

function sampleAtTime(trace: ChargingTraceData, tMin: number, ambTempC: number): LiveCellTelemetry {
  const s = trace.series;
  const times = s.time_min;
  if (!times.length) {
    return {
      soc: 0.1,
      socEst: 0.1,
      tempC: ambTempC,
      ambientTempC: ambTempC,
      cRate: 0,
      voltageV: 3.0,
      strategy: trace.strategy,
    };
  }
  if (tMin >= times[times.length - 1]) {
    const last = times.length - 1;
    return {
      soc: s.soc_true[last],
      socEst: s.soc_est[last],
      tempC: s.temp_c[last],
      ambientTempC: ambTempC,
      cRate: 0.0, // charge complete
      voltageV: s.voltage_v[last],
      strategy: trace.strategy,
    };
  }

  let idx = 0;
  for (let i = 0; i < times.length - 1; i++) {
    if (times[i + 1] >= tMin) {
      idx = i;
      break;
    }
  }
  const t0 = times[idx];
  const t1 = times[idx + 1] ?? t0;
  const frac = t1 > t0 ? (tMin - t0) / (t1 - t0) : 0;
  const lerp = (arr: number[]) => arr[idx] + (arr[idx + 1] - arr[idx]) * frac;

  return {
    soc: lerp(s.soc_true),
    socEst: lerp(s.soc_est),
    tempC: lerp(s.temp_c),
    ambientTempC: ambTempC,
    cRate: lerp(s.current_c_rate),
    voltageV: lerp(s.voltage_v),
    strategy: trace.strategy,
  };
}

export const AdaptiveChargingLab: React.FC<AdaptiveChargingLabProps> = ({ precomputedTraces }) => {
  const [selectedPreset, setSelectedPreset] = useState<string>("na_ion_soh80");
  const [customPair, setCustomPair] = useState<{
    cccv: ChargingTraceData;
    adaptive: ChargingTraceData;
    ambTemp: number;
    label: string;
  } | null>(null);

  // Live simulation form state
  const [simChem, setSimChem] = useState<"li_ion" | "na_ion">("na_ion");
  const [simSoh, setSimSoh] = useState<number>(0.78);
  const [simAmbTemp, setSimAmbTemp] = useState<number>(28);
  const [simTargetSoc, setSimTargetSoc] = useState<number>(0.9);
  const [simRunning, setSimRunning] = useState<boolean>(false);

  // 3D viewport toggles
  const [compareMode, setCompareMode] = useState<boolean>(true);
  const [xrayCutaway, setXrayCutaway] = useState<boolean>(true);
  const [autoRotate, setAutoRotate] = useState<boolean>(true);

  // Playback state
  const [playing, setPlaying] = useState<boolean>(true);
  const [speed, setSpeed] = useState<number>(4);
  const [playTimeMin, setPlayTimeMin] = useState<number>(0);

  const activeData = useMemo(() => {
    if (customPair) {
      return customPair;
    }
    const cccv = precomputedTraces[`${selectedPreset}_CC-CV`];
    const adaptive = precomputedTraces[`${selectedPreset}_Adaptive_ML`];
    const presetMeta = PRESETS.find((p) => p.key === selectedPreset);
    return {
      cccv,
      adaptive,
      ambTemp: 25,
      label: presetMeta?.label ?? selectedPreset,
    };
  }, [customPair, precomputedTraces, selectedPreset]);

  const maxTimeMin = useMemo(() => {
    const t1 = activeData.cccv?.metrics.charge_time_min ?? 60;
    const t2 = activeData.adaptive?.metrics.charge_time_min ?? 60;
    return Math.max(t1, t2);
  }, [activeData]);

  // Smooth playback loop
  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    let raf = 0;
    const step = (now: number) => {
      const dtSec = (now - last) / 1000;
      last = now;
      setPlayTimeMin((prev) => {
        const next = prev + dtSec * speed * 1.4;
        return next > maxTimeMin ? 0 : next;
      });
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, maxTimeMin]);

  const handleSelectPreset = (key: string) => {
    setCustomPair(null);
    setSelectedPreset(key);
    setPlayTimeMin(0);
    setPlaying(true);
  };

  const handleRunCustomSim = async () => {
    setSimRunning(true);
    try {
      const res = await api.simulateCharging({
        chemistry: simChem,
        soh_true: simSoh,
        ambient_temp_c: simAmbTemp,
        soc_start: 0.1,
        soc_target: simTargetSoc,
      });
      setCustomPair({
        cccv: res.cc_cv,
        adaptive: res.adaptive_ml,
        ambTemp: simAmbTemp,
        label: `Custom Live Sim · ${simChem === "li_ion" ? "Li-ion 18650" : "Na-ion Prismatic"} (${Math.round(
          simSoh * 100
        )}% SoH @ ${simAmbTemp}°C)`,
      });
      setPlayTimeMin(0);
      setPlaying(true);
    } finally {
      setSimRunning(false);
    }
  };

  if (!activeData.cccv || !activeData.adaptive) {
    return <div className="panel">Loading Objective 5 charging traces from urop_core/outputs...</div>;
  }

  const chem = activeData.adaptive.chemistry;
  const cccvNow = sampleAtTime(activeData.cccv, playTimeMin, activeData.ambTemp);
  const adaptNow = sampleAtTime(activeData.adaptive, playTimeMin, activeData.ambTemp);

  const mCC = activeData.cccv.metrics;
  const mAd = activeData.adaptive.metrics;

  const tempReductionPct =
    ((mCC.peak_temp_rise_c - mAd.peak_temp_rise_c) / Math.max(mCC.peak_temp_rise_c, 0.01)) * 100;
  const degReductionPct =
    ((mCC.estimated_degradation_pct - mAd.estimated_degradation_pct) /
      Math.max(mCC.estimated_degradation_pct, 1e-6)) *
    100;
  const effGain = mAd.energy_efficiency_pct - mCC.energy_efficiency_pct;

  // Merge time grids cleanly for synchronized Recharts visualization
  const chartData = useMemo(() => {
    const nPts = 70;
    const rows = [];
    for (let i = 0; i <= nPts; i++) {
      const t = (i / nPts) * maxTimeMin;
      const c = sampleAtTime(activeData.cccv, t, activeData.ambTemp);
      const a = sampleAtTime(activeData.adaptive, t, activeData.ambTemp);
      rows.push({
        time: Number(t.toFixed(1)),
        cc_cRate: Number(c.cRate.toFixed(3)),
        ad_cRate: Number(a.cRate.toFixed(3)),
        cc_temp: Number(c.tempC.toFixed(2)),
        ad_temp: Number(a.tempC.toFixed(2)),
        cc_volt: Number(c.voltageV.toFixed(3)),
        ad_volt: Number(a.voltageV.toFixed(3)),
        ad_soc_true: Number((a.soc * 100).toFixed(1)),
        ad_soc_est: Number((a.socEst * 100).toFixed(1)),
      });
    }
    return rows;
  }, [activeData, maxTimeMin]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      {/* Scenario Selector Bar */}
      <div className="panel" style={{ padding: "14px 18px" }}>
        <div className="panel-header" style={{ marginBottom: 10 }}>
          <div>
            <h2 className="panel-title">
              <BatteryCharging size={18} className="text-green" />
              Objective 5: 3D Adaptive Charging Digital Twin (ML/TL-Informed vs Conventional CC-CV)
            </h2>
            <p className="panel-subtitle">
              Real-time 3D cutaway visualization of{" "}
              <strong style={{ color: "#e2e8f0" }}>
                {chem === "li_ion" ? "18650 Cylindrical Lithium-Ion (2.5–4.2 V)" : "Prismatic Sodium-Ion (1.5–3.9 V)"}
              </strong>{" "}
              driven by <code className="mono">urop_core/outputs/charging_traces.json</code> &amp; live FastAPI physics bridge.
            </p>
          </div>
          <span className={`tag ${chem === "li_ion" ? "tag-li" : "tag-na"}`}>
            Active: {activeData.label}
          </span>
        </div>

        <div className="btn-group">
          {PRESETS.map((p) => (
            <button
              key={p.key}
              className={`btn ${p.chem === "na_ion" ? "amber" : ""} ${
                !customPair && selectedPreset === p.key ? "active" : ""
              }`}
              onClick={() => handleSelectPreset(p.key)}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Hero 3D Stage + Live Simulator Control Sidebar */}
      <div className="lab-hero-grid">
        {/* Left: Three.js 3D Digital Twin Stage */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div className="stage-wrapper">
            <BatteryCell3D
              chemistry={chem}
              cccv={cccvNow}
              adaptive={adaptNow}
              compareMode={compareMode}
              xrayCutaway={xrayCutaway}
              autoRotate={autoRotate}
            />

            {/* Top Floating 3D Controls */}
            <div className="stage-hud-top">
              <div className="hud-badge">
                <div className="hud-badge-title">3D WebGL Digital Twin · Drag to Orbit · Scroll to Zoom</div>
                <div style={{ fontSize: 12, color: "#cbd5e1" }}>
                  Ion flux speed = live C-rate · Core glow = ΔT thermal stress · Red halo = high-SoC plating risk
                </div>
              </div>
              <div className="btn-group">
                <button
                  className={`btn ${compareMode ? "active" : ""}`}
                  onClick={() => setCompareMode((v) => !v)}
                >
                  {compareMode ? "Dual Twin (CC-CV vs ML)" : "Single Cell Inspect"}
                </button>
                <button
                  className={`btn ${xrayCutaway ? "active" : ""}`}
                  onClick={() => setXrayCutaway((v) => !v)}
                >
                  <Eye size={13} />
                  {xrayCutaway ? "X-Ray Cutaway" : "Solid Casing"}
                </button>
                <button
                  className={`btn ${autoRotate ? "active" : ""}`}
                  onClick={() => setAutoRotate((v) => !v)}
                >
                  <RotateCcw size={13} />
                  Orbit
                </button>
              </div>
            </div>

            {/* Bottom Floating Live Telemetry HUD */}
            <div className="stage-hud-bottom">
              {compareMode && (
                <div className="hud-badge" style={{ borderColor: "rgba(244, 63, 94, 0.5)" }}>
                  <div className="hud-badge-title text-red">LEFT CELL · CONVENTIONAL CC-CV</div>
                  <div className="hud-badge-row mono">
                    <span>
                      SoC <strong>{(cccvNow.soc * 100).toFixed(1)}%</strong>
                    </span>
                    <span className="text-red">
                      {cccvNow.tempC.toFixed(1)}°C
                    </span>
                    <span>{cccvNow.cRate.toFixed(2)}C</span>
                    <span>{cccvNow.voltageV.toFixed(2)}V</span>
                  </div>
                </div>
              )}

              <div className="hud-badge" style={{ borderColor: "rgba(16, 185, 129, 0.55)" }}>
                <div className="hud-badge-title text-green">
                  {compareMode ? "RIGHT CELL · " : ""}ADAPTIVE ML CONTROLLER
                </div>
                <div className="hud-badge-row mono">
                  <span>
                    SoC <strong>{(adaptNow.soc * 100).toFixed(1)}%</strong>
                  </span>
                  <span className="text-green">
                    {adaptNow.tempC.toFixed(1)}°C
                  </span>
                  <span>{adaptNow.cRate.toFixed(2)}C</span>
                  <span>{adaptNow.voltageV.toFixed(2)}V</span>
                </div>
              </div>
            </div>
          </div>

          {/* Synchronized Timeline Playback Bar */}
          <div className="playback-bar">
            <button className="btn primary" onClick={() => setPlaying((p) => !p)}>
              {playing ? <Pause size={14} /> : <Play size={14} />}
              {playing ? "Pause" : "Play"}
            </button>
            <button
              className="btn"
              onClick={() => {
                setPlayTimeMin(0);
                setPlaying(true);
              }}
            >
              <RotateCcw size={13} />
            </button>
            <div style={{ flex: 1, minWidth: 160 }}>
              <input
                type="range"
                min={0}
                max={maxTimeMin}
                step={0.2}
                value={playTimeMin}
                onChange={(e) => {
                  setPlaying(false);
                  setPlayTimeMin(Number(e.target.value));
                }}
              />
            </div>
            <span className="mono" style={{ fontSize: 12.5, minWidth: 108, textAlign: "right" }}>
              t = {playTimeMin.toFixed(1)} / {maxTimeMin.toFixed(1)} min
            </span>
            <div className="btn-group">
              {[1, 2, 4, 8].map((s) => (
                <button key={s} className={`btn ${speed === s ? "active" : ""}`} onClick={() => setSpeed(s)}>
                  {s}x
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right: Live What-If Backend Simulator & Head-to-Head KPIs */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {/* Head-to-Head Gains Scorecard */}
          <div className="panel">
            <h3 className="panel-title" style={{ marginBottom: 12 }}>
              <Sparkles size={15} className="text-green" />
              Adaptive ML Gains vs CC-CV
            </h3>
            <div className="grid-2" style={{ gap: 10 }}>
              <div className="kpi-card">
                <div className="kpi-label">Peak ΔT Reduction</div>
                <div className="kpi-value mono text-green">-{tempReductionPct.toFixed(1)}%</div>
                <div className="kpi-sub mono">
                  {mCC.peak_temp_rise_c.toFixed(2)}°C → {mAd.peak_temp_rise_c.toFixed(2)}°C
                </div>
              </div>
              <div className="kpi-card">
                <div className="kpi-label">Degradation Cut</div>
                <div className="kpi-value mono text-green">-{degReductionPct.toFixed(1)}%</div>
                <div className="kpi-sub mono">
                  {mCC.estimated_degradation_pct.toFixed(4)}% → {mAd.estimated_degradation_pct.toFixed(4)}%
                </div>
              </div>
              <div className="kpi-card">
                <div className="kpi-label">Energy Efficiency</div>
                <div className="kpi-value mono text-cyan">{mAd.energy_efficiency_pct.toFixed(2)}%</div>
                <div className="kpi-sub mono">
                  {effGain >= 0 ? "+" : ""}
                  {effGain.toFixed(2)}% vs CC-CV ({mCC.energy_efficiency_pct.toFixed(2)}%)
                </div>
              </div>
              <div className="kpi-card">
                <div className="kpi-label">Safety &amp; Charge Time</div>
                <div className="kpi-value mono" style={{ fontSize: 18 }}>
                  {mAd.charge_time_min.toFixed(1)} min
                </div>
                <div className="kpi-sub" style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <ShieldCheck size={13} className="text-green" />
                  Limits 100% safe (CC-CV: {mCC.charge_time_min.toFixed(1)}m)
                </div>
              </div>
            </div>
          </div>

          {/* Interactive Live Simulation Controls (Calls FastAPI POST /api/charging/simulate) */}
          <div className="panel" style={{ flex: 1 }}>
            <h3 className="panel-title" style={{ marginBottom: 4 }}>
              <Sliders size={15} className="text-cyan" />
              Live What-If Charging Simulator
            </h3>
            <p className="panel-subtitle" style={{ marginBottom: 14 }}>
              Calls FastAPI <code className="mono">POST /api/charging/simulate</code> to run a fresh coupled
              electro-thermal-aging simulation in real time.
            </p>

            <div className="control-row">
              <div className="control-label">
                <span>Battery Chemistry</span>
                <span className="mono">{simChem === "li_ion" ? "Li-ion (18650)" : "Na-ion (Prismatic)"}</span>
              </div>
              <div className="btn-group">
                <button
                  className={`btn ${simChem === "li_ion" ? "active" : ""}`}
                  style={{ flex: 1, justifyContent: "center" }}
                  onClick={() => setSimChem("li_ion")}
                >
                  Li-ion (2.5–4.2V)
                </button>
                <button
                  className={`btn amber ${simChem === "na_ion" ? "active" : ""}`}
                  style={{ flex: 1, justifyContent: "center" }}
                  onClick={() => setSimChem("na_ion")}
                >
                  Na-ion (1.5–3.9V)
                </button>
              </div>
            </div>

            <div className="control-row">
              <div className="control-label">
                <span>Cell State of Health (SoH)</span>
                <span className="mono text-cyan">{Math.round(simSoh * 100)}%</span>
              </div>
              <input
                type="range"
                min={0.7}
                max={1.0}
                step={0.02}
                value={simSoh}
                onChange={(e) => setSimSoh(Number(e.target.value))}
              />
            </div>

            <div className="control-row">
              <div className="control-label">
                <span>Ambient Temperature</span>
                <span className="mono text-amber">{simAmbTemp}°C</span>
              </div>
              <input
                type="range"
                min={5}
                max={36}
                step={1}
                value={simAmbTemp}
                onChange={(e) => setSimAmbTemp(Number(e.target.value))}
              />
            </div>

            <div className="control-row">
              <div className="control-label">
                <span>Target Charge SoC</span>
                <span className="mono text-green">{Math.round(simTargetSoc * 100)}%</span>
              </div>
              <input
                type="range"
                min={0.8}
                max={0.95}
                step={0.01}
                value={simTargetSoc}
                onChange={(e) => setSimTargetSoc(Number(e.target.value))}
              />
            </div>

            <button
              className="btn primary"
              style={{ width: "100%", justifyContent: "center", padding: "10px 14px", marginTop: 4 }}
              onClick={handleRunCustomSim}
              disabled={simRunning}
            >
              <Zap size={15} />
              {simRunning ? "Simulating Electro-Thermal ODE..." : "Run Live 3D Simulation"}
            </button>
          </div>
        </div>
      </div>

      {/* Synchronized Telemetry Charts (4 Panels) */}
      <div className="grid-2">
        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">
              <Zap size={15} className="text-amber" />
              Charging Current Schedule (C-Rate)
            </h3>
          </div>
          <ResponsiveContainer width="100%" height={235}>
            <LineChart data={chartData}>
              <CartesianGrid stroke="rgba(148,163,184,0.12)" strokeDasharray="3 3" />
              <XAxis dataKey="time" stroke="#94a3b8" fontSize={11} unit="m" />
              <YAxis stroke="#94a3b8" fontSize={11} unit="C" domain={[0, 1.2]} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <ReferenceLine x={Number(playTimeMin.toFixed(1))} stroke="#38bdf8" strokeDasharray="4 4" />
              <Line type="stepAfter" dataKey="cc_cRate" name="CC-CV (C)" stroke="#f43f5e" strokeWidth={2} dot={false} />
              <Line
                type="stepAfter"
                dataKey="ad_cRate"
                name="Adaptive ML (C)"
                stroke="#10b981"
                strokeWidth={2.4}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">
              <Flame size={15} className="text-red" />
              Cell Thermal Trajectory (°C)
            </h3>
          </div>
          <ResponsiveContainer width="100%" height={235}>
            <LineChart data={chartData}>
              <CartesianGrid stroke="rgba(148,163,184,0.12)" strokeDasharray="3 3" />
              <XAxis dataKey="time" stroke="#94a3b8" fontSize={11} unit="m" />
              <YAxis stroke="#94a3b8" fontSize={11} unit="°C" domain={["auto", "auto"]} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <ReferenceLine x={Number(playTimeMin.toFixed(1))} stroke="#38bdf8" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="cc_temp" name="CC-CV Temp" stroke="#f43f5e" strokeWidth={2} dot={false} />
              <Line
                type="monotone"
                dataKey="ad_temp"
                name="Adaptive ML Temp"
                stroke="#10b981"
                strokeWidth={2.4}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">
              <Gauge size={15} className="text-cyan" />
              Terminal Voltage Envelope (V)
            </h3>
          </div>
          <ResponsiveContainer width="100%" height={235}>
            <LineChart data={chartData}>
              <CartesianGrid stroke="rgba(148,163,184,0.12)" strokeDasharray="3 3" />
              <XAxis dataKey="time" stroke="#94a3b8" fontSize={11} unit="m" />
              <YAxis stroke="#94a3b8" fontSize={11} unit="V" domain={["auto", "auto"]} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <ReferenceLine x={Number(playTimeMin.toFixed(1))} stroke="#38bdf8" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="cc_volt" name="CC-CV (V)" stroke="#f43f5e" strokeWidth={2} dot={false} />
              <Line
                type="monotone"
                dataKey="ad_volt"
                name="Adaptive ML (V)"
                stroke="#38bdf8"
                strokeWidth={2.4}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">
              <Activity size={15} className="text-green" />
              Closed-Loop SoC Observer (True vs ML/TL Estimated)
            </h3>
            <span className="mono" style={{ fontSize: 12, color: "#94a3b8" }}>
              Observer MAE = {(mAd.soc_observer_mae * 100).toFixed(2)}%
            </span>
          </div>
          <ResponsiveContainer width="100%" height={235}>
            <LineChart data={chartData}>
              <CartesianGrid stroke="rgba(148,163,184,0.12)" strokeDasharray="3 3" />
              <XAxis dataKey="time" stroke="#94a3b8" fontSize={11} unit="m" />
              <YAxis stroke="#94a3b8" fontSize={11} unit="%" domain={[0, 100]} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <ReferenceLine x={Number(playTimeMin.toFixed(1))} stroke="#38bdf8" strokeDasharray="4 4" />
              <Line
                type="monotone"
                dataKey="ad_soc_true"
                name="True SoC (%)"
                stroke="#38bdf8"
                strokeWidth={2.3}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="ad_soc_est"
                name="ML/TL Estimated SoC (%)"
                stroke="#f59e0b"
                strokeWidth={2}
                strokeDasharray="5 4"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="disclaimer-banner">
        <strong>Proposal Section 5.7 Scientific Note:</strong> Adaptive charging performance is evaluated through a
        coupled electro-thermal-aging simulation respecting chemistry-specific cut-off voltages (4.2 V Li-ion / 3.9 V
        Na-ion) and thermal bounds (45 °C). It serves as a validated digital-twin simulation study rather than physical
        hardware cycler validation.
      </div>
    </div>
  );
};
