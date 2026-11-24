import type {
  ChargingMetrics,
  ChargingTraceData,
  CrossRow,
  DomainShiftRow,
  LiveSimulationResponse,
  OverviewSummary,
  PcaTaskCloud,
  TransferRow,
  WithinRow,
} from "./types";

async function getJson<T>(url: string): Promise<T> {
  const resp = await fetch(url);
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`API ${url} failed (${resp.status}): ${txt}`);
  }
  return resp.json() as Promise<T>;
}

export const api = {
  overview: () => getJson<{ summary: OverviewSummary; figures: string[] }>("/api/overview"),
  chargingTraces: () =>
    getJson<{ traces: Record<string, ChargingTraceData>; summary: ChargingMetrics[] }>("/api/charging/traces"),
  pcaManifold: () =>
    getJson<{ manifold: Record<"SoC" | "SoH", PcaTaskCloud>; domain_shift: DomainShiftRow[] }>("/api/pca-manifold"),
  withinTable: () => getJson<{ rows: WithinRow[] }>("/api/tables/within_chemistry"),
  crossTable: () => getJson<{ rows: CrossRow[] }>("/api/tables/cross_chemistry"),
  transferTable: () => getJson<{ rows: TransferRow[] }>("/api/tables/transfer_learning"),
  report: () => getJson<{ markdown: string }>("/api/report"),
  simulateCharging: async (payload: {
    chemistry: "li_ion" | "na_ion";
    soh_true: number;
    ambient_temp_c: number;
    soc_start: number;
    soc_target: number;
    q_nom?: number | null;
  }): Promise<LiveSimulationResponse> => {
    const resp = await fetch("/api/charging/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      throw new Error(`Simulation failed (${resp.status})`);
    }
    return resp.json() as Promise<LiveSimulationResponse>;
  },
};
