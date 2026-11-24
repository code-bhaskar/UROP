export interface ChargingMetrics {
  chemistry: "li_ion" | "na_ion";
  scenario: string;
  soh_true: number;
  soh_est: number;
  strategy: "CC-CV" | "Adaptive_ML";
  charge_time_min: number;
  peak_temp_c: number;
  peak_temp_rise_c: number;
  mean_temp_rise_c: number;
  energy_efficiency_pct: number;
  estimated_degradation_pct: number;
  soc_observer_mae: number;
  max_voltage_v: number;
  safety_limits_respected: boolean;
}

export interface ChargingSeries {
  time_min: number[];
  soc_true: number[];
  soc_est: number[];
  voltage_v: number[];
  current_c_rate: number[];
  temp_c: number[];
}

export interface ChargingTraceData {
  chemistry: "li_ion" | "na_ion";
  scenario: string;
  strategy: "CC-CV" | "Adaptive_ML";
  metrics: ChargingMetrics;
  series: ChargingSeries;
}

export interface LiveSimulationResponse {
  request: {
    chemistry: "li_ion" | "na_ion";
    soh_true: number;
    ambient_temp_c: number;
    soc_start: number;
    soc_target: number;
    q_nom: number | null;
    observer_noise_std: number;
  };
  spec: {
    name: string;
    v_min: number;
    v_max: number;
    v_nom: number;
    q_nom: number;
    t_max_safe: number;
  };
  cc_cv: ChargingTraceData;
  adaptive_ml: ChargingTraceData;
  deltas: {
    temp_rise_reduction_pct: number;
    degradation_reduction_pct: number;
    efficiency_gain_pct: number;
    charge_time_delta_min: number;
  };
}

export interface PcaTaskCloud {
  raw_li: [number, number, number][];
  raw_na: [number, number, number][];
  aligned_li: [number, number, number][];
  aligned_na: [number, number, number][];
  raw_explained_var: number[];
  aligned_explained_var: number[];
}

export interface DomainShiftRow {
  task: "SoC" | "SoH";
  representation: string;
  mmd2: number;
  mean_wasserstein_1: number;
  mean_psi: number;
}

export interface WithinRow {
  task: "SoC" | "SoH";
  chemistry: "li_ion" | "na_ion";
  model: string;
  mae: number;
  rmse: number;
  r2: number;
  train_time_s: number;
  infer_time_ms: number;
}

export interface CrossRow {
  task: "SoC" | "SoH";
  direction: "li_ion_to_na_ion" | "na_ion_to_li_ion";
  tier: string;
  model: string;
  mae: number;
  rmse: number;
  r2: number;
}

export interface TransferRow {
  task: "SoC" | "SoH";
  direction: "li_ion_to_na_ion" | "na_ion_to_li_ion";
  target_fraction: number;
  n_target_samples: number;
  method: string;
  train_time_s: number;
  mae: number;
  rmse: number;
  r2: number;
}

export interface OverviewSummary {
  data_mode: string;
  elapsed_seconds: number;
  dataset_inventory: {
    total_files: number;
    pulled_files: number;
    pointer_files: number;
    total_declared_mb: number;
  };
  quality_audit: {
    timeseries: {
      bronze_rows: number;
      silver_rows: number;
      duplicates_removed: number;
      bounds_and_null_removed: number;
      isolation_forest_anomalies_removed: number;
      retention_rate_pct: number;
    };
    cycles: {
      bronze_rows: number;
      silver_rows: number;
      removed: number;
    };
  };
  duckdb_analytics: {
    chemistry_summary: Record<string, string | number>[];
    aging_summary: Record<string, string | number>[];
  };
  data_mining: {
    kmeans_silhouette: number;
    domain_shift: DomainShiftRow[];
    top_mi_soc_li: { feature: string; mutual_information: number }[];
    top_mi_soc_na: { feature: string; mutual_information: number }[];
    top_mi_soh_li: { feature: string; mutual_information: number }[];
    top_mi_soh_na: { feature: string; mutual_information: number }[];
  };
  unit_counts?: {
    SoC?: { li_ion?: number; na_ion?: number };
    SoH?: { li_ion?: number; na_ion?: number };
  };
  evaluation_modes?: {
    SoC_cross_chemistry?: string;
    SoH_cross_chemistry?: string;
    na_soc_within?: string;
    na_soh_within?: string;
  };
  compatibility_notes?: string[];
}
