/**
 * Typed API responses.
 *
 * These mirror the FastAPI OpenAPI contract at 06_BACKEND/openapi.json. They are
 * hand-written rather than generated so the fields the UI actually depends on
 * are explicit and reviewable — several of them (`band_basis`, `expected_accuracy`,
 * `caveat`) exist specifically to carry an honesty statement from the backend to
 * the screen, and they must never be silently dropped.
 */

export interface ModelCard {
  model_name: string
  blend_formula: string
  blend_weight_direct: number
  blend_weight_recursive: number
  objective: string
  n_estimators: number
  seed: number
  status: string
  validation_rmse: number
  validation_mae: number
  validation_window: string
  validation_n: number
  forecast_origin: string
  forecast_dates: string
  horizon_days: number
  n_series: number
  model_direct_sha256: string
  model_recursive_sha256: string
  forecast_sha256: string
  db_built_at: string
}

export type CapabilityCategory = 'implemented' | 'rejected' | 'not_supported'

export interface Capability {
  name: string
  category: CapabilityCategory
  detail: string
  evidence: string | null
}

export interface CapabilityMatrix {
  implemented: Capability[]
  rejected: Capability[]
  not_supported: Capability[]
}

export interface Provenance {
  model_direct_sha256: string
  model_recursive_sha256: string
  forecast_sha256: string
  db_built_at: string
  row_counts: Record<string, number>
  backtest_origins: string[]
  sources: Record<string, string>
}

export interface Readiness {
  ready: boolean
  degraded: boolean
  environment: string
  version: string
  detail: {
    data_dir: string
    artefacts: Record<string, { path: string; exists: boolean; size_mb: number | null }>
    tables: Record<string, number>
    history_queryable: boolean
    backtest_queryable: boolean
    errors: string[]
  }
  cache: { entries: number; live: number }
}

export interface LevelInfo {
  level: string
  label: string
  node_count: number
  columns: string[]
}

export interface HierarchyNode {
  level: string
  node_id: string
  label: string
  n_series: number
  mean_daily_sales: number | null
}

export interface SeriesSummary {
  series_idx: number
  id: string
  item_id: string
  dept_id: string
  cat_id: string
  store_id: string
  state_id: string
  volume_tier: string
  regime: string
  mean_daily_sales: number
  zero_pct: number
}

export interface SeriesDetail extends SeriesSummary {
  total_units: number
  adi: number
  cv2: number
  regime_explanation: string
}

export interface HistoryPoint {
  date: string
  day_idx: number
  sales: number
  sell_price: number | null
  event_name: string | null
  snap: number
}

export interface SeriesHistory {
  series: SeriesDetail
  history: HistoryPoint[]
  from_date: string
  to_date: string
}

export interface ForecastPoint {
  date: string
  day_idx: number
  horizon: number
  yhat: number
  /** Empirical backtest error band. NOT a model-produced interval. */
  lower: number | null
  upper: number | null
}

export interface SeriesForecast {
  series: SeriesDetail
  origin_day: string
  origin_date: string
  forecast: ForecastPoint[]
  total_28d: number
  /** The backend's own statement of what the band is. Always render it. */
  band_basis: string | null
  band_regime: string | null
}

export interface MeasuredAccuracy {
  measured_level: string
  n_groups: number
  rmse: number
  mae: number
  wape: number
  accuracy_pct: number
  basis: string
}

export interface AggregateForecast {
  level: string
  node_id: string
  n_series: number
  origin_day: string
  forecast: { date: string; day_idx: number; horizon: number; yhat: number }[]
  total_28d: number
  expected_accuracy: MeasuredAccuracy | null
  history?: { date: string; day_idx: number; sales: number }[]
}

export interface BacktestWindow {
  origin_idx: number
  origin_day: string
  origin_date: string
  window_start: string
  window_end: string
  n_predictions: number
  rmse: number
  mae: number
  wape: number
  bias: number
  accuracy_pct: number
  rmse_direct_member: number
  rmse_recursive_member: number
  member_residual_correlation: number
  is_primary_validation_window: boolean
}

export interface LevelAccuracy {
  level: string
  n_groups: number
  rmse: number
  mae: number
  wape: number
  accuracy_pct: number
}

export interface HorizonPoint {
  horizon: number
  rmse: number
  mae: number
  bias: number
  mean_actual: number
}

export interface RegimeAccuracy {
  regime: string
  n: number
  n_series: number
  rmse: number
  mae: number
  bias: number
  mean_actual: number
  zero_rate_pct: number
  share_of_squared_error_pct: number
}

export interface VolumeTier {
  volume_tier: string
  n: number
  n_series: number
  rmse: number
  mae: number
  bias: number
  mean_actual: number
  share_of_squared_error_pct: number
  share_of_rows_pct: number
}

export interface Occurrence {
  origin_idx: number
  origin_day: string
  window: string
  threshold: number
  rule: string
  n: number
  confusion_matrix: {
    true_positive: number
    false_positive: number
    false_negative: number
    true_negative: number
  }
  accuracy: number
  precision: number
  recall: number
  f1: number
  base_rate: number
  always_predict_no_demand_accuracy: number
  caveat: string
}

export interface MemberComparison {
  origin_idx: number
  origin_day: string
  window: string
  members: { name: string; weight: number; rmse: number; mae: number }[]
  blend: { name: string; rmse: number; mae: number }
  gain_vs_best_member: number
  residual_correlation: number
  why_it_works: string
}

export interface SeriesBacktest {
  series: Partial<SeriesSummary>
  origin_idx: number
  origin_day: string
  window: string
  points: {
    date: string
    day_idx: number
    horizon: number
    actual: number
    predicted: number
    predicted_direct: number
    predicted_recursive: number
    error: number
  }[]
  rmse: number
  mae: number
  total_actual: number
  total_predicted: number
  basis: string
}

export interface AggregateBacktest {
  level: string
  node_id: string
  n_series: number
  origin_idx: number
  origin_day: string
  window: string
  points: {
    date: string
    day_idx: number
    horizon: number
    actual: number
    predicted: number
    error: number
  }[]
  total_actual: number
  total_predicted: number
  wape: number | null
  accuracy_pct: number | null
  note: string
}

export interface Mover {
  series_idx: number
  id: string
  item_id: string
  store_id: string
  dept_id: string
  cat_id: string
  regime: string
  volume_tier: string
  forecast_total_28d: number
  forecast_daily: number
  recent_daily_28d: number
  delta_daily: number
  delta_28d: number
  delta_pct: number | null
}

export interface TopMovers {
  level: string
  node_id: string
  n_series_considered: number
  basis: string
  rising?: Mover[]
  falling?: Mover[]
}

export interface PortfolioSummary {
  level: string
  node_id: string
  n_series: number
  forecast_total_28d: number
  forecast_daily_avg: number
  recent_total_28d: number
  change_vs_recent: number
  change_pct: number | null
  avg_zero_day_pct: number
  regime_mix: { regime: string; n_series: number }[]
  expected_accuracy: MeasuredAccuracy | null
}

export interface PlanningSummary {
  series: SeriesDetail
  horizon_days: number
  expected_total: number
  expected_daily: number
  planning_range: { low: number; high: number; basis: string }
  recent_28d_actual: number | null
  change_vs_recent: number | null
  weekly_breakdown: { week: number; days: string; expected: number }[]
  regime: string
  caveats: string[]
}

export interface InferenceStatus {
  available: boolean
  enabled: boolean
  reasons: string[]
  frozen_origin: string
  models_cached: boolean
  estimated_runtime_seconds: number
  supported_operations: string[]
  refused_operations: Record<string, string>
  runtime: { loaded: boolean; [k: string]: unknown }
  jobs: {
    running: number
    max_concurrent: number
    tracked_jobs: number
    durable: boolean
    note: string
  }
}

export interface InferenceJob {
  job_id: string
  kind: string
  status: 'queued' | 'running' | 'succeeded' | 'failed'
  progress: number
  message: string
  submitted_at: number
  started_at: number | null
  finished_at: number | null
  result: VerificationResult | null
  error: { type: string; message: string; traceback_tail: string } | null
}

export interface VerificationResult {
  verdict: 'MATCH' | 'MISMATCH'
  tolerance: number
  max_abs_diff: number
  mean_abs_diff: number
  n_predictions: number
  n_series: number
  horizon: number
  origin_day: string
  blend_weight_direct: number
  recomputed_total: number
  artefact_total: number
  leakage_checks: Record<string, boolean>
  timings_seconds: Record<string, number>
  models: Record<string, unknown>
  interpretation: string
}

export interface ErrorBand {
  regime: string
  horizon: number
  q05: number
  q25: number
  q50: number
  q75: number
  q95: number
  n: number
  normalised_sd: number
}

// --------------------------------------------------------------------------
// AI Forecast Assistant
//
// The assistant is an explanatory layer. `grounded` and `ungrounded_numbers`
// are the anti-hallucination signal: the backend checks every figure in the
// answer against the context it supplied, and the UI must surface a failure
// rather than hide it.
// --------------------------------------------------------------------------

export interface GenAIStatus {
  available: boolean
  enabled: boolean
  provider: string
  model: string | null
  reasons: string[]
  key_configured: boolean
  max_question_chars: number
  guarantees: string[]
  refusals: Record<string, string>
}

export interface GenAISuggestions {
  context: string
  suggestions: string[]
}

export interface AskRequest {
  question: string
  store_id?: string | null
  item_id?: string | null
  level?: string
  node_id?: string
}

export interface AskResponse {
  answer: string
  intent: string
  model: string
  grounded: boolean
  ungrounded_numbers: number[]
  injection_suspected: boolean
  context_keys: string[]
  elapsed_ms: number
  disclaimer: string
  /** True when the reply hit the output token limit and was cut short. */
  truncated?: boolean
  /** True when local policy answered and no AI provider was called at all. */
  refused?: boolean
  /** secret_extraction | instruction_override | forecast_mutation | model_mutation */
  refusal_category?: string | null
}
