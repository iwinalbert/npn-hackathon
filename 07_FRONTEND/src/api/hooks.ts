/**
 * Data hooks.
 *
 * One hook per backend endpoint. Every screen reads through these, so there is
 * exactly one place where a query key, a cache policy or a URL lives.
 *
 * Cache policy reflects what the data actually is: the model is frozen and the
 * forecast is a fixed quantity, so almost everything here is immutable for the
 * lifetime of the page. Long stale times are correct, not lazy.
 */
import { useQuery, type UseQueryResult } from '@tanstack/react-query'

import { API_BASE, ApiError, apiFetch, post } from './client'
import type {
  AggregateBacktest, AggregateForecast, BacktestWindow, CapabilityMatrix,
  ErrorBand, HierarchyNode, HorizonPoint, InferenceJob, InferenceStatus,
  LevelAccuracy, LevelInfo, MemberComparison, ModelCard, Occurrence,
  PlanningSummary, PortfolioSummary, Provenance, Readiness, RegimeAccuracy,
  AskRequest, AskResponse, GenAIStatus, GenAISuggestions,
  SeriesBacktest, SeriesDetail, SeriesForecast, SeriesHistory, SeriesSummary,
  TopMovers, VolumeTier,
} from './types'

/** Frozen data: never refetch within a session. */
const FROZEN = { staleTime: Infinity, gcTime: Infinity } as const
/** Live-ish data (readiness, jobs). */
const LIVE = { staleTime: 5_000 } as const

// --- meta -------------------------------------------------------------------

export const useModelCard = (): UseQueryResult<ModelCard> =>
  useQuery({ queryKey: ['model'], queryFn: () => apiFetch<ModelCard>('/meta/model'), ...FROZEN })

export const useCapabilities = (): UseQueryResult<CapabilityMatrix> =>
  useQuery({ queryKey: ['capabilities'], queryFn: () => apiFetch<CapabilityMatrix>('/meta/capabilities'), ...FROZEN })

export const useProvenance = (): UseQueryResult<Provenance> =>
  useQuery({ queryKey: ['provenance'], queryFn: () => apiFetch<Provenance>('/meta/provenance'), ...FROZEN })

export const useReadiness = (): UseQueryResult<Readiness> =>
  useQuery({ queryKey: ['ready'], queryFn: () => apiFetch<Readiness>('/ready'), ...LIVE, retry: 1 })

// --- hierarchy --------------------------------------------------------------

export const useLevels = (): UseQueryResult<LevelInfo[]> =>
  useQuery({ queryKey: ['levels'], queryFn: () => apiFetch<LevelInfo[]>('/hierarchy/levels'), ...FROZEN })

export const useNodes = (level: string, parentLevel?: string, parentId?: string) =>
  useQuery({
    queryKey: ['nodes', level, parentLevel ?? null, parentId ?? null],
    queryFn: () => apiFetch<HierarchyNode[]>('/hierarchy/nodes', {
      level, parent_level: parentLevel, parent_id: parentId, limit: 500,
    }),
    enabled: Boolean(level),
    ...FROZEN,
  })

export const useSearch = (term: string) =>
  useQuery({
    queryKey: ['search', term],
    queryFn: () => apiFetch<SeriesSummary[]>('/hierarchy/search', { q: term, limit: 25 }),
    enabled: term.trim().length >= 2,
    staleTime: 60_000,
  })

export const useAggregate = (level: string, nodeId: string, historyDays = 90) =>
  useQuery({
    queryKey: ['aggregate', level, nodeId, historyDays],
    queryFn: () => apiFetch<AggregateForecast>('/hierarchy/aggregate', {
      level, node_id: nodeId, history_days: historyDays,
    }),
    enabled: Boolean(level && nodeId),
    ...FROZEN,
  })

// --- series -----------------------------------------------------------------

export const useSeriesList = (filters: Record<string, string | undefined>, limit = 100) =>
  useQuery({
    queryKey: ['series-list', filters, limit],
    queryFn: () => apiFetch<SeriesSummary[]>('/series', { ...filters, limit }),
    ...FROZEN,
  })

export const useSeriesDetail = (store?: string, item?: string) =>
  useQuery({
    queryKey: ['series', store, item],
    queryFn: () => apiFetch<SeriesDetail>(`/series/${store}/${item}`),
    enabled: Boolean(store && item),
    ...FROZEN,
  })

export const useSeriesHistory = (store?: string, item?: string, days = 90) =>
  useQuery({
    queryKey: ['history', store, item, days],
    queryFn: () => apiFetch<SeriesHistory>(`/series/${store}/${item}/history`, { days }),
    enabled: Boolean(store && item),
    ...FROZEN,
  })

export const useSeriesForecast = (store?: string, item?: string) =>
  useQuery({
    queryKey: ['forecast', store, item],
    queryFn: () => apiFetch<SeriesForecast>(`/series/${store}/${item}/forecast`),
    enabled: Boolean(store && item),
    ...FROZEN,
  })

// --- accuracy ---------------------------------------------------------------

export const useWindows = (): UseQueryResult<BacktestWindow[]> =>
  useQuery({ queryKey: ['windows'], queryFn: () => apiFetch<BacktestWindow[]>('/accuracy/windows'), ...FROZEN })

export const useLevelAccuracy = () =>
  useQuery({
    queryKey: ['level-accuracy'],
    queryFn: () => apiFetch<{ levels: LevelAccuracy[]; note: string }>('/accuracy/levels'),
    ...FROZEN,
  })

export const useHorizonAccuracy = (origin = 1912) =>
  useQuery({
    queryKey: ['horizon', origin],
    queryFn: () => apiFetch<HorizonPoint[]>('/accuracy/horizon', { origin_idx: origin }),
    ...FROZEN,
  })

export const useRegimeAccuracy = (origin = 1912) =>
  useQuery({
    queryKey: ['regimes', origin],
    queryFn: () => apiFetch<{ origin_idx: number; regimes: RegimeAccuracy[]; note: string }>(
      '/accuracy/regimes', { origin_idx: origin }),
    ...FROZEN,
  })

export const useVolumeTiers = (origin = 1912) =>
  useQuery({
    queryKey: ['volume-tiers', origin],
    queryFn: () => apiFetch<{ tiers: VolumeTier[]; tier_definition: string; note: string }>(
      '/accuracy/volume-tiers', { origin_idx: origin }),
    ...FROZEN,
  })

export const useOccurrence = (origin = 1912) =>
  useQuery({
    queryKey: ['occurrence', origin],
    queryFn: () => apiFetch<Occurrence>('/accuracy/occurrence', { origin_idx: origin }),
    ...FROZEN,
  })

export const useMembers = (origin = 1912) =>
  useQuery({
    queryKey: ['members', origin],
    queryFn: () => apiFetch<MemberComparison>('/accuracy/members', { origin_idx: origin }),
    ...FROZEN,
  })

export const useErrorBands = (regime?: string) =>
  useQuery({
    queryKey: ['error-bands', regime ?? 'all'],
    queryFn: () => apiFetch<{ bands: ErrorBand[]; basis: string; measured_coverage: number; disclaimer: string }>(
      '/accuracy/error-bands', { regime }),
    ...FROZEN,
  })

export const useSeriesBacktest = (store?: string, item?: string, origin = 1912) =>
  useQuery({
    queryKey: ['series-backtest', store, item, origin],
    queryFn: () => apiFetch<SeriesBacktest>(`/accuracy/backtest/${store}/${item}`, { origin_idx: origin }),
    enabled: Boolean(store && item),
    ...FROZEN,
  })

export const useAggregateBacktest = (level: string, nodeId: string, origin = 1912) =>
  useQuery({
    queryKey: ['agg-backtest', level, nodeId, origin],
    queryFn: () => apiFetch<AggregateBacktest>('/accuracy/backtest', {
      level, node_id: nodeId, origin_idx: origin,
    }),
    enabled: Boolean(level && nodeId),
    ...FROZEN,
  })

// --- insights ---------------------------------------------------------------

export const usePortfolio = (level = 'total', nodeId = 'ALL') =>
  useQuery({
    queryKey: ['portfolio', level, nodeId],
    queryFn: () => apiFetch<PortfolioSummary>('/insights/summary', { level, node_id: nodeId }),
    ...FROZEN,
  })

export const useTopMovers = (level = 'total', nodeId = 'ALL', limit = 15) =>
  useQuery({
    queryKey: ['movers', level, nodeId, limit],
    queryFn: () => apiFetch<TopMovers>('/insights/top-movers', { level, node_id: nodeId, limit }),
    ...FROZEN,
  })

export const usePlanning = (store?: string, item?: string) =>
  useQuery({
    queryKey: ['planning', store, item],
    queryFn: () => apiFetch<PlanningSummary>(`/insights/planning/${store}/${item}`),
    enabled: Boolean(store && item),
    ...FROZEN,
  })

// --- inference --------------------------------------------------------------

export const useInferenceStatus = () =>
  useQuery({ queryKey: ['inference-status'], queryFn: () => apiFetch<InferenceStatus>('/inference/status'), ...LIVE })

export const startVerification = () => post<{ job_id: string }>('/inference/verify')

export const useInferenceJob = (jobId: string | null) =>
  useQuery({
    queryKey: ['inference-job', jobId],
    queryFn: () => apiFetch<InferenceJob>(`/inference/jobs/${jobId}`),
    enabled: Boolean(jobId),
    // Poll while the job is in flight; stop as soon as it settles.
    refetchInterval: (q) => {
      const s = (q.state.data as InferenceJob | undefined)?.status
      return s === 'succeeded' || s === 'failed' ? false : 1500
    },
  })

// --- AI assistant -----------------------------------------------------------

export const useGenAIStatus = () =>
  useQuery({
    queryKey: ['genai-status'],
    queryFn: () => apiFetch<GenAIStatus>('/genai/status'),
    staleTime: 60_000,
    retry: 1,
  })

export const useGenAISuggestions = (store?: string | null, item?: string | null) =>
  useQuery({
    queryKey: ['genai-suggestions', store ?? null, item ?? null],
    queryFn: () => apiFetch<GenAISuggestions>('/genai/suggestions', {
      store_id: store ?? undefined, item_id: item ?? undefined,
    }),
    staleTime: Infinity,
  })

/**
 * Ask the assistant.
 *
 * A plain async function rather than a mutation hook: the page owns the
 * conversation list, and TanStack's mutation state would duplicate it.
 */
export async function askAssistant(body: AskRequest): Promise<AskResponse> {
  const res = await fetch(`${API_BASE}/genai/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    let parsed: Record<string, unknown> = {}
    try { parsed = await res.json() } catch { /* non-JSON error body */ }
    throw new ApiError(
      res.status,
      (parsed.error as string) ?? 'http_error',
      (parsed.message as string) ?? res.statusText,
      parsed.context as Record<string, unknown> | undefined,
      parsed.request_id as string | undefined,
    )
  }
  return (await res.json()) as AskResponse
}
