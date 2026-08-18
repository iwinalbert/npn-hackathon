import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import type { ReactNode } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { Overview } from '../pages/Overview'
import { AppShell } from '../components/layout/AppShell'

const MODEL = {
  model_name: 'Direct+Recursive Tweedie Blend',
  blend_formula: '0.60*direct(38f) + 0.40*recursive(32f)',
  blend_weight_direct: 0.6, blend_weight_recursive: 0.4,
  objective: 'tweedie', n_estimators: 400, seed: 42, status: 'FROZEN',
  validation_rmse: 2.0929, validation_mae: 1.0395,
  validation_window: 'd_1914-d_1941', validation_n: 853720,
  forecast_origin: 'd_1941', forecast_dates: '2016-05-23 .. 2016-06-19',
  horizon_days: 28, n_series: 30490,
  model_direct_sha256: 'a'.repeat(64), model_recursive_sha256: 'b'.repeat(64),
  forecast_sha256: 'c'.repeat(64), db_built_at: '2026-08-15T00:00:00Z',
}

const LEVELS = {
  levels: [
    { level: 'L1_total', n_groups: 1, rmse: 1589, mae: 1246, wape: 0.0283, accuracy_pct: 97.2 },
    { level: 'L3_store', n_groups: 10, rmse: 305, mae: 233, wape: 0.053, accuracy_pct: 94.7 },
    { level: 'L10_item', n_groups: 3049, rmse: 8.5, mae: 4.2, wape: 0.292, accuracy_pct: 70.8 },
    { level: 'L12_store_item', n_groups: 30490, rmse: 2.0929, mae: 1.0395, wape: 0.7205, accuracy_pct: 28.0 },
  ],
  note: 'Always quote the level that matches the decision.',
}

const OCCURRENCE = {
  origin_idx: 1912, origin_day: 'd_1913', window: 'x', threshold: 0.5,
  rule: 'actual event = actual sales > 0', n: 853720,
  confusion_matrix: { true_positive: 313824, false_positive: 182682, false_negative: 75171, true_negative: 282043 },
  accuracy: 0.698, precision: 0.6321, recall: 0.8068, f1: 0.7088,
  base_rate: 0.4556, always_predict_no_demand_accuracy: 0.5444,
  caveat: 'The model was never trained to classify.',
}

const PROVENANCE = {
  model_direct_sha256: 'a'.repeat(64), model_recursive_sha256: 'b'.repeat(64),
  forecast_sha256: 'c'.repeat(64), db_built_at: '2026-08-15T00:00:00Z',
  row_counts: { backtest: 6829760, forecast: 853720, series: 30490 },
  backtest_origins: ['d_1601', 'd_1629', 'd_1679', 'd_1707', 'd_1750', 'd_1778', 'd_1885', 'd_1913'],
  sources: { forecast: 'predictions/final_forecast/...' },
}

const READY = {
  ready: true, degraded: false, environment: 'test', version: '2.0.0',
  detail: { data_dir: '', artefacts: {}, tables: { series: 30490 }, history_queryable: true, backtest_queryable: true, errors: [] },
  cache: { entries: 0, live: 0 },
}

function routeFor(url: string): unknown {
  if (url.includes('/meta/model')) return MODEL
  if (url.includes('/meta/provenance')) return PROVENANCE
  if (url.includes('/accuracy/levels')) return LEVELS
  if (url.includes('/accuracy/occurrence')) return OCCURRENCE
  if (url.includes('/ready')) return READY
  return {}
}

function mockApi(failing = false) {
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    if (failing) {
      return {
        ok: false, status: 503,
        statusText: 'Service Unavailable',
        json: async () => ({ error: 'service_unavailable', message: 'not built' }),
        headers: { get: () => null },
      }
    }
    return {
      ok: true, status: 200, statusText: 'OK',
      json: async () => routeFor(url),
      headers: { get: () => null },
    }
  }))
}

function wrap(node: ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>{node}</MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('Overview page', () => {
  beforeEach(() => vi.unstubAllGlobals())

  it('shows a loading state before data arrives', () => {
    mockApi()
    wrap(<Overview />)
    expect(screen.getAllByRole('status').length).toBeGreaterThan(0)
  })

  it('renders the product name and subtitle for judges', async () => {
    mockApi()
    wrap(<Overview />)
    expect(screen.getByRole('heading', { name: /Retail Demand Forecasting/i })).toBeInTheDocument()
    expect(screen.getByText(/Walmart M5 · Hierarchical 28-Day Forecasting/i)).toBeInTheDocument()
  })

  it('renders EXACTLY the metrics the API returned, not rounded guesses', async () => {
    mockApi()
    wrap(<Overview />)
    await waitFor(() => expect(screen.getByText('2.0929')).toBeInTheDocument())
    expect(screen.getByText('1.0395')).toBeInTheDocument()
    expect(screen.getByText('30,490')).toBeInTheDocument()
    expect(screen.getByText('FROZEN')).toBeInTheDocument()
  })

  it('shows occurrence metrics as computed, without inflating them', async () => {
    mockApi()
    wrap(<Overview />)
    await waitFor(() => expect(screen.getByText('69.8%')).toBeInTheDocument())
    expect(screen.getByText('80.7%')).toBeInTheDocument()
    expect(screen.getByText('63.2%')).toBeInTheDocument()
  })

  it('carries the backend caveat about occurrence metrics to the screen', async () => {
    mockApi()
    wrap(<Overview />)
    await waitFor(() =>
      expect(screen.getByText(/never trained to classify/i)).toBeInTheDocument())
  })

  it('shows accuracy at several levels rather than one global number', async () => {
    mockApi()
    wrap(<Overview />)
    await waitFor(() => expect(screen.getByText('97.2%')).toBeInTheDocument())
    expect(screen.getByText('94.7%')).toBeInTheDocument()
    expect(screen.getByText('28.0%')).toBeInTheDocument()
  })

  it('renders an error state and NO fabricated numbers when the API fails', async () => {
    mockApi(true)
    wrap(<Overview />)
    await waitFor(() => expect(screen.getAllByRole('alert').length).toBeGreaterThan(0))
    expect(screen.queryByText('2.0929')).not.toBeInTheDocument()
    expect(screen.queryByText('97.2%')).not.toBeInTheDocument()
    expect(screen.getAllByText(/not ready/i).length).toBeGreaterThan(0)
  })
})

describe('AppShell navigation', () => {
  beforeEach(() => vi.unstubAllGlobals())

  it('exposes every top-level destination as a link', async () => {
    mockApi()
    wrap(<AppShell />)
    const names = screen.getAllByRole('link').map((el) => el.textContent ?? '')
    for (const label of ['Overview', 'Forecast', 'Hierarchy', 'Insights',
                         'AI Assistant']) {
      expect(names.some((n) => n.startsWith(label))).toBe(true)
    }
  })

  it('does not advertise routes the app no longer serves', async () => {
    mockApi()
    wrap(<AppShell />)
    const hrefs = screen.getAllByRole('link').map((el) => el.getAttribute('href'))
    for (const gone of ['/accuracy', '/validation', '/model', '/methodology']) {
      expect(hrefs).not.toContain(gone)
    }
  })

  it('shows the product branding in the header', async () => {
    mockApi()
    wrap(<AppShell />)
    expect(screen.getByRole('heading', { name: 'Retail Demand Forecasting' })).toBeInTheDocument()
  })

  it('reports API status instead of silently pretending to be healthy', async () => {
    mockApi(true)
    wrap(<AppShell />)
    await waitFor(
      () => expect(screen.getByText(/API offline/i)).toBeInTheDocument(),
      { timeout: 5000 },
    )
  })

  it('has a labelled main navigation landmark', async () => {
    mockApi()
    wrap(<AppShell />)
    expect(screen.getByRole('navigation', { name: /main/i })).toBeInTheDocument()
  })
})
