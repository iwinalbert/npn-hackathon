/** Number and date formatting. Pure functions — unit-tested. */

export const nf = (v: number, digits = 0) =>
  v.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })

/** Compact units for headline figures: 1309021 -> "1.31M". */
export function compact(v: number): string {
  const a = Math.abs(v)
  if (a >= 1e9) return `${(v / 1e9).toFixed(2)}B`
  if (a >= 1e6) return `${(v / 1e6).toFixed(2)}M`
  if (a >= 1e3) return `${(v / 1e3).toFixed(1)}K`
  return nf(v, a < 10 && a % 1 !== 0 ? 2 : 0)
}

export const pct = (v: number, digits = 1) => `${v.toFixed(digits)}%`

export const signed = (v: number, digits = 2) =>
  `${v >= 0 ? '+' : ''}${v.toFixed(digits)}`

/** "2016-05-23" -> "23 May" */
export function shortDate(iso: string): string {
  const d = new Date(`${iso}T00:00:00Z`)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', timeZone: 'UTC' })
}

/** "2016-05-23" -> "23 May 2016" */
export function longDate(iso: string): string {
  const d = new Date(`${iso}T00:00:00Z`)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleDateString('en-GB', {
    day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC',
  })
}

/** "HOBBIES_1_001" -> "Hobbies 1 001", for display without losing the id. */
export const humanise = (id: string) =>
  id.toLowerCase().replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())

export const REGIME_COLORS: Record<string, string> = {
  smooth: '#3FB98B',
  erratic: '#D9A63C',
  intermittent: '#4EA8F0',
  lumpy: '#E0665F',
  'never sold': '#6B7C99',
}

export const TIER_ORDER = ['very low', 'low', 'medium', 'high']

/**
 * One day on the demand chart.
 *
 * Two series share the x-axis, and a day carries only what the API actually
 * supplies for it:
 *
 *   'history'   observed actual only — before the compared window
 *   'compared'  actual AND prediction — the held-out backtest window, the only
 *               stretch where the two can honestly be put side by side
 *   'forecast'  prediction only — the delivered 28 days, which have not
 *               happened yet and so have no actual to compare against
 *
 * Nothing is carried across a phase boundary: a value the API did not supply
 * stays null and the chart draws a gap rather than a plausible-looking line.
 */
export interface DemandPoint {
  date: string
  actual: number | null
  predicted: number | null
  lower: number | null
  upper: number | null
  /** Recharts needs an explicit [low, high] pair to render a band area. */
  band: [number, number] | null
  phase: 'history' | 'compared' | 'forecast'
}

export interface DemandSeriesInput {
  /** Observed daily sales up to the forecast origin. */
  history?: { date: string; sales: number }[]
  /** Held-out window where the model's prediction and the actual both exist. */
  backtest?: { date: string; actual: number; predicted: number }[]
  /** The delivered forecast: prediction only. */
  forecast?: { date: string; yhat: number; lower?: number | null; upper?: number | null }[]
}

/** Merge the three sources into one date-ordered series for charting. */
export function buildDemandSeries(
  { history = [], backtest = [], forecast = [] }: DemandSeriesInput,
): DemandPoint[] {
  const byDate = new Map<string, DemandPoint>()

  const at = (date: string): DemandPoint => {
    let p = byDate.get(date)
    if (!p) {
      p = {
        date, actual: null, predicted: null, lower: null, upper: null,
        band: null, phase: 'history',
      }
      byDate.set(date, p)
    }
    return p
  }

  for (const h of history) at(h.date).actual = h.sales

  for (const b of backtest) {
    const p = at(b.date)
    p.actual = b.actual
    p.predicted = b.predicted
    p.phase = 'compared'
  }

  for (const f of forecast) {
    const p = at(f.date)
    p.predicted = f.yhat
    p.lower = f.lower ?? null
    p.upper = f.upper ?? null
    p.band = f.lower != null && f.upper != null ? [f.lower, f.upper] : null
    p.phase = 'forecast'
  }

  return [...byDate.values()].sort((a, b) => a.date.localeCompare(b.date))
}
