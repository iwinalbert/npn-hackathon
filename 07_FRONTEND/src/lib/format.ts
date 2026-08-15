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
 * Merge observed history and forecast into one continuous series for charting.
 *
 * The two must share an x-axis but stay visually distinct, so each point
 * carries EITHER `actual` OR `forecast`, never both — except at the origin,
 * which is duplicated into both so the lines join instead of showing a gap.
 */
export interface CombinedPoint {
  date: string
  actual: number | null
  forecast: number | null
  lower: number | null
  upper: number | null
  /** Recharts needs an explicit [low, high] pair to render a band area. */
  band: [number, number] | null
  isForecast: boolean
}

export function combineHistoryAndForecast(
  history: { date: string; sales: number }[],
  forecast: { date: string; yhat: number; lower?: number | null; upper?: number | null }[],
): CombinedPoint[] {
  const out: CombinedPoint[] = history.map((h) => ({
    date: h.date,
    actual: h.sales,
    forecast: null,
    lower: null,
    upper: null,
    band: null,
    isForecast: false,
  }))

  // Join the lines: the last observed point also seeds the forecast line.
  const last = history.at(-1)
  if (last && out.length > 0) {
    out[out.length - 1] = { ...out[out.length - 1], forecast: last.sales }
  }

  for (const f of forecast) {
    out.push({
      date: f.date,
      actual: null,
      forecast: f.yhat,
      lower: f.lower ?? null,
      upper: f.upper ?? null,
      band: f.lower != null && f.upper != null ? [f.lower, f.upper] : null,
      isForecast: true,
    })
  }
  return out
}
