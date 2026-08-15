import { useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'

import {
  usePlanning, useSearch, useSeriesForecast, useSeriesHistory,
} from '../api/hooks'
import { ForecastChart } from '../components/charts/ForecastChart'
import {
  Async, Badge, Card, Caveat, EmptyState, ErrorState, Explain, Metric, Spinner,
} from '../components/ui'
import {
  REGIME_COLORS, combineHistoryAndForecast, compact, humanise, longDate, nf, pct, signed,
} from '../lib/format'

const HISTORY_OPTIONS = [
  { days: 30, label: '30d' },
  { days: 90, label: '90d' },
  { days: 180, label: '6m' },
  { days: 365, label: '1y' },
]

/** Sensible default so the page is never empty on first load. */
const DEFAULT = { store: 'CA_3', item: 'FOODS_3_090' }

function SeriesSearch({
  onPick, current,
}: { onPick: (store: string, item: string) => void; current: string }) {
  const [term, setTerm] = useState('')
  const [debounced, setDebounced] = useState('')

  useEffect(() => {
    const t = setTimeout(() => setDebounced(term), 250)
    return () => clearTimeout(t)
  }, [term])

  const results = useSearch(debounced)

  return (
    <div>
      <label htmlFor="series-search" className="block text-xs font-medium text-ink-muted">
        Find a product or store
      </label>
      <input
        id="series-search"
        type="search"
        value={term}
        onChange={(e) => setTerm(e.target.value)}
        placeholder="e.g. FOODS_3_090, HOBBIES, CA_3"
        autoComplete="off"
        className="mt-1.5 w-full rounded border border-line bg-base px-3 py-2 text-sm
                   text-ink placeholder:text-ink-dim transition-colors
                   focus:border-forecast"
      />

      <div className="mt-2 min-h-[1.25rem]">
        {results.isFetching && <Spinner label="Searching" />}
        {debounced.length >= 2 && !results.isFetching && results.data?.length === 0 && (
          <p className="text-xs text-ink-dim">No match for “{debounced}”.</p>
        )}
      </div>

      {results.data && results.data.length > 0 && (
        <ul className="mt-1 max-h-72 divide-y divide-line overflow-y-auto rounded border border-line">
          {results.data.map((s) => {
            const key = `${s.store_id}/${s.item_id}`
            const active = key === current
            return (
              <li key={s.series_idx}>
                <button
                  type="button"
                  onClick={() => onPick(s.store_id, s.item_id)}
                  aria-current={active ? 'true' : undefined}
                  className={`flex w-full items-center justify-between gap-3 px-3 py-2 text-left
                              transition-colors ${
                                active ? 'bg-accentSoft' : 'hover:bg-elevated'
                              }`}
                >
                  <span className="min-w-0">
                    <span className="block truncate text-xs font-medium text-ink">
                      {s.item_id}
                    </span>
                    <span className="block text-[11px] text-ink-dim">
                      {s.store_id} · {s.dept_id}
                    </span>
                  </span>
                  <span className="tnum shrink-0 text-[11px] text-ink-muted">
                    {s.mean_daily_sales.toFixed(1)}/day
                  </span>
                </button>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

export function Forecast() {
  const [params, setParams] = useSearchParams()
  const store = params.get('store') ?? DEFAULT.store
  const item = params.get('item') ?? DEFAULT.item
  const [days, setDays] = useState(90)

  const history = useSeriesHistory(store, item, days)
  const forecast = useSeriesForecast(store, item)
  const planning = usePlanning(store, item)

  const pick = (s: string, i: string) => setParams({ store: s, item: i })

  const combined = useMemo(() => {
    if (!history.data || !forecast.data) return []
    return combineHistoryAndForecast(
      history.data.history.map((h) => ({ date: h.date, sales: h.sales })),
      forecast.data.forecast.map((f) => ({
        date: f.date, yhat: f.yhat, lower: f.lower, upper: f.upper,
      })),
    )
  }, [history.data, forecast.data])

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-ink">Forecast</h1>
        <p className="mt-1 text-sm text-ink-muted">
          Observed demand and the next 28 days, for one product in one store.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[18rem_1fr]">
        {/* --- selection -------------------------------------------- */}
        <div className="space-y-4">
          <Card title="Select a series">
            <SeriesSearch onPick={pick} current={`${store}/${item}`} />
          </Card>

          {forecast.data && (
            <Card title="Series profile">
              <dl className="space-y-3 text-xs">
                <div>
                  <dt className="text-ink-dim">Product</dt>
                  <dd className="mt-0.5 font-medium text-ink">{humanise(item)}</dd>
                  <dd className="font-mono text-[10px] text-ink-dim">{item}</dd>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <dt className="text-ink-dim">Store</dt>
                    <dd className="mt-0.5 font-medium text-ink">{store}</dd>
                  </div>
                  <div>
                    <dt className="text-ink-dim">Department</dt>
                    <dd className="mt-0.5 font-medium text-ink">
                      {forecast.data.series.dept_id}
                    </dd>
                  </div>
                </div>
                <div>
                  <dt className="text-ink-dim">Demand pattern</dt>
                  <dd className="mt-1">
                    <span
                      className="inline-flex items-center gap-1.5 rounded border px-2 py-0.5 text-[11px] font-medium"
                      style={{
                        borderColor: `${REGIME_COLORS[forecast.data.series.regime]}66`,
                        color: REGIME_COLORS[forecast.data.series.regime],
                      }}
                    >
                      {forecast.data.series.regime}
                    </span>
                  </dd>
                  <dd className="mt-1.5 leading-relaxed text-ink-muted">
                    {forecast.data.series.regime_explanation}
                  </dd>
                </div>
                <div className="grid grid-cols-2 gap-3 border-t border-line pt-3">
                  <div>
                    <dt className="text-ink-dim">Days with no sale</dt>
                    <dd className="tnum mt-0.5 font-medium text-ink">
                      {pct(forecast.data.series.zero_pct)}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-ink-dim">Avg daily</dt>
                    <dd className="tnum mt-0.5 font-medium text-ink">
                      {forecast.data.series.mean_daily_sales.toFixed(2)}
                    </dd>
                  </div>
                </div>
              </dl>
            </Card>
          )}
        </div>

        {/* --- chart ------------------------------------------------- */}
        <div className="space-y-6">
          <Card
            title={`${item} · ${store}`}
            subtitle={
              forecast.data
                ? `History up to ${longDate(forecast.data.origin_date)}, then 28 forecast days`
                : undefined
            }
            actions={
              <div className="flex items-center gap-3">
                <Link
                  to={`/assistant?store=${store}&item=${item}`}
                  className="rounded border border-forecast/50 bg-forecast/10 px-2.5 py-1
                             text-[11px] font-medium text-forecast transition-colors
                             hover:bg-forecast/20"
                >
                  Ask AI about this forecast
                </Link>
              <div className="flex gap-1" role="group" aria-label="History range">
                {HISTORY_OPTIONS.map((o) => (
                  <button
                    key={o.days}
                    type="button"
                    onClick={() => setDays(o.days)}
                    aria-pressed={days === o.days}
                    className={`rounded border px-2 py-1 text-[11px] font-medium transition-colors ${
                      days === o.days
                        ? 'border-forecast/50 bg-forecast/10 text-forecast'
                        : 'border-line text-ink-muted hover:bg-elevated'
                    }`}
                  >
                    {o.label}
                  </button>
                ))}
              </div>
              </div>
            }
          >
            {history.isLoading || forecast.isLoading ? (
              <div className="space-y-3">
                <Spinner label="Loading forecast" />
                <div className="skeleton h-[320px] w-full" />
              </div>
            ) : forecast.isError ? (
              <ErrorState error={forecast.error} onRetry={() => forecast.refetch()} />
            ) : history.isError ? (
              <ErrorState error={history.error} onRetry={() => history.refetch()} />
            ) : combined.length === 0 ? (
              <EmptyState title="No data for this series" />
            ) : (
              <>
                <ForecastChart
                  data={combined}
                  originDate={forecast.data!.origin_date}
                  height={330}
                />
                <Explain>
                  The solid line is what actually sold. The dashed line is the model's
                  forecast, starting the day after the marked origin. The shaded band
                  is not a model output — it is how wrong this model has actually been
                  on similar series in past tests, so it shows the range you should
                  plan around.
                </Explain>
                {forecast.data?.band_basis && <Caveat>{forecast.data.band_basis}</Caveat>}
              </>
            )}
          </Card>

          {/* --- planning ------------------------------------------- */}
          <Async query={planning} height="h-40">
            {(p) => (
              <Card
                title="Planning view"
                subtitle="What to expect over the next 28 days"
                actions={<Badge variant="info">{p.regime}</Badge>}
              >
                <dl className="grid grid-cols-2 gap-x-6 gap-y-5 md:grid-cols-4">
                  <Metric
                    label="Expected total"
                    value={compact(p.expected_total)}
                    unit="units"
                    tone="forecast"
                    hint="Sum of the 28 daily forecasts."
                  />
                  <Metric
                    label="Plan-around range"
                    value={`${compact(p.planning_range.low)} – ${compact(p.planning_range.high)}`}
                    size="sm"
                    hint="Where past errors of this size usually landed."
                  />
                  <Metric
                    label="Previous 28 days"
                    value={p.recent_28d_actual != null ? compact(p.recent_28d_actual) : '—'}
                    size="sm"
                    hint="What actually sold in the 28 days before the origin."
                  />
                  <Metric
                    label="Change"
                    value={p.change_vs_recent != null ? signed(p.change_vs_recent, 0) : '—'}
                    size="sm"
                    tone={
                      p.change_vs_recent == null ? 'default'
                        : p.change_vs_recent >= 0 ? 'good' : 'warn'
                    }
                    hint="Forecast total versus the previous 28 days."
                  />
                </dl>

                <div className="mt-5 grid grid-cols-4 gap-2">
                  {p.weekly_breakdown.map((w) => (
                    <div key={w.week} className="rounded border border-line bg-base px-3 py-2">
                      <p className="text-[10px] uppercase tracking-wide text-ink-dim">
                        Week {w.week}
                      </p>
                      <p className="tnum mt-1 text-sm font-semibold text-ink">
                        {nf(w.expected, w.expected < 10 ? 1 : 0)}
                      </p>
                      <p className="text-[10px] text-ink-dim">days {w.days}</p>
                    </div>
                  ))}
                </div>

                <Caveat>{p.planning_range.basis}</Caveat>

                <ul className="mt-3 space-y-1.5">
                  {p.caveats.map((c) => (
                    <li key={c} className="flex gap-2 text-xs leading-relaxed text-ink-muted">
                      <span aria-hidden className="text-ink-dim">·</span>
                      <span>{c}</span>
                    </li>
                  ))}
                </ul>
              </Card>
            )}
          </Async>
        </div>
      </div>
    </div>
  )
}
