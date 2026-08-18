import { useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'

import {
  usePlanning, useSearch, useSeriesBacktest, useSeriesForecast, useSeriesHistory,
} from '../api/hooks'
import { ActualVsPredictedChart } from '../components/charts/ActualVsPredictedChart'
import {
  Async, Badge, Card, Caveat, EmptyState, ErrorState, Explain, Metric, Spinner, VizStage,
} from '../components/ui'
import {
  REGIME_COLORS, buildDemandSeries, compact, humanise, longDate, nf, pct, signed,
} from '../lib/format'

const HISTORY_OPTIONS = [
  { days: 28, label: '28d' },
  { days: 90, label: '90d' },
  { days: 180, label: '6m' },
  { days: 365, label: '1y' },
]

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
  const [days, setDays] = useState(28)

  const history = useSeriesHistory(store, item, days)
  const forecast = useSeriesForecast(store, item)
  const backtest = useSeriesBacktest(store, item)
  const planning = usePlanning(store, item)

  const pick = (s: string, i: string) => setParams({ store: s, item: i })

  const series = useMemo(() => buildDemandSeries({
    history: history.data?.history.map((h) => ({ date: h.date, sales: h.sales })),
    backtest: backtest.data?.points.map((p) => ({
      date: p.date, actual: p.actual, predicted: p.predicted,
    })),
    forecast: forecast.data?.forecast.map((f) => ({
      date: f.date, yhat: f.yhat, lower: f.lower, upper: f.upper,
    })),
  }), [history.data, backtest.data, forecast.data])

  const forecastStart = forecast.data?.forecast[0]?.date
  const loading = history.isLoading || forecast.isLoading

  return (
    <div className="mx-auto max-w-[1600px] space-y-6">
      <VizStage
        title="Actual vs Predicted Demand — 28-Day Forecast"
        description={
          <>
            What actually sold, drawn against what the model said would sell. Where
            both lines run together the model is being scored on days it never saw
            in training; after <span className="font-medium text-ink">Forecast Start</span>{' '}
            only the prediction exists, because those 28 days have not happened yet.
          </>
        }
        meta={
          forecast.data && (
            <>
              <span className="font-medium text-ink-muted">{item}</span> · {store}
              {backtest.data && <> · Compared on held-out window {backtest.data.window}</>}
              {' '}· Forecast from {longDate(forecast.data.forecast[0].date)} to{' '}
              {longDate(forecast.data.forecast.at(-1)!.date)}
            </>
          )
        }
        actions={
          <div className="flex flex-wrap items-center gap-3">
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
        footer={
          <>
            <p className="text-xs leading-relaxed text-ink-muted">
              The shaded band is not a model output — it is how wrong this model has
              actually been on similar series in past tests, so it shows the range
              you should plan around.
            </p>
            {backtest.data && (
              <p className="mt-1.5 text-[11px] leading-relaxed text-ink-dim">
                {backtest.data.basis}
              </p>
            )}
          </>
        }
      >
        {loading ? (
          <div className="space-y-3">
            <Spinner label="Loading demand chart" />
            <div className="skeleton h-[480px] w-full" />
          </div>
        ) : forecast.isError ? (
          <ErrorState error={forecast.error} onRetry={() => forecast.refetch()} />
        ) : history.isError ? (
          <ErrorState error={history.error} onRetry={() => history.refetch()} />
        ) : series.length === 0 ? (
          <EmptyState title="No data for this series" />
        ) : (
          <ActualVsPredictedChart
            data={series}
            forecastStartDate={forecastStart}
            height="clamp(400px, 56vh, 680px)"
            everyDate
          />
        )}
      </VizStage>

      {/* --- supporting: how close it was, and what to plan around ----- */}
      <div className="grid gap-6 xl:grid-cols-2">
        <Async query={backtest} height="h-40">
          {(b) => (
            <Card
              title="How close was it, last time we could check?"
              subtitle={`Held-out window ${b.window} — the 28 days before the forecast starts`}
            >
              <dl className="grid grid-cols-2 gap-x-6 gap-y-5 sm:grid-cols-4">
                <Metric
                  label="Actual total"
                  value={compact(b.total_actual)}
                  unit="units"
                  hint="What this series really sold over those 28 days."
                />
                <Metric
                  label="Predicted total"
                  value={compact(b.total_predicted)}
                  unit="units"
                  tone="forecast"
                  hint="What the model said it would sell."
                />
                <Metric
                  label="RMSE"
                  value={b.rmse.toFixed(2)}
                  size="sm"
                  hint="Average daily miss, penalising big misses hardest."
                />
                <Metric
                  label="MAE"
                  value={b.mae.toFixed(2)}
                  size="sm"
                  hint="The typical daily miss in units."
                />
              </dl>
              <Explain>
                These are the two lines on the chart, added up. Daily accuracy on a
                single product is modest — most days sell nothing — which is exactly
                why the plan-around range below matters more than any single day.
              </Explain>
            </Card>
          )}
        </Async>

        <Async query={planning} height="h-40">
          {(p) => (
            <Card
              title="Planning view"
              subtitle="What to expect over the next 28 days"
              actions={<Badge variant="info">{p.regime}</Badge>}
            >
              <dl className="grid grid-cols-2 gap-x-6 gap-y-5 sm:grid-cols-4">
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

              <div className="mt-5 grid grid-cols-2 gap-2 sm:grid-cols-4">
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

      {/* --- selection, below the chart it feeds ---------------------- */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Select a series" subtitle="Search, then the chart above updates">
          <SeriesSearch onPick={pick} current={`${store}/${item}`} />
        </Card>

        {forecast.data && (
          <Card title="Series profile">
            <dl className="grid gap-3 text-xs sm:grid-cols-2">
              <div className="sm:col-span-2">
                <dt className="text-ink-dim">Product</dt>
                <dd className="mt-0.5 font-medium text-ink">{humanise(item)}</dd>
                <dd className="font-mono text-[10px] text-ink-dim">{item}</dd>
              </div>
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
              <div className="sm:col-span-2">
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
              <div className="border-t border-line pt-3">
                <dt className="text-ink-dim">Days with no sale</dt>
                <dd className="tnum mt-0.5 font-medium text-ink">
                  {pct(forecast.data.series.zero_pct)}
                </dd>
              </div>
              <div className="border-t border-line pt-3">
                <dt className="text-ink-dim">Avg daily</dt>
                <dd className="tnum mt-0.5 font-medium text-ink">
                  {forecast.data.series.mean_daily_sales.toFixed(2)}
                </dd>
              </div>
            </dl>
          </Card>
        )}
      </div>
    </div>
  )
}
