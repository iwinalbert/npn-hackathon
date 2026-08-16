import { useMemo, useState } from 'react'
import { useAggregate, useLevelAccuracy, useNodes } from '../api/hooks'
import { HierarchyDiagram } from '../components/diagrams'
import { Async, Card, Explain, Metric } from '../components/ui'
import { combineHistoryAndForecast, compact, pct } from '../lib/format'
import { ForecastChart } from '../components/charts/ForecastChart'

/** Levels worth drilling through in a UI, in the order a planner thinks. */
const LEVELS = [
  { key: 'total', label: 'Whole chain' },
  { key: 'state', label: 'State' },
  { key: 'store', label: 'Store' },
  { key: 'category', label: 'Category' },
  { key: 'department', label: 'Department' },
  { key: 'store_department', label: 'Store × Dept' },
  { key: 'item', label: 'Item (all stores)' },
]

export function Hierarchy() {
  const [level, setLevel] = useState('store')
  const [nodeId, setNodeId] = useState('CA_3')

  const nodes = useNodes(level)
  const agg = useAggregate(level, nodeId, 90)
  const levelAcc = useLevelAccuracy()

  // Keep the selected node valid when the level changes.
  const options = nodes.data ?? []
  const effectiveNode = useMemo(() => {
    if (level === 'total') return 'ALL'
    if (options.some((o) => o.node_id === nodeId)) return nodeId
    return options[0]?.node_id ?? ''
  }, [level, nodeId, options])

  const combined = useMemo(() => {
    if (!agg.data) return []
    return combineHistoryAndForecast(
      (agg.data.history ?? []).map((h) => ({ date: h.date, sales: h.sales })),
      agg.data.forecast.map((f) => ({ date: f.date, yhat: f.yhat })),
    )
  }, [agg.data])

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-ink">Hierarchy</h1>
        <p className="mt-1 text-sm text-ink-muted">
          The model forecasts individual store-item pairs. Everything above is the sum of those.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr_20rem]">
        <div className="space-y-6">
          {/* --- selector ------------------------------------------- */}
          <Card title="Choose a level">
            <div className="flex flex-wrap gap-1.5" role="group" aria-label="Aggregation level">
              {LEVELS.map((l) => (
                <button
                  key={l.key}
                  type="button"
                  onClick={() => {
                    setLevel(l.key)
                    setNodeId(l.key === 'total' ? 'ALL' : '')
                  }}
                  aria-pressed={level === l.key}
                  className={`rounded border px-2.5 py-1.5 text-xs font-medium transition-colors ${
                    level === l.key
                      ? 'border-forecast/50 bg-forecast/10 text-forecast'
                      : 'border-line text-ink-muted hover:bg-elevated'
                  }`}
                >
                  {l.label}
                </button>
              ))}
            </div>

            {level !== 'total' && (
              <div className="mt-4">
                <label htmlFor="node-select" className="block text-xs font-medium text-ink-muted">
                  Then pick one
                </label>
                {nodes.isLoading ? (
                  <div className="skeleton mt-1.5 h-9 w-full" />
                ) : (
                  <select
                    id="node-select"
                    value={effectiveNode}
                    onChange={(e) => setNodeId(e.target.value)}
                    className="mt-1.5 w-full rounded border border-line bg-base px-3 py-2
                               text-sm text-ink transition-colors focus:border-forecast"
                  >
                    {options.map((o) => (
                      <option key={o.node_id} value={o.node_id}>
                        {o.node_id} — {o.n_series.toLocaleString()} series
                      </option>
                    ))}
                  </select>
                )}
              </div>
            )}
          </Card>

          {/* --- aggregate forecast --------------------------------- */}
          <Async query={agg} height="h-80">
            {(a) => (
              <Card
                title={`${a.node_id} · 28-day outlook`}
                subtitle={`Sum of ${a.n_series.toLocaleString()} store-item forecasts`}
              >
                <dl className="mb-5 grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-3">
                  <Metric
                    label="Forecast total"
                    value={compact(a.total_28d)}
                    unit="units"
                    tone="forecast"
                    hint="Expected demand over the next 28 days."
                  />
                  <Metric
                    label="Daily average"
                    value={compact(a.total_28d / 28)}
                    size="sm"
                    hint="Forecast total divided by 28."
                  />
                  {a.expected_accuracy && (
                    <Metric
                      label="Accuracy at this level"
                      value={pct(a.expected_accuracy.accuracy_pct)}
                      size="sm"
                      tone="good"
                      hint="Measured on held-out data at this exact aggregation."
                    />
                  )}
                </dl>

                {combined.length > 0 && (
                  <ForecastChart
                    data={combined}
                    originDate={a.forecast[0]?.date ?? ''}
                    height={280}
                    showBand={false}
                    yLabel="units / day"
                  />
                )}

                <Explain>
                  Because every aggregate is an exact sum of the bottom-level
                  forecasts, the numbers here can never contradict the individual
                  product forecasts. Accuracy improves as you aggregate: independent
                  errors on individual products cancel each other out.
                </Explain>

                {a.expected_accuracy && (
                  <p className="mt-2 text-[11px] text-ink-dim">{a.expected_accuracy.basis}</p>
                )}
              </Card>
            )}
          </Async>
        </div>

        {/* --- side: structure + accuracy ladder -------------------- */}
        <div className="space-y-6">
          <Card title="How aggregation works">
            <HierarchyDiagram />
          </Card>

          <Async query={levelAcc} height="h-64">
            {(d) => {
              const max = Math.max(...d.levels.map((l) => l.accuracy_pct))
              return (
                <Card title="Accuracy by level" subtitle="Measured, held-out">
                  <ul className="space-y-2">
                    {d.levels.map((l) => (
                      <li key={l.level}>
                        <div className="flex items-baseline justify-between gap-2">
                          <span className="truncate text-[11px] text-ink-muted">
                            {l.level.replace(/^L\d+_/, '').replace(/_/g, ' ')}
                          </span>
                          <span className="tnum text-[11px] font-semibold text-ink">
                            {pct(l.accuracy_pct)}
                          </span>
                        </div>
                        <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-elevated">
                          <div
                            className="h-full rounded-full bg-forecast/70"
                            style={{ width: `${(l.accuracy_pct / max) * 100}%` }}
                          />
                        </div>
                      </li>
                    ))}
                  </ul>
                  <Explain>{d.note}</Explain>
                </Card>
              )
            }}
          </Async>
        </div>
      </div>
    </div>
  )
}
