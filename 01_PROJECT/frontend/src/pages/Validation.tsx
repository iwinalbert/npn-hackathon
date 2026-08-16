import { useEffect, useMemo, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, CartesianGrid as Grid, ComposedChart, Legend, Line,
  ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'

import {
  useAggregateBacktest, useNodes, useSearch, useSeriesBacktest, useWindows,
} from '../api/hooks'
import { ValidationDiagram } from '../components/diagrams'
import { Async, Badge, Card, Caveat, Explain, Metric, Spinner } from '../components/ui'
import { compact, longDate, nf, pct, shortDate } from '../lib/format'

const axis = { stroke: '#6B7C99', tick: { fontSize: 11 }, tickLine: false } as const

function BacktestTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  const p = payload[0].payload
  return (
    <div className="rounded border border-line-strong bg-surface px-3 py-2 shadow-lg">
      <p className="text-xs font-semibold text-ink">{longDate(String(label))}</p>
      <dl className="mt-1.5 space-y-1 text-xs">
        <div className="flex justify-between gap-6">
          <dt className="text-ink-muted">Actually sold</dt>
          <dd className="tnum font-semibold text-actual">{nf(p.actual, 2)}</dd>
        </div>
        <div className="flex justify-between gap-6">
          <dt className="text-ink-muted">Model predicted</dt>
          <dd className="tnum font-semibold text-forecast">{nf(p.predicted, 2)}</dd>
        </div>
        <div className="flex justify-between gap-6 border-t border-line pt-1">
          <dt className="text-ink-muted">Miss</dt>
          <dd className={`tnum font-semibold ${p.error >= 0 ? 'text-warn' : 'text-bad'}`}>
            {p.error >= 0 ? '+' : ''}{nf(p.error, 2)}
          </dd>
        </div>
      </dl>
    </div>
  )
}

/**
 * The "check it against reality" page.
 *
 * Every window here has a KNOWN outcome, which is exactly what makes it useful
 * and exactly why it must never be labelled as live accuracy.
 */
export function Validation() {
  const windows = useWindows()
  const [origin, setOrigin] = useState(1912)
  const [mode, setMode] = useState<'aggregate' | 'series'>('aggregate')

  const [level] = useState('store')
  const [nodeId, setNodeId] = useState('CA_3')
  const nodes = useNodes(level)

  const [term, setTerm] = useState('')
  const [debounced, setDebounced] = useState('')
  useEffect(() => {
    const t = setTimeout(() => setDebounced(term), 250)
    return () => clearTimeout(t)
  }, [term])
  const results = useSearch(debounced)
  const [series, setSeries] = useState({ store: 'CA_3', item: 'FOODS_3_090' })

  const aggBt = useAggregateBacktest(level, nodeId, origin)
  const serBt = useSeriesBacktest(
    mode === 'series' ? series.store : undefined,
    mode === 'series' ? series.item : undefined,
    origin,
  )

  const active = mode === 'aggregate' ? aggBt : serBt
  const points = useMemo(() => active.data?.points ?? [], [active.data])

  const selectedWindow = windows.data?.find((w) => w.origin_idx === origin)

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-ink">Validation</h1>
        <p className="mt-1 text-sm text-ink-muted">
          Replay a past 28-day window where the real outcome is known, and compare it
          against what the model predicted.
        </p>
      </div>

      <Caveat>
        These are <strong>historical validation results</strong>. For each window the
        model was rebuilt using only data available before that window began, then
        scored against what actually happened. This is not live deployment accuracy.
      </Caveat>

      {/* --- window picker ------------------------------------------- */}
      <Async query={windows} height="h-32">
        {(w) => (
          <Card title="Choose a validation window" subtitle="Eight independent periods with known outcomes">
            <div className="flex flex-wrap gap-2" role="group" aria-label="Validation window">
              {w.map((x) => (
                <button
                  key={x.origin_idx}
                  type="button"
                  onClick={() => setOrigin(x.origin_idx)}
                  aria-pressed={origin === x.origin_idx}
                  className={`rounded border px-3 py-2 text-left transition-colors ${
                    origin === x.origin_idx
                      ? 'border-forecast/50 bg-forecast/10'
                      : 'border-line hover:bg-elevated'
                  }`}
                >
                  <span className="block text-xs font-medium text-ink">
                    {shortDate(x.window_start)} – {shortDate(x.window_end)}
                  </span>
                  <span className="tnum block text-[10px] text-ink-dim">
                    RMSE {x.rmse.toFixed(3)}
                    {x.is_primary_validation_window && ' · primary'}
                  </span>
                </button>
              ))}
            </div>

            {selectedWindow && (
              <dl className="mt-5 grid grid-cols-2 gap-x-6 gap-y-4 border-t border-line pt-4 md:grid-cols-5">
                <Metric label="RMSE" value={selectedWindow.rmse.toFixed(4)} size="sm" tone="forecast" />
                <Metric label="MAE" value={selectedWindow.mae.toFixed(4)} size="sm" />
                <Metric label="WAPE" value={selectedWindow.wape.toFixed(4)} size="sm" />
                <Metric label="Bias" value={selectedWindow.bias.toFixed(4)} size="sm"
                  tone={Math.abs(selectedWindow.bias) < 0.05 ? 'good' : 'warn'} />
                <Metric label="Predictions" value={compact(selectedWindow.n_predictions)} size="sm" />
              </dl>
            )}
          </Card>
        )}
      </Async>

      {/* --- scope ---------------------------------------------------- */}
      <Card
        title="What to inspect"
        actions={
          <div className="flex gap-1" role="group" aria-label="Scope">
            {(['aggregate', 'series'] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                aria-pressed={mode === m}
                className={`rounded border px-2.5 py-1 text-[11px] font-medium transition-colors ${
                  mode === m
                    ? 'border-forecast/50 bg-forecast/10 text-forecast'
                    : 'border-line text-ink-muted hover:bg-elevated'
                }`}
              >
                {m === 'aggregate' ? 'A whole store' : 'A single product'}
              </button>
            ))}
          </div>
        }
      >
        {mode === 'aggregate' ? (
          <div>
            <label htmlFor="bt-node" className="block text-xs font-medium text-ink-muted">
              Store
            </label>
            <select
              id="bt-node"
              value={nodeId}
              onChange={(e) => setNodeId(e.target.value)}
              className="mt-1.5 w-full max-w-xs rounded border border-line bg-base px-3 py-2
                         text-sm text-ink focus:border-forecast"
            >
              {(nodes.data ?? []).map((o) => (
                <option key={o.node_id} value={o.node_id}>{o.node_id}</option>
              ))}
            </select>
          </div>
        ) : (
          <div className="max-w-md">
            <label htmlFor="bt-search" className="block text-xs font-medium text-ink-muted">
              Search a product
            </label>
            <input
              id="bt-search"
              type="search"
              value={term}
              onChange={(e) => setTerm(e.target.value)}
              placeholder="e.g. FOODS_3_090"
              className="mt-1.5 w-full rounded border border-line bg-base px-3 py-2 text-sm
                         text-ink placeholder:text-ink-dim focus:border-forecast"
            />
            {results.isFetching && <div className="mt-2"><Spinner label="Searching" /></div>}
            {results.data && results.data.length > 0 && (
              <ul className="mt-2 max-h-52 divide-y divide-line overflow-y-auto rounded border border-line">
                {results.data.map((s) => (
                  <li key={s.series_idx}>
                    <button
                      type="button"
                      onClick={() => setSeries({ store: s.store_id, item: s.item_id })}
                      className="flex w-full justify-between px-3 py-2 text-left text-xs
                                 transition-colors hover:bg-elevated"
                    >
                      <span className="text-ink">{s.item_id}</span>
                      <span className="text-ink-dim">{s.store_id}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
            <p className="mt-2 text-[11px] text-ink-dim">
              Showing <span className="text-ink-muted">{series.item}</span> in{' '}
              <span className="text-ink-muted">{series.store}</span>
            </p>
          </div>
        )}
      </Card>

      {/* --- the replay ---------------------------------------------- */}
      <Async query={active as never} height="h-80">
        {() => (
          <Card
            title="Predicted versus actual"
            subtitle={active.data?.window}
            actions={
              mode === 'series' && serBt.data ? (
                <Badge variant="info">RMSE {serBt.data.rmse.toFixed(3)}</Badge>
              ) : mode === 'aggregate' && aggBt.data?.accuracy_pct != null ? (
                <Badge variant="good">{pct(aggBt.data.accuracy_pct)} accurate</Badge>
              ) : null
            }
          >
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={points} margin={{ top: 8, right: 12, left: 4, bottom: 4 }}>
                <Grid stroke="#243149" strokeDasharray="2 4" vertical={false} />
                <XAxis dataKey="date" tickFormatter={shortDate} {...axis}
                  axisLine={{ stroke: '#243149' }} minTickGap={24} />
                <YAxis {...axis} axisLine={false} width={48} />
                <Tooltip content={<BacktestTooltip />} cursor={{ stroke: '#31415F' }} />
                <Legend verticalAlign="top" height={28} iconType="plainline"
                  wrapperStyle={{ fontSize: 11, color: '#9AA9C2' }} />
                <Line dataKey="actual" name="Actually sold" stroke="#C6D2E4"
                  strokeWidth={1.8} dot={false} isAnimationActive={false} />
                <Line dataKey="predicted" name="Model predicted" stroke="#4EA8F0"
                  strokeWidth={2} strokeDasharray="5 3" dot={false} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>

            <div className="mt-5">
              <p className="mb-2 text-xs font-medium text-ink-muted">
                Daily miss (predicted − actual)
              </p>
              <ResponsiveContainer width="100%" height={110}>
                <BarChart data={points} margin={{ top: 0, right: 12, left: 4, bottom: 0 }}>
                  <CartesianGrid stroke="#243149" strokeDasharray="2 4" vertical={false} />
                  <XAxis dataKey="date" tickFormatter={shortDate} {...axis}
                    axisLine={{ stroke: '#243149' }} minTickGap={24} />
                  <YAxis {...axis} axisLine={false} width={48} />
                  <Tooltip content={<BacktestTooltip />} cursor={{ fill: '#1B2740' }} />
                  <ReferenceLine y={0} stroke="#31415F" />
                  <Bar dataKey="error" name="miss" fill="#4EA8F0" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {active.data && (
              <dl className="mt-5 grid grid-cols-2 gap-x-6 gap-y-4 border-t border-line pt-4 sm:grid-cols-4">
                <Metric label="Total actual" value={compact(active.data.total_actual)} size="sm" />
                <Metric label="Total predicted" value={compact(active.data.total_predicted)} size="sm" />
                <Metric
                  label="Difference"
                  value={compact(active.data.total_predicted - active.data.total_actual)}
                  size="sm"
                  tone={
                    Math.abs(active.data.total_predicted - active.data.total_actual) /
                      Math.max(active.data.total_actual, 1) < 0.05 ? 'good' : 'warn'
                  }
                  hint="Over the whole 28-day window."
                />
                {mode === 'series' && serBt.data && (
                  <Metric label="MAE" value={serBt.data.mae.toFixed(3)} size="sm" />
                )}
                {mode === 'aggregate' && aggBt.data?.wape != null && (
                  <Metric label="WAPE" value={aggBt.data.wape.toFixed(4)} size="sm" />
                )}
              </dl>
            )}

            <Explain>
              {mode === 'aggregate'
                ? 'Totals for a whole store track closely because independent errors on individual products cancel out. This is the level most ordering decisions are actually made at.'
                : 'A single product is far noisier than a store total. Judge the model here on whether it tracks the general level and shape, not on matching individual days exactly.'}
            </Explain>
            {'basis' in (active.data ?? {}) && (
              <p className="mt-2 text-[11px] text-ink-dim">{(active.data as any).basis}</p>
            )}
            {'note' in (active.data ?? {}) && (
              <p className="mt-2 text-[11px] text-ink-dim">{(active.data as any).note}</p>
            )}
          </Card>
        )}
      </Async>

      <Card title="How validation works">
        <ValidationDiagram />
      </Card>
    </div>
  )
}
