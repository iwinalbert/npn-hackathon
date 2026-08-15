import { useState } from 'react'
import { Link } from 'react-router-dom'

import { useNodes, usePortfolio, useTopMovers } from '../api/hooks'
import { Async, Badge, Card, Explain, Metric } from '../components/ui'
import { REGIME_COLORS, compact, nf, pct, signed } from '../lib/format'
import type { Mover } from '../api/types'

function MoverRow({ m, direction }: { m: Mover; direction: 'up' | 'down' }) {
  return (
    <li>
      <Link
        to={`/forecast?store=${m.store_id}&item=${m.item_id}`}
        className="flex items-center gap-3 rounded px-2 py-2 transition-colors hover:bg-elevated"
      >
        <span
          aria-hidden
          className={`w-1 self-stretch rounded-full ${direction === 'up' ? 'bg-good' : 'bg-warn'}`}
        />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-xs font-medium text-ink">{m.item_id}</span>
          <span className="block text-[11px] text-ink-dim">
            {m.store_id} · {m.dept_id}
          </span>
        </span>
        <span className="shrink-0 text-right">
          <span
            className={`tnum block text-xs font-semibold ${
              direction === 'up' ? 'text-good' : 'text-warn'
            }`}
          >
            {signed(m.delta_28d, 0)} units
          </span>
          <span className="tnum block text-[11px] text-ink-dim">
            {m.recent_daily_28d.toFixed(1)} → {m.forecast_daily.toFixed(1)}/day
          </span>
        </span>
      </Link>
    </li>
  )
}

/**
 * Decision support: the last step of the flow the product teaches.
 *
 * Every figure here is arithmetic over the frozen forecast and the observed
 * history — there is no separate "insight model", and nothing is hard-coded.
 */
export function Insights() {
  const [level, setLevel] = useState('total')
  const [nodeId, setNodeId] = useState('ALL')
  const nodes = useNodes(level === 'total' ? 'store' : level)

  const portfolio = usePortfolio(level, nodeId)
  const movers = useTopMovers(level, nodeId, 12)

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-ink">Insights</h1>
        <p className="mt-1 text-sm text-ink-muted">
          Where demand is shifting, and what deserves attention first.
        </p>
      </div>

      {/* --- scope --------------------------------------------------- */}
      <Card title="Scope">
        <div className="flex flex-wrap items-end gap-4">
          <div>
            <span className="block text-xs font-medium text-ink-muted">Level</span>
            <div className="mt-1.5 flex gap-1" role="group" aria-label="Level">
              {[
                { k: 'total', l: 'Whole chain' },
                { k: 'store', l: 'One store' },
              ].map((o) => (
                <button
                  key={o.k}
                  type="button"
                  onClick={() => {
                    setLevel(o.k)
                    setNodeId(o.k === 'total' ? 'ALL' : 'CA_3')
                  }}
                  aria-pressed={level === o.k}
                  className={`rounded border px-2.5 py-1.5 text-xs font-medium transition-colors ${
                    level === o.k
                      ? 'border-forecast/50 bg-forecast/10 text-forecast'
                      : 'border-line text-ink-muted hover:bg-elevated'
                  }`}
                >
                  {o.l}
                </button>
              ))}
            </div>
          </div>

          {level === 'store' && (
            <div>
              <label htmlFor="ins-store" className="block text-xs font-medium text-ink-muted">
                Store
              </label>
              <select
                id="ins-store"
                value={nodeId}
                onChange={(e) => setNodeId(e.target.value)}
                className="mt-1.5 rounded border border-line bg-base px-3 py-1.5 text-sm
                           text-ink focus:border-forecast"
              >
                {(nodes.data ?? []).map((o) => (
                  <option key={o.node_id} value={o.node_id}>{o.node_id}</option>
                ))}
              </select>
            </div>
          )}
        </div>
      </Card>

      {/* --- portfolio ----------------------------------------------- */}
      <Async query={portfolio} height="h-48">
        {(p) => (
          <Card
            title="Next 28 days at a glance"
            subtitle={`${nf(p.n_series)} store-item series in scope`}
            actions={
              p.expected_accuracy ? (
                <Badge variant="good">
                  {pct(p.expected_accuracy.accuracy_pct)} accurate at this level
                </Badge>
              ) : null
            }
          >
            <dl className="grid grid-cols-2 gap-x-6 gap-y-5 md:grid-cols-4">
              <Metric
                label="Forecast demand"
                value={compact(p.forecast_total_28d)}
                unit="units"
                tone="forecast"
                hint="Total expected over the next 28 days."
              />
              <Metric
                label="Previous 28 days"
                value={compact(p.recent_total_28d)}
                size="sm"
                hint="What actually sold in the 28 days before the origin."
              />
              <Metric
                label="Change"
                value={p.change_pct != null ? `${signed(p.change_pct, 1)}%` : '—'}
                size="sm"
                tone={p.change_pct == null ? 'default' : p.change_pct >= 0 ? 'good' : 'warn'}
                hint="Forecast versus the previous period."
              />
              <Metric
                label="Days with no sale"
                value={pct(p.avg_zero_day_pct)}
                size="sm"
                hint="Averaged across every series in scope."
              />
            </dl>

            <div className="mt-5">
              <p className="mb-2 text-xs font-medium text-ink-muted">Demand patterns in scope</p>
              <div className="flex h-3 overflow-hidden rounded-full">
                {p.regime_mix.map((r) => (
                  <div
                    key={r.regime}
                    title={`${r.regime}: ${nf(r.n_series)} series`}
                    style={{
                      width: `${(r.n_series / p.n_series) * 100}%`,
                      background: REGIME_COLORS[r.regime] ?? '#6B7C99',
                    }}
                  />
                ))}
              </div>
              <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
                {p.regime_mix.map((r) => (
                  <li key={r.regime} className="flex items-center gap-1.5 text-[11px]">
                    <span
                      aria-hidden
                      className="h-2 w-2 rounded-full"
                      style={{ background: REGIME_COLORS[r.regime] ?? '#6B7C99' }}
                    />
                    <span className="text-ink-muted">{r.regime}</span>
                    <span className="tnum text-ink-dim">{pct((r.n_series / p.n_series) * 100, 0)}</span>
                  </li>
                ))}
              </ul>
            </div>

            <Explain>
              Most products in this dataset sell on only some days. The mix above is why
              a single accuracy number would be misleading — a portfolio dominated by
              intermittent items behaves very differently from one dominated by
              steady sellers.
            </Explain>
          </Card>
        )}
      </Async>

      {/* --- movers -------------------------------------------------- */}
      <Async query={movers} height="h-96">
        {(m) => (
          <div className="grid gap-6 lg:grid-cols-2">
            <Card
              title="Rising fastest"
              subtitle="Largest forecast increase against the last 28 days"
            >
              {m.rising && m.rising.length > 0 ? (
                <ul className="-mx-2 divide-y divide-line">
                  {m.rising.map((x) => (
                    <MoverRow key={x.series_idx} m={x} direction="up" />
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-ink-dim">Nothing rising in this scope.</p>
              )}
            </Card>

            <Card
              title="Falling fastest"
              subtitle="Largest forecast decrease against the last 28 days"
            >
              {m.falling && m.falling.length > 0 ? (
                <ul className="-mx-2 divide-y divide-line">
                  {m.falling.map((x) => (
                    <MoverRow key={x.series_idx} m={x} direction="down" />
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-ink-dim">Nothing falling in this scope.</p>
              )}
            </Card>

            <div className="lg:col-span-2">
              <Explain>{m.basis}</Explain>
            </div>
          </div>
        )}
      </Async>
    </div>
  )
}
