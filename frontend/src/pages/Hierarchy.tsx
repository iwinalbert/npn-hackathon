import { useMemo, useRef, useState } from 'react'

import {
  useAggregate, useAggregateBacktest, useLevelAccuracy, useLevels, useModelCard,
  useNodeTotals, useNodes,
} from '../api/hooks'
import { ActualVsPredictedChart } from '../components/charts/ActualVsPredictedChart'
import { HierarchyFlow, type FlowBranch, type FlowNode } from '../components/charts/HierarchyFlow'
import { Async, Badge, Card, Explain, Metric, VizStage } from '../components/ui'
import { buildDemandSeries, compact, nf, pct } from '../lib/format'
import type { AggregateForecast, HierarchyNode } from '../api/types'

const LEVELS = [
  { key: 'total', label: 'Whole chain' },
  { key: 'state', label: 'State' },
  { key: 'store', label: 'Store' },
  { key: 'category', label: 'Category' },
  { key: 'department', label: 'Department' },
  { key: 'store_department', label: 'Store × Dept' },
  { key: 'item', label: 'Item (all stores)' },
]

function toFlowNodes(
  nodes: HierarchyNode[] | undefined,
  totals: Record<string, AggregateForecast>,
): FlowNode[] {
  return (nodes ?? []).map((n) => ({
    id: n.node_id,
    label: n.node_id,
    nSeries: n.n_series,
    total: totals[n.node_id]?.total_28d ?? null,
  }))
}

export function Hierarchy() {
  const [level, setLevel] = useState('store')
  const [nodeId, setNodeId] = useState('CA_3')
  const drillRef = useRef<HTMLDivElement>(null)

  const levels = useLevels()
  const model = useModelCard()
  const states = useNodes('state')
  const stores = useNodes('store')
  const categories = useNodes('category')
  const departments = useNodes('department')

  const chain = useNodeTotals('total', ['ALL'])
  const stateTotals = useNodeTotals('state', (states.data ?? []).map((n) => n.node_id))
  const storeTotals = useNodeTotals('store', (stores.data ?? []).map((n) => n.node_id))
  const catTotals = useNodeTotals('category', (categories.data ?? []).map((n) => n.node_id))
  const deptTotals = useNodeTotals('department', (departments.data ?? []).map((n) => n.node_id))

  const chainTotal = chain.byId.ALL?.total_28d ?? null
  const bottomSeries =
    levels.data?.find((l) => l.level === 'series')?.node_count
    ?? model.data?.n_series
    ?? 0

  const branches: FlowBranch[] = [
    {
      key: 'geography',
      title: 'By geography — where demand happens',
      caption: 'summed by geography',
      levels: [
        { key: 'store', title: 'Store', shape: 'box', nodes: toFlowNodes(stores.data, storeTotals.byId) },
        { key: 'state', title: 'State', shape: 'pill', nodes: toFlowNodes(states.data, stateTotals.byId) },
      ],
    },
    {
      key: 'product',
      title: 'By product — what is being sold',
      caption: 'summed by product',
      levels: [
        { key: 'department', title: 'Department', shape: 'box', nodes: toFlowNodes(departments.data, deptTotals.byId) },
        { key: 'category', title: 'Category', shape: 'pill', nodes: toFlowNodes(categories.data, catTotals.byId) },
      ],
    },
  ]

  const nodes = useNodes(level)
  const options = nodes.data ?? []
  const effectiveNode = useMemo(() => {
    if (level === 'total') return 'ALL'
    if (options.some((o) => o.node_id === nodeId)) return nodeId
    return options[0]?.node_id ?? ''
  }, [level, nodeId, options])

  const agg = useAggregate(level, effectiveNode, 0)
  const aggBacktest = useAggregateBacktest(level, effectiveNode)
  const levelAcc = useLevelAccuracy()

  const series = useMemo(() => buildDemandSeries({
    backtest: aggBacktest.data?.points.map((p) => ({
      date: p.date, actual: p.actual, predicted: p.predicted,
    })),
    forecast: agg.data?.forecast.map((f) => ({ date: f.date, yhat: f.yhat })),
  }), [aggBacktest.data, agg.data])

  const select = (nextLevel: string, nextNode: string) => {
    setLevel(nextLevel)
    setNodeId(nextNode)
    drillRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  return (
    <div className="mx-auto max-w-[1600px] space-y-6">
      <VizStage
        title="Hierarchical Demand Forecast"
        description={
          <>
            How lower-level store-item forecasts aggregate into higher-level demand.
            The model predicts one level only — every figure above it is an exact sum
            of those predictions, summed two ways: by geography and by product.
          </>
        }
        actions={<Badge variant="info">28-day totals</Badge>}
        meta="Click any node to chart it below."
        footer={
          <p className="text-xs leading-relaxed text-ink-muted">
            Because every aggregate is an exact sum of the bottom-level forecasts, the
            numbers here can never contradict the individual product forecasts — and
            accuracy improves as you go up, because independent errors on individual
            products cancel each other out.
          </p>
        }
      >
        <HierarchyFlow
          branches={branches}
          bottom={{
            label: 'Store × Item',
            nSeries: bottomSeries,
            horizonDays: model.data?.horizon_days ?? 28,
          }}
          chainTotal={chainTotal}
          selected={{ level, nodeId: effectiveNode }}
          onSelect={select}
        />
      </VizStage>

      {/* --- the selected node, charted the same way as the Forecast page --- */}
      <div ref={drillRef} className="scroll-mt-20">
        <Card
          title={`Actual vs Predicted Demand — ${effectiveNode}`}
          subtitle={
            agg.data
              ? [
                  `Sum of ${nf(agg.data.n_series)} store-item forecasts`,
                  aggBacktest.data && `compared on held-out window ${aggBacktest.data.window}`,
                ].filter(Boolean).join(' · ')
              : 'Loading node'
          }
          actions={
            <div className="flex flex-wrap items-center justify-end gap-2">
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
                    className={`rounded border px-2.5 py-1 text-[11px] font-medium transition-colors ${
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
                <select
                  aria-label="Node"
                  value={effectiveNode}
                  onChange={(e) => setNodeId(e.target.value)}
                  className="rounded border border-line bg-base px-2 py-1 text-xs text-ink
                             transition-colors focus:border-forecast"
                >
                  {options.map((o) => (
                    <option key={o.node_id} value={o.node_id}>
                      {o.node_id} — {nf(o.n_series)} series
                    </option>
                  ))}
                </select>
              )}
            </div>
          }
        >
          <Async query={agg} height="h-96">
            {(a) => (
              <>
                <dl className="mb-5 grid grid-cols-2 gap-x-6 gap-y-5 sm:grid-cols-4">
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
                  {aggBacktest.data?.accuracy_pct != null && (
                    <Metric
                      label="This node, held out"
                      value={pct(aggBacktest.data.accuracy_pct)}
                      size="sm"
                      tone="good"
                      hint="Measured on this node alone, over the window charted below."
                    />
                  )}
                </dl>

                {series.length > 0 ? (
                  <ActualVsPredictedChart
                    data={series}
                    forecastStartDate={a.forecast[0]?.date}
                    height="clamp(320px, 44vh, 560px)"
                    showBand={false}
                    everyDate
                  />
                ) : (
                  <div className="skeleton h-80 w-full" />
                )}

                {aggBacktest.data && <Explain>{aggBacktest.data.note}</Explain>}
                {a.expected_accuracy && (
                  <p className="mt-2 text-[11px] text-ink-dim">{a.expected_accuracy.basis}</p>
                )}
              </>
            )}
          </Async>
        </Card>
      </div>

      {/* --- supporting reference ---------------------------------------- */}
      <div className="grid gap-6 lg:grid-cols-2">
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

        <Async query={levels} height="h-64">
          {(ls) => (
            <Card title="The levels this project implements" subtitle="Node counts, from the data itself">
              <ul className="divide-y divide-line text-xs">
                {ls.map((l) => (
                  <li key={l.level} className="flex items-center justify-between gap-4 py-1.5">
                    <span className="min-w-0">
                      <span className="block truncate font-medium text-ink">{l.label}</span>
                      <span className="block font-mono text-[10px] text-ink-dim">{l.level}</span>
                    </span>
                    <span className="tnum shrink-0 text-ink-muted">{nf(l.node_count)} nodes</span>
                  </li>
                ))}
              </ul>
              <Explain>
                Twelve levels, one forecast. The diagram above draws four of them plus
                the chain total and the bottom level; the rest are the same sum taken
                over a different grouping.
              </Explain>
            </Card>
          )}
        </Async>
      </div>
    </div>
  )
}
