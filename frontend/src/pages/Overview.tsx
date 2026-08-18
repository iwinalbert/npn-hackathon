import { Link } from 'react-router-dom'

import { useLevelAccuracy, useModelCard, useOccurrence, useProvenance } from '../api/hooks'
import { PipelineDiagram } from '../components/diagrams'
import { Async, Badge, Card, Explain, Metric } from '../components/ui'
import { compact, nf, pct } from '../lib/format'

export function Overview() {
  const model = useModelCard()
  const prov = useProvenance()
  const levels = useLevelAccuracy()
  const occ = useOccurrence()

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      {/* --- masthead ------------------------------------------------- */}
      <div className="animate-slideUp">
        <h1 className="text-2xl font-semibold tracking-tight text-ink">
          Retail Demand Forecasting
        </h1>
        <p className="mt-1 text-sm text-ink-muted">
          Walmart M5 · Hierarchical 28-Day Forecasting
        </p>
        <p className="mt-3 max-w-2xl text-sm leading-relaxed text-ink-muted">
          This system predicts how many units each product will sell, in each store,
          for each of the next 28 days — 30,490 store-item combinations in total.
          It handles hierarchical roll-ups, price and calendar context, and demand
          that is zero on most days.
        </p>
      </div>

      {/* --- headline metrics ----------------------------------------- */}
      <Async query={model} height="h-32">
        {(m) => (
          <Card
            title="The deployed model"
            subtitle={m.blend_formula}
            actions={<Badge variant={m.status === 'FROZEN' ? 'info' : 'warn'}>{m.status}</Badge>}
          >
            <dl className="grid grid-cols-2 gap-x-6 gap-y-6 md:grid-cols-4">
              <Metric
                label="Forecast horizon"
                value={m.horizon_days}
                unit="days"
                hint={`Covers ${m.forecast_dates}.`}
              />
              <Metric
                label="Series forecast"
                value={nf(m.n_series)}
                hint="One forecast per product, per store."
              />
              <Metric
                label="RMSE"
                value={m.validation_rmse.toFixed(4)}
                tone="forecast"
                hint="Average size of the forecast miss, penalising big misses hardest. Lower is better."
              />
              <Metric
                label="MAE"
                value={m.validation_mae.toFixed(4)}
                hint="The typical miss in units, treating all misses equally. Lower is better."
              />
            </dl>
            <Explain>
              Measured on {nf(m.validation_n)} held-out predictions from{' '}
              {m.validation_window} — days the model never saw during training.
              For scale: always guessing zero scores 3.92, and a simple 28-day
              average scores 2.24.
            </Explain>
          </Card>
        )}
      </Async>

      {/* --- accuracy depends on level -------------------------------- */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Async query={levels} height="h-64">
          {(d) => {
            const pick = (k: string) => d.levels.find((l) => l.level.includes(k))
            const rows = [
              { label: 'Whole chain, per day', l: pick('L1_total') },
              { label: 'One store, per day', l: pick('L3_store') },
              { label: 'One item, all stores', l: pick('L10_item') },
              { label: 'One item in one store', l: pick('L12_store_item') },
            ].filter((r) => r.l)
            return (
              <Card
                title="Accuracy depends on what you are deciding"
                subtitle="The same forecast, measured at four different levels"
              >
                <ul className="space-y-2.5">
                  {rows.map((r) => (
                    <li key={r.label} className="flex items-center gap-3">
                      <span className="w-44 shrink-0 text-xs text-ink-muted">{r.label}</span>
                      <div className="h-2 flex-1 overflow-hidden rounded-full bg-elevated">
                        <div
                          className="h-full rounded-full bg-forecast transition-all"
                          style={{ width: `${Math.max(r.l!.accuracy_pct, 2)}%` }}
                        />
                      </div>
                      <span className="tnum w-14 shrink-0 text-right text-xs font-semibold text-ink">
                        {pct(r.l!.accuracy_pct)}
                      </span>
                    </li>
                  ))}
                </ul>
                <Explain>
                  Individual store-item forecasts are hard because most days sell
                  nothing. Those errors are largely independent, so they cancel out
                  when you add them up — which is why a store-level plan is far more
                  reliable than any single product line. Always read the number that
                  matches your decision.
                </Explain>
              </Card>
            )
          }}
        </Async>

        <Card
          title="How a forecast is produced"
          subtitle="Two different models, blended"
        >
          <PipelineDiagram />
        </Card>
      </div>

      {/* --- occurrence + provenance ---------------------------------- */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Async query={occ} height="h-40">
          {(o) => (
            <Card
              title="Does it spot the days that actually sell?"
              subtitle={`Threshold: a forecast of ${o.threshold} units or more counts as "expect a sale"`}
            >
              <dl className="grid grid-cols-2 gap-x-6 gap-y-5 sm:grid-cols-4">
                <Metric label="Recall" value={pct(o.recall * 100)} size="sm"
                  hint="Of the days that did sell, how many were flagged." />
                <Metric label="Precision" value={pct(o.precision * 100)} size="sm"
                  hint="Of the days flagged, how many really sold." />
                <Metric label="F1" value={o.f1.toFixed(3)} size="sm"
                  hint="Balance of the two above." />
                <Metric label="Accuracy" value={pct(o.accuracy * 100)} size="sm"
                  hint={`Beats always-guess-nothing (${pct(o.always_predict_no_demand_accuracy * 100)}).`} />
              </dl>
              <Explain>{o.caveat}</Explain>
            </Card>
          )}
        </Async>

        <Async query={prov} height="h-40">
          {(p) => (
            <Card title="What is behind these numbers" subtitle="Every figure traces to a stored artefact">
              <dl className="space-y-2.5 text-xs">
                <div className="flex justify-between gap-4">
                  <dt className="text-ink-muted">Validation windows with real outcomes</dt>
                  <dd className="tnum font-semibold text-ink">{p.backtest_origins.length}</dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt className="text-ink-muted">Backtest predictions scored</dt>
                  <dd className="tnum font-semibold text-ink">{compact(p.row_counts.backtest ?? 0)}</dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt className="text-ink-muted">Forecast values served</dt>
                  <dd className="tnum font-semibold text-ink">{compact(p.row_counts.forecast ?? 0)}</dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt className="text-ink-muted">Model fingerprint</dt>
                  <dd className="font-mono text-[10px] text-ink-muted">
                    {p.model_direct_sha256.slice(0, 12)}…
                  </dd>
                </div>
              </dl>
              <Explain>
                The deployed model files are hash-checked on every request, and the
                served forecast is verified against the frozen artefact before it
                reaches this page.
              </Explain>
            </Card>
          )}
        </Async>
      </div>

      {/* --- where to go next ----------------------------------------- */}
      <Card title="Start here">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {[
            { to: '/forecast', t: 'See a forecast', d: 'Pick any product and store, and view its next 28 days against its history.' },
            { to: '/hierarchy', t: 'Roll it up', d: 'Aggregate from a single item to a department, a store, or the whole chain.' },
            { to: '/insights', t: 'See what needs attention', d: 'The products and stores where the next 28 days move the most.' },
            { to: '/assistant', t: 'Ask a question', d: 'Query the forecast in plain language and get a grounded answer.' },
          ].map((c) => (
            <Link
              key={c.to}
              to={c.to}
              className="card card-hover block p-4 transition-transform hover:-translate-y-px"
            >
              <p className="text-sm font-semibold text-ink">{c.t}</p>
              <p className="mt-1.5 text-xs leading-relaxed text-ink-muted">{c.d}</p>
            </Link>
          ))}
        </div>
      </Card>
    </div>
  )
}
