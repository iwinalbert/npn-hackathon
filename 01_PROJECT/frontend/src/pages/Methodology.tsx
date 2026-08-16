import { useCapabilities, useProvenance, useRegimeAccuracy } from '../api/hooks'
import { DataRowDiagram } from '../components/diagrams'
import { Async, Badge, Card, Caveat, Explain } from '../components/ui'
import { REGIME_COLORS, compact, nf, pct } from '../lib/format'
import type { Capability } from '../api/types'

function CapabilityList({
  items, tone,
}: { items: Capability[]; tone: 'good' | 'warn' | 'neutral' }) {
  const border = {
    good: 'border-l-good/50',
    warn: 'border-l-warn/50',
    neutral: 'border-l-line-strong',
  }[tone]
  return (
    <ul className="space-y-3">
      {items.map((c) => (
        <li key={c.name} className={`border-l-2 pl-3 ${border}`}>
          <p className="text-xs font-semibold text-ink">{c.name}</p>
          <p className="mt-1 text-[11px] leading-relaxed text-ink-muted">{c.detail}</p>
          {c.evidence && (
            <p className="mt-1 font-mono text-[10px] leading-relaxed text-ink-dim">
              {c.evidence}
            </p>
          )}
        </li>
      ))}
    </ul>
  )
}

/**
 * The page that makes this a research demonstrator rather than a dashboard.
 *
 * The capability matrix is fetched from the backend, not written here, so the
 * claims on screen can never drift from what the system actually supports.
 */
export function Methodology() {
  const caps = useCapabilities()
  const regimes = useRegimeAccuracy()
  const prov = useProvenance()

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-ink">Methodology</h1>
        <p className="mt-1 text-sm text-ink-muted">
          The data, what the model can and cannot do, and what the research tried and
          rejected.
        </p>
      </div>

      {/* --- the data ------------------------------------------------- */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="What the data is">
          <DataRowDiagram />
          <Explain>
            The Walmart M5 dataset: 3,049 products across 10 stores in California,
            Texas and Wisconsin, recorded daily for 1,941 days (2011–2016). That is
            30,490 store-item series and about 59.2 million daily observations.
          </Explain>
        </Card>

        <Async query={regimes} height="h-64">
          {(r) => {
            const total = r.regimes.reduce((s, x) => s + x.n_series, 0)
            return (
              <Card
                title="Why this is hard: intermittent demand"
                subtitle="Most products do not sell every day"
              >
                <ul className="space-y-2.5">
                  {r.regimes
                    .filter((x) => x.regime !== 'never sold')
                    .sort((a, b) => b.n_series - a.n_series)
                    .map((x) => (
                      <li key={x.regime}>
                        <div className="flex items-baseline justify-between gap-2">
                          <span className="flex items-center gap-1.5 text-xs text-ink">
                            <span aria-hidden className="h-2 w-2 rounded-full"
                              style={{ background: REGIME_COLORS[x.regime] }} />
                            {x.regime}
                          </span>
                          <span className="tnum text-[11px] text-ink-dim">
                            {nf(x.n_series)} series · {pct(x.zero_rate_pct)} zero days
                          </span>
                        </div>
                        <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-elevated">
                          <div className="h-full rounded-full"
                            style={{
                              width: `${(x.n_series / total) * 100}%`,
                              background: REGIME_COLORS[x.regime],
                              opacity: 0.7,
                            }} />
                        </div>
                      </li>
                    ))}
                </ul>
                <Explain>
                  Many store-item combinations sell nothing on most days. Ordinary
                  forecasting methods assume a smooth, continuous quantity and cope
                  badly with that. The model handles it with a Tweedie objective, which
                  is built for data that is mostly zeros with occasional positive counts
                  — not by a separate specialist model.
                </Explain>
              </Card>
            )
          }}
        </Async>
      </div>

      {/* --- capability matrix ---------------------------------------- */}
      <Async query={caps} height="h-96">
        {(c) => (
          <div className="space-y-6">
            <Card
              title="What this system genuinely does"
              subtitle="Every claim below is backed by a recorded experiment"
              actions={<Badge variant="good">{c.implemented.length} capabilities</Badge>}
            >
              <CapabilityList items={c.implemented} tone="good" />
            </Card>

            <Card
              title="What the research tried and rejected"
              subtitle="Measured, found not to help, and deliberately left out"
              actions={<Badge variant="warn">{c.rejected.length} rejected</Badge>}
            >
              <CapabilityList items={c.rejected} tone="warn" />
              <Explain>
                These are not oversights. Each was implemented, measured against the
                champion, and turned down because the evidence did not support it.
                Publishing the rejections is what separates a validated result from a
                list of techniques.
              </Explain>
            </Card>

            <Card
              title="What this system does not do"
              subtitle="Limits of the model and of the data itself"
              actions={<Badge variant="bad">{c.not_supported.length} limits</Badge>}
            >
              <CapabilityList items={c.not_supported} tone="neutral" />
              <Caveat>
                In particular there is <strong>no price what-if simulator</strong>. The
                model uses price as context for forecasting, not as a causal lever.
                When price changes were simulated during research, the predicted
                response was inconsistent and sometimes economically backwards, so
                exposing a price slider would have produced confident nonsense.
              </Caveat>
            </Card>
          </div>
        )}
      </Async>

      {/* --- covariates ----------------------------------------------- */}
      <Card title="External information the model uses">
        <div className="grid gap-4 sm:grid-cols-3">
          {[
            {
              t: 'Price',
              s: 'Used',
              tone: 'good' as const,
              d: 'Weekly selling price, plus how it compares to that product’s own recent average. Known in advance for the whole forecast window.',
            },
            {
              t: 'Calendar & holidays',
              s: 'Used',
              tone: 'good' as const,
              d: 'Day of week, month, and named events such as holidays and sporting occasions. Published years ahead, so using them is legitimate.',
            },
            {
              t: 'Promotions',
              s: 'Not available',
              tone: 'warn' as const,
              d: 'The dataset contains no promotion field at all. A discount proxy was tested and carried no additional signal, so nothing here claims to model promotions.',
            },
          ].map((x) => (
            <div key={x.t} className="rounded border border-line bg-base p-4">
              <div className="flex items-center justify-between gap-2">
                <p className="text-xs font-semibold text-ink">{x.t}</p>
                <Badge variant={x.tone}>{x.s}</Badge>
              </div>
              <p className="mt-2 text-[11px] leading-relaxed text-ink-muted">{x.d}</p>
            </div>
          ))}
        </div>
        <Explain>
          SNAP benefit days are also a model input, matched to each store's own state.
          Every one of these is known ahead of time — nothing the model uses for a future
          day requires forecasting that input first.
        </Explain>
      </Card>

      {/* --- limitations ---------------------------------------------- */}
      <Card title="Honest limitations" subtitle="Where this system is weak, stated plainly">
        <ul className="space-y-3">
          {[
            {
              t: 'Validation is not deployment accuracy',
              d: 'Every figure on this site comes from historical windows with known outcomes. The delivered 28-day forecast has no recorded outcome, so its true accuracy is unknown and is never quoted.',
            },
            {
              t: 'The error range is measured, not predicted',
              d: 'The model produces a single number per day, not a probability distribution. The shaded ranges are how wrong this model has been on similar products in past tests, which is useful but is not a statistical prediction interval.',
            },
            {
              t: 'Stockouts are invisible',
              d: 'A recorded zero may mean nobody wanted the product, or that it was not on the shelf. The dataset cannot distinguish the two, so neither can the model, and demand may be understated for items that ran out.',
            },
            {
              t: 'Accuracy is modest at the individual product level',
              d: 'About 28% at store-item-day. That is a property of forecasting sparse retail demand, not a defect — but it means single-product forecasts should inform decisions, not make them alone.',
            },
            {
              t: 'RMSE was prioritised over MAE',
              d: 'The chosen blend improves RMSE while slightly worsening MAE against a single model. That trade was deliberate, because the cost of a large miss in inventory is worse than several small ones.',
            },
            {
              t: 'One dataset, one retailer, one period',
              d: 'Everything here is validated on Walmart M5 data ending in 2016. Nothing establishes how it would transfer to another retailer, another category mix, or a different demand environment.',
            },
          ].map((l) => (
            <li key={l.t} className="border-l-2 border-l-line-strong pl-3">
              <p className="text-xs font-semibold text-ink">{l.t}</p>
              <p className="mt-1 text-[11px] leading-relaxed text-ink-muted">{l.d}</p>
            </li>
          ))}
        </ul>
      </Card>

      {/* --- provenance ------------------------------------------------ */}
      <Async query={prov} height="h-32">
        {(p) => (
          <Card title="Where every number on this site comes from">
            <dl className="space-y-2 text-[11px]">
              {Object.entries(p.sources).map(([k, v]) => (
                <div key={k} className="flex flex-wrap justify-between gap-2 border-b border-line pb-2 last:border-0">
                  <dt className="font-medium text-ink-muted">{k}</dt>
                  <dd className="break-all text-right font-mono text-ink-dim">{v}</dd>
                </div>
              ))}
            </dl>
            <Explain>
              {p.backtest_origins.length} validation windows ·{' '}
              {compact(p.row_counts.backtest ?? 0)} scored backtest predictions ·{' '}
              {compact(p.row_counts.forecast ?? 0)} forecast values. The research layer is
              read-only: this application cannot modify a model, a dataset or a recorded
              result.
            </Explain>
          </Card>
        )}
      </Async>
    </div>
  )
}
