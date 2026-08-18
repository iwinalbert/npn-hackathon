import type { ReactNode } from 'react'

function Node({
  children, tone = 'default', sub,
}: { children: ReactNode; tone?: 'default' | 'accent' | 'muted'; sub?: string }) {
  const tones = {
    default: 'border-line-strong bg-elevated text-ink',
    accent: 'border-forecast/50 bg-forecast/10 text-ink',
    muted: 'border-line bg-surface text-ink-muted',
  }
  return (
    <div className={`rounded border px-3 py-2 text-center ${tones[tone]}`}>
      <div className="text-xs font-semibold leading-tight">{children}</div>
      {sub && <div className="mt-0.5 text-[10px] leading-tight text-ink-dim">{sub}</div>}
    </div>
  )
}

function Arrow({ label }: { label?: string }) {
  return (
    <div className="flex flex-col items-center py-1" aria-hidden>
      <div className="h-4 w-px bg-line-strong" />
      {label && <span className="py-0.5 text-[10px] text-ink-dim">{label}</span>}
      <svg width="10" height="6" viewBox="0 0 10 6" className="fill-line-strong">
        <path d="M5 6L0 0h10z" />
      </svg>
    </div>
  )
}

export function PipelineDiagram() {
  return (
    <figure>
      <figcaption className="sr-only">
        Forecasting pipeline: sales history plus price and calendar context become
        lag and rolling features, which two LightGBM models turn into a blended
        28-day forecast.
      </figcaption>
      <div className="mx-auto max-w-md">
        <div className="grid grid-cols-3 gap-2">
          <Node tone="muted" sub="1,941 days">Sales history</Node>
          <Node tone="muted" sub="weekly">Price</Node>
          <Node tone="muted" sub="events, SNAP">Calendar</Node>
        </div>
        <Arrow label="frozen at the forecast origin" />
        <Node sub="lags, rolling means, recency, per-series shape">Features</Node>
        <Arrow />
        <div className="grid grid-cols-2 gap-2">
          <Node sub="38 features · all 28 days at once">Direct model</Node>
          <Node sub="32 features · 1 day, rolled 28×">Recursive model</Node>
        </div>
        <Arrow label="0.60 / 0.40" />
        <Node tone="accent" sub="30,490 store-item series">28-day forecast</Node>
      </div>
    </figure>
  )
}

export function HierarchyDiagram() {
  const rows = [
    { label: 'Chain total', n: '1' },
    { label: 'State', n: '3' },
    { label: 'Store', n: '10' },
    { label: 'Department', n: '7' },
    { label: 'Item', n: '3,049' },
    { label: 'Store × Item', n: '30,490', accent: true },
  ]
  return (
    <figure>
      <figcaption className="sr-only">
        The forecast is produced at the bottom store-item level and summed upward,
        so every aggregate is consistent with its parts.
      </figcaption>
      <div className="mx-auto max-w-sm space-y-1.5">
        {rows.map((r, i) => (
          <div key={r.label}>
            <div
              className={`flex items-center justify-between rounded border px-3 py-1.5 ${
                r.accent
                  ? 'border-forecast/50 bg-forecast/10'
                  : 'border-line bg-surface'
              }`}
              style={{ marginInline: `${(rows.length - 1 - i) * 6}px` }}
            >
              <span className="text-xs font-medium text-ink">{r.label}</span>
              <span className="tnum text-[11px] text-ink-dim">{r.n}</span>
            </div>
          </div>
        ))}
        <p className="pt-2 text-center text-[11px] leading-relaxed text-ink-muted">
          The model predicts only the <span className="text-forecast">bottom level</span>.
          Everything above is an exact sum of those predictions.
        </p>
      </div>
    </figure>
  )
}

export function ValidationDiagram() {
  return (
    <figure>
      <figcaption className="sr-only">
        Validation cuts the timeline at a forecast origin: the model sees only data
        before it and is scored on the 28 days after it.
      </figcaption>
      <div className="mx-auto max-w-lg">
        <div className="flex items-stretch gap-1 text-center">
          <div className="flex-[3] rounded-l border border-line bg-surface px-3 py-3">
            <div className="text-xs font-semibold text-ink">Training</div>
            <div className="mt-0.5 text-[10px] text-ink-dim">everything up to the origin</div>
          </div>
          <div className="flex w-px items-center bg-warn" aria-hidden />
          <div className="flex-[1] rounded-r border border-forecast/40 bg-forecast/10 px-3 py-3">
            <div className="text-xs font-semibold text-ink">Scored</div>
            <div className="mt-0.5 text-[10px] text-ink-dim">the next 28 days</div>
          </div>
        </div>
        <div className="mt-1 flex text-[10px] text-ink-dim">
          <div className="flex-[3] text-right pr-1">forecast origin ↑</div>
          <div className="flex-[1]" />
        </div>
        <p className="mt-3 text-[11px] leading-relaxed text-ink-muted">
          Every feature is frozen at the origin. Nothing after it reaches the model —
          verified by overwriting all post-origin sales with a nonsense value and
          checking that every feature comes back bit-for-bit identical.
        </p>
      </div>
    </figure>
  )
}

export function DataRowDiagram() {
  const fields = [
    { k: 'store', v: 'CA_3', note: '10 stores, 3 states' },
    { k: 'item', v: 'FOODS_3_090', note: '3,049 products' },
    { k: 'department', v: 'FOODS_3', note: '7 departments' },
    { k: 'date', v: '2016-05-22', note: '1,941 days' },
    { k: 'units sold', v: '112', note: 'the target' },
    { k: 'price', v: '$1.60', note: 'weekly, known ahead' },
    { k: 'event', v: 'none', note: 'holidays, SNAP' },
  ]
  return (
    <div className="overflow-hidden rounded border border-line">
      <div className="border-b border-line bg-elevated px-3 py-2 text-[11px] font-semibold text-ink-muted">
        One row = one product, in one store, on one day
      </div>
      <dl className="divide-y divide-line">
        {fields.map((f) => (
          <div key={f.k} className="grid grid-cols-[7rem_1fr_auto] items-baseline gap-3 px-3 py-2">
            <dt className="text-[11px] uppercase tracking-wide text-ink-dim">{f.k}</dt>
            <dd className="tnum text-xs font-medium text-ink">{f.v}</dd>
            <dd className="text-[10px] text-ink-dim">{f.note}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}
