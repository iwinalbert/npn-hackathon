import { compact, nf, pct } from '../../lib/format'


export interface FlowNode {
  id: string
  label: string
  total: number | null
  nSeries: number
}

export interface FlowLevel {
  key: string
  title: string
  shape: 'box' | 'pill'
  nodes: FlowNode[]
}

export interface FlowBranch {
  key: string
  title: string
  caption: string
  levels: FlowLevel[]
}

interface Props {
  branches: FlowBranch[]
  bottom: { label: string; nSeries: number; horizonDays: number }
  chainTotal: number | null
  selected?: { level: string; nodeId: string }
  onSelect?: (level: string, nodeId: string) => void
}

function FlowArrow({ label }: { label?: string }) {
  return (
    <div className="flex flex-col items-center py-1.5" aria-hidden>
      <div className="h-4 w-px bg-line-strong" />
      {label && (
        <span className="py-1 text-center text-[10px] uppercase tracking-wider text-ink-dim">
          {label}
        </span>
      )}
      <svg width="11" height="7" viewBox="0 0 11 7" className="fill-line-strong">
        <path d="M5.5 7 L0 0 h11 z" />
      </svg>
    </div>
  )
}

function NodeChip({
  node, shape, level, share, selected, onSelect,
}: {
  node: FlowNode
  shape: 'box' | 'pill'
  level: string
  share: number | null
  selected: boolean
  onSelect?: (level: string, nodeId: string) => void
}) {
  const radius = shape === 'pill' ? 'rounded-full' : 'rounded'
  return (
    <li className="group relative">
      <button
        type="button"
        onClick={() => onSelect?.(level, node.id)}
        aria-pressed={selected}
        className={`w-full min-w-[6.5rem] border px-3 py-2 text-left transition-colors ${radius} ${
          selected
            ? 'border-forecast/60 bg-forecast/10'
            : 'border-line bg-base hover:border-line-strong hover:bg-elevated'
        }`}
      >
        <span className="block truncate text-xs font-semibold text-ink">{node.label}</span>
        <span className="tnum mt-0.5 block text-[11px] text-ink-muted">
          {node.total != null ? `${compact(node.total)} units` : '…'}
        </span>
        {share != null && (
          <span
            aria-hidden
            className={`mt-1.5 block h-1 overflow-hidden bg-elevated ${
              shape === 'pill' ? 'rounded-full' : 'rounded-sm'
            }`}
          >
            <span
              className="block h-full rounded-full bg-forecast/70"
              style={{ width: `${Math.max(share * 100, 2)}%` }}
            />
          </span>
        )}
      </button>

      {/* Hover detail, styled like the chart tooltip so the two read as one app. */}
      <div
        role="tooltip"
        className="pointer-events-none absolute bottom-full left-1/2 z-20 mb-1.5 hidden
                   w-max max-w-[15rem] -translate-x-1/2 rounded border border-line-strong
                   bg-surface px-3 py-2 shadow-lg group-hover:block group-focus-within:block"
      >
        <p className="text-xs font-semibold text-ink">{node.label}</p>
        <dl className="mt-1.5 space-y-1 text-xs">
          <div className="flex justify-between gap-4">
            <dt className="text-ink-muted">28-day forecast</dt>
            <dd className="tnum font-semibold text-forecast">
              {node.total != null ? `${nf(Math.round(node.total))} units` : 'loading…'}
            </dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt className="text-ink-muted">Share of chain</dt>
            <dd className="tnum text-ink">{share != null ? pct(share * 100) : '—'}</dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt className="text-ink-muted">Store-item series</dt>
            <dd className="tnum text-ink">{nf(node.nSeries)}</dd>
          </div>
        </dl>
        <p className="mt-1.5 border-t border-line pt-1.5 text-[10px] text-ink-dim">
          Sum of this node's bottom-level forecasts. Click to chart it.
        </p>
      </div>
    </li>
  )
}

function LevelRow({
  level, chainTotal, selected, onSelect,
}: {
  level: FlowLevel
  chainTotal: number | null
  selected?: { level: string; nodeId: string }
  onSelect?: (level: string, nodeId: string) => void
}) {
  const nodes = [...level.nodes].sort((a, b) => (b.total ?? -1) - (a.total ?? -1))
  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-ink-dim">
          {level.title}
        </p>
        <p className="tnum text-[10px] text-ink-dim">{nodes.length} nodes</p>
      </div>
      <ul className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-4">
        {nodes.map((n) => (
          <NodeChip
            key={n.id}
            node={n}
            shape={level.shape}
            level={level.key}
            share={chainTotal && n.total != null ? n.total / chainTotal : null}
            selected={selected?.level === level.key && selected?.nodeId === n.id}
            onSelect={onSelect}
          />
        ))}
      </ul>
    </div>
  )
}

export function HierarchyFlow({
  branches, bottom, chainTotal, selected, onSelect,
}: Props) {
  return (
    <figure className="m-0">
      <figcaption className="sr-only">
        Diagram. The model forecasts {nf(bottom.nSeries)} store-item series. Those
        forecasts are summed two ways — by geography into stores and states, and by
        product into departments and categories — and both paths arrive at the same
        chain total.
      </figcaption>

      {/* --- the only level the model predicts ------------------------- */}
      <div className="rounded-lg border border-forecast/40 bg-forecast/5 px-4 py-3 text-center">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-forecast">
          Bottom level · the only level the model predicts
        </p>
        <p className="mt-1 text-sm font-semibold text-ink">
          {bottom.label} — {nf(bottom.nSeries)} forecasts
        </p>
        <p className="mt-0.5 text-xs text-ink-muted">
          One forecast per product, per store, for each of the next {bottom.horizonDays} days
        </p>
      </div>

      {/* --- two roll-up paths ----------------------------------------- */}
      <div className="grid gap-x-6 lg:grid-cols-2">
        {branches.map((b) => (
          <FlowArrow key={`arrow-${b.key}`} label={b.caption} />
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {branches.map((b) => (
          <div key={b.key} className="rounded-lg border border-line bg-base/40 p-3 sm:p-4">
            <p className="mb-3 text-xs font-semibold text-ink">{b.title}</p>
            {b.levels.map((level, i) => (
              <div key={level.key}>
                {i > 0 && <FlowArrow label="sums into" />}
                <LevelRow
                  level={level}
                  chainTotal={chainTotal}
                  selected={selected}
                  onSelect={onSelect}
                />
              </div>
            ))}
          </div>
        ))}
      </div>

      {/* --- both paths converge --------------------------------------- */}
      <div className="grid gap-x-6 lg:grid-cols-2">
        {branches.map((b) => (
          <FlowArrow key={`join-${b.key}`} label="sums into" />
        ))}
      </div>

      <div className="mx-auto max-w-md rounded-lg border-2 border-forecast/50 bg-forecast/10 px-5 py-4 text-center">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-ink-dim">
          Chain total · next {bottom.horizonDays} days
        </p>
        <p className="tnum mt-1 text-metric font-semibold text-forecast">
          {chainTotal != null ? compact(chainTotal) : '…'}
          <span className="ml-1.5 text-sm font-normal text-ink-muted">units</span>
        </p>
        <p className="mt-1 text-xs leading-relaxed text-ink-muted">
          Both paths sum the same {nf(bottom.nSeries)} forecasts, so both land on this
          number. That is what makes the hierarchy coherent — there is no
          reconciliation step to disagree about.
        </p>
      </div>
    </figure>
  )
}
