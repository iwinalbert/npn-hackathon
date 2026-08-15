import {
  Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts'

import {
  useHorizonAccuracy, useMembers, useOccurrence, useRegimeAccuracy,
  useVolumeTiers, useWindows,
} from '../api/hooks'
import { Async, Badge, Card, Caveat, Explain, Metric } from '../components/ui'
import { REGIME_COLORS, compact, nf, pct } from '../lib/format'

const axis = { stroke: '#6B7C99', tick: { fontSize: 11 }, tickLine: false } as const
const tooltipStyle = {
  contentStyle: {
    background: '#131C2E', border: '1px solid #31415F', borderRadius: 4, fontSize: 12,
  },
  labelStyle: { color: '#E8EDF6' },
} as const

/** Confusion matrix rendered as a labelled 2×2 grid, not colour alone. */
function ConfusionMatrix({ cm, n }: { cm: Record<string, number>; n: number }) {
  const cells = [
    { k: 'true_positive', label: 'Predicted a sale, and it sold', tone: 'good' },
    { k: 'false_positive', label: 'Predicted a sale, nothing sold', tone: 'warn' },
    { k: 'false_negative', label: 'Predicted nothing, but it sold', tone: 'bad' },
    { k: 'true_negative', label: 'Predicted nothing, nothing sold', tone: 'good' },
  ] as const
  const toneClass = {
    good: 'border-good/30 bg-good/5',
    warn: 'border-warn/30 bg-warn/5',
    bad: 'border-bad/30 bg-bad/5',
  }
  return (
    <div className="grid grid-cols-2 gap-2">
      {cells.map((c) => (
        <div key={c.k} className={`rounded border px-3 py-2.5 ${toneClass[c.tone]}`}>
          <p className="tnum text-lg font-semibold text-ink">{compact(cm[c.k] ?? 0)}</p>
          <p className="tnum text-[10px] text-ink-dim">
            {pct(((cm[c.k] ?? 0) / n) * 100)} of days
          </p>
          <p className="mt-1 text-[11px] leading-snug text-ink-muted">{c.label}</p>
        </div>
      ))}
    </div>
  )
}

export function Accuracy() {
  const windows = useWindows()
  const horizon = useHorizonAccuracy()
  const regimes = useRegimeAccuracy()
  const tiers = useVolumeTiers()
  const occ = useOccurrence()
  const members = useMembers()

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-ink">Accuracy</h1>
        <p className="mt-1 text-sm text-ink-muted">
          Historical validation performance, measured on days the model never saw
          during training.
        </p>
      </div>

      <Caveat>
        Everything on this page is <strong>historical validation performance</strong>,
        not live deployment accuracy. The delivered 28-day forecast covers days that
        have no recorded outcome, so no accuracy figure can honestly be quoted
        against it.
      </Caveat>

      {/* --- windows -------------------------------------------------- */}
      <Async query={windows} height="h-64">
        {(w) => {
          const primary = w.find((x) => x.is_primary_validation_window) ?? w[0]
          const mean = (k: 'rmse' | 'mae') =>
            w.reduce((s, x) => s + x[k], 0) / w.length
          return (
            <Card
              title="Performance across 8 independent time windows"
              subtitle="Each is a separate 28-day period with known outcomes"
            >
              <dl className="mb-5 grid grid-cols-2 gap-x-6 gap-y-5 md:grid-cols-4">
                <Metric label="RMSE (primary window)" value={primary.rmse.toFixed(4)} tone="forecast"
                  hint="Average miss, penalising large misses hardest." />
                <Metric label="MAE (primary window)" value={primary.mae.toFixed(4)}
                  hint="Typical miss in units." />
                <Metric label="Mean RMSE, all windows" value={mean('rmse').toFixed(4)} size="sm"
                  hint="Consistency across different seasons." />
                <Metric label="Bias" value={primary.bias.toFixed(4)} size="sm"
                  tone={Math.abs(primary.bias) < 0.05 ? 'good' : 'warn'}
                  hint="Negative means it slightly under-forecasts on average." />
              </dl>

              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <caption className="sr-only">
                    Validation metrics for each of the eight backtest windows
                  </caption>
                  <thead>
                    <tr className="border-b border-line text-left text-ink-dim">
                      <th scope="col" className="py-2 pr-3 font-medium">Window</th>
                      <th scope="col" className="py-2 pr-3 font-medium">Period</th>
                      <th scope="col" className="py-2 pr-3 text-right font-medium">RMSE</th>
                      <th scope="col" className="py-2 pr-3 text-right font-medium">MAE</th>
                      <th scope="col" className="py-2 pr-3 text-right font-medium">WAPE</th>
                      <th scope="col" className="py-2 text-right font-medium">Bias</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-line">
                    {w.map((x) => (
                      <tr key={x.origin_idx} className="transition-colors hover:bg-elevated">
                        <td className="py-2 pr-3 font-mono text-[11px] text-ink-muted">
                          {x.origin_day}
                          {x.is_primary_validation_window && (
                            <span className="ml-2"><Badge variant="info">primary</Badge></span>
                          )}
                        </td>
                        <td className="py-2 pr-3 text-ink-muted">
                          {x.window_start} → {x.window_end}
                        </td>
                        <td className="tnum py-2 pr-3 text-right font-medium text-ink">
                          {x.rmse.toFixed(4)}
                        </td>
                        <td className="tnum py-2 pr-3 text-right text-ink-muted">
                          {x.mae.toFixed(4)}
                        </td>
                        <td className="tnum py-2 pr-3 text-right text-ink-muted">
                          {x.wape.toFixed(4)}
                        </td>
                        <td className="tnum py-2 text-right text-ink-muted">
                          {x.bias.toFixed(4)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <Explain>
                Eight windows rather than one matters: a model can look good on a
                single lucky month. WAPE is total error as a share of total demand —
                0.72 means the errors add up to 72% of the units actually sold at this
                very granular level.
              </Explain>
            </Card>
          )
        }}
      </Async>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* --- horizon ------------------------------------------------ */}
        <Async query={horizon} height="h-64">
          {(h) => (
            <Card
              title="Does accuracy decay further out?"
              subtitle="Error by days ahead, within the 28-day horizon"
            >
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={h} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="#243149" strokeDasharray="2 4" vertical={false} />
                  <XAxis dataKey="horizon" {...axis} axisLine={{ stroke: '#243149' }}
                    label={{ value: 'days ahead', position: 'insideBottom', offset: -2,
                      style: { fontSize: 10, fill: '#6B7C99' } }} />
                  <YAxis {...axis} axisLine={false} width={40} />
                  <Tooltip {...tooltipStyle}
                    formatter={(v: number, n: string) => [v.toFixed(4), n.toUpperCase()]}
                    labelFormatter={(l) => `Day ${l} ahead`} />
                  <Line dataKey="rmse" name="rmse" stroke="#4EA8F0" strokeWidth={2}
                    dot={false} isAnimationActive={false} />
                  <Line dataKey="mae" name="mae" stroke="#9AA9C2" strokeWidth={1.5}
                    strokeDasharray="4 3" dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
              <Explain>
                Error stays remarkably flat across the horizon — day 28 is barely worse
                than day 1. That is a property of this design: all 28 days are predicted
                in one shot from the same frozen information, so nothing degrades as the
                forecast walks forward. The visible bumps are weekly seasonality.
              </Explain>
            </Card>
          )}
        </Async>

        {/* --- volume tiers ------------------------------------------ */}
        <Async query={tiers} height="h-64">
          {(t) => (
            <Card
              title="Where the error actually lives"
              subtitle="By how much a product typically sells"
            >
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={t.tiers} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="#243149" strokeDasharray="2 4" vertical={false} />
                  <XAxis dataKey="volume_tier" {...axis} axisLine={{ stroke: '#243149' }} />
                  <YAxis {...axis} axisLine={false} width={40} />
                  <Tooltip {...tooltipStyle}
                    formatter={(v: number, n: string) =>
                      [n === 'rmse' ? v.toFixed(4) : `${v.toFixed(1)}%`,
                       n === 'rmse' ? 'RMSE' : 'Share of total error']} />
                  <Bar dataKey="rmse" name="rmse" radius={[3, 3, 0, 0]}>
                    {t.tiers.map((x) => (
                      <Cell key={x.volume_tier}
                        fill={x.volume_tier === 'high' ? '#E0665F' : '#4EA8F0'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="mt-3 space-y-1.5">
                {t.tiers.map((x) => (
                  <div key={x.volume_tier} className="flex items-center gap-2 text-[11px]">
                    <span className="w-16 text-ink-muted">{x.volume_tier}</span>
                    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-elevated">
                      <div className="h-full rounded-full bg-bad/70"
                        style={{ width: `${x.share_of_squared_error_pct}%` }} />
                    </div>
                    <span className="tnum w-24 text-right text-ink-dim">
                      {pct(x.share_of_rows_pct)} of rows
                    </span>
                    <span className="tnum w-20 text-right font-medium text-ink">
                      {pct(x.share_of_squared_error_pct)} err
                    </span>
                  </div>
                ))}
              </div>
              <Explain>{t.note}</Explain>
            </Card>
          )}
        </Async>
      </div>

      {/* --- regimes ------------------------------------------------- */}
      <Async query={regimes} height="h-64">
        {(r) => (
          <Card
            title="Performance by demand pattern"
            subtitle="Products behave very differently, and the model is scored on each kind"
          >
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {r.regimes.filter((x) => x.regime !== 'never sold').map((x) => (
                <div key={x.regime} className="rounded border border-line bg-base p-3">
                  <div className="flex items-center gap-2">
                    <span aria-hidden className="h-2 w-2 rounded-full"
                      style={{ background: REGIME_COLORS[x.regime] }} />
                    <span className="text-xs font-semibold text-ink">{x.regime}</span>
                  </div>
                  <p className="tnum mt-2 text-lg font-semibold text-ink">{x.rmse.toFixed(3)}</p>
                  <p className="text-[10px] text-ink-dim">RMSE</p>
                  <dl className="mt-2 space-y-0.5 text-[11px] text-ink-muted">
                    <div className="flex justify-between">
                      <dt>Series</dt><dd className="tnum">{nf(x.n_series)}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt>Zero-sale days</dt><dd className="tnum">{pct(x.zero_rate_pct)}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt>Share of error</dt><dd className="tnum">{pct(x.share_of_squared_error_pct)}</dd>
                    </div>
                  </dl>
                </div>
              ))}
            </div>
            <Explain>
              These are the Syntetos-Boylan demand classes. <em>Intermittent</em> products
              sell rarely but predictably; <em>lumpy</em> ones are irregular in both timing
              and size and are the hardest to forecast. Reporting error separately per class
              stops an easy majority from hiding a weak minority.
            </Explain>
          </Card>
        )}
      </Async>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* --- occurrence -------------------------------------------- */}
        <Async query={occ} height="h-64">
          {(o) => (
            <Card
              title="Spotting the days that sell"
              subtitle={o.rule}
            >
              <dl className="mb-4 grid grid-cols-4 gap-3">
                <Metric label="Accuracy" value={pct(o.accuracy * 100)} size="sm" />
                <Metric label="Precision" value={pct(o.precision * 100)} size="sm" />
                <Metric label="Recall" value={pct(o.recall * 100)} size="sm" />
                <Metric label="F1" value={o.f1.toFixed(3)} size="sm" />
              </dl>
              <ConfusionMatrix cm={o.confusion_matrix} n={o.n} />
              <Explain>
                <strong>Recall</strong> answers: of the days that really sold, how many did
                the model flag? <strong>Precision</strong> answers: of the days it flagged,
                how many really sold. This model leans toward recall — it would rather warn
                you about a sale that does not happen than miss one that does.
              </Explain>
              <Caveat>{o.caveat}</Caveat>
            </Card>
          )}
        </Async>

        {/* --- members ----------------------------------------------- */}
        <Async query={members} height="h-64">
          {(m) => {
            const rows = [
              ...m.members.map((x) => ({ name: x.name, rmse: x.rmse, blend: false })),
              { name: m.blend.name, rmse: m.blend.rmse, blend: true },
            ]
            return (
              <Card title="Why two models instead of one" subtitle={m.window}>
                <ResponsiveContainer width="100%" height={160}>
                  <BarChart data={rows} layout="vertical"
                    margin={{ top: 0, right: 40, left: 0, bottom: 0 }}>
                    <XAxis type="number" domain={['dataMin - 0.02', 'dataMax + 0.01']} hide />
                    <YAxis type="category" dataKey="name" width={150}
                      stroke="#6B7C99" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
                    <Tooltip {...tooltipStyle} formatter={(v: number) => [v.toFixed(4), 'RMSE']} />
                    <Bar dataKey="rmse" radius={[0, 3, 3, 0]}>
                      {rows.map((r) => (
                        <Cell key={r.name} fill={r.blend ? '#3FB98B' : '#4EA8F0'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <dl className="mt-3 grid grid-cols-2 gap-4">
                  <Metric label="Blend gain" value={m.gain_vs_best_member.toFixed(4)} size="sm"
                    tone="good" hint="RMSE improvement over the better single model." />
                  <Metric label="Error correlation" value={m.residual_correlation.toFixed(3)}
                    size="sm" hint="Below 1.0 means the two models fail differently." />
                </dl>
                <Explain>{m.why_it_works}</Explain>
              </Card>
            )
          }}
        </Async>
      </div>
    </div>
  )
}
