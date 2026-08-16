import {
  Area, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'

import { type CombinedPoint, longDate, shortDate } from '../../lib/format'

/**
 * The product's primary chart: observed history flowing into a 28-day forecast.
 *
 * DESIGN DECISIONS
 * ----------------
 * * History is a SOLID line in a neutral tone; the forecast is a DASHED line in
 *   the single accent colour. The distinction is carried by dash pattern and
 *   position as well as colour, so it survives colour-blindness and greyscale.
 * * A labelled reference line marks the forecast origin — the moment the model
 *   stops seeing data and starts predicting. That boundary is the single most
 *   important thing to communicate.
 * * The shaded band is drawn UNDER the forecast line and is explicitly labelled
 *   as observed backtest error, never as a model interval.
 */
interface Props {
  data: CombinedPoint[]
  originDate: string
  height?: number
  showBand?: boolean
  /** Units label for the y-axis, e.g. "units/day". */
  yLabel?: string
}

function ForecastTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  const p: CombinedPoint = payload[0].payload
  return (
    <div className="rounded border border-line-strong bg-surface px-3 py-2 shadow-lg">
      <p className="text-xs font-semibold text-ink">{longDate(String(label))}</p>
      <dl className="mt-1.5 space-y-1 text-xs">
        {p.actual != null && (
          <div className="flex items-center justify-between gap-6">
            <dt className="text-ink-muted">Actual sold</dt>
            <dd className="tnum font-semibold text-actual">{p.actual.toFixed(0)}</dd>
          </div>
        )}
        {p.forecast != null && p.isForecast && (
          <div className="flex items-center justify-between gap-6">
            <dt className="text-ink-muted">Forecast</dt>
            <dd className="tnum font-semibold text-forecast">{p.forecast.toFixed(2)}</dd>
          </div>
        )}
        {p.lower != null && p.upper != null && (
          <div className="flex items-center justify-between gap-6">
            <dt className="text-ink-muted">Typical error range</dt>
            <dd className="tnum text-ink-muted">
              {p.lower.toFixed(1)} – {p.upper.toFixed(1)}
            </dd>
          </div>
        )}
      </dl>
      {p.isForecast && (
        <p className="mt-1.5 border-t border-line pt-1.5 text-[10px] text-ink-dim">
          Predicted — no actual exists for this date
        </p>
      )}
    </div>
  )
}

export function ForecastChart({
  data, originDate, height = 320, showBand = true, yLabel = 'units / day',
}: Props) {
  const hasBand = showBand && data.some((d) => d.band != null)

  return (
    <div>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 8, right: 12, left: 4, bottom: 4 }}>
          <CartesianGrid stroke="#243149" strokeDasharray="2 4" vertical={false} />
          <XAxis
            dataKey="date"
            tickFormatter={shortDate}
            stroke="#6B7C99"
            tick={{ fontSize: 11 }}
            minTickGap={28}
            tickLine={false}
            axisLine={{ stroke: '#243149' }}
          />
          <YAxis
            stroke="#6B7C99"
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={44}
            label={{
              value: yLabel, angle: -90, position: 'insideLeft',
              style: { fontSize: 10, fill: '#6B7C99' },
            }}
          />
          <Tooltip content={<ForecastTooltip />} cursor={{ stroke: '#31415F' }} />
          <Legend
            verticalAlign="top"
            height={28}
            iconType="plainline"
            wrapperStyle={{ fontSize: 11, color: '#9AA9C2' }}
          />

          {hasBand && (
            <Area
              dataKey="band"
              name="Typical error range (observed)"
              stroke="none"
              fill="#4EA8F0"
              fillOpacity={0.12}
              isAnimationActive={false}
              connectNulls
            />
          )}

          <Line
            dataKey="actual"
            name="Actual sales"
            stroke="#C6D2E4"
            strokeWidth={1.6}
            dot={false}
            connectNulls={false}
            isAnimationActive={false}
          />
          <Line
            dataKey="forecast"
            name="Forecast (28 days)"
            stroke="#4EA8F0"
            strokeWidth={2}
            strokeDasharray="5 3"
            dot={false}
            connectNulls
            isAnimationActive={false}
          />

          <ReferenceLine
            x={originDate}
            stroke="#D9A63C"
            strokeDasharray="3 3"
            label={{
              value: 'forecast origin',
              position: 'insideTopRight',
              fill: '#D9A63C',
              fontSize: 10,
            }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
