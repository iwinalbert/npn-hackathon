import { useId, useMemo } from 'react'
import {
  Area, CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'

import { type DemandPoint, compact, longDate, shortDate } from '../../lib/format'

const COLOR = {
  actual: '#C6D2E4',    // observed reality — the app's neutral "actual" tone
  predicted: '#4EA8F0', // the model
  horizon: '#7CC6FF',   // far end of the horizon, lighter as certainty thins
  origin: '#D9A63C',
  grid: '#243149',
  axis: '#6B7C99',
}

export interface Props {
  data: DemandPoint[]
  forecastStartDate?: string
  height?: number | string
  showBand?: boolean
  yLabel?: string
  everyDate?: boolean
  pxPerDate?: number
}

const tickLabel = (v: number) => (Math.abs(v) >= 1000 ? compact(v) : String(Number(v.toFixed(2))))

function niceStep(raw: number): number {
  if (!Number.isFinite(raw) || raw <= 0) return 1
  const mag = 10 ** Math.floor(Math.log10(raw))
  const norm = raw / mag
  const step = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10
  return step * mag
}

function niceTicks(max: number, count = 5): number[] {
  const step = niceStep(max / (count - 1))
  return Array.from({ length: count }, (_, i) => Number((i * step).toPrecision(12)))
}

function DemandTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  const p: DemandPoint = payload[0].payload
  const gap = p.actual != null && p.predicted != null ? p.predicted - p.actual : null

  return (
    <div className="max-w-[16rem] rounded border border-line-strong bg-surface px-3 py-2 shadow-lg">
      <p className="text-xs font-semibold text-ink">{longDate(String(label))}</p>
      <dl className="mt-2 space-y-1 text-xs">
        <div className="flex items-center justify-between gap-4">
          <dt className="flex items-center gap-1.5 text-ink-muted">
            <span aria-hidden className="h-0.5 w-3 rounded-full" style={{ background: COLOR.actual }} />
            Actual Demand
          </dt>
          <dd className="tnum font-semibold" style={{ color: COLOR.actual }}>
            {p.actual != null ? tickLabel(p.actual) : '—'}
          </dd>
        </div>
        <div className="flex items-center justify-between gap-4">
          <dt className="flex items-center gap-1.5 text-ink-muted">
            <span
              aria-hidden
              className="h-0.5 w-3 rounded-full"
              style={{ background: COLOR.predicted }}
            />
            Predicted Demand
          </dt>
          <dd className="tnum font-semibold" style={{ color: COLOR.predicted }}>
            {p.predicted != null ? tickLabel(p.predicted) : '—'}
          </dd>
        </div>
        {gap != null && (
          <div className="flex items-center justify-between gap-4 border-t border-line pt-1">
            <dt className="text-ink-muted">Difference</dt>
            <dd className={`tnum font-semibold ${gap >= 0 ? 'text-good' : 'text-warn'}`}>
              {gap >= 0 ? '+' : ''}{tickLabel(gap)}
            </dd>
          </div>
        )}
        {p.lower != null && p.upper != null && (
          <div className="flex items-center justify-between gap-4">
            <dt className="text-ink-muted">Typical error range</dt>
            <dd className="tnum text-ink-muted">
              {tickLabel(p.lower)} – {tickLabel(p.upper)}
            </dd>
          </div>
        )}
      </dl>
      <p className="mt-2 border-t border-line pt-1.5 text-[10px] leading-snug text-ink-dim">
        {p.phase === 'forecast'
          ? 'Predicted — this day has not happened, so there is no actual'
          : p.phase === 'compared'
            ? 'Held-out window — the model never saw this day in training'
            : 'Observed demand'}
      </p>
    </div>
  )
}

function ChartLegend({ hasBand, hasCompared }: { hasBand: boolean; hasCompared: boolean }) {
  return (
    <ul className="mb-3 flex flex-wrap items-center gap-x-5 gap-y-2 px-1 text-xs text-ink-muted">
      <li className="flex items-center gap-2">
        <svg width="34" height="10" aria-hidden className="shrink-0">
          <line x1="0" y1="5" x2="34" y2="5" stroke={COLOR.actual} strokeWidth="2.6" />
          <circle cx="17" cy="5" r="3.4" fill={COLOR.actual} />
        </svg>
        <span className="font-medium text-ink">Actual Demand</span>
      </li>
      <li className="flex items-center gap-2">
        <svg width="34" height="10" aria-hidden className="shrink-0">
          <line
            x1="0" y1="5" x2="34" y2="5"
            stroke={COLOR.predicted} strokeWidth="2.6" strokeDasharray="7 5"
          />
          <path d="M17 1.2 L20.6 5 L17 8.8 L13.4 5 Z" fill={COLOR.predicted} />
        </svg>
        <span className="font-medium text-ink">Predicted Demand</span>
      </li>
      {hasBand && (
        <li className="flex items-center gap-2">
          <span
            aria-hidden
            className="h-3 w-7 shrink-0 rounded-sm border border-forecast/30 bg-forecast/15"
          />
          Typical error range
        </li>
      )}
      {hasCompared && (
        <li className="flex items-center gap-2">
          <span aria-hidden className="h-3 w-7 shrink-0 rounded-sm bg-elevated" />
          Compared window (held out)
        </li>
      )}
      <li className="flex items-center gap-2">
        <svg width="10" height="12" aria-hidden className="shrink-0">
          <line
            x1="5" y1="0" x2="5" y2="12"
            stroke={COLOR.origin} strokeWidth="1.6" strokeDasharray="3 3"
          />
        </svg>
        Forecast Start
      </li>
    </ul>
  )
}

function ActualDot(props: any) {
  const { cx, cy, payload, index } = props
  if (cx == null || cy == null || payload?.actual == null) return <g key={`a-empty-${index}`} />
  return <circle key={`a-${index}`} cx={cx} cy={cy} r={3} fill={COLOR.actual} />
}

function PredictedDot(props: any) {
  const { cx, cy, payload, index } = props
  if (cx == null || cy == null || payload?.predicted == null) return <g key={`p-empty-${index}`} />
  return (
    <path
      key={`p-${index}`}
      d={`M ${cx} ${cy - 3.6} L ${cx + 3.6} ${cy} L ${cx} ${cy + 3.6} L ${cx - 3.6} ${cy} Z`}
      fill={COLOR.predicted}
    />
  )
}

function endLabel(atIndex: number, text: string, color: string, dy: number) {
  return function EndLabel(props: any) {
    const { x, y, index } = props
    if (index !== atIndex || x == null || y == null) return <g key={`skip-${index}`} />
    return (
      <g key={`end-${text}`} transform={`translate(${x}, ${y})`} aria-hidden>
        <line x1={-8} y1={dy + 4} x2={-2} y2={-2} stroke={color} strokeWidth={1} opacity={0.6} />
        <text
          x={-10}
          y={dy}
          textAnchor="end"
          fill={color}
          fontSize={12}
          fontWeight={600}
          style={{ paintOrder: 'stroke', stroke: '#131C2E', strokeWidth: 4 }}
        >
          {text}
        </text>
      </g>
    )
  }
}

export function ActualVsPredictedChart({
  data, forecastStartDate, height = 420, showBand = true,
  yLabel = 'Demand (units / day)', everyDate = true, pxPerDate = 18,
}: Props) {
  const gradientId = `${useId()}-predicted`
  const hasBand = showBand && data.some((d) => d.band != null)

  const firstCompared = data.findIndex((d) => d.phase === 'compared')
  const firstForecast = data.findIndex((d) => d.phase === 'forecast')
  const lastComparedDate = firstForecast > 0 ? data[firstForecast - 1]?.date : data.at(-1)?.date
  const startDate = forecastStartDate ?? (firstForecast >= 0 ? data[firstForecast]?.date : undefined)

  const lastActual = useMemo(
    () => data.reduce((last, d, i) => (d.actual != null ? i : last), -1), [data])
  const lastPredicted = useMemo(
    () => data.reduce((last, d, i) => (d.predicted != null ? i : last), -1), [data])

  const yTicks = useMemo(() => {
    let max = 0
    for (const d of data) {
      if (d.actual != null) max = Math.max(max, d.actual)
      if (d.predicted != null) max = Math.max(max, d.predicted)
      if (d.upper != null) max = Math.max(max, d.upper)
    }
    return niceTicks(max)
  }, [data])

  const axisProps = {
    stroke: COLOR.axis,
    tick: { fontSize: 11 },
    tickLine: false,
    axisLine: false as const,
    width: 52,
    tickFormatter: tickLabel,
    domain: [0, yTicks.at(-1) ?? 1] as [number, number],
    ticks: yTicks,
    label: {
      value: yLabel, angle: -90, position: 'insideLeft' as const,
      style: { fontSize: 11, fill: COLOR.axis, textAnchor: 'middle' as const },
    },
  }

  const plotHeight = typeof height === 'number' ? `${height}px` : height
  const margin = { top: 12, right: 20, left: 4, bottom: 4 }
  const xHeight = everyDate ? 62 : 30

  return (
    <figure className="m-0">
      <figcaption className="sr-only">
        Line chart. Actual demand is a solid line with round markers; predicted
        demand is a dashed line with diamond markers. They overlap on the
        held-out comparison window and only the prediction continues past the
        forecast start marker.
      </figcaption>

      <ChartLegend hasBand={hasBand} hasCompared={firstCompared >= 0} />

      <div className="relative">
        {/*
          The plot scrolls sideways, which would take the y-axis with it and
          leave every value unreadable. So the axis is drawn a second time, on
          an opaque strip pinned to the left edge, sharing the ticks, height and
          margins of the real one underneath.
        */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-y-0 left-0 z-10 w-[56px] overflow-hidden bg-surface"
        >
          {/*
            Rendered at a normal width and then clipped to the axis: given a
            zero-width plot area Recharts falls back to bare domain endpoints
            instead of the tick set the real chart is using, and the labels
            would stop matching the gridlines.
          */}
          <div style={{ width: 260, height: plotHeight }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={data} margin={margin}>
                <XAxis dataKey="date" height={xHeight} tick={false} axisLine={false} tickLine={false} />
                <YAxis {...axisProps} />
                <Line dataKey="actual" stroke="none" dot={false} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="overflow-x-auto pb-1">
          <div
            style={{
              height: plotHeight,
              minWidth: everyDate ? `max(100%, ${data.length * pxPerDate}px)` : undefined,
            }}
          >
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={data} margin={margin}>
                <defs>
                  <linearGradient id={gradientId} x1="0" y1="0" x2="1" y2="0">
                    <stop offset={0} stopColor={COLOR.predicted} />
                    <stop offset={1} stopColor={COLOR.horizon} />
                  </linearGradient>
                </defs>

                <CartesianGrid stroke={COLOR.grid} strokeDasharray="2 4" vertical={false} />

                {/* The two named regions, tinted — identity without relying on
                    colour vision alone. */}
                {firstCompared >= 0 && lastComparedDate && (
                  <ReferenceArea
                    x1={data[firstCompared].date}
                    x2={lastComparedDate}
                    fill="#1B2740"
                    fillOpacity={0.4}
                    stroke="none"
                  />
                )}
                {firstForecast > 0 && (
                  <ReferenceArea
                    x1={data[firstForecast].date}
                    x2={data.at(-1)!.date}
                    fill={COLOR.predicted}
                    fillOpacity={0.05}
                    stroke="none"
                  />
                )}

                <XAxis
                  dataKey="date"
                  tickFormatter={shortDate}
                  stroke={COLOR.axis}
                  tick={{ fontSize: everyDate ? 10 : 11 }}
                  interval={everyDate ? 0 : 'preserveStartEnd'}
                  minTickGap={everyDate ? 0 : 28}
                  angle={everyDate ? -60 : 0}
                  textAnchor={everyDate ? 'end' : 'middle'}
                  height={xHeight}
                  padding={everyDate ? { left: 22, right: 10 } : undefined}
                  tickLine={false}
                  axisLine={{ stroke: COLOR.grid }}
                />
                <YAxis {...axisProps} tick={false} />
                <Tooltip content={<DemandTooltip />} cursor={{ stroke: '#31415F' }} />

                {hasBand && (
                  <Area
                    dataKey="band"
                    stroke="none"
                    fill={COLOR.predicted}
                    fillOpacity={0.12}
                    activeDot={false}
                    isAnimationActive={false}
                    connectNulls
                  />
                )}

                <Line
                  dataKey="actual"
                  name="Actual Demand"
                  stroke={COLOR.actual}
                  strokeWidth={2.6}
                  dot={<ActualDot />}
                  activeDot={{ r: 5, strokeWidth: 0 }}
                  label={lastActual >= 0
                    ? endLabel(lastActual, 'Actual Demand', COLOR.actual, -14)
                    : undefined}
                  connectNulls={false}
                  isAnimationActive={false}
                />
                <Line
                  dataKey="predicted"
                  name="Predicted Demand"
                  stroke={`url(#${gradientId})`}
                  strokeWidth={2.6}
                  strokeDasharray="7 5"
                  dot={<PredictedDot />}
                  activeDot={{ r: 5, strokeWidth: 0 }}
                  label={lastPredicted >= 0
                    ? endLabel(lastPredicted, 'Predicted Demand', COLOR.predicted, 22)
                    : undefined}
                  connectNulls={false}
                  isAnimationActive={false}
                />

                {startDate && (
                  <ReferenceLine
                    x={startDate}
                    stroke={COLOR.origin}
                    strokeWidth={1.6}
                    strokeDasharray="4 3"
                    label={{
                      value: 'Forecast Start',
                      position: 'insideTopRight',
                      fill: COLOR.origin,
                      fontSize: 11,
                      fontWeight: 600,
                    }}
                  />
                )}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <p className="mt-2 text-center text-xs font-medium text-ink-dim">
        Date
        {everyDate && data.length > 40 && (
          <span className="ml-2 font-normal">· every day is labelled — scroll sideways for more</span>
        )}
      </p>
    </figure>
  )
}
