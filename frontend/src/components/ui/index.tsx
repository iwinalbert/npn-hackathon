import type { ReactNode } from 'react'

import { ApiError } from '../../api/client'


interface CardProps {
  title?: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
  bodyClassName?: string
}

export function Card({
  title, subtitle, actions, children, className = '', bodyClassName = 'p-5',
}: CardProps) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <header className="flex items-start justify-between gap-4 border-b border-line px-5 pb-3 pt-4">
          <div className="min-w-0">
            {title && (
              <h2 className="text-sm font-semibold tracking-tight text-ink">{title}</h2>
            )}
            {subtitle && (
              <p className="mt-1 text-xs leading-relaxed text-ink-muted">{subtitle}</p>
            )}
          </div>
          {actions && <div className="shrink-0">{actions}</div>}
        </header>
      )}
      <div className={bodyClassName}>{children}</div>
    </section>
  )
}


export function VizStage({
  title, description, meta, actions, footer, children,
}: {
  title: string
  description: ReactNode
  meta?: ReactNode
  actions?: ReactNode
  footer?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="card animate-slideUp">
      <header className="border-b border-line px-5 pb-4 pt-5 sm:px-6">
        <div className="flex flex-wrap items-start justify-between gap-x-6 gap-y-3">
          <div className="min-w-0 max-w-3xl">
            <h1 className="text-lg font-semibold tracking-tight text-ink sm:text-xl">
              {title}
            </h1>
            <p className="mt-2 text-sm leading-relaxed text-ink-muted">{description}</p>
          </div>
          {actions && <div className="shrink-0">{actions}</div>}
        </div>
        {meta && <p className="mt-3 text-xs text-ink-dim">{meta}</p>}
      </header>

      <div className="px-3 py-4 sm:px-5 sm:py-5">{children}</div>

      {footer && (
        <div className="border-t border-line px-5 py-3 sm:px-6">{footer}</div>
      )}
    </section>
  )
}


const TONE: Record<string, string> = {
  default: 'text-ink',
  good: 'text-good',
  warn: 'text-warn',
  bad: 'text-bad',
  forecast: 'text-forecast',
}

export function Metric({
  label, value, unit, hint, tone = 'default', size = 'md',
}: {
  label: string
  value: ReactNode
  unit?: string
  hint?: string
  tone?: keyof typeof TONE
  size?: 'sm' | 'md'
}) {
  return (
    <div className="min-w-0">
      <dt className="text-xs font-medium uppercase tracking-wide text-ink-dim">{label}</dt>
      <dd
        className={`mt-1.5 tnum font-semibold ${TONE[tone]} ${
          size === 'md' ? 'text-metric' : 'text-xl'
        }`}
      >
        {value}
        {unit && <span className="ml-1 text-sm font-normal text-ink-muted">{unit}</span>}
      </dd>
      {hint && <p className="mt-1.5 text-xs leading-relaxed text-ink-muted">{hint}</p>}
    </div>
  )
}


const VARIANTS = {
  neutral: 'border-line-strong text-ink-muted',
  good: 'border-good/40 bg-good/5 text-good',
  warn: 'border-warn/40 bg-warn/5 text-warn',
  bad: 'border-bad/40 bg-bad/5 text-bad',
  info: 'border-forecast/40 bg-forecast/5 text-forecast',
} as const

export function Badge({
  children, variant = 'neutral',
}: { children: ReactNode; variant?: keyof typeof VARIANTS }) {
  return (
    <span
      className={`inline-flex items-center rounded border px-2 py-0.5 text-[11px]
                  font-medium tracking-wide ${VARIANTS[variant]}`}
    >
      {children}
    </span>
  )
}


export function Spinner({ label = 'Loading' }: { label?: string }) {
  return (
    <span role="status" className="inline-flex items-center gap-2 text-sm text-ink-muted">
      <span
        aria-hidden
        className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-line-strong border-t-forecast"
      />
      {label}
    </span>
  )
}

export function SkeletonBlock({ height = 'h-48' }: { height?: string }) {
  return <div className={`skeleton w-full ${height}`} aria-hidden />
}

export function LoadingPanel({
  height = 'h-48', label = 'Loading',
}: { height?: string; label?: string }) {
  return (
    <div className="space-y-3">
      <Spinner label={label} />
      <SkeletonBlock height={height} />
    </div>
  )
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const api = error instanceof ApiError ? error : null
  const message = api ? api.userMessage : (error as Error)?.message ?? 'Unexpected error'
  const unreachable = api?.status === 503 || api?.status === 0
  return (
    <div role="alert" className="rounded border border-bad/40 bg-bad/5 px-4 py-3">
      <p className="text-sm font-medium text-bad">{message}</p>
      {unreachable && (
        <p className="mt-1.5 text-xs text-ink-muted">
          The API may not be running, or its data layer has not been built. Start it with{' '}
          <code className="font-mono text-ink">python tasks.py api</code>.
        </p>
      )}
      {api?.context?.hint ? (
        <p className="mt-1.5 text-xs text-ink-muted">{String(api.context.hint)}</p>
      ) : null}
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-3 rounded border border-line-strong px-3 py-1.5 text-xs
                     font-medium text-ink transition-colors hover:bg-elevated"
        >
          Try again
        </button>
      )}
      {api?.requestId && (
        <p className="mt-2 font-mono text-[10px] text-ink-dim">request {api.requestId}</p>
      )}
    </div>
  )
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded border border-dashed border-line px-4 py-8 text-center">
      <p className="text-sm font-medium text-ink-muted">{title}</p>
      {children && <div className="mt-1.5 text-xs text-ink-dim">{children}</div>}
    </div>
  )
}

export function Explain({ children }: { children: ReactNode }) {
  return (
    <p className="mt-3 border-l-2 border-line-strong pl-3 text-xs leading-relaxed text-ink-muted">
      {children}
    </p>
  )
}

export function Caveat({ children }: { children: ReactNode }) {
  return (
    <p className="mt-3 rounded border border-warn/30 bg-warn/5 px-3 py-2 text-xs leading-relaxed text-ink-muted">
      <span className="font-semibold text-warn">Note </span>
      {children}
    </p>
  )
}


export function Async<T>({
  query, children, height = 'h-48', empty,
}: {
  query: { data?: T; isLoading: boolean; isError: boolean; error: unknown; refetch: () => void }
  children: (data: T) => ReactNode
  height?: string
  empty?: (data: T) => boolean
}) {
  if (query.isLoading) return <LoadingPanel height={height} />
  if (query.isError) return <ErrorState error={query.error} onRetry={() => query.refetch()} />
  if (query.data === undefined) return <EmptyState title="No data available" />
  if (empty?.(query.data)) return <EmptyState title="Nothing to show for this selection" />
  return <div className="animate-fadeIn">{children(query.data)}</div>
}
