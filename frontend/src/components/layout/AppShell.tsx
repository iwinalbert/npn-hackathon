import { useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'

import { useModelCard, useReadiness } from '../../api/hooks'
import { Badge } from '../ui'
import { FloatingAIAssistant } from './FloatingAIAssistant'

const NAV = [
  {
    group: 'Forecasting',
    items: [
      { to: '/', label: 'Overview', end: true, desc: 'What this system does' },
      { to: '/forecast', label: 'Forecast', desc: 'History and 28-day forecast per store-item' },
      { to: '/hierarchy', label: 'Hierarchy', desc: 'Roll up from item to chain' },
      { to: '/insights', label: 'Insights', desc: 'What needs attention' },
      { to: '/assistant', label: 'AI Assistant', desc: 'Ask about forecasts in plain language' },
    ],
  },
]

function StatusPill() {
  const { data, isLoading, isError } = useReadiness()
  if (isLoading) return <Badge>checking…</Badge>
  if (isError || !data?.ready) return <Badge variant="bad">API offline</Badge>
  if (data.degraded) return <Badge variant="warn">degraded</Badge>
  return <Badge variant="good">API ready</Badge>
}

export function AppShell() {
  const [open, setOpen] = useState(false)
  const { data: model } = useModelCard()

  return (
    <div className="flex min-h-full flex-col">
      {/* ---------------------------------------------------------------- */}
      <header className="sticky top-0 z-30 border-b border-line bg-base/95 backdrop-blur">
        <div className="flex items-center gap-4 px-4 py-3 lg:px-6">
          <button
            type="button"
            className="rounded border border-line px-2 py-1 text-xs text-ink-muted lg:hidden"
            aria-expanded={open}
            aria-controls="main-nav"
            onClick={() => setOpen((v) => !v)}
          >
            Menu
          </button>

          <div className="min-w-0 flex-1">
            <h1 className="truncate text-sm font-semibold tracking-tight text-ink">
              Retail Demand Forecasting
            </h1>
            <p className="truncate text-xs text-ink-muted">
              Walmart M5 · Hierarchical 28-Day Forecasting
            </p>
          </div>

          <div className="hidden items-center gap-3 sm:flex">
            {model && (
              <span className="tnum text-xs text-ink-muted">
                RMSE <span className="font-semibold text-ink">{model.validation_rmse.toFixed(4)}</span>
                <span className="mx-1.5 text-ink-dim">·</span>
                MAE <span className="font-semibold text-ink">{model.validation_mae.toFixed(4)}</span>
              </span>
            )}
            <StatusPill />
          </div>
        </div>
      </header>

      <div className="flex flex-1">
        {/* -------------------------------------------------------------- */}
        <nav
          id="main-nav"
          aria-label="Main"
          className={`${open ? 'block' : 'hidden'} w-full shrink-0 border-b border-line
                      bg-surface lg:block lg:w-60 lg:border-b-0 lg:border-r`}
        >
          <div className="space-y-6 px-3 py-5 lg:sticky lg:top-[57px]">
            {NAV.map((group) => (
              <div key={group.group}>
                <p className="px-2 pb-2 text-[10px] font-semibold uppercase tracking-wider text-ink-dim">
                  {group.group}
                </p>
                <ul className="space-y-0.5">
                  {group.items.map((item) => (
                    <li key={item.to}>
                      <NavLink
                        to={item.to}
                        end={item.end}
                        onClick={() => setOpen(false)}
                        className={({ isActive }) =>
                          `block rounded px-2.5 py-2 transition-colors ${
                            isActive
                              ? 'bg-accentSoft text-ink'
                              : 'text-ink-muted hover:bg-elevated hover:text-ink'
                          }`
                        }
                      >
                        {({ isActive }) => (
                          <>
                            <span
                              className={`block text-sm ${isActive ? 'font-semibold' : 'font-medium'}`}
                            >
                              {item.label}
                            </span>
                            <span className="mt-0.5 block text-[11px] leading-snug text-ink-dim">
                              {item.desc}
                            </span>
                          </>
                        )}
                      </NavLink>
                    </li>
                  ))}
                </ul>
              </div>
            ))}

            <div className="border-t border-line px-2 pt-4">
              <p className="text-[11px] leading-relaxed text-ink-dim">
                Model is <span className="font-semibold text-ink-muted">frozen</span>. All figures
                come from held-out validation — never from the delivered forecast window, which has
                no ground truth.
              </p>
            </div>
          </div>
        </nav>

        {/* -------------------------------------------------------------- */}
        {/*
          The extra bottom padding is what keeps the floating button from being
          a nuisance: the last row of any page can always be scrolled clear of
          it, so it never permanently covers a control or a chart axis.
        */}
        <main className="min-w-0 flex-1 px-4 pb-24 pt-6 lg:px-8">
          <Outlet />
        </main>
      </div>

      {/* Rendered once here, so every routed page gets it without opting in. */}
      <FloatingAIAssistant />
    </div>
  )
}
