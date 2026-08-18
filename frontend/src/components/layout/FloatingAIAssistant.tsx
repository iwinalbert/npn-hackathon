import { Link, useLocation } from 'react-router-dom'

export function FloatingAIAssistant() {
  const { pathname } = useLocation()
  const onAssistant = pathname === '/assistant'

  return (
    <div
      className="group pointer-events-none fixed bottom-4 right-4 z-40 flex flex-col
                 items-end gap-1.5 sm:bottom-6 sm:right-6 sm:flex-row sm:items-center
                 sm:gap-2.5"
    >
      <span
        aria-hidden
        className="pointer-events-none translate-x-1 select-none whitespace-nowrap rounded
                   border border-line bg-surface px-2.5 py-1.5 text-xs font-medium
                   text-ink opacity-0 shadow-lg shadow-black/30 transition-all
                   duration-150 ease-out group-hover:translate-x-0 group-hover:opacity-100
                   group-focus-within:translate-x-0 group-focus-within:opacity-100"
      >
        AI Assistant
      </span>

      <Link
        to="/assistant"
        aria-label="Open AI Assistant"
        aria-current={onAssistant ? 'page' : undefined}
        title="AI Assistant"
        className={`pointer-events-auto flex h-11 w-11 items-center justify-center
                    rounded-full border text-xs font-semibold tracking-wide
                    shadow-lg shadow-black/30 transition-all duration-150 ease-out
                    hover:-translate-y-0.5 sm:h-12 sm:w-12 sm:text-sm ${
                      onAssistant
                        ? 'border-forecast/60 bg-accentSoft text-forecast'
                        : 'border-line-strong bg-surface text-ink-muted '
                          + 'hover:border-forecast/50 hover:bg-elevated hover:text-ink'
                    }`}
      >
        AI
      </Link>
    </div>
  )
}
