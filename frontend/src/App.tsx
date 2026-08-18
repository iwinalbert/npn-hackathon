import { Suspense, lazy } from 'react'
import { Route, Routes } from 'react-router-dom'

import { AppShell } from './components/layout/AppShell'
import { LoadingPanel } from './components/ui'
import { Overview } from './pages/Overview'

const Forecast = lazy(() => import('./pages/Forecast').then((m) => ({ default: m.Forecast })))
const Hierarchy = lazy(() => import('./pages/Hierarchy').then((m) => ({ default: m.Hierarchy })))
const Insights = lazy(() => import('./pages/Insights').then((m) => ({ default: m.Insights })))
const Assistant = lazy(() => import('./pages/Assistant').then((m) => ({ default: m.Assistant })))

function NotFound() {
  return (
    <div className="mx-auto max-w-lg py-16 text-center">
      <p className="text-sm font-semibold text-ink">Page not found</p>
      <p className="mt-1.5 text-xs text-ink-muted">
        That route does not exist in this application.
      </p>
    </div>
  )
}

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Overview />} />
        <Route
          path="forecast"
          element={
            <Suspense fallback={<LoadingPanel height="h-96" label="Loading forecast explorer" />}>
              <Forecast />
            </Suspense>
          }
        />
        <Route
          path="hierarchy"
          element={
            <Suspense fallback={<LoadingPanel height="h-96" label="Loading hierarchy" />}>
              <Hierarchy />
            </Suspense>
          }
        />
        <Route
          path="insights"
          element={
            <Suspense fallback={<LoadingPanel height="h-96" label="Loading insights" />}>
              <Insights />
            </Suspense>
          }
        />
        <Route
          path="assistant"
          element={
            <Suspense fallback={<LoadingPanel height="h-96" label="Loading assistant" />}>
              <Assistant />
            </Suspense>
          }
        />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  )
}
