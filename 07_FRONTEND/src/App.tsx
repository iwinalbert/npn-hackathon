import { Suspense, lazy } from 'react'
import { Route, Routes } from 'react-router-dom'

import { AppShell } from './components/layout/AppShell'
import { LoadingPanel } from './components/ui'
import { Overview } from './pages/Overview'

/**
 * Overview loads eagerly because it is the landing page and must paint fast.
 * Every other route is code-split: the chart library is the largest dependency
 * and no evaluator needs it before they navigate.
 */
const Forecast = lazy(() => import('./pages/Forecast').then((m) => ({ default: m.Forecast })))
const Hierarchy = lazy(() => import('./pages/Hierarchy').then((m) => ({ default: m.Hierarchy })))
const Accuracy = lazy(() => import('./pages/Accuracy').then((m) => ({ default: m.Accuracy })))
const Validation = lazy(() => import('./pages/Validation').then((m) => ({ default: m.Validation })))
const Insights = lazy(() => import('./pages/Insights').then((m) => ({ default: m.Insights })))
const Model = lazy(() => import('./pages/Model').then((m) => ({ default: m.Model })))
const Methodology = lazy(() => import('./pages/Methodology').then((m) => ({ default: m.Methodology })))

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
          path="accuracy"
          element={
            <Suspense fallback={<LoadingPanel height="h-96" label="Loading accuracy" />}>
              <Accuracy />
            </Suspense>
          }
        />
        <Route
          path="validation"
          element={
            <Suspense fallback={<LoadingPanel height="h-96" label="Loading validation" />}>
              <Validation />
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
          path="model"
          element={
            <Suspense fallback={<LoadingPanel height="h-96" label="Loading model" />}>
              <Model />
            </Suspense>
          }
        />
        <Route
          path="methodology"
          element={
            <Suspense fallback={<LoadingPanel height="h-96" label="Loading methodology" />}>
              <Methodology />
            </Suspense>
          }
        />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  )
}
