import { useState } from 'react'

import {
  startVerification, useInferenceJob, useInferenceStatus, useMembers,
  useModelCard, useProvenance,
} from '../api/hooks'
import { PipelineDiagram, ValidationDiagram } from '../components/diagrams'
import { Async, Badge, Card, Caveat, Explain, ErrorState, Metric, Spinner } from '../components/ui'
import { compact, nf } from '../lib/format'

/** The live model-verification control. */
function Verification() {
  const status = useInferenceStatus()
  const [jobId, setJobId] = useState<string | null>(null)
  const [startError, setStartError] = useState<unknown>(null)
  const job = useInferenceJob(jobId)

  const start = async () => {
    setStartError(null)
    try {
      const { job_id } = await startVerification()
      setJobId(job_id)
    } catch (e) {
      setStartError(e)
    }
  }

  const running = job.data?.status === 'queued' || job.data?.status === 'running'
  const result = job.data?.result

  return (
    <Card
      title="Verify the model, live"
      subtitle="Reload the frozen model, re-run it, and check it reproduces the published forecast"
      actions={
        status.data?.available ? (
          <button
            type="button"
            onClick={start}
            disabled={running}
            className="rounded border border-forecast/50 bg-forecast/10 px-3 py-1.5 text-xs
                       font-medium text-forecast transition-colors hover:bg-forecast/20
                       disabled:cursor-not-allowed disabled:opacity-50"
          >
            {running ? 'Running…' : 'Run verification'}
          </button>
        ) : null
      }
    >
      {status.isLoading && <Spinner label="Checking availability" />}

      {status.data && !status.data.available && (
        <div className="rounded border border-warn/30 bg-warn/5 px-3 py-2.5">
          <p className="text-xs font-medium text-warn">Live inference is not available here</p>
          <ul className="mt-1.5 space-y-0.5">
            {status.data.reasons.map((r) => (
              <li key={r} className="text-[11px] text-ink-muted">· {r}</li>
            ))}
          </ul>
          <p className="mt-2 text-[11px] text-ink-dim">
            Everything else on this site still works — the forecast and all accuracy
            figures are served from stored artefacts.
          </p>
        </div>
      )}

      {startError ? <ErrorState error={startError} onRetry={start} /> : null}

      {running && job.data && (
        <div className="mt-1">
          <div className="flex items-center justify-between text-xs">
            <span className="text-ink-muted">{job.data.message}</span>
            <span className="tnum text-ink-dim">{job.data.progress}%</span>
          </div>
          <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-elevated">
            <div
              className="h-full rounded-full bg-forecast transition-all duration-500"
              style={{ width: `${job.data.progress}%` }}
            />
          </div>
          <p className="mt-2 text-[11px] text-ink-dim">
            Takes about 45 seconds: it loads the sales panel, rebuilds every feature,
            runs both models and blends them.
          </p>
        </div>
      )}

      {job.data?.status === 'failed' && (
        <div role="alert" className="rounded border border-bad/40 bg-bad/5 px-3 py-2.5">
          <p className="text-xs font-medium text-bad">Verification failed</p>
          <p className="mt-1 text-[11px] text-ink-muted">{job.data.error?.message}</p>
        </div>
      )}

      {result && (
        <div className="animate-slideUp">
          <div
            className={`rounded border px-4 py-3 ${
              result.verdict === 'MATCH'
                ? 'border-good/40 bg-good/5'
                : 'border-bad/40 bg-bad/5'
            }`}
          >
            <div className="flex items-center gap-2">
              <Badge variant={result.verdict === 'MATCH' ? 'good' : 'bad'}>
                {result.verdict}
              </Badge>
              <span className="text-xs text-ink-muted">
                across {nf(result.n_predictions)} predictions
              </span>
            </div>
            <p className="mt-2 text-xs leading-relaxed text-ink-muted">
              {result.interpretation}
            </p>
          </div>

          <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-4">
            <Metric
              label="Largest difference"
              value={result.max_abs_diff === 0 ? '0' : result.max_abs_diff.toExponential(2)}
              size="sm"
              tone={result.max_abs_diff <= result.tolerance ? 'good' : 'bad'}
              hint="Between the live run and the published forecast."
            />
            <Metric label="Series" value={compact(result.n_series)} size="sm" />
            <Metric
              label="Total run time"
              value={`${result.timings_seconds.total?.toFixed(0)}s`}
              size="sm"
              hint="Data load, both models, and the comparison."
            />
            <Metric
              label="Blend weight"
              value={result.blend_weight_direct.toFixed(2)}
              size="sm"
              hint="Weight on the direct model, as frozen."
            />
          </dl>

          <Explain>
            This is not a stored result. The API just reloaded the model files from
            disk, rebuilt every feature from the raw sales history, ran both models,
            combined them, and compared the output against the published forecast
            value by value.
          </Explain>
        </div>
      )}

      {status.data?.refused_operations && (
        <div className="mt-5 border-t border-line pt-4">
          <p className="text-xs font-medium text-ink-muted">What this endpoint refuses to do</p>
          <ul className="mt-2 space-y-1.5">
            {Object.entries(status.data.refused_operations).map(([k, v]) => (
              <li key={k} className="text-[11px] leading-relaxed text-ink-dim">
                <span className="font-mono text-ink-muted">{k.replace(/_/g, ' ')}</span> — {v}
              </li>
            ))}
          </ul>
        </div>
      )}
    </Card>
  )
}

export function Model() {
  const card = useModelCard()
  const members = useMembers()
  const prov = useProvenance()

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-ink">Model</h1>
        <p className="mt-1 text-sm text-ink-muted">
          What the forecasting model is, and how you can check it yourself.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="How a forecast is produced">
          <PipelineDiagram />
          <Explain>
            Two models look at the same information in different ways. The{' '}
            <strong>direct</strong> model predicts all 28 days at once from what it knew
            on day zero. The <strong>recursive</strong> model predicts one day, feeds that
            prediction back in, and repeats 28 times. Their mistakes are not the same
            mistakes, so combining them is more reliable than either alone.
          </Explain>
        </Card>

        <Async query={card} height="h-64">
          {(m) => (
            <Card
              title="Specification"
              actions={<Badge variant="info">{m.status}</Badge>}
            >
              <dl className="space-y-2.5 text-xs">
                {[
                  ['Algorithm', 'LightGBM (gradient-boosted decision trees)'],
                  ['Objective', m.objective],
                  ['Combination', m.blend_formula],
                  ['Trees per model', nf(m.n_estimators)],
                  ['Random seed', String(m.seed)],
                  ['Horizon', `${m.horizon_days} days`],
                  ['Series covered', nf(m.n_series)],
                  ['Validation window', m.validation_window],
                  ['Forecast period', m.forecast_dates],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-4 border-b border-line pb-2 last:border-0">
                    <dt className="text-ink-dim">{k}</dt>
                    <dd className="text-right font-medium text-ink">{v}</dd>
                  </div>
                ))}
              </dl>
              <Explain>
                <strong>Gradient-boosted trees</strong> build many small decision trees,
                each correcting the previous one's mistakes. The <strong>Tweedie</strong>{' '}
                objective is chosen because retail demand is mostly zeros with occasional
                positive counts — a shape ordinary regression handles badly.
              </Explain>
            </Card>
          )}
        </Async>
      </div>

      <Verification />

      <div className="grid gap-6 lg:grid-cols-2">
        <Async query={members} height="h-48">
          {(m) => (
            <Card title="Why the two models are combined" subtitle={m.window}>
              <dl className="space-y-3">
                {m.members.map((x) => (
                  <div key={x.name} className="flex items-center justify-between gap-4">
                    <div className="min-w-0">
                      <p className="text-xs font-medium text-ink">{x.name}</p>
                      <p className="text-[11px] text-ink-dim">
                        weight {x.weight.toFixed(2)}
                      </p>
                    </div>
                    <span className="tnum text-sm font-semibold text-ink-muted">
                      {x.rmse.toFixed(4)}
                    </span>
                  </div>
                ))}
                <div className="flex items-center justify-between gap-4 border-t border-line pt-3">
                  <p className="text-xs font-semibold text-good">{m.blend.name}</p>
                  <span className="tnum text-sm font-semibold text-good">
                    {m.blend.rmse.toFixed(4)}
                  </span>
                </div>
              </dl>
              <Explain>{m.why_it_works}</Explain>
            </Card>
          )}
        </Async>

        <Card title="How leakage was prevented">
          <ValidationDiagram />
          <Explain>
            Leakage is when a model accidentally sees the future it is supposed to
            predict — the most common way a forecasting result turns out to be
            fiction. Here every feature is frozen at the forecast origin, and the
            check is empirical: post-origin sales are overwritten with a nonsense
            value and every feature must come back bit-for-bit identical.
          </Explain>
        </Card>
      </div>

      <Async query={prov} height="h-40">
        {(p) => (
          <Card title="Artefact fingerprints" subtitle="What exactly is deployed">
            <dl className="space-y-2 text-xs">
              {[
                ['Direct model', p.model_direct_sha256],
                ['Recursive model', p.model_recursive_sha256],
                ['Published forecast', p.forecast_sha256],
              ].map(([k, v]) => (
                <div key={k} className="flex flex-wrap items-baseline justify-between gap-2">
                  <dt className="text-ink-muted">{k}</dt>
                  <dd className="break-all font-mono text-[10px] text-ink-dim">{v}</dd>
                </div>
              ))}
            </dl>
            <Caveat>
              These SHA-256 fingerprints are checked by the backend test suite against the
              frozen model manifest. If anyone swapped or retrained a model, the tests fail
              — the deployment cannot quietly change what it serves.
            </Caveat>
          </Card>
        )}
      </Async>
    </div>
  )
}
