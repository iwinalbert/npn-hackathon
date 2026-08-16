import { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import { askAssistant, useGenAIStatus, useGenAISuggestions, useSeriesDetail } from '../api/hooks'
import type { AskResponse } from '../api/types'
import { ApiError } from '../api/client'
import { Badge, Card, Caveat, Explain, Spinner } from '../components/ui'

interface Turn {
  id: number
  question: string
  answer?: AskResponse
  error?: unknown
  pending: boolean
}

/**
 * Shown only when the status endpoint could not be read. The backend is the
 * source of truth for these — this exists so the explanatory panel still says
 * something true when the API is unreachable, not to duplicate policy.
 */
const FALLBACK_REFUSALS: Record<string, string> = {
  modifying_forecasts: 'the model is frozen; no write path exists',
  price_what_if: 'the model uses price as context, not as a causal lever',
  prediction_intervals: 'the model emits point forecasts only',
  live_accuracy_claims: 'no ground truth exists for the delivered forecast window',
}

/**
 * Used only if /genai/suggestions fails while the assistant itself is up.
 * A prompt list is the difference between an obvious UI and an empty box, so it
 * should not disappear because one auxiliary request failed.
 */
const FALLBACK_SUGGESTIONS_SERIES = [
  'Explain this forecast in plain language',
  'Is demand increasing, decreasing or stable?',
  'How much should I plan to stock over the next 28 days?',
  'How accurate has the model been on this product?',
]

const FALLBACK_SUGGESTIONS_CHAIN = [
  'Which items need attention right now?',
  'How accurate is this model, and what does RMSE 2.09 mean?',
  'Explain the difference between direct and recursive forecasting',
  'How does the model handle products that rarely sell?',
]

/** `**bold**` → <strong>. Everything else stays literal text. */
function inline(text: string) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith('**') && part.endsWith('**') && part.length > 4
      ? <strong key={i} className="font-medium text-ink">{part.slice(2, -2)}</strong>
      : <span key={i}>{part}</span>,
  )
}

/**
 * Renders an answer as paragraphs and bullet lists.
 *
 * The model writes light markdown — `* **Planning Range:** …` — despite being
 * told not to use headings. Rendering that as raw text put literal asterisks on
 * screen. This is a deliberately minimal renderer, not a markdown engine: bold
 * and bullets are what actually appears, and anything else is left as text
 * rather than interpreted.
 */
function AnswerText({ text }: { text: string }) {
  const blocks: Array<{ type: 'p' | 'ul'; lines: string[] }> = []
  for (const raw of text.split('\n')) {
    const line = raw.trim()
    if (!line) continue
    const bullet = /^[*-]\s+/.test(line)
    const last = blocks[blocks.length - 1]
    if (bullet) {
      const item = line.replace(/^[*-]\s+/, '')
      if (last?.type === 'ul') last.lines.push(item)
      else blocks.push({ type: 'ul', lines: [item] })
    } else if (last?.type === 'p') {
      last.lines.push(line)
    } else {
      blocks.push({ type: 'p', lines: [line] })
    }
  }

  return (
    <>
      {blocks.map((b, i) =>
        b.type === 'ul' ? (
          <ul key={i} className={`space-y-1.5 ${i ? 'mt-3' : ''}`}>
            {b.lines.map((l, j) => (
              <li key={j} className="flex gap-2 text-sm leading-relaxed text-ink">
                <span aria-hidden className="mt-2 h-1 w-1 shrink-0 rounded-full bg-ink-dim" />
                <span>{inline(l)}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p key={i} className={`text-sm leading-relaxed text-ink ${i ? 'mt-3' : ''}`}>
            {inline(b.lines.join(' '))}
          </p>
        ),
      )}
    </>
  )
}

/**
 * AI Forecast Assistant.
 *
 * Deliberately framed as an analytical tool, not a chatbot: every answer shows
 * which data it was built from, how long it took, and whether every figure in it
 * traced back to that data. A failed grounding check is shown to the user rather
 * than hidden — an assistant that admits an unverifiable number is worth far
 * more than one that sounds confident.
 */
export function Assistant() {
  const [params, setParams] = useSearchParams()
  const store = params.get('store')
  const item = params.get('item')

  const status = useGenAIStatus()
  const suggestions = useGenAISuggestions(store, item)
  const series = useSeriesDetail(store ?? undefined, item ?? undefined)

  const [question, setQuestion] = useState('')
  const [turns, setTurns] = useState<Turn[]>([])
  const nextId = useRef(1)
  const endRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [turns])

  const available = status.data?.available === true
  const busy = turns.some((t) => t.pending)
  const prompts = suggestions.data?.suggestions?.length
    ? suggestions.data.suggestions
    : (store && item ? FALLBACK_SUGGESTIONS_SERIES : FALLBACK_SUGGESTIONS_CHAIN)

  async function send(text: string) {
    const q = text.trim()
    if (!q || busy || !available) return
    const id = nextId.current++
    setTurns((t) => [...t, { id, question: q, pending: true }])
    setQuestion('')
    try {
      const answer = await askAssistant({
        question: q,
        store_id: store ?? undefined,
        item_id: item ?? undefined,
      })
      setTurns((t) => t.map((x) => (x.id === id ? { ...x, answer, pending: false } : x)))
    } catch (err) {
      setTurns((t) => t.map((x) => (x.id === id ? { ...x, error: err, pending: false } : x)))
    }
  }

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-ink">
          AI Forecast Assistant
        </h1>
        <p className="mt-1 text-sm text-ink-muted">
          Ask about the forecasts in plain language. Every answer is built from
          verified backend data — the assistant cannot change anything.
        </p>
      </div>

      {/* --- scope ------------------------------------------------------ */}
      {store && item && (
        <div className="flex flex-wrap items-center gap-3 rounded border border-forecast/30
                        bg-forecast/5 px-4 py-3">
          <Badge variant="info">focused</Badge>
          <span className="text-sm text-ink">
            <span className="font-semibold">{item}</span>
            <span className="mx-1.5 text-ink-dim">in</span>
            <span className="font-semibold">{store}</span>
            {series.data && (
              <span className="ml-2 text-xs text-ink-muted">
                {series.data.dept_id} · {series.data.regime}
              </span>
            )}
          </span>
          <button
            type="button"
            onClick={() => setParams({})}
            className="ml-auto rounded border border-line-strong px-2.5 py-1 text-xs
                       text-ink-muted transition-colors hover:bg-elevated hover:text-ink"
          >
            Clear focus
          </button>
        </div>
      )}

      {/* --- unavailable ------------------------------------------------ */}
      {status.isLoading && <Spinner label="Checking assistant availability" />}

      {/*
        The status probe FAILING is different from the assistant being
        unavailable, and it needs its own branch. Without one, `status.data` is
        undefined, `isLoading` is false, `available` is false — every section
        below is gated on one of those, so the page renders the heading and
        nothing else. That is not a hypothetical: an API process started before
        this feature existed answers /health with 200 and /genai/status with
        404, which produced exactly that blank page.
      */}
      {status.isError && (
        <Card title="Cannot reach the assistant service">
          <p className="text-sm text-ink">
            The frontend could not read{' '}
            <code className="font-mono text-xs text-ink-muted">/api/v1/genai/status</code>
            {' '}from the API.
          </p>
          <p className="mt-2 text-xs text-ink-muted">
            {status.error instanceof ApiError
              ? `${status.error.userMessage}${status.error.status ? ` (HTTP ${status.error.status})` : ''}`
              : 'The request did not complete.'}
          </p>
          <ul className="mt-3 space-y-1.5 text-xs text-ink-muted">
            <li>
              · <span className="text-ink">The API is not running.</span> Start it
              with <code className="font-mono text-ink">python tasks.py api</code>.
            </li>
            <li>
              · <span className="text-ink">The API is running an older build.</span>{' '}
              A 404 here means the process predates the assistant endpoints —
              restart it so the <code className="font-mono text-ink">/genai</code>{' '}
              routes are registered.
            </li>
          </ul>
          <div className="mt-4">
            <button
              type="button"
              onClick={() => { void status.refetch(); void suggestions.refetch() }}
              className="rounded border border-forecast/50 bg-forecast/10 px-3 py-1.5
                         text-xs font-medium text-forecast transition-colors
                         hover:bg-forecast/20"
            >
              Try again
            </button>
          </div>
          <Explain>
            Only the assistant is affected. Every forecast, accuracy figure and
            chart elsewhere in this application is served by other endpoints and
            continues to work.
          </Explain>
        </Card>
      )}

      {status.data && !available && (
        <Card title="The assistant is not configured in this deployment">
          <ul className="space-y-1">
            {status.data.reasons.map((r) => (
              <li key={r} className="text-xs text-ink-muted">· {r}</li>
            ))}
          </ul>
          <Explain>
            The assistant is optional. Every forecast, accuracy figure and chart in
            this application works without it — it only adds a natural-language
            explanation layer on top. To enable it, set{' '}
            <code className="font-mono text-ink">GEMINI_API_KEY</code> in the API
            environment and restart the service. The key is used server-side only
            and is never sent to the browser.
          </Explain>
        </Card>
      )}

      {/* --- conversation ------------------------------------------------ */}
      {available && (
        <Card
          title="Ask about your forecasts"
          subtitle={`Answers are grounded in ${store && item ? `${item} · ${store}` : 'chain-wide'} data`}
          actions={status.data?.model
            ? <Badge>{status.data.model}</Badge>
            : null}
        >
          {turns.length === 0 && (
            <div className="rounded border border-dashed border-line px-4 py-6">
              <p className="text-xs font-medium text-ink-muted">
                Try one of these
              </p>
              <div className="mt-3 flex flex-wrap gap-2">
                {prompts.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => send(s)}
                    className="rounded border border-line px-3 py-1.5 text-xs text-ink-muted
                               transition-colors hover:border-forecast/50
                               hover:bg-forecast/10 hover:text-ink"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {turns.length > 0 && (
            <ol className="space-y-5">
              {turns.map((t) => (
                <li key={t.id} className="animate-slideUp">
                  <div className="flex justify-end">
                    <p className="max-w-[85%] rounded border border-line-strong bg-elevated
                                  px-3 py-2 text-sm text-ink">
                      {t.question}
                    </p>
                  </div>

                  <div className="mt-3">
                    {t.pending && <Spinner label="Analysing verified data" />}

                    {t.error != null && (
                      <div role="alert" className="rounded border border-bad/40 bg-bad/5 px-3 py-2.5">
                        <p className="text-xs font-medium text-bad">
                          {t.error instanceof ApiError ? t.error.userMessage : 'Request failed'}
                        </p>
                        {t.error instanceof ApiError && t.error.context?.remedy ? (
                          <p className="mt-1 text-[11px] text-ink-muted">
                            {String(t.error.context.remedy)}
                          </p>
                        ) : null}
                      </div>
                    )}

                    {t.answer && (
                      <div>
                        <div className="rounded border border-line bg-base px-4 py-3">
                          <AnswerText text={t.answer.answer} />
                        </div>

                        {/* provenance strip */}
                        <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1.5
                                        text-[11px] text-ink-dim">
                          {t.answer.refused ? (
                            <>
                              <span className="rounded border border-good/40 bg-good/10
                                               px-1.5 py-0.5 font-medium text-good">
                                answered locally — no AI call
                              </span>
                              <span aria-hidden>·</span>
                              <span>
                                policy:{' '}
                                <span className="font-mono text-ink-muted">
                                  {t.answer.refusal_category}
                                </span>
                              </span>
                            </>
                          ) : (
                            <>
                              <span>
                                data:{' '}
                                <span className="font-mono text-ink-muted">{t.answer.intent}</span>
                              </span>
                              <span aria-hidden>·</span>
                              <span>{t.answer.elapsed_ms} ms</span>
                              <span aria-hidden>·</span>
                              {t.answer.grounded ? (
                                <span className="text-good">
                                  all figures traced to backend data
                                </span>
                              ) : (
                                <span className="text-warn">
                                  {t.answer.ungrounded_numbers.length} figure
                                  {t.answer.ungrounded_numbers.length === 1 ? '' : 's'} could not be traced
                                </span>
                              )}
                            </>
                          )}
                          {t.answer.injection_suspected && (
                            <>
                              <span aria-hidden>·</span>
                              <span className="text-warn">input flagged</span>
                            </>
                          )}
                        </div>

                        {t.answer.truncated && (
                          <Caveat>
                            This answer reached the model's output limit and was cut
                            short. Ask a narrower question, or raise{' '}
                            <code className="font-mono">NPN_GENAI_MAX_OUTPUT_TOKENS</code>{' '}
                            in the API environment.
                          </Caveat>
                        )}

                        {!t.answer.grounded && !t.answer.refused && (
                          <Caveat>
                            This answer contains {t.answer.ungrounded_numbers.length} number
                            {t.answer.ungrounded_numbers.length === 1 ? '' : 's'} (
                            {t.answer.ungrounded_numbers.slice(0, 4).join(', ')}) that the
                            backend could not match to the data it supplied. Treat those
                            figures as unverified and check them against the Forecast or
                            Accuracy pages.
                          </Caveat>
                        )}
                      </div>
                    )}
                  </div>
                </li>
              ))}
            </ol>
          )}

          <div ref={endRef} />

          {/* --- input ---------------------------------------------------- */}
          <form
            className="mt-5 border-t border-line pt-4"
            onSubmit={(e) => {
              e.preventDefault()
              send(question)
            }}
          >
            <label htmlFor="assistant-input" className="sr-only">
              Ask about your forecasts
            </label>
            <div className="flex gap-2">
              <textarea
                id="assistant-input"
                ref={inputRef}
                rows={2}
                value={question}
                maxLength={status.data?.max_question_chars ?? 800}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    send(question)
                  }
                }}
                placeholder="Ask about your forecasts…  (Enter to send, Shift+Enter for a new line)"
                className="min-h-[3rem] flex-1 resize-y rounded border border-line bg-base
                           px-3 py-2 text-sm text-ink placeholder:text-ink-dim
                           transition-colors focus:border-forecast"
              />
              <button
                type="submit"
                disabled={!question.trim() || busy}
                className="self-end rounded border border-forecast/50 bg-forecast/10 px-4 py-2
                           text-xs font-medium text-forecast transition-colors
                           hover:bg-forecast/20 disabled:cursor-not-allowed
                           disabled:opacity-40"
              >
                {busy ? 'Thinking…' : 'Ask'}
              </button>
            </div>

            {turns.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-2">
                {prompts.slice(0, 3).map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => send(s)}
                    disabled={busy}
                    className="rounded border border-line px-2.5 py-1 text-[11px] text-ink-muted
                               transition-colors hover:border-forecast/50 hover:text-ink
                               disabled:opacity-40"
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}
          </form>
        </Card>
      )}

      {/*
        --- what it will and will not do ---------------------------------
        Rendered whenever the probe has settled, success or failure. These two
        panels are static explanations of the design; gating them on a
        successful API call was what let the whole page go blank.
      */}
      {!status.isLoading && (
        <div className="grid gap-6 lg:grid-cols-2">
          <Card title="How this assistant works">
            <ol className="space-y-2.5">
              {[
                'Your question is sent to this application’s own backend, never directly to an AI provider.',
                'The backend decides which verified data the question needs and retrieves it.',
                'Only that small structured summary is sent to the language model — never the dataset.',
                'The reply is checked: every figure in it must trace back to the data supplied.',
              ].map((s, i) => (
                <li key={s} className="flex gap-3">
                  <span className="tnum mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center
                                   rounded-full border border-line-strong text-[10px] text-ink-dim">
                    {i + 1}
                  </span>
                  <span className="text-xs leading-relaxed text-ink-muted">{s}</span>
                </li>
              ))}
            </ol>
            <Explain>
              The assistant translates and explains. It never computes a forecast,
              and every number it can quote was calculated by the forecasting
              system itself.
            </Explain>
          </Card>

          <Card title="What it will not do">
            <ul className="space-y-2.5">
              {Object.entries(status.data?.refusals ?? FALLBACK_REFUSALS).map(([k, v]) => (
                <li key={k} className="border-l-2 border-l-line-strong pl-3">
                  <p className="text-xs font-medium text-ink">
                    {k.replace(/_/g, ' ')}
                  </p>
                  <p className="mt-0.5 text-[11px] leading-relaxed text-ink-muted">{v}</p>
                </li>
              ))}
            </ul>
            <Caveat>
              The forecasting model is frozen. The assistant has no write path to any
              forecast, model file or dataset — it is an explanatory layer only.
            </Caveat>
          </Card>
        </div>
      )}
    </div>
  )
}
