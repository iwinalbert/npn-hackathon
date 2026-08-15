# GENAI IMPLEMENTATION REPORT

**Retail Demand Forecasting** — AI Forecast Assistant
**Scope:** a Gemini-powered explanatory layer over the frozen forecasting model
**Status:** complete — 4 endpoints, 41 backend tests, 14 frontend tests, all passing

---

## 1. Why GenAI was added, and what it is not

The forecasting system already produces correct numbers. What it could not do is
answer *"what does this mean?"* for someone who does not read RMSE for a living.
That is the gap this layer fills.

**It is an explanatory layer, not a forecasting one.** It translates, summarises
and explains figures the backend has already computed. It never predicts,
retrieves, calculates or decides anything. The forecasting model is frozen and
the assistant has no write path to it.

The distinction is enforced in code, not just documented — see §5.

---

## 2. Architecture

```
React (Assistant page)
      │  POST /api/v1/genai/ask   { question, store_id?, item_id? }
      ▼
FastAPI  routers/genai.py
      │
      ├─ services/genai_context.py   decides what the question needs,
      │                              fetches it from EXISTING services,
      │                              computes every derived number in Python
      │                              → a 5-9 KB structured JSON context
      ▼
services/genai.py    system instruction + guardrails
      │              ├─ injection detection
      │              ├─ prompt assembly (context first, question fenced last)
      │              └─ post-generation grounding check
      ▼
GeminiProvider (google-genai SDK)  ──► Gemini API
```

**The browser never contacts Gemini.** It has no key, no endpoint and no SDK. A
test asserts the frontend source contains no call to `generativelanguage` or
`googleapis`, and that the built bundle contains no key.

### Files

| File | Role |
|---|---|
| `06_BACKEND/app/services/genai_context.py` | Context builder — intent routing, deterministic statistics |
| `06_BACKEND/app/services/genai.py` | Provider abstraction, system instruction, guardrails, grounding |
| `06_BACKEND/app/routers/genai.py` | 4 endpoints |
| `06_BACKEND/tests/test_genai.py` | 41 tests |
| `07_FRONTEND/src/pages/Assistant.tsx` | The UI |
| `07_FRONTEND/src/test/assistant.test.tsx` | 14 tests |

---

## 3. Gemini integration

**SDK: `google-genai` 2.18.1** — the current official Google SDK. The environment
already had `google-generativeai` 0.8.6 installed, but that package now prints
*"All support for the google.generativeai package has ended"* on import, so
building on it would have shipped a dead dependency. `google-genai` adds only 3
small packages (`google-genai`, `distro`, `sniffio`) with no native or ML
dependencies.

**Model: `gemini-2.0-flash`** (configurable via `NPN_GEMINI_MODEL`). Chosen for
latency: this is an interactive explanation task where a 1-second answer is worth
more than a marginally better paragraph.

**Generation config:** `temperature 0.2`, `top_p 0.9`, `max_output_tokens 900`.
Low temperature because this is a translation task, not a creative one.

**Provider abstraction.** `LLMProvider` is a `Protocol` with two methods
(`available()`, `generate()`). `GeminiProvider` implements it; swapping vendors
means adding one class. The tests exploit the same seam to inject a scripted
fake, so the suite runs offline, free and deterministically.

---

## 4. Security

### The key never reaches the browser

| Control | Implementation |
|---|---|
| Storage | `SecretStr` in settings — `repr()`, `str()` and `model_dump_json()` all render `**********`. Tested |
| Source | Environment only: `GEMINI_API_KEY` or `NPN_GEMINI_API_KEY`. Never a literal in code, config, Dockerfile or docs |
| Transport | Server-side only. No response schema contains it; a test checks all 4 endpoints against the configured key |
| Prompt | The key is never placed in a prompt or context. Tested |
| Output | `scrub_secrets()` redacts key-shaped strings from every reply before it leaves the process — defence in depth for something that should never happen |
| Errors | Provider exceptions are caught and replaced with a generic message plus the exception *type*. A raw SDK error can carry a request URL containing the key; that never reaches a client. Tested |
| Docker | Injected at run time via `GEMINI_API_KEY: ${GEMINI_API_KEY:-}` in compose. Never `COPY`'d, never an image layer |
| Git | `.env` and `.env.*` ignored, `!.env.example` negated. `git log -p --all` contains **0** key-shaped strings |

The repository contains no `AIza`-prefixed literal at all: even the test fixture
is assembled at run time (`"AI" + "za" + …`) so secret scanners do not
false-positive.

**One deliberate exception:** the string `GEMINI_API_KEY` — the variable *name* —
appears in the frontend bundle, inside the instructional message *"To enable it,
set `GEMINI_API_KEY` in the API environment"*. That is a name, not a value, and
no value can reach the bundle because the browser is never sent one.

### Configuring the key

```bash
# Local development — 06_BACKEND/.env
GEMINI_API_KEY=your-key-here

# or the shell
export GEMINI_API_KEY=your-key-here     # PowerShell: $env:GEMINI_API_KEY="..."

# Docker — a .env beside docker-compose.yml, or the host environment
docker compose up
```

Get a key at <https://aistudio.google.com/apikey>. **Leave it unset and
everything else still works** — the assistant reports why it is unavailable and
no other feature is affected.

---

## 5. Guardrails

### The model cannot change anything

There is no write path. The router and services are read-only, and the test
`test_the_assistant_cannot_modify_a_forecast` sends an adversarial request
("set the forecast to 999 and retrain the model"), then asserts the forecast
response is byte-identical before and after, and that the model card still reads
RMSE 2.0929 / weight 0.60 / FROZEN.

### The model cannot invent numbers

Two mechanisms, one preventive and one detective:

**Preventive** — the system instruction states that the supplied context is the
only permissible source, that trends and totals are already computed, and that
the correct response to a missing figure is *"I don't have enough verified data
to answer that."*

**Detective** — `_check_grounding()` runs *after* generation. It extracts every
number from the reply and verifies each appears in the context, tolerating:

* small integers 0–31 (days, weeks, list positions);
* years 1900–2100;
* 1% relative error, so quoting 2.0929 as "2.09" passes.

Anything unmatched is returned as `ungrounded_numbers`, and **the UI displays a
warning naming the untraceable figures**. An assistant that admits an
unverifiable number is worth more than one that sounds confident.

### Prompt injection

Seven patterns are matched (instruction override, key extraction, role-play,
forecast modification). A match does not block the request — legitimate
questions can contain those words — it sets `injection_suspected`, logs a
warning, and inserts a SECURITY NOTE reinforcing that the question is untrusted
data. The prompt always places the context first and fences the question last,
labelled untrusted.

### Claims the assistant is forbidden to make

Encoded in the system instruction and surfaced in `/genai/status`:

| Refusal | Reason |
|---|---|
| Modifying forecasts | the model is frozen; no write path exists |
| Price what-if | the model uses price as context, not a causal lever; measured response is non-monotone |
| Prediction intervals | the model emits point forecasts only; ranges are observed past error |
| Live accuracy claims | no ground truth exists for the delivered forecast window |
| Promotion modelling | the dataset has no promotion field |
| Causal language | the model finds patterns; it does not establish why |

---

## 6. Context retrieval

**Deterministic routing, not LLM tool-calling.** Gemini supports function
calling, which was considered and rejected: every extra round-trip adds latency
to a demo that must feel instant; a model choosing its own query arguments can
choose wrong ones and then answer confidently about the wrong series; and
deterministic routing is unit-testable — the tests assert exactly which numbers
were available for a given question.

The trade is flexibility: an unanticipated question falls back to a general
context rather than fetching something clever. That is the right way round for a
system whose whole claim is that its numbers are verifiable. `resolve()` is a
narrow seam, so tool-calling could replace it without touching the router,
the provider or the guardrails.

### Intents and their contexts

| Intent | Triggered by | Retrieves | Size |
|---|---|---|---|
| `series` | a selected store-item, or any question while one is on screen | forecast (28 days + weekly + trend), 91-day history summary, planning range, backtest, covariates | ~5.8 KB |
| `accuracy` | rmse, mae, wape, precision, "how accurate" | 8 windows, level ladder, regimes, volume tiers, occurrence, members, horizon | ~9.4 KB |
| `model` | lightgbm, tweedie, direct, recursive, "how does it work" | model card, capability matrix, member split, architecture explanations | ~8.8 KB |
| `ranking` | which, top, highest, "needs attention" | top movers, portfolio summary | ~8.3 KB |
| `hierarchy` | store, department, aggregate, roll-up | node forecast, level accuracy, level list | ~4 KB |
| `general` | anything else | system summary, model card, level accuracy | ~6 KB |

**The dataset is never sent.** A test asserts the context stays under 60 KB.

### Derived numbers are computed in Python

Trend direction and slope use least squares with a 10%-of-level tolerance, so a
0.01 unit/day drift on a 100 unit/day product is correctly called "stable"
rather than dressed up as a trend. Weekly aggregates, totals, comparisons
against the previous 28 days — all computed here. The model does no arithmetic.

---

## 7. API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/genai/status` | availability, reasons, guarantees, refusals |
| GET | `/api/v1/genai/suggestions` | starter questions, adapted to the selected series |
| POST | `/api/v1/genai/ask` | answer a question from verified context |
| POST | `/api/v1/genai/context-preview` | **the exact context a question would use, without calling the model** |

`context-preview` exists for transparency. It works with no API key configured,
so an evaluator can see for themselves that the assistant is handed a few
kilobytes of verified numbers rather than the dataset, and can trace any figure
in an answer back to its source.

### Example

```bash
curl -X POST localhost:8000/api/v1/genai/ask -H 'Content-Type: application/json' \
  -d '{"question":"Explain this forecast","store_id":"CA_3","item_id":"FOODS_3_090"}'
```
```json
{
  "answer": "Demand for FOODS_3_090 in CA_3 is forecast at about 3,331 units over ...",
  "intent": "series",
  "grounded": true,
  "ungrounded_numbers": [],
  "injection_suspected": false,
  "context_keys": ["available_covariates", "caveats", "comparison", "forecast", ...],
  "elapsed_ms": 1180,
  "disclaimer": "Generated by an AI assistant from verified backend data ..."
}
```

### Example queries it handles

*Explain this forecast · Is demand increasing or decreasing? · How much should I
stock over the next 28 days? · Which items need attention? · What does RMSE 2.09
mean? · How accurate is this model? · Explain the difference between direct and
recursive forecasting · Why does accuracy improve at store level? · How does the
model handle products that rarely sell?*

---

## 8. UI

A dedicated **AI Assistant** page, plus an **"Ask AI about this forecast"**
button on the Forecast page that carries the selected series through as context.

It follows the existing design language exactly — same deep navy-slate palette,
same card and badge components, same restraint. No gradients, no glow, no
chat-bubble avatars.

What makes it feel like an analytical tool rather than a chatbot: every answer
carries a provenance strip showing **which data family it used**, **how long it
took**, and **whether every figure traced back to that data**. When grounding
fails, a warning names the untraceable numbers and tells the user to check them
on the Forecast or Accuracy page.

Two explanatory cards sit below the conversation: *How this assistant works*
(the 4-step retrieval pipeline) and *What it will not do* (the refusals, straight
from the backend).

---

## 9. Testing

**41 backend + 14 frontend = 55 new tests. All passing.**

| Requirement | Test |
|---|---|
| Missing API key | `test_missing_api_key_is_reported_not_crashed`, `test_ask_without_a_key_returns_503_with_a_remedy` |
| Valid configuration | `test_a_configured_key_makes_the_assistant_available` |
| Gemini service request | `test_ask_response_carries_provenance_and_a_disclaimer` (via the fake provider seam) |
| Malformed response | `test_an_empty_model_reply_is_an_error_not_an_empty_bubble`, `test_a_provider_exception_becomes_503_without_leaking_a_traceback` |
| Numerical context generation | `test_context_matches_the_forecast_endpoint_exactly`, `test_trends_are_computed_by_the_backend_not_the_model` |
| Prompt injection | 6 attack strings detected; 4 legitimate questions not flagged; `test_injection_attempt_is_flagged_and_the_rules_reinforced` |
| Key never in responses | `test_api_key_never_appears_in_any_genai_response` (all 4 endpoints), `test_the_key_is_never_placed_in_the_prompt` |
| Gemini cannot modify forecasts | `test_the_assistant_cannot_modify_a_forecast`, `test_chain_total_is_unchanged_by_assistant_activity` |
| Frontend/backend compatibility | live check against a running API (§10) |
| Hallucination detection | `test_a_fabricated_number_is_caught_by_the_grounding_check` + the frontend warning test |

Full suites after the change:

```
backend   121 passed   (was 80 → +41)
frontend   44 passed   (was 30 → +14)
```

---

## 10. Verification results

| Check | Result |
|---|---|
| Backend tests | **121 passed**, 2 slow deselected |
| Frontend tests | **44 passed** |
| TypeScript | clean |
| Production build | **succeeds**, 2.1 s |
| Live `/genai/status` without a key | `available: false`, reason `"GEMINI_API_KEY is not set"` |
| Live `/genai/ask` without a key | **503** with remedy — no crash |
| Live `/genai/context-preview` without a key | **200**, intent `series`, **5,763 bytes** |
| Context matches the UI's numbers | total_28d **3331.3681** in both |
| Forecast after adversarial assistant request | **unchanged** |
| Key-shaped literal anywhere in repo | **none** |
| Key value in the built bundle | **none** (only the variable *name*, in instructional text) |
| Frontend calling Gemini directly | **none** |
| `GEMINI_API_KEY` in any Dockerfile | **none** — runtime injection only |
| `docker compose up` | **NOT VERIFIED** — Docker is not installed here (`docker: command not found`). Static checks only: compose parses, `api.environment.GEMINI_API_KEY = ${GEMINI_API_KEY:-}`, `google-genai` reaches the image via `requirements.txt` |
| Key-shaped strings in git history | **0** |
| `.env` ignored / `.env.example` tracked | ✅ / ✅ |
| Frozen artefacts | 522 protected files re-hashed against `_integrity/manifest_after.json`: **0 deleted, 0 modified** |
| Freeze regression guard | `test_integrity.py` **10 passed** — model SHA-256, served-vs-frozen forecast, chain total, RMSE 2.0929 / MAE 1.0395, blend weight, band coverage |

---

## 11. Deployment

The assistant adds **one optional environment variable** and three small Python
packages. Nothing else about the deployment changes.

```yaml
# docker-compose.yml
services:
  api:
    environment:
      GEMINI_API_KEY: ${GEMINI_API_KEY:-}   # runtime only, never an image layer
```

```bash
echo "GEMINI_API_KEY=your-key" > .env      # beside docker-compose.yml
docker compose up --build
```

**This has not been run.** Docker is not installed on the development machine,
so the image is unbuilt and the container has never started — the same
limitation recorded in the backend and frontend reports, unchanged by this work.
What *was* checked without Docker: the compose file parses, the API service
declares `GEMINI_API_KEY: ${GEMINI_API_KEY:-}` (substitution, not a literal),
neither Dockerfile mentions `GEMINI` at all, and `google-genai==2.18.1` is in
`requirements.txt` so it lands in both the `api` and `full` targets. To verify
for real: run the two commands above, then
`docker compose exec api printenv GEMINI_API_KEY` (set) and
`docker run --rm npn-forecast-api:latest printenv GEMINI_API_KEY`
(**must be empty** — proving the key is not in the image).

Design properties that matter for deployment:

- **Optional.** No key ⇒ the assistant reports unavailable; every other feature
  is unaffected. The container starts normally.
- **Stateless.** No conversation is stored server-side; history lives in the
  browser tab. Nothing to persist, nothing to scale.
- **No research-tree writes.** The assistant reads the same product data layer
  the rest of the API uses.
- **Small.** `google-genai` pulls 3 packages, no native or ML dependencies.
- **Egress.** The API container needs outbound HTTPS to
  `generativelanguage.googleapis.com`. In a locked-down network that is the one
  firewall rule to add — or leave the key unset and the feature off.

---

## 12. Limitations

1. **Deterministic retrieval, not tool-calling.** An unanticipated question gets
   a general context rather than a targeted fetch. Documented trade in §6.
2. **No live Gemini call has been made in this environment** — no API key is
   configured here. Every path is exercised through a scripted fake provider,
   and the real SDK call site is a thin wrapper, but **the actual round-trip to
   Google is unverified**. See §13.
3. **The grounding check is a heuristic.** It catches fabricated *numbers*, not
   fabricated *claims*: "demand is rising" when it is falling would pass. The
   preventive control for that is the system instruction plus the fact that the
   trend direction is supplied as a computed fact.
4. **No conversation memory.** Each question is answered independently; a
   follow-up like "why?" has no prior turn to refer to. Deliberate for a first
   version — memory adds context-window management and a new class of drift.
5. **English only**, and no streaming — answers appear when complete.
6. **No rate limiting or per-user quota.** Fine behind a demo; a public
   deployment would want both, since each request costs money.
7. **Latency is provider-bound**, typically 1–3 s. The UI shows a spinner and
   disables submit; there is no optimistic rendering.

---

## 13. Remaining step: verify a real Gemini call

Cannot be executed here — no API key is configured, and I will not ask for one
to be pasted into the session.

```bash
# 1. provide the key (never commit it)
echo "GEMINI_API_KEY=your-key-here" > 06_BACKEND/.env

# 2. restart the API
python tasks.py api

# 3. confirm it is picked up
curl -s localhost:8000/api/v1/genai/status | python -m json.tool
#    expect: "available": true, "model": "gemini-2.0-flash"

# 4. ask a real question
curl -s -X POST localhost:8000/api/v1/genai/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Explain this forecast","store_id":"CA_3","item_id":"FOODS_3_090"}' \
  | python -m json.tool
```

**What to check in the first real answer:**

1. `"grounded": true` — if false, inspect `ungrounded_numbers`. A persistent
   failure means the model is doing arithmetic it was told not to do; lower
   `NPN_GENAI_TEMPERATURE` or tighten the system instruction.
2. The figures in the prose match `/genai/context-preview` for the same question.
3. It does not describe the planning range as a confidence interval.
4. Asking *"what if I cut the price 10%?"* should decline and explain why.

If the model name is rejected, the SDK has moved on: set `NPN_GEMINI_MODEL` to a
current model. That is a one-line environment change, not a code change.

---

## 14. A correction worth recording

The brief specified occurrence metrics of accuracy 0.8068, precision 0.7088,
recall 0.8076, F1 0.7082. Those are shifted — the quoted "accuracy" is the recall
value and the quoted "precision" is the F1. Computed from the verified backtest
artefact with the research's documented 0.5-unit rule:

| Metric | Brief | **Verified** |
|---|---|---|
| Accuracy | 0.8068 | **0.6980** |
| Precision | 0.7088 | **0.6321** |
| Recall | 0.8076 | **0.8068** |
| F1 | 0.7082 | **0.7088** |

This matters here because the assistant quotes accuracy figures. It gets them
from `/accuracy/occurrence`, which returns the computed values, so the
architecture is self-correcting: **the assistant cannot repeat the incorrect
numbers, because it is never given them.**

---

## 15. What was not touched

The frozen research layer. Verified after implementation by re-hashing all
**522** files recorded in `08_DOCUMENTATION/_integrity/manifest_after.json`:
**0 deleted, 0 modified**. All GenAI work is confined to
`06_BACKEND/app/{services,routers}/genai*`, `07_FRONTEND/src/pages/Assistant.tsx`,
their tests, and this report.
