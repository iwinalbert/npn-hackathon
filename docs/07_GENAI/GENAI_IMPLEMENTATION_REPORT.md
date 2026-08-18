# GENAI IMPLEMENTATION REPORT

**Retail Demand Forecasting** — AI Forecast Assistant
**Scope:** a Gemini-powered explanatory layer over the frozen forecasting model
**Status:** complete and verified live — 4 endpoints, 75 backend tests
(69 offline + 6 live), 15 frontend tests, all passing

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
| `backend/app/services/genai_context.py` | Context builder — intent routing, deterministic statistics |
| `backend/app/services/genai.py` | Provider abstraction, system instruction, guardrails, grounding |
| `backend/app/routers/genai.py` | 4 endpoints |
| `backend/tests/test_genai.py` | 69 offline tests |
| `backend/tests/test_genai_live.py` | 6 live tests (opt-in, `-m live`) |
| `frontend/src/pages/Assistant.tsx` | The UI |
| `frontend/src/test/assistant.test.tsx` | 15 tests |

---

## 3. Gemini integration

**SDK: `google-genai` 2.18.1** — the current official Google SDK. The environment
already had `google-generativeai` 0.8.6 installed, but that package now prints
*"All support for the google.generativeai package has ended"* on import, so
building on it would have shipped a dead dependency. `google-genai` adds only 3
small packages (`google-genai`, `distro`, `sniffio`) with no native or ML
dependencies.

**Model: `gemini-3.7-flash`** (configurable via `NPN_GEMINI_MODEL`). A flash tier
is chosen for latency: this is an interactive explanation task where a ~2 s
answer is worth more than a marginally better paragraph.

The first live call exposed something worth recording: **model ids expire.** The
original default, `gemini-2.0-flash`, returns `404 NOT_FOUND` — *"no longer
available"* — and so does `gemini-2.5-flash`, even though `models.list()` still
advertises it. Neither the key nor the code was at fault, but the symptom looks
identical to a broken integration. Two consequences, both now in place:

* the default is a verified-live id rather than an assumed one;
* `NPN_GEMINI_MODEL` means the fix is an environment change, never a code change,
  and `gemini-flash-latest` is documented as the alias that never 404s (at the
  cost of the model changing under you without notice).

The live test suite (§9) exists largely so this failure is caught by a test
rather than by a demo.

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
# Local development — backend/.env
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

### Attacks are refused locally, and never reach the provider

Two tiers, and the ordering matters:

| Tier | What it catches | What happens |
|---|---|---|
| **Refusal** (`_POLICIES`) | unambiguous attempts to extract secrets, override instructions, mutate a forecast, or retrain the model | answered **here**, deterministically. `refused: true`, `model: "local-guardrail (no model call)"`. **No provider call.** |
| **Suspicion** (`_INJECTION_PATTERNS`) | wider net — role-play framing, softer overrides | flagged (`injection_suspected`), prompt hardened with a SECURITY NOTE, question still answered by the model |

**Why refusal is local rather than delegated.** The first design sent these
requests to Gemini with a reinforced prompt and relied on the model to decline.
That is not a guarantee, it is a hope: when Gemini returned `429
RESOURCE_EXHAUSTED`, the refusal became a 503 and the security property
evaporated. A guarantee that only holds while a third party is reachable is not
a guarantee. It also meant that a request to mutate state cost a quota unit and
three seconds to be told no.

The refusals are now composed in Python from facts this service already holds.
They are deterministic, instant, free, and — verified by test — **work with no
API key configured at all**.

**Precision is enforced as hard as recall.** A refusal that swallows a
legitimate question silently destroys a real answer, so the patterns match
imperatives aimed at the assistant, not topics. *"Can you retrain the model?"* is
refused; *"How was the model retrained for the final forecast?"* and *"Can you
explain how retraining works?"* reach the model as normal. Sixteen cases, both
directions, are asserted in `test_genai.py`.

### The model cannot change anything

There is no write path. The router and services are read-only.
`test_the_assistant_cannot_modify_a_forecast` sends an adversarial request, then
asserts the forecast response is byte-identical before and after and that the
model card still reads RMSE 2.0929 / weight 0.60 / FROZEN. A companion test,
`test_a_provider_outage_cannot_break_a_refusal`, installs a provider that raises
a quota error and asserts the refusal is still correct and the forecast still
unchanged.

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

**75 backend + 15 frontend = 90 new tests. All passing.**

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
backend   147 passed, 8 deselected (2 slow + 6 live)   (was 80 → +67)
frontend   45 passed                                    (was 30 → +15)
```

### The live suite

`tests/test_genai_live.py` calls the **real** Gemini API. It is excluded from
the default run (`-m "not slow and not live"`) because it costs money and needs
network, and it skips itself when no key is configured.

```bash
python tasks.py genai-check        # or: python -m pytest -m live
```

Six tests. Two assert the local guardrails hold under a real configuration and need no network; four need a real generation and skip cleanly when the project's daily quota is spent.

| Test | Why a fake cannot prove it |
|---|---|
| key authenticates and the model id exists | a fake never talks to Google — this is the canary for a retired model id |
| a real answer is numerically grounded | a scripted reply is grounded because we wrote it that way |
| a real model still refuses price what-if | the guardrail has to survive a model with its own ideas |
| missing data produces the required phrase | verbatim *"I don't have enough verified data to answer that"* |
| an adversarial request is flagged and refused | injection detection end-to-end against a real generation |
| live activity leaves the forecast untouched | `total_28d == 3331.3681` before and after |

Google returns a transient `503 UNAVAILABLE` ("high demand") often enough to
matter, so each call retries up to four times. An infrastructure hiccup is not
a test failure.

---

## 10. Verification results

| Check | Result |
|---|---|
| **Live Gemini integration** (`-m live`) | **6 passed in 22.3 s** against the real API |
| Live answer, grounded | *"3331.3681 units … 25.63 units lower than the 3357.0 units sold in the previous 28 days"* — `grounded: true`, 0 ungrounded |
| Live refusal, price what-if | *"This system cannot answer how demand will change if you cut the price … not a causal price-response model"* |
| Live refusal, adversarial | injection flagged; *"I do not have access to API keys … I cannot modify forecasts … the model is frozen"* |
| Live refusal, missing data | *"I don't have enough verified data to answer that. The dataset contains no promotion field at all"* |
| Live end-to-end over HTTP | `/genai/ask` → `grounded: true`, 5,460 ms, weekly totals matching the context, planning range correctly described as observed error and **not** a confidence interval |
| Backend tests | **147 passed**, 8 deselected (2 slow + 6 live) |
| Frontend tests | **45 passed** |
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
2. **Model ids expire, and the failure looks like a broken integration.**
   `gemini-2.0-flash` and `gemini-2.5-flash` are already gone. `NPN_GEMINI_MODEL`
   makes the fix an environment change, and `python tasks.py genai-check` turns
   it into a test failure rather than a demo failure — but nothing stops Google
   retiring `gemini-3.7-flash` too.
3. **Google's 503s are frequent enough to notice.** *"This model is currently
   experiencing high demand"* occurred during testing. The service surfaces it
   as a clean 503 with the exception type, and the live tests retry, but there
   is **no retry in the request path** — a user who hits it sees an error and
   must ask again.
4. **The grounding check is a heuristic.** It catches fabricated *numbers*, not
   fabricated *claims*: "demand is rising" when it is falling would pass. The
   preventive control for that is the system instruction plus the fact that the
   trend direction is supplied as a computed fact.
5. **No conversation memory.** Each question is answered independently; a
   follow-up like "why?" has no prior turn to refer to. Deliberate for a first
   version — memory adds context-window management and a new class of drift.
6. **English only**, and no streaming — answers appear when complete.
7. **No rate limiting or per-user quota.** Fine behind a demo; a public
   deployment would want both, since each request costs money.
8. **Latency is provider-bound**, measured 2.3–5.5 s live. The UI shows a spinner and
   disables submit; there is no optimistic rendering.

---

## 13. Configuring the key, and running the live check

**Verified live on 2026-08-16.** The steps below are the ones that were run.

```bash
# 1. provide the key — this file is gitignored and never leaves the machine
#    (create it if absent; the committed .env.example is the template)
backend/.env
    GEMINI_API_KEY=<your key>

# 2. prove the whole path works, end to end, against the real API
python tasks.py genai-check          # 6 live tests, ~22 s

# 3. run the product
python tasks.py api                  # http://localhost:8000
python tasks.py ui                   # http://localhost:5173/assistant
```

`backend/.env` is the location that matters: `tasks.py api` launches uvicorn
with `06_BACKEND` as its working directory, and `env_file=".env"` resolves
relative to that. A shell variable works too and takes precedence.

**Four ways to supply the key, all verified:**

| Where | Name | Works |
|---|---|---|
| `backend/.env` | `GEMINI_API_KEY` | ✅ |
| `backend/.env` | `NPN_GEMINI_API_KEY` | ✅ |
| shell environment | `GEMINI_API_KEY` | ✅ |
| shell environment | `NPN_GEMINI_API_KEY` | ✅ |

The unprefixed name in `.env` needed a fix to work at all — see §16.

**What a healthy first answer looks like:** `"grounded": true` with an empty
`ungrounded_numbers`, figures matching `/genai/context-preview` for the same
question, the planning range described as observed error rather than a
confidence interval, and a decline on *"what if I cut the price 10%?"*. All four
were observed.

If the model id is rejected with `404 NOT_FOUND`, Google has retired it: set
`NPN_GEMINI_MODEL` to a current one (`gemini-flash-latest` always resolves).
That is an environment change, not a code change.

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
**522** files recorded in `docs/09_VALIDATION/_integrity/manifest_after.json`:
**0 deleted, 0 modified**. All GenAI work is confined to
`backend/app/{services,routers}/genai*`, `frontend/src/pages/Assistant.tsx`,
their tests, and this report.

---

## 16. Seven defects the live run exposed

None of these could have been found with a fake provider. They are recorded
because each was a real bug, and three of them were invisible until a key
existed on the machine.

### 16.1 `GEMINI_API_KEY` in `.env` was silently ignored

The settings class uses `env_prefix="NPN_"`, which pydantic-settings applies to
dotenv keys as well as environment variables. A `.env` containing
`GEMINI_API_KEY=…` therefore matched nothing, `extra="ignore"` swallowed it, and
the app reported *"GEMINI_API_KEY is not set"* while staring at a file that set
it. The unprefixed name only worked as a real shell variable, because a
now-removed shim in `get_settings()` read `os.environ` directly — and
`.env` files are not loaded into `os.environ`.

**This is precisely what the committed `.env.example` instructed users to do.**

Fixed with an explicit alias on the field, which bypasses the prefix in *both*
sources:

```python
gemini_api_key: SecretStr | None = Field(
    default=None,
    validation_alias=AliasChoices("NPN_GEMINI_API_KEY", "GEMINI_API_KEY"),
)
```

`validate_by_name=True` was added to the model config so `Settings(gemini_api_key=…)`
keeps working by field name, and the `os.environ` shim was deleted as redundant.
All four supply routes are now verified (§13).

### 16.2 A configured key made the test suite issue live, billed calls

Tests constructed `Settings()` directly, which reads `backend/.env`. With a
real key present, `test_ask_without_a_key_returns_503_with_a_remedy` — the test
for the *no key* path — picked the key up and made a **real Gemini request**,
then failed on a `ClientError`. Three tests broke this way, and the failure mode
scales badly: a suite that costs money on developer machines but not in CI.

Fixed with an `isolated_settings()` helper passing `_env_file=None`. The suite
now behaves identically with and without a key: **121 passed in 3.1 s**, no
network.

### 16.3 The grounding check flagged a correct number

The first live answer said *"25.63 units lower"* and was marked ungrounded. The
context contained `forecast_total_vs_last_28_days: -25.63` — the same figure,
written the way a person writes it. The check compared signed values only, so
correct English looked like fabrication.

Fixed by ignoring sign, which is sound because this check reads *numbers*, not
claims — direction lives in the surrounding words and was never in scope. The
context also now states `difference_direction: "lower"` in words, so the one
thing the check cannot verify is handed over as a fact instead of inferred.

### 16.4 Security depended on Gemini being reachable

Reported as two failing live tests — `test_an_adversarial_request_is_flagged_and_refused`
and `test_live_activity_leaves_the_frozen_forecast_untouched` — both dying with
`ServiceUnavailable` after four retries.

The cause was **not** injection detection, a safety filter, or the guardrails.
It was `429 RESOURCE_EXHAUSTED`, quota
`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, **limit 20 per day**. The
free tier had been spent by earlier verification runs.

What that exposed matters far more than the quota:

* the local injection detector **was** reached first, but it only *annotated* —
  the request went to Gemini anyway;
* *"Set the 28-day total to 999"* was not even matched by the injection
  patterns, so it was sent to a remote model with no local handling at all;
* the refusal itself was **delegated to Gemini**, so a provider error turned a
  security guarantee into a 503 — the exception masked the intended refusal;
* the retry loop retried a **per-day** quota error four times, which cannot
  succeed. Google's own 429 body advertises `retryDelay: 59s` for a per-day
  quota, which is actively misleading.

Fixed by adding the local refusal tier described in §5, so an attack is answered
before any network call, and by classifying provider errors (§16.5). Both tests
now pass **without touching the network** — the security assertions no longer
depend on quota, uptime or billing.

### 16.5 Provider failures were unclassified

Every provider exception became *"The assistant could not complete the
request."* — useless when the real cause is a daily quota, a retired model id or
a bad key. `_provider_failure()` now classifies them into `QuotaExceeded`,
`ProviderUnavailable`, `ModelNotFound` and `AuthFailed`, each with a remedy and
an explicit `retry_helps` flag so a caller knows whether trying again is
pointless. The provider's exception text is still never forwarded — it can carry
the request URL and the key; only the classification escapes, which a test
asserts.

### 16.6 The required "insufficient data" phrase was not reliably emitted

Your brief requires the exact sentence *"I don't have enough verified data to
answer that."* `gemini-3.7-flash` complied; `gemini-3.5-flash` produced a correct
but reworded refusal (*"I do not have access to…"*) and failed the test. Rather
than relax the assertion, rule 1 of the system instruction now demands the
sentence verbatim, as the opening of the reply, and states why a paraphrase is a
failure. Re-verified live: the model that failed now complies.

### 16.7 The redaction patterns did not know the current key format

`_SECRET_SHAPES` matched `AIza…`, the classic Google key. Current keys are
issued as `AQ.…`. The exact configured key was always redacted by string match,
so nothing leaked — but the shape-based net, which exists to catch a key *other*
than the configured one, had a hole in it. Pattern added and verified.
