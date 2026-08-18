# CI/CD

Pipeline: [`../../.github/workflows/ci.yml`](../../.github/workflows/ci.yml)
(at the repository root because GitHub Actions requires that path).

---

## The constraint that shapes the whole pipeline

**The 130 MB product data layer is not in git, and neither are the frozen model
binaries it is generated from.**

A CI runner therefore cannot build it, and the API cannot reach `ready: true`
there. Any pipeline that pretends otherwise is either permanently red or
quietly meaningless.

So this pipeline is explicit about the split, and the `deploy-gate` job prints
it into every run summary so a green tick is never read as more assurance than
it is.

| Verified in CI | Verified only on a host |
|---|---|
| compose validity, build-context hygiene, secret hygiene | the frozen-forecast canary (`3331.3681`) |
| frontend tests, typecheck, production build | data-dependent backend tests |
| no key in the built bundle | the API reaching `ready: true` |
| images build (both targets + frontend) | end-to-end serving correctness |
| containers boot, run as uid 10001 | |
| no secret in any image layer | |
| the API **degrades correctly** with no data | |
| SPA deep-link fallback through nginx | |

The host-side half is the release checklist in
[`runbook.md`](runbook.md). Both halves are required before a release.

---

## Jobs

```
preflight ──┬──► images ──┐
            │             │
backend ────┼─────────────┼──► deploy-gate
            │             │
frontend ───┴─────────────┘
```

### 1. `preflight` — config, context, secrets

Runs `infra/scripts/preflight.py --skip-data`. Fast, and it catches the
failures a successful `docker build` does **not** catch:

- a compose file that no longer parses, or lost a healthcheck or restart policy
- a research mount that stopped being read-only
- a **new top-level directory leaking into the build context** — this already
  happened once, when `study-materials` was added
- a literal secret in a compose file
- a committed API key in any tracked file
- an unpinned base image, a missing `HEALTHCHECK`, a `USER` that moved after
  `CMD`

`--skip-data` is not a way of hiding a failure: it acknowledges that the data
layer legitimately cannot exist on a runner, while every other section still
gates the build.

### 2. `backend` — import, schema, non-data tests

Data-dependent tests skip here (`conftest.py` skips when `product.duckdb` is
absent), and the summary says so rather than hiding it behind a green tick.
What this job does prove:

- **the app imports** — catches a bad import, a missing package, or a settings
  field that will not validate, all of which would otherwise first appear as a
  container that starts and immediately dies
- **`openapi.json` is current** — it is committed and consumed by the frontend,
  so drift should be a review comment, not a surprise
- every test that does not need data passes

### 3. `frontend` — typecheck, tests, build

Fully meaningful on CI; no data layer needed. `npm ci` (not `install`) so the
lockfile is honoured exactly.

The last step greps the built bundle for a Google API key shape. Vite inlines
any `VITE_`-prefixed variable at build time, so a mis-named environment variable
is a realistic way to put a secret in front of every user. This is the check
that catches it.

### 4. `images` — build, boot, harden

Builds all three images for real, then asserts what static analysis could only
argue:

| Check | Asserts |
|---|---|
| `docker run --entrypoint id` | uid **10001** in both backend targets |
| `printenv GEMINI_API_KEY` | empty — no secret baked into the image |
| filesystem grep over `/app` | no key shape in any layer |
| boot with **no data mount** | `/health` 200, `/ready` **false with a reason** |
| frontend boot + `/forecast` | 200 — the SPA fallback works |

The "degrades correctly" check is the honest version of "does it work on CI".
The data layer is absent, so `ready` **must** be false — and it must say so in a
structured way rather than crashing or hanging. A regression there is exactly
what turns a missing mount into an unexplained hang in production.

Both containers are put on a user-defined network. nginx resolves `proxy_pass`
upstreams at startup once `envsubst` has made the host a literal, so on the
default bridge it would refuse to boot with `host not found in upstream`.

Layer caching uses `type=gha`, so an unchanged dependency layer is not rebuilt.

### 5. `deploy-gate` — the verdict

Runs with `if: always()`, prints a result table plus an explicit **"Not proven
by this run"** section listing the host-side checks, and fails if any upstream
job failed.

---

## Running the pipeline locally

Everything CI does, you can do on your machine — and with the data layer
present, you get the checks CI cannot run:

```bash
python tasks.py preflight              # = the preflight job, but WITH data
python tasks.py test                   # = backend job, but data tests RUN
python tasks.py ui-test                # = frontend job
python tasks.py ui-build
python tasks.py docker-up              # = images job
python tasks.py smoke                  # <- what CI structurally cannot do
```

Or in one command: `python tasks.py verify-all`.

---

## No CD, deliberately

The pipeline builds and verifies. It does not deploy.

There is no deployment target, no cloud account and no registry configured for
this project, and a deploy job pointing at nothing is worse than no deploy job —
it looks like a capability that does not exist.

To add one when there is a target, the shape is:

```yaml
  publish:
    needs: [deploy-gate]
    if: github.ref == 'refs/heads/main'
    permissions:
      contents: read
      packages: write          # or id-token: write for OIDC to a cloud
    steps:
      - uses: docker/login-action@v3
        with: { registry: ghcr.io, username: ${{ github.actor }},
                password: ${{ secrets.GITHUB_TOKEN }} }
      - uses: docker/build-push-action@v6
        with:
          context: .
          file: backend/Dockerfile
          target: api
          push: true
          tags: ghcr.io/${{ github.repository }}/api:${{ github.sha }}
```

Tag with `github.sha`, not `latest` — `latest` makes rollback ambiguous.

**Whatever deploys, the deploy is not finished until `smoke_test.py` passes
against the deployed URL:**

```bash
python infra/scripts/smoke_test.py --api-url https://... --web-url https://...
```

It is stdlib-only and exits non-zero on failure, so it works as a deployment
gate anywhere.

---

## Adding a check

Put it where it belongs:

| Kind of check | Where |
|---|---|
| Static — config, files, hygiene | `infra/scripts/preflight.py` |
| Live — a running deployment | `infra/scripts/smoke_test.py` |
| Application behaviour | the backend or frontend test suite |
| Image property (uid, layers, size) | the `images` job in `ci.yml` |

Both scripts share a shape: `record()` / `s.ok()`, `s.warn()`, `s.fail()`, a
`--json` mode, and exit 1 on any FAIL. WARNs never fail a run — if something
should block a release, make it a FAIL.

Prefer preflight over a CI-only step: it runs identically on a laptop, so a
developer hits it before pushing rather than after.

---

## Status

The pipeline has **never executed** — this repository has no git remote yet, so
no runner has ever picked it up. What has been verified locally is every step it
runs:

| Step | Verified locally |
|---|---|
| `preflight --skip-data` | passes |
| backend tests | 149 passed |
| `openapi.json` current | no diff |
| frontend typecheck | clean |
| frontend tests | 62 passed |
| frontend build | clean, 5.0 s |
| no key in `dist/` | none found |
| workflow YAML | parses; job graph correct |
| image build / boot | **not verified — Docker is not installed here** |

Expect the first real run to need adjustment in the `images` job, since it is the
only one whose steps have never executed anywhere.
