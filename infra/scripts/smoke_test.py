"""
Post-deploy smoke test. Run this against a RUNNING stack.

    python infra/scripts/smoke_test.py                    # default local stack
    python infra/scripts/smoke_test.py --api-url http://localhost:8000
    python infra/scripts/smoke_test.py --skip-web         # API only (host dev run)
    python infra/scripts/smoke_test.py --json             # machine-readable

Standard library only -- no pip install. That matters because this has to be
runnable from a CI runner, a bastion host, or `docker compose exec` inside the
API container itself, none of which are guaranteed to have httpx.

WHAT IT ASSERTS, AND WHY EACH ONE EARNED ITS PLACE
--------------------------------------------------
A deploy can pass "the container is running" and still be broken in ways users
see immediately. Each check below corresponds to a failure that a process-level
health probe does not catch:

  ready vs health   health only proves the process answers. `ready` proves the
                    125 MB data layer is actually queryable. A stack with an
                    empty mount passes health forever.
  frozen forecast   proves the deployment is serving the RIGHT data, not merely
                    some data. A stale or half-built product.duckdb answers 200
                    with wrong numbers.
  SPA deep link     /forecast must return the app, not 404. This breaks the
                    instant nginx's try_files is wrong, and only on refresh --
                    so clicking around in a browser will not find it.
  proxy path        the browser reaches the API only through the frontend. If
                    API_HOST is wrong, the SPA loads perfectly and every request
                    inside it fails.
  secret scan       asserts no API key appears in any response body.

Exit 0 = deployment is good. 1 = at least one FAIL.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

DEFAULT_API = "http://localhost:8000"
DEFAULT_WEB = "http://localhost:8080"

# The frozen champion's 28-day total for this series. Asserted, not printed:
# it is the cheapest end-to-end proof that the deployed data layer is the
# validated one. Source: MODEL_FREEZE.md / DOCKER_IMPLEMENTATION_REPORT sec. 8.
CANARY_SERIES = ("CA_3", "FOODS_3_090")
CANARY_TOTAL_28D = 3331.3681
CANARY_TOLERANCE = 0.001

# Google API key shape. Deliberately narrow: a generic "long base64-ish string"
# regex matches artefact hashes and produces noise nobody investigates.
SECRET_RX = re.compile(r"AIza[0-9A-Za-z_\-]{35}")

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
RESULTS: list[dict] = []


def record(name: str, status: str, detail: str = "", ms: float | None = None):
    RESULTS.append({"name": name, "status": status, "detail": detail,
                    "ms": round(ms, 1) if ms is not None else None})


def get(url: str, timeout: float, method: str = "GET"):
    """Fetch a URL. Returns (status, body, elapsed_ms) -- never raises."""
    req = urllib.request.Request(url, method=method,
                                 headers={"User-Agent": "npn-smoke/1.0"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", errors="replace")
            return r.status, body, dict(r.headers), (time.perf_counter() - t0) * 1000
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, body, dict(e.headers), (time.perf_counter() - t0) * 1000
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return None, str(e), {}, (time.perf_counter() - t0) * 1000


def wait_for(url: str, timeout: float, attempts: int, delay: float) -> bool:
    """
    Poll until the stack answers.

    A smoke test that runs the instant `up` returns is testing a container that
    has not finished booting. Rather than a blind sleep, poll: fast when the
    stack is already warm, patient when it is not.
    """
    for i in range(attempts):
        status, _, _, _ = get(url, timeout)
        if status == 200:
            if i:
                print(f"  (ready after {i * delay:.0f}s)", file=sys.stderr)
            return True
        time.sleep(delay)
    return False


# ---------------------------------------------------------------------------
def check_api(base: str, timeout: float) -> list[str]:
    """Returns the response bodies collected, for the secret scan."""
    bodies: list[str] = []
    api = f"{base}/api/v1"

    # --- liveness ---------------------------------------------------------
    status, body, _, ms = get(f"{api}/health", timeout)
    bodies.append(body)
    if status == 200:
        record("api: /health", PASS, "200", ms)
    else:
        record("api: /health", FAIL, f"got {status}: {body[:200]}", ms)
        return bodies                       # nothing else can pass

    # --- readiness: the check that actually matters -----------------------
    status, body, _, ms = get(f"{api}/ready", timeout)
    bodies.append(body)
    if status != 200:
        record("api: /ready", FAIL, f"got {status}", ms)
    else:
        try:
            doc = json.loads(body)
        except ValueError:
            record("api: /ready", FAIL, "response is not JSON", ms)
            doc = {}
        if doc.get("ready") is True:
            record("api: /ready", PASS, "ready=true", ms)
        else:
            record("api: /ready", FAIL,
                   f"ready={doc.get('ready')} -- the data layer is not "
                   f"queryable. Detail: {json.dumps(doc)[:300]}", ms)

    # --- the canary: right data, not just some data -----------------------
    store, item = CANARY_SERIES
    status, body, _, ms = get(f"{api}/series/{store}/{item}/forecast", timeout)
    bodies.append(body)
    if status != 200:
        record("api: frozen forecast", FAIL, f"got {status}", ms)
    else:
        try:
            doc = json.loads(body)
            total = float(doc.get("total_28d"))
        except (ValueError, TypeError):
            record("api: frozen forecast", FAIL,
                   f"no numeric total_28d in {body[:200]}", ms)
        else:
            delta = abs(total - CANARY_TOTAL_28D)
            if delta <= CANARY_TOLERANCE:
                record("api: frozen forecast", PASS,
                       f"{store}/{item} total_28d={total} OK", ms)
            else:
                record("api: frozen forecast", FAIL,
                       f"total_28d={total}, expected {CANARY_TOTAL_28D} "
                       f"(delta {delta:.4f}) -- the deployed data layer is NOT "
                       f"the validated one", ms)

    # --- model provenance --------------------------------------------------
    status, body, _, ms = get(f"{api}/meta/model", timeout)
    bodies.append(body)
    record("api: /meta/model", PASS if status == 200 else FAIL,
           "200" if status == 200 else f"got {status}", ms)

    # --- optional subsystems: must ANSWER, need not be enabled ------------
    for path, label in (("/genai/status", "genai"),
                        ("/inference/status", "inference")):
        status, body, _, ms = get(f"{api}{path}", timeout)
        bodies.append(body)
        if status != 200:
            record(f"api: {path}", FAIL, f"got {status}", ms)
            continue
        try:
            doc = json.loads(body)
        except ValueError:
            record(f"api: {path}", FAIL, "not JSON", ms)
            continue
        # Both endpoints report availability under different key names; either
        # answer is a pass, because both subsystems are optional by design.
        avail = doc.get("available", doc.get("configured", doc.get("enabled")))
        record(f"api: {path}", PASS,
               f"available={avail}" if avail else
               f"unavailable (by design if not configured) -- {label}", ms)

    # --- docs ---------------------------------------------------------------
    # Parsed, not just status-checked. When the base URL is the frontend (the
    # --prod topology, where port 8000 is unpublished), nginx does not proxy
    # this path -- its SPA fallback answers 200 with index.html. A bare status
    # check would call that a pass.
    status, body, _, ms = get(f"{base}/openapi.json", timeout)
    if status != 200:
        record("api: /openapi.json", WARN, f"got {status}", ms)
    else:
        try:
            spec = json.loads(body)
        except ValueError:
            record("api: /openapi.json", WARN,
                   "200 but not JSON -- expected when testing through the "
                   "frontend proxy, which does not expose this path", ms)
        else:
            record("api: /openapi.json", PASS,
                   f"{len(spec.get('paths', {}))} paths", ms)

    return bodies


def check_web(base: str, timeout: float) -> list[str]:
    bodies: list[str] = []

    status, body, headers, ms = get(f"{base}/healthz", timeout)
    if status == 200:
        record("web: /healthz", PASS, "200", ms)
    else:
        record("web: /healthz", FAIL, f"got {status}", ms)
        return bodies

    # --- the SPA itself ----------------------------------------------------
    status, body, headers, ms = get(f"{base}/", timeout)
    bodies.append(body)
    if status == 200 and "<div id=\"root\"" in body or (
            status == 200 and "<html" in body.lower()):
        record("web: /", PASS, f"200 | {len(body):,} bytes", ms)
    else:
        record("web: /", FAIL, f"got {status}", ms)

    cc = headers.get("Cache-Control", "")
    if "no-cache" in cc or "no-store" in cc:
        record("web: index.html not cached", PASS, cc)
    else:
        record("web: index.html not cached", WARN,
               f"Cache-Control={cc!r} -- a deploy may not be picked up")

    # --- client-side routing ------------------------------------------------
    # The single most common nginx misconfiguration in an SPA deployment, and
    # invisible until a user refreshes on a sub-route.
    status, body, _, ms = get(f"{base}/forecast", timeout)
    bodies.append(body)
    if status == 200 and "<html" in body.lower():
        record("web: SPA deep link /forecast", PASS, "200, serves the app", ms)
    else:
        record("web: SPA deep link /forecast", FAIL,
               f"got {status} -- try_files fallback is wrong", ms)

    # --- the proxy ----------------------------------------------------------
    status, body, _, ms = get(f"{base}/api/v1/health", timeout)
    bodies.append(body)
    if status == 200:
        record("web: /api proxy -> API", PASS, "200", ms)
    else:
        record("web: /api proxy -> API", FAIL,
               f"got {status} -- API_HOST is wrong, or the API is unreachable "
               f"on the internal network", ms)

    # --- security headers ---------------------------------------------------
    _, _, headers, _ = get(f"{base}/", timeout)
    for header, expected in (("X-Content-Type-Options", "nosniff"),
                             ("X-Frame-Options", "SAMEORIGIN"),
                             ("Referrer-Policy", "no-referrer")):
        got = headers.get(header, "")
        if expected.lower() in got.lower():
            record(f"web: {header}", PASS, got)
        else:
            record(f"web: {header}", WARN, f"missing or unexpected: {got!r}")

    return bodies


def check_secrets(bodies: list[str]) -> None:
    hits = sum(len(SECRET_RX.findall(b or "")) for b in bodies)
    if hits:
        record("no API key in any response body", FAIL,
               f"{hits} key-shaped string(s) found -- INVESTIGATE IMMEDIATELY")
    else:
        record("no API key in any response body", PASS,
               f"scanned {len(bodies)} responses")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-url", default=DEFAULT_API,
                    help=f"API base URL (default {DEFAULT_API})")
    ap.add_argument("--web-url", default=DEFAULT_WEB,
                    help=f"frontend base URL (default {DEFAULT_WEB})")
    ap.add_argument("--skip-web", action="store_true",
                    help="API only -- for a host dev run with no frontend container")
    ap.add_argument("--skip-api", action="store_true",
                    help="frontend only")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--wait", type=int, default=0, metavar="SECONDS",
                    help="poll for the stack to come up before testing")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    api = args.api_url.rstrip("/")
    web = args.web_url.rstrip("/")

    if args.wait:
        target = f"{api}/api/v1/health" if not args.skip_api else f"{web}/healthz"
        attempts = max(1, int(args.wait / 3))
        if not wait_for(target, args.timeout, attempts, 3.0):
            print(f"stack did not answer {target} within {args.wait}s",
                  file=sys.stderr)
            return 1

    bodies: list[str] = []
    if not args.skip_api:
        bodies += check_api(api, args.timeout)
    if not args.skip_web:
        bodies += check_web(web, args.timeout)
    check_secrets(bodies)

    failures = [r for r in RESULTS if r["status"] == FAIL]
    warnings = [r for r in RESULTS if r["status"] == WARN]

    if args.json:
        print(json.dumps({"ok": not failures, "failures": len(failures),
                          "warnings": len(warnings), "checks": RESULTS}, indent=2))
        return 1 if failures else 0

    mark = {PASS: "  ok  ", WARN: " warn ", FAIL: " FAIL "}
    print(f"\nSmoke test\n  api {api}\n  web {'(skipped)' if args.skip_web else web}\n")
    for r in RESULTS:
        ms = f"  [{r['ms']:.0f} ms]" if r["ms"] else ""
        detail = f"  {r['detail']}" if r["detail"] else ""
        print(f"[{mark[r['status']]}] {r['name']}{detail}{ms}")

    print(f"\n{'=' * 60}")
    print(f"{len(RESULTS) - len(failures) - len(warnings)} passed | "
          f"{len(warnings)} warnings | {len(failures)} failures")
    if failures:
        print("\nSMOKE TEST FAILED -- this deployment is not serving correctly:")
        for r in failures:
            print(f"  - {r['name']}: {r['detail']}")
        print("\nTriage: infra/docs/troubleshooting.md")
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
