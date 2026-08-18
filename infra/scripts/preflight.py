"""
Pre-deploy checks. Everything here runs WITHOUT Docker.

    python infra/scripts/preflight.py            # all checks
    python infra/scripts/preflight.py --json     # machine-readable

This is the gate that runs before a build, in CI and on a laptop. It answers one
question: "if I run `docker compose up` right now, will it work, and will the
image be the one I meant to ship?"

It deliberately needs no Docker daemon, because the two failures it exists to
catch are both invisible to a successful build:

  * a MISSING DATA LAYER, which does not fail the build at all -- the API starts,
    /ready reports not-ready, its healthcheck never passes, and the frontend
    waits for service_healthy forever. That looks like a hang.
  * a LEAKING BUILD CONTEXT, which does not fail the build either. It just
    quietly ships gigabytes, or a secret, into an image layer.

Exit code 0 = safe to build. 1 = at least one FAIL. WARNs never fail the run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


# ---------------------------------------------------------------------------
# result plumbing
# ---------------------------------------------------------------------------
@dataclass
class Check:
    name: str
    status: str
    detail: str = ""


@dataclass
class Section:
    title: str
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(name, status, detail))

    def ok(self, name: str, detail: str = "") -> None:
        self.add(name, PASS, detail)

    def warn(self, name: str, detail: str = "") -> None:
        self.add(name, WARN, detail)

    def fail(self, name: str, detail: str = "") -> None:
        self.add(name, FAIL, detail)


SECTIONS: list[Section] = []


def section(title: str) -> Section:
    s = Section(title)
    SECTIONS.append(s)
    return s


# ---------------------------------------------------------------------------
# 1. product data layer
# ---------------------------------------------------------------------------
# Named separately from the size floors so a truncated download is caught too:
# a 0-byte product.duckdb passes "exists" and fails everything afterwards.
REQUIRED_DATA = {
    "product.duckdb": 5_000_000,
    "history.parquet": 5_000_000,
    "backtest.parquet": 5_000_000,
}


def check_data_layer(skip: bool = False) -> None:
    s = section("Product data layer")
    if skip:
        # CI runners legitimately have no data layer: it is 130 MB, gitignored,
        # and generated from frozen artefacts that are themselves not in git.
        # Every other section still applies there, which is the point of the
        # flag -- config checks are exactly what CI should be enforcing.
        s.warn("skipped", "--skip-data: not checked on this run")
        return

    data_dir = ROOT / "backend" / "data"

    if not data_dir.is_dir():
        s.fail("data directory", f"{data_dir} does not exist "
                                 f"-- run: python tasks.py build-db")
        return

    total = 0
    for name, floor in REQUIRED_DATA.items():
        p = data_dir / name
        if not p.is_file():
            s.fail(name, "MISSING -- run: python tasks.py build-db")
            continue
        size = p.stat().st_size
        total += size
        if size < floor:
            s.fail(name, f"only {size:,} bytes -- looks truncated; rebuild it")
        else:
            s.ok(name, f"{size / 1e6:,.1f} MB")

    if total:
        s.ok("total mounted at /data/product", f"{total / 1e6:,.1f} MB")


# ---------------------------------------------------------------------------
# 2. compose configuration
# ---------------------------------------------------------------------------
COMPOSE_FILES = [
    "docker-compose.yml",
    "docker-compose.inference.yml",
    "infra/compose/docker-compose.prod.yml",
]

# A literal key here would mean the secret is in the repository. The only
# acceptable form is compose interpolation from the host environment.
SECRET_KEYS = ("GEMINI_API_KEY", "NPN_GEMINI_API_KEY")


def _load_yaml(path: Path):
    try:
        import yaml
    except ImportError:
        return None, "PyYAML not installed (pip install pyyaml)"
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")), None
    except Exception as exc:                              # noqa: BLE001
        return None, str(exc)


def check_compose() -> None:
    s = section("Compose configuration")
    docs: dict[str, dict] = {}

    for rel in COMPOSE_FILES:
        p = ROOT / rel
        if not p.is_file():
            s.fail(rel, "missing")
            continue
        doc, err = _load_yaml(p)
        if err:
            s.fail(rel, f"does not parse: {err}")
            continue
        docs[rel] = doc or {}
        s.ok(rel, f"parses | {len(doc.get('services', {}))} service(s)")

    base = docs.get("docker-compose.yml")
    if not base:
        return

    services = base.get("services", {})

    for name, svc in services.items():
        # Restart policy: without it a crashed container stays down silently.
        if svc.get("restart"):
            s.ok(f"{name}: restart policy", svc["restart"])
        else:
            s.fail(f"{name}: restart policy", "not set")

        # Healthcheck: the frontend's depends_on condition is worthless without
        # one on the API, and no orchestrator can route safely without it.
        if svc.get("healthcheck"):
            s.ok(f"{name}: healthcheck", "present")
        else:
            s.fail(f"{name}: healthcheck", "not set")

        if "no-new-privileges:true" in (svc.get("security_opt") or []):
            s.ok(f"{name}: no-new-privileges", "set")
        else:
            s.warn(f"{name}: no-new-privileges", "not set")

        limits = (svc.get("deploy", {}).get("resources", {})
                     .get("limits", {}))
        if limits.get("memory"):
            s.ok(f"{name}: memory limit", str(limits["memory"]))
        else:
            s.warn(f"{name}: memory limit", "unbounded")

        if svc.get("privileged"):
            s.fail(f"{name}: privileged", "privileged containers are not allowed")
        if svc.get("network_mode") == "host":
            s.fail(f"{name}: network_mode", "host networking defeats isolation")

    # Frozen artefacts and the product data layer must never be writable by the
    # running system.
    #
    # Only the SOURCE side of a bind is inspected. Matching anywhere in the
    # string is wrong: the container-side path `/research/...` also appears on
    # the named scratch volumes, which are writable on purpose (they absorb the
    # mkdir()s pipeline/config.py performs at import time). A named volume has
    # no host path at all, so it is skipped rather than judged.
    protected_sources = ("./research/", "./backend/data")
    for rel, doc in docs.items():
        for name, svc in (doc.get("services") or {}).items():
            for vol in svc.get("volumes") or []:
                if not isinstance(vol, str):
                    continue
                src = vol.split(":", 1)[0]
                if not src.startswith((".", "/")):
                    continue                      # named volume, not a bind
                if not src.startswith(protected_sources):
                    continue
                if vol.endswith(":ro"):
                    s.ok(f"{rel} {name}: read-only mount", src)
                else:
                    s.fail(f"{rel} {name}: mount is writable", vol)

    # The Gemini key must be interpolated, never literal.
    raw = "\n".join((ROOT / f).read_text(encoding="utf-8")
                    for f in COMPOSE_FILES if (ROOT / f).is_file())
    # Strip ${...} spans first. Without this the check reports its own answer as
    # a violation: `GEMINI_API_KEY: ${GEMINI_API_KEY:-}` contains a second,
    # inner occurrence of the name followed by `:-}`, which looks literal.
    raw = re.sub(r"\$\{[^}]*\}", "", raw)
    # [ \t]* rather than \s*: with the interpolation stripped the line is now
    # `GEMINI_API_KEY:` with nothing after it, and \s* would cross the newline
    # and match the next line's content as if it were the value.
    literal = [k for k in SECRET_KEYS
               if re.search(rf"{k}[ \t]*[:=][ \t]*(?!\$)\S+", raw)]
    if literal:
        s.fail("secret interpolation",
               f"literal value found for {', '.join(literal)}")
    else:
        s.ok("secret interpolation", "GEMINI_API_KEY comes from the environment")


# ---------------------------------------------------------------------------
# 3. build context (.dockerignore simulation)
# ---------------------------------------------------------------------------
# The five paths the backend Dockerfile actually COPYs. If any is excluded the
# build fails; if anything large or secret is INcluded, the image is wrong.
REQUIRED_IN_CONTEXT = [
    "backend/requirements.txt",
    "backend/requirements-inference.txt",
    "backend/app/main.py",
    "backend/scripts/build_product_db.py",
    "research/pipeline/config.py",
]

FORBIDDEN_IN_CONTEXT = [
    "backend/.env",
    "backend/data/product.duckdb",
]

# Above this, something has leaked. The measured context is ~0.4 MB.
CONTEXT_BUDGET_MB = 5.0


def _pattern_to_regex(pat: str) -> re.Pattern[str]:
    """
    Translate one .dockerignore pattern into a regex.

    Docker matches with Go's filepath.Match plus `**`. The rule that matters
    most here is that a pattern matching a DIRECTORY excludes everything under
    it, so `research/**` and `research` must both take the subtree.
    """
    out, i = [], 0
    while i < len(pat):
        c = pat[i]
        if c == "*":
            if pat[i:i + 2] == "**":
                out.append(".*")
                i += 2
                if pat[i:i + 1] == "/":
                    i += 1
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(c))
        i += 1
    # Trailing: a match on the path itself also matches its children.
    return re.compile("^" + "".join(out) + "(/.*)?$")


def _parse_dockerignore(path: Path) -> list[tuple[bool, re.Pattern[str], str]]:
    rules = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        if negate:
            line = line[1:]
        line = line.strip().rstrip("/")
        if not line:
            continue
        rules.append((negate, _pattern_to_regex(line), line))
    return rules


def _included(rel: str, rules) -> bool:
    """Last matching rule wins -- Docker's own semantics."""
    included = True
    for negate, rx, _ in rules:
        if rx.match(rel):
            included = negate
    return included


def check_build_context() -> None:
    s = section("Backend build context")
    di = ROOT / ".dockerignore"
    if not di.is_file():
        s.fail(".dockerignore", "missing -- the whole repository would be sent")
        return

    rules = _parse_dockerignore(di)
    s.ok(".dockerignore", f"{len(rules)} rules")

    # Any top-level directory with no rule at all leaks wholesale. This is the
    # exact failure the allow-list is meant to prevent, and it reappears every
    # time someone adds a new top-level folder.
    unruled = []
    for entry in sorted(ROOT.iterdir()):
        # .git is the one directory Docker's builder handles itself.
        if not entry.is_dir() or entry.name == ".git":
            continue
        if _included(entry.name + "/probe", rules):
            unruled.append(entry.name)
    if unruled:
        s.fail("top-level coverage",
               "no exclusion rule covers: " + ", ".join(unruled)
               + " -- these would be sent to the daemon")
    else:
        s.ok("top-level coverage", "every top-level directory has a rule")

    # Walk what would actually be sent, pruning subtrees nothing can re-include.
    negations = [lit for neg, _, lit in rules if neg]
    sent_files, sent_bytes = 0, 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel_dir = Path(dirpath).relative_to(ROOT).as_posix()
        if rel_dir == ".":
            rel_dir = ""
        keep = []
        for d in dirnames:
            sub = f"{rel_dir}/{d}" if rel_dir else d
            if sub in {".git"}:
                continue
            # Descend only if something in the subtree could still be included.
            if _included(f"{sub}/probe", rules) or any(
                    n.startswith(sub) for n in negations):
                keep.append(d)
        dirnames[:] = keep
        for f in filenames:
            rel = f"{rel_dir}/{f}" if rel_dir else f
            if _included(rel, rules):
                sent_files += 1
                try:
                    sent_bytes += (ROOT / rel).stat().st_size
                except OSError:
                    pass

    mb = sent_bytes / 1e6
    detail = f"{sent_files} files | {mb:,.2f} MB"
    if mb > CONTEXT_BUDGET_MB:
        s.fail("context size", f"{detail} -- over the {CONTEXT_BUDGET_MB} MB "
                               f"budget; something is leaking")
    else:
        s.ok("context size", detail)

    for rel in REQUIRED_IN_CONTEXT:
        if not (ROOT / rel).exists():
            s.warn(f"required: {rel}", "file does not exist in the repository")
        elif _included(rel, rules):
            s.ok(f"required: {rel}", "included")
        else:
            s.fail(f"required: {rel}", "EXCLUDED -- the build will fail")

    for rel in FORBIDDEN_IN_CONTEXT:
        if _included(rel, rules):
            s.fail(f"forbidden: {rel}", "would be sent to the daemon")
        else:
            s.ok(f"forbidden: {rel}", "excluded")


# ---------------------------------------------------------------------------
# 4. secrets
# ---------------------------------------------------------------------------
def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def check_secrets() -> None:
    s = section("Secrets")

    tracked = _git("ls-files").splitlines()
    if not tracked:
        s.warn("git", "not a git checkout, or git is unavailable -- skipped")
        return

    env_tracked = [f for f in tracked
                   if Path(f).name == ".env" or Path(f).name.startswith(".env.")
                   and not f.endswith(".example")]
    if env_tracked:
        s.fail(".env files", "TRACKED IN GIT: " + ", ".join(env_tracked))
    else:
        s.ok(".env files", "none tracked")

    # Google API keys are AIza + 35 chars. Narrow on purpose: a broad
    # "looks like a token" regex fires on every hash in the research tree.
    key_rx = re.compile(r"AIza[0-9A-Za-z_\-]{35}")
    hits = []
    for f in tracked:
        p = ROOT / f
        if not p.is_file() or p.stat().st_size > 2_000_000:
            continue
        try:
            if key_rx.search(p.read_text(encoding="utf-8", errors="ignore")):
                hits.append(f)
        except OSError:
            continue
    if hits:
        s.fail("API keys in tracked files", ", ".join(hits[:5]))
    else:
        s.ok("API keys in tracked files", "none found")

    if (ROOT / ".env").is_file():
        s.ok(".env for compose", "present -- GEMINI_API_KEY will be injected")
    else:
        s.warn(".env for compose",
               "absent -- the stack runs, the AI assistant reports unavailable")


# ---------------------------------------------------------------------------
# 5. images
# ---------------------------------------------------------------------------
DOCKERFILES = {
    "backend": "backend/Dockerfile",
    "frontend": "frontend/Dockerfile",
}


def check_dockerfiles() -> None:
    s = section("Dockerfiles")
    for name, rel in DOCKERFILES.items():
        p = ROOT / rel
        if not p.is_file():
            s.fail(name, f"{rel} missing")
            continue
        text = p.read_text(encoding="utf-8")

        # An unpinned base image makes a build unreproducible between runs.
        # `FROM base AS deps-api` refers to an earlier STAGE, not a registry
        # image, so only external references are candidates for pinning.
        stages = {m.lower() for m in re.findall(r"\bAS\s+(\S+)", text, re.I)}
        bases = [b for b in re.findall(r"^FROM\s+(\S+)", text, re.M)
                 if b.lower() not in stages]
        unpinned = [b for b in bases if ":" not in b or b.endswith(":latest")]
        if unpinned:
            s.fail(f"{name}: pinned base images", ", ".join(unpinned))
        else:
            s.ok(f"{name}: pinned base images", ", ".join(sorted(set(bases))))

        if "HEALTHCHECK" in text:
            s.ok(f"{name}: HEALTHCHECK", "present")
        else:
            s.fail(f"{name}: HEALTHCHECK", "absent")

        if name == "backend":
            # USER must precede CMD in every runtime stage, or the process runs
            # as root regardless of the USER line existing somewhere above.
            for target in ("api", "full"):
                m = re.search(rf"AS {target}\b(.*?)(?=^FROM |\Z)", text,
                              re.S | re.M)
                if not m:
                    s.fail(f"backend: target {target}", "stage not found")
                    continue
                stage = m.group(1)
                iu, ic = stage.find("USER app"), stage.find("CMD")
                if iu == -1:
                    s.fail(f"backend: target {target} non-root", "no USER app")
                elif ic != -1 and iu > ic:
                    s.fail(f"backend: target {target} non-root",
                           "USER comes after CMD")
                else:
                    s.ok(f"backend: target {target} non-root", "USER app")


# ---------------------------------------------------------------------------
# 6. toolchain
# ---------------------------------------------------------------------------
def check_toolchain() -> None:
    s = section("Toolchain")
    import shutil

    v = sys.version_info
    if (v.major, v.minor) >= (3, 11):
        s.ok("python", f"{v.major}.{v.minor}.{v.micro}")
    else:
        s.fail("python", f"{v.major}.{v.minor} -- 3.11+ required")

    for tool, required in (("docker", False), ("node", False), ("npm", False)):
        path = shutil.which(tool)
        if path:
            s.ok(tool, path)
        elif required:
            s.fail(tool, "not on PATH")
        else:
            s.warn(tool, "not on PATH -- container/frontend steps unavailable")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--skip-data", action="store_true",
                    help="skip the product data layer checks (for CI, where the "
                         "130 MB gitignored data layer does not exist)")
    args = ap.parse_args()

    check_data_layer(skip=args.skip_data)
    for fn in (check_compose, check_build_context,
               check_secrets, check_dockerfiles, check_toolchain):
        fn()

    failures = [c for s in SECTIONS for c in s.checks if c.status == FAIL]
    warnings = [c for s in SECTIONS for c in s.checks if c.status == WARN]

    if args.json:
        print(json.dumps({
            "ok": not failures,
            "failures": len(failures),
            "warnings": len(warnings),
            "sections": [{"title": s.title,
                          "checks": [vars(c) for c in s.checks]}
                         for s in SECTIONS],
        }, indent=2))
        return 1 if failures else 0

    mark = {PASS: "  ok  ", WARN: " warn ", FAIL: " FAIL "}
    for s in SECTIONS:
        print(f"\n{s.title}\n{'-' * len(s.title)}")
        for c in s.checks:
            detail = f"  {c.detail}" if c.detail else ""
            print(f"[{mark[c.status]}] {c.name}{detail}")

    total = sum(len(s.checks) for s in SECTIONS)
    print(f"\n{'=' * 60}")
    print(f"{total - len(failures) - len(warnings)} passed | "
          f"{len(warnings)} warnings | {len(failures)} failures")
    if failures:
        print("\nPREFLIGHT FAILED -- do not build:")
        for c in failures:
            print(f"  - {c.name}: {c.detail}")
        return 1
    print("PREFLIGHT PASSED -- safe to build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
