"""
Cross-platform task runner.

    python tasks.py <command>

`make` is not installed on the Windows machine this project is developed and
demonstrated on, so the Makefile alone would strand the most important user.
This runner exposes the same commands everywhere with no extra tooling.

    python tasks.py help
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "06_BACKEND"
FRONTEND = ROOT / "07_FRONTEND"
PY = sys.executable


def _run(cmd: list[str], cwd: Path = ROOT) -> int:
    print(f"$ {' '.join(str(c) for c in cmd)}   (in {cwd})", flush=True)
    return subprocess.call(cmd, cwd=str(cwd))


def build_db() -> int:
    """Build the product database from the frozen research artefacts."""
    return _run([PY, str(BACKEND / "scripts" / "build_product_db.py")])


def api() -> int:
    """Run the API on http://localhost:8000 (interactive docs at /docs)."""
    return _run([PY, "-m", "uvicorn", "app.main:app", "--reload",
                 "--port", "8000"], cwd=BACKEND)


def test() -> int:
    """Run the backend test suite."""
    return _run([PY, "-m", "pytest"], cwd=BACKEND)


def verify_integrity() -> int:
    """Prove that no protected research artefact has changed."""
    script = ROOT / "scripts" / "08_organization" / "61_integrity_manifest.py"
    rc = _run([PY, str(script), "after"])
    return rc or _run([PY, str(script), "compare"])


def openapi() -> int:
    """Write the OpenAPI schema to 06_BACKEND/openapi.json."""
    sys.path.insert(0, str(BACKEND))
    import json

    from app.main import app                                   # noqa: PLC0415
    out = BACKEND / "openapi.json"
    out.write_text(json.dumps(app.openapi(), indent=2), encoding="utf-8")
    paths = len(app.openapi()["paths"])
    print(f"wrote {out.relative_to(ROOT)}  ({paths} paths)")
    return 0


def _npm(*args: str) -> int:
    """Run npm in the frontend directory. Uses shell=True on Windows because
    npm ships as npm.cmd there and is not directly executable otherwise."""
    import platform
    cmd = ["npm", *args]
    if platform.system() == "Windows":
        print(f"$ {' '.join(cmd)}   (in {FRONTEND})", flush=True)
        return subprocess.call(" ".join(cmd), cwd=str(FRONTEND), shell=True)
    return _run(cmd, cwd=FRONTEND)


def ui() -> int:
    """Run the frontend dev server on http://localhost:5173."""
    return _npm("run", "dev")


def ui_build() -> int:
    """Build the production frontend bundle into 07_FRONTEND/dist."""
    return _npm("run", "build")


def ui_test() -> int:
    """Run the frontend test suite."""
    return _npm("run", "test")


def ui_install() -> int:
    """Install frontend dependencies."""
    return _npm("ci")


def verify_all() -> int:
    """Run every check: backend tests, frontend tests, build, integrity."""
    steps = [
        ("backend tests", test),
        ("frontend tests", ui_test),
        ("frontend build", ui_build),
        ("artefact integrity", verify_integrity),
    ]
    failures = []
    bar = "=" * 60
    for name, fn in steps:
        print(f"\n{bar}\n{name}\n{bar}", flush=True)
        if fn() != 0:
            failures.append(name)
    print(f"\n{bar}")
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


def clean_db() -> int:
    """Delete the product database. It is always rebuildable."""
    removed = 0
    for p in (BACKEND / "data").glob("product.duckdb*"):
        p.unlink()
        print(f"removed {p.name}")
        removed += 1
    if not removed:
        print("nothing to remove")
    return 0


COMMANDS = {
    "build-db": build_db,
    "api": api,
    "test": test,
    "ui": ui,
    "ui-install": ui_install,
    "ui-build": ui_build,
    "ui-test": ui_test,
    "verify-all": verify_all,
    "verify-integrity": verify_integrity,
    "openapi": openapi,
    "clean-db": clean_db,
}


def help_() -> int:
    print(__doc__.strip())
    print("\nCommands:")
    width = max(len(k) for k in COMMANDS)
    for name, fn in COMMANDS.items():
        print(f"  {name:<{width}}  {(fn.__doc__ or '').strip()}")
    return 0


COMMANDS["help"] = help_


def main() -> int:
    if len(sys.argv) < 2:
        return help_()
    cmd = sys.argv[1]
    fn = COMMANDS.get(cmd)
    if fn is None:
        print(f"unknown command '{cmd}'\n")
        return help_() or 1
    return fn()


if __name__ == "__main__":
    sys.exit(main())
