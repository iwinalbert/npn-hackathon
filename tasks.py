
from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
PY = sys.executable


def _run(cmd: list[str], cwd: Path = ROOT) -> int:
    print(f"$ {' '.join(str(c) for c in cmd)}   (in {cwd})", flush=True)
    return subprocess.call(cmd, cwd=str(cwd))


def build_db() -> int:
    return _run([PY, str(BACKEND / "scripts" / "build_product_db.py")])


def _whats_on(port: int) -> str | None:
    import json
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/openapi.json", timeout=2) as r:
            spec = json.load(r)
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None

    paths = spec.get("paths", {})
    genai = [p for p in paths if "/genai" in p]
    detail = f"{len(paths)} routes"
    if not genai:
        detail += ", NO /genai routes - this build predates the AI assistant"
    return detail


def _listener_pids(port: int) -> list[int]:
    pids: list[int] = []
    if platform.system() == "Windows":
        out = subprocess.run(["netstat", "-ano"], capture_output=True,
                             text=True).stdout
        for line in out.splitlines():
            parts = line.split()
            if (len(parts) >= 5 and parts[0] == "TCP"
                    and parts[1].endswith(f":{port}") and parts[3] == "LISTENING"):
                pids.append(int(parts[4]))
        for pid in list(pids):
            kids = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-CimInstance Win32_Process -Filter 'ParentProcessId={pid}'"
                 f" | Select-Object -ExpandProperty ProcessId) -join ' '"],
                capture_output=True, text=True).stdout.split()
            pids += [int(k) for k in kids if k.isdigit()]
    else:
        out = subprocess.run(["lsof", "-ti", f"tcp:{port}"],
                             capture_output=True, text=True).stdout
        pids += [int(p) for p in out.split() if p.isdigit()]
    return sorted(set(pids))


def stop_api() -> int:
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

    if not _whats_on(port) and not _listener_pids(port):
        print(f"nothing is listening on port {port}")
        return 0

    for pid in _listener_pids(port):
        if platform.system() == "Windows":
            rc = subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                                capture_output=True, text=True)
            ok = rc.returncode == 0
        else:
            import signal
            try:
                os.kill(pid, signal.SIGKILL)
                ok = True
            except OSError:
                ok = False
        print(f"  pid {pid}: {'stopped' if ok else 'already gone'}")

    time.sleep(1.5)
    if _listener_pids(port) or _whats_on(port):
        print(f"port {port} is STILL occupied — run as Administrator, or reboot")
        return 1
    print(f"port {port} is free")
    return 0


def api() -> int:
    port = sys.argv[2] if len(sys.argv) > 2 else "8000"

    existing = _whats_on(int(port))
    if existing:
        bar = "!" * 70
        print(f"\n{bar}")
        print(f"Port {port} is ALREADY serving an API: {existing}")
        print("Free it, then start again:")
        print(f"    python tasks.py stop-api {port}")
        print(f"    python tasks.py api {'' if port == '8000' else port}".rstrip())
        print(f"{bar}\n")
        return 1

    return _run([PY, "-m", "uvicorn", "app.main:app", "--reload",
                 "--port", str(port)], cwd=BACKEND)


def test() -> int:
    return _run([PY, "-m", "pytest"], cwd=BACKEND)


def genai_check() -> int:
    return _run([PY, "-m", "pytest", "-m", "live", "-v"], cwd=BACKEND)


def _overlays() -> list[str]:
    files = ["-f", "docker-compose.yml"]
    if "--inference" in sys.argv:
        files += ["-f", "docker-compose.inference.yml"]
    if "--prod" in sys.argv:
        files += ["-f", "infra/compose/docker-compose.prod.yml"]
    return files


def _compose(*args: str) -> int:
    import shutil
    if shutil.which("docker") is None:
        bar = "!" * 70
        print(f"\n{bar}")
        print("Docker is not installed or not on PATH.")
        print("  Install Docker Desktop (Windows/macOS) or docker + the compose")
        print("  plugin (Linux), then run this again.")
        print("  Development without Docker is unaffected:")
        print("      python tasks.py api        python tasks.py ui")
        print(f"{bar}\n")
        return 1

    return _run(["docker", "compose", *_overlays(), *args])


def docker_config() -> int:
    return _compose("config")


def docker_build() -> int:
    return _compose("build")


def docker_up() -> int:
    if preflight() != 0:
        print("\npreflight failed - not starting the stack "
              "(see the failures above)")
        return 1

    rc = _compose("up", "-d", "--build")
    if rc == 0:
        prod = "--prod" in sys.argv
        print("\n  app  -> http://localhost:8080")
        if prod:
            print("  api  -> internal only (--prod removes the published port)")
        else:
            print("  api  -> http://localhost:8000/docs")
        print("  logs -> python tasks.py docker-logs")
        print(f"  next -> python tasks.py smoke{' --prod' if prod else ''}")
    return rc


def docker_down() -> int:
    return _compose("down")


def docker_logs() -> int:
    return _compose("logs", "-f", "--tail", "100")


def docker_ps() -> int:
    return _compose("ps")


def preflight() -> int:
    cmd = [PY, str(ROOT / "infra" / "scripts" / "preflight.py")]
    if "--skip-data" in sys.argv:
        cmd.append("--skip-data")
    if "--json" in sys.argv:
        cmd.append("--json")
    return _run(cmd)


def smoke() -> int:
    cmd = [PY, str(ROOT / "infra" / "scripts" / "smoke_test.py")]

    if "--api" in sys.argv:
        cmd += ["--api-url", sys.argv[sys.argv.index("--api") + 1]]
    elif "--prod" in sys.argv:
        cmd += ["--api-url", "http://localhost:8080"]

    for flag in ("--skip-web", "--skip-api", "--json"):
        if flag in sys.argv:
            cmd.append(flag)
    if "--wait" in sys.argv:
        cmd += ["--wait", sys.argv[sys.argv.index("--wait") + 1]]
    return _run(cmd)


def verify_integrity() -> int:
    script = (ROOT / "research" / "scripts" / "08_organization"
              / "61_integrity_manifest.py")
    rc = _run([PY, str(script), "after"])
    return rc or _run([PY, str(script), "compare"])


def openapi() -> int:
    sys.path.insert(0, str(BACKEND))
    import json

    from app.main import app                                   # noqa: PLC0415
    out = BACKEND / "openapi.json"
    out.write_text(json.dumps(app.openapi(), indent=2), encoding="utf-8")
    paths = len(app.openapi()["paths"])
    print(f"wrote {out.relative_to(ROOT)}  ({paths} paths)")
    return 0


def _npm(*args: str) -> int:
    cmd = ["npm", *args]
    if platform.system() == "Windows":
        print(f"$ {' '.join(cmd)}   (in {FRONTEND})", flush=True)
        return subprocess.call(" ".join(cmd), cwd=str(FRONTEND), shell=True)
    return _run(cmd, cwd=FRONTEND)


def ui() -> int:
    if len(sys.argv) > 2:
        target = f"http://127.0.0.1:{int(sys.argv[2])}"
        os.environ["VITE_DEV_API_TARGET"] = target
        print(f"proxying /api to {target}")
    return _npm("run", "dev")


def ui_build() -> int:
    return _npm("run", "build")


def ui_test() -> int:
    return _npm("run", "test")


def ui_install() -> int:
    return _npm("ci")


def verify_all() -> int:
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
    "stop-api": stop_api,
    "test": test,
    "genai-check": genai_check,
    "ui": ui,
    "ui-install": ui_install,
    "ui-build": ui_build,
    "ui-test": ui_test,
    "docker-config": docker_config,
    "docker-build": docker_build,
    "docker-up": docker_up,
    "docker-down": docker_down,
    "docker-logs": docker_logs,
    "docker-ps": docker_ps,
    "preflight": preflight,
    "smoke": smoke,
    "verify-all": verify_all,
    "verify-integrity": verify_integrity,
    "openapi": openapi,
    "clean-db": clean_db,
}


def help_() -> int:
    print((__doc__ or "tasks.py -- project task runner").strip())
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
