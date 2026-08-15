"""
INTEGRITY MANIFEST — a hash fingerprint of every protected artefact.

READ-ONLY. Trains nothing, moves nothing, deletes nothing. It only reads files
and writes a manifest JSON.

WHY THIS EXISTS
---------------
The project reorganisation must be provably non-destructive. Asserting "nothing
was touched" is worth very little; a SHA-256 of every protected file taken
before and after, and compared, is worth a great deal.

Run it twice:

    python scripts/08_organization/61_integrity_manifest.py before
    ... reorganise ...
    python scripts/08_organization/61_integrity_manifest.py after
    python scripts/08_organization/61_integrity_manifest.py compare

WHAT COUNTS AS PROTECTED
------------------------
  data/raw/                 the five immutable competition CSVs
  data/processed/           build output, not regenerated
  models/champion/          the shipped model artefacts
  models/experiments/       every experimental model
  predictions/final_forecast/  the deliverable
  predictions/validation/   every backtest prediction file
  experiments/registry/     the experiment ledger records
  experiments/artifacts/    result tables and diagnostics

Anything the reorganisation is ALLOWED to add (the numbered folders, new
documentation) is deliberately excluded, so a clean comparison means "no
protected artefact changed", not "nothing happened".
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "pipeline" / "config.py").exists())

PROTECTED = [
    "data/raw",
    "data/processed",
    "models/champion",
    "models/experiments",
    "predictions/final_forecast",
    "predictions/validation",
    "experiments/registry",
    "experiments/artifacts",
    "pipeline",
    "scripts",
    "docs",
    "reports",
    "MY_RESEARCH_PAPER",
]

OUT_DIR = ROOT / "08_DOCUMENTATION" / "_integrity"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build() -> dict:
    entries = {}
    for rel in PROTECTED:
        base = ROOT / rel
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            r = p.relative_to(ROOT).as_posix()
            entries[r] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    return entries


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "before"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if mode == "compare":
        b = json.loads((OUT_DIR / "manifest_before.json").read_text())["files"]
        a = json.loads((OUT_DIR / "manifest_after.json").read_text())["files"]
        missing = sorted(set(b) - set(a))
        added = sorted(set(a) - set(b))
        changed = sorted(k for k in set(b) & set(a)
                         if b[k]["sha256"] != a[k]["sha256"])
        print(f"  files before      : {len(b)}")
        print(f"  files after       : {len(a)}")
        print(f"  MISSING (deleted) : {len(missing)}")
        for k in missing[:20]:
            print(f"      - {k}")
        print(f"  CHANGED (rewritten): {len(changed)}")
        for k in changed[:20]:
            print(f"      ~ {k}")
        print(f"  added under protected roots: {len(added)}")
        for k in added[:20]:
            print(f"      + {k}")
        ok = not missing and not changed
        result = {"files_before": len(b), "files_after": len(a),
                  "missing": missing, "changed": changed, "added": added,
                  "PASS_no_protected_artefact_lost_or_modified": ok}
        (OUT_DIR / "integrity_comparison.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n  -> {'PASS' if ok else 'FAIL'}: "
              f"{'no protected artefact was deleted or modified' if ok else 'PROTECTED ARTEFACTS CHANGED'}")
        return 0 if ok else 1

    entries = build()
    total = sum(v["bytes"] for v in entries.values())
    path = OUT_DIR / f"manifest_{mode}.json"
    path.write_text(json.dumps(
        {"mode": mode, "project_root": str(ROOT), "n_files": len(entries),
         "total_bytes": total, "files": entries}, indent=2), encoding="utf-8")
    print(f"  {mode}: {len(entries)} files, {total/1e9:.2f} GB -> {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
