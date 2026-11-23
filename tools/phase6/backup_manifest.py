"""Issue 6A Phase 6 — checkpoint backup manifest and verification (read-only on the originals).

The MADQL-v2 training outputs (and every checkpoint Phase 6 depends on) live only under the
git-ignored ``results/``. This tool never copies, moves or modifies them. It:

* ``manifest``: writes a SHA-256 manifest of every file in the protected directories;
* ``verify --root R``: re-hashes the files under ``R`` (the original tree, or a backup copy that
  keeps the same relative paths) and compares them with the manifest;
* ``record --root R --label L``: verifies the copy at ``R`` (every file, and the manifest itself
  at ``R/results/dev6a/phase6/backup/manifest_sha256.json``) and appends it to
  ``backup_record.json``, which the Phase 6 freeze requires (two destinations, D12).

The copy itself is made by the user (destination of their choice, never ``models/``), e.g.::

    python tools/phase6/backup_manifest.py manifest --out results/dev6a/phase6/backup
    robocopy results\\dev6a\\phase4\\training <DEST>\\results\\dev6a\\phase4\\training /E /COPY:DT /R:1 /W:1
    python tools/phase6/backup_manifest.py verify --out results/dev6a/phase6/backup --root <DEST>
    python tools/phase6/backup_manifest.py verify --out results/dev6a/phase6/backup --root .
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: Directories whose files Phase 6 depends on (checkpoints, run provenance, selection).
PROTECTED = (
    "results/dev6a/phase4/training",  # the 9 MADQL-v2 runs (best.pt, step checkpoints, logs)
    "results/dev6a/phase4/select",  # selection.json (selects n3)
    "results/training/madql_seed1",
    "results/training/madql_seed2",
    "results/training/madql_seed3",
    "results/training/ippo_seed1",
    "results/training/ippo_seed2",
    "results/training/ippo_seed3",
    "results/training/mappo_seed1",
    "results/training/mappo_seed2",
    "results/training/mappo_seed3",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(root: Path = Path(".")) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for d in PROTECTED:
        for p in sorted((root / d).rglob("*")):
            if p.is_file():
                rel = p.relative_to(root).as_posix()
                files[rel] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    return {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "directories": list(PROTECTED),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(v["bytes"] for v in files.values()),
        "madql_v2_best": {
            f"{c}_seed{k}": files[key]["sha256"]
            for c in ("n3", "t8", "n3t8")
            for k in (1, 2, 3)
            if (key := f"results/dev6a/phase4/training/{c}_seed{k}/best.pt") in files
        },
    }


def verify(man: dict[str, Any], root: Path) -> dict[str, Any]:
    missing, changed = [], []
    for rel, v in man["files"].items():
        p = root / rel
        if not p.is_file():
            missing.append(rel)
        elif p.stat().st_size != v["bytes"] or sha256(p) != v["sha256"]:
            changed.append(rel)
    return {"root": str(root), "checked": len(man["files"]), "missing": missing,
            "changed": changed, "ok": not missing and not changed}  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("manifest", "verify", "record"))
    ap.add_argument("--label", default="")
    ap.add_argument("--out", type=Path, default=Path("results/dev6a/phase6/backup"))
    ap.add_argument("--root", type=Path, default=Path("."))
    args = ap.parse_args(argv)
    path = args.out / "manifest_sha256.json"
    if args.cmd == "manifest":
        if path.exists():
            raise SystemExit(f"{path} exists; a manifest is never overwritten")
        man = manifest()
        args.out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(man, indent=1, sort_keys=True), "utf-8")
        print(json.dumps({k: man[k] for k in ("file_count", "total_bytes")}))
        return 0
    res = verify(json.loads(path.read_text("utf-8")), args.root)
    print(json.dumps({k: res[k] for k in ("root", "checked", "ok")}),
          len(res["missing"]), len(res["changed"]))  # fmt: skip
    if args.cmd == "record":
        if not res["ok"]:
            raise SystemExit("not recorded: the copy does not match the manifest")
        copied = args.root / path
        if not copied.is_file() or sha256(copied) != sha256(path):
            raise SystemExit("not recorded: the manifest itself is not in the copy")
        rec_path = args.out / "backup_record.json"
        rec = json.loads(rec_path.read_text("utf-8")) if rec_path.exists() else {"destinations": []}
        if any(d["path"] == str(args.root.resolve()) for d in rec["destinations"]):
            raise SystemExit("this destination is already recorded")
        rec["destinations"].append({
            "path": str(args.root.resolve()),
            "label": args.label,
            "recorded_at": dt.datetime.now().isoformat(timespec="seconds"),
            "manifest_sha256": sha256(path),
            "verified": {"checked": res["checked"], "ok": res["ok"],
                         "missing": len(res["missing"]), "changed": len(res["changed"])},
        })  # fmt: skip
        rec_path.write_text(json.dumps(rec, indent=1), "utf-8")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
