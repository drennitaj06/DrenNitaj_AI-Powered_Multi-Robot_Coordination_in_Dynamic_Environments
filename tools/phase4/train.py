"""Issue 6A Phase 4 — MADQL-v2 candidate training (protocol: the Phase 4 document §3).

Every run is the historical Issue #5 configuration (``configs/training/madql.toml``: 1 M
environment steps, 4 copies, sampler seed 0, the approved curriculum, A\\*, obs-v1, the model §7
reward) with only the candidate's learning-rule fields changed, and run seed 1, 2 or 3 (the
historical seeds, so each candidate run starts from the same initial network, exploration stream
and scenario order as its historical counterpart). Validation and the checkpoint rule are the
historical ones (block A, greedy, completion then discounted base return → ``best.pt``); only the
number of validation worker processes differs, which does not affect learning (results keep the
task order; the reproduction check ``repro.py`` covers the unchanged path).

New outputs only: ``results/dev6a/phase4/training/<candidate>_seed<k>/`` plus ``manifest.json``.
Existing checkpoints are never read or written by this tool.

Usage::

    python tools/phase4/train.py manifest --out results/dev6a/phase4/training
    python tools/phase4/train.py run --cand n3 --seed 1 --out results/dev6a/phase4/training
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

BASE = Path("configs/training/madql.toml")
#: Candidates (fixed in the Phase 4 document §3.2 before any training): the historical
#: configuration is the cell (n_step 1, target_period 2 000).
CANDIDATES: dict[str, dict[str, int]] = {
    "n3": {"n_step": 3, "target_period": 2_000},
    "t8": {"n_step": 1, "target_period": 8_000},
    "n3t8": {"n_step": 3, "target_period": 8_000},
}
SEEDS = (1, 2, 3)
VALIDATION_WORKERS = 2
CODE = (
    "src/robocoord/marl/madql.py",
    "src/robocoord/marl/replay.py",
    "src/robocoord/marl/trainer.py",
    "src/robocoord/marl/env.py",
    "src/robocoord/marl/reward.py",
    "src/robocoord/marl/observation.py",
    "src/robocoord/marl/scenarios.py",
    "src/robocoord/marl/evaluation.py",
    "src/robocoord/marl/networks.py",
    "src/robocoord/marl/checkpoint.py",
    "configs/training/madql.toml",
    "tools/phase4/train.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config(cand: str, seed: int) -> Any:
    from robocoord.marl.trainer import load_training_config

    cfg = load_training_config(BASE, run_seed=seed)
    fields = CANDIDATES[cand]
    return replace(
        cfg,
        name=f"madql-v2-{cand}",
        madql=replace(cfg.madql, n_step=fields["n_step"], target_period=fields["target_period"]),
        validation=replace(cfg.validation, workers=VALIDATION_WORKERS),
    )


def run_dir(out: Path, cand: str, seed: int) -> Path:
    return out / f"{cand}_seed{seed}"


def manifest(out: Path) -> dict[str, Any]:
    import platform

    import numpy
    import torch

    out.mkdir(parents=True, exist_ok=True)
    m = {
        "written_at": dt.datetime.now().isoformat(timespec="seconds"),
        "base_config": str(BASE),
        "candidates": CANDIDATES,
        "seeds": list(SEEDS),
        "validation_workers": VALIDATION_WORKERS,
        "runs": {
            f"{c}_seed{s}": {"config": config(c, s).to_dict(), "out": str(run_dir(out, c, s))}
            for c in CANDIDATES
            for s in SEEDS
        },
        "code_sha256": {c: sha256(Path(c)) for c in CODE},
        "environment": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }
    path = out / "manifest.json"
    if path.exists():
        old: dict[str, Any] = json.loads(path.read_text("utf-8"))
        if old["code_sha256"] != m["code_sha256"] or old["runs"] != m["runs"]:
            raise SystemExit("manifest.json exists with a different code or configuration")
        return old
    path.write_text(json.dumps(m, indent=1, sort_keys=True), "utf-8")
    return m


def run(out: Path, cand: str, seed: int) -> dict[str, Any]:
    from robocoord.marl.trainer import train

    m = json.loads((out / "manifest.json").read_text("utf-8"))
    now = {c: sha256(Path(c)) for c in CODE}
    if now != m["code_sha256"]:
        raise SystemExit(
            f"code changed since the manifest: {[c for c in CODE if now[c] != m['code_sha256'][c]]}"
        )
    cfg = config(cand, seed)
    if cfg.to_dict() != m["runs"][f"{cand}_seed{seed}"]["config"]:
        raise SystemExit("configuration differs from the manifest")
    summary = train(cfg, run_dir(out, cand, seed))
    best = run_dir(out, cand, seed) / "best.pt"
    summary["best_sha256"] = sha256(best)
    (run_dir(out, cand, seed) / "phase4_run.json").write_text(
        json.dumps(summary, indent=1), "utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("manifest", "run"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cand", choices=tuple(CANDIDATES))
    ap.add_argument("--seed", type=int, choices=SEEDS)
    args = ap.parse_args(argv)
    if args.cmd == "manifest":
        m = manifest(args.out)
        print(json.dumps({"runs": len(m["runs"]), "code": len(m["code_sha256"])}))
        return 0
    if args.cand is None or args.seed is None:
        raise SystemExit("--cand and --seed are required")
    print(json.dumps(run(args.out, args.cand, args.seed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
