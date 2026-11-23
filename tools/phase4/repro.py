"""Issue 6A Phase 4 — reproduction check of the MADQL trainer with the option off.

Runs the historical configuration (``configs/training/madql.toml``, run seed ``--seed``, the full
1 M-step schedule) with the current code until the first validation (50 000 environment steps)
has been written, then stops, and compares the outputs with the stored Issue #5 run (read-only):

* ``step_00050000.pt`` (inference checkpoint incl. meta) and ``step_00050000.train.pt`` — SHA-256
  and every tensor;
* the 50 k validation record (summary, selection key, and every field of every Issue #5 episode
  record; the current evaluation adds the Issue #6 fields, which the Issue #5 records predate)
  and every training-log line up to 50 k (wall-clock fields excluded);
* ``config.json``'s configuration and hash.

Usage::

    python tools/phase4/repro.py --seed 1 --out results/dev6a/phase4/repro
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

STOP_AT = 50_000
WALL = ("wall_s", "validation_s", "train_steps_per_s", "elapsed_s")


class _StopError(Exception):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lines(path: Path, upto: int) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text("utf-8").splitlines():
        r = json.loads(line)
        if r.get("env_steps", 0) > upto:
            break
        out.append({k: v for k, v in r.items() if k not in WALL})
    return out


def _validation_equal(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Summary and key equal; every non-timing field of every old record present and equal."""
    recs_ok = len(old["records"]) == len(new["records"]) and all(
        all(k in b and a[k] == b[k] for k in a if k != "wall_s")
        for a, b in zip(old["records"], new["records"], strict=True)
    )
    added = sorted(set(new["records"][0]) - set(old["records"][0])) if new["records"] else []
    return {
        "summary_equal": old["summary"] == new["summary"],
        "key_equal": old["key"] == new["key"],
        "records": len(old["records"]),
        "old_fields_equal": recs_ok,
        "fields_added_by_current_evaluation": added,
        "equal": old["summary"] == new["summary"] and old["key"] == new["key"] and recs_ok,
    }


def main(argv: Sequence[str] | None = None) -> int:
    from robocoord.marl.trainer import load_training_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args(argv)
    hist = Path(f"results/training/madql_seed{args.seed}")
    out = args.out / f"madql_seed{args.seed}"
    cfg = load_training_config("configs/training/madql.toml", run_seed=args.seed)
    if not args.compare_only:
        _train_until(cfg, out)
    return compare(hist, out, args.seed)


def _train_until(cfg: Any, out: Path) -> None:
    from robocoord.marl.trainer import Trainer

    trainer = Trainer(cfg, out)
    original = trainer.validate

    def validate(rrt: bool) -> Any:
        entry = original(rrt)
        if trainer.env_steps >= STOP_AT:
            raise _StopError
        return entry

    trainer.validate = validate  # type: ignore[method-assign]
    with contextlib.suppress(_StopError):
        trainer.train()
    trainer._log.close()
    trainer._val.close()


def compare(hist: Path, out: Path, seed: int) -> int:
    import torch

    res: dict[str, Any] = {"seed": seed, "stop_at": STOP_AT}
    for name in (f"step_{STOP_AT:08d}.pt", f"step_{STOP_AT:08d}.train.pt"):
        a, b = hist / "checkpoints" / name, out / "checkpoints" / name
        sa, sb = torch.load(a, weights_only=False), torch.load(b, weights_only=False)
        tens = _tensors_equal(sa, sb)
        res[name] = {"sha256_equal": sha256(a) == sha256(b), "tensors_equal": tens}
    c_old = json.loads((hist / "config.json").read_text("utf-8"))
    c_new = json.loads((out / "config.json").read_text("utf-8"))
    res["config_equal"] = c_old["config"] == c_new["config"]
    res["config_hash_equal"] = c_old["hash"] == c_new["hash"]
    res["train_log_equal"] = _lines(hist / "train_log.jsonl", STOP_AT) == _lines(
        out / "train_log.jsonl", STOP_AT
    )
    v_old = json.loads((hist / "validation.jsonl").read_text("utf-8").splitlines()[0])
    v_new = json.loads((out / "validation.jsonl").read_text("utf-8").splitlines()[0])
    res["validation"] = _validation_equal(v_old, v_new)
    res["identical"] = bool(
        res["config_hash_equal"]
        and res["train_log_equal"]
        and res["validation"]["equal"]
        and all(
            res[n]["tensors_equal"]
            for n in (f"step_{STOP_AT:08d}.pt", f"step_{STOP_AT:08d}.train.pt")
        )
    )
    (out / "repro.json").write_text(json.dumps(res, indent=1), "utf-8")
    print(json.dumps(res))
    return 0 if res["identical"] else 1


def _tensors_equal(a: Any, b: Any) -> bool:
    import torch

    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and set(a) == set(b)
            and all(_tensors_equal(a[k], b[k]) for k in a if k != "meta")
        )
    if isinstance(a, (list, tuple)):
        return (
            isinstance(b, (list, tuple))
            and len(a) == len(b)
            and all(_tensors_equal(x, y) for x, y in zip(a, b, strict=True))
        )
    return bool(a == b)


if __name__ == "__main__":
    sys.exit(main())
