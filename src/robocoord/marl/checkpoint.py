"""Checkpoints (ADR-9; test X-7). Requires PyTorch.

* **Training checkpoints** (``*.train.pt``): online and target networks, optimiser state and the
  update counter. They stay in ``results/`` (git-ignored).
* **Inference checkpoints** (``*.pt`` + ``*.json`` sidecar): the policy network ``state_dict``
  only — the MADQL Q-network (``mlp-q-v1``) or the PPO actor (``mlp-pi-v1``, Issue #6; masked
  logits) — plus a sidecar with the ADR-9 fields (algorithm and variant; model, observation and
  reward specifications and versions; config hash and full config; seeds; environment steps;
  validation score; experiment id and code version) and the file's SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import robocoord
from robocoord.marl.config import N_ACTIONS, ObsParams
from robocoord.marl.networks import build_mlp

OBS_SPEC_VERSION = "obs-v1"  # model §6.1, 7×7×7 + 13
REWARD_SPEC_VERSION = "reward-v1"  # model §7 base + potential shaping + team mixing
MODEL_SPEC_VERSION = "mlp-q-v1"  # MADQL Q-network (Issue #5): masked arg-max of Q-values
ACTOR_SPEC_VERSION = "mlp-pi-v1"  # PPO actor (Issue #6): masked categorical over logits
#: Supported inference model specifications (additive table; both are 356 → hidden → 5 MLPs and
#: their greedy action is the masked arg-max of the outputs).
MODEL_SPECS: dict[str, str] = {MODEL_SPEC_VERSION: "q-values", ACTOR_SPEC_VERSION: "logits"}
SIDECAR_FIELDS = (
    "algorithm",
    "variant",
    "model",
    "observation",
    "reward",
    "config_hash",
    "config",
    "seeds",
    "env_steps",
    "validation",
    "experiment_id",
    "code_version",
    "sha256",
)


def config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_inference(
    path: str | Path,
    net: nn.Module,
    *,
    hidden: tuple[int, ...],
    obs: ObsParams,
    meta: dict[str, Any],
    model_version: str = MODEL_SPEC_VERSION,
) -> Path:
    """Write ``path`` (weights) and its ``.json`` sidecar; returns the ``.pt`` path."""
    if model_version not in MODEL_SPECS:
        raise ValueError(f"unknown model specification {model_version!r}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().clone() for k, v in net.state_dict().items()}, p)
    sidecar = {
        **meta,
        "model": {
            "version": model_version,
            "input": obs.dim,
            "hidden": list(hidden),
            "output": N_ACTIONS,
            "activation": "relu",
            "parameters": sum(v.numel() for v in net.state_dict().values()),
        },
        "observation": {
            "version": OBS_SPEC_VERSION,
            "radius": obs.radius,
            "path_cells": obs.path_cells,
            "max_priority": obs.max_priority,
            "dim": obs.dim,
        },
        "code_version": robocoord.__version__,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "file_bytes": p.stat().st_size,
        "sha256": file_sha256(p),
    }
    sidecar.setdefault("reward", {"version": REWARD_SPEC_VERSION})
    p.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True), "utf-8")
    return p


def load_inference(path: str | Path) -> tuple[nn.Sequential, dict[str, Any], ObsParams]:
    """Load weights and sidecar; verifies the SHA-256 and the model specification."""
    p = Path(path)
    meta = json.loads(p.with_suffix(".json").read_text("utf-8"))
    if file_sha256(p) != meta["sha256"]:
        raise ValueError(f"{p}: SHA-256 does not match its sidecar")
    model = meta["model"]
    if model["version"] not in MODEL_SPECS or meta["observation"]["version"] != OBS_SPEC_VERSION:
        raise ValueError(f"{p}: unsupported model or observation specification")
    o = meta["observation"]
    obs = ObsParams(radius=o["radius"], path_cells=o["path_cells"], max_priority=o["max_priority"])
    net = build_mlp(obs.dim, tuple(model["hidden"]), model["output"])
    net.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
    net.eval()
    return net, meta, obs


def save_training(path: str | Path, state: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, p)
    return p


def load_training(path: str | Path) -> dict[str, Any]:
    state: dict[str, Any] = torch.load(Path(path), map_location="cpu", weights_only=False)
    return state
