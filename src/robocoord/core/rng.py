"""Deterministic random-stream derivation (ADR-8).

One master seed is expanded into named child streams in a fixed, documented order. Changing how
much randomness one component consumes never shifts another component's sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Fixed order of named streams. Append new names at the end only (never reorder).
STREAM_NAMES: tuple[str, ...] = ("map", "obstacles", "tasks", "planner", "policy", "torch")


@dataclass(frozen=True, slots=True)
class SeedStreams:
    """Child seed sequences for every named stream of one run."""

    master_seed: int
    sequences: dict[str, np.random.SeedSequence]

    def generator(self, name: str) -> np.random.Generator:
        """A fresh generator for stream ``name`` (always starts at the same state)."""
        return np.random.default_rng(self._seq(name))

    def generators(self, name: str, count: int) -> list[np.random.Generator]:
        """``count`` independent generators derived from stream ``name`` (e.g. one per obstacle)."""
        children = np.random.SeedSequence(
            entropy=self._seq(name).entropy, spawn_key=(*self._seq(name).spawn_key, 1_000_003)
        ).spawn(count)
        return [np.random.default_rng(c) for c in children]

    def int_seed(self, name: str) -> int:
        """A 63-bit integer seed for stream ``name`` (e.g. for ``torch.manual_seed``)."""
        return int(self._seq(name).generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))

    def _seq(self, name: str) -> np.random.SeedSequence:
        try:
            return self.sequences[name]
        except KeyError:
            raise KeyError(f"unknown stream {name!r}; known: {STREAM_NAMES}") from None


def make_streams(master_seed: int) -> SeedStreams:
    """Derive the named streams for ``master_seed`` (non-negative integer)."""
    if master_seed < 0:
        raise ValueError("master seed must be non-negative")
    root = np.random.SeedSequence(master_seed)
    children = root.spawn(len(STREAM_NAMES))
    return SeedStreams(master_seed, dict(zip(STREAM_NAMES, children, strict=True)))
