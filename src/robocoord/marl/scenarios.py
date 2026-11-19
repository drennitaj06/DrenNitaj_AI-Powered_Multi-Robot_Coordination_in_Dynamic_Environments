"""Scenario distributions, the curriculum, and the seed blocks of Issue #5. NumPy only.

Every scenario comes from the existing generator
:func:`robocoord.experiments.allocation_runs.e5_scenario` (a generator name from Issue #4, not an
experiment label). Seed blocks (``05`` §B.1):

* training scenario seeds 0–9 999, consumed in one fixed pseudo-random order that is the same
  for every training run (``training_seed_order``);
* validation block A 10 000–10 019: periodic validation and checkpoint selection;
* validation block B 10 020–10 099: development comparisons only, never selection or tuning;
* 20 000–20 999: reserved for the official E5/E6 evaluation of Issue #8; never used here.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from robocoord.core.errors import ConfigError
from robocoord.experiments.allocation_runs import e5_scenario
from robocoord.scenario.spec import ScenarioSpec

TRAIN_SEEDS = range(0, 10_000)
BLOCK_A = range(10_000, 10_020)
BLOCK_B = range(10_020, 10_100)
RESERVED_TEST = range(20_000, 21_000)
#: RRT* subset of each block: the first 10 seeds of block A, the first 20 of block B.
BLOCK_A_RRT = range(10_000, 10_010)
BLOCK_B_RRT = range(10_020, 10_040)

_BLOCKS = {"train": TRAIN_SEEDS, "A": BLOCK_A, "B": BLOCK_B}


class SeedBlockError(ValueError):
    """A seed was requested outside the block that the operation is allowed to use."""


def check_seeds(seeds: Iterable[int], block: str) -> list[int]:
    """Return the seeds if every one lies in ``block`` ("train", "A" or "B"), else raise."""
    allowed = _BLOCKS[block]
    out = [int(s) for s in seeds]
    bad = [s for s in out if s not in allowed]
    if bad:
        raise SeedBlockError(
            f"seeds {bad[:5]} are outside block {block} ({allowed.start}–{allowed.stop - 1})"
        )
    return out


def training_seed_order(sampler_seed: int = 0) -> list[int]:
    """The fixed order in which training scenario seeds are consumed (identical across runs)."""
    rng = np.random.default_rng(np.random.SeedSequence(entropy=sampler_seed, spawn_key=(5, 5)))
    return [int(s) for s in rng.permutation(len(TRAIN_SEEDS))]


@dataclass(frozen=True, slots=True)
class ScenarioDist:
    """A distribution over ``e5_scenario`` instances. Values in the tuples are drawn uniformly,
    so a repeated value carries more weight (e.g. ``robots = (4, 5, 6, 7, 8, 8)``)."""

    name: str
    dynamics: str
    robots: tuple[int, ...]
    sizes: tuple[int, ...]
    densities: tuple[float, ...]
    obstacles: tuple[int, ...]
    max_steps: int
    tasks_per_robot: int = 2
    arrival_rate: float = 0.05
    arrivals_until: int = 200
    n_events: int = 6
    event_duration: int = 20

    def __post_init__(self) -> None:
        if self.dynamics not in ("static", "obstacles", "full"):
            raise ConfigError(f"scenario.{self.name}.dynamics", f"unknown {self.dynamics!r}")
        for field_name in ("robots", "sizes", "densities", "obstacles"):
            if not getattr(self, field_name):
                raise ConfigError(f"scenario.{self.name}.{field_name}", "must not be empty")
        if self.max_steps < 1:
            raise ConfigError(f"scenario.{self.name}.max_steps", "must be >= 1")

    def parameters(self, seed: int) -> dict[str, float | int]:
        """The values drawn for ``seed``: a function of ``(name, seed)`` only."""
        rng = np.random.default_rng([zlib.crc32(self.name.encode()), seed])
        return {
            "robots": int(rng.choice(self.robots)),
            "size": int(rng.choice(self.sizes)),
            "density": float(rng.choice(self.densities)),
            "obstacles": int(rng.choice(self.obstacles)),
        }

    def sample(self, seed: int) -> ScenarioSpec:
        p = self.parameters(seed)
        return e5_scenario(
            seed,
            self.dynamics,
            size=int(p["size"]),
            density=float(p["density"]),
            robots=int(p["robots"]),
            tasks_per_robot=self.tasks_per_robot,
            obstacles=int(p["obstacles"]),
            max_steps=self.max_steps,
            arrival_rate=self.arrival_rate,
            arrivals_until=self.arrivals_until,
            n_events=self.n_events,
            event_duration=self.event_duration,
        )


@dataclass(frozen=True, slots=True)
class Stage:
    """A curriculum stage: the share of the step budget and its scenario distribution."""

    fraction: float
    dist: ScenarioDist


DENSITIES = (0.10, 0.15, 0.20)

#: Approved curriculum (design R7): static 2–4 robots → moving obstacles → full dynamics 4–8.
CURRICULUM: tuple[Stage, ...] = (
    Stage(0.2, ScenarioDist("stage1", "static", (2, 3, 4), (16,), DENSITIES, (0,), 200)),
    Stage(0.3, ScenarioDist("stage2", "obstacles", (2, 3, 4), (16, 24), DENSITIES, (2, 4), 300)),
    Stage(
        0.5,
        ScenarioDist("stage3", "full", (4, 5, 6, 7, 8, 8), (24, 32, 32), DENSITIES, (4, 8), 400),
    ),
)

#: Exact validation / development-comparison configurations (design D): 32×32, density 0.15,
#: 8 robots, 16 scheduled tasks, horizon 400; 8 obstacles for the dynamic levels; arrivals at
#: rate 0.05 until step 200 and 6 map events of 20 steps in the full level.
LEVELS: dict[str, ScenarioDist] = {
    lvl: ScenarioDist(f"eval-{lvl}", lvl, (8,), (32,), (0.15,), (0 if lvl == "static" else 8,), 400)
    for lvl in ("static", "obstacles", "full")
}


def stage_at(step: int, total: int, curriculum: Sequence[Stage] = CURRICULUM) -> int:
    """Index of the curriculum stage active at environment step ``step`` of ``total``."""
    acc = 0.0
    for k, st in enumerate(curriculum):
        acc += st.fraction
        if step < acc * total:
            return k
    return len(curriculum) - 1
