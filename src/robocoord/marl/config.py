"""Typed parameters of the MARL environment (model §6–7; algorithms §6, §8)."""

from __future__ import annotations

from dataclasses import dataclass, field

from robocoord.core.errors import ConfigError


@dataclass(frozen=True, slots=True)
class ObsParams:
    """Egocentric observation (model §6.1).

    * ``radius`` — Chebyshev window radius ``r`` (window ``(2r+1)²``; default 3, i.e. 7×7);
    * ``path_cells`` — ``L_o``, the number of own reference-path cells after the robot's cell;
    * ``max_priority`` — ``P``, the normaliser of priority levels (``π / P``).
    """

    radius: int = 3
    path_cells: int = 6
    max_priority: int = 3

    def __post_init__(self) -> None:
        if self.radius < 1:
            raise ConfigError("obs.radius", "must be >= 1")
        if self.path_cells < 1:
            raise ConfigError("obs.path_cells", "must be >= 1")
        if self.max_priority < 1:
            raise ConfigError("obs.max_priority", "must be >= 1")

    @property
    def window(self) -> int:
        return 2 * self.radius + 1

    @property
    def grid_size(self) -> int:
        """Number of window values: ``C_o · (2r+1)²``."""
        return N_CHANNELS * self.window * self.window

    @property
    def dim(self) -> int:
        """``d_o = C_o (2r+1)² + 13`` (356 for ``r = 3``)."""
        return self.grid_size + N_SCALARS


#: Window channels (model §6.1).
N_CHANNELS = 7
CH_STATIC, CH_DYN, CH_DYN_PRED, CH_ROBOT, CH_ROBOT_PRIO, CH_PATH, CH_GOAL = range(N_CHANNELS)
#: Channels holding 0/1 values; ``CH_ROBOT_PRIO`` holds ``level / P``.
BINARY_CHANNELS = (CH_STATIC, CH_DYN, CH_DYN_PRED, CH_ROBOT, CH_PATH, CH_GOAL)
#: Scalar vector ``u_i`` (model §6.1): goal offset (2), capped distance (1), mode one-hot (3),
#: own task priority (1), suggested-action one-hot (5), plan-valid flag (1).
N_SCALARS = 13
N_ACTIONS = 5


@dataclass(frozen=True, slots=True)
class RewardParams:
    """Reward weights (model §7, initial defaults) and the discount used by the shaping term."""

    c_step: float = 0.01
    zeta: float = 1.0
    r_pick: float = 0.5
    r_del: float = 1.0
    c_rr: float = 1.0
    c_ro: float = 1.0
    c_sh: float = 0.5
    beta: float = 0.3
    gamma: float = 0.99

    def __post_init__(self) -> None:
        if not 0.0 <= self.beta <= 1.0:
            raise ConfigError("reward.beta", "must be in [0, 1]")
        if not 0.0 < self.gamma < 1.0:
            raise ConfigError("reward.gamma", "must be in (0, 1)")
        for name in ("c_step", "zeta", "r_pick", "r_del", "c_rr", "c_ro", "c_sh"):
            if getattr(self, name) < 0:
                raise ConfigError(f"reward.{name}", "must be >= 0")


@dataclass(frozen=True, slots=True)
class EnvConfig:
    """System components inside the environment (approved training defaults, D2):
    A\\*, auction, endpoint recovery on, deadlock resolver off, shield off."""

    planner: str = "astar"
    allocator: str = "auction"
    endpoint: bool = True
    deadlock: bool = False
    shield: str = "off"
    rrt_iterations: int | None = None
    obs: ObsParams = field(default_factory=ObsParams)
    reward: RewardParams = field(default_factory=RewardParams)
    global_blocks: int = 8

    def __post_init__(self) -> None:
        if self.planner not in ("astar", "rrtstar"):
            raise ConfigError("env.planner", f"expected astar or rrtstar, got {self.planner!r}")
        if self.allocator not in ("auction", "sequential"):
            raise ConfigError(
                "env.allocator", f"expected auction or sequential, got {self.allocator!r}"
            )
        if self.shield not in ("off", "predictive"):
            raise ConfigError("env.shield", f"expected off or predictive, got {self.shield!r}")
        if self.global_blocks < 1:
            raise ConfigError("env.global_blocks", "must be >= 1")
