"""Tests of the Block C development evaluation (runner guard, validity gate, frozen analysis)."""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "selection"))
import block_c
import block_c_analysis as bca

SEEDS = tuple(range(11_000, 11_012))
PICKS = {run: {"old": "step_00100000.pt", "new": "step_00200000.pt"} for run in bca.RUNS}
FPS = {lvl: {s: f"{lvl}-{s}" for s in SEEDS} for lvl in bca.LEVELS}


def rec(run: str, arm: str, lvl: str, s: int, **kw: Any) -> dict[str, Any]:
    r: dict[str, Any] = {
        **bca.EXPECTED, "label": run, "checkpoint": PICKS[run][arm], "arm": arm, "level": lvl,
        "seed": s, "robots": 4, "max_steps": 100, "instance_sha256": f"{lvl}-{s}",
        "status": "ok", "steps": 100, "released": 10, "delivered": 9, "conflicts_total": 3,
        "conflicts": {"RRv": 3, "RRs": 0, "ROv": 0, "ROs": 0, "INV": 0},
        "physical_collisions": 0, "terminated": False, "base_return": 1.0,
        "mean_flow_time": 20.0,
    }  # fmt: skip
    r.update(kw)
    return r


def records(seed: int = 0, effect: float = 0.0) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    out = []
    for run in bca.RUNS:
        for lvl in bca.LEVELS:
            for s in SEEDS:
                for arm in bca.ARMS:
                    mu = 6.0 * (1 - effect) if arm == "new" and run == "mappo_seed1" else 6.0
                    k = int(rng.negative_binomial(2.0, 2.0 / (2.0 + mu)))
                    rel = int(rng.integers(8, 14))
                    d = rel - int(rng.integers(0, 2))
                    out.append(rec(run, arm, lvl, s, released=rel, delivered=d,
                                   conflicts_total=k, conflicts={"RRv": k, "INV": 0},
                                   base_return=float(rng.normal()),
                                   terminated=d == rel))  # fmt: skip
    return out


def build(recs: list[dict[str, Any]]) -> bca.Data:
    return bca.build(recs, PICKS, FPS, seeds=SEEDS)


# --------------------------------------------------------------------------- constants, guard


def test_frozen_constants() -> None:
    assert bca.RUNS == ("mappo_seed1", "mappo_seed2", "mappo_seed3", "ippo_seed1")
    assert range(11_000, 11_120) == bca.BLOCK_C
    frozen = (bca.ALPHA, bca.N_BOOT, bca.BOOT_SEED, bca.QUANTILE_METHOD)
    assert frozen == (0.05, 10_000, 0, "higher")
    assert bca.PRIMARY == ("completion", "conflicts_per_episode")
    assert len(bca.SECONDARY) == 3 and not set(bca.PRIMARY) & set(bca.SECONDARY)
    assert bca.EXPECTED["block"] == "C" and bca.EXPECTED["mode"] == "greedy"
    for r in block_c.FORBIDDEN.values():
        assert not set(r) & set(bca.BLOCK_C)
    assert block_c.FORBIDDEN["D"] == range(12_000, 12_200)


@pytest.mark.parametrize("s", [11_000, 11_059, 11_119])
def test_guard_accepts_block_c(s: int) -> None:
    assert block_c.guard([s], "C") == [s]


@pytest.mark.parametrize("s", [12_000, 12_199, 20_000, 10_000, 10_050, 13_000, 11_120, 10_999, 0])
def test_guard_rejects_everything_else(s: int) -> None:
    with pytest.raises(ValueError):
        block_c.guard([s], "C")


def test_guard_rejects_other_block_labels() -> None:
    with pytest.raises(ValueError):
        block_c.guard([11_000], "A")


def test_worker_refuses_block_d_before_generating_the_scenario() -> None:
    pytest.importorskip("robocoord")
    from robocoord.marl.evaluation import EvalTask

    task = EvalTask("ppo", "full", 12_000, "astar", "C", "x.pt", label="mappo_seed1")
    with pytest.raises(ValueError, match="block D"):
        block_c._run(("old", task))


def test_tasks_are_paired_and_cover_block_c_only() -> None:
    pytest.importorskip("robocoord")
    pk = {"runs": {r: {"old": f"step_{r}_old.pt", "new": f"step_{r}_new.pt"} for r in bca.RUNS}}
    jobs = block_c.tasks(pk)
    assert len(jobs) == 4 * 2 * 3 * 120 == 2_880
    by: dict[tuple[str, str, int], dict[str, Any]] = {}
    for arm, t in jobs:
        assert t.seed in bca.BLOCK_C and t.block == "C" and t.controller == "ppo"
        assert (t.planner, t.mode, t.shield, t.allocator) == ("astar", "greedy", "off", "auction")
        by.setdefault((t.label, t.level, t.seed), {})[arm] = t
    assert len(by) == 1_440
    for (run, _, _), pair in by.items():
        old, new = pair["old"], pair["new"]
        assert old.checkpoint.endswith(f"{run}_old.pt") and new.checkpoint.endswith(f"{run}_new.pt")
        for f in ("controller", "level", "seed", "planner", "block", "shield", "rrt_iterations",
                  "label", "allocator", "mode", "replicate"):  # fmt: skip
            assert getattr(old, f) == getattr(new, f)
    assert {t.label for _, t in jobs} == set(bca.RUNS)


# --------------------------------------------------------------------------- statistics


def test_holm_known_values() -> None:
    adj = bca.holm([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert bca.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert bca.holm([]) == []


def test_holm_is_monotone_and_dominates_raw() -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        p = list(rng.uniform(0, 0.2, size=8))
        adj = bca.holm(p)
        assert all(a >= x for a, x in zip(adj, p, strict=True))
        order = np.argsort(p, kind="stable")
        assert all(adj[order[i]] <= adj[order[i + 1]] for i in range(7))


def test_paired_ratio_reduces_to_paired_mean_for_unit_denominator() -> None:
    rng = np.random.default_rng(2)
    yo, yn = rng.poisson(5, 40).astype(float), rng.poisson(4, 40).astype(float)
    one = np.ones(40)
    d, se = bca.paired_ratio(yo, one, yn, one)
    assert float(d) == pytest.approx(np.mean(yn - yo))
    assert float(se) == pytest.approx(np.std(yn - yo, ddof=1) / math.sqrt(40))


def test_paired_ratio_hand_example() -> None:
    yo, xo = np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 4.0])
    yn, xn = np.array([2.0, 2.0, 3.0]), np.array([2.0, 2.0, 4.0])
    d, se = bca.paired_ratio(yo, xo, yn, xn)
    to, tn = 6 / 8, 7 / 8
    uo = (yo - to * xo) / (8 / 3)
    un = (yn - tn * xn) / (8 / 3)
    assert float(d) == pytest.approx(tn - to)
    assert float(se) == pytest.approx(np.std(un - uo, ddof=1) / math.sqrt(3))


def test_bootstrap_identical_arms_is_degenerate_null() -> None:
    y = np.arange(1.0, 21.0)
    one = np.ones(20)
    inf = bca.bootstrap_t(bca.paired_ratio, [y, one, y.copy(), one], bca.boot_index(20, 499))
    assert (inf.estimate, inf.se, inf.ci, inf.p) == (0.0, 0.0, (0.0, 0.0), 1.0)


def test_bootstrap_constant_shift_has_smallest_p() -> None:
    y = np.arange(1.0, 21.0)
    one = np.ones(20)
    inf = bca.bootstrap_t(bca.paired_ratio, [y, one, y + 2.0, one], bca.boot_index(20, 499))
    assert inf.estimate == pytest.approx(2.0) and inf.se == pytest.approx(0.0, abs=1e-12)
    assert inf.p == pytest.approx(1 / 500)


def test_bootstrap_is_deterministic_and_ci_brackets_estimate() -> None:
    rng = np.random.default_rng(3)
    one = np.ones(60)
    a = [rng.poisson(5, 60).astype(float), one, rng.poisson(4, 60).astype(float), one]
    i1 = bca.bootstrap_t(bca.paired_ratio, a, bca.boot_index(60, 999))
    i2 = bca.bootstrap_t(bca.paired_ratio, a, bca.boot_index(60, 999))
    assert i1 == i2
    assert i1.ci[0] < i1.estimate < i1.ci[1] and 0 < i1.p <= 1
    assert i1.q > 1.5


def test_bootstrap_t_is_calibrated_under_a_skewed_paired_null() -> None:
    """Exchangeable arms, sparse negative-binomial conflicts: rejection rate near α."""
    rng = np.random.default_rng(4)
    n, reps = 120, 300
    idx = bca.boot_index(n, 999, 5)
    rej = 0
    for _ in range(reps):
        mu = rng.gamma(0.5, 8.0, n)
        a = rng.poisson(mu).astype(float)
        b = rng.poisson(mu).astype(float)
        one = np.ones(n)
        rej += bca.bootstrap_t(bca.paired_ratio, [a, one, b, one], idx).p <= 0.05
    assert rej / reps < 0.09


# --------------------------------------------------------------------------- validity gate


def test_complete_records_are_evaluable_and_sum_correctly() -> None:
    recs = records()
    d = build(recs)
    assert all(d.evaluable.values()) and not any(d.problems.values())
    a = d.arrays["ippo_seed1"]["new"]
    want = sum(r["delivered"] for r in recs if r["label"] == "ippo_seed1" and r["arm"] == "new")
    assert a["delivered"].sum() == want
    assert a["robot_steps"].sum() == 4 * 100 * 3 * len(SEEDS)
    assert a["episodes"].sum() == 3 * len(SEEDS)


def test_record_order_does_not_matter() -> None:
    recs = records()
    rev = list(reversed(recs))
    r1 = bca.analyse(build(recs), n_boot=199)
    r2 = bca.analyse(build(rev), n_boot=199)
    assert r1 == r2


@pytest.mark.parametrize(
    ("mutate", "word"),
    [
        (lambda rs: rs.pop(5), "missing"),
        (lambda rs: rs.append(copy.deepcopy(rs[5])), "duplicate"),
        (lambda rs: rs[5].update(status="error", error_type="PhysicalCollisionError"), "failed"),
        (lambda rs: rs[5].update(status="error", error_type="InvalidTransitionError"), "failed"),
        (lambda rs: rs[5].update(delivered=99), "invalid"),
        (lambda rs: rs[5].update(conflicts_total=-1), "invalid"),
        (lambda rs: rs[5].update(conflicts_total=2.5), "invalid"),
        (lambda rs: rs[5].update(released=None), "invalid"),
        (lambda rs: rs[5].update(steps=101), "invalid"),
        (lambda rs: rs[5].update(base_return=float("nan")), "invalid"),
        (lambda rs: rs[5].update(conflicts={"RRv": 0}), "invalid"),
        (lambda rs: rs[5].update(terminated=True, delivered=1), "invalid"),
    ],
)
def test_gate_marks_the_run_not_evaluable(mutate: Any, word: str) -> None:
    recs = records()
    run = recs[5]["label"]
    mutate(recs)
    d = build(recs)
    assert not d.evaluable[run]
    assert any(p.startswith(word) for p in d.problems[run])
    assert all(v for r, v in d.evaluable.items() if r != run)
    assert len(d.complete_seeds[run]) == len(SEEDS) - 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(mode="stochastic"),
        lambda r: r.update(shield="on"),
        lambda r: r.update(planner="rrtstar"),
        lambda r: r.update(block="D"),
        lambda r: r.update(checkpoint="step_00999999.pt"),
        lambda r: r.update(label="ippo_seed2"),
        lambda r: r.update(seed=12_000),
        lambda r: r.update(level="warehouse"),
        lambda r: r.update(instance_sha256="other"),
        lambda r: r.update(robots=5),
        lambda r: r.update(max_steps=99),
    ],
)
def test_configuration_or_pairing_mismatch_aborts(mutate: Any) -> None:
    recs = records()
    mutate(recs[7])
    with pytest.raises(bca.ConfigError):
        build(recs)


# --------------------------------------------------------------------------- analysis


def test_analysis_structure_and_holm_families() -> None:
    res = bca.analyse(build(records()), n_boot=499)
    assert len(res["families"]["primary"]) == 8 and len(res["families"]["secondary"]) == 12
    assert len(res["per_level"]) == 4 * 3 * 5 and len(res["across_run_mean"]) == 5
    for fam in res["families"].values():
        raw = [h["p"] for h in fam]
        assert [h["p_holm"] for h in fam] == pytest.approx(bca.holm(raw))
        for h in fam:
            assert h["estimate"] == pytest.approx(h["new"] - h["old"])
            assert h["ci95"][0] <= h["estimate"] <= h["ci95"][1]
            assert h["reject"] == (h["p_holm"] <= 0.05)
    assert all("p" not in r for r in res["per_level"])


def test_planted_conflict_reduction_is_detected_for_that_run_only() -> None:
    res = bca.analyse(build(records(seed=7, effect=0.8)), n_boot=999)
    prim = res["families"]["primary"]
    conf = {h["run"]: h for h in prim if h["outcome"] == "conflicts_per_episode"}
    assert conf["mappo_seed1"]["reject"] and conf["mappo_seed1"]["estimate"] < 0
    assert conf["mappo_seed1"]["relative_change"] == pytest.approx(
        conf["mappo_seed1"]["new"] / conf["mappo_seed1"]["old"] - 1
    )


def test_not_evaluable_run_enters_holm_with_p_one_and_uses_complete_seeds() -> None:
    recs = records(seed=7, effect=0.8)
    recs = [r for r in recs if not (r["label"] == "mappo_seed1" and r["seed"] == 11_003
                                    and r["arm"] == "new" and r["level"] == "full")]  # fmt: skip
    res = bca.analyse(build(recs), n_boot=499)
    for h in res["families"]["primary"]:
        if h["run"] == "mappo_seed1":
            assert not h["evaluable"] and h["p_for_holm"] == 1.0 and not h["reject"]
            assert h["n_seeds"] == len(SEEDS) - 1
        else:
            assert h["evaluable"] and h["n_seeds"] == len(SEEDS)
    assert res["n_seeds_used"]["mappo_seed1"] == len(SEEDS) - 1
    assert all(r["n_seeds"] == len(SEEDS) - 1 for r in res["across_run_mean"])


def test_safety_counts() -> None:
    recs = records()
    recs[0].update(status="error", error_type="PhysicalCollisionError")
    recs[2].update(conflicts={"RRv": recs[2]["conflicts_total"] - 1, "INV": 1})
    recs[4].update(terminated=False, delivered=recs[4]["released"] - 2)
    rows = bca.safety(recs, PICKS)
    assert len(rows) == 4 * 2 * 4
    tot = {k: sum(r[k] for r in rows if r["level"] == "all")
           for k in ("episodes", "failed", "invalid_moves_INV", "physical_collisions")}  # fmt: skip
    assert tot == {"episodes": len(recs), "failed": 1, "invalid_moves_INV": 1,
                   "physical_collisions": 0}  # fmt: skip
    row = next(r for r in rows if r["run"] == recs[0]["label"] and r["arm"] == recs[0]["arm"]
               and r["level"] == "all")  # fmt: skip
    assert row["error_types"] == {"PhysicalCollisionError": 1}
    ends = ("terminated_all_delivered", "horizon_with_undelivered", "horizon_all_delivered")
    for r in rows:
        if r["level"] == "all":
            assert r["completed"] == sum(r[k] for k in ends) + r["invalid_records"]


def test_same_weights_ignores_archive_name_but_not_values(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    sd = {"0.weight": torch.arange(6.0).reshape(2, 3), "0.bias": torch.zeros(2)}
    a, b, c, d = (tmp_path / n for n in ("best.pt", "step_00000001.pt", "x.pt", "y.pt"))
    torch.save(sd, a)
    torch.save({k: v.clone() for k, v in sd.items()}, b)
    assert a.read_bytes() != b.read_bytes()
    assert block_c.same_weights(a, b)
    torch.save({**sd, "0.bias": torch.tensor([0.0, 1e-7])}, c)
    assert not block_c.same_weights(a, c)
    torch.save({**sd, "0.bias": torch.zeros(2, dtype=torch.float64)}, d)
    assert not block_c.same_weights(a, d)
