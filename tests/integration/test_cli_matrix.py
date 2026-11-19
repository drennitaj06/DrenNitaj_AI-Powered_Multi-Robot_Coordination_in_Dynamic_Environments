"""End-to-end ``simulate`` across the 2 × 2 × 2 × 2 planner × allocator × controller × shield
matrix (Issue #4 stage 7; acceptance criteria 8, 9, 11, 12) and the demo-scenario evidence.

Every capability a demo scenario claims is asserted on its actual trace or metrics here.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import json
from pathlib import Path
from typing import Any

import pytest

from robocoord.cli import main, resolve_deadlock, resolve_endpoint
from robocoord.core.errors import ConfigError

ROOT = Path(__file__).resolve().parents[2]
SCEN = ROOT / "configs" / "scenarios"
MATRIX = list(
    itertools.product(
        ("astar", "rrtstar"),
        ("auction", "sequential"),
        ("reference", "baseline"),
        ("off", "predictive"),
    )
)
#: Designated delivery demos (criterion 9): the baseline must deliver every task.
#: demo_delivery and random_warehouse are dynamic (moving obstacles; map events / arrivals),
#: headon_two_lane is static, preemption has a patrolling obstacle.
DELIVERY_DEMOS = [
    "demo_delivery",  # dynamic, in the fast suite (CI)
    "headon_two_lane",  # static, in the fast suite (CI)
    pytest.param("random_warehouse", marks=pytest.mark.slow),  # long suite: pytest -m slow
    pytest.param("preemption", marks=pytest.mark.slow),
]


def cli(*args: str) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(list(args))
    return code, buf.getvalue()


def simulate_json(
    name: str, planner: str, allocator: str, controller: str, shield: str, *extra: str
) -> dict[str, Any]:
    code, out = cli(
        "simulate",
        "--scenario",
        str(SCEN / f"{name}.toml"),
        "--planner",
        planner,
        "--allocator",
        allocator,
        "--controller",
        controller,
        "--shield",
        shield,
        "--json",
        *extra,
    )
    assert code == 0
    return json.loads(out)  # type: ignore[no-any-return]


def trace_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------- the 16 combinations


@pytest.mark.parametrize(("planner", "allocator", "controller", "shield"), MATRIX)
@pytest.mark.parametrize("name", DELIVERY_DEMOS)
def test_matrix_runs_end_to_end(
    name: str, planner: str, allocator: str, controller: str, shield: str
) -> None:
    m = simulate_json(name, planner, allocator, controller, shield)
    assert m["physical_collisions"] == 0 and m["tasks_released"] > 0
    extra = m["extra"]
    assert extra["planning"]["allocation"]["allocator"] == allocator
    assert extra["controller"]["controller"] == controller
    assert ("shield" in extra) == (shield == "predictive")
    assert ("deadlock" in extra["planning"]) == (controller == "baseline")  # --deadlock auto
    if controller == "baseline":
        assert m["tasks_delivered"] == m["tasks_released"]  # criterion 9 on the designated demos
        c = m["conflicts"]
        assert c["RRv"] == c["RRs"] == c["ROs"] == c["INV"] == 0
    else:
        assert m["conflicts_avoided"] == 0 and m["conflicts_introduced"] == 0  # X-10


# ---------------------------------------------------------------- demo evidence from traces


def run_trace(tmp_path: Path, name: str, *args: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = tmp_path / f"{name}.jsonl"
    code, out = cli(
        "simulate", "--scenario", str(SCEN / f"{name}.toml"), "--json", "--trace", str(path), *args
    )
    assert code == 0
    return json.loads(out), trace_records(path)


@pytest.mark.parametrize("allocator", ["auction", "sequential"])
def test_preemption_demo(tmp_path: Path, allocator: str) -> None:
    m, recs = run_trace(tmp_path, "preemption", "--planner", "astar", "--allocator", allocator)
    at3 = next(r for r in recs if r.get("type") == "step" and r["t"] == 4)  # the step from t = 3
    assert at3["alloc"]["unassigned"] == [[0, 0]] and [0, 1] in at3["alloc"]["assigned"]
    assert at3["plans"][0][:4] == [0, "T1", "ok", 0]  # robot 0 replans to the urgent pickup
    assert m["extra"]["planning"]["allocation"]["preemptions"] == 1
    assert m["tasks_delivered"] == 3
    picked_by = {
        task: robot for r in recs if r.get("type") == "step" for robot, task in r["picked"]
    }
    # the urgent task is done by robot 0; the pre-empted task is re-auctioned to robot 1 once
    # robot 1's delivery triggers an auction
    assert picked_by == {0: 1, 1: 0, 2: 1}
    reassign = next(r for r in recs if r.get("type") == "step" and r["t"] == 7)
    assert (
        reassign["alloc"]["assigned"] == [[1, 0]] and "delivered" in reassign["alloc"]["triggers"]
    )


def test_two_lane_deadlock_demo(tmp_path: Path) -> None:
    m, recs = run_trace(tmp_path, "headon_two_lane", "--planner", "astar", "--allocator", "auction")
    events = [e for r in recs if "deadlock" in r for e in r["deadlock"]]
    assert [e[0] for e in events] == ["detected_persistent", "detours_started", "released"]
    t7 = [p for r in recs for p in r.get("plans", []) if p[1] == "T7"]
    assert len(t7) == 2 and m["tasks_delivered"] == 2 and m["conflicts_total"] == 0
    off, _ = run_trace(
        tmp_path,
        "headon_two_lane",
        "--planner",
        "astar",
        "--allocator",
        "auction",
        "--deadlock",
        "off",
    )
    assert off["tasks_delivered"] == 0 and off["conflicts_total"] == 0  # the stage 5A behaviour


def test_single_lane_unresolved_demo(tmp_path: Path) -> None:
    m, recs = run_trace(
        tmp_path, "headon_single_lane", "--planner", "rrtstar", "--allocator", "sequential"
    )
    events = [e[0] for r in recs if "deadlock" in r for e in r["deadlock"]]
    assert events == ["detected_persistent", "no_passing_place"]
    assert m["tasks_delivered"] == 0 and m["conflicts_total"] == 0
    assert m["extra"]["planning"]["deadlock"]["unresolved_at_end"] == 1


def test_shield_demo(tmp_path: Path) -> None:
    """Reference follower: the shield turns 37 proposed robot–obstacle conflicts into 37
    interventions (the obstacle pins itself in the single lane, so nothing is delivered)."""
    off, _ = run_trace(
        tmp_path,
        "shield_corridor",
        "--planner",
        "astar",
        "--allocator",
        "auction",
        "--controller",
        "reference",
    )
    on, recs = run_trace(
        tmp_path,
        "shield_corridor",
        "--planner",
        "astar",
        "--allocator",
        "auction",
        "--controller",
        "reference",
        "--shield",
        "predictive",
    )
    assert off["conflicts"]["ROv"] + off["conflicts"]["ROs"] > 0 and "shield" not in off["extra"]
    assert on["conflicts_total"] == 0 and on["extra"]["shield"]["interventions"] > 0
    assert on["extra"]["shield"]["prevented"] == off["conflicts_total"]
    assert any("shield" in r for r in recs) and recs[0]["shield"] == "predictive"


def test_dynamic_demo_has_moving_obstacles_map_events_and_parallel_robots(tmp_path: Path) -> None:
    m, recs = run_trace(tmp_path, "demo_delivery", "--planner", "rrtstar", "--allocator", "auction")
    steps = [r for r in recs if r.get("type") == "step"]
    assert len({tuple(map(tuple, r["obstacles"])) for r in steps}) > 1  # obstacles move
    assert any("map_events" in r for r in steps)
    busy = [sum(1 for rob in r["robots"] if rob[2] != "I") for r in steps]
    assert max(busy) >= 2 and m["tasks_delivered"] == 4  # several robots work at once


# ---------------------------------------------------------------- CLI behaviour


def test_text_output_lists_issue4_statistics() -> None:
    code, out = cli(
        "simulate",
        "--scenario",
        str(SCEN / "headon_two_lane.toml"),
        "--planner",
        "astar",
        "--allocator",
        "auction",
        "--shield",
        "predictive",
        "--no-animate",
    )
    assert code == 0
    for text in (
        "allocation (auction)",
        "deadlock resolver: detected 1+0",
        "controller (baseline)",
        "shield (predictive)",
    ):
        assert text in out
    assert "coordinator: auction+astar+detour" in out


def test_configuration_validation(capsys: pytest.CaptureFixture[str]) -> None:
    demo = str(SCEN / "demo_delivery.toml")
    assert (
        main(["simulate", "--scenario", demo, "--allocator", "auction", "--json"]) == 2
    )  # no planner
    assert "--allocator needs --planner" in capsys.readouterr().err
    assert (
        main(
            [
                "simulate",
                "--scenario",
                demo,
                "--planner",
                "astar",
                "--allocator",
                "auction",
                "--controller",
                "scripted",
            ]
        )
        == 2
    )
    assert main(["simulate", "--scenario", demo, "--planner", "astar", "--deadlock", "on"]) == 2
    assert (
        main(
            [
                "simulate",
                "--scenario",
                str(SCEN / "random_warehouse.toml"),
                "--controller",
                "baseline",
            ]
        )
        == 2
    )
    assert resolve_deadlock("auto", "baseline", "auction") and not resolve_deadlock(
        "auto", "reference", "auction"
    )
    assert not resolve_deadlock("auto", "baseline", None) and resolve_deadlock(
        "on", "reference", "auction"
    )
    with pytest.raises(ConfigError):
        resolve_deadlock("on", "baseline", None)


@pytest.mark.parametrize(("planner", "allocator", "controller", "shield"), MATRIX)
def test_matrix_with_endpoint_recovery_on_and_off(
    planner: str, allocator: str, controller: str, shield: str
) -> None:
    """Endpoint recovery is on by default with an allocator, for every controller, and
    ``--endpoint off`` removes it (no T8 trigger, no statistics, earlier coordinator name)."""
    on = simulate_json("demo_delivery", planner, allocator, controller, shield)
    off = simulate_json(
        "demo_delivery", planner, allocator, controller, shield, "--endpoint", "off"
    )
    assert (
        "endpoint" in on["extra"]["planning"]
        and "T8" in on["extra"]["planning"]["calls_by_trigger"]
    )
    assert "endpoint" not in off["extra"]["planning"]
    assert "T8" not in off["extra"]["planning"]["calls_by_trigger"]
    for m in (on, off):
        assert m["physical_collisions"] == 0
        if controller == "baseline":
            assert m["tasks_delivered"] == m["tasks_released"]
            c = m["conflicts"]
            assert c["RRv"] == c["RRs"] == c["ROs"] == c["INV"] == 0


def test_endpoint_option_validation_and_text(capsys: pytest.CaptureFixture[str]) -> None:
    demo = str(SCEN / "demo_delivery.toml")
    assert main(["simulate", "--scenario", demo, "--planner", "astar", "--endpoint", "on"]) == 2
    assert "--endpoint on needs --allocator" in capsys.readouterr().err
    assert resolve_endpoint(None, "auction") and not resolve_endpoint(None, None)
    assert not resolve_endpoint("off", "sequential") and resolve_endpoint("on", "auction")
    with pytest.raises(ConfigError):
        resolve_endpoint("on", None)
    code, out = cli(
        "simulate",
        "--scenario",
        demo,
        "--planner",
        "astar",
        "--allocator",
        "auction",
        "--controller",
        "reference",
        "--no-animate",
    )
    assert code == 0 and "endpoint recovery: blockages temporary" in out
    assert "coordinator: auction+astar+endpoint" in out
    code, out = cli(
        "simulate",
        "--scenario",
        demo,
        "--planner",
        "astar",
        "--allocator",
        "auction",
        "--endpoint",
        "off",
        "--no-animate",
    )
    assert code == 0 and "endpoint recovery" not in out and "+endpoint" not in out


def test_fixture_path_unchanged_and_accepts_new_controllers() -> None:
    """Without --allocator the Issue #3 fixture coordinator is used (default controller
    ``follow``); the reference follower reproduces it exactly."""
    demo = str(SCEN / "demo_delivery.toml")
    code_f, follow = cli("simulate", "--scenario", demo, "--planner", "astar", "--json")
    code_r, ref = cli(
        "simulate", "--scenario", demo, "--planner", "astar", "--controller", "reference", "--json"
    )
    assert code_f == code_r == 0
    f, r = json.loads(follow), json.loads(ref)
    r["extra"].pop("controller")
    assert f == r


def test_deterministic_cli_traces(tmp_path: Path) -> None:
    texts = []
    for k in range(2):
        path = tmp_path / f"t{k}.jsonl"
        code, _ = cli(
            "simulate",
            "--scenario",
            str(SCEN / "random_warehouse.toml"),
            "--planner",
            "rrtstar",
            "--allocator",
            "sequential",
            "--shield",
            "predictive",
            "--json",
            "--trace",
            str(path),
        )
        assert code == 0
        texts.append(path.read_bytes())
    assert texts[0] == texts[1]
