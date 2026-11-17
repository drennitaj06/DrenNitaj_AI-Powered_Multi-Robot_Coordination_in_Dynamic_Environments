"""CLI behaviour, in-process (fast suite)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robocoord.cli import main
from robocoord.sim.trace import read_trace

ROOT = Path(__file__).resolve().parents[2]
DEMO = str(ROOT / "configs" / "scenarios" / "demo_delivery.toml")
RANDOM = str(ROOT / "configs" / "scenarios" / "random_warehouse.toml")


def test_simulate_headless_prints_metrics(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["simulate", "--scenario", DEMO, "--no-animate"]) == 0
    out = capsys.readouterr().out
    assert "all tasks delivered" in out
    assert "delivered 4/4" in out and "physical collisions 0" in out


def test_simulate_animation_frames(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(["simulate", "--scenario", DEMO, "--delay", "0", "--no-clear", "--every", "10"]) == 0
    )
    out = capsys.readouterr().out
    assert out.count("legend:") >= 3  # initial frame + periodic + final
    assert "DELIVERED task" in out or "delivered" in out
    assert "\x1b[" not in out  # no ANSI codes without --color/clearing


def test_animation_has_no_escape_codes_when_output_is_not_a_terminal(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Default animation clears the screen only on a real TTY; captured/piped output must stay
    clean (regression: it used to emit clear-screen codes into files and pipes)."""
    assert main(["simulate", "--scenario", DEMO, "--delay", "0", "--every", "15"]) == 0
    assert "\x1b[" not in capsys.readouterr().out


def test_no_coordinator_note_explains_zero_deliveries(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["simulate", "--scenario", RANDOM, "--controller", "random", "--no-animate"]) == 0
    out = capsys.readouterr().out
    assert "delivered 0/" in out and "no task coordinator is attached" in out
    assert main(["simulate", "--scenario", DEMO, "--no-animate"]) == 0
    assert "no task coordinator" not in capsys.readouterr().out


def test_corridor_demo_cli_reports_conflicts_and_rates(capsys: pytest.CaptureFixture[str]) -> None:
    corridor = str(ROOT / "configs" / "scenarios" / "corridor_queue.toml")
    assert main(["simulate", "--scenario", corridor, "--no-animate"]) == 0
    out = capsys.readouterr().out
    assert "delivered 3/3" in out and "'RRv': 4" in out and "blocked moves 4" in out
    assert "rates per 100 robot-steps: conflicts 13.33" in out


def test_simulate_json_and_trace(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    trace = tmp_path / "t.jsonl"
    assert main(["simulate", "--scenario", DEMO, "--json", "--trace", str(trace)]) == 0
    metrics = json.loads(capsys.readouterr().out)
    assert metrics["tasks_delivered"] == 4 and metrics["physical_collisions"] == 0
    recs = read_trace(trace)
    assert recs[-1]["metrics"] == metrics


def test_random_controller_and_seed_override(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(["simulate", "--scenario", RANDOM, "--controller", "random", "--json", "--seed", "4"])
        == 0
    )
    metrics = json.loads(capsys.readouterr().out)
    assert metrics["steps"] == 80 and metrics["conflicts"]["INV"] == 0


def test_validate_and_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", "--scenario", DEMO]) == 0
    assert "OK: demo-delivery" in capsys.readouterr().out
    bad = tmp_path / "bad.toml"
    bad.write_text(
        '[scenario]\nname="b"\n[map]\nrows=["..."]\n[[robots]]\nstart=[5,0]\n', encoding="utf-8"
    )
    assert main(["validate", "--scenario", str(bad)]) == 2
    assert "robots[0].start" in capsys.readouterr().err
    # scripted controller without a [script] section
    assert main(["simulate", "--scenario", RANDOM, "--controller", "scripted", "--no-animate"]) == 2
