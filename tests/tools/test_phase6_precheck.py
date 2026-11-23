"""Phase 6 pre-run check (Amendment A1), independent of the git-ignored Stage 3a freeze.

``tests/tools/test_phase6.py`` is hashed by the Phase 6 freeze record and is not edited. Its
pre-run check test reads the real Stage 3a freeze under ``results/`` and is skipped where that
file is absent (``tests/conftest.py``). These tests run the real ``check()`` everywhere, with a
temporary Stage 3a freeze built from the library's ``FROZEN_PARAMS``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
for sub in ("selection", "phase3", "phase1", "phase2", "phase4", "phase6"):
    sys.path.insert(0, str(ROOT / "tools" / sub))

import phase3 as p3  # noqa: E402
import phase6 as p6  # noqa: E402


class ScenarioGenerationError(RuntimeError):
    """Raised if the pre-run check generates any scenario."""


@pytest.fixture
def no_generation(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Every scenario generation (any block, any seed) raises and is recorded."""
    from robocoord.marl.scenarios import ScenarioDist

    calls: list[int] = []

    def never(self: Any, seed: int) -> Any:
        calls.append(int(seed))
        raise ScenarioGenerationError(f"scenario {seed} generated during the pre-run check")

    monkeypatch.setattr(ScenarioDist, "sample", never)
    return calls


def _stub_precheck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, tamper: bool = False
) -> None:
    """A temporary Stage 3a freeze from ``FROZEN_PARAMS`` (optionally with one parameter changed)
    and the same stubs as the frozen test for the protocol freeze, scan, checkpoints and backups."""
    from robocoord.control.supervisor import FROZEN_PARAMS

    params = {p: {k: getattr(v, k) for k in v.__slots__} for p, v in FROZEN_PARAMS.items()}
    if tamper:
        params["astar"]["hold"] += 1
    freeze = tmp_path / "stage3a_freeze.json"
    freeze.write_text(json.dumps({"frozen_params": params}), "utf-8")
    monkeypatch.setattr(p3, "STAGE3A_FREEZE", freeze)
    monkeypatch.setattr(
        p3, "STAGE3A_FREEZE_SHA256", hashlib.sha256(freeze.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        p6, "verify_freeze", lambda out: {"design": {"episodes": {"total": 22_080}}}
    )
    monkeypatch.setattr(p6, "scan_results", lambda out: {"clean": True, "files_scanned": 0})
    monkeypatch.setattr(p6, "checkpoints", lambda: {"ok": True, "policies": {}})
    monkeypatch.setattr(p6, "backup_status", lambda: {"ok": True})
    (tmp_path / "plan_freeze.json").write_text("{}", "utf-8")


def test_precheck_never_generates_or_fingerprints_block_v(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_generation: list[int]
) -> None:
    _stub_precheck(tmp_path, monkeypatch)
    res = p6.check(tmp_path, authorized=True)
    assert res["ok"] and res["stage3a_freeze_ok"] and res["episodes"] == 22_080
    assert no_generation == []  # no scenario of any block was generated
    text = (tmp_path / "check.json").read_text("utf-8")
    assert "fingerprints" not in text and "instances" not in json.loads(text)
    assert "deferred to run" in json.loads(text)["block_v_instance_check"]
    assert not (tmp_path / "instance_check.json").exists()


def test_precheck_fails_when_stage3a_parameters_differ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_generation: list[int]
) -> None:
    _stub_precheck(tmp_path, monkeypatch, tamper=True)
    res = p6.check(tmp_path, authorized=True)
    assert not res["stage3a_freeze_ok"] and not res["ok"]
    assert no_generation == []
    assert not (tmp_path / "instance_check.json").exists()


def test_precheck_fails_when_stage3a_freeze_hash_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_generation: list[int]
) -> None:
    _stub_precheck(tmp_path, monkeypatch)
    monkeypatch.setattr(p3, "STAGE3A_FREEZE_SHA256", "0" * 64)
    res = p6.check(tmp_path, authorized=True)
    assert not res["stage3a_freeze_ok"] and not res["ok"]
    assert no_generation == []
    assert not (tmp_path / "instance_check.json").exists()
