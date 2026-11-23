"""Shared pytest configuration.

Hypothesis runs derandomized (fixed examples) so that the suite is fully reproducible (05 §A.1).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings

settings.register_profile(
    "default",
    derandomize=True,
    deadline=None,
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("default")

#: The real Stage 3a freeze (git-ignored ``results/``). ``tests/tools/test_phase6.py`` is hashed by
#: the Phase 6 freeze record and is therefore not edited; its pre-run check test reads this file
#: and is skipped only where the file is absent (e.g. CI). ``tests/tools/test_phase6_precheck.py``
#: covers the same Amendment A1 property everywhere with an injected freeze.
STAGE3A_FREEZE = Path(__file__).resolve().parents[1] / "results/dev6a/phase3/stage3a/freeze.json"
NEEDS_STAGE3A_FREEZE = (
    "tests/tools/test_phase6.py::test_precheck_never_generates_or_fingerprints_block_v",
)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if STAGE3A_FREEZE.exists():
        return
    skip = pytest.mark.skip(
        reason="the real Stage 3a freeze is absent (git-ignored results/); the A1 pre-run check "
        "property is covered by tests/tools/test_phase6_precheck.py"
    )
    for item in items:
        if item.nodeid in NEEDS_STAGE3A_FREEZE:
            item.add_marker(skip)
