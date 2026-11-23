"""``ppo_trainer.peak_rss_mb``: platform-dependent peak-memory reading (Windows API or getrusage).

CI runs the fast suite on Ubuntu and Windows, so each platform branch is exercised there.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("torch")

from robocoord.marl.ppo_trainer import peak_rss_mb


def test_peak_rss_mb_reports_this_process() -> None:
    mb = peak_rss_mb()
    assert mb is not None
    assert 1.0 < mb < 1_000_000.0  # a Python process with NumPy and torch loaded


@pytest.mark.skipif(sys.platform == "win32", reason="getrusage branch (POSIX only)")
def test_peak_rss_mb_matches_getrusage() -> None:
    import resource

    mb = peak_rss_mb()
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # read after: ru_maxrss never falls
    scale = 2**20 if sys.platform == "darwin" else 2**10
    assert mb is not None and mb <= kb / scale


def test_peak_rss_mb_returns_none_without_the_platform_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pretend to be the other platform: its API is missing here (no ``ctypes.WinDLL`` off
    # Windows, no ``resource`` module on Windows), and the reading must degrade to None.
    monkeypatch.setattr(sys, "platform", "linux" if sys.platform == "win32" else "win32")
    assert peak_rss_mb() is None
