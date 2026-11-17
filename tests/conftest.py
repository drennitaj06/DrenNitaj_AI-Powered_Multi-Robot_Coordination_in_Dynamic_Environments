"""Shared pytest configuration.

Hypothesis runs derandomized (fixed examples) so that the suite is fully reproducible (05 §A.1).
"""

from __future__ import annotations

from hypothesis import HealthCheck, settings

settings.register_profile(
    "default",
    derandomize=True,
    deadline=None,
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("default")
