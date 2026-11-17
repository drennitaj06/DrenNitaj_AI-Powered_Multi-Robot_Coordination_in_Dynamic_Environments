"""Scenario specification, TOML loading, validation and synthetic generators."""

from robocoord.scenario.loader import build_scenario, load_scenario
from robocoord.scenario.spec import ScenarioError, ScenarioSpec

__all__ = ["ScenarioError", "ScenarioSpec", "build_scenario", "load_scenario"]
