"""Shared configuration error (architecture §9)."""

from __future__ import annotations


class ConfigError(ValueError):
    """An invalid configuration value. ``field`` names the offending key (e.g. ``planner.name``)."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field = field
