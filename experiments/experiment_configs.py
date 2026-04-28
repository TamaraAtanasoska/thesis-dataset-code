"""
Experiment configuration protocol and base class.

Each experiment module (exp0–exp6, including exp4b) exports a ``CONFIG`` object that conforms
to ``ExperimentConfig``. The shared runner in ``runner.py`` uses this protocol
to build prompts, call the model, and parse responses without knowing
experiment-specific details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class ExperimentConfig(Protocol):
    """What the runner expects from each experiment definition."""

    name: str
    description: str
    response_format: dict[str, str] | None
    output_columns: list[str]

    def build_messages(self, row: Any, **kwargs) -> list[dict[str, str]]:
        """Return ``[{"role": ..., "content": ...}, ...]`` for the API call."""
        ...

    def parse_response(self, raw_text: str, row: Any, **kwargs) -> dict[str, Any]:
        """Extract experiment-specific output columns from the raw model response."""
        ...


@dataclass
class BaseExperimentConfig:
    """
    Convenience base that satisfies ``ExperimentConfig``.

    Subclass or instantiate directly, providing ``_build_messages_fn`` and
    ``_parse_response_fn`` callables.
    """

    name: str = ""
    description: str = ""
    response_format: dict[str, str] | None = field(default_factory=lambda: {"type": "json_object"})
    output_columns: list[str] = field(default_factory=list)

    _build_messages_fn: Callable[..., list[dict[str, str]]] | None = field(
        default=None, repr=False
    )
    _parse_response_fn: Callable[..., dict[str, Any]] | None = field(
        default=None, repr=False
    )

    def build_messages(self, row: Any, **kwargs) -> list[dict[str, str]]:
        if self._build_messages_fn is None:
            raise NotImplementedError(f"build_messages not set for {self.name}")
        return self._build_messages_fn(row, **kwargs)

    def parse_response(self, raw_text: str, row: Any, **kwargs) -> dict[str, Any]:
        if self._parse_response_fn is None:
            raise NotImplementedError(f"parse_response not set for {self.name}")
        return self._parse_response_fn(raw_text, row, **kwargs)
