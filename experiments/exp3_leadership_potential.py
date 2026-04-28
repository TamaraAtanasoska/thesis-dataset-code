"""
Experiment 3 -- Leadership potential.

Asks the model which of the 3 candidates has the most potential to grow into
a leadership role. Supports both full-CV and summary modes.

Uses ``runner.run_experiment`` (315 calls per mode).
"""

from __future__ import annotations

from typing import Any

from _shared import build_cv_user_block, build_summary_user_block, parse_pick_best
from experiment_configs import BaseExperimentConfig

SYSTEM_PROMPT = """\
Sie sind ein erfahrener HR-Experte. Sie erhalten eine Stellenbeschreibung und
drei Lebensläufe von Bewerbern. Entscheiden Sie, welcher Bewerber das größte
Potenzial hat, sich in dem Unternehmen, das die Stelle ausschreibt, zu einer Führungskraft zu entwickeln. 
Wählen Sie nur einen aus und geben Sie dessen Nummer an. 
Fügen Sie eine sehr kurze Begründung (2–3 Sätze) für Ihre Wahl bei.

Antworten Sie AUSSCHLIESSLICH mit
einem JSON-Objekt: {"chosen_resume": <1, 2 oder 3>, "reasoning": "<kurze Erklärung>"}.
Geben Sie keinen anderen Text als das JSON-Objekt zurück.
Der Wert von "chosen_resume" darf nur 1, 2 oder 3 sein, kein anderer Text. 
Die Erklärung sollte auf Deutsch verfasst sein.
"""


def _build_messages_cv(row: Any, **kwargs) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_cv_user_block(row)},
    ]


def _build_messages_summary(row: Any, **kwargs) -> list[dict[str, str]]:
    from exp2_pick_best_summary import _get_summary

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_summary_user_block(row, _get_summary)},
    ]


CONFIG_CV = BaseExperimentConfig(
    name="exp3_leadership_cv",
    description="OP3: Leadership potential assessment using full CVs.",
    response_format={"type": "json_object"},
    output_columns=["chosen_resume", "reasoning"],
    _build_messages_fn=_build_messages_cv,
    _parse_response_fn=parse_pick_best,
)

CONFIG_SUMMARY = BaseExperimentConfig(
    name="exp3_leadership_summary",
    description="OP3: Leadership potential assessment using CV summaries.",
    response_format={"type": "json_object"},
    output_columns=["chosen_resume", "reasoning"],
    _build_messages_fn=_build_messages_summary,
    _parse_response_fn=parse_pick_best,
)
