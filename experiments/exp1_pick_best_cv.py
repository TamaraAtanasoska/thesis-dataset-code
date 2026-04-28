"""
Experiment 1 -- Pick the best candidate (full CVs).

Presents the model with 1 JD + 3 augmented CVs and asks it to choose the best
candidate. Uses ``runner.run_experiment`` (one call per wide row = 315 total).
"""

from __future__ import annotations

from typing import Any

from _shared import build_cv_user_block, parse_pick_best
from experiment_configs import BaseExperimentConfig

SYSTEM_PROMPT = """\
Sie sind ein erfahrener Rekrutierer. Sie erhalten eine Stellenbeschreibung und drei
Lebensläufe von Bewerbern. Wählen Sie den besten Kandidaten für die Stelle aus. 
Geben Sie lediglich die Nummer des besten Bewerbers zurück. 
Fügen Sie eine sehr kurze Begründung (2–3 Sätze) für Ihre Wahl bei. 

Antworten Sie AUSSCHLIESSLICH mit
einem JSON-Objekt: {"chosen_resume": <1, 2 oder 3>, "reasoning": "<kurze Erklärung>"}.
Geben Sie keinen anderen Text als das JSON-Objekt zurück.
Der Wert von "chosen_resume" darf nur 1, 2 oder 3 sein, kein anderer Text. 
Die Erklärung sollte auf Deutsch verfasst sein.
"""


def _build_messages(row: Any, **kwargs) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_cv_user_block(row)},
    ]


CONFIG = BaseExperimentConfig(
    name="exp1_pick_best_cv",
    description="OP1: Pick the best candidate from 3 full CVs for a job.",
    response_format={"type": "json_object"},
    output_columns=["chosen_resume", "reasoning"],
    _build_messages_fn=_build_messages,
    _parse_response_fn=parse_pick_best,
)
