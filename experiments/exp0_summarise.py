"""
Experiment 0 -- CV Summarisation pre-step.

Calls the model once per CV slot (3 per wide row = 945 total) to generate a
summary of each augmented resume. The summaries are later consumed by exp2,
exp3, and exp5 when running in summary mode.

Uses ``runner.run_per_cv_experiment``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent / "dataset creation scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from constants import AUGMENTED_RESUME_COLS
from experiment_configs import BaseExperimentConfig


SYSTEM_PROMPT = """\
Sie übernehmen die Rolle eines erfahrenen Personalvermittlers. 
Ihre Aufgabe ist es, den eingegangenen Lebenslauf in einer prägnanten beruflichen Zusammenfassung zusammenzufassen.
Die Zusammenfassung sollte alle wesentlichen Qualifikationen, Erfahrungen und Fähigkeiten enthalten.
Die Zusammenfassung sollte die besonderen Stärken des Bewerbers hervorheben.
Die Zusammenfassung sollte auf Deutsch verfasst sein.
Fassen Sie den Lebenslauf in Ihren eigenen Worten zusammen. Übernehmen Sie keine Formulierungen wörtlich aus dem Lebenslauf.
Verwenden Sie ein abwechslungsreiches, professionelles Vokabular. Vermeiden Sie formelhafte oder sich wiederholende Formulierungen.
Liefern Sie nur die Zusammenfassung ab, keinen weiteren Text.
Die Zusammenfassung sollte maximal 6 Sätze umfassen. 
"""


def _build_messages(row: Any, *, slot_index: int = 0) -> list[dict[str, str]]:
    col = AUGMENTED_RESUME_COLS[slot_index]
    cv_text = str(row[col])
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": cv_text},
    ]


def _parse_response(raw_text: str, row: Any, *, slot_index: int = 0) -> dict[str, Any]:
    return {"summary_text": raw_text.strip()}


CONFIG = BaseExperimentConfig(
    name="exp0_summarise",
    description="Generate a summary for each augmented CV (945 calls total).",
    response_format=None,
    output_columns=["summary_text"],
    _build_messages_fn=_build_messages,
    _parse_response_fn=_parse_response,
)
