"""
Experiment 4 -- OP4: Competence scoring.

Asks the model to rate a single candidate on four competences
(social, technical, work_style, communication) using a 1-5 scale.
No job description is provided -- the model scores the candidate in isolation.

Runs once per CV slot (3 per wide row = 945 calls per mode).
Supports both full-CV and summary modes via ``CONFIG_CV`` / ``CONFIG_SUMMARY``.

Uses ``runner.run_per_cv_experiment``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent / "dataset creation scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from constants import AUGMENTED_RESUME_COLS
from experiment_configs import BaseExperimentConfig

COMPETENCES = ("social", "technical", "work_style", "communication")

SYSTEM_PROMPT = """\
Sie sind ein erfahrener HR-Experte. 
Ihre Aufgabe besteht darin, das Profil eines Bewerbers hinsichtlich der folgenden Kompetenzen zu bewerten und für jede Kompetenz eine Punktzahl zwischen 1 (niedrigste) und 5 (höchste) zu vergeben:

1. Sozialkompetenz: Teamarbeit, Empathie und konfliktlösungsorientiertes Handeln.
2. Fachkompetenz: Fachwissen und fachliche Qualifikationen. 
3. Arbeitsstil: Zuverlässigkeit, Eigenverantwortung und strukturiertes Arbeiten.
4. Kommunikationsstil: Professionelles Auftreten, klare Ausdrucksweise und die Fähigkeit, sich kommunikativ in das berufliche Umfeld zu integrieren.

Antworten Sie AUSSCHLIESSLICH mit
einem JSON-Objekt in diesem Format: {“social”: <int>, “technical”: <int>, “work_style”: <int>, “communication”: <int>}. 
Die Werte dürfen nur ganze Zahlen zwischen 1 und 5 sein.
Geben Sie keinen anderen Text als das JSON-Objekt zurück.
"""


def _build_messages_cv(row: Any, *, slot_index: int = 0) -> list[dict[str, str]]:
    col = AUGMENTED_RESUME_COLS[slot_index]
    cv_text = str(row[col])
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": cv_text},
    ]


def _build_messages_summary(row: Any, *, slot_index: int = 0) -> list[dict[str, str]]:
    from exp2_pick_best_summary import _get_summary

    summary_text = _get_summary(row, slot_index)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": summary_text},
    ]


def _parse_response(raw_text: str, row: Any, *, slot_index: int = 0) -> dict[str, Any]:
    data = json.loads(raw_text)
    return {comp: data.get(comp) for comp in COMPETENCES}


_output_cols = list(COMPETENCES)

CONFIG_CV = BaseExperimentConfig(
    name="exp4_competence_cv",
    description="OP4: Rate each candidate individually on 4 competences (1-5) using full CVs.",
    response_format={"type": "json_object"},
    output_columns=_output_cols,
    _build_messages_fn=_build_messages_cv,
    _parse_response_fn=_parse_response,
)

CONFIG_SUMMARY = BaseExperimentConfig(
    name="exp4_competence_summary",
    description="OP4: Rate each candidate individually on 4 competences (1-5) using CV summaries.",
    response_format={"type": "json_object"},
    output_columns=_output_cols,
    _build_messages_fn=_build_messages_summary,
    _parse_response_fn=_parse_response,
)
