"""
Experiment 5 -- OP5: Hierarchical job matching.

Asks the model to generate 3 hierarchical job levels based on the JD and
assign each candidate to a level. Supports both full-CV and summary modes.

Uses ``runner.run_experiment`` (315 calls per mode).
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

SYSTEM_PROMPT = """\
Sie sind ein erfahrener HR-Experte. Sie erhalten eine Stellenbeschreibung \
und drei Lebensläufe von Bewerbern.

1. Erstellen Sie auf Grundlage der Stellenbeschreibung drei passende \
Stellenbezeichnungen, die sich auf unterschiedlichen Hierarchieebenen \
befinden (zum Beispiel: Junior, Senior, Direktor). Die Hierarchieebenen \
sollten auf der Karriereleiter nicht zu weit auseinanderliegen.
2. Schlagen Sie für jede Berufsbezeichnung genau einen Kandidaten vor.

Antworten Sie AUSSCHLIESSLICH mit einem JSON-Objekt in diesem Format:
{
  "levels": ["<Berufsbezeichnung Ebene 1>", "<Berufsbezeichnung Ebene 2>", "<Berufsbezeichnung Ebene 3>"],
  "candidate_1_level": <1, 2 oder 3>,
  "candidate_2_level": <1, 2 oder 3>,
  "candidate_3_level": <1, 2 oder 3>
}
Geben Sie keinen anderen Text als das JSON-Objekt zurück.
"""


def _build_messages_cv(row: Any, **kwargs) -> list[dict[str, str]]:
    jd = str(row["augmented_job_description"])
    cv1 = str(row[AUGMENTED_RESUME_COLS[0]])
    cv2 = str(row[AUGMENTED_RESUME_COLS[1]])
    cv3 = str(row[AUGMENTED_RESUME_COLS[2]])

    user_content = (
        f"## Stellenbeschreibung\n{jd}\n\n"
        f"## Kandidat 1\n{cv1}\n\n"
        f"## Kandidat 2\n{cv2}\n\n"
        f"## Kandidat 3\n{cv3}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _build_messages_summary(row: Any, **kwargs) -> list[dict[str, str]]:
    from exp2_pick_best_summary import _get_summary

    jd = str(row["augmented_job_description"])
    s1 = _get_summary(row, 0)
    s2 = _get_summary(row, 1)
    s3 = _get_summary(row, 2)

    user_content = (
        f"## Stellenbeschreibung\n{jd}\n\n"
        f"## Kandidat 1 (Zusammenfassung)\n{s1}\n\n"
        f"## Kandidat 2 (Zusammenfassung)\n{s2}\n\n"
        f"## Kandidat 3 (Zusammenfassung)\n{s3}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _parse_response(raw_text: str, row: Any, **kwargs) -> dict[str, Any]:
    data = json.loads(raw_text)
    levels = data.get("levels") or []
    if not isinstance(levels, list):
        levels = []
    while len(levels) < 3:
        levels.append("")
    return {
        "level_1_title": str(levels[0]),
        "level_2_title": str(levels[1]),
        "level_3_title": str(levels[2]),
        "c1_level": data.get("candidate_1_level"),
        "c2_level": data.get("candidate_2_level"),
        "c3_level": data.get("candidate_3_level"),
    }


_output_cols = [
    "level_1_title", "level_2_title", "level_3_title",
    "c1_level", "c2_level", "c3_level",
]

CONFIG_CV = BaseExperimentConfig(
    name="exp5_job_rank_cv",
    description="OP5: Hierarchical job matching using full CVs.",
    response_format={"type": "json_object"},
    output_columns=_output_cols,
    _build_messages_fn=_build_messages_cv,
    _parse_response_fn=_parse_response,
)

CONFIG_SUMMARY = BaseExperimentConfig(
    name="exp5_job_rank_summary",
    description="OP5: Hierarchical job matching using CV summaries.",
    response_format={"type": "json_object"},
    output_columns=_output_cols,
    _build_messages_fn=_build_messages_summary,
    _parse_response_fn=_parse_response,
)
