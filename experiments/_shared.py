"""Shared helpers for experiment modules — eliminates duplication."""

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

NO_CHOICE = -1


def parse_pick_best(raw_text: str, row: Any, **kwargs) -> dict[str, Any]:
    """Parse JSON with ``chosen_resume`` and ``reasoning`` (exp1/2/3/6)."""
    data = json.loads(raw_text)
    chosen = data.get("chosen_resume")
    return {
        "chosen_resume": int(chosen) if chosen is not None else NO_CHOICE,
        "reasoning": str(data.get("reasoning") or ""),
    }


def build_cv_user_block(row: Any, jd_col: str = "augmented_job_description") -> str:
    """Format JD + 3 full CVs as a user message string."""
    jd = str(row[jd_col])
    cv1 = str(row[AUGMENTED_RESUME_COLS[0]])
    cv2 = str(row[AUGMENTED_RESUME_COLS[1]])
    cv3 = str(row[AUGMENTED_RESUME_COLS[2]])
    return (
        f"## Stellenbeschreibung\n{jd}\n\n"
        f"## Kandidat 1\n{cv1}\n\n"
        f"## Kandidat 2\n{cv2}\n\n"
        f"## Kandidat 3\n{cv3}"
    )


def build_summary_user_block(
    row: Any,
    get_summary_fn,
    jd_col: str = "augmented_job_description",
) -> str:
    """Format JD + 3 summaries as a user message string."""
    jd = str(row[jd_col])
    s1 = get_summary_fn(row, 0)
    s2 = get_summary_fn(row, 1)
    s3 = get_summary_fn(row, 2)
    return (
        f"## Stellenbeschreibung\n{jd}\n\n"
        f"## Kandidat 1 (Zusammenfassung)\n{s1}\n\n"
        f"## Kandidat 2 (Zusammenfassung)\n{s2}\n\n"
        f"## Kandidat 3 (Zusammenfassung)\n{s3}"
    )
