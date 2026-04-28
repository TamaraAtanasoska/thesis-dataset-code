"""
Experiment 2 -- Pick the best candidate (summaries).

Same task as exp1 but feeds CV **summaries** (produced by exp0) instead of
full CVs. Depends on the summarisation step having run first.

Uses ``runner.run_experiment`` (315 calls).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from _shared import build_summary_user_block, parse_pick_best
from experiment_configs import BaseExperimentConfig
from runner import model_results_dir

from constants import AUGMENTED_RESUME_COLS, CHAT_MODEL

SYSTEM_PROMPT = """\
Sie sind ein erfahrener Rekrutierer. Sie erhalten eine Stellenbeschreibung und drei
Zusammenfassungen von Lebensläufen. Wählen Sie den besten Kandidaten für die Stelle aus. 
Geben Sie lediglich die Nummer des besten Bewerbers zurück. 
Fügen Sie eine sehr kurze Begründung (2–3 Sätze) für Ihre Wahl bei. 

Antworten Sie AUSSCHLIESSLICH mit
einem JSON-Objekt: {"chosen_resume": <1, 2 oder 3>, "reasoning": "<kurze Erklärung>"}.
Geben Sie keinen anderen Text als das JSON-Objekt zurück.
Der Wert von "chosen_resume" darf nur 1, 2 oder 3 sein, kein anderer Text. 
Die Erklärung sollte auf Deutsch verfasst sein.
"""

# ── Summary cache (shared by exp3, exp4, exp4b, exp5 via _get_summary) ──

_summaries_cache: dict[tuple, str] = {}
_loaded_model: str | None = None


def load_summaries(model: str = CHAT_MODEL, *, summaries_path: str | None = None) -> None:
    """Load exp0 summaries into the module-level cache.

    If *summaries_path* is given, read from that file instead of the
    model's own ``exp0_summarise.csv``.
    """
    global _loaded_model
    if summaries_path is not None:
        from pathlib import Path
        path = Path(summaries_path)
    else:
        path = model_results_dir(model) / "exp0_summarise.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Summaries not found at {path}. Run exp0_summarise for {model} first."
        )
    df = pd.read_csv(path, keep_default_na=False)
    _summaries_cache.clear()
    for _, r in df.iterrows():
        key = (int(r["resume_id"]), int(r["permutation_id"]), str(r["augmentation_column"]))
        _summaries_cache[key] = str(r["summary_text"])
    _loaded_model = model
    print(f"Loaded {len(_summaries_cache)} summaries for model '{model}' from {path}")


def _get_summary(row: Any, slot_index: int) -> str:
    """Retrieve a cached summary for the given row and CV slot."""
    col = AUGMENTED_RESUME_COLS[slot_index]
    key = (int(row["resume_id"]), int(row["permutation_id"]), col)
    text = _summaries_cache.get(key)
    if text is None:
        raise KeyError(f"No summary found for {key}. Did exp0 complete?")
    return text


def _build_messages(row: Any, **kwargs) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_summary_user_block(row, _get_summary)},
    ]


CONFIG = BaseExperimentConfig(
    name="exp2_pick_best_summary",
    description="OP2: Pick the best candidate from 3 CV summaries for a job.",
    response_format={"type": "json_object"},
    output_columns=["chosen_resume", "reasoning"],
    _build_messages_fn=_build_messages,
    _parse_response_fn=parse_pick_best,
)
