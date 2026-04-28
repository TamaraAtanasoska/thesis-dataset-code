"""Shared constants for the dataset creation pipeline.

Centralises column names, the model identifier, and other values that were previously
duplicated across multiple scripts. Each pipeline module still owns its own
``SYSTEM_INSTRUCTIONS`` (the prompts are different per step).
"""

from __future__ import annotations

CHAT_MODEL = "command-a-03-2025"

# Pre-augmentation resume columns (legacy names; note trailing space on "resume 2 ")
COL_RESUME = "resume"
COL_RESUME2 = "resume 2 "
COL_RESUME3 = "resume 3"
RESUME_SOURCE_COLS = (COL_RESUME, COL_RESUME2, COL_RESUME3)

# Post-augmentation resume columns (after placeholder / language-religion edits)
AUGMENTED_RESUME_COLS = ("augmented_resume_1", "augmented_resume_2", "augmented_resume_3")

# Design-key triple used for merges, checkpoints, and deduplication
ROW_ID_COLS = ("resume_id", "permutation_id", "baseline_at_pos")

# Parsed marker traits in long-format analysis CSV (hyphenated column names)
COL_ASSUMED_ORIGIN = "assumed-origin"
COL_IMMIGRATION_HISTORY = "immigration-history"
MARKER_TRAIT_COLS = (
    COL_ASSUMED_ORIGIN,
    COL_IMMIGRATION_HISTORY,
    "uni",
    "religion",
    "language",
)

# Cohere response_format for structured JSON output
RESPONSE_FORMAT_JSON: dict[str, str] = {"type": "json_object"}
