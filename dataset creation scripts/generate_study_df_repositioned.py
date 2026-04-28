"""
Construction of ``permutation_id`` and ``baseline_at_pos`` (and rotated CV texts / markers).

This module implements the **study design expansion**: 105 input rows (one triple of CV
texts per person) → **315 rows** by taking each person’s three linguistic variants
(Original / Rephrased 1 / Rephrased 2) and **crossing them** with **three physical
rotations** of which identity sits in column 1, 2, and 3.

The rotation logic, treatment pools, marker strings, and use of the global ``random``
module are kept **algorithmically identical** to the original script so that, with the
same input CSV and the same RNG state, you get the same ``permutation_id``,
``baseline_at_pos``, ``markers``, and column contents.

**Identifiers (what they mean for analysis)**

- **``permutation_id``** ∈ {1, 2, 3}: which **cyclic shift** was applied to both
  (identities, source texts) together for this row. It is ``p_idx + 1`` where
  ``p_idx`` ∈ {0, 1, 2}:

  - ``permutation_id == 1``: shift 0 — slot 1 gets identity/text that was index 0,
    slot 2 gets index 1, slot 3 gets index 2.
  - ``permutation_id == 2``: shift 1 — slot 1 gets what was index 1, etc.
  - ``permutation_id == 3``: shift 2 — slot 1 gets what was index 2, etc.

  So the **baseline identity** (German / Germany-born / …) moves to a **different
  physical column** depending on ``permutation_id``.

- **``baseline_at_pos``** ∈ {1, 2, 3}: **after** that rotation, in which **column
  position** (1 = first resume column, 2 = second, 3 = third) the **Baseline**
  persona (the neutral German reference identity) appears. This is what you need to
  align “same text slot” vs “same identity role” across permutations.

**Input columns (required)**

- ``resume``, ``resume 2 `` (note trailing space), ``resume 3``: the three linguistic
  versions for that person, in fixed semantic order **before** rotation.

**Output columns (added / overwritten)**

- ``resume``, ``resume 2 ``, ``resume 3``: texts **after** rotation for this
  ``permutation_id`` (aligned with markers).
- ``markers``: three blocks separated by ``\\n\\n``, one per column, describing the
  identity in that column for this row.
- ``permutation_id``, ``baseline_at_pos``: as above.

For full detail see the docstring of :func:`generate_study_df_repositioned`.
"""

from __future__ import annotations

import random
from itertools import product
from typing import Any

import pandas as pd

from constants import COL_RESUME, COL_RESUME2, COL_RESUME3, RESUME_SOURCE_COLS

# Pool sizes: 70 treatment-A draws + 140 treatment-B draws = 210 = 2 × 105 persons
N_POOL_A = 70
N_POOL_BC = 140
N_PERSONS = 105
N_PERMUTATIONS = 3

# Version labels attached to *text slots* in marker blocks (linguistic role)
VERSION_INDEX_TO_LABEL = {0: "Original", 1: "Rephrased 1", 2: "Rephrased 2"}


def _build_treatment_pool() -> list[tuple[Any, ...]]:
    """
    Build the shuffled list of 210 treatment tuples consumed two-at-a-time per person.

    **Structure (unchanged from original)**

    - **Cluster A** (70 slots): nationalities ``Turkish``, ``Syrian``, ``Iraq`` crossed
      with origins, universities, and religion including ``Volunteering (Muslim NGO)``.
    - **Cluster B/C** (140 slots): nationalities ``Russian``, ``Ukrainian``, …,
      ``Japanese`` with the same origin × uni grid but religion fixed to
      ``None (Neutral)``.

    Lists are truncated to exactly ``N_POOL_A`` and ``N_POOL_BC`` by repeating full
    Cartesian products and slicing — same arithmetic as the legacy script.
    """
    cluster_a = ["Turkish", "Syrian", "Iraq"]
    cluster_bc = ["Russian", "Ukrainian", "Serbian", "Indian", "Chinese", "Japanese"]
    origins = ["Foreign-born", "Germany-born (Migrationshintergrund)"]
    unis = ["German University", "Foreign University"]

    combos_a = list(
        product(cluster_a, origins, unis, ["Volunteering (Muslim NGO)", "None (Neutral)"])
    )
    pool_a = (combos_a * (N_POOL_A // len(combos_a) + 1))[:N_POOL_A]

    combos_bc = list(product(cluster_bc, origins, unis, ["None (Neutral)"]))
    pool_bc = (combos_bc * (N_POOL_BC // len(combos_bc) + 1))[:N_POOL_BC]

    treatment_pool = pool_a + pool_bc
    random.shuffle(treatment_pool)
    return treatment_pool


def _assign_language_for_identity(ident: dict[str, str]) -> None:
    """
    Mutate ``ident`` in place with a ``lang`` key (side effect, legacy behavior).

    **Rules (unchanged)**

    - Baseline is handled separately (always ``Fluent``).
    - For **Treatment A** and **Treatment B**: if the **immigration-history** marker is ``Foreign-born``,
      language is chosen at random between ``C2 Verhandlungssicher`` and ``None``;
      otherwise (Germany-born Migrationshintergrund) language is ``Fluent``.

    Uses :func:`random.choice` in the same loop order as the original (only for
    ``identities[1:]``, two identities per person).
    """
    if ident["ori"] == "Foreign-born":
        ident["lang"] = random.choice(["C2 Verhandlungssicher", "None"])
    else:
        ident["lang"] = "Fluent"


def _rotate_lists(items: list, shift: int) -> list:
    """Left-rotate ``items`` by ``shift`` positions (``shift`` in {0,1,2})."""
    return items[shift:] + items[:shift]


def _make_marker_blocks(
    rotated_idents: list[dict[str, str]],
    p_idx: int,
) -> list[str]:
    """
    Build the three marker strings for columns 1..3 after rotation.

    Each block starts with a line ``resume {pos} ({version_label}):`` where
    ``version_label`` encodes **which linguistic variant** (Original vs rephrases)
    is physically in that column **for this permutation**, not which identity.

    The index formula ``current_version_idx = ((pos - 1) + p_idx) % 3`` ties the
    label to the rotation: when ``p_idx`` increases, which column shows "Original"
    shifts accordingly.
    """
    blocks: list[str] = []
    for pos in range(1, 4):
        ident = rotated_idents[pos - 1]
        current_version_idx = ((pos - 1) + p_idx) % 3
        v_label = VERSION_INDEX_TO_LABEL[current_version_idx]
        block = (
            f"resume {pos} ({v_label}):\n"
            f"- assumed-origin: {ident['nat']}\n"
            f"- immigration-history: {ident['ori']}\n"
            f"- uni: {ident['uni']}\n"
            f"- religion: {ident['rel']}\n"
            f"- language: {ident['lang']}"
        )
        blocks.append(block)
    return blocks


def generate_study_df_repositioned(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand **105** base rows to **315** by crossing each row with three permutations.

    **What this function does (step by step)**

    1. **Treatment randomization (once per call)**  
       Build ``treatment_pool`` (210 tuples), shuffle it globally with
       ``random.shuffle``, then iterate with ``next(...)`` **twice per input row** to
       assign Treatment A and Treatment B identities. Order of consumption matches the
       original: row 0 takes tuple 0 and 1, row 1 takes 2 and 3, …

    2. **Three identities per person**  
       - **Baseline**: fixed German reference (neutral religion, fluent language).  
       - **Treatment A**: from first drawn tuple (assumed-origin, immigration-history, uni, religion).  
       - **Treatment B**: from second drawn tuple.

    3. **Language on treatments**  
       For the two non-baseline identities, apply :func:`_assign_language_for_identity`
       (random choice for Foreign-born immigration-history). Baseline language is set to ``Fluent``.

    4. **Three output rows per input row** (``p_idx`` = 0, 1, 2)  
       For each ``p_idx``, **left-rotate** both ``identities`` and ``source_texts`` by
       ``p_idx`` positions so that column ``resume`` / ``resume 2 `` / ``resume 3``
       always match the identity described in the corresponding marker block.

    5. **Metadata**  
       - ``permutation_id = p_idx + 1``  
       - ``baseline_at_pos``: 1-based index of the slot where the Baseline identity
         ended up after this rotation (search in ``rotated_idents`` for ``type ==
         "Baseline"``).

    **Reproducibility**

    The implementation uses the global ``random`` module (``shuffle``, ``choice``) like
    the legacy code. To reproduce a past run you must set the same RNG state **before**
    calling this function, e.g. ``random.seed(…)`` if you had recorded the seed.

    **Parameters**
    ----------
    df
        Dataframe with exactly **105** rows expected by the design (not enforced here),
        and columns ``resume``, ``resume 2 ``, ``resume 3`` plus any other columns to
        copy through (each copied to all three expanded rows).

    **Returns**
    -------
    DataFrame
        **315** rows: for each original row, three rows distinguished by
        ``permutation_id`` ∈ {1,2,3} and differing ``baseline_at_pos``, ``markers``,
        and rotated resume texts.
    """
    for c in RESUME_SOURCE_COLS:
        if c not in df.columns:
            raise ValueError(f"Input df must contain column {c!r}")

    treatment_pool = _build_treatment_pool()
    treatment_iter = iter(treatment_pool)

    expanded_rows: list[dict[Any, Any]] = []

    for _i, row in df.iterrows():
        source_texts = [row[COL_RESUME], row[COL_RESUME2], row[COL_RESUME3]]

        t1 = next(treatment_iter)
        t2 = next(treatment_iter)

        identities: list[dict[str, str]] = [
            {
                "type": "Baseline",
                "nat": "German",
                "ori": "Germany-born",
                "uni": "German University",
                "rel": "None (Neutral)",
            },
            {
                "type": "Treatment A",
                "nat": t1[0],
                "ori": t1[1],
                "uni": t1[2],
                "rel": t1[3],
            },
            {
                "type": "Treatment B",
                "nat": t2[0],
                "ori": t2[1],
                "uni": t2[2],
                "rel": t2[3],
            },
        ]

        for ident in identities[1:]:
            _assign_language_for_identity(ident)
        identities[0]["lang"] = "Fluent"

        for p_idx in range(N_PERMUTATIONS):
            rotated_idents = _rotate_lists(identities, p_idx)
            rotated_texts = _rotate_lists(list(source_texts), p_idx)

            new_row = row.copy()
            new_row[COL_RESUME] = rotated_texts[0]
            new_row[COL_RESUME2] = rotated_texts[1]
            new_row[COL_RESUME3] = rotated_texts[2]

            marker_blocks = _make_marker_blocks(rotated_idents, p_idx)
            new_row["permutation_id"] = p_idx + 1
            new_row["baseline_at_pos"] = next(
                idx for idx, x in enumerate(rotated_idents, start=1) if x["type"] == "Baseline"
            )
            new_row["markers"] = "\n\n".join(marker_blocks)
            expanded_rows.append(new_row)

    return pd.DataFrame(expanded_rows)


__all__ = [
    "generate_study_df_repositioned",
]
