"""
Replace CV placeholders using Cohere: for each of three resume columns, request a JSON map
from placeholder names to values (given markers + shared placeholder list), then substitute
into the CV text in Python.

After all model-based replacements, :func:`patch_plz_first_occurrence` runs on the three
``augmented_resume_*`` columns: any remaining ``[PLZ]`` is filled using the **first**
German city name (from a fixed map) that appears in that cell’s text, matching the
follow-up script you used when the model often omitted ``[PLZ]``.

The implementation matches the **legacy Colab script** in behavior: same system prompt,
same ``user_msg`` shape, same ``co.chat`` arguments (including ``response_format``), same
JSON parsing, same string replacement rules, same ``time.sleep``, checkpoint cadence, and
output columns.

Environment: set ``COHERE_API_KEY``, or pass a ``client`` into
:func:`run_placeholder_augmentation`.
"""

from __future__ import annotations

import json
import os
import random
import re
import string
import time
from typing import Any

import pandas as pd

from constants import (
    AUGMENTED_RESUME_COLS,
    CHAT_MODEL,
    COL_RESUME,
    COL_RESUME2,
    COL_RESUME3,
    RESPONSE_FORMAT_JSON,
    RESUME_SOURCE_COLS,
)
from paths import DATA_DIR, checkpoint_path_for

SYSTEM_INSTRUCTIONS = """### SYSTEM-ANWEISUNG: GENERIERUNG DIVERSER IDENTITÄTS-DATEN
Du bist ein kreativer Daten-Generator für eine wissenschaftliche Audit-Studie. Deine Aufgabe ist es, für eine übermittelte Liste von PLATZHALTERN realistische, männliche Identitätsdaten zu erstellen, die exakt zu den MARKERN passen.

### STRATEGISCHE VORGABEN:
- OUTPUT-FORMAT: Antworte AUSSCHLIESSLICH mit einem validen JSON-Objekt. Keine Einleitung, kein Fließtext.
- GESCHLECHT: Alle generierten Daten müssen MÄNNLICH sein (Namen, Titel, Bezeichnungen).
- ALPHA-SEED: Nutze den bereitgestellten Buchstaben zur Variation von Namen und Städten.
- WOHNSITZ-REGEL (WICHTIG): Da alle Kandidaten aktuell in Deutschland leben, müssen Platzhalter für den aktuellen Wohnsitz und Kontakt (z. B. [ADRESSE], [STADT], [POSTLEITZAHL], [TELEFON], [BUNDESLAND], [WOHNUNG], [EMAIL]) IMMER deutsche Daten und Formate enthalten.

### REGELN FÜR DIE IDENTITÄTS-MARKER:
1. NATIONALITÄT: Wähle klangvolle, realistische männliche Namen passend zur 'assumed-origin'. Variiere die Namen bei jedem Aufruf stark.
2. HERKUNFT (Origin):
   - [GEBURTSORT]: Wenn 'Foreign-born', wähle eine Stadt im Herkunftsland. Wenn 'Germany-born', wähle eine Stadt in Deutschland.
   - [SCHULE]: Wenn 'Foreign-born', eine Schule im Ausland. Wenn 'Germany-born', eine Schule in Deutschland.
3. BILDUNG (Uni):
   - 'German University': Eine reale Universität in Deutschland.
   - 'Foreign University': Eine reale Universität im Herkunftsland der Nationalität.
4. KONTAKT-DETAILS:
   - [TELEFON]: Nutze deutsche Mobilfunk- oder Festnetznummern.
   - [EMAIL]: Erstelle eine passende E-Mail-Adresse basierend auf dem generierten Namen (z. B. vorname.nachname@provider.de).

### HINWEIS:
- Erzeuge für JEDEN Platzhalter in der übergebenen Liste einen individuellen, passenden Wert.
"""


# City → PLZ (first occurrence in text wins for resolving [PLZ]); same map as the legacy patch script.
PLZ_CITY_MAP: dict[str, str] = {
    "München": "80331",
    "Berlin": "10117",
    "Hamburg": "20095",
    "Köln": "50667",
    "Frankfurt": "60311",
    "Stuttgart": "70173",
    "Düsseldorf": "40213",
    "Dortmund": "44135",
    "Essen": "45127",
    "Leipzig": "04109",
    "Bremen": "28195",
    "Dresden": "01067",
    "Hannover": "30159",
    "Nürnberg": "90402",
    "Duisburg": "47051",
    "Bochum": "44787",
    "Wuppertal": "42103",
    "Bielefeld": "33602",
    "Bonn": "53111",
    "Münster": "48143",
    "Karlsruhe": "76133",
    "Mannheim": "68159",
    "Augsburg": "86150",
    "Wiesbaden": "65183",
    "Gelsenkirchen": "45879",
    "Mönchengladbach": "41061",
    "Braunschweig": "38100",
    "Chemnitz": "09111",
    "Kiel": "24103",
    "Aachen": "52062",
    "Halle": "06108",
    "Magdeburg": "39104",
    "Freiburg": "79098",
    "Krefeld": "47798",
    "Mainz": "55116",
}


def _patch_plz_in_cell(content: str, city_map: dict[str, str]) -> tuple[str, bool]:
    """
    If ``content`` contains ``[PLZ]``, pick the postcode for the **earliest** matching
    city from ``city_map`` (case-insensitive) and replace all ``[PLZ]`` with it.

    Returns
    -------
    (new_text, applied)
        ``applied`` is True when a replacement was made, False when ``[PLZ]`` was present
        but no mapped city was found (text unchanged).
    """
    if "[PLZ]" not in content:
        return content, False

    found_city_plz: str | None = None
    earliest_pos = float("inf")

    for city, plz in city_map.items():
        match = re.search(re.escape(city), content, re.IGNORECASE)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()
            found_city_plz = plz

    if found_city_plz:
        return content.replace("[PLZ]", found_city_plz), True
    return content, False


def patch_plz_first_occurrence(
    df: pd.DataFrame,
    *,
    resume_cols: list[str] | None = None,
    city_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    For each augmented CV cell that still contains ``[PLZ]``, find which **mapped**
    German city name occurs **earliest** in the text (case-insensitive) and replace **all**
    ``[PLZ]`` in that cell with that city’s postcode.

    Mutates ``df`` in place (same as the original helper). Prints patch counts.

    Parameters
    ----------
    df
        Must contain the augmented resume columns (default ``augmented_resume_1`` … ``_3``).
    resume_cols
        Columns to scan; defaults to :data:`AUGMENTED_RESUME_COLS`.
    city_map
        Defaults to :data:`PLZ_CITY_MAP`.

    Returns
    -------
    The same ``df`` reference, mutated.
    """
    cols = resume_cols or AUGMENTED_RESUME_COLS
    cmap = city_map if city_map is not None else PLZ_CITY_MAP

    patch_count = 0
    no_fix = 0

    for idx, row in df.iterrows():
        for col in cols:
            if col not in df.columns:
                continue
            content = str(row[col])

            if "[PLZ]" not in content:
                continue

            new_text, applied = _patch_plz_in_cell(content, cmap)
            if applied:
                df.at[idx, col] = new_text
                patch_count += 1
            else:
                no_fix += 1

    print("Processing complete.")
    print(f"   - Patched with valid PLZ: {patch_count}")
    print(f"   - Not fixed: {no_fix}")
    return df


def report_leftover_plz(
    df: pd.DataFrame,
    *,
    resume_cols: list[str] | None = None,
) -> int:
    """
    Count augmented cells that still contain the literal ``[PLZ]`` after patching.
    (Replaces ad-hoc ``check_for_leftovers`` when that helper is not defined.)
    """
    cols = resume_cols or AUGMENTED_RESUME_COLS
    n = 0
    for _idx, row in df.iterrows():
        for col in cols:
            if col in df.columns and "[PLZ]" in str(row[col]):
                n += 1
    print(f"Cells still containing [PLZ]: {n}")
    return n


def _normalize_placeholder_key(placeholder: str) -> str:
    """Legacy behavior: keys must match bracketed tokens in the CV text."""
    return placeholder if placeholder.startswith("[") else f"[{placeholder}]"


def _apply_replacement_map(
    current_text: str,
    replacement_map: dict[str, Any],
) -> str:
    """Substitute each key in the map into ``current_text`` (string values)."""
    for placeholder, replacement_value in replacement_map.items():
        p_key = _normalize_placeholder_key(placeholder)
        current_text = current_text.replace(p_key, str(replacement_value))
    return current_text


def _cohere_placeholder_json(
    client,
    *,
    instructions: str,
    alpha_seed: str,
    marker_block: str,
    placeholder_list: Any,
) -> str:
    """
    Single chat call: system instructions + user message with ALPHA-SEED, MARKERS, placeholders.
    Returns raw JSON string from the assistant message.
    """
    user_msg = (
        f"ALPHA-SEED: {alpha_seed}\n"
        f"MARKERS: {marker_block}\n"
        f"GENERATE VALUES FOR THESE PLACEHOLDERS: {placeholder_list}"
    )
    response = client.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_msg},
        ],
        response_format=RESPONSE_FORMAT_JSON,
    )
    return response.message.content[0].text


def run_placeholder_augmentation(
    df: pd.DataFrame,
    instructions: str,
    output_file: str = str(DATA_DIR / "augmented_cvs.csv"),
    client=None,
    *,
    patch_plz: bool = True,
    report_plz_leftovers: bool = True,
) -> pd.DataFrame:
    """
    For each dataframe row, call the model three times (once per CV slot) and build
    ``augmented_resume_1` … ``_3`` with placeholders replaced.

    After all rows are augmented, :func:`patch_plz_first_occurrence` runs on the result
    (unless ``patch_plz=False``) so ``[PLZ]`` is resolved from the earliest city name in
    each cell. Optionally :func:`report_leftover_plz` prints how many cells still contain
    ``[PLZ]``. Intermediate checkpoint CSVs (every 10 index labels) are written **before**
    the PLZ patch; the final ``output_file`` is fully patched.

    Parameters
    ----------
    df
        Must include ``markers``, ``placeholders``, ``jd_placeholders``, ``jd``,
        ``resume_id``, ``permutation_id``, ``baseline_at_pos``, and the three resume
        columns ``resume``, ``resume 2 ``, ``resume 3``.
    instructions
        System prompt (same role as ``SYSTEM_INSTRUCTIONS`` in the legacy script).
    output_file
        Final CSV path; checkpoint every 10 **index labels** ``idx`` is written to
        ``checkpoint_<output_file>`` (same naming as the original).
    client
        ``cohere.Client`` instance. If ``None``, uses ``COHERE_API_KEY`` from the
        environment.
    patch_plz
        If ``True`` (default), run ``[PLZ]`` post-processing on the three augmented columns.
    report_plz_leftovers
        If ``True`` (default), print how many augmented cells still contain ``[PLZ]``.

    Returns
    -------
    DataFrame
        Assembled rows with ``augmented_resume_1` … ``_3``; also written to ``output_file``.
    """
    import cohere

    if client is None:
        key = os.environ.get("COHERE_API_KEY")
        if not key:
            raise RuntimeError("Set COHERE_API_KEY or pass client=")
        client = cohere.Client(key)

    required = [
        "markers",
        "placeholders",
        "jd_placeholders",
        "jd",
        "resume_id",
        "permutation_id",
        "baseline_at_pos",
        *RESUME_SOURCE_COLS,
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"df must contain column {c!r}")

    final_rows: list[dict[str, Any]] = []
    alphabet = string.ascii_uppercase
    n_rows = len(df)

    print(f"Starting placeholder-based augmentation for {n_rows} rows...")

    for idx, row in df.iterrows():
        print(f"Processing Row {idx + 1}/{n_rows} (ID: {row['resume_id']})...")

        marker_sections = row["markers"].split("\n\n")
        resumes = [row[COL_RESUME], row[COL_RESUME2], row[COL_RESUME3]]
        placeholder_list = row["placeholders"]

        augmented_cvs: list[str] = []

        for i in range(3):
            alpha_seed = random.choice(alphabet)

            try:
                raw_json = _cohere_placeholder_json(
                    client,
                    instructions=instructions,
                    alpha_seed=alpha_seed,
                    marker_block=marker_sections[i],
                    placeholder_list=placeholder_list,
                )

                if idx < 5:
                    print(raw_json)

                replacement_map = json.loads(raw_json)
                current_text = str(resumes[i])
                current_text = _apply_replacement_map(current_text, replacement_map)
                augmented_cvs.append(current_text)

            except Exception as e:
                print(f"Error at Row {idx}, CV {i + 1}: {e}")
                augmented_cvs.append("REPLACEMENT_FAILED")

            time.sleep(0.1)

        new_row = {
            "resume_id": row["resume_id"],
            "placeholders": row["placeholders"],
            "jd_placeholders": row["jd_placeholders"],
            "jd": row["jd"],
            "permutation_id": row["permutation_id"],
            "baseline_at_pos": row["baseline_at_pos"],
            "markers": row["markers"],
            "augmented_resume_1": augmented_cvs[0],
            "augmented_resume_2": augmented_cvs[1],
            "augmented_resume_3": augmented_cvs[2],
        }
        final_rows.append(new_row)

        if (idx + 1) % 10 == 0:
            pd.DataFrame(final_rows).to_csv(
                str(checkpoint_path_for(output_file)), index=False
            )

    output_df = pd.DataFrame(final_rows)
    if patch_plz:
        patch_plz_first_occurrence(output_df)
    if report_plz_leftovers:
        report_leftover_plz(output_df)
    output_df.to_csv(output_file, index=False)
    print(f"Success! Augmented CVs saved to {output_file}")
    return output_df


__all__ = [
    "CHAT_MODEL",
    "SYSTEM_INSTRUCTIONS",
    "COL_RESUME",
    "COL_RESUME2",
    "COL_RESUME3",
    "RESUME_SOURCE_COLS",
    "PLZ_CITY_MAP",
    "patch_plz_first_occurrence",
    "report_leftover_plz",
    "run_placeholder_augmentation",
]
