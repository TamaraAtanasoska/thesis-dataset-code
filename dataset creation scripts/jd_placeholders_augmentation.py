"""
Fill job-description placeholders via Cohere JSON responses (even/odd row index → small vs
large German city logic in the system prompt).

Behavior matches the **legacy script**: same ``SYSTEM_INSTRUCTIONS``, same ``user_msg``
layout (``INDEX``, placeholder list, JD text), same ``co.chat`` arguments, same key
normalization and ``[KEY]`` replacement, same ``city_type`` / ``assigned_city`` rules,
``time.sleep(0.05)``, and same fallback on exception (row copied without JD changes).

Set ``COHERE_API_KEY`` or pass ``client=cohere.Client(...)``.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import pandas as pd

from constants import CHAT_MODEL, RESPONSE_FORMAT_JSON

SYSTEM_INSTRUCTIONS = """### SYSTEM-ANWEISUNG: GENERIERUNG VON BRANCHENSPEZIFISCHEN ARBEITGEBER-DATEN
Du bist ein Daten-Generator für eine wissenschaftliche Audit-Studie. Deine Aufgabe ist es, Platzhalter in einer STELLENANZEIGE mit realistischen deutschen Daten zu füllen.

### GEOGRAFISCHE LOGIK (STRICKT EINHALTEN):
Du erhältst einen INDEX für die aktuelle Zeile:
- Wenn der INDEX GERADE (even) ist (0, 2, 4...): Wähle eine reale deutsche KLEINSTADT oder eine LÄNDLICHE REGION (unter 50.000 Einwohner).
- Wenn der INDEX UNGERADE (odd) ist (1, 3, 5...): Wähle eine reale deutsche GROSSSTADT (über 250.000 Einwohner).
- Die Stadt muss in Deutschland liegen und zur Branche der Anzeige passen.

### FORMAT-REGELN FÜR PLATZHALTER:
- Die Platzhalter im Text sind durch eckige Klammern gekennzeichnet, z. B. [FIRMA], [STADT], [STRASSE].
- Du musst für JEDEN übermittelten Platzhalter einen passenden Wert generieren.

### STRIKTE AUSGABE-VORGABE:
- Antworte AUSSCHLIESSLICH mit einem validen JSON-Objekt.
- Erstelle KEINE Einleitung, KEINEN Fließtext und KEINE Erklärungen.
- Das JSON-Objekt muss die Platzhalter-Namen als Keys enthalten (ohne die Klammern im Key, z. B. {"FIRMA": "Beispiel GmbH"}).
- Nenne zwingend den Schlüssel "STADT" im JSON.

### BRANCHEN-FIT:
- Der Firmenname ([FIRMA]) und die Details müssen professionell und glaubwürdig für die jeweilige Branche der Stellenanzeige sein.
"""

def _build_jd_user_message(idx: Any, placeholders_needed: Any, jd_text: str) -> str:
    """Same string layout as the original ``user_msg``."""
    return (
        f"INDEX: {idx}\n"
        f"LIST OF PLACEHOLDERS TO FILL: {placeholders_needed}\n\n"
        f"JOB DESCRIPTION CONTEXT:\n{jd_text}"
    )


def _apply_jd_replacements(jd_text: str, replacements: dict[str, Any]) -> str:
    """Replace ``[KEY]`` tokens using JSON keys (strip optional brackets from keys)."""
    augmented_jd = jd_text
    for key, val in replacements.items():
        clean_key = key.strip("[]")
        placeholder_tag = f"[{clean_key}]"
        augmented_jd = augmented_jd.replace(placeholder_tag, str(val))
    return augmented_jd


def _city_type_from_index(idx: Any) -> str:
    """Even index → Small/Rural, odd → Big/Diverse (same ``idx % 2`` test as the original)."""
    return "Small/Rural" if idx % 2 == 0 else "Big/Diverse"


def _assigned_city_from_replacements(replacements: dict[str, Any]) -> str:
    """First value whose key uppercases to STADT, else ``Unknown``."""
    return next((v for k, v in replacements.items() if k.upper() == "STADT"), "Unknown")


def _cohere_jd_json(client, *, instructions: str, user_msg: str) -> str:
    response = client.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_msg},
        ],
        response_format=RESPONSE_FORMAT_JSON,
    )
    return response.message.content[0].text


def augment_jds_final_version(
    df: pd.DataFrame,
    instructions: str,
    client=None,
) -> pd.DataFrame:
    """
    One API call per row: JSON map of placeholder keys → values, then substitute into ``jd``.

    Parameters
    ----------
    df
        Must include ``jd``, ``jd_placeholders``, and ``resume_id`` (for progress prints).
    instructions
        Typically :data:`SYSTEM_INSTRUCTIONS`.
    client
        ``cohere.Client``; if ``None``, built from ``COHERE_API_KEY``.

    Returns
    -------
    New dataframe with updated ``jd``, plus ``city_type`` and ``assigned_city`` on success.
    On API/parse errors, the original row dict is appended unchanged.
    """
    import cohere

    if client is None:
        key = os.environ.get("COHERE_API_KEY")
        if not key:
            raise RuntimeError("Set COHERE_API_KEY or pass client=")
        client = cohere.Client(key)

    for c in ("jd", "jd_placeholders", "resume_id"):
        if c not in df.columns:
            raise ValueError(f"df must contain column {c!r}")

    final_rows: list[dict[str, Any]] = []
    n = len(df)

    print(f"Starting JD Augmentation for {n} rows...")

    for idx, row in df.iterrows():
        print(f"Processing JD for row {idx + 1}/{n} (Resume ID: {row['resume_id']})...")

        jd_text = str(row["jd"])
        placeholders_needed = row["jd_placeholders"]
        user_msg = _build_jd_user_message(idx, placeholders_needed, jd_text)

        try:
            raw = _cohere_jd_json(client, instructions=instructions, user_msg=user_msg)
            replacements = json.loads(raw)

            augmented_jd = _apply_jd_replacements(jd_text, replacements)
            city_type = _city_type_from_index(idx)
            assigned_city = _assigned_city_from_replacements(replacements)

            new_row = row.to_dict()
            new_row.update(
                {
                    "jd": augmented_jd,
                    "city_type": city_type,
                    "assigned_city": assigned_city,
                }
            )
            final_rows.append(new_row)

        except Exception as e:
            print(f"Error at Index {idx}: {e}")
            final_rows.append(row.to_dict())

        time.sleep(0.05)

    return pd.DataFrame(final_rows)


__all__ = [
    "SYSTEM_INSTRUCTIONS",
    "augment_jds_final_version",
]
