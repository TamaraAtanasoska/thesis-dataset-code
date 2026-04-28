"""
Generate ``resume 2 `` and ``resume 3`` from ``resume`` via a single Cohere chat that
returns two lexically distinct German CV versions (“Version 1” / “Version 2”).

This mirrors the **original Colab-style script**: same prompt string, same
``co.chat(..., messages=[...])`` shape (two user turns, no system message), same
``response.message.content[0].text`` access, and the same string splits to recover
the two CVs. Refactoring only structures the code; it does not change those pieces.

**Typical use**

.. code-block:: python

    import cohere
    import pandas as pd
    from rephrase_resume_variants import rephrase_resumes_two_versions

    co = cohere.Client(os.environ["COHERE_API_KEY"])
    df = pd.read_csv("rephrased_cvs_full.csv")
    rephrase_resumes_two_versions(df, client=co)

Environment: ``COHERE_API_KEY`` must be set if you omit ``client``.
"""

from __future__ import annotations

import os
import sys
import pandas as pd

from constants import CHAT_MODEL, COL_RESUME, COL_RESUME2, COL_RESUME3

REPHRASE_PROMPT_TEXT = (
    "Formulieren Sie den folgenden Lebenslauf zweimal um. Die Umformulierung sollte zu "
    "zwei Versionen des ursprünglichen Lebenslaufs führen, die inhaltlich identisch, aber "
    "lexikalisch unterschiedlich sind. Ändern Sie den Wortschatz und die Satzstrukturen, "
    "behalten Sie jedoch alle Angaben zu fachlichen und sozialen Kompetenzen, den "
    "quantifizierten Ergebnissen, dem Verantwortungsumfang sowie sonstigen sachlichen "
    "Informationen unverändert bei. Die Platzhalter, also die Wörter in eckigen Klammern, "
    "sollten sowohl inhaltlich als auch in ihrer Anzahl unverändert gegenüber dem Original "
    "bleiben. Fügen Sie keine neuen Informationen hinzu. Die Ergebnisse sollten stilistisch "
    "dem Original entsprechen, d. h. es sollte sich um ein gültiges, professionelles Format "
    "für einen deutschen Lebenslauf handeln. Geben Sie nur den umformulierten Inhalt der "
    "Lebensläufe aus und kennzeichnen Sie die Versionen mit „Version 1“ und „Version 2“."
)

def parse_double_rephrase_response(response_text: str) -> tuple[str | None, str | None]:
    """
    Split the model reply into Version 1 and Version 2 bodies.

    **Logic (unchanged from the original loop)**

    The model is asked to label output with the substrings ``Version 1`` and ``Version 2``.
    We split on ``Version 1``, take everything after it, then split on ``Version 2``:
    text before the second split is Version 1, text after is Version 2. Leading/trailing
    whitespace, colons, and asterisks are stripped the same way as in the original.

    Returns
    -------
    (version_1, version_2)
        Both ``None`` if either marker is missing from ``response_text``.
    """
    if "Version 1" not in response_text or "Version 2" not in response_text:
        return None, None

    parts_v1 = response_text.split("Version 1", 1)
    remaining_text = parts_v1[1]
    parts_v2 = remaining_text.split("Version 2", 1)

    cv_version_1 = parts_v2[0].strip().strip(":").strip("*").strip()
    cv_version_2 = parts_v2[1].strip().strip(":").strip("*").strip()
    return cv_version_1, cv_version_2


def _cohere_double_rephrase_raw(client, resume_to_rephrase: str) -> str:
    """
    Single chat call: instruction prompt + CV text. Returns raw assistant string.

    **API shape (legacy)**

    Two consecutive ``user`` messages — first the task, then the CV — and no ``temperature``
    argument, matching the snippet you ran in Colab.
    """
    res = client.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "user", "content": REPHRASE_PROMPT_TEXT},
            {"role": "user", "content": resume_to_rephrase},
        ],
    )
    return res.message.content[0].text


def rephrase_resumes_two_versions(
    df: pd.DataFrame,
    client=None,
    *,
    resume_col: str = COL_RESUME,
    resume2_col: str = COL_RESUME2,
    resume3_col: str = COL_RESUME3,
    id_col: str = "resume_id",
    skip_if_both_filled: bool = True,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """
    For each row, if needed, call Cohere once and fill the two rephrased columns.

    Parameters
    ----------
    df
        Must contain ``resume_col``, ``resume2_col``, ``resume3_col``, and ``id_col``
        (for logging). **Mutated in place** via ``df.at[...]``, same as the original script.
    client
        ``cohere.Client`` instance. If ``None``, builds ``cohere.Client(os.environ["COHERE_API_KEY"])``.
    resume_col, resume2_col, resume3_col
        Column names; defaults match legacy data (note space in ``resume 2 ``).
    id_col
        Used only in ``print`` messages on success, failure, or parse error.
    skip_if_both_filled
        If ``True``, skip a row when **both** rephrase columns are already non-null (original behavior).
    max_rows
        Process at most this many rows that would otherwise be processed (useful for pilots).

    Returns
    -------
    The same dataframe reference (mutated).
    """
    import cohere

    if client is None:
        key = os.environ.get("COHERE_API_KEY")
        if not key:
            raise RuntimeError("Set COHERE_API_KEY or pass client=")
        client = cohere.Client(key)

    required = [resume_col, resume2_col, resume3_col, id_col]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"df must contain column {c!r}")

    processed = 0
    for index, row in df.iterrows():
        if max_rows is not None and processed >= max_rows:
            break

        if skip_if_both_filled and pd.notnull(row[resume2_col]) and pd.notnull(row[resume3_col]):
            continue

        resume_to_rephrase = row[resume_col]

        try:
            response_text = _cohere_double_rephrase_raw(client, str(resume_to_rephrase))
            print(f"Rephrased CV for {id_col} {row[id_col]}")

            cv_version_1, cv_version_2 = parse_double_rephrase_response(response_text)

            if cv_version_1 is not None and cv_version_2 is not None:
                df.at[index, resume2_col] = cv_version_1
                df.at[index, resume3_col] = cv_version_2
            else:
                print(f"Could not parse versions for ID {row[id_col]}")
        except Exception as e:
            print(f"Error processing ID {row[id_col]}: {e}", file=sys.stderr)

        processed += 1

    return df


__all__ = [
    "REPHRASE_PROMPT_TEXT",
    "parse_double_rephrase_response",
    "rephrase_resumes_two_versions",
]
