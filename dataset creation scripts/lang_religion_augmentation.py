"""
Language and religion CV augmentation via Cohere (Step 5 of the pipeline).

Provides the main ``add_language_and_religion`` function (with resume / checkpoint /
max_rows support), plus retry helpers for failed slots and section-mismatch reruns.

The ``SYSTEM_INSTRUCTIONS`` prompt is specific to this step (CV editor role for
Sprachen + Ehrenamt/Religion). It must not be changed without re-running the pipeline.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from constants import AUGMENTED_RESUME_COLS, CHAT_MODEL, ROW_ID_COLS
from paths import DATA_DIR, checkpoint_path_for

RESUME_COLS = list(AUGMENTED_RESUME_COLS)

SYSTEM_INSTRUCTIONS = """\
### ROLLE: PRÄZISER CV-EDITOR (KORREKTUR-MODUS)
Deine Aufgabe ist es, einen Lebenslauf (CV) basierend auf Markern zu aktualisieren und bestehende Widersprüche zu korrigieren.

### SPRACH-MAPPING:
- Falls [SPRACHE] "Fluent" ist -> Nutze im CV: "Deutsch: Muttersprache"
- Falls [SPRACHE] "C2 Verhandlungssicher" ist -> Nutze im CV: "Deutsch: C2 (Verhandlungssicher)"
### ARBEITSSCHRITTE:

1. ABSCHNITT SPRACHEN (STRIKTE ERSETZUNG):
   - WENN [SPRACHE] NICHT "None" IST:
     - Suche im Text nach einem Sprach-Abschnitt (z. B. "Sprachen", "Sprachkenntnisse", "Fremdsprachen").
     - Wenn gefunden: Entferne ALLE vorhandenen Angaben zu "Deutsch" (z. B. "Grundkenntnisse", "B1", etc.) aus diesem Abschnitt und ersetze sie durch den neuen Wert aus dem [SPRACH-MAPPING]. 
     - Es darf am Ende nur EINE Angabe für Deutsch im Abschnitt stehen.
     - Wenn KEIN Abschnitt existiert: Erstelle am Ende (VOR den Referenzen) eine neue Überschrift "Sprachkenntnisse" und füge den Wert ein.
   - WENN [SPRACHE] "None" IST:
     - Entferne den gesamten Sprach-Abschnitt vollständig.

2. ABSCHNITT EHRENAMT (RELIGION):
   - WENN [RELIGION] "Volunteering (Muslim NGO)" ist:
     - Suche nach einem Abschnitt für Engagement (z. B. "Engagement", "Ehrenamt").
     - Wenn gefunden: Füge eine Zeile hinzu, die ein spezifisches Engagement beschreibt. Wähle dabei eine Tätigkeit, die STIMMIG ZUM BERUFLICHEN PROFIL des Lebenslaufs passt (z. B. eher administrative Aufgaben für Büroberufe oder praktische Hilfe für technische Berufe). Wähle variierend aus folgenden Bereichen:
        * Soziales/Bildung (z. B. Hausaufgabenhilfe, Jugendarbeit)
        * Verwaltung/Organisation (z. B. Kassenwart, Unterstützung in der Mitgliederverwaltung)
        * Events (z. B. Organisation von Wohltätigkeitsveranstaltungen oder Gemeindefesten)
        * Praktische Hilfe (z. B. handwerkliche Instandhaltung oder Logistik bei Hilfsprojekten)
        Beispiel für das Format: "- Ehrenamtlicher Helfer in der lokalen Moscheegemeinde (Bereich: Veranstaltungsmanagement)".     
     - Wenn NICHT gefunden: Erstelle vor den Referenzen eine neue Überschrift "Ehrenamtliches Engagement" und füge die Zeile dort ein.
   - WENN [RELIGION] "None" IST:
     - Ändere im Bereich Engagement absolut nichts.

### INTEGRITÄTS-REGELN:
- ABSCHNITTSERHALT (Pflicht): Jeder im [CV_TEXT] erkennbare inhaltliche Abschnitt (typische Überschriften z. B. Berufserfahrung, Ausbildung, Kenntnisse/Qualifikationen, Projekte, Referenzen, Engagement, persönliche Daten, Zertifikate, Interessen usw.) muss im Ausgangstext wieder als klar erkennbarer Abschnitt vorkommen. Entferne, verschmelze oder streiche keine vollständigen Abschnitte außerhalb der Sprach-Logik. Überschriften-Format (z. B. **fett**, # Markdown) wie im Original beibehalten, sofern vorhanden. Ausnahme nur für Sprachen/Sprachkenntnisse/Fremdsprachen: dort gelten ausschließlich die obigen Regeln zu [SPRACHE] (einschließlich vollständiger Entfernung des Sprach-Abschnitts bei „None").
- Überschreibe alte Sprachniveaus für Deutsch zwingend. Es dürfen keine doppelten Angaben (z. B. "Grundkenntnisse" UND "Muttersprache") stehen bleiben.
- Verändere KEIN EINZIGES WORT in der Berufserfahrung oder Ausbildung.
- Gib AUSSCHLIESSLICH den aktualisierten Lebenslauf-Text zurück. Keine Kommentare.
"""

CHAT_TEMPERATURE = 0.1

# ---------------------------------------------------------------------------
# Cohere client placeholder — set by the notebook via ``init_client(co)``
# before calling any function that hits the API.
# ---------------------------------------------------------------------------
_co = None


def init_client(client) -> None:
    """Register the notebook-level ``cohere.Client`` for API calls."""
    global _co
    _co = client


def _get_client():
    if _co is None:
        raise RuntimeError(
            "Call lang_religion_augmentation.init_client(co) before using API functions."
        )
    return _co


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_language_religion(marker_block: str) -> tuple[str, str]:
    lang_val, rel_val = "None", "None"
    for line in marker_block.split("\n"):
        low = line.lower()
        if "language:" in low:
            lang_val = line.split(":", 1)[-1].strip()
        if "religion:" in low:
            rel_val = line.split(":", 1)[-1].strip()
    return lang_val, rel_val


def _cohere_update_one_resume(
    col: str,
    current_cv_text: str,
    lang_val: str,
    rel_val: str,
    system_instructions: str,
    request_options: dict | None = None,
    log_context: str = "",
) -> tuple[str, str | None]:
    co = _get_client()
    user_msg = (
        f"[SPRACHE]: {lang_val}\n"
        f"[RELIGION]: {rel_val}\n"
        f"[CV_TEXT]:\n{current_cv_text}"
    )
    try:
        kwargs = dict(
            model=CHAT_MODEL,
            message=user_msg,
            preamble=system_instructions,
            temperature=CHAT_TEMPERATURE,
        )
        if request_options:
            kwargs["request_options"] = request_options
        response = co.chat(**kwargs)
        text = response.text.strip()
        return col, text
    except Exception as e:
        prefix = f"[{log_context}] " if log_context else ""
        print(f"{prefix}Error for column {col}: {e}", file=sys.stderr)
        return col, None


def _todo_mask_vs_done(df, done_df, row_id_cols: tuple[str, ...]) -> pd.Series:
    """Boolean mask: True where df row is not yet present in done_df (ids compared as str)."""
    for c in row_id_cols:
        if c not in done_df.columns:
            raise ValueError(f"Resume file missing id column {c!r}")
    done_keys = done_df[list(row_id_cols)].drop_duplicates().copy()
    for c in row_id_cols:
        done_keys[c] = done_keys[c].astype(str)
    left = df[list(row_id_cols)].copy()
    for c in row_id_cols:
        left[c] = left[c].astype(str)
    merged = left.merge(done_keys.assign(_done=1), on=list(row_id_cols), how="left")
    return merged["_done"].isna()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def add_language_and_religion(
    df,
    system_instructions,
    output_file=str(DATA_DIR / "added_lang_and_rel_cvs.csv"),
    checkpoint_every: int = 20,
    max_workers_per_row: int = 3,
    row_id_cols: tuple[str, ...] = ROW_ID_COLS,
    max_rows: int | None = None,
    resume: bool = False,
    sort_output_by_ids: bool = True,
):
    """
    For each row, update three resume variants in parallel (network-bound speedup).

    Pilot: resume=False, max_rows=5 -- first 5 rows only, writes output_file.
    Continue: resume=True -- loads output_file or checkpoint, skips matching
    row_id_cols, processes remaining rows, concatenates, writes output_file again.
    """
    checkpoint_path = str(checkpoint_path_for(output_file))

    done_df = None
    if resume:
        out_df = (
            pd.read_csv(output_file)
            if os.path.isfile(output_file) and os.path.getsize(output_file) > 0
            else None
        )
        ck_df = (
            pd.read_csv(checkpoint_path)
            if os.path.isfile(checkpoint_path) and os.path.getsize(checkpoint_path) > 0
            else None
        )
        if out_df is not None and ck_df is not None:
            done_df = ck_df if len(ck_df) > len(out_df) else out_df
            src = checkpoint_path if done_df is ck_df else output_file
            print(f"Resume: loaded {len(done_df)} rows from {src!r} (picked longer of output vs checkpoint)")
        elif ck_df is not None:
            done_df = ck_df
            print(f"Resume: loaded {len(done_df)} rows from {checkpoint_path!r}")
        elif out_df is not None:
            done_df = out_df
            print(f"Resume: loaded {len(done_df)} rows from {output_file!r}")

    if done_df is not None and len(done_df) > 0:
        mask = _todo_mask_vs_done(df, done_df, row_id_cols)
        todo_df = df.loc[mask].copy()
        final_rows: list[dict] = done_df.to_dict("records")
        print(
            f"Already saved: {len(done_df)} | Still to run this batch: {len(todo_df)} (of {len(df)} in input)"
        )
    else:
        todo_df = df.copy()
        final_rows = []
        if resume:
            print("Resume requested but no non-empty output/checkpoint found; running full input.")

    if max_rows is not None:
        todo_df = todo_df.iloc[: int(max_rows)].copy()
        print(f"max_rows={max_rows!r} → processing {len(todo_df)} rows in this invocation")

    total_rows = len(todo_df)
    workers = min(max_workers_per_row, len(RESUME_COLS))

    if total_rows == 0:
        print("Nothing new to process.")
        final_df = pd.DataFrame(final_rows)
        if len(final_df) and sort_output_by_ids:
            final_df = final_df.sort_values(list(row_id_cols)).reset_index(drop=True)
        if len(final_df):
            final_df.to_csv(output_file, index=False)
        return final_df

    n_prior = len(final_rows)
    print(
        f"API pass: {total_rows} row(s) this run, up to {workers} parallel calls per row | "
        f"rows already in file: {n_prior}"
    )

    for row_num, (idx, row) in enumerate(todo_df.iterrows(), start=1):
        row_keys_str = f"resume_id={row['resume_id']}"
        marker_sections = row["markers"].split("\n\n")
        while len(marker_sections) < len(RESUME_COLS):
            marker_sections.append("")

        updated_resumes: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i, col in enumerate(RESUME_COLS):
                current_cv_text = str(row[col])
                m_text = marker_sections[i]
                lang_val, rel_val = _parse_language_religion(m_text)
                futures.append(
                    executor.submit(
                        _cohere_update_one_resume,
                        col,
                        current_cv_text,
                        lang_val,
                        rel_val,
                        system_instructions,
                        None,
                        row_keys_str,
                    )
                )
            for fut in as_completed(futures):
                col_name, text = fut.result()
                if text is not None:
                    updated_resumes[col_name] = text
                else:
                    updated_resumes[col_name] = row[col_name]

        pct = row_num / total_rows * 100
        sys.stdout.write(
            f"\r[{row_num}/{total_rows}] {pct:5.1f}% | df_index={idx} | {row_keys_str} | total {n_prior + row_num}"
        )
        sys.stdout.flush()

        new_row = row.to_dict()
        new_row.update(updated_resumes)
        final_rows.append(new_row)

        if row_num % checkpoint_every == 0:
            pd.DataFrame(final_rows).to_csv(checkpoint_path, index=False)

    final_df = pd.DataFrame(final_rows)
    if sort_output_by_ids:
        final_df = final_df.sort_values(list(row_id_cols)).reset_index(drop=True)
    final_df.to_csv(output_file, index=False)

    print("\n" + "=" * 60)
    print(f"Done. Saved to: {output_file}")

    return final_df


# ---------------------------------------------------------------------------
# Retry / finder / section-mismatch rerun
# ---------------------------------------------------------------------------


def _slot_expects_augmentation(marker_block: str) -> bool:
    """
    True if this resume slot's markers imply the editor should change the CV
    (non-None language and/or non-neutral religion).
    """
    lang_val, rel_val = _parse_language_religion(marker_block)
    lang = (lang_val or "").strip().lower()
    rel_l = (rel_val or "").strip().lower()
    lang_is_none = lang in ("none", "")
    rel_is_neutral = rel_l == "none" or ("none" in rel_l and "neutral" in rel_l)
    return (not lang_is_none) or (not rel_is_neutral)


def find_unaugmented_resume_slots(
    output_df: pd.DataFrame,
    input_df: pd.DataFrame,
    row_id_cols: tuple[str, ...] = ROW_ID_COLS,
    resume_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Slots where output still equals input **and** markers say that slot should have
    been edited. ``df_index`` is the row label from ``input_df``.
    """
    cols = resume_cols or RESUME_COLS
    need = list(row_id_cols) + cols
    for name, df in [("output", output_df), ("input", input_df)]:
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise ValueError(f"{name} df missing columns: {miss}")
    if "markers" not in output_df.columns:
        raise ValueError("output_df must include a 'markers' column")

    left = output_df[[*row_id_cols, *cols, "markers"]].copy()
    right = input_df[[*row_id_cols, *cols]].copy()
    right["df_index"] = right.index
    right = right.rename(columns={c: f"{c}_in" for c in cols})
    for k in row_id_cols:
        left[k] = left[k].astype(str)
        right[k] = right[k].astype(str)
    merged = left.merge(right, on=list(row_id_cols), how="inner")

    records: list[dict] = []
    for col in cols:
        col_in = f"{col}_in"
        for _, row in merged.iterrows():
            if str(row[col]) != str(row[col_in]):
                continue
            marker_sections = str(row["markers"]).split("\n\n")
            while len(marker_sections) < len(RESUME_COLS):
                marker_sections.append("")
            slot_i = RESUME_COLS.index(col)
            if not _slot_expects_augmentation(marker_sections[slot_i]):
                continue
            records.append(
                {
                    "df_index": row["df_index"],
                    **{k: row[k] for k in row_id_cols},
                    "resume_column": col,
                }
            )

    if not records:
        return pd.DataFrame(columns=["df_index", *row_id_cols, "resume_column"])
    return pd.DataFrame(records).drop_duplicates().reset_index(drop=True)


def retry_failed_resume_slots(
    output_df: pd.DataFrame,
    input_df: pd.DataFrame,
    system_instructions: str,
    row_id_cols: tuple[str, ...] = ROW_ID_COLS,
    output_file: str = str(DATA_DIR / "added_lang_and_rel_cvs.csv"),
    chat_timeout_seconds: int = 180,
    max_retries_per_call: int = 4,
    sleep_between_retries_s: float = 2.0,
) -> pd.DataFrame:
    """Re-call Cohere only for failed slots; sequential calls; longer per-request timeout."""
    failed = find_unaugmented_resume_slots(output_df, input_df, row_id_cols)
    if failed.empty:
        print("No unaugmented slots found.")
        return output_df

    req_opts = {"timeout_in_seconds": chat_timeout_seconds, "max_retries": 2}
    out = output_df.copy()
    nfail = len(failed)
    print(f"Retrying {nfail} slot(s); HTTP timeout up to {chat_timeout_seconds}s per attempt")

    for i, fr in enumerate(failed.itertuples(index=False), start=1):
        key = {k: getattr(fr, k) for k in row_id_cols}
        col = fr.resume_column
        mask = pd.Series(True, index=out.index)
        for k, v in key.items():
            mask &= out[k].astype(str) == str(v)
        hits = out.loc[mask]
        if len(hits) != 1:
            print(
                f"Skip (expected 1 row): df_index={getattr(fr, 'df_index', '?')} resume_id={key['resume_id']} → {len(hits)} matches",
                file=sys.stderr,
            )
            continue
        idx = hits.index[0]
        row = out.loc[idx]
        marker_sections = str(row["markers"]).split("\n\n")
        while len(marker_sections) < len(RESUME_COLS):
            marker_sections.append("")
        slot_i = RESUME_COLS.index(col)
        lang_val, rel_val = _parse_language_religion(marker_sections[slot_i])
        current_cv_text = str(row[col])

        text = None
        for attempt in range(1, max_retries_per_call + 1):
            _, text = _cohere_update_one_resume(
                col,
                current_cv_text,
                lang_val,
                rel_val,
                system_instructions,
                request_options=req_opts,
            )
            if text is not None:
                break
            time.sleep(sleep_between_retries_s * attempt)
        if text is not None:
            out.at[idx, col] = text
            print(
                f"[{i}/{nfail}] OK df_index={fr.df_index} resume_id={key['resume_id']} {col}"
            )
        else:
            print(
                f"[{i}/{nfail}] still failing df_index={fr.df_index} resume_id={key['resume_id']} {col}",
                file=sys.stderr,
            )

    out.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")
    return out


def rerun_api_for_section_mismatch_report(
    mismatch_df,
    augmented_df,
    input_df,
    system_instructions: str,
    row_id_cols: tuple[str, ...] = ROW_ID_COLS,
    output_file: str = str(DATA_DIR / "added_lang_and_rel_cvs.csv"),
    chat_timeout_seconds: int = 240,
    max_retries_per_call: int = 4,
    sleep_between_retries_s: float = 2.0,
    use_input_df_text: bool = True,
    exclude_slots=None,
):
    """
    Re-call Cohere for each (row keys x resume_column) listed in the section-mismatch
    report. If ``use_input_df_text`` is True (default), the model sees the
    pre-augmentation CV text from ``input_df``.
    """
    if isinstance(mismatch_df, str):
        mismatch_df = pd.read_csv(mismatch_df)
    need = [*row_id_cols, "resume_column"]
    miss = [c for c in need if c not in mismatch_df.columns]
    if miss:
        raise ValueError(f"mismatch_df missing columns: {miss}")

    slots = mismatch_df[need].drop_duplicates().reset_index(drop=True)

    if exclude_slots is not None and len(exclude_slots):
        ex = (
            pd.DataFrame(exclude_slots)
            if isinstance(exclude_slots, list)
            else exclude_slots.copy()
        )
        ex_cols = list(row_id_cols) + ["resume_column"]
        bad = [c for c in ex_cols if c not in ex.columns]
        if bad:
            raise ValueError(f"exclude_slots missing columns: {bad}")
        ex = ex[ex_cols].drop_duplicates()
        for k in row_id_cols:
            ex[k] = ex[k].astype(str)
            slots[k] = slots[k].astype(str)
        slots = slots.merge(ex.assign(_ex=1), on=ex_cols, how="left")
        n_ex = int(slots["_ex"].notna().sum())
        slots = slots[slots["_ex"].isna()].drop(columns=["_ex"]).reset_index(drop=True)
        if n_ex:
            print(f"Excluded {n_ex} slot(s) from re-run (exclude_slots)")

    ns = len(slots)
    if ns == 0:
        print("No rows left to re-run after exclusions.")
        return augmented_df

    req_opts = {"timeout_in_seconds": chat_timeout_seconds, "max_retries": 2}
    out = augmented_df.copy()
    print(
        f"Section-mismatch re-run: {ns} unique slot(s); "
        f"timeout {chat_timeout_seconds}s; "
        f"source text = {'input_df (pre-aug)' if use_input_df_text else 'augmented cell'}"
    )

    for i, fr in enumerate(slots.itertuples(index=False), start=1):
        key = {k: getattr(fr, k) for k in row_id_cols}
        col = fr.resume_column
        if col not in RESUME_COLS:
            print(f"Skip unknown resume column {col!r}", file=sys.stderr)
            continue

        mask_out = pd.Series(True, index=out.index)
        mask_in = pd.Series(True, index=input_df.index)
        for k, v in key.items():
            mask_out &= out[k].astype(str) == str(v)
            mask_in &= input_df[k].astype(str) == str(v)
        hits = out.loc[mask_out]
        hits_in = input_df.loc[mask_in]
        if len(hits) != 1:
            print(f"Skip augmented: {key} → {len(hits)} row(s)", file=sys.stderr)
            continue
        if len(hits_in) != 1:
            print(f"Skip input_df: {key} → {len(hits_in)} row(s)", file=sys.stderr)
            continue

        idx = hits.index[0]
        row_out = out.loc[idx]
        row_in = hits_in.iloc[0]

        marker_sections = str(row_out["markers"]).split("\n\n")
        while len(marker_sections) < len(RESUME_COLS):
            marker_sections.append("")
        slot_i = RESUME_COLS.index(col)
        lang_val, rel_val = _parse_language_religion(marker_sections[slot_i])

        current_cv_text = (
            str(row_in[col]) if use_input_df_text else str(row_out[col])
        )

        text = None
        for attempt in range(1, max_retries_per_call + 1):
            _, text = _cohere_update_one_resume(
                col,
                current_cv_text,
                lang_val,
                rel_val,
                system_instructions,
                request_options=req_opts,
            )
            if text is not None:
                break
            time.sleep(sleep_between_retries_s * attempt)
        if text is not None:
            out.at[idx, col] = text
            print(f"[{i}/{ns}] OK resume_id={key['resume_id']} {col}")
        else:
            print(
                f"[{i}/{ns}] FAIL resume_id={key['resume_id']} {col}",
                file=sys.stderr,
            )

    out.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")
    return out
