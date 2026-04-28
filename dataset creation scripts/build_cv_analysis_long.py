"""
Build a long-format table for analysis: one row per (design row × augmented CV slot).

Reads the wide CSV (e.g. added_lang_and_rel_cvs.csv), splits `markers` into three blocks
aligned with augmented_resume_1..3, and parses each block into columns (assumed-origin,
immigration-history, uni, religion, language, marker_header, plus any other `key: value` lines).
Legacy marker lines ``- nationality:`` / ``- origin:`` are mapped to the new names.

When loading the long CSV in pandas, use ``keep_default_na=False`` so the literal word
``None`` in ``language`` / ``religion`` is not turned into NaN.

Usage:
  python build_cv_analysis_long.py
  python build_cv_analysis_long.py -i added_lang_and_rel_cvs.csv -o final_CV_dataset_long.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from constants import AUGMENTED_RESUME_COLS, ROW_ID_COLS
from paths import DATA_DIR

# First line like: resume 1 (Original):  or  resume 2 (Rephrased 1):
_MARKER_HEADER = re.compile(r"^\s*resume\s+\d+\s*\(", re.I)

# ---------------------------------------------------------------------------
# JD-level extractors (profession title + seniority classification)
# ---------------------------------------------------------------------------

_PROFESSION_RE = re.compile(
    r"(?:\*{0,2})(?:Stellen|Berufs)bezeichnung(?:\*{0,2})\s*[:\uff1a]\s*(.+)",
    re.IGNORECASE,
)

_SENIORITY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Director / VP", re.compile(
        r"(?:Direktor|Vizepräsident|Vice President|Director|VP)", re.I)),
    ("Senior / Lead", re.compile(
        r"(?:Senior|Leitend|Lead|Sr\.)", re.I)),
    ("Manager", re.compile(
        r"(?:Manager|Leiter)", re.I)),
    ("Junior / Entry", re.compile(
        r"(?:Junior|Jr\.|Assistent|Einstieg|Associate|Angehend)", re.I)),
    ("Specialist / Analyst", re.compile(
        r"(?:Spezialist|Analyst|Koordinator|Berater|Experte|Prüfer)", re.I)),
]


def _extract_profession(jd_text: str) -> str | None:
    """Extract job title from the JD text."""
    m = _PROFESSION_RE.search(jd_text)
    if m:
        return m.group(1).strip().strip("*").strip()
    lines = [l.strip() for l in jd_text.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return None
    first_low = lines[0].strip().strip("*").strip().lower()
    if first_low in ("stellenbezeichnung", "berufsbezeichnung"):
        return lines[1].strip().strip("*").strip()
    if any(kw in lines[1].lower() for kw in (
        "unternehmensübersicht", "stellenzusammenfassung",
        "unternehmensüberblick", "unternehmensprofil",
        "job title", "abteilung",
    )):
        return lines[0].strip().strip("*").strip()
    return None


def _classify_seniority(title: str) -> str:
    """Map a job title to a seniority bucket."""
    for label, pat in _SENIORITY_PATTERNS:
        if pat.search(title):
            return label
    return "Other"


def parse_marker_block(block: str) -> dict[str, str]:
    """Parse one markers block (one CV slot). Bullet lines are `- key: value` or `key: value`."""
    out: dict[str, str] = {}
    lines = block.strip().split("\n")
    if lines:
        first = lines[0].strip()
        if _MARKER_HEADER.match(first):
            out["marker_header"] = first.rstrip(":").strip()
            lines = lines[1:]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("-"):
            line = line[1:].strip()
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        key = k.strip().lower().replace(" ", "_")
        # Legacy wide-table markers (pre-rename)
        if key == "nationality":
            key = "assumed-origin"
        elif key == "origin":
            key = "immigration-history"
        out[key] = v.strip()
    return out


def split_marker_sections(markers: str, n_slots: int) -> list[str]:
    """Split the full `markers` cell into n_slots blocks (blank-line separated)."""
    if markers is None or (isinstance(markers, float) and pd.isna(markers)):
        return [""] * n_slots
    parts = str(markers).split("\n\n")
    while len(parts) < n_slots:
        parts.append("")
    return parts[:n_slots]


def wide_to_long_analysis(
    df: pd.DataFrame,
    *,
    resume_cols: list[str] | None = None,
    row_id_cols: tuple[str, ...] = ROW_ID_COLS,
) -> pd.DataFrame:
    """
    Wide augmented CSV -> long DataFrame with `augmented_resume_text`, `marker_block`,
    parsed marker fields, and `augmentation_index` 1..3.
    """
    cols = resume_cols or list(AUGMENTED_RESUME_COLS)
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c!r}")
    if "markers" not in df.columns:
        raise ValueError("Missing column 'markers'")

    id_present = [c for c in row_id_cols if c in df.columns]
    # Columns copied unchanged to each long row (same value for all 3 slots)
    meta_cols = [
        c
        for c in df.columns
        if c not in cols and c != "markers"
    ]

    rows: list[dict] = []
    has_jd = "augmented_job_description" in df.columns
    for wide_pos, (_, row) in enumerate(df.iterrows()):
        sections = split_marker_sections(row["markers"], len(cols))

        profession: str | None = None
        seniority: str | None = None
        if has_jd:
            jd_text = str(row["augmented_job_description"])
            profession = _extract_profession(jd_text)
            seniority = _classify_seniority(profession) if profession else None

        for slot_i, col in enumerate(cols):
            idx = slot_i + 1
            block = sections[slot_i]
            parsed = parse_marker_block(block)
            resume_text = row[col]
            rec: dict = {
                "wide_row_pos": wide_pos,
                "augmentation_index": idx,
                "augmentation_column": col,
                "marker_block": block,
                "augmented_resume_text": resume_text,
                "resume_char_length": len(str(resume_text)),
                "jd_profession": profession,
                "jd_seniority": seniority,
            }
            for k in id_present:
                rec[k] = row[k]
            for k in meta_cols:
                rec[k] = row[k]
            rec.update(parsed)
            rows.append(rec)

    long_df = pd.DataFrame(rows)
    # Stable observation id for merges (string keys)
    parts = []
    for c in id_present:
        parts.append(long_df[c].astype(str))
    parts.append(long_df["augmentation_index"].astype(str))
    long_df.insert(0, "observation_id", parts[0].str.cat(parts[1:], sep="_"))

    # Sensible column order: ids, design, text, marker_block, parsed keys
    front = [
        "observation_id",
        "wide_row_pos",
        *id_present,
        "augmentation_index",
        "augmentation_column",
        "augmented_resume_text",
        "resume_char_length",
        "jd_profession",
        "jd_seniority",
        "marker_block",
    ]
    rest = [c for c in long_df.columns if c not in front]
    preferred_parsed = [
        "marker_header",
        "assumed-origin",
        "immigration-history",
        "uni",
        "religion",
        "language",
    ]
    tail = [c for c in preferred_parsed if c in rest] + sorted(
        c for c in rest if c not in preferred_parsed
    )
    long_df = long_df[[c for c in front if c in long_df.columns] + tail]
    return long_df


def build_from_csv(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    row_id_cols: tuple[str, ...] = ROW_ID_COLS,
) -> pd.DataFrame:
    input_path = Path(input_path)
    df = pd.read_csv(input_path)
    long_df = wide_to_long_analysis(df, row_id_cols=row_id_cols)
    if output_path is not None:
        out = Path(output_path)
        long_df.to_csv(out, index=False)
        print(f"Wrote {len(long_df)} rows -> {out.resolve()}")
    return long_df


def main() -> None:
    p = argparse.ArgumentParser(description="Wide augmented CVs -> long analysis CSV")
    p.add_argument(
        "-i",
        "--input",
        default=str(DATA_DIR / "added_lang_and_rel_cvs.csv"),
        help="Wide CSV path",
    )
    p.add_argument(
        "-o",
        "--output",
        default=str(DATA_DIR / "final_CV_dataset_long.csv"),
        help="Long analysis CSV path",
    )
    args = p.parse_args()
    build_from_csv(args.input, args.output)


if __name__ == "__main__":
    main()
