#!/usr/bin/env python3
"""
Compare original vs augmented CV texts: report rows where the augmented version
is missing section headers that appear in the original.

Language-related sections (Sprachen, Sprachkenntnisse, etc.) are excluded from
the requirement, since the augmentation prompt may remove them entirely.

Headings are matched after normalizing markdown (``#``, ``**…**``, ``*…*``) and
simple subtitle splits (em dash), so formatting differences should not cause
false "missing section" flags.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Iterable

import pandas as pd

from constants import AUGMENTED_RESUME_COLS, ROW_ID_COLS
from paths import DATA_DIR

# Canonical section buckets: (canonical_id, keywords matching heading start / whole line)
# Order: more specific before general (e.g. sprachkenntnisse before kenntnisse).
SECTION_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    (
        "language",
        (
            "sprachkenntnisse",
            "sprachen",
            "fremdsprachen",
            "language skills",
            "languages",
        ),
    ),
    (
        "work",
        (
            "berufserfahrung",
            "berufliche erfahrung",
            "berufliche tätigkeit",
            "werdegang",
            "beruflicher werdegang",
            "praktische erfahrung",
            "praktika",
            "praktikum",
        ),
    ),
    (
        "education",
        (
            "ausbildung",
            "studium",
            "akademischer werdegang",
            "akademische ausbildung",
            "schulbildung",
            "schulausbildung",
        ),
    ),
    (
        "skills_general",
        (
            "it-kenntnisse",
            "edv-kenntnisse",
            "computerkenntnisse",
            "softwarekenntnisse",
            "digitale kompetenzen",
            "fachkenntnisse",
            "qualifikationen",
            "kompetenzen",
            "fähigkeiten",
            "kenntnisse und fähigkeiten",
        ),
    ),
    (
        "engagement",
        (
            "ehrenamt",
            "ehrenamtliches engagement",
            "soziales engagement",
            "engagement",
            "freiwilligenarbeit",
        ),
    ),
    (
        "personal",
        (
            "persönliche daten",
            "persönliches profil",
            "über mich",
            "profil",
            "kurzprofil",
            "steckbrief",
        ),
    ),
    (
        "references",
        (
            "referenzen",
            "anlagen",
        ),
    ),
    (
        "projects",
        ("projekte", "projektübersicht", "ausgewählte projekte"),
    ),
    (
        "certs",
        ("zertifikate", "zertifizierungen", "weiterbildungen", "fortbildungen"),
    ),
    (
        "publications",
        ("publikationen", "veröffentlichungen"),
    ),
    (
        "interests",
        ("interessen", "hobbys", "private interessen"),
    ),
]

# "kenntnisse" alone is ambiguous; treat only if not already classified
_FALLBACK_KEYWORDS = ("kenntnisse",)

LANGUAGE_CANONICALS = {"language"}

# Human-readable names for report columns (canonical id -> description)
CANONICAL_LABELS: dict[str, str] = {
    "language": "Language / Sprachen (ignored in requirement)",
    "work": "Work experience — Berufserfahrung, Werdegang, Praktika, …",
    "education": "Education — Ausbildung, Studium, …",
    "skills_general": "Skills / Kenntnisse — IT, Qualifikationen, Fähigkeiten, …",
    "engagement": "Engagement / Ehrenamt",
    "personal": "Personal / Profil — Persönliche Daten, Über mich, …",
    "references": "References / Referenzen, Anlagen",
    "projects": "Projects / Projekte",
    "certs": "Certificates / Zertifikate, Weiterbildung",
    "publications": "Publications / Publikationen",
    "interests": "Interests / Hobbys",
}


def format_missing_sections_readable(missing: Iterable[str]) -> str:
    """Semicolon-separated labels for missing canonical section ids."""
    parts = []
    for cid in sorted(missing):
        parts.append(CANONICAL_LABELS.get(cid, cid))
    return "; ".join(parts)


def _normalize_heading_line(line: str) -> str:
    """Strip markdown / noise so ``**Berufserfahrung**`` and ``Berufserfahrung`` match."""
    s = line.strip()
    s = re.sub(r"^#+\s*", "", s).strip()
    # unwrap ** ... ** (repeat for nested junk)
    for _ in range(8):
        prev = s
        s = re.sub(r"^\*\*\s*(.+?)\s*\*\*\s*$", r"\1", s, flags=re.DOTALL)
        if s == prev:
            break
    # single-asterisk wrap *Heading* (whole line)
    s = s.strip()
    s_star = re.sub(r"^\*([^*]+)\*$", r"\1", s)
    if s_star != s:
        s = s_star.strip()
    s = s.strip("*").strip()
    s = s.lower().rstrip(":")
    # title often has an em dash subtitle: "über mich – kurzprofil"
    s = re.split(r"\s+[—–]\s+|\s+\|\s+", s, maxsplit=1)[0].strip()
    s = re.sub(r"\s+[—–\-]\s*$", "", s)
    return s.strip()


def _line_looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 120:
        return False
    # Sentence-like (likely body text)
    if s.endswith((".", "!", "?", ";")) and len(s) > 55:
        return False
    if len(s) < 2:
        return False
    # Markdown heading: do not treat leading ** as a bullet list
    if s.startswith("**"):
        return True
    if s.startswith("#"):
        return True
    # list bullets / enumerations (single * or - followed by space)
    if re.match(r"^[\d\)\]\}\•►▪▸]+\s", s):
        return False
    if re.match(r"^[\-\*]\s+\S", s):
        return False
    return True


def _classify_heading(norm: str) -> str | None:
    """Return canonical section id, or None if not matched."""
    if not norm:
        return None
    head = re.split(r"\s+[—–]\s+|\s+\|\s+", norm, maxsplit=1)[0].strip()
    for cid, kws in SECTION_GROUPS:
        for kw in kws:
            if head == kw or head.startswith(kw + " ") or head.startswith(kw + "\t"):
                return cid
            if norm == kw or norm.startswith(kw + " ") or norm.startswith(kw + "\t"):
                return cid
    for kw in _FALLBACK_KEYWORDS:
        if head == kw or head.startswith(kw + " ") or head.startswith(kw + "\t"):
            return "skills_general"
        if norm == kw or norm.startswith(kw + " ") or norm.startswith(kw + "\t"):
            return "skills_general"
    return None


def extract_section_canonicals(text: str) -> set[str]:
    """Collect canonical section ids implied by heading-like lines in the CV."""
    if not isinstance(text, str) or not text.strip():
        return set()
    text = text.replace("\r\n", "\n")
    found: set[str] = set()

    # First line of each non-empty paragraph
    for block in re.split(r"\n\s*\n+", text):
        block = block.strip()
        if not block:
            continue
        first = block.split("\n", 1)[0]
        if _line_looks_like_heading(first):
            n = _normalize_heading_line(first)
            cid = _classify_heading(n)
            if cid:
                found.add(cid)

    # Standalone heading lines (same line repeated blocks already covered; catches lone headers)
    for line in text.split("\n"):
        ls = line.strip()
        if not _line_looks_like_heading(ls):
            continue
        n = _normalize_heading_line(ls)
        cid = _classify_heading(n)
        if cid:
            found.add(cid)

    return found


def required_sections_for_comparison(orig_text: str) -> set[str]:
    """Sections the original has that we expect to survive augmentation (excl. language)."""
    all_c = extract_section_canonicals(orig_text)
    return {c for c in all_c if c not in LANGUAGE_CANONICALS}


def missing_sections(orig_text: str, aug_text: str) -> set[str]:
    req = required_sections_for_comparison(orig_text)
    aug = extract_section_canonicals(aug_text)
    return req - aug


def _aug_index_label_for_keys(
    augmented_df: pd.DataFrame,
    row_keys: list[str],
    key_row: pd.Series,
) -> object | None:
    """Row label in ``augmented_df`` matching ``row_keys`` (avoids merge column quirks)."""
    m = pd.Series(True, index=augmented_df.index)
    for k in row_keys:
        m &= augmented_df[k].astype(str) == str(key_row[k])
    hits = augmented_df.index[m]
    return hits[0] if len(hits) else None


def compare_dataframes(
    original_df: pd.DataFrame,
    augmented_df: pd.DataFrame,
    row_keys: Iterable[str] = ROW_ID_COLS,
    resume_cols: Iterable[str] = AUGMENTED_RESUME_COLS,
) -> pd.DataFrame:
    """Return one row per problematic (keys, column) with missing canonical sections."""
    row_keys = list(row_keys)
    resume_cols = list(resume_cols)

    for c in row_keys + resume_cols:
        if c not in original_df.columns:
            raise ValueError(f"original_df missing column {c!r}")
        if c not in augmented_df.columns:
            raise ValueError(f"augmented_df missing column {c!r}")

    o = original_df[row_keys + resume_cols].copy()
    a = augmented_df[row_keys + resume_cols].copy()
    for k in row_keys:
        o[k] = o[k].astype(str)
        a[k] = a[k].astype(str)

    merged = o.merge(
        a,
        on=row_keys,
        how="inner",
        suffixes=("_orig", "_aug"),
    )

    records: list[dict] = []
    for col in resume_cols:
        co, ca = f"{col}_orig", f"{col}_aug"
        for _, row in merged.iterrows():
            miss = missing_sections(str(row[co]), str(row[ca]))
            if not miss:
                continue
            rec = {k: row[k] for k in row_keys}
            rec["aug_df_index"] = _aug_index_label_for_keys(
                augmented_df, row_keys, row
            )
            rec["resume_column"] = col
            rec["n_missing"] = len(miss)
            rec["missing_sections_readable"] = format_missing_sections_readable(miss)
            rec["missing_canonical_sections"] = ",".join(sorted(miss))
            records.append(rec)

    if not records:
        return pd.DataFrame(
            columns=[
                *row_keys,
                "aug_df_index",
                "resume_column",
                "n_missing",
                "missing_sections_readable",
                "missing_canonical_sections",
            ]
        )
    out = pd.DataFrame(records)
    return (
        out.sort_values(
            by=["aug_df_index", "resume_column"],
            na_position="last",
        ).reset_index(drop=True)
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--original",
        default=str(DATA_DIR / "augmented_cvs_full.csv"),
        help="CSV with pre-augmentation CVs (same columns as notebook input).",
    )
    p.add_argument(
        "--augmented",
        default=str(DATA_DIR / "added_lang_and_rel_cvs.csv"),
        help="CSV with augmented CVs (notebook output).",
    )
    p.add_argument(
        "--out",
        default=str(DATA_DIR / "cv_section_mismatch_report.csv"),
        help="Write report CSV (empty if no issues).",
    )
    args = p.parse_args()

    if not os.path.isfile(args.original):
        print(f"Missing original CSV: {args.original}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.augmented):
        print(f"Missing augmented CSV: {args.augmented}", file=sys.stderr)
        return 1

    orig = pd.read_csv(args.original)
    aug = pd.read_csv(args.augmented)

    # Notebook drops jd and merges JD fields into pre_language frame; original file may still have jd
    common_resume = [c for c in AUGMENTED_RESUME_COLS if c in orig.columns and c in aug.columns]
    if len(common_resume) != len(AUGMENTED_RESUME_COLS):
        missing = set(AUGMENTED_RESUME_COLS) - set(common_resume)
        print(f"Warning: missing resume columns: {missing}", file=sys.stderr)

    resume_use = common_resume if common_resume else list(AUGMENTED_RESUME_COLS)
    report = compare_dataframes(orig, aug, resume_cols=resume_use)
    report.to_csv(args.out, index=False)

    _o = orig[list(ROW_ID_COLS)].astype(str)
    _a = aug[list(ROW_ID_COLS)].astype(str)
    n_aligned = len(_o.merge(_a, on=list(ROW_ID_COLS), how="inner"))
    print(f"Rows aligned on row keys: {n_aligned}")
    print(f"Problematic (keys × column) slots: {len(report)}")
    if len(report):
        print(report.head(20).to_string(index=False))
        if len(report) > 20:
            print(f"... ({len(report) - 20} more in {args.out})")
    else:
        print("No missing non-language sections detected (by heading heuristics).")
    print(f"Full report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
