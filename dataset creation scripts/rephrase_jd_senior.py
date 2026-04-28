"""
Rephrase each job description one hierarchical level above.

Reads ``augmented_job_description`` from the experiment dataset, calls Cohere
to produce a senior-level version, and writes the result back as a new column
``augmented_job_description_senior``.

Deduplicates so each unique JD text is only rephrased once, then broadcasts
back to all rows that share it.  Checkpoints after every LLM call.

Usage (from the repo root)::

    python "dataset creation scripts/rephrase_jd_senior.py"

Requires ``COHERE_API_KEY`` env var.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _THIS_DIR.parent / "experiments"
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from runner import call_model, init_client, CHAT_MODEL, PROVIDER_COHERE

DATA_DIR = _THIS_DIR.parent / "data"
WIDE_CSV = DATA_DIR / "final_CV_dataset_experiments_one_row.csv"
LONG_CSV = DATA_DIR / "final_CV_dataset_long.csv"
CACHE_FILE = DATA_DIR / "jd_senior_cache.json"

COL_ORIG = "augmented_job_description"
COL_SENIOR = "augmented_job_description_senior"

SYSTEM_PROMPT = """\
Sie sind ein erfahrener HR-Experte. Sie erhalten eine Stellenbeschreibung. \
Ihre Aufgabe ist es, diese so umzuformulieren, dass sie sich auf eine Position \
bezieht, die eine Hierarchiestufe höher liegt.

Wichtig:
- Ändern Sie keine sachlichen Angaben wie Firmenname, Adresse, Standort usw.
- Ändern Sie weder die beschriebene Berufsbezeichnung noch die Branche.
- Die Bezeichnung für die hierarchisch höhere Ebene als in der ursprünglichen \
Stellenbeschreibung ist branchenspezifisch. Verwenden Sie die entsprechenden Begriffe.
- Behalten Sie alle Anforderungen der ursprünglichen Stellenbeschreibung bei, \
heben Sie sie lediglich eine Stufe an.
- Behalten Sie die Wortwahl und den Schreibstil der ursprünglichen \
Stellenbeschreibung bei.
- Geben Sie nur die Stellenbeschreibung zurück und nichts anderes.
"""


def _jd_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _load_cache() -> dict[str, str]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"  WARNING: corrupt cache file ({exc}), starting fresh")
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def rephrase_jd(client, jd_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": jd_text},
    ]
    return call_model(client, messages=messages, model=CHAT_MODEL)


def run() -> None:
    client = init_client(PROVIDER_COHERE)
    wide = pd.read_csv(WIDE_CSV, keep_default_na=False)

    unique_jds = wide[COL_ORIG].unique()
    print(f"Unique JDs to rephrase: {len(unique_jds)} (from {len(wide)} rows)")

    cache = _load_cache()
    done = sum(1 for jd in unique_jds if _jd_hash(jd) in cache)
    print(f"Already cached: {done}/{len(unique_jds)}")

    for i, jd_text in enumerate(unique_jds, 1):
        h = _jd_hash(jd_text)
        if h in cache:
            continue
        print(f"[{i}/{len(unique_jds)}] Rephrasing JD (hash={h[:8]})...")
        try:
            senior_jd = rephrase_jd(client, jd_text)
            if not senior_jd.strip():
                print(f"  WARNING: empty response for hash={h[:8]}, skipping")
                continue
            cache[h] = senior_jd
            _save_cache(cache)
            time.sleep(0.1)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

    print(f"\nCache complete: {len(cache)} entries. Writing columns...")

    wide[COL_SENIOR] = wide[COL_ORIG].map(lambda jd: cache.get(_jd_hash(jd), ""))
    missing = (wide[COL_SENIOR] == "").sum()
    if missing:
        print(f"  WARNING: {missing} rows have no senior JD (LLM errors)")
    wide.to_csv(WIDE_CSV, index=False)
    print(f"  Wrote {COL_SENIOR} to {WIDE_CSV}")

    long = pd.read_csv(LONG_CSV, keep_default_na=False)
    if COL_ORIG in long.columns:
        long[COL_SENIOR] = long[COL_ORIG].map(lambda jd: cache.get(_jd_hash(jd), ""))
        long.to_csv(LONG_CSV, index=False)
        print(f"  Wrote {COL_SENIOR} to {LONG_CSV}")
    else:
        jd_map = wide[["resume_id", COL_SENIOR]].drop_duplicates("resume_id")
        long = long.merge(jd_map, on="resume_id", how="left")
        long.to_csv(LONG_CSV, index=False)
        print(f"  Merged {COL_SENIOR} via resume_id into {LONG_CSV}")

    print("Done.")


if __name__ == "__main__":
    run()
