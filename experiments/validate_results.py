"""
Validate experiment result files for missing, incomplete, or erroneous output.

Usage:
    python validate_results.py                          # check all models
    python validate_results.py command-a-03-2025        # check one model

Can also be imported and used from a notebook:
    from validate_results import validate_model, validate_all, rerun_bad_rows
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent / "dataset creation scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from paths import DATA_DIR, RESULTS_DIR

EXPECTED_WIDE_ROWS = 315
EXPECTED_PER_CV_ROWS = 945  # 315 * 3

# These resume_ids have JD/CV mismatches in the original JobResQA source data.
# Models correctly refuse to pick a candidate, so rerunning is pointless.
MISMATCH_RESUME_IDS = {1275, 1285, 1295, 1301, 1336, 1369, 1373}

_COMPETENCES = ("social", "technical", "work_style", "communication")

EXPERIMENT_SPECS: dict[str, dict[str, Any]] = {
    "exp0_summarise": {
        "type": "per_cv",
        "expected_rows": EXPECTED_PER_CV_ROWS,
        "key_cols": ["resume_id", "permutation_id", "augmentation_column"],
        "output_cols": ["summary_text"],
    },
    "exp1_pick_best_cv": {
        "type": "wide",
        "expected_rows": EXPECTED_WIDE_ROWS,
        "key_cols": ["resume_id", "permutation_id"],
        "output_cols": ["chosen_resume", "reasoning"],
    },
    "exp2_pick_best_summary": {
        "type": "wide",
        "expected_rows": EXPECTED_WIDE_ROWS,
        "key_cols": ["resume_id", "permutation_id"],
        "output_cols": ["chosen_resume", "reasoning"],
    },
    "exp3_leadership_cv": {
        "type": "wide",
        "expected_rows": EXPECTED_WIDE_ROWS,
        "key_cols": ["resume_id", "permutation_id"],
        "output_cols": ["chosen_resume", "reasoning"],
    },
    "exp3_leadership_summary": {
        "type": "wide",
        "expected_rows": EXPECTED_WIDE_ROWS,
        "key_cols": ["resume_id", "permutation_id"],
        "output_cols": ["chosen_resume", "reasoning"],
    },
    "exp4_competence_cv": {
        "type": "per_cv",
        "expected_rows": EXPECTED_PER_CV_ROWS,
        "key_cols": ["resume_id", "permutation_id", "augmentation_column"],
        "output_cols": list(_COMPETENCES),
    },
    "exp4_competence_summary": {
        "type": "per_cv",
        "expected_rows": EXPECTED_PER_CV_ROWS,
        "key_cols": ["resume_id", "permutation_id", "augmentation_column"],
        "output_cols": list(_COMPETENCES),
    },
    "exp4b_competence_comparative_cv": {
        "type": "wide",
        "expected_rows": EXPECTED_WIDE_ROWS,
        "key_cols": ["resume_id", "permutation_id"],
        "output_cols": [f"c{c}_{comp}" for c in (1, 2, 3) for comp in _COMPETENCES],
    },
    "exp4b_competence_comparative_summary": {
        "type": "wide",
        "expected_rows": EXPECTED_WIDE_ROWS,
        "key_cols": ["resume_id", "permutation_id"],
        "output_cols": [f"c{c}_{comp}" for c in (1, 2, 3) for comp in _COMPETENCES],
    },
    "exp5_job_rank_cv": {
        "type": "wide",
        "expected_rows": EXPECTED_WIDE_ROWS,
        "key_cols": ["resume_id", "permutation_id"],
        "output_cols": ["level_1_title", "level_2_title", "level_3_title",
                        "c1_level", "c2_level", "c3_level"],
    },
    "exp5_job_rank_summary": {
        "type": "wide",
        "expected_rows": EXPECTED_WIDE_ROWS,
        "key_cols": ["resume_id", "permutation_id"],
        "output_cols": ["level_1_title", "level_2_title", "level_3_title",
                        "c1_level", "c2_level", "c3_level"],
    },
    "exp6_senior_jd_cv": {
        "type": "wide",
        "expected_rows": EXPECTED_WIDE_ROWS,
        "key_cols": ["resume_id", "permutation_id"],
        "output_cols": ["chosen_resume", "reasoning"],
    },
    "exp6_senior_jd_summary": {
        "type": "wide",
        "expected_rows": EXPECTED_WIDE_ROWS,
        "key_cols": ["resume_id", "permutation_id"],
        "output_cols": ["chosen_resume", "reasoning"],
    },
}


# ── Validation ────────────────────────────────────────────────────────

def _check_int_range(df, row, key_cols, col_name, lo, hi, report):
    """Append to report if *col_name* is outside [lo, hi]."""
    val = row.get(col_name)
    if pd.isna(val) or str(val).strip() == "":
        return
    try:
        v = int(float(val))
        if v < lo or v > hi:
            entry = {c: row[c] for c in key_cols}
            entry["issue"] = f"{col_name}={v}, expected {lo}-{hi}"
            report["invalid_output_rows"].append(entry)
    except (ValueError, TypeError):
        entry = {c: row[c] for c in key_cols}
        entry["issue"] = f"{col_name}={val!r}, not parseable"
        report["invalid_output_rows"].append(entry)


def _check_experiment(
    df: pd.DataFrame,
    spec: dict[str, Any],
    exp_name: str,
) -> dict[str, Any]:
    """Check one experiment result file and return a report dict."""
    report: dict[str, Any] = {
        "experiment": exp_name,
        "rows_found": len(df),
        "rows_expected": spec["expected_rows"],
        "missing_rows": 0,
        "error_rows": [],
        "null_output_rows": [],
        "invalid_output_rows": [],
        "duplicate_rows": 0,
    }

    key_cols = [c for c in spec["key_cols"] if c in df.columns]

    # ── Duplicate rows ────────────────────────────────────────────
    if key_cols:
        report["duplicate_rows"] = int(df.duplicated(subset=key_cols, keep="last").sum())

    # ── Missing rows ──────────────────────────────────────────────
    report["missing_rows"] = max(0, spec["expected_rows"] - len(df))

    # ── Error rows (non-empty error column) ───────────────────────
    if "error" in df.columns:
        error_mask = df["error"].astype(str).str.strip().ne("")
        report["error_rows"] = [
            {col: row[col] for col in key_cols + ["error"]}
            for _, row in df[error_mask].iterrows()
        ]

    # ── Null / empty output columns ───────────────────────────────
    for _, row in df.iterrows():
        nulls = [
            col for col in spec["output_cols"]
            if col not in df.columns
            or pd.isna(row.get(col))
            or str(row.get(col, "")).strip() == ""
        ]
        if nulls:
            entry = {col: row[col] for col in key_cols}
            entry["null_columns"] = nulls
            report["null_output_rows"].append(entry)

    # ── Value-range checks (skip rows that already have errors) ───
    has_chosen = (
        exp_name.startswith("exp1") or exp_name.startswith("exp2")
        or exp_name.startswith("exp3") or exp_name.startswith("exp6")
    )
    has_isolated_scores = exp_name.startswith("exp4_")
    has_comparative_scores = exp_name.startswith("exp4b_")
    has_levels = exp_name.startswith("exp5")

    for _, row in df.iterrows():
        if "error" in df.columns and str(row.get("error", "")).strip():
            continue

        if has_chosen:
            _check_int_range(df, row, key_cols, "chosen_resume", 1, 3, report)

        if has_isolated_scores:
            for comp in _COMPETENCES:
                _check_int_range(df, row, key_cols, comp, 1, 5, report)

        if has_comparative_scores:
            for cand in (1, 2, 3):
                for comp in _COMPETENCES:
                    _check_int_range(df, row, key_cols, f"c{cand}_{comp}", 1, 5, report)

        if has_levels:
            for lc in ("c1_level", "c2_level", "c3_level"):
                _check_int_range(df, row, key_cols, lc, 1, 3, report)

    return report


def _print_report(report: dict[str, Any]) -> None:
    name = report["experiment"]
    found = report["rows_found"]
    expected = report["rows_expected"]
    missing = report["missing_rows"]
    errors = report["error_rows"]
    nulls = report["null_output_rows"]
    invalids = report["invalid_output_rows"]
    dupes = report["duplicate_rows"]

    has_issues = missing > 0 or errors or nulls or invalids or dupes > 0
    status = "ISSUES" if has_issues else "OK"
    print(f"\n  [{name}] {found}/{expected} rows  — {status}")

    if missing > 0:
        print(f"    Missing: {missing} rows")
    if dupes > 0:
        print(f"    Duplicates: {dupes} rows (last kept)")
    for label, items in [("Errors", errors), ("Null outputs", nulls),
                         ("Invalid values", invalids)]:
        if items:
            print(f"    {label}: {len(items)} rows")
            for item in items[:5]:
                print(f"      {item}")
            if len(items) > 5:
                print(f"      ... and {len(items) - 5} more")


def get_bad_keys(report: dict[str, Any], spec: dict[str, Any]) -> list[tuple]:
    """Extract the unique keys of all problematic rows for potential rerun."""
    key_cols = spec["key_cols"]
    bad: set[tuple] = set()
    for bucket in ("error_rows", "null_output_rows", "invalid_output_rows"):
        for entry in report[bucket]:
            bad.add(tuple(entry.get(c) for c in key_cols))
    return sorted(bad)


# ── Model-level validation ────────────────────────────────────────────

def validate_model(model: str) -> dict[str, dict[str, Any]]:
    """Validate all experiment results for one model. Returns {exp_name: report}."""
    model_dir = RESULTS_DIR / model
    if not model_dir.exists():
        print(f"\nModel directory not found: {model_dir}")
        return {}

    print(f"\n{'='*60}")
    print(f"  Validating: {model}")
    print(f"{'='*60}")

    reports = {}
    for exp_name, spec in EXPERIMENT_SPECS.items():
        path = model_dir / f"{exp_name}.csv"
        if not path.exists():
            print(f"\n  [{exp_name}] FILE NOT FOUND")
            continue

        df = pd.read_csv(path, keep_default_na=False)
        report = _check_experiment(df, spec, exp_name)
        reports[exp_name] = report
        _print_report(report)

        bad = get_bad_keys(report, spec)
        if bad:
            print(f"    Keys to rerun: {len(bad)}")

    missing_exps = set(EXPERIMENT_SPECS) - set(reports)
    if missing_exps:
        print(f"\n  Missing experiments ({len(missing_exps)}/{len(EXPERIMENT_SPECS)}):")
        for m in sorted(missing_exps):
            print(f"    - {m}")

    total_issues = sum(
        len(r["error_rows"]) + len(r["null_output_rows"])
        + len(r["invalid_output_rows"]) + r["missing_rows"]
        for r in reports.values()
    )
    print(f"\n  Total issues: {total_issues}")
    return reports


def validate_all() -> dict[str, dict[str, dict[str, Any]]]:
    """Validate all models found in RESULTS_DIR."""
    all_reports = {}
    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        return all_reports

    model_dirs = sorted(d for d in RESULTS_DIR.iterdir() if d.is_dir())
    if not model_dirs:
        print(f"No model directories found in {RESULTS_DIR}")
        return all_reports

    for model_dir in model_dirs:
        reports = validate_model(model_dir.name)
        if reports:
            all_reports[model_dir.name] = reports

    return all_reports


# ── Rerun helpers ─────────────────────────────────────────────────────

# Experiments that need exp2's summary cache loaded before reruns.
# exp6 summary uses its own cache and is handled separately below.
_SUMMARY_EXPERIMENTS_EXP2 = {
    "exp2_pick_best_summary",
    "exp3_leadership_summary",
    "exp4_competence_summary",
    "exp4b_competence_comparative_summary",
    "exp5_job_rank_summary",
}

_SUMMARY_EXPERIMENTS_EXP6 = {
    "exp6_senior_jd_summary",
}


def _load_config(exp_name: str):
    """Import and return the CONFIG object for an experiment.

    Does NOT reload the module so that runtime state (e.g. the loaded
    summaries cache in exp2) is preserved.
    """
    import importlib

    module_map = {
        "exp0_summarise": ("exp0_summarise", "CONFIG"),
        "exp1_pick_best_cv": ("exp1_pick_best_cv", "CONFIG"),
        "exp2_pick_best_summary": ("exp2_pick_best_summary", "CONFIG"),
        "exp3_leadership_cv": ("exp3_leadership_potential", "CONFIG_CV"),
        "exp3_leadership_summary": ("exp3_leadership_potential", "CONFIG_SUMMARY"),
        "exp4_competence_cv": ("exp4_competence_scoring", "CONFIG_CV"),
        "exp4_competence_summary": ("exp4_competence_scoring", "CONFIG_SUMMARY"),
        "exp4b_competence_comparative_cv": ("exp4b_competence_comparative", "CONFIG_CV"),
        "exp4b_competence_comparative_summary": ("exp4b_competence_comparative", "CONFIG_SUMMARY"),
        "exp5_job_rank_cv": ("exp5_job_rank", "CONFIG_CV"),
        "exp5_job_rank_summary": ("exp5_job_rank", "CONFIG_SUMMARY"),
        "exp6_senior_jd_cv": ("exp6_senior_jd", "CONFIG_CV"),
        "exp6_senior_jd_summary": ("exp6_senior_jd", "CONFIG_SUMMARY"),
    }
    if exp_name not in module_map:
        raise ValueError(f"Unknown experiment: {exp_name}")

    mod_name, attr = module_map[exp_name]
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


def _cast_to_col_dtype(series: pd.Series, val: Any) -> Any:
    """Cast *val* to match the dtype of *series* so assignment doesn't raise."""
    try:
        if pd.api.types.is_integer_dtype(series.dtype):
            return int(val)
        if pd.api.types.is_float_dtype(series.dtype):
            return float(val)
    except (ValueError, TypeError):
        pass
    return str(val)


_SHARED_SUMMARIES = DATA_DIR / "shared_summaries" / "exp0_summarise_cohere.csv"


def _ensure_summaries_loaded(exp_name: str, model: str) -> None:
    """Load summary caches needed by the given experiment before reruns.

    Falls back to the shared Cohere summaries if the model has no own exp0 file.
    """
    own_path = RESULTS_DIR / model / "exp0_summarise.csv"
    kw = {} if own_path.exists() else {"summaries_path": str(_SHARED_SUMMARIES)}

    if exp_name in _SUMMARY_EXPERIMENTS_EXP2:
        import exp2_pick_best_summary
        if exp2_pick_best_summary._loaded_model != model:
            exp2_pick_best_summary.load_summaries(model, **kw)

    if exp_name in _SUMMARY_EXPERIMENTS_EXP6:
        import exp6_senior_jd
        if exp6_senior_jd._loaded_model != model:
            exp6_senior_jd.load_summaries(model, **kw)


def rerun_bad_rows(
    model: str,
    exp_name: str,
    client=None,
    *,
    api_key: str | None = None,
    sleep_seconds: float = 0.05,
) -> int:
    """
    Rerun only the bad rows for a single experiment and patch the results CSV.

    Returns the number of rows successfully rerun.
    """
    import time
    from constants import AUGMENTED_RESUME_COLS
    from runner import call_model, init_client, model_results_dir

    spec = EXPERIMENT_SPECS[exp_name]
    config = _load_config(exp_name)
    key_cols = spec["key_cols"]

    result_path = model_results_dir(model) / f"{exp_name}.csv"
    if not result_path.exists():
        print(f"No result file found at {result_path}")
        return 0

    df_result = pd.read_csv(result_path, keep_default_na=False)
    report = _check_experiment(df_result, spec, exp_name)
    bad_keys = get_bad_keys(report, spec)

    if not bad_keys:
        print(f"[{exp_name}] No bad rows to rerun.")
        return 0

    # Skip rows from mismatched resume_ids — they will never produce valid
    # output because the JDs don't match the CVs in the source data.
    rid_index = key_cols.index("resume_id") if "resume_id" in key_cols else None
    if rid_index is not None:
        original_count = len(bad_keys)
        bad_keys = [k for k in bad_keys if int(k[rid_index]) not in MISMATCH_RESUME_IDS]
        skipped = original_count - len(bad_keys)
        if skipped:
            print(f"[{exp_name}] Skipping {skipped} rows from mismatched resume_ids")

    if not bad_keys:
        print(f"[{exp_name}] No fixable bad rows to rerun (all are JD/CV mismatches).")
        return 0

    print(f"[{exp_name}] {len(bad_keys)} bad rows to rerun for model '{model}'")

    if client is None:
        client = init_client(api_key=api_key)

    _ensure_summaries_loaded(exp_name, model)

    df_wide = pd.read_csv(
        DATA_DIR / "final_CV_dataset_experiments_one_row.csv",
        keep_default_na=False,
    )

    # Normalise keys to strings for per_cv experiments (augmentation_column
    # is a string, but resume_id/permutation_id may load as int from CSV).
    if spec["type"] == "per_cv" and "augmentation_column" in key_cols:
        bad_key_set = {tuple(str(v) for v in k) for k in bad_keys}
    else:
        bad_key_set = {tuple(k) for k in bad_keys}

    fixed = 0
    for _, row in df_wide.iterrows():

        # ── Per-CV experiments: one API call per CV slot ──────────
        if spec["type"] == "per_cv":
            for slot_i, col in enumerate(AUGMENTED_RESUME_COLS):
                key = tuple(
                    col if c == "augmentation_column" else str(row.get(c))
                    for c in key_cols
                )
                if key not in bad_key_set:
                    continue

                print(f"  Rerunning {key}...")
                try:
                    messages = config.build_messages(row, slot_index=slot_i)
                    raw = call_model(
                        client, messages=messages, model=model,
                        response_format=config.response_format,
                    )
                    parsed = config.parse_response(raw, row, slot_index=slot_i)

                    # Build a boolean mask matching this row in the results CSV
                    mask = True
                    for kc in key_cols:
                        if kc == "augmentation_column":
                            mask = mask & (df_result[kc] == col)
                        else:
                            mask = mask & (df_result[kc].astype(str) == str(row[kc]))

                    for out_col, val in parsed.items():
                        df_result.loc[mask, out_col] = _cast_to_col_dtype(df_result[out_col], val)
                    df_result.loc[mask, "raw_response"] = raw
                    df_result.loc[mask, "error"] = ""
                    fixed += 1
                except Exception as e:
                    print(f"  Failed again for {key}: {e}")

                time.sleep(sleep_seconds)

        # ── Wide experiments: one API call per row ────────────────
        else:
            rid = row.get("resume_id")
            pid = row.get("permutation_id")
            key = (rid, pid)
            if key not in bad_key_set:
                continue

            print(f"  Rerunning {key}...")
            try:
                messages = config.build_messages(row)
                raw = call_model(
                    client, messages=messages, model=model,
                    response_format=config.response_format,
                )
                parsed = config.parse_response(raw, row)

                mask = (
                    (df_result["resume_id"].astype(str) == str(rid))
                    & (df_result["permutation_id"].astype(str) == str(pid))
                )
                for out_col, val in parsed.items():
                    df_result.loc[mask, out_col] = _cast_to_col_dtype(df_result[out_col], val)
                df_result.loc[mask, "raw_response"] = raw
                df_result.loc[mask, "error"] = ""
                fixed += 1
            except Exception as e:
                print(f"  Failed again for {key}: {e}")

            time.sleep(sleep_seconds)

    df_result.to_csv(result_path, index=False)
    print(f"[{exp_name}] Fixed {fixed}/{len(bad_keys)} rows. Saved to {result_path}")
    return fixed


def rerun_all_bad(model: str, client=None, **kwargs) -> dict[str, int]:
    """Rerun bad rows for all experiments of a model. Returns {exp_name: fixed_count}."""
    reports = validate_model(model)
    results = {}
    for exp_name, report in reports.items():
        spec = EXPERIMENT_SPECS[exp_name]
        bad = get_bad_keys(report, spec)
        if bad:
            results[exp_name] = rerun_bad_rows(model, exp_name, client=client, **kwargs)
    if not results:
        print(f"\nNo bad rows found for {model}. All clean!")
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        validate_model(sys.argv[1])
    else:
        validate_all()
