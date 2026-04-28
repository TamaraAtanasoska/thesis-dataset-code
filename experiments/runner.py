"""
Shared experiment runner: API calls, checkpointing, and result recording.

Iterates over the wide dataset rows, delegates prompt construction and response
parsing to an ``ExperimentConfig``, and writes results (with identity markers)
to a CSV with row-by-row checkpointing.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent / "dataset creation scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from constants import CHAT_MODEL
from paths import DATA_DIR, RESULTS_DIR

if TYPE_CHECKING:
    from experiment_configs import ExperimentConfig

IDENTITY_COLS = (
    "resume_id",
    "permutation_id",
    "baseline_at_pos",
    "markers",
    "city_type",
    "assigned_city",
)


def model_results_dir(model: str = CHAT_MODEL) -> Path:
    """``RESULTS_DIR / <model>/`` -- all outputs for one model live here."""
    d = RESULTS_DIR / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def _checkpoint_path(output_path: Path) -> Path:
    return output_path.parent / f"checkpoint_{output_path.name}"


def _load_completed_keys(checkpoint_path: Path) -> set[tuple]:
    """Return set of (resume_id, permutation_id) already completed."""
    if not checkpoint_path.exists():
        return set()
    df = pd.read_csv(checkpoint_path)
    if "resume_id" in df.columns and "permutation_id" in df.columns:
        return set(zip(df["resume_id"], df["permutation_id"]))
    return set()


PROVIDER_COHERE = "cohere"
PROVIDER_OPENROUTER = "openrouter"


def init_client(
    provider: str = PROVIDER_COHERE,
    api_key: str | None = None,
):
    """
    Create an API client for the given provider.

    Supported providers:
      - ``"cohere"``     — uses ``COHERE_API_KEY`` env var
      - ``"openrouter"`` — uses ``OPENROUTER_API_KEY`` env var
    """
    if provider == PROVIDER_COHERE:
        import cohere
        key = api_key or os.environ.get("COHERE_API_KEY")
        if not key:
            raise RuntimeError("Set COHERE_API_KEY or pass api_key=")
        return cohere.ClientV2(key)

    if provider == PROVIDER_OPENROUTER:
        from openai import OpenAI
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("Set OPENROUTER_API_KEY or pass api_key=")
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

    raise ValueError(f"Unknown provider: {provider!r}")


# keep old name as alias so the running notebook doesn't break on next reload
init_cohere_client = init_client


def call_model(
    client,
    *,
    messages: list[dict[str, str]],
    model: str = CHAT_MODEL,
    response_format: dict[str, str] | None = None,
) -> str:
    """
    Single model call. Auto-detects provider from client type and returns
    the raw text response.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format

    try:
        import cohere
        is_cohere = isinstance(client, (cohere.ClientV2, cohere.Client))
    except ImportError:
        is_cohere = False

    if is_cohere:
        response = client.chat(**kwargs)
        return response.message.content[0].text
    else:
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


def run_experiment(
    df: pd.DataFrame,
    config: ExperimentConfig,
    client,
    *,
    output_path: Path | str | None = None,
    model: str = CHAT_MODEL,
    sleep_seconds: float = 0.05,
) -> pd.DataFrame:
    """
    Run an experiment over every row of the wide dataset.

    Parameters
    ----------
    df
        Wide augmented dataset (315 rows).
    config
        An ``ExperimentConfig`` that provides ``build_messages``,
        ``parse_response``, ``name``, ``response_format``, and
        ``output_columns``.
    client
        API client (from ``init_client``).
    output_path
        Where to write the final CSV. Defaults to
        ``RESULTS_DIR / "{config.name}_{model}.csv"``.
    model
        Model identifier passed to ``call_model``.
    sleep_seconds
        Delay between API calls for rate limiting.

    Returns
    -------
    DataFrame with identity columns + experiment output columns.
    """
    if output_path is None:
        output_path = model_results_dir(model) / f"{config.name}.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cp_path = _checkpoint_path(output_path)
    completed = _load_completed_keys(cp_path)

    id_cols_present = [c for c in IDENTITY_COLS if c in df.columns]
    results: list[dict[str, Any]] = []
    n = len(df)

    print(f"[{config.name}] Starting experiment ({n} rows, "
          f"{len(completed)} already done)...")

    for row_num, (idx, row) in enumerate(df.iterrows(), 1):
        key = (row.get("resume_id"), row.get("permutation_id"))
        if key in completed:
            continue

        print(f"[{config.name}] Row {row_num}/{n} "
              f"(resume_id={key[0]}, perm={key[1]})...")

        base = {c: row[c] for c in id_cols_present}
        base["model"] = model

        try:
            messages = config.build_messages(row)
            raw = call_model(
                client,
                messages=messages,
                model=model,
                response_format=config.response_format,
            )
            parsed = config.parse_response(raw, row)
            base.update(parsed)
            base["raw_response"] = raw
            base["error"] = ""
        except Exception as e:
            print(f"[{config.name}] Error at row {row_num}: {e}")
            base["error"] = str(e)
            base["raw_response"] = ""
            for col in config.output_columns:
                base.setdefault(col, None)

        results.append(base)

        # Append to checkpoint: write header only on the first row
        cp_df = pd.DataFrame([base])
        cp_df.to_csv(
            cp_path,
            mode="a",
            header=not cp_path.exists(),
            index=False,
        )

        time.sleep(sleep_seconds)

    if cp_path.exists():
        full_df = pd.read_csv(cp_path)
    else:
        full_df = pd.DataFrame(results)

    full_df.to_csv(output_path, index=False)
    print(f"[{config.name}] Done. Wrote {len(full_df)} rows -> {output_path}")

    if cp_path.exists():
        cp_path.unlink()
        print(f"[{config.name}] Checkpoint cleaned up.")

    return full_df


def run_per_cv_experiment(
    df: pd.DataFrame,
    config: ExperimentConfig,
    client,
    *,
    output_path: Path | str | None = None,
    model: str = CHAT_MODEL,
    sleep_seconds: float = 0.05,
) -> pd.DataFrame:
    """
    Like ``run_experiment`` but calls the model once per CV slot (3 times per
    wide row). Used by the summarisation experiment (exp0) and per-CV competence
    scoring (exp4).

    The config's ``build_messages(row, slot_index=i)`` receives the slot index
    (0, 1, 2) and ``parse_response`` receives it too.
    """
    from constants import AUGMENTED_RESUME_COLS

    if output_path is None:
        output_path = model_results_dir(model) / f"{config.name}.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cp_path = _checkpoint_path(output_path)
    completed: set[tuple] = set()
    if cp_path.exists():
        cp_df = pd.read_csv(cp_path)
        if all(c in cp_df.columns for c in ("resume_id", "permutation_id", "augmentation_column")):
            completed = set(zip(
                cp_df["resume_id"],
                cp_df["permutation_id"],
                cp_df["augmentation_column"],
            ))

    id_cols_present = [c for c in IDENTITY_COLS if c in df.columns]
    n = len(df)

    print(f"[{config.name}] Starting per-CV experiment ({n} rows x 3 slots, "
          f"{len(completed)} already done)...")

    for row_num, (idx, row) in enumerate(df.iterrows(), 1):
        for slot_i, col in enumerate(AUGMENTED_RESUME_COLS):
            key = (row.get("resume_id"), row.get("permutation_id"), col)
            if key in completed:
                continue

            print(f"[{config.name}] Row {row_num}/{n}, slot {col}...")

            base: dict[str, Any] = {c: row[c] for c in id_cols_present}
            base["model"] = model
            base["augmentation_column"] = col

            try:
                messages = config.build_messages(row, slot_index=slot_i)
                raw = call_model(
                    client,
                    messages=messages,
                    model=model,
                    response_format=config.response_format,
                )
                parsed = config.parse_response(raw, row, slot_index=slot_i)
                base.update(parsed)
                base["raw_response"] = raw
                base["error"] = ""
            except Exception as e:
                print(f"[{config.name}] Error at row {row_num} slot {col}: {e}")
                base["error"] = str(e)
                base["raw_response"] = ""
                for col_name in config.output_columns:
                    base.setdefault(col_name, None)

            # Append to checkpoint: write header only on the first row
            cp_row = pd.DataFrame([base])
            cp_row.to_csv(
                cp_path,
                mode="a",
                header=not cp_path.exists(),
                index=False,
            )

            time.sleep(sleep_seconds)

    if cp_path.exists():
        full_df = pd.read_csv(cp_path)
    else:
        full_df = pd.DataFrame()

    full_df.to_csv(output_path, index=False)
    print(f"[{config.name}] Done. Wrote {len(full_df)} rows -> {output_path}")

    if cp_path.exists():
        cp_path.unlink()

    return full_df
