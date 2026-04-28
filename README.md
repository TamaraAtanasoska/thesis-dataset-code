# Thesis: dataset, dataset creation scripts and experiment pipelines

More detailed documentation soon.

## Layout

| Path | Purpose |
|------|---------|
| `data/` | Contains the study dataset (long and wide versions), augmented from the German subset of [JobResQA benchmark](https://github.com/Avature/jobresqa-benchmark). |
| `dataset creation scripts/` | Utilities used to prepare and augment the dataset, including the prompts used for each step in the augmentation. They are costly and not optimised for API calls, as they were used in a slow, exploratory process where each augmentation step was executed and evaluated separately. For less costly runs, both the scripts and prompts should be condensed and merged. |
| `experiments/` | Runnable experiment definitions (`exp*.py`), shared config (`experiment_configs.py`), orchestration (`runner.py`, `run_experiments.ipynb`), and validation. The experiments include the translated German versions of the prompts used and the code for shaping the result data by model tested. |

Paths listed in `.gitignore` (for example local editor settings and generated chart exports) are intentionally not versioned. 

## Requirements

Use **Python 3.10 or newer**. Install dependencies from the repository root:

```bash
pip install ".[notebook]"
```

The `[notebook]` extra adds Jupyter tooling (`ipykernel`, `jupyterlab`). Omit it for a minimal environment:

```bash
pip install .
```

API-based scripts expect provider keys in the environment (for example `COHERE_API_KEY` or `OPENROUTER_API_KEY`), as described in `experiments/runner.py`.

## Running experiments

From the repository root, run the experiment modules or the shared runner as documented in `experiments/` (for example `python experiments/runner.py` with the intended experiment configuration, or the workflow in `experiments/run_experiments.ipynb`).