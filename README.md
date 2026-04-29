# Thesis: dataset, dataset creation scripts and experiment pipelines

From the abstract:
The current Generative AI (GenAI) evaluation landscape is often critiqued for lacking rigour, particularly in assessing abstract, contested concepts. This has motivated multidisciplinary approaches grounded in fields with a tradition of evaluating such concepts, such as the social sciences. We demonstrate such an approach using the framework by [Wallach et al. (2025)](https://proceedings.mlr.press/v267/wallach25a.html) to assess the extent of cultural essentialism and cultural stereotypes in the German workplace, in a study on the use of GenAI tools by human resource professionals for hiring and candidate evaluation. We find that the framework provides a solid and rigorous foundation for multidisciplinary research.

More detailed documentation soon.

If you are interested in the resulting data from the experiments or have questions about the analysis, please feel free to reach out. 

## Layout

| Path | Purpose |
|------|---------|
| [data/](data/) | Contains the study dataset (long and wide versions), augmented from the German subset of [JobResQA benchmark](https://github.com/Avature/jobresqa-benchmark). |
| [dataset creation scripts/](dataset%20creation%20scripts/) | Utilities used to prepare and augment the dataset, including the prompts used for each step in the augmentation. |
| [experiments/](experiments/) | Runnable experiment definitions (`exp*.py`), shared config (`experiment_configs.py`), orchestration (`runner.py`, `run_experiments.ipynb`), and validation. The experiments include the translated German versions of the prompts used and the code for shaping the result data by model tested. |

Paths listed in `.gitignore` (for example local editor settings and generated chart exports) are intentionally not versioned. 

The [dataset creation scripts/](dataset%20creation%20scripts/) were crated iteratively, originally as individual scripts and part of a longer discovery phase. They were refactored in a runnable collection with the help of a coding assistant from the separate notebook scripts. There is an example notebook in the folder with saved output where some parts were rerun after the dataset creation to serve as example. However, the scripts are currently costly to run, and they can very likely be combined in a one shot for augmenting each row of original [JobResQA benchmark](https://github.com/Avature/jobresqa-benchmark).

There is a current plan to continue the work by including more languages and cultural perspectives than German, so the scripts will get an updated version.

If yuo are interested in the English versions of the experiment prompts, please feel free to reach out. 

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
