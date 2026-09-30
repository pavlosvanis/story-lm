# Development

Install the project and its development tools with `uv sync --locked`.
The default development group includes Ruff and Matplotlib for experiment plots.

## Python style

Use four spaces, double quotes, and a 120-character line length. Ruff formats
Python files and sorts imports. Docstrings use Google-style `Args`, `Returns`,
`Yields`, and `Raises` sections when they clarify the interface. Type hints carry
type information; documentation explains meaning, shapes, and behavior.

Comments explain algorithm choices, numerical stability, or tensor shapes.
Keep comments accurate and avoid narrating operations whose meaning is obvious.
Preserve attribution for inherited code and fixtures.

To apply formatting and import sorting:

```sh
uv run ruff check --select I --fix src experiments tests train_bpe_tinystories.py
uv run ruff format src experiments tests train_bpe_tinystories.py
```

To check the current files:

```sh
uv run ruff check src experiments tests train_bpe_tinystories.py
uv run ruff format --check src experiments tests train_bpe_tinystories.py
```

## Verification

```sh
uv run pytest
uv run python -m experiments.scripts.generate_text --max-new-tokens 32
```

The tokenizer comparisons use tiktoken's GPT-2 reference encoding, which may
download its vocabulary and merges on the first run. The generation command uses
the existing model and tokenizer under `artifacts/` and writes a sample under
`experiments/results/generation/`.

## Experiment records

Experiment settings live in `experiments/config.py`, and JSON serialization lives
in `experiments/save_results.py`. Executable workflows live in
`experiments/scripts/`; recorded outputs and local checkpoints live in
`experiments/results/`. Checkpoints remain ignored by Git.

Run experiment workflows as modules from the repository root. For example:

```sh
uv run python -m experiments.scripts.benchmark_tokenizer throughput
uv run python -m experiments.scripts.plot_experiments
```

The plotting workflow reads the saved results and writes figures to `figures/`.

Pass the repository root to `save_experiment_results` as `project_root`.
The logger records project-relative paths without changing the configuration
used for training. External absolute paths are reduced to filenames; provide
those datasets separately when reproducing an experiment.

Local homework PDFs, caches, and virtual environments are ignored. Ignoring a
file does not remove a copy that was already tracked; review those separately
when preparing a commit.
