# StoryLM

**A small story-writing language model built and trained from scratch in PyTorch.**

StoryLM connects a byte-level BPE tokenizer, a decoder-only Transformer, a custom training pipeline, and a local browser demo. The selected model reached **1.6303 per-token validation cross-entropy** on TinyStories after **40,960,000 training token predictions**.

The project documents the decisions behind that model through learning-rate sweeps, fixed-token-budget batch comparisons, and architecture ablations. Trained weights and the matching tokenizer are included, so the demo can run without downloading the training corpus or retraining.

## Try the model

Requires Python **3.11–3.13**, [uv](https://docs.astral.sh/uv/), and a browser. From the repository root:

```bash
uv sync --locked
uv run storylm gui
```

The command loads the saved model and opens a local browser page. Enter a story beginning, select **Continue the story**. **Save** downloads the document as `name.txt`; the filename field contains the name without its extension. Keep the terminal open while using the demo and press **Ctrl+C** there to stop it.

Generation runs in a background server thread. The letter-by-letter reveal happens after the model returns its completion; it is a display effect rather than live token streaming. The interface uses Python's standard library and browser JavaScript. Node.js is only needed for the optional JavaScript tests.

![StoryLM.png](screenshots%2FStoryLM.png)

For command-line generation:

```bash
uv run storylm generate --prompt "A fox discovered a hidden door." --max-new-tokens 256 --temperature 0.8 --top-p 0.9 --seed 42
```

Device selection defaults to MPS, then CUDA, then CPU. Use `--device cpu`, `--device mps`, or `--device cuda` to select a device explicitly. The GUI also accepts `--artifacts-dir` and `--device`.

## What I built

- **Tokenization:** byte-level BPE training, encoding and decoding, special-token handling, and corpus preparation.
- **Model:** causal Transformer attention, rotary positional embeddings, pre-norm RMSNorm, and SwiGLU feed-forward blocks.
- **Optimization:** AdamW, gradient clipping, warmup and cosine learning-rate scheduling, validation, and checkpointing.
- **Experiment tooling:** controlled sweeps, architecture ablations, saved result records, and figures.
- **Inference:** a reusable generator, temperature and nucleus sampling, command-line generation, and a local writing interface.
- **Artifact workflows:** cached data preparation, validation of existing model artifacts, and training/export into a separate destination.

The implementation uses PyTorch tensor operations and autograd. The published checkpoint was trained for this project, rather than adapted from pretrained language-model weights.

## Model and training configuration

| Setting | Selected checkpoint |
| --- | --- |
| Dataset | TinyStories V2, GPT-4 training and validation text |
| Tokenizer | Byte-level BPE, 10,000 tokens |
| Context length | 256 tokens |
| Transformer blocks | 4 |
| Model width / attention heads | 512 / 16 |
| Feed-forward network | SwiGLU, width 1,344 |
| Normalization / positions | Pre-norm RMSNorm / RoPE |
| Batch size / training steps | 64 / 2,500 |
| Training token predictions | 40,960,000 |
| Maximum learning rate | 0.003 |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.1 |
| Precision / original device | Float32 / Apple MPS |
| Final validation cross-entropy | **1.6303** |

The learning rate uses 50 warmup steps followed by cosine decay to 10% of the maximum. The exported weights are from the final training step.

## Experimental results

The batch-32 reference model used 5,000 steps and the same total token budget. Architecture comparisons below use that reference, rather than the selected batch-64 model.

| Experiment | Validation cross-entropy | Finding |
| --- | ---: | --- |
| Batch-32 baseline | 1.6554 | Pre-norm RMSNorm, RoPE, SwiGLU, learning rate 0.003 |
| Batch 64 | **1.6303** | Best completed batch-size run at the fixed token budget |
| SiLU, approximately matched parameter count | 1.6424 | Small improvement in one seed; not combined with batch 64 |
| No RMSNorm, learning rate 0.001 | 1.6929 | Lower learning rate restored stability |
| Post-norm | 1.7116 | Slower convergence and higher final loss |
| No explicit positional encoding | 1.8284 | Learned useful patterns, but underperformed RoPE |

Removing RMSNorm at learning rate 0.003 produced NaN losses. The learning-rate sweep also separated poor convergence at 0.01 from unstable behavior at 0.1.

![Validation loss versus processed token predictions for the batch-size comparison](figures/batch_size_validation.png)

*Batch sizes were compared at the same 40,960,000-token prediction budget. Batch 64 achieved the lowest final validation loss among the completed runs.*

See [the experiment report](docs/experiments.md) for the full tables, eight figures, generated sample, and interpretation.

## Reproduce the workflow

The demo and CLI generation need these four matching artifacts:

```text
artifacts/
├── final_model/
│   ├── config.json
│   └── weights.pt
├── tinystories_vocab.pkl
└── tinystories_merges.pkl
```

The weights are approximately 87 MiB. Raw text and prepared training arrays are excluded from the repository and are unnecessary for inference.

To prepare training data:

```bash
uv run storylm prepare
```

Preparation reuses existing valid tokenizer files and token arrays. Missing raw splits are downloaded from a pinned TinyStories revision and checked against their expected sizes and SHA-256 hashes. Missing token arrays are then encoded. The training text is approximately 2.2 GB; encoding builds the complete output token array in memory, so this step requires substantially more resources than the demo.

To train a new model while preserving the included checkpoint:

```bash
uv run storylm train --artifacts-dir artifacts/retrained
```

The training workflow prepares missing data and exports the selected configuration. If a completed model already exists at the destination, it validates and reuses it. Incomplete model output directories are rejected. Checkpoints and loss history are saved, but automatic checkpoint resume is not implemented.

Training settings are defined in [`src/storylm/training/config.py`](src/storylm/training/config.py), including the model architecture, batch size, token budget, and optimizer settings. To experiment with different settings, edit that file and train into a new, unused artifacts directory. Changing the training configuration does not retrain or replace an existing completed model.

The separate [`experiments/config.py`](experiments/config.py) contains reference settings for the sweep and ablation scripts. The exported `artifacts/final_model/config.json` records the architecture associated with its saved weights and is used when loading that model for inference.

To use the new export:

```bash
uv run storylm gui --artifacts-dir artifacts/retrained
```

To regenerate the experiment figures from the saved records:

```bash
uv run python -m experiments.scripts.plot_experiments
```

Original experiments used MPS and float32. A fixed seed supports repeatable runs within a setup; it does not guarantee identical losses or samples across devices and software versions.

## Code map

| Location | Purpose |
| --- | --- |
| [`src/storylm/tokenization/`](src/storylm/tokenization/) | BPE training, tokenizer, and corpus encoding |
| [`src/storylm/model/`](src/storylm/model/) | Transformer components and architecture variants |
| [`src/storylm/training/`](src/storylm/training/) | Optimizer, schedules, checkpoints, and training workflow |
| [`src/storylm/data/`](src/storylm/data/) | Dataset downloads, preparation, and next-token batch sampling |
| [`src/storylm/inference/`](src/storylm/inference/) | Artifact loading and autoregressive generation |
| [`src/storylm/gui.py`](src/storylm/gui.py) | Local browser interface and inference server |
| [`experiments/scripts/`](experiments/scripts/) | Sweeps, ablations, benchmarks, and plotting |
| [`experiments/results/`](experiments/results/) | Saved experiment measurements and samples |
| [`tests/`](tests/) | Component and workflow tests, including inherited reference fixtures |

## Validation and limitations

Run the checks from the repository root:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The GUI suite covers model loading/retry, request validation, concurrent generation, failures, and browser control behavior. Its JavaScript cases require Node.js and are reported as skipped when Node is unavailable.

StoryLM produces readable individual sentences, but longer stories can repeat events or lose narrative consistency. Its 256-token context and roughly 41-million-token training budget constrain what it can learn. The experiments do not establish performance on broader language tasks, and small single-seed differences should not be treated as general architecture rankings. The browser server is a local showcase bound to loopback, rather than a public hosting service.

## Attribution and license

StoryLM began as Boston University CS599 coursework adapted from [Stanford CS336's language-modeling assignment](https://github.com/stanford-cs336/assignment1-basics). I implemented the tokenizer, Transformer, and training pipeline, ran the experiments, and extended the project with reusable artifact workflows, CLI commands, and a browser demo. Inherited starter materials, tests, and reference fixtures retain their upstream attribution.

Training uses [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), by Ronen Eldan and Yuanzhi Li ([paper](https://arxiv.org/abs/2305.07759)). The [pinned dataset card](https://huggingface.co/datasets/roneneldan/TinyStories/blob/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md) specifies **CDLA-Sharing-1.0**; that dataset license is separate from this project's MIT software license.

See [LICENSE.md](LICENSE.md)
