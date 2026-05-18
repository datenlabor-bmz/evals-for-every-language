# Results snapshots

This directory holds the latest aggregated output of `evals/main.py`. It is the inspection point for the dashboard's data, and it is republished to the Hugging Face Space on every push to `main`.

## Files

| File | What it is | Source of truth? |
| --- | --- | --- |
| `results.json` | Mean score per `(model, bcp_47, task, metric)`, with `origin` tagging human vs. machine-translated data. | A snapshot — the dataset `fair-forward/evals-for-every-language-results` on the HuggingFace Hub is the authoritative copy and may be more current than what's committed here. |
| `languages.json` | Per-language metadata (BCP-47, name, autonym, speaker count, Glottolog family, FLORES+/FLEURS/CommonVoice subset paths, `in_benchmark` flag). | Snapshot of `fair-forward/evals-for-every-language-languages`. |
| `models.json` | Per-model metadata (id, provider, cost per 1M tokens, license, training-on-prompts flag, creation date, HF id, parameter count). | Snapshot of `fair-forward/evals-for-every-language-models`. |
| `results-detailed.json` | Per-sample log (one row per `sentence_nr`). Used by the dashboard to compute bootstrap confidence intervals. **Not committed** — too large; fetch from `fair-forward/evals-for-every-language-results-detailed`. | — |
| `model_failure_stats*.csv` | Ad-hoc analysis exports. **Not committed.** | — |

## How to load

```python
import json
rows = json.load(open("results/results.json"))
# or
from datasets import load_dataset
df = load_dataset("fair-forward/evals-for-every-language-results")["train"].to_pandas()
```

## Refreshing

`evals/main.py` regenerates all four `*.json` files locally and pushes the same four tables to their Hugging Face datasets. After a run, commit the changed files in `results/` and push to `main`; the `.github/workflows/huggingface-upload.yml` workflow republishes the repo tree to the Space.
