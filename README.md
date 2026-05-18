---
title: languagebench
emoji: 🌍
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 8000
license: cc-by-sa-4.0
short_description: AI model evaluations for every language in the world.
datasets:
- openlanguagedata/flores_plus
- CohereForAI/Global-MMLU
- masakhane/afrimmlu
- masakhane/afrimgsm
- masakhane/uhura-truthfulqa
models:
- openai/gpt-5
- anthropic/claude-opus-4.5
- google/gemini-3-pro-preview
- meta-llama/llama-3.3-70b-instruct
- deepseek/deepseek-v3.2-exp
- mistralai/mistral-medium-3.1
- google/gemma-3-27b-it
- microsoft/phi-4
tags:
- leaderboard
- submission:manual
- test:public
- judge:auto
- modality:text
- modality:artefacts
- eval:generation
- language:English
- language:German
- language:Chinese
- language:Hindi
- language:Spanish
- language:Arabic
- language:Swahili
- language:Yoruba
---

<!-- The lists above are HF Spaces discoverability facets, not the full evaluation set. The canonical model list lives in evals/models.py (~42 curated frontier models); datasets are wired in evals/datasets_/. -->


<!--
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference 
For tag meaning, see https://huggingface.co/spaces/leaderboards/LeaderboardsExplorer
-->


[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-purple)](https://huggingface.co/spaces/fair-forward/languagebench)

# languagebench 🌍

_AI model evaluations for every language in the world_

## Inspect the latest results

The most recent end-to-end evaluation snapshot lives in [`results/`](results/):

- `results/results.json` — aggregated scores per (model, language, task, metric)
- `results/languages.json` — language metadata (BCP-47 code, name, speaker count, family, script)
- `results/models.json` — model metadata (provider, size, license, cost, creation date)

These are the same tables the dashboard renders. For programmatic access, including the per-sample log with confidence-interval data, pull the canonical Hugging Face datasets:

```python
from datasets import load_dataset
results = load_dataset("fair-forward/evals-for-every-language-results")["train"].to_pandas()
detailed = load_dataset("fair-forward/evals-for-every-language-results-detailed")["train"].to_pandas()
```

## Evaluate

### Local Development
```bash
uv sync --group dev
uv run evals/main.py
```

## Explore

```bash
uv run evals/backend.py
cd frontend && npm i && npm start
```

## System Architecture

See [notes/system-architecture-diagram.md](notes/system-architecture-diagram.md) for the complete system architecture diagram and component descriptions. The accompanying paper is [_The AI Language Proficiency Monitor – Tracking the Progress of LLMs on Multilingual Benchmarks_](https://arxiv.org/abs/2507.08538) (Pomerenke, Nothnagel, & Ostermann, 2025).
