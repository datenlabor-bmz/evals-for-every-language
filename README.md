---
title: AI Language Monitor
emoji: 🌍
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 8000
license: cc-by-sa-4.0
short_description: Evaluating LLM performance across all human languages.
datasets:
- openlanguagedata/flores_plus
- google/fleurs
- mozilla-foundation/common_voice_1_0
- CohereForAI/Global-MMLU
models:
- meta-llama/Llama-3.3-70B-Instruct
- mistralai/Mistral-Small-24B-Instruct-2501
- deepseek-ai/DeepSeek-V3
- microsoft/phi-4
- openai/whisper-large-v3
- google/gemma-3-27b-it
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
---

<!--
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference 
For tag meaning, see https://huggingface.co/spaces/leaderboards/LeaderboardsExplorer
-->


[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-purple)](https://huggingface.co/spaces/datenlabor-bmz/ai-language-monitor)

# AI Language Monitor 🌍

_Tracking language proficiency of AI models for every language_

## System Architecture

The AI Language Monitor evaluates language models across 100+ languages using a comprehensive pipeline that combines model discovery, automated evaluation, and real-time visualization.

> **Detailed Architecture**: See [system_architecture_diagram.md](system_architecture_diagram.md) for the complete system architecture diagram and component descriptions.

**Key Features:**
- **Model Discovery**: Combines curated models with real-time trending models via web scraping
- **Multi-Task Evaluation**: 7 tasks across 100+ languages with origin tracking (human vs machine-translated)
- **Scalable Architecture**: Dual deployment (local/GitHub vs Google Cloud)
- **Real-time Visualization**: Interactive web interface with country-level insights

## Evaluate

### Local Development
```bash
uv run --extra dev evals/main.py
```

## Explore

```bash
uv run evals/backend.py
cd frontend && npm i && npm start
```
