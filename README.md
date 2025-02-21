---
title: AI Language Monitor
emoji: 🌍
colorFrom: purple
colorTo: pink
sdk: gradio
license: cc-by-sa-4.0
short_description: Evaluating LLM performance across all human languages.
datasets:
- openlanguagedata/flores_plus
- mozilla-foundation/common_voice_1_0
models:
- meta-llama/Llama-3.3-70B-Instruct
- mistralai/Mistral-Small-24B-Instruct-2501
- deepseek-ai/DeepSeek-V3
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
---

<!--
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference 
For tag meaning, see https://huggingface.co/spaces/leaderboards/LeaderboardsExplorer
-->

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-purple)](https://huggingface.co/spaces/datenlabor-bmz/ai-language-monitor)

# AI Language Monitor 🌍

Benchmarking all big AI models on all benchmarkable languages.

Sources:

1. For AI models: [OpenRouter](https://openrouter.ai/)
2. For language benchmarks: [FLORES+](https://github.com/openlanguagedata/flores)
3. For language statistics: [Wikidata](https://gist.github.com/unhammer/3e8f2e0f79972bf5008a4c970081502d), [Ethnologue](https://www.ethnologue.com/browse/names/)

[UI sketch](https://www.tldraw.com/ro/5YkWi9dfBixOkQ4FV23zA?d=v192.-1.2090.1569.page)
