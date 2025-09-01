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

```mermaid
flowchart TD
    %% Model Sources
    A1["important_models<br/>Static Curated List"] --> D[load_models]
    A2["get_historical_popular_models<br/>Web Scraping - Top 20"] --> D
    A3["get_current_popular_models<br/>Web Scraping - Top 10"] --> D
    A4["blocklist<br/>Exclusions"] --> D
    
    %% Model Processing
    D --> |"Combine & Dedupe"| E["Dynamic Model List<br/>~40-50 models"]
    E --> |get_or_metadata| F["OpenRouter API<br/>Model Metadata"]
    F --> |get_hf_metadata| G["HuggingFace API<br/>Model Details"]
    G --> H["Enriched Model DataFrame"]
    H --> |Save| I[models.json]
    
    %% Model Validation & Cost Filtering
    H --> |"Validate Models<br/>Check API Availability"| H1["Valid Models Only<br/>Cost ≤ $20/1M tokens"]
    H1 --> |"Timeout Protection<br/>120s for Large Models"| H2["Robust Model List"]
    
    %% Language Data
    J["languages.py<br/>BCP-47 + Population"] --> K["Top 100 Languages"]
    
    %% Task Registry with Unified Prompting
    L["tasks.py<br/>7 Evaluation Tasks"] --> M["Task Functions<br/>Unified English Zero-Shot"]
    M --> M1["translation_from/to<br/>BLEU + ChrF"]
    M --> M2["classification<br/>Accuracy"]
    M --> M3["mmlu<br/>Accuracy"]
    M --> M4["arc<br/>Accuracy"] 
    M --> M5["truthfulqa<br/>Accuracy"]
    M --> M6["mgsm<br/>Accuracy"]
    
    %% On-the-fly Translation with Origin Tagging
    subgraph OTF [On-the-fly Dataset Translation]
        direction LR
        DS_raw["Raw English Dataset<br/>(e.g., MMLU)"] --> Google_Translate["Google Translate API"]
        Google_Translate --> DS_translated["Translated Dataset<br/>(e.g., German MMLU)<br/>Origin: 'machine'"]
        DS_native["Native Dataset<br/>(e.g., German MMLU)<br/>Origin: 'human'"]
    end
    
    %% Evaluation Pipeline
    H2 --> |"models ID"| N["main.py / main_gcs.py<br/>evaluate"]
    K --> |"languages bcp_47"| N
    L --> |"tasks.items"| N
    N --> |"Filter by model.tasks"| O["Valid Combinations<br/>Model × Language × Task"]
    O --> |"10 samples each"| P["Evaluation Execution<br/>Batch Processing"]
    
    %% Task Execution with Origin Tracking
    P --> Q1[translate_and_evaluate<br/>Origin: 'human']
    P --> Q2[classify_and_evaluate<br/>Origin: 'human']
    P --> Q3[mmlu_and_evaluate<br/>Origin: 'human'/'machine']
    P --> Q4[arc_and_evaluate<br/>Origin: 'human'/'machine']
    P --> Q5[truthfulqa_and_evaluate<br/>Origin: 'human'/'machine']
    P --> Q6[mgsm_and_evaluate<br/>Origin: 'human'/'machine']
    
    %% API Calls with Error Handling
    Q1 --> |"complete() API<br/>Rate Limiting"| R["OpenRouter<br/>Model Inference"]
    Q2 --> |"complete() API<br/>Rate Limiting"| R
    Q3 --> |"complete() API<br/>Rate Limiting"| R
    Q4 --> |"complete() API<br/>Rate Limiting"| R
    Q5 --> |"complete() API<br/>Rate Limiting"| R
    Q6 --> |"complete() API<br/>Rate Limiting"| R
    
    %% Results Processing with Origin Aggregation
    R --> |Scores| S["Result Aggregation<br/>Mean by model+lang+task+origin"]
    S --> |Save| T[results.json]
    
    %% Backend & Frontend with Origin-Specific Metrics
    T --> |Read| U[backend.py]
    I --> |Read| U
    U --> |make_model_table| V["Model Rankings<br/>Origin-Specific Metrics"]
    U --> |make_country_table| W["Country Aggregation"]
    U --> |"API Endpoint"| X["FastAPI /api/data<br/>arc_accuracy_human<br/>arc_accuracy_machine"]
    X --> |"JSON Response"| Y["Frontend React App"]
    
    %% UI Components
    Y --> Z1["WorldMap.js<br/>Country Visualization"]
    Y --> Z2["ModelTable.js<br/>Model Rankings"]
    Y --> Z3["LanguageTable.js<br/>Language Coverage"]
    Y --> Z4["DatasetTable.js<br/>Task Performance"]
    
    %% Data Sources with Origin Information
    subgraph DS ["Data Sources"]
        DS1["Flores-200<br/>Translation Sentences<br/>Origin: 'human'"]
        DS2["MMLU/AfriMMLU<br/>Knowledge QA<br/>Origin: 'human'"]
        DS3["ARC<br/>Science Reasoning<br/>Origin: 'human'"]
        DS4["TruthfulQA<br/>Truthfulness<br/>Origin: 'human'"]
        DS5["MGSM<br/>Math Problems<br/>Origin: 'human'"]
    end
    
    DS1 --> Q1
    DS2 --> Q3
    DS3 --> Q4
    DS4 --> Q5
    DS5 --> Q6
    
    DS_translated --> Q3
    DS_translated --> Q4
    DS_translated --> Q5
    
    DS_native --> Q3
    DS_native --> Q4
    DS_native --> Q5
    
    %% Styling - Neutral colors that work in both dark and light modes
    classDef modelSource fill:#f8f9fa,stroke:#6c757d,color:#212529
    classDef evaluation fill:#e9ecef,stroke:#495057,color:#212529
    classDef api fill:#dee2e6,stroke:#6c757d,color:#212529
    classDef storage fill:#d1ecf1,stroke:#0c5460,color:#0c5460
    classDef frontend fill:#f8d7da,stroke:#721c24,color:#721c24
    classDef translation fill:#d4edda,stroke:#155724,color:#155724
    
    class A1,A2,A3,A4 modelSource
    class Q1,Q2,Q3,Q4,Q5,Q6,P evaluation
    class R,F,G,X api
    class T,I storage
    class Y,Z1,Z2,Z3,Z4 frontend
    class Google_Translate,DS_translated,DS_native translation
```

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

### Google Cloud Deployment
```bash
uv run --extra dev evals/main_gcs.py
```

## Explore

```bash
uv run evals/backend.py
cd frontend && npm i && npm start
```
