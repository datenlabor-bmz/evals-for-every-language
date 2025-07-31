# AI Language Monitor - System Architecture

This diagram shows the complete data flow from model discovery through evaluation to frontend visualization.

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
    
    %% Language Data
    J["languages.py<br/>BCP-47 + Population"] --> K["Top 100 Languages"]
    
    %% Task Registry
    L["tasks.py<br/>7 Evaluation Tasks"] --> M["Task Functions"]
    M --> M1["translation_from/to<br/>BLEU + ChrF"]
    M --> M2["classification<br/>Accuracy"]
    M --> M3["mmlu<br/>Accuracy"]
    M --> M4["arc<br/>Accuracy"] 
    M --> M5["truthfulqa<br/>Accuracy"]
    M --> M6["mgsm<br/>Accuracy"]
    
    %% On-the-fly Translation
    subgraph OTF [On-the-fly Dataset Translation]
        direction LR
        DS_raw["Raw English Dataset<br/>(e.g., MMLU)"] --> Google_Translate["Google Translate API"]
        Google_Translate --> DS_translated["Translated Dataset<br/>(e.g., German MMLU)"]
    end
    
    %% Evaluation Pipeline
    H --> |"models ID"| N["main.py evaluate"]
    K --> |"languages bcp_47"| N
    L --> |"tasks.items"| N
    N --> |"Filter by model.tasks"| O["Valid Combinations<br/>Model × Language × Task"]
    O --> |"10 samples each"| P["Evaluation Execution"]
    
    %% Task Execution
    P --> Q1[translate_and_evaluate]
    P --> Q2[classify_and_evaluate]
    P --> Q3[mmlu_and_evaluate]
    P --> Q4[arc_and_evaluate]
    P --> Q5[truthfulqa_and_evaluate]
    P --> Q6[mgsm_and_evaluate]
    
    %% API Calls
    Q1 --> |"complete() API"| R["OpenRouter<br/>Model Inference"]
    Q2 --> |"complete() API"| R
    Q3 --> |"complete() API"| R
    Q4 --> |"complete() API"| R
    Q5 --> |"complete() API"| R
    Q6 --> |"complete() API"| R
    
    %% Results Processing
    R --> |Scores| S["Result Aggregation<br/>Mean by model+lang+task"]
    S --> |Save| T[results.json]
    
    %% Backend & Frontend
    T --> |Read| U[backend.py]
    I --> |Read| U
    U --> |make_model_table| V["Model Rankings"]
    U --> |make_country_table| W["Country Aggregation"]
    U --> |"API Endpoint"| X["FastAPI /api/data"]
    X --> |"JSON Response"| Y["Frontend React App"]
    
    %% UI Components
    Y --> Z1["WorldMap.js<br/>Country Visualization"]
    Y --> Z2["ModelTable.js<br/>Model Rankings"]
    Y --> Z3["LanguageTable.js<br/>Language Coverage"]
    Y --> Z4["DatasetTable.js<br/>Task Performance"]
    
    %% Data Sources
    subgraph DS ["Data Sources"]
        DS1["Flores-200<br/>Translation Sentences"]
        DS2["MMLU/AfriMMLU<br/>Knowledge QA"]
        DS3["ARC<br/>Science Reasoning"]
        DS4["TruthfulQA<br/>Truthfulness"]
        DS5["MGSM<br/>Math Problems"]
    end
    
    DS1 --> Q1
    DS2 --> Q3
    DS3 --> Q4
    DS4 --> Q5
    DS5 --> Q6
    
    DS_translated --> Q3
    DS_translated --> Q4
    DS_translated --> Q5
    
    %% Styling
    classDef modelSource fill:#e1f5fe
    classDef evaluation fill:#f3e5f5
    classDef api fill:#fff3e0
    classDef storage fill:#e8f5e8
    classDef frontend fill:#fce4ec
    
    class A1,A2,A3,A4 modelSource
    class Q1,Q2,Q3,Q4,Q5,Q6,P evaluation
    class R,F,G,X api
    class T,I storage
    class Y,Z1,Z2,Z3,Z4 frontend
```

## Architecture Components

### 🔵 Model Discovery (Blue)
- **Static Curated Models**: Handpicked important models for comprehensive evaluation
- **Dynamic Popular Models**: Real-time discovery of trending models via web scraping
- **Quality Control**: Blocklist for problematic or incompatible models
- **Metadata Enrichment**: Rich model information from OpenRouter and HuggingFace APIs

### 🟣 Evaluation Pipeline (Purple)
- **7 Active Tasks**: Translation (bidirectional), Classification, MMLU, ARC, TruthfulQA, MGSM
- **Combinatorial Approach**: Systematic evaluation across Model × Language × Task combinations
- **Sample-based**: 10 evaluations per combination for statistical reliability
- **Unified API**: All tasks use OpenRouter's `complete()` function for consistency

### 🟠 API Integration (Orange)
- **OpenRouter**: Primary model inference API for all language model tasks
- **HuggingFace**: Model metadata and open-source model information
- **Google Translate**: Specialized translation API for comparison baseline

### 🟢 Data Storage (Green)
- **results.json**: Aggregated evaluation scores and metrics
- **models.json**: Dynamic model list with metadata
- **languages.json**: Language information with population data

### 🟡 Frontend Visualization (Pink)
- **WorldMap**: Interactive country-level language proficiency visualization
- **ModelTable**: Ranked model performance leaderboard
- **LanguageTable**: Language coverage and speaker statistics
- **DatasetTable**: Task-specific performance breakdowns

## Data Flow Summary

1. **Model Discovery**: Combine curated + trending models → enrich with metadata
2. **Evaluation Setup**: Generate all valid Model × Language × Task combinations
3. **Task Execution**: Run evaluations using appropriate datasets and APIs
4. **Result Processing**: Aggregate scores and save to JSON files
5. **Backend Serving**: FastAPI serves processed data via REST API
6. **Frontend Display**: React app visualizes data through interactive components

This architecture enables scalable, automated evaluation of AI language models across diverse languages and tasks while providing real-time insights through an intuitive web interface. 