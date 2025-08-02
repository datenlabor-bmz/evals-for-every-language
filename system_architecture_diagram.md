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
    
    %% Styling - Dark theme friendly colors
    classDef modelSource fill:#2d3748,stroke:#4a5568,color:#e2e8f0
    classDef evaluation fill:#553c9a,stroke:#805ad5,color:#e9d8fd
    classDef api fill:#c05621,stroke:#ed8936,color:#fed7aa
    classDef storage fill:#22543d,stroke:#38a169,color:#c6f6d5
    classDef frontend fill:#702459,stroke:#b83280,color:#fed7e2
    classDef translation fill:#2c5282,stroke:#3182ce,color:#bee3f8
    
    class A1,A2,A3,A4 modelSource
    class Q1,Q2,Q3,Q4,Q5,Q6,P evaluation
    class R,F,G,X api
    class T,I storage
    class Y,Z1,Z2,Z3,Z4 frontend
    class Google_Translate,DS_translated,DS_native translation
```

## Architecture Components

### 🔵 Model Discovery (Dark Blue)
- **Static Curated Models**: Handpicked important models for comprehensive evaluation
- **Dynamic Popular Models**: Real-time discovery of trending models via web scraping
- **Quality Control**: Blocklist for problematic or incompatible models
- **Model Validation**: API availability checks and cost filtering (≤$20/1M tokens)
- **Timeout Protection**: 120s timeout for large/reasoning models, 60s for others
- **Metadata Enrichment**: Rich model information from OpenRouter and HuggingFace APIs

### 🟣 Evaluation Pipeline (Purple)
- **7 Active Tasks**: Translation (bidirectional), Classification, MMLU, ARC, TruthfulQA, MGSM
- **Unified English Zero-Shot Prompting**: All tasks use English instructions with target language content
- **Origin Tagging**: Distinguishes between human-translated ('human') and machine-translated ('machine') data
- **Combinatorial Approach**: Systematic evaluation across Model × Language × Task combinations
- **Sample-based**: 10 evaluations per combination for statistical reliability
- **Batch Processing**: 50 tasks per batch with rate limiting and error resilience
- **Dual Deployment**: `main.py` for local/GitHub, `main_gcs.py` for Google Cloud with GCS storage

### 🟠 API Integration (Orange)
- **OpenRouter**: Primary model inference API for all language model tasks
- **Rate Limiting**: Intelligent batching and delays to prevent API overload
- **Error Handling**: Graceful handling of timeouts, rate limits, and model unavailability
- **HuggingFace**: Model metadata and open-source model information
- **Google Translate**: Specialized translation API for on-the-fly dataset translation

### 🟢 Data Storage (Green)
- **results.json**: Aggregated evaluation scores with origin-specific metrics
- **models.json**: Dynamic model list with metadata and validation status
- **languages.json**: Language information with population data

### 🟡 Frontend Visualization (Pink)
- **WorldMap**: Interactive country-level language proficiency visualization
- **ModelTable**: Ranked model performance leaderboard with origin-specific columns
- **LanguageTable**: Language coverage and speaker statistics
- **DatasetTable**: Task-specific performance breakdowns with human/machine distinction

### 🔵 Translation & Origin Tracking (Blue)
- **On-the-fly Translation**: Google Translate API for languages without native benchmarks
- **Origin Tagging**: Automatic classification of data sources (human vs. machine translated)
- **Separate Metrics**: Frontend displays distinct scores for human and machine-translated data

## Data Flow Summary

1. **Model Discovery**: Combine curated + trending models → validate API availability → enrich with metadata
2. **Evaluation Setup**: Generate all valid Model × Language × Task combinations with origin tracking
3. **Task Execution**: Run evaluations using unified English prompting and appropriate datasets
4. **Result Processing**: Aggregate scores by model+language+task+origin and save to JSON files
5. **Backend Serving**: FastAPI serves processed data with origin-specific metrics via REST API
6. **Frontend Display**: React app visualizes data through interactive components with transparency indicators

This architecture enables scalable, automated evaluation of AI language models across diverse languages and tasks while providing real-time insights through an intuitive web interface with methodological transparency. 