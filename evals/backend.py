import json
import os

import numpy as np
import pandas as pd
import uvicorn

from countries import make_country_table
from datasets_.util import load
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

scores = load("results")
languages = load("languages")
models = load("models")


def mean(lst):
    return sum(lst) / len(lst) if lst else None


task_metrics = [
    "translation_from_bleu",
    "translation_to_bleu",
    "classification_accuracy",
    "mmlu_accuracy",
    "arc_accuracy",
    "truthfulqa_accuracy",
    "mgsm_accuracy",
]


def compute_normalized_average(df, metrics):
    """Compute average of min-max normalized metric columns."""
    normalized_df = df[metrics].copy()
    for col in metrics:
        if col in normalized_df.columns:
            col_min = normalized_df[col].min()
            col_max = normalized_df[col].max()
            if col_max > col_min:  # Avoid division by zero
                normalized_df[col] = (normalized_df[col] - col_min) / (
                    col_max - col_min
                )
            else:
                normalized_df[col] = 0  # If all values are the same, set to 0
    return normalized_df.mean(axis=1, skipna=False)


def make_model_table(scores_df, models):
    scores_df = scores_df.copy()
    # Create a combined task_metric for origin
    scores_df["task_metric_origin"] = (
        scores_df["task"] + "_" + scores_df["metric"] + "_" + scores_df["origin"]
    )

    # Pivot to get scores for each origin-specific metric
    scores_pivot = scores_df.pivot_table(
        index="model",
        columns="task_metric_origin",
        values="score",
        aggfunc="mean",
    )

    # Create the regular task_metric for the main average calculation
    scores_df["task_metric"] = scores_df["task"] + "_" + scores_df["metric"]
    main_pivot = scores_df.pivot_table(
        index="model", columns="task_metric", values="score", aggfunc="mean"
    )

    # Merge the two pivots
    df = pd.merge(main_pivot, scores_pivot, on="model", how="outer")

    for metric in task_metrics:
        if metric not in df.columns:
            df[metric] = np.nan

    df["average"] = compute_normalized_average(df, task_metrics)

    # Add flag if any machine-origin data was used
    machine_presence = scores_df[scores_df["origin"] == "machine"].groupby(["model", "task_metric"]).size()
    for metric in task_metrics:
        df[f"{metric}_contains_machine"] = df.index.map(lambda m: (m, metric) in machine_presence.index)
    df = df.sort_values(by="average", ascending=False).reset_index()
    df = pd.merge(df, models, left_on="model", right_on="id", how="left")
    df["rank"] = df.index + 1

    # Dynamically find all metric columns to include
    final_cols = df.columns
    metric_cols = [m for m in final_cols if any(tm in m for tm in task_metrics)]

    df["creation_date"] = df["creation_date"].apply(lambda x: x.isoformat() if x else None)

    df = df[
        [
            "rank",
            "model",
            "name",
            "provider_name",
            "hf_id",
            "creation_date",
            "size",
            "type",
            "license",
            "cost",
            "average",
            *sorted(list(set(metric_cols))),
        ]
    ]
    return df


def make_language_table(scores_df, languages):
    scores_df = scores_df.copy()
    scores_df["task_metric"] = scores_df["task"] + "_" + scores_df["metric"]
    
    # Pivot scores
    score_pivot = scores_df.pivot_table(
        index="bcp_47", columns="task_metric", values="score", aggfunc="mean"
    )
    
    # Pivot origins (first origin since each task+lang combo has only one)
    origin_pivot = scores_df.pivot_table(
        index="bcp_47", columns="task_metric", values="origin", aggfunc="first"
    )
    origin_pivot = origin_pivot.add_suffix("_origin")
    
    df = pd.merge(score_pivot, origin_pivot, on="bcp_47", how="outer")
    
    for metric in task_metrics:
        if metric not in df.columns:
            df[metric] = np.nan

    df["average"] = compute_normalized_average(df, task_metrics)
    df = pd.merge(languages, df, on="bcp_47", how="outer")
    df = df.sort_values(by="speakers", ascending=False)

    # Dynamically find all metric columns to include
    final_cols = df.columns
    metric_cols = [m for m in final_cols if any(tm in m for tm in task_metrics)]

    df = df[
        [
            "bcp_47",
            "language_name",
            "autonym",
            "speakers",
            "family",
            "average",
            "in_benchmark",
            *sorted(list(set(metric_cols))),
        ]
    ]
    return df


def make_language_tier_history(scores_df, languages, models):
    # Rank languages by speakers
    ranked_langs = languages.sort_values(by="speakers", ascending=False).reset_index(drop=True)
    
    # Define tiers
    tier_ranges = {
        "Top 1": (0, 1),
        "Top 2-20": (1, 20),
        "Top 20-200": (19, 500),
    }
    
    # Calculate model-language proficiency scores
    scores_df = scores_df.copy()
    scores_df["task_metric"] = scores_df["task"] + "_" + scores_df["metric"]
    
    # Pivot to get model-language-metric scores
    pivot = scores_df.pivot_table(
        index=["model", "bcp_47"],
        columns="task_metric",
        values="score",
        aggfunc="mean"
    )
    
    # Ensure all task_metrics columns exist
    for metric in task_metrics:
        if metric not in pivot.columns:
            pivot[metric] = np.nan
    
    # Calculate proficiency score for each model-language pair
    pivot["proficiency_score"] = compute_normalized_average(pivot, task_metrics)
    pivot = pivot.reset_index()
    
    # Create all tier-level aggregations (allowing overlapping tiers)
    all_tier_scores = []
    for tier_name, (start, end) in tier_ranges.items():
        tier_langs = ranked_langs.iloc[start:end]["bcp_47"].tolist()
        tier_data = pivot[pivot["bcp_47"].isin(tier_langs)]
        tier_scores = tier_data.groupby("model")["proficiency_score"].mean().reset_index()
        tier_scores["tier"] = tier_name
        all_tier_scores.append(tier_scores)
    
    tier_scores = pd.concat(all_tier_scores, ignore_index=True)
    
    # Merge with models data
    tier_scores = pd.merge(tier_scores, models, left_on="model", right_on="id", how="left")
    
    # Select relevant columns
    tier_scores = tier_scores[
        ["model", "name", "provider_name", "creation_date", "size", "tier", "proficiency_score"]
    ]
    
    tier_scores["creation_date"] = tier_scores["creation_date"].apply(
        lambda x: x.isoformat() if x else None
    )
    
    return tier_scores


def make_license_history(scores_df, models):
    scores_df = scores_df.copy()
    scores_df["task_metric"] = scores_df["task"] + "_" + scores_df["metric"]
    
    # Pivot to get model-level scores
    pivot = scores_df.pivot_table(
        index="model",
        columns="task_metric",
        values="score",
        aggfunc="mean"
    )
    
    # Ensure all task_metrics columns exist
    for metric in task_metrics:
        if metric not in pivot.columns:
            pivot[metric] = np.nan
    
    # Calculate proficiency score for each model
    pivot["proficiency_score"] = compute_normalized_average(pivot, task_metrics)
    pivot = pivot.reset_index()
    
    # Merge with models data
    df = pd.merge(pivot, models, left_on="model", right_on="id", how="left")
    
    # Classify as commercial or open
    df["license_type"] = df["type"].apply(
        lambda x: "Open-source" if x == "open-source" else "Commercial"
    )
    
    # Select relevant columns
    df = df[
        ["model", "name", "provider_name", "creation_date", "size", "license_type", "proficiency_score"]
    ]
    
    df["creation_date"] = df["creation_date"].apply(
        lambda x: x.isoformat() if x else None
    )
    
    return df


app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)


def serialize(df):
    return df.replace({np.nan: None}).to_dict(orient="records")


@app.post("/api/data")
async def data(request: Request):
    body = await request.body()
    data = json.loads(body)
    selected_languages = data.get("selectedLanguages", {})
    
    # Identify which metrics have machine translations available
    machine_translated_metrics = {
        f"{row['task']}_{row['metric']}" 
        for _, row in scores.iterrows() 
        if row["origin"] == "machine"
    }

    # Filter by selected languages if provided
    df = scores[scores["bcp_47"].isin(lang["bcp_47"] for lang in selected_languages)] if selected_languages else scores
    
    if len(df) == 0:
        model_table = pd.DataFrame()
        countries = pd.DataFrame()
    else:
        model_table = make_model_table(df, models)
        countries = make_country_table(make_language_table(df, languages))
    
    language_table = make_language_table(scores, languages)
    language_tier_history = make_language_tier_history(scores, languages, models)
    license_history = make_license_history(scores, models)
    datasets_df = pd.read_json("data/datasets.json")
    
    return JSONResponse(content={
        "model_table": serialize(model_table),
        "language_table": serialize(language_table),
        "dataset_table": serialize(datasets_df),
        "countries": serialize(countries),
        "machine_translated_metrics": list(machine_translated_metrics),
        "language_tier_history": serialize(language_tier_history),
        "license_history": serialize(license_history),
    })


# Only serve static files if build directory exists
if os.path.exists("frontend/build"):
    app.mount("/", StaticFiles(directory="frontend/build", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
