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
from joblib.memory import Memory

cache = Memory(location=".cache", verbose=0).cache

scores = load("results")
scores_detailed = load("results-detailed")
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
    "mgsm_accuracy",
]


def compute_normalized_average(df, metrics):
    """Compute simple average across metric columns without normalization."""
    return df[metrics].mean(axis=1, skipna=False)


@cache
def compute_bootstrap_ci(
    data_hash, group_cols_tuple, n_bootstrap=1000, ci_level=0.95, seed=42
):
    """Compute bootstrap CIs for grouped data. Cached based on data hash."""
    # This function is called with the actual data passed separately via _ci_cache
    df, group_cols = _ci_cache[data_hash]
    np.random.seed(seed)
    percentiles = [(1 - ci_level) / 2 * 100, (1 + ci_level) / 2 * 100]

    def bootstrap_group(group):
        scores = group["score"].values
        if len(scores) == 0:
            return pd.Series({"ci_lower": None, "ci_upper": None})
        bootstrap_means = [
            np.random.choice(scores, len(scores), replace=True).mean()
            for _ in range(n_bootstrap)
        ]
        ci_lower, ci_upper = np.percentile(bootstrap_means, percentiles)
        return pd.Series({"ci_lower": ci_lower, "ci_upper": ci_upper})

    result = df.groupby(group_cols, as_index=False).apply(
        bootstrap_group, include_groups=False
    )
    result.columns = group_cols + ["ci_lower", "ci_upper"]
    return result


# Thread-safe cache for passing DataFrames to cached function
_ci_cache = {}


def add_confidence_intervals(df, scores_df_detailed, group_col, metrics):
    """DRY helper to add CI columns for metrics and average to a dataframe."""
    if scores_df_detailed is None or scores_df_detailed.empty:
        return df

    detailed = scores_df_detailed.copy()
    detailed["task_metric"] = detailed["task"] + "_" + detailed["metric"]

    # Add CI for each metric
    for metric in metrics:
        metric_data = detailed[detailed["task_metric"] == metric]
        if not metric_data.empty:
            # Create hash based on data shape, groups, and statistics
            group_stats = (
                metric_data.groupby(group_col)["score"]
                .agg(["count", "mean", "std"])
                .round(6)
            )
            data_hash = hash(
                (
                    metric,
                    group_col,
                    len(metric_data),
                    tuple(group_stats.index),
                    tuple(map(tuple, group_stats.values)),
                )
            )
            _ci_cache[data_hash] = (metric_data, [group_col])
            ci_df = compute_bootstrap_ci(data_hash, (group_col,))
            ci_df = ci_df.rename(
                columns={
                    "ci_lower": f"{metric}_ci_lower",
                    "ci_upper": f"{metric}_ci_upper",
                }
            )
            df = pd.merge(df, ci_df, on=group_col, how="left")

    # Add CI for average
    avg_data = detailed[detailed["task_metric"].isin(metrics)]
    if not avg_data.empty:
        # Create hash based on data shape, groups, and statistics
        group_stats = (
            avg_data.groupby(group_col)["score"].agg(["count", "mean", "std"]).round(6)
        )
        data_hash = hash(
            (
                "average",
                group_col,
                len(avg_data),
                tuple(group_stats.index),
                tuple(map(tuple, group_stats.values)),
            )
        )
        _ci_cache[data_hash] = (avg_data, [group_col])
        avg_ci_df = compute_bootstrap_ci(data_hash, (group_col,))
        avg_ci_df = avg_ci_df.rename(
            columns={"ci_lower": "average_ci_lower", "ci_upper": "average_ci_upper"}
        )
        df = pd.merge(df, avg_ci_df, on=group_col, how="left")

    return df


def make_model_table(scores_df, models, scores_df_detailed=None):
    scores_df = scores_df.copy()
    scores_df["task_metric"] = scores_df["task"] + "_" + scores_df["metric"]
    scores_df["task_metric_origin"] = (
        scores_df["task_metric"] + "_" + scores_df["origin"]
    )

    # Pivot scores
    main_pivot = scores_df.pivot_table(
        index="model", columns="task_metric", values="score", aggfunc="mean"
    )
    scores_pivot = scores_df.pivot_table(
        index="model", columns="task_metric_origin", values="score", aggfunc="mean"
    )
    df = pd.merge(main_pivot, scores_pivot, on="model", how="outer")

    # Fill missing metrics and compute average
    for metric in task_metrics:
        df[metric] = df.get(metric, np.nan)
    df["average"] = compute_normalized_average(df, task_metrics)
    df = add_confidence_intervals(df, scores_df_detailed, "model", task_metrics)

    # Add machine-origin flags
    machine_presence = (
        scores_df[scores_df["origin"] == "machine"]
        .groupby(["model", "task_metric"])
        .size()
    )
    for metric in task_metrics:
        df[f"{metric}_contains_machine"] = df.index.map(
            lambda m: (m, metric) in machine_presence.index
        )

    # Sort and add metadata
    df = df.sort_values(by="average", ascending=False).reset_index()
    df = pd.merge(df, models, left_on="model", right_on="id", how="left")
    df["rank"] = df.index + 1
    df["creation_date"] = df["creation_date"].apply(
        lambda x: x.isoformat() if x else None
    )

    # Select columns dynamically
    metric_cols = [m for m in df.columns if any(tm in m for tm in task_metrics)]
    avg_ci_cols = [
        c for c in df.columns if c in ["average_ci_lower", "average_ci_upper"]
    ]

    return df[
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
            *avg_ci_cols,
            *sorted(set(metric_cols)),
        ]
    ]


def make_language_table(scores_df, languages, scores_df_detailed=None):
    scores_df = scores_df.copy()
    scores_df["task_metric"] = scores_df["task"] + "_" + scores_df["metric"]

    # Pivot scores and origins
    score_pivot = scores_df.pivot_table(
        index="bcp_47", columns="task_metric", values="score", aggfunc="mean"
    )
    origin_pivot = scores_df.pivot_table(
        index="bcp_47", columns="task_metric", values="origin", aggfunc="first"
    )
    origin_pivot = origin_pivot.add_suffix("_origin")
    df = pd.merge(score_pivot, origin_pivot, on="bcp_47", how="outer")

    # Fill missing metrics and compute average
    for metric in task_metrics:
        df[metric] = df.get(metric, np.nan)
    df["average"] = compute_normalized_average(df, task_metrics)

    # For language table, we need to compute scores from detailed data to match CI calculation
    # (CI is computed from all samples, so score should be too)
    if scores_df_detailed is not None and not scores_df_detailed.empty:
        detailed = scores_df_detailed.copy()
        detailed["task_metric"] = detailed["task"] + "_" + detailed["metric"]
        detailed_pivot = detailed.pivot_table(
            index="bcp_47", columns="task_metric", values="score", aggfunc="mean"
        )
        for metric in task_metrics:
            if metric in detailed_pivot.columns:
                df[metric] = detailed_pivot[metric]
        df["average"] = compute_normalized_average(df, task_metrics)

    df = add_confidence_intervals(df, scores_df_detailed, "bcp_47", task_metrics)

    # Merge with language metadata and sort
    df = pd.merge(languages, df, on="bcp_47", how="outer").sort_values(
        by="speakers", ascending=False
    )

    # Select columns dynamically
    metric_cols = [m for m in df.columns if any(tm in m for tm in task_metrics)]
    avg_ci_cols = [
        c for c in df.columns if c in ["average_ci_lower", "average_ci_upper"]
    ]

    return df[
        [
            "bcp_47",
            "language_name",
            "autonym",
            "speakers",
            "family",
            "average",
            *avg_ci_cols,
            "in_benchmark",
            *sorted(set(metric_cols)),
        ]
    ]


def make_language_tier_history(scores_df, languages, models):
    ranked_langs = languages.sort_values(by="speakers", ascending=False).reset_index(
        drop=True
    )
    tier_ranges = {"Top 1": (0, 1), "Top 2-20": (1, 20), "Top 20-200": (19, 500)}

    # Calculate model-language proficiency scores
    scores_df = scores_df.copy()
    scores_df["task_metric"] = scores_df["task"] + "_" + scores_df["metric"]
    pivot = scores_df.pivot_table(
        index=["model", "bcp_47"], columns="task_metric", values="score", aggfunc="mean"
    )
    for metric in task_metrics:
        pivot[metric] = pivot.get(metric, np.nan)
    pivot["proficiency_score"] = compute_normalized_average(pivot, task_metrics)
    pivot = pivot.reset_index()

    # Aggregate by tier
    tier_scores = pd.concat(
        [
            pivot[pivot["bcp_47"].isin(ranked_langs.iloc[start:end]["bcp_47"])]
            .groupby("model")["proficiency_score"]
            .mean()
            .reset_index()
            .assign(tier=tier_name)
            for tier_name, (start, end) in tier_ranges.items()
        ],
        ignore_index=True,
    )

    tier_scores = pd.merge(
        tier_scores, models, left_on="model", right_on="id", how="left"
    )
    tier_scores["creation_date"] = tier_scores["creation_date"].apply(
        lambda x: x.isoformat() if x else None
    )

    return tier_scores[
        [
            "model",
            "name",
            "provider_name",
            "creation_date",
            "size",
            "tier",
            "proficiency_score",
        ]
    ]


def make_license_history(scores_df, models):
    scores_df = scores_df.copy()
    scores_df["task_metric"] = scores_df["task"] + "_" + scores_df["metric"]

    # Pivot and compute proficiency
    pivot = scores_df.pivot_table(
        index="model", columns="task_metric", values="score", aggfunc="mean"
    )
    for metric in task_metrics:
        pivot[metric] = pivot.get(metric, np.nan)
    pivot["proficiency_score"] = compute_normalized_average(pivot, task_metrics)

    # Merge and classify
    df = pd.merge(
        pivot.reset_index(), models, left_on="model", right_on="id", how="left"
    )
    df["license_type"] = df["type"].apply(
        lambda x: "Open-source" if x == "open-source" else "Commercial"
    )
    df["creation_date"] = df["creation_date"].apply(
        lambda x: x.isoformat() if x else None
    )

    return df[
        [
            "model",
            "name",
            "provider_name",
            "creation_date",
            "size",
            "license_type",
            "proficiency_score",
        ]
    ]


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
    df = (
        scores[scores["bcp_47"].isin(lang["bcp_47"] for lang in selected_languages)]
        if selected_languages
        else scores
    )
    df_detailed = (
        scores_detailed[
            scores_detailed["bcp_47"].isin(
                lang["bcp_47"] for lang in selected_languages
            )
        ]
        if selected_languages
        else scores_detailed
    )

    if len(df) == 0:
        model_table = pd.DataFrame()
        countries = pd.DataFrame()
    else:
        model_table = make_model_table(df, models, df_detailed)
        countries = make_country_table(make_language_table(df, languages, df_detailed))

    language_table = make_language_table(scores, languages, scores_detailed)
    language_tier_history = make_language_tier_history(scores, languages, models)
    license_history = make_license_history(scores, models)
    datasets_df = pd.read_json("data/datasets.json")

    return JSONResponse(
        content={
            "model_table": serialize(model_table),
            "language_table": serialize(language_table),
            "dataset_table": serialize(datasets_df),
            "countries": serialize(countries),
            "machine_translated_metrics": list(machine_translated_metrics),
            "language_tier_history": serialize(language_tier_history),
            "license_history": serialize(license_history),
        }
    )


# Only serve static files if build directory exists
if os.path.exists("frontend/build"):
    app.mount("/", StaticFiles(directory="frontend/build", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
