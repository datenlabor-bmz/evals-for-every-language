import asyncio
import json

import numpy as np
import pandas as pd
from rich import print
from tqdm.asyncio import tqdm_asyncio
from languages import languages
from tasks import tasks
from models import models, model_fast

# ===== config =====

n_sentences = 30
langs_eval = languages.iloc[:10]
langs_eval_detailed = languages.iloc[:2]
transcription_langs_eval = languages.iloc[:10]
transcription_langs_eval_detailed = languages.iloc[:5]

# ===== run evaluation and aggregate results =====


async def evaluate():
    print("running evaluations")
    results = [
        task(model, original_language.bcp_47, i)
        for task in tasks
        for i in range(n_sentences)
        for original_language in langs_eval.itertuples()
        for model in models["id"]
        if original_language.in_benchmark
        and (
            model == model_fast
            or original_language.bcp_47 in langs_eval_detailed.bcp_47.values
        )
    ]
    return await tqdm_asyncio.gather(*results, miniters=1)


def aggregate(results):
    results = pd.DataFrame([r for rs in results for r in rs])
    results = (
        results.groupby(["model", "bcp_47", "task", "metric"]).mean().reset_index()
    )
    lang_results = (
        results.groupby(["bcp_47", "task", "metric"])
        .agg({"score": "mean", "model": "nunique"})
        .reset_index()
    )
    lang_results = pd.merge(languages, lang_results, on="bcp_47", how="outer")
    model_results = (
        results.groupby(["model", "task", "metric"])
        .agg({"score": "mean", "bcp_47": "nunique"})
        .reset_index()
    )
    task_results = (
        results.groupby(["task", "metric"])
        .agg({"score": "mean", "bcp_47": "nunique", "model": "nunique"})
        .reset_index()
    )
    return results, lang_results, model_results, task_results


def mean(lst):
    return sum(lst) / len(lst) if lst else None


def fmt_name(s):
    return (
        " ".join(w.capitalize() for w in s.split("-"))
        .replace("Gpt", "GPT")
        .replace("ai", "AI")
    )


def serialize(df):
    return df.replace({np.nan: None}).to_dict(orient="records")


def make_model_table(df):
    df["task_metric"] = df["task"] + "_" + df["metric"]
    df = df.drop(columns=["task", "metric"])
    task_metrics = df["task_metric"].unique()
    df = df.pivot(index="model", columns="task_metric", values="score").fillna(0)
    df["average"] = df[task_metrics].mean(axis=1)
    df = df.sort_values(by="average", ascending=False).reset_index()
    for row in [*task_metrics, "average"]:
        df[row] = df[row].round(2)
    df = pd.merge(df, models, left_on="model", right_on="id", how="left")
    df["creation_date"] = df["creation_date"].dt.strftime("%Y-%m-%d")
    df["provider"] = df["model"].str.split("/").str[0].apply(fmt_name)
    df["model"] = df["model"].str.split("/").str[1].apply(fmt_name)
    df["rank"] = df.index + 1
    df = df[["rank", "provider", "model", "hf_id", "creation_date", "size", "type", "license", "average", *task_metrics]]
    return df


def make_language_table(df):
    df["task_metric"] = df["task"] + "_" + df["metric"]
    df = df.drop(columns=["task", "metric"])
    task_metrics = df["task_metric"].unique()
    df = df.pivot(index="bcp_47", columns="task_metric", values="score").fillna(0).reset_index()
    df["average"] = df[task_metrics].mean(axis=1)
    for row in [*task_metrics, "average"]:
        df[row] = df[row].round(2)
    df = pd.merge(languages, df, on="bcp_47", how="outer")
    df = df.sort_values(by="speakers", ascending=False)
    df = df[["language_name", "speakers", "family", "average", "in_benchmark", *task_metrics]]
    return df

async def main():
    results = await evaluate()
    results, lang_results, model_results, task_results = aggregate(results)
    all_results = {
        "tasks": serialize(task_results),
        "models": serialize(model_results),
        "languages": serialize(lang_results),
        "scores": serialize(results),
    }
    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    datasets_df = pd.read_json("data/datasets.json")
    all_tables = {
        "model_table": serialize(make_model_table(model_results)),
        "language_table": serialize(make_language_table(lang_results)),
        "dataset_table": serialize(datasets_df),
    }
    with open("frontend/public/results.json", "w") as f:
        json.dump(all_tables, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())
