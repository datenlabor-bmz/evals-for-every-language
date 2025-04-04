import pandas as pd
from countries import make_country_table

make_country_table = make_country_table


def aggregate(results):
    results = (
        results.groupby(["model", "bcp_47", "task", "metric"]).mean().reset_index()
    )
    lang_results = (
        results.groupby(["bcp_47", "task", "metric"])
        .agg({"score": "mean", "model": "nunique"})
        .reset_index()
    )
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


def make_model_table(df, models):
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
    df["rank"] = df.index + 1
    df = df[
        [
            "rank",
            "model",
            "hf_id",
            "creation_date",
            "size",
            "type",
            "license",
            "average",
            *task_metrics,
        ]
    ]
    return df


def make_language_table(df, languages):
    df["task_metric"] = df["task"] + "_" + df["metric"]
    df = df.drop(columns=["task", "metric"])
    task_metrics = df["task_metric"].unique()
    df = (
        df.pivot(index="bcp_47", columns="task_metric", values="score")
        .fillna(0)
        .reset_index()
    )
    df["average"] = df[task_metrics].mean(axis=1)
    for row in [*task_metrics, "average"]:
        df[row] = df[row].round(2)
    df = pd.merge(languages, df, on="bcp_47", how="outer")
    df = df.sort_values(by="speakers", ascending=False)
    df = df[
        [
            "bcp_47",
            "language_name",
            "autonym",
            "speakers",
            "family",
            "average",
            "in_benchmark",
            *task_metrics,
        ]
    ]
    return df