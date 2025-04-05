import json

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from languages import languages
from models import models
from countries import make_country_table

def mean(lst):
    return sum(lst) / len(lst) if lst else None


def make_model_table(df, models):
    df = (
        df.groupby(["model", "task", "metric"])
        .agg({"score": "mean", "bcp_47": "nunique"})
        .reset_index()
    )
    df["task_metric"] = df["task"] + "_" + df["metric"]
    df = df.drop(columns=["task", "metric"])
    task_metrics = df["task_metric"].unique()
    df = df.pivot(index="model", columns="task_metric", values="score").fillna(0)
    df["average"] = df[task_metrics].mean(axis=1)
    df = df.sort_values(by="average", ascending=False).reset_index()
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
    df = (
        df.groupby(["bcp_47", "task", "metric"])
        .agg({"score": "mean", "model": "nunique"})
        .reset_index()
    )
    df["task_metric"] = df["task"] + "_" + df["metric"]
    df = df.drop(columns=["task", "metric"])
    task_metrics = df["task_metric"].unique()
    df = (
        df.pivot(index="bcp_47", columns="task_metric", values="score")
        .fillna(0)
        .reset_index()
    )
    df["average"] = df[task_metrics].mean(axis=1)
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


app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

with open("results.json", "r") as f:
    results = pd.DataFrame(json.load(f))


def serialize(df):
    return df.replace({np.nan: None}).to_dict(orient="records")


@app.post("/api/data")
async def data(request: Request):
    body = await request.body()
    data = json.loads(body)
    selected_languages = data.get("selectedLanguages", {})
    df = (
        results.groupby(["model", "bcp_47", "task", "metric"]).mean().reset_index()
    )
    # lang_results = pd.merge(languages, lang_results, on="bcp_47", how="outer")
    language_table = make_language_table(df, languages)
    datasets_df = pd.read_json("data/datasets.json")
    countries = make_country_table(language_table)
    if selected_languages:
        # the filtering is only applied for the model table
        df = df[df["bcp_47"].isin(lang["bcp_47"] for lang in selected_languages)]
    model_table = make_model_table(df, models)
    all_tables = {
        "model_table": serialize(model_table),
        "language_table": serialize(language_table),
        "dataset_table": serialize(datasets_df),
        "countries": serialize(countries),
    }
    return JSONResponse(content=all_tables)

app.mount("/", StaticFiles(directory="frontend/public", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
