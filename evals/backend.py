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
from tables import aggregate, make_country_table, make_language_table, make_model_table

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
    df = results
    if selected_languages:
        df = df[df["bcp_47"].isin(l["bcp_47"] for l in selected_languages)]

    _, lang_results, model_results, task_results = aggregate(df)
    print(lang_results)
    # lang_results = pd.merge(languages, lang_results, on="bcp_47", how="outer")
    model_table = make_model_table(model_results, models)
    language_table = make_language_table(lang_results, languages)
    datasets_df = pd.read_json("data/datasets.json")
    countries = make_country_table(language_table)
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
