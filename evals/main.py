import asyncio
import time
from datetime import timedelta
from os import environ

import pandas as pd
from languages import languages
from models import models
from rich import print
from tasks import tasks
from tqdm.asyncio import tqdm_asyncio
from datasets_.util import load, save

n_sentences = int(environ.get("N_SENTENCES", 1))  # 20))
n_languages = int(environ.get("N_LANGUAGES", 2))  # 150))
n_models = int(environ.get("N_MODELS", 50))  # -1))
stop_on_error = bool(environ.get("STOP_ON_ERROR", True))

async def evaluate():
    start_time = time.time()

    # get all combinations that need evaluation
    combis = [
        (task_name, model, lang.bcp_47, i)
        for i in range(n_sentences)
        for model in models.iloc[:n_models]["id"]
        for lang in languages.head(n_languages).itertuples()
        for task_name, task in tasks.items()
        if task_name in models[models["id"] == model]["tasks"].iloc[0]
    ]
    combis = pd.DataFrame(combis, columns=["task", "model", "bcp_47", "sentence_nr"])

    old_results = load("results-detailed")
    old_models = load("models")
    old_languages = load("languages")
    if not old_results.empty:
        # Filter out already evaluated combinations
        completed = set(old_results[["task", "model", "bcp_47", "sentence_nr"]].apply(tuple, axis=1))
        # set + combis is faster than merge (locally it made a difference when loading all data/tasks into memory)
        mask = ~combis.apply(
            lambda row: (row["task"], row["model"], row["bcp_47"], row["sentence_nr"]) in completed, axis=1
        )
        combis = combis[mask]

    print(f"Running {len(combis)} evaluation tasks...")

    # batching (asyncio.gather + rate-limiting can in principle run everything at once, but in practice batching is more efficient / necessary)
    batch_size = 1000
    results = []
    for i in range(0, len(combis), batch_size):
        batch = combis[i : i + batch_size]
        batch_results = await tqdm_asyncio.gather(
            *[
                tasks[task_name](model, bcp_47, sentence_nr)
                for _, (task_name, model, bcp_47, sentence_nr) in batch.iterrows()
            ],
            # return_exceptions=not stop_on_error,
        )
        results.extend(batch_results)
    results = [a for l in results for a in l]
    results = pd.DataFrame(results)
    
    # Merge with existing results
    updated_models = models
    updated_languages = languages
    if not old_results.empty:
        results = pd.concat([old_results, results])
        results = results.drop_duplicates(
            subset=["task", "model", "bcp_47", "metric", "sentence_nr"]
        ).sort_values(by=["model", "task", "bcp_47", "metric"])
        updated_models = (
            pd.concat([old_models, models]).drop_duplicates(subset=["id"], keep="last")
            .sort_values("id", ascending=True)
        )
        updated_languages = (
            pd.concat([old_languages, languages])
            .drop_duplicates(subset=["bcp_47"], keep="last")
            .sort_values("speakers", ascending=False)
        )
    # Aggregate results (over sentence number)
    results_agg = (
        results.groupby(["model", "bcp_47", "task", "metric"])
        .agg({"score": "mean", "origin": "first"})
        .reset_index()
    )
    save(results, "results-detailed")
    save(results_agg, "results")
    save(updated_models, "models")
    save(updated_languages, "languages")
    elapsed = time.time() - start_time
    print(f"Evaluation completed in {str(timedelta(seconds=int(elapsed)))}")


if __name__ == "__main__":
    results = asyncio.run(evaluate())
