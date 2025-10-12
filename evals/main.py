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
from pathlib import Path

n_sentences = int(environ.get("N_SENTENCES", 1))  # 20))
n_languages = int(environ.get("N_LANGUAGES", 2))  # 150))
n_models = int(environ.get("N_MODELS", 5))  # -1))
stop_on_error = bool(environ.get("STOP_ON_ERROR", True))


async def evaluate():
    start_time = time.time()
    if Path("results/results.json").exists():
        old_models = pd.read_json("results/models.json")
        old_languages = pd.read_json("results/languages.json")
        old_results = pd.read_json("results/results.json")
    else:
        old_models = pd.DataFrame()
        old_languages = pd.DataFrame()
        old_results = pd.DataFrame()

    # get all combinations that need evaluation
    combis = [
        (task_name, model, lang.bcp_47)
        for model in models.iloc[:n_models]["id"]
        for lang in languages.head(n_languages).itertuples()
        for task_name, task in tasks.items()
        if task_name in models[models["id"] == model]["tasks"].iloc[0]
    ]
    combis = pd.DataFrame(combis, columns=["model", "bcp_47", "task"])
    if not old_results.empty:
        # Filter out already evaluated combinations
        completed = set(old_results[["model", "bcp_47", "task"]].apply(tuple, axis=1))
        # set + combis is faster than merge (locally it made a difference when loading all data/tasks into memory)
        mask = ~combis.apply(
            lambda row: (row["model"], row["bcp_47"], row["task"]) in completed, axis=1
        )
        combis = combis[mask]

    # add sentence numbers
    all_tasks = []
    for i in range(n_sentences):
        for task_name, model, bcp_47 in combis.itertuples(index=False):
            all_tasks.append((tasks[task_name], model, bcp_47, i))

    print(f"Running {len(all_tasks)} evaluation tasks...")

    # batching (asyncio.gather + rate-limiting can in principle run everything at once, but in practice batching is more efficient / necessary)
    batch_size = 1000
    results = []
    for i in range(0, len(all_tasks), batch_size):
        batch = all_tasks[i : i + batch_size]
        batch_results = await tqdm_asyncio.gather(
            *[
                task_func(model, bcp_47, sentence_nr)
                for task_func, model, bcp_47, sentence_nr in batch
            ],
            # return_exceptions=not stop_on_error,
        )
        results.extend(batch_results)
    results = [a for l in results for a in l]
    results = pd.DataFrame(results)
    args = dict(orient="records", indent=2, force_ascii=False)
    results.to_json("results/results_detailed.json", **args)

    # Aggregate results (over sentence number)
    results = (
        results.groupby(["model", "bcp_47", "task", "metric", "origin"])
        .agg({"score": "mean"})
        .reset_index()
    )

    # Merge with existing results
    updated_models = models
    updated_languages = languages
    if not old_results.empty:
        results = pd.concat([old_results, results])
        results = results.drop_duplicates(
            subset=["model", "bcp_47", "task", "metric", "origin"]
        )
        updated_models = (
            pd.concat([old_models, models]).drop_duplicates(subset=["id"], keep="last")
            # .sort_values("creation_date", ascending=True)
        )
        updated_languages = (
            pd.concat([old_languages, languages])
            .drop_duplicates(subset=["bcp_47"], keep="last")
            .sort_values("speakers", ascending=False)
        )
    results = results.sort_values(by=["model", "bcp_47", "task", "metric"])
    results.to_json("results/results.json", **args)
    updated_models.to_json("results/models.json", **args)
    updated_languages.to_json("results/languages.json", **args)
    elapsed = time.time() - start_time
    print(f"Evaluation completed in {str(timedelta(seconds=int(elapsed)))}")


if __name__ == "__main__":
    results = asyncio.run(evaluate())
