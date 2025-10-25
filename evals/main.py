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
from tqdm import tqdm

n_sentences = int(environ.get("N_SENTENCES", 10))  # 20))
n_languages = int(environ.get("N_LANGUAGES", 2))  # 150))
n_models = int(environ.get("N_MODELS", 100))  # -1))
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

    # Load cached results and filter out completed combinations
    old_results = load("results-detailed")
    if not old_results.empty:
        completed = set(old_results[["task", "model", "bcp_47", "sentence_nr"]].apply(tuple, axis=1))
        combis = combis[~combis.apply(lambda row: tuple(row) in completed, axis=1)]

    print(f"Running {len(combis)} evaluation tasks...")

    # batching (asyncio.gather + rate-limiting can in principle run everything at once, but in practice batching is more efficient / necessary)
    batch_size = 1000
    batch_results = [
        await tqdm_asyncio.gather(
            *[tasks[task_name](model, bcp_47, sentence_nr)
              for _, (task_name, model, bcp_47, sentence_nr) in batch.iterrows()]
        )
        for i in tqdm(range(0, len(combis), batch_size), colour='blue', desc='Batches')
        for batch in [combis[i:i + batch_size]]
    ]
    results = pd.DataFrame([r for batch in batch_results for result in batch for r in result])
    
    # Merge with cached results (immutable log)
    all_results = pd.concat([old_results, results]).drop_duplicates(
        subset=["task", "model", "bcp_47", "metric", "sentence_nr"]
    ) if not old_results.empty else results
    
    # Filter to current models × languages and aggregate
    current_models = set(models["id"])
    current_languages = set(languages["bcp_47"])
    results_agg = (
        all_results[all_results["model"].isin(current_models) & all_results["bcp_47"].isin(current_languages)]
        .groupby(["model", "bcp_47", "task", "metric"])
        .agg({"score": "mean", "origin": "first"})
        .reset_index()
    )
    
    save(all_results, "results-detailed")
    save(results_agg, "results")
    save(models, "models")
    save(languages, "languages")
    elapsed = time.time() - start_time
    print(f"Evaluation completed in {str(timedelta(seconds=int(elapsed)))}")


if __name__ == "__main__":
    results = asyncio.run(evaluate())
