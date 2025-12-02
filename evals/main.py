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
from datasets_.util import load, save, get_valid_task_languages
from tqdm import tqdm

n_sentences = int(environ.get("N_SENTENCES", 10))
n_languages = int(environ.get("N_LANGUAGES", 1000))
n_models = int(environ.get("N_MODELS", 40))

async def evaluate():
    start_time = time.time()

    # Pre-compute model tasks to avoid O(n²) lookups
    model_tasks = models.set_index("id")["tasks"].to_dict()
    
    # Pre-compute valid languages for each task
    valid_task_langs = {task_name: get_valid_task_languages(task_name) for task_name in tasks}
    
    # get all combinations that need evaluation (filtering invalid lang×task combos)
    combis = [
        (task_name, model, lang.bcp_47, i)
        for i in range(n_sentences)
        for lang in languages.head(n_languages).itertuples()
        for task_name, task in tasks.items()
        for model in models.iloc[:n_models]["id"]
        if task_name in model_tasks[model] and lang.bcp_47 in valid_task_langs[task_name]
    ]
    combis = pd.DataFrame(combis, columns=["task", "model", "bcp_47", "sentence_nr"])

    # Load cached results and filter out completed combinations
    old_results = load("results-detailed")
    if not old_results.empty:
        # Only treat status==\"ok\" (or missing status) as completed.
        if "status" in old_results.columns:
            ok_mask = old_results["status"].isna() | (old_results["status"] == "ok")
            completed_df = old_results.loc[ok_mask, ["task", "model", "bcp_47", "sentence_nr"]]
        else:
            completed_df = old_results[["task", "model", "bcp_47", "sentence_nr"]]
        completed = set(completed_df.apply(tuple, axis=1))
        combis = combis[~combis.apply(lambda row: tuple(row) in completed, axis=1)]

    print(f"Running {len(combis)} evaluation tasks...")

    # batching (asyncio.gather + rate-limiting can in principle run everything at once, but in practice batching is more efficient / necessary)
    batch_size = 2000
    batch_results = [
        await tqdm_asyncio.gather(
            *[tasks[task_name](model, bcp_47, sentence_nr)
              for _, (task_name, model, bcp_47, sentence_nr) in batch.iterrows()]
        )
        for i in tqdm(range(0, len(combis), batch_size), colour='blue', desc='Batches')
        for batch in [combis[i:i + batch_size]]
    ]
    results = [r for batch in batch_results for result in batch for r in result]
    results = pd.DataFrame(results) if results else pd.DataFrame(columns=["task", "model", "bcp_47", "metric", "sentence_nr", "score", "origin"])
    
    # Merge with cached results (immutable log, prefer latest results on conflict)
    all_results = pd.concat([old_results, results]).drop_duplicates(
        subset=["task", "model", "bcp_47", "metric", "sentence_nr"],
        keep="last",
    ) if not old_results.empty else results
    
    # Filter to current models × languages and aggregate.
    # Only aggregate over successful evaluations (status == \"ok\" or missing).
    current_models = set(models.iloc[:n_models]["id"])
    current_languages = set(languages.head(n_languages)["bcp_47"])
    if "status" in all_results.columns:
        valid_mask = all_results["status"].isna() | (all_results["status"] == "ok")
        valid_results = all_results[valid_mask]
    else:
        valid_results = all_results

    results_agg = (
        valid_results[valid_results["model"].isin(current_models) & valid_results["bcp_47"].isin(current_languages)]
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
