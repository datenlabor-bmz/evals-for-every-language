import asyncio

import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from languages import languages
from models import models
from tasks import tasks

# ===== config =====

n_sentences = 10
n_languages = 20
n_models = 25

# ===== run evaluation and aggregate results =====


async def evaluate():
    print("running evaluations")
    old_results = pd.read_json("results.json")
    results = [
        task(task, model, lang, i)
        for task_name, task in tasks.items()
        for i in range(n_sentences)
        for lang in languages.iloc[:n_languages].itertuples()
        for model in models["id"].iloc[:n_models]
        if len(
            old_results[
                (old_results["model"] == model)
                & (old_results["bcp_47"] == lang.bcp_47)
                & (old_results["task"] == task_name)
                & (old_results["sentence_nr"] == i)
            ]
        )
        == 0
    ]
    results = await tqdm_asyncio.gather(*results, miniters=1)
    results = [r for group in results for r in group]
    results = pd.DataFrame(results)
    results = pd.concat([old_results, results])
    args = dict(orient="records", indent=2, force_ascii=False)
    results.to_json("results.json", **args)
    pd.DataFrame(models).to_json("models.json", **args)
    pd.DataFrame(languages).to_json("languages.json", **args)


if __name__ == "__main__":
    results = asyncio.run(evaluate())
