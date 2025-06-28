import asyncio
from time import time

import pandas as pd
from languages import languages
from models import models
from tasks import tasks
from tqdm.asyncio import tqdm_asyncio

# ===== config =====

n_sentences = 10

# ===== run evaluation and aggregate results =====


async def evaluate():
    # FIXME we should not need this for-loop, but it helps
    for n_languages in range(100, 101):
        print(f"running evaluations for {n_languages} languages")
        old_results = pd.read_json("results.json")
        old_models = pd.read_json("models.json")
        # get all combinations of model, language and task
        combis = [
            (model, lang.bcp_47, task_name)
            for model in models["id"]
            for lang in languages.iloc[:n_languages].itertuples()
            for task_name, task in tasks.items()
            if task_name in models[models["id"] == model]["tasks"].iloc[0]
        ]
        # filter out combinations that have already been evaluated
        combis = pd.DataFrame(combis, columns=["model", "bcp_47", "task"])
        combis = combis.merge(old_results, on=["model", "bcp_47", "task"], how="left")
        combis = combis[combis["metric"].isna()][["model", "bcp_47", "task"]]
        # run evaluations
        results = [
            tasks[task_name](model, bcp_47, i)
            for i in range(n_sentences)
            for model, bcp_47, task_name in combis.itertuples(index=False)
        ]
        results = await tqdm_asyncio.gather(*results, miniters=1)
        results = [r for group in results for r in group]
        args = dict(orient="records", indent=2, force_ascii=False)
        if results:
            # aggregate results
            results = pd.DataFrame(results)
            results = (
                results.groupby(["model", "bcp_47", "task", "metric"])
                .agg({"score": "mean"})
                .reset_index()
            )
            # save results
            results = pd.concat([old_results, results])
            results = results.sort_values(by=["model", "bcp_47", "task", "metric"])
            results.to_json("results.json", **args)
        # save up-to-date info on models and languages
        all_models = pd.concat([pd.DataFrame(models), old_models])
        all_models = all_models.drop_duplicates(subset=["id"]).sort_values(by=["id"])
        all_models.to_json("models.json", **args)
        pd.DataFrame(languages).to_json("languages.json", **args)


if __name__ == "__main__":
    results = asyncio.run(evaluate())
