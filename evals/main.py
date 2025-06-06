import asyncio
from time import time

t0 = time()

import pandas as pd
from languages import languages

print(f"loaded languages in {time() - t0:.2f}s")
t0 = time()

from models import models

print(f"loaded models in {time() - t0:.2f}s")
t0 = time()

from tasks import tasks

print(f"loaded tasks in {time() - t0:.2f}s")
t0 = time()

from tqdm.asyncio import tqdm_asyncio

# ===== config =====

n_sentences = 10
n_models = 35


# ===== run evaluation and aggregate results =====


async def evaluate():
    # FIXME we should not need this for loop, but it helps
    for n_languages in range(0, 200):
        print(f"running evaluations for {n_languages} languages")
        old_results = pd.read_json("results.json")
        old_models = pd.read_json("models.json")
        # get all combinations of model, language and task
        combis = [
            (model, lang.bcp_47, task_name)
            for task_name, task in tasks.items()
            for lang in languages.iloc[:n_languages].itertuples()
            for model in models["id"].iloc[:n_models]
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
        all_models = pd.concat([old_models, pd.DataFrame(models)])
        all_models = all_models.drop_duplicates(subset=["id"]).sort_values(by=["id"])
        all_models.to_json("models.json", **args)
        pd.DataFrame(languages).to_json("languages.json", **args)


if __name__ == "__main__":
    results = asyncio.run(evaluate())
