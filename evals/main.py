import asyncio
import json

import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from languages import languages
from models import models
from tasks import tasks

# ===== config =====

n_sentences = 10
n_languages = 40
n_models = 25

# ===== run evaluation and aggregate results =====


async def evaluate():
    print("running evaluations")
    results = [
        task(model, lang.bcp_47, i)
        for task in tasks
        for i in range(n_sentences)
        for lang in languages.iloc[:n_languages].itertuples()
        for model in models["id"].iloc[:n_models]
        if lang.in_benchmark # TODO
    ]
    return await tqdm_asyncio.gather(*results, miniters=1)

def serialize(df):
    return df.replace({np.nan: None, pd.NA: None}).to_dict(orient="records")

async def main():
    models["creation_date"] = models["creation_date"].apply(lambda x: x.isoformat())
    results = await evaluate()
    results = [r for group in results for r in group]
    results = {
        "languages": serialize(languages),
        "models": serialize(models),
        "scores": results,
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())
