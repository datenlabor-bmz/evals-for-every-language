import asyncio
import json

import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from languages import languages
from models import model_fast, models
from tasks import tasks

# ===== config =====

n_sentences = 30
langs_eval = languages.iloc[:30]
langs_eval_detailed = languages.iloc[:2]
transcription_langs_eval = languages.iloc[:10]
transcription_langs_eval_detailed = languages.iloc[:5]

# ===== run evaluation and aggregate results =====


async def evaluate():
    print("running evaluations")
    results = [
        task(model, original_language.bcp_47, i)
        for task in tasks
        for i in range(n_sentences)
        for original_language in langs_eval.itertuples()
        for model in models["id"]
        if original_language.in_benchmark
        and (
            model == model_fast
            or original_language.bcp_47 in langs_eval_detailed.bcp_47.values
        )
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
