import asyncio
import pandas as pd
import time
from datetime import datetime, timedelta
from tqdm.asyncio import tqdm_asyncio
from models import models
from tasks import tasks
from languages import languages

results = pd.DataFrame()


async def evaluate():
    # FIXME we should not need this for-loop, but it helps
    n_sentences = 10  # Full evaluation
    start_time = time.time()
    print(f"🚀 Starting full evaluation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Evaluating {n_sentences} sentences per task")
    
    # Evaluate all languages with benchmark data (194 total)
    benchmark_languages = languages[languages["in_benchmark"]]
    for n_languages in range(10, min(len(benchmark_languages) + 1, 151), 10):
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
        # run evaluations in batches to prevent HTTP pool exhaustion
        all_tasks = [
            (tasks[task_name], model, bcp_47, i)
            for i in range(n_sentences)
            for model, bcp_47, task_name in combis.itertuples(index=False)
        ]
        
        print(f"⏳ Processing {len(all_tasks)} evaluation tasks in batches...")
        
        batch_size = 50  # Process 50 tasks at a time
        all_results = []
        
        for i in range(0, len(all_tasks), batch_size):
            batch = all_tasks[i:i+batch_size]
            print(f"📦 Processing batch {i//batch_size + 1}/{(len(all_tasks) + batch_size - 1)//batch_size} ({len(batch)} tasks)")
            
            batch_coroutines = [task_func(model, bcp_47, sentence_nr) for task_func, model, bcp_47, sentence_nr in batch]
            batch_results = await asyncio.gather(*batch_coroutines, return_exceptions=True)
            all_results.extend(batch_results)
            
            # Small delay between batches to avoid overwhelming the API
            await asyncio.sleep(1)
        
        results = all_results
        # Filter out exceptions and flatten results
        valid_results = []
        exception_count = 0
        for r in results:
            if isinstance(r, Exception):
                exception_count += 1
                continue
            if isinstance(r, list):
                valid_results.extend(r)
            else:
                valid_results.append(r)
        
        print(f"⚠️  Encountered {exception_count} API errors (model unavailable/rate limits)")
        print(f"✅ Successfully processed {len(valid_results)} evaluations")
        
        # Save partial results even if some failed
        if valid_results:
            results = valid_results
            args = dict(orient="records", indent=2, force_ascii=False)
            pd.DataFrame(results).to_json("results.json", **args)
            print(f"💾 Saved {len(valid_results)} results to results.json")
        else:
            print("⚠️  No valid results to save - all API calls failed")
            
        # Continue with next batch even if this one had errors
        
        # Time estimation
        elapsed = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        if n_languages < 100:
            remaining_batches = (100 - n_languages) // 10
            estimated_remaining = elapsed * remaining_batches / (n_languages // 10)
            eta = datetime.now() + timedelta(seconds=estimated_remaining)
            print(f"⏱️  Batch completed in {elapsed_str}. ETA for full run: {eta.strftime('%H:%M:%S')}")
        else:
            print(f"✅ Full evaluation completed in {elapsed_str}")
            print(f"🎉 Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    results = asyncio.run(evaluate())
