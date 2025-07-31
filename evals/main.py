import asyncio
import pandas as pd
import time
import os
from datetime import datetime, timedelta
from tqdm.asyncio import tqdm_asyncio
from models import models
from tasks import tasks
from languages import languages

results = pd.DataFrame()


async def evaluate():
    # FIXME we should not need this for-loop, but it helps
    n_sentences = 2  # Testing with 2 sentences first
    start_time = time.time()
    print(f"🚀 Starting full evaluation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Evaluating {n_sentences} sentences per task")
    
    # Evaluate top languages by speakers (configurable via MAX_LANGUAGES env var)
    max_languages = int(os.environ.get("MAX_LANGUAGES", 5))  # Default 5 for quick testing
    top_languages = languages.head(max_languages)  # Top N by population
    print(f"🌍 Evaluating top {len(top_languages)} languages by speakers (max: {max_languages})")
    
    # For testing, just use all available languages up to max_languages
    for n_languages in [min(max_languages, len(top_languages))]:
        print(f"running evaluations for {n_languages} languages")
        old_results = pd.read_json("results.json")
        old_models = pd.read_json("models.json")
        # get all combinations of model, language and task
        combis = [
            (model, lang.bcp_47, task_name)
            for model in models["id"]
            for lang in top_languages.iloc[:n_languages].itertuples()
            for task_name, task in tasks.items()
            if task_name in models[models["id"] == model]["tasks"].iloc[0]
        ]
        # filter out combinations that have already been evaluated
        combis = pd.DataFrame(combis, columns=["model", "bcp_47", "task"])
        combis = combis.merge(old_results, on=["model", "bcp_47", "task"], how="left")
        combis = combis[combis["metric"].isna()][["model", "bcp_47", "task"]]
        # run evaluations in batches to prevent HTTP pool exhaustion
        all_tasks = []
        for i in range(n_sentences):
            for model, bcp_47, task_name in combis.itertuples(index=False):
                # All tasks now use the same signature
                all_tasks.append((tasks[task_name], model, bcp_47, i))
        
        print(f"⏳ Processing {len(all_tasks)} evaluation tasks in batches...")
        
        batch_size = 50  # Process 50 tasks at a time
        all_results = []
        
        for i in range(0, len(all_tasks), batch_size):
            batch = all_tasks[i:i+batch_size]
            print(f"📦 Processing batch {i//batch_size + 1}/{(len(all_tasks) + batch_size - 1)//batch_size} ({len(batch)} tasks)")
            
            batch_coroutines = []
            for task_data in batch:
                task_func, model, bcp_47, sentence_nr = task_data
                batch_coroutines.append(task_func(model, bcp_47, sentence_nr))
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
            
            # Aggregate results like main branch
            results_df = pd.DataFrame(results)
            if len(results_df) > 0:
                results_df = (
                    results_df.groupby(["model", "bcp_47", "task", "metric"])
                    .agg({"score": "mean"})
                    .reset_index()
                )
                # Merge with old results
                old_results = pd.read_json("results.json")
                results_df = pd.concat([old_results, results_df])
                results_df = results_df.sort_values(by=["model", "bcp_47", "task", "metric"])
                results_df.to_json("results.json", **args)
                print(f"💾 Saved {len(results_df)} aggregated results to results.json")
            else:
                print("⚠️  No valid results to aggregate")
        else:
            print("⚠️  No valid results to save - all API calls failed")
            
        # Save up-to-date info on models and languages (like main branch)
        all_models = pd.concat([pd.DataFrame(models), old_models])
        all_models = all_models.drop_duplicates(subset=["id"]).sort_values(by=["id"])
        all_models.to_json("models.json", **args)
        pd.DataFrame(languages).to_json("languages.json", **args)
            
        # Continue with next batch even if this one had errors
        
        # Time estimation
        elapsed = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        if n_languages < max_languages:
            remaining_batches = (max_languages - n_languages) // 10
            batch_count = max(1, n_languages // 10)  # Avoid division by zero
            estimated_remaining = elapsed * remaining_batches / batch_count
            eta = datetime.now() + timedelta(seconds=estimated_remaining)
            print(f"⏱️  Batch completed in {elapsed_str}. ETA for full run: {eta.strftime('%H:%M:%S')}")
        else:
            print(f"✅ Full evaluation completed in {elapsed_str}")
            print(f"🎉 Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    results = asyncio.run(evaluate())
