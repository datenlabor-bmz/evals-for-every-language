import asyncio
import pandas as pd
import time
from datetime import datetime, timedelta
from models import models
from tasks import tasks
from languages import languages
import os

async def evaluate():
    # Configuration - easily adjustable defaults
    n_sentences = int(os.environ.get("N_SENTENCES", 20))     # Default: 20 sentences per task
    max_languages = int(os.environ.get("MAX_LANGUAGES", 150))  # Default: 150 top languages
    single_model = os.environ.get("SINGLE_MODEL")            # Optional: run only one specific model
    test_mode = os.environ.get("TEST", "").lower() in ("1", "true", "yes")  # Optional: skip results loading/saving
    
    # Keep original DataFrames for saving metadata
    original_models_df = pd.DataFrame(models)
    original_languages_df = pd.DataFrame(languages)
    
    # Create working copies for single evaluation runs
    models_df = original_models_df.copy()
    languages_df = original_languages_df.copy()
    top_languages = languages.head(max_languages)
    
    # Filter to single model if specified (only affects evaluation, not saving)
    if single_model:
        models_df = models_df[models_df["id"] == single_model]
        if len(models_df) == 0:
            print(f"Error: Model '{single_model}' not found. Available models:")
            for model_id in original_models_df["id"]:
                print(f"  {model_id}")
            return pd.DataFrame()

    print(f"Starting evaluation: {len(models_df)} models, {len(top_languages)} languages, {n_sentences} sentences per task")
    if test_mode:
        print("TEST MODE: Skipping results loading/saving")
    start_time = time.time()
    
    # Load existing results to avoid re-evaluation (skip in test mode)
    if test_mode:
        old_results = pd.DataFrame(columns=["model", "bcp_47", "task", "metric", "origin", "score"])
    else:
        try:
            old_results = pd.read_json("results.json")
            if old_results.empty:
                old_results = pd.DataFrame(columns=["model", "bcp_47", "task", "metric", "origin", "score"])
        except FileNotFoundError:
            old_results = pd.DataFrame(columns=["model", "bcp_47", "task", "metric", "origin", "score"])
    
    # Get all combinations that need evaluation
    combis = [
        (model, lang.bcp_47, task_name)
        for model in models_df["id"]
        for lang in top_languages.itertuples()
        for task_name, task in tasks.items()
        if task_name in models_df[models_df["id"] == model]["tasks"].iloc[0]
    ]
    
    # Filter out already evaluated combinations
    combis = pd.DataFrame(combis, columns=["model", "bcp_47", "task"])
    combis = combis.merge(old_results, on=["model", "bcp_47", "task"], how="left")
    combis = combis[combis["metric"].isna()][["model", "bcp_47", "task"]]
    
    # Create all evaluation tasks
    all_tasks = []
    for i in range(n_sentences):
        for model, bcp_47, task_name in combis.itertuples(index=False):
            all_tasks.append((tasks[task_name], model, bcp_47, i))
    
    print(f"Running {len(all_tasks)} evaluation tasks...")
    
    # Run all tasks with simple asyncio.gather, but stop on first error
    try:
        results = await asyncio.gather(
            *[task_func(model, bcp_47, sentence_nr) for task_func, model, bcp_47, sentence_nr in all_tasks],
            return_exceptions=False  # This will raise on first exception
        )
        
        # Process results - no exceptions should reach here
        valid_results = []
        for r in results:
            if isinstance(r, list):
                valid_results.extend(r)
            else:
                valid_results.append(r)

        print(f"Completed: {len(valid_results)} valid results")
        
    except Exception as e:
        print(f"EVALUATION STOPPED - API Error occurred:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        return pd.DataFrame()
    
        # Save results (skip in test mode)
        if valid_results:
            results_df = pd.DataFrame(valid_results)
            
            # Aggregate results
            results_df = (
                results_df.groupby(["model", "bcp_47", "task", "metric", "origin"])
                .agg({"score": "mean"})
                .reset_index()
            )
            
            if not test_mode:
                args = dict(orient="records", indent=2, force_ascii=False)
                
                # Merge with existing results
                if not old_results.empty:
                    results_df = pd.concat([old_results, results_df])
                    results_df = results_df.drop_duplicates(subset=["model", "bcp_47", "task", "metric", "origin"])
                
                results_df = results_df.sort_values(by=["model", "bcp_47", "task", "metric"])
                results_df.to_json("results.json", **args)
                
                # Save model and language info (always save complete metadata, not filtered)
                original_models_df.to_json("models.json", **args)
                original_languages_df.to_json("languages.json", **args)
            else:
                print("TEST MODE: Skipping results saving")
            
            elapsed = time.time() - start_time
            print(f"Evaluation completed in {str(timedelta(seconds=int(elapsed)))}")
            
            return results_df
    
    return pd.DataFrame()


if __name__ == "__main__":
    results = asyncio.run(evaluate())
