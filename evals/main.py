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
    n_sentences = int(
        os.environ.get("N_SENTENCES", 20)
    )  # Default: 20 sentences per task
    max_languages = int(
        os.environ.get("MAX_LANGUAGES", 150)
    )  # Default: 150 top languages
    single_model = os.environ.get(
        "SINGLE_MODEL"
    )  # Optional: run only one specific model
    test_mode = os.environ.get("TEST", "").lower() in (
        "1",
        "true",
        "yes",
    )  # Optional: skip results loading/saving

    # Keep original DataFrames for saving metadata - distinction added for single model test runs.
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

    print(
        f"Starting evaluation: {len(models_df)} models, {len(top_languages)} languages, {n_sentences} sentences per task"
    )
    if test_mode:
        print("TEST MODE: Skipping results loading/saving")
    start_time = time.time()

    # Load existing results to avoid re-evaluation (skip in test mode)
    if test_mode:
        old_results = pd.DataFrame(
            columns=["model", "bcp_47", "task", "metric", "origin", "score"]
        )
    else:
        old_results = pd.read_json("results/results.json")

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
    if not old_results.empty:
        completed = set(old_results[["model", "bcp_47", "task"]].apply(tuple, axis=1))
        # set + combis is faster than merge (locally it made a difference for me when loading all data/tasks into memory)
        mask = ~combis.apply(
            lambda row: (row["model"], row["bcp_47"], row["task"]) in completed, axis=1
        )
        combis = combis[mask]

    # Create all evaluation tasks
    all_tasks = []
    for i in range(n_sentences):
        for model, bcp_47, task_name in combis.itertuples(index=False):
            all_tasks.append((tasks[task_name], model, bcp_47, i))

    print(f"Running {len(all_tasks)} evaluation tasks...")

    # For single model runs, we stop immediately on first API error to inspect.
    # For full evaluations, we continue despite errors to get maximum coverage.
    stop_on_error = single_model is not None

    # Process tasks in batches to avoid memory issues (for full evaluation locally that helped a lot)
    batch_size = 1000
    all_results = []

    try:
        for i in range(0, len(all_tasks), batch_size):
            batch = all_tasks[i : i + batch_size]
            batch_results = await asyncio.gather(
                *[
                    task_func(model, bcp_47, sentence_nr)
                    for task_func, model, bcp_47, sentence_nr in batch
                ],
                return_exceptions=not stop_on_error,
            )
            all_results.extend(batch_results)

        results = all_results

        # Process results and logging API errors separately to understand what are the main issues.
        valid_results = []
        errors = []

        for i, r in enumerate(results):
            if isinstance(r, Exception):
                if i < len(all_tasks):
                    task_info = all_tasks[i]
                    errors.append(f"{task_info[1]},{task_info[2]},{str(r)}")
            elif isinstance(r, list):
                valid_results.extend(r)
            elif r is not None:
                valid_results.append(r)

        # log errors and store
        if errors:
            with open("errors.log", "w") as f:
                f.write("model,task,error\n")
                for error in errors:
                    f.write(error + "\n")

        # Track model completion (TO BE DELETED - was for local run only)
        if valid_results:
            completed_models = set()
            for result in valid_results:
                if isinstance(result, dict) and "model" in result:
                    model = result["model"]
                    if model not in completed_models:
                        completed_models.add(model)
                        print(f"Completed: {model}")

        print(f"Completed: {len(valid_results)} valid results, {len(errors)} errors")

    # this is for local single model runs - for testing and development
    except Exception as e:
        print(f"EVALUATION STOPPED - API Error occurred:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        return pd.DataFrame()

    # Save results (skipped in test mode as we do not want to overwrite existing results)
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
                results_df = results_df.drop_duplicates(
                    subset=["model", "bcp_47", "task", "metric", "origin"]
                )

            results_df = results_df.sort_values(
                by=["model", "bcp_47", "task", "metric"]
            )
            results_df.to_json("results/results.json", **args)

            # Save model and language info (always save complete metadata, not filtered)
            original_models_df.to_json("results/models.json", **args)
            original_languages_df.to_json("results/languages.json", **args)
        else:
            print("TEST MODE: Skipping results saving")

        elapsed = time.time() - start_time
        print(f"Evaluation completed in {str(timedelta(seconds=int(elapsed)))}")

        return results_df

    return pd.DataFrame()


if __name__ == "__main__":
    results = asyncio.run(evaluate())
