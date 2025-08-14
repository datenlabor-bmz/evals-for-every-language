#!/usr/bin/env python3
"""
Focused evaluation script for premium models only
"""

import asyncio
import pandas as pd
import time
import os
from datetime import datetime, timedelta
from models import models
from tasks import tasks
from languages import languages
import json

# Focus only on premium models
FOCUSED_MODELS = [
    "anthropic/claude-sonnet-4",      # Claude Sonnet 4 ($15)
    "openai/gpt-5",                   # GPT-5 ($10) - NEW!
    # "anthropic/claude-opus-4.1",    # Not yet available
    # "x-ai/grok-4"                   # Not yet available
]
results = pd.DataFrame()

def save_checkpoint(results_df, models_df, languages_df, batch_num, total_batches):
    """Save current progress as checkpoint with atomic writes"""
    try:
        args = dict(orient="records", indent=2, force_ascii=False)
        
        # Save current results with atomic write
        if len(results_df) > 0:
            # Write to temporary file first, then rename (atomic)
            temp_file = "results_focused.json"
            results_df.to_json(temp_file, **args)
            import os
            os.replace(temp_file, "results_focused.json")
            print(f"💾 Checkpoint saved: {len(results_df):,} results (batch {batch_num}/{total_batches})")
        
        # Save model and language info with atomic writes
        temp_models = "models_focused.json"
        temp_languages = "languages_focused.json"
        models_df.to_json(temp_models, **args)
        languages_df.to_json(temp_languages, **args)
        os.replace(temp_models, "models_focused.json")
        os.replace(temp_languages, "languages_focused.json")
        
        # Save checkpoint metadata
        checkpoint_info = {
            "last_batch": batch_num,
            "total_batches": total_batches,
            "timestamp": datetime.now().isoformat(),
            "results_count": len(results_df),
            "models_count": len(models_df),
            "languages_count": len(languages_df),
            "focused_models": FOCUSED_MODELS
        }
        temp_checkpoint = "checkpoint_focused.json"
        with open(temp_checkpoint, "w") as f:
            json.dump(checkpoint_info, f, indent=2)
        os.replace(temp_checkpoint, "checkpoint_focused.json")
        
        print(f"✅ All files saved atomically - safe to push to remote")
            
    except Exception as e:
        print(f"⚠️  Failed to save checkpoint: {e}")
        import traceback
        traceback.print_exc()

def load_checkpoint():
    """Load previous checkpoint if available"""
    try:
        if os.path.exists("checkpoint_focused.json"):
            with open("checkpoint_focused.json", "r") as f:
                checkpoint = json.load(f)
            print(f"📂 Found checkpoint from batch {checkpoint['last_batch']}/{checkpoint['total_batches']}")
            return checkpoint
    except Exception as e:
        print(f"⚠️  Failed to load checkpoint: {e}")
    return None

async def evaluate():
    # Configuration
    n_sentences = int(os.environ.get("N_SENTENCES", 20))
    max_languages = int(os.environ.get("MAX_LANGUAGES", 150))
    
    # Load models and filter to focused ones
    models_df = pd.DataFrame(models)
    focused_models_df = models_df[models_df["id"].isin(FOCUSED_MODELS)].copy()
    
    if len(focused_models_df) == 0:
        print("❌ No focused models found in models list")
        return
    
    print(f"🎯 Focused Evaluation: {len(focused_models_df)} Premium Models")
    print(f"🚀 Starting focused evaluation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Evaluating {n_sentences} sentences per task")
    print(f"🌍 Evaluating top {max_languages} languages by speakers")
    
    # Load checkpoint if available
    checkpoint = load_checkpoint()
    start_batch = 0
    if checkpoint:
        start_batch = checkpoint['last_batch']
        print(f"🔄 Resuming from batch {start_batch}")
    
    # Load existing results
    try:
        old_results = pd.read_json("results_focused.json")
        if old_results.empty:
            old_results = pd.DataFrame(columns=["model", "bcp_47", "task", "metric", "origin", "score"])
    except FileNotFoundError:
        old_results = pd.DataFrame(columns=["model", "bcp_47", "task", "metric", "origin", "score"])
    
    # Get top languages
    top_languages = languages.head(max_languages)
    
    # Generate combinations for focused models only
    combis = [
        (model, lang.bcp_47, task_name)
        for model in focused_models_df["id"]
        for lang in top_languages.itertuples()
        for task_name, task in tasks.items()
        if task_name in focused_models_df[focused_models_df["id"] == model]["tasks"].iloc[0]
    ]
    
    # Filter out combinations that have already been evaluated
    combis = pd.DataFrame(combis, columns=["model", "bcp_47", "task"])
    combis = combis.merge(old_results, on=["model", "bcp_47", "task"], how="left")
    combis = combis[combis["metric"].isna()][["model", "bcp_47", "task"]]
    
    # Create tasks
    all_tasks = []
    for i in range(n_sentences):
        for model, bcp_47, task_name in combis.itertuples(index=False):
            all_tasks.append((tasks[task_name], model, bcp_47, i))
    
    print(f"⏳ Processing {len(all_tasks):,} evaluation tasks in batches...")
    
    # Batch processing
    batch_size = 200  # Optimized batch size
    total_batches = (len(all_tasks) + batch_size - 1) // batch_size
    
    # Show evaluation scope
    print(f"🎯 Evaluation Scope:")
    print(f"  Models: {len(focused_models_df)} premium models")
    print(f"  Languages: {len(top_languages)} (top {max_languages} by speakers)")
    print(f"  Tasks: {len(tasks)} (translation, classification, MMLU, ARC, TruthfulQA, MGSM)")
    print(f"  Sentences per task: {n_sentences}")
    print(f"  Total combinations: {len(all_tasks):,}")
    print(f"  Batches: {total_batches:,} (batch size: {batch_size})")
    print(f"  Estimated time: {total_batches * 10.5 / 3600:.1f} hours")
    print(f"  Starting from batch: {start_batch}")
    
    all_results = []
    start_time = time.time()
    
    for i in range(start_batch * batch_size, len(all_tasks), batch_size):
        batch = all_tasks[i:i+batch_size]
        current_batch = i // batch_size + 1
        
        print(f"📦 Processing batch {current_batch}/{total_batches} ({len(batch)} tasks)")
        
        # Show what's being evaluated in this batch
        batch_summary = {}
        for task_data in batch:
            task_func, model, bcp_47, sentence_nr = task_data
            if hasattr(task_func, 'func'):
                task_name = task_func.func.__name__.replace('_and_evaluate', '')
            else:
                task_name = task_func.__name__.replace('_and_evaluate', '')
            
            if task_name not in batch_summary:
                batch_summary[task_name] = set()
            batch_summary[task_name].add(bcp_47)
        
        for task_name, languages_set in batch_summary.items():
            lang_list = ', '.join(sorted(languages_set))
            print(f"  🔄 {task_name}: {lang_list}")
        
        # Execute batch
        batch_coroutines = []
        for task_data in batch:
            task_func, model, bcp_47, sentence_nr = task_data
            batch_coroutines.append(task_func(model, bcp_47, sentence_nr))
        
        try:
            batch_results = await asyncio.gather(*batch_coroutines, return_exceptions=True)
            all_results.extend(batch_results)
            
            # Process batch results
            valid_results = []
            exception_count = 0
            for r in batch_results:
                if isinstance(r, Exception):
                    exception_count += 1
                    continue
                if isinstance(r, list):
                    valid_results.extend(r)
                else:
                    valid_results.append(r)
            
            if valid_results:
                # Aggregate results
                batch_df = pd.DataFrame(valid_results)
                if len(batch_df) > 0:
                    batch_df = (
                        batch_df.groupby(["model", "bcp_47", "task", "metric", "origin"])
                        .agg({"score": "mean"})
                        .reset_index()
                    )
                    # Merge with existing results
                    all_results_df = pd.concat([old_results, batch_df])
                    all_results_df = all_results_df.drop_duplicates(subset=["model", "bcp_47", "task", "metric", "origin"])
                    all_results_df = all_results_df.sort_values(by=["model", "bcp_47", "task", "metric"])
                    
                    # Save checkpoint
                    save_checkpoint(all_results_df, focused_models_df, top_languages, current_batch, total_batches)
                    
                    # Update old_results for next batch
                    old_results = all_results_df
            
            print(f"✅ Batch {current_batch} completed: {len(valid_results)} valid results, {exception_count} errors")
            
            # Progress monitoring
            progress = (current_batch / total_batches) * 100
            elapsed = time.time() - start_time
            if current_batch > 0:
                avg_time_per_batch = elapsed / current_batch
                remaining_batches = total_batches - current_batch
                eta_seconds = remaining_batches * avg_time_per_batch
                eta = datetime.now() + timedelta(seconds=int(eta_seconds))
                print(f"📊 Progress: {progress:.1f}% | ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            print(f"❌ Batch {current_batch} failed: {e}")
            # Save checkpoint even on failure
            if len(all_results) > 0:
                results_df = pd.DataFrame(all_results)
                save_checkpoint(results_df, focused_models_df, top_languages, current_batch, total_batches)
            continue
        
        # Reduced delay between batches
        await asyncio.sleep(0.5)
    
    # Final aggregation and save
    results = all_results
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
    
    print(f"⚠️  Encountered {exception_count} API errors")
    print(f"✅ Successfully processed {len(valid_results)} evaluations")
    
    # Save final results
    if valid_results:
        results_df = pd.DataFrame(valid_results)
        if len(results_df) > 0:
            results_df = (
                results_df.groupby(["model", "bcp_47", "task", "metric", "origin"])
                .agg({"score": "mean"})
                .reset_index()
            )
            # Merge with old results
            all_results_df = pd.concat([old_results, results_df])
            all_results_df = all_results_df.drop_duplicates(subset=["model", "bcp_47", "task", "metric", "origin"])
            all_results_df = all_results_df.sort_values(by=["model", "bcp_47", "task", "metric"])
            all_results_df.to_json("results_focused.json", orient="records", indent=2, force_ascii=False)
            print(f"💾 Final results saved: {len(all_results_df)} evaluations")
    
    # Clean up checkpoint file on successful completion
    if os.path.exists("checkpoint_focused.json"):
        os.remove("checkpoint_focused.json")
        print("🧹 Cleaned up checkpoint file")
    
    elapsed = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    print(f"✅ Focused evaluation completed in {elapsed_str}")
    print(f"🎉 Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results

if __name__ == "__main__":
    results = asyncio.run(evaluate()) 