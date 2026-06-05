import asyncio
import time
from datetime import timedelta
from os import environ

import pandas as pd
from languages import languages
from models import (
    models,
    AUTO_BLOCKLIST_MIN_ATTEMPTS,
    AUTO_BLOCKLIST_FAIL_PCT_THRESHOLD,
)
from rich import print
from tasks import tasks
from tqdm.asyncio import tqdm_asyncio
from datasets_.util import load, save, save_local_only, get_valid_task_languages
from tqdm import tqdm

# Canonical scale used in the nightly workflow. Reduced scale (smaller
# N_LANGUAGES or N_MODELS) is OK for local validation, but pushing the
# aggregated `results` dataset back to HF in that mode would truncate the
# published table — see CANONICAL_*_FOR_PUSH and the guard in save() below.
CANONICAL_N_LANGUAGES_FOR_PUSH = 1000
CANONICAL_N_MODELS_FOR_PUSH = 100  # nightly uses 150; bar is "covers the full cohort"

n_sentences = int(environ.get("N_SENTENCES", 10))
n_languages = int(environ.get("N_LANGUAGES", 1000))
n_models = int(environ.get("N_MODELS", 40))

# When n_languages or n_models is smaller than canonical, the filter in
# `results_agg` below would discard most rows and overwrite the public HF
# aggregate. Detect that and downgrade to local-only writes.
ALLOW_HF_PUSH_RESULTS = (
    n_languages >= CANONICAL_N_LANGUAGES_FOR_PUSH
    and n_models >= CANONICAL_N_MODELS_FOR_PUSH
)

def publishable_models(log_df, covered_models):
    """Models safe to include in the published aggregate.

    `covered_models` must already be COVERAGE-COMPLETE (every expected
    task × language × sentence attempted). The caller guarantees this — a
    model is only added once its full matrix is done. Coverage is the load-
    bearing guard: a sparsely-evaluated model (e.g. 5 of 200 languages) would
    otherwise show an inflated mean and jump the leaderboard, which is exactly
    the failure that once put a small model at rank #1.

    On top of coverage we drop models that are *broken* (mostly errors over a
    meaningful sample) rather than merely low-scoring — auto_blocklist removes
    them from the cohort on the next run."""
    if "status" not in log_df.columns:
        return set(covered_models)
    keep = set()
    in_scope = log_df[log_df["model"].isin(covered_models)]
    for mid, grp in in_scope.groupby("model"):
        total = len(grp)
        failed = (grp["status"] == "error").sum()
        if (total >= AUTO_BLOCKLIST_MIN_ATTEMPTS
                and failed / total * 100 >= AUTO_BLOCKLIST_FAIL_PCT_THRESHOLD):
            continue
        keep.add(mid)
    return keep


def checkpoint(log_df, covered_models, cohort_languages, note):
    """Push the immutable log + the aggregate (healthy, fully-covered models
    only) to HuggingFace (or locally if below scale)."""
    if "status" in log_df.columns:
        valid = log_df[log_df["status"].isna() | (log_df["status"] == "ok")]
    else:
        valid = log_df
    keep = publishable_models(log_df, covered_models)
    agg = (
        valid[valid["model"].isin(keep) & valid["bcp_47"].isin(cohort_languages)]
        .groupby(["model", "bcp_47", "task", "metric"])
        .agg({"score": "mean", "origin": "first"})
        .reset_index()
    )
    # results-detailed is append-merged (immutable log); safe at any scale.
    save(log_df, "results-detailed")
    # The aggregated tables are filtered by cohort_models × cohort_languages, so
    # a partial-scale run would truncate the published view — push only from a
    # full-scale run; otherwise local-only.
    if ALLOW_HF_PUSH_RESULTS:
        save(agg, "results")
        save(models, "models")
        save(languages, "languages")
    else:
        save_local_only(agg, "results")
        save_local_only(models, "models")
        save_local_only(languages, "languages")
    print(f"  ✓ checkpoint after {note}: {len(keep)} models published, "
          f"{len(log_df)} detailed rows "
          f"({'HF' if ALLOW_HF_PUSH_RESULTS else 'local-only'})")
    return agg


async def evaluate():
    start_time = time.time()

    # Pre-compute model tasks to avoid O(n²) lookups
    model_tasks = models.set_index("id")["tasks"].to_dict()
    
    # Pre-compute valid languages for each task
    valid_task_langs = {task_name: get_valid_task_languages(task_name) for task_name in tasks}
    
    # get all combinations that need evaluation (filtering invalid lang×task combos)
    combis = [
        (task_name, model, lang.bcp_47, i)
        for i in range(n_sentences)
        for lang in languages.head(n_languages).itertuples()
        for task_name, task in tasks.items()
        for model in models.iloc[:n_models]["id"]
        if task_name in model_tasks[model] and lang.bcp_47 in valid_task_langs[task_name]
    ]
    combis = pd.DataFrame(combis, columns=["task", "model", "bcp_47", "sentence_nr"])

    # Load cached results and filter out completed combinations
    old_results = load("results-detailed")
    if not old_results.empty:
        # Only treat status==\"ok\" (or missing status) as completed.
        if "status" in old_results.columns:
            ok_mask = old_results["status"].isna() | (old_results["status"] == "ok")
            completed_df = old_results.loc[ok_mask, ["task", "model", "bcp_47", "sentence_nr"]]
        else:
            completed_df = old_results[["task", "model", "bcp_47", "sentence_nr"]]
        completed = set(completed_df.apply(tuple, axis=1))
        combis = combis[~combis.apply(lambda row: tuple(row) in completed, axis=1)]

    print(f"Running {len(combis)} evaluation tasks across {combis['model'].nunique()} models...")

    current_models = set(models.iloc[:n_models]["id"])
    current_languages = set(languages.head(n_languages)["bcp_47"])

    # We evaluate ONE MODEL AT A TIME and checkpoint to HuggingFace after each
    # model finishes its ENTIRE matrix (every task × language × sentence
    # attempted). This buys two things:
    #   1. Progress survives interruption — the GitHub-hosted runner's hard 6h
    #      cap, or a local laptop sleeping. The next run skips models already
    #      fully logged (status=="ok" rows), so a large onboarding completes
    #      across however many runs it takes instead of losing everything.
    #   2. A model only enters the PUBLISHED aggregate once it has full
    #      benchmark coverage, so a half-evaluated model can never show an
    #      inflated score from a sparse sample (the failure mode that once put
    #      a small model at rank #1).
    all_results = old_results.copy() if not old_results.empty else pd.DataFrame(
        columns=["task", "model", "bcp_47", "metric", "sentence_nr", "score", "origin", "status"]
    )
    dedup_keys = ["task", "model", "bcp_47", "metric", "sentence_nr"]
    batch_size = 2000

    # Cohort order, so each checkpoint boundary is a fully-computed model.
    # Models already fully cached have no pending combis and are skipped.
    pending_models = [m for m in models.iloc[:n_models]["id"].tolist()
                      if (combis["model"] == m).any()]

    # A model is COVERAGE-COMPLETE iff it has zero pending combis: either it was
    # already fully evaluated before this run (not in pending_models), or this
    # run finishes its full matrix below. Only coverage-complete models may be
    # published — this is what stops a model with a handful of evaluated
    # languages from showing an inflated mean and jumping the leaderboard.
    covered = current_models - set(pending_models)
    print(f"{len(covered)} models already fully covered; "
          f"{len(pending_models)} pending this run")
    results_agg = None
    for mi, model_id in enumerate(pending_models, 1):
        model_combis = combis[combis["model"] == model_id]
        print(f"[{mi}/{len(pending_models)}] {model_id}: {len(model_combis)} new samples")
        model_out = []
        for i in tqdm(range(0, len(model_combis), batch_size),
                      colour="blue", desc=model_id):
            batch = model_combis.iloc[i:i + batch_size]
            batch_res = await tqdm_asyncio.gather(
                *[tasks[task_name](model, bcp_47, sentence_nr)
                  for _, (task_name, model, bcp_47, sentence_nr) in batch.iterrows()]
            )
            model_out.extend(r for result in batch_res for r in result)
        model_df = pd.DataFrame(model_out) if model_out else pd.DataFrame(columns=all_results.columns)

        if not model_df.empty and "status" in model_df.columns:
            err = (model_df["status"] != "ok").mean()
            if err > 0.8:
                # Logged so auto_blocklist sees it; publish-health filter keeps
                # it out of the aggregate this run.
                print(f"  ⚠ {model_id}: {err:.0%} of new rows errored — logged, not published")

        all_results = pd.concat([all_results, model_df]).drop_duplicates(
            subset=dedup_keys, keep="last"
        )
        # This model's full matrix is now attempted → coverage-complete.
        covered.add(model_id)
        results_agg = checkpoint(all_results, covered, current_languages, model_id)

    if results_agg is None:
        # Everything was already cached — still refresh the published tables
        # from the existing log (e.g. cohort/cost metadata may have changed).
        results_agg = checkpoint(all_results, covered, current_languages, "no new work")

    elapsed = time.time() - start_time
    print(f"Evaluation completed in {str(timedelta(seconds=int(elapsed)))}")


if __name__ == "__main__":
    results = asyncio.run(evaluate())
