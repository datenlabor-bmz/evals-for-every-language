import os
from pathlib import Path

import pandas as pd
from datasets import Dataset, get_dataset_config_names, load_dataset
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub.errors import RepositoryNotFoundError
from joblib.memory import Memory
from langcodes import standardize_tag

cache = Memory(location=".cache", verbose=0).cache
TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

# Macrolanguage mappings: when standardize_tag returns a macrolanguage,
# map it to the preferred specific variant for consistency across datasets.
# This ensures results from different benchmarks use the same language code.
MACROLANGUAGE_MAPPINGS = {
    "no": "nb",  # Norwegian -> Norwegian Bokmål (most widely used variant)
    # Add more mappings here if they cause duplicate entries in the languages table:
    # "ms": "zsm",  # Malay -> Standard Malay (if both appear in population data)
    # "ar": "arb",  # Arabic -> Standard Arabic (if both appear in population data)
    # "zh": "cmn",  # Chinese -> Mandarin Chinese (if both appear in population data)
    # Check LANGUAGE_SPEAKING_POPULATION to see which macrolanguages need mapping
}


def standardize_bcp47(tag: str, macro: bool = True) -> str:
    """Standardize a BCP-47 tag with consistent macrolanguage handling."""
    
    standardized = standardize_tag(tag, macro=macro)
    return MACROLANGUAGE_MAPPINGS.get(standardized, standardized)


@cache
def _get_dataset_config_names(dataset, **kwargs):
    return get_dataset_config_names(dataset, **kwargs)


@cache
def _load_dataset(dataset, subset, **kwargs):
    return load_dataset(dataset, subset, **kwargs)


# Cache individual dataset items to avoid reloading entire datasets
@cache
def _get_dataset_item(dataset, subset, split, index, **kwargs):
    """Load a single item from a dataset efficiently"""
    ds = load_dataset(dataset, subset, split=split, **kwargs)
    return ds[index] if index < len(ds) else None


def load(fname: str):
    try:
        ds = load_dataset(f"fair-forward/evals-for-every-language-{fname}", token=TOKEN)
        return ds["train"].to_pandas()
    except (DatasetNotFoundError, RepositoryNotFoundError, KeyError):
        return pd.DataFrame()


def save(df: pd.DataFrame, fname: str):
    df = df.drop(columns=["__index_level_0__"], errors="ignore")
    ds = Dataset.from_pandas(df)
    ds.push_to_hub(f"fair-forward/evals-for-every-language-{fname}", token=TOKEN)
    Path("results").mkdir(exist_ok=True)
    df.to_json(f"results/{fname}.json", orient="records", force_ascii=False, indent=2)


def get_valid_task_languages(task_name: str) -> set:
    """Return set of bcp_47 codes that have data available for the given task."""
    from datasets_.flores import flores, splits
    from datasets_.mmlu import tags_afrimmlu, tags_global_mmlu, tags_mmlu_autotranslated
    from datasets_.arc import tags_uhura_arc_easy, tags_uhura_arc_easy_translated
    from datasets_.truthfulqa import tags_uhura_truthfulqa
    from datasets_.mgsm import tags_mgsm, tags_afrimgsm, tags_gsm8kx, tags_gsm_autotranslated
    
    if task_name in ["translation_from", "translation_to", "classification"]:
        return set(flores["bcp_47"])
    elif task_name == "mmlu":
        return set([*tags_afrimmlu.keys(), *tags_global_mmlu.keys(), *tags_mmlu_autotranslated.keys()])
    elif task_name == "arc":
        return set([*tags_uhura_arc_easy.keys(), *tags_uhura_arc_easy_translated.keys()])
    elif task_name == "truthfulqa":
        return set(tags_uhura_truthfulqa.keys())
    elif task_name == "mgsm":
        return set([*tags_mgsm.keys(), *tags_afrimgsm.keys(), *tags_gsm8kx.keys(), *tags_gsm_autotranslated.keys()])
    return set()
