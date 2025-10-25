import os

import pandas as pd
from datasets import Dataset, get_dataset_config_names, load_dataset
from datasets.exceptions import DatasetNotFoundError
from joblib.memory import Memory

cache = Memory(location=".cache", verbose=0).cache
TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")


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
    from huggingface_hub.errors import RepositoryNotFoundError
    try:
        ds = load_dataset(f"fair-forward/evals-for-every-language-{fname}", token=TOKEN)
        return ds["train"].to_pandas()
    except (DatasetNotFoundError, RepositoryNotFoundError, KeyError):
        return pd.DataFrame()


def save(df: pd.DataFrame, fname: str):
    ds = Dataset.from_pandas(df)
    ds.push_to_hub(f"fair-forward/evals-for-every-language-{fname}", token=TOKEN)
    ds.to_json(f"results/{fname}.json", lines=False, force_ascii=False, indent=2)
