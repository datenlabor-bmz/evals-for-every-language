from datasets import get_dataset_config_names, load_dataset
from joblib.memory import Memory

cache = Memory(location=".cache", verbose=0).cache


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
