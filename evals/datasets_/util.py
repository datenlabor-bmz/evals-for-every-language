from datasets import get_dataset_config_names, load_dataset
from joblib.memory import Memory

cache = Memory(location=".cache", verbose=0).cache


@cache
def _get_dataset_config_names(dataset):
    return get_dataset_config_names(dataset)


@cache
def _load_dataset(dataset, subset, **kwargs):
    return load_dataset(dataset, subset, **kwargs)
