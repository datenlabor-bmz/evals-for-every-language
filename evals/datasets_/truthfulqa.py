import random
from collections import Counter, defaultdict

from langcodes import Language, standardize_tag
from rich import print

from datasets_.util import _get_dataset_config_names, _load_dataset

slug_uhura_truthfulqa = "masakhane/uhura-truthfulqa"
tags_uhura_truthfulqa = {
    standardize_tag(a.split("_")[0], macro=True): a for a in _get_dataset_config_names(slug_uhura_truthfulqa)
    if a.endswith("multiple_choice")
}


def add_choices(row):
    row["choices"] = row["mc1_targets"]["choices"]
    row["labels"] = row["mc1_targets"]["labels"]
    return row


def load_truthfulqa(language_bcp_47, nr):
    if language_bcp_47 in tags_uhura_truthfulqa.keys():
        ds = _load_dataset(slug_uhura_truthfulqa, tags_uhura_truthfulqa[language_bcp_47])
        ds = ds.map(add_choices)
        examples = ds["train"]
        task = ds["test"][nr]
        return "masakhane/uhura-truthfulqa", examples, task
    else:
        return None, None, None
