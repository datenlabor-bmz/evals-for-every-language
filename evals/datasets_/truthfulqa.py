import random
from collections import Counter, defaultdict

from langcodes import Language, standardize_tag
from rich import print
from tqdm import tqdm
import asyncio
from tqdm.asyncio import tqdm_asyncio
import os

from datasets import Dataset, load_dataset
from models import translate_google, get_google_supported_languages

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
        ds = _load_dataset(
            slug_uhura_truthfulqa, tags_uhura_truthfulqa[language_bcp_47]
        )
        ds = ds.map(add_choices)
        task = ds["test"][nr]
        return "masakhane/uhura-truthfulqa", task
    else:
        return None, None



def translate_truthfulqa(languages):
    human_translated = [*tags_uhura_truthfulqa.keys()]
    untranslated = [
        lang
        for lang in languages["bcp_47"].values[:100]
        if lang not in human_translated and lang in get_google_supported_languages()
    ]
    n_samples = 10

    slug = "fair-forward/truthfulqa-autotranslated"
    for lang in tqdm(untranslated):
        # check if already exists on hub
        try:
            ds_lang = load_dataset(slug, lang)
        except (ValueError, Exception):
            print(f"Translating {lang}...")
            for split in ["train", "test"]:
                ds = _load_dataset(slug_uhura_truthfulqa, tags_uhura_truthfulqa["en"], split=split)
                samples = []
                if split == "train":
                    samples.extend(ds)
                else:
                    for i in range(n_samples):
                        task = ds[i]
                        samples.append(task)
                questions_tr = [
                    translate_google(s["question"], "en", lang) for s in samples
                ]
                questions_tr = asyncio.run(tqdm_asyncio.gather(*questions_tr))
                choices_texts_concatenated = []
                for s in samples:
                    for choice in eval(s["choices"]):
                        choices_texts_concatenated.append(choice)
                choices_tr = [
                    translate_google(c, "en", lang) for c in choices_texts_concatenated
                ]
                choices_tr = asyncio.run(tqdm_asyncio.gather(*choices_tr))
                # group into chunks of 4
                choices_tr = [
                    choices_tr[i : i + 4] for i in range(0, len(choices_tr), 4)
                ]

                ds_lang = Dataset.from_dict(
                    {
                        "subject": [s["subject"] for s in samples],
                        "question": questions_tr,
                        "choices": choices_tr,
                        "answer": [s["answer"] for s in samples],
                    }
                )
                ds_lang.push_to_hub(
                    slug,
                    split=split,
                    config_name=lang,
                    token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"),
                )
                ds_lang.to_json(
                    f"data/translations/mmlu/{lang}_{split}.json",
                    lines=False,
                    force_ascii=False,
                    indent=2,
                )
