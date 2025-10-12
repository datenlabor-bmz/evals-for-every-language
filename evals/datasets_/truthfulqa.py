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
slug_truthfulqa_autotranslated = "fair-forward/truthfulqa-autotranslated"

tags_uhura_truthfulqa = {
    standardize_tag(a.split("_")[0], macro=True): a
    for a in _get_dataset_config_names(slug_uhura_truthfulqa)
    if a.endswith("multiple_choice")
}

tags_truthfulqa_autotranslated = {
    standardize_tag(a, macro=True): a
    for a in _get_dataset_config_names(slug_truthfulqa_autotranslated)
}
tags_truthfulqa_autotranslated = {}


def add_choices(row):
    row["choices"] = row["mc1_targets"]["choices"]
    row["labels"] = row["mc1_targets"]["labels"]
    return row


async def load_truthfulqa(language_bcp_47, nr):
    if language_bcp_47 in tags_uhura_truthfulqa.keys():
        ds = _load_dataset(
            slug_uhura_truthfulqa, tags_uhura_truthfulqa[language_bcp_47]
        )
        ds = ds.map(add_choices)
        task = ds["test"][nr]
        # Ensure there is a correct answer before returning the task
        if 1 not in task["labels"]:
            return None, None, None
        return "masakhane/uhura-truthfulqa", task, "human"
    elif language_bcp_47 in tags_truthfulqa_autotranslated.keys():
        # Load from auto-translated dataset (same samples as translation)
        ds = _load_dataset(slug_truthfulqa_autotranslated, language_bcp_47)
        test_split = ds["test"] if "test" in ds else ds
        task = test_split[nr]
        # Ensure there is a correct answer before returning the task
        if 1 not in task.get("labels", []):
            return None, None, None
        return slug_truthfulqa_autotranslated, task, "machine"
    # TODO: add Okapi, TruthfulQA-X @Jonas
    else:
        return None, None, None


def translate_truthfulqa(languages):
    human_translated = [*tags_uhura_truthfulqa.keys()]
    untranslated = [
        lang
        for lang in languages["bcp_47"].values[:150]
        if lang not in human_translated and lang in get_google_supported_languages()
    ]
    n_samples = 20

    # Set fixed seed for consistent sample selection across all languages
    random.seed(42)

    slug = "fair-forward/truthfulqa-autotranslated"
    for lang in tqdm(untranslated):
        # check if already exists on hub
        try:
            ds_lang = load_dataset(slug, lang)
        except (ValueError, Exception):
            print(f"Translating {lang}...")
            for split in ["train", "test"]:
                ds = _load_dataset(
                    slug_uhura_truthfulqa, tags_uhura_truthfulqa["en"], split=split
                )
                samples = []
                if split == "train":
                    samples.extend(ds)
                else:
                    # Use the same 20 samples that the evaluation pipeline uses (indices 0-19)
                    for i in range(min(n_samples, len(ds))):
                        task = ds[i]
                        samples.append(task)

                # Translate questions
                questions_tr = [
                    translate_google(s["question"], "en", lang) for s in samples
                ]
                questions_tr = asyncio.run(tqdm_asyncio.gather(*questions_tr))

                # Translate choices for each sample
                all_choices_tr = []
                all_labels = []

                for s in samples:
                    # Get choices from mc1_targets
                    choices = s["mc1_targets"]["choices"]
                    labels = s["mc1_targets"]["labels"]

                    # Translate choices
                    choices_tr = [
                        translate_google(choice, "en", lang) for choice in choices
                    ]
                    choices_tr = asyncio.run(tqdm_asyncio.gather(*choices_tr))

                    all_choices_tr.append(choices_tr)
                    all_labels.append(labels)

                ds_lang = Dataset.from_dict(
                    {
                        "question": questions_tr,
                        "choices": all_choices_tr,
                        "labels": all_labels,
                    }
                )
                ds_lang.push_to_hub(
                    slug,
                    split=split,
                    config_name=lang,
                    token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"),
                )
                ds_lang.to_json(
                    f"data/translations/truthfulqa/{lang}_{split}.json",
                    lines=False,
                    force_ascii=False,
                    indent=2,
                )
