import random

from rich import print
from models import translate_google, get_google_supported_languages
from tqdm import tqdm
from datasets import load_dataset, Dataset
import asyncio
from tqdm.asyncio import tqdm_asyncio
import os

from datasets_.util import _get_dataset_config_names, _load_dataset, standardize_bcp47

slug_uhura_arc_easy = "masakhane/uhura-arc-easy"
tags_uhura_arc_easy = {
    standardize_bcp47(a.split("_")[0]): a
    for a in _get_dataset_config_names(slug_uhura_arc_easy)
    if not a.endswith("unmatched")
}


random.seed(42)
id_sets_train = [
    set(_load_dataset(slug_uhura_arc_easy, tag, split="train")["id"])
    for tag in tags_uhura_arc_easy.values()
]
common_ids_train = list(sorted(set.intersection(*id_sets_train)))
random.shuffle(common_ids_train)
id_sets_test = [
    set(_load_dataset(slug_uhura_arc_easy, tag, split="test")["id"])
    for tag in tags_uhura_arc_easy.values()
]
common_ids_test = list(sorted(set.intersection(*id_sets_test)))
random.shuffle(common_ids_test)

slug_uhura_arc_easy_translated = "fair-forward/arc-easy-autotranslated"
tags_uhura_arc_easy_translated = {
    standardize_bcp47(a.split("_")[0]): a
    for a in _get_dataset_config_names(slug_uhura_arc_easy_translated)
}


def add_choices(row):
    row["choices"] = row["choices"]["text"]
    return row


def load_uhura_arc_easy(language_bcp_47, nr):
    if language_bcp_47 in tags_uhura_arc_easy.keys():
        ds = _load_dataset(slug_uhura_arc_easy, tags_uhura_arc_easy[language_bcp_47])
        ds = ds.map(add_choices)
        ds = ds.rename_column("answerKey", "answer")
        task = ds["test"].filter(lambda x: x["id"] == common_ids_test[nr])[0]
        return "masakhane/uhura-arc-easy", task, "human"
    if language_bcp_47 in tags_uhura_arc_easy_translated.keys():
        ds = _load_dataset(
            slug_uhura_arc_easy_translated,
            tags_uhura_arc_easy_translated[language_bcp_47],
        )
        ds = ds.rename_column("answerKey", "answer")
        task = ds["test"].filter(lambda x: x["id"] == common_ids_test[nr])[0]
        return "fair-forward/arc-easy-autotranslated", task, "machine"
    else:
        return None, None, None


def translate_arc(languages):
    human_translated = tags_uhura_arc_easy.keys()
    untranslated = [
        lang
        for lang in languages["bcp_47"].values
        if lang not in human_translated and lang in get_google_supported_languages()
    ]
    n_samples = 10
    train_ids = common_ids_train[: n_samples + 3]
    en_train = _load_dataset(
        slug_uhura_arc_easy, subset=tags_uhura_arc_easy["en"], split="train"
    )
    en_train = en_train.filter(lambda x: x["id"] in train_ids)
    test_ids = common_ids_test[:n_samples]
    en_test = _load_dataset(
        slug_uhura_arc_easy, subset=tags_uhura_arc_easy["en"], split="test"
    )
    en_test = en_test.filter(lambda x: x["id"] in test_ids)
    data = {"train": en_train, "test": en_test}

    slug = "fair-forward/arc-easy-autotranslated"
    for lang in tqdm(untranslated):
        # check if already exists on hub
        try:
            ds_lang = load_dataset(slug, lang)
        except (ValueError, Exception):
            print(f"Translating {lang}...")
            for split, data_en in data.items():
                questions_tr = [
                    translate_google(q, "en", lang) for q in data_en["question"]
                ]
                questions_tr = asyncio.run(tqdm_asyncio.gather(*questions_tr))
                choices_texts_concatenated = []
                for choice in data_en["choices"]:
                    for option in choice["text"]:
                        choices_texts_concatenated.append(option)
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
                        "id": data_en["id"],
                        "question": questions_tr,
                        "choices": choices_tr,
                        "answerKey": data_en["answerKey"],
                    }
                )
                ds_lang.push_to_hub(
                    slug,
                    split=split,
                    config_name=lang,
                    token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"),
                )
                ds_lang.to_json(
                    f"data/translations/arc/{lang}_{split}.json",
                    lines=False,
                    force_ascii=False,
                    indent=2,
                )
