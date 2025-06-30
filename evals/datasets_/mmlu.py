import asyncio
import os
import random
from collections import Counter, defaultdict

from datasets import Dataset, load_dataset
from datasets_.util import _get_dataset_config_names, _load_dataset
from langcodes import Language, standardize_tag
from models import google_supported_languages, translate_google
from rich import print
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio


def print_counts(slug, subjects_dev, subjects_test):
    print(
        f"{slug:<25} {len(list(set(subjects_test))):>3} test categories, {len(subjects_test):>6} samples, {len(list(set(subjects_dev))):>3} dev categories, {len(subjects_dev):>6} dev samples"
    )


def print_datasets_analysis():
    print("Category counts and sample counts per dataset:")
    slug1 = "masakhane/afrimmlu"
    ds1 = _load_dataset(slug1, "eng")
    print_counts(slug1, ds1["dev"]["subject"], ds1["test"]["subject"])
    langs1 = _get_dataset_config_names(slug1)
    langs1 = [standardize_tag(a, macro=True) for a in langs1]

    slug2 = "openai/MMMLU"  # does not have dev set! – but: these languages are all also present in Global-MMLU
    ds2 = _load_dataset(slug2, "FR_FR")
    print_counts(slug2, [], ds2["test"]["Subject"])
    langs2 = _get_dataset_config_names(slug2)
    langs2 = [a.split("_")[0].lower() for a in langs2]
    langs2.remove("default")

    slug3 = "CohereForAI/Global-MMLU"
    ds3 = _load_dataset(slug3, "en")
    print_counts(slug3, ds3["dev"]["subject"], ds3["test"]["subject"])
    langs3 = _get_dataset_config_names(slug3)
    langs3 = [standardize_tag(a, macro=True) for a in langs3]

    slug4 = "lighteval/okapi_mmlu"
    ds4 = _load_dataset(slug4, "ar", trust_remote_code=True)
    print_counts(
        slug4,
        [a.split("/")[0] for a in ds4["dev"]["id"]],
        [a.split("/")[0] for a in ds4["test"]["id"]],
    )
    langs4 = _get_dataset_config_names(slug4)

    slug5 = "Eurolingua/mmlux"
    subsets = _get_dataset_config_names(slug5)
    subjects = set(a.rsplit("_", 1)[0] for a in subsets)
    rows_test = [
        _load_dataset(slug5, subset)["test"]["id"]
        for subset in subsets
        if "_DA" in subset
    ]
    rows_test = [a.split("/")[0] for l in rows_test for a in l]
    rows_dev = [
        _load_dataset(slug5, subset)["dev"]["id"]
        for subset in subsets
        if "_DA" in subset
    ]
    rows_dev = [a.split("/")[0] for l in rows_dev for a in l]
    print_counts(slug5, rows_dev, rows_test)
    langs5 = list(set(a.rsplit("_", 1)[1].split("-")[0].lower() for a in subsets))

    langs = langs1 + langs2 + langs3 + langs4 + langs5
    lang_datasets = defaultdict(list)
    for slug, langs_list in [
        (slug1, langs1),
        (slug2, langs2),
        (slug3, langs3),
        (slug4, langs4),
        (slug5, langs5),
    ]:
        for lang in langs_list:
            lname = Language.get(lang).display_name()
            lang_datasets[lname].append(slug)
    print("Datasets per language:")
    print(sorted(lang_datasets.items()))
    print(len(set(langs)))

    print("Datasets per language for languages that are not in Global-MMLU:")
    print(
        sorted(
            (lang, datasets)
            for lang, datasets in lang_datasets.items()
            if slug3 not in datasets
        )
    )
    print(
        Counter(
            dataset
            for ds_list in lang_datasets.values()
            for dataset in ds_list
            if slug3 not in ds_list
        )
    )
    print(list(set(ds1["test"]["subject"])))


# based on this analysis:
# - we drop the OpenAI dataset, since it does not have a dev set, and since every language that it has is also present in Global-MMLU
# - we stick to the 5 categories of the AfriMMLU dataset, since this is the most restricted dataset, and these 5 categories are present in all datasets, so this is good for comparability

# AfriMMLU is human-translated, but has only 5 task categories
# Global-MMLU is mixed-translated, specifically those 15 languages are that are also present in Global-MMLU-Lite, which are mostly from MMMLU; otherwise translated using Google Translate
# Okapi-MMLU is translated using ChatGPT (version unclear)
# MMLUX is translated using DeepL
# Therefore, the priority is: AfriMMLU, Global-MMLU, MMLUX, Okapi-MMLU

# print_datasets_analysis()


def parse_choices(row):
    if not isinstance(row["choices"], list):
        row["choices"] = eval(row["choices"])
    return row


def add_choices(row):
    row["choices"] = [
        row["option_a"],
        row["option_b"],
        row["option_c"],
        row["option_d"],
    ]
    return row


tags_afrimmlu = {
    standardize_tag(a, macro=True): a
    for a in _get_dataset_config_names("masakhane/afrimmlu")
}
tags_global_mmlu = {
    standardize_tag(a, macro=True): a
    for a in _get_dataset_config_names("CohereForAI/Global-MMLU")
}
tags_okapi = _get_dataset_config_names("lighteval/okapi_mmlu")
tags_mmlux = set(
    a.rsplit("_", 1)[1].split("-")[0].lower()
    for a in _get_dataset_config_names("Eurolingua/mmlux", trust_remote_code=True)
)
tags_mmlu_autotranslated = _get_dataset_config_names("fair-forward/mmlu-autotranslated")

categories = sorted(
        list(set(_load_dataset("masakhane/afrimmlu", "eng")["dev"]["subject"]))
    )


def load_mmlu(language_bcp_47, nr):
    category = categories[nr % len(categories)]
    if language_bcp_47 in tags_afrimmlu.keys():
        ds = _load_dataset("masakhane/afrimmlu", tags_afrimmlu[language_bcp_47])
        ds = ds.map(parse_choices)
        examples = ds["dev"].filter(lambda x: x["subject"] == category)
        task = ds["test"].filter(lambda x: x["subject"] == category)[nr]
        return "masakhane/afrimmlu", examples, task
    elif language_bcp_47 in tags_global_mmlu.keys():
        ds = _load_dataset("CohereForAI/Global-MMLU", tags_global_mmlu[language_bcp_47])
        ds = ds.map(add_choices)
        examples = ds["dev"].filter(lambda x: x["subject"] == category)
        task = ds["test"].filter(lambda x: x["subject"] == category)[nr]
        return "CohereForAI/Global-MMLU", examples, task
    elif language_bcp_47 in tags_mmlu_autotranslated:
        ds = _load_dataset("fair-forward/mmlu-autotranslated", language_bcp_47)
        examples = ds["dev"].filter(lambda x: x["subject"] == category)
        task = ds["test"].filter(lambda x: x["subject"] == category)[nr]
        return "fair-forward/mmlu-autotranslated", examples, task
    else:
        return None, None, None


def translate_mmlu(languages):
    human_translated = [*tags_afrimmlu.keys(), *tags_global_mmlu.keys()]
    untranslated = [
        lang
        for lang in languages["bcp_47"].values[:100]
        if lang not in human_translated and lang in google_supported_languages
    ]
    n_samples = 10

    slug = "fair-forward/mmlu-autotranslated"
    for lang in tqdm(untranslated):
        # check if already exists on hub
        try:
            ds_lang = load_dataset(slug, lang)
        except (ValueError, Exception):
            print(f"Translating {lang}...")
            for split in ["dev", "test"]:
                ds = _load_dataset("masakhane/afrimmlu", "eng", split=split)
                samples = []
                for category in categories:
                    if split == "dev":
                        samples.extend(ds.filter(lambda x: x["subject"] == category))
                    else:
                        for i in range(n_samples):
                            task = ds.filter(lambda x: x["subject"] == category)[i]
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
