import random
from collections import Counter, defaultdict

from langcodes import Language, standardize_tag
from rich import print

from .util import _get_dataset_config_names, _load_dataset
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


def load_mmlu(language_bcp_47, nr):
    categories = sorted(
        list(set(_load_dataset("masakhane/afrimmlu", "eng")["dev"]["subject"]))
    )
    category = categories[nr % len(categories)]
    random.seed(nr)
    i = random.randint(0, 100)
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
        for a in _get_dataset_config_names("Eurolingua/mmlux")
    )
    if language_bcp_47 in tags_afrimmlu:
        ds = _load_dataset("masakhane/afrimmlu", tags_afrimmlu[language_bcp_47])
        ds = ds.map(parse_choices)
        examples = ds["dev"].filter(lambda x: x["subject"] == category)
        task = ds["test"].filter(lambda x: x["subject"] == category)[i]
        return "masakhane/afrimmlu", examples, task
    elif language_bcp_47 in tags_global_mmlu:
        ds = _load_dataset("CohereForAI/Global-MMLU", tags_global_mmlu[language_bcp_47])
        ds = ds.map(add_choices)
        examples = ds["dev"].filter(lambda x: x["subject"] == category)
        task = ds["test"].filter(lambda x: x["subject"] == category)[i]
        return "CohereForAI/Global-MMLU", examples, task
    elif language_bcp_47 in tags_okapi:
        ds = _load_dataset(
            "lighteval/okapi_mmlu", language_bcp_47, trust_remote_code=True
        )
        examples = ds["dev"].filter(lambda x: x["subject"] == category)
        task = ds["test"].filter(lambda x: x["id"] == f"{category}/test/{i}")[0]
        return "lighteval/okapi_mmlu", examples, task
    elif language_bcp_47 in tags_mmlux:
        # loading this is more complicated, todo
        return None, None, None
    else:
        return None, None, None
