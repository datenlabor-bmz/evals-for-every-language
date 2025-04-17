from joblib.memory import Memory
from datasets import load_dataset, get_dataset_config_names
from rich import print
from langcodes import standardize_tag, Language
from collections import defaultdict, Counter
cache = Memory(location=".cache", verbose=0).cache

@cache
def _get_dataset_config_names(dataset):
    return get_dataset_config_names(dataset)

@cache
def _load_dataset(dataset, subset, **kwargs):
    return load_dataset(dataset, subset, **kwargs)

def print_counts(slug,subjects_dev, subjects_test):
    print(f"{slug:<25} {len(list(set(subjects_test))):>3} test categories, {len(subjects_test):>6} samples, {len(list(set(subjects_dev))):>3} dev categories, {len(subjects_dev):>6} dev samples")

def print_datasets_analysis():
    print("Category counts and sample counts per dataset:")
    slug1 = "masakhane/afrimmlu"
    ds1 = _load_dataset(slug1, "eng")
    print_counts(slug1, ds1["dev"]["subject"], ds1["test"]["subject"])
    langs1 = _get_dataset_config_names(slug1)
    langs1 = [standardize_tag(a, macro=True) for a in langs1]

    slug2 = "openai/MMMLU" # does not have dev set! – but: these languages are all also present in Global-MMLU
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
    print_counts(slug4, [a.split("/")[0] for a in ds4["dev"]["id"]], [a.split("/")[0] for a in ds4["test"]["id"]])
    langs4 = _get_dataset_config_names(slug4)


    slug5 = "Eurolingua/mmlux"
    subsets = _get_dataset_config_names(slug5)
    subjects = set(a.rsplit("_", 1)[0] for a in subsets)
    rows_test = [_load_dataset(slug5, subset)["test"]["id"] for subset in subsets if "_DA" in subset]
    rows_test = [a.split("/")[0] for l in rows_test for a in l]
    rows_dev = [_load_dataset(slug5, subset)["dev"]["id"] for subset in subsets if "_DA" in subset]
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
    print(sorted((lang, datasets) for lang, datasets in lang_datasets.items() if  slug3 not in datasets))
    print(Counter(dataset for ds_list in lang_datasets.values() for dataset in ds_list if  slug3 not in ds_list))
    print(list(set(ds1["test"]["subject"])))

# based on this analysis:
# - we drop the OpenAI dataset, since it does not have a dev set, and since every language that it has is also present in Global-MMLU
# - we stick to the 5 categories of the AfriMMLU dataset, since this is the most restricted dataset, and these 5 categories are present in all datasets, so this is good for comparability

# AfriMMLU is human-translated, but has only 5 task categories
# Global-MMLU is partially human-translated, specifically those 15 languages are that are also present in Global-MMLU-Lite, which are mostly from MMMLU; otherwise translated using Google Translate
# Okapi-MMLU is translated using ChatGPT (version unclear)
# MMLUX is translated using DeepL
# Therefore, the priority is: AfriMMLU, Global-MMLU, Okapi-MMLU, MMLUX

print_datasets_analysis()

def load_mmlu(language_bcp_47):
    pass
