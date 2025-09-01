import asyncio
import os
import random

from datasets import Dataset, load_dataset
from datasets_.util import _get_dataset_config_names, _load_dataset, cache
from langcodes import Language, standardize_tag
from models import get_google_supported_languages, translate_google
from rich import print
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

slug_mgsm = "juletxara/mgsm"
tags_mgsm = {
    standardize_tag(a, macro=True): a for a in _get_dataset_config_names(slug_mgsm)
}
slug_afrimgsm = "masakhane/afrimgsm"
tags_afrimgsm = {
    standardize_tag(a, macro=True): a for a in _get_dataset_config_names(slug_afrimgsm)
}
slug_gsm8kx = "Eurolingua/gsm8kx"
tags_gsm8kx = {
    standardize_tag(a, macro=True): a
    for a in _get_dataset_config_names(slug_gsm8kx, trust_remote_code=True)
}
slug_gsm_autotranslated = "fair-forward/gsm-autotranslated"
tags_gsm_autotranslated = {
    standardize_tag(a, macro=True): a
    for a in _get_dataset_config_names(slug_gsm_autotranslated)
}


def parse_number(i):
    if isinstance(i, int):
        return i
    try:
        return int(i.replace(",", "").replace(".", ""))
    except ValueError:
        return None


@cache
def _get_mgsm_item(dataset_slug, subset_tag, nr, trust_remote_code=False):
    """Cache individual MGSM items efficiently"""
    try:
        ds = _load_dataset(dataset_slug, subset=subset_tag, split="test", trust_remote_code=trust_remote_code)
        if nr >= len(ds):
            return None
        
        row = ds[nr]
        
        # Post-process based on dataset type
        if dataset_slug == slug_gsm8kx:
            row["answer_number"] = row["answer"].split("####")[1].strip()
        
        return row
    except Exception:
        # Dataset doesn't exist or doesn't have test split
        return None


def load_mgsm(language_bcp_47, nr):
    if language_bcp_47 in tags_mgsm.keys():
        item = _get_mgsm_item(slug_mgsm, tags_mgsm[language_bcp_47], nr)
        return slug_mgsm, item, "human" if item else (None, None, None)
    elif language_bcp_47 in tags_afrimgsm.keys():
        item = _get_mgsm_item(slug_afrimgsm, tags_afrimgsm[language_bcp_47], nr)
        return slug_afrimgsm, item, "human" if item else (None, None, None)
    elif language_bcp_47 in tags_gsm8kx.keys():
        item = _get_mgsm_item(slug_gsm8kx, tags_gsm8kx[language_bcp_47], nr, trust_remote_code=True)
        return slug_gsm8kx, item, "machine" if item else (None, None, None)
    elif language_bcp_47 in tags_gsm_autotranslated.keys():
        item = _get_mgsm_item(slug_gsm_autotranslated, tags_gsm_autotranslated[language_bcp_47], nr)
        return slug_gsm_autotranslated, item, "machine" if item else (None, None, None)
    else:
        return None, None, None


def translate_mgsm(languages):
    human_translated = [*tags_mgsm.keys(), *tags_afrimgsm.keys()]
    untranslated = [
        lang
        for lang in languages["bcp_47"].values[:100]
        if lang not in human_translated and lang in get_google_supported_languages()
    ]
    en = _load_dataset(slug_mgsm, subset=tags_mgsm["en"], split="test")
    slug = "fair-forward/gsm-autotranslated"
    for lang in tqdm(untranslated):
        # check if already exists on hub
        try:
            ds_lang = load_dataset(slug, lang, split="test")
        except ValueError:
            print(f"Translating {lang}...")
            questions_tr = [translate_google(q, "en", lang) for q in en["question"]]
            questions_tr = asyncio.run(tqdm_asyncio.gather(*questions_tr))
            ds_lang = Dataset.from_dict(
                {
                    "question": questions_tr,
                    "answer": en["answer"],
                    "answer_number": en["answer_number"],
                    "equation_solution": en["equation_solution"],
                }
            )
            ds_lang.push_to_hub(
                slug,
                split="test",
                config_name=lang,
                token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"),
            )
            ds_lang.to_json(
                f"data/translations/mgsm/{lang}.json", lines=False, force_ascii=False, indent=2
            )
