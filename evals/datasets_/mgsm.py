import asyncio
import os

from datasets import Dataset, load_dataset
from datasets_.util import _get_dataset_config_names, _load_dataset
from langcodes import standardize_tag
from models import google_supported_languages, translate_google
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


def load_mgsm(language_bcp_47, nr):
    if language_bcp_47 in tags_mgsm.keys():
        ds = _load_dataset(slug_mgsm, subset=tags_mgsm[language_bcp_47], split="test")
        return slug_mgsm, ds[nr]
    elif language_bcp_47 in tags_afrimgsm.keys():
        ds = _load_dataset(
            slug_afrimgsm, subset=tags_afrimgsm[language_bcp_47], split="test"
        )
        return slug_afrimgsm, ds[nr]
    elif language_bcp_47 in tags_gsm_autotranslated.keys():
        ds = _load_dataset(
            slug_gsm_autotranslated, subset=tags_gsm_autotranslated[language_bcp_47], split="test"
        )
        return slug_gsm_autotranslated, ds[nr]
    elif language_bcp_47 in tags_gsm8kx.keys():
        row = _load_dataset(
            slug_gsm8kx,
            subset=tags_gsm8kx[language_bcp_47],
            split="test",
            trust_remote_code=True,
        )[nr]
        row["answer_number"] = row["answer"].split("####")[1].strip()
        return slug_gsm8kx, row
    else:
        return None, None


def translate_mgsm(languages):
    human_translated = [*tags_mgsm.keys(), *tags_afrimgsm.keys()]
    untranslated = [
        lang
        for lang in languages["bcp_47"].values[:100]
        if lang not in human_translated and lang in google_supported_languages
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
