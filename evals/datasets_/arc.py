import random
from collections import Counter, defaultdict

from langcodes import Language, standardize_tag
from rich import print
from models import translate_google, google_supported_languages

from datasets_.util import _get_dataset_config_names, _load_dataset

slug_uhura_arc_easy = "masakhane/uhura-arc-easy"
tags_uhura_arc_easy = {
    standardize_tag(a.split("_")[0], macro=True): a for a in _get_dataset_config_names(slug_uhura_arc_easy)
    if not a.endswith("unmatched")
}


random.seed(42)
id_sets_train = [set(_load_dataset(slug_uhura_arc_easy, tag, split="train")["id"]) for tag in tags_uhura_arc_easy.values()]
common_ids_train = list(sorted(set.intersection(*id_sets_train)))
random.shuffle(common_ids_train)
id_sets_test = [set(_load_dataset(slug_uhura_arc_easy, tag, split="test")["id"]) for tag in tags_uhura_arc_easy.values()]
common_ids_test = list(sorted(set.intersection(*id_sets_test)))
random.shuffle(common_ids_test)




def add_choices(row):
    row["choices"] = row["choices"]["text"]
    return row


def load_uhura_arc_easy(language_bcp_47, nr):
    print(language_bcp_47, tags_uhura_arc_easy.keys())
    if language_bcp_47 in tags_uhura_arc_easy.keys():
        ds = _load_dataset(slug_uhura_arc_easy, tags_uhura_arc_easy[language_bcp_47])
        ds = ds.map(add_choices)
        ds = ds.rename_column("answerKey", "answer")
        train_ids = common_ids_train[nr:nr+3]
        examples = ds["train"].filter(lambda x: x["id"] in train_ids)
        task = ds["test"].filter(lambda x: x["id"] == common_ids_test[nr])[0]
        return "masakhane/uhura-arc-easy", examples, task
    else:
        return None, None, None

def translate_arc(languages):
    human_translated = tags_uhura_arc_easy.keys()
    untranslated = [
        lang
        for lang in languages["bcp_47"].values[:100]
        if lang not in human_translated and lang in google_supported_languages
    ]
    n_samples = 10
    en_train = _load_dataset(slug_uhura_arc_easy, subset=tags_uhura_arc_easy["en"], split="test")
    train_ids = common_ids_train[:n_samples+3]
    examples = en_train["train"].filter(lambda x: x["id"] in train_ids)
    task = en_train["test"].filter(lambda x: x["id"] == common_ids_test[nr])[0]
    return "masakhane/uhura-arc-easy", examples, task
    

    # slug = "fair-forward/gsm-autotranslated"
    # for lang in tqdm(untranslated):
    #     # check if already exists on hub
    #     try:
    #         ds_lang = load_dataset(slug, lang, split="test")
    #     except ValueError:
    #         print(f"Translating {lang}...")
    #         questions_tr = [translate_google(q, "en", lang) for q in en["question"]]
    #         questions_tr = asyncio.run(tqdm_asyncio.gather(*questions_tr))
    #         ds_lang = Dataset.from_dict(
    #             {
    #                 "question": questions_tr,
    #                 "answer": en["answer"],
    #                 "answer_number": en["answer_number"],
    #                 "equation_solution": en["equation_solution"],
    #             }
    #         )
    #         ds_lang.push_to_hub(
    #             slug,
    #             split="test",
    #             config_name=lang,
    #             token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"),
    #         )
    #         ds_lang.to_json(
    #             f"data/mgsm/{lang}.json", lines=False, force_ascii=False, indent=2
    #         )
