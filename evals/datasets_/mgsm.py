from datasets_.util import _get_dataset_config_names, _load_dataset
from langcodes import Language, standardize_tag

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
