import re

import pandas as pd
from datasets_.commonvoice import commonvoice
from datasets_.fleurs import fleurs
from datasets_.flores import flores
from joblib.memory import Memory
from langcodes import Language, standardize_tag
from language_data.population_data import LANGUAGE_SPEAKING_POPULATION

cache = Memory(location=".cache", verbose=0).cache

# load general language data
languages = {
    lang: pop
    for lang, pop in LANGUAGE_SPEAKING_POPULATION.items()
    if not re.match(r".*-[A-Z]{2}$", lang)
}
languages = pd.DataFrame(list(languages.items()), columns=["bcp_47", "speakers"])
languages["language_name"] = languages["bcp_47"].apply(
    lambda x: Language.get(x).display_name()
)
languages["autonym"] = languages["bcp_47"].apply(
    lambda x: Language.get(x).autonym().title()
)

glottolog = pd.read_csv(
    "data/glottolog_languoid.csv/languoid.csv", na_values=[""], keep_default_na=False
)  # Min _Nan_ Chinese is not N/A!
glottolog["bcp_47"] = glottolog["iso639P3code"].apply(
    lambda x: standardize_tag(x, macro=True) if not pd.isna(x) else None
)

@cache
def language_family(bcp_47):
    languoid = glottolog[glottolog["bcp_47"] == bcp_47].iloc[0]
    if pd.isna(languoid["family_id"]):
        return None
    family = glottolog[glottolog["id"] == languoid["family_id"]].iloc[0]
    return family["name"]

languages["family"] = languages["bcp_47"].apply(language_family)

# load script codes and names
scripts = pd.read_csv("data/ScriptCodes.csv").rename(
    columns={"Code": "iso15924", "English Name": "script_name"}
)

def script_name(iso15924):
    return scripts[scripts["iso15924"] == iso15924]["script_name"].values[0]


# merge data
# always "left" because keep it simple for now
languages = pd.merge(languages, flores, on="bcp_47", how="left")
languages = pd.merge(languages, fleurs, on="bcp_47", how="left")
languages = pd.merge(languages, commonvoice, on="bcp_47", how="left")
languages["in_benchmark"] = languages["bcp_47"].isin(flores["bcp_47"])
languages = languages.sort_values(by="speakers", ascending=False)
