import re
from collections import defaultdict
from joblib.memory import Memory
import pandas as pd
from language_data.population_data import LANGUAGE_SPEAKING_POPULATION

cache = Memory(location=".cache", verbose=0).cache


def population(bcp_47):
    items = {
        re.sub(r"^[a-z]+-", "", lang): pop
        for lang, pop in LANGUAGE_SPEAKING_POPULATION.items()
        if re.match(rf"^{bcp_47}-[A-Z]{{2}}$", lang)
    }
    return items


@cache
def make_country_table(language_table):
    countries = defaultdict(list)
    for lang in language_table.itertuples():
        for country, speaker_pop in population(lang.bcp_47).items():
            countries[country].append(
                {
                    "name": lang.language_name,
                    "bcp_47": lang.bcp_47,
                    "population": speaker_pop,
                    "score": lang.average if not pd.isna(lang.average) else None,
                }
            )
    for country, languages in countries.items():
        speaker_pop = sum(entry["population"] for entry in languages)

        if speaker_pop < 1000:  # Grey out low-population countries
            score = None  # This will make them appear grey on the map
        else:
            score = (
                sum((entry["score"] or 0) * entry["population"] for entry in languages)
                / speaker_pop
            )

        countries[country] = {
            "score": score,
            "languages": languages,
        }
    countries = [{"iso2": country, **data} for country, data in countries.items()]
    return pd.DataFrame(countries)
