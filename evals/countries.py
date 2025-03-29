import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import pandas as pd
from language_data.population_data import LANGUAGE_SPEAKING_POPULATION
from language_data.util import data_filename


def get_population_data():
    filename = data_filename("supplementalData.xml")
    root = ET.fromstring(open(filename).read())
    territories = root.findall("./territoryInfo/territory")

    data = {}
    for territory in territories:
        t_code = territory.attrib["type"]
        t_population = float(territory.attrib["population"])
        data[t_code] = t_population
    return data


def population(bcp_47):
    items = {
        re.sub(r"^[a-z]+-", "", lang): pop
        for lang, pop in LANGUAGE_SPEAKING_POPULATION.items()
        if re.match(rf"^{bcp_47}-[A-Z]{{2}}$", lang)
    }
    return items


def make_country_table(language_table):
    countries = defaultdict(list)
    for lang in language_table.itertuples():
        for country, speaker_pop in population(lang.bcp_47).items():
            countries[country].append(
                {
                    "name": lang.language_name,
                    "bcp_47": lang.bcp_47,
                    "population": speaker_pop,
                    "score": lang.average if not pd.isna(lang.average) else 0,
                }
            )
    for country, languages in countries.items():
        speaker_pop = sum(entry["population"] for entry in languages)
        score = (
            sum(entry["score"] * entry["population"] for entry in languages)
            / speaker_pop
        )
        countries[country] = {
            "score": score,
            "languages": languages,
        }
    countries = [{"iso2": country, **data} for country, data in countries.items()]
    return pd.DataFrame(countries)
