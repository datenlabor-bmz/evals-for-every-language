from langcodes import Language, standardize_tag
import pandas as pd
import os
import re

flores_dir = "data/floresp-v2.0-rc.3/dev"

def flores_sentences(language) -> list[str] | None:
    try:
        return open(f"{flores_dir}/dev.{language.flores_path}").readlines()
    except FileNotFoundError:
        return None

def aggregate_flores_paths(flores_paths):
    # takes a list of paths from the same language but different scripts
    # returns the one with the largest writing population
    if len(flores_paths) == 1:
        return flores_paths.values[0]
    populations = [
        Language.get(standardize_tag(x, macro=True)).writing_population()
        for x in flores_paths.values
    ]
    return flores_paths.values[populations.index(max(populations))]

flores = pd.DataFrame(
    [f.split(".")[1] for f in os.listdir(flores_dir)],
    columns=["flores_path"],
)
flores["bcp_47"] = flores["flores_path"].apply(
    lambda x: standardize_tag(x, macro=True),
)
# ignore script (language is language)
flores["bcp_47"] = flores["bcp_47"].apply(
    lambda x: re.sub(r"-[A-Z][a-z]+$", "", x)
)
flores = (
    flores.groupby("bcp_47")
    .agg({"flores_path": aggregate_flores_paths})
    .reset_index()
)

