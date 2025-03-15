import re
from datetime import date

import pandas as pd
from joblib.memory import Memory
from langcodes import standardize_tag
from requests import get

cache = Memory(location=".cache", verbose=0).cache


# load CommonVoice stats
@cache  # cache for 1 day
def get_commonvoice_stats(date: date):
    return get("https://commonvoice.mozilla.org/api/v1/stats/languages").json()


commonvoice = pd.DataFrame(get_commonvoice_stats(date.today())).rename(
    columns={"locale": "commonvoice_locale", "validatedHours": "commonvoice_hours"}
)[["commonvoice_locale", "commonvoice_hours"]]
# ignore country (language is language) (in practive this is only relevant to zh-CN/zh-TW/zh-HK)
commonvoice["bcp_47"] = commonvoice["commonvoice_locale"].apply(
    lambda x: re.sub(r"-[A-Z]{2}$", "", x)
)
commonvoice["bcp_47"] = commonvoice["bcp_47"].apply(
    lambda x: standardize_tag(x, macro=True)
)  # this does not really seem to get macrolanguages though, e.g. not for Quechua
commonvoice = (
    commonvoice.groupby("bcp_47")
    .agg({"commonvoice_hours": "sum", "commonvoice_locale": "first"})
    .reset_index()
)
