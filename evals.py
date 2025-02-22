import asyncio
import json
import os
import re
from datetime import date
from os import getenv

import evaluate
import pandas as pd
import requests
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from joblib.memory import Memory
from langcodes import Language, standardize_tag
from language_data.population_data import LANGUAGE_SPEAKING_POPULATION
from openai import AsyncOpenAI
from requests import get
from tqdm.asyncio import tqdm_asyncio
from transformers import NllbTokenizer

# config
models = [
    "openai/gpt-4o-mini",  # 0.6$/M tokens
    # "anthropic/claude-3.5-haiku", # 4$/M tokens -> too expensive
    "meta-llama/llama-3.3-70b-instruct",  # 0.3$/M tokens
    "mistralai/mistral-small-24b-instruct-2501",  # 0.14$/M tokens
    "google/gemini-2.0-flash-001",  # 0.4$/M tokens
    # "qwen/qwen-turbo", # 0.2$/M tokens; recognizes "inappropriate content"
    "deepseek/deepseek-chat",  # 0.9$/M tokens
    "microsoft/phi-4",  # 0.07$/M tokens
]
fast_model = "meta-llama/llama-3.3-70b-instruct"
n_sentences = 30

# setup
load_dotenv()
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
)
cache = Memory(location=".cache", verbose=0).cache
bleu = evaluate.load("bleu")
# bertscore = evaluate.load("bertscore")
tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
rate_limit = AsyncLimiter(max_rate=20, time_period=1)


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

# load script codes and names
scripts = pd.read_csv("data/ScriptCodes.csv").rename(
    columns={"Code": "iso15924", "English Name": "script_name"}
)


def population(bcp_47):
    items = {
        re.sub(r"^[a-z]+-", "", lang): pop
        for lang, pop in LANGUAGE_SPEAKING_POPULATION.items()
        if re.match(rf"^{bcp_47}-[A-Z]{{2}}$", lang)
    }
    return items


def script_name(iso15924):
    return scripts[scripts["iso15924"] == iso15924]["script_name"].values[0]


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


# load benchmark languages and scripts
benchmark_dir = "data/floresp-v2.0-rc.3/dev"
benchmark_languages = pd.DataFrame(
    [f.split(".")[1] for f in os.listdir(benchmark_dir)],
    columns=["flores_path"],
)
benchmark_languages["bcp_47"] = benchmark_languages["flores_path"].apply(
    lambda x: standardize_tag(x, macro=True),
)
# ignore script (language is language)
benchmark_languages["bcp_47"] = benchmark_languages["bcp_47"].apply(
    lambda x: re.sub(r"-[A-Z][a-z]+$", "", x)
)
benchmark_languages = (
    benchmark_languages.groupby("bcp_47")
    .agg({"flores_path": aggregate_flores_paths})
    .reset_index()
)


# load CommonVoice stats
@cache  # cache for 1 day
def get_commonvoice_stats(date: date):
    return get("https://commonvoice.mozilla.org/api/v1/stats/languages").json()


commonvoice_stats = pd.DataFrame(get_commonvoice_stats(date.today())).rename(
    columns={"locale": "commonvoice_locale", "validatedHours": "commonvoice_hours"}
)[["commonvoice_locale", "commonvoice_hours"]]
# ignore country (language is language) (in practive this is only relevant to zh-CN/zh-TW/zh-HK)
commonvoice_stats["bcp_47"] = commonvoice_stats["commonvoice_locale"].apply(
    lambda x: re.sub(r"-[A-Z]{2}$", "", x)
)
commonvoice_stats["bcp_47"] = commonvoice_stats["bcp_47"].apply(
    lambda x: standardize_tag(x, macro=True)
)  # this does not really seem to get macrolanguages though, e.g. not for Quechua
commonvoice_stats = (
    commonvoice_stats.groupby("bcp_47")
    .agg({"commonvoice_hours": "sum", "commonvoice_locale": "first"})
    .reset_index()
)

# merge data
languages = pd.merge(
    languages, benchmark_languages, on="bcp_47", how="left"
)  # "left" because keep it simple for now
languages = pd.merge(
    languages, commonvoice_stats, on="bcp_47", how="left"
)  # "left" because keep it simple for now
languages["in_benchmark"] = languages["bcp_47"].isin(benchmark_languages["bcp_47"])

languages = languages.sort_values(by="speakers", ascending=False)
languages = languages.iloc[:30]

# sample languages to translate to
target_languages = languages[languages["in_benchmark"]].sample(
    n=n_sentences, weights="speakers", replace=True, random_state=42
)
# sample languages to analyze with all models
detailed_languages = languages[languages["in_benchmark"]].sample(n=10, random_state=42)


# utils
def check_rate_limit():
    print(
        requests.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {getenv('OPENROUTER_API_KEY')}"},
        ).json()
    )
    models = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {getenv('OPENROUTER_API_KEY')}"},
    ).json()["data"]
    model = next((m for m in models if m["id"] == "google/gemini-flash-1.5"), None)
    print(model)


@cache
async def complete(**kwargs):
    async with rate_limit:
        response = await client.chat.completions.create(**kwargs)
    if not response.choices:
        raise Exception(response)
    return response


def load_sentences(language):
    return open(f"{benchmark_dir}/dev.{language.flores_path}").readlines()


@cache
async def translate_and_evaluate(model, original_language_bcp_47, sentence_nr):
    original_language = languages[languages["bcp_47"] == original_language_bcp_47].iloc[
        0
    ]
    target_language = target_languages.iloc[sentence_nr]
    original_sentence = load_sentences(original_language)[sentence_nr].strip()
    target_sentence = load_sentences(target_language)[sentence_nr].strip()
    script = script_name(target_language.flores_path.split("_")[1])
    reply = await complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"Translate the following text to the {target_language.language_name} language; use the {script} script; reply only with the translation:\n\n{original_sentence}",
            }
        ],
        temperature=0,
        max_tokens=1024,
    )
    prediction = reply.choices[0].message.content.strip()
    score = bleu.compute(
        predictions=[prediction],
        references=[target_sentence],
        tokenizer=tokenizer.tokenize,
    )
    return {
        "model": model,
        "bcp_47": original_language["bcp_47"],
        "bleu": score["bleu"],
        "sentence_nr": sentence_nr,
    }


def mean(lst):
    return sum(lst) / len(lst) if lst else 0


# evaluation!
async def main():
    scores = [
        translate_and_evaluate(model, original_language.bcp_47, i)
        for i in range(n_sentences)
        for original_language in languages.itertuples()
        for model in models
        if original_language.in_benchmark
        and (
            model == fast_model
            or original_language.bcp_47 in detailed_languages.bcp_47.values
        )
    ]
    scores = await tqdm_asyncio.gather(*scores, miniters=1)
    results = []
    for language in languages.itertuples():
        results_for_language = []
        for model in models:
            results_for_model = [
                score
                for score in scores
                if score["bcp_47"] == language.bcp_47 and score["model"] == model
            ]
            if results_for_model:
                bleu = mean([s["bleu"] for s in results_for_model])
                results_for_language.append(
                    {
                        "model": model,
                        "bleu": bleu,
                    }
                )
        if results_for_language:
            results.append(
                {
                    "language_name": language.language_name,
                    "bcp_47": language.bcp_47,
                    "speakers": language.speakers,
                    "scores": results_for_language,
                    "bleu": mean([s["bleu"] for s in results_for_language]),
                    "commonvoice_hours": language.commonvoice_hours
                    if not pd.isna(language.commonvoice_hours)
                    else None,
                    "commonvoice_locale": language.commonvoice_locale
                    if not pd.isna(language.commonvoice_locale)
                    else None,
                    "population": population(language.bcp_47),
                }
            )
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # check_rate_limit()
    asyncio.run(main())
