import asyncio
import json
import os
import re
from os import getenv

import evaluate
import pandas as pd
import requests
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from joblib.memory import Memory
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from transformers import NllbTokenizer
from datetime import date
from requests import get
from language_data.population_data import LANGUAGE_SPEAKING_POPULATION
from langcodes import standardize_tag, Language

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
bertscore = evaluate.load("bertscore")
tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
rate_limit = AsyncLimiter(max_rate=20, time_period=1)


def reorder(language_name):
    if "," in language_name and "(" not in language_name:
        return language_name.split(",")[1] + " " + language_name.split(",")[0]
    return language_name


# load general language data
languages = {
    lang: pop
    for lang, pop in LANGUAGE_SPEAKING_POPULATION.items()
    if not re.match(r".*-[A-Z]{2}$", lang)
}
languages = pd.DataFrame(list(languages.items()), columns=["bcp_47", "speakers"])
languages["name"] = languages["bcp_47"].apply(lambda x: Language.get(x).display_name())

# load script codes and names
scripts = pd.read_csv("data/ScriptCodes.csv").rename(
    columns={"Code": "iso15924", "English Name": "script_name"}
)


def script_name(iso15924):
    return scripts[scripts["iso15924"] == iso15924]["script_name"].values[0]


# load benchmark languages and scripts
benchmark_dir = "data/floresp-v2.0-rc.3/dev"
benchmark_languages = pd.DataFrame(
    [f.split(".")[1].split("_", 1) for f in os.listdir(benchmark_dir)],
    columns=["iso639_3", "iso15924"],
)
benchmark_languages["bcp_47"] = benchmark_languages.apply(
    lambda row: standardize_tag(row["iso639_3"] + "-" + row["iso15924"], macro=True),
    axis=1,
)
# ignore script (language is language)
benchmark_languages["bcp_47"] = benchmark_languages["bcp_47"].apply(
    lambda x: re.sub(r"-[A-Z][a-z]+$", "", x)
)
benchmark_languages = (
    benchmark_languages.groupby("bcp_47")
    .agg({"iso639_3": "first", "iso15924": "first"})
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
languages = languages.iloc[:10]

# sample languages to translate to
target_languages = languages[languages["in_benchmark"]].sample(
    n=n_sentences, weights="speakers", replace=True, random_state=42
)
# sample languages to analyze with all models
detailed_languages = languages[languages["in_benchmark"]].sample(n=3, random_state=42)


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


async def translate(model, target_language, sentence):
    script = script_name(target_language.iso15924)
    reply = await complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"Translate the following text to the {target_language.name} language; use the {script} script; reply only with the translation:\n\n{sentence}",
            }
        ],
        temperature=0,
        max_tokens=1024,
    )
    return reply.choices[0].message.content


def mean(l):
    return sum(l) / len(l) if l else 0


def load_sentences(language):
    return open(
        f"{benchmark_dir}/dev.{language.iso639_3}_{language.iso15924}"
    ).readlines()


# evaluation!
async def main():
    results = []
    for language in list(languages.itertuples()):
        scores = []
        if language.in_benchmark:
            original_sentences = load_sentences(language)[:n_sentences]
            for model in models:
                if (
                    model != fast_model
                    and language.bcp_47 not in detailed_languages.bcp_47.values
                ):
                    continue
                predictions = [
                    translate(
                        model,
                        language,
                        sentence,
                    )
                    for sentence, language in zip(
                        original_sentences, target_languages.itertuples()
                    )
                ]
                predictions = await tqdm_asyncio.gather(
                    *predictions,
                    miniters=1,
                    desc=f"{language.name} {model.split('/')[0]}",
                )
                target_sentences = [
                    load_sentences(lang)[i]
                    for i, lang in enumerate(target_languages.itertuples())
                ]
                metrics_bleu = bleu.compute(
                    predictions=predictions,
                    references=target_sentences,
                    tokenizer=tokenizer.tokenize,
                )
                # metrics_bert = bertscore.compute(
                #     predictions=predictions,
                #     references=target_sentences,
                #     model_type="distilbert-base-uncased",
                # )
                scores.append(
                    {
                        "model": model,
                        "bleu": metrics_bleu["bleu"],
                        # "bert_score": mean(metrics_bert["f1"]),
                    }
                )
        results.append(
            {
                "language_name": language.name,
                "bcp_47": language.bcp_47,
                "speakers": language.speakers if not pd.isna(language.speakers) else 0,
                "scores": scores,
                "bleu": mean([s["bleu"] for s in scores]) if scores else None,
                # "bert_score": mean([s["bert_score"] for s in scores]),
                "commonvoice_hours": language.commonvoice_hours,
                "commonvoice_locale": language.commonvoice_locale,
            }
        )
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # check_rate_limit()
    asyncio.run(main())
