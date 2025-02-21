import asyncio
import json
import os
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

# config
models = [
    "openai/gpt-4o-mini", # 0.6$/M tokens
    # "anthropic/claude-3.5-haiku", # 4$/M tokens -> too expensive
    "meta-llama/llama-3.3-70b-instruct", # 0.3$/M tokens
    "mistralai/mistral-small-24b-instruct-2501", # 0.14$/M tokens
    "google/gemini-2.0-flash-001", # 0.4$/M tokens
    # "qwen/qwen-turbo", # 0.2$/M tokens; recognizes "inappropriate content"
    "deepseek/deepseek-chat", # 0.9$/M tokens
    "microsoft/phi-4", # 0.07$/M tokens
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

# load benchmark languages and scripts
benchmark_dir = "data/floresp-v2.0-rc.3/dev"
benchmark_languages = pd.DataFrame(
    [f.split(".")[1].split("_", 1) for f in os.listdir(benchmark_dir)],
    columns=["language_code", "script_code"],
)
# hack: drop additional script codes for languages with multiple scripts
benchmark_languages = benchmark_languages.groupby("language_code").head(1)
benchmark_languages["in_benchmark"] = True

# load Ethnologue language names
language_names = (
    pd.read_csv("data/LanguageCodes.tab", sep="\t")
    .rename(columns={"LangID": "language_code", "Name": "language_name"})[
        ["language_code", "language_name"]
    ]
    .assign(language_name=lambda df: df["language_name"].apply(reorder).str.strip())
)

# load Wikidata speaker stats
language_stats = (
    pd.read_csv("data/languages.tsv", sep="\t")
    .rename(columns={"iso639_3": "language_code", "maxSpeakers": "speakers"})[
        ["language_code", "speakers", "iso639_1"]
    ]
    .dropna(subset=["language_code"])
)
language_stats["speakers"] = pd.to_numeric(language_stats["speakers"], errors="coerce")
ignored_languages = [
    "zho",  # Chinese -> use Mandarin (cmn) instead
    "ara",  # Arabic -> use Standard Arabic (arb) instead
    "pus",  # Pashto -> use Nothern / Central / Southern Pashto instead (pbt / pst / pbu)
    "fas",  # Persian -> use Iranian Persian (pes) instead
    "msa",  # Malay -> use Indonesian (ind) instead
]
language_stats = language_stats[
    ~language_stats["language_code"].isin(ignored_languages)
]

# load unicode script names
script_names = pd.read_csv("data/ScriptCodes.csv").rename(
    columns={"Code": "script_code", "English Name": "script_name"}
)[["script_code", "script_name"]]

# merge data
languages = pd.merge(language_stats, language_names, on="language_code", how="outer")
languages = pd.merge(benchmark_languages, languages, on="language_code", how="outer")
languages = pd.merge(languages, script_names, on="script_code", how="left")
languages["in_benchmark"] = languages["in_benchmark"].fillna(False)
languages = languages.sort_values(by="speakers", ascending=False)
languages = languages.iloc[:30]

# retrieve CommonVoice stats
@cache # cache for 1 day
def get_commonvoice_stats(date: date):
    return get("https://commonvoice.mozilla.org/api/v1/stats/languages").json()

commonvoice_stats = pd.DataFrame(get_commonvoice_stats(date.today()))

# sample languages to translate to
target_languages = languages[languages["in_benchmark"]].sample(
    n=n_sentences, weights="speakers", replace=True, random_state=42
)
# sample languages to analyze with all models
detailed_languages = languages[languages["in_benchmark"]].sample(
    n=10, random_state=42
)


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


@cache
async def translate(model, target_language, target_script, sentence):
    reply = await complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"Translate the following text to the {target_language} language; use the {target_script} script; reply only with the translation:\n\n{sentence}",
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
        f"{benchmark_dir}/dev.{language.language_code}_{language.script_code}"
    ).readlines()


# evaluation!
async def main():
    results = []
    for language in list(languages.itertuples()):
        name = (
            language.language_name
            if not pd.isna(language.language_name)
            else language.language_code
        )
        print(name)
        scores = []
        if language.in_benchmark:
            original_sentences = load_sentences(language)[:n_sentences]
            for model in models:
                if (
                    model != fast_model
                    and language.language_code
                    not in detailed_languages.language_code.values
                ):
                    continue
                
                print(model)
                predictions = [
                    translate(
                        model, language.language_name, language.script_name, sentence
                    )
                    for sentence, language in zip(original_sentences, target_languages.itertuples())
                ]
                predictions = await tqdm_asyncio.gather(*predictions, miniters=1)
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
        commonvoice_hours = commonvoice_stats[commonvoice_stats["locale"] == language.iso639_1]["validatedHours"].values
        commonvoice_hours = commonvoice_hours[0] if commonvoice_hours.size > 0 else "N/A"
        results.append(
            {
                "language_name": name,
                "language_code": language.language_code,
                "speakers": language.speakers if not pd.isna(language.speakers) else 0,
                "scores": scores,
                "bleu": mean([s["bleu"] for s in scores]) if scores else None,
                # "bert_score": mean([s["bert_score"] for s in scores]),
                "commonvoice_hours": commonvoice_hours,
            }
        )
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # check_rate_limit()
    asyncio.run(main())
