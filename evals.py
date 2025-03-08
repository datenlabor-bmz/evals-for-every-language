import asyncio
import json
import os
import random
import re
from datetime import date
from os import getenv

import evaluate
import pandas as pd
import requests
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from joblib.memory import Memory
from langcodes import Language, standardize_tag
from language_data.population_data import LANGUAGE_SPEAKING_POPULATION
from openai import AsyncOpenAI
from pyglottolog import Glottolog
from requests import get
from rich import print
from tqdm.asyncio import tqdm_asyncio
from transformers import NllbTokenizer
from huggingface_hub import InferenceClient

# config
models = [
    "openai/gpt-4o-mini",  # 0.6$/M tokens
    # "anthropic/claude-3.5-haiku", # 4$/M tokens -> too expensive for dev
    "meta-llama/llama-3.3-70b-instruct",  # 0.3$/M tokens
    "mistralai/mistral-small-24b-instruct-2501",  # 0.14$/M tokens
    "google/gemini-2.0-flash-001",  # 0.4$/M tokens
    # "qwen/qwen-turbo", # 0.2$/M tokens; recognizes "inappropriate content"
    # "deepseek/deepseek-chat",  # 0.9$/M tokens
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
chrf = evaluate.load("chrf")
tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
rate_limit = AsyncLimiter(max_rate=20, time_period=1)


@cache
def transcribe(filename, model="elevenlabs/scribe_v1"):
    provider, modelname = model.split("/")
    with open(filename, "rb") as f:
        audio = f.read()
    match provider:
        case "elevenlabs":
            client = ElevenLabs(api_key=getenv("ELEVENLABS_API_KEY"))
            response = client.speech_to_text.convert(model_id=modelname, file=audio)
            return response.text
        case "openai":
            client = InferenceClient(api_key=getenv("HUGGINGFACE_ACCESS_TOKEN"))
            output = client.automatic_speech_recognition(model=model, audio=audio)
            return output.text
        case _:
            raise ValueError(f"Model {model} not supported")


print(transcribe("data/test.m4a", "openai/whisper-large-v3-turbo"))
exit()


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


glottolog = Glottolog("data/glottolog-5.1")


@cache
def language_family(iso_639_3):
    languoid = glottolog.languoid(iso_639_3)
    return languoid.family.name if languoid else None


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

languages = languages.sort_values(by="speakers", ascending=False).iloc[:20]

# sample languages to translate to
target_languages = languages[languages["in_benchmark"]].sample(
    n=n_sentences, weights="speakers", replace=True, random_state=42
)
# sample languages to analyze with all models
detailed_languages = languages[languages["in_benchmark"]].sample(n=1, random_state=42)


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
    bleu_score = bleu.compute(
        predictions=[prediction],
        references=[target_sentence],
        tokenizer=tokenizer.tokenize,
    )
    chrf_score = chrf.compute(predictions=[prediction], references=[target_sentence])
    return {
        "model": model,
        "bcp_47": original_language["bcp_47"],
        "mt_bleu": bleu_score["bleu"],
        "mt_chrf": chrf_score["score"],
        "sentence_nr": sentence_nr,
    }


metadata = pd.read_csv("data/floresp-v2.0-rc.3/metadata_dev.tsv", sep="\t")


@cache
async def classify_and_evaluate(model, language_bcp_47, nr):
    language = languages[languages["bcp_47"] == language_bcp_47].iloc[0]
    sentences = pd.DataFrame(load_sentences(language), columns=["text"])
    sentences = pd.concat([metadata, sentences], axis=1)
    sentences = sentences.dropna(subset=["topic"])
    sentences["topic"] = sentences["topic"].str.lower()
    paragraphs = (
        sentences.groupby("URL").agg({"text": " ".join, "topic": "first"}).reset_index()
    )
    top_topics = paragraphs.value_counts("topic").head(5).index
    paragraphs = paragraphs[paragraphs["topic"].isin(top_topics)]
    examples = pd.concat(
        [
            paragraphs[paragraphs["topic"] == t].sample(n=5, random_state=42)
            for t in top_topics
        ]
    ).sample(frac=1, random_state=42)
    test_paragraphs = paragraphs[~paragraphs["URL"].isin(examples["URL"])].sample(
        frac=1, random_state=42
    )
    test_paragraph = test_paragraphs.iloc[nr]

    def topic_to_number(topic):
        return top_topics.get_loc(topic)

    messages = []
    for example in examples.itertuples():
        messages += [
            {"role": "user", "content": example.text},
            {"role": "assistant", "content": str(topic_to_number(example.topic))},
        ]
    reply = await complete(
        model=model,
        messages=[
            *messages,
            {
                "role": "user",
                "content": test_paragraph.text,
            },
        ],
        temperature=0,
        max_tokens=5,
    )
    try:
        prediction = int(reply.choices[0].message.content.strip())
    except ValueError:
        prediction = -1
    return {
        "model": model,
        "bcp_47": language["bcp_47"],
        "true": topic_to_number(test_paragraph.topic),
        "pred": prediction,
        "sentence_nr": nr,
    }


def corrupt_sentence(sentence):
    # replace 5% of the sentence with <mask>
    mask_length = round(len(sentence) * 0.05)
    start = random.randint(0, len(sentence) - mask_length)
    end = start + mask_length
    return sentence[:start] + "<mask>" + sentence[end:]


@cache
async def mlm_and_evaluate(model, language_bcp_47, nr):
    language = languages[languages["bcp_47"] == language_bcp_47].iloc[0]
    sentences = pd.DataFrame(load_sentences(language), columns=["text"])
    sentences["corrupt_text"] = sentences["text"].apply(corrupt_sentence)
    examples = sentences.sample(n=10, random_state=42)
    test_sentences = sentences[~sentences["text"].isin(examples["text"])].sample(
        frac=1, random_state=42
    )
    test_sentence = test_sentences.iloc[nr]
    messages = []
    for example in examples.itertuples():
        messages += [
            {"role": "user", "content": example.corrupt_text},
            {"role": "assistant", "content": example.text},
        ]
    reply = await complete(
        model=model,
        messages=[
            *messages,
            {
                "role": "user",
                "content": test_sentence.corrupt_text,
            },
        ],
        temperature=0,
        max_tokens=1024,
    )
    prediction = reply.choices[0].message.content.strip()
    chrf_score = chrf.compute(predictions=[prediction], references=[test_sentence.text])
    return {
        "model": model,
        "bcp_47": language["bcp_47"],
        "mlm_chrf": chrf_score["score"],
        "sentence_nr": nr,
    }


def mean(lst):
    return sum(lst) / len(lst) if lst else 0


# evaluation!
async def main():
    print("evaluate translation")
    translation_scores = [
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
    translation_scores = await tqdm_asyncio.gather(*translation_scores, miniters=1)
    print("evaluate classification")
    classification_scores = [
        classify_and_evaluate(model, language.bcp_47, i)
        for i in range(n_sentences)
        for language in languages.itertuples()
        for model in models
        if language.in_benchmark
        and (model == fast_model or language.bcp_47 in detailed_languages.bcp_47.values)
    ]
    classification_scores = await tqdm_asyncio.gather(
        *classification_scores, miniters=1
    )
    print("evaluate masked language modeling")
    mlm_scores = [
        mlm_and_evaluate(model, language.bcp_47, i)
        for i in range(n_sentences)
        for language in languages.itertuples()
        for model in models
        if language.in_benchmark
        and (model == fast_model or language.bcp_47 in detailed_languages.bcp_47.values)
    ]
    mlm_scores = await tqdm_asyncio.gather(*mlm_scores, miniters=1)
    all_results = []
    for language in languages.itertuples():
        results = []
        for model in models:
            translations_for_model = [
                score
                for score in translation_scores
                if score["bcp_47"] == language.bcp_47 and score["model"] == model
            ]
            classifications_for_model = [
                score
                for score in classification_scores
                if score["bcp_47"] == language.bcp_47 and score["model"] == model
            ]
            mlm_for_model = [
                score
                for score in mlm_scores
                if score["bcp_47"] == language.bcp_47 and score["model"] == model
            ]
            mt_bleu = mean([s["mt_bleu"] for s in translations_for_model])
            mt_chrf = mean([s["mt_chrf"] for s in translations_for_model])
            cls_acc = mean([s["true"] == s["pred"] for s in classifications_for_model])
            mlm_chrf = mean([s["mlm_chrf"] for s in mlm_for_model])
            overall_score = (mt_chrf / 100 + cls_acc + mlm_chrf / 100) / 3
            if translations_for_model:
                results.append(
                    {
                        "model": model,
                        "mt_bleu": mt_bleu,
                        "mt_chrf": mt_chrf,
                        "cls_acc": cls_acc,
                        "mlm_chrf": mlm_chrf,
                        "overall_score": overall_score,
                    }
                )
        if results:
            all_results.append(
                {
                    "language_name": language.language_name,
                    "bcp_47": language.bcp_47,
                    "speakers": language.speakers,
                    "scores": results,
                    "mt_bleu": mean([s["mt_bleu"] for s in results]),
                    "mt_chrf": mean([s["mt_chrf"] for s in results]),
                    "cls_acc": mean([s["cls_acc"] for s in results]),
                    "mlm_chrf": mean([s["mlm_chrf"] for s in results]),
                    "overall_score": mean([s["overall_score"] for s in results]),
                    "commonvoice_hours": language.commonvoice_hours
                    if not pd.isna(language.commonvoice_hours)
                    else None,
                    "commonvoice_locale": language.commonvoice_locale
                    if not pd.isna(language.commonvoice_locale)
                    else None,
                    "population": population(language.bcp_47),
                    "language_family": language_family(
                        language.flores_path.split("_")[0]
                    ),
                }
            )
    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # check_rate_limit()
    asyncio.run(main())
