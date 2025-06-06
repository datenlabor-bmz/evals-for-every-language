import json
import re
from collections import defaultdict
from datetime import date
from os import getenv

import pandas as pd
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from elevenlabs import AsyncElevenLabs
from huggingface_hub import AsyncInferenceClient, HfApi
from joblib.memory import Memory
from openai import AsyncOpenAI, PermissionDeniedError
from requests import HTTPError, get

# for development purposes, all languages will be evaluated on the fast models
# and only a sample of languages will be evaluated on all models
important_models = [
    "meta-llama/llama-4-maverick",  # 0.6$
    "meta-llama/llama-3.3-70b-instruct",  # 0.3$
    "meta-llama/llama-3.1-70b-instruct",  # 0.3$
    "meta-llama/llama-3-70b-instruct",  # 0.4$
    # "meta-llama/llama-2-70b-chat", # 0.9$; not properly supported by OpenRouter
    # "openai/gpt-4.1",  # 8$
    "openai/gpt-4.1-mini",  # 1.6$
    "openai/gpt-4.1-nano",  # 0.4$
    "openai/gpt-4o-mini",  # 0.6$
    # "openai/gpt-4o-2024-11-20", # 10$
    # "openai/gpt-3.5-turbo-0613",  # 2$
    # "openai/gpt-3.5-turbo",  # 1.5$
    # "anthropic/claude-3.5-haiku", # 4$ -> too expensive for dev
    "mistralai/mistral-small-3.1-24b-instruct",  # 0.3$
    "mistralai/mistral-saba",  # 0.6$
    "mistralai/mistral-nemo",  # 0.08$
    "google/gemini-2.5-flash-preview",  # 0.6$
    "google/gemini-2.0-flash-lite-001",  # 0.3$
    "google/gemma-3-27b-it",  # 0.2$
    # "qwen/qwen-turbo", # 0.2$; recognizes "inappropriate content"
    # "qwen/qwq-32b",  # 0.2$
    # "qwen/qwen-2.5-72b-instruct",  # 0.39$
    # "qwen/qwen-2-72b-instruct",  # 0.9$
    "deepseek/deepseek-chat-v3-0324",  # 1.1$
    "deepseek/deepseek-chat",  # 0.89$
    "microsoft/phi-4",  # 0.07$
    "microsoft/phi-4-multimodal-instruct",  # 0.1$
    "amazon/nova-micro-v1",  # 0.09$
]

blocklist = [
    "microsoft/wizardlm-2-8x22b"  # temporarily rate-limited
]

transcription_models = [
    "elevenlabs/scribe_v1",
    "openai/whisper-large-v3",
    # "openai/whisper-small",
    # "facebook/seamless-m4t-v2-large",
]

cache = Memory(location=".cache", verbose=0).cache


@cache
def get_models(date: date):
    return get("https://openrouter.ai/api/frontend/models").json()["data"]


def get_model(permaslug):
    models = get_models(date.today())
    slugs = [
        m
        for m in models
        if m["permaslug"] == permaslug
        and m["endpoint"]
        and not m["endpoint"]["is_free"]
    ]
    if len(slugs) == 0:
        # the problem is that free models typically have very high rate-limiting
        print(f"no non-free model found for {permaslug}")
    return slugs[0] if len(slugs) >= 1 else None


@cache
def get_historical_popular_models(date: date):
    raw = get("https://openrouter.ai/rankings").text
    data = re.search(r'{\\"data\\":(.*),\\"isPercentage\\"', raw).group(1)
    data = json.loads(data.replace("\\", ""))
    counts = defaultdict(int)
    for day in data:
        for model, count in day["ys"].items():
            if model.startswith("openrouter") or model == "Others":
                continue
            counts[model.split(":")[0]] += count
    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    models = [get_model(model) for model, _ in counts]
    return [m for m in models if m]


@cache
def get_current_popular_models(date: date):
    raw = get("https://openrouter.ai/rankings?view=day").text.replace("\\", "")
    data = re.search(r'"rankingData":(.*),"rankingType":"day"', raw).group(1)
    data = json.loads(data)
    data = sorted(data, key=lambda x: x["total_prompt_tokens"], reverse=True)
    models = [get_model(model["model_permaslug"]) for model in data]
    return [m for m in models if m]


load_dotenv()
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
)

openrouter_rate_limit = AsyncLimiter(max_rate=20, time_period=1)
elevenlabs_rate_limit = AsyncLimiter(max_rate=2, time_period=1)
huggingface_rate_limit = AsyncLimiter(max_rate=5, time_period=1)


@cache
async def complete(**kwargs) -> str | None:
    async with openrouter_rate_limit:
        try:
            response = await client.chat.completions.create(**kwargs)
        except PermissionDeniedError as e:
            print(e)
            return None
    if not response.choices:
        raise Exception(response)
    return response.choices[0].message.content.strip()


@cache
async def transcribe_elevenlabs(path, model):
    modelname = model.split("/")[-1]
    client = AsyncElevenLabs(api_key=getenv("ELEVENLABS_API_KEY"))
    async with elevenlabs_rate_limit:
        with open(path, "rb") as file:
            response = await client.speech_to_text.convert(
                model_id=modelname, file=file
            )
    return response.text


@cache
async def transcribe_huggingface(path, model):
    client = AsyncInferenceClient(api_key=getenv("HUGGINGFACE_ACCESS_TOKEN"))
    async with huggingface_rate_limit:
        output = await client.automatic_speech_recognition(model=model, audio=path)
    return output.text


async def transcribe(path, model="elevenlabs/scribe_v1"):
    provider, modelname = model.split("/")
    match provider:
        case "elevenlabs":
            return await transcribe_elevenlabs(path, modelname)
        case "openai" | "facebook":
            return await transcribe_huggingface(path, model)
        case _:
            raise ValueError(f"Model {model} not supported")


def get_or_metadata(id):
    # get metadata from OpenRouter
    models = get_models(date.today())
    metadata = next((m for m in models if m["slug"] == id), None)
    return metadata


api = HfApi()


@cache
def get_hf_metadata(row):
    # get metadata from the HuggingFace API
    empty = {
        "hf_id": None,
        "creation_date": None,
        "size": None,
        "type": "closed-source",
        "license": None,
    }
    if not row:
        return empty
    id = row["hf_slug"] or row["slug"].split(":")[0]
    if not id:
        return empty
    try:
        info = api.model_info(id)
        license = (
            (info.card_data.license or "")
            .replace("-", " ")
            .replace("mit", "MIT")
            .title()
        )
        return {
            "hf_id": info.id,
            "creation_date": info.created_at,
            "size": info.safetensors.total if info.safetensors else None,
            "type": "open-source",
            "license": license,
        }
    except HTTPError:
        return empty


def get_cost(row):
    cost = float(row["endpoint"]["pricing"]["completion"])
    return round(cost * 1_000_000, 2)


@cache
def load_models(date: date):
    popular_models = (
        get_historical_popular_models(date.today())[:20]
        + get_current_popular_models(date.today())[:10]
    )
    popular_models = [m["slug"] for m in popular_models]
    models = set(important_models + popular_models) - set(blocklist)
    models = pd.DataFrame(sorted(list(models)), columns=["id"])
    or_metadata = models["id"].apply(get_or_metadata)
    hf_metadata = or_metadata.apply(get_hf_metadata)
    creation_date_hf = pd.to_datetime(hf_metadata.str["creation_date"]).dt.date
    creation_date_or = pd.to_datetime(
        or_metadata.str["created_at"].str.split("T").str[0]
    ).dt.date

    models = models.assign(
        name=or_metadata.str["short_name"]
        .str.replace(" (free)", "")
        .str.replace(" (self-moderated)", ""),
        provider_name=or_metadata.str["name"].str.split(": ").str[0],
        cost=or_metadata.apply(get_cost),
        hf_id=hf_metadata.str["hf_id"],
        size=hf_metadata.str["size"],
        type=hf_metadata.str["type"],
        license=hf_metadata.str["license"],
        creation_date=creation_date_hf.combine_first(creation_date_or),
    )
    # models = models[models["cost"] <= 2.0].reset_index(drop=True)
    return models


models = load_models(date.today())
