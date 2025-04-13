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
from openai import AsyncOpenAI
from requests import HTTPError, get

# for development purposes, all languages will be evaluated on the fast models
# and only a sample of languages will be evaluated on all models
models = [
    "openai/gpt-4o-mini",  # 0.6$/M tokens
    # "anthropic/claude-3.5-haiku", # 4$/M tokens -> too expensive for dev
    "meta-llama/llama-4-maverick",  # 0.6$/M tokens
    "meta-llama/llama-3.3-70b-instruct",  # 0.3$/M tokens
    "meta-llama/llama-3.1-70b-instruct",  # 0.3$/M tokens
    "meta-llama/llama-3-70b-instruct",  # 0.4$/M tokens
    "mistralai/mistral-small-3.1-24b-instruct",  # 0.3$/M tokens
    # "mistralai/mistral-saba", # 0.6$/M tokens
    # "mistralai/mistral-nemo", # 0.08$/M tokens
    "google/gemini-2.0-flash-001",  # 0.4$/M tokens
    # "google/gemini-2.0-flash-lite-001",  # 0.3$/M tokens
    "google/gemma-3-27b-it",  # 0.2$/M tokens
    # "qwen/qwen-turbo", # 0.2$/M tokens; recognizes "inappropriate content"
    "qwen/qwq-32b",  # 0.2$/M tokens
    "deepseek/deepseek-chat-v3-0324",  # 1.1$/M tokens
    # "microsoft/phi-4",  # 0.07$/M tokens; only 16k tokens context
    "microsoft/phi-4-multimodal-instruct",  # 0.1$/M tokens
    "amazon/nova-micro-v1",  # 0.09$/M tokens
    # "openGPT-X/Teuken-7B-instruct-research-v0.4",  # not on OpenRouter
]

transcription_models = [
    "elevenlabs/scribe_v1",
    "openai/whisper-large-v3",
    # "openai/whisper-small",
    # "facebook/seamless-m4t-v2-large",
]

cache = Memory(location=".cache", verbose=0).cache


@cache
def get_popular_models(date: date):
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
    return [model for model, _ in counts]


pop_models = get_popular_models(date.today())
models += [m for m in pop_models if m not in models][:1]

load_dotenv()
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
)

openrouter_rate_limit = AsyncLimiter(max_rate=20, time_period=1)
elevenlabs_rate_limit = AsyncLimiter(max_rate=2, time_period=1)
huggingface_rate_limit = AsyncLimiter(max_rate=5, time_period=1)


@cache
async def complete(**kwargs):
    async with openrouter_rate_limit:
        response = await client.chat.completions.create(**kwargs)
    if not response.choices:
        raise Exception(response)
    return response


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


models = pd.DataFrame(models, columns=["id"]).iloc[:3]


@cache
def get_or_metadata(id):
    # get metadata from OpenRouter
    response = cache(get)("https://openrouter.ai/api/frontend/models/")
    models = response.json()["data"]
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
        "type": "Commercial",
        "license": None,
    }
    if not row:
        return empty
    id = row["hf_slug"] or row["slug"].split(":")[0]
    if not id:
        return empty
    try:
        info = api.model_info(id)
        license = info.card_data.license.replace("-", " ").replace("mit", "MIT").title()
        return {
            "hf_id": info.id,
            "creation_date": info.created_at,
            "size": info.safetensors.total if info.safetensors else None,
            "type": "Open",
            "license": license,
        }
    except HTTPError:
        return empty


or_metadata = models["id"].apply(get_or_metadata)
hf_metadata = or_metadata.apply(get_hf_metadata)


def get_cost(row):
    cost = float(row["endpoint"]["pricing"]["completion"])
    return round(cost * 1_000_000, 2)


exists = or_metadata.apply(lambda x: x is not None)
models, or_metadata, hf_metadata = (
    models[exists],
    or_metadata[exists],
    hf_metadata[exists],
)
creation_date_hf = pd.to_datetime(hf_metadata.str["creation_date"]).dt.date
creation_date_or = pd.to_datetime(
    or_metadata.str["created_at"].str.split("T").str[0]
).dt.date

models = models.assign(
    name=or_metadata.str["short_name"],
    provider_name=or_metadata.str["name"].str.split(": ").str[0],
    cost=or_metadata.apply(get_cost),
    hf_id=hf_metadata.str["hf_id"],
    size=hf_metadata.str["size"],
    type=hf_metadata.str["type"],
    license=hf_metadata.str["license"],
    creation_date=creation_date_hf.combine_first(creation_date_or),
)
