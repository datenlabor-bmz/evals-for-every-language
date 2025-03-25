from os import getenv

import pandas as pd
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from elevenlabs import AsyncElevenLabs
from huggingface_hub import AsyncInferenceClient, HfApi
from joblib.memory import Memory
from openai import AsyncOpenAI
from requests import HTTPError

# for development purposes, all languages will be evaluated on the fast models
# and only a sample of languages will be evaluated on all models
models = [
    "openai/gpt-4o-mini",  # 0.6$/M tokens
    # "anthropic/claude-3.5-haiku", # 4$/M tokens -> too expensive for dev
    "meta-llama/llama-3.3-70b-instruct",  # 0.3$/M tokens
    "meta-llama/llama-3.1-70b-instruct",  # 0.3$/M tokens
    "meta-llama/llama-3-70b-instruct", # 0.4$/M tokens
    "mistralai/mistral-small-24b-instruct-2501",  # 0.14$/M tokens
    "mistralai/mistral-nemo",
    "google/gemini-2.0-flash-001",  # 0.4$/M tokens
    "google/gemini-2.0-flash-lite-001",  # 0.3$/M tokens
    "google/gemma-3-27b-it",  # 0.2$/M tokens
    # "qwen/qwen-turbo", # 0.2$/M tokens; recognizes "inappropriate content"
    "qwen/qwq-32b",
    # "deepseek/deepseek-chat",  # 1.3$/M tokens
    # "microsoft/phi-4",  # 0.07$/M tokens; only 16k tokens context
    "microsoft/phi-4-multimodal-instruct",
    "amazon/nova-micro-v1",  # 0.09$/M tokens
    # "openGPT-X/Teuken-7B-instruct-research-v0.4",  # not on OpenRouter
]
model_fast = "meta-llama/llama-3.3-70b-instruct"

transcription_models = [
    "elevenlabs/scribe_v1",
    "openai/whisper-large-v3",
    # "openai/whisper-small",
    # "facebook/seamless-m4t-v2-large",
]
transcription_model_fast = "elevenlabs/scribe_v1"

load_dotenv()
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
)

cache = Memory(location=".cache", verbose=0).cache
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


models = pd.DataFrame(models, columns=["id"])

api = HfApi()

def get_metadata(id):
    try:
        info = api.model_info(id)
        license = info.card_data.license.replace("-", " ").replace("mit", "MIT").title()
        return {
            "hf_id": info.id,
            "creation_date": info.created_at,
            "size": info.safetensors.total,
            "type": "Open",
            "license": license,
        }
    except HTTPError:
        return {
            "hf_id": None,
            "creation_date": None,
            "size": None,
            "type": "Commercial",
            "license": None,
        }

models["hf_id"] = models["id"].apply(get_metadata).str["hf_id"]
models["creation_date"] = models["id"].apply(get_metadata).str["creation_date"]
models["creation_date"] = pd.to_datetime(models["creation_date"])
models["size"] = models["id"].apply(get_metadata).str["size"]
models["type"] = models["id"].apply(get_metadata).str["type"]
models["license"] = models["id"].apply(get_metadata).str["license"]
