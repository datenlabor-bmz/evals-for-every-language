import re
from datetime import date
from os import getenv
from pathlib import Path

import pandas as pd
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from google.cloud import translate_v2 as translate
from huggingface_hub import AsyncInferenceClient, HfApi
from joblib.memory import Memory
from openai import AsyncOpenAI, BadRequestError
from requests import HTTPError, get

# for development purposes, all languages will be evaluated on the fast models
# and only a sample of languages will be evaluated on all models
important_models = [
    "allenai/olmo-3.1-32b-instruct", 
    "meta-llama/llama-4-maverick",  # 0.6$
    "meta-llama/llama-3.3-70b-instruct",  # 0.3$
    "meta-llama/llama-3.1-70b-instruct",  # 0.3$
    "meta-llama/llama-3-70b-instruct",  # 0.4$
    # "meta-llama/llama-2-70b-chat", # 0.9$; not properly supported by OpenRouter
    "openai/gpt-5.4", # 15$
    # "openai/gpt-5.3", # 15$
    "openai/gpt-5.2",
    "openai/gpt-5.1",
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/gpt-4.1",  # 8$
    "openai/gpt-4o",  # 10$
    "openai/gpt-3.5-turbo", # $1.50
    "openai/gpt-oss-120b",
    "anthropic/claude-opus-4.6",  # 25$
    "anthropic/claude-opus-4.5",  # 25$
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-opus-4.1",  # 15$
    "anthropic/claude-sonnet-4",
    "anthropic/claude-3.7-sonnet",  # 15$
    "anthropic/claude-3.5-sonnet",
    "mistralai/mistral-small-3.2-24b-instruct",  # 0.3$
    "mistralai/mistral-medium-3.1",
    "mistralai/mistral-saba",  # 0.6$
    "mistralai/mistral-nemo",  # 0.08$
    "google/gemini-3.1-pro-preview", #12$
    "google/gemini-3-pro-preview", # 12$
    "google/gemini-2.5-pro", # $10
    "google/gemini-2.5-flash",  # 0.6$
    "google/gemini-2.5-flash-lite",  # 0.3$
    "google/gemma-3-27b-it",  # 0.2$
    # "x-ai/grok-4", # $15
    # "minimax/minimax-m2.5",  # 1,1$; ~53% ok, content=None often
    # "moonshotai/kimi-k2.5",  # privacy filter or content=None 
    "cohere/command-a",
    # "qwen/qwen3-32b",
    # "qwen/qwen3-235b-a22b",
    "qwen/qwen3-30b-a3b",  # 0.29$
    "deepseek/deepseek-v3.2-exp",
    "microsoft/phi-4",  # 0.07$
    "amazon/nova-premier-v1", # 12.5$
    "amazon/nova-pro-v1",  # 0.09$
    "moonshotai/kimi-k2",  # 0.6$
    "baidu/ernie-4.5-300b-a47b",
    # Added 2026-05-19 — new-generation flagships (one per family; auto-discovery handles the rest)
    "openai/gpt-5.5",  # $30/M output; gpt-5.5-pro is $180/M, beyond cap
    "anthropic/claude-opus-4.7",
    "deepseek/deepseek-v4-pro",
    "x-ai/grok-4.20",
    "mistralai/mistral-medium-3.5",
    "moonshotai/kimi-k2.6",
    "google/gemini-3.1-flash-lite",
]

blocklist = [
    "google/gemini-2.5-pro-preview",
    # "google/gemini-2.5-pro",
    "google/gemini-2.5-flash-preview",
    "google/gemini-2.5-flash-lite-preview",
    "google/gemini-2.5-flash-preview-04-17",
    "google/gemini-2.5-flash-preview-05-20",
    "google/gemini-2.5-flash-lite-preview-06-17",
    "google/gemini-2.5-pro-preview-06-05",
    "google/gemini-2.5-pro-preview-05-06",
    "perplexity/sonar-deep-research",
    "perplexity/sonar-reasoning",
    "perplexity/sonar-reasoning-pro",
    "qwen/qwen3-vl-30b-a3b-thinking",
    "alpindale/goliath-120b",
    "z-ai/glm-4.6",  # ~33% ok, content=None often
    "qwen/qwen3-235b-a22b",  # ~60% ok, content=None often
]

# Hard upper bound on per-token output cost. Models above this are dropped
# (validated in get_or_metadata + discover_new_models + load_models filter).
# Raised 2026-05-19 from $25 -> $30 to accommodate GPT-5.5 ($30/M output).
COST_CAP_PER_1M = 30.0

transcription_models = [
    "elevenlabs/scribe_v1",
    "openai/whisper-large-v3",
    # "openai/whisper-small",
    # "facebook/seamless-m4t-v2-large",
]

cache = Memory(location=".cache", verbose=0).cache


@cache
def load_or_metadata(date: date):
    return get("https://openrouter.ai/api/frontend/models").json()["data"]


def get_or_metadata(permaslug):
    models = load_or_metadata(date.today())
    slugs = [
        m
        for m in models
        if (m["permaslug"] == permaslug or m["slug"] == permaslug)
        and m["endpoint"]
        and not m["endpoint"]["is_free"]
        # OpenRouter privacy is provider-specific, so only keep models that
        # have at least one non-free provider that does not train on prompts.
        and m["endpoint"]["provider_info"]["dataPolicy"]["training"] is False
    ]
    if len(slugs) == 0:
        print(f"no appropriate model (not free and privacy-compatible) found for {permaslug}")
    return slugs[0] if len(slugs) >= 1 else None


# Strip numeric version tokens AND date-snapshot suffixes from a slug to
# derive a model "family" key. Size-tier suffixes (-pro, -mini, -flash,
# -lite, -opus, -haiku, ...) stay so flagship and cheap-tier variants form
# separate families.
#
# Examples:
#   openai/gpt-5.5-pro                -> openai/gpt-pro
#   openai/gpt-5.4-mini               -> openai/gpt-mini
#   anthropic/claude-opus-4.7         -> anthropic/claude-opus
#   deepseek/deepseek-v4-pro          -> deepseek/deepseek-pro
#   meta-llama/llama-3.3-70b-instruct -> meta-llama/llama-70b-instruct
#   qwen/qwen3-235b-a22b-07-25        -> qwen/qwen-235b-a22b  (date suffix stripped)
#   bytedance-seed/seed-1.6-20250625  -> bytedance-seed/seed
_DATE_SUFFIX_RE = re.compile(
    r"-(20\d{6}|20\d{2}-\d{2}-\d{2}|\d{2}-\d{2}|\d{4})$"
)
_VERSION_SUFFIX_RE = re.compile(r"[-]?v?\d+(\.\d+)*(-exp|-instruct)?(?=($|-))")


def _family_key(slug: str) -> str:
    # Strip trailing date snapshots first (so version regex can match cleanly).
    while True:
        new = _DATE_SUFFIX_RE.sub("", slug)
        if new == slug:
            break
        slug = new
    return _VERSION_SUFFIX_RE.sub("", slug)


# Providers we trust to ship general-purpose text LLMs. Adding a new vendor
# here is the explicit human gate for auto-discovery.
_DISCOVERY_PROVIDER_ALLOWLIST = frozenset({
    "openai", "anthropic", "google", "meta-llama", "mistralai", "deepseek",
    "x-ai", "qwen", "alibaba", "cohere", "amazon", "moonshotai", "baidu",
    "allenai", "microsoft", "liquid", "ibm-granite", "nvidia", "rekaai",
    "stepfun", "tencent", "z-ai", "bytedance-seed", "ai21", "nousresearch",
    "perplexity", "arcee-ai", "deepcogito", "prime-intellect", "writer",
    "upstage", "openrouter",
})

# Skip these substrings anywhere in the slug — covers transient snapshots,
# non-text modalities, and task-specialised variants.
_DISCOVERY_SKIP_TAGS = (
    "-preview", "-beta", "-experimental", ":free", "-latest",
    "-vision", "-vl", "-image", "-audio", "-tts", "-stt", "-embed",
    "-asr", "-transcribe", "-search", "rerank", "-ocr", "-edit",
    "coder", "codex", "devstral", "codestral",
    "-thinking", "-reasoning", "-think", "-deep-research", "deepresearch",
    "-multi-agent", "safeguard",
)

# Skip these whole product families (named non-text models).
_DISCOVERY_SKIP_PRODUCTS = (
    "whisper", "voxtral", "chirp", "kokoro", "orpheus", "zonos", "csm-",
    "sora", "veo-", "wan-", "seedance", "seedream", "flux.", "imagine",
    "kling", "hailuo", "riverflow", "recraft", "morph-",
    "bge-", "gte-", "e5-", "multilingual-e5",
)


@cache
def discover_new_models(date: date) -> list[str]:
    """Surface OpenRouter models matching inclusion rules; pick the flagship per family.

    Flagship = highest-cost non-blocked variant within a family. If a model's
    flagship gets auto-blocklisted, the next-most-expensive variant takes its
    place on the next call (auto_blocklist is consulted before the dedupe step).
    """
    try:
        catalog = load_or_metadata(date)
    except Exception as e:
        print(f"[discover_new_models] OpenRouter catalog fetch failed: {e}; skipping")
        return []

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365)
    curated_families = {_family_key(s) for s in important_models}
    blocked_families = {_family_key(s) for s in blocklist}
    blocked = set(blocklist) | set(load_auto_blocklist(date))

    candidates = []
    for m in catalog:
        slug = m.get("permaslug") or m.get("slug")
        if not slug or slug in blocked:
            continue
        if slug.startswith("~"):
            continue  # OpenRouter alias slugs like "~anthropic/claude-opus-latest"
        provider = slug.split("/", 1)[0] if "/" in slug else ""
        if provider not in _DISCOVERY_PROVIDER_ALLOWLIST:
            continue
        if any(tag in slug for tag in _DISCOVERY_SKIP_TAGS):
            continue
        slug_lower = slug.lower()
        if any(prod in slug_lower for prod in _DISCOVERY_SKIP_PRODUCTS):
            continue
        if not m.get("endpoint"):
            continue
        if m["endpoint"].get("is_free"):
            continue
        try:
            trains = m["endpoint"]["provider_info"]["dataPolicy"]["training"]
        except (TypeError, KeyError):
            continue
        if trains is not False:
            continue
        try:
            cost_per_1m = float(m["endpoint"]["pricing"]["completion"]) * 1_000_000
        except (TypeError, KeyError, ValueError):
            continue
        if cost_per_1m > COST_CAP_PER_1M:
            continue
        try:
            created = pd.to_datetime(m["created_at"], utc=True)
        except (TypeError, ValueError, KeyError):
            continue
        if created < cutoff:
            continue
        family = _family_key(slug)
        if family in curated_families:
            continue  # already represented in important_models — don't duplicate
        if family in blocked_families:
            continue  # date-suffixed snapshot of a blocklisted slug
        candidates.append((slug, created, family, cost_per_1m))

    # Dedupe: pick flagship per family (highest cost wins; newer wins on ties).
    by_family: dict[str, tuple[str, tuple]] = {}
    for slug, created, family, cost in candidates:
        rank = (-cost, -created.timestamp())
        if family not in by_family or rank < by_family[family][1]:
            by_family[family] = (slug, rank)
    return sorted(s for s, _ in by_family.values())


def get_translation_models():
    return pd.DataFrame(
        [
            {
                "id": "google/translate-v2",
                "name": "Google Translate",
                "provider_name": "Google",
                "cost": 20.0,
                "train_on_prompts": False,  # they don't do it in the API
                "size": None,
                "type": "closed-source",
                "license": None,
                "tasks": ["translation_from", "translation_to"],
            }
        ]
    )


load_dotenv()
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
)

openrouter_rate_limit = AsyncLimiter(max_rate=20, time_period=1)
elevenlabs_rate_limit = AsyncLimiter(max_rate=2, time_period=1)
huggingface_rate_limit = AsyncLimiter(max_rate=5, time_period=1)
google_rate_limit = AsyncLimiter(max_rate=10, time_period=1)


@cache
async def complete(**kwargs) -> str | None:
    async with openrouter_rate_limit:
        try:
            response = await client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            if "filtered" in e.message:
                return None
            raise e
    if not response.choices:
        raise Exception(response)
    return response.choices[0].message.content.strip()


translate_client = translate.Client()


def get_google_supported_languages():
    return [l["language"] for l in translate_client.get_languages()]


@cache
async def translate_google(text, source_language, target_language):
    async with google_rate_limit:
        response = translate_client.translate(
            text, source_language=source_language, target_language=target_language
        )
    return response["translatedText"]


# @cache
# async def transcribe_elevenlabs(path, model):
#     modelname = model.split("/")[-1]
#     client = AsyncElevenLabs(api_key=getenv("ELEVENLABS_API_KEY"))
#     async with elevenlabs_rate_limit:
#         with open(path, "rb") as file:
#             response = await client.speech_to_text.convert(
#                 model_id=modelname, file=file
#             )
#     return response.text


# @cache
# async def transcribe_huggingface(path, model):
#     client = AsyncInferenceClient(api_key=getenv("HUGGINGFACE_ACCESS_TOKEN"))
#     async with huggingface_rate_limit:
#         output = await client.automatic_speech_recognition(model=model, audio=path)
#     return output.text


# async def transcribe(path, model="elevenlabs/scribe_v1"):
#     provider, modelname = model.split("/")
#     match provider:
#         case "elevenlabs":
#             return await transcribe_elevenlabs(path, modelname)
#         case "openai" | "facebook":
#             return await transcribe_huggingface(path, model)
#         case _:
#             raise ValueError(f"Model {model} not supported")


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
        license = ""
        if (
            info.card_data
            and hasattr(info.card_data, "license")
            and info.card_data.license
        ):
            license = (
                info.card_data.license.replace("-", " ").replace("mit", "MIT").title()
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
    try:
        cost = float(row["endpoint"]["pricing"]["completion"])
        return round(cost * 1_000_000, 2)
    except (TypeError, KeyError):
        return None


def get_training_policy(row):
    # get openrouter info whether the provider may train on prompts
    # (this needs to be thoroughly avoided for our benchmark prompts!)
    return row["endpoint"]["provider_info"]["dataPolicy"]["training"]


# Auto-blocklist thresholds: a model is auto-excluded if it has attempted at
# least MIN_ATTEMPTS evaluations and FAIL_PCT_THRESHOLD% or more returned an
# error (content=None / filtered / etc.). Matches the manual blocklist's
# existing quality bar ("~33% ok, content=None often" -> 67% fail -> blocked).
AUTO_BLOCKLIST_MIN_ATTEMPTS = 100
AUTO_BLOCKLIST_FAIL_PCT_THRESHOLD = 50.0


def compute_model_health() -> pd.DataFrame:
    """Per-model success/failure stats from results-detailed. Empty DF on miss."""
    from datasets_.util import load

    detailed = load("results-detailed")
    if detailed.empty or "status" not in detailed.columns:
        return pd.DataFrame(
            columns=["model", "total", "failed", "failed_pct", "score_nonfailed"]
        )
    is_error = (detailed["status"] != "ok").rename("is_error")
    grouped = pd.DataFrame(
        {
            "total": detailed.groupby("model").size(),
            "failed": is_error.groupby(detailed["model"]).sum(),
            "score_nonfailed": detailed[~is_error].groupby("model")["score"].mean(),
        }
    ).reset_index()
    grouped["failed_pct"] = grouped["failed"] / grouped["total"] * 100
    return grouped.sort_values("failed_pct", ascending=False)


@cache
def load_auto_blocklist(date: date) -> list[str]:
    """Models past the failure threshold in observed history. Empty on first run."""
    try:
        health = compute_model_health()
    except Exception as e:
        print(f"[auto_blocklist] failed to load history: {e}; using empty list")
        return []
    if health.empty:
        return []
    bad = health[
        (health["total"] >= AUTO_BLOCKLIST_MIN_ATTEMPTS)
        & (health["failed_pct"] >= AUTO_BLOCKLIST_FAIL_PCT_THRESHOLD)
    ]
    return sorted(bad["model"].tolist())


@cache
def load_models(date: date) -> pd.DataFrame:
    auto_discovered = discover_new_models(date)
    auto_blocked = set(load_auto_blocklist(date))

    # Manual curation wins: important_models override the auto-blocklist
    # (the warning gives a human a nudge to investigate the quality regression).
    override = set(important_models) & auto_blocked
    if override:
        print(
            f"[load_models] important_models override auto_blocklist (kept anyway): "
            f"{sorted(override)}"
        )
    if auto_blocked - override:
        print(
            f"[load_models] auto_blocklist excluding: "
            f"{sorted(auto_blocked - override)}"
        )
    if auto_discovered:
        print(f"[load_models] auto_discovered added: {auto_discovered}")

    all_model_candidates = (
        (set(important_models) | (set(auto_discovered) - auto_blocked))
        - set(blocklist)
    )

    # Snapshot health stats for inspection (small enough to track in git).
    try:
        health = compute_model_health()
        if not health.empty:
            Path("results").mkdir(exist_ok=True)
            health.to_json(
                "results/model_health.json",
                orient="records",
                indent=2,
                force_ascii=False,
            )
    except Exception as e:
        print(f"[load_models] failed to snapshot model_health.json: {e}")

    # Validate models exist on OpenRouter before including them
    valid_models = []

    for model_id in all_model_candidates:
        metadata = get_or_metadata(model_id)
        if metadata is not None:
            valid_models.append(model_id)

    models = pd.DataFrame(sorted(valid_models), columns=["id"])
    or_metadata = models["id"].apply(get_or_metadata)  # TODO this is double-doubled
    hf_metadata = or_metadata.apply(get_hf_metadata)
    creation_date_hf = pd.to_datetime(hf_metadata.str["creation_date"]).dt.date
    creation_date_or = pd.to_datetime(
        or_metadata.str["created_at"].str.split("T").str[0]
    ).dt.date

    models = models.assign(
        name=or_metadata.str["short_name"]
        .str.replace(" (free)", "")
        .str.replace(" (self-moderated)", "")
        .str.replace(r"\s*\([^)]*\)\s*$", "", regex=True),
        provider_name=or_metadata.str["name"].str.split(": ").str[0],
        # openrouter_metadata=or_metadata.astype(str),
        cost=or_metadata.apply(get_cost),
        train_on_prompts=or_metadata.apply(get_training_policy),
        hf_id=hf_metadata.str["hf_id"],
        size=hf_metadata.str["size"],
        type=hf_metadata.str["type"],
        license=hf_metadata.str["license"],
        creation_date=creation_date_hf.combine_first(creation_date_or),
    )
    models.to_json(
        "models_unfiltered.json", orient="records", indent=2, force_ascii=False
    )
    # Filter out expensive models to keep costs reasonable.
    # Log any manually-curated entries that get dropped here so the user knows why.
    too_expensive = models[models["cost"] > COST_CAP_PER_1M]
    important_dropped = too_expensive[too_expensive["id"].isin(important_models)]
    for _, row in important_dropped.iterrows():
        print(
            f"[load_models] dropping {row['id']} from cohort: "
            f"cost ${row['cost']}/M > cap ${COST_CAP_PER_1M}/M"
        )
    models = models[models["cost"] <= COST_CAP_PER_1M].reset_index(drop=True)
    models["tasks"] = [
        [
            "translation_from",
            "translation_to",
            "classification",
            "mmlu",
            "arc",
            "truthfulqa",
            "mgsm",
        ]
    ] * len(models)
    models = pd.concat([models, get_translation_models()])
    return models


models = load_models(date.today())
