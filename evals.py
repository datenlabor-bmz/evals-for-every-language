import asyncio
import json
import os
import random
import re
import tarfile
from datetime import date
from os import getenv
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import requests
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from elevenlabs import AsyncElevenLabs
from huggingface_hub import AsyncInferenceClient
from joblib.memory import Memory
from langcodes import Language, standardize_tag
from language_data.population_data import LANGUAGE_SPEAKING_POPULATION
from openai import AsyncOpenAI
from requests import get
from rich import print
from tqdm.asyncio import tqdm_asyncio
from transformers import NllbTokenizer

# ===== config =====

# for development purposes, all languages will be evaluated on the fast models
# and only a sample of languages will be evaluated on all models
models = [
    "openai/gpt-4o-mini",  # 0.6$/M tokens
    # "anthropic/claude-3.5-haiku", # 4$/M tokens -> too expensive for dev
    "meta-llama/llama-3.3-70b-instruct",  # 0.3$/M tokens
    "mistralai/mistral-small-24b-instruct-2501",  # 0.14$/M tokens
    "google/gemini-2.0-flash-001",  # 0.4$/M tokens
    # "qwen/qwen-turbo", # 0.2$/M tokens; recognizes "inappropriate content"
    # "deepseek/deepseek-chat",  # 0.9$/M tokens
    # "microsoft/phi-4",  # 0.07$/M tokens; only 16k tokens context
]
model_fast = "meta-llama/llama-3.3-70b-instruct"
n_languages = 50
n_detailed_languages = 10
n_sentences = 30

transcription_models = [
    "elevenlabs/scribe_v1",
    "openai/whisper-large-v3",
    # "openai/whisper-small",
    # "facebook/seamless-m4t-v2-large",
]
transcription_model_fast = "elevenlabs/scribe_v1"
transcription_n_languages = 10
transcription_n_detailed_languages = 5

# ===== setup =====

load_dotenv()
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
)
cache = Memory(location=".cache", verbose=0).cache
bleu = evaluate.load("bleu")
chrf = evaluate.load("chrf")
wer = evaluate.load("wer")
tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
openrouter_rate_limit = AsyncLimiter(max_rate=20, time_period=1)
elevenlabs_rate_limit = AsyncLimiter(max_rate=2, time_period=1)
huggingface_rate_limit = AsyncLimiter(max_rate=5, time_period=1)

# ===== load metadata =====

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


glottolog = pd.read_csv(
    "data/glottolog_languoid.csv/languoid.csv", na_values=[""], keep_default_na=False
)  # Min _Nan_ Chinese is not N/A!
glottolog["bcp_47"] = glottolog["iso639P3code"].apply(
    lambda x: standardize_tag(x, macro=True) if not pd.isna(x) else None
)


@cache
def language_family(bcp_47):
    languoid = glottolog[glottolog["bcp_47"] == bcp_47].iloc[0]
    if pd.isna(languoid["family_id"]):
        return None
    family = glottolog[glottolog["id"] == languoid["family_id"]].iloc[0]
    return family["name"]


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

fleurs_tags = "af_za,am_et,ar_eg,as_in,ast_es,az_az,be_by,bg_bg,bn_in,bs_ba,ca_es,ceb_ph,ckb_iq,cmn_hans_cn,cs_cz,cy_gb,da_dk,de_de,el_gr,en_us,es_419,et_ee,fa_ir,ff_sn,fi_fi,fil_ph,fr_fr,ga_ie,gl_es,gu_in,ha_ng,he_il,hi_in,hr_hr,hu_hu,hy_am,id_id,ig_ng,is_is,it_it,ja_jp,jv_id,ka_ge,kam_ke,kea_cv,kk_kz,km_kh,kn_in,ko_kr,ky_kg,lb_lu,lg_ug,ln_cd,lo_la,lt_lt,luo_ke,lv_lv,mi_nz,mk_mk,ml_in,mn_mn,mr_in,ms_my,mt_mt,my_mm,nb_no,ne_np,nl_nl,nso_za,ny_mw,oc_fr,om_et,or_in,pa_in,pl_pl,ps_af,pt_br,ro_ro,ru_ru,sd_in,sk_sk,sl_si,sn_zw,so_so,sr_rs,sv_se,sw_ke,ta_in,te_in,tg_tj,th_th,tr_tr,uk_ua,umb_ao,ur_pk,uz_uz,vi_vn,wo_sn,xh_za,yo_ng,yue_hant_hk,zu_za"
fleurs = pd.DataFrame(fleurs_tags.split(","), columns=["fleurs_tag"])
fleurs["bcp_47"] = fleurs["fleurs_tag"].apply(
    lambda x: standardize_tag(x.rsplit("_")[0], macro=True)
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
    languages, fleurs, on="bcp_47", how="left"
)  # "left" because keep it simple for now
languages = pd.merge(
    languages, commonvoice_stats, on="bcp_47", how="left"
)  # "left" because keep it simple for now
languages["in_benchmark"] = languages["bcp_47"].isin(benchmark_languages["bcp_47"])

languages = languages.sort_values(by="speakers", ascending=False)

# sample languages to translate to
target_languages = languages[languages["in_benchmark"]].sample(
    n=n_sentences, weights="speakers", replace=True, random_state=42
)
langs_eval = languages.iloc[:n_languages]
langs_eval_detailed = languages.iloc[:n_detailed_languages]
transcription_langs_eval = languages.iloc[:transcription_n_languages]
transcription_langs_eval_detailed = languages.iloc[:transcription_n_detailed_languages]


def download_file(url, path):
    response = requests.get(url)
    with open(path, "wb") as f:
        f.write(response.content)


def download_fleurs():
    # the huggingface loader does not allow loading only the dev set, so do it manually
    for language in transcription_langs_eval.itertuples():
        tar_url = f"https://huggingface.co/datasets/google/fleurs/resolve/main/data/{language.fleurs_tag}/audio/dev.tar.gz"
        tar_path = Path(f"data/fleurs/{language.fleurs_tag}/audio/dev.tar.gz")
        audio_path = Path(f"data/fleurs/{language.fleurs_tag}/audio")
        if not audio_path.exists():
            print(f"Downloading {tar_url} to {tar_path}")
            tar_path.parent.mkdir(parents=True, exist_ok=True)
            download_file(tar_url, tar_path)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=audio_path)
        tsv_url = f"https://huggingface.co/datasets/google/fleurs/resolve/main/data/{language.fleurs_tag}/dev.tsv"
        tsv_path = Path(f"data/fleurs/{language.fleurs_tag}/dev.tsv")
        if not tsv_path.exists():
            print(f"Downloading {tsv_url} to {tsv_path}")
            tsv_path.parent.mkdir(parents=True, exist_ok=True)
            download_file(tsv_url, tsv_path)


# ===== define tasks and metrics =====


@cache
async def complete(**kwargs):
    async with openrouter_rate_limit:
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
    if prediction.strip():
        bleu_score = bleu.compute(
            predictions=[prediction],
            references=[target_sentence],
            tokenizer=tokenizer.tokenize,
        )
    else:
        bleu_score = {"bleu": 0}
    chrf_score = chrf.compute(predictions=[prediction], references=[target_sentence])
    return [
        {
            "model": model,
            "bcp_47": original_language["bcp_47"],
            "task": "translation",
            "metric": metric,
            "score": score,
            "sentence_nr": sentence_nr,
        }
        for metric, score in zip(
            ["bleu", "chrf"], [bleu_score["bleu"], chrf_score["score"] / 100]
        )
    ]


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
        pred = int(reply.choices[0].message.content.strip())
    except ValueError:
        pred = -1
    true = topic_to_number(test_paragraph.topic)
    return [
        {
            "model": model,
            "bcp_47": language["bcp_47"],
            "task": "classification",
            "metric": "accuracy",
            "score": int(pred == true),
            "sentence_nr": nr,
        }
    ]


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
    return [
        {
            "model": model,
            "bcp_47": language["bcp_47"],
            "task": "language_modeling",
            "metric": "chrf",
            "score": chrf_score["score"] / 100,
            "sentence_nr": nr,
        }
    ]


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


@cache
async def transcribe_and_evaluate(model, language_bcp_47, nr):
    language = languages[languages["bcp_47"] == language_bcp_47].iloc[0]
    fleurs = pd.read_csv(
        f"data/fleurs/{language.fleurs_tag}/dev.tsv",
        sep="\t",
        names=[
            "id",
            "fname",
            "raw_transcription",
            "transcription",
            "words",
            "id2",
            "gender",
        ],
    )
    item = fleurs.iloc[nr]
    path = f"data/fleurs/{language.fleurs_tag}/audio/dev/{item.fname}"
    pred = await transcribe(path, model=model)
    wer_score = wer.compute(predictions=[pred], references=[item.transcription])
    return [
        {
            "model": model,
            "bcp_47": language["bcp_47"],
            "task": "asr",
            "metric": "wer",
            "score": wer_score,
            "sentence_nr": nr,
        }
    ]


tasks = [
    translate_and_evaluate,
    classify_and_evaluate,
    mlm_and_evaluate,
    # transcribe_and_evaluate,
]

# ===== run evaluation and aggregate results =====


def mean(lst):
    return sum(lst) / len(lst) if lst else None


async def main():
    print("running evaluations")
    results = [
        task(model, original_language.bcp_47, i)
        for task in tasks
        for i in range(n_sentences)
        for original_language in langs_eval.itertuples()
        for model in models
        if original_language.in_benchmark
        and (
            model == model_fast
            or original_language.bcp_47 in langs_eval_detailed.bcp_47.values
        )
    ]
    results = await tqdm_asyncio.gather(*results, miniters=1)
    results = pd.DataFrame([r for rs in results for r in rs])
    results = (
        results.groupby(["model", "bcp_47", "task", "metric"]).mean().reset_index()
    )
    lang_results = (
        results.groupby(["bcp_47", "task", "metric"])
        .agg({"score": "mean", "model": "nunique"})
        .reset_index()
    )
    lang_results = pd.merge(languages, lang_results, on="bcp_47", how="outer")
    model_results = (
        results.groupby(["model", "task", "metric"])
        .agg({"score": "mean", "bcp_47": "nunique"})
        .reset_index()
    )
    task_results = (
        results.groupby(["task", "metric"])
        .agg({"score": "mean", "bcp_47": "nunique", "model": "nunique"})
        .reset_index()
    )
    all_results = {
        "tasks": task_results.replace({np.nan:None}).to_dict(orient="records"),
        "models": model_results.replace({np.nan:None}).to_dict(orient="records"),
        "languages": lang_results.replace({np.nan:None}).to_dict(orient="records"),
        "scores": results.replace({np.nan:None}).to_dict(orient="records"),
    }
    with open("frontend/public/results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    download_fleurs()
    asyncio.run(main())
