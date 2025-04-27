import random
from functools import partial

import evaluate
import pandas as pd
import sentencepiece as spm
from datasets_.flores import flores_sentences
from datasets_.mmlu import load_mmlu
from joblib.memory import Memory
from languages import languages, script_name
from models import complete, transcribe

cache = Memory(location=".cache", verbose=0).cache
bleu = evaluate.load("bleu")
chrf = evaluate.load("chrf")
wer = evaluate.load("wer")
tokenizer = spm.SentencePieceProcessor(
    model_file="data/spbleu/flores200_sacrebleu_tokenizer_spm.model"
)

# sample languages to translate to
target_languages = languages[languages["in_benchmark"]].sample(
    frac=1, weights="speakers", replace=True, random_state=42
)


@cache
async def translate_and_evaluate(model, bcp_47, sentence_nr, mode="from"):
    original_language = languages[languages["bcp_47"] == bcp_47].iloc[0]
    target_language = target_languages.iloc[sentence_nr]
    match mode:
        case "from":
            pass
        case "to":
            original_language, target_language = target_language, original_language
    original_sentence = flores_sentences(original_language)[sentence_nr].strip()
    target_sentence = flores_sentences(target_language)[sentence_nr].strip()
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
            "bcp_47": bcp_47,
            "task": f"translation_{mode}",
            "metric": metric,
            "score": score,
            "sentence_nr": sentence_nr,
        }
        for metric, score in (
            ("bleu", bleu_score["bleu"]),
            ("chrf", chrf_score["score"] / 100),
        )
    ]


metadata = pd.read_csv("data/floresp-v2.0-rc.3/metadata_dev.tsv", sep="\t")


@cache
async def classify_and_evaluate(model, bcp_47, nr):
    language = languages[languages["bcp_47"] == bcp_47].iloc[0]
    sentences = pd.DataFrame(flores_sentences(language), columns=["text"])
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
            paragraphs[paragraphs["topic"] == t].sample(n=1, random_state=42)
            for t in top_topics
        ]
    ).sample(frac=1, random_state=nr)
    test_paragraphs = paragraphs[~paragraphs["URL"].isin(examples["URL"])].sample(
        frac=1, random_state=42
    )
    test_paragraph = test_paragraphs.iloc[nr]

    def format_prompt(text):
        return f"{text}\n\nTopic: {'|'.join(top_topics)}?"

    messages = []
    for example in examples.itertuples():
        messages += [
            {"role": "user", "content": format_prompt(example.text)},
            {"role": "assistant", "content": example.topic},
        ]
    # some models have poor tokenization for some languages, and the prompt for this task is relatively long, so it sometimes exceeds the context window
    # this is not just to blame on the context window but mostly on the model's tokenization, so we assign 0 accuracy in this case
    try:
        reply = await complete(
            model=model,
            messages=[
                *messages,
                {
                    "role": "user",
                    "content": format_prompt(test_paragraph.text),
                },
            ],
            temperature=0,
            max_tokens=30,
        )
        response = reply.choices[0].message.content.strip().lower()
        true = test_paragraph.topic
        others = [t for t in top_topics if t != true]
        acc = int(
            response.startswith(true)
            or (true in response and not any(o in response for o in others))
        )
    except Exception as e:
        if "`inputs` tokens + `max_new_tokens` must be <= 4097" in str(e):
            print(f"Max tokens exceeded for {model} in {bcp_47}")
            acc = 0
        else:
            raise e
    return [
        {
            "model": model,
            "bcp_47": bcp_47,
            "task": "classification",
            "metric": "accuracy",
            "score": acc,
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
    sentences = pd.DataFrame(flores_sentences(language), columns=["text"])
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
async def mmlu_and_evaluate(model, language_bcp_47, nr):
    ds_name, examples, task = load_mmlu(language_bcp_47, nr)
    if not task:
        return []

    def format_item(item):
        return f"""{item["question"]}
        
        A: {item["choices"][0]}
        B: {item["choices"][1]}
        C: {item["choices"][2]}
        D: {item["choices"][3]}
        
        A|B|C|D?"""

    messages = []
    for example in examples:
        messages += [
            {"role": "user", "content": format_item(example)},
            {"role": "assistant", "content": example["answer"]},
        ]
    messages += [{"role": "user", "content": format_item(task)}]
    reply = await complete(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1,
    )
    acc = int(reply.choices[0].message.content[:1].strip() == task["answer"])
    return [
        {
            "model": model,
            "bcp_47": language_bcp_47,
            "task": "mmlu",
            "metric": "accuracy",
            "score": acc,
            "sentence_nr": nr,
        }
    ]


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
    partial(translate_and_evaluate, mode="from"),
    partial(translate_and_evaluate, mode="to"),
    classify_and_evaluate,
    # mlm_and_evaluate,
    mmlu_and_evaluate,
    # transcribe_and_evaluate,
]
