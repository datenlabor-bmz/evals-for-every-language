import random

import evaluate
import pandas as pd
from joblib.memory import Memory
from transformers import NllbTokenizer
from languages import languages, script_name
from datasets_.flores import flores_sentences
from models import complete, transcribe
cache = Memory(location=".cache", verbose=0).cache
bleu = evaluate.load("bleu")
chrf = evaluate.load("chrf")
wer = evaluate.load("wer")
tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")


# sample languages to translate to
target_languages = languages[languages["in_benchmark"]].sample(
    frac=1, weights="speakers", replace=True, random_state=42
)

@cache
async def translate_and_evaluate(model, original_language_bcp_47, sentence_nr):
    original_language = languages[languages["bcp_47"] == original_language_bcp_47].iloc[
        0
    ]
    target_language = target_languages.iloc[sentence_nr]
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