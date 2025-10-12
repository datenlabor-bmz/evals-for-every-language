import random
from functools import partial

import evaluate
import pandas as pd
import sentencepiece as spm
from datasets_.arc import load_uhura_arc_easy
from datasets_.flores import flores_sentences
from datasets_.mgsm import load_mgsm, parse_number
from datasets_.mmlu import load_mmlu
from datasets_.truthfulqa import load_truthfulqa
from google.cloud import translate_v2 as translate
from langcodes import closest_supported_match
from languages import languages, script_name
from models import complete, translate_google

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

translate_client = translate.Client()
supported_languages = [l["language"] for l in translate_client.get_languages()]


async def translate_and_evaluate(model, bcp_47, sentence_nr, mode="from"):
    original_language = languages[languages["bcp_47"] == bcp_47].iloc[0]
    target_language = target_languages.iloc[sentence_nr]
    match mode:
        case "from":
            pass
        case "to":
            original_language, target_language = target_language, original_language
    if (
        flores_sentences(original_language) is None
        or flores_sentences(target_language) is None
    ):
        return []
    original_sentence = flores_sentences(original_language)["text"][sentence_nr].strip()
    target_sentence = flores_sentences(target_language)["text"][sentence_nr].strip()
    script = script_name(target_language.flores_path.split("_")[1])
    translation_prompt = f"Translate the following text to the {target_language.language_name} language; use the {script} script; reply only with the translation:\n\n{original_sentence}"
    if model == "google/translate-v2":
        original_language = closest_supported_match(
            original_language, supported_languages
        )
        target_language = closest_supported_match(target_language, supported_languages)
        if original_language == target_language:
            prediction = original_sentence
        elif original_language is None or target_language is None:
            prediction = None
        else:
            prediction = await translate_google(
                original_sentence, original_language.bcp_47, target_language.bcp_47
            )
    else:
        prediction = await complete(
            model=model,
            messages=[{"role": "user", "content": translation_prompt}],
            temperature=0,
            max_tokens=1024,
        )
    if prediction:
        bleu_score = bleu.compute(
            predictions=[prediction],
            references=[target_sentence],
            tokenizer=tokenizer.tokenize,
        )
        chrf_score = chrf.compute(
            predictions=[prediction], references=[target_sentence]
        )
    else:
        bleu_score = {"bleu": 0}
        chrf_score = {"score": 0}

    return [
        {
            "model": model,
            "bcp_47": bcp_47,
            "task": f"translation_{mode}",
            "metric": metric,
            "score": score,
            "origin": "human",  # FLORES+ is human-translated
            "sentence_nr": sentence_nr,
            "prompt": translation_prompt,
            "response": prediction,
        }
        for metric, score in (
            ("bleu", bleu_score["bleu"]),
            ("chrf", chrf_score["score"] / 100),
        )
    ]


async def classify_and_evaluate(model, bcp_47, nr):
    language = languages[languages["bcp_47"] == bcp_47].iloc[0]
    sentences = flores_sentences(language)
    if sentences is None:
        return []
    sentences = sentences.dropna(subset=["topic"])
    sentences["topic"] = sentences["topic"].str.lower()
    paragraphs = (
        sentences.groupby("url").agg({"text": " ".join, "topic": "first"}).reset_index()
    )
    top_topics = paragraphs.value_counts("topic").head(5).index
    paragraphs = paragraphs[paragraphs["topic"].isin(top_topics)]
    test_paragraph = paragraphs.sample(n=1, random_state=nr).iloc[0]

    prompt = f"""Classify the following text into one of these topics: {", ".join(top_topics)}.
Reply with only the topic name.

Text:
{test_paragraph.text}
"""
    response = await complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=30,
    )

    pred = response.lower().strip() if response else ""
    true = test_paragraph.topic.lower().strip()
    others = [t for t in top_topics if t != true]
    acc = (
        int(
            pred.startswith(true)
            or (true in pred and not any(o in pred for o in others))
        )
        if pred
        else 0
    )

    return [
        {
            "model": model,
            "bcp_47": bcp_47,
            "task": "classification",
            "metric": "accuracy",
            "score": acc,
            "origin": "human",  # FLORES+ is human-translated
            "sentence_nr": nr,
            "prompt": prompt,
            "response": pred,
        }
    ]


def corrupt_sentence(sentence):
    # replace 5% of the sentence with <mask>
    mask_length = round(len(sentence) * 0.05)
    start = random.randint(0, len(sentence) - mask_length)
    end = start + mask_length
    return sentence[:start] + "<mask>" + sentence[end:]


async def mlm_and_evaluate(model, language_bcp_47, nr):
    language = languages[languages["bcp_47"] == language_bcp_47].iloc[0]
    sentences = flores_sentences(language)
    if sentences is None:
        return []
    sentences = pd.DataFrame(sentences, columns=["text"])
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
    prediction = await complete(
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


def format_multiple_choice(item):
    return f"""{item["question"]}
    
    A: {item["choices"][0]}
    B: {item["choices"][1]}
    C: {item["choices"][2]}
    D: {item["choices"][3]}"""


async def mmlu_and_evaluate(model, language_bcp_47, nr):
    ds_name, task, origin = await load_mmlu(language_bcp_47, nr)
    if not task:
        return []
    prompt = f"""Solve the following multiple choice question. Reason step-by-step and then write the final answer as a single letter.

Response format: <reasoning> #### <letter>

---

{format_multiple_choice(task)}"""
    response = await complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1024,
    )
    if response and "####" in response:
        answer = response.split("####")[-1].strip()
        acc = int(answer[:1] == task["answer"])
    else:
        acc = 0

    return [
        {
            "model": model,
            "bcp_47": language_bcp_47,
            "task": "mmlu",
            "metric": "accuracy",
            "score": acc,
            "origin": origin,  # Add origin tag to results
            "sentence_nr": nr,
            "prompt": prompt,
            "response": response,
        }
    ]


async def arc_and_evaluate(model, language_bcp_47, nr):
    ds_name, task, origin = load_uhura_arc_easy(language_bcp_47, nr)
    if not task:
        return []

    prompt = f"""Solve the following multiple choice question. Reason step-by-step and then write the final answer as a single letter.

Response format: <reasoning> #### <letter>

---

{format_multiple_choice(task)}"""
    response = await complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1024,
    )
    if response and "####" in response:
        answer = response.split("####")[-1].strip()
        acc = int(answer[:1] == task["answer"])
    else:
        acc = 0
    return [
        {
            "model": model,
            "bcp_47": language_bcp_47,
            "task": "arc",
            "metric": "accuracy",
            "score": acc,
            "origin": origin,
            "sentence_nr": nr,
            "prompt": prompt,
            "response": response,
        }
    ]


letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def shuffle_choices_and_labels(item):
    indices = list(range(len(item["choices"])))
    random.shuffle(indices)
    item["choices"] = [item["choices"][i] for i in indices]
    item["labels"] = [item["labels"][i] for i in indices]
    return item


def format_multiple_choice_truthfulqa(item):
    text = item["question"] + "\n\n"
    for i, choice in enumerate(item["choices"]):
        text += f"{letters[i]}: {choice}\n"
    return text


async def truthfulqa_and_evaluate(model, language_bcp_47, nr):
    ds_name, task, origin = await load_truthfulqa(language_bcp_47, nr)
    if not task:
        return []

    # Find the correct answer
    correct_choice_index = task["labels"].index(1)
    answer = letters[correct_choice_index]

    prompt = f"""Answer the following multiple choice question. Reason step-by-step and then write the final answer as a single letter.

Response format: <reasoning> #### <letter>

---

{format_multiple_choice_truthfulqa(task)}"""
    response = await complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1024,  # Increased for reasoning
    )
    if response and "####" in response:
        pred_answer = response.split("####")[-1].strip()
        acc = int(pred_answer[:1].upper() == answer)
    else:
        acc = 0

    return [
        {
            "model": model,
            "bcp_47": language_bcp_47,
            "task": "truthfulqa",
            "metric": "accuracy",
            "score": acc,
            "origin": origin,
            "sentence_nr": nr,
            "prompt": prompt,
            "response": response,
        }
    ]


async def mgsm_and_evaluate(model, language_bcp_47, nr):
    ds_slug, question, origin = load_mgsm(language_bcp_47, nr)
    if not question:
        return []

    prompt = f"""Solve the following math problem. Reason step-by-step and then write the final answer as a number.

Response format: <reasoning> #### <number>

---

{question["question"]}"""
    response = await complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1024,
    )
    if response and "####" in response:
        number = response.split("####")[1].strip()
        accuracy = int(parse_number(number) == parse_number(question["answer_number"]))
    else:
        accuracy = 0

    return [
        {
            "model": model,
            "bcp_47": language_bcp_47,
            "task": "mgsm",
            "metric": "accuracy",
            "score": accuracy,
            "origin": origin,
            "sentence_nr": nr,
            "prompt": prompt,
            "response": response,
        }
    ]


# async def transcribe_and_evaluate(model, language_bcp_47, nr):
#     language = languages[languages["bcp_47"] == language_bcp_47].iloc[0]
#     fleurs = pd.read_csv(
#         f"data/fleurs/{language.fleurs_tag}/dev.tsv",
#         sep="\t",
#         names=[
#             "id",
#             "fname",
#             "raw_transcription",
#             "transcription",
#             "words",
#             "id2",
#             "gender",
#         ],
#     )
#     item = fleurs.iloc[nr]
#     path = f"data/fleurs/{language.fleurs_tag}/audio/dev/{item.fname}"
#     pred = await transcribe(path, model=model)
#     wer_score = wer.compute(predictions=[pred], references=[item.transcription])
#     return [
#         {
#             "model": model,
#             "bcp_47": language["bcp_47"],
#             "task": "asr",
#             "metric": "wer",
#             "score": wer_score,
#             "sentence_nr": nr,
#         }
#     ]


tasks = {
    "translation_from": partial(translate_and_evaluate, mode="from"),
    "translation_to": partial(translate_and_evaluate, mode="to"),
    "classification": classify_and_evaluate,
    "mmlu": mmlu_and_evaluate,
    "arc": arc_and_evaluate,
    "truthfulqa": truthfulqa_and_evaluate,
    "mgsm": mgsm_and_evaluate,
}
