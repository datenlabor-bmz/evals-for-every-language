import random
from functools import partial
from textwrap import dedent

import evaluate
import pandas as pd
import sentencepiece as spm
from datasets_.flores import flores_sentences
from datasets_.mgsm import load_mgsm, parse_number
from datasets_.mmlu import load_mmlu
from datasets_.arc import load_uhura_arc_easy
from datasets_.truthfulqa import load_truthfulqa
from google.cloud import translate_v2 as translate
from langcodes import closest_supported_match
from languages import languages, script_name
from models import complete, transcribe, translate_google

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
            messages=[
                {
                    "role": "user",
                    "content": f"Translate the following text to the {target_language.language_name} language; use the {script} script; reply only with the translation:\n\n{original_sentence}",
                }
            ],
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
            "sentence_nr": sentence_nr,
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
    examples = pd.concat(
        [
            paragraphs[paragraphs["topic"] == t].sample(n=1, random_state=42)
            for t in top_topics
        ]
    ).sample(frac=1, random_state=nr)
    test_paragraphs = paragraphs[~paragraphs["url"].isin(examples["url"])].sample(
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
        pred = await complete(
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
        true = test_paragraph.topic
        others = [t for t in top_topics if t != true]
        acc = (
            int(
                pred.startswith(true)
                or (true in pred and not any(o in pred for o in others))
            )
            if pred
            else 0
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
    D: {item["choices"][3]}
    
    A|B|C|D?"""


async def mmlu_and_evaluate(model, language_bcp_47, nr):
    ds_name, examples, task = load_mmlu(language_bcp_47, nr)
    if not task:
        return []

    messages = []
    for example in examples:
        messages += [
            {"role": "user", "content": format_multiple_choice(example)},
            {"role": "assistant", "content": example["answer"]},
        ]
    messages += [{"role": "user", "content": format_multiple_choice(task)}]
    try:
        response = await complete(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=1,
        )
        if response:
            acc = int(response[:1].strip() == task["answer"])
        else:
            acc = 0
    except Exception as e:
        if "ResponsibleAIPolicyViolation" in str(e):
            acc = 0
        else:
            raise e
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


async def arc_and_evaluate(model, language_bcp_47, nr):
    ds_name, examples, task = load_uhura_arc_easy(language_bcp_47, nr)
    if not task:
        return []

    messages = []
    for example in examples:
        messages += [
            {"role": "user", "content": format_multiple_choice(example)},
            {"role": "assistant", "content": example["answer"]},
        ]
    messages += [{"role": "user", "content": format_multiple_choice(task)}]
    try:
        response = await complete(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=1,
        )
        if response:
            acc = int(response[:1].strip() == task["answer"])
        else:
            acc = 0
    except Exception as e:
        if "ResponsibleAIPolicyViolation" in str(e):
            acc = 0
        else:
            raise e
    return [
        {
            "model": model,
            "bcp_47": language_bcp_47,
            "task": "arc",
            "metric": "accuracy",
            "score": acc,
            "sentence_nr": nr,
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
    text += "|".join(letters[: len(item["choices"])]) + "?"
    return text


async def truthfulqa_and_evaluate(model, language_bcp_47, nr):
    ds_name, examples, task = load_truthfulqa(language_bcp_47, nr)
    if not task:
        return []
    task = shuffle_choices_and_labels(task)
    answer = letters[task["labels"].index(1)]
    messages = []
    for example in examples:
        example = shuffle_choices_and_labels(example)
        messages += [
            {"role": "user", "content": format_multiple_choice_truthfulqa(example)},
            {"role": "assistant", "content": letters[example["labels"].index(1)]},
        ]
    messages += [{"role": "user", "content": format_multiple_choice_truthfulqa(task)}]
    try:
        response = await complete(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=1,
        )
        if response:
            acc = int(response[:1].strip() == answer)
        else:
            acc = 0
    except Exception as e:
        if "ResponsibleAIPolicyViolation" in str(e):
            acc = 0
        else:
            raise e
    return [
        {
            "model": model,
            "bcp_47": language_bcp_47,
            "task": "truthfulqa",
            "metric": "accuracy",
            "score": acc,
            "sentence_nr": nr,
        }
    ]


async def mgsm_and_evaluate(model, language_bcp_47, nr):
    system_prompt = """
    Solve the math problem. Use reasoning, and finally give the answer as a number.
    Response format: <reasoning> #### <number>
    """
    system_prompt = dedent(system_prompt).strip()
    ds_slug, question = load_mgsm(language_bcp_47, nr)
    if not question:
        return []
    response = await complete(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question["question"]},
        ],
        temperature=0,
        max_tokens=1024,
    )
    if response and len(response.split("####")) == 2:
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
            "sentence_nr": nr,
        }
    ]


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


tasks = {
    "translation_from": partial(translate_and_evaluate, mode="from"),
    "translation_to": partial(translate_and_evaluate, mode="to"),
    "classification": classify_and_evaluate,
    # "mlm": mlm_and_evaluate,
    "mmlu": mmlu_and_evaluate,
    "arc": arc_and_evaluate,
    "truthfulqa": truthfulqa_and_evaluate,
    "mgsm": mgsm_and_evaluate,
    # "asr": transcribe_and_evaluate,
}
