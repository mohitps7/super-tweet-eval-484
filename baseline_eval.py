import argparse
import os
from typing import List, Dict

import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM

from tqdm.auto import tqdm

def load_data(dataset_name: str, config_name: str):
    ds = load_dataset(dataset_name, config_name)
    if "test" not in ds:
        raise ValueError(f"No test split for {dataset_name} config {config_name}")
    return ds["test"]


def run_inference(
    texts: List[str],
    model_name: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    batch_size: int = 32,
):
    model.to(device)
    model.eval()
    preds = []

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Inferencing {model_name}"):
        batch_texts = texts[i: i + batch_size]
        enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    return preds


def run_topic_generation(
    texts: List[str],
    model_name: str,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 32,
) -> List[List[str]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    gen_pipe  = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device.type == "cuda" else -1,
    )
    outs = gen_pipe(texts, max_length=max_length, batch_size=batch_size)
    return [
        [lbl.strip().lower() for lbl in out["generated_text"].split(",") if lbl.strip()]
        for out in outs
    ]


def run_sentiment_hate_generation(
    texts: List[str],
    model_name: str,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 32,
) -> List[str]:
    pipe = pipeline(
        "text2text-generation",
        model=model_name,
        tokenizer=model_name,
        device=0 if device.type == "cuda" else -1,
    )
    outs = pipe(texts, max_length=max_length, batch_size=batch_size)
    return [out["generated_text"].strip().lower() for out in outs]

def compute_single_metrics(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                                     zero_division=0)
    _, _, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)

    labels = [
        "strongly negative",
        "negative",
        "negative or neutral",
        "positive",
        "strongly positive"
    ]

    report_dict = classification_report(
        y_true, y_pred, zero_division=0
    )

    print(report_dict)

    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
    }

def compute_hate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    labels = [
        "hate_gender", "hate_race", "hate_sexuality", "hate_religion",
        "hate_origin", "hate_disability", "hate_age", "not_hate"
    ]
    macro_f1 = f1_score(y_true, y_pred, labels=list(range(len(labels))), average="macro", zero_division=0)

    y_true_bin = [0 if y == 7 else 1 for y in y_true]  # 7 is not_hate
    y_pred_bin = [0 if y == 7 else 1 for y in y_pred]
    micro_f1 = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)

    combined = (macro_f1 + micro_f1) / 2.0

    return {
        "f1_macro": macro_f1,
        "f1_micro": micro_f1,
        "f1_combined": combined,
    }

TOPIC_LABELS = [
        "arts_&_culture", "business_&_entrepreneurs", "celebrity_&_pop_culture",
        "diaries_&_daily_life", "family", "fashion_&_style", "film_tv_& video",
        "fitness_&_health", "food_&_dining", "gaming", "learning_&_educational",
        "music", "news_&_social_concern", "other_hobbies", "relationships",
        "science_&_technology", "sports", "travel_&_adventure", "youth_&_student_life"
]

def evaluate_topic_multi(pred_lists, split="test"):
    ds = load_dataset("cardiffnlp/super_tweeteval", "tweet_topic", split=split)
    gold_labels = ds["gold_label_list"]

    label_names = TOPIC_LABELS

    predictions = []
    for labs in pred_lists:
        vec = [0] * len(label_names)
        for lbl in labs:
            if lbl in label_names:
                vec[label_names.index(lbl)] = 1
        predictions.append(vec)

    f1m = f1_score(gold_labels, predictions, average="macro", zero_division=0)
    return {"f1_macro": f1m}


def fuzzy_match_sentiment(pred: str, label_set: List[str]) -> int:
    pred = pred.strip().lower()
    for i, label in enumerate(label_set):
        if label in pred:
            return i
    return 2  # default to 'negative or neutral'

def main():
    parser = argparse.ArgumentParser(description="Baseline inference for TweetEval tasks")
    parser.add_argument("--tasks", nargs="+",
                        choices=["sentiment", "topic", "hate"],
                        default=["sentiment", "topic", "hate"],
                        )
    parser.add_argument("--model-sentiment", default="cardiffnlp/twitter-roberta-base-topic-sentiment-latest")
    parser.add_argument("--model-topic", default="cardiffnlp/twitter-roberta-base-topic-latest")
    parser.add_argument("--model-hate", default="cardiffnlp/twitter-roberta-base-hate-latest-st")
    parser.add_argument("--save_csv", default=None)
    parser.add_argument("--sentiment-gen-model", default=None)
    parser.add_argument("--topic-gen-model", default=None)
    parser.add_argument("--hate-gen-model", default=None)

    args = parser.parse_args()

    task_configs = {
        "sentiment": {
            "dataset": "cardiffnlp/super_tweeteval",
            "config": "tweet_sentiment",
            "text_key": "text",
            "label_key": "gold_label",
            "model": args.model_sentiment,
        },
        "hate": {
            "dataset": "cardiffnlp/super_tweeteval",
            "config": "tweet_hate",
            "text_key": "text",
            "label_key": "gold_label",
            "model": args.model_hate,
        },
        "topic": {
            "dataset": "cardiffnlp/super_tweeteval",
            "config": "tweet_topic",
            "text_key": "text",
            "label_key": "gold_label_list",
            "model": args.model_topic,
        },
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = []

    topic_pipe = None
    if "topic" in args.tasks:
        topic_pipe = pipeline(
            "text-classification",
            model=args.model_topic,
            tokenizer=args.model_topic,
            function_to_apply="sigmoid",
            return_all_scores=True,
            top_k=None,
            device=0 if torch.cuda.is_available() else -1,
        )

    for task in args.tasks:
        cfg = task_configs[task]
        print(f"\nRunning task: {task}")

        if task in ("sentiment", "hate"):
            test_ds = load_data(cfg["dataset"], cfg["config"])
            texts = [ex[cfg["text_key"]] for ex in test_ds]
            y_true = [ex[cfg["label_key"]] for ex in test_ds]
            tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
            model = AutoModelForSequenceClassification.from_pretrained(cfg["model"])

            if task == "sentiment" and args.sentiment_gen_model:
                sentiment_labels = [
                    "strongly negative", "negative", "negative or neutral",
                    "positive", "strongly positive"
                ]
                label_str = ", ".join(sentiment_labels)

                prompts = [
                    f"Classify the sentiment of this tweet. Choose exactly one of the following labels:\n{label_str}.\n\nTweet: \"{text}\"\nLabel:"
                    for text in texts
                ]

                preds_text = run_sentiment_hate_generation(
                    prompts,
                    args.sentiment_gen_model,
                    device,
                    batch_size=32,
                    max_length=16,
                )
                label2id = {label: i for i, label in enumerate(sentiment_labels)}
                preds = [fuzzy_match_sentiment(pred, sentiment_labels) for pred in preds_text]
            elif task == "hate" and args.hate_gen_model:
                hate_labels = [
                    "hate_gender", "hate_race", "hate_sexuality", "hate_religion",
                    "hate_origin", "hate_disability", "hate_age", "not_hate"
                ]
                label_str = ", ".join(hate_labels)

                prompts = [
                    f"Classify this tweet as ONE of these following hate labels: ({label_str}):\n\nTweet: \"{text}\"\nLabel:"
                    for text in texts
                ]

                preds_text = run_sentiment_hate_generation(
                    prompts,
                    args.hate_gen_model,
                    device,
                    batch_size=32,
                    max_length=32,
                )

                hate_labels = [
                    "hate_gender", "hate_race", "hate_sexuality", "hate_religion",
                    "hate_origin", "hate_disability", "hate_age", "not_hate"
                ]
                label2id = {label: i for i, label in enumerate(hate_labels)}
                preds = [label2id.get(pred.split()[0], 7) for pred in preds_text]
            else:
                preds = run_inference(texts, cfg["model"], tokenizer, model, device, batch_size=32)

            if task == "hate":
                metrics = compute_hate_metrics(y_true, preds)
            else:
                metrics = compute_single_metrics(y_true, preds)
        else:
            ds = load_dataset(
           "cardiffnlp/super_tweeteval", "tweet_topic", split="test"
               )
            texts = ds["text"]

            label_str = ", ".join(TOPIC_LABELS)

            prompts = [
                f"Classify this tweet into one or more topics (choose from: {label_str}):\n\nTweet: \"{text}\"\nLabel (comma-separated):"
                for text in texts
            ]
            if args.topic_gen_model:
                preds_lists = run_topic_generation(
                    prompts,
                    args.topic_gen_model,
                    device,
                    batch_size=32,
                    max_length=32,
                )
            else:
                batch_out = topic_pipe(texts, batch_size=32)
                preds_lists = [
                   [d["label"] for d in single if d["score"] > 0.5]
                   for single in batch_out
                ]
            metrics = evaluate_topic_multi(preds_lists, split="test")

        print(f"Metrics for {task}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        summary.append({"task": task, **metrics})

    df = pd.DataFrame(summary)
    print(df.to_markdown(index=False))

    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f"Saved metrics to {args.save_csv}")


if __name__ == "__main__":
    main()
