import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from google import genai
from google.genai import types as gtypes

from openai import OpenAI, APIError, RateLimitError, Timeout
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import tiktoken
from tqdm import tqdm

from noise_utils import add_noise_to_dataset

logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class AppConfig:
    api_key: str
    provider: str = "openai"
    model: str = "gpt-4.1-nano"  # "gemini-2.0-flash-lite-001"
    max_samples: Optional[int] = 0
    tasks: Tuple[str, ...] = ("topic", "sentiment", "hate")
    throttle: float = 0.0
    output_path: Path = Path("results.json")
    log_level: str = "INFO"
    few_shot: bool = False
    num_few_shot: int = 3
    add_noise: bool = False
    noise_level: str = "medium"

    def __post_init__(self):
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            stream=sys.stdout,
        )


def build_logit_bias(labels: List[str]) -> Dict[str, int]:
    enc = tiktoken.get_encoding("cl100k_base")
    bias: Dict[str, int] = {}
    for lbl in labels:
        tokens = enc.encode(" " + lbl)
        for tid in tokens:
            bias[str(tid)] = 8
    return bias


class Task:
    def __init__(self, name: str, labels: List[str], prompt: str, default: str, max_tokens: int):
        self.name = name
        self.labels = labels
        self.prompt_tpl = prompt
        self.default = default
        self.logit_bias = build_logit_bias(labels)
        self.max_tokens = max_tokens
        self.few_shot_examples: List[Tuple[str, str]] = []

    def set_few_shot_examples(self, examples: List[Tuple[str, str]]):
        self.few_shot_examples = examples

    def make_prompt(self, text: str) -> str:
        shots = ""
        for xtext, xlabel in self.few_shot_examples:
            shots += f"Tweet: \"{xtext}\"\nLabel: {xlabel}\n\n"
        return shots + self.prompt_tpl.format(text=text, labels=", ".join(self.labels))

    def parse(self, resp: Optional[str]) -> str:
        if not resp:
            return self.default
        resp = resp.strip().lower()

        for lbl in self.labels:
            if resp == lbl.lower():
                return lbl.lower()
        for lbl in self.labels:
            if lbl.lower() in resp:
                return lbl.lower()

        # fallback: guess the first label that appears in text
        for word in resp.split():
            word = word.strip(".,:;\"'")
            if word in [label.lower() for label in self.labels]:
                return word

        return self.default


class GPTClient:
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.logger = logging.getLogger("GPTClient")

    def get(self, prompt: str, logit_bias: Dict[str, int], max_tokens: int) -> Optional[str]:
        delay = 1.0
        for i in range(5):
            try:
                rsp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    logit_bias=logit_bias,
                    max_tokens=max_tokens
                )
                return rsp.choices[0].message.content.strip()
            except RateLimitError as e:
                self.logger.warning(f"Rate limit, back off {delay}s: {e}")
            except (APIError, Timeout) as e:
                self.logger.warning(f"API error, back off {delay}s: {e}")
            time.sleep(delay)
            delay = min(delay*1.5, 30.0)
        self.logger.error("All retries failed")
        return None


class GeminiClient:
    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.logger = logging.getLogger("GeminiClient")

    def get(self, prompt: str, logit_bias: Dict[str, int], max_tokens: int) -> Optional[str]:
        delay = 1.0
        for i in range(5):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=gtypes.GenerateContentConfig(
                        temperature=0,
                    ),
                )
                return resp.text.strip()
            except Exception as e:
                self.logger.warning(f"API error, back off {delay}s: {e}")
            time.sleep(delay)
            delay = min(delay*1.5, 30.0)

        self.logger.error("All retries failed")
        return None


class Pipeline:
    def __init__(self, config: AppConfig, gpt_client: GPTClient, tasks: Dict[str, Task]):
        self.cfg = config
        self.client = gpt_client
        self.tasks = tasks
        self.logger = logging.getLogger("Pipeline")
        self.results: Dict[str, Dict[str, List[Any]]] = {
            name: {"true": [], "pred": []} for name in tasks
        }

    def _process_single(self,
                        name: str,
                        task: Task,
                        row: Dict[str, Any],
                        idx: int
                        ) -> Tuple[int, Any, Any, bool]:
        text = row["text"]
        if name == "topic":
            true_idxs = row["gold_label_list"]
            gt = [task.labels[i].lower() for i in true_idxs]
        else:
            lbl_idx = row["gold_label"]
            gt = task.labels[lbl_idx].lower()

        prompt = task.make_prompt(text)
        resp = self.client.get(prompt, task.logit_bias, task.max_tokens)

        if name == "topic":
            pred = [lbl.strip().lower() for lbl in (resp or "").split(",") if lbl.strip()]
        else:
            pred = task.parse(resp)

        if self.cfg.throttle:
            time.sleep(self.cfg.throttle)

        skipped = resp is None
        return idx, gt, pred, skipped

    def run(self):
        for name, task in self.tasks.items():
            ds = load_dataset_for(name, self.cfg.max_samples, self.cfg)
            self.logger.info(f"Running '{name}' on {len(ds)} examples")

            n = len(ds)
            truths = [None] * n
            preds = [None] * n

            if self.cfg.provider == "gemini":
                max_workers = 12
            else:
                max_workers = 6

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._process_single, name, task, row, idx): idx
                    for idx, row in enumerate(ds)
                }

                for fut in tqdm(as_completed(futures), total=n, desc=name.capitalize()):
                    idx, gt, pred, skipped = fut.result()
                    truths[idx] = gt
                    preds[idx] = pred
                    if skipped:
                        self.logger.error(f"#{idx} skipped (no response)")

            self.results[name]["true"] = truths
            self.results[name]["pred"] = preds

        with open(self.cfg.output_path.with_suffix(".json"), "w") as f:
            json.dump(self.results, f, indent=2)
        self.logger.info(f"Output saved to: {self.cfg.output_path}")

    def eval(self):
        all_metrics = []

        for name, task in self.tasks.items():
            self.logger.info(f"{name.upper()}")

            if name == "sentiment":
                y_true = [lbl.lower() for lbl in self.results[name]["true"]]
                y_pred = [lbl.lower() for lbl in self.results[name]["pred"]]
                labels = [lbl.lower() for lbl in task.labels]

                report_dict = classification_report(
                    y_true, y_pred, labels=labels, zero_division=0
                )

                print(report_dict)

                for lbl, met in report_dict.items():
                    if isinstance(met, dict):
                        all_metrics.append({
                            "task": name,
                            "label": lbl,
                            "precision": met["precision"],
                            "recall": met["recall"],
                            "f1_score": met["f1-score"],
                            "support": met["support"]
                        })
                continue

            if name == "hate":
                y_true = [lbl.lower() for lbl in self.results[name]["true"]]
                y_pred = [lbl.lower() for lbl in self.results[name]["pred"]]

                subclasses = [lbl.lower() for lbl in task.labels]
                y_true_idx = [subclasses.index(lbl) for lbl in y_true]
                y_pred_idx = [subclasses.index(lbl) for lbl in y_pred]
                macro_f1 = f1_score(
                    y_true_idx, y_pred_idx,
                    average="macro", zero_division=0
                )

                y_true_bin = [0 if lbl == "not_hate" else 1 for lbl in y_true]
                y_pred_bin = [0 if lbl == "not_hate" else 1 for lbl in y_pred]
                micro_f1 = f1_score(
                    y_true_bin, y_pred_bin,
                    average="micro", zero_division=0
                )

                combined_f1 = (micro_f1 + macro_f1) / 2
                self.logger.info(f"micro-F1 (binary): {micro_f1:.4f}")
                self.logger.info(f"macro-F1 (8-way) : {macro_f1:.4f}")
                self.logger.info(f"combined-F1     : {combined_f1:.4f}")

                all_metrics.append({
                    "task": name,
                    "label": "combined-F1",
                    "precision": "",
                    "recall": "",
                    "f1_score": combined_f1,
                    "support": "",
                })

                continue

            pred_lists = self.results[name]["pred"]

            ds = load_dataset_for("topic", self.cfg.max_samples, self.cfg)
            gold_labels = ds["gold_label_list"]

            label_names = [
                "arts_&_culture", "business_&_entrepreneurs", "celebrity_&_pop_culture",
                "diaries_&_daily_life", "family", "fashion_&_style", "film_tv_& video",
                "fitness_&_health", "food_&_dining", "gaming", "learning_&_educational",
                "music", "news_&_social_concern", "other_hobbies", "relationships",
                "science_&_technology", "sports", "travel_&_adventure", "youth_&_student_life"
            ]

            predictions = []
            for labs in pred_lists:
                vec = [0] * len(label_names)
                for lbl in labs:
                    if lbl in label_names:
                        vec[label_names.index(lbl)] = 1
                predictions.append(vec)

            f1m = f1_score(gold_labels, predictions, average="macro", zero_division=0)
            self.logger.info(f"Macro-F1 : {f1m:.4f}")

            all_metrics.append({
                "task": name,
                "label": "macro-f1",
                "precision": "",
                "recall": "",
                "f1_score": f1m,
                "support": ""
            })

        metrics_df = pd.DataFrame(all_metrics)
        csv_path = self.cfg.output_path.with_suffix(".metrics.csv")
        metrics_df.to_csv(csv_path, index=False)
        self.logger.info(f"Metrics saved to: {csv_path}")


def get_few_shot_examples(task_name: str, labels: List[str], num: int) -> List[Tuple[str, str]]:
    if num <= 0:
        return []

    config_map = {
        "sentiment": "tweet_sentiment",
        "hate": "tweet_hate",
        "topic": "tweet_topic",
    }
    label_field = "gold_label" if task_name != "topic" else "gold_label_list"
    text_field = "text"

    ds = load_dataset("cardiffnlp/super_tweeteval", config_map[task_name], split="train")
    ds = ds.shuffle(seed=42)
    examples = []

    for ex in ds:
        text = ex[text_field]
        if task_name == "topic":
            label_idxs = ex[label_field]
            labels_str = ", ".join([labels[i] for i in label_idxs])
        else:
            label_idx = ex[label_field]
            labels_str = labels[label_idx]

        examples.append((text, labels_str))
        if len(examples) >= num:
            break

    return examples


def build_tasks(config: AppConfig) -> Dict[str, Task]:
    topics = [
        "arts_&_culture", "business_&_entrepreneurs", "celebrity_&_pop_culture",
        "diaries_&_daily_life", "family", "fashion_&_style", "film_tv_& video",
        "fitness_&_health", "food_&_dining", "gaming", "learning_&_educational",
        "music", "news_&_social_concern", "other_hobbies", "relationships",
        "science_&_technology", "sports", "travel_&_adventure", "youth_&_student_life"
    ]
    sentiments = [
        "strongly negative",
        "negative",
        "negative or neutral",
        "positive",
        "strongly positive"
    ]

    hates = [
        "hate_gender", "hate_race", "hate_sexuality", "hate_religion",
        "hate_origin", "hate_disability", "hate_age", "not_hate"
    ]

    zero_shot_topic = (
        "Classify this tweet into one or more topics (choose from {labels}):\n\nTweet: \"{text}\"\n"
        "Label (comma-separated):"
    )

    zero_shot_hate = (
        "Classify this tweet as ONE of these following hate labels: ({labels}):\n\nTweet: \"{text}\"\nLabel:"
    )

    zero_shot_sentiment = (
        "Classify the sentiment of this tweet. Choose exactly one of the following labels:\n"
        "{labels}.\n\nTweet: \"{text}\"\n\nLabel:"
    )

    topic_task = Task(
            "topic", topics,
            zero_shot_topic,
            default=topics[0],
            max_tokens=15 if config.few_shot else 10
        )
    topic_task.set_few_shot_examples(get_few_shot_examples("topic", topics, config.num_few_shot))

    sentiment_task = Task(
            "sentiment", sentiments,
            zero_shot_sentiment,
            default="negative or neutral",
            max_tokens=7 if config.few_shot else 5
        )
    sentiment_task.set_few_shot_examples(get_few_shot_examples("sentiment", sentiments, config.num_few_shot))

    hate_task = Task(
        "hate", hates,
        zero_shot_hate,
        default="not_hate",
        max_tokens=1 if config.few_shot else 1
    )
    hate_task.set_few_shot_examples(get_few_shot_examples("hate", hates, config.num_few_shot))

    return {
        "topic": topic_task,
        "sentiment": sentiment_task,
        "hate": hate_task,
    }


def load_dataset_for(task_name: str, max_samples: Optional[int], config: Optional[AppConfig] = None):
    dataset_map = {
        "sentiment": "tweet_sentiment",
        "topic": "tweet_topic",
        "hate": "tweet_hate",
    }

    if task_name not in dataset_map:
        raise ValueError(f"Unknown task: {task_name}")

    ds = load_dataset("cardiffnlp/super_tweeteval", dataset_map[task_name], split="test")

    if config and config.add_noise:
        print("Adding noise:", config.noise_level)
        ds = add_noise_to_dataset(ds, level=config.noise_level)

    if max_samples > 0:
        ds = ds.select(range(min(len(ds), max_samples)))
    return ds


def parse_args() -> AppConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["gemini", "openai"], default="openai")
    p.add_argument("--api-key", required=True)
    p.add_argument("--model", default="gpt-4.1-nano")  # default="gemini-2.0-flash-lite-001")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--throttle", type=float, default=0.0)
    p.add_argument("--output-path", type=Path, default=Path("results.json"))
    p.add_argument(
        "--tasks", nargs="+",
        choices=["topic", "sentiment", "hate"],
        default=["topic", "sentiment", "hate"],
    )
    p.add_argument("--num-few-shot", type=int, default=3)
    p.add_argument("--few-shot", action="store_true", help="Use few-shot prompting")
    p.add_argument("--add-noise", action="store_true")
    p.add_argument("--noise-level", choices=["light", "medium", "heavy"], default="medium")

    args = p.parse_args()

    if not args.few_shot:
        args.num_few_shot = 0

    print(args)

    return AppConfig(
        api_key=args.api_key,
        provider=args.provider,
        model=args.model,
        max_samples=args.max_samples,
        throttle=args.throttle,
        output_path=args.output_path,
        tasks=tuple(args.tasks),
        few_shot=args.few_shot,
        num_few_shot=args.num_few_shot,
        add_noise=args.add_noise,
        noise_level=args.noise_level,
    )


if __name__ == "__main__":
    cfg = parse_args()

    if cfg.provider == "gemini":
        client = GeminiClient(cfg.api_key, cfg.model)
    else:
        client = GPTClient(cfg.api_key, cfg.model)

    all_tasks = build_tasks(cfg)
    selected = {n: all_tasks[n] for n in cfg.tasks}
    pipeline = Pipeline(cfg, client, selected)
    pipeline.run()
    pipeline.eval()
