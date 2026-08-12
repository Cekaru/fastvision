"""Benchmark tasks (VQAv2 / GQA / TextVQA / POPE) and their metrics.

Each task yields ``Record`` items and declares which scoring function
applies. Hugging Face ``datasets`` is only imported when a real task is
loaded; the ``synthetic`` task needs no downloads and exists so the whole
harness can be smoke-tested on CPU.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Records and tasks
# ---------------------------------------------------------------------------


@dataclass
class Record:
    image: Any  # PIL.Image.Image
    question: str
    answers: list[str]  # ground-truth answers (1 for exact-match tasks, 10 for VQA)


@dataclass
class Task:
    name: str
    records: list[Record]
    metric: Callable[[str, list[str]], float]
    prompt_suffix: str = field(
        default="\nAnswer the question using a single word or phrase."
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "im": "i'm", "ive": "i've", "isnt": "isn't", "itd": "it'd", "itll": "it'll",
    "lets": "let's", "maam": "ma'am", "mightve": "might've", "mustve": "must've",
    "shant": "shan't", "shed": "she'd", "shes": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "somebodyd": "somebody'd", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someonell": "someone'll",
    "someones": "someone's", "somethingd": "something'd", "somethingll": "something'll",
    "thats": "that's", "thered": "there'd", "therere": "there're", "theres": "there's",
    "theyd": "they'd", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
    "twas": "'twas", "wasnt": "wasn't", "wed": "we'd", "weve": "we've",
    "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's",
    "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd", "wholl": "who'll", "whos": "who's",
    "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "yall": "y'all",
    "youd": "you'd", "youll": "you'll", "youre": "you're", "youve": "you've",
}
_NUMBER_WORDS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[;/\[\]\"{}()=+\\_\-><@`?,!]")
_PERIOD = re.compile(r"(?<!\d)\.(?!\d)")  # strip periods except decimals


def normalize_answer(text: str) -> str:
    """Official VQA answer normalization (punctuation, articles, numbers)."""
    text = text.strip().lower().replace("\n", " ").replace("\t", " ")
    text = _PUNCT.sub("", text)
    text = _PERIOD.sub("", text)
    words = []
    for word in text.split():
        word = _NUMBER_WORDS.get(word, word)
        if word in _ARTICLES:
            continue
        words.append(_CONTRACTIONS.get(word, word))
    return " ".join(words)


def vqa_accuracy(prediction: str, answers: list[str]) -> float:
    """Official VQA soft accuracy: min(#humans that said it / 3, 1)."""
    pred = normalize_answer(prediction)
    gts = [normalize_answer(a) for a in answers]
    matches = sum(g == pred for g in gts)
    return min(matches / 3.0, 1.0)


def exact_match(prediction: str, answers: list[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(normalize_answer(a) == pred for a in answers))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_hf(path: str, split: str, limit: int | None, **kwargs):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "benchmark tasks need the 'datasets' package: pip install fastvision[bench]"
        ) from exc
    # With a row cap, stream so only ``limit`` rows are fetched instead of
    # downloading the whole split first (VQAv2 validation is multiple GB —
    # the difference between a minutes-long and an hours-long run on Colab).
    if limit is not None:
        from itertools import islice

        try:
            stream = load_dataset(path, split=split, streaming=True, **kwargs)
            rows = list(islice(stream, limit))
            if rows:
                return rows
        except Exception:  # pragma: no cover - source may not support streaming
            pass  # fall back to a full (non-streaming) download below
    ds = load_dataset(path, split=split, **kwargs)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def _vqa_answers(raw) -> list[str]:
    # lmms-lab VQAv2 stores answers as list[{"answer": ...}]; textvqa as list[str]
    if raw and isinstance(raw[0], dict):
        return [a["answer"] for a in raw]
    return list(raw)


def load_vqav2(split: str = "validation", limit: int | None = None) -> Task:
    ds = _load_hf("lmms-lab/VQAv2", split, limit)
    records = [
        Record(row["image"], row["question"], _vqa_answers(row["answers"]))
        for row in ds
    ]
    return Task("vqav2", records, vqa_accuracy)


def load_textvqa(split: str = "validation", limit: int | None = None) -> Task:
    ds = _load_hf("lmms-lab/textvqa", split, limit)
    records = [
        Record(row["image"], row["question"], _vqa_answers(row["answers"]))
        for row in ds
    ]
    return Task("textvqa", records, vqa_accuracy)


def load_gqa(split: str = "testdev", limit: int | None = None) -> Task:
    # GQA ships questions and images as separate configs keyed by image id.
    questions = _load_hf("lmms-lab/GQA", split, limit, name=f"{split}_balanced_instructions")
    images = _load_hf("lmms-lab/GQA", split, None, name=f"{split}_balanced_images")
    by_id = {row["id"]: row["image"] for row in images}
    records = [
        Record(by_id[row["imageId"]], row["question"], [row["answer"]])
        for row in questions
    ]
    return Task("gqa", records, exact_match)


def load_pope(split: str = "test", limit: int | None = None) -> Task:
    ds = _load_hf("lmms-lab/POPE", split, limit)
    records = [Record(row["image"], row["question"], [row["answer"]]) for row in ds]
    return Task(
        "pope", records, exact_match,
        prompt_suffix="\nAnswer the question using a single word or phrase.",
    )


def load_synthetic(split: str = "test", limit: int | None = 8) -> Task:
    """Random noise images + fixed answers; exercises the harness offline."""
    import numpy as np

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError("synthetic task needs Pillow: pip install pillow") from exc

    rng = np.random.default_rng(0)
    records = []
    for i in range(limit or 8):
        arr = rng.integers(0, 255, size=(336, 336, 3), dtype=np.uint8)
        records.append(
            Record(Image.fromarray(arr), f"What is in image {i}?", ["noise"])
        )
    return Task("synthetic", records, exact_match)


TASKS: dict[str, Callable[..., Task]] = {
    "vqav2": load_vqav2,
    "textvqa": load_textvqa,
    "gqa": load_gqa,
    "pope": load_pope,
    "synthetic": load_synthetic,
}


def load_task(name: str, split: str | None = None, limit: int | None = None) -> Task:
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; available: {sorted(TASKS)}")
    kwargs: dict = {"limit": limit}
    if split is not None:
        kwargs["split"] = split
    return TASKS[name](**kwargs)
