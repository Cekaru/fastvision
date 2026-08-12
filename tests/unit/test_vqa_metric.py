"""Answer normalization and scoring functions."""

import pytest

from benchmarks.datasets import exact_match, normalize_answer, vqa_accuracy


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("The Dog!", "dog"),
        ("  two ", "2"),
        ("dont", "don't"),
        ("A cat.", "cat"),
        ("1.5", "1.5"),  # decimals keep their period
        ("red, white and blue", "red white and blue"),
    ],
)
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw) == expected


def test_vqa_accuracy_soft():
    answers = ["dog"] * 3 + ["cat"] * 7
    assert vqa_accuracy("dog", answers) == 1.0  # 3 matches / 3 caps at 1
    assert vqa_accuracy("Cat", answers) == 1.0
    assert vqa_accuracy("bird", answers) == 0.0


def test_vqa_accuracy_partial():
    answers = ["dog"] * 2 + ["cat"] * 8
    assert vqa_accuracy("the dog", answers) == pytest.approx(2 / 3)


def test_exact_match():
    assert exact_match("Yes.", ["yes"]) == 1.0
    assert exact_match("no", ["yes"]) == 0.0
