"""Text preprocessing and similarity utilities."""

from __future__ import annotations

import math
import re
import string

def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase terms."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = re.split(r"\s+", text.strip())
    return [t for t in tokens if t]


def dense_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two dense vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(v * v for v in vec_a))
    mag_b = math.sqrt(sum(v * v for v in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)
