"""Semantic similarity with sentence-transformers.

Keyword matching alone misses meaning ("built REST APIs" vs "backend web
services"), so we also compare the resume and JD as dense embeddings.
`all-MiniLM-L6-v2` is small (~90 MB), fast on CPU, and good enough for
document-level similarity, which keeps the app runnable on a laptop or a free
hosting tier.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

# The model truncates input after 256 word-pieces, so anything past that is
# silently ignored. ~150 words stays safely under the limit.
CHUNK_WORDS = 150


def load_model() -> SentenceTransformer:
    """Load the embedding model. The app wraps this in st.cache_resource so
    the model is only loaded once per server process, not on every rerun."""
    return SentenceTransformer(MODEL_NAME)


def chunk_text(text: str, max_words: int = CHUNK_WORDS) -> list[str]:
    """Split text into chunks of at most `max_words` words."""
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def embed_text(model: SentenceTransformer, text: str) -> np.ndarray:
    """Embed a long document as the average of its chunk embeddings.

    Averaging lets every part of a two-page resume contribute instead of only
    the first 256 tokens. We re-normalize afterwards so the result is a unit
    vector and cosine similarity is just a dot product.
    """
    chunks = chunk_text(text) or [""]
    vectors = model.encode(chunks, normalize_embeddings=True)
    mean = vectors.mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm else mean


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, clipped to 0..1.

    Raw cosine can be slightly negative, but for scoring "unrelated" and
    "opposite" mean the same thing, so we floor it at 0.
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))


def text_similarity(model: SentenceTransformer, text_a: str, text_b: str) -> float:
    """Semantic similarity between two documents, from 0 to 1."""
    return cosine_similarity(embed_text(model, text_a), embed_text(model, text_b))


def weakest_bullets(
    model: SentenceTransformer, bullets: list[str], jd_text: str, n: int = 5
) -> list[str]:
    """Return the `n` bullets least similar to the JD, weakest first.

    These are the bullets that benefit most from a rewrite, and limiting the
    count keeps the LLM prompt short and the response fast.
    """
    if not bullets:
        return []
    jd_vec = embed_text(model, jd_text)
    bullet_vecs = model.encode(bullets, normalize_embeddings=True)
    scores = bullet_vecs @ jd_vec
    order = np.argsort(scores)[:n]
    return [bullets[i] for i in order]
