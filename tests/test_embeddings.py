"""Embedding helpers tested with a tiny fake model, so tests don't download
the real sentence-transformers model."""

import numpy as np
import pytest

from matcher.embeddings import chunk_text, cosine_similarity, embed_text, weakest_bullets


class FakeModel:
    """'Embeds' text as counts of a few vocabulary words."""

    VOCAB = ["python", "docker", "cooking"]

    def encode(self, texts, normalize_embeddings=True):
        vecs = np.array([[t.lower().count(w) for w in self.VOCAB] for t in texts], dtype=float)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return np.divide(vecs, norms, out=np.zeros_like(vecs), where=norms > 0)


def test_chunk_text_respects_max_words():
    chunks = chunk_text(" ".join(["word"] * 350), max_words=150)
    assert [len(c.split()) for c in chunks] == [150, 150, 50]


def test_cosine_similarity_is_clipped():
    assert cosine_similarity(np.array([1.0, 0]), np.array([-1.0, 0])) == 0.0
    assert cosine_similarity(np.array([1.0, 1]), np.array([2.0, 2])) == pytest.approx(1.0)
    assert cosine_similarity(np.zeros(2), np.array([1.0, 0])) == 0.0


def test_embed_text_returns_unit_vector():
    vec = embed_text(FakeModel(), "python docker " * 200)
    assert np.linalg.norm(vec) == pytest.approx(1.0)


def test_weakest_bullets_orders_least_similar_first():
    bullets = ["Wrote python services", "Cooking for the team", "Python and docker"]
    weakest = weakest_bullets(FakeModel(), bullets, "python docker", n=2)
    assert weakest == ["Cooking for the team", "Wrote python services"]
