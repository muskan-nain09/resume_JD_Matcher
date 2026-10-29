"""JobStore tests use an in-memory fake Pinecone client; no API key needed."""

from types import SimpleNamespace
from unittest import mock

import numpy as np

from matcher import job_store
from matcher.job_store import EMBEDDING_DIM, JobStore


class FakeIndex:
    def __init__(self):
        self.vectors = []

    def upsert(self, vectors):
        self.vectors.extend(vectors)

    def query(self, vector, top_k, include_metadata):
        q = np.array(vector)
        scored = sorted(self.vectors, key=lambda v: -float(np.dot(q, v["values"])))
        return SimpleNamespace(matches=[
            SimpleNamespace(score=float(np.dot(q, v["values"])), metadata=v["metadata"])
            for v in scored[:top_k]
        ])


class FakeClient:
    def __init__(self, exists=True):
        self.exists = exists
        self.created = None
        self.index = FakeIndex()

    def has_index(self, name):
        return self.exists

    def create_index(self, **kwargs):
        self.created = kwargs

    def Index(self, name):
        return self.index


def _unit(i):
    v = np.zeros(EMBEDDING_DIM)
    v[i] = 1.0
    return v


def test_is_enabled_depends_on_api_key():
    with mock.patch.dict("os.environ", {"PINECONE_API_KEY": "abc"}):
        assert job_store.is_enabled()
    with mock.patch.dict("os.environ", {"PINECONE_API_KEY": ""}):
        assert not job_store.is_enabled()


def test_creates_index_when_missing():
    client = FakeClient(exists=False)
    JobStore(api_key="x", index_name="jobs", client=client)
    assert client.created["dimension"] == EMBEDDING_DIM
    assert client.created["metric"] == "cosine"


def test_save_and_find_best_matches():
    client = FakeClient()
    store = JobStore(api_key="x", client=client)
    store.save_job("ML Engineer", "Acme", "x" * 5000, _unit(0))
    store.save_job("Chef", "Bistro", "Cook food", _unit(1))

    saved = client.index.vectors[0]["metadata"]
    assert len(saved["jd_preview"]) == job_store.JD_PREVIEW_CHARS

    matches = store.find_matches(_unit(0), top_k=5)
    assert [m.title for m in matches] == ["ML Engineer", "Chef"]
    assert matches[0].score == 1.0
    assert matches[0].company == "Acme"
