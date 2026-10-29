"""Saved job descriptions in Pinecone (optional feature).

Why a vector database? To find which of many saved jobs best fit a resume,
we need a nearest-neighbour search over embeddings. For a handful of jobs a
Python loop would do, but Pinecone shows how this scales to thousands and
persists between sessions without us running a database.

The feature is only enabled when PINECONE_API_KEY is set, so the core app has
no hard dependency on an external service.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

import numpy as np

EMBEDDING_DIM = 384  # output size of all-MiniLM-L6-v2
DEFAULT_INDEX = "jd-matcher"
# Pinecone metadata is capped at 40 KB per vector. A preview is all the UI needs.
JD_PREVIEW_CHARS = 1000
# Free-tier serverless region.
CLOUD, REGION = "aws", "us-east-1"


@dataclass
class JobMatch:
    title: str
    company: str
    score: float  # cosine similarity, 0..1
    jd_preview: str


def is_enabled() -> bool:
    return bool(os.getenv("PINECONE_API_KEY", "").strip())


class JobStore:
    """Thin wrapper around a Pinecone index of job descriptions."""

    def __init__(self, api_key: str, index_name: str = DEFAULT_INDEX, client=None):
        # `client` lets tests pass a fake instead of calling Pinecone.
        if client is None:
            from pinecone import Pinecone

            client = Pinecone(api_key=api_key)
        self._ensure_index(client, index_name)
        self.index = client.Index(index_name)

    @staticmethod
    def _ensure_index(client, index_name: str) -> None:
        """Create the index on first use, so setup is just "add an API key"."""
        if client.has_index(index_name):
            return
        from pinecone import ServerlessSpec

        # Cosine matches how we compare embeddings everywhere else.
        client.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )

    @classmethod
    def from_env(cls) -> "JobStore":
        return cls(
            api_key=os.environ["PINECONE_API_KEY"],
            index_name=os.getenv("PINECONE_INDEX", DEFAULT_INDEX),
        )

    def save_job(self, title: str, company: str, jd_text: str, embedding: np.ndarray) -> str:
        """Store one JD embedding with its metadata. Returns the new id."""
        job_id = str(uuid.uuid4())
        self.index.upsert(vectors=[{
            "id": job_id,
            "values": [float(x) for x in embedding],
            "metadata": {
                "title": title,
                "company": company,
                "jd_preview": jd_text[:JD_PREVIEW_CHARS],
            },
        }])
        return job_id

    def find_matches(self, resume_embedding: np.ndarray, top_k: int = 5) -> list[JobMatch]:
        """Return the `top_k` saved jobs most similar to the resume."""
        result = self.index.query(
            vector=[float(x) for x in resume_embedding],
            top_k=top_k,
            include_metadata=True,
        )
        matches = []
        for m in result.matches:
            meta = m.metadata or {}
            matches.append(JobMatch(
                title=meta.get("title", "Untitled"),
                company=meta.get("company", ""),
                score=max(0.0, min(1.0, float(m.score))),
                jd_preview=meta.get("jd_preview", ""),
            ))
        return matches
