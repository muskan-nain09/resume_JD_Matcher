"""LLM provider abstraction and resume-bullet rewrite suggestions.

The rest of the app only calls `provider.generate(prompt) -> str`, so swapping
Ollama for Hugging Face (or adding another provider later) never touches the
UI or the prompt logic. We call both APIs with plain `requests` instead of an
SDK or LangChain: it's two HTTP endpoints, and fewer dependencies keeps the
project easy to deploy and explain.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol

import requests

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
# Hugging Face's router exposes an OpenAI-compatible chat endpoint for
# models hosted by its inference providers.
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

# Low temperature: we want faithful edits of the user's bullets, not creativity.
TEMPERATURE = 0.3
MAX_TOKENS = 800
TIMEOUT_SECONDS = 120
MAX_BULLETS = 5
MAX_JD_CHARS = 4000  # keeps the prompt within small models' context windows


class LLMError(Exception):
    """Raised with a user-friendly message when the LLM can't be used."""


class LLMProvider(Protocol):
    name: str

    def generate(self, prompt: str) -> str: ...


class OllamaProvider:
    """Local models via Ollama's HTTP API. Free and private: the resume never
    leaves the user's machine."""

    name = "Ollama"

    def __init__(self, url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_OLLAMA_MODEL):
        self.url = url.rstrip("/")
        self.model = model

    def generate(self, prompt: str) -> str:
        try:
            resp = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": TEMPERATURE},
                },
                timeout=TIMEOUT_SECONDS,
            )
        except requests.ConnectionError as exc:
            raise LLMError(
                f"Couldn't reach Ollama at {self.url}. Is it running? (`ollama serve`)"
            ) from exc
        except requests.Timeout as exc:
            raise LLMError("Ollama took too long to respond.") from exc
        if resp.status_code == 404:
            raise LLMError(
                f"Ollama model '{self.model}' isn't installed. Run `ollama pull {self.model}`."
            )
        if not resp.ok:
            raise LLMError(f"Ollama returned an error ({resp.status_code}).")
        return resp.json().get("response", "")


class HuggingFaceProvider:
    """Hosted models via the Hugging Face Inference API, for deployments
    where running Ollama isn't possible (e.g. Streamlit Community Cloud)."""

    name = "Hugging Face"

    def __init__(self, token: str, model: str = DEFAULT_HF_MODEL):
        if not token:
            raise LLMError("HF_API_TOKEN is not set, so Hugging Face can't be used.")
        self.token = token
        self.model = model

    def generate(self, prompt: str) -> str:
        try:
            resp = requests.post(
                HF_CHAT_URL,
                headers={"Authorization": f"Bearer {self.token}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                },
                timeout=TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise LLMError("Couldn't reach the Hugging Face Inference API.") from exc
        if resp.status_code in (401, 403):
            raise LLMError("Hugging Face rejected the API token. Check HF_API_TOKEN.")
        if not resp.ok:
            raise LLMError(f"Hugging Face returned an error ({resp.status_code}).")
        try:
            return resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError("Unexpected response from Hugging Face.") from exc


def get_provider() -> LLMProvider:
    """Choose a provider from the LLM_PROVIDER env variable (default: ollama)."""
    choice = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if choice == "huggingface":
        return HuggingFaceProvider(
            token=os.getenv("HF_API_TOKEN", ""),
            model=os.getenv("HF_MODEL", DEFAULT_HF_MODEL),
        )
    if choice == "ollama":
        return OllamaProvider(
            url=os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL),
            model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        )
    raise LLMError(f"Unknown LLM_PROVIDER '{choice}'. Use 'ollama' or 'huggingface'.")


def generate(prompt: str) -> str:
    """Convenience wrapper: generate text with the configured provider."""
    return get_provider().generate(prompt)


# ---------- rewrite suggestions ----------

@dataclass
class Rewrite:
    original: str
    suggestion: str | None  # None if the model skipped this bullet


def build_rewrite_prompt(jd_text: str, missing_keywords: list[str], bullets: list[str]) -> str:
    """Build the rewrite prompt.

    The rules against inventing numbers and experience are the most important
    part: a resume with fabricated claims is worse than a weak one. We ask
    for `[X%]`-style placeholders so the user fills in real figures.
    A numbered output format is easy for small models to follow and easy for
    us to parse reliably.
    """
    numbered = "\n".join(f"{i}. {b}" for i, b in enumerate(bullets, start=1))
    keywords = ", ".join(missing_keywords) if missing_keywords else "(none)"
    return f"""You are an expert resume coach. Rewrite each resume bullet point below so it is a stronger match for the job description.

Rules:
- Start each bullet with a strong action verb (e.g. Built, Led, Automated, Reduced).
- Include relevant missing keywords ONLY where they truthfully fit what the original bullet describes. Never add a skill or tool the bullet doesn't support.
- NEVER invent numbers, metrics, or experience. If a metric would help, insert a placeholder such as [X%], [N users], or [$X] for the candidate to fill in.
- Keep each bullet to one sentence, under 30 words.

Job description:
\"\"\"
{jd_text[:MAX_JD_CHARS]}
\"\"\"

Missing keywords: {keywords}

Resume bullets:
{numbered}

Respond with exactly {len(bullets)} lines, one per bullet, in the same order, formatted as "<number>. <rewritten bullet>". Do not add any other text."""


_NUMBERED_LINE = re.compile(r"^\s*\**\s*(\d+)\s*[.):]\s*(.+?)\s*$")


def parse_rewrites(response: str, originals: list[str]) -> list[Rewrite]:
    """Match numbered lines in the response back to the original bullets.

    Models sometimes add an intro line, wrap bullets in quotes, or use bold
    markdown, so we only trust lines that start with a valid number.
    """
    suggestions: dict[int, str] = {}
    for line in response.splitlines():
        m = _NUMBERED_LINE.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        text = m.group(2).strip().strip("*").strip().strip('"').strip()
        if 1 <= idx <= len(originals) and text and idx not in suggestions:
            suggestions[idx] = text
    return [Rewrite(original=b, suggestion=suggestions.get(i)) for i, b in enumerate(originals, 1)]


def suggest_rewrites(
    provider: LLMProvider, jd_text: str, missing_keywords: list[str], bullets: list[str]
) -> list[Rewrite]:
    """Ask the LLM to rewrite up to MAX_BULLETS bullets. Raises LLMError."""
    bullets = bullets[:MAX_BULLETS]
    if not bullets:
        return []
    prompt = build_rewrite_prompt(jd_text, missing_keywords, bullets)
    return parse_rewrites(provider.generate(prompt), bullets)
