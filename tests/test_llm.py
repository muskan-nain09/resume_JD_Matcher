"""LLM tests use fake providers and mocked HTTP; Ollama never needs to run."""

from unittest import mock

import pytest
import requests

from matcher import llm


class FakeProvider:
    name = "fake"

    def __init__(self, response):
        self.response = response
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.response


def test_prompt_contains_rules_keywords_and_bullets():
    prompt = llm.build_rewrite_prompt("JD text", ["Kubernetes", "MLflow"], ["Did A", "Did B"])
    assert "NEVER invent numbers" in prompt
    assert "[X%]" in prompt
    assert "Kubernetes, MLflow" in prompt
    assert "1. Did A\n2. Did B" in prompt


def test_parse_rewrites_tolerates_messy_output():
    response = (
        "Sure! Here are your bullets:\n"
        "1. **Built a model in [X] weeks**\n"
        '2) "Automated reports with Python"\n'
        "4. Out of range, ignored\n"
    )
    rewrites = llm.parse_rewrites(response, ["orig 1", "orig 2", "orig 3"])
    assert [r.suggestion for r in rewrites] == [
        "Built a model in [X] weeks", "Automated reports with Python", None,
    ]
    assert [r.original for r in rewrites] == ["orig 1", "orig 2", "orig 3"]


def test_suggest_rewrites_limits_bullets():
    provider = FakeProvider("1. A\n2. B\n3. C\n4. D\n5. E")
    rewrites = llm.suggest_rewrites(provider, "jd", [], [f"bullet {i}" for i in range(8)])
    assert len(rewrites) == llm.MAX_BULLETS
    assert "6. bullet 5" not in provider.prompts[0]


@pytest.mark.parametrize("env, expected", [
    ({"LLM_PROVIDER": "ollama"}, llm.OllamaProvider),
    ({}, llm.OllamaProvider),
    ({"LLM_PROVIDER": "huggingface", "HF_API_TOKEN": "hf_x"}, llm.HuggingFaceProvider),
])
def test_get_provider_selects_from_env(env, expected):
    with mock.patch.dict("os.environ", env, clear=True):
        assert isinstance(llm.get_provider(), expected)


def test_huggingface_without_token_raises_friendly_error():
    with mock.patch.dict("os.environ", {"LLM_PROVIDER": "huggingface"}, clear=True):
        with pytest.raises(llm.LLMError, match="HF_API_TOKEN"):
            llm.get_provider()


def test_unreachable_ollama_raises_llm_error():
    with mock.patch("requests.post", side_effect=requests.ConnectionError):
        with pytest.raises(llm.LLMError, match="Couldn't reach Ollama"):
            llm.OllamaProvider().generate("hi")


def test_ollama_returns_response_text():
    fake = mock.Mock(ok=True, status_code=200)
    fake.json.return_value = {"response": "1. Better bullet"}
    with mock.patch("requests.post", return_value=fake) as post:
        assert llm.OllamaProvider(model="llama3").generate("hi") == "1. Better bullet"
    assert post.call_args.kwargs["json"]["model"] == "llama3"
