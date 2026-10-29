"""Shared fixtures. spaCy and the skills list are loaded once per test session
because loading spaCy takes a second or two."""

from pathlib import Path

import pytest

from matcher import keywords

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "sample_data"


@pytest.fixture(scope="session")
def skills():
    return keywords.load_skills()


@pytest.fixture(scope="session")
def nlp():
    return keywords.load_nlp()
