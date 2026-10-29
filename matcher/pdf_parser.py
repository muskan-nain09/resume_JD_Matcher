"""Extract and clean text from a resume PDF.

We use pdfplumber because it reads the PDF's text layer directly (no OCR).
That is deliberate: applicant tracking systems (ATS) also read the text layer,
so if we can't read a resume, an ATS probably can't either, and the user
should know.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import BinaryIO

import pdfplumber

NO_TEXT_MESSAGE = (
    "This PDF has no readable text. It may be a scanned image, "
    "which most ATS systems also can't read."
)

# Characters that resume templates commonly use to start a bullet point.
BULLET_CHARS = "•●▪■◦‣○►▸➢➤✓✔*-–—"
BULLET_PATTERN = re.compile(rf"^\s*[{re.escape(BULLET_CHARS)}]\s*")
CID_PATTERN = re.compile(r"\(cid:\d+\)")
CID_AT_START = re.compile(r"^\(cid:\d+\)\s*")

# Bullets shorter than this are usually fragments ("- Python") rather than
# achievement statements, so they aren't worth rewriting.
MIN_BULLET_WORDS = 4


class PDFParseError(Exception):
    """Raised when a PDF can't be turned into usable text."""


@dataclass
class ParsedResume:
    text: str
    bullets: list[str] = field(default_factory=list)


def parse_resume(file: str | BinaryIO) -> ParsedResume:
    """Read a PDF (path or file-like object) and return its text and bullets."""
    raw = extract_text(file)
    text = clean_text(raw)
    if not text.strip():
        raise PDFParseError(NO_TEXT_MESSAGE)
    return ParsedResume(text=text, bullets=extract_bullets(text))


def extract_text(file: str | BinaryIO) -> str:
    """Return the raw text of every page, joined by newlines."""
    try:
        with pdfplumber.open(file) as pdf:
            # extract_text() returns None for pages without a text layer.
            pages = [page.extract_text() or "" for page in pdf.pages]
    except Exception as exc:  # pdfplumber raises several different types
        raise PDFParseError(f"Could not read this file as a PDF ({exc}).") from exc
    return "\n".join(pages)


def clean_text(raw: str) -> str:
    """Normalize unicode and whitespace while keeping line breaks.

    Line breaks are kept on purpose: they are how we find bullet points later.
    """
    # NFKC turns ligatures like "ﬁ" into "fi" and fancy spaces into plain ones.
    text = unicodedata.normalize("NFKC", raw)
    # Drop control and private-use characters (icon fonts often produce these),
    # but keep newlines.
    text = "".join(
        ch for ch in text
        if ch == "\n" or unicodedata.category(ch) not in {"Cc", "Cf", "Co", "Cs"}
    )
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    # pdfminer writes "(cid:127)" for glyphs it can't map to unicode. Resume
    # templates hit this most often with custom bullet symbols, so a cid at
    # the start of a line becomes a bullet; anywhere else it is just noise.
    lines = [CID_AT_START.sub("• ", line) for line in lines]
    lines = [CID_PATTERN.sub("", line).strip() for line in lines]
    # Collapse runs of blank lines into one.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def extract_bullets(text: str) -> list[str]:
    """Split resume text into bullet points.

    PDFs wrap long bullets onto several lines. A line that doesn't start with
    a bullet marker but starts with a lowercase letter is treated as the
    continuation of the previous bullet. It's a simple heuristic, but it
    handles the common case without needing layout analysis.
    """
    bullets: list[str] = []
    in_bullet = False
    for line in text.split("\n"):
        if BULLET_PATTERN.match(line):
            bullets.append(BULLET_PATTERN.sub("", line, count=1))
            in_bullet = True
        elif in_bullet and line and line[0].islower():
            bullets[-1] += " " + line
        else:
            in_bullet = False
    return [b.strip() for b in bullets if len(b.split()) >= MIN_BULLET_WORDS]
