import io

import pypdfium2 as pdfium
import pytest

from matcher.pdf_parser import (
    NO_TEXT_MESSAGE, PDFParseError, clean_text, extract_bullets, parse_resume,
)

from .conftest import SAMPLE_DIR


def _blank_pdf() -> io.BytesIO:
    """A valid PDF with a page but no text layer, which is what a scanned
    (image-only) resume looks like to a text extractor."""
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(612, 792)
    buf = io.BytesIO()
    pdf.save(buf)
    buf.seek(0)
    return buf


def test_sample_pdf_returns_text_and_bullets():
    resume = parse_resume(str(SAMPLE_DIR / "sample_resume.pdf"))
    assert "Data Scientist" in resume.text
    assert "scikit-learn" in resume.text
    assert len(resume.bullets) >= 5
    # The first bullet wraps onto a second line in the PDF; it must be rejoined.
    assert any(b.endswith("prioritize outreach") for b in resume.bullets)


def test_pdf_without_text_raises_clear_error():
    with pytest.raises(PDFParseError) as exc_info:
        parse_resume(_blank_pdf())
    assert str(exc_info.value) == NO_TEXT_MESSAGE


def test_non_pdf_raises_parse_error():
    with pytest.raises(PDFParseError):
        parse_resume(io.BytesIO(b"definitely not a pdf"))


def test_clean_text_normalizes_whitespace_and_cid_glyphs():
    raw = "Skills:\t  Python   and ﬁne-tuning\n(cid:127) Built a model\n\n\n\nEnd (cid:3)"
    assert clean_text(raw) == "Skills: Python and fine-tuning\n• Built a model\n\nEnd"


def test_extract_bullets_handles_markers_and_wrapping():
    text = "\n".join([
        "Experience",
        "• Built a churn model that helped the",
        "retention team prioritize outreach",
        "- Wrote SQL queries for reporting",
        "* Python",                      # too short to be a real bullet
        "Education",
    ])
    assert extract_bullets(text) == [
        "Built a churn model that helped the retention team prioritize outreach",
        "Wrote SQL queries for reporting",
    ]
