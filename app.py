"""Streamlit UI for the Resume–JD Matcher.

This file only handles layout and state. All the actual logic lives in the
`matcher` package so it can be tested without Streamlit.

Streamlit re-runs this whole script on every interaction, so:
- heavy objects (models) are cached with st.cache_resource, and
- analysis results are kept in st.session_state so they don't disappear
  when the user clicks something else.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from matcher import embeddings, keywords, scoring
from matcher.pdf_parser import ParsedResume, PDFParseError, parse_resume

st.set_page_config(page_title="Resume–JD Matcher", page_icon="📄", layout="wide")

BAND_COLORS = {"green": "green", "amber": "orange", "red": "red"}


# ---------- cached resources (loaded once per server process) ----------

@st.cache_resource(show_spinner="Loading embedding model...")
def get_model():
    return embeddings.load_model()


@st.cache_resource(show_spinner="Loading language model...")
def get_nlp():
    return keywords.load_nlp()


@st.cache_resource
def get_skills():
    return keywords.load_skills()


# ---------- analysis ----------

@dataclass
class Analysis:
    resume: ParsedResume
    jd_text: str
    score: scoring.ScoreBreakdown
    keywords: keywords.KeywordResult


def run_analysis(resume: ParsedResume, jd_text: str) -> Analysis:
    similarity = embeddings.text_similarity(get_model(), resume.text, jd_text)
    kw = keywords.compare_keywords(jd_text, resume.text, get_skills(), get_nlp())
    score = scoring.compute_score(similarity, len(kw.matched_keywords), kw.total)
    return Analysis(resume=resume, jd_text=jd_text, score=score, keywords=kw)


# ---------- rendering helpers ----------

def render_tags(items: list[str], color: str) -> None:
    if items:
        st.markdown(" ".join(f":{color}-badge[{item}]" for item in items))
    else:
        st.caption("None")


def render_score(score: scoring.ScoreBreakdown) -> None:
    color = BAND_COLORS[scoring.score_band(score.final_score)]
    st.markdown(f"## Match score: :{color}[{score.final_score} / 100]")
    st.progress(score.final_score / 100)

    col1, col2 = st.columns(2)
    col1.metric("Semantic similarity", f"{score.similarity:.0%}",
                help="How close the resume's overall meaning is to the JD (embeddings).")
    coverage = "n/a" if score.keyword_coverage is None else f"{score.keyword_coverage:.0%}"
    col2.metric("Keyword coverage", coverage,
                help="Share of JD keywords that also appear in the resume.")
    st.caption(
        f"Score = {scoring.SIMILARITY_WEIGHT:.0%} × similarity + "
        f"{scoring.KEYWORD_WEIGHT:.0%} × keyword coverage"
    )


def render_keywords(kw: keywords.KeywordResult) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"✅ Matched keywords ({len(kw.matched_keywords)})")
        render_tags(kw.matched_keywords, "green")
    with col2:
        st.subheader(f"❌ Missing keywords ({len(kw.missing_keywords)})")
        render_tags(kw.missing_keywords, "red")


# ---------- page ----------

st.title("📄 Resume–JD Matcher")
st.caption("Compare your resume with a job description and see what's missing.")

match_tab, = st.tabs(["Match"])

with match_tab:
    left, right = st.columns(2)
    resume_file = left.file_uploader("Resume (PDF)", type=["pdf"])
    jd_text = right.text_area("Job description", height=200,
                              placeholder="Paste the full job description here...")

    ready = resume_file is not None and jd_text.strip()
    if st.button("Analyze", type="primary", disabled=not ready):
        try:
            resume = parse_resume(resume_file)
        except PDFParseError as exc:
            st.session_state.pop("analysis", None)
            st.error(str(exc))
        else:
            with st.spinner("Analyzing resume..."):
                st.session_state["analysis"] = run_analysis(resume, jd_text)

    analysis: Analysis | None = st.session_state.get("analysis")
    if analysis:
        st.divider()
        render_score(analysis.score)
        st.divider()
        render_keywords(analysis.keywords)
