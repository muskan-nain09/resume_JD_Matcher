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
from dotenv import load_dotenv

from matcher import embeddings, job_store, keywords, llm, scoring
from matcher.pdf_parser import ParsedResume, PDFParseError, parse_resume

load_dotenv()
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


@st.cache_resource(show_spinner="Connecting to Pinecone...")
def get_job_store():
    return job_store.JobStore.from_env()


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


def load_rewrites(analysis: Analysis) -> None:
    """Generate suggestions once per analysis and store them in session state.

    Any LLM failure is stored as a message instead of raised, so the score and
    keywords above still display when Ollama isn't running.
    """
    bullets = embeddings.weakest_bullets(
        get_model(), analysis.resume.bullets, analysis.jd_text, n=llm.MAX_BULLETS
    )
    if not bullets:
        st.session_state["rewrites"] = (
            "No bullet points were found in the resume (lines starting with •, -, or *)."
        )
        return
    try:
        provider = llm.get_provider()
        with st.spinner(f"Writing suggestions with {provider.name}..."):
            st.session_state["rewrites"] = llm.suggest_rewrites(
                provider, analysis.jd_text, analysis.keywords.missing_keywords, bullets
            )
    except llm.LLMError as exc:
        st.session_state["rewrites"] = f"Rewrite suggestions are unavailable right now. {exc}"


def render_rewrites() -> None:
    st.subheader("✍️ Rewrite suggestions")
    rewrites = st.session_state.get("rewrites")
    if isinstance(rewrites, str):
        st.info(rewrites)
        return
    st.caption("Your weakest bullets compared with the JD. "
               "Fill in any [placeholders] with your real numbers; never guess.")
    for r in rewrites:
        col1, col2 = st.columns(2)
        col1.markdown(f"**Original**\n\n{r.original}")
        col2.markdown(f"**Suggested**\n\n{r.suggestion or '_No suggestion returned._'}")
        st.divider()


def render_saved_jobs(resume_file) -> None:
    try:
        store = get_job_store()
    except Exception as exc:  # bad key, network error, etc.
        st.error(f"Couldn't connect to Pinecone: {exc}")
        return

    st.subheader("Save a job description")
    with st.form("save_job", clear_on_submit=True):
        title = st.text_input("Job title")
        company = st.text_input("Company")
        saved_jd = st.text_area("Job description", height=150)
        if st.form_submit_button("Save job"):
            if not (title.strip() and saved_jd.strip()):
                st.warning("A job title and description are required.")
            else:
                with st.spinner("Saving..."):
                    store.save_job(title.strip(), company.strip(), saved_jd,
                                   embeddings.embed_text(get_model(), saved_jd))
                st.success(f"Saved “{title.strip()}”.")

    st.divider()
    st.subheader("Find best matching jobs")
    if resume_file is None:
        st.info("Upload your resume in the Match tab first.")
        return
    if st.button("Find matches"):
        try:
            resume = parse_resume(resume_file)
        except PDFParseError as exc:
            st.error(str(exc))
            return
        with st.spinner("Searching saved jobs..."):
            matches = store.find_matches(embeddings.embed_text(get_model(), resume.text))
        if not matches:
            st.info("No saved jobs yet. Save one above.")
        for m in matches:
            label = f"{m.score:.0%} · {m.title}" + (f" at {m.company}" if m.company else "")
            with st.expander(label):
                st.caption(m.jd_preview)


# ---------- page ----------

st.title("📄 Resume–JD Matcher")
st.caption("Compare your resume with a job description and see what's missing.")

# The Saved Jobs tab only exists when Pinecone is configured; the core
# matching feature never depends on it.
pinecone_enabled = job_store.is_enabled()
if not pinecone_enabled:
    st.caption("💡 Saved Jobs is turned off. Set `PINECONE_API_KEY` in your `.env` "
               "to save job descriptions and find your best matches.")
tabs = st.tabs(["Match", "Saved Jobs"] if pinecone_enabled else ["Match"])
match_tab = tabs[0]

with match_tab:
    left, right = st.columns(2)
    resume_file = left.file_uploader("Resume (PDF)", type=["pdf"])
    jd_text = right.text_area("Job description", height=200,
                              placeholder="Paste the full job description here...")

    ready = resume_file is not None and jd_text.strip()
    if st.button("Analyze", type="primary", disabled=not ready):
        # A new analysis invalidates old suggestions.
        st.session_state.pop("analysis", None)
        st.session_state.pop("rewrites", None)
        try:
            resume = parse_resume(resume_file)
        except PDFParseError as exc:
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
        st.divider()
        if "rewrites" not in st.session_state:
            load_rewrites(analysis)
        render_rewrites()

if pinecone_enabled:
    with tabs[1]:
        render_saved_jobs(resume_file)
