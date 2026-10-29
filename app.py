"""Streamlit UI for the Resume–JD Matcher."""

import streamlit as st

from matcher import embeddings
from matcher.pdf_parser import PDFParseError, parse_resume

st.set_page_config(page_title="Resume–JD Matcher", page_icon="📄")


@st.cache_resource(show_spinner="Loading embedding model...")
def get_model():
    return embeddings.load_model()


st.title("📄 Resume–JD Matcher")

resume_file = st.file_uploader("Resume (PDF)", type=["pdf"])
jd_text = st.text_area("Job description", height=250)

if st.button("Analyze", type="primary", disabled=not (resume_file and jd_text.strip())):
    try:
        resume = parse_resume(resume_file)
    except PDFParseError as exc:
        st.error(str(exc))
        st.stop()
    with st.spinner("Comparing resume and job description..."):
        similarity = embeddings.text_similarity(get_model(), resume.text, jd_text)
    score = round(similarity * 100)
    st.metric("Similarity score", f"{score} / 100")
    st.progress(score / 100)
