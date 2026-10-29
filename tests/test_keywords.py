import pytest

from matcher.keywords import compare_keywords, extract_noun_phrases, find_skills

JD = """Senior Data Engineer
- Build data pipelines with Python, Apache Airflow and Spark
- Deploy services on AWS using Docker and Kubernetes
- Strong experience with PostgreSQL and a solid understanding of event-driven architecture
"""


def test_known_jd_returns_expected_skills(skills):
    found = set(find_skills(JD, skills))
    expected = {"Python", "Apache Airflow", "Apache Spark", "Amazon Web Services",
                "Docker", "Kubernetes", "PostgreSQL", "Data Pipelines"}
    assert expected <= found


@pytest.mark.parametrize("text, skill", [
    ("Used sklearn for modeling", "scikit-learn"),
    ("Used scikit learn for modeling", "scikit-learn"),
    ("Used Scikit-Learn for modeling", "scikit-learn"),
    ("Backend in NodeJS", "Node.js"),
    ("Backend in node.js", "Node.js"),
    ("Deployed on K8s", "Kubernetes"),
    ("Worked with Postgres", "PostgreSQL"),
    ("Experience with LLMs", "Large Language Models"),
    ("Strong C++ skills", "C++"),
    ("Services written in Go", "Go"),
])
def test_variants_match_canonical_skill(skills, text, skill):
    assert skill in find_skills(text, skills)


@pytest.mark.parametrize("text, not_skill", [
    ("We go above and beyond", "Go"),     # short names are case-sensitive
    ("Collaborate with R&D teams", "R"),  # "R" must not match inside "R&D"
    ("Strong C++ skills", "C#"),
    ("PySpark jobs", "Python"),           # whole-word only
])
def test_no_false_positives(skills, text, not_skill):
    assert not_skill not in find_skills(text, skills)


def test_noun_phrases_filter_generic_words(nlp, skills):
    phrases = extract_noun_phrases(JD, nlp, skills).values()
    assert "event driven architecture" in phrases
    for phrase in phrases:
        for generic in ("experience", "understanding", "team"):
            assert generic not in phrase.split()


def test_compare_splits_matched_and_missing(skills, nlp):
    resume = "Built ETL data pipeline jobs in Python and Spark, deployed with Docker."
    result = compare_keywords(JD, resume, skills, nlp)
    assert {"Python", "Apache Spark", "Docker", "Data Pipelines"} <= set(result.matched_keywords)
    assert {"Kubernetes", "PostgreSQL", "Amazon Web Services"} <= set(result.missing_keywords)
    assert not set(result.matched_keywords) & set(result.missing_keywords)
    assert result.total == len(result.matched_keywords) + len(result.missing_keywords)


def test_phrase_matching_uses_lemmas(skills, nlp):
    jd = "You will own payment processing infrastructure."
    resume = "Maintained the payment processing infrastructures for checkout."
    result = compare_keywords(jd, resume, skills, nlp)
    assert "payment processing infrastructure" in result.matched_keywords
