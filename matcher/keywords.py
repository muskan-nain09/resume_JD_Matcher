"""Extract keywords from a job description and check which ones the resume has.

Two complementary methods:

1. **Curated skills list** (`data/skills.txt`): high precision. If the JD says
   "Kubernetes", we know that's a skill, and aliases ("K8s") are handled.
2. **spaCy noun chunks**: catch domain phrases the list can't anticipate,
   such as "demand forecasting" or "recommendation engine". These are noisier,
   so we filter out generic words like "experience" or "team".

Skills are checked against the resume with the same regex used on the JD.
Noun phrases are compared by *lemma* so "data pipelines" in the JD matches
"data pipeline" in the resume.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from spacy.language import Language

DEFAULT_SKILLS_PATH = Path(__file__).resolve().parent.parent / "data" / "skills.txt"
SPACY_MODEL = "en_core_web_sm"

# Aliases this short ("Go", "R", "AI") are matched case-sensitively, otherwise
# every "go" or "r" in normal prose would count as a skill.
CASE_SENSITIVE_MAX_LEN = 2

# A skill must not be glued to these characters on either side. \w alone isn't
# enough because "C" should not match inside "C++" or "C#", and "R" should not
# match "R&D".
BOUNDARY = r"[\w+#&]"

# Words that show up in almost every JD but say nothing about required skills.
GENERIC_WORDS = {
    "ability", "abilities", "experience", "experiences", "team", "teams",
    "skill", "skills", "knowledge", "understanding", "familiarity",
    "proficiency", "year", "years", "candidate", "candidates", "role", "roles",
    "position", "job", "work", "opportunity", "opportunities", "company",
    "responsibility", "responsibilities", "requirement", "requirements",
    "qualification", "qualifications", "plus", "bonus", "degree", "background",
    "environment", "culture", "benefit", "benefits", "salary", "day", "time",
    "way", "part", "level", "variety", "range", "number", "type", "kind",
    "thing", "lot", "area", "areas", "field", "fields", "best", "practice",
    "practices", "member", "members", "people", "world", "product", "products",
    "solution", "solutions", "tool", "tools", "technology", "technologies",
    "platform", "platforms", "system", "systems", "stack", "need", "needs",
    "goal", "goals", "impact", "growth", "communication", "collaboration",
    "passion", "mindset", "attitude", "detail", "details", "hands", "role",
    "something", "anything", "everything", "others", "ownership", "stakeholder",
    "stakeholders", "manager", "managers", "engineer", "engineers", "scientist",
    "scientists", "developer", "developers", "similar", "related", "etc",
    "e.g", "i.e", "nice", "strong", "solid", "excellent", "good", "great",
    "proven", "demonstrated", "relevant", "preferred", "required", "familiar",
    "hands-on", "plus", "new", "other", "various", "multiple", "high", "key",
    "core", "own", "end", "production", "prototype", "self", "fast", "paced",
}

# Named entities that are about the employer or logistics, not skills.
SKIP_ENTITY_TYPES = {"ORG", "GPE", "LOC", "PERSON", "DATE", "TIME", "MONEY",
                     "PERCENT", "CARDINAL", "ORDINAL", "QUANTITY", "NORP", "FAC"}

MAX_PHRASE_WORDS = 3

# JDs are full of bulleted lists; markers confuse the parser at line start.
BULLET_PREFIX = re.compile(r"^\s*[•●▪■◦‣○►▸➢➤✓✔*\-–—]\s*")


@dataclass
class Skill:
    name: str
    patterns: list[re.Pattern] = field(default_factory=list)

    def found_in(self, text: str) -> bool:
        return any(p.search(text) for p in self.patterns)


@dataclass
class KeywordResult:
    matched_keywords: list[str]
    missing_keywords: list[str]

    @property
    def total(self) -> int:
        return len(self.matched_keywords) + len(self.missing_keywords)


def _alias_pattern(alias: str) -> re.Pattern:
    """Build a whole-word regex for one alias.

    Spaces and hyphens are interchangeable and optional between words, so
    "scikit-learn", "scikit learn" and "scikitlearn" all match one alias.
    """
    parts = [re.escape(p) for p in re.split(r"[\s\-]+", alias.strip()) if p]
    body = r"[\s\-]*".join(parts)
    flags = 0 if len(alias) <= CASE_SENSITIVE_MAX_LEN else re.IGNORECASE
    return re.compile(rf"(?<!{BOUNDARY}){body}(?!{BOUNDARY})", flags)


def load_skills(path: str | Path = DEFAULT_SKILLS_PATH) -> list[Skill]:
    """Parse skills.txt: one skill per line, aliases separated by '|'."""
    skills = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        aliases = [a.strip() for a in line.split("|") if a.strip()]
        skills.append(Skill(name=aliases[0], patterns=[_alias_pattern(a) for a in aliases]))
    return skills


def find_skills(text: str, skills: list[Skill]) -> list[str]:
    """Return canonical names of every skill mentioned in `text`."""
    return [s.name for s in skills if s.found_in(text)]


def load_nlp() -> Language:
    """Load the small English spaCy model. Only the parser, tagger, lemmatizer
    and NER are needed, which is everything in en_core_web_sm."""
    import spacy

    return spacy.load(SPACY_MODEL)


def _lemma_text(doc) -> str:
    return " ".join(t.lemma_.lower() for t in doc if not t.is_punct and not t.is_space)


def extract_noun_phrases(
    text: str, nlp: Language, skills: list[Skill] | None = None
) -> dict[str, str]:
    """Return {lemmatized phrase: display phrase} for meaningful noun chunks.

    Rules, each chosen to cut a specific kind of noise:
    - Parse line by line, so a chunk never runs across two bullet points.
    - Ignore the first word of a bullet line. JD bullets start with a verb
      ("Deploy models...") that the small spaCy model often tags as a noun.
    - Keep only the part of a chunk after its last "and"/"or"/comma, so
      "recommendation and forecasting products" doesn't merge two ideas.
    - Drop stop words, generic words, and named entities (company, dates).
    - Keep only 2-3 word phrases. Single tech words are the skills list's job;
      single words from noun chunks were mostly noise ("data", "models").
    - Skip phrases made only of proper nouns, which are usually company or
      product names that aren't in the skills list.
    - Skip chunks that contain a known skill, so it isn't counted twice.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    is_bullet = [bool(BULLET_PREFIX.match(line)) for line in lines]
    docs = nlp.pipe(BULLET_PREFIX.sub("", line) for line in lines)
    phrases: dict[str, str] = {}
    for doc, bullet in zip(docs, is_bullet):
        for chunk in doc.noun_chunks:
            span = chunk
            breaks = [t.i for t in chunk if t.pos_ == "CCONJ" or t.text in {",", ";"}]
            if breaks:
                span = doc[breaks[-1] + 1:chunk.end]
            if skills and find_skills(span.text, skills):
                continue
            # VERB is allowed because participles inside a noun chunk act as
            # modifiers: "event-driven architecture", "automated retraining".
            tokens = [
                t for t in span
                if t.is_alpha and not t.is_stop
                and not (bullet and t.i == 0)
                and t.ent_type_ not in SKIP_ENTITY_TYPES
                and t.lemma_.lower() not in GENERIC_WORDS
                and t.text.lower() not in GENERIC_WORDS
                and t.pos_ in {"NOUN", "PROPN", "ADJ", "VERB"}
            ]
            if not 2 <= len(tokens) <= MAX_PHRASE_WORDS:
                continue
            if tokens[-1].pos_ not in {"NOUN", "PROPN"}:
                continue
            if all(t.pos_ == "PROPN" for t in tokens):
                continue
            lemma = " ".join(t.lemma_.lower() for t in tokens)
            phrases.setdefault(lemma, " ".join(t.text.lower() for t in tokens))
    return phrases


def compare_keywords(
    jd_text: str, resume_text: str, skills: list[Skill], nlp: Language
) -> KeywordResult:
    """Extract JD keywords and split them into matched / missing in the resume."""
    matched, missing = [], []

    for skill in skills:
        if skill.found_in(jd_text):
            (matched if skill.found_in(resume_text) else missing).append(skill.name)

    resume_lemmas = _lemma_text(nlp(resume_text))
    for lemma, display in extract_noun_phrases(jd_text, nlp, skills).items():
        found = re.search(rf"(?<!\w){re.escape(lemma)}(?!\w)", resume_lemmas)
        (matched if found else missing).append(display)

    return KeywordResult(matched_keywords=matched, missing_keywords=missing)
