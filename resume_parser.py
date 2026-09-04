"""
Resume Parser — turns a resume (PDF or plain text) into a clean list of
skills, using a predefined skills taxonomy.

WHY THIS APPROACH (taxonomy-based, not free-form NER)
------------------------------------------------------
General-purpose NER models aren't trained to reliably detect "skills" as
a category, and training a custom one needs labeled data we don't have.
Instead we scan the resume text against a curated list of known skills
(skills_taxonomy.json). This is fast, explainable, and good enough for
a hackathon demo — and it's the same first-pass approach real resume
screening tools use before anything fancier.

TWO-LAYER MATCHING
-------------------
Layer 1 (exact/fuzzy text match): catches skills written close to their
    taxonomy form, e.g. "python", "Python3", "PYTHON" all match "Python".
Layer 2 (semantic fallback, using the same embedding model as the
    skill-gap engine): catches skills phrased differently, e.g. resume
    says "built REST based APIs" -> still matches "REST APIs".

OUTPUT
------
parse_resume("some_resume.pdf") -> ["Python", "SQL", "Communication", ...]
This list is what you feed directly into skill_gap_engine.compute_skill_gap().
"""

import json
import re
from pathlib import Path

import pdfplumber
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------------------------
# Setup: load taxonomy + embedding model once
# ---------------------------------------------------------------------------
_TAXONOMY_PATH = Path(__file__).parent / "skills_taxonomy.json"

with open(_TAXONOMY_PATH, "r", encoding="utf-8") as f:
    _TAXONOMY_SKILLS = json.load(f)["skills"]

# Reuse the SAME model as skill_gap_engine.py so behaviour is consistent
# and we don't load two separate copies of the model in the same process.
_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Precompute embeddings for every taxonomy skill ONCE (not per resume) —
# this is what makes Layer 2 fast even though it runs on every parse call.
_TAXONOMY_EMBEDDINGS = _MODEL.encode(_TAXONOMY_SKILLS, convert_to_tensor=True)

SEMANTIC_MATCH_THRESHOLD = 0.6  # stricter than the gap engine's threshold,
                                 # since resume text is noisier than a clean skill list


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------
def _extract_text_from_pdf(pdf_path: str) -> str:
    """Pulls all text out of a PDF resume, page by page."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def _split_into_lines(text: str) -> list[str]:
    """Breaks resume text into short chunks (lines/phrases) for scanning."""
    # Split on newlines, bullets, commas, and pipes — resumes use all of these
    raw_pieces = re.split(r"[\n•,|;]+", text)
    return [p.strip() for p in raw_pieces if len(p.strip()) > 1]


# ---------------------------------------------------------------------------
# Layer 1: exact / fuzzy text match
# ---------------------------------------------------------------------------
def _exact_match_skills(full_text: str) -> set[str]:
    """Checks which taxonomy skills appear (case-insensitively) in the text."""
    text_lower = full_text.lower()
    found = set()
    for skill in _TAXONOMY_SKILLS:
        # word-boundary-ish check so "R" doesn't match inside "ारray" etc.
        pattern = re.escape(skill.lower())
        if re.search(rf"\b{pattern}\b", text_lower):
            found.add(skill)
    return found


# ---------------------------------------------------------------------------
# Layer 2: semantic fallback match
# ---------------------------------------------------------------------------
def _semantic_match_skills(lines: list[str], already_found: set[str]) -> set[str]:
    """
    For lines that didn't already trigger an exact match, check if their
    MEANING is close to any taxonomy skill (catches differently-worded
    mentions like "built REST based APIs" -> "REST APIs").
    """
    if not lines:
        return set()

    found = set()
    line_embeddings = _MODEL.encode(lines, convert_to_tensor=True)

    # Compare every line against every taxonomy skill at once
    similarity_matrix = util.cos_sim(line_embeddings, _TAXONOMY_EMBEDDINGS)

    for line_idx, line in enumerate(lines):
        best_idx = int(similarity_matrix[line_idx].argmax())
        best_score = float(similarity_matrix[line_idx][best_idx])
        matched_skill = _TAXONOMY_SKILLS[best_idx]

        if best_score >= SEMANTIC_MATCH_THRESHOLD and matched_skill not in already_found:
            found.add(matched_skill)

    return found


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------
def parse_resume(source: str, is_pdf: bool = True) -> list[str]:
    """
    Extracts a clean, deduplicated skill list from a resume.

    Args:
        source: path to a PDF file, OR raw resume text (if is_pdf=False)
        is_pdf: set False to pass plain text directly (useful for testing
                or if you're pulling text from a pasted textarea in the UI)

    Returns:
        Sorted list of skill names found in the resume, e.g.
        ["Communication", "Excel", "Python", "SQL"]
    """
    full_text = _extract_text_from_pdf(source) if is_pdf else source

    if not full_text.strip():
        return []

    # Layer 1: exact/fuzzy match across the whole text
    exact_matches = _exact_match_skills(full_text)

    # Layer 2: semantic match on lines that might phrase skills differently
    lines = _split_into_lines(full_text)
    semantic_matches = _semantic_match_skills(lines, already_found=exact_matches)

    all_skills = exact_matches | semantic_matches
    return sorted(all_skills)


# ---------------------------------------------------------------------------
# DEMO — run this file directly to test on plain text (no PDF needed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_resume_text = """
    John Doe
    Aspiring Data Analyst

    Skills:
    - Proficient in Python and basic SQL
    - Experience with Excel for data reporting
    - Built REST based APIs during internship
    - Strong communication and team leadership skills

    Education: Diploma in Computer Applications
    """

    skills = parse_resume(sample_resume_text, is_pdf=False)
    print("Extracted skills:", skills)
