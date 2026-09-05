"""
Recommendation Engine — suggests courses to close skill gaps, using the
SAME semantic embedding approach as skill_gap_engine.py.

WHY SEMANTIC (not a fixed skill -> course dictionary)
-------------------------------------------------------
A fixed lookup table (e.g. {"SQL": "SQL Course"}) only works for skills
you explicitly typed in by hand, and breaks the moment a missing skill's
name doesn't exactly match a key in that dictionary.

Instead, this engine embeds each missing skill and each course's taught
skills, then finds the course whose taught skills are semantically
CLOSEST to the missing skill — even if the wording differs.

This also means the system "adapts" automatically: add a new course to
course_dataset.json, and it's immediately considered for any matching
missing skill, with zero code changes. No manual re-mapping needed.

HOW IT WORKS
------------
1. Load course_dataset.json — each course has a name + list of skills
   it teaches.
2. Flatten this into a single list of (skill, course_name) pairs, so a
   course teaching 3 skills becomes 3 separate embeddable entries.
3. Embed every one of those taught-skill entries ONCE at startup.
4. For each missing skill passed in:
   a. Embed it
   b. Compare against all taught-skill embeddings (cosine similarity)
   c. Recommend the course behind the closest match (if above threshold)
5. Return a mapping: { missing_skill: recommended_course_name }

This file depends only on course_dataset.json and sentence-transformers —
it doesn't import anything from skill_gap_engine.py, so it can be tested
completely on its own.
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------------------------
# Setup: load course data + embedding model once
# ---------------------------------------------------------------------------
_COURSE_DATA_PATH = Path(__file__).parent / "course_dataset.json"

with open(_COURSE_DATA_PATH, "r", encoding="utf-8") as f:
    _COURSES = json.load(f)["courses"]

# Reuse the same lightweight model used across the project for consistency.
_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# How close a missing skill needs to be to a course's taught skill to
# count as a valid recommendation. Skills that don't clear this bar get
# no recommendation rather than a bad/irrelevant one.
RECOMMENDATION_THRESHOLD = 0.5


def _build_skill_to_course_index():
    """
    Flattens course_dataset.json into two parallel lists:
        _flat_skills:  ["Python", "SQL", "Database Management", ...]
        _flat_courses: ["Python Programming Fundamentals - NSDC",
                         "SQL for Data Analysis - NSDC",
                         "SQL for Data Analysis - NSDC", ...]
    i.e. every (taught_skill, course_name) pair the dataset defines,
    so a course teaching 3 skills appears 3 times, once per skill.

    Also precomputes embeddings for every taught skill, ONCE, so lookups
    at request time are fast.
    """
    flat_skills = []
    flat_courses = []

    for course in _COURSES:
        for skill in course["teaches_skills"]:
            flat_skills.append(skill)
            flat_courses.append(course["course_name"])

    embeddings = _MODEL.encode(flat_skills, convert_to_tensor=True)
    return flat_skills, flat_courses, embeddings


_FLAT_SKILLS, _FLAT_COURSES, _SKILL_EMBEDDINGS = _build_skill_to_course_index()


def recommend_courses(missing_skills: list[str]) -> dict:
    """
    Given a list of missing skills (from skill_gap_engine's output),
    returns the best-matching course recommendation for each.

    Args:
        missing_skills: e.g. ["Power BI", "AWS", "Teamwork"]

    Returns:
        {
            "Power BI": {"recommended_course": "Power BI for Business
                          Analytics - NSDC", "confidence": 1.0},
            "AWS": {"recommended_course": "AWS Cloud Fundamentals - NSDC",
                     "confidence": 1.0},
            "Teamwork": {"recommended_course": "Workplace Communication &
                          Soft Skills - NSDC", "confidence": 1.0}
        }
        A skill with no course above the confidence threshold maps to
        {"recommended_course": None, "confidence": <best score found>}.
    """
    if not missing_skills:
        return {}

    recommendations = {}
    missing_embeddings = _MODEL.encode(missing_skills, convert_to_tensor=True)

    for i, skill in enumerate(missing_skills):
        similarities = util.cos_sim(missing_embeddings[i], _SKILL_EMBEDDINGS)[0]
        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])

        if best_score >= RECOMMENDATION_THRESHOLD:
            recommendations[skill] = {
                "recommended_course": _FLAT_COURSES[best_idx],
                "confidence": round(best_score, 3),
            }
        else:
            recommendations[skill] = {
                "recommended_course": None,
                "confidence": round(best_score, 3),
            }

    return recommendations


def print_recommendations(recommendations: dict) -> None:
    """Pretty-prints recommendation results to the console."""
    print("\nRecommended courses to close the gap:")
    for skill, rec in recommendations.items():
        if rec["recommended_course"]:
            print(f"  📘 {skill:<20} -> {rec['recommended_course']} "
                  f"(confidence: {rec['confidence']})")
        else:
            print(f"  ⚠️  {skill:<20} -> No strong course match found "
                  f"(best score: {rec['confidence']})")


# ---------------------------------------------------------------------------
# DEMO — run this file directly to test it standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_missing_skills = ["Power BI", "AWS", "Teamwork", "Cloud Computing"]

    result = recommend_courses(sample_missing_skills)
    print_recommendations(result)
