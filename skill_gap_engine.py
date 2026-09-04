"""
Skill-Gap Engine — Core AI logic for the SIH 2026 project
"Ai powered tracker for skill gap recognition and Employment outcome"
Team: Void pioneers | PS ID: 26135

WHAT THIS FILE DOES
--------------------
Given:
    1. A trainee's skills (list of strings)
    2. A job's required skills (list of strings)

It computes:
    - An overall match score (%)
    - Which required skills the trainee already covers (matched)
    - Which required skills are missing / weak (the "gap")

HOW IT WORKS
------------
1. Load a pretrained sentence-embedding model (all-MiniLM-L6-v2).
   This turns each skill (a short phrase) into a vector of numbers
   that captures its MEANING, not just its exact spelling.
   e.g. "ML" and "Machine Learning" end up close together in vector space,
   even though they share no letters — this is why we use embeddings
   instead of plain keyword matching.

2. Embed the trainee's skills and the job's required skills SEPARATELY,
   one skill at a time (not the whole list as one blob). This is what
   lets us report WHICH specific skills are missing, not just one
   overall percentage.

3. For every required skill, find the trainee skill it's closest to
   (cosine similarity — a standard way to measure how "similar" two
   vectors are, from -1 to 1, where 1 = identical meaning).

4. If the best match score is above a threshold (default 0.55), we
   count that skill as COVERED. If it's below, we count it as a GAP.

5. Overall match % = (number of covered skills) / (total required skills)

This whole thing is deliberately simple and explainable — every number
in the output can be traced back to a specific comparison, which is
exactly the "explainability layer" mentioned in our architecture slide.

RUN THIS FILE DIRECTLY to see a demo with sample data (see bottom of file).
"""

from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------------------------
# STEP 1: Load the embedding model once (reused for every comparison)
# ---------------------------------------------------------------------------
# 'all-MiniLM-L6-v2' is small, fast, and runs fine on a laptop CPU —
# good for a live hackathon demo where you don't want a 10-second delay
# per request.
_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# How similar two skills need to be (0 to 1) to count as "the same skill".
# Tune this if you see too many false matches/misses during testing.
MATCH_THRESHOLD = 0.55


def compute_skill_gap(trainee_skills: list[str], required_skills: list[str]) -> dict:
    """
    Compare a trainee's skills against a job's required skills.

    Args:
        trainee_skills: e.g. ["Python", "Excel", "basic SQL", "communication"]
        required_skills: e.g. ["Python", "SQL", "Power BI", "AWS", "communication"]

    Returns:
        A dictionary with the match score and a detailed breakdown of
        every required skill (covered or missing), e.g.:

        {
            "match_score_percent": 60.0,
            "matched_skills": ["Python", "communication"],
            "missing_skills": ["Power BI", "AWS"],
            "details": [
                {"required_skill": "Python", "status": "covered",
                 "closest_trainee_skill": "Python", "similarity": 1.0},
                ...
            ]
        }
    """
    if not required_skills:
        raise ValueError("required_skills list cannot be empty")
    if not trainee_skills:
        # No trainee skills at all -> everything is a gap
        return {
            "match_score_percent": 0.0,
            "matched_skills": [],
            "missing_skills": list(required_skills),
            "details": [
                {"required_skill": s, "status": "missing",
                 "closest_trainee_skill": None, "similarity": 0.0}
                for s in required_skills
            ],
        }

    # --- STEP 2: Embed each skill individually ---
    # encode() returns one vector per string in the input list.
    trainee_embeddings = _MODEL.encode(trainee_skills, convert_to_tensor=True)
    required_embeddings = _MODEL.encode(required_skills, convert_to_tensor=True)

    matched_skills = []
    missing_skills = []
    details = []

    # --- STEP 3 & 4: For each required skill, find its best match ---
    for i, req_skill in enumerate(required_skills):
        # Compare this one required skill's embedding against ALL trainee
        # skill embeddings at once -> a row of similarity scores.
        similarities = util.cos_sim(required_embeddings[i], trainee_embeddings)[0]

        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])
        best_trainee_skill = trainee_skills[best_idx]

        is_covered = best_score >= MATCH_THRESHOLD
        status = "covered" if is_covered else "missing"

        if is_covered:
            matched_skills.append(req_skill)
        else:
            missing_skills.append(req_skill)

        details.append({
            "required_skill": req_skill,
            "status": status,
            "closest_trainee_skill": best_trainee_skill,
            "similarity": round(best_score, 3),
        })

    # --- STEP 5: Overall match percentage ---
    match_score_percent = round(100 * len(matched_skills) / len(required_skills), 1)

    return {
        "match_score_percent": match_score_percent,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "details": details,
    }


def print_report(trainee_name: str, job_title: str, result: dict) -> None:
    """Pretty-prints a skill-gap result to the console (used by the demo)."""
    print("=" * 60)
    print(f"Trainee: {trainee_name}")
    print(f"Target Role: {job_title}")
    print("=" * 60)
    print(f"Overall Match Score: {result['match_score_percent']}%\n")

    print("Skill-by-skill breakdown:")
    for d in result["details"]:
        mark = "✅" if d["status"] == "covered" else "❌"
        print(f"  {mark} {d['required_skill']:<15} "
              f"(closest trainee skill: '{d['closest_trainee_skill']}', "
              f"similarity: {d['similarity']})")

    print(f"\nMissing skills (the gap): {result['missing_skills'] or 'None — full match!'}")
    print()


# ---------------------------------------------------------------------------
# DEMO — run this file directly (`python skill_gap_engine.py`) to see it work
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Sample trainee (this would normally come from the resume parser, Step 2)
    trainee_1 = ["Python", "Excel", "basic SQL", "communication", "MS Word"]

    # Sample job requirement (this would normally come from the job dataset, Step 3)
    job_data_analyst = ["Python", "SQL", "Power BI", "AWS", "communication"]

    result = compute_skill_gap(trainee_1, job_data_analyst)
    print_report("Trainee A (sample resume)", "Data Analyst", result)

    # A second example to show it generalizes
    trainee_2 = ["Java", "Spring Boot", "REST APIs", "Git", "teamwork"]
    job_backend_dev = ["Java", "Spring", "SQL", "Docker", "teamwork"]

    result_2 = compute_skill_gap(trainee_2, job_backend_dev)
    print_report("Trainee B (sample resume)", "Backend Developer", result_2)
