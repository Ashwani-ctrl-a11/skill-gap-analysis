"""
Cohort Analytics — the piece that answers "impact of skilling initiatives"
from the problem statement, not just individual skill gaps.

WHAT THIS FILE DOES
--------------------
Reads trainee_records.json (profile + employment status for every trainee),
runs each trainee's actual resume through the SAME resume_parser.py and
skill_gap_engine.py already built and tested — comparing each trainee
against their own target_job — and produces two things:

1. AGGREGATE SUMMARY per training program, e.g.:
       "PMKVY - Python Backend Development: 5 trainees,
        60% employed, 40% employed & in-field, avg skill match 72%"
   This is the number that actually proves whether a skilling program
   is working — not just one person's result.

2. PER-TRAINEE DETAIL TABLE, e.g.:
       Trainee One | Python Backend Dev | Employed | In-field | 83% | Missing: Teamwork
   This is the "who is struggling and with what" view an admin/
   counsellor would actually use to decide who needs follow-up training.

WHY THIS FILE DOESN'T DUPLICATE ANY LOGIC
-------------------------------------------
It doesn't re-implement skill matching — it just CALLS parse_resume()
and compute_skill_gap(), exactly like main.py does. This file's only
new logic is: loop over every trainee, and aggregate the results.
"""

import json
from pathlib import Path
from statistics import mean

from resume_parser import parse_resume
from skill_gap_engine import compute_skill_gap

TRAINEE_DATA_PATH = Path("trainee_records.json")
JOB_DATA_PATH = Path("job_dataset.json")
RESUME_FOLDER = Path("sample_resumes")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_trainees() -> dict:
    with open(TRAINEE_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["trainees"]


def _load_jobs() -> dict:
    with open(JOB_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["jobs"]


# ---------------------------------------------------------------------------
# Per-trainee analysis
# ---------------------------------------------------------------------------
def analyze_trainee(trainee_id: str, trainee: dict, jobs: dict) -> dict:
    """
    Runs one trainee's resume through the existing parser + skill-gap
    engine against their own target_job, and combines it with their
    stored profile/employment data into one flat record.
    """
    resume_path = RESUME_FOLDER / trainee["resume_file"]

    if not resume_path.exists():
        # Don't crash the whole report if one resume file is missing —
        # just flag it and move on, so the rest of the cohort still shows.
        return {
            "trainee_id": trainee_id,
            "name": trainee["name"],
            "training_program": trainee["training_program"],
            "target_job": trainee["target_job"],
            "employment_status": trainee["employment_status"],
            "matches_trained_field": (
                trainee["employment_details"]["matches_trained_field"]
                if trainee.get("employment_details") else None
            ),
            "skill_match_percent": None,
            "missing_skills": None,
            "error": f"Resume file not found: {resume_path}",
        }

    trainee_skills = parse_resume(str(resume_path), is_pdf=True)
    required_skills = jobs[trainee["target_job"]]["skills"]
    gap_result = compute_skill_gap(trainee_skills, required_skills)

    matches_field = None
    if trainee["employment_status"] == "Employed" and trainee.get("employment_details"):
        matches_field = trainee["employment_details"]["matches_trained_field"]

    return {
        "trainee_id": trainee_id,
        "name": trainee["name"],
        "training_program": trainee["training_program"],
        "target_job": trainee["target_job"],
        "employment_status": trainee["employment_status"],
        "matches_trained_field": matches_field,
        "skill_match_percent": gap_result["match_score_percent"],
        "missing_skills": gap_result["missing_skills"],
        "error": None,
    }


# ---------------------------------------------------------------------------
# Aggregate summary, grouped by training program
# ---------------------------------------------------------------------------
def build_cohort_report() -> dict:
    """
    Returns:
        {
            "per_trainee": [ ...list of analyze_trainee() results... ],
            "by_program": {
                "PMKVY - Python Backend Development": {
                    "total_trainees": 1,
                    "employed": 1,
                    "employed_percent": 100.0,
                    "unemployed": 0,
                    "employed_in_field": 1,
                    "employed_in_field_percent": 100.0,
                    "average_skill_match_percent": 83.3
                },
                ...
            }
        }
    """
    trainees = _load_trainees()
    jobs = _load_jobs()

    per_trainee_results = [
        analyze_trainee(tid, t, jobs) for tid, t in trainees.items()
    ]

    # Group results by training_program
    by_program: dict[str, list[dict]] = {}
    for result in per_trainee_results:
        by_program.setdefault(result["training_program"], []).append(result)

    program_summaries = {}
    for program, results in by_program.items():
        total = len(results)
        employed = sum(1 for r in results if r["employment_status"] == "Employed")
        in_field = sum(1 for r in results if r["matches_trained_field"] is True)

        valid_scores = [r["skill_match_percent"] for r in results
                         if r["skill_match_percent"] is not None]
        avg_score = round(mean(valid_scores), 1) if valid_scores else None

        program_summaries[program] = {
            "total_trainees": total,
            "employed": employed,
            "employed_percent": round(100 * employed / total, 1) if total else 0.0,
            "unemployed": total - employed,
            "employed_in_field": in_field,
            "employed_in_field_percent": round(100 * in_field / total, 1) if total else 0.0,
            "average_skill_match_percent": avg_score,
        }

    return {
        "per_trainee": per_trainee_results,
        "by_program": program_summaries,
    }


# ---------------------------------------------------------------------------
# Console display
# ---------------------------------------------------------------------------
def print_cohort_report(report: dict) -> None:
    print("=" * 70)
    print("IMPACT SUMMARY BY TRAINING PROGRAM")
    print("=" * 70)
    for program, stats in report["by_program"].items():
        print(f"\n{program}")
        print(f"  Total trainees:           {stats['total_trainees']}")
        print(f"  Employed:                 {stats['employed']} "
              f"({stats['employed_percent']}%)")
        print(f"  Unemployed:               {stats['unemployed']}")
        print(f"  Employed & in-field:      {stats['employed_in_field']} "
              f"({stats['employed_in_field_percent']}%)")
        avg = stats['average_skill_match_percent']
        print(f"  Average skill match:      {avg}%" if avg is not None
              else "  Average skill match:      N/A")

    print("\n" + "=" * 70)
    print("PER-TRAINEE DETAIL")
    print("=" * 70)
    for r in report["per_trainee"]:
        print(f"\n{r['name']} ({r['trainee_id']})")
        print(f"  Program:          {r['training_program']}")
        if r["error"]:
            print(f"  ⚠️  {r['error']}")
            continue
        print(f"  Employment:       {r['employment_status']}"
              + (f" (in-field: {r['matches_trained_field']})"
                 if r["matches_trained_field"] is not None else ""))
        print(f"  Skill match:      {r['skill_match_percent']}%")
        print(f"  Missing skills:   {r['missing_skills'] or 'None'}")


# ---------------------------------------------------------------------------
# DEMO — run this file directly to see the full cohort report
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    report = build_cohort_report()
    print_cohort_report(report)
