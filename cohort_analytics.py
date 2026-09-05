"""
Cohort Analytics — the piece that answers "impact of skilling initiatives"
from the problem statement, not just individual skill gaps.

WHAT THIS FILE DOES
--------------------
Reads trainee_records.json — including each trainee's check-in history
(dated skill-gap + employment snapshots, tracked from the point their
programme started) — and produces:

1. AGGREGATE SUMMARY per training program, e.g.:
       "PMKVY - Python Backend Development: 5 trainees,
        60% employed, 40% employed & in-field, avg skill match 72%"
   This is the number that actually proves whether a skilling program
   is working — not just one person's result.

2. PER-TRAINEE DETAIL TABLE, e.g.:
       Trainee One | Python Backend Dev | Employed | In-field | 83%
       (+28.3 pts since programme start) | Missing: Teamwork
   This is the "who is struggling, with what, and are they improving"
   view an admin/counsellor would actually use to decide who needs
   follow-up training.

WHY THIS FILE DOESN'T DUPLICATE ANY LOGIC
-------------------------------------------
It reads the LATEST recorded check-in for each trainee — the same
check-in data api.py's /trainee/{id}/checkin endpoint computes via
parse_resume() + compute_skill_gap() when it's recorded. This file does
not re-run the AI pipeline; it reports on results already computed and
stored, so a cohort report reflects each trainee's most recent check-in,
not a fresh re-scan on every page load.

For any trainee with NO check-in history yet (e.g. records added by hand
without going through the API), it falls back to computing live from
their resume, so the report never silently skips someone.
"""

import json
from pathlib import Path
from statistics import mean

from resume_parser import parse_resume, ResumeTextExtractionError
from skill_gap_engine import compute_skill_gap

TRAINEE_DATA_PATH = Path("trainee_records.json")
JOB_DATA_PATH = Path("job_dataset.json")
RESUME_FOLDER = Path("sample_resumes")

UNASSIGNED_PROGRAM_LABEL = "No programme assigned yet"


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
    Builds one flat summary record for a trainee, combining their stored
    profile with the latest available skill-gap snapshot — preferring
    their recorded check-in history (fast, already computed), and only
    falling back to a live resume re-scan if no check-in exists yet.
    """
    program = trainee.get("training_program") or UNASSIGNED_PROGRAM_LABEL
    base = {
        "trainee_id": trainee_id,
        "name": trainee["name"],
        "training_program": program,
        "target_job": trainee.get("target_job"),
        "employment_status": trainee.get("employment_status"),
        "matches_trained_field": (
            trainee["employment_details"]["matches_trained_field"]
            if trainee.get("employment_details") else None
        ),
    }

    if not trainee.get("target_job"):
        return {
            **base,
            "skill_match_percent": None,
            "missing_skills": None,
            "checkin_count": 0,
            "skill_match_trend": None,
            "error": "Trainee has not completed their profile yet.",
        }

    checkins = trainee.get("checkins", [])

    if checkins:
        # Prefer recorded history — already computed, and this is what
        # gives us the "trend since programme start" figure.
        first_checkin = checkins[0]
        latest_checkin = checkins[-1]

        trend = None
        if len(checkins) > 1:
            trend = round(
                latest_checkin["skill_match_percent"] - first_checkin["skill_match_percent"], 1
            )

        return {
            **base,
            "skill_match_percent": latest_checkin["skill_match_percent"],
            "missing_skills": latest_checkin["missing_skills"],
            "checkin_count": len(checkins),
            "skill_match_trend": trend,
            "error": None,
        }

    # No check-in history recorded yet (e.g. hand-added data) — fall back
    # to computing live, so this trainee still appears in the report.
    resume_file = trainee.get("resume_file")
    resume_path = RESUME_FOLDER / resume_file if resume_file else None

    if not resume_path or not resume_path.exists():
        return {
            **base,
            "skill_match_percent": None,
            "missing_skills": None,
            "checkin_count": 0,
            "skill_match_trend": None,
            "error": f"Resume file not found: {resume_path}",
        }

    try:
        trainee_skills = parse_resume(str(resume_path), is_pdf=True)
    except ResumeTextExtractionError as e:
        return {
            **base,
            "skill_match_percent": None,
            "missing_skills": None,
            "checkin_count": 0,
            "skill_match_trend": None,
            "error": str(e),
        }

    required_skills = jobs[trainee["target_job"]]["skills"]
    gap_result = compute_skill_gap(trainee_skills, required_skills)

    return {
        **base,
        "skill_match_percent": gap_result["match_score_percent"],
        "missing_skills": gap_result["missing_skills"],
        "checkin_count": 0,
        "skill_match_trend": None,
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
                    "average_skill_match_percent": 83.3,
                    "average_skill_match_trend": 28.3
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

        valid_trends = [r["skill_match_trend"] for r in results
                          if r["skill_match_trend"] is not None]
        avg_trend = round(mean(valid_trends), 1) if valid_trends else None

        program_summaries[program] = {
            "total_trainees": total,
            "employed": employed,
            "employed_percent": round(100 * employed / total, 1) if total else 0.0,
            "unemployed": total - employed,
            "employed_in_field": in_field,
            "employed_in_field_percent": round(100 * in_field / total, 1) if total else 0.0,
            "average_skill_match_percent": avg_score,
            "average_skill_match_trend": avg_trend,
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
        trend = stats['average_skill_match_trend']
        if trend is not None:
            sign = "+" if trend >= 0 else ""
            print(f"  Avg. change since start:  {sign}{trend} pts")

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
        print(f"  Skill match:      {r['skill_match_percent']}%"
              + (f" ({'+' if r['skill_match_trend'] >= 0 else ''}{r['skill_match_trend']} pts since start)"
                 if r["skill_match_trend"] is not None else ""))
        print(f"  Check-ins recorded: {r['checkin_count']}")
        print(f"  Missing skills:   {r['missing_skills'] or 'None'}")


# ---------------------------------------------------------------------------
# DEMO — run this file directly to see the full cohort report
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    report = build_cohort_report()
    print_cohort_report(report)
