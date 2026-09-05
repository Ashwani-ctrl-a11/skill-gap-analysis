"""
api.py — FastAPI backend that wraps all the existing, already-tested
logic (resume_parser, skill_gap_engine, recommendation_engine,
cohort_analytics) behind HTTP endpoints a browser/dashboard can call.

NOTHING about the existing files changes. This file only ADDS a web
layer on top of them — it imports and calls the same functions you
already tested standalone in the terminal.

DATA STORAGE (MVP-level, intentional)
--------------------------------------
Trainee data is stored in trainee_records.json — a plain file, read and
rewritten on every signup/update. This is fine for a hackathon demo with
a handful of trainees. In production this would move to a proper
database (PostgreSQL, already in the original tech stack) to handle
concurrent writes safely at scale.

RUNNING THIS FILE
-----------------
    uvicorn api:app --reload

Then open http://127.0.0.1:8000/docs in a browser — FastAPI
auto-generates an interactive test page for every endpoint below,
so you can try signup/submit/results/cohort without building any
frontend yet.
"""

import json
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from resume_parser import parse_resume, ResumeTextExtractionError
from skill_gap_engine import compute_skill_gap
from recommendation_engine import recommend_courses
from cohort_analytics import build_cohort_report

TRAINEE_DATA_PATH = Path("trainee_records.json")
JOB_DATA_PATH = Path("job_dataset.json")
RESUME_FOLDER = Path("sample_resumes")
RESUME_FOLDER.mkdir(exist_ok=True)

# --- Upload validation limits (MVP-level guardrails) ------------------------
# A real PDF file starts with these bytes ("%PDF-"). Checking this — instead
# of trusting the filename extension or the browser-supplied content-type,
# either of which can be spoofed or simply wrong — is a minimal but real
# check that what was uploaded is actually a PDF before we try to parse it.
PDF_MAGIC_BYTES = b"%PDF-"
MAX_RESUME_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB — generous for a text resume,
                                           # small enough to block accidental
                                           # huge/wrong-file uploads

app = FastAPI(title="Skill Gap & Employment Outcome Tracker API")

# Allows a frontend running on a different port (e.g. a React dev server)
# to call this API from the browser. Fine for an MVP; in production this
# would be locked down to a specific domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Small helpers — read/write the same JSON files main.py and
# cohort_analytics.py already use
# ---------------------------------------------------------------------------
def _load_trainees() -> dict:
    with open(TRAINEE_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_trainees(data: dict) -> None:
    with open(TRAINEE_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_jobs() -> dict:
    with open(JOB_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["jobs"]


def _find_trainee_by_username(data: dict, username: str) -> Optional[tuple[str, dict]]:
    """Returns (trainee_id, trainee_record) for a given username, or None."""
    for tid, t in data["trainees"].items():
        if t.get("username") == username:
            return tid, t
    return None


def _next_trainee_id(data: dict) -> str:
    """Generates the next sequential trainee_XXX id."""
    existing_numbers = [
        int(tid.split("_")[1]) for tid in data["trainees"].keys()
        if tid.startswith("trainee_") and tid.split("_")[1].isdigit()
    ]
    next_num = (max(existing_numbers) + 1) if existing_numbers else 1
    return f"trainee_{next_num:03d}"


def _validate_and_save_resume(resume: UploadFile, trainee_id: str) -> str:
    """
    Reads an uploaded resume fully into memory, checks it's actually a PDF
    (by magic bytes, not just filename/content-type) and under the size
    cap, then writes it to sample_resumes/. Raises HTTPException on any
    validation failure. Returns the saved filename.
    """
    contents = resume.file.read()

    if len(contents) > MAX_RESUME_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Resume file is too large "
                   f"(max {MAX_RESUME_SIZE_BYTES // (1024*1024)} MB).",
        )

    if not contents.startswith(PDF_MAGIC_BYTES):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file does not appear to be a valid PDF.",
        )

    resume_filename = f"{trainee_id}.pdf"
    resume_path = RESUME_FOLDER / resume_filename
    with open(resume_path, "wb") as f:
        f.write(contents)

    return resume_filename


def _compute_checkin(trainee: dict, jobs: dict) -> dict:
    """
    Runs the existing resume_parser + skill_gap_engine against a trainee's
    CURRENT resume + target_job, and packages the result as one dated
    check-in entry. This is the same computation get_result() returns —
    factored out here so both the results endpoint and the check-in/
    history endpoints share one implementation instead of duplicating it.

    Raises ResumeTextExtractionError if the PDF has no readable text
    (see resume_parser.py) — callers should catch this and turn it into
    a clear HTTP error rather than a confusing 0% result.
    """
    resume_path = RESUME_FOLDER / trainee["resume_file"]
    if not resume_path.exists():
        raise HTTPException(status_code=404, detail="Resume file missing on server.")

    trainee_skills = parse_resume(str(resume_path), is_pdf=True)
    required_skills = jobs[trainee["target_job"]]["skills"]
    gap_result = compute_skill_gap(trainee_skills, required_skills)

    return {
        "date": date.today().isoformat(),
        "employment_status": trainee.get("employment_status"),
        "job_title": (trainee.get("employment_details") or {}).get("job_title"),
        "matches_trained_field": (trainee.get("employment_details") or {}).get("matches_trained_field"),
        "skill_match_percent": gap_result["match_score_percent"],
        "missing_skills": gap_result["missing_skills"],
    }


# ---------------------------------------------------------------------------
# 1. Signup — just a username + name, no password (MVP-only, as agreed)
# ---------------------------------------------------------------------------
@app.post("/signup")
def signup(username: str = Form(...), name: str = Form(...)):
    data = _load_trainees()

    if _find_trainee_by_username(data, username):
        raise HTTPException(status_code=400, detail="Username already taken.")

    new_id = _next_trainee_id(data)
    data["trainees"][new_id] = {
        "username": username,
        "name": name,
        "resume_file": None,
        "training_program": None,
        "target_job": None,
        "training_institute": None,
        "training_start_date": None,
        "training_duration_months": None,
        "education_level": None,
        "location": {"state": None, "district": None},
        "category": None,
        "preferred_job_sector": None,
        "prior_experience_years": None,
        "internship": {"done": False, "organization": None,
                        "duration_months": None, "role": None},
        "employment_status": None,
        "employment_details": None,
        "checkins": [],
    }
    _save_trainees(data)
    return {"trainee_id": new_id, "username": username, "name": name}


# ---------------------------------------------------------------------------
# 2. Login — look up an existing trainee by username
# ---------------------------------------------------------------------------
@app.post("/login")
def login(username: str = Form(...)):
    data = _load_trainees()
    result = _find_trainee_by_username(data, username)
    if not result:
        raise HTTPException(status_code=404, detail="Username not found.")
    trainee_id, trainee = result
    return {"trainee_id": trainee_id, **trainee}


# ---------------------------------------------------------------------------
# 3. Submit / update profile + resume upload
# ---------------------------------------------------------------------------
@app.post("/trainee/{trainee_id}/profile")
def submit_profile(
    trainee_id: str,
    target_job: str = Form(...),
    training_program: str = Form(...),
    training_institute: str = Form(""),
    training_start_date: str = Form(""),
    training_duration_months: int = Form(0),
    education_level: str = Form(""),
    state: str = Form(""),
    district: str = Form(""),
    category: str = Form(""),
    preferred_job_sector: str = Form(""),
    prior_experience_years: int = Form(0),
    internship_done: bool = Form(False),
    internship_organization: str = Form(""),
    internship_duration_months: int = Form(0),
    internship_role: str = Form(""),
    employment_status: str = Form("Unemployed"),
    employment_job_title: str = Form(""),
    matches_trained_field: bool = Form(False),
    resume: UploadFile = File(...),
):
    data = _load_trainees()
    if trainee_id not in data["trainees"]:
        raise HTTPException(status_code=404, detail="Trainee ID not found.")

    jobs = _load_jobs()
    if target_job not in jobs:
        raise HTTPException(
            status_code=400,
            detail=f"'{target_job}' is not a recognised job. "
                   f"Available jobs: {list(jobs.keys())}",
        )

    # Validates it's actually a PDF and under the size cap before saving.
    resume_filename = _validate_and_save_resume(resume, trainee_id)

    employment_details = None
    if employment_status == "Employed":
        employment_details = {
            "job_title": employment_job_title,
            "employed_since": None,
            "matches_trained_field": matches_trained_field,
        }

    data["trainees"][trainee_id].update({
        "resume_file": resume_filename,
        "training_program": training_program,
        "target_job": target_job,
        "training_institute": training_institute,
        "training_start_date": training_start_date,
        "training_duration_months": training_duration_months,
        "education_level": education_level,
        "location": {"state": state, "district": district},
        "category": category,
        "preferred_job_sector": preferred_job_sector,
        "prior_experience_years": prior_experience_years,
        "internship": {
            "done": internship_done,
            "organization": internship_organization or None,
            "duration_months": internship_duration_months or None,
            "role": internship_role or None,
        },
        "employment_status": employment_status,
        "employment_details": employment_details,
    })

    # Record this submission as check-in #1 — the "program start" baseline
    # snapshot that later check-ins (weekly/monthly) get compared against.
    jobs = _load_jobs()
    try:
        first_checkin = _compute_checkin(data["trainees"][trainee_id], jobs)
        data["trainees"][trainee_id]["checkins"] = [first_checkin]
    except ResumeTextExtractionError as e:
        _save_trainees(data)  # keep the profile fields, just skip the check-in
        raise HTTPException(status_code=400, detail=str(e))

    _save_trainees(data)

    return {"message": "Profile updated successfully.", "trainee_id": trainee_id}


# ---------------------------------------------------------------------------
# 4. Get a trainee's skill-gap + recommendation result
# ---------------------------------------------------------------------------
@app.get("/trainee/{trainee_id}/result")
def get_result(trainee_id: str):
    data = _load_trainees()
    if trainee_id not in data["trainees"]:
        raise HTTPException(status_code=404, detail="Trainee ID not found.")

    trainee = data["trainees"][trainee_id]
    if not trainee.get("resume_file") or not trainee.get("target_job"):
        raise HTTPException(
            status_code=400,
            detail="Trainee has not submitted a profile/resume yet.",
        )

    jobs = _load_jobs()
    resume_path = RESUME_FOLDER / trainee["resume_file"]
    if not resume_path.exists():
        raise HTTPException(status_code=404, detail="Resume file missing on server.")

    try:
        trainee_skills = parse_resume(str(resume_path), is_pdf=True)
    except ResumeTextExtractionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    required_skills = jobs[trainee["target_job"]]["skills"]
    gap_result = compute_skill_gap(trainee_skills, required_skills)
    recommendations = recommend_courses(gap_result["missing_skills"])

    return {
        "trainee_id": trainee_id,
        "name": trainee["name"],
        "target_job": trainee["target_job"],
        "parsed_skills": trainee_skills,
        "match_score_percent": gap_result["match_score_percent"],
        "matched_skills": gap_result["matched_skills"],
        "missing_skills": gap_result["missing_skills"],
        "skill_details": gap_result["details"],  # includes per-skill proficiency tiers
        "recommendations": recommendations,
    }


# ---------------------------------------------------------------------------
# 4b. Record a new check-in — a periodic (weekly/monthly) re-snapshot of a
#     trainee's skill gap + employment status, tracked from the point their
#     programme started (check-in #1, created automatically in the profile
#     endpoint above) through however many follow-ups get recorded over time.
# ---------------------------------------------------------------------------
@app.post("/trainee/{trainee_id}/checkin")
def add_checkin(
    trainee_id: str,
    employment_status: str = Form("Unemployed"),
    employment_job_title: str = Form(""),
    matches_trained_field: bool = Form(False),
    resume: Optional[UploadFile] = File(None),
):
    """
    Records a new check-in for a trainee already in the system. If a new
    resume is uploaded, it replaces the stored one and skills are
    re-parsed from it; otherwise the existing resume on file is re-scored
    (useful for a pure employment-status update with no new resume).
    """
    data = _load_trainees()
    if trainee_id not in data["trainees"]:
        raise HTTPException(status_code=404, detail="Trainee ID not found.")

    trainee = data["trainees"][trainee_id]
    if not trainee.get("target_job"):
        raise HTTPException(
            status_code=400,
            detail="Trainee has not completed their initial profile yet.",
        )

    if resume is not None:
        resume_filename = _validate_and_save_resume(resume, trainee_id)
        trainee["resume_file"] = resume_filename

    employment_details = None
    if employment_status == "Employed":
        employment_details = {
            "job_title": employment_job_title,
            "employed_since": None,
            "matches_trained_field": matches_trained_field,
        }
    trainee["employment_status"] = employment_status
    trainee["employment_details"] = employment_details

    jobs = _load_jobs()
    try:
        new_checkin = _compute_checkin(trainee, jobs)
    except ResumeTextExtractionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    trainee.setdefault("checkins", []).append(new_checkin)
    _save_trainees(data)

    return {"message": "Check-in recorded.", "checkin": new_checkin}


# ---------------------------------------------------------------------------
# 4c. Full check-in history for a trainee — the "from program start,
#     tracked at intervals" timeline.
# ---------------------------------------------------------------------------
@app.get("/trainee/{trainee_id}/history")
def get_history(trainee_id: str):
    data = _load_trainees()
    if trainee_id not in data["trainees"]:
        raise HTTPException(status_code=404, detail="Trainee ID not found.")

    trainee = data["trainees"][trainee_id]
    return {
        "trainee_id": trainee_id,
        "name": trainee["name"],
        "target_job": trainee.get("target_job"),
        "checkins": trainee.get("checkins", []),
    }


# ---------------------------------------------------------------------------
# 5. Admin — full cohort analytics report
# ---------------------------------------------------------------------------
@app.get("/admin/cohort")
def get_cohort_report():
    return build_cohort_report()


# ---------------------------------------------------------------------------
# 6. List available jobs (grouped by sector) — for populating dropdowns
# ---------------------------------------------------------------------------
@app.get("/jobs")
def list_jobs():
    jobs = _load_jobs()
    by_sector: dict[str, list[str]] = {}
    for title, info in jobs.items():
        by_sector.setdefault(info["sector"], []).append(title)
    for titles in by_sector.values():
        titles.sort()
    return by_sector


@app.get("/")
def root():
    return {"status": "API is running. Visit /docs to try the endpoints."}
