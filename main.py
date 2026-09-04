"""
main.py — Interactive entry point for the MVP pipeline.

FLOW:
    1. Scan sample_resumes/ folder -> let user pick (or search for) a trainee
    2. Read job_dataset.json -> let user pick (or search for) a job,
       browsed grouped by sector
    3. Run: resume -> [resume_parser.py] -> skill list
            job title -> [job_dataset.json] -> required skill list
            both -> [skill_gap_engine.py] -> match score + gap report
    4. Print the report
    5. Ask "compare another?" and loop, so you can demo multiple
       combinations in one run without restarting the script.

Nothing about resume_parser.py or skill_gap_engine.py changes here —
this file only changes HOW the trainee/job inputs get chosen.
"""

import json
from pathlib import Path

from resume_parser import parse_resume
from skill_gap_engine import compute_skill_gap, print_report

RESUME_FOLDER = Path("sample_resumes")
JOB_DATA_PATH = Path("job_dataset.json")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_jobs() -> dict:
    """Reads job_dataset.json -> { job_title: {"sector": ..., "skills": [...]} }"""
    with open(JOB_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["jobs"]


def list_resumes() -> list[Path]:
    """Finds all PDF resumes in the sample_resumes/ folder."""
    if not RESUME_FOLDER.exists():
        return []
    return sorted(RESUME_FOLDER.glob("*.pdf"))


# ---------------------------------------------------------------------------
# Job selection: grouped by sector, with search
# ---------------------------------------------------------------------------
def choose_job(jobs: dict) -> str:
    """
    Shows all jobs grouped by sector with continuous numbering.
    User can either type a number directly, or type a search term
    (e.g. "nurse") to filter the list down first.

    Returns the chosen job title (a key from `jobs`).
    """
    # Build sector -> [job titles] groups, and a flat numbered index
    sectors: dict[str, list[str]] = {}
    for title, info in jobs.items():
        sectors.setdefault(info["sector"], []).append(title)
    for titles in sectors.values():
        titles.sort()

    numbered_jobs = []  # index i (0-based) -> job title
    for sector in sorted(sectors):
        for title in sectors[sector]:
            numbered_jobs.append(title)

    def print_all_jobs(job_list: list[str]):
        print("\nAvailable jobs by sector:")
        current_sector = None
        for title in job_list:
            sector = jobs[title]["sector"]
            if sector != current_sector:
                print(f"\n[{sector}]")
                current_sector = sector
            idx = numbered_jobs.index(title) + 1
            print(f"  {idx}. {title}")

    print_all_jobs(numbered_jobs)

    while True:
        choice = input(
            "\nEnter a job number, or type part of a job name to search: "
        ).strip()

        # Try as a direct number first
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(numbered_jobs):
                return numbered_jobs[idx]
            print(f"Please enter a number between 1 and {len(numbered_jobs)}.")
            continue

        # Otherwise treat it as a search term
        matches = [t for t in numbered_jobs if choice.lower() in t.lower()]
        if not matches:
            print(f"No jobs found matching '{choice}'. Try again.")
            continue
        if len(matches) == 1:
            print(f"Matched: {matches[0]}")
            return matches[0]

        print("\nMatches found:")
        for i, m in enumerate(matches, start=1):
            print(f"  {i}. {m}")
        sub_choice = input("Enter number to confirm: ").strip()
        if sub_choice.isdigit() and 1 <= int(sub_choice) <= len(matches):
            return matches[int(sub_choice) - 1]
        print("Invalid selection, let's try again.")


# ---------------------------------------------------------------------------
# Trainee selection (same number-or-search pattern, simpler list)
# ---------------------------------------------------------------------------
def choose_resume(resumes: list[Path]) -> Path:
    print("\nAvailable trainees:")
    for i, path in enumerate(resumes, start=1):
        print(f"  {i}. {path.name}")

    while True:
        choice = input(
            "\nEnter a trainee number, or type part of the filename to search: "
        ).strip()

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(resumes):
                return resumes[idx]
            print(f"Please enter a number between 1 and {len(resumes)}.")
            continue

        matches = [r for r in resumes if choice.lower() in r.name.lower()]
        if not matches:
            print(f"No trainees found matching '{choice}'. Try again.")
            continue
        if len(matches) == 1:
            print(f"Matched: {matches[0].name}")
            return matches[0]

        print("\nMatches found:")
        for i, m in enumerate(matches, start=1):
            print(f"  {i}. {m.name}")
        sub_choice = input("Enter number to confirm: ").strip()
        if sub_choice.isdigit() and 1 <= int(sub_choice) <= len(matches):
            return matches[int(sub_choice) - 1]
        print("Invalid selection, let's try again.")


# ---------------------------------------------------------------------------
# Pipeline (same logic as before, just parameterized)
# ---------------------------------------------------------------------------
def run_pipeline(resume_path: Path, job_title: str, jobs: dict) -> dict:
    trainee_skills = parse_resume(str(resume_path), is_pdf=True)
    print(f"\nParsed skills from resume: {trainee_skills}")

    required_skills = jobs[job_title]["skills"]
    result = compute_skill_gap(trainee_skills, required_skills)
    print_report(trainee_name=resume_path.name, job_title=job_title, result=result)
    return result


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    jobs = load_jobs()
    resumes = list_resumes()

    if not resumes:
        print(f"No PDF resumes found in '{RESUME_FOLDER}/'. "
              f"Add at least one .pdf file there and run again.")
        return

    while True:
        resume_path = choose_resume(resumes)
        job_title = choose_job(jobs)

        run_pipeline(resume_path, job_title, jobs)

        again = input("Compare another? (y/n): ").strip().lower()
        if again != "y":
            print("Done.")
            break


if __name__ == "__main__":
    main()
