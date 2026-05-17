"""
Database utility for saving scraped jobs.
This is a stub; integrate with your actual DB (e.g., via SQLAlchemy, Django ORM, or direct API call to Java backend).
"""
import json
from datetime import datetime

DB_PATH = "scraped_jobs.json"  # Replace with real DB integration

def save_jobs(jobs):
    """Append new jobs to the local JSON file (replace with real DB logic)."""
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except Exception:
        existing = []
    # Avoid duplicates by apply_link
    seen = {job['apply_link'] for job in existing}
    new_jobs = [job for job in jobs if job['apply_link'] not in seen]
    if not new_jobs:
        return 0
    existing.extend(new_jobs)
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    return len(new_jobs)

def get_jobs():
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

if __name__ == "__main__":
    print(f"Loaded {len(get_jobs())} jobs from DB.")
