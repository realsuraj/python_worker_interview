"""
API endpoint to serve scraped jobs (to be integrated with FastAPI or Flask if needed).
Stub for now: returns all jobs from the local DB.
"""
from fastapi import FastAPI
from db import get_jobs

app = FastAPI()

@app.get("/scraped-jobs")
def scraped_jobs():
    return {"jobs": get_jobs()}

# To run: uvicorn api:app --reload
