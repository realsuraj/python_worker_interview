"""
Daily job scraping scheduler.
"""
import time
from scraper import scrape_naukri_jobs
from db import save_jobs

SCRAPE_INTERVAL_SECONDS = 24 * 60 * 60  # Once per day

if __name__ == "__main__":
    while True:
        print("[JobScraper] Starting daily scrape...")
        jobs = scrape_naukri_jobs()
        count = save_jobs(jobs)
        print(f"[JobScraper] Saved {count} new jobs from Naukri.")
        # TODO: Add calls for Indeed, Monster, etc.
        print(f"[JobScraper] Sleeping for {SCRAPE_INTERVAL_SECONDS // 3600} hours.")
        time.sleep(SCRAPE_INTERVAL_SECONDS)
