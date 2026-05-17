"""
Job Scraper for Naukri, Indeed, Monster, and others.
Fetches new jobs daily and saves them to the database.
"""
import requests
from bs4 import BeautifulSoup
from datetime import datetime

class ScrapedJob:
    def __init__(self, title, company, location, summary, apply_link, portal, posted_at=None):
        self.title = title
        self.company = company
        self.location = location
        self.summary = summary
        self.apply_link = apply_link
        self.portal = portal
        self.posted_at = posted_at or datetime.utcnow().isoformat()

    def to_dict(self):
        return {
            'title': self.title,
            'company': self.company,
            'location': self.location,
            'summary': self.summary,
            'apply_link': self.apply_link,
            'portal': self.portal,
            'posted_at': self.posted_at,
        }

# Example stub for Naukri scraping (expand for other portals)
def scrape_naukri_jobs(query="software engineer", location="", max_results=20):
    url = f"https://www.naukri.com/{query.replace(' ', '-')}-jobs-in-{location}" if location else f"https://www.naukri.com/{query.replace(' ', '-')}-jobs"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, "html.parser")
    jobs = []
    for job_card in soup.select("article.jobTuple"):
        title = job_card.select_one("a.title").get_text(strip=True) if job_card.select_one("a.title") else ""
        company = job_card.select_one("a.subTitle").get_text(strip=True) if job_card.select_one("a.subTitle") else ""
        location = job_card.select_one("li.location").get_text(strip=True) if job_card.select_one("li.location") else ""
        summary = job_card.select_one("div.job-description").get_text(strip=True) if job_card.select_one("div.job-description") else ""
        apply_link = job_card.select_one("a.title")['href'] if job_card.select_one("a.title") else ""
        jobs.append(ScrapedJob(title, company, location, summary, apply_link, "naukri").to_dict())
        if len(jobs) >= max_results:
            break
    return jobs

# TODO: Add similar functions for Indeed, Monster, etc.

if __name__ == "__main__":
    jobs = scrape_naukri_jobs()
    for job in jobs:
        print(job)
