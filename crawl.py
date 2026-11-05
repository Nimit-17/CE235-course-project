import json
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMPANIES = ROOT / "companies.json"
UA = "ce235-job-snapshot (course project; contact 24B0718)"


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        text = data.strip()
        if text:
            self.parts.append(text)

    def text(self):
        return " ".join(self.parts)


def html_to_text(value):
    if not value:
        return ""
    parser = TextExtractor()
    parser.feed(value)
    parser.close()
    return parser.text()


def fetch_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def greenhouse_rows(company, payload, fetched_at):
    rows = []
    for job in payload.get("jobs", []):
        location = job.get("location") or {}
        rows.append(
            {
                "source": "greenhouse",
                "company": company,
                "job_id": str(job.get("id")),
                "title": job.get("title") or "",
                "url": job.get("absolute_url") or "",
                "location": location.get("name") or "",
                "text": html_to_text(job.get("content")),
                "fetched_at": fetched_at,
            }
        )
    return rows


def lever_rows(company, payload, fetched_at):
    rows = []
    for job in payload:
        categories = job.get("categories") or {}
        location = categories.get("location") or job.get("country") or ""
        rows.append(
            {
                "source": "lever",
                "company": company,
                "job_id": str(job.get("id")),
                "title": job.get("text") or "",
                "url": job.get("hostedUrl") or job.get("applyUrl") or "",
                "location": location,
                "text": (job.get("descriptionPlain") or html_to_text(job.get("description"))).strip(),
                "fetched_at": fetched_at,
            }
        )
    return rows


def ashby_rows(company, payload, fetched_at):
    rows = []
    for job in payload.get("jobs", []):
        if job.get("isListed") is False:
            continue
        rows.append(
            {
                "source": "ashby",
                "company": company,
                "job_id": str(job.get("id")),
                "title": job.get("title") or "",
                "url": job.get("jobUrl") or job.get("applyUrl") or "",
                "location": job.get("location") or "",
                "text": (job.get("descriptionPlain") or html_to_text(job.get("descriptionHtml"))).strip(),
                "fetched_at": fetched_at,
            }
        )
    return rows


SOURCES = {
    "greenhouse": {
        "url": "https://boards-api.greenhouse.io/v1/boards/{company}/jobs?content=true",
        "rows": greenhouse_rows,
    },
    "lever": {
        "url": "https://api.lever.co/v0/postings/{company}?mode=json",
        "rows": lever_rows,
    },
    "ashby": {
        "url": "https://api.ashbyhq.com/posting-api/job-board/{company}",
        "rows": ashby_rows,
    },
}


def main():
    boards = json.loads(COMPANIES.read_text(encoding="utf-8"))
    day = date.today().isoformat()
    fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    out_dir = ROOT / "data" / day
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    jobs_path = out_dir / "jobs.jsonl"
    summary = []
    with jobs_path.open("w", encoding="utf-8") as jobs_file:
        for source, companies in boards.items():
            spec = SOURCES[source]
            for company in companies:
                url = spec["url"].format(company=company)
                try:
                    payload = fetch_json(url)
                except urllib.error.HTTPError as error:
                    summary.append((source, company, f"HTTP {error.code}"))
                    time.sleep(0.4)
                    continue
                except urllib.error.URLError as error:
                    summary.append((source, company, f"failed: {error.reason}"))
                    time.sleep(0.4)
                    continue

                raw_path = raw_dir / source / f"{company}.json"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(json.dumps(payload), encoding="utf-8")

                rows = spec["rows"](company, payload, fetched_at)
                for row in rows:
                    jobs_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                summary.append((source, company, len(rows)))
                time.sleep(0.4)

    total = 0
    for source, company, count in summary:
        print(f"{source:12} {company:20} {count}")
        if isinstance(count, int):
            total += count
    print(f"saved {total} jobs to {jobs_path}")


if __name__ == "__main__":
    main()
