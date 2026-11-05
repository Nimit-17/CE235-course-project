import json
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UA = "ce235-job-snapshot (course project; contact 24B0718)"
EDGE_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]
CDX = "https://web.archive.org/cdx/search/cdx"
PATTERNS = {
    "greenhouse": [
        "job-boards.greenhouse.io/{company}/jobs/*",
        "boards.greenhouse.io/{company}/jobs/*",
    ],
    "lever": ["jobs.lever.co/{company}/*"],
    "ashby": ["jobs.ashbyhq.com/{company}/*"],
}
JOB_URL = {
    "greenhouse": re.compile(r"/jobs/\d+$", re.I),
    "lever": re.compile(r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I),
    "ashby": re.compile(r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I),
}


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if self.skip:
            return
        text = " ".join(data.split())
        if text:
            self.parts.append(text)

    def text(self):
        return " ".join(self.parts)


def html_to_text(html):
    parser = TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


def edge_path():
    for path in EDGE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("Microsoft Edge was not found")


def http_text(url):
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            html = response.read().decode("utf-8", "replace")
            return response.status, response.geturl(), html_to_text(html)
    except urllib.error.HTTPError as error:
        html = error.read().decode("utf-8", "replace")
        return error.code, url, html_to_text(html)


def rendered_text(url):
    profile = Path(tempfile.mkdtemp(prefix="edge-job-"))
    try:
        command = [
            str(edge_path()),
            "--headless",
            "--disable-gpu",
            "--no-first-run",
            f"--user-data-dir={profile}",
            "--virtual-time-budget=12000",
            "--dump-dom",
            url,
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=45,
            check=False,
        )
        html = completed.stdout.decode("utf-8", "replace")
        return html_to_text(html)
    finally:
        shutil.rmtree(profile, ignore_errors=True)


def page_text(source, url):
    if source == "lever":
        status, final, text = http_text(url)
        return status, final, text
    text = rendered_text(url)
    return None, url, text


def cdx_urls(pattern):
    query = urllib.parse.urlencode(
        {
            "url": pattern,
            "output": "json",
            "fl": "original,timestamp,statuscode",
            "filter": "statuscode:200",
            "collapse": "urlkey",
            "limit": "40",
        }
    )
    request = urllib.request.Request(f"{CDX}?{query}", headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read().decode("utf-8", "replace").strip()
    if not payload:
        return []
    rows = json.loads(payload)
    return [row[0] for row in rows[1:]]


def normalize(url):
    return url.split("?")[0].rstrip("/")


def removed_urls(source, company, live):
    found = []
    seen = set()
    for pattern in PATTERNS[source]:
        try:
            originals = cdx_urls(pattern.format(company=company))
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
            continue
        for original in originals:
            base = normalize(original)
            if base in live or base in seen:
                continue
            if not JOB_URL[source].search(base):
                continue
            seen.add(base)
            found.append(base)
            if len(found) == 2:
                return found
        time.sleep(0.4)
    return found


def main():
    day = "2026-10-30"
    jobs_path = ROOT / "data" / day / "jobs.jsonl"
    rows = [json.loads(line) for line in jobs_path.read_text(encoding="utf-8").splitlines() if line]
    live = {normalize(row["url"]) for row in rows}
    boards = json.loads((ROOT / "companies.json").read_text(encoding="utf-8"))
    fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    out_dir = ROOT / "data" / day
    closed_path = out_dir / "closed.jsonl"
    rendered_live_path = out_dir / "rendered_live.jsonl"

    closed_rows = []
    rendered_rows = []
    for source, companies in boards.items():
        for company in companies:
            urls = removed_urls(source, company, live)
            print(f"{source} {company}: {len(urls)} removed urls")
            for url in urls:
                try:
                    status, final, text = page_text(source, url)
                except (subprocess.TimeoutExpired, urllib.error.URLError, OSError) as error:
                    print(f"  fail {url} {error}")
                    continue
                print(f"  closed text {len(text)} {url}")
                closed_rows.append(
                    {
                        "source": source,
                        "company": company,
                        "url": final or url,
                        "status": status,
                        "label": "closed",
                        "text": text,
                        "fetched_at": fetched_at,
                    }
                )
                time.sleep(0.3)

            example = next(row for row in rows if row["source"] == source and row["company"] == company)
            try:
                status, final, text = page_text(source, example["url"])
            except (subprocess.TimeoutExpired, urllib.error.URLError, OSError) as error:
                print(f"  live fail {example['url']} {error}")
                continue
            print(f"  live text {len(text)} {example['url']}")
            rendered_rows.append(
                {
                    "source": source,
                    "company": company,
                    "url": final or example["url"],
                    "status": status,
                    "label": "live",
                    "text": text,
                    "fetched_at": fetched_at,
                }
            )
            time.sleep(0.3)

    closed_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in closed_rows),
        encoding="utf-8",
    )
    rendered_live_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rendered_rows),
        encoding="utf-8",
    )
    print(f"saved {len(closed_rows)} closed and {len(rendered_rows)} rendered live")


if __name__ == "__main__":
    main()
