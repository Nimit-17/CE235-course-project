import json
import shutil
import subprocess
import tempfile
import time
import urllib.error
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
CLOSED_PHRASES = (
    "no longer open",
    "no longer available",
    "might have closed",
    "has been removed",
    "job not found",
    "job you requested was not found",
    "page not found",
    "position has been filled",
    "couldn't find anything",
    "sorry, but we can't find",
)


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
        completed = subprocess.run(command, capture_output=True, timeout=45, check=False)
        return html_to_text(completed.stdout.decode("utf-8", "replace"))
    finally:
        shutil.rmtree(profile, ignore_errors=True)


def page_text(source, url):
    if source == "lever":
        status, final, text = http_text(url)
        return status, final, text
    return None, url, rendered_text(url)


def looks_closed(text):
    lowered = text.lower()
    return any(phrase in lowered for phrase in CLOSED_PHRASES)


def main():
    labeled = ROOT / "data" / "labeled_temporal.jsonl"
    rows = [json.loads(line) for line in labeled.read_text(encoding="utf-8").splitlines() if line]
    closed = [row for row in rows if row["label"] == "closed"]
    fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    out = ROOT / "data" / "closed_from_diff.jsonl"
    kept = []

    for index, row in enumerate(closed, 1):
        url = row["url"]
        if not url:
            continue
        try:
            status, final, text = page_text(row["source"], url)
        except Exception as error:
            print(f"[{index}/{len(closed)}] fail {row['company']} {error}")
            continue
        ok = looks_closed(text)
        print(
            f"[{index}/{len(closed)}] {row['source']:12} {row['company']:10} "
            f"len={len(text):5} closed={ok} {url[:80]}"
        )
        if not ok or len(text) < 40:
            time.sleep(0.3)
            continue
        kept.append(
            {
                "source": row["source"],
                "company": row["company"],
                "job_id": row["job_id"],
                "title": row.get("title") or "",
                "url": final or url,
                "status": status,
                "label": "closed",
                "text": text,
                "fetched_at": fetched_at,
                "origin": "temporal_diff",
            }
        )
        time.sleep(0.3)

    out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept),
        encoding="utf-8",
    )
    print(f"saved {len(kept)} closed pages to {out}")


if __name__ == "__main__":
    main()
