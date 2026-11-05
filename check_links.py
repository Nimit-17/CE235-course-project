"""
Classify job posting URLs from a text file.

Usage:
  python check_links.py links.txt
  python check_links.py links.txt -o results/my_run.csv

links.txt: one URL per line. Blank lines and # comments are ignored.
Writes a CSV (default results/check_links.csv) and prints a short table.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

from classify import load, classify_url

ROOT = Path(__file__).resolve().parent


def read_urls(path: Path) -> list[str]:
    urls = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("http://") or text.startswith("https://"):
            if text not in seen:
                urls.append(text)
                seen.add(text)
            continue
        raise SystemExit(f"Not a URL (use http/https): {text}")
    if not urls:
        raise SystemExit(f"No URLs found in {path}")
    return urls


def main():
    parser = argparse.ArgumentParser(description="Classify job URLs from a .txt file")
    parser.add_argument("links_file", type=Path, help="Text file with one URL per line")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="CSV output path (default: results/check_links_YYYYMMDD_HHMMSS.csv)",
    )
    args = parser.parse_args()

    urls = read_urls(args.links_file)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = args.output or (ROOT / "results" / f"check_links_{stamp}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model... ({len(urls)} URLs)")
    tokenizer, model, labels = load()

    fieldnames = [
        "label",
        "live",
        "closed",
        "reason",
        "url",
        "final_url",
        "preview",
    ]
    rows = []
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, url in enumerate(urls, 1):
            print(f"[{index}/{len(urls)}] {url}")
            try:
                result = classify_url(url, tokenizer, model, labels)
            except Exception as error:
                result = {
                    "label": "uncertain",
                    "live": None,
                    "closed": None,
                    "reason": f"error: {error}",
                    "final_url": url,
                    "preview": "",
                }
            row = {
                "label": result.get("label"),
                "live": result.get("live"),
                "closed": result.get("closed"),
                "reason": result.get("reason"),
                "url": url,
                "final_url": result.get("final_url") or url,
                "preview": (result.get("preview") or "").replace("\n", " ")[:240],
            }
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            print(
                f"  -> {row['label']}  live={row['live']} closed={row['closed']}  "
                f"({row['reason']})"
            )
            time.sleep(0.2)

    print("\n==== SUMMARY ====")
    print(f"{'label':10} {'live':6} {'closed':6} url")
    for row in rows:
        live = "" if row["live"] is None else f"{row['live']:.3f}"
        closed = "" if row["closed"] is None else f"{row['closed']:.3f}"
        print(f"{row['label']:10} {live:6} {closed:6} {row['url']}")
    counts = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print("counts:", counts)
    print("saved", out)


if __name__ == "__main__":
    main()
