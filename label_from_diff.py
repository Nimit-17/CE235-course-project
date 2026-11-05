import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT / "data" / "2026-10-30" / "jobs.jsonl"
NEW_DIRS = sorted((ROOT / "data").glob("20*"))


def load(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


def key(row):
    return (row["source"], row["company"], str(row["job_id"]))


def main():
    new_day = None
    for day_dir in reversed(NEW_DIRS):
        jobs = day_dir / "jobs.jsonl"
        if jobs.exists() and day_dir.name != "2026-10-30":
            new_day = day_dir
            break
    if new_day is None:
        raise SystemExit("No second crawl found under data/")

    old_rows = load(OLD)
    new_rows = load(new_day / "jobs.jsonl")
    old_map = {key(row): row for row in old_rows}
    new_keys = {key(row) for row in new_rows}

    live = []
    closed = []
    for k, row in old_map.items():
        text = (row.get("text") or "").strip()
        if len(text) < 40:
            continue
        item = {
            "source": row["source"],
            "company": row["company"],
            "job_id": row["job_id"],
            "title": row.get("title") or "",
            "url": row.get("url") or "",
            "location": row.get("location") or "",
            "text": text,
            "first_seen": "2026-10-30",
            "second_crawl": new_day.name,
        }
        if k in new_keys:
            item["label"] = "live"
            live.append(item)
        else:
            item["label"] = "closed"
            closed.append(item)

    out = ROOT / "data" / "labeled_temporal.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in closed + live:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_source = Counter(row["source"] for row in closed)
    by_company = Counter((row["source"], row["company"]) for row in closed)
    print(f"old={len(old_rows)} new={len(new_rows)} day={new_day.name}")
    print(f"temporal live={len(live)} closed={len(closed)}")
    print("closed by source:", dict(by_source))
    for (source, company), count in sorted(by_company.items()):
        print(f"  {source:12} {company:12} closed={count}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
