import json
import sys
from pathlib import Path

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TOP_CHARS = 700


def load():
    labels = json.loads((MODEL_DIR / "labels.json").read_text(encoding="utf-8"))
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model, labels


def classify(text, tokenizer, model, labels):
    cleaned = " ".join(text.split())
    if len(cleaned) < 12:
        return {"label": "uncertain", "live": None, "closed": None}
    encoded = tokenizer(
        cleaned,
        truncation=True,
        max_length=labels["max_length"],
        return_tensors="pt",
    )
    with torch.no_grad():
        probs = torch.softmax(model(**encoded).logits, dim=-1)[0]
    live = float(probs[0])
    closed = float(probs[1])
    top = max(live, closed)
    if top < labels["uncertain_below"]:
        name = "uncertain"
    else:
        name = "live" if live >= closed else "closed"
    return {"label": name, "live": round(live, 3), "closed": round(closed, 3)}


def _focus_closed_banner(text):
    """Keep the closed notice; drop the openings list that follows on soft-closed boards."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    cut_markers = (
        "current openings",
        "view all jobs",
        "view more jobs",
        "you may also",
        "create a job alert",
        "similar jobs",
        "other openings",
    )
    cut_at = None
    for marker in cut_markers:
        idx = lowered.find(marker)
        if idx >= 20:
            cut_at = idx if cut_at is None else min(cut_at, idx)
    focused = cleaned[:cut_at].strip() if cut_at is not None else cleaned[:TOP_CHARS]
    return focused[:TOP_CHARS]


def _useful_dialog(text):
    """Drop Greenhouse phone-country dropdowns and other noisy widgets."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    if "results found" in lowered and ("afghanistan" in lowered or "ål and" in lowered or "aland" in lowered):
        return ""
    if "afghanistan" in lowered and "+93" in cleaned:
        return ""
    if len(cleaned) > 400 and lowered.count("+") > 20:
        return ""
    return cleaned


def _banner_text(fetched):
    dialog = _useful_dialog(fetched.get("dialog_text") or "")
    early = fetched.get("early_text") or ""
    # Prefer early page text; only keep dialog when it looks like a closed toast.
    parts = []
    if dialog:
        parts.append(dialog)
    if early:
        parts.append(early)
    if not parts and fetched.get("text"):
        parts.append(fetched["text"][:TOP_CHARS])
    return _focus_closed_banner(" ".join(parts))


def classify_fetched(fetched, tokenizer, model, labels):
    """Two-pass: prefer closed from early/dialog/top; else use full page."""
    if fetched.get("posting_gone"):
        banner = _banner_text(fetched) or fetched.get("text") or "The job posting was removed."
        result = classify(banner[:TOP_CHARS], tokenizer, model, labels)
        # Redirect off the job URL is strong evidence the posting is gone.
        if result["label"] != "closed":
            result = {
                "label": "closed",
                "live": result.get("live"),
                "closed": max(result.get("closed") or 0.0, 0.9),
            }
        result["reason"] = "posting_gone_redirect"
        result["banner"] = classify(banner[:TOP_CHARS], tokenizer, model, labels)
        result["full"] = classify(fetched.get("text") or banner, tokenizer, model, labels)
        return result

    if fetched.get("blocked") or not fetched.get("ok"):
        return {
            "label": "uncertain",
            "live": None,
            "closed": None,
            "reason": "fetch_blocked_or_empty",
        }

    full_text = fetched.get("text") or ""
    banner = _banner_text(fetched) or full_text[:TOP_CHARS]
    top = classify(banner[:TOP_CHARS], tokenizer, model, labels)
    full = classify(full_text, tokenizer, model, labels)

    # Soft-close fix: trust a closed top read only when the banner text is a real notice,
    # not a random widget, and confidence is solid.
    banner_l = banner.lower()
    banner_looks_closed = any(
        phrase in banner_l
        for phrase in (
            "no longer open",
            "no longer available",
            "might have closed",
            "job not found",
            "not accepting",
            "doesn't exist",
            "does not exist",
            "has been removed",
            "position filled",
        )
    )
    if (
        top["label"] == "closed"
        and (top.get("closed") or 0) >= 0.6
        and banner_looks_closed
    ):
        chosen = dict(top)
        chosen["reason"] = "top_closed_overrides_full"
    elif full["label"] == "closed":
        chosen = dict(full)
        chosen["reason"] = "full_closed"
    else:
        chosen = dict(full)
        chosen["reason"] = "full_page"

    chosen["banner"] = top
    chosen["full"] = full
    return chosen


def classify_url(url, tokenizer=None, model=None, labels=None):
    from fetch_page import fetch_page

    if tokenizer is None:
        tokenizer, model, labels = load()
    fetched = fetch_page(url)
    result = classify_fetched(fetched, tokenizer, model, labels)
    result["method"] = fetched.get("method")
    result["final_url"] = fetched.get("final_url")
    result["error"] = fetched.get("error")
    result["preview"] = (fetched.get("early_text") or fetched.get("text") or "")[:220]
    result["dialog_preview"] = (fetched.get("dialog_text") or "")[:160]
    return result


def main():
    tokenizer, model, labels = load()
    if len(sys.argv) > 1 and sys.argv[1].startswith("http"):
        result = classify_url(sys.argv[1], tokenizer, model, labels)
        print(result["label"], result)
        return
    if len(sys.argv) > 1:
        result = classify(" ".join(sys.argv[1:]), tokenizer, model, labels)
        print(result["label"], result)
        return
    samples = [
        ("data/2026-10-30/jobs.jsonl", "saved open ad"),
        ("data/2026-10-30/closed.jsonl", "saved closed page"),
    ]
    for path, title in samples:
        file_path = ROOT / path
        if not file_path.exists():
            continue
        row = json.loads(file_path.read_text(encoding="utf-8").splitlines()[0])
        result = classify(row["text"], tokenizer, model, labels)
        print(title, result["label"], {k: result[k] for k in ("live", "closed")})


if __name__ == "__main__":
    main()
