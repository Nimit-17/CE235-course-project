import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

ROOT = Path(__file__).resolve().parent
SEED = 7
MAX_LENGTH = 160
CLOSED_PHRASES = [
    r"no longer open",
    r"no longer available",
    r"might have closed",
    r"has been removed",
    r"job not found",
    r"page not found",
    r"position has been filled",
    r"position filled",
    r"couldn't find anything",
    r"this role is closed",
    r"role is closed",
    r"not accepting applications",
]


def load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def clean(text):
    return " ".join((text or "").split())


def looks_closed_phrase(text):
    lowered = text.lower()
    return any(re.search(p, lowered) for p in CLOSED_PHRASES)


def load_unique():
    closed = []
    seen = set()
    for path in [
        ROOT / "data" / "closed_from_diff.jsonl",
        ROOT / "data" / "2026-10-30" / "closed.jsonl",
    ]:
        for row in load_jsonl(path):
            text = clean(row.get("text"))
            if len(text) < 40 or text in seen:
                continue
            if not looks_closed_phrase(text):
                continue
            seen.add(text)
            closed.append(
                {
                    "text": text,
                    "label": "closed",
                    "label_id": 1,
                    "source": row.get("source"),
                    "company": row.get("company"),
                }
            )

    live = []
    seen = set()
    for path in [
        ROOT / "data" / "2026-11-03" / "jobs.jsonl",
        ROOT / "data" / "2026-10-30" / "rendered_live.jsonl",
    ]:
        for row in load_jsonl(path):
            text = clean(row.get("text"))
            if len(text) < 40 or text in seen:
                continue
            if looks_closed_phrase(text) or "security verification" in text.lower():
                continue
            seen.add(text)
            live.append(
                {
                    "text": text,
                    "label": "live",
                    "label_id": 0,
                    "source": row.get("source"),
                    "company": row.get("company"),
                }
            )
    random.seed(SEED)
    random.shuffle(live)
    return closed, live


def scores(y_true, y_pred):
    return {
        "n": len(y_true),
        "closed_n": int(sum(1 for y in y_true if y == 1)),
        "live_n": int(sum(1 for y in y_true if y == 0)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "live_precision": round(float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)), 3),
        "live_recall": round(float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)), 3),
        "live_f1": round(float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)), 3),
        "closed_precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 3),
        "closed_recall": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 3),
        "closed_f1": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 3),
    }


def keyword_predict(texts):
    return [1 if looks_closed_phrase(text) else 0 for text in texts]


def tfidf_loo(closed, live_train):
    """Leave-one-closed-template-out for fair closed-class score."""
    preds = []
    truths = []
    for i, held in enumerate(closed):
        train_closed = [row for j, row in enumerate(closed) if j != i]
        train = train_closed + live_train
        test = [held] + live_train[:30]
        vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
        x_train = vectorizer.fit_transform([row["text"] for row in train])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(x_train, [row["label_id"] for row in train])
        x_test = vectorizer.transform([row["text"] for row in test])
        pred = clf.predict(x_test)
        preds.extend(pred.tolist())
        truths.extend([row["label_id"] for row in test])
    return scores(truths, preds)


def distilbert_predict(texts, model_dir):
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    preds = []
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            preds.append(int(model(**encoded).logits.argmax(dim=-1).item()))
    return preds


def by_source(rows, pred):
    out = {}
    for source in sorted({row["source"] for row in rows if row.get("source")}):
        idx = [i for i, row in enumerate(rows) if row.get("source") == source]
        y = [rows[i]["label_id"] for i in idx]
        p = [pred[i] for i in idx]
        out[source] = {
            "n": len(idx),
            "closed": sum(1 for v in y if v == 1),
            "accuracy": round(float(accuracy_score(y, p)), 3),
            "live_f1": round(float(f1_score(y, p, pos_label=0, zero_division=0)), 3),
            "closed_f1": round(float(f1_score(y, p, pos_label=1, zero_division=0)), 3),
        }
    return out


def main():
    closed, live = load_unique()
    live_test = live[:200]
    live_train = live[200:500]
    eval_rows = closed + live_test
    y_true = [row["label_id"] for row in eval_rows]
    texts = [row["text"] for row in eval_rows]
    print(f"unique_closed={len(closed)} live_pool={len(live)} eval_n={len(eval_rows)}")

    results = {}
    kw = keyword_predict(texts)
    results["keyword"] = scores(y_true, kw)
    results["keyword"]["by_source"] = by_source(eval_rows, kw)
    print("keyword", {k: results["keyword"][k] for k in results["keyword"] if k != "by_source"})

    results["tfidf_logreg_loo"] = tfidf_loo(closed, live_train)
    print("tfidf_loo", results["tfidf_logreg_loo"])

    model_dir = ROOT / "model"
    if model_dir.exists():
        db = distilbert_predict(texts, model_dir)
        results["distilbert"] = scores(y_true, db)
        results["distilbert"]["by_source"] = by_source(eval_rows, db)
        print("distilbert", {k: results["distilbert"][k] for k in results["distilbert"] if k != "by_source"})
        # Note: DistilBERT saw these closed templates in training; keyword did not need training.
        results["distilbert"]["note"] = (
            "Closed templates were seen in training; live pages are held out. "
            "Treat closed scores as train recall, live scores as holdout."
        )

    temporal = load_jsonl(ROOT / "data" / "labeled_temporal.jsonl")
    summary = {
        "crawls": {
            "2026-10-30": 1082,
            "2026-11-03": len(load_jsonl(ROOT / "data" / "2026-11-03" / "jobs.jsonl")),
        },
        "temporal_labels": dict(Counter(row["label"] for row in temporal)),
        "closed_urls_with_closed_text": len(load_jsonl(ROOT / "data" / "closed_from_diff.jsonl")),
        "unique_closed_page_templates": len(closed),
        "results": results,
        "note": (
            "Temporal closed = left the board between crawls. "
            "Classifier uses fetched closed page text vs live ad text."
        ),
    }
    out = ROOT / "results" / "metrics.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
