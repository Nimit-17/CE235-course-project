import json
import random
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MAX_LENGTH = 160
SEED = 7
LIVE_SAMPLE = 300
CLOSED_PHRASES = (
    "no longer open",
    "no longer available",
    "this job is no longer available",
    "might have closed",
    "has been removed",
    "job not found",
    "job you requested was not found",
    "page not found",
    "position has been filled",
    "couldn't find anything",
    "doesn't exist",
    "does not exist",
)


class PageSet(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(row["label_id"], dtype=torch.long)
        return item


def load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def looks_closed(text):
    lowered = text.lower()
    return any(phrase in lowered for phrase in CLOSED_PHRASES)


def looks_blocked(text):
    lowered = text.lower()
    return "security verification" in lowered or "just a moment" in lowered


def unique_rows(rows, label):
    kept = []
    seen = set()
    for row in rows:
        text = " ".join((row.get("text") or "").split())
        if len(text) < 40 or text in seen:
            continue
        if label == "closed" and not looks_closed(text):
            continue
        if label == "live" and (looks_closed(text) or looks_blocked(text)):
            continue
        seen.add(text)
        kept.append(
            {
                "text": text,
                "label": label,
                "label_id": 0 if label == "live" else 1,
                "source": row.get("source"),
                "company": row.get("company"),
            }
        )
    return kept


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    closed_rows = unique_rows(
        load_jsonl(ROOT / "data" / "closed_from_diff.jsonl")
        + load_jsonl(ROOT / "data" / "2026-10-30" / "closed.jsonl")
        + load_jsonl(ROOT / "data" / "soft_closed.jsonl"),
        "closed",
    )
    live_pool = unique_rows(
        load_jsonl(ROOT / "data" / "2026-11-03" / "jobs.jsonl")
        + load_jsonl(ROOT / "data" / "2026-10-30" / "rendered_live.jsonl"),
        "live",
    )
    if not closed_rows:
        raise SystemExit("No closed pages found")

    random.shuffle(live_pool)
    n_hold = max(40, int(0.2 * min(len(live_pool), LIVE_SAMPLE + 80)))
    held_out = live_pool[:n_hold]
    live_rows = live_pool[n_hold : n_hold + LIVE_SAMPLE]

    copies = max(1, len(live_rows) // max(1, len(closed_rows)))
    train_rows = live_rows + closed_rows * copies
    random.shuffle(train_rows)
    print(
        f"train live={len(live_rows)} closed_unique={len(closed_rows)} "
        f"closed_copies={copies} holdout_live={len(held_out)}"
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
    )
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loader = DataLoader(PageSet(train_rows, tokenizer), batch_size=8, shuffle=True)

    for epoch in range(1, 4):
        total = 0.0
        seen = 0
        for batch in loader:
            labels = batch.pop("labels")
            optimizer.zero_grad()
            logits = model(**batch).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * labels.size(0)
            seen += labels.size(0)
        print(f"epoch {epoch} loss {total / seen:.4f}")

    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in DataLoader(PageSet(held_out, tokenizer), batch_size=8):
            labels = batch.pop("labels")
            preds = model(**batch).logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
    print(f"holdout live accuracy {correct / len(held_out):.3f} on {len(held_out)}")

    closed_correct = 0
    with torch.no_grad():
        for batch in DataLoader(PageSet(closed_rows, tokenizer), batch_size=8):
            labels = batch.pop("labels")
            preds = model(**batch).logits.argmax(dim=-1)
            closed_correct += (preds == labels).sum().item()
    print(f"closed unique recalled {closed_correct}/{len(closed_rows)}")

    MODEL_DIR.mkdir(exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    (MODEL_DIR / "labels.json").write_text(
        json.dumps(
            {
                "id2label": {"0": "live", "1": "closed"},
                "uncertain_below": 0.6,
                "max_length": MAX_LENGTH,
            }
        ),
        encoding="utf-8",
    )
    print(f"saved model to {MODEL_DIR}")


if __name__ == "__main__":
    main()
