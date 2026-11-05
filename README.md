# Detecting live vs closed job postings from page text

CE 235 Artificial Intelligence & Data Science course project.  
Nimit Limbachiya, 24B0718. Individual submission.

Repository: https://github.com/Nimit-17/CE235-course-project

Classifies a rendered job-posting page as **live**, **closed**, or **uncertain** from page text. Many closed postings still return HTTP 200, so status codes alone are not enough.

## Dataset

Two public ATS crawls (Greenhouse, Lever, Ashby) for 10 companies:

| Crawl | Jobs |
|---|---|
| 2026-10-30 | 1082 |
| 2026-11-03 | 1072 |

Processed files live under `data/`. Temporal labels: jobs that left the board between crawls vs jobs still listed. Classifier training uses fetched closed-page text vs live ad text.

## Scripts

```text
python crawl.py
python label_from_diff.py
python fetch_closed_now.py
python train_model.py
python evaluate.py
python classify.py "The job you are looking for is no longer open."
python classify.py https://...
python check_links.py links.txt
```

Metrics: `results/metrics.json`. Saved DistilBERT weights: `model/` (Git LFS).

## Methods compared

1. Keyword heuristic on closed phrases  
2. TF-IDF + logistic regression  
3. Fine-tuned DistilBERT  

## Dependencies

See `requirements.txt`. First Playwright fetch may need `playwright install` (or system Edge).

## Moodle

Upload `report.pdf` and `results_video.mp4` on Moodle. Full code, processed data, and model are in this repository (Moodle file-size limit is 20 MB).
