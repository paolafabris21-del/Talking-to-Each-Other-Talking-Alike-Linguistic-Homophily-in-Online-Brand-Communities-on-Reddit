"""
Comment-level sentiment with cardiffnlp/twitter-roberta-base-sentiment, then a
user-level majority vote. Tests whether sentiment varies across communities
(chi-square + Cramer's V). Saves results/{subreddit}_sentiment.json.

The model (~500 MB) downloads on first run. It uses the Apple MPS or CUDA backend
when available, otherwise the CPU. Comments are streamed from disk and classified
in bounded chunks, so memory stays low even on the full six-month dataset; progress
is printed per chunk and each subreddit is saved on completion, so an interrupted
run resumes from where it stopped.
"""

import json
import os
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import chi2_contingency
from transformers import pipeline

from bots import is_bot

SUBREDDITS  = ["apple", "google", "samsung"]
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
MODEL_NAME  = "cardiffnlp/twitter-roberta-base-sentiment"
BATCH_SIZE  = 32
CHUNK       = 2000        # comments classified between progress prints
MAX_CHARS   = 280
LABEL_MAP   = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
SENTIMENTS  = ["Positive", "Neutral", "Negative"]


def pick_device():
    """Apple MPS or CUDA if available, otherwise CPU (-1)."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return -1


def load_pipeline():
    device = pick_device()
    print(f"Loading sentiment model (device={device})...")
    clf = pipeline(
        "text-classification",
        model=MODEL_NAME, tokenizer=MODEL_NAME,
        truncation=True, max_length=128, top_k=1,
        device=device, batch_size=BATCH_SIZE,
    )
    print("Model loaded.")
    return clf


def qualifies(comment, partition):
    a, b = comment.get("author", ""), comment.get("body", "")
    return (a in partition and not is_bot(a)
            and b not in {"[deleted]", "[removed]"} and len(b.split()) >= 3)


def count_comments(path, partition):
    total = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if qualifies(json.loads(line), partition):
                total += 1
    return total


def run_sentiment(subreddit, clf):
    out_path = os.path.join(RESULTS_DIR, f"{subreddit}_sentiment.json")
    if os.path.exists(out_path):
        print(f"[{subreddit}] already done - skipping")
        return

    with open(os.path.join(RESULTS_DIR, f"{subreddit}_communities.json")) as fh:
        partition = json.load(fh)["partition"]

    path = os.path.join(DATA_DIR, f"{subreddit}_comments.jsonl")
    total = count_comments(path, partition)
    print(f"[{subreddit}] {total:,} comments to classify")

    user_counts = defaultdict(Counter)   # user -> label counts
    buf_text, buf_owner, done = [], [], 0

    def flush():
        nonlocal done
        if not buf_text:
            return
        for owner, pred in zip(buf_owner, clf(buf_text)):
            top = pred[0] if isinstance(pred, list) else pred
            user_counts[owner][LABEL_MAP[top["label"]]] += 1
        done += len(buf_text)
        print(f"  [{subreddit}] {done:,}/{total:,} classified", flush=True)
        buf_text.clear()
        buf_owner.clear()

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            if qualifies(c, partition):
                buf_text.append(c["body"][:MAX_CHARS])
                buf_owner.append(c["author"])
                if len(buf_text) >= CHUNK:
                    flush()
    flush()

    user_sentiment = {u: cnt.most_common(1)[0][0] for u, cnt in user_counts.items()}

    comm_counts = defaultdict(Counter)
    for user, label in user_sentiment.items():
        comm_counts[int(partition[user])][label] += 1
    comm_ids = sorted(comm_counts)
    table = np.array([[comm_counts[c][s] for s in SENTIMENTS] for c in comm_ids])

    chi2, pval, dof, _ = chi2_contingency(table)
    n, k = table.sum(), min(table.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * k))) if n * k > 0 else 0.0
    print(f"[{subreddit}] chi2={chi2:.2f} dof={dof} p={pval:.4f} V={cramers_v:.3f}")

    result = {
        "contingency_table": table.tolist(),
        "communities": comm_ids,
        "chi2": float(chi2),
        "p_value": float(pval),
        "dof": int(dof),
        "cramers_v": cramers_v,
    }
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    with open(os.path.join(RESULTS_DIR, f"{subreddit}_user_sentiment.json"), "w") as fh:
        json.dump(user_sentiment, fh)
    print(f"[{subreddit}] saved -> {out_path}")


if __name__ == "__main__":
    clf = load_pipeline()
    for sub in SUBREDDITS:
        run_sentiment(sub, clf)
