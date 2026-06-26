"""
TF-IDF user vectors and pairwise cosine similarity per subreddit. Each user is
one document (all their comments). Saves the similarity matrix and the matching
GCC-ordered user list to results/.
"""

import json
import os
import pickle
import re
from collections import defaultdict

import numpy as np

from bots import is_bot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SUBREDDITS  = ["apple", "google", "samsung"]
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

TFIDF_PARAMS = dict(
    min_df        = 2,       # ignore terms appearing in < 2 user-docs
    max_df        = 0.90,    # ignore very common terms (top 10%)
    sublinear_tf  = True,    # apply log(1+tf)
    ngram_range   = (1, 1),  # unigrams only
    stop_words    = "english",
    max_features  = 10_000,
)

MIN_COMMENTS = 5


def clean_text(text: str) -> str:
    """Basic Reddit text cleaning."""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)          # remove URLs
    text = re.sub(r"/u/\w+", " ", text)           # remove user mentions
    text = re.sub(r"/r/\w+", " ", text)           # remove subreddit mentions
    text = re.sub(r"[^a-z\s]", " ", text)         # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_user_documents(subreddit: str, gcc_nodes: set) -> dict[str, str]:
    """Load and aggregate all comments per GCC user."""
    path = os.path.join(DATA_DIR, f"{subreddit}_comments.jsonl")
    user_docs: dict[str, list[str]] = defaultdict(list)

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            author = c.get("author", "")
            body   = c.get("body", "")
            if (not is_bot(author)
                    and body not in {"[deleted]", "[removed]"}
                    and author in gcc_nodes):
                user_docs[author].append(clean_text(body))

    return {u: " ".join(docs) for u, docs in user_docs.items() if docs}


def compute_similarity(subreddit: str) -> None:
    print(f"  r/{subreddit} - TF-IDF & cosine similarity")

    # Load GCC
    pkl_path = os.path.join(RESULTS_DIR, f"{subreddit}_network.pkl")
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    GCC = data["graph"]
    gcc_nodes = set(GCC.nodes())
    print(f"  GCC nodes: {len(gcc_nodes):,}")

    # Build per-user documents
    user_docs = load_user_documents(subreddit, gcc_nodes)
    # Keep only users present in GCC
    users = [u for u in GCC.nodes() if u in user_docs]
    docs  = [user_docs[u] for u in users]
    print(f"  Users with text: {len(users):,}")

    # Fit TF-IDF
    vectoriser = TfidfVectorizer(**TFIDF_PARAMS)
    X = vectoriser.fit_transform(docs)   # shape: (n_users, n_features)
    print(f"  TF-IDF matrix  : {X.shape[0]} users x {X.shape[1]} features")

    # Pairwise cosine similarity
    sim_matrix = cosine_similarity(X)    # shape: (n_users, n_users)
    print(f"  Similarity matrix shape: {sim_matrix.shape}")
    # Mask diagonal for descriptive stats
    np.fill_diagonal(sim_matrix, np.nan)
    print(f"  Mean off-diagonal similarity: {np.nanmean(sim_matrix):.4f}")
    print(f"  Std off-diagonal similarity : {np.nanstd(sim_matrix):.4f}")
    np.fill_diagonal(sim_matrix, 0.0)    # restore 0 on diagonal for downstream use

    # Save
    np.save(os.path.join(RESULTS_DIR, f"{subreddit}_similarity_matrix.npy"), sim_matrix)
    with open(os.path.join(RESULTS_DIR, f"{subreddit}_users.json"), "w") as fh:
        json.dump(users, fh)
    # Also save vectoriser and X for LDA step
    with open(os.path.join(RESULTS_DIR, f"{subreddit}_tfidf.pkl"), "wb") as fh:
        pickle.dump({"vectoriser": vectoriser, "X": X, "users": users}, fh)

    print("  Saved similarity matrix and user list.")


if __name__ == "__main__":
    for sub in SUBREDDITS:
        compute_similarity(sub)
    print("\nDone.")
