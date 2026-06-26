"""
LDA topic modelling per detected community (5 topics each), to characterise the
vocabulary of each community. Saves results/{subreddit}_lda_topics.json.
"""

import json

from bots import is_bot
import os
import re
from collections import defaultdict

from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS


SUBREDDITS  = ["apple", "google", "samsung"]
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
N_TOPICS    = 5
LDA_PASSES  = 20
LDA_SEED    = 42
TOP_N_WORDS = 10


EXTRA_STOPS = {"like", "just", "good", "really", "think", "use", "time",
               "want", "need", "also", "one", "still", "new", "get", "got",
               "would", "could", "know", "see", "well", "actually", "even",
               "yeah", "thing", "things", "way", "much", "lot", "ve", "don"}


def clean_tokenise(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens
              if t not in STOPWORDS
              and t not in EXTRA_STOPS
              and len(t) > 2]
    return tokens


def load_community_docs(subreddit: str) -> tuple[dict[int, list], dict]:
    """Return {community_id: [tokenised_doc, ...]} for GCC users."""
    # Load community partition
    with open(os.path.join(RESULTS_DIR, f"{subreddit}_communities.json")) as fh:
        comm_data = json.load(fh)
    partition = {k: v for k, v in comm_data["partition"].items()}  # user->comm (str keys)

    # Load raw comments
    path = os.path.join(DATA_DIR, f"{subreddit}_comments.jsonl")
    user_tokens: dict[str, list] = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            author = c.get("author", "")
            body   = c.get("body", "")
            if (author in partition
                    and not is_bot(author)
                    and body not in {"[deleted]", "[removed]"}):
                user_tokens[author].extend(clean_tokenise(body))

    # Group by community (each user is one document)
    comm_docs: dict[int, list] = defaultdict(list)
    for user, comm_str in partition.items():
        comm_id = int(comm_str)
        if user in user_tokens and user_tokens[user]:
            comm_docs[comm_id].append(user_tokens[user])

    return comm_docs, comm_data


def fit_lda(docs: list[list[str]], n_topics: int = N_TOPICS) -> tuple[LdaModel, corpora.Dictionary]:
    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=2, no_above=0.95)
    corpus     = [dictionary.doc2bow(doc) for doc in docs]
    lda = LdaModel(
        corpus      = corpus,
        id2word     = dictionary,
        num_topics  = n_topics,
        passes      = LDA_PASSES,
        random_state= LDA_SEED,
        alpha       = "auto",
        eta         = "auto",
    )
    return lda, dictionary


def run_lda(subreddit: str) -> dict:
    print(f"  r/{subreddit} - LDA topic modelling")

    comm_docs, comm_data = load_community_docs(subreddit)
    all_topics = {}

    for comm_id in sorted(comm_docs.keys()):
        docs = comm_docs[comm_id]
        print(f"  Community {comm_id}: {len(docs)} user-documents")
        if len(docs) < 3:
            print("    (too few documents - skipping LDA)")
            all_topics[comm_id] = []
            continue

        lda, dictionary = fit_lda(docs)
        topics_for_comm = []
        for t_idx in range(N_TOPICS):
            top_words = [w for w, _ in lda.show_topic(t_idx, topn=TOP_N_WORDS)]
            probs     = [float(p) for _, p in lda.show_topic(t_idx, topn=TOP_N_WORDS)]
            topics_for_comm.append({"topic": t_idx, "words": top_words, "probs": probs})

        all_topics[comm_id] = topics_for_comm

        # Print top words per topic
        for t in topics_for_comm:
            print(f"    Topic {t['topic']}: {', '.join(t['words'][:6])}")

    # Serialise
    result = {str(k): v for k, v in all_topics.items()}
    out_path = os.path.join(RESULTS_DIR, f"{subreddit}_lda_topics.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"  Results saved -> {out_path}")
    return all_topics


if __name__ == "__main__":
    for sub in SUBREDDITS:
        run_lda(sub)
    print("\nDone.")
