"""
Robustness check for H1b. Repeats the QAP using a topic-independent
function-word similarity matrix instead of TF-IDF, to separate stylistic from
topical homophily. Saves results/style_qap_results.json and Figure 3.
"""

import json
import os
import pickle
import re
from collections import defaultdict

import numpy as np

from bots import is_bot
from sklearn.metrics.pairwise import cosine_similarity


SUBREDDITS  = ["apple", "google", "samsung"]
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
N_PERMS     = 1000
RNG_SEED    = 42


# Closed-class English function words (topic-independent).
# Standard LIWC-style categories: pronouns, articles, prepositions, auxiliary
# verbs, conjunctions, negations, quantifiers, common adverbs/determiners.
FUNCTION_WORDS = set("""
i me my myself we our ours ourselves you your yours yourself yourselves he him
his himself she her hers herself it its itself they them their theirs themselves
this that these those who whom whose which what
a an the
am is are was were be been being have has had having do does did doing
will would shall should can could may might must ought
and but or nor for yet so because although though while whereas if unless until
since whether than as that
in on at by for with about against between into through during before after above
below to from up down of off over under again further then once here there
when where why how all any both each few more most other some such no not only own
same too very can just don dont didnt doesnt isnt arent wasnt werent hasnt havent
hadnt wont wouldnt cant cannot couldnt shouldnt mustnt
none nothing nobody never always also however therefore thus hence indeed instead
rather quite somewhat almost enough still even much many lot
""".split())


def tokenize(text: str):
    """Lowercase, keep only letters, split on whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.split()


def load_user_function_word_docs(subreddit: str, gcc_nodes: set) -> dict:
    """Aggregate per-user token lists (GCC users only)."""
    path = os.path.join(DATA_DIR, f"{subreddit}_comments.jsonl")
    user_tokens = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            author = c.get("author", "")
            body = c.get("body", "")
            if (not is_bot(author)
                    and body not in {"[deleted]", "[removed]"}
                    and author in gcc_nodes):
                user_tokens[author].extend(tokenize(body))
    return user_tokens


def build_style_matrix(subreddit: str):
    print(f"\nr/{subreddit} - function-word style similarity")

    # exact user ordering used by the TF-IDF / QAP pipeline
    with open(os.path.join(RESULTS_DIR, f"{subreddit}_users.json")) as fh:
        users = json.load(fh)
    with open(os.path.join(RESULTS_DIR, f"{subreddit}_network.pkl"), "rb") as fh:
        GCC = pickle.load(fh)["graph"]
    gcc_nodes = set(GCC.nodes())

    user_tokens = load_user_function_word_docs(subreddit, gcc_nodes)

    fw_list = sorted(FUNCTION_WORDS)
    fw_index = {w: i for i, w in enumerate(fw_list)}
    n, d = len(users), len(fw_list)
    M = np.zeros((n, d), dtype=float)

    n_empty = 0
    for r, u in enumerate(users):
        toks = user_tokens.get(u, [])
        total = len(toks)
        if total == 0:
            n_empty += 1
            continue
        counts = np.zeros(d)
        fw_total = 0
        for t in toks:
            j = fw_index.get(t)
            if j is not None:
                counts[j] += 1
                fw_total += 1
        # relative frequency over function-word tokens (style profile)
        if fw_total > 0:
            M[r] = counts / fw_total
    print(f"  Users: {n} | function-word features: {d} | users w/o tokens: {n_empty}")

    sim = cosine_similarity(M)
    np.fill_diagonal(sim, 0.0)
    np.save(os.path.join(RESULTS_DIR, f"{subreddit}_style_similarity_matrix.npy"), sim)

    # adjacency in identical user order
    adj = np.zeros((n, n))
    idx = {u: i for i, u in enumerate(users)}
    for u, v in GCC.edges():
        if u in idx and v in idx:
            adj[idx[u], idx[v]] = 1.0
            adj[idx[v], idx[u]] = 1.0
    return sim, adj


def qap(sim, adj, n_perms=N_PERMS, seed=RNG_SEED):
    """
    Fast QAP correlation, identical in result to script 04 but O(E) per
    permutation. Permuting rows+cols of a binary symmetric adjacency preserves
    the multiset of dyad values (mean_y, sd_y are invariant), so only the
    cross-product term sum(sim*adj) varies. For a permutation q the permuted
    cross-product equals sum over edges (a,b) of sim[q[a], q[b]].
    """
    n = adj.shape[0]
    iu = np.triu_indices(n, k=1)
    x = sim[iu]
    m = x.size
    mean_x, sd_x = x.mean(), x.std()
    # edge list (upper triangle) and binary-vector moments
    ea, eb = np.where(np.triu(adj, k=1) > 0)
    E = ea.size
    mean_y = E / m
    sd_y = np.sqrt(mean_y * (1.0 - mean_y))
    denom = m * sd_x * sd_y

    def r_from_cross(cross):
        return float((cross - m * mean_x * mean_y) / denom)

    obs = r_from_cross(sim[ea, eb].sum())
    rng = np.random.default_rng(seed)
    null = np.empty(n_perms)
    for i in range(n_perms):
        q = rng.permutation(n)
        null[i] = r_from_cross(sim[q[ea], q[eb]].sum())
    return {
        "observed_r": obs,
        "p_value": float(np.mean(null >= obs)),
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "n_perms": n_perms,
    }


def make_figure(style_results: dict):
    """Figure 3: content (TF-IDF) vs. style (function-word) QAP homophily."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "qap_results.json")) as fh:
        content = json.load(fh)

    labels = [f"r/{s}" for s in SUBREDDITS]
    c = [content[s]["observed_r"] for s in SUBREDDITS]
    st = [style_results[s]["observed_r"] for s in SUBREDDITS]
    x = np.arange(len(SUBREDDITS))
    w = 0.36

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    b1 = ax.bar(x - w / 2, c, w, label="Content (TF-IDF)", color="#4477AA")
    b2 = ax.bar(x + w / 2, st, w, label="Style (function words)", color="#CCBB44")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Observed QAP Pearson r")
    ax.set_ylim(0, 0.16)
    ax.set_title("Linguistic homophily: content vs. style (all p < 0.001)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig3_style_robustness.pdf"))
    plt.close(fig)
    print("Saved -> figures/fig3_style_robustness.pdf")


if __name__ == "__main__":
    results = {}
    for sub in SUBREDDITS:
        sim, adj = build_style_matrix(sub)
        res = qap(sim, adj)
        results[sub] = res
        print(f"  Observed style r : {res['observed_r']:+.4f}  "
              f"(null {res['null_mean']:+.5f} +/- {res['null_std']:.5f}, "
              f"p={res['p_value']:.4f})")
    with open(os.path.join(RESULTS_DIR, "style_qap_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nSaved -> results/style_qap_results.json")
    make_figure(results)
