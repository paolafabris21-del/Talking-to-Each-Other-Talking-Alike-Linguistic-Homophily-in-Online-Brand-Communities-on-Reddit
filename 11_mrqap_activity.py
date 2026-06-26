"""
MRQAP with an activity control (H1b). Reports the partial correlation of
linguistic similarity with adjacency, controlling for users' comment counts,
using the double-semi-partialing permutation of Dekker, Krackhardt & Snijders
(2007). Saves results/mrqap_results.json.

    python 11_mrqap_activity.py [subreddit ...]
"""

import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

from bots import is_bot

SUBREDDITS_ALL = ["apple", "google", "samsung"]
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
N_PERMS     = 1000
RNG_SEED    = 42



def comment_counts(subreddit: str, gcc_nodes: set) -> dict:
    """Number of (filtered) comments per GCC user -- the activity covariate."""
    counts = defaultdict(int)
    path = os.path.join(DATA_DIR, f"{subreddit}_comments.jsonl")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            a, b = c.get("author", ""), c.get("body", "")
            if not is_bot(a) and b not in {"[deleted]", "[removed]"} and a in gcc_nodes:
                counts[a] += 1
    return counts


def partial_qap(Y, Sim, Act, n_perms=N_PERMS, seed=RNG_SEED):
    """
    DSP MRQAP. Returns the partial correlation of Sim with Y controlling for Act,
    plus a permutation null distribution and one-tailed p-value.
    All inputs are n x n symmetric matrices.
    """
    n = Y.shape[0]
    iu = np.triu_indices(n, k=1)
    y = Y[iu].astype(np.float64)
    s = Sim[iu].astype(np.float64)
    a = Act[iu].astype(np.float64)

    Xc = np.column_stack([np.ones_like(a), a])          # controls: [1, activity]
    Mc = np.linalg.inv(Xc.T @ Xc) @ Xc.T                # 2 x m projection operator

    def residualise(v):
        return v - Xc @ (Mc @ v)

    s_r = residualise(s)                                # similarity | controls
    y_r = residualise(y)                                # adjacency  | controls

    def pcorr(u, v):
        du, dv = np.sqrt(u @ u), np.sqrt(v @ v)
        return float((u @ v) / (du * dv)) if du > 0 and dv > 0 else 0.0

    obs = pcorr(s_r, y_r)                                # partial correlation

    # residual matrix of Sim (symmetric), to be permuted by node relabelling
    E = np.zeros((n, n), dtype=np.float32)
    E[iu] = s_r.astype(np.float32)
    E = E + E.T

    rng = np.random.default_rng(seed)
    null = np.empty(n_perms)
    for i in range(n_perms):
        p = rng.permutation(n)
        ep = E[p][:, p][iu].astype(np.float64)
        ep_r = residualise(ep)                          # re-residualise on controls
        null[i] = pcorr(ep_r, y_r)

    return {
        "partial_r": obs,
        "p_value": float(np.mean(null >= obs)),
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "n_perms": n_perms,
    }


def build_matrices(subreddit: str):
    with open(os.path.join(RESULTS_DIR, f"{subreddit}_users.json")) as fh:
        users = json.load(fh)
    with open(os.path.join(RESULTS_DIR, f"{subreddit}_network.pkl"), "rb") as fh:
        GCC = pickle.load(fh)["graph"]
    n = len(users)
    idx = {u: i for i, u in enumerate(users)}

    Y = np.zeros((n, n))
    for u, v in GCC.edges():
        if u in idx and v in idx:
            Y[idx[u], idx[v]] = Y[idx[v], idx[u]] = 1.0

    counts = comment_counts(subreddit, set(GCC.nodes()))
    logc = np.array([np.log(max(counts.get(u, 1), 1)) for u in users])
    Act = logc[:, None] + logc[None, :]                 # log(c_i)+log(c_j)
    np.fill_diagonal(Act, 0.0)

    Sim_content = np.load(os.path.join(RESULTS_DIR, f"{subreddit}_similarity_matrix.npy"))
    style_path = os.path.join(RESULTS_DIR, f"{subreddit}_style_similarity_matrix.npy")
    Sim_style = np.load(style_path) if os.path.exists(style_path) else None
    return Y, Act, Sim_content, Sim_style


if __name__ == "__main__":
    subs = sys.argv[1:] or SUBREDDITS_ALL
    out_path = os.path.join(RESULTS_DIR, "mrqap_results.json")
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as fh:
            results = json.load(fh)

    for sub in subs:
        print(f"\nr/{sub} - MRQAP (similarity | activity)")
        Y, Act, Sc, Ss = build_matrices(sub)
        rec = {}
        rc = partial_qap(Y, Sc, Act)
        rec["content"] = rc
        print(f"  Content: partial r = {rc['partial_r']:+.4f}  "
              f"(null {rc['null_mean']:+.5f} +/- {rc['null_std']:.5f}, p={rc['p_value']:.4f})")
        if Ss is not None:
            rs = partial_qap(Y, Ss, Act)
            rec["style"] = rs
            print(f"  Style  : partial r = {rs['partial_r']:+.4f}  "
                  f"(null {rs['null_mean']:+.5f} +/- {rs['null_std']:.5f}, p={rs['p_value']:.4f})")
        results[sub] = rec
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"  saved -> {os.path.basename(out_path)}")
