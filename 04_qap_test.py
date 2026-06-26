"""
QAP correlation test for linguistic homophily (H1).

Correlates the binary adjacency matrix of each GCC with the pairwise TF-IDF
cosine-similarity matrix, and assesses significance with 1,000 node-label
permutations (Krackhardt, 1988). Saves the observed coefficient, the permutation
summary, and the full null distribution (used by 09_figures_final.py, Figure 2).

    python 04_qap_test.py
"""

import json
import os
import pickle

import numpy as np

SUBREDDITS  = ["apple", "google", "samsung"]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
N_PERMS     = 1000
RNG_SEED    = 42


def upper_tri(M):
    return M[np.triu_indices(M.shape[0], k=1)]


def qap_test(sim, adj, n_perms=N_PERMS, seed=RNG_SEED):
    """
    Observed Pearson r between similarity and adjacency over all dyads, plus the
    permutation null distribution.

    Permuting rows+cols of the binary adjacency keeps the number of edges fixed,
    so the dyadic mean and sd of the adjacency are invariant and only the
    cross-product term changes. For a node permutation q the permuted
    cross-product equals the sum over edges (a,b) of sim[q[a], q[b]], which makes
    each permutation O(E) rather than O(n^2), giving the same statistic.
    """
    n = adj.shape[0]
    x = upper_tri(sim).astype(np.float64)
    m = x.size
    mean_x, sd_x = x.mean(), x.std()

    ea, eb = np.where(np.triu(adj, k=1) > 0)   # edge list
    n_edges = ea.size
    mean_y = n_edges / m
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

    p_value = float(np.mean(null >= obs))
    return obs, p_value, null


def run_qap(subreddit):
    print(f"\nr/{subreddit} - QAP test")

    sim = np.load(os.path.join(RESULTS_DIR, f"{subreddit}_similarity_matrix.npy"))
    with open(os.path.join(RESULTS_DIR, f"{subreddit}_users.json")) as fh:
        users = json.load(fh)
    with open(os.path.join(RESULTS_DIR, f"{subreddit}_network.pkl"), "rb") as fh:
        GCC = pickle.load(fh)["graph"]

    n = len(users)
    adj = np.zeros((n, n))
    idx = {u: i for i, u in enumerate(users)}
    for u, v in GCC.edges():
        if u in idx and v in idx:
            adj[idx[u], idx[v]] = adj[idx[v], idx[u]] = 1.0

    obs, p_value, null = qap_test(sim, adj)
    np.save(os.path.join(RESULTS_DIR, f"{subreddit}_qap_null.npy"), null)

    print(f"  observed r = {obs:+.4f}")
    print(f"  null mean / sd = {null.mean():.5f} / {null.std():.5f}")
    print(f"  p (one-tailed) = {p_value:.4f}")
    return {
        "observed_r": obs,
        "p_value": p_value,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "n_perms": N_PERMS,
    }


if __name__ == "__main__":
    results = {sub: run_qap(sub) for sub in SUBREDDITS}
    with open(os.path.join(RESULTS_DIR, "qap_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    print("\nSummary")
    for sub, r in results.items():
        sig = "**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else "")
        print(f"  r/{sub:<8} r={r['observed_r']:+.4f}  p={r['p_value']:.4f} {sig}")
