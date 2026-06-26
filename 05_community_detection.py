"""
Louvain community detection on the weighted GCC. Runs 10 times with different
seeds, keeps the best partition by modularity, and measures stability with mean
ARI and NMI across runs. Saves results/{subreddit}_communities.json.
"""

import json
import os
import pickle
import itertools

import community as community_louvain          # python-louvain
import networkx as nx
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


SUBREDDITS  = ["apple", "google", "samsung"]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
N_RUNS      = 10
BASE_SEED   = 42


def labels_from_partition(partition: dict, nodes: list) -> np.ndarray:
    return np.array([partition[n] for n in nodes])


def run_louvain_multi(G: nx.Graph, n_runs: int = N_RUNS) -> tuple[dict, float, list]:
    """
    Run Louvain n_runs times; return best partition, its modularity, and all partitions.
    """
    partitions   = []
    modularities = []
    for i in range(n_runs):
        part = community_louvain.best_partition(G, weight="weight", random_state=BASE_SEED + i)
        mod  = community_louvain.modularity(part, G, weight="weight")
        partitions.append(part)
        modularities.append(mod)

    best_idx = int(np.argmax(modularities))
    return partitions[best_idx], modularities[best_idx], partitions


def stability_scores(partitions: list, nodes: list) -> tuple[float, float]:
    """Mean ARI and NMI across all pairwise run comparisons."""
    ari_scores = []
    nmi_scores = []
    for p1, p2 in itertools.combinations(partitions, 2):
        l1 = labels_from_partition(p1, nodes)
        l2 = labels_from_partition(p2, nodes)
        ari_scores.append(adjusted_rand_score(l1, l2))
        nmi_scores.append(normalized_mutual_info_score(l1, l2))
    return float(np.mean(ari_scores)), float(np.mean(nmi_scores))


def detect_communities(subreddit: str) -> dict:
    print(f"  r/{subreddit} - Louvain community detection")

    with open(os.path.join(RESULTS_DIR, f"{subreddit}_network.pkl"), "rb") as fh:
        data = pickle.load(fh)
    GCC   = data["graph"]
    nodes = list(GCC.nodes())

    best_part, best_mod, all_parts = run_louvain_multi(GCC)
    n_communities = len(set(best_part.values()))
    mean_ari, mean_nmi = stability_scores(all_parts, nodes)

    print(f"  Communities detected : {n_communities}")
    print(f"  Best modularity      : {best_mod:.4f}")
    print(f"  Mean ARI (10 runs)   : {mean_ari:.4f}")
    print(f"  Mean NMI (10 runs)   : {mean_nmi:.4f}")

    # Per-community stats
    comm_to_nodes: dict[int, list] = {}
    for node, comm in best_part.items():
        comm_to_nodes.setdefault(comm, []).append(node)

    comm_stats = {}
    for comm_id, members in comm_to_nodes.items():
        sub_G = GCC.subgraph(members)
        degs  = [d for _, d in GCC.degree(members)]
        comm_stats[comm_id] = {
            "size"            : len(members),
            "pct"             : len(members) / len(nodes) * 100,
            "internal_edges"  : sub_G.number_of_edges(),
            "avg_internal_deg": float(np.mean(degs)),
        }
        print(f"    Community {comm_id:2d}: {len(members):4d} nodes ({len(members)/len(nodes)*100:.1f}%)")

    result = {
        "partition"    : best_part,
        "modularity"   : best_mod,
        "n_communities": n_communities,
        "mean_ari"     : mean_ari,
        "mean_nmi"     : mean_nmi,
        "comm_stats"   : comm_stats,
    }

    out_path = os.path.join(RESULTS_DIR, f"{subreddit}_communities.json")
    # partition keys are strings in JSON
    result_serializable = result.copy()
    result_serializable["partition"] = {str(k): v for k, v in best_part.items()}
    result_serializable["comm_stats"] = {str(k): v for k, v in comm_stats.items()}
    with open(out_path, "w") as fh:
        json.dump(result_serializable, fh, indent=2)
    print(f"  Saved -> {out_path}")
    return result


if __name__ == "__main__":
    summary = {}
    all_comm_stats = {}

    for sub in SUBREDDITS:
        res = detect_communities(sub)
        summary[sub] = {
            "n_communities": res["n_communities"],
            "modularity"   : res["modularity"],
            "mean_ari"     : res["mean_ari"],
            "mean_nmi"     : res["mean_nmi"],
        }
        all_comm_stats[sub] = res["comm_stats"]

    # Summary table
    print("\n=== COMMUNITY DETECTION SUMMARY ===")
    hdr = f"{'Subreddit':<12} {'N comms':>8} {'Modularity':>11} {'ARI':>7} {'NMI':>7}"
    print(hdr)
    print("-" * 48)
    for sub, s in summary.items():
        print(f"r/{sub:<10} {s['n_communities']:>8} {s['modularity']:>11.4f} "
              f"{s['mean_ari']:>7.4f} {s['mean_nmi']:>7.4f}")
