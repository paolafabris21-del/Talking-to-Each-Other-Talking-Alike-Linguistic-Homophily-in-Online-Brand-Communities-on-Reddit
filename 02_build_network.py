"""
Build the user reply network for each subreddit: a directed graph from
parent_id links, symmetrised to an undirected weighted graph, keeping the giant
connected component. Saves the GCC and descriptive stats to results/.
"""

import json
import os
import pickle
from collections import defaultdict

import networkx as nx
import numpy as np

from bots import is_bot


SUBREDDITS  = ["apple", "google", "samsung"]
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MIN_COMMENTS = 5   # minimum comments per user to be included


def load_comments(subreddit: str) -> list[dict]:
    path = os.path.join(DATA_DIR, f"{subreddit}_comments.jsonl")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def build_network(subreddit: str) -> dict:
    print(f"  r/{subreddit}")

    comments = load_comments(subreddit)
    print(f"  Raw comments      : {len(comments):,}")

    #filter bots and removed content
    comments = [
        c for c in comments
        if not is_bot(c.get("author", ""))
        and c.get("body") not in {"[deleted]", "[removed]"}
        and c.get("author", "")
    ]
    print(f"  After filtering   : {len(comments):,}")

    #build author lookup
    id_to_author: dict[str, str] = {c["id"]: c["author"] for c in comments}

    #filter to active users (>= MIN_COMMENTS)
    user_counts: dict[str, int] = defaultdict(int)
    for c in comments:
        user_counts[c["author"]] += 1
    active = {u for u, n in user_counts.items() if n >= MIN_COMMENTS}
    print(f"  Active users (>={MIN_COMMENTS}): {len(active):,}")

    # build edge weights: an edge (u, v) exists if u replied to a comment written by v.
    edge_weight: dict[tuple, int] = defaultdict(int)
    for c in comments:
        author = c["author"]
        if author not in active:
            continue
        parent_id = c.get("parent_id", "")
        # parent_id starts with "t1_" for comment replies, "t3_" for post replies
        if not parent_id.startswith("t1_"):
            continue
        raw_parent_id = parent_id[3:]  # strip "t1_" prefix; id_to_author keys have no prefix
        parent_author = id_to_author.get(raw_parent_id)
        if parent_author and parent_author in active and parent_author != author:
            edge_weight[(author, parent_author)] += 1

    # directed graph
    G_dir = nx.DiGraph()
    for (u, v), w in edge_weight.items():
        G_dir.add_edge(u, v, weight=w)
    print(f"  Directed graph    : {G_dir.number_of_nodes()} nodes, {G_dir.number_of_edges()} edges")

    # undirected weighted graph
    G = nx.Graph()
    for (u, v), w in edge_weight.items():
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)
    print(f"  Undirected graph  : {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    #giant connected component GCC
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    GCC = G.subgraph(components[0]).copy()
    coverage = GCC.number_of_nodes() / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    print(f"  GCC               : {GCC.number_of_nodes()} nodes, {GCC.number_of_edges()} edges ({coverage*100:.1f}% of nodes)")

    #descriptive statisticS
    degrees = [d for _, d in GCC.degree()]
    weights = [d["weight"] for _, _, d in GCC.edges(data=True)]
    stats = {
        "n_nodes"          : GCC.number_of_nodes(),
        "n_edges"          : GCC.number_of_edges(),
        "density"          : nx.density(GCC),
        "avg_degree"       : float(np.mean(degrees)) if degrees else 0.0,
        "max_degree"       : int(np.max(degrees)) if degrees else 0,
        "avg_edge_weight"  : float(np.mean(weights)) if weights else 0.0,
        "n_components_full": len(components),
        "gcc_coverage"     : coverage,
        "avg_clustering"   : nx.average_clustering(GCC),
    }
    for k, v in stats.items():
        print(f"    {k:<22}: {v:.4f}" if isinstance(v, float) else f"    {k:<22}: {v}")

    #save
    pkl_path = os.path.join(RESULTS_DIR, f"{subreddit}_network.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump({"graph": GCC, "stats": stats}, fh)

    edgelist_path = os.path.join(RESULTS_DIR, f"{subreddit}_edgelist.csv")
    nx.write_weighted_edgelist(GCC, edgelist_path, delimiter=",")
    print(f"  Saved -> {pkl_path}")
    print(f"  Saved -> {edgelist_path}")

    return stats


if __name__ == "__main__":
    all_stats = {}
    for sub in SUBREDDITS:
        all_stats[sub] = build_network(sub)

    out_path = os.path.join(RESULTS_DIR, "network_stats.json")
    with open(out_path, "w") as fh:
        json.dump(all_stats, fh, indent=2)
    print(f"\nNetwork stats saved -> {out_path}")

    print("\n=== NETWORK SUMMARY ===")
    hdr = f"{'Subreddit':<12} {'Nodes':>7} {'Edges':>7} {'Density':>9} {'AvgDeg':>8} {'AvgClust':>10} {'GCC%':>7}"
    print(hdr)
    print("-" * len(hdr))
    for sub, s in all_stats.items():
        print(
            f"r/{sub:<10} {s['n_nodes']:>7,} {s['n_edges']:>7,} "
            f"{s['density']:>9.5f} {s['avg_degree']:>8.2f} "
            f"{s['avg_clustering']:>10.4f} {s['gcc_coverage']*100:>6.1f}%"
        )
