"""
Generate the paper figures (1-4) from the saved results. Run after the analysis
steps are complete. Each figure is written as a vector PDF.

    python3 09_figures_final.py
"""

import json
import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

# Paths
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

SUBREDDITS = ["apple", "google", "samsung"]

# Global style - minimalist academic
BRAND_COLORS = {
    "apple":   "#555555",   # neutral dark gray
    "google":  "#4285F4",   # Google blue
    "samsung": "#1428A0",   # Samsung deep blue
}
BRAND_LABELS = {
    "apple":   "r/apple",
    "google":  "r/google",
    "samsung": "r/samsung",
}

# Muted qualitative palette for communities (ColorBrewer Set2-derived)
COMM_PALETTE = [
    "#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854",
    "#FFD92F", "#E5C494", "#B3B3B3", "#1B9E77", "#D95F02",
    "#7570B3", "#E7298A", "#66A61E", "#E6AB02", "#A6761D",
    "#B15928", "#6A3D9A", "#FF7F00", "#33A02C", "#1F78B4",
    "#FB9A99", "#E31A1C", "#FDBF6F", "#CAB2D6", "#B2DF8A",
    "#A6CEE3", "#FFFF99", "#BEBADA", "#FFFFB3", "#80B1D3",
]

SENTIMENT_COLORS = {
    "Positive": "#2166AC",   # blue
    "Neutral":  "#D9D9D9",   # light gray
    "Negative": "#D6604D",   # muted red
}

mpl.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#EEEEEE",
    "grid.linewidth":     0.6,
    "axes.axisbelow":     True,
})


# Helper: save figure as vector PDF
def save_fig(fig, name: str) -> None:
    path = os.path.join(FIGURES_DIR, f"{name}.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved -> figures/{name}.pdf")


# Figure 1 - Network visualisations (3-panel)
def fig1_networks() -> None:
    print("\n[Figure 1] Network visualisations...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, sub in zip(axes, SUBREDDITS):
        with open(os.path.join(RESULTS_DIR, f"{sub}_network.pkl"), "rb") as fh:
            net_data = pickle.load(fh)
        with open(os.path.join(RESULTS_DIR, f"{sub}_communities.json")) as fh:
            comm_data = json.load(fh)

        GCC       = net_data["graph"]
        partition = {k: int(v) for k, v in comm_data["partition"].items()}
        n_comms   = comm_data["n_communities"]
        Q         = comm_data["modularity"]

        node_colors = [
            COMM_PALETTE[partition.get(n, 0) % len(COMM_PALETTE)]
            for n in GCC.nodes()
        ]
        # Node size proportional to degree, clipped for readability
        degrees    = dict(GCC.degree())
        node_sizes = [max(4, min(60, 2 * degrees[n])) for n in GCC.nodes()]
        edge_alpha = max(0.04, min(0.15, 300 / GCC.number_of_edges()))

        pos = nx.spring_layout(
            GCC, weight="weight", seed=42,
            k=2.0 / np.sqrt(GCC.number_of_nodes())
        )

        nx.draw_networkx_edges(
            GCC, pos, ax=ax,
            alpha=edge_alpha, width=0.4, edge_color="#AAAAAA"
        )
        nx.draw_networkx_nodes(
            GCC, pos, ax=ax,
            node_color=node_colors, node_size=node_sizes,
            alpha=0.85, linewidths=0.0
        )

        ax.set_title(
            f"{BRAND_LABELS[sub]}\n"
            f"$n$={GCC.number_of_nodes():,}  "
            f"$e$={GCC.number_of_edges():,}  "
            f"$Q$={Q:.3f}  "
            f"$K$={n_comms}",
            fontsize=10, pad=8
        )
        ax.axis("off")
        ax.set_facecolor("white")
        ax.grid(False)

    fig.suptitle(
        "Figure 1. User Interaction Networks Coloured by Louvain Community",
        fontsize=11, y=1.01, fontweight="normal"
    )
    fig.tight_layout()
    save_fig(fig, "fig1_networks")
    plt.close()


# Figure 2 - QAP null distributions (3-panel)
def fig2_qap() -> None:
    print("\n[Figure 2] QAP null distributions...")

    with open(os.path.join(RESULTS_DIR, "qap_results.json")) as fh:
        qap_meta = json.load(fh)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)

    for ax, sub in zip(axes, SUBREDDITS):
        meta  = qap_meta[sub]
        obs_r = meta["observed_r"]

        # actual permutation null distribution saved by 04_qap_test.py
        null = np.load(os.path.join(RESULTS_DIR, f"{sub}_qap_null.npy"))

        color = BRAND_COLORS[sub]
        ax.hist(null, bins=35, color=color, alpha=0.55, edgecolor="none",
                label="Permuted $r$")
        ax.axvline(obs_r, color="#CC3333", lw=1.8, linestyle="--",
                   label=f"Observed $r$ = {obs_r:.3f}")
        ax.axvline(0, color="#999999", lw=0.8, linestyle=":")

        ax.set_title(BRAND_LABELS[sub], fontweight="bold")
        ax.set_xlabel("Pearson $r$ (permuted)")
        ax.set_ylabel("Count" if sub == "apple" else "")
        ax.legend(frameon=False, fontsize=8.5)

        # Annotate p-value (top-left, away from observed r line on the right)
        ax.text(0.03, 0.95, "$p$ < 0.001", transform=ax.transAxes,
                ha="left", va="top", fontsize=9,
                color="#CC3333", fontweight="bold")

    fig.suptitle(
        "Figure 2. QAP Null Distributions and Observed Linguistic Homophily Coefficients",
        fontsize=11, y=1.02, fontweight="normal"
    )
    fig.tight_layout()
    save_fig(fig, "fig2_qap")
    plt.close()


# Figure 3 - Cross-brand comparison
def fig3_crossbrand() -> None:
    print("\n[Figure 3] Cross-brand comparison...")

    with open(os.path.join(RESULTS_DIR, "qap_results.json")) as fh:
        qap = json.load(fh)

    comm_stats = {}
    for sub in SUBREDDITS:
        with open(os.path.join(RESULTS_DIR, f"{sub}_communities.json")) as fh:
            d = json.load(fh)
        with open(os.path.join(RESULTS_DIR, f"{sub}_sentiment.json")) as fh:
            s = json.load(fh)
        comm_stats[sub] = {
            "modularity": d["modularity"],
            "n_comm":     d["n_communities"],
            "mean_ari":   d["mean_ari"],
            "mean_nmi":   d["mean_nmi"],
            "cramers_v":  s["cramers_v"],
            "sent_p":     s["p_value"],
        }

    labels = [BRAND_LABELS[s] for s in SUBREDDITS]
    colors = [BRAND_COLORS[s] for s in SUBREDDITS]
    x      = np.arange(len(SUBREDDITS))
    width  = 0.55

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel A: QAP observed r
    bars = axes[0].bar(x, [qap[s]["observed_r"] for s in SUBREDDITS],
                       width=width, color=colors, alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("Observed Pearson $r$")
    axes[0].set_title("(A) Linguistic Homophily (QAP)")
    axes[0].set_ylim(0, 0.18)
    for bar, sub in zip(bars, SUBREDDITS):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.003,
                     f"{qap[sub]['observed_r']:.3f}",
                     ha="center", va="bottom", fontsize=9)
    # Add p < 0.001 annotation for all
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     0.005, "$p$<.001", ha="center", va="bottom",
                     fontsize=7.5, color="white", fontweight="bold")

    # Panel B: Modularity + ARI
    w2 = 0.25
    axes[1].bar(x - w2/2, [comm_stats[s]["modularity"] for s in SUBREDDITS],
                width=w2, color=colors, alpha=0.85, label="Modularity $Q$")
    axes[1].bar(x + w2/2, [comm_stats[s]["mean_ari"] for s in SUBREDDITS],
                width=w2, color=colors, alpha=0.45, hatch="//", edgecolor="white",
                label="Mean ARI")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("Score")
    axes[1].set_title("(B) Community Structure")
    axes[1].set_ylim(0, 0.85)
    axes[1].legend(frameon=False)

    # Panel C: Cramér's V (sentiment heterogeneity)
    bar_colors_sent = []
    for sub in SUBREDDITS:
        bar_colors_sent.append(BRAND_COLORS[sub] if comm_stats[sub]["sent_p"] < 0.05
                               else "#AAAAAA")
    bars3 = axes[2].bar(x, [comm_stats[s]["cramers_v"] for s in SUBREDDITS],
                        width=width, color=bar_colors_sent, alpha=0.85)
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, rotation=15, ha="right")
    axes[2].set_ylabel("Cramér's $V$")
    axes[2].set_title("(C) Sentiment Heterogeneity\nacross Communities")
    axes[2].set_ylim(0, 0.30)
    for bar, sub in zip(bars3, SUBREDDITS):
        p = comm_stats[sub]["sent_p"]
        sig = ("$p$ = " + f"{p:.3f}" + " *") if p < 0.05 else f"$p$ = {p:.3f}"
        axes[2].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.005, sig,
                     ha="center", va="bottom", fontsize=8.5)

    fig.suptitle(
        "Figure 3. Cross-Brand Comparison of Linguistic Homophily, "
        "Community Structure, and Sentiment Heterogeneity",
        fontsize=10.5, y=1.02, fontweight="normal"
    )
    fig.tight_layout()
    save_fig(fig, "fig3_crossbrand")
    plt.close()


# Figure 4 - Sentiment distribution by community (3-panel, top 12 + Other)
def fig4_sentiment(top_n: int = 12) -> None:
    print("\n[Figure 4] Sentiment distributions...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=False)

    for ax, sub in zip(axes, SUBREDDITS):
        with open(os.path.join(RESULTS_DIR, f"{sub}_sentiment.json")) as fh:
            sent_data = json.load(fh)
        with open(os.path.join(RESULTS_DIR, f"{sub}_communities.json")) as fh:
            comm_data = json.load(fh)

        comm_sizes = {int(k): v["size"] for k, v in comm_data["comm_stats"].items()}
        comm_ids   = sent_data["communities"]
        ct         = sent_data["contingency_table"]
        chi2       = sent_data["chi2"]
        pval       = sent_data["p_value"]
        V          = sent_data["cramers_v"]

        # Sort by size descending
        order = sorted(range(len(comm_ids)),
                       key=lambda i: comm_sizes.get(comm_ids[i], 0), reverse=True)

        rows_top  = order[:top_n]
        rows_rest = order[top_n:]

        pos_pct, neu_pct, neg_pct, labels = [], [], [], []

        for i in rows_top:
            row   = ct[i]
            total = sum(row)
            if total == 0:
                continue
            pos_pct.append(row[0] / total * 100)
            neu_pct.append(row[1] / total * 100)
            neg_pct.append(row[2] / total * 100)
            labels.append(f"C{comm_ids[i]}  (n={total})")

        # "Other" row: aggregate remaining communities
        if rows_rest:
            agg = [sum(ct[i][j] for i in rows_rest) for j in range(3)]
            total_other = sum(agg)
            n_other_comms = len(rows_rest)
            if total_other > 0:
                pos_pct.append(agg[0] / total_other * 100)
                neu_pct.append(agg[1] / total_other * 100)
                neg_pct.append(agg[2] / total_other * 100)
                labels.append(f"Other ({n_other_comms} comms, n={total_other})")

        y = np.arange(len(labels))

        # Stacked horizontal bars: Positive | Neutral | Negative
        ax.barh(y, pos_pct, color=SENTIMENT_COLORS["Positive"],
                alpha=0.88, label="Positive", height=0.65)
        ax.barh(y, neu_pct, color=SENTIMENT_COLORS["Neutral"],
                alpha=0.88, label="Neutral",  height=0.65, left=pos_pct)
        ax.barh(y, neg_pct, color=SENTIMENT_COLORS["Negative"],
                alpha=0.88, label="Negative", height=0.65,
                left=[p + n for p, n in zip(pos_pct, neu_pct)])

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Users (%)", fontsize=9)
        ax.axvline(50, color="#CCCCCC", lw=0.8, linestyle=":")
        ax.invert_yaxis()
        ax.grid(axis="x", color="#EEEEEE", linewidth=0.6)
        ax.grid(axis="y", visible=False)

        # Separate "Other" row visually
        if rows_rest:
            ax.axhline(top_n - 0.5, color="#AAAAAA", lw=0.8, linestyle="--")

        sig_str = f"$p$ = {pval:.3f}" + (" *" if pval < 0.05 else "")
        ax.set_title(
            f"{BRAND_LABELS[sub]}\n"
            f"$\\chi^2$({sent_data['dof']}) = {chi2:.1f},  {sig_str},  $V$ = {V:.3f}",
            fontsize=10, pad=6
        )

    fig.suptitle(
        "Figure 4. Sentiment Distribution by Louvain Community\n"
        "(top 12 communities by size; remaining communities aggregated as 'Other')",
        fontsize=10.5, y=1.02, fontweight="normal"
    )

    # Single legend below the panels, outside the plot area
    handles = [
        mpatches.Patch(color=SENTIMENT_COLORS["Positive"], alpha=0.88, label="Positive"),
        mpatches.Patch(color=SENTIMENT_COLORS["Neutral"],  alpha=0.88, label="Neutral"),
        mpatches.Patch(color=SENTIMENT_COLORS["Negative"], alpha=0.88, label="Negative"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), frameon=False,
               fontsize=9, handlelength=1.2, handleheight=0.9)

    fig.tight_layout()
    save_fig(fig, "fig4_sentiment")
    plt.close()


if __name__ == "__main__":
    fig1_networks()
    fig2_qap()
    fig3_crossbrand()
    fig4_sentiment()
    print("\nAll figures saved to figures/")
