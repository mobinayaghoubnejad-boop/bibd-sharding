"""Fig. 7: multi-epoch cross-shard ratio and migration (reads results/hybrid_*_raw.csv).
Colour = partitioner used at epoch 0; dashed = BIBD-constrained updates at epochs 1-5."""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Inter", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN = "#2060C0", "#E0603A", "#2A9D6F"
SERIES = [  # method, label, colour, linestyle, marker
    ("Louvain", "Louvain (rebuild)", ORANGE, "-", "s"),
    ("Hybrid", "Hybrid (Louvain → BIBD)", ORANGE, "--", "D"),
    ("TxAllo-C", "TxAllo-C (G + A)", GREEN, "-", "^"),
    ("Hybrid-TxAllo", "Hybrid (TxAllo-C → BIBD)", GREEN, "--", "v"),
    ("BIBD", "Proposed (BIBD)", BLUE, "-", "o"),
]
syn = pd.read_csv("results/hybrid_synthetic_raw.csv"); sc = pd.read_csv("results/hybrid_scale_raw.csv")
sets = [(syn[syn.N == 310], "N = 310, S = 31 (15 runs)"), (sc, "N = 100,000, S = 31 (3 runs)")]
fig, axs = plt.subplots(2, 2, figsize=(10.2, 7.2), sharex=True)
for r, (d, title) in enumerate(sets):
    for c, met in enumerate(["rho_cross", "migration"]):
        ax = axs[r, c]
        for m, lab, col, ls, mk in SERIES:
            g = d[d.method == m].groupby("epoch")[met].agg(["mean", "std"])
            if met == "migration":
                g = g.loc[1:]
            ax.plot(g.index, g["mean"], color=col, ls=ls, marker=mk, ms=7, lw=2, label=lab,
                    markeredgecolor="white", markeredgewidth=1)
            ax.fill_between(g.index, g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0),
                            color=col, alpha=0.10, lw=0)
        ax.set_ylim(0, 1.05); ax.grid(axis="y", color="#e5e5e5", lw=0.8); ax.set_axisbelow(True)
        ax.set_xticks(range(6)); ax.set_title(title, fontsize=11)
        ax.set_ylabel("ρ_cross on next window" if met == "rho_cross" else "Migration (fraction of accounts)")
        if r == 1:
            ax.set_xlabel("Epoch")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig("results/fig7_hybrid.png", dpi=200)
