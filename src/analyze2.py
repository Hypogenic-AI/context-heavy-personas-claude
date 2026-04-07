"""
Analysis of Experiment 2: Persona Generation Distinctiveness
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

plt.rcParams.update({
    'font.size': 11,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})


def main():
    with open(RESULTS_DIR / "experiment2_results.json") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} records from Experiment 2")
    print(f"Personas: {df['persona'].nunique()}")
    print(f"K values: {sorted(df['k'].unique())}")

    # Pivot table
    pivot = df.pivot_table(index="persona", columns="k", values="mean_distinctiveness")
    print("\nDistinctiveness scores (persona x K):")
    print(pivot.round(2).to_string())

    # Plot 1: Line plots of distinctiveness vs K
    fig, ax = plt.subplots(figsize=(10, 6))
    personas = sorted(df["persona"].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(personas)))

    for i, persona in enumerate(personas):
        sub = df[df["persona"] == persona].sort_values("k")
        ax.plot(sub["k"], sub["mean_distinctiveness"], "o-",
                label=persona.replace("-", " ")[:35], color=colors[i],
                linewidth=2, markersize=5)

    ax.set_xlabel("Number of In-Context Examples (K)", fontsize=13)
    ax.set_ylabel("Persona Distinctiveness (0-10)", fontsize=13)
    ax.set_title("Persona Generation Distinctiveness vs. Context Length\n(LLM-as-Judge, 0=generic, 10=highly distinctive)", fontsize=13)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0, 1, 3, 10, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(3, 10.5)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "exp2_distinctiveness_curves.png")
    plt.close(fig)
    print("Saved: exp2_distinctiveness_curves.png")

    # Plot 2: Heatmap
    pivot_sorted = pivot.sort_values(by=list(pivot.columns), ascending=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot_sorted, annot=True, fmt=".1f", cmap="YlOrRd",
                vmin=4, vmax=10, ax=ax, linewidths=0.5)
    ax.set_title("Persona Distinctiveness Heatmap\n(0=generic, 10=highly distinctive)")
    ax.set_xlabel("Number of In-Context Examples (K)")
    labels = [t.get_text().replace("-", " ")[:40] for t in ax.get_yticklabels()]
    ax.set_yticklabels(labels, fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "exp2_heatmap.png")
    plt.close(fig)
    print("Saved: exp2_heatmap.png")

    # Plot 3: Mean distinctiveness vs K
    mean_by_k = df.groupby("k")["mean_distinctiveness"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(mean_by_k["k"], mean_by_k["mean"], yerr=mean_by_k["std"],
                fmt="o-", color="steelblue", linewidth=2, markersize=6, capsize=4)
    ax.set_xlabel("Number of In-Context Examples (K)", fontsize=13)
    ax.set_ylabel("Mean Distinctiveness Score", fontsize=13)
    ax.set_title("Average Persona Distinctiveness vs. Context Length", fontsize=13)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0, 1, 3, 10, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "exp2_mean_distinctiveness.png")
    plt.close(fig)
    print("Saved: exp2_mean_distinctiveness.png")

    # Statistical analysis
    print("\n" + "=" * 60)
    print("EXPERIMENT 2 STATISTICAL ANALYSIS")
    print("=" * 60)

    # Compute gain from K=0 to K=100 per persona
    gains = []
    for persona in personas:
        sub = df[df["persona"] == persona]
        d0 = sub[sub["k"] == 0]["mean_distinctiveness"].values[0]
        d100 = sub[sub["k"] == 100]["mean_distinctiveness"].values[0]
        gains.append({"persona": persona, "d_at_0": d0, "d_at_100": d100, "gain": d100 - d0})
    gains_df = pd.DataFrame(gains)

    print("\nDistinctiveness gain (K=0 → K=100):")
    for _, row in gains_df.sort_values("gain", ascending=False).iterrows():
        print(f"  {row['persona']:40s}: {row['d_at_0']:.1f} → {row['d_at_100']:.1f} (gain: {row['gain']:+.1f})")

    # Test if gains are significantly positive
    t_stat, p_val = stats.ttest_1samp(gains_df["gain"].values, 0, alternative="greater")
    print(f"\nOne-sample t-test (mean gain > 0): t={t_stat:.3f}, p={p_val:.4f}")
    print(f"Mean gain: {gains_df['gain'].mean():.2f} ± {gains_df['gain'].std():.2f}")

    # Kruskal-Wallis on K effect
    k_groups = [df[df["k"] == k]["mean_distinctiveness"].values for k in sorted(df["k"].unique())]
    h_stat, p_kw = stats.kruskal(*k_groups)
    print(f"\nKruskal-Wallis (effect of K): H={h_stat:.3f}, p={p_kw:.4f}")

    # Correlation: K vs distinctiveness
    corr, p_corr = stats.spearmanr(df["k"], df["mean_distinctiveness"])
    print(f"Spearman correlation (K vs distinctiveness): rho={corr:.3f}, p={p_corr:.4f}")

    # Identify which personas benefit most from context
    significant_gain = gains_df[gains_df["gain"] > 0.5]
    print(f"\nPersonas with >0.5 distinctiveness gain from context: {len(significant_gain)}/{len(personas)}")

    # Save
    gains_df.to_csv(RESULTS_DIR / "exp2_gains.csv", index=False)
    stats_results = {
        "ttest_t": float(t_stat), "ttest_p": float(p_val),
        "kruskal_H": float(h_stat), "kruskal_p": float(p_kw),
        "spearman_rho": float(corr), "spearman_p": float(p_corr),
        "mean_gain": float(gains_df["gain"].mean()),
        "std_gain": float(gains_df["gain"].std()),
    }
    with open(RESULTS_DIR / "exp2_stats.json", "w") as f:
        json.dump(stats_results, f, indent=2)
    print("\nSaved analysis results.")


if __name__ == "__main__":
    main()
