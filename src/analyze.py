"""
Analysis of Context Heavy Personas Experiment
=============================================
Generates visualizations and statistical analysis of persona elicitation
accuracy as a function of in-context example count K.
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
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def load_results():
    """Load experiment results into a DataFrame."""
    with open(RESULTS_DIR / "experiment_results.json") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df


def compute_summary(df):
    """Compute mean accuracy per persona per K (averaged over seeds)."""
    summary = df.groupby(["persona", "k"]).agg(
        mean_acc=("accuracy", "mean"),
        std_acc=("accuracy", "std"),
        n_runs=("accuracy", "count"),
    ).reset_index()
    summary["std_acc"] = summary["std_acc"].fillna(0)
    return summary


def plot_all_curves(summary):
    """Plot accuracy vs K for all personas (spaghetti plot)."""
    fig, ax = plt.subplots(figsize=(12, 7))

    personas = summary["persona"].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(personas)))

    for i, persona in enumerate(sorted(personas)):
        sub = summary[summary["persona"] == persona].sort_values("k")
        ax.plot(sub["k"], sub["mean_acc"], marker="o", markersize=3,
                label=persona.replace("-", " ")[:40], color=colors[i], alpha=0.7, linewidth=1.5)

    ax.set_xlabel("Number of In-Context Examples (K)", fontsize=13)
    ax.set_ylabel("Persona Elicitation Accuracy", fontsize=13)
    ax.set_title("Persona Elicitation Accuracy vs. Context Length", fontsize=14)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0, 1, 3, 5, 10, 25, 50, 100, 200])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0.3, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random chance")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, ncol=1)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "all_persona_curves.png")
    plt.close(fig)
    print("Saved: all_persona_curves.png")


def plot_heatmap(summary):
    """Heatmap of accuracy by persona and K."""
    pivot = summary.pivot_table(index="persona", columns="k", values="mean_acc")
    pivot = pivot.sort_values(by=pivot.columns.tolist(), ascending=False)

    fig, ax = plt.subplots(figsize=(12, 14))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.3, vmax=1.0,
                ax=ax, cbar_kws={"label": "Accuracy"}, linewidths=0.5)
    ax.set_title("Persona Elicitation Accuracy Heatmap\n(rows=personas, columns=K)", fontsize=13)
    ax.set_xlabel("Number of In-Context Examples (K)")
    ax.set_ylabel("")
    # Make persona names readable
    labels = [t.get_text().replace("-", " ")[:45] for t in ax.get_yticklabels()]
    ax.set_yticklabels(labels, fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "accuracy_heatmap.png")
    plt.close(fig)
    print("Saved: accuracy_heatmap.png")


def plot_mean_curve(summary):
    """Plot mean accuracy across all personas vs K with error band."""
    grouped = summary.groupby("k").agg(
        grand_mean=("mean_acc", "mean"),
        grand_std=("mean_acc", "std"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grouped["k"], grouped["grand_mean"], "o-", color="steelblue", linewidth=2, markersize=6)
    ax.fill_between(grouped["k"],
                    grouped["grand_mean"] - grouped["grand_std"],
                    grouped["grand_mean"] + grouped["grand_std"],
                    alpha=0.2, color="steelblue")
    ax.set_xlabel("Number of In-Context Examples (K)", fontsize=13)
    ax.set_ylabel("Mean Persona Elicitation Accuracy", fontsize=13)
    ax.set_title("Average Persona Accuracy vs. Context Length\n(mean ± 1 SD across personas)", fontsize=13)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0, 1, 3, 5, 10, 25, 50, 100, 200])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "mean_accuracy_curve.png")
    plt.close(fig)
    print("Saved: mean_accuracy_curve.png")


def classify_personas(summary):
    """Classify personas into easy/medium/hard based on their K-sensitivity."""
    results = []
    for persona in summary["persona"].unique():
        sub = summary[summary["persona"] == persona].sort_values("k")
        acc_at_0 = sub[sub["k"] == 0]["mean_acc"].values[0] if 0 in sub["k"].values else 0.5
        acc_at_3 = sub[sub["k"] == 3]["mean_acc"].values[0] if 3 in sub["k"].values else acc_at_0
        acc_at_max = sub["mean_acc"].max()
        acc_at_k10 = sub[sub["k"] == 10]["mean_acc"].values[0] if 10 in sub["k"].values else acc_at_3

        # Gain from K=0 to K=3
        early_gain = acc_at_3 - acc_at_0
        # Gain from K=10 to max K
        late_gain = acc_at_max - acc_at_k10
        # Total gain
        total_gain = acc_at_max - acc_at_0

        # K at which 80% accuracy is first reached
        k_to_80 = None
        for _, row in sub.iterrows():
            if row["mean_acc"] >= 0.80:
                k_to_80 = row["k"]
                break

        results.append({
            "persona": persona,
            "acc_at_0": acc_at_0,
            "acc_at_3": acc_at_3,
            "acc_at_10": acc_at_k10,
            "acc_at_max": acc_at_max,
            "early_gain": early_gain,
            "late_gain": late_gain,
            "total_gain": total_gain,
            "k_to_80": k_to_80,
        })

    taxonomy = pd.DataFrame(results)

    # Classify
    def classify(row):
        if row["acc_at_3"] >= 0.80:
            return "Easy (K≤3)"
        elif row["acc_at_10"] >= 0.80:
            return "Medium (K≤10)"
        elif row["acc_at_max"] >= 0.80:
            return "Hard (K>10)"
        else:
            return "Very Hard (never ≥80%)"
    taxonomy["category"] = taxonomy.apply(classify, axis=1)

    return taxonomy


def plot_taxonomy(taxonomy):
    """Plot persona difficulty taxonomy."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Category counts
    cats = taxonomy["category"].value_counts()
    colors_map = {
        "Easy (K≤3)": "#2ecc71",
        "Medium (K≤10)": "#f39c12",
        "Hard (K>10)": "#e74c3c",
        "Very Hard (never ≥80%)": "#8e44ad",
    }
    cat_colors = [colors_map.get(c, "gray") for c in cats.index]
    axes[0].bar(range(len(cats)), cats.values, color=cat_colors)
    axes[0].set_xticks(range(len(cats)))
    axes[0].set_xticklabels(cats.index, rotation=15, ha="right", fontsize=9)
    axes[0].set_ylabel("Number of Personas")
    axes[0].set_title("Persona Difficulty Distribution")

    # Late gain distribution
    axes[1].hist(taxonomy["late_gain"], bins=15, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("Late Gain (accuracy improvement K=10 → max K)")
    axes[1].set_ylabel("Number of Personas")
    axes[1].set_title("Distribution of Late-Onset Accuracy Gains\n(positive = context-heavy evidence)")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "persona_taxonomy.png")
    plt.close(fig)
    print("Saved: persona_taxonomy.png")


def statistical_tests(summary, taxonomy):
    """Run statistical tests for the hypothesis."""
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)

    # Test 1: Is there a significant effect of K on accuracy?
    k_groups = [summary[summary["k"] == k]["mean_acc"].values for k in sorted(summary["k"].unique())]
    stat, p_kruskal = stats.kruskal(*k_groups)
    print(f"\n1. Kruskal-Wallis test (effect of K on accuracy):")
    print(f"   H-statistic = {stat:.3f}, p = {p_kruskal:.6f}")

    # Test 2: Is accuracy at K=100+ significantly higher than K=3?
    acc_3 = summary[summary["k"] == 3]["mean_acc"].values
    k_max = summary["k"].max()
    acc_max = summary[summary["k"] == k_max]["mean_acc"].values
    stat_wil, p_wil = stats.wilcoxon(acc_max, acc_3, alternative="greater")
    print(f"\n2. Wilcoxon signed-rank test (K={k_max} > K=3):")
    print(f"   statistic = {stat_wil:.3f}, p = {p_wil:.6f}")
    effect_size = np.mean(acc_max - acc_3)
    print(f"   Mean improvement: {effect_size:.4f}")

    # Test 3: Are there personas with significant late gains?
    print(f"\n3. Context-heavy persona analysis:")
    context_heavy = taxonomy[taxonomy["late_gain"] > 0.05]
    print(f"   Personas with >5% accuracy gain after K=10: {len(context_heavy)}/{len(taxonomy)}")
    if len(context_heavy) > 0:
        for _, row in context_heavy.iterrows():
            print(f"     - {row['persona']}: late_gain = {row['late_gain']:.3f}")

    # Test 4: One-sample t-test on late gains (is mean > 0?)
    late_gains = taxonomy["late_gain"].values
    t_stat, p_ttest = stats.ttest_1samp(late_gains, 0, alternative="greater")
    print(f"\n4. One-sample t-test (mean late gain > 0):")
    print(f"   t = {t_stat:.3f}, p = {p_ttest:.6f}")
    print(f"   Mean late gain = {np.mean(late_gains):.4f} ± {np.std(late_gains):.4f}")

    # Test 5: Correlation between zero-shot accuracy and total gain
    corr, p_corr = stats.spearmanr(taxonomy["acc_at_0"], taxonomy["total_gain"])
    print(f"\n5. Spearman correlation (zero-shot accuracy vs total gain from ICL):")
    print(f"   rho = {corr:.3f}, p = {p_corr:.6f}")

    return {
        "kruskal_H": stat, "kruskal_p": p_kruskal,
        "wilcoxon_stat": stat_wil, "wilcoxon_p": p_wil,
        "mean_improvement_k3_to_max": effect_size,
        "n_context_heavy_5pct": len(context_heavy),
        "late_gain_ttest_t": t_stat, "late_gain_ttest_p": p_ttest,
        "mean_late_gain": float(np.mean(late_gains)),
        "spearman_rho": corr, "spearman_p": p_corr,
    }


def plot_scatter_gain(taxonomy):
    """Scatter: zero-shot accuracy vs late gain."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_map = {
        "Easy (K≤3)": "#2ecc71",
        "Medium (K≤10)": "#f39c12",
        "Hard (K>10)": "#e74c3c",
        "Very Hard (never ≥80%)": "#8e44ad",
    }
    for cat, color in colors_map.items():
        sub = taxonomy[taxonomy["category"] == cat]
        if len(sub) > 0:
            ax.scatter(sub["acc_at_0"], sub["late_gain"], c=color, label=cat,
                      s=60, alpha=0.8, edgecolors="white", linewidth=0.5)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Zero-shot Accuracy (K=0)", fontsize=12)
    ax.set_ylabel("Late Gain (K=10 → max K)", fontsize=12)
    ax.set_title("Zero-shot Accuracy vs. Late-Onset Gains", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "scatter_gain.png")
    plt.close(fig)
    print("Saved: scatter_gain.png")


def main():
    print("Loading results...")
    df = load_results()
    print(f"Loaded {len(df)} result records")

    summary = compute_summary(df)
    print(f"Summary: {len(summary)} persona-K combinations")

    # Generate all visualizations
    plot_all_curves(summary)
    plot_heatmap(summary)
    plot_mean_curve(summary)

    taxonomy = classify_personas(summary)
    plot_taxonomy(taxonomy)
    plot_scatter_gain(taxonomy)

    # Save taxonomy
    taxonomy.to_csv(RESULTS_DIR / "persona_taxonomy.csv", index=False)
    print("Saved: persona_taxonomy.csv")

    # Statistical tests
    stats_results = statistical_tests(summary, taxonomy)

    # Save stats
    with open(RESULTS_DIR / "statistical_tests.json", "w") as f:
        json.dump(stats_results, f, indent=2)
    print("\nSaved: statistical_tests.json")

    # Print summary table
    print("\n" + "="*60)
    print("PERSONA TAXONOMY SUMMARY")
    print("="*60)
    print(taxonomy.sort_values("late_gain", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
