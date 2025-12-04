#!/usr/bin/env python3
"""
Level 3: Analysis and Visualization for Byzantine Robustness Paper

Generates figures and tables for the paper comparing aggregation strategies
under various Byzantine attacks.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


# Configuration
STRATEGIES = ["fedavg", "fedmedian", "fedtrimmed"]
ATTACKS = ["none", "random_noise", "sign_flipping", "alie", "ipm"]
BYZANTINE_RATIOS = [0.1, 0.2, 0.3]

STRATEGY_LABELS = {
    "fedavg": "FedAvg",
    "fedmedian": "FedMedian",
    "fedtrimmed": "FedTrimmedAvg",
}

ATTACK_LABELS = {
    "none": "No Attack",
    "random_noise": "Random Noise",
    "sign_flipping": "Sign Flip",
    "alie": "ALIE",
    "ipm": "IPM",
}


def load_results(results_dir: str) -> List[Dict]:
    """Load all result JSON files from directory"""
    results = []
    results_path = Path(results_dir)

    for f in results_path.glob("*_result.json"):
        with open(f) as fp:
            results.append(json.load(fp))

    return results


def results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Convert results list to pandas DataFrame"""
    rows = []
    for r in results:
        if "error" in r:
            continue
        rows.append({
            "strategy": r["strategy"],
            "attack": r["attack"],
            "byzantine_ratio": r["byzantine_ratio"],
            "final_accuracy": r["final_accuracy"],
            "best_accuracy": r["best_accuracy"],
            "num_clients": r.get("num_clients", 50),
            "num_byzantine": r.get("num_byzantine", 0),
        })
    return pd.DataFrame(rows)


def create_heatmap(df: pd.DataFrame, output_dir: str, byzantine_ratio: float = 0.2):
    """Create heatmap of accuracy by strategy and attack"""
    # Filter to specific Byzantine ratio
    filtered = df[df["byzantine_ratio"] == byzantine_ratio]

    # Pivot table
    pivot = filtered.pivot_table(
        values="final_accuracy",
        index="strategy",
        columns="attack",
        aggfunc="mean"
    )

    # Reorder
    pivot = pivot.reindex(STRATEGIES)
    pivot = pivot[ATTACKS]

    # Rename
    pivot.index = [STRATEGY_LABELS.get(s, s) for s in pivot.index]
    pivot.columns = [ATTACK_LABELS.get(a, a) for a in pivot.columns]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Accuracy (%)"},
        ax=ax,
        linewidths=0.5,
    )

    ax.set_title(f"Accuracy by Strategy and Attack ({int(byzantine_ratio*100)}% Byzantine)")
    ax.set_xlabel("Attack Type")
    ax.set_ylabel("Aggregation Strategy")

    plt.tight_layout()
    output_path = Path(output_dir) / f"heatmap_byz{int(byzantine_ratio*100)}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")
    return pivot


def create_attack_impact_chart(df: pd.DataFrame, output_dir: str):
    """Create bar chart showing attack impact on each strategy"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for idx, strategy in enumerate(STRATEGIES):
        ax = axes[idx]
        strategy_df = df[(df["strategy"] == strategy) & (df["byzantine_ratio"] == 0.2)]

        # Get accuracies by attack
        accuracies = []
        for attack in ATTACKS:
            acc = strategy_df[strategy_df["attack"] == attack]["final_accuracy"].values
            if len(acc) > 0:
                accuracies.append(acc[0])
            else:
                accuracies.append(0)

        colors = ["green" if a == "none" else "coral" for a in ATTACKS]

        bars = ax.bar(
            [ATTACK_LABELS.get(a, a) for a in ATTACKS],
            accuracies,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_title(STRATEGY_LABELS.get(strategy, strategy))
        ax.set_ylim(0, 100)
        ax.set_ylabel("Accuracy (%)" if idx == 0 else "")
        ax.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{acc:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.suptitle("Attack Impact on Aggregation Strategies (20% Byzantine)")
    plt.tight_layout()

    output_path = Path(output_dir) / "attack_impact_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


def create_byzantine_scaling_chart(df: pd.DataFrame, output_dir: str):
    """Show how accuracy degrades with increasing Byzantine ratio"""
    fig, axes = plt.subplots(1, len(ATTACKS) - 1, figsize=(16, 4), sharey=True)

    # Skip "none" attack
    attacks_to_plot = [a for a in ATTACKS if a != "none"]

    for idx, attack in enumerate(attacks_to_plot):
        ax = axes[idx]

        for strategy in STRATEGIES:
            strategy_df = df[(df["strategy"] == strategy) & (df["attack"] == attack)]

            # Sort by byzantine ratio
            strategy_df = strategy_df.sort_values("byzantine_ratio")

            ax.plot(
                strategy_df["byzantine_ratio"] * 100,
                strategy_df["final_accuracy"],
                marker="o",
                label=STRATEGY_LABELS.get(strategy, strategy),
                linewidth=2,
            )

        ax.set_title(f"{ATTACK_LABELS.get(attack, attack)}")
        ax.set_xlabel("Byzantine Clients (%)")
        ax.set_ylabel("Accuracy (%)" if idx == 0 else "")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle("Accuracy vs Byzantine Ratio by Attack Type")
    plt.tight_layout()

    output_path = Path(output_dir) / "byzantine_scaling.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


def create_convergence_comparison(results_dir: str, output_dir: str, byzantine_ratio: float = 0.2):
    """Plot convergence curves for different strategies under each attack"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Load round-by-round accuracies
    for idx, attack in enumerate(ATTACKS):
        ax = axes[idx]

        for strategy in STRATEGIES:
            # Find result file
            result_file = Path(results_dir) / f"{strategy}_{attack}_byz{int(byzantine_ratio*100)}_result.json"

            if result_file.exists():
                with open(result_file) as f:
                    result = json.load(f)

                if "round_accuracies" in result:
                    rounds = range(1, len(result["round_accuracies"]) + 1)
                    ax.plot(
                        rounds,
                        result["round_accuracies"],
                        label=STRATEGY_LABELS.get(strategy, strategy),
                        linewidth=2,
                    )

        ax.set_title(ATTACK_LABELS.get(attack, attack))
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Hide unused subplot
    axes[-1].axis('off')

    plt.suptitle(f"Convergence Under Different Attacks ({int(byzantine_ratio*100)}% Byzantine)")
    plt.tight_layout()

    output_path = Path(output_dir) / "convergence_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


def create_robustness_summary_table(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Create summary table showing robustness metrics"""
    # Calculate metrics for each strategy
    summary_rows = []

    for strategy in STRATEGIES:
        strategy_df = df[df["strategy"] == strategy]

        # Baseline accuracy (no attack)
        baseline = strategy_df[strategy_df["attack"] == "none"]["final_accuracy"].mean()

        # Under attack accuracies
        attack_accuracies = {}
        for attack in ATTACKS:
            if attack != "none":
                acc = strategy_df[strategy_df["attack"] == attack]["final_accuracy"].mean()
                attack_accuracies[attack] = acc

        # Calculate robustness metrics
        mean_attacked = np.mean(list(attack_accuracies.values()))
        min_attacked = min(attack_accuracies.values())
        robustness_drop = baseline - mean_attacked
        worst_case_drop = baseline - min_attacked

        summary_rows.append({
            "Strategy": STRATEGY_LABELS.get(strategy, strategy),
            "Baseline (No Attack)": f"{baseline:.1f}%",
            "Mean Under Attack": f"{mean_attacked:.1f}%",
            "Worst Case": f"{min_attacked:.1f}%",
            "Mean Drop": f"-{robustness_drop:.1f}pp",
            "Worst Drop": f"-{worst_case_drop:.1f}pp",
        })

    summary_df = pd.DataFrame(summary_rows)

    # Save to CSV
    output_path = Path(output_dir) / "robustness_summary.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    # Print table
    print("\n" + "=" * 80)
    print("ROBUSTNESS SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80 + "\n")

    return summary_df


def create_detailed_results_table(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Create detailed results table for paper appendix"""
    # Pivot to wide format
    pivot = df.pivot_table(
        values="final_accuracy",
        index=["strategy", "byzantine_ratio"],
        columns="attack",
        aggfunc="mean"
    ).round(1)

    # Rename
    pivot.index = pivot.index.set_levels(
        [[STRATEGY_LABELS.get(s, s) for s in pivot.index.levels[0]],
         [f"{int(r*100)}%" for r in pivot.index.levels[1]]],
    )
    pivot.columns = [ATTACK_LABELS.get(a, a) for a in pivot.columns]

    # Save
    output_path = Path(output_dir) / "detailed_results.csv"
    pivot.to_csv(output_path)
    print(f"Saved: {output_path}")

    return pivot


def generate_latex_table(df: pd.DataFrame, output_dir: str, byzantine_ratio: float = 0.2):
    """Generate LaTeX table for paper"""
    filtered = df[df["byzantine_ratio"] == byzantine_ratio]

    # Pivot
    pivot = filtered.pivot_table(
        values="final_accuracy",
        index="strategy",
        columns="attack",
        aggfunc="mean"
    ).round(1)

    # Reorder and rename
    pivot = pivot.reindex(STRATEGIES)
    pivot = pivot[[a for a in ATTACKS if a in pivot.columns]]

    latex_rows = []
    latex_rows.append("\\begin{table}[h]")
    latex_rows.append("\\centering")
    latex_rows.append("\\caption{Test accuracy (\\%) under different Byzantine attacks with " +
                     f"{int(byzantine_ratio*100)}\\% malicious clients}}")
    latex_rows.append("\\begin{tabular}{l" + "c" * len(pivot.columns) + "}")
    latex_rows.append("\\toprule")

    # Header
    header = "Strategy & " + " & ".join([ATTACK_LABELS.get(a, a) for a in pivot.columns]) + " \\\\"
    latex_rows.append(header)
    latex_rows.append("\\midrule")

    # Data rows
    for strategy in pivot.index:
        values = [f"{pivot.loc[strategy, a]:.1f}" for a in pivot.columns]
        row = f"{STRATEGY_LABELS.get(strategy, strategy)} & " + " & ".join(values) + " \\\\"
        latex_rows.append(row)

    latex_rows.append("\\bottomrule")
    latex_rows.append("\\end{tabular}")
    latex_rows.append("\\label{tab:attack_results}")
    latex_rows.append("\\end{table}")

    latex_content = "\n".join(latex_rows)

    output_path = Path(output_dir) / "results_table.tex"
    with open(output_path, 'w') as f:
        f.write(latex_content)

    print(f"Saved: {output_path}")
    print("\nLaTeX Table:")
    print(latex_content)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze Byzantine robustness results')
    parser.add_argument('--results_dir', type=str, default='./results/paper',
                        help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str, default='./results/paper/figures',
                        help='Directory for output figures')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    print(f"Loading results from {args.results_dir}...")
    results = load_results(args.results_dir)

    if not results:
        print("No results found. Run experiments first.")
        return

    print(f"Found {len(results)} result files")

    # Convert to DataFrame
    df = results_to_dataframe(results)
    print(f"Loaded {len(df)} experiments")

    # Generate all outputs
    print("\nGenerating figures and tables...")

    # Heatmaps for each Byzantine ratio
    for byz_ratio in BYZANTINE_RATIOS:
        if byz_ratio in df["byzantine_ratio"].values:
            create_heatmap(df, args.output_dir, byz_ratio)

    # Other figures
    create_attack_impact_chart(df, args.output_dir)
    create_byzantine_scaling_chart(df, args.output_dir)
    create_convergence_comparison(args.results_dir, args.output_dir)

    # Tables
    create_robustness_summary_table(df, args.output_dir)
    create_detailed_results_table(df, args.output_dir)
    generate_latex_table(df, args.output_dir)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
