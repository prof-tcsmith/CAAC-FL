#!/usr/bin/env python3
"""
Aggregate results from baseline experiments and compute statistical analysis.

Features:
- Load results from multiple runs
- Compute mean, std, 95% confidence intervals
- Perform statistical significance tests (ANOVA, t-tests)
- Generate summary tables and visualizations
"""

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


DATASETS = ['mnist', 'fashion_mnist', 'cifar10']
STRATEGIES = ['fedavg', 'fedmean', 'fedmedian']
CONDITIONS = ['iid_equal', 'iid_unequal']


def confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate mean and confidence interval.

    Args:
        data: List of values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    n = len(data)
    if n < 2:
        return np.mean(data), np.mean(data), np.mean(data)

    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h


def load_results(results_dir: str) -> pd.DataFrame:
    """
    Load all experiment results into a DataFrame.

    Args:
        results_dir: Base directory containing results

    Returns:
        DataFrame with all experiment results
    """
    records = []

    for dataset in DATASETS:
        for condition in CONDITIONS:
            pattern = os.path.join(results_dir, dataset, condition, '*.json')
            files = glob.glob(pattern)

            for filepath in files:
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)

                    records.append({
                        'dataset': data.get('dataset', dataset),
                        'strategy': data.get('strategy'),
                        'condition': data.get('condition', condition),
                        'seed': data.get('seed'),
                        'final_accuracy': data.get('final_accuracy'),
                        'best_accuracy': data.get('best_accuracy'),
                        'final_loss': data.get('final_loss'),
                        'num_rounds': data.get('num_rounds'),
                        'filepath': filepath,
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not load {filepath}: {e}")

    return pd.DataFrame(records)


def aggregate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute aggregated statistics for each configuration.

    Args:
        df: DataFrame with individual run results

    Returns:
        DataFrame with aggregated statistics
    """
    aggregated = []

    for dataset in df['dataset'].unique():
        for condition in df['condition'].unique():
            for strategy in df['strategy'].unique():
                subset = df[
                    (df['dataset'] == dataset) &
                    (df['condition'] == condition) &
                    (df['strategy'] == strategy)
                ]

                if len(subset) == 0:
                    continue

                accuracies = subset['final_accuracy'].values
                mean, ci_low, ci_high = confidence_interval(accuracies)

                aggregated.append({
                    'dataset': dataset,
                    'condition': condition,
                    'strategy': strategy,
                    'n_runs': len(subset),
                    'mean_accuracy': mean,
                    'std_accuracy': np.std(accuracies),
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'min_accuracy': np.min(accuracies),
                    'max_accuracy': np.max(accuracies),
                    'best_accuracy_mean': np.mean(subset['best_accuracy'].values),
                })

    return pd.DataFrame(aggregated)


def perform_anova(df: pd.DataFrame, dataset: str, condition: str) -> Dict:
    """
    Perform one-way ANOVA to test if strategy differences are significant.

    Args:
        df: DataFrame with individual run results
        dataset: Dataset name
        condition: Condition name

    Returns:
        Dict with ANOVA results
    """
    subset = df[(df['dataset'] == dataset) & (df['condition'] == condition)]

    groups = []
    strategy_names = []
    for strategy in STRATEGIES:
        data = subset[subset['strategy'] == strategy]['final_accuracy'].values
        if len(data) > 0:
            groups.append(data)
            strategy_names.append(strategy)

    if len(groups) < 2:
        return {'error': 'Not enough groups for ANOVA'}

    f_stat, p_value = stats.f_oneway(*groups)

    return {
        'dataset': dataset,
        'condition': condition,
        'f_statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'strategies': strategy_names,
        'group_means': [np.mean(g) for g in groups],
    }


def pairwise_ttest(df: pd.DataFrame, dataset: str, condition: str) -> List[Dict]:
    """
    Perform pairwise t-tests between strategies.

    Args:
        df: DataFrame with individual run results
        dataset: Dataset name
        condition: Condition name

    Returns:
        List of pairwise comparison results
    """
    subset = df[(df['dataset'] == dataset) & (df['condition'] == condition)]

    results = []
    for i, strat1 in enumerate(STRATEGIES):
        for strat2 in STRATEGIES[i+1:]:
            data1 = subset[subset['strategy'] == strat1]['final_accuracy'].values
            data2 = subset[subset['strategy'] == strat2]['final_accuracy'].values

            if len(data1) < 2 or len(data2) < 2:
                continue

            t_stat, p_value = stats.ttest_ind(data1, data2)

            # Cohen's d effect size
            pooled_std = np.sqrt(((len(data1)-1)*np.std(data1)**2 + (len(data2)-1)*np.std(data2)**2) /
                                  (len(data1) + len(data2) - 2))
            cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0

            results.append({
                'dataset': dataset,
                'condition': condition,
                'strategy1': strat1,
                'strategy2': strat2,
                'mean1': np.mean(data1),
                'mean2': np.mean(data2),
                'diff': np.mean(data1) - np.mean(data2),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'cohens_d': cohens_d,
            })

    return results


def generate_summary_table(agg_df: pd.DataFrame, output_path: str):
    """
    Generate a formatted summary table in Markdown.

    Args:
        agg_df: Aggregated statistics DataFrame
        output_path: Path to save the table
    """
    with open(output_path, 'w') as f:
        f.write("# Baseline Aggregation Comparison: Summary Statistics\n\n")

        for dataset in DATASETS:
            f.write(f"## {dataset.upper().replace('_', '-')}\n\n")

            for condition in CONDITIONS:
                f.write(f"### {condition.replace('_', ' ').title()}\n\n")

                subset = agg_df[(agg_df['dataset'] == dataset) &
                               (agg_df['condition'] == condition)]

                if len(subset) == 0:
                    f.write("*No results available*\n\n")
                    continue

                f.write("| Strategy | Mean Acc (%) | Std | 95% CI | N |\n")
                f.write("|----------|-------------|-----|--------|---|\n")

                for _, row in subset.sort_values('mean_accuracy', ascending=False).iterrows():
                    f.write(f"| {row['strategy'].upper()} | "
                           f"{row['mean_accuracy']:.2f} | "
                           f"{row['std_accuracy']:.2f} | "
                           f"[{row['ci_low']:.2f}, {row['ci_high']:.2f}] | "
                           f"{row['n_runs']} |\n")

                f.write("\n")

        f.write("---\n")
        f.write("*95% CI = 95% Confidence Interval*\n")

    print(f"Summary table saved to: {output_path}")


def generate_comparison_plot(agg_df: pd.DataFrame, output_dir: str):
    """
    Generate comparison bar charts with error bars.

    Args:
        agg_df: Aggregated statistics DataFrame
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: All datasets comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, dataset in enumerate(DATASETS):
        ax = axes[i]
        subset = agg_df[agg_df['dataset'] == dataset]

        x = np.arange(len(CONDITIONS))
        width = 0.25
        colors = {'fedavg': '#3498db', 'fedmean': '#2ecc71', 'fedmedian': '#e74c3c'}

        for j, strategy in enumerate(STRATEGIES):
            strat_data = subset[subset['strategy'] == strategy]
            means = [strat_data[strat_data['condition'] == c]['mean_accuracy'].values[0]
                     if len(strat_data[strat_data['condition'] == c]) > 0 else 0
                     for c in CONDITIONS]
            errs = [strat_data[strat_data['condition'] == c]['std_accuracy'].values[0]
                    if len(strat_data[strat_data['condition'] == c]) > 0 else 0
                    for c in CONDITIONS]

            bars = ax.bar(x + j * width, means, width, label=strategy.upper(),
                         yerr=errs, capsize=3, color=colors[strategy])

        ax.set_xlabel('Data Distribution')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title(dataset.upper().replace('_', '-'))
        ax.set_xticks(x + width)
        ax.set_xticklabels(['IID-Equal', 'IID-Unequal'])
        ax.legend(loc='lower right')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_all_datasets.png'), dpi=150)
    plt.close()

    # Plot 2: Cross-dataset comparison heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, condition in enumerate(CONDITIONS):
        ax = axes[idx]

        # Create matrix
        matrix = np.zeros((len(DATASETS), len(STRATEGIES)))
        for i, dataset in enumerate(DATASETS):
            for j, strategy in enumerate(STRATEGIES):
                val = agg_df[(agg_df['dataset'] == dataset) &
                            (agg_df['condition'] == condition) &
                            (agg_df['strategy'] == strategy)]['mean_accuracy']
                matrix[i, j] = val.values[0] if len(val) > 0 else 0

        im = ax.imshow(matrix, cmap='RdYlGn', vmin=50, vmax=100)

        ax.set_xticks(range(len(STRATEGIES)))
        ax.set_xticklabels([s.upper() for s in STRATEGIES])
        ax.set_yticks(range(len(DATASETS)))
        ax.set_yticklabels([d.upper().replace('_', '-') for d in DATASETS])
        ax.set_title(condition.replace('_', ' ').title())

        # Add text annotations
        for i in range(len(DATASETS)):
            for j in range(len(STRATEGIES)):
                ax.text(j, i, f'{matrix[i, j]:.1f}%',
                       ha='center', va='center', fontsize=10, fontweight='bold')

        plt.colorbar(im, ax=ax, label='Accuracy (%)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_comparison.png'), dpi=150)
    plt.close()

    print(f"Plots saved to: {output_dir}")


def generate_convergence_plots(results_dir: str, output_dir: str):
    """
    Generate convergence trajectory plots.

    Args:
        results_dir: Directory containing raw results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    colors = {'fedavg': '#3498db', 'fedmean': '#2ecc71', 'fedmedian': '#e74c3c'}

    for dataset in DATASETS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for idx, condition in enumerate(CONDITIONS):
            ax = axes[idx]

            for strategy in STRATEGIES:
                pattern = os.path.join(results_dir, dataset, condition, f'{strategy}_*.json')
                files = glob.glob(pattern)

                if not files:
                    continue

                # Load all trajectories
                trajectories = []
                for filepath in files:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        if 'test_accuracy' in data:
                            trajectories.append(data['test_accuracy'])

                if not trajectories:
                    continue

                # Align trajectories and compute mean/std
                min_len = min(len(t) for t in trajectories)
                aligned = np.array([t[:min_len] for t in trajectories])
                mean_traj = np.mean(aligned, axis=0)
                std_traj = np.std(aligned, axis=0)

                rounds = np.arange(len(mean_traj))
                ax.plot(rounds, mean_traj, label=strategy.upper(),
                       color=colors[strategy], linewidth=2)
                ax.fill_between(rounds, mean_traj - std_traj, mean_traj + std_traj,
                              alpha=0.2, color=colors[strategy])

            ax.set_xlabel('Communication Round')
            ax.set_ylabel('Test Accuracy (%)')
            ax.set_title(f'{dataset.upper().replace("_", "-")} - {condition.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'convergence_{dataset}.png'), dpi=150)
        plt.close()

    print(f"Convergence plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate baseline experiment results')
    parser.add_argument('--results_dir', type=str, default='./results/baseline',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='./analysis',
                        help='Directory for output files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Aggregating Baseline Experiment Results")
    print("=" * 60)

    # Load all results
    print("\nLoading results...")
    df = load_results(args.results_dir)
    print(f"  Loaded {len(df)} experiment runs")

    if len(df) == 0:
        print("No results found. Run experiments first.")
        return

    # Compute aggregated statistics
    print("\nComputing statistics...")
    agg_df = aggregate_statistics(df)

    # Save raw aggregated data
    agg_df.to_csv(os.path.join(args.output_dir, 'aggregated_results.csv'), index=False)
    print(f"  Saved: aggregated_results.csv")

    # Generate summary table
    generate_summary_table(agg_df, os.path.join(args.output_dir, 'summary_statistics.md'))

    # Statistical tests
    print("\nPerforming statistical tests...")
    anova_results = []
    ttest_results = []

    for dataset in DATASETS:
        for condition in CONDITIONS:
            # ANOVA
            anova = perform_anova(df, dataset, condition)
            if 'error' not in anova:
                anova_results.append(anova)
                print(f"  {dataset}/{condition}: ANOVA p={anova['p_value']:.4f} "
                      f"({'significant' if anova['significant'] else 'not significant'})")

            # Pairwise t-tests
            ttests = pairwise_ttest(df, dataset, condition)
            ttest_results.extend(ttests)

    # Save statistical test results
    if anova_results:
        pd.DataFrame(anova_results).to_csv(
            os.path.join(args.output_dir, 'anova_results.csv'), index=False)
    if ttest_results:
        pd.DataFrame(ttest_results).to_csv(
            os.path.join(args.output_dir, 'ttest_results.csv'), index=False)

    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_comparison_plot(agg_df, os.path.join(args.output_dir, 'figures'))
    generate_convergence_plots(args.results_dir, os.path.join(args.output_dir, 'figures'))

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)

    # Print quick summary
    print("\nQuick Summary (Mean Accuracy %):")
    pivot = agg_df.pivot_table(
        values='mean_accuracy',
        index=['dataset', 'condition'],
        columns='strategy'
    )
    print(pivot.round(2).to_string())


if __name__ == "__main__":
    main()
