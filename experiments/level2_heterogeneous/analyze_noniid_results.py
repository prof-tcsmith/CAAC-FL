#!/usr/bin/env python3
"""
Analyze Level 2 Non-IID experiment results.

Generates summary statistics, comparison tables, and visualizations
for the non-IID aggregation comparison study.
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'noniid')
ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), 'analysis')


def load_all_results(results_dir=RESULTS_DIR):
    """Load all experiment results from JSON files."""
    results = []

    pattern = os.path.join(results_dir, '**', '*.json')
    for filepath in glob.glob(pattern, recursive=True):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")

    return results


def create_summary_dataframe(results):
    """Create a summary DataFrame from results."""
    rows = []
    for r in results:
        rows.append({
            'dataset': r['dataset'],
            'strategy': r['strategy'],
            'alpha': r['alpha'],
            'seed': r['seed'],
            'final_accuracy': r['final_accuracy'],
            'best_accuracy': r['best_accuracy'],
            'final_loss': r['final_loss'],
            'heterogeneity_kl': r.get('heterogeneity_kl', np.nan),
            'total_time': r['total_time_seconds'],
        })

    return pd.DataFrame(rows)


def compute_statistics(df, groupby_cols):
    """Compute summary statistics grouped by specified columns."""
    stats_df = df.groupby(groupby_cols).agg({
        'final_accuracy': ['mean', 'std', 'min', 'max', 'count'],
        'best_accuracy': ['mean', 'std'],
        'final_loss': ['mean', 'std'],
        'heterogeneity_kl': ['mean'],
    }).round(2)

    # Flatten column names
    stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]

    # Compute 95% CI
    n = stats_df['final_accuracy_count']
    std = stats_df['final_accuracy_std']
    stats_df['final_accuracy_ci95'] = 1.96 * std / np.sqrt(n)

    return stats_df.reset_index()


def generate_markdown_summary(df, output_path):
    """Generate markdown summary file."""

    with open(output_path, 'w') as f:
        f.write("# Level 2 Non-IID Aggregation Comparison: Summary Statistics\n\n")

        for dataset in df['dataset'].unique():
            f.write(f"## {dataset.upper().replace('_', '-')}\n\n")

            for alpha in sorted(df['alpha'].unique()):
                subset = df[(df['dataset'] == dataset) & (df['alpha'] == alpha)]

                if len(subset) == 0:
                    continue

                f.write(f"### Alpha = {alpha}\n\n")

                stats = compute_statistics(subset, ['strategy'])

                f.write("| Strategy | Mean Acc (%) | Std | 95% CI | N |\n")
                f.write("|----------|-------------|-----|--------|---|\n")

                for _, row in stats.sort_values('final_accuracy_mean', ascending=False).iterrows():
                    ci = row['final_accuracy_ci95']
                    ci_low = row['final_accuracy_mean'] - ci
                    ci_high = row['final_accuracy_mean'] + ci
                    f.write(f"| {row['strategy'].upper()} | {row['final_accuracy_mean']:.2f} | "
                            f"{row['final_accuracy_std']:.2f} | [{ci_low:.2f}, {ci_high:.2f}] | "
                            f"{int(row['final_accuracy_count'])} |\n")

                f.write("\n")

            f.write("\n")

        f.write("---\n")
        f.write("*95% CI = 95% Confidence Interval*\n")
        f.write(f"\n*Generated from {len(df)} experiments*\n")

    print(f"Summary written to: {output_path}")


def compare_with_level1(noniid_df, level1_path=None):
    """Compare non-IID results with Level 1 IID baseline."""

    if level1_path is None:
        level1_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'level1_fundamentals', 'results', 'baseline'
        )

    # Load Level 1 results
    level1_results = []
    pattern = os.path.join(level1_path, '**', '*.json')
    for filepath in glob.glob(pattern, recursive=True):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                level1_results.append(data)
        except:
            pass

    if not level1_results:
        print("Warning: No Level 1 results found for comparison")
        return None

    # Create Level 1 DataFrame
    l1_rows = []
    for r in level1_results:
        l1_rows.append({
            'dataset': r['dataset'],
            'strategy': r['strategy'],
            'condition': r['condition'],
            'seed': r['seed'],
            'final_accuracy': r['final_accuracy'],
        })

    l1_df = pd.DataFrame(l1_rows)

    # Compute comparison
    comparison = []

    for dataset in noniid_df['dataset'].unique():
        for strategy in noniid_df['strategy'].unique():
            # Non-IID (alpha=0.5) results
            noniid = noniid_df[(noniid_df['dataset'] == dataset) &
                               (noniid_df['strategy'] == strategy) &
                               (noniid_df['alpha'] == 0.5)]

            # IID-Equal results from Level 1
            iid_equal = l1_df[(l1_df['dataset'] == dataset) &
                              (l1_df['strategy'] == strategy) &
                              (l1_df['condition'] == 'iid_equal')]

            # IID-Unequal results from Level 1
            iid_unequal = l1_df[(l1_df['dataset'] == dataset) &
                                (l1_df['strategy'] == strategy) &
                                (l1_df['condition'] == 'iid_unequal')]

            if len(noniid) > 0 and len(iid_equal) > 0:
                comparison.append({
                    'dataset': dataset,
                    'strategy': strategy,
                    'iid_equal_acc': iid_equal['final_accuracy'].mean(),
                    'iid_unequal_acc': iid_unequal['final_accuracy'].mean() if len(iid_unequal) > 0 else np.nan,
                    'noniid_acc': noniid['final_accuracy'].mean(),
                    'iid_vs_noniid_delta': iid_equal['final_accuracy'].mean() - noniid['final_accuracy'].mean(),
                })

    return pd.DataFrame(comparison)


def run_statistical_tests(df):
    """Run ANOVA and post-hoc tests for each dataset."""

    results = {}

    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]

        # Group by strategy
        groups = [subset[subset['strategy'] == s]['final_accuracy'].values
                  for s in subset['strategy'].unique()]

        if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
            # One-way ANOVA
            f_stat, p_value = stats.f_oneway(*groups)

            results[dataset] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }

    return results


def main():
    print("=" * 60)
    print("Level 2 Non-IID Results Analysis")
    print("=" * 60)

    # Load results
    results = load_all_results()

    if not results:
        print("No results found. Run experiments first.")
        return

    print(f"Loaded {len(results)} experiment results")

    # Create DataFrame
    df = create_summary_dataframe(results)

    # Create analysis directory
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    # Generate summary
    summary_path = os.path.join(ANALYSIS_DIR, 'summary_statistics.md')
    generate_markdown_summary(df, summary_path)

    # Save raw statistics
    stats_df = compute_statistics(df, ['dataset', 'alpha', 'strategy'])
    stats_path = os.path.join(ANALYSIS_DIR, 'detailed_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"Detailed statistics saved to: {stats_path}")

    # Compare with Level 1
    comparison_df = compare_with_level1(df)
    if comparison_df is not None and len(comparison_df) > 0:
        comparison_path = os.path.join(ANALYSIS_DIR, 'level1_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Level 1 comparison saved to: {comparison_path}")

        print("\n--- IID vs Non-IID Comparison ---")
        print(comparison_df.to_string(index=False))

    # Statistical tests
    print("\n--- Statistical Tests (ANOVA) ---")
    test_results = run_statistical_tests(df)
    for dataset, result in test_results.items():
        sig = "***" if result['significant'] else ""
        print(f"{dataset}: F={result['f_statistic']:.2f}, p={result['p_value']:.4f} {sig}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
