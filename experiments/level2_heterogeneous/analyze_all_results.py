#!/usr/bin/env python3
"""
Comprehensive analysis of all federated learning experiments.
Analyzes both weight-sharing and gradient-sharing strategies across all conditions.
"""

import json
import os
from pathlib import Path
import numpy as np
from collections import defaultdict
import pandas as pd

RESULTS_DIR = Path("results/noniid")

WEIGHT_STRATEGIES = ['fedavg', 'fedmean', 'fedmedian']
GRADIENT_STRATEGIES = ['fedsgd', 'fedadam', 'fedtrimmed']
ALL_STRATEGIES = WEIGHT_STRATEGIES + GRADIENT_STRATEGIES

CONDITIONS = ['noniid_equal', 'noniid_unequal']
DATASETS = ['mnist', 'fashion_mnist', 'cifar10']
CLIENT_COUNTS = [10, 25, 50]


def load_results():
    """Load all experimental results."""
    results = []

    for json_file in RESULTS_DIR.rglob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Extract info from path if not in data
            path_parts = str(json_file).split('/')
            condition = next((p for p in path_parts if p in CONDITIONS), data.get('condition'))

            # Get final and best accuracy
            acc_history = data.get('test_accuracy', [])
            final_acc = acc_history[-1] if acc_history else 0
            best_acc = max(acc_history) if acc_history else 0

            results.append({
                'dataset': data.get('dataset'),
                'strategy': data.get('strategy'),
                'condition': condition,
                'num_clients': data.get('num_clients'),
                'seed': data.get('seed'),
                'final_accuracy': final_acc,
                'best_accuracy': best_acc,
                'alpha': data.get('alpha', 0.5),
            })
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return pd.DataFrame(results)


def analyze_by_strategy_type(df):
    """Compare weight vs gradient strategies."""
    df = df.copy()
    df['strategy_type'] = df['strategy'].apply(
        lambda s: 'weight' if s in WEIGHT_STRATEGIES else 'gradient'
    )

    print("\n" + "="*80)
    print("WEIGHT vs GRADIENT STRATEGY COMPARISON")
    print("="*80)

    summary = df.groupby(['strategy_type', 'dataset', 'condition', 'num_clients']).agg({
        'final_accuracy': ['mean', 'std', 'count'],
        'best_accuracy': ['mean', 'std']
    }).round(2)

    print(summary)
    return summary


def analyze_by_strategy(df):
    """Detailed analysis by individual strategy."""
    print("\n" + "="*80)
    print("INDIVIDUAL STRATEGY PERFORMANCE")
    print("="*80)

    for dataset in DATASETS:
        print(f"\n### {dataset.upper()} ###")

        for condition in CONDITIONS:
            print(f"\n{condition}:")

            subset = df[(df['dataset'] == dataset) & (df['condition'] == condition)]

            summary = subset.groupby(['strategy', 'num_clients']).agg({
                'final_accuracy': ['mean', 'std'],
            }).round(2)

            # Pivot for better readability
            pivot = subset.groupby(['strategy', 'num_clients'])['final_accuracy'].mean().unstack()
            print(pivot.round(2).to_string())


def find_best_strategies(df):
    """Find best performing strategies for each condition."""
    print("\n" + "="*80)
    print("BEST STRATEGIES BY CONDITION")
    print("="*80)

    for dataset in DATASETS:
        print(f"\n### {dataset.upper()} ###")

        for condition in CONDITIONS:
            for nc in CLIENT_COUNTS:
                subset = df[(df['dataset'] == dataset) &
                           (df['condition'] == condition) &
                           (df['num_clients'] == nc)]

                if subset.empty:
                    continue

                best = subset.groupby('strategy')['final_accuracy'].mean().sort_values(ascending=False)

                print(f"\n{condition}, {nc} clients:")
                for i, (strat, acc) in enumerate(best.head(3).items()):
                    marker = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                    print(f"  {marker} {strat}: {acc:.2f}%")


def compare_weight_vs_gradient(df):
    """Direct comparison of weight vs gradient strategies."""
    print("\n" + "="*80)
    print("WEIGHT vs GRADIENT HEAD-TO-HEAD")
    print("="*80)

    df = df.copy()
    df['strategy_type'] = df['strategy'].apply(
        lambda s: 'weight' if s in WEIGHT_STRATEGIES else 'gradient'
    )

    comparison = []

    for dataset in DATASETS:
        for condition in CONDITIONS:
            for nc in CLIENT_COUNTS:
                subset = df[(df['dataset'] == dataset) &
                           (df['condition'] == condition) &
                           (df['num_clients'] == nc)]

                if subset.empty:
                    continue

                weight_mean = subset[subset['strategy_type'] == 'weight']['final_accuracy'].mean()
                gradient_mean = subset[subset['strategy_type'] == 'gradient']['final_accuracy'].mean()

                diff = weight_mean - gradient_mean
                winner = "weight" if diff > 0 else "gradient"

                comparison.append({
                    'dataset': dataset,
                    'condition': condition,
                    'num_clients': nc,
                    'weight_acc': weight_mean,
                    'gradient_acc': gradient_mean,
                    'diff': diff,
                    'winner': winner
                })

    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))

    # Summary
    weight_wins = (comp_df['winner'] == 'weight').sum()
    gradient_wins = (comp_df['winner'] == 'gradient').sum()

    print(f"\n--- Summary ---")
    print(f"Weight strategies win: {weight_wins}/{len(comp_df)}")
    print(f"Gradient strategies win: {gradient_wins}/{len(comp_df)}")
    print(f"Average difference (weight - gradient): {comp_df['diff'].mean():.2f}%")

    return comp_df


def generate_summary_statistics(df):
    """Generate comprehensive summary statistics."""
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY STATISTICS")
    print("="*80)

    # Overall stats
    print(f"\nTotal experiments: {len(df)}")
    print(f"Unique strategies: {df['strategy'].nunique()}")
    print(f"Datasets: {df['dataset'].unique().tolist()}")
    print(f"Conditions: {df['condition'].unique().tolist()}")
    print(f"Client counts: {sorted(df['num_clients'].unique())}")

    # Per-strategy overall performance
    print("\n--- Overall Strategy Performance (mean ± std) ---")
    overall = df.groupby('strategy')['final_accuracy'].agg(['mean', 'std', 'count'])
    overall = overall.sort_values('mean', ascending=False)
    for strat, row in overall.iterrows():
        print(f"  {strat}: {row['mean']:.2f} ± {row['std']:.2f}% (n={int(row['count'])})")

    return overall


def analyze_client_scaling_effect(df):
    """Analyze how client count affects performance."""
    print("\n" + "="*80)
    print("CLIENT SCALING EFFECT")
    print("="*80)

    for dataset in DATASETS:
        print(f"\n### {dataset.upper()} ###")

        subset = df[df['dataset'] == dataset]
        scaling = subset.groupby(['strategy', 'num_clients'])['final_accuracy'].mean().unstack()

        # Calculate drop from c10 to c50
        if 10 in scaling.columns and 50 in scaling.columns:
            scaling['drop_10_to_50'] = scaling[10] - scaling[50]

        print(scaling.round(2).to_string())


def save_summary_csv(df, filename="comprehensive_analysis.csv"):
    """Save summary to CSV."""
    summary = df.groupby(['dataset', 'condition', 'num_clients', 'strategy']).agg({
        'final_accuracy': ['mean', 'std', 'count'],
        'best_accuracy': ['mean']
    }).round(2)

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    summary.to_csv(filename, index=False)
    print(f"\nSummary saved to {filename}")


if __name__ == "__main__":
    print("Loading all experimental results...")
    df = load_results()
    print(f"Loaded {len(df)} results")

    if len(df) == 0:
        print("No results found!")
        exit(1)

    # Run all analyses
    generate_summary_statistics(df)
    analyze_by_strategy(df)
    find_best_strategies(df)
    comp_df = compare_weight_vs_gradient(df)
    analyze_client_scaling_effect(df)

    # Save summary
    save_summary_csv(df)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
