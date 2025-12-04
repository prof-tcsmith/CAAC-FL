"""
Level 5a: CAAC-FL Results Analysis

This script analyzes the comprehensive experiment results and generates:
1. Summary statistics tables
2. Comparison plots (CAAC-FL vs FedAvg baseline)
3. Detection performance analysis
4. Convergence curves
5. Cold-start effectiveness analysis

Usage:
    python analyze_results.py [--results_dir ./results/comprehensive]
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from glob import glob
from collections import defaultdict

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plots will be skipped.")


def load_all_results(results_dir: str) -> list:
    """Load all JSON result files from directory."""
    results = []
    json_files = glob(os.path.join(results_dir, "*.json"))

    for f in json_files:
        if 'all_experiments' in f or 'summary' in f:
            continue
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                data['filename'] = os.path.basename(f)
                results.append(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    print(f"Loaded {len(results)} experiment results")
    return results


def results_to_dataframe(results: list) -> pd.DataFrame:
    """Convert results list to pandas DataFrame."""
    rows = []

    for r in results:
        config = r.get('config', {})
        summary = r.get('summary', {})
        detection = summary.get('detection_stats', {})

        row = {
            'experiment': r.get('experiment_name', r.get('filename', '')),
            'strategy': r.get('strategy', 'caacfl' if config.get('use_caacfl', True) else 'fedavg'),
            'attack': config.get('attack', 'unknown'),
            'byzantine_ratio': config.get('byzantine_ratio', 0),
            'num_byzantine': config.get('num_byzantine', 0),
            'seed': config.get('seed', 0),
            'num_rounds': config.get('num_rounds', 0),
            'num_clients': config.get('num_clients', 0),
            'alpha': config.get('alpha', 0.5),
            'final_accuracy': summary.get('final_accuracy', 0),
            'best_accuracy': summary.get('best_accuracy', 0),
            'final_loss': summary.get('final_loss', 0),
            'total_time': summary.get('total_time', 0),
            'detection_rate': detection.get('detection_rate'),
            'total_detections': detection.get('total_detections', 0),
            'false_positives': detection.get('total_false_positives', 0),
        }

        # Extract round-by-round data
        rounds_data = r.get('rounds', [])
        if rounds_data:
            row['accuracies'] = [rd['accuracy'] for rd in rounds_data]
            row['losses'] = [rd['loss'] for rd in rounds_data]

            # Early round accuracy (cold-start period)
            early_accs = [rd['accuracy'] for rd in rounds_data[:5]]
            row['early_accuracy_mean'] = np.mean(early_accs) if early_accs else 0

            # Late round accuracy
            late_accs = [rd['accuracy'] for rd in rounds_data[-5:]]
            row['late_accuracy_mean'] = np.mean(late_accs) if late_accs else 0

        rows.append(row)

    return pd.DataFrame(rows)


def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate aggregated summary statistics."""
    # Group by strategy, attack, byzantine_ratio
    grouped = df.groupby(['strategy', 'attack', 'byzantine_ratio']).agg({
        'final_accuracy': ['mean', 'std', 'min', 'max'],
        'best_accuracy': ['mean', 'std'],
        'detection_rate': ['mean', 'std'],
        'false_positives': ['mean', 'std'],
        'total_time': 'mean',
        'seed': 'count'
    }).round(2)

    # Flatten column names
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.rename(columns={'seed_count': 'n_runs'})

    return grouped.reset_index()


def generate_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate CAAC-FL vs FedAvg comparison table."""
    rows = []

    for attack in df['attack'].unique():
        for byz_ratio in df['byzantine_ratio'].unique():
            caacfl = df[(df['strategy'] == 'caacfl') &
                       (df['attack'] == attack) &
                       (df['byzantine_ratio'] == byz_ratio)]
            fedavg = df[(df['strategy'] == 'fedavg') &
                       (df['attack'] == attack) &
                       (df['byzantine_ratio'] == byz_ratio)]

            if len(caacfl) > 0 and len(fedavg) > 0:
                row = {
                    'attack': attack,
                    'byzantine_ratio': byz_ratio,
                    'caacfl_accuracy': f"{caacfl['final_accuracy'].mean():.2f} ± {caacfl['final_accuracy'].std():.2f}",
                    'fedavg_accuracy': f"{fedavg['final_accuracy'].mean():.2f} ± {fedavg['final_accuracy'].std():.2f}",
                    'improvement': caacfl['final_accuracy'].mean() - fedavg['final_accuracy'].mean(),
                    'caacfl_detection_rate': f"{caacfl['detection_rate'].mean():.1f}%" if caacfl['detection_rate'].notna().any() else "N/A",
                }
                rows.append(row)
            elif len(caacfl) > 0:
                row = {
                    'attack': attack,
                    'byzantine_ratio': byz_ratio,
                    'caacfl_accuracy': f"{caacfl['final_accuracy'].mean():.2f} ± {caacfl['final_accuracy'].std():.2f}",
                    'fedavg_accuracy': "N/A",
                    'improvement': None,
                    'caacfl_detection_rate': f"{caacfl['detection_rate'].mean():.1f}%" if caacfl['detection_rate'].notna().any() else "N/A",
                }
                rows.append(row)

    return pd.DataFrame(rows)


def plot_accuracy_comparison(df: pd.DataFrame, output_dir: str):
    """Plot accuracy comparison between CAAC-FL and FedAvg."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    attacks = [a for a in df['attack'].unique() if a != 'none']
    byz_ratios = sorted([r for r in df['byzantine_ratio'].unique() if r > 0])

    for idx, attack in enumerate(attacks[:4]):
        ax = axes[idx // 2, idx % 2]

        caacfl_accs = []
        fedavg_accs = []
        x_labels = []

        for byz_ratio in byz_ratios:
            caacfl = df[(df['strategy'] == 'caacfl') &
                       (df['attack'] == attack) &
                       (df['byzantine_ratio'] == byz_ratio)]
            fedavg = df[(df['strategy'] == 'fedavg') &
                       (df['attack'] == attack) &
                       (df['byzantine_ratio'] == byz_ratio)]

            if len(caacfl) > 0:
                caacfl_accs.append((caacfl['final_accuracy'].mean(), caacfl['final_accuracy'].std()))
            else:
                caacfl_accs.append((0, 0))

            if len(fedavg) > 0:
                fedavg_accs.append((fedavg['final_accuracy'].mean(), fedavg['final_accuracy'].std()))
            else:
                fedavg_accs.append((0, 0))

            x_labels.append(f"{int(byz_ratio*100)}%")

        x = np.arange(len(x_labels))
        width = 0.35

        caacfl_means = [a[0] for a in caacfl_accs]
        caacfl_stds = [a[1] for a in caacfl_accs]
        fedavg_means = [a[0] for a in fedavg_accs]
        fedavg_stds = [a[1] for a in fedavg_accs]

        bars1 = ax.bar(x - width/2, caacfl_means, width, yerr=caacfl_stds,
                       label='CAAC-FL', color='steelblue', capsize=3)
        bars2 = ax.bar(x + width/2, fedavg_means, width, yerr=fedavg_stds,
                       label='FedAvg', color='coral', capsize=3)

        ax.set_xlabel('Byzantine Ratio')
        ax.set_ylabel('Final Accuracy (%)')
        ax.set_title(f'{attack.replace("_", " ").title()} Attack')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.legend()
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved: accuracy_comparison.png")


def plot_convergence_curves(df: pd.DataFrame, output_dir: str):
    """Plot convergence curves for different conditions."""
    if not HAS_MATPLOTLIB:
        return

    # Find experiments with accuracy history
    df_with_history = df[df['accuracies'].apply(lambda x: isinstance(x, list) and len(x) > 0)]

    if len(df_with_history) == 0:
        print("No accuracy history available for convergence plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Clean (no attack) comparison
    ax = axes[0, 0]
    clean_caacfl = df_with_history[(df_with_history['strategy'] == 'caacfl') &
                                    (df_with_history['attack'] == 'none')]
    clean_fedavg = df_with_history[(df_with_history['strategy'] == 'fedavg') &
                                    (df_with_history['attack'] == 'none')]

    if len(clean_caacfl) > 0:
        accs = np.array(list(clean_caacfl['accuracies']))
        mean_acc = accs.mean(axis=0)
        std_acc = accs.std(axis=0)
        rounds = np.arange(1, len(mean_acc) + 1)
        ax.plot(rounds, mean_acc, label='CAAC-FL', color='steelblue')
        ax.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2, color='steelblue')

    if len(clean_fedavg) > 0:
        accs = np.array(list(clean_fedavg['accuracies']))
        mean_acc = accs.mean(axis=0)
        std_acc = accs.std(axis=0)
        rounds = np.arange(1, len(mean_acc) + 1)
        ax.plot(rounds, mean_acc, label='FedAvg', color='coral')
        ax.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2, color='coral')

    ax.set_xlabel('Round')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('No Attack (Clean)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2-4: Different attacks at 20% Byzantine
    attacks_to_plot = ['sign_flipping', 'random_noise', 'alie']
    for idx, attack in enumerate(attacks_to_plot):
        ax = axes[(idx + 1) // 2, (idx + 1) % 2]

        attack_caacfl = df_with_history[(df_with_history['strategy'] == 'caacfl') &
                                         (df_with_history['attack'] == attack) &
                                         (df_with_history['byzantine_ratio'] == 0.2)]
        attack_fedavg = df_with_history[(df_with_history['strategy'] == 'fedavg') &
                                         (df_with_history['attack'] == attack) &
                                         (df_with_history['byzantine_ratio'] == 0.2)]

        if len(attack_caacfl) > 0:
            accs = np.array(list(attack_caacfl['accuracies']))
            mean_acc = accs.mean(axis=0)
            std_acc = accs.std(axis=0)
            rounds = np.arange(1, len(mean_acc) + 1)
            ax.plot(rounds, mean_acc, label='CAAC-FL', color='steelblue')
            ax.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2, color='steelblue')

        if len(attack_fedavg) > 0:
            accs = np.array(list(attack_fedavg['accuracies']))
            mean_acc = accs.mean(axis=0)
            std_acc = accs.std(axis=0)
            rounds = np.arange(1, len(mean_acc) + 1)
            ax.plot(rounds, mean_acc, label='FedAvg', color='coral')
            ax.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2, color='coral')

        ax.set_xlabel('Round')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{attack.replace("_", " ").title()} (20% Byzantine)')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_curves.png'), dpi=150)
    plt.close()
    print(f"Saved: convergence_curves.png")


def plot_detection_performance(df: pd.DataFrame, output_dir: str):
    """Plot Byzantine detection performance."""
    if not HAS_MATPLOTLIB:
        return

    caacfl_df = df[df['strategy'] == 'caacfl']

    if len(caacfl_df) == 0 or caacfl_df['detection_rate'].isna().all():
        print("No detection data available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Detection rate by attack type
    ax = axes[0]
    attacks = [a for a in caacfl_df['attack'].unique() if a != 'none']

    for attack in attacks:
        attack_data = caacfl_df[(caacfl_df['attack'] == attack) &
                                (caacfl_df['detection_rate'].notna())]
        if len(attack_data) > 0:
            grouped = attack_data.groupby('byzantine_ratio')['detection_rate'].agg(['mean', 'std'])
            ax.errorbar(grouped.index * 100, grouped['mean'], yerr=grouped['std'],
                       label=attack.replace('_', ' ').title(), marker='o', capsize=3)

    ax.set_xlabel('Byzantine Ratio (%)')
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title('Byzantine Detection Rate by Attack Type')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)

    # Plot 2: False positive rate
    ax = axes[1]
    for attack in attacks:
        attack_data = caacfl_df[(caacfl_df['attack'] == attack)]
        if len(attack_data) > 0:
            # Calculate FP rate: FP / (total rounds * honest clients)
            attack_data = attack_data.copy()
            honest_clients = attack_data['num_clients'] - attack_data['num_byzantine']
            total_honest_rounds = honest_clients * attack_data['num_rounds']
            attack_data['fp_rate'] = (attack_data['false_positives'] / total_honest_rounds) * 100

            grouped = attack_data.groupby('byzantine_ratio')['fp_rate'].agg(['mean', 'std'])
            ax.errorbar(grouped.index * 100, grouped['mean'], yerr=grouped['std'],
                       label=attack.replace('_', ' ').title(), marker='o', capsize=3)

    ax.set_xlabel('Byzantine Ratio (%)')
    ax.set_ylabel('False Positive Rate (%)')
    ax.set_title('False Positive Rate by Attack Type')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_performance.png'), dpi=150)
    plt.close()
    print(f"Saved: detection_performance.png")


def plot_cold_start_analysis(df: pd.DataFrame, output_dir: str):
    """Analyze cold-start mitigation effectiveness."""
    if not HAS_MATPLOTLIB:
        return

    df_with_history = df[df['accuracies'].apply(lambda x: isinstance(x, list) and len(x) > 0)]

    if len(df_with_history) == 0:
        print("No accuracy history available for cold-start analysis")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Compare early rounds (1-5) vs late rounds (last 5)
    strategies = ['caacfl', 'fedavg']
    attacks = [a for a in df['attack'].unique() if a != 'none']

    x = np.arange(len(attacks))
    width = 0.2

    for i, strategy in enumerate(strategies):
        early_improvements = []
        late_improvements = []

        for attack in attacks:
            attack_data = df_with_history[(df_with_history['strategy'] == strategy) &
                                           (df_with_history['attack'] == attack) &
                                           (df_with_history['byzantine_ratio'] == 0.2)]

            if len(attack_data) > 0:
                early_improvements.append(attack_data['early_accuracy_mean'].mean())
                late_improvements.append(attack_data['late_accuracy_mean'].mean())
            else:
                early_improvements.append(0)
                late_improvements.append(0)

        offset = (i - 0.5) * width * 2
        bars = ax.bar(x + offset, late_improvements, width,
                     label=f'{strategy.upper()} (Final)', alpha=0.8)

    ax.set_xlabel('Attack Type')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Final Round Accuracy Under Attack (20% Byzantine)')
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace('_', '\n') for a in attacks])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cold_start_analysis.png'), dpi=150)
    plt.close()
    print(f"Saved: cold_start_analysis.png")


def generate_latex_table(df: pd.DataFrame, output_dir: str):
    """Generate LaTeX table for paper."""
    comparison = generate_comparison_table(df)

    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\caption{CAAC-FL vs FedAvg Accuracy Comparison}\n"
    latex += "\\begin{tabular}{llccc}\n"
    latex += "\\toprule\n"
    latex += "Attack & Byz. Ratio & CAAC-FL & FedAvg & Improvement \\\\\n"
    latex += "\\midrule\n"

    for _, row in comparison.iterrows():
        imp = row['improvement']
        imp_str = f"+{imp:.2f}" if imp and imp > 0 else (f"{imp:.2f}" if imp else "N/A")
        latex += f"{row['attack'].replace('_', ' ')} & {int(row['byzantine_ratio']*100)}\\% & "
        latex += f"{row['caacfl_accuracy']} & {row['fedavg_accuracy']} & {imp_str} \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    latex_file = os.path.join(output_dir, 'results_table.tex')
    with open(latex_file, 'w') as f:
        f.write(latex)
    print(f"Saved: results_table.tex")


def main():
    parser = argparse.ArgumentParser(description='Analyze CAAC-FL experiment results')
    parser.add_argument('--results_dir', type=str, default='./results/comprehensive',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for analysis (default: results_dir/analysis)')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, 'analysis')

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("CAAC-FL Results Analysis")
    print("=" * 70)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

    # Load results
    results = load_all_results(args.results_dir)

    if not results:
        print("No results found!")
        return

    # Convert to DataFrame
    df = results_to_dataframe(results)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Strategies: {df['strategy'].unique()}")
    print(f"Attacks: {df['attack'].unique()}")
    print(f"Byzantine ratios: {df['byzantine_ratio'].unique()}")

    # Generate summary tables
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    summary = generate_summary_table(df)
    print(summary.to_string())
    summary.to_csv(os.path.join(args.output_dir, 'summary_statistics.csv'), index=False)

    print("\n" + "=" * 70)
    print("CAAC-FL vs FedAvg COMPARISON")
    print("=" * 70)

    comparison = generate_comparison_table(df)
    print(comparison.to_string())
    comparison.to_csv(os.path.join(args.output_dir, 'comparison_table.csv'), index=False)

    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    plot_accuracy_comparison(df, args.output_dir)
    plot_convergence_curves(df, args.output_dir)
    plot_detection_performance(df, args.output_dir)
    plot_cold_start_analysis(df, args.output_dir)

    # Generate LaTeX table
    generate_latex_table(df, args.output_dir)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
