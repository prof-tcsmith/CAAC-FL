#!/usr/bin/env python3
"""
Comprehensive Analysis of All Federated Learning Experiments (Level 1 & Level 2)

This script analyzes results from:
- Level 1: IID experiments (Equal and Unequal client sizes, varying client counts)
- Level 2: Non-IID experiments (Dirichlet label skew with α ∈ {0.1, 0.5, 1.0})

Generates comprehensive visualizations and summary statistics.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11

def load_experiment_results(base_dir):
    """Load all experiment results from Level 1 and Level 2."""
    experiments = []

    # Level 1 results
    level1_dir = Path(base_dir) / "level1_fundamentals" / "results" / "comprehensive"
    if level1_dir.exists():
        for json_file in level1_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
                data['experiment_file'] = json_file.name
                experiments.append(data)

    # Level 2 results
    level2_dir = Path(base_dir) / "level2_heterogeneous" / "results" / "comprehensive"
    if level2_dir.exists():
        for json_file in level2_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
                data['experiment_file'] = json_file.name
                experiments.append(data)

    return experiments

def parse_experiment_metadata(exp):
    """Parse experiment metadata from filename."""
    filename = exp.get('experiment_file', '')

    # Parse Level 1 experiments: level1_{partition}_{aggregation}_c{clients}_metrics.json
    if 'level1' in filename:
        level = 'Level 1'
        parts = filename.replace('level1_', '').replace('_metrics.json', '').split('_')

        # Extract partition (iid-equal or iid-unequal)
        if 'iid-equal' in filename:
            partition = 'iid-equal'
            remaining = filename.split('iid-equal_')[1]
        elif 'iid-unequal' in filename:
            partition = 'iid-unequal'
            remaining = filename.split('iid-unequal_')[1]
        else:
            partition = 'iid-equal'
            remaining = '_'.join(parts[1:])

        # Extract aggregation and clients
        parts2 = remaining.replace('_metrics.json', '').split('_')
        aggregation = parts2[0] if len(parts2) > 0 else 'unknown'

        # Extract client count
        client_part = [p for p in parts2 if p.startswith('c')]
        num_clients = int(client_part[0][1:]) if client_part else 50

        alpha = None
        data_type = f"IID-{partition.split('-')[1].capitalize()}"

    # Parse Level 2 experiments: level2_noniid_{aggregation}_a{alpha}_c{clients}_metrics.json
    elif 'level2' in filename:
        level = 'Level 2'
        partition = 'non-iid'

        # Extract aggregation
        if 'fedavg' in filename:
            aggregation = 'fedavg'
        elif 'fedmedian' in filename:
            aggregation = 'fedmedian'
        elif 'krum' in filename:
            aggregation = 'krum'
        else:
            aggregation = 'unknown'

        # Extract alpha
        import re
        alpha_match = re.search(r'_a([0-9.]+)_', filename)
        alpha = float(alpha_match.group(1)) if alpha_match else 0.5

        # Extract clients
        client_match = re.search(r'_c(\d+)_', filename)
        num_clients = int(client_match.group(1)) if client_match else 50

        data_type = f'Non-IID (α={alpha})'

    else:
        level = 'Unknown'
        partition = 'unknown'
        aggregation = 'unknown'
        num_clients = 50
        alpha = None
        data_type = 'Unknown'

    return {
        'level': level,
        'partition': partition,
        'aggregation': aggregation,
        'num_clients': num_clients,
        'alpha': alpha,
        'data_type': data_type,
        'filename': filename
    }

def is_experiment_failed(exp):
    """Check if experiment failed to train (stuck at random chance accuracy)."""
    test_acc = exp.get('test_accuracy', [])
    if not test_acc or len(test_acc) < 5:
        return True

    # Check if accuracy never improved beyond random chance (10% for 10 classes)
    # Allow small variance for random fluctuations
    max_acc = max(test_acc)
    return max_acc < 15.0  # If never exceeded 15%, it failed

def create_comprehensive_summary(experiments):
    """Create comprehensive summary DataFrame of all experiments."""
    summary_data = []
    failed_experiments = []

    for exp in experiments:
        metadata = parse_experiment_metadata(exp)

        # Get final accuracy
        test_acc = exp.get('test_accuracy', [])
        final_acc = test_acc[-1] if test_acc else 0.0

        # Check if experiment failed
        failed = is_experiment_failed(exp)
        if failed:
            failed_experiments.append(f"{metadata['aggregation'].upper()} ({metadata['data_type']})")

        summary_data.append({
            'Level': metadata['level'],
            'Data Type': metadata['data_type'],
            'Aggregation': metadata['aggregation'].upper(),
            'Clients': metadata['num_clients'],
            'Alpha': metadata['alpha'] if metadata['alpha'] is not None else 'N/A',
            'Final Accuracy (%)': round(final_acc, 2) if not failed else 'FAILED',
            'Partition': metadata['partition'],
            'Status': 'FAILED' if failed else 'OK'
        })

    df = pd.DataFrame(summary_data)
    df = df.sort_values(['Level', 'Data Type', 'Aggregation', 'Clients'])
    return df, failed_experiments

def plot_noniid_comparison(experiments, output_dir):
    """Plot Non-IID performance across different alpha values and aggregation methods."""
    # Filter for Level 2 Non-IID experiments (excluding failed ones)
    noniid_exps = [exp for exp in experiments
                   if parse_experiment_metadata(exp)['level'] == 'Level 2'
                   and not is_experiment_failed(exp)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Group by alpha and aggregation
    alpha_values = [0.1, 0.5, 1.0]
    aggregations = ['fedavg', 'fedmedian', 'krum']
    colors = {'fedavg': '#1f77b4', 'fedmedian': '#ff7f0e', 'krum': '#2ca02c'}
    markers = {'fedavg': 'o', 'fedmedian': 's', 'krum': '^'}

    # Left plot: Convergence curves for α=0.1 (extreme heterogeneity)
    ax1 = axes[0]
    for agg in aggregations:
        for exp in noniid_exps:
            meta = parse_experiment_metadata(exp)
            if meta['aggregation'] == agg and meta['alpha'] == 0.1:
                test_acc = exp.get('test_accuracy', [])
                rounds = list(range(1, len(test_acc) + 1))
                ax1.plot(rounds, test_acc,
                        marker=markers[agg], markersize=4, linewidth=2.5,
                        label=f'{agg.upper()}', color=colors[agg], alpha=0.8)
                break

    ax1.set_xlabel('Communication Round', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Non-IID Convergence (α=0.1, Extreme Heterogeneity)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 21)

    # Right plot: Final accuracy vs alpha for all methods
    ax2 = axes[1]
    final_accs = {agg: [] for agg in aggregations}

    for alpha in alpha_values:
        for agg in aggregations:
            found = False
            for exp in noniid_exps:
                meta = parse_experiment_metadata(exp)
                if meta['aggregation'] == agg and meta['alpha'] == alpha:
                    test_acc = exp.get('test_accuracy', [])
                    final_acc = test_acc[-1] if test_acc else 0.0
                    final_accs[agg].append(final_acc)
                    found = True
                    break
            if not found:
                # Add NaN for missing data
                final_accs[agg].append(np.nan)

    x_pos = np.arange(len(alpha_values))
    # Filter out aggregations with all NaN values
    valid_aggs = [agg for agg in aggregations if not all(np.isnan(final_accs[agg]))]
    width = 0.8 / len(valid_aggs) if valid_aggs else 0.25

    for i, agg in enumerate(valid_aggs):
        offset = (i - len(valid_aggs)/2 + 0.5) * width
        # Filter out NaN values for this aggregation
        valid_data = [v if not np.isnan(v) else 0 for v in final_accs[agg]]
        ax2.bar(x_pos + offset, valid_data, width,
               label=f'{agg.upper()}', color=colors[agg], alpha=0.8, edgecolor='black')

        # Add value labels on bars (skip NaN values)
        for j, v in enumerate(final_accs[agg]):
            if not np.isnan(v):
                ax2.text(x_pos[j] + offset, v + 0.5, f'{v:.1f}%',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_xlabel('Dirichlet Concentration Parameter (α)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Final Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Non-IID Performance vs. Data Heterogeneity', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['0.1\n(Extreme)', '0.5\n(Moderate)', '1.0\n(Mild)'])
    ax2.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max([max(accs) for accs in final_accs.values()]) + 5)

    plt.tight_layout()
    plt.savefig(output_dir / 'noniid_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'noniid_comprehensive_analysis.png'}")
    plt.close()

def plot_krum_analysis(experiments, output_dir):
    """Detailed analysis of Krum performance across all conditions."""
    # Extract Krum experiments (excluding failed ones)
    krum_exps = [exp for exp in experiments
                 if parse_experiment_metadata(exp)['aggregation'] == 'krum'
                 and not is_experiment_failed(exp)]

    # Check if we have any successful Krum experiments
    if not krum_exps:
        print(f"   ⚠ WARNING: No successful Krum experiments found. Skipping Krum analysis plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Krum convergence across different alphas (top-left)
    ax1 = axes[0, 0]
    alpha_values = [0.1, 0.5, 1.0]
    colors_alpha = {0.1: '#d62728', 0.5: '#ff7f0e', 1.0: '#2ca02c'}

    for exp in krum_exps:
        meta = parse_experiment_metadata(exp)
        if meta['alpha'] is not None:
            test_acc = exp.get('test_accuracy', [])
            rounds = list(range(1, len(test_acc) + 1))
            ax1.plot(rounds, test_acc,
                    marker='o', markersize=4, linewidth=2.5,
                    label=f'α={meta["alpha"]}', color=colors_alpha[meta['alpha']], alpha=0.8)

    ax1.set_xlabel('Communication Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Krum: Convergence vs. Data Heterogeneity', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 21)

    # Plot 2: Krum vs Baselines (α=0.1) (top-right)
    ax2 = axes[0, 1]
    aggregations = ['fedavg', 'fedmedian', 'krum']
    colors = {'fedavg': '#1f77b4', 'fedmedian': '#ff7f0e', 'krum': '#2ca02c'}
    markers = {'fedavg': 'o', 'fedmedian': 's', 'krum': '^'}

    for agg in aggregations:
        for exp in experiments:
            meta = parse_experiment_metadata(exp)
            if meta['aggregation'] == agg and meta['alpha'] == 0.1:
                test_acc = exp.get('test_accuracy', [])
                rounds = list(range(1, len(test_acc) + 1))
                ax2.plot(rounds, test_acc,
                        marker=markers[agg], markersize=4, linewidth=2.5,
                        label=f'{agg.upper()}', color=colors[agg], alpha=0.8)
                break

    ax2.set_xlabel('Communication Round', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Krum vs. Baselines (α=0.1, Extreme Heterogeneity)', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 21)

    # Plot 3: Final accuracy comparison across alphas (bottom-left)
    ax3 = axes[1, 0]

    # Collect data
    final_accs = {agg: [] for agg in aggregations}
    alpha_vals = [0.1, 0.5, 1.0]

    for alpha in alpha_vals:
        for agg in aggregations:
            for exp in experiments:
                meta = parse_experiment_metadata(exp)
                if meta['aggregation'] == agg and meta['alpha'] == alpha:
                    test_acc = exp.get('test_accuracy', [])
                    final_acc = test_acc[-1] if test_acc else 0.0
                    final_accs[agg].append(final_acc)
                    break

    x_pos = np.arange(len(alpha_vals))
    width = 0.25

    for i, agg in enumerate(aggregations):
        offset = (i - 1) * width
        bars = ax3.bar(x_pos + offset, final_accs[agg], width,
                      label=f'{agg.upper()}', color=colors[agg], alpha=0.8, edgecolor='black')

        # Highlight Krum bars
        if agg == 'krum':
            for bar in bars:
                bar.set_linewidth(2.5)

        # Add value labels
        for j, v in enumerate(final_accs[agg]):
            ax3.text(x_pos[j] + offset, v + 0.5, f'{v:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax3.set_xlabel('Data Heterogeneity (α)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Final Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Final Performance Comparison', fontsize=13, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(['0.1', '0.5', '1.0'])
    ax3.legend(loc='lower right', fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Relative performance (Krum advantage over FedAvg) (bottom-right)
    ax4 = axes[1, 1]

    krum_advantage = []
    for alpha in alpha_vals:
        krum_acc = None
        fedavg_acc = None

        for exp in experiments:
            meta = parse_experiment_metadata(exp)
            if meta['alpha'] == alpha:
                test_acc = exp.get('test_accuracy', [])
                final_acc = test_acc[-1] if test_acc else 0.0

                if meta['aggregation'] == 'krum':
                    krum_acc = final_acc
                elif meta['aggregation'] == 'fedavg':
                    fedavg_acc = final_acc

        if krum_acc is not None and fedavg_acc is not None:
            krum_advantage.append(krum_acc - fedavg_acc)

    colors_bars = ['#d62728' if adv < 0 else '#2ca02c' for adv in krum_advantage]
    bars = ax4.bar(x_pos, krum_advantage, color=colors_bars, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, v in enumerate(krum_advantage):
        y_pos = v + 0.2 if v > 0 else v - 0.2
        ax4.text(x_pos[i], y_pos, f'{v:+.2f}%',
                ha='center', va='bottom' if v > 0 else 'top',
                fontsize=11, fontweight='bold')

    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax4.set_xlabel('Data Heterogeneity (α)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Krum Advantage over FedAvg (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Krum Relative Performance', fontsize=13, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(['0.1\n(Extreme)', '0.5\n(Moderate)', '1.0\n(Mild)'])
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'krum_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'krum_comprehensive_analysis.png'}")
    plt.close()

def plot_grand_heatmap_all(experiments, output_dir):
    """Create comprehensive heatmap of all experiments."""
    # Prepare data (exclude failed experiments)
    heatmap_data = []

    for exp in experiments:
        if is_experiment_failed(exp):
            continue

        meta = parse_experiment_metadata(exp)
        test_acc = exp.get('test_accuracy', [])
        final_acc = test_acc[-1] if test_acc else 0.0

        row_label = f"{meta['data_type']}, {meta['num_clients']} clients"
        col_label = meta['aggregation'].upper()

        heatmap_data.append({
            'Row': row_label,
            'Column': col_label,
            'Accuracy': final_acc
        })

    df = pd.DataFrame(heatmap_data)
    pivot = df.pivot_table(values='Accuracy', index='Row', columns='Column', aggfunc='mean')

    # Reorder columns
    col_order = ['FEDAVG', 'FEDMEAN', 'FEDMEDIAN', 'KRUM']
    pivot = pivot[[col for col in col_order if col in pivot.columns]]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=pivot.min().min() - 2, vmax=pivot.max().max() + 2,
                cbar_kws={'label': 'Test Accuracy (%)'},
                linewidths=1, linecolor='gray', ax=ax)

    ax.set_xlabel('Aggregation Strategy', fontsize=13, fontweight='bold')
    ax.set_ylabel('Experimental Configuration', fontsize=13, fontweight='bold')
    ax.set_title('Comprehensive Performance Heatmap: All Experiments',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'grand_heatmap_all_experiments.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'grand_heatmap_all_experiments.png'}")
    plt.close()

def plot_iid_vs_noniid_comparison(experiments, output_dir):
    """Compare IID-Equal vs Non-IID performance for common methods."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Methods available in both IID and Non-IID
    common_methods = ['fedavg', 'fedmedian']
    colors = {'fedavg': '#1f77b4', 'fedmedian': '#ff7f0e'}
    markers = {'fedavg': 'o', 'fedmedian': 's'}

    # Left plot: IID-Equal (50 clients)
    ax1 = axes[0]
    for method in common_methods:
        for exp in experiments:
            meta = parse_experiment_metadata(exp)
            if (meta['aggregation'] == method and
                meta['partition'] == 'iid-equal' and
                meta['num_clients'] == 50):
                test_acc = exp.get('test_accuracy', [])
                rounds = list(range(1, len(test_acc) + 1))
                ax1.plot(rounds, test_acc,
                        marker=markers[method], markersize=4, linewidth=2.5,
                        label=f'{method.upper()} (IID-Equal)',
                        color=colors[method], alpha=0.8)
                break

    ax1.set_xlabel('Communication Round', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('IID-Equal Performance (50 clients)', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 21)

    # Right plot: Non-IID α=0.5 (50 clients)
    ax2 = axes[1]
    for method in common_methods:
        for exp in experiments:
            meta = parse_experiment_metadata(exp)
            if (meta['aggregation'] == method and
                meta['alpha'] == 0.5):
                test_acc = exp.get('test_accuracy', [])
                rounds = list(range(1, len(test_acc) + 1))
                ax2.plot(rounds, test_acc,
                        marker=markers[method], markersize=4, linewidth=2.5,
                        label=f'{method.upper()} (Non-IID α=0.5)',
                        color=colors[method], alpha=0.8, linestyle='--')
                break

    ax2.set_xlabel('Communication Round', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Non-IID Performance (50 clients, α=0.5)', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 21)

    plt.tight_layout()
    plt.savefig(output_dir / 'iid_vs_noniid_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'iid_vs_noniid_comparison.png'}")
    plt.close()

def main():
    """Main analysis pipeline."""
    print("=" * 70)
    print("COMPREHENSIVE FEDERATED LEARNING EXPERIMENT ANALYSIS")
    print("=" * 70)

    base_dir = Path(__file__).parent

    # Output directory
    output_dir = base_dir / "papers" / "federated-aggregation-comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all experiments
    print("\n[1/6] Loading experiment results...")
    experiments = load_experiment_results(base_dir)
    print(f"   ✓ Loaded {len(experiments)} experiments")

    # Create comprehensive summary
    print("\n[2/6] Creating comprehensive summary...")
    summary_df, failed_experiments = create_comprehensive_summary(experiments)
    summary_path = output_dir / 'all_experiments_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"   ✓ Saved: {summary_path}")
    print("\n" + summary_df.to_string(index=False))

    # Warn about failed experiments
    if failed_experiments:
        print("\n   ⚠ WARNING: The following experiments FAILED to train (stuck at random chance):")
        for failed in failed_experiments:
            print(f"      - {failed}")
        print("   These experiments will be EXCLUDED from visualizations.")

    # Generate visualizations
    print("\n[3/6] Generating Non-IID comparison plots...")
    plot_noniid_comparison(experiments, output_dir)

    print("\n[4/6] Generating Krum comprehensive analysis...")
    plot_krum_analysis(experiments, output_dir)

    print("\n[5/6] Generating grand heatmap...")
    plot_grand_heatmap_all(experiments, output_dir)

    print("\n[6/6] Generating IID vs Non-IID comparison...")
    plot_iid_vs_noniid_comparison(experiments, output_dir)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - all_experiments_summary.csv")
    print("  - noniid_comprehensive_analysis.png")
    print("  - krum_comprehensive_analysis.png")
    print("  - grand_heatmap_all_experiments.png")
    print("  - iid_vs_noniid_comparison.png")
    print()

if __name__ == "__main__":
    main()
