"""
Comprehensive Analysis Script for Multi-Dimensional Federated Learning Experiments

Analyzes results across:
1. Data Distribution: IID-Equal vs IID-Unequal vs Non-IID
2. Aggregation Methods: FedAvg vs FedMean vs FedMedian vs Krum
3. Client Counts: 10 vs 25 vs 50
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

def load_experiment_data(results_dir='./results/comprehensive'):
    """Load all experiment results"""
    results_dir = Path(results_dir)
    experiments = {}

    # Find all JSON result files
    for json_file in results_dir.glob('*.json'):
        exp_name = json_file.stem.replace('_metrics', '')
        with open(json_file, 'r') as f:
            data = json.load(f)
            experiments[exp_name] = data

    print(f"Loaded {len(experiments)} experiments")
    return experiments


def parse_experiment_name(exp_name):
    """Extract metadata from experiment name"""
    parts = exp_name.split('_')

    metadata = {
        'level': None,
        'partition': None,
        'aggregation': None,
        'clients': None,
        'alpha': None
    }

    if 'level1' in exp_name:
        metadata['level'] = 'level1'
        # level1_iid-equal_fedavg_c50 or level1_iid-unequal_fedmean_c25
        if 'iid-equal' in exp_name:
            metadata['partition'] = 'IID-Equal'
        elif 'iid-unequal' in exp_name:
            metadata['partition'] = 'IID-Unequal'

        for agg in ['fedavg', 'fedmean', 'fedmedian']:
            if agg in exp_name:
                metadata['aggregation'] = agg.capitalize()

    elif 'level2' in exp_name:
        metadata['level'] = 'level2'
        metadata['partition'] = 'Non-IID'
        # level2_noniid_fedavg_a0.5_c50
        for agg in ['fedavg', 'fedmedian', 'krum']:
            if agg in exp_name:
                metadata['aggregation'] = agg.capitalize()

        # Extract alpha value
        for part in parts:
            if part.startswith('a'):
                try:
                    metadata['alpha'] = float(part[1:])
                except:
                    pass

    # Extract client count
    for part in parts:
        if part.startswith('c') and part[1:].isdigit():
            metadata['clients'] = int(part[1:])

    return metadata


def create_comparison_plots(experiments, output_dir='./results/comprehensive'):
    """Create comprehensive comparison plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Organize data
    data_records = []
    for exp_name, exp_data in experiments.items():
        metadata = parse_experiment_name(exp_name)
        if metadata['aggregation'] and 'test_accuracy' in exp_data:
            final_acc = exp_data['test_accuracy'][-1]
            rounds = len(exp_data['test_accuracy']) - 1

            data_records.append({
                **metadata,
                'experiment': exp_name,
                'final_accuracy': final_acc,
                'rounds': rounds,
                'test_accuracy_series': exp_data['test_accuracy'],
                'test_loss_series': exp_data['test_loss']
            })

    df = pd.DataFrame(data_records)
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"\nTotal experiments: {len(df)}")
    print(f"\nBreakdown:")
    print(f"  Partitions: {df['partition'].value_counts().to_dict()}")
    print(f"  Aggregation: {df['aggregation'].value_counts().to_dict()}")
    print(f"  Client counts: {df['clients'].value_counts().to_dict()}")

    # Plot 1: IID-Equal vs IID-Unequal (Test H1: FedAvg weighting advantage)
    print("\n" + "=" * 70)
    print("PLOT 1: IID-Equal vs IID-Unequal (FedAvg Weighting Test)")
    print("=" * 70)

    iid_data = df[df['partition'].isin(['IID-Equal', 'IID-Unequal']) &
                   (df['clients'] == 50)]

    if len(iid_data) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))

        for _, row in iid_data.iterrows():
            label = f"{row['partition']} - {row['aggregation']}"
            style = '--' if row['partition'] == 'IID-Equal' else '-'
            ax.plot(row['test_accuracy_series'], label=label, linestyle=style, linewidth=2)

        ax.set_xlabel('Communication Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Hypothesis Test: FedAvg Weighting Advantage\n' +
                    'IID-Equal (dashed) vs IID-Unequal (solid)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'comparison_iid_equal_vs_unequal.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: comparison_iid_equal_vs_unequal.png")
        plt.close()

    # Plot 2: Non-IID Comparison (Different Alpha Values)
    print("\nPLOT 2: Non-IID Heterogeneity (Different Alpha Values)")
    print("-" * 70)

    noniid_data = df[df['partition'] == 'Non-IID']

    if len(noniid_data) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # By Alpha
        for alpha in sorted(noniid_data['alpha'].dropna().unique()):
            subset = noniid_data[noniid_data['alpha'] == alpha]
            for _, row in subset.iterrows():
                label = f"α={alpha} - {row['aggregation']}"
                axes[0].plot(row['test_accuracy_series'], label=label, linewidth=2)

        axes[0].set_xlabel('Communication Round', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        axes[0].set_title('Non-IID Performance by Dirichlet α', fontsize=14, fontweight='bold')
        axes[0].legend(loc='lower right', fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # Final Accuracy Bar Chart
        pivot = noniid_data.pivot_table(values='final_accuracy',
                                        index='aggregation',
                                        columns='alpha',
                                        aggfunc='first')

        pivot.plot(kind='bar', ax=axes[1], width=0.8)
        axes[1].set_ylabel('Final Test Accuracy (%)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Aggregation Method', fontsize=12, fontweight='bold')
        axes[1].set_title('Final Accuracy by α and Aggregation', fontsize=14, fontweight='bold')
        axes[1].legend(title='Dirichlet α', fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

        plt.tight_layout()
        plt.savefig(output_dir / 'comparison_noniid_alpha.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: comparison_noniid_alpha.png")
        plt.close()

    # Plot 3: Client Scaling
    print("\nPLOT 3: Client Scaling (10, 25, 50 clients)")
    print("-" * 70)

    scaling_data = df[df['partition'] == 'IID-Equal']

    if len(scaling_data) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Convergence curves
        for agg in scaling_data['aggregation'].unique():
            subset = scaling_data[scaling_data['aggregation'] == agg].sort_values('clients')
            for _, row in subset.iterrows():
                label = f"{row['aggregation']} - {row['clients']} clients"
                axes[0].plot(row['test_accuracy_series'], label=label, linewidth=2)

        axes[0].set_xlabel('Communication Round', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        axes[0].set_title('Convergence vs Client Count', fontsize=14, fontweight='bold')
        axes[0].legend(loc='lower right', fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # Final accuracy vs clients
        pivot2 = scaling_data.pivot_table(values='final_accuracy',
                                          index='clients',
                                          columns='aggregation',
                                          aggfunc='first')

        pivot2.plot(kind='line', ax=axes[1], marker='o', markersize=8, linewidth=2)
        axes[1].set_xlabel('Number of Clients', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Final Test Accuracy (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Scalability: Final Accuracy vs Client Count',
                         fontsize=14, fontweight='bold')
        axes[1].legend(title='Aggregation', fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'comparison_client_scaling.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: comparison_client_scaling.png")
        plt.close()

    # Plot 4: Grand Comparison Heatmap
    print("\nPLOT 4: Grand Comparison Heatmap")
    print("-" * 70)

    # Create summary table
    summary_data = []
    for _, row in df.iterrows():
        partition_label = row['partition']
        if row['partition'] == 'Non-IID' and row['alpha']:
            partition_label = f"Non-IID(α={row['alpha']})"

        summary_data.append({
            'Partition': partition_label,
            'Aggregation': row['aggregation'],
            'Clients': row['clients'],
            'Final Accuracy': row['final_accuracy']
        })

    summary_df = pd.DataFrame(summary_data)

    # Create pivot for main conditions
    pivot_main = summary_df[summary_df['Clients'] == 50].pivot_table(
        values='Final Accuracy',
        index='Partition',
        columns='Aggregation',
        aggfunc='first'
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_main, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=70, vmax=80, ax=ax, cbar_kws={'label': 'Test Accuracy (%)'})
    ax.set_title('Final Test Accuracy: All Conditions (50 clients)',
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Data Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Aggregation Method', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_grand_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: heatmap_grand_comparison.png")
    plt.close()

    # Create CSV summary
    summary_df.to_csv(output_dir / 'comprehensive_summary.csv', index=False)
    print(f"\n✓ Saved: comprehensive_summary.csv")

    return summary_df


def print_hypothesis_analysis(summary_df):
    """Print analysis of experimental hypotheses"""
    print("\n" + "=" * 70)
    print("HYPOTHESIS ANALYSIS")
    print("=" * 70)

    # H1: FedAvg vs FedMean with Unequal Sizes
    print("\nH1: FedAvg Weighting Advantage (IID-Unequal)")
    print("-" * 70)
    unequal = summary_df[(summary_df['Partition'] == 'IID-Unequal') &
                         (summary_df['Clients'] == 50)]

    if len(unequal) > 0:
        fedavg_acc = unequal[unequal['Aggregation'] == 'Fedavg']['Final Accuracy'].values
        fedmean_acc = unequal[unequal['Aggregation'] == 'Fedmean']['Final Accuracy'].values

        if len(fedavg_acc) > 0 and len(fedmean_acc) > 0:
            diff = fedavg_acc[0] - fedmean_acc[0]
            print(f"  FedAvg:  {fedavg_acc[0]:.2f}%")
            print(f"  FedMean: {fedmean_acc[0]:.2f}%")
            print(f"  Difference: {diff:+.2f}%")

            if diff > 0.5:
                print(f"  ✅ H1 CONFIRMED: FedAvg outperforms FedMean with unequal sizes")
            elif diff < -0.5:
                print(f"  ❌ H1 REJECTED: FedMean outperforms FedAvg (unexpected)")
            else:
                print(f"  ⚠️  H1 INCONCLUSIVE: Difference too small ({diff:.2f}%)")

    # H2: Non-IID Robustness
    print("\nH2: Robustness to Non-IID Data")
    print("-" * 70)
    iid_equal = summary_df[(summary_df['Partition'] == 'IID-Equal') &
                           (summary_df['Clients'] == 50)]
    noniid = summary_df[(summary_df['Partition'].str.contains('Non-IID')) &
                        (summary_df['Clients'] == 50)]

    for agg in ['Fedavg', 'Fedmedian']:
        if len(iid_equal) > 0 and len(noniid) > 0:
            iid_acc = iid_equal[iid_equal['Aggregation'] == agg]['Final Accuracy'].values
            noniid_acc = noniid[noniid['Aggregation'] == agg]['Final Accuracy'].mean()

            if len(iid_acc) > 0:
                degradation = iid_acc[0] - noniid_acc
                print(f"\n  {agg}:")
                print(f"    IID:     {iid_acc[0]:.2f}%")
                print(f"    Non-IID: {noniid_acc:.2f}% (avg)")
                print(f"    Degradation: {degradation:.2f}%")

    # H3: Client Scaling
    print("\nH3: Scalability Analysis")
    print("-" * 70)
    scaling = summary_df[summary_df['Partition'] == 'IID-Equal']

    for agg in ['Fedavg', 'Fedmean', 'Fedmedian']:
        subset = scaling[scaling['Aggregation'] == agg].sort_values('Clients')
        if len(subset) > 1:
            print(f"\n  {agg}:")
            for _, row in subset.iterrows():
                print(f"    {row['Clients']:3d} clients: {row['Final Accuracy']:.2f}%")


def main():
    print("=" * 70)
    print("COMPREHENSIVE EXPERIMENTAL ANALYSIS")
    print("=" * 70)

    # Load all experiment data
    experiments = load_experiment_data()

    if len(experiments) == 0:
        print("\n⚠️  No experiment results found in ./results/comprehensive/")
        print("Please run experiments first using: bash run_comprehensive_experiments.sh")
        return

    # Create comparison plots
    summary_df = create_comparison_plots(experiments)

    # Print hypothesis analysis
    print_hypothesis_analysis(summary_df)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - comparison_iid_equal_vs_unequal.png")
    print("  - comparison_noniid_alpha.png")
    print("  - comparison_client_scaling.png")
    print("  - heatmap_grand_comparison.png")
    print("  - comprehensive_summary.csv")
    print("\nThese results can be used to update the research paper.")


if __name__ == "__main__":
    main()
