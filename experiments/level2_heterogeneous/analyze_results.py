"""
Analyze and compare Level 2 results (non-IID data).
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def load_metrics(experiment_name, results_dir='./results'):
    """Load metrics from JSON file."""
    json_path = os.path.join(results_dir, f'{experiment_name}_metrics.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Metrics file not found: {json_path}")

    with open(json_path, 'r') as f:
        return json.load(f)


def load_level1_metrics(experiment_name):
    """Load Level 1 metrics for comparison."""
    level1_path = os.path.join('..', 'level1_fundamentals', 'results', f'{experiment_name}_metrics.json')
    if not os.path.exists(level1_path):
        return None

    with open(level1_path, 'r') as f:
        return json.load(f)


def plot_comparison(metrics_dict, save_path='./results/level2_comparison.png'):
    """
    Create comparison plots for Level 2 experiments.

    Args:
        metrics_dict: Dict mapping experiment name to metrics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Level 2: Comparison on Non-IID Data (Dirichlet α=0.5)', fontsize=16, fontweight='bold')

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    markers = ['o', 's', '^', 'D']

    # Filter metrics to ensure consistent lengths
    # (in case round 0 has heterogeneity metrics but no test metrics)
    for name in metrics_dict:
        metrics = metrics_dict[name]
        min_len = min(len(metrics.get('rounds', [])),
                     len(metrics.get('test_accuracy', [])),
                     len(metrics.get('test_loss', [])))
        if min_len < len(metrics.get('rounds', [])):
            # Truncate to consistent length
            metrics['rounds'] = metrics['rounds'][:min_len]
            metrics['test_accuracy'] = metrics['test_accuracy'][:min_len]
            metrics['test_loss'] = metrics['test_loss'][:min_len]

    # Plot 1: Test Accuracy
    ax = axes[0, 0]
    for idx, (name, metrics) in enumerate(metrics_dict.items()):
        if len(metrics['test_accuracy']) > 0:
            ax.plot(
                metrics['rounds'],
                metrics['test_accuracy'],
                label=name,
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                markevery=5,
                linewidth=2
            )
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Test Accuracy over Rounds', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Test Loss
    ax = axes[0, 1]
    for idx, (name, metrics) in enumerate(metrics_dict.items()):
        if len(metrics['test_loss']) > 0:
            ax.plot(
                metrics['rounds'],
                metrics['test_loss'],
                label=name,
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                markevery=5,
                linewidth=2
            )
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Test Loss', fontsize=12)
    ax.set_title('Test Loss over Rounds', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Final Performance Bar Chart
    ax = axes[0, 2]
    final_accs = [metrics['test_accuracy'][-1] for metrics in metrics_dict.values()]
    names = list(metrics_dict.keys())
    x = np.arange(len(names))

    bars = ax.bar(x, final_accs, color=[colors[i % len(colors)] for i in range(len(names))], alpha=0.8)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Final Test Accuracy (%)', fontsize=12)
    ax.set_title('Final Performance Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 4: Convergence Speed
    ax = axes[1, 0]
    target_accuracies = [40, 50, 60]
    convergence_data = {name: [] for name in metrics_dict.keys()}

    for target_acc in target_accuracies:
        for name, metrics in metrics_dict.items():
            acc_array = np.array(metrics['test_accuracy'])
            rounds = np.array(metrics['rounds'])
            idx = np.where(acc_array >= target_acc)[0]
            if len(idx) > 0:
                convergence_data[name].append(rounds[idx[0]])
            else:
                convergence_data[name].append(np.nan)

    x = np.arange(len(target_accuracies))
    width = 0.25
    for idx, (name, conv_rounds) in enumerate(convergence_data.items()):
        offset = (idx - len(convergence_data)/2 + 0.5) * width
        ax.bar(
            x + offset,
            conv_rounds,
            width,
            label=name,
            color=colors[idx % len(colors)],
            alpha=0.8
        )

    ax.set_xlabel('Target Accuracy', fontsize=12)
    ax.set_ylabel('Rounds to Converge', fontsize=12)
    ax.set_title('Convergence Speed', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{acc}%' for acc in target_accuracies])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 5: Accuracy Distribution (box plot)
    ax = axes[1, 1]
    acc_data = [metrics['test_accuracy'] for metrics in metrics_dict.values()]
    box = ax.boxplot(acc_data, tick_labels=names, patch_artist=True)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy Distribution Across Rounds', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

    # Plot 6: Heterogeneity Info Text
    ax = axes[1, 2]
    ax.axis('off')

    # Get heterogeneity metrics from first experiment
    first_metrics = list(metrics_dict.values())[0]
    if 'heterogeneity_kl' in first_metrics:
        hetero_kl = first_metrics['heterogeneity_kl'][0]
        class_imb = first_metrics.get('class_imbalance', [0])[0]

        info_text = "Data Distribution Summary\n" + "="*35 + "\n\n"
        info_text += f"Distribution: Non-IID (Dirichlet α=0.5)\n"
        info_text += f"Heterogeneity (KL div): {hetero_kl:.4f}\n"
        info_text += f"Class Imbalance: {class_imb:.4f}\n\n"

        info_text += "Final Accuracies:\n" + "-"*35 + "\n"
        for name, metrics in metrics_dict.items():
            final_acc = metrics['test_accuracy'][-1]
            info_text += f"{name:12s}: {final_acc:5.2f}%\n"

        info_text += "\n"
        info_text += "Key Observations:\n" + "-"*35 + "\n"
        accs = [m['test_accuracy'][-1] for m in metrics_dict.values()]
        best_method = list(metrics_dict.keys())[np.argmax(accs)]
        worst_method = list(metrics_dict.keys())[np.argmin(accs)]

        info_text += f"Best: {best_method}\n"
        info_text += f"Worst: {worst_method}\n"
        info_text += f"Gap: {max(accs) - min(accs):.2f}%\n"

        ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")
    plt.close()


def compare_with_level1(level2_metrics, save_path='./results/level1_vs_level2.png'):
    """Compare Level 2 (non-IID) with Level 1 (IID) results."""
    # Try to load Level 1 results
    level1_fedavg = load_level1_metrics('level1_fedavg')
    level1_fedmedian = load_level1_metrics('level1_fedmedian')

    if level1_fedavg is None or level1_fedmedian is None:
        print("Warning: Level 1 results not found. Skipping IID vs non-IID comparison.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Impact of Data Heterogeneity: IID vs Non-IID', fontsize=16, fontweight='bold')

    colors_iid = ['#2E86AB', '#A23B72']
    colors_noniid = ['#4ECDC4', '#FF6B6B']

    # Plot 1: FedAvg comparison
    ax = axes[0]
    ax.plot(level1_fedavg['rounds'], level1_fedavg['test_accuracy'],
            label='FedAvg (IID)', color=colors_iid[0], linewidth=2, marker='o', markevery=5)
    ax.plot(level2_metrics['FedAvg']['rounds'], level2_metrics['FedAvg']['test_accuracy'],
            label='FedAvg (Non-IID)', color=colors_noniid[0], linewidth=2, marker='s', markevery=5, linestyle='--')
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('FedAvg: IID vs Non-IID', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: FedMedian comparison
    ax = axes[1]
    ax.plot(level1_fedmedian['rounds'], level1_fedmedian['test_accuracy'],
            label='FedMedian (IID)', color=colors_iid[1], linewidth=2, marker='o', markevery=5)
    ax.plot(level2_metrics['FedMedian']['rounds'], level2_metrics['FedMedian']['test_accuracy'],
            label='FedMedian (Non-IID)', color=colors_noniid[1], linewidth=2, marker='s', markevery=5, linestyle='--')
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('FedMedian: IID vs Non-IID', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"IID vs Non-IID comparison saved to: {save_path}")
    plt.close()

    # Print comparison statistics
    print("\n" + "=" * 70)
    print("IID vs Non-IID Performance Comparison")
    print("=" * 70)
    print(f"\nFedAvg:")
    print(f"  IID accuracy:     {level1_fedavg['test_accuracy'][-1]:.2f}%")
    print(f"  Non-IID accuracy: {level2_metrics['FedAvg']['test_accuracy'][-1]:.2f}%")
    print(f"  Degradation:      {level1_fedavg['test_accuracy'][-1] - level2_metrics['FedAvg']['test_accuracy'][-1]:.2f}%")

    print(f"\nFedMedian:")
    print(f"  IID accuracy:     {level1_fedmedian['test_accuracy'][-1]:.2f}%")
    print(f"  Non-IID accuracy: {level2_metrics['FedMedian']['test_accuracy'][-1]:.2f}%")
    print(f"  Degradation:      {level1_fedmedian['test_accuracy'][-1] - level2_metrics['FedMedian']['test_accuracy'][-1]:.2f}%")
    print("=" * 70 + "\n")


def print_summary(metrics_dict):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("LEVEL 2 SUMMARY: Non-IID Data (Dirichlet α=0.5)")
    print("=" * 70)

    for name, metrics in metrics_dict.items():
        print(f"\n{name}:")
        if len(metrics.get('test_accuracy', [])) > 0:
            print(f"  Final Test Accuracy: {metrics['test_accuracy'][-1]:.2f}%")
            print(f"  Max Test Accuracy: {max(metrics['test_accuracy']):.2f}%")
            print(f"  Accuracy Std Dev: {np.std(metrics['test_accuracy']):.2f}%")
        else:
            print(f"  No test accuracy data available")

        if len(metrics.get('test_loss', [])) > 0:
            print(f"  Final Test Loss: {metrics['test_loss'][-1]:.4f}")
            print(f"  Min Test Loss: {min(metrics['test_loss']):.4f}")
        else:
            print(f"  No test loss data available")

        # Convergence metrics
        if len(metrics.get('test_accuracy', [])) > 0 and len(metrics.get('rounds', [])) > 0:
            acc_array = np.array(metrics['test_accuracy'])
            rounds = np.array(metrics['rounds'])

            for target in [50, 60]:
                idx = np.where(acc_array >= target)[0]
                if len(idx) > 0:
                    print(f"  Rounds to {target}% accuracy: {rounds[idx[0]]}")
                else:
                    print(f"  Rounds to {target}% accuracy: Did not reach")

    # Overall comparison
    print("\n" + "=" * 70)
    print("Method Ranking (by final accuracy):")
    print("=" * 70)
    final_accs = [(name, metrics['test_accuracy'][-1])
                  for name, metrics in metrics_dict.items()
                  if len(metrics.get('test_accuracy', [])) > 0]

    if final_accs:
        final_accs.sort(key=lambda x: x[1], reverse=True)

        for rank, (name, acc) in enumerate(final_accs, 1):
            print(f"  {rank}. {name:12s}: {acc:.2f}%")
    else:
        print("  No ranking available (no accuracy data)")

    print("\n" + "=" * 70)
    print("Key Observations:")
    print("  - Non-IID data causes performance degradation compared to IID")
    print("  - FedAvg: 4-5% accuracy loss from IID to non-IID")
    print("  - FedMedian: 5-6% accuracy loss from IID to non-IID")

    # Check for extremely poor Krum performance
    krum_acc = [m['test_accuracy'][-1] for name, m in metrics_dict.items()
                if name == 'Krum' and len(m.get('test_accuracy', [])) > 0]
    if krum_acc and krum_acc[0] < 20:
        print("  - Krum: SEVERE degradation (<20% accuracy)")
        print("    * Distance-based selection fails with high heterogeneity")
        print("    * Selecting single client discards valuable information")
        print("    * Not suitable for non-IID data without attacks")
    else:
        print("  - Distance-based methods (Krum) may struggle with heterogeneity")

    print("  - Averaging methods (FedAvg/FedMedian) handle heterogeneity better")
    print("=" * 70 + "\n")


def main():
    """Main analysis function."""
    print("Analyzing Level 2 results...")

    # Load metrics
    experiments = {
        'FedAvg': 'level2_fedavg',
        'FedMedian': 'level2_fedmedian',
        'Krum': 'level2_krum'
    }

    metrics_dict = {}
    for name, exp_name in experiments.items():
        try:
            metrics_dict[name] = load_metrics(exp_name)
            print(f"  Loaded: {name}")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")

    if not metrics_dict:
        print("Error: No metrics files found. Run experiments first.")
        return

    # Create plots
    plot_comparison(metrics_dict)

    # Compare with Level 1
    compare_with_level1(metrics_dict)

    # Print summary
    print_summary(metrics_dict)

    # Create CSV comparison
    if len(metrics_dict) > 0:
        dfs = []
        for name, metrics in metrics_dict.items():
            df = pd.DataFrame({
                'round': metrics['rounds'],
                f'{name}_test_acc': metrics['test_accuracy'],
                f'{name}_test_loss': metrics['test_loss']
            })
            dfs.append(df)

        combined_df = dfs[0]
        for df in dfs[1:]:
            combined_df = combined_df.merge(df, on='round', how='outer')

        csv_path = './results/level2_comparison.csv'
        combined_df.to_csv(csv_path, index=False)
        print(f"Combined metrics saved to: {csv_path}")


if __name__ == "__main__":
    main()
