"""
Analyze and compare Level 1 results.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def load_metrics(experiment_name):
    """Load metrics from JSON file."""
    json_path = f'./results/{experiment_name}_metrics.json'
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Metrics file not found: {json_path}")

    with open(json_path, 'r') as f:
        return json.load(f)


def plot_comparison(metrics_dict, save_path='./results/level1_comparison.png'):
    """
    Create comparison plots for multiple experiments.

    Args:
        metrics_dict: Dict mapping experiment name to metrics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Level 1: FedAvg vs FedMedian Comparison (IID Data)', fontsize=16)

    colors = ['#2E86AB', '#A23B72', '#F18F01']
    markers = ['o', 's', '^']

    # Plot 1: Test Accuracy
    ax = axes[0, 0]
    for idx, (name, metrics) in enumerate(metrics_dict.items()):
        ax.plot(
            metrics['rounds'],
            metrics['test_accuracy'],
            label=name,
            color=colors[idx],
            marker=markers[idx],
            markevery=5,
            linewidth=2
        )
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Test Accuracy over Rounds', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Test Loss
    ax = axes[0, 1]
    for idx, (name, metrics) in enumerate(metrics_dict.items()):
        ax.plot(
            metrics['rounds'],
            metrics['test_loss'],
            label=name,
            color=colors[idx],
            marker=markers[idx],
            markevery=5,
            linewidth=2
        )
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Test Loss', fontsize=12)
    ax.set_title('Test Loss over Rounds', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Convergence Speed
    ax = axes[1, 0]
    target_accuracies = [50, 60, 70]
    convergence_data = {name: [] for name in metrics_dict.keys()}

    for target_acc in target_accuracies:
        for name, metrics in metrics_dict.items():
            acc_array = np.array(metrics['test_accuracy'])
            rounds = np.array(metrics['rounds'])
            # Find first round where accuracy exceeds target
            idx = np.where(acc_array >= target_acc)[0]
            if len(idx) > 0:
                convergence_data[name].append(rounds[idx[0]])
            else:
                convergence_data[name].append(np.nan)

    x = np.arange(len(target_accuracies))
    width = 0.35
    for idx, (name, conv_rounds) in enumerate(convergence_data.items()):
        ax.bar(
            x + idx * width,
            conv_rounds,
            width,
            label=name,
            color=colors[idx],
            alpha=0.8
        )

    ax.set_xlabel('Target Accuracy', fontsize=12)
    ax.set_ylabel('Rounds to Converge', fontsize=12)
    ax.set_title('Convergence Speed Comparison', fontsize=13)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f'{acc}%' for acc in target_accuracies])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Final Performance Bar Chart
    ax = axes[1, 1]
    final_accs = [metrics['test_accuracy'][-1] for metrics in metrics_dict.values()]
    final_losses = [metrics['test_loss'][-1] for metrics in metrics_dict.values()]

    x = np.arange(len(metrics_dict))
    width = 0.35

    ax2 = ax.twinx()
    bars1 = ax.bar(x - width/2, final_accs, width, label='Test Accuracy', color=colors[0], alpha=0.8)
    bars2 = ax2.bar(x + width/2, final_losses, width, label='Test Loss', color=colors[1], alpha=0.8)

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Final Test Accuracy (%)', fontsize=12, color=colors[0])
    ax2.set_ylabel('Final Test Loss', fontsize=12, color=colors[1])
    ax.set_title('Final Performance Comparison', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics_dict.keys()), rotation=15, ha='right')
    ax.tick_params(axis='y', labelcolor=colors[0])
    ax2.tick_params(axis='y', labelcolor=colors[1])

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")
    plt.close()


def print_summary(metrics_dict):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("LEVEL 1 SUMMARY: FedAvg vs FedMedian (IID Data)")
    print("=" * 70)

    for name, metrics in metrics_dict.items():
        print(f"\n{name}:")
        print(f"  Final Test Accuracy: {metrics['test_accuracy'][-1]:.2f}%")
        print(f"  Final Test Loss: {metrics['test_loss'][-1]:.4f}")
        print(f"  Max Test Accuracy: {max(metrics['test_accuracy']):.2f}%")
        print(f"  Min Test Loss: {min(metrics['test_loss']):.4f}")

        # Convergence to 70%
        acc_array = np.array(metrics['test_accuracy'])
        rounds = np.array(metrics['rounds'])
        idx_70 = np.where(acc_array >= 70.0)[0]
        if len(idx_70) > 0:
            print(f"  Rounds to 70% accuracy: {rounds[idx_70[0]]}")
        else:
            print(f"  Rounds to 70% accuracy: Did not reach")

    # Comparison
    if len(metrics_dict) == 2:
        names = list(metrics_dict.keys())
        acc_diff = metrics_dict[names[0]]['test_accuracy'][-1] - metrics_dict[names[1]]['test_accuracy'][-1]
        print(f"\nAccuracy Difference ({names[0]} - {names[1]}): {acc_diff:+.2f}%")

        if abs(acc_diff) < 1.0:
            print("  → Similar performance (as expected with IID data, no attacks)")
        elif acc_diff > 0:
            print(f"  → {names[0]} performed slightly better")
        else:
            print(f"  → {names[1]} performed slightly better")

    print("\n" + "=" * 70)
    print("\nKey Observations:")
    print("  - Both methods should achieve similar accuracy (~75-80%) with IID data")
    print("  - This validates the experimental setup before introducing attacks")
    print("  - Slight differences may be due to randomness in training")
    print("=" * 70 + "\n")


def main():
    """Main analysis function."""
    print("Analyzing Level 1 results...")

    # Load metrics
    experiments = {
        'FedAvg': 'level1_fedavg',
        'FedMedian': 'level1_fedmedian'
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

    # Print summary
    print_summary(metrics_dict)

    # Create CSV comparison
    if len(metrics_dict) > 0:
        # Combine all metrics into a single DataFrame
        dfs = []
        for name, metrics in metrics_dict.items():
            df = pd.DataFrame({
                'round': metrics['rounds'],
                f'{name}_test_acc': metrics['test_accuracy'],
                f'{name}_test_loss': metrics['test_loss']
            })
            dfs.append(df)

        # Merge on round number
        combined_df = dfs[0]
        for df in dfs[1:]:
            combined_df = combined_df.merge(df, on='round', how='outer')

        csv_path = './results/level1_comparison.csv'
        combined_df.to_csv(csv_path, index=False)
        print(f"Combined metrics saved to: {csv_path}")


if __name__ == "__main__":
    main()
