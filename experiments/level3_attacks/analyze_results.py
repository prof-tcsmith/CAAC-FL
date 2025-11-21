"""
Level 3 Results Analysis: Byzantine Attack Impact

Analyzes and visualizes the impact of Byzantine attacks on different
aggregation methods. Compares performance across:
- 4 aggregation methods: FedAvg, FedMedian, Krum, Trimmed Mean
- 3 attack scenarios: No Attack, Random Noise, Sign Flipping

Generates comprehensive comparison plots and statistical summaries.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10


def load_experiment_results(results_dir='./results'):
    """
    Load all experiment results

    Returns:
        Dictionary with structure: {method: {attack: metrics}}
    """
    results_dir = Path(results_dir)

    methods = ['fedavg', 'fedmedian', 'krum', 'trimmed_mean']
    attacks = ['no_attack', 'random_noise', 'sign_flipping']

    all_results = {}

    for method in methods:
        all_results[method] = {}

        for attack in attacks:
            filename = f'level3_{method}_{attack}_metrics.json'
            filepath = results_dir / filename

            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Extract metrics
                rounds = data.get('rounds', [])
                test_accuracy = data.get('test_accuracy', [])
                test_loss = data.get('test_loss', [])

                # Filter to ensure consistent lengths
                min_len = min(len(rounds), len(test_accuracy), len(test_loss))
                if min_len > 0:
                    all_results[method][attack] = {
                        'rounds': rounds[:min_len],
                        'test_accuracy': test_accuracy[:min_len],
                        'test_loss': test_loss[:min_len],
                    }
                else:
                    print(f"Warning: No data for {method} with {attack}")
                    all_results[method][attack] = None
            else:
                print(f"Warning: File not found: {filepath}")
                all_results[method][attack] = None

    return all_results


def plot_attack_impact(all_results, save_path='./results/level3_attack_impact.png'):
    """
    Create comprehensive attack impact visualization

    Creates a 3x4 grid showing:
    - Row 1: Test accuracy over rounds for each method
    - Row 2: Test loss over rounds for each method
    - Row 3: Attack impact summary (bar charts)
    """
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    methods = ['fedavg', 'fedmedian', 'krum', 'trimmed_mean']
    method_names = ['FedAvg', 'FedMedian', 'Krum', 'Trimmed Mean']
    attacks = ['no_attack', 'random_noise', 'sign_flipping']
    attack_names = ['No Attack', 'Random Noise', 'Sign Flipping']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    # Row 1: Test Accuracy Curves
    for col, (method, method_name) in enumerate(zip(methods, method_names)):
        ax = fig.add_subplot(gs[0, col])

        for attack, attack_name, color in zip(attacks, attack_names, colors):
            data = all_results.get(method, {}).get(attack)
            if data:
                rounds = data['rounds']
                accuracy = data['test_accuracy']
                ax.plot(rounds, accuracy, label=attack_name, color=color,
                       linewidth=2, marker='o', markersize=3, markevery=5)

        ax.set_title(f'{method_name}: Test Accuracy', fontsize=12, fontweight='bold')
        ax.set_xlabel('Round')
        ax.set_ylabel('Test Accuracy (%)')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])

    # Row 2: Test Loss Curves
    for col, (method, method_name) in enumerate(zip(methods, method_names)):
        ax = fig.add_subplot(gs[1, col])

        for attack, attack_name, color in zip(attacks, attack_names, colors):
            data = all_results.get(method, {}).get(attack)
            if data:
                rounds = data['rounds']
                loss = data['test_loss']
                ax.plot(rounds, loss, label=attack_name, color=color,
                       linewidth=2, marker='s', markersize=3, markevery=5)

        ax.set_title(f'{method_name}: Test Loss', fontsize=12, fontweight='bold')
        ax.set_xlabel('Round')
        ax.set_ylabel('Test Loss')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    # Row 3: Attack Impact Summary
    # Collect final accuracies
    final_accuracies = {}
    for method in methods:
        final_accuracies[method] = {}
        for attack in attacks:
            data = all_results.get(method, {}).get(attack)
            if data and len(data['test_accuracy']) > 0:
                final_accuracies[method][attack] = data['test_accuracy'][-1]
            else:
                final_accuracies[method][attack] = 0

    # Plot 1: Accuracy by Attack Type
    ax = fig.add_subplot(gs[2, 0])
    x = np.arange(len(methods))
    width = 0.25

    for i, (attack, attack_name, color) in enumerate(zip(attacks, attack_names, colors)):
        values = [final_accuracies[method][attack] for method in methods]
        ax.bar(x + i*width, values, width, label=attack_name, color=color, alpha=0.8)

    ax.set_ylabel('Final Test Accuracy (%)')
    ax.set_title('Final Accuracy: Attack Comparison', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(method_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])

    # Plot 2: Degradation from Baseline
    ax = fig.add_subplot(gs[2, 1])

    degradation_data = []
    for method in methods:
        baseline = final_accuracies[method]['no_attack']
        for attack in ['random_noise', 'sign_flipping']:
            degradation = baseline - final_accuracies[method][attack]
            degradation_data.append({
                'method': method,
                'attack': attack,
                'degradation': degradation
            })

    df_deg = pd.DataFrame(degradation_data)

    for i, attack in enumerate(['random_noise', 'sign_flipping']):
        values = [df_deg[(df_deg['method'] == m) & (df_deg['attack'] == attack)]['degradation'].values[0]
                 for m in methods]
        ax.bar(x + i*0.35, values, 0.35,
              label=attack_names[i+1], color=colors[i+1], alpha=0.8)

    ax.set_ylabel('Accuracy Degradation (%)')
    ax.set_title('Attack Impact (Degradation)', fontweight='bold')
    ax.set_xticks(x + 0.175)
    ax.set_xticklabels(method_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Plot 3: Method Ranking
    ax = fig.add_subplot(gs[2, 2])

    # Calculate average performance across attacks
    avg_performance = []
    for method, method_name in zip(methods, method_names):
        avg = np.mean([final_accuracies[method][attack] for attack in attacks])
        avg_performance.append((method_name, avg))

    avg_performance.sort(key=lambda x: x[1], reverse=True)
    names, values = zip(*avg_performance)

    colors_rank = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    ax.barh(names, values, color=colors_rank, alpha=0.8)
    ax.set_xlabel('Average Accuracy (%)')
    ax.set_title('Average Performance Ranking', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim([0, 100])

    # Plot 4: Robustness Score
    ax = fig.add_subplot(gs[2, 3])

    # Robustness = (Accuracy under worst attack) / (Baseline accuracy)
    robustness_scores = []
    for method, method_name in zip(methods, method_names):
        baseline = final_accuracies[method]['no_attack']
        worst_attack = min(final_accuracies[method]['random_noise'],
                          final_accuracies[method]['sign_flipping'])
        robustness = (worst_attack / baseline * 100) if baseline > 0 else 0
        robustness_scores.append((method_name, robustness))

    robustness_scores.sort(key=lambda x: x[1], reverse=True)
    names, values = zip(*robustness_scores)

    ax.barh(names, values, color='#9b59b6', alpha=0.8)
    ax.set_xlabel('Robustness Score (%)')
    ax.set_title('Robustness Score\n(Worst Attack / Baseline)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim([0, 100])

    # Add main title
    fig.suptitle('Level 3: Byzantine Attack Impact Analysis\n'
                'Non-IID Data (Dirichlet α=0.5) | 15 Clients | 20% Byzantine',
                fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nAttack impact plot saved to: {save_path}")
    plt.close()


def print_summary(all_results):
    """Print detailed summary statistics"""
    print("\n" + "=" * 80)
    print("LEVEL 3: BYZANTINE ATTACK IMPACT ANALYSIS")
    print("=" * 80)

    methods = ['fedavg', 'fedmedian', 'krum', 'trimmed_mean']
    method_names = ['FedAvg', 'FedMedian', 'Krum', 'Trimmed Mean']
    attacks = ['no_attack', 'random_noise', 'sign_flipping']
    attack_names = ['No Attack', 'Random Noise', 'Sign Flipping']

    print("\nFinal Test Accuracies (%):")
    print("-" * 80)

    # Create table
    for method, method_name in zip(methods, method_names):
        print(f"\n{method_name}:")
        for attack, attack_name in zip(attacks, attack_names):
            data = all_results.get(method, {}).get(attack)
            if data and len(data['test_accuracy']) > 0:
                final_acc = data['test_accuracy'][-1]
                print(f"  {attack_name:20s}: {final_acc:6.2f}%")
            else:
                print(f"  {attack_name:20s}: No data")

    print("\n" + "-" * 80)
    print("Attack Impact (Degradation from Baseline):")
    print("-" * 80)

    for method, method_name in zip(methods, method_names):
        baseline_data = all_results.get(method, {}).get('no_attack')
        if baseline_data and len(baseline_data['test_accuracy']) > 0:
            baseline = baseline_data['test_accuracy'][-1]
            print(f"\n{method_name} (Baseline: {baseline:.2f}%):")

            for attack, attack_name in zip(['random_noise', 'sign_flipping'],
                                          ['Random Noise', 'Sign Flipping']):
                data = all_results.get(method, {}).get(attack)
                if data and len(data['test_accuracy']) > 0:
                    attacked = data['test_accuracy'][-1]
                    degradation = baseline - attacked
                    pct_degradation = (degradation / baseline * 100) if baseline > 0 else 0
                    print(f"  {attack_name:20s}: -{degradation:5.2f}% ({pct_degradation:5.1f}% loss)")

    print("\n" + "-" * 80)
    print("Method Rankings:")
    print("-" * 80)

    # Rank by average performance
    avg_performance = []
    for method, method_name in zip(methods, method_names):
        accuracies = []
        for attack in attacks:
            data = all_results.get(method, {}).get(attack)
            if data and len(data['test_accuracy']) > 0:
                accuracies.append(data['test_accuracy'][-1])
        if accuracies:
            avg = np.mean(accuracies)
            avg_performance.append((method_name, avg))

    avg_performance.sort(key=lambda x: x[1], reverse=True)
    print("\nBy Average Accuracy:")
    for rank, (method_name, avg) in enumerate(avg_performance, 1):
        print(f"  {rank}. {method_name:15s}: {avg:6.2f}%")

    # Rank by robustness
    robustness_scores = []
    for method, method_name in zip(methods, method_names):
        baseline_data = all_results.get(method, {}).get('no_attack')
        if baseline_data and len(baseline_data['test_accuracy']) > 0:
            baseline = baseline_data['test_accuracy'][-1]

            worst_acc = baseline
            for attack in ['random_noise', 'sign_flipping']:
                data = all_results.get(method, {}).get(attack)
                if data and len(data['test_accuracy']) > 0:
                    worst_acc = min(worst_acc, data['test_accuracy'][-1])

            robustness = (worst_acc / baseline * 100) if baseline > 0 else 0
            robustness_scores.append((method_name, robustness, worst_acc))

    robustness_scores.sort(key=lambda x: x[1], reverse=True)
    print("\nBy Robustness (Worst Attack / Baseline):")
    for rank, (method_name, robustness, worst_acc) in enumerate(robustness_scores, 1):
        print(f"  {rank}. {method_name:15s}: {robustness:5.1f}% (worst: {worst_acc:.2f}%)")

    print("\n" + "=" * 80)
    print("KEY OBSERVATIONS:")
    print("=" * 80)

    # Analyze results
    print("\n1. Most Robust Method:")
    if robustness_scores:
        best_method, best_robustness, best_worst = robustness_scores[0]
        print(f"   {best_method} maintained {best_robustness:.1f}% of baseline performance")
        print(f"   under worst attack (final accuracy: {best_worst:.2f}%)")

    print("\n2. Most Vulnerable Method:")
    if robustness_scores:
        worst_method, worst_robustness, worst_acc = robustness_scores[-1]
        print(f"   {worst_method} dropped to {worst_robustness:.1f}% of baseline performance")
        print(f"   under worst attack (final accuracy: {worst_acc:.2f}%)")

    print("\n3. Attack Severity Ranking:")
    # Calculate average impact of each attack across all methods
    attack_impacts = {}
    for attack, attack_name in zip(['random_noise', 'sign_flipping'],
                                   ['Random Noise', 'Sign Flipping']):
        degradations = []
        for method in methods:
            baseline_data = all_results.get(method, {}).get('no_attack')
            attack_data = all_results.get(method, {}).get(attack)
            if baseline_data and attack_data:
                if (len(baseline_data['test_accuracy']) > 0 and
                    len(attack_data['test_accuracy']) > 0):
                    baseline = baseline_data['test_accuracy'][-1]
                    attacked = attack_data['test_accuracy'][-1]
                    degradations.append(baseline - attacked)
        if degradations:
            attack_impacts[attack_name] = np.mean(degradations)

    sorted_attacks = sorted(attack_impacts.items(), key=lambda x: x[1], reverse=True)
    for rank, (attack_name, avg_deg) in enumerate(sorted_attacks, 1):
        print(f"   {rank}. {attack_name}: -{avg_deg:.2f}% average degradation")

    print("\n" + "=" * 80)


def save_summary_csv(all_results, save_path='./results/level3_summary.csv'):
    """Save summary results to CSV"""
    methods = ['fedavg', 'fedmedian', 'krum', 'trimmed_mean']
    method_names = ['FedAvg', 'FedMedian', 'Krum', 'TrimmedMean']
    attacks = ['no_attack', 'random_noise', 'sign_flipping']

    rows = []
    for method, method_name in zip(methods, method_names):
        row = {'method': method_name}
        for attack in attacks:
            data = all_results.get(method, {}).get(attack)
            if data and len(data['test_accuracy']) > 0:
                row[f'{attack}_accuracy'] = data['test_accuracy'][-1]
                row[f'{attack}_loss'] = data['test_loss'][-1]
            else:
                row[f'{attack}_accuracy'] = None
                row[f'{attack}_loss'] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"\nSummary CSV saved to: {save_path}")


def main():
    """Main analysis function"""
    print("=" * 80)
    print("Level 3: Byzantine Attack Impact Analysis")
    print("=" * 80)

    # Load results
    print("\nLoading experiment results...")
    all_results = load_experiment_results('./results')

    # Check which experiments are available
    print("\nAvailable experiments:")
    for method in all_results:
        for attack in all_results[method]:
            if all_results[method][attack]:
                print(f"  ✓ {method} with {attack}")
            else:
                print(f"  ✗ {method} with {attack} (missing)")

    # Generate plots
    print("\nGenerating attack impact visualization...")
    plot_attack_impact(all_results)

    # Print summary
    print_summary(all_results)

    # Save CSV
    save_summary_csv(all_results)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
