#!/usr/bin/env python3
"""
Generate comparison visualization for Standard Krum vs Multi-Krum
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Results data
results = {
    'Standard Krum': {
        0.1: {'accuracy': 10.07, 'kl': 1.3950},
        0.5: {'accuracy': 31.93, 'kl': 0.5898},
        1.0: {'accuracy': 42.14, 'kl': 0.3330},
    },
    'Multi-Krum': {
        0.1: {'accuracy': 60.69, 'kl': 1.3950},
        0.5: {'accuracy': 67.12, 'kl': 0.5898},
        1.0: {'accuracy': 69.56, 'kl': 0.3330},
    },
    'FedAvg': {
        0.1: {'accuracy': 66.77, 'kl': 1.3950},
        0.5: {'accuracy': 69.62, 'kl': 0.5898},
        1.0: {'accuracy': 70.49, 'kl': 0.3330},
    },
    'FedMedian': {
        0.1: {'accuracy': 43.72, 'kl': 1.3950},
        0.5: {'accuracy': 63.79, 'kl': 0.5898},
        1.0: {'accuracy': 67.19, 'kl': 0.3330},
    },
}

alphas = [0.1, 0.5, 1.0]
alpha_labels = ['α=0.1\n(extreme)', 'α=0.5\n(moderate)', 'α=1.0\n(mild)']

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Color scheme
colors = {
    'Standard Krum': '#e74c3c',  # Red
    'Multi-Krum': '#2ecc71',      # Green
    'FedAvg': '#3498db',          # Blue
    'FedMedian': '#f39c12',       # Orange
}

# Plot 1: Bar chart comparison
ax1 = axes[0]
x = np.arange(len(alphas))
width = 0.2
multiplier = 0

for method, data in results.items():
    accuracies = [data[alpha]['accuracy'] for alpha in alphas]
    offset = width * multiplier
    bars = ax1.bar(x + offset, accuracies, width, label=method, color=colors[method])
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    multiplier += 1

ax1.set_xlabel('Dirichlet α (Heterogeneity Level)', fontsize=12)
ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
ax1.set_title('Standard Krum vs Multi-Krum on Non-IID Data', fontsize=14, fontweight='bold')
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(alpha_labels)
ax1.legend(loc='upper left')
ax1.set_ylim(0, 85)
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Random chance')

# Plot 2: Improvement visualization
ax2 = axes[1]
std_acc = [results['Standard Krum'][a]['accuracy'] for a in alphas]
multi_acc = [results['Multi-Krum'][a]['accuracy'] for a in alphas]
improvements = [m - s for m, s in zip(multi_acc, std_acc)]

bars = ax2.bar(alpha_labels, improvements, color='#27ae60', edgecolor='black', linewidth=1.5)
for bar, imp, std, multi in zip(bars, improvements, std_acc, multi_acc):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'+{imp:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
            f'{std:.1f}% → {multi:.1f}%', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

ax2.set_xlabel('Dirichlet α (Heterogeneity Level)', fontsize=12)
ax2.set_ylabel('Accuracy Improvement (%)', fontsize=12)
ax2.set_title('Multi-Krum Improvement over Standard Krum', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 60)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('krum_standard_vs_multi_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("Generated: krum_standard_vs_multi_comparison.png")

# Also create a convergence comparison plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Load convergence data
convergence_data = {}
for variant in ['std', 'multi']:
    convergence_data[variant] = {}
    for alpha in [0.1, 0.5, 1.0]:
        if variant == 'std':
            filepath = f'../../level2_heterogeneous/results/krum_standard/level2_noniid_krum_std_a{alpha}_c50_metrics.json'
        else:
            filepath = f'../../level2_heterogeneous/results/multikrum/level2_noniid_krum_a{alpha}_c50_metrics.json'
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                convergence_data[variant][alpha] = data['test_accuracy']
        except FileNotFoundError:
            print(f"Warning: {filepath} not found")
            convergence_data[variant][alpha] = None

for i, alpha in enumerate([0.1, 0.5, 1.0]):
    ax = axes[i]

    if convergence_data['std'][alpha]:
        rounds = list(range(len(convergence_data['std'][alpha])))
        ax.plot(rounds, convergence_data['std'][alpha], 'r-o', label='Standard Krum (m=1)',
                markersize=4, linewidth=2)

    if convergence_data['multi'][alpha]:
        rounds = list(range(len(convergence_data['multi'][alpha])))
        ax.plot(rounds, convergence_data['multi'][alpha], 'g-s', label='Multi-Krum (m=48)',
                markersize=4, linewidth=2)

    ax.set_xlabel('Communication Round', fontsize=11)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title(f'α={alpha} ({"Extreme" if alpha==0.1 else "Moderate" if alpha==0.5 else "Mild"} Heterogeneity)',
                fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 80)
    ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5)

plt.suptitle('Convergence Comparison: Standard Krum vs Multi-Krum', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('krum_convergence_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("Generated: krum_convergence_comparison.png")
print("\nDone! Generated comparison figures.")
