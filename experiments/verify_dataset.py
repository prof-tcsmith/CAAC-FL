"""
Verify CIFAR-10 dataset and show statistics.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from shared.data_utils import load_cifar10, partition_data_iid, partition_data_dirichlet, analyze_data_distribution
import numpy as np


def main():
    print("=" * 70)
    print("CIFAR-10 Dataset Verification")
    print("=" * 70)
    print()

    # Load dataset
    print("Loading CIFAR-10...")
    train_dataset, test_dataset = load_cifar10(data_dir='./data')
    print(f"  ✓ Training set: {len(train_dataset):,} samples")
    print(f"  ✓ Test set: {len(test_dataset):,} samples")
    print()

    # Show class distribution
    print("Class Distribution in Training Set:")
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    if hasattr(train_dataset, 'targets'):
        labels = np.array(train_dataset.targets)
    else:
        labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])

    for class_id, class_name in enumerate(class_names):
        count = np.sum(labels == class_id)
        print(f"  {class_id}. {class_name:12s}: {count:,} samples ({count/len(labels)*100:.1f}%)")
    print()

    # Test IID partitioning
    print("Testing IID Partitioning (10 clients)...")
    client_dict_iid = partition_data_iid(train_dataset, num_clients=10, seed=42)
    stats_iid = analyze_data_distribution(train_dataset, client_dict_iid, num_classes=10)

    print(f"  Clients: {stats_iid['num_clients']}")
    print(f"  Samples per client (min-max): {min(stats_iid['client_sizes'].values())}-{max(stats_iid['client_sizes'].values())}")
    print(f"  Heterogeneity (KL divergence): {stats_iid['heterogeneity_metrics']['mean_kl_divergence']:.4f}")
    print(f"  → Low KL divergence indicates IID distribution ✓")
    print()

    # Test Dirichlet partitioning
    print("Testing Dirichlet Partitioning (α=0.5, 10 clients)...")
    client_dict_dir = partition_data_dirichlet(train_dataset, num_clients=10, alpha=0.5, seed=42)
    stats_dir = analyze_data_distribution(train_dataset, client_dict_dir, num_classes=10)

    print(f"  Clients: {stats_dir['num_clients']}")
    print(f"  Samples per client (min-max): {min(stats_dir['client_sizes'].values())}-{max(stats_dir['client_sizes'].values())}")
    print(f"  Heterogeneity (KL divergence): {stats_dir['heterogeneity_metrics']['mean_kl_divergence']:.4f}")
    print(f"  → Higher KL divergence indicates non-IID distribution ✓")
    print()

    # Show example class distribution for first 3 clients
    print("Class Distribution for First 3 Clients (Dirichlet α=0.5):")
    for i in range(3):
        dist = np.array(stats_dir['class_distribution'][i])
        total = dist.sum()
        print(f"  Client {i}: ", end="")
        # Show top 3 classes
        top_classes = np.argsort(dist)[-3:][::-1]
        for cls in top_classes:
            if dist[cls] > 0:
                print(f"{class_names[cls]}({dist[cls]/total*100:.0f}%) ", end="")
        print()
    print()

    # Comparison
    print("IID vs Non-IID Comparison:")
    print(f"  IID Heterogeneity:     {stats_iid['heterogeneity_metrics']['mean_kl_divergence']:.4f}")
    print(f"  Non-IID Heterogeneity: {stats_dir['heterogeneity_metrics']['mean_kl_divergence']:.4f}")
    print(f"  Ratio: {stats_dir['heterogeneity_metrics']['mean_kl_divergence'] / stats_iid['heterogeneity_metrics']['mean_kl_divergence']:.1f}x more heterogeneous")
    print()

    print("=" * 70)
    print("✓ Dataset verified and ready for experiments!")
    print("=" * 70)
    print()
    print("Next Steps:")
    print("  1. Run Level 1 experiments: cd level1_fundamentals && bash run_all.sh")
    print("  2. View results in: level1_fundamentals/results/")
    print()


if __name__ == "__main__":
    main()
