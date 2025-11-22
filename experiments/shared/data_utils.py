"""
Data loading and partitioning utilities.
"""

# Suppress PyTorch pin_memory deprecation warnings (from PyTorch internals)
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='torch.utils.data')

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Dict
from collections import defaultdict


def load_cifar10(data_dir='./data'):
    """
    Load CIFAR-10 dataset.

    Args:
        data_dir: Directory to store/load data

    Returns:
        train_dataset, test_dataset
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    return train_dataset, test_dataset


def partition_data_iid(dataset, num_clients, seed=42):
    """
    Partition dataset into IID subsets for each client.

    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        seed: Random seed for reproducibility

    Returns:
        dict: {client_id: list of indices}
    """
    np.random.seed(seed)

    num_items = len(dataset)
    indices = np.random.permutation(num_items)
    shard_size = num_items // num_clients

    client_dict = {}
    for i in range(num_clients):
        start = i * shard_size
        end = start + shard_size if i < num_clients - 1 else num_items
        client_dict[i] = indices[start:end].tolist()

    return client_dict


def partition_data_dirichlet(dataset, num_clients, alpha=0.5, seed=42):
    """
    Partition dataset into non-IID subsets using Dirichlet distribution.

    Based on the method from:
    - Hsu et al., "Measuring the Effects of Non-Identical Data Distribution"
    - Li et al., "An Experimental Study of Byzantine-Robust Aggregation Schemes"

    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter
               - Lower alpha = more heterogeneous
               - alpha=0.1: highly non-IID
               - alpha=0.5: moderately non-IID
               - alpha=1.0: slightly non-IID
               - alpha=100: nearly IID
        seed: Random seed

    Returns:
        dict: {client_id: list of indices}
    """
    np.random.seed(seed)

    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        # Extract labels by iterating
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    num_classes = len(np.unique(labels))
    num_items = len(dataset)

    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    # Initialize client data indices
    client_dict = {i: [] for i in range(num_clients)}

    # For each class, distribute samples to clients using Dirichlet
    for class_id in range(num_classes):
        indices = class_indices[class_id]
        np.random.shuffle(indices)

        # Sample from Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_clients)

        # Distribute indices according to proportions
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)

        for client_id in range(num_clients):
            start = 0 if client_id == 0 else proportions[client_id - 1]
            end = proportions[client_id]
            client_dict[client_id].extend(indices[start:end])

    # Shuffle each client's data
    for client_id in client_dict:
        np.random.shuffle(client_dict[client_id])

    return client_dict


def partition_data_power_law(dataset, num_clients, alpha=1.5, seed=42):
    """
    Partition dataset with power-law distributed sizes (for Level 5).

    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        alpha: Power law exponent (default 1.5)
        seed: Random seed

    Returns:
        dict: {client_id: list of indices}
    """
    np.random.seed(seed)

    # Generate power-law distributed sizes
    sizes = np.random.power(alpha, num_clients)
    sizes = sizes / sizes.sum()  # Normalize to sum to 1
    sizes = (sizes * len(dataset)).astype(int)

    # Ensure minimum size and adjust
    min_size = 50
    sizes = np.maximum(sizes, min_size)

    # Adjust to match total dataset size
    if sizes.sum() > len(dataset):
        # Scale down proportionally
        sizes = (sizes * len(dataset) / sizes.sum()).astype(int)

    # Shuffle indices
    indices = np.random.permutation(len(dataset))

    # Assign to clients
    client_dict = {}
    start = 0
    for i in range(num_clients):
        end = start + sizes[i]
        if i == num_clients - 1:  # Last client gets remaining
            end = len(dataset)
        client_dict[i] = indices[start:end].tolist()
        start = end

    return client_dict


def analyze_data_distribution(dataset, client_dict, num_classes=10):
    """
    Analyze and visualize data distribution across clients.

    Args:
        dataset: PyTorch dataset
        client_dict: Dictionary mapping client_id to indices
        num_classes: Number of classes

    Returns:
        dict: Statistics about data distribution
    """
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    stats = {
        'num_clients': len(client_dict),
        'num_classes': num_classes,
        'client_sizes': {},
        'class_distribution': {},
        'heterogeneity_metrics': {}
    }

    # Per-client statistics
    class_dist_matrix = np.zeros((len(client_dict), num_classes))

    for client_id, indices in client_dict.items():
        client_labels = labels[indices]

        # Size
        stats['client_sizes'][client_id] = len(indices)

        # Class distribution
        class_counts = np.bincount(client_labels, minlength=num_classes)
        stats['class_distribution'][client_id] = class_counts.tolist()
        class_dist_matrix[client_id] = class_counts

    # Heterogeneity metrics
    # 1. Normalized class distribution matrix
    class_dist_normalized = class_dist_matrix / (class_dist_matrix.sum(axis=1, keepdims=True) + 1e-8)

    # 2. KL divergence from uniform distribution
    uniform_dist = np.ones(num_classes) / num_classes
    kl_divs = []
    for client_dist in class_dist_normalized:
        kl_div = np.sum(client_dist * np.log((client_dist + 1e-8) / (uniform_dist + 1e-8)))
        kl_divs.append(kl_div)

    stats['heterogeneity_metrics']['mean_kl_divergence'] = float(np.mean(kl_divs))
    stats['heterogeneity_metrics']['std_kl_divergence'] = float(np.std(kl_divs))

    # 3. Class imbalance (std dev of class proportions)
    class_proportions_std = np.std(class_dist_normalized, axis=0)
    stats['heterogeneity_metrics']['mean_class_imbalance'] = float(np.mean(class_proportions_std))

    return stats


def create_dataloaders(dataset, client_dict, batch_size=32, num_workers=2):
    """
    Create DataLoader for each client.

    Args:
        dataset: PyTorch dataset
        client_dict: Dictionary mapping client_id to indices
        batch_size: Batch size for training
        num_workers: Number of workers for data loading

    Returns:
        dict: {client_id: DataLoader}
    """
    dataloaders = {}

    for client_id, indices in client_dict.items():
        subset = Subset(dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,  # Disabled to avoid PyTorch deprecation warnings
            persistent_workers=True if num_workers > 0 else False,  # Keep workers alive
            prefetch_factor=2 if num_workers > 0 else None,  # Prefetch 2 batches per worker
        )
        dataloaders[client_id] = loader

    return dataloaders


if __name__ == "__main__":
    print("Testing data loading and partitioning...")

    # Load data
    train_dataset, test_dataset = load_cifar10()
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Test IID partitioning
    print("\n=== IID Partitioning ===")
    client_dict_iid = partition_data_iid(train_dataset, num_clients=10)
    stats_iid = analyze_data_distribution(train_dataset, client_dict_iid)
    print(f"Client sizes: {list(stats_iid['client_sizes'].values())}")
    print(f"Heterogeneity (KL div): {stats_iid['heterogeneity_metrics']['mean_kl_divergence']:.4f}")

    # Test Dirichlet partitioning
    print("\n=== Dirichlet Partitioning (α=0.5) ===")
    client_dict_dir = partition_data_dirichlet(train_dataset, num_clients=10, alpha=0.5)
    stats_dir = analyze_data_distribution(train_dataset, client_dict_dir)
    print(f"Client sizes: {list(stats_dir['client_sizes'].values())}")
    print(f"Heterogeneity (KL div): {stats_dir['heterogeneity_metrics']['mean_kl_divergence']:.4f}")

    # Show class distribution for first few clients
    print("\nClass distribution for first 3 clients:")
    for i in range(3):
        dist = np.array(stats_dir['class_distribution'][i])
        print(f"  Client {i}: {dist}")

    # Test power-law partitioning
    print("\n=== Power-Law Partitioning ===")
    client_dict_power = partition_data_power_law(train_dataset, num_clients=10)
    stats_power = analyze_data_distribution(train_dataset, client_dict_power)
    sizes = list(stats_power['client_sizes'].values())
    print(f"Client sizes: {sizes}")
    print(f"Size range: {min(sizes)} - {max(sizes)}")
    print(f"Size std dev: {np.std(sizes):.1f}")
