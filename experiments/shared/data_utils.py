"""
Data loading and partitioning utilities.

Supports:
- MNIST (28x28 grayscale, 10 classes)
- Fashion-MNIST (28x28 grayscale, 10 classes)
- CIFAR-10 (32x32 RGB, 10 classes)
"""

# Suppress PyTorch pin_memory deprecation warnings (from PyTorch internals)
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='torch.utils.data')

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


# Dataset metadata for reference
DATASET_INFO = {
    'mnist': {
        'num_classes': 10,
        'input_channels': 1,
        'input_size': (28, 28),
        'train_samples': 60000,
        'test_samples': 10000,
    },
    'fashion_mnist': {
        'num_classes': 10,
        'input_channels': 1,
        'input_size': (28, 28),
        'train_samples': 60000,
        'test_samples': 10000,
    },
    'cifar10': {
        'num_classes': 10,
        'input_channels': 3,
        'input_size': (32, 32),
        'train_samples': 50000,
        'test_samples': 10000,
    },
}


def load_dataset(dataset_name: str, data_dir: str = './data'):
    """
    Load a dataset by name.

    Args:
        dataset_name: One of 'mnist', 'fashion_mnist', 'cifar10'
        data_dir: Directory to store/load data

    Returns:
        train_dataset, test_dataset
    """
    dataset_name = dataset_name.lower().replace('-', '_')

    if dataset_name == 'mnist':
        return load_mnist(data_dir)
    elif dataset_name == 'fashion_mnist':
        return load_fashion_mnist(data_dir)
    elif dataset_name == 'cifar10':
        return load_cifar10(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Supported: {list(DATASET_INFO.keys())}")


def load_mnist(data_dir='./data'):
    """
    Load MNIST dataset.

    Args:
        data_dir: Directory to store/load data

    Returns:
        train_dataset, test_dataset
    """
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform_train
    )

    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    return train_dataset, test_dataset


def load_fashion_mnist(data_dir='./data'):
    """
    Load Fashion-MNIST dataset.

    Args:
        data_dir: Directory to store/load data

    Returns:
        train_dataset, test_dataset
    """
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    train_dataset = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform_train
    )

    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    return train_dataset, test_dataset


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


def partition_data_iid_unequal(dataset, num_clients, size_variation=0.5, seed=42):
    """
    Partition dataset into IID subsets with unequal sizes.

    Data is randomly distributed (IID) but clients have different dataset sizes.
    This tests FedAvg's weighting advantage vs unweighted FedMean.

    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        size_variation: Controls size heterogeneity
                        0.5 = moderate variation (sizes: 0.5x to 1.5x average)
                        1.0 = high variation (sizes: near 0x to 2.0x average)
        seed: Random seed for reproducibility

    Returns:
        dict: {client_id: list of indices}

    Example:
        For 50k samples, 50 clients, variation=0.5:
        - Average: 1000 samples/client
        - Range: ~500 to ~1500 samples/client
    """
    np.random.seed(seed)
    num_items = len(dataset)

    # Generate random size proportions using Dirichlet distribution
    # Lower alpha = more heterogeneous sizes
    # Higher alpha = more homogeneous sizes
    alpha_param = 1.0 / size_variation
    concentration = np.ones(num_clients) * alpha_param
    size_proportions = np.random.dirichlet(concentration)

    # Convert proportions to actual sizes
    client_sizes = (size_proportions * num_items).astype(int)

    # Adjust last client to ensure sum equals total
    client_sizes[-1] = num_items - client_sizes[:-1].sum()

    # Ensure no client has zero samples
    client_sizes = np.maximum(client_sizes, 1)

    # Randomly shuffle all indices (IID property)
    indices = np.random.permutation(num_items)

    # Partition indices according to sizes
    client_dict = {}
    start_idx = 0
    for client_id in range(num_clients):
        end_idx = start_idx + client_sizes[client_id]
        client_dict[client_id] = indices[start_idx:end_idx].tolist()
        start_idx = end_idx

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


def partition_data_dirichlet_equal(dataset, num_clients, alpha=0.5, seed=42):
    """
    Partition dataset into non-IID subsets with EQUAL sample counts per client.

    Uses Dirichlet distribution for label heterogeneity but enforces that each
    client receives exactly the same number of samples. This allows isolating
    the effect of label heterogeneity from quantity heterogeneity.

    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter
               - Lower alpha = more heterogeneous label distribution
               - alpha=0.1: highly non-IID
               - alpha=0.5: moderately non-IID
               - alpha=1.0: slightly non-IID
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
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    num_classes = len(np.unique(labels))
    num_items = len(dataset)
    samples_per_client = num_items // num_clients

    # Group indices by class and shuffle
    class_indices = {c: [] for c in range(num_classes)}
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    for c in range(num_classes):
        np.random.shuffle(class_indices[c])

    # Track how many samples remain in each class
    class_remaining = {c: len(class_indices[c]) for c in range(num_classes)}
    class_pointers = {c: 0 for c in range(num_classes)}

    # Generate Dirichlet proportions for each client (their class preferences)
    client_proportions = np.random.dirichlet([alpha] * num_classes, size=num_clients)

    # Initialize client data indices
    client_dict = {i: [] for i in range(num_clients)}

    # Assign samples to each client
    for client_id in range(num_clients):
        # Calculate how many samples this client wants from each class
        props = client_proportions[client_id]

        # Scale proportions to get target counts per class
        target_counts = (props * samples_per_client).astype(int)

        # Adjust to ensure we get exactly samples_per_client
        diff = samples_per_client - target_counts.sum()
        if diff > 0:
            # Add to classes with highest fractional parts
            fractional = (props * samples_per_client) - target_counts
            add_to = np.argsort(fractional)[-diff:]
            target_counts[add_to] += 1
        elif diff < 0:
            # Remove from classes with lowest fractional parts (but > 0)
            for _ in range(-diff):
                candidates = np.where(target_counts > 0)[0]
                if len(candidates) > 0:
                    fractional = (props * samples_per_client) - target_counts
                    remove_from = candidates[np.argmin(fractional[candidates])]
                    target_counts[remove_from] -= 1

        # Assign samples from each class
        for class_id in range(num_classes):
            want = target_counts[class_id]
            available = class_remaining[class_id]
            take = min(want, available)

            if take > 0:
                start = class_pointers[class_id]
                end = start + take
                client_dict[client_id].extend(class_indices[class_id][start:end])
                class_pointers[class_id] = end
                class_remaining[class_id] -= take

            # If we couldn't get enough from this class, we need to get from others
            shortfall = want - take
            if shortfall > 0:
                # Get from classes that still have samples
                for other_class in range(num_classes):
                    if other_class == class_id or shortfall == 0:
                        continue
                    other_available = class_remaining[other_class]
                    other_take = min(shortfall, other_available)
                    if other_take > 0:
                        start = class_pointers[other_class]
                        end = start + other_take
                        client_dict[client_id].extend(
                            class_indices[other_class][start:end]
                        )
                        class_pointers[other_class] = end
                        class_remaining[other_class] -= other_take
                        shortfall -= other_take

        # Shuffle this client's data
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
