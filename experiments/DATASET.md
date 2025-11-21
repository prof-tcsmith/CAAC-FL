# CIFAR-10 Dataset Information

## Overview

The CIFAR-10 dataset has been successfully downloaded and verified for use in CAAC-FL experiments.

## Dataset Statistics

- **Total Training Samples**: 50,000
- **Total Test Samples**: 10,000
- **Image Size**: 32×32×3 (RGB)
- **Number of Classes**: 10
- **Dataset Size**: ~170 MB (compressed), ~178 MB (extracted)
- **Format**: Python pickle files

## Class Distribution

The dataset is perfectly balanced with 5,000 samples per class:

| Class ID | Class Name | Training Samples | Percentage |
|----------|------------|------------------|------------|
| 0 | airplane | 5,000 | 10.0% |
| 1 | automobile | 5,000 | 10.0% |
| 2 | bird | 5,000 | 10.0% |
| 3 | cat | 5,000 | 10.0% |
| 4 | deer | 5,000 | 10.0% |
| 5 | dog | 5,000 | 10.0% |
| 6 | frog | 5,000 | 10.0% |
| 7 | horse | 5,000 | 10.0% |
| 8 | ship | 5,000 | 10.0% |
| 9 | truck | 5,000 | 10.0% |

## Directory Structure

```
experiments/data/
├── cifar-10-python.tar.gz          # Original download (163 MB)
└── cifar-10-batches-py/            # Extracted dataset
    ├── batches.meta                # Metadata
    ├── data_batch_1                # Training batch 1 (10,000 samples)
    ├── data_batch_2                # Training batch 2 (10,000 samples)
    ├── data_batch_3                # Training batch 3 (10,000 samples)
    ├── data_batch_4                # Training batch 4 (10,000 samples)
    ├── data_batch_5                # Training batch 5 (10,000 samples)
    ├── test_batch                  # Test set (10,000 samples)
    └── readme.html                 # Dataset documentation
```

## Data Preprocessing

The dataset is automatically preprocessed by `shared/data_utils.py`:

### Training Set Transforms
```python
transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # Random crop with padding
    transforms.RandomHorizontalFlip(),         # Random horizontal flip
    transforms.ToTensor(),                      # Convert to tensor
    transforms.Normalize(                       # Normalize
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])
```

### Test Set Transforms
```python
transforms.Compose([
    transforms.ToTensor(),                      # Convert to tensor
    transforms.Normalize(                       # Normalize
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])
```

## Partitioning Methods Available

### 1. IID Partitioning
- **Function**: `partition_data_iid()`
- **Heterogeneity**: KL divergence ≈ 0.0007 (very low)
- **Use**: Level 1 experiments
- **Characteristics**: Equal-sized partitions, uniform class distribution

### 2. Dirichlet Partitioning
- **Function**: `partition_data_dirichlet(alpha=0.5)`
- **Heterogeneity**: KL divergence ≈ 0.7475 (1138x more than IID)
- **Use**: Levels 2-5 experiments
- **Characteristics**: Variable-sized partitions, non-uniform class distribution
- **Example Distribution** (first 3 clients):
  - Client 0: truck(41%), horse(26%), dog(10%)
  - Client 1: deer(54%), frog(19%), ship(8%)
  - Client 2: automobile(39%), deer(32%), frog(18%)

### 3. Power-Law Partitioning
- **Function**: `partition_data_power_law()`
- **Use**: Level 5 (extreme heterogeneity)
- **Characteristics**: Highly variable partition sizes following power-law distribution

## Verification

Run `python verify_dataset.py` to:
- ✓ Verify dataset integrity
- ✓ Show class distribution
- ✓ Test IID partitioning
- ✓ Test Dirichlet partitioning
- ✓ Compare heterogeneity metrics

## Download Information

- **Download Date**: 2025-11-20
- **Source**: PyTorch torchvision.datasets.CIFAR10
- **Download Speed**: ~32 MB/s average
- **Total Download Time**: ~5 seconds

## Usage in Experiments

The dataset is automatically loaded by all experiment scripts:

```python
from shared.data_utils import load_cifar10

# Load dataset
train_dataset, test_dataset = load_cifar10(data_dir='./data')

# Partition for federated learning
client_dict = partition_data_iid(train_dataset, num_clients=10)
# or
client_dict = partition_data_dirichlet(train_dataset, num_clients=10, alpha=0.5)
```

## References

- **Dataset Paper**: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009
- **Official Website**: https://www.cs.toronto.edu/~kriz/cifar.html
- **PyTorch Documentation**: https://pytorch.org/vision/stable/datasets.html#cifar10

## Citation

If using CIFAR-10 in publications, cite:

```bibtex
@techreport{Krizhevsky2009,
  author = {Alex Krizhevsky},
  title = {Learning Multiple Layers of Features from Tiny Images},
  institution = {University of Toronto},
  year = {2009}
}
```
