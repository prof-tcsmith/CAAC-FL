"""
Download CIFAR-10 dataset for experiments.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from shared.data_utils import load_cifar10
import torch


def main():
    print("=" * 60)
    print("CIFAR-10 Dataset Download")
    print("=" * 60)
    print()

    # Set download directory
    data_dir = './data'
    print(f"Dataset will be downloaded to: {os.path.abspath(data_dir)}")
    print()

    # Download dataset
    print("Downloading CIFAR-10 dataset...")
    print("(This may take a few minutes on first run)")
    print("-" * 60)

    train_dataset, test_dataset = load_cifar10(data_dir=data_dir)

    print("-" * 60)
    print()
    print("✓ Download complete!")
    print()
    print("Dataset Information:")
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Test samples: {len(test_dataset):,}")
    print(f"  Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)")
    print(f"  Image size: 32x32x3 (RGB)")
    print(f"  Dataset size: ~170 MB")
    print()
    print(f"Dataset location: {os.path.abspath(data_dir)}/cifar-10-batches-py/")
    print()

    # Show sample statistics
    print("Sample from training set:")
    sample_img, sample_label = train_dataset[0]
    print(f"  Image tensor shape: {sample_img.shape}")
    print(f"  Label: {sample_label}")
    print()

    print("=" * 60)
    print("Dataset ready for experiments!")
    print("=" * 60)


if __name__ == "__main__":
    main()
